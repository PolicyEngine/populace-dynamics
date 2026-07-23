"""The C2 firm-size seam: readers -> ``firms/banding.py``.

The seam Max flagged before the C1/C2 freeze (#192, #208): the
person-side readers and the target-side mapper name the same concept
in different code spaces, so wiring one into the other without an
adapter mis-bands silently. ``NOEMP 6`` is 1000+ while IPUMS
``FIRMSIZE 6`` is 50-99 — a factor-20 error that raises nothing.

The freeze closed this in code (``banding.py`` grew
``noemp_to_canonical`` and refuses to guess a coding) but not in the
pipeline: no reader imported it. These tests pin the wiring, and are
the "one seam test that round-trips every NOEMP code through to
``CanonicalBand`` and asserts the intended bands" from the #192
thread.

Three properties, in the order they'd fail if the seam reopened:

1. every raw code maps, and to the band the dictionary evidence says;
2. the readers emit *that* mapping rather than a private one;
3. the two CPS code spaces stay distinguishable, so a caller holding
   the wrong one gets an error rather than a plausible wrong band.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import asec_firm_size, sipp_jobs
from populace_dynamics.firms import banding
from populace_dynamics.firms.banding import (
    CanonicalBand,
    cps_firmsize_to_canonical,
    noemp_to_canonical,
    sipp_empsize_to_canonical,
)

# The intended bands, stated here independently of the mapper's
# tables so that editing one of them cannot silently move the other.
# Codes 5 (500-999) and 6 (1000+) both land in B500_PLUS: canonical
# C2 is coarser than NOEMP above 500, by design.
NOEMP_INTENDED: dict[int, CanonicalBand] = {
    1: CanonicalBand.LT10,
    2: CanonicalBand.B10_49,
    3: CanonicalBand.B50_99,
    4: CanonicalBand.B100_499,
    5: CanonicalBand.B500_PLUS,
    6: CanonicalBand.B500_PLUS,
}

#: EMPSIZE code -> intended canonical span label. SIPP's inclusive
#: upper bounds straddle the canonical edges by one integer, so most
#: codes are inexact and must render as a joined run.
EMPSIZE_INTENDED: dict[int, str] = {
    1: "1-9",
    2: "10-49",
    3: "10-49|50-99",
    4: "50-99|100-499",
    5: "100-499",
    6: "100-499|500+",
    7: "500+",
    8: "500+",
}


class TestNoempRoundTrip:
    """Every NOEMP code, every vintage, through to CanonicalBand."""

    @pytest.mark.parametrize("year", [2011, 2018, 2019, 2024, 2025])
    @pytest.mark.parametrize("code", sorted(NOEMP_INTENDED))
    def test_every_code_maps_to_the_intended_band(self, code, year):
        span = noemp_to_canonical(code, year)
        assert span is not None
        assert span.exact, (
            f"NOEMP {code} straddles a canonical edge in {year}; "
            "every ASEC firm-size band is supposed to nest one."
        )
        assert span.band is NOEMP_INTENDED[code]

    @pytest.mark.parametrize("year", [2011, 2018, 2019, 2024, 2025])
    def test_niu_is_none_not_a_band(self, year):
        # Code 0 must not fall into LT10; a NIU banded as "1-9" would
        # move non-workers into the smallest firm-size cell.
        assert noemp_to_canonical(0, year) is None

    def test_the_50_edge_holds_across_the_phantom_relabeling(self):
        # The 2019+ dictionaries relabel codes 2/3 as 10-24 / 25-99.
        # If that relabeling were ever taken at face value, code 3
        # would stop resolving the 50 edge and every ACA-threshold
        # cell would silently change population (#192).
        for year in (2017, 2018, 2019, 2024):
            assert noemp_to_canonical(3, year).band is CanonicalBand.B50_99
            assert noemp_to_canonical(2, year).band is CanonicalBand.B10_49


class TestReadersEmitTheCanonicalMapping:
    """The readers must not band independently of banding.py."""

    @pytest.mark.parametrize("year", [2011, 2018, 2019, 2025])
    def test_asec_map_is_the_banding_map(self, year):
        emitted = asec_firm_size.noemp_canonical_map(year)
        assert emitted == {
            code: NOEMP_INTENDED[code].label for code in NOEMP_INTENDED
        }

    def test_asec_reader_emits_canonical_band(self, tmp_path):
        rows = [{"NOEMP": code} for code in sorted(NOEMP_INTENDED)]
        frame = pd.DataFrame(
            [
                {
                    "PERIDNUM": 10_000 + i,
                    "I_NOEMP": 0,
                    "LJCW": 1,
                    "INDUSTRY": 770,
                    "WEIND": 4,
                    "WKSWORK": 52,
                    "WORKYN": 1,
                    "MARSUPWT": 100_000,
                    **row,
                }
                for i, row in enumerate(rows)
            ]
        )
        path = tmp_path / "pppub24.csv"
        frame.to_csv(path, index=False)

        out = asec_firm_size.read_asec_firm_size(2024, path=path)
        assert "canonical_band" in out.columns
        for code, band in NOEMP_INTENDED.items():
            got = out.loc[out["noemp"] == code, "canonical_band"]
            assert list(got) == [band.label]

    def test_sipp_spans_are_the_banding_spans(self):
        assert sipp_jobs.EMPSIZE_CANONICAL_SPANS == EMPSIZE_INTENDED
        for code, label in EMPSIZE_INTENDED.items():
            assert sipp_empsize_to_canonical(code).label == label

    def test_sipp_straddling_codes_stay_inexact(self):
        # Collapsing 3/4/6 to a single band would invent precision
        # SIPP's inclusive upper bounds do not have.
        exact = sipp_jobs.EMPSIZE_CANONICAL_EXACT
        assert [c for c, e in exact.items() if not e] == [3, 4, 6]
        for code in (3, 4, 6):
            assert banding.SPAN_LABEL_SEPARATOR in (
                sipp_jobs.EMPSIZE_CANONICAL_SPANS[code]
            )


class TestTheCodeSpacesStayDistinguishable:
    """A caller holding the wrong coding must not get a band."""

    def test_noemp_and_ipums_disagree_where_documented(self):
        # The collision that motivated the seam: same integer, same
        # vintage, different band. If these ever agree, the mapper
        # has lost the distinction and the adapter is a no-op.
        assert noemp_to_canonical(6, 2016).band is CanonicalBand.B500_PLUS
        assert (
            cps_firmsize_to_canonical(6, 2016, coding="ipums_firmsize").band
            is CanonicalBand.B50_99
        )

    def test_coding_has_no_default(self):
        with pytest.raises(TypeError):
            cps_firmsize_to_canonical(6, 2016)

    def test_unknown_coding_raises(self):
        with pytest.raises(ValueError, match="Unknown coding"):
            cps_firmsize_to_canonical(6, 2016, coding="noemp")

    def test_pre_2011_refuses_rather_than_borrowing_a_vintage(self):
        with pytest.raises(ValueError, match="2011"):
            noemp_to_canonical(2, 2010)


class TestOneVocabulary:
    """Person side and target side share one band vocabulary."""

    def test_labels_are_the_canonical_set(self):
        assert [b.label for b in banding.CANONICAL_BANDS] == [
            "1-9",
            "10-49",
            "50-99",
            "100-499",
            "500+",
        ]

    def test_asec_canonical_labels_are_a_subset(self):
        vocabulary = {b.label for b in banding.CANONICAL_BANDS}
        emitted = set(asec_firm_size.noemp_canonical_map(2024).values())
        assert emitted <= vocabulary

    def test_sipp_exact_spans_are_in_the_vocabulary(self):
        vocabulary = {b.label for b in banding.CANONICAL_BANDS}
        for code, exact in sipp_jobs.EMPSIZE_CANONICAL_EXACT.items():
            label = sipp_jobs.EMPSIZE_CANONICAL_SPANS[code]
            assert (label in vocabulary) is exact

    def test_establishment_and_firm_columns_are_named_apart(self):
        # SIPP measures establishment size (#192 finding 1). If both
        # readers emitted "canonical_band", a join would silently
        # treat a location headcount as an enterprise headcount.
        assert "canonical_band" not in sipp_jobs.__all__
        assert hasattr(asec_firm_size, "noemp_canonical_map")


REAL_SIPP = Path("~/PolicyEngine/sipp-data").expanduser()


@pytest.mark.skipif(not REAL_SIPP.is_dir(), reason="SIPP pu files not staged")
def test_real_sipp_bands_every_valid_code():
    """On real data, no valid EMPSIZE code falls through unbanded."""
    months = sipp_jobs.read_sipp_job_months(2023)
    valid = months["empsize_code"].isin(sorted(EMPSIZE_INTENDED))
    assert months.loc[valid, "estab_size_band"].notna().all()
    # And nothing outside the universe acquires a band.
    assert months.loc[~valid, "estab_size_band"].isna().all()
