"""Tests for the canonical firm-size banding (contract IC2).

Checks the properties the contract promises: canonical bands
partition the positive integers; every raw source code maps to
exactly one contiguous ``BandSpan``; within each source vintage the
mapped intervals are non-overlapping and cover the source's domain;
and the specific reconciliation facts recorded in ADR 0003 (SUSB and
every ASEC firm-size vintage nest exactly; the 2019+ ASEC "25 to 99"
label is a phantom relabeling of the 50-99 band, #192) hold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from populace_dynamics.firms import banding
from populace_dynamics.firms.banding import (
    BDS_FSIZE_INTERVALS,
    CANONICAL_BANDS,
    CPS_FIRMSIZE_INTERVALS_2011_2018,
    LEHD_FIRMSIZE_INTERVALS,
    SIPP_EMPSIZE_INTERVALS,
    SUSB_ENTRSIZE_INTERVALS,
    CanonicalBand,
    band_of_count,
    bds_fsize_to_canonical,
    cps_firmsize_to_canonical,
    lehd_firmsize_to_canonical,
    sipp_empsize_to_canonical,
    susb_entrsize_to_canonical,
)

ALL_INTERVAL_TABLES = {
    "cps_2011_2018": CPS_FIRMSIZE_INTERVALS_2011_2018,
    "cps_standard": banding.CPS_FIRMSIZE_INTERVALS_STANDARD,
    "sipp": SIPP_EMPSIZE_INTERVALS,
    "susb": SUSB_ENTRSIZE_INTERVALS,
    "lehd": LEHD_FIRMSIZE_INTERVALS,
    "bds": BDS_FSIZE_INTERVALS,
}

# CPS standard code 3 ("Under 25", 1988-1991 only) overlaps codes 1/2
# by value but never co-occurs with them within a vintage; the
# non-overlap invariant is checked with it removed (see banding.py).
VINTAGE_EXCLUSIVE_CODES = {"cps_standard": {3}}


# ---------------------------------------------------------------
# Canonical band set
# ---------------------------------------------------------------


def test_canonical_bands_partition_positive_integers():
    bands = list(CANONICAL_BANDS)
    assert bands[0].lo == 1
    assert bands[-1].hi == math.inf
    for prev, nxt in zip(bands, bands[1:], strict=False):
        assert nxt.lo == prev.hi + 1  # contiguous, no gap or overlap


def test_fifty_is_a_band_edge():
    """The ACA/state-mandate threshold must be an edge (ADR 0003)."""
    assert CanonicalBand.B50_99.lo == 50


@pytest.mark.parametrize(
    ("count", "band"),
    [
        (1, CanonicalBand.LT10),
        (9, CanonicalBand.LT10),
        (10, CanonicalBand.B10_49),
        (49, CanonicalBand.B10_49),
        (50, CanonicalBand.B50_99),
        (499, CanonicalBand.B100_499),
        (500, CanonicalBand.B500_PLUS),
        (10**7, CanonicalBand.B500_PLUS),
    ],
)
def test_band_of_count(count, band):
    assert band_of_count(count) is band


def test_band_of_count_rejects_nonpositive():
    with pytest.raises(ValueError):
        band_of_count(0)


# ---------------------------------------------------------------
# Generic mapping properties: total, single-span, non-overlapping
# ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_INTERVAL_TABLES))
def test_source_intervals_non_overlapping(name):
    excluded = VINTAGE_EXCLUSIVE_CODES.get(name, set())
    intervals = sorted(
        iv
        for code, iv in ALL_INTERVAL_TABLES[name].items()
        if code not in excluded
    )
    for (lo_a, hi_a), (lo_b, _) in zip(intervals, intervals[1:], strict=False):
        assert hi_a < lo_b, f"{name}: [{lo_a},{hi_a}] overlaps {lo_b}"


def _code_spans(name):
    """(code, span) for every dispatchable code in the named table.

    Vintage-exclusive codes (cps_standard code 3, dispatchable only in
    the 1988-1991 regime this module does not serve) are skipped.
    """
    excluded = VINTAGE_EXCLUSIVE_CODES.get(name, set())
    codes = [c for c in ALL_INTERVAL_TABLES[name] if c not in excluded]
    if name == "cps_2011_2018":
        spans = [
            cps_firmsize_to_canonical(c, 2015, coding="ipums_firmsize")
            for c in codes
        ]
    elif name == "cps_standard":
        spans = [
            cps_firmsize_to_canonical(c, 2020, coding="ipums_firmsize")
            for c in codes
        ]
    elif name == "sipp":
        spans = [sipp_empsize_to_canonical(c) for c in codes]
    elif name == "susb":
        spans = [susb_entrsize_to_canonical(c) for c in codes]
    elif name == "lehd":
        spans = [lehd_firmsize_to_canonical(c) for c in codes]
    else:
        spans = [bds_fsize_to_canonical(c) for c in codes]
    return list(zip(codes, spans, strict=True))


@pytest.mark.parametrize("name", sorted(ALL_INTERVAL_TABLES))
def test_every_code_maps_to_one_contiguous_span(name):
    for _code, span in _code_spans(name):
        assert span is not None
        assert len(span.bands) >= 1
        order = [CANONICAL_BANDS.index(b) for b in span.bands]
        assert order == list(range(order[0], order[0] + len(order)))


@pytest.mark.parametrize("name", sorted(ALL_INTERVAL_TABLES))
def test_span_matches_interval_containment(name):
    """The span is exactly the canonical bands the interval touches."""
    table = ALL_INTERVAL_TABLES[name]
    for code, span in _code_spans(name):
        lo, hi = table[code]
        expected = tuple(
            b for b in CANONICAL_BANDS if b.lo <= hi and b.hi >= lo
        )
        assert span.bands == expected, (name, code)


#: The full code domain each source table must carry — a completeness
#: pin so silently dropping (or adding) a source code fails, which the
#: property tests above cannot catch (they only walk whatever is
#: present). Verified against the primary sources (ADR 0003).
EXPECTED_CODE_DOMAINS = {
    "cps_2011_2018": {1, 4, 6, 7, 8, 9},
    "cps_standard": {1, 2, 3, 5, 7, 8, 9},
    "sipp": {1, 2, 3, 4, 5, 6, 7, 8},
    "susb": {
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "31",
        "22",
        "23",
        "24",
        "25",
    },
    "lehd": {1, 2, 3, 4, 5},
    "bds": set(BDS_FSIZE_INTERVALS),
}


@pytest.mark.parametrize("name", sorted(ALL_INTERVAL_TABLES))
def test_source_code_domain_is_complete(name):
    assert set(ALL_INTERVAL_TABLES[name]) == EXPECTED_CODE_DOMAINS[name]


# ---------------------------------------------------------------
# Recorded reconciliation facts (ADR 0003)
# ---------------------------------------------------------------


def test_susb_detail_classes_nest_exactly():
    for code in SUSB_ENTRSIZE_INTERVALS:
        assert susb_entrsize_to_canonical(code).exact, code


def test_susb_detail_classes_cover_everything():
    intervals = sorted(SUSB_ENTRSIZE_INTERVALS.values())
    assert intervals[0][0] == 1
    assert intervals[-1][1] == math.inf
    for (_, hi_a), (lo_b, _) in zip(intervals, intervals[1:], strict=False):
        assert lo_b == hi_a + 1


def test_susb_subtotals_return_none():
    for code in banding.SUSB_SUBTOTAL_CODES:
        assert susb_entrsize_to_canonical(code) is None


def test_cps_2011_2018_nests_exactly():
    for code in CPS_FIRMSIZE_INTERVALS_2011_2018:
        span = cps_firmsize_to_canonical(code, 2015, coding="ipums_firmsize")
        assert span.exact, code
    assert (
        cps_firmsize_to_canonical(6, 2011, coding="ipums_firmsize").band
        is CanonicalBand.B50_99
    )


def test_cps_post2019_code5_is_exact_fifty_to_99():
    # IPUMS 2019+ code 5 is *labelled* "25 to 99" but the relabeling is
    # phantom (#192): it carries the 50-99 band and resolves the 50 edge
    # exactly, like the 2011-2018 vintage.
    span = cps_firmsize_to_canonical(5, 2023, coding="ipums_firmsize")
    assert span.exact
    assert span.band is CanonicalBand.B50_99


def test_cps_vintage_switching():
    ipums = {"coding": "ipums_firmsize"}
    # Code 4 (10-49) exists only in the 2011-2018 vintage.
    assert cps_firmsize_to_canonical(4, 2015, **ipums).bands == (
        CanonicalBand.B10_49,
    )
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(4, 2023, **ipums)
    # Code 2 (labelled "10-24"; empirically 10-49) exists only 2019+.
    assert (
        cps_firmsize_to_canonical(2, 2023, **ipums).band
        is CanonicalBand.B10_49
    )
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(2, 2015, **ipums)


def test_cps_coding_is_required():
    """No default coding: the both-spaces footgun (code 4/6 in
    2011-2018, code 5 in 2019+) cannot mis-band via an omitted arg."""
    with pytest.raises(TypeError):
        cps_firmsize_to_canonical(4, 2015)


def test_cps_niu_returns_none():
    assert cps_firmsize_to_canonical(0, 2023, coding="ipums_firmsize") is None


def test_sipp_missing_and_inclusive_bounds():
    assert sipp_empsize_to_canonical(-9) is None
    assert sipp_empsize_to_canonical(1).band is CanonicalBand.LT10
    # "26 to 50" straddles the 50 edge by its inclusive upper bound.
    assert sipp_empsize_to_canonical(3).bands == (
        CanonicalBand.B10_49,
        CanonicalBand.B50_99,
    )


def test_lehd_margins_return_none():
    assert lehd_firmsize_to_canonical(0) is None
    assert lehd_firmsize_to_canonical("N") is None
    assert lehd_firmsize_to_canonical(2).band is CanonicalBand.B10_49
    assert lehd_firmsize_to_canonical(5).band is CanonicalBand.B500_PLUS


def test_bds_20_99_straddles_fifty():
    span = bds_fsize_to_canonical("d) 20 to 99")
    assert span.bands == (CanonicalBand.B10_49, CanonicalBand.B50_99)
    assert bds_fsize_to_canonical("j) 10000+").exact


def test_unknown_codes_raise():
    with pytest.raises(KeyError):
        sipp_empsize_to_canonical(99)
    with pytest.raises(KeyError):
        susb_entrsize_to_canonical("77")
    with pytest.raises(KeyError):
        bds_fsize_to_canonical("k) other")
    with pytest.raises(KeyError):
        lehd_firmsize_to_canonical(9)


# ---------------------------------------------------------------
# Vintage-impossible IPUMS FIRMSIZE codes must raise (ADR 0003 #5)
# ---------------------------------------------------------------


def test_cps_firmsize_rejects_vintage_impossible_codes():
    ipums = {"coding": "ipums_firmsize"}
    # Code 3 ("Under 25") exists only in the 1988-1991 vintage, which
    # this module does not dispatch; it must not borrow a 2019+ interval.
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(3, 2023, **ipums)
    # Codes 4/6 are 2011-2018-only in IPUMS FIRMSIZE.
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(4, 2023, **ipums)
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(6, 2023, **ipums)
    # Codes 2/5 are standard-only.
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(2, 2015, **ipums)
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(5, 2015, **ipums)


def test_cps_firmsize_rejects_unserved_years():
    # 1988-1991 "Under 25" vintage, and 1992-2010 (now refused on the
    # IPUMS leg too, symmetric with NOEMP — only 2011+ is dispatchable).
    with pytest.raises(ValueError):
        cps_firmsize_to_canonical(1, 1989, coding="ipums_firmsize")
    with pytest.raises(ValueError):
        cps_firmsize_to_canonical(1, 2005, coding="ipums_firmsize")


def test_cps_firmsize_rejects_unknown_coding():
    with pytest.raises(ValueError):
        cps_firmsize_to_canonical(1, 2015, coding="raw")


# ---------------------------------------------------------------
# NOEMP is a distinct coding from IPUMS FIRMSIZE (seam with #194)
# ---------------------------------------------------------------


def test_noemp_bands_2011_2018():
    # NOEMP code numbering differs from IPUMS FIRMSIZE: NOEMP 2 is
    # 10-49, but IPUMS code 2 does not occur in the 2011-2018 vintage.
    assert banding.noemp_to_canonical(2, 2015).band is CanonicalBand.B10_49
    assert banding.noemp_to_canonical(3, 2015).band is CanonicalBand.B50_99
    assert banding.noemp_to_canonical(4, 2015).band is CanonicalBand.B100_499
    assert banding.noemp_to_canonical(6, 2015).band is CanonicalBand.B500_PLUS
    assert banding.noemp_to_canonical(0, 2015) is None


def test_noemp_bands_2019_plus_match_2011_2018():
    # The 2019+ "10-24"/"25-99" dictionary labels are a phantom
    # relabeling (#192); NOEMP codes 2/3 carry the same 10-49/50-99
    # bands as 2011-2018 and resolve the 50 edge exactly.
    assert banding.noemp_to_canonical(2, 2023).band is CanonicalBand.B10_49
    assert banding.noemp_to_canonical(3, 2023).band is CanonicalBand.B50_99


def test_noemp_and_firmsize_codings_disagree():
    """The same integer is a different band under each coding — the
    collision the explicit `coding` argument exists to prevent."""
    # Code 4, 2011-2018: NOEMP -> 100-499, IPUMS FIRMSIZE -> 10-49.
    assert banding.noemp_to_canonical(4, 2015).band is CanonicalBand.B100_499
    assert (
        cps_firmsize_to_canonical(4, 2015, coding="ipums_firmsize").band
        is CanonicalBand.B10_49
    )
    # Routing NOEMP through the explicit census_noemp coding agrees
    # with the direct entry point.
    assert (
        cps_firmsize_to_canonical(4, 2015, coding="census_noemp").band
        is CanonicalBand.B100_499
    )


def test_noemp_rejects_pre_2011():
    with pytest.raises(ValueError):
        banding.noemp_to_canonical(1, 2005)


# ---------------------------------------------------------------
# Helper edge cases (ADR 0003 review, green suggestions)
# ---------------------------------------------------------------


def test_span_rejects_inverted_interval():
    with pytest.raises(ValueError):
        banding._span(50, 40)


def test_band_of_count_rejects_non_integer():
    with pytest.raises(ValueError):
        band_of_count(9.5)
    # A whole-valued float is fine.
    assert band_of_count(9.0) is CanonicalBand.LT10


def test_band_of_count_accepts_numpy_integers():
    # A Series.map(band_of_count) hands over numpy integer dtypes, which
    # are not Python ``int`` but are ``numbers.Integral`` (F2).
    assert band_of_count(np.int64(50)) is CanonicalBand.B50_99
    assert band_of_count(np.int32(9)) is CanonicalBand.LT10
    # numpy bool is still rejected, like Python bool.
    with pytest.raises(ValueError):
        band_of_count(np.bool_(True))


def test_lehd_non_numeric_raises_keyerror():
    with pytest.raises(KeyError):
        lehd_firmsize_to_canonical("abc")
