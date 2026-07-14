"""Tests for the canonical firm-size banding (contract C2).

Checks the properties the contract promises: canonical bands
partition the positive integers; every raw source code maps to
exactly one contiguous ``BandSpan``; within each source vintage the
mapped intervals are non-overlapping and cover the source's domain;
and the specific reconciliation facts recorded in ADR 0003 (SUSB and
2011-2018 ASEC nest exactly; the 2019+ ASEC 25-99 band straddles the
50 edge) hold.
"""

from __future__ import annotations

import math

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


def _spans(name):
    if name == "cps_2011_2018":
        return [
            cps_firmsize_to_canonical(c, 2015)
            for c in CPS_FIRMSIZE_INTERVALS_2011_2018
        ]
    if name == "cps_standard":
        return [
            cps_firmsize_to_canonical(c, 2020)
            for c in banding.CPS_FIRMSIZE_INTERVALS_STANDARD
        ]
    if name == "sipp":
        return [sipp_empsize_to_canonical(c) for c in SIPP_EMPSIZE_INTERVALS]
    if name == "susb":
        return [susb_entrsize_to_canonical(c) for c in SUSB_ENTRSIZE_INTERVALS]
    if name == "lehd":
        return [lehd_firmsize_to_canonical(c) for c in LEHD_FIRMSIZE_INTERVALS]
    return [bds_fsize_to_canonical(c) for c in BDS_FSIZE_INTERVALS]


@pytest.mark.parametrize("name", sorted(ALL_INTERVAL_TABLES))
def test_every_code_maps_to_one_contiguous_span(name):
    for span in _spans(name):
        assert span is not None
        assert len(span.bands) >= 1
        order = [CANONICAL_BANDS.index(b) for b in span.bands]
        assert order == list(range(order[0], order[0] + len(order)))


@pytest.mark.parametrize("name", sorted(ALL_INTERVAL_TABLES))
def test_span_matches_interval_containment(name):
    """The span is exactly the canonical bands the interval touches."""
    table = ALL_INTERVAL_TABLES[name]
    for code, span in zip(table, _spans(name), strict=True):
        lo, hi = table[code]
        expected = tuple(
            b for b in CANONICAL_BANDS if b.lo <= hi and b.hi >= lo
        )
        assert span.bands == expected, (name, code)


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
        span = cps_firmsize_to_canonical(code, 2015)
        assert span.exact, code
    assert cps_firmsize_to_canonical(6, 2011).band is CanonicalBand.B50_99


def test_cps_post2019_25_99_straddles_fifty():
    span = cps_firmsize_to_canonical(5, 2023)
    assert not span.exact
    assert span.bands == (CanonicalBand.B10_49, CanonicalBand.B50_99)
    with pytest.raises(ValueError):
        _ = span.band  # ambiguity must be handled explicitly


def test_cps_vintage_switching():
    # Code 4 (10-49) exists only in the 2011-2018 vintage.
    assert cps_firmsize_to_canonical(4, 2015).bands == (CanonicalBand.B10_49,)
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(4, 2023)
    # Code 2 (10-24) exists only outside 2011-2018.
    assert cps_firmsize_to_canonical(2, 2023).exact
    with pytest.raises(KeyError):
        cps_firmsize_to_canonical(2, 2015)


def test_cps_niu_returns_none():
    assert cps_firmsize_to_canonical(0, 2023) is None


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
