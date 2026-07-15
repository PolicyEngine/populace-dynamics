"""Tests for the committed employer-firm target extracts and loaders.

The extracts in ``data/external/`` are small aggregate derivatives of
published Census/LEHD files (provenance:
``data/external/employer_firm_target_sources.md``), so these tests
always run. They pin the accounting identities the calibration and
gate machinery will rely on, and cross-check the extracts against the
banding mappings.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace_dynamics.firms import banding, targets

SECTORS = {
    "11",
    "21",
    "22",
    "23",
    "31-33",
    "42",
    "44-45",
    "48-49",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "61",
    "62",
    "71",
    "72",
    "81",
}


# ---------------------------------------------------------------
# SUSB
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def susb():
    return targets.load_susb_sector_size()


def test_susb_sectors_and_size_classes(susb):
    assert set(susb["naics_sector"]) == SECTORS | {"--", "99"}
    detail = susb[
        ~susb["entrsize_code"].isin(banding.SUSB_SUBTOTAL_CODES | {"01"})
    ]
    assert set(detail["entrsize_code"]) == set(banding.SUSB_ENTRSIZE_INTERVALS)


def test_susb_detail_employment_sums_to_total(susb):
    us = susb[susb["naics_sector"] == "--"]
    total = us.loc[us["entrsize_code"] == "01", "employment"].iloc[0]
    detail = us[
        ~us["entrsize_code"].isin(banding.SUSB_SUBTOTAL_CODES | {"01"})
    ]
    assert detail["employment"].sum() == total
    shares = detail["employment"] / total
    assert np.isclose(shares.sum(), 1.0)
    assert ((shares >= 0) & (shares <= 1)).all()


def test_susb_sector_totals_sum_to_us_total(susb):
    sector_totals = susb[
        (susb["naics_sector"] != "--") & (susb["entrsize_code"] == "01")
    ]
    assert (
        sector_totals["employment"].sum() == targets.SUSB_TOTAL_EMPLOYMENT_2022
    )


def test_susb_per_sector_detail_sums_to_sector_total(susb):
    """Each sector's detail size classes stack to that sector's own
    ENTRSIZE 01 total (not just the US margin)."""
    detail_codes = set(banding.SUSB_ENTRSIZE_INTERVALS)
    sectors = susb[
        (susb["naics_sector"] != "--") & (susb["naics_sector"] != "99")
    ]
    for sector, rows in sectors.groupby("naics_sector"):
        total = rows.loc[rows["entrsize_code"] == "01", "employment"]
        detail = rows[rows["entrsize_code"].isin(detail_codes)]
        assert detail["employment"].sum() == total.iloc[0], sector


def test_susb_canonical_band_shares(susb):
    """Detail classes roll up through the banding module exactly."""
    us = susb[
        (susb["naics_sector"] == "--")
        & ~susb["entrsize_code"].isin(banding.SUSB_SUBTOTAL_CODES | {"01"})
    ].copy()
    us["band"] = us["entrsize_code"].map(
        lambda c: banding.susb_entrsize_to_canonical(c).band.name
    )
    by_band = us.groupby("band")["employment"].sum()
    assert set(by_band.index) == {b.name for b in banding.CANONICAL_BANDS}
    assert by_band.sum() == targets.SUSB_TOTAL_EMPLOYMENT_2022
    # Large-firm dominance: the 500+ band holds the majority of US
    # private employment (published SUSB fact, ~55-62%).
    share_500 = by_band[banding.CanonicalBand.B500_PLUS.name] / by_band.sum()
    assert 0.5 < share_500 < 0.7


# ---------------------------------------------------------------
# BDS
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def bds():
    return targets.load_bds_firm_size()


def test_bds_shape_and_rates(bds):
    assert len(bds) == 45 * 10  # 1978-2022 x 10 size categories
    for col in ("job_creation_rate", "job_destruction_rate"):
        observed = bds[col].dropna()
        assert (observed >= 0).all()
        assert (observed <= 100).all()  # DHS rates per 100
    # Reallocation is the sum of creation and destruction rates.
    ok = bds.dropna(
        subset=[
            "job_creation_rate",
            "job_destruction_rate",
            "reallocation_rate",
        ]
    )
    assert np.allclose(
        ok["reallocation_rate"],
        # DHS reallocation = JC + JD - |net|.
        ok["job_creation_rate"]
        + ok["job_destruction_rate"]
        - ok["net_job_creation_rate"].abs(),
        atol=0.15,  # published rounding to 3 decimals
    )


def test_bds_categories_map_through_banding(bds):
    for cat in bds["fsize"].unique():
        span = banding.bds_fsize_to_canonical(cat)
        assert len(span.bands) >= 1


# ---------------------------------------------------------------
# QWI
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def qwi():
    return targets.load_qwi_firmsize_sector()


def test_qwi_schema_and_coverage(qwi):
    assert set(qwi["industry"]) == SECTORS | {"00"}
    assert set(qwi["firmsize"]) == {1, 2, 3, 4, 5}
    assert qwi["year"].min() == 2015
    assert qwi["year"].max() >= 2024


def test_qwi_rates_and_earnings(qwi):
    for col in ("hire_rate", "separation_rate"):
        observed = qwi[col].dropna()
        assert ((observed >= 0) & (observed <= 1)).all()
        # Quarterly hire/separation rates are materially positive.
        assert observed.median() > 0.02
    assert (qwi["EarnS"].dropna() > 0).all()


def test_qwi_firm_size_earnings_gradient(qwi):
    """E7's sign: mean earnings rise with firm size (500+ vs <20),
    on the all-industry margin."""
    us = qwi[qwi["industry"] == "00"]
    small = us.loc[us["firmsize"] == 1, "EarnS"].mean()
    large = us.loc[us["firmsize"] == 5, "EarnS"].mean()
    assert large > small


def test_qwi_size_margins_sum_to_reasonable_total(qwi):
    """Detail size classes stack to a plausible US private total."""
    us = qwi[
        (qwi["industry"] == "00")
        & (qwi["year"] == 2023)
        & (qwi["quarter"] == 1)
    ]
    total_emp = us["Emp"].sum()
    assert 1.0e8 < total_emp < 1.6e8  # US private beginning-of-quarter jobs


# ---------------------------------------------------------------
# J2J
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def j2j():
    return targets.load_j2j_firmsize_sector()


def test_j2j_schema_and_coverage(j2j):
    assert set(j2j["industry"]) == SECTORS | {"00"}
    assert set(j2j["firmsize"]) == {1, 2, 3, 4, 5}
    assert j2j["year"].min() == 2015


def test_j2j_rates_bounded(j2j):
    for col in ("j2j_hire_rate", "j2j_separation_rate"):
        observed = j2j[col].dropna()
        assert ((observed >= 0) & (observed <= 1)).all()
        assert observed.median() > 0.01


def test_j2j_flows_are_subsets_of_market_flows(j2j):
    """J2J hires are a component of all main-job hires; EE moves are
    a component of J2J."""
    ok = j2j.dropna(subset=["MHire", "J2JHire", "EEHire"])
    assert (ok["J2JHire"] <= ok["MHire"]).all()
    assert (ok["EEHire"] <= ok["J2JHire"]).all()
    ok = j2j.dropna(subset=["MSep", "J2JSep", "EESep"])
    assert (ok["J2JSep"] <= ok["MSep"]).all()
    assert (ok["EESep"] <= ok["J2JSep"]).all()


@pytest.mark.parametrize(
    "loader",
    [
        targets.load_susb_sector_size,
        targets.load_bds_firm_size,
        targets.load_qwi_firmsize_sector,
        targets.load_j2j_firmsize_sector,
    ],
)
def test_loaders_return_independent_copies(loader):
    """A caller mutating a loaded frame must not corrupt a later load
    (the cached frame is shared; the public loader returns a copy)."""
    first = loader()
    first.iloc[0, 0] = None
    first["_scratch_col"] = 1
    second = loader()
    assert "_scratch_col" not in second.columns
    assert second.iloc[0, 0] is not None


def test_lehd_labels_match_banding_intervals(qwi, j2j):
    """Committed firmsize labels agree with the pinned banding
    intervals (e.g. code 3 is '50-249 Employees')."""
    for df in (qwi, j2j):
        pairs = df[["firmsize", "firmsize_label"]].drop_duplicates()
        for code, label in zip(
            pairs["firmsize"], pairs["firmsize_label"], strict=True
        ):
            lo, hi = banding.LEHD_FIRMSIZE_INTERVALS[int(code)]
            if hi == float("inf"):
                assert label.startswith(f"{lo}+") or "500+" in label
            else:
                assert str(int(hi)) in label
