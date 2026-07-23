"""Tests for the committed employer-firm target extracts and loaders.

The extracts in ``data/external/`` are small aggregate derivatives of
published Census/LEHD files (provenance:
``data/external/employer_firm_target_sources.md``), so these tests
always run. They pin the accounting identities the calibration and
gate machinery will rely on, and cross-check the extracts against the
banding mappings.
"""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------
# J2J sex x age (gate E2 reference)
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def j2j_sexage():
    return targets.load_j2j_sexage()


def test_j2j_sexage_schema_and_coverage(j2j_sexage):
    assert set(j2j_sexage["sex"]) == {0, 1, 2}
    assert set(j2j_sexage["agegrp"]) == {f"A0{i}" for i in range(9)}
    assert j2j_sexage["year"].min() == 2015
    assert j2j_sexage["year"].max() >= 2024
    # Full grid: one row per quarter x sex x age group.
    quarters = j2j_sexage[["year", "quarter"]].drop_duplicates()
    assert len(j2j_sexage) == len(quarters) * 3 * 9


def test_j2j_sexage_rates_bounded(j2j_sexage):
    for col in (
        "hire_rate",
        "separation_rate",
        "j2j_hire_rate",
        "j2j_separation_rate",
    ):
        observed = j2j_sexage[col].dropna()
        assert ((observed >= 0) & (observed <= 1)).all()
        assert observed.median() > 0.01


def test_j2j_sexage_margins_stack(j2j_sexage):
    """Detail sex and age cells stack to the published margins
    (exactly: LEHD margins here are published, not derived)."""
    cell = j2j_sexage.set_index(["year", "quarter", "sex", "agegrp"])["MainB"]
    total = cell.loc[2019, 1, 0, "A00"]
    by_sex = sum(cell.loc[2019, 1, s, "A00"] for s in (1, 2))
    by_age = sum(cell.loc[2019, 1, 0, f"A0{i}"] for i in range(1, 9))
    # LEHD noise infusion rounds each published cell independently,
    # so margins agree only to a few jobs out of ~130 million.
    assert abs(by_sex - total) <= 5
    assert abs(by_age - total) <= 5


def test_j2j_sexage_age_gradient(j2j_sexage):
    """E2's sign: young workers churn faster -- the 19-21 group's
    J2J hire rate exceeds the 55-64 group's, both sexes pooled."""
    us = j2j_sexage[j2j_sexage["sex"] == 0]
    young = us.loc[us["agegrp"] == "A02", "j2j_hire_rate"].mean()
    older = us.loc[us["agegrp"] == "A07", "j2j_hire_rate"].mean()
    assert young > 2 * older


# ---------------------------------------------------------------
# J2JOD origin x destination firm size (gate E11 reference)
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def j2jod():
    return targets.load_j2jod_firmsize()


def test_j2jod_schema_and_coverage(j2jod):
    assert set(j2jod["firmsize"]) == {0, 1, 2, 3, 4, 5}
    assert set(j2jod["firmsize_orig"]) == {0, 1, 2, 3, 4, 5}
    assert j2jod["year"].min() == 2015
    assert j2jod["year"].max() >= 2024
    # Full 6 x 6 grid per quarter.
    quarters = j2jod[["year", "quarter"]].drop_duplicates()
    assert len(j2jod) == len(quarters) * 36


def test_j2jod_identity_j2j_is_ee_plus_aqhire(j2jod):
    ok = j2jod.dropna(subset=["EE", "AQHire", "J2J"])
    assert (ok["J2J"] == ok["EE"] + ok["AQHire"]).all()
    ok = j2jod.dropna(subset=["EES", "AQHireS", "J2JS"])
    assert (ok["J2JS"] == ok["EES"] + ok["AQHireS"]).all()


def test_j2jod_detail_window(j2jod):
    """The full national origin x destination detail is published
    only for 2015Q1-2016Q1 (later quarters carry status flag 11: a
    state coverage gap blocks the national aggregate; provenance
    note entry 6). Pin the window so a re-fetch that changes it
    fails loudly."""
    detail = j2jod[(j2jod["firmsize"] > 0) & (j2jod["firmsize_orig"] > 0)]
    published = detail[detail["EE"].notna()]
    assert set(zip(published["year"], published["quarter"], strict=True)) == {
        (2015, 1),
        (2015, 2),
        (2015, 3),
        (2015, 4),
        (2016, 1),
    }
    assert detail.loc[detail["EE"].isna(), "sEE"].isin([11]).all()
    # The margins stay published for every quarter.
    margin = j2jod[(j2jod["firmsize"] == 0) & (j2jod["firmsize_orig"] == 0)]
    assert margin["EE"].notna().all()


def test_j2jod_detail_cells_below_margins(j2jod):
    """Each detail origin x destination cell is bounded by both of
    its one-sided margins (the margins aggregate the detail)."""
    cell = j2jod.set_index(["year", "quarter", "firmsize_orig", "firmsize"])[
        "EE"
    ]
    for o in range(1, 6):
        for d in range(1, 6):
            detail = cell.loc[2015, 3, o, d]
            if np.isnan(detail):
                continue
            assert detail <= cell.loc[2015, 3, o, 0]
            assert detail <= cell.loc[2015, 3, 0, d]


def test_j2jod_large_firms_dominate_flows(j2jod):
    """E11's sign: the 500+ x 500+ corner carries the largest
    detail flow (large firms dominate both ends of the ladder)."""
    us = j2jod[
        (j2jod["year"] == 2015)
        & (j2jod["quarter"] == 3)
        & (j2jod["firmsize"] > 0)
        & (j2jod["firmsize_orig"] > 0)
    ]
    top = us.loc[us["EE"].idxmax()]
    assert top["firmsize"] == 5 and top["firmsize_orig"] == 5


@pytest.mark.parametrize(
    "loader",
    [
        targets.load_susb_sector_size,
        targets.load_bds_firm_size,
        targets.load_qwi_firmsize_sector,
        targets.load_j2j_firmsize_sector,
        targets.load_j2j_sexage,
        targets.load_j2jod_firmsize,
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


# ---------------------------------------------------------------
# The archived LED response and the content-based integrity pin
# ---------------------------------------------------------------


def _fetch_module():
    """Import the fetch script (scripts/ is not an installed package)."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fetch_employer_firm_targets.py"
    )
    spec = importlib.util.spec_from_file_location("_fetch_eft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_led_j2jod_archive_is_committed():
    """The only-ever detail window must not depend on the live tool.

    The LED Extraction Tool re-runs the query against whatever
    release is current, and the 2015Q1-2016Q1 origin x destination
    detail is the only window in which this cross is published at
    all. If the archive goes missing, that window is unrecoverable.
    """
    fetch = _fetch_module()
    assert fetch.LED_J2JOD_ARCHIVE.exists()


def test_led_j2jod_content_digest_matches_the_pin():
    fetch = _fetch_module()
    digest = fetch.led_j2jod_content_digest(fetch.LED_J2JOD_ARCHIVE)
    assert digest == fetch.LED_J2JOD_CONTENT_SHA256


def test_content_digest_ignores_column_order_but_not_values(tmp_path):
    """The property that makes the pin worth having.

    The byte digest originally pinned on this extract broke within
    six days purely because the tool reordered its measure columns,
    with all 1,476 values unchanged. A pin that fires on cosmetic
    reordering trains the maintainer to re-pin on sight, so a real
    revision arrives indistinguishable from the false alarms.
    """
    import pandas as pd

    fetch = _fetch_module()
    frame = pd.read_csv(fetch.LED_J2JOD_ARCHIVE, low_memory=False)

    shuffled = tmp_path / "shuffled.csv"
    frame[list(reversed(frame.columns))].to_csv(shuffled, index=False)
    assert (
        fetch.led_j2jod_content_digest(shuffled)
        == fetch.LED_J2JOD_CONTENT_SHA256
    )

    revised = tmp_path / "revised.csv"
    bumped = frame.copy()
    bumped.loc[bumped.index[0], "EE"] += 1
    bumped.to_csv(revised, index=False)
    assert (
        fetch.led_j2jod_content_digest(revised)
        != fetch.LED_J2JOD_CONTENT_SHA256
    )


def test_committed_extract_rebuilds_from_the_archive(tmp_path):
    """Byte-equality of the committed extract from the archived raw."""
    import pandas as pd

    fetch = _fetch_module()
    raw = pd.read_csv(fetch.fetch_led_j2jod(tmp_path), low_memory=False)
    keep = raw[raw["year"] >= fetch.LEHD_START_YEAR].copy()
    id_cols = ["year", "quarter", "firmsize_orig", "firmsize"]
    flag_cols = [f"s{m}" for m in fetch.J2JOD_MEASURES]
    out = keep[id_cols + fetch.J2JOD_MEASURES + flag_cols].copy()
    out.insert(
        4, "firmsize_orig_label", fetch._firmsize_label(out["firmsize_orig"])
    )
    out.insert(5, "firmsize_label", fetch._firmsize_label(out["firmsize"]))
    out = out.sort_values(id_cols).reset_index(drop=True)

    rebuilt = tmp_path / "rebuilt.csv"
    out.to_csv(rebuilt, index=False, float_format="%.10g")
    committed = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "external"
        / "j2jod_us_firmsize_od_2015on.csv"
    )
    assert rebuilt.read_text() == committed.read_text()
