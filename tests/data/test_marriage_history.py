"""Tests for the Marriage History reader (mh85_23).

Two tiers, matching the rest of the data suite: an always-runnable tier
that builds a miniature synthetic MH product (so it exercises the exact
label-verification and colspec path with no staged files) and tests the
pure episode helper, plus an integration tier that reads the real staged
file and pins its shape, domains, keys, and join coverage. The
integration tier skips when the PSID files are absent (e.g. in CI).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import family, marriage

from .psid_fixtures import write_product

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_marriage = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23" / "MH85_23.txt").is_file(),
    reason="PSID marriage history file (mh85_23) not staged",
)
needs_earnings = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files (earnings panel) not staged",
)

# Six synthetic MH records exercising every code path: a never-married
# placeholder, an intact marriage to a joinable spouse, a divorce with a
# non-sample spouse (MH8 in 800-995), a widowhood, an NA-order/DK-year
# record, and a separation. Columns are the full 20-variable layout so
# label verification sees the whole file.
_MH_ROWS: list[tuple[str, int, str, list[int]]] = [
    ("MH1", 1, "RELEASE NUMBER", [2, 2, 2, 2, 2, 2]),
    ("MH2", 4, "1968 INTERVIEW NUMBER OF INDIVIDUAL", [7, 7, 9, 9, 11, 13]),
    ("MH3", 3, "PERSON NUMBER OF INDIVIDUAL", [1, 2, 1, 1, 3, 2]),
    ("MH4", 1, "SEX OF INDIVIDUAL", [1, 2, 1, 1, 2, 2]),
    ("MH5", 2, "MONTH INDIVIDUAL BORN", [6, 3, 1, 1, 98, 4]),
    ("MH6", 4, "YEAR INDIVIDUAL BORN", [1950, 1955, 1948, 1948, 9998, 1965]),
    ("MH7", 4, "1968 INTERVIEW NUMBER OF SPOUSE", [0, 8, 9, 10, 12, 14]),
    ("MH8", 3, "PERSON NUMBER OF SPOUSE", [0, 1, 800, 5, 7, 2]),
    ("MH9", 2, "ORDER OF THIS MARRIAGE", [99, 1, 2, 1, 98, 1]),
    ("MH10", 2, "MONTH MARRIED", [99, 6, 4, 7, 98, 5]),
    ("MH11", 4, "YEAR MARRIED", [9999, 1980, 1975, 1968, 9998, 1990]),
    ("MH12", 1, "STATUS OF THIS MARRIAGE", [9, 1, 4, 3, 8, 5]),
    ("MH13", 2, "MONTH WIDOWED OR DIVORCED", [99, 99, 8, 2, 98, 99]),
    (
        "MH14",
        4,
        "YEAR WIDOWED OR DIVORCED",
        [9999, 9999, 1990, 1972, 9998, 9999],
    ),
    ("MH15", 2, "MONTH SEPARATED", [99, 99, 3, 99, 98, 6]),
    ("MH16", 4, "YEAR SEPARATED", [9999, 9999, 1988, 9999, 9998, 1995]),
    (
        "MH17",
        4,
        "YEAR MOST RECENTLY REPORTED MARRIAGE",
        [2019, 2021, 2005, 2005, 1999, 2011],
    ),
    (
        "MH18",
        2,
        "NUMBER OF MARRIAGES OF THIS INDIVIDUAL",
        [0, 1, 2, 2, 98, 1],
    ),
    ("MH19", 1, "LAST KNOWN MARITAL STATUS", [2, 1, 4, 4, 8, 5]),
    ("MH20", 2, "NUMBER OF MARRIAGE RECORDS", [1, 1, 2, 2, 1, 1]),
]


@pytest.fixture
def mini_marriage_dir(tmp_path: Path) -> Path:
    write_product(tmp_path / "mh85_23", "MH85_23.sps", "MH85_23.txt", _MH_ROWS)
    return tmp_path


# --- always-runnable: synthetic product + pure helpers ----------------


def test_build_person_ids_and_spouse_joinability(mini_marriage_dir: Path):
    mh = marriage.marriage_history(data_dir=mini_marriage_dir)
    assert list(mh["person_id"]) == [7001, 7002, 9001, 9001, 11003, 13002]
    # Spouse id only for joinable PSID persons; 0/0 (never married) and
    # MH8==800 (non-sample spouse) become NA.
    spouse = mh["spouse_person_id"]
    assert pd.isna(spouse.iloc[0])  # never married, MH7/MH8 == 0
    assert spouse.iloc[1] == 8001
    assert pd.isna(spouse.iloc[2])  # MH8 == 800, non-sample spouse
    assert spouse.iloc[3] == 10005
    assert spouse.iloc[4] == 12007
    assert mh["spouse_person_id"].dtype == "Int64"


def test_build_decodes_status_and_flags_placeholder(mini_marriage_dir: Path):
    mh = marriage.marriage_history(data_dir=mini_marriage_dir)
    assert list(mh["how_ended"]) == [
        "never_married",
        "intact",
        "divorce",
        "widowhood",
        "unknown",
        "separated",
    ]
    assert list(mh["is_marriage"]) == [False, True, True, True, True, True]
    assert list(mh["last_known_status"]) == [
        "never_married",
        "married",
        "divorced",
        "divorced",
        "unknown",
        "separated",
    ]
    assert list(mh["sex"]) == [
        "male",
        "female",
        "male",
        "male",
        "female",
        "female",
    ]


def test_build_maps_missing_sentinels_to_na(mini_marriage_dir: Path):
    mh = marriage.marriage_history(data_dir=mini_marriage_dir)
    # Never-married placeholder: order 99, year 9999, month 99 -> NA.
    assert pd.isna(mh["marriage_order"].iloc[0])
    assert pd.isna(mh["start_year"].iloc[0])
    assert pd.isna(mh["start_month"].iloc[0])
    # NA-order (98) and DK year (9998) row.
    assert pd.isna(mh["marriage_order"].iloc[4])
    assert pd.isna(mh["start_year"].iloc[4])
    assert pd.isna(mh["birth_year"].iloc[4])  # 9998
    # Real values survive.
    assert list(mh["marriage_order"].dropna()) == [1, 2, 1, 1]
    assert mh["start_year"].iloc[1] == 1980
    assert mh["separation_year"].iloc[2] == 1988
    assert mh["end_year"].iloc[2] == 1990


def test_marriage_episodes_coalesces_end_and_orders():
    """Pure helper on synthetic rows: end-year coalescing + ordering."""
    records = pd.DataFrame(
        {
            "person_id": [900, 900, 700, 130, 110],
            "marriage_order": pd.array([2, 1, 1, 1, pd.NA], dtype="Int64"),
            "start_year": pd.array(
                [1975, 1968, 1980, 1990, 2000], dtype="Int64"
            ),
            "start_month": pd.array([4, 7, 6, 5, 1], dtype="Int64"),
            "end_year": pd.array(
                [1990, 1972, pd.NA, pd.NA, pd.NA], dtype="Int64"
            ),
            "separation_year": pd.array(
                [pd.NA, pd.NA, pd.NA, 1995, pd.NA], dtype="Int64"
            ),
            "how_ended": [
                "divorce",
                "widowhood",
                "intact",
                "separated",
                "unknown",
            ],
            "spouse_person_id": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "last_known_status": [
                "divorced",
                "divorced",
                "married",
                "separated",
                "unknown",
            ],
            "is_marriage": [True, True, True, True, True],
        }
    )
    ep = marriage.marriage_episodes(records)
    # Ordered by (person_id, marriage_order); NA order sorts last.
    assert list(ep["person_id"]) == [110, 130, 700, 900, 900]
    by = ep.set_index(["person_id", "marriage_order"], drop=False)
    # Divorce/widowhood use end_year; separated uses separation_year;
    # intact and unknown have no episode end.
    assert by.loc[(900, 1), "episode_end_year"] == 1972
    assert by.loc[(900, 2), "episode_end_year"] == 1990
    assert by.loc[(130, 1), "episode_end_year"] == 1995  # separated
    assert pd.isna(by.loc[(700, 1), "episode_end_year"])  # intact
    assert by.loc[(900, 1), "episode_duration_years"] == 4
    assert by.loc[(130, 1), "episode_duration_years"] == 5


def test_marriage_episodes_excludes_placeholders():
    records = pd.DataFrame(
        {
            "person_id": [1, 2],
            "marriage_order": pd.array([pd.NA, 1], dtype="Int64"),
            "start_year": pd.array([pd.NA, 1990], dtype="Int64"),
            "start_month": pd.array([pd.NA, 6], dtype="Int64"),
            "end_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "separation_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "how_ended": ["never_married", "intact"],
            "spouse_person_id": pd.array([pd.NA, 9], dtype="Int64"),
            "last_known_status": ["never_married", "married"],
            "is_marriage": [False, True],
        }
    )
    ep = marriage.marriage_episodes(records)
    assert list(ep["person_id"]) == [2]


def test_label_mismatch_raises(mini_marriage_dir: Path):
    sps = mini_marriage_dir / "mh85_23" / "MH85_23.sps"
    sps.write_text(
        sps.read_text().replace(
            "STATUS OF THIS MARRIAGE", "STATE OF THIS MARRIAGE"
        )
    )
    with pytest.raises(ValueError, match="does not match"):
        marriage.marriage_history(data_dir=mini_marriage_dir)


def test_undocumented_status_code_raises(tmp_path: Path):
    rows = [list(f) for f in _MH_ROWS]
    for field in rows:
        if field[0] == "MH12":
            # Code 6 is not in the documented MH12 domain.
            field[3] = [9, 1, 6, 3, 8, 5]
    write_product(
        tmp_path / "mh85_23",
        "MH85_23.sps",
        "MH85_23.txt",
        [tuple(f) for f in rows],
    )
    with pytest.raises(ValueError, match="undocumented code"):
        marriage.marriage_history(data_dir=tmp_path)


# --- integration: real staged mh85_23 ---------------------------------

# Pinned on the staged Release 2 file (December 2025), verified
# 2026-07-07.
_N_RECORDS = 65_226
_N_PERSONS = 55_732
_N_MARRIAGES = 42_666
_N_PLACEHOLDERS = 22_560
_N_JOINABLE_SPOUSE = 32_522
_N_REAL_ORDER = 42_053


@needs_marriage
def test_real_shape_and_record_pins():
    mh = marriage.marriage_history()
    assert len(mh) == _N_RECORDS
    assert mh["person_id"].nunique() == _N_PERSONS
    assert int(mh["is_marriage"].sum()) == _N_MARRIAGES
    assert int((~mh["is_marriage"]).sum()) == _N_PLACEHOLDERS
    assert int(mh["spouse_person_id"].notna().sum()) == _N_JOINABLE_SPOUSE
    assert int(mh["marriage_order"].notna().sum()) == _N_REAL_ORDER


@needs_marriage
def test_real_code_domains_are_documented():
    mh = marriage.marriage_history()
    assert set(mh["how_ended"].dropna()) == {
        "intact",
        "widowhood",
        "divorce",
        "separated",
        "other",
        "unknown",
        "never_married",
    }
    assert set(mh["last_known_status"].dropna()) == {
        "married",
        "never_married",
        "widowed",
        "divorced",
        "separated",
        "unknown",
    }
    assert set(mh["sex"].dropna()) == {"male", "female"}
    assert int(mh["marriage_order"].min()) == 1
    assert int(mh["marriage_order"].max()) == 13


@needs_marriage
def test_real_keys_and_chronology():
    mh = marriage.marriage_history()
    real = mh[mh["marriage_order"].notna()]
    assert not real.duplicated(["person_id", "marriage_order"]).any()
    ep = marriage.marriage_episodes(mh)
    both = ep.dropna(subset=["start_year", "episode_end_year"])
    assert (both["episode_end_year"] >= both["start_year"]).all()
    # Real marriage years fall in a sane window.
    assert int(mh["start_year"].min()) >= 1900
    assert int(mh["start_year"].max()) <= 2023


@needs_marriage
@needs_earnings
def test_real_join_coverage_vs_earnings_panel():
    """Share of earnings-panel persons carrying a marriage record.

    Pinned at 0.911 (any record) / 0.712 (an actual marriage) on the
    2026-07-07 staged data; tolerance absorbs a future PSID refresh.
    """
    earn_persons = set(family.family_earnings_panel().person_id.unique())
    mh = marriage.marriage_history()
    any_share = len(earn_persons & set(mh["person_id"])) / len(earn_persons)
    real_share = len(
        earn_persons & set(mh[mh["is_marriage"]]["person_id"])
    ) / len(earn_persons)
    assert any_share == pytest.approx(0.911, abs=0.03)
    assert real_share == pytest.approx(0.712, abs=0.03)
