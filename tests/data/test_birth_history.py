"""Tests for the Childbirth & Adoption History reader (cah85_23).

Always-runnable tier: a miniature synthetic CAH product plus the pure
``birth_events`` helper. Integration tier: the real staged file, pinned
on shape, domains, keys, and earnings-panel join coverage; skipped when
the PSID files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import births, family

from .psid_fixtures import write_product

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_births = pytest.mark.skipif(
    not (REAL_DATA / "cah85_23" / "CAH85_23.txt").is_file(),
    reason="PSID childbirth/adoption file (cah85_23) not staged",
)
needs_earnings = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files (earnings panel) not staged",
)

# Five synthetic CAH records: a joinable-child birth, a non-sample-child
# birth of a deceased child, an adoption with an adoptive relationship, a
# no-children denial placeholder, and a DK-date birth. The full read-set
# of variables is present so label verification sees the whole layout.
_CAH_ROWS: list[tuple[str, int, str, list[int]]] = [
    ("CAH1", 1, "RELEASE NUMBER", [2, 2, 2, 2, 2]),
    ("CAH2", 1, "RECORD TYPE", [1, 1, 2, 1, 1]),
    (
        "CAH3",
        4,
        "1968 INTERVIEW NUMBER OF PARENT",
        [20, 20, 20, 30, 40],
    ),
    ("CAH4", 3, "PERSON NUMBER OF PARENT", [1, 1, 1, 2, 3]),
    ("CAH5", 1, "SEX OF PARENT", [2, 2, 2, 1, 2]),
    ("CAH6", 2, "MONTH PARENT BORN", [5, 5, 5, 98, 7]),
    (
        "CAH7",
        4,
        "YEAR PARENT BORN",
        [1960, 1960, 1960, 9998, 1970],
    ),
    (
        "CAH8",
        1,
        "MARITAL STATUS OF MOTHER WHEN IND BORN",
        [1, 1, 9, 9, 8],
    ),
    ("CAH9", 2, "BIRTH ORDER", [1, 2, 99, 99, 98]),
    (
        "CAH10",
        4,
        "1968 INTERVIEW NUMBER OF CHILD",
        [20, 20, 20, 0, 40],
    ),
    ("CAH11", 3, "PERSON NUMBER OF CHILD", [3, 800, 5, 0, 4]),
    ("CAH12", 1, "OS1. SEX OF CHILD", [1, 2, 1, 9, 8]),
    ("CAH13", 2, "OS2. MONTH CHILD BORN", [6, 8, 3, 99, 98]),
    (
        "CAH15",
        4,
        "OS2. YEAR CHILD BORN",
        [1985, 1990, 2000, 9999, 9998],
    ),
    (
        "CAH24",
        2,
        "OS5. WHERE CHILD WAS WHEN LAST REPORTED",
        [1, 6, 1, 99, 98],
    ),
    (
        "CAH25",
        2,
        "OS6. MONTH CHILD MOVED OUT OR DIED",
        [99, 4, 99, 99, 98],
    ),
    (
        "CAH26",
        4,
        "OS6. YEAR CHILD MOVED OUT OR DIED",
        [9999, 2015, 9999, 9999, 9998],
    ),
    (
        "CAH114",
        4,
        "YR MOST RECENTLY REPORTED NUMBER OF KIDS",
        [2019, 2019, 2019, 2015, 2013],
    ),
    (
        "CAH115",
        4,
        "YEAR MOST RECENTLY REPORTED THIS CHILD",
        [2019, 2019, 2019, 9999, 2013],
    ),
    (
        "CAH116",
        2,
        "NUMBER OF NATURAL OR ADOPTED CHILDREN",
        [2, 2, 2, 0, 98],
    ),
    (
        "CAH117",
        2,
        "RELATIONSHIP TO ADOPTIVE PARENT",
        [99, 99, 33, 99, 99],
    ),
    (
        "CAH118",
        2,
        "NUMBER OF BIRTH OR ADOPTION RECORDS",
        [3, 3, 3, 1, 1],
    ),
]


@pytest.fixture
def mini_birth_dir(tmp_path: Path) -> Path:
    write_product(
        tmp_path / "cah85_23", "CAH85_23.sps", "CAH85_23.txt", _CAH_ROWS
    )
    return tmp_path


# --- always-runnable --------------------------------------------------


def test_build_person_ids_and_child_joinability(mini_birth_dir: Path):
    bh = births.birth_history(data_dir=mini_birth_dir)
    assert list(bh["parent_person_id"]) == [
        20001,
        20001,
        20001,
        30002,
        40003,
    ]
    child = bh["child_person_id"]
    assert child.iloc[0] == 20003
    assert pd.isna(child.iloc[1])  # CAH11 == 800, non-sample child
    assert child.iloc[2] == 20005
    assert pd.isna(child.iloc[3])  # no child (denial placeholder)
    assert child.iloc[4] == 40004
    assert bh["child_person_id"].dtype == "Int64"


def test_build_decodes_and_flags_events(mini_birth_dir: Path):
    bh = births.birth_history(data_dir=mini_birth_dir)
    assert list(bh["record_type"]) == [
        "birth",
        "birth",
        "adoption",
        "birth",
        "birth",
    ]
    assert list(bh["is_event"]) == [True, True, True, False, True]
    # Deceased child surfaces via CAH24 code 6.
    assert bh["where_child_last_reported"].iloc[1] == "deceased"
    assert bh["moved_out_or_died_year"].iloc[1] == 2015
    # Adoptive relationship kept only on the adoption record.
    assert bh["adoptive_relationship_code"].iloc[2] == 33
    assert pd.isna(bh["adoptive_relationship_code"].iloc[0])


def test_build_maps_missing_sentinels_to_na(mini_birth_dir: Path):
    bh = births.birth_history(data_dir=mini_birth_dir)
    # Denial placeholder: no child sex, no birth date/order.
    assert pd.isna(bh["child_sex"].iloc[3])
    assert pd.isna(bh["birth_year"].iloc[3])
    assert pd.isna(bh["birth_order"].iloc[3])
    assert pd.isna(bh["mother_marital_status_at_birth"].iloc[3])
    # DK-date birth row.
    assert pd.isna(bh["birth_year"].iloc[4])  # 9998
    assert pd.isna(bh["child_sex"].iloc[4])  # CAH12 == 8
    assert bh["mother_marital_status_at_birth"].iloc[4] == "unknown"
    assert pd.isna(bh["parent_birth_year"].iloc[3])  # 9998
    # Real values survive.
    assert list(bh["birth_year"].dropna()) == [1985, 1990, 2000]
    assert list(bh["birth_order"].dropna()) == [1, 2]
    assert list(bh["record_type"].map({"birth": 1, "adoption": 2})) == [
        1,
        1,
        2,
        1,
        1,
    ]


def test_birth_events_orders_and_excludes_placeholders():
    """Pure helper on synthetic rows: order by parent then birth date."""
    records = pd.DataFrame(
        {
            "parent_person_id": [20001, 20001, 20001, 40003, 30002],
            "birth_year": pd.array(
                [2000, 1985, 1990, pd.NA, pd.NA], dtype="Int64"
            ),
            "birth_order": pd.array([pd.NA, 1, 2, 1, pd.NA], dtype="Int64"),
            "record_type": [
                "adoption",
                "birth",
                "birth",
                "birth",
                "birth",
            ],
            "is_event": [True, True, True, True, False],
        }
    )
    ev = births.birth_events(records)
    # Placeholder (is_event False) dropped; parent 20001 ordered by year.
    assert list(ev["parent_person_id"]) == [20001, 20001, 20001, 40003]
    assert list(ev["birth_year"].dropna()) == [1985, 1990, 2000]
    assert len(ev) == 4


def test_label_mismatch_raises(mini_birth_dir: Path):
    sps = mini_birth_dir / "cah85_23" / "CAH85_23.sps"
    sps.write_text(sps.read_text().replace("RECORD TYPE", "RECORD KIND"))
    with pytest.raises(ValueError, match="does not match"):
        births.birth_history(data_dir=mini_birth_dir)


def test_undocumented_where_code_raises(tmp_path: Path):
    rows = [list(f) for f in _CAH_ROWS]
    for field in rows:
        if field[0] == "CAH24":
            # Code 55 is not in the documented CAH24 domain.
            field[3] = [1, 55, 1, 99, 98]
    write_product(
        tmp_path / "cah85_23",
        "CAH85_23.sps",
        "CAH85_23.txt",
        [tuple(f) for f in rows],
    )
    with pytest.raises(ValueError, match="undocumented code"):
        births.birth_history(data_dir=tmp_path)


# --- integration: real staged cah85_23 --------------------------------

# Pinned on the staged Release 2 file (December 2025), verified
# 2026-07-07.
_N_RECORDS = 148_739
_N_PARENTS = 55_567
_N_EVENTS = 90_065
_N_PLACEHOLDERS = 58_674
_N_JOINABLE_CHILD = 67_819
_N_CHILD_BIRTH = 66_776
_N_CHILD_ADOPTION = 1_043
_N_DECEASED = 2_154


@needs_births
def test_real_shape_and_record_pins():
    bh = births.birth_history()
    assert len(bh) == _N_RECORDS
    assert bh["parent_person_id"].nunique() == _N_PARENTS
    assert int(bh["is_event"].sum()) == _N_EVENTS
    assert int((~bh["is_event"]).sum()) == _N_PLACEHOLDERS
    assert int(bh["child_person_id"].notna().sum()) == _N_JOINABLE_CHILD
    by_type = bh[bh["child_person_id"].notna()]["record_type"].value_counts()
    assert int(by_type["birth"]) == _N_CHILD_BIRTH
    assert int(by_type["adoption"]) == _N_CHILD_ADOPTION


@needs_births
def test_real_code_domains_are_documented():
    bh = births.birth_history()
    assert set(bh["record_type"].dropna()) == {"birth", "adoption"}
    assert set(bh["child_sex"].dropna()) == {"male", "female"}
    assert set(bh["parent_sex"].dropna()) == {"male", "female"}
    assert set(bh["mother_marital_status_at_birth"].dropna()) <= set(
        births.MOTHER_MARITAL_STATUS.values()
    )
    assert set(bh["where_child_last_reported"].dropna()) <= set(
        births.WHERE_CHILD_LAST_REPORTED.values()
    )
    observed = set(bh["adoptive_relationship_code"].dropna().astype(int))
    assert observed <= set(births.ADOPTIVE_RELATIONSHIP)
    # Deceased children are the survivorship anchor.
    assert (
        int((bh["where_child_last_reported"] == "deceased").sum())
        == _N_DECEASED
    )
    assert int(bh["birth_order"].min()) == 1
    assert int(bh["birth_order"].max()) == 18


@needs_births
def test_real_no_duplicate_parent_child_event():
    bh = births.birth_history()
    events = bh[bh["child_person_id"].notna()]
    assert not events.duplicated(
        ["parent_person_id", "child_person_id", "record_type"]
    ).any()
    # Birth years fall in a sane window.
    assert int(bh["birth_year"].min()) >= 1900
    assert int(bh["birth_year"].max()) <= 2023


@needs_births
@needs_earnings
def test_real_join_coverage_vs_earnings_panel():
    """Share of earnings-panel persons who appear as a parent.

    Pinned at 0.911 (any record) / 0.726 (an actual event) on the
    2026-07-07 staged data; tolerance absorbs a future PSID refresh.
    """
    earn_persons = set(family.family_earnings_panel().person_id.unique())
    bh = births.birth_history()
    any_share = len(earn_persons & set(bh["parent_person_id"])) / len(
        earn_persons
    )
    event_share = len(
        earn_persons & set(bh[bh["is_event"]]["parent_person_id"])
    ) / len(earn_persons)
    assert any_share == pytest.approx(0.911, abs=0.03)
    assert event_share == pytest.approx(0.726, abs=0.03)
