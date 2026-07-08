"""Tests for the marital-transition + fertility panel construction.

Two tiers, matching the rest of the data suite. An always-runnable tier
exercises the pure construction logic on synthetic
:func:`populace_dynamics.data.marriage.marriage_history`-shaped frames
(state machine, discrete-time hazard convention, occupancy, ASFR), and an
integration tier reads the real staged history files and pins the panel's
shape. The integration tier skips when the PSID files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import marriage, transitions

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_marriage = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23" / "MH85_23.txt").is_file(),
    reason="PSID marriage history file (mh85_23) not staged",
)
needs_births = pytest.mark.skipif(
    not (REAL_DATA / "cah85_23" / "CAH85_23.txt").is_file(),
    reason="PSID childbirth history file (cah85_23) not staged",
)
needs_ind = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID individual file (ind2023er) not staged",
)

# Full marriage_history column set, so marriage.marriage_episodes runs.
_MH_COLUMNS = [
    "person_id",
    "sex",
    "birth_year",
    "birth_month",
    "marriage_order",
    "spouse_person_id",
    "start_year",
    "start_month",
    "end_year",
    "end_month",
    "separation_year",
    "separation_month",
    "how_ended",
    "last_known_status",
    "most_recent_report_year",
    "n_marriages",
    "n_records",
    "is_marriage",
]
_INT = "Int64"


def _mh(rows: list[dict]) -> pd.DataFrame:
    """Build a marriage_history-shaped frame from partial row dicts."""
    filled = []
    for r in rows:
        rec = {c: r.get(c) for c in _MH_COLUMNS}
        rec.setdefault("birth_month", pd.NA)
        rec.setdefault("start_month", pd.NA)
        rec.setdefault("end_month", pd.NA)
        rec.setdefault("separation_month", pd.NA)
        rec.setdefault("separation_year", pd.NA)
        rec.setdefault("spouse_person_id", pd.NA)
        rec.setdefault("n_records", 1)
        filled.append(rec)
    df = pd.DataFrame(filled, columns=_MH_COLUMNS)
    for col in (
        "birth_year",
        "marriage_order",
        "start_year",
        "end_year",
        "separation_year",
        "n_marriages",
        "most_recent_report_year",
        "spouse_person_id",
    ):
        df[col] = df[col].astype(_INT)
    df["sex"] = df["sex"].astype("string")
    df["how_ended"] = df["how_ended"].astype("string")
    df["last_known_status"] = df["last_known_status"].astype("string")
    df["is_marriage"] = df["is_marriage"].astype(bool)
    return df


def _deaths(mapping: dict[int, int | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": list(mapping),
            "death_year": pd.array(list(mapping.values()), dtype=_INT),
        }
    )


# Four persons spanning every path: never-married; one intact marriage; a
# marry->divorce->remarry->widowhood career; and a widowhood ended by the
# person's own death.
_ROWS = [
    dict(
        person_id=1,
        sex="female",
        birth_year=1970,
        how_ended="never_married",
        last_known_status="never_married",
        most_recent_report_year=2020,
        n_marriages=0,
        is_marriage=False,
    ),
    dict(
        person_id=2,
        sex="male",
        birth_year=1960,
        marriage_order=1,
        start_year=1985,
        how_ended="intact",
        last_known_status="married",
        most_recent_report_year=2015,
        n_marriages=1,
        is_marriage=True,
    ),
    dict(
        person_id=3,
        sex="female",
        birth_year=1955,
        marriage_order=1,
        start_year=1980,
        end_year=1990,
        how_ended="divorce",
        last_known_status="widowed",
        most_recent_report_year=2018,
        n_marriages=2,
        is_marriage=True,
    ),
    dict(
        person_id=3,
        sex="female",
        birth_year=1955,
        marriage_order=2,
        start_year=1995,
        end_year=2005,
        how_ended="widowhood",
        last_known_status="widowed",
        most_recent_report_year=2018,
        n_marriages=2,
        is_marriage=True,
    ),
    dict(
        person_id=4,
        sex="male",
        birth_year=1950,
        marriage_order=1,
        start_year=1975,
        end_year=1985,
        how_ended="widowhood",
        last_known_status="widowed",
        most_recent_report_year=2010,
        n_marriages=1,
        is_marriage=True,
    ),
]
_WEIGHTS = pd.Series({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})


# --- always-runnable: person attributes / censoring -------------------
def test_person_attributes_censoring():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: 2000})
    attrs = transitions.person_attributes(records, deaths, _WEIGHTS)
    by = attrs.set_index("person_id")
    # censor_year = min(most_recent_report_year, death, MAX_YEAR).
    assert by.loc[1, "censor_year"] == 2020
    assert by.loc[2, "censor_year"] == 2015
    assert by.loc[4, "censor_year"] == 2000  # own death precedes MRR 2010
    assert by.loc[1, "start_exposure_year"] == 1970 + transitions.START_AGE
    assert by.loc[3, "n_marriages"] == 2


# --- always-runnable: state machine + transitions ---------------------
def test_marital_state_machine_and_transitions():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: 2000})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    py = panel.person_years

    def state(pid, year):
        row = py[(py.person_id == pid) & (py.year == year)]
        return row.marital_state.iloc[0]

    # Person 1 never marries: never_married throughout.
    assert set(py[py.person_id == 1].marital_state) == {"never_married"}
    # Person 2: never_married up to and including the marriage year (the
    # transition is IN 1985; the pre-transition state holds at t), married
    # from 1986.
    assert state(2, 1984) == "never_married"
    assert state(2, 1985) == "never_married"
    assert state(2, 1986) == "married"
    # Person 3: married 1980 (from 1981), divorced 1990 (from 1991),
    # married again 1995 (from 1996), widowed 2005 (from 2006).
    assert state(3, 1985) == "married"
    assert state(3, 1992) == "divorced"
    assert state(3, 2000) == "married"
    assert state(3, 2008) == "widowed"

    ev = panel.events.set_index(["person_id", "transition"])
    # First marriages at the order-1 start, aged from birth.
    assert ev.loc[(2, "first_marriage"), "year"] == 1985
    assert ev.loc[(2, "first_marriage"), "age"] == 25
    # Divorce dated at the marriage end, carrying its duration.
    assert ev.loc[(3, "divorce"), "year"] == 1990
    assert ev.loc[(3, "divorce"), "marriage_duration"] == 10
    # Remarriage carries years since the prior dissolution (1995 - 1990).
    assert ev.loc[(3, "remarriage"), "year"] == 1995
    assert ev.loc[(3, "remarriage"), "years_since_dissolution"] == 5
    # Widowhood dated at the marriage end, aged from birth.
    assert ev.loc[(3, "widowhood"), "year"] == 2005
    assert ev.loc[(3, "widowhood"), "age"] == 50


def test_duration_and_ysd_columns_track_state():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: 2000})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    py = panel.person_years.set_index(["person_id", "year"])
    # Person 3 married 1980: duration at 1985 is 5 (1985 - 1980).
    assert py.loc[(3, 1985), "marriage_duration"] == 5
    assert pd.isna(py.loc[(3, 1985), "years_since_dissolution"])
    # After the 1990 divorce: ysd at 1993 is 3, no marriage duration.
    assert py.loc[(3, 1993), "years_since_dissolution"] == 3
    assert pd.isna(py.loc[(3, 1993), "marriage_duration"])


def test_own_death_censors_exposure():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: 2000})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    p4 = panel.person_years[panel.person_years.person_id == 4]
    assert int(p4.year.max()) == 2000  # dies 2000, no person-years after


def test_require_weight_drops_zero_weight_persons():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: None})
    weights = pd.Series({1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0})
    kept = transitions.build_marital_panel(records, deaths, weights)
    assert 1 not in set(kept.attrs.person_id)
    everyone = transitions.build_marital_panel(
        records, deaths, weights, require_weight=False
    )
    assert 1 in set(everyone.attrs.person_id)


def test_hazard_cells_hand_computed_first_marriage():
    """First-marriage hazard = weighted events / never-married exposure."""
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: None})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    cells = transitions.hazard_cells(panel, weighted=True)
    # Persons 2 and 4 (male) each marry at age 25. With the discrete-time
    # convention the marriage year is their only never-married person-year
    # in the 25-34 band, so both the numerator (2 events) and the band
    # exposure (2 person-years) are 2 -> a degenerate hazard of 1.0.
    cell = cells["first_marriage.25-34|male"]
    assert cell["n_events"] == 2
    assert cell["rate"] == pytest.approx(1.0)
    # Their never-married 18-24 person-years carry no first marriage.
    assert cells["first_marriage.18-24|male"]["n_events"] == 0
    assert cells["first_marriage.18-24|male"]["rate"] == 0.0
    # No 45+ female first marriage in the fixture.
    assert cells["first_marriage.45+|female"]["n_events"] == 0


def test_occupancy_ever_married_and_mean_marriages():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: None})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    occ = transitions.occupancy_cells(panel, weighted=True)
    # Persons observed to 40: all four (censor >= birth+40). Ever married
    # by 40: persons 2,3,4 (married in their 20s), person 1 never -> 3/4
    # female? by sex: females = {1 (never), 3 (married)} -> 1/2 = 0.5.
    assert occ["ever_married_by_40|female"]["rate"] == pytest.approx(0.5)
    # Males = {2, 4}, both married by 40 -> 1.0.
    assert occ["ever_married_by_40|male"]["rate"] == pytest.approx(1.0)
    # Mean lifetime marriages among ever-married females (person 3): 2.
    assert occ["mean_lifetime_marriages|female"]["rate"] == pytest.approx(2.0)


def test_unweighted_matches_weighted_when_weights_equal():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: None})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    w = transitions.hazard_cells(panel, weighted=True)
    u = transitions.hazard_cells(panel, weighted=False)
    # All weights are 1.0, so weighted and unweighted rates coincide.
    for key in w:
        assert w[key]["rate"] == pytest.approx(u[key]["rate"]), key


# --- always-runnable: fertility ---------------------------------------
_BH_COLUMNS = [
    "parent_person_id",
    "parent_sex",
    "parent_birth_year",
    "record_type",
    "child_person_id",
    "child_sex",
    "birth_year",
    "birth_order",
    "is_event",
]


def _bh(rows: list[dict]) -> pd.DataFrame:
    filled = []
    for r in rows:
        rec = {c: r.get(c) for c in _BH_COLUMNS}
        rec.setdefault("child_person_id", pd.NA)
        rec.setdefault("child_sex", pd.NA)
        rec.setdefault("birth_order", pd.NA)
        filled.append(rec)
    df = pd.DataFrame(filled, columns=_BH_COLUMNS)
    for col in (
        "parent_birth_year",
        "child_person_id",
        "birth_year",
        "birth_order",
    ):
        df[col] = df[col].astype(_INT)
    df["record_type"] = df["record_type"].astype("string")
    df["parent_sex"] = df["parent_sex"].astype("string")
    df["is_event"] = df["is_event"].astype(bool)
    return df


def test_fertility_asfr_and_completed():
    records = _mh(_ROWS)
    deaths = _deaths({1: None, 2: None, 3: None, 4: None})
    panel = transitions.build_marital_panel(records, deaths, _WEIGHTS)
    # Woman 3 (born 1955) has two births at ages 25 (1980) and 30 (1985);
    # woman 1 (born 1970) has one birth at age 22 (1992).
    bh = _bh(
        [
            dict(
                parent_person_id=3,
                parent_sex="female",
                parent_birth_year=1955,
                record_type="birth",
                birth_year=1980,
                is_event=True,
            ),
            dict(
                parent_person_id=3,
                parent_sex="female",
                parent_birth_year=1955,
                record_type="birth",
                birth_year=1985,
                is_event=True,
            ),
            dict(
                parent_person_id=1,
                parent_sex="female",
                parent_birth_year=1970,
                record_type="birth",
                birth_year=1992,
                is_event=True,
            ),
        ]
    )
    fert = transitions.build_fertility_panel(panel, bh)
    assert len(fert.births) == 3
    cells = transitions.fertility_cells(fert, weighted=True)
    # ASFR 25-29 has woman 3's age-25 birth (band 25-29) -> positive.
    assert cells["asfr.25-29"]["n_events"] == 1
    assert cells["asfr.20-24"]["n_events"] == 1  # woman 1 at 22
    # Completed fertility: women observed to 45 = persons 1 and 3. Woman 3
    # (1950s cohort) has 2 births; woman 1 (1970s) has 1.
    assert cells["completed_fertility.c1950s"]["rate"] == pytest.approx(2.0)
    assert cells["completed_fertility.c1970s"]["rate"] == pytest.approx(1.0)


# --- integration: real staged history files ---------------------------
@needs_marriage
@needs_births
@needs_ind
def test_real_panel_shape_pins():
    from populace_dynamics.data import births as births_mod
    from populace_dynamics.data import deaths as deaths_mod
    from populace_dynamics.data import panels

    mh = marriage.marriage_history()
    dr = deaths_mod.read_death_records()
    demo = panels.demographic_panel()
    demo_pos = demo[demo.weight > 0]
    pw = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    panel = transitions.build_marital_panel(mh, dr, pw)
    # Pinned on the 2026-07-07 staged data; tolerances absorb a refresh.
    assert panel.person_years.person_id.nunique() == pytest.approx(
        41409, abs=500
    )
    assert len(panel.person_years) == pytest.approx(1_164_692, rel=0.02)
    assert set(panel.events.transition.unique()) == {
        "first_marriage",
        "remarriage",
        "divorce",
        "widowhood",
    }
    # Every event age is at least the marriageable start age.
    assert int(panel.events.age.min()) >= transitions.START_AGE
    # Reference moments cover the expected cell count: the round-1 v2 set
    # (41 original + 4 aggregate hazards + 4 origin-split remarriage + 5
    # cohort ever-married + 8 dissolved-state stock shares = 62).
    fert = transitions.build_fertility_panel(panel, births_mod.birth_history())
    cells = transitions.reference_moments(panel, fert)
    assert len(cells) == 62
    # The added families are present.
    for key in (
        "widowhood.45+|male",
        "widowhood.45-64|female",
        "first_marriage.35+|female",
        "remarriage.after_divorce",
        "remarriage.widowed_under60",
        "ever_married_by_40.c1980s",
        "share_widowed.75+|female",
        "share_divorced.55-64|male",
    ):
        assert key in cells, key
