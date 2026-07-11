"""Tests for the UK dynamics layer (ADR 0002).

Ports the core coverage of archived policyengine-uk-data#346's test
suite onto :class:`UKPanelDataset` synthetic fixtures, plus new tests
pinning the two review fixes: orphaned-entity pruning after person
removal, and weight-proportional immigration donors. All hermetic —
no FRS or UKHLS microdata is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import ukhls
from populace_dynamics.uk import ons_rates
from populace_dynamics.uk.advance_year import advance_year
from populace_dynamics.uk.dataset import (
    UKPanelDataset,
    classify_panel_ids,
    prune_orphaned_entities,
)
from populace_dynamics.uk.demographic_ageing import age_dataset
from populace_dynamics.uk.household_transitions import (
    apply_children_leaving_home,
    apply_employment_transitions,
    apply_income_decile_transitions,
    apply_marriages,
    apply_migration,
    apply_separations,
)


def make_dataset(
    n_couples: int = 20,
    n_singles: int = 20,
    n_children: int = 10,
    seed: int = 0,
) -> UKPanelDataset:
    """Synthetic population: couples (benunit of 2 adults), singles,
    and children attached to the first households."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for i in range(n_couples):
        for sex, age_off in (("MALE", 0), ("FEMALE", -2)):
            rows.append(
                {
                    "person_id": pid,
                    "person_benunit_id": i,
                    "person_household_id": i,
                    "age": int(30 + (i % 30) + age_off),
                    "gender": sex,
                    "employment_income": float(rng.integers(15_000, 60_000)),
                    "employment_status": "FT_EMPLOYED",
                }
            )
            pid += 1
    for j in range(n_singles):
        bu = n_couples + j
        rows.append(
            {
                "person_id": pid,
                "person_benunit_id": bu,
                "person_household_id": bu,
                "age": int(22 + (j % 40)),
                "gender": "MALE" if j % 2 == 0 else "FEMALE",
                "employment_income": float(rng.integers(0, 40_000)),
                "employment_status": "FT_EMPLOYED",
            }
        )
        pid += 1
    for k in range(n_children):
        rows.append(
            {
                "person_id": pid,
                "person_benunit_id": k,
                "person_household_id": k,
                "age": int(rng.integers(0, 15)),
                "gender": "MALE" if k % 2 == 0 else "FEMALE",
                "employment_income": 0.0,
                "employment_status": "CHILD",
            }
        )
        pid += 1

    person = pd.DataFrame(rows)
    bu_ids = sorted(person["person_benunit_id"].unique())
    hh_ids = sorted(person["person_household_id"].unique())
    benunit = pd.DataFrame({"benunit_id": bu_ids, "benunit_weight": 1000.0})
    household = pd.DataFrame(
        {
            "household_id": hh_ids,
            "household_weight": 1000.0,
            "region": ["LONDON"] * len(hh_ids),
        }
    )
    return UKPanelDataset(person=person, benunit=benunit, household=household)


# ---------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------


def test_dataset_validates_ids():
    ds = make_dataset()
    ds.validate_ids()


def test_dataset_missing_column_raises():
    ds = make_dataset()
    with pytest.raises(ValueError, match="missing column"):
        UKPanelDataset(
            person=ds.person.drop(columns=["gender"]),
            benunit=ds.benunit,
            household=ds.household,
        )


def test_prune_orphaned_entities():
    ds = make_dataset()
    ds.person = ds.person[ds.person["person_household_id"] != 0].reset_index(
        drop=True
    )
    pruned = prune_orphaned_entities(ds)
    assert 0 not in set(pruned.household["household_id"])
    assert 0 not in set(pruned.benunit["benunit_id"])
    pruned.validate_ids()


def test_classify_panel_ids():
    ds = make_dataset()
    after = ds.copy()
    after.person = after.person[after.person["person_id"] != 0]
    new_row = ds.person.iloc[[1]].assign(person_id=99_999)
    after.person = pd.concat([after.person, new_row], ignore_index=True)
    result = classify_panel_ids(ds, after)
    assert 0 in result["removed"]
    assert 99_999 in result["added"]
    assert 1 in result["survivors"]


# ---------------------------------------------------------------
# Demographic ageing
# ---------------------------------------------------------------


def test_age_increment_and_id_preservation():
    ds = make_dataset()
    aged = age_dataset(ds, 1, mortality_rates={}, fertility_rates={})
    assert (
        aged.person["age"].to_numpy() == ds.person["age"].to_numpy() + 1
    ).all()
    assert set(aged.person["person_id"]) == set(ds.person["person_id"])


def test_zero_years_is_copy():
    ds = make_dataset()
    aged = age_dataset(ds, 0)
    pd.testing.assert_frame_equal(aged.person, ds.person)


def test_negative_years_raises():
    with pytest.raises(ValueError, match="non-negative"):
        age_dataset(make_dataset(), -1)


def test_mortality_removes_people():
    ds = make_dataset()
    aged = age_dataset(
        ds,
        1,
        mortality_rates={a: 1.0 for a in range(200)},
        fertility_rates={},
    )
    assert aged.person.empty


def test_mortality_prunes_orphaned_entities():
    """Review fix: killing every member of a household must drop its
    benunit and household rows too."""
    ds = make_dataset(n_couples=5, n_singles=5, n_children=0)
    # Kill only age 22 (single j=0, benunit/household 5, sole member).
    target = ds.person[ds.person["age"] == 22]
    assert len(target) == 1
    hh = int(target["person_household_id"].iloc[0])
    bu = int(target["person_benunit_id"].iloc[0])
    aged = age_dataset(ds, 1, mortality_rates={22: 1.0}, fertility_rates={})
    assert hh not in set(aged.household["household_id"])
    assert bu not in set(aged.benunit["benunit_id"])
    # Everyone else's entities survive.
    assert len(aged.household) == len(ds.household) - 1
    aged.validate_ids()


def test_sex_specific_mortality():
    ds = make_dataset(n_couples=50, n_singles=0, n_children=0)
    rates = {"MALE": {a: 1.0 for a in range(200)}, "FEMALE": {}}
    aged = age_dataset(ds, 1, mortality_rates=rates, fertility_rates={})
    assert (aged.person["gender"] == "FEMALE").all()


def test_fertility_adds_newborns_in_mothers_household():
    ds = make_dataset(n_couples=50, n_singles=0, n_children=0)
    aged = age_dataset(
        ds,
        1,
        seed=1,
        mortality_rates={},
        fertility_rates={a: 1.0 for a in range(15, 50)},
    )
    newborns = aged.person[aged.person["age"] == 1]  # born then aged +1
    mothers = ds.person[
        (ds.person["gender"] == "FEMALE") & ds.person["age"].between(15, 49)
    ]
    assert len(newborns) == len(mothers)
    assert set(newborns["person_household_id"]) <= set(
        mothers["person_household_id"]
    )
    # Fresh non-colliding IDs.
    assert newborns["person_id"].min() > ds.person["person_id"].max()
    aged.validate_ids()


def test_ageing_deterministic_same_seed():
    ds = make_dataset()
    a = age_dataset(ds, 3, seed=7)
    b = age_dataset(ds, 3, seed=7)
    pd.testing.assert_frame_equal(a.person, b.person)


# ---------------------------------------------------------------
# Marriage
# ---------------------------------------------------------------


def test_marriages_merge_benunits():
    ds = make_dataset(n_couples=0, n_singles=40, n_children=0)
    rates = {
        "MALE": {a: 1.0 for a in range(18, 121)},
        "FEMALE": {a: 1.0 for a in range(18, 121)},
    }
    out = apply_marriages(
        ds, marriage_rates=rates, rng=np.random.default_rng(0)
    )
    counts = out.person.groupby("person_benunit_id").size()
    assert (counts == 2).sum() > 0  # some couples formed
    assert len(out.benunit) < len(ds.benunit)  # units merged
    out.validate_ids()


def test_marriages_disabled_with_empty_rates():
    ds = make_dataset()
    out = apply_marriages(ds, marriage_rates={})
    pd.testing.assert_frame_equal(out.person, ds.person)


def test_marriage_weights_folded():
    ds = make_dataset(n_couples=0, n_singles=2, n_children=0)
    ds.person.loc[:, "age"] = 30
    rates = {
        "MALE": {30: 1.0},
        "FEMALE": {30: 1.0},
    }
    out = apply_marriages(
        ds, marriage_rates=rates, rng=np.random.default_rng(0)
    )
    assert len(out.benunit) == 1
    assert float(out.benunit["benunit_weight"].iloc[0]) == 2000.0


# ---------------------------------------------------------------
# Separation
# ---------------------------------------------------------------


def test_separations_split_couples():
    ds = make_dataset(n_couples=30, n_singles=0, n_children=0)
    out = apply_separations(
        ds,
        separation_rates={a: 1.0 for a in range(121)},
        rng=np.random.default_rng(0),
    )
    counts = out.person.groupby("person_benunit_id").size()
    assert (counts == 1).all()  # every couple split
    assert len(out.benunit) == 2 * len(ds.benunit)
    out.validate_ids()


def test_separation_mover_is_male_by_default():
    ds = make_dataset(n_couples=1, n_singles=0, n_children=0)
    out = apply_separations(
        ds,
        separation_rates={a: 1.0 for a in range(121)},
        rng=np.random.default_rng(0),
    )
    male = out.person[out.person["gender"] == "MALE"].iloc[0]
    female = out.person[out.person["gender"] == "FEMALE"].iloc[0]
    assert int(female["person_household_id"]) == 0  # stays
    assert int(male["person_household_id"]) != 0  # moves out


# ---------------------------------------------------------------
# Children leaving home
# ---------------------------------------------------------------


def test_children_leaving_home():
    ds = make_dataset(n_couples=5, n_singles=0, n_children=0)
    # Add an adult dependent (age 20) to benunit 0's couple.
    dep = ds.person.iloc[[0]].copy()
    dep["person_id"] = 9_000
    dep["age"] = 20
    ds.person = pd.concat([ds.person, dep], ignore_index=True)
    out = apply_children_leaving_home(
        ds,
        leaving_home_rates={20: 1.0},
        rng=np.random.default_rng(0),
    )
    leaver = out.person[out.person["person_id"] == 9_000].iloc[0]
    assert int(leaver["person_benunit_id"]) not in set(
        ds.person["person_benunit_id"]
    )
    assert int(leaver["person_household_id"]) not in set(
        ds.person["person_household_id"]
    )
    out.validate_ids()


# ---------------------------------------------------------------
# Migration
# ---------------------------------------------------------------


def test_migration_inflow_adds_people():
    ds = make_dataset(n_couples=0, n_singles=100, n_children=0)
    out = apply_migration(
        ds,
        net_migration_rates={a: 0.5 for a in range(18, 70)},
        rng=np.random.default_rng(0),
    )
    assert len(out.person) > len(ds.person)
    added = classify_panel_ids(ds, out)["added"]
    assert added
    out.validate_ids()


def test_migration_outflow_prunes_entities():
    ds = make_dataset(n_couples=0, n_singles=50, n_children=0)
    out = apply_migration(
        ds,
        net_migration_rates={a: -0.9 for a in range(0, 121)},
        rng=np.random.default_rng(0),
    )
    assert len(out.person) < len(ds.person)
    # Review fix: single-person households of emigrants are gone.
    out.validate_ids()
    live_hh = set(out.person["person_household_id"])
    assert set(out.household["household_id"]) == live_hh


def test_migration_donors_are_weight_proportional():
    """Review fix: a donor with 9x the household weight should be
    cloned far more often than its low-weight same-age peer."""
    # Inflow is capped at the cohort size, so use a 200-person cohort:
    # 100 heavy-weight donors (income 10k) and 100 light (income 20k).
    n = 200
    person = pd.DataFrame(
        {
            "person_id": range(n),
            "person_benunit_id": range(n),
            "person_household_id": range(n),
            "age": [25] * n,
            "gender": ["MALE"] * n,
            "employment_income": [10_000.0] * (n // 2) + [20_000.0] * (n // 2),
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": range(n), "benunit_weight": [1.0] * n}
    )
    household = pd.DataFrame(
        {
            "household_id": range(n),
            "household_weight": [9000.0] * (n // 2) + [1000.0] * (n // 2),
            "region": ["LONDON"] * n,
        }
    )
    ds = UKPanelDataset(person=person, benunit=benunit, household=household)
    out = apply_migration(
        ds,
        net_migration_rates={25: 1.0},
        rng=np.random.default_rng(0),
    )
    clones = out.person[out.person["person_id"] >= n]
    n_heavy = int((clones["employment_income"] == 10_000.0).sum())
    n_light = int((clones["employment_income"] == 20_000.0).sum())
    assert n_heavy + n_light > 50
    share_heavy = n_heavy / (n_heavy + n_light)
    assert 0.8 < share_heavy < 1.0  # ~0.9 expected


def test_migration_deterministic_same_seed():
    ds = make_dataset()
    a = apply_migration(ds, rng=np.random.default_rng(3))
    b = apply_migration(ds, rng=np.random.default_rng(3))
    pd.testing.assert_frame_equal(a.person, b.person)


# ---------------------------------------------------------------
# Employment / income transitions
# ---------------------------------------------------------------


def test_retirement_at_spa():
    ds = make_dataset()
    ds.person.loc[0, "age"] = 70
    ds.person.loc[0, "employment_income"] = 30_000.0
    out = apply_employment_transitions(ds, rng=np.random.default_rng(0))
    assert float(out.person.loc[0, "employment_income"]) == 0.0
    assert out.person.loc[0, "employment_status"] == "RETIRED"


def test_wage_drift():
    ds = make_dataset(n_couples=1, n_singles=0, n_children=0)
    ds.person["age"] = 40
    ds.person["employment_income"] = 10_000.0
    out = apply_employment_transitions(
        ds,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.05,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(out.person["employment_income"], 10_500.0)


def test_ukhls_driven_transitions_move_states():
    ds = make_dataset(n_couples=100, n_singles=0, n_children=0)
    # Everyone employed; force IN_WORK -> UNEMPLOYED for all cells.
    bands = [
        "16-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
    ]
    rates = {
        (b, s, "IN_WORK"): {"UNEMPLOYED": 1.0}
        for b in bands
        for s in ("MALE", "FEMALE")
    }
    out = apply_employment_transitions(
        ds,
        ukhls_rates=rates,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    working_age = out.person["age"].astype(int) < 66
    assert (out.person.loc[working_age, "employment_income"] == 0.0).all()


def test_income_decile_transitions_with_committed_tables():
    ds = make_dataset(n_couples=200, n_singles=0, n_children=0)
    rates = ukhls.load_income_decile_transitions()
    out = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(0)
    )
    before = ds.person["employment_income"].to_numpy()
    after = out.person["employment_income"].to_numpy()
    assert (before != after).any()  # some people moved
    assert (after >= 0).all()


# ---------------------------------------------------------------
# ONS rates (committed workbooks)
# ---------------------------------------------------------------


def test_ons_mortality_shape_and_sanity():
    rates = ons_rates.get_mortality_rates(2023)
    for sex in ("MALE", "FEMALE"):
        assert set(range(0, 101)) <= set(rates[sex])
        assert all(0 <= q <= 1 for q in rates[sex].values())
    # Adult male mortality exceeds female at 60.
    assert rates["MALE"][60] > rates["FEMALE"][60]
    # Old-age mortality far exceeds young-adult mortality.
    assert rates["MALE"][90] > 50 * rates["MALE"][25]


def test_ons_mortality_unisex_average():
    rates = ons_rates.get_mortality_rates(2023)
    uni = ons_rates.get_mortality_rates_unisex(2023)
    expected = 0.5 * rates["MALE"][50] + 0.5 * rates["FEMALE"][50]
    assert abs(uni[50] - expected) < 1e-12


def test_ons_fertility_shape_and_sanity():
    rates = ons_rates.get_fertility_rates(2024)
    assert set(rates) <= set(range(15, 50))
    assert 44 in rates  # "40 and over" capped to 40-44
    assert all(0 <= p < 0.2 for p in rates.values())
    # Fertility peaks around 30, far above the age-45 tail.
    assert rates[30] > 3 * rates[44]


def test_ons_fertility_year_fallback():
    latest = ons_rates.get_fertility_rates(None)
    future = ons_rates.get_fertility_rates(2050)
    assert future == latest


# ---------------------------------------------------------------
# advance_year composition
# ---------------------------------------------------------------


def test_advance_year_end_to_end():
    ds = make_dataset(n_couples=100, n_singles=100, n_children=50)
    out = advance_year(
        ds,
        seed=0,
        mortality_rates=ons_rates.get_mortality_rates(2023),
        fertility_rates=ons_rates.get_fertility_rates(2024),
        ukhls_employment_rates=ukhls.load_employment_transitions(),
        ukhls_decile_rates=ukhls.load_income_decile_transitions(),
    )
    assert out.fiscal_year == ds.fiscal_year + 1
    out.validate_ids()
    moved = classify_panel_ids(ds, out)
    assert moved["survivors"]  # population continuity


def test_advance_year_deterministic():
    ds = make_dataset(n_couples=50, n_singles=50, n_children=20)
    a = advance_year(ds, seed=42)
    b = advance_year(ds, seed=42)
    pd.testing.assert_frame_equal(a.person, b.person)
    pd.testing.assert_frame_equal(a.household, b.household)


def test_advance_year_does_not_mutate_input():
    ds = make_dataset()
    before = ds.person.copy()
    advance_year(ds, seed=0)
    pd.testing.assert_frame_equal(ds.person, before)


def test_advance_year_uprate_hook():
    ds = make_dataset(n_couples=5, n_singles=0, n_children=0)

    def double_income(d, year):
        d.person["employment_income"] *= 2.0
        return d

    out = advance_year(
        ds,
        seed=0,
        mortality_rates={},
        fertility_rates={},
        marriage_rates={},
        separation_rates={},
        leaving_home_rates={},
        net_migration_rates={},
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.0,
        uprate=double_income,
    )
    working = ds.person["age"].astype(int) < 66
    assert np.allclose(
        out.person.loc[working, "employment_income"],
        ds.person.loc[working, "employment_income"] * 2.0,
    )
