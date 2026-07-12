"""Adapter from the certified family-transition registry to households.

Fit and spouse mapping are copied from
``household_composition_sim.py:315-357,612-642``.  The paternal shadow process
is copied from ``household_composition_sim.py:465-547``.  This adapter imports
the maintained family-transition registry directly and never imports a frozen
household-composition candidate module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models.family_transitions.components.fertility import (
    FERTILITY_AGE_HI,
    FERTILITY_AGE_LO,
    fertility_probabilities,
)


def fit_family_transitions(
    mpanel: transitions.MaritalPanel,
    demographic_panel: pd.DataFrame,
    marriage_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    marriage_order_map: pd.DataFrame,
    train_ids: set[int],
) -> ft.FittedFamilyTransitions:
    """Fit candidate 16 on the train complement.

    Copied from ``household_composition_sim.py:332-351``.
    """
    context = ft.FitContext(
        panel=mpanel,
        demographic_panel=demographic_panel,
        marriage_records=marriage_records,
        birth_records=birth_records,
        marriage_order_map=marriage_order_map,
        train_ids=frozenset(train_ids),
    )
    return ft.REGISTRY.fit(ft.CANDIDATE_16, context)


def fit_male_gap(fitted: ft.FittedFamilyTransitions) -> float:
    """Return the male-to-female spousal age gap used by the shadow.

    Copied from ``household_composition_sim.py:315-329``.
    """
    gaps = fitted.spousal_age_gaps.get("male", {})
    band0 = gaps.get(0)
    if band0 is None or len(band0) == 0:
        all_values = [
            value
            for value in gaps.values()
            if value is not None and len(value)
        ]
        if not all_values:
            return -2.0
        return float(np.concatenate(all_values).mean())
    return float(np.asarray(band0, dtype=np.float64).mean())


def spouse_from_marital(
    side_a_pw: pd.DataFrame, sim_years: pd.DataFrame
) -> np.ndarray:
    """Map simulated marital state to the legal-spouse occupancy.

    Copied from ``household_composition_sim.py:612-642``.
    """
    pw = side_a_pw.reset_index(drop=True)
    state = sim_years.set_index(["person_id", "year"])["marital_state"]
    idx = pd.MultiIndex.from_arrays(
        [pw["person_id"].to_numpy(), pw["year"].to_numpy()]
    )
    matched = state.reindex(idx)
    sim_spouse = (matched == "married").to_numpy()
    covered = matched.notna().to_numpy()
    obs_init = (
        pw.groupby("person_id")["coresident_spouse"]
        .transform("first")
        .to_numpy()
        .astype(bool)
    )
    return np.where(covered, sim_spouse, obs_init)


def marital_binary(state: pd.Series | np.ndarray) -> np.ndarray:
    """Binarize marital state for custodial conditioning.

    Copied from ``household_composition_sim_v3.py:128-131``.
    """
    series = pd.Series(state)
    return np.where(series.to_numpy() == "married", "married", "not_married")


def simulated_marital_binary(
    sim_years: pd.DataFrame, side_a_pw: pd.DataFrame
) -> pd.DataFrame:
    """Align simulated binary marital state to household person-waves.

    Copied from ``household_composition_sim_v3.py:612-628``.
    """
    sim = sim_years[["person_id", "year", "marital_state"]].copy()
    sim["marital"] = marital_binary(sim["marital_state"])
    pw = side_a_pw[["person_id", "year"]].merge(
        sim[["person_id", "year", "marital"]],
        on=["person_id", "year"],
        how="left",
    )
    return pw[["person_id", "year", "marital"]]


def father_marital_by_year(
    mpanel: transitions.MaritalPanel,
) -> pd.DataFrame:
    """Return observed binary marital state by father-year.

    Copied from ``household_composition_sim_v3.py:257-261``.
    """
    py = mpanel.person_years[["person_id", "year", "marital_state"]].copy()
    py["marital"] = marital_binary(py["marital_state"])
    return py[["person_id", "year", "marital"]]


def paternal_births(
    mpanel_sim_years: pd.DataFrame,
    attrs: pd.DataFrame,
    male_gap: float,
    fert_lookup: np.ndarray,
    fert_decade_map: dict[int, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw shadow paternal births for unlinked men.

    Copied from ``household_composition_sim.py:465-547``.
    """
    men = attrs[attrs["sex"] == "male"][["person_id", "birth_year"]]
    if not len(men):
        return pd.DataFrame({"parent_person_id": [], "birth_year": []}).astype(
            {"parent_person_id": "int64", "birth_year": "int64"}
        )
    married = mpanel_sim_years[mpanel_sim_years["marital_state"] == "married"][
        ["person_id", "year"]
    ]
    married = married[married["person_id"].isin(set(men["person_id"]))]
    if not len(married):
        return pd.DataFrame({"parent_person_id": [], "birth_year": []}).astype(
            {"parent_person_id": "int64", "birth_year": "int64"}
        )
    birth_year_by = men.set_index("person_id")["birth_year"]
    married = married.assign(
        birth_year_person=married["person_id"].map(birth_year_by)
    )
    married = married[married["birth_year_person"].notna()]
    married["wife_age"] = (
        married["year"] - married["birth_year_person"] + male_gap
    )
    married = married[
        (married["wife_age"] >= FERTILITY_AGE_LO)
        & (married["wife_age"] <= FERTILITY_AGE_HI)
    ].sort_values(["person_id", "year"])
    if not len(married):
        return pd.DataFrame({"parent_person_id": [], "birth_year": []}).astype(
            {"parent_person_id": "int64", "birth_year": "int64"}
        )
    married["decade_idx"] = (
        (married["birth_year_person"] // 10 * 10)
        .astype("int64")
        .map(lambda decade: fert_decade_map.get(int(decade), -1))
        .to_numpy()
    )
    married["wife_age_int"] = np.rint(married["wife_age"].to_numpy()).astype(
        np.int64
    )
    man_ids = np.sort(married["person_id"].unique())
    man_index = {int(person): index for index, person in enumerate(man_ids)}
    married["man_idx"] = married["person_id"].map(man_index).to_numpy()
    parity = np.zeros(len(man_ids), dtype=np.int64)
    out_person: list[int] = []
    out_year: list[int] = []
    for year, group in married.groupby("year", sort=True):
        midx = group["man_idx"].to_numpy()
        probability = fertility_probabilities(
            group["wife_age_int"].to_numpy(),
            parity[midx],
            group["decade_idx"].to_numpy(),
            fert_lookup,
        )
        uniform = rng.random(len(group))
        born = uniform < probability
        born_idx = midx[born]
        out_person.extend(int(man_ids[index]) for index in born_idx)
        out_year.extend([int(year)] * int(born.sum()))
        np.add.at(parity, born_idx, 1)
    return pd.DataFrame(
        {
            "parent_person_id": np.array(out_person, dtype=np.int64),
            "birth_year": np.array(out_year, dtype=np.int64),
        }
    )
