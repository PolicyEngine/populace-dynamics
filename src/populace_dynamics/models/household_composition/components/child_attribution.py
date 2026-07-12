"""Maternal, shadow, and custodial linked-child attribution.

This resolved component copies the effective lineage from:

* ``household_composition_sim_v2.py:212-236`` (all father links),
* ``household_composition_sim_v3.py:93-131,177-221,257-338`` (joinable links,
  roster pairs, and the original custodial basis),
* ``household_composition_sim_v4.py:119-123,146-152,416-534`` (single-year
  age/era custodial refinement),
* ``household_composition_sim_v5.py:264-376`` (child-record basis),
* ``household_composition_sim_v6.py:150-152,510-527`` (the final 0--4 basis
  revert), and
* ``household_composition_sim_v7.py:100-126,223-570,674-791`` (enumeration
  conditioning and episode-persistence frailty).

Superseded per-wave custodial draw functions are omitted; their final
probability resolver and exposure ordering are copied into the v7 draw below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import births, transitions
from populace_dynamics.data import household_composition as hc
from populace_dynamics.models.household_composition.common import weighted_rate
from populace_dynamics.models.household_composition.components.marital_core_adapter import (
    father_marital_by_year as _father_marital_by_year,
)

CHILD_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 12),
    (13, 17),
    (18, 24),
    (25, 60),
)
CHILD_CORESIDENCE_MAX_AGE = 60
SPELL_CHILD_MAX_AGE = 17
CUSTODIAL_ERA_SLICES = ("pre-1997", "1997-2009", "2010-2023")
CUSTODIAL_REVERT_BAND = (0, 4)
MIN_STRATUM_N = 20
MARRIED = "married"
NOT_MARRIED = "not_married"
DELTA_STREAM_TAG_V7 = 0xC7
EPISODE_BUCKETS = ("1", "2", "3", "4+")
CHILD_BAND_LABELS = tuple(hc.band_label(lo, hi) for lo, hi in CHILD_AGE_BANDS)


def child_band(age: int) -> str | None:
    """Return the custodial age-band label.

    Copied from ``household_composition_sim_v3.py:120-125``.
    """
    for lo, hi in CHILD_AGE_BANDS:
        if lo <= age <= hi:
            return hc.band_label(lo, hi)
    return None


def child_band_bounds(age: int) -> tuple[int, int]:
    """Return custodial band bounds.

    Copied from ``household_composition_sim_v4.py:530-534``.
    """
    for lo, hi in CHILD_AGE_BANDS:
        if lo <= age <= hi:
            return lo, hi
    return CHILD_AGE_BANDS[-1]


def era_of_year(year: int) -> str:
    """Map a year to the locked era slice.

    Copied from ``household_composition_sim_v4.py:146-152``.
    """
    if year <= 1996:
        return "pre-1997"
    if year <= 2009:
        return "1997-2009"
    return "2010-2023"


def father_link_births(birth_records: pd.DataFrame) -> pd.DataFrame:
    """Return all dated biological father-birth links.

    Copied from ``household_composition_sim_v2.py:212-236``.
    """
    events = births.birth_events(birth_records)
    father = events[
        (events["parent_sex"] == "male")
        & (events["record_type"] == "birth")
        & events["birth_year"].notna()
    ]
    out = pd.DataFrame(
        {
            "parent_person_id": father["parent_person_id"].astype("int64"),
            "birth_year": father["birth_year"].astype("int64"),
        }
    )
    return out.reset_index(drop=True)


def father_link_births_with_child(
    birth_records: pd.DataFrame,
) -> pd.DataFrame:
    """Return dated biological father links with a joinable child ID.

    Copied from ``household_composition_sim_v3.py:177-200``.
    """
    events = births.birth_events(birth_records)
    father = events[
        (events["parent_sex"] == "male")
        & (events["record_type"] == "birth")
        & events["birth_year"].notna()
        & events["child_person_id"].notna()
    ]
    return pd.DataFrame(
        {
            "parent_person_id": father["parent_person_id"].astype("int64"),
            "child_person_id": father["child_person_id"].astype("int64"),
            "birth_year": father["birth_year"].astype("int64"),
        }
    ).reset_index(drop=True)


def parent_child_coresidence_pairs(
    relationship_map: pd.DataFrame,
) -> pd.DataFrame:
    """Return observed parent-child coresidence pairs.

    Copied from ``household_composition_sim_v3.py:203-221``.
    """
    codes = hc.CORESIDENCE_LINKS["coresident_child"]
    links = relationship_map[relationship_map["ego_rel_to_alter"].isin(codes)]
    out = links[["interview_year", "ego_person_id", "alter_person_id"]].rename(
        columns={
            "interview_year": "year",
            "ego_person_id": "parent_person_id",
            "alter_person_id": "child_person_id",
        }
    )
    return out.drop_duplicates().reset_index(drop=True)


def father_marital_by_year(
    mpanel: transitions.MaritalPanel,
) -> pd.DataFrame:
    """Expose the frozen father-year marital input.

    Copied from ``household_composition_sim_v3.py:257-261``.
    """
    return _father_marital_by_year(mpanel)


def fit_custodial_era(
    person_waves: pd.DataFrame,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    train_ids: set[int],
) -> tuple[
    dict[tuple[int, str, str], float],
    dict[tuple[int, str], float],
    dict[tuple[str, str], float],
    float,
    dict[str, Any],
]:
    """Fit custodial probability by age, era, and father marital state.

    Copied from ``household_composition_sim_v4.py:416-513``.
    """
    links = father_links_child[
        father_links_child["parent_person_id"].isin(train_ids)
    ]
    father_waves = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "weight"]
    ].rename(columns={"person_id": "parent_person_id"})
    exposure = links.merge(father_waves, on="parent_person_id", how="inner")
    if not len(exposure):
        return {}, {}, {}, 0.0, {"n_exposure": 0, "n_coresident": 0}
    exposure["child_age"] = exposure["year"] - exposure["birth_year"]
    exposure = exposure[
        (exposure["child_age"] >= 0)
        & (exposure["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    exposure["era"] = exposure["year"].map(era_of_year)
    exposure = exposure.merge(
        marital_by_year.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    exposure["marital"] = exposure["marital"].fillna(NOT_MARRIED)
    pairs = parent_pairs.assign(_cores=True)
    exposure = exposure.merge(
        pairs,
        on=["parent_person_id", "child_person_id", "year"],
        how="left",
    )
    exposure["coresident"] = exposure["_cores"].fillna(False).astype(bool)
    weight = exposure["weight"].to_numpy(np.float64)
    event = exposure["coresident"].to_numpy(np.float64)
    overall = (
        float((weight * event).sum() / weight.sum())
        if weight.sum() > 0
        else 0.0
    )

    def rate(frame: pd.DataFrame) -> float:
        return weighted_rate(frame, frame["coresident"].to_numpy(float))

    band_marital: dict[tuple[str, str], float] = {}
    for lo, hi in CHILD_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for marital in (MARRIED, NOT_MARRIED):
            sub = exposure[
                (exposure["child_age"] >= lo)
                & (exposure["child_age"] <= hi)
                & (exposure["marital"] == marital)
            ]
            band_marital[(band, marital)] = rate(sub) if len(sub) else overall
    age_marital: dict[tuple[int, str], float] = {}
    custodial: dict[tuple[int, str, str], float] = {}
    for age in range(0, CHILD_CORESIDENCE_MAX_AGE + 1):
        age_rows = exposure[exposure["child_age"] == age]
        for marital in (MARRIED, NOT_MARRIED):
            age_state = age_rows[age_rows["marital"] == marital]
            if len(age_state) >= MIN_STRATUM_N:
                age_marital[(age, marital)] = rate(age_state)
            for era in CUSTODIAL_ERA_SLICES:
                sub = age_state[age_state["era"] == era]
                if len(sub) >= MIN_STRATUM_N:
                    custodial[(age, era, marital)] = rate(sub)
    return (
        custodial,
        age_marital,
        band_marital,
        overall,
        {
            "n_exposure": int(len(exposure)),
            "n_coresident": int(exposure["coresident"].sum()),
            "overall_coresidence_rate": round(overall, 5),
            "n_age_era_marital_cells": int(len(custodial)),
            "n_age_marital_cells": int(len(age_marital)),
        },
    )


def build_child_record_exposure(
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    demographic_panel: pd.DataFrame,
    relationship_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build the child-record custodial exposure.

    Copied from ``household_composition_sim_v5.py:264-339``.
    """
    father = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ]
    child_ids = set(int(value) for value in father["child_person_id"].unique())
    enumerated = (
        relationship_map[["interview_year", "ego_person_id"]]
        .drop_duplicates()
        .rename(
            columns={
                "interview_year": "year",
                "ego_person_id": "child_person_id",
            }
        )
    )
    enumerated = enumerated[
        enumerated["child_person_id"].isin(child_ids)
    ].copy()
    father_by_child = father.drop_duplicates("child_person_id").set_index(
        "child_person_id"
    )
    enumerated["birth_year"] = enumerated["child_person_id"].map(
        father_by_child["birth_year"]
    )
    enumerated["father_id"] = enumerated["child_person_id"].map(
        father_by_child["parent_person_id"]
    )
    enumerated["child_age"] = enumerated["year"] - enumerated["birth_year"]
    enumerated = enumerated[
        (enumerated["child_age"] >= 0)
        & (enumerated["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    enumerated["child_band"] = enumerated["child_age"].map(child_band)
    enumerated = enumerated[enumerated["child_band"].notna()]
    demo = demographic_panel[["person_id", "period", "weight"]].rename(
        columns={"person_id": "child_person_id", "period": "year"}
    )
    enumerated = enumerated.merge(
        demo, on=["child_person_id", "year"], how="left"
    )
    enumerated = enumerated[enumerated["weight"].fillna(0) > 0].copy()
    pair_father = parent_pairs.rename(
        columns={"parent_person_id": "father_id"}
    ).assign(_wf=True)
    enumerated = enumerated.merge(
        pair_father,
        on=["father_id", "child_person_id", "year"],
        how="left",
    )
    enumerated["with_father"] = enumerated["_wf"].fillna(False).astype(bool)
    enumerated = enumerated.drop(columns="_wf")
    enumerated = enumerated.merge(
        marital_by_year.rename(columns={"person_id": "father_id"}),
        on=["father_id", "year"],
        how="left",
    )
    enumerated["marital"] = enumerated["marital"].fillna(NOT_MARRIED)
    return enumerated[
        [
            "father_id",
            "child_person_id",
            "year",
            "child_band",
            "weight",
            "with_father",
            "marital",
        ]
    ].reset_index(drop=True)


def fit_custodial_child_record(
    child_record_exposure: pd.DataFrame, train_ids: set[int]
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Fit child-record custodial rates.

    Copied from ``household_composition_sim_v5.py:342-376``.
    """
    sub = child_record_exposure[
        child_record_exposure["father_id"].isin(train_ids)
    ]
    rates: dict[tuple[str, str], float] = {}
    for lo, hi in CHILD_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for marital in (MARRIED, NOT_MARRIED):
            mask = (sub["child_band"] == band) & (sub["marital"] == marital)
            frame = sub[mask]
            rates[(band, marital)] = (
                weighted_rate(frame, frame["with_father"].to_numpy(np.float64))
                if len(frame)
                else 0.0
            )
    return rates, {
        "n_child_record_waves_train": int(len(sub)),
        "not_married_child_record_by_band": {
            hc.band_label(lo, hi): round(
                rates[(hc.band_label(lo, hi), NOT_MARRIED)], 5
            )
            for lo, hi in CHILD_AGE_BANDS
        },
    }


def custodial_probability(
    fitted: Any, age: int, era: str, marital: str
) -> float:
    """Return candidate 9's resolved per-wave custody probability.

    Copied from ``household_composition_sim_v6.py:510-527`` and its candidate-4
    fallback ``household_composition_sim_v4.py:516-527``.
    """
    if marital == NOT_MARRIED:
        lo, hi = child_band_bounds(age)
        if (lo, hi) != CUSTODIAL_REVERT_BAND:
            band = hc.band_label(lo, hi)
            if (band, NOT_MARRIED) in fitted.custodial_child_record:
                return fitted.custodial_child_record[(band, NOT_MARRIED)]
    if (age, era, marital) in fitted.custodial_era:
        return fitted.custodial_era[(age, era, marital)]
    if (age, marital) in fitted.custodial_age_marital:
        return fitted.custodial_age_marital[(age, marital)]
    band = hc.band_label(*child_band_bounds(age))
    if (band, marital) in fitted.custodial_band_marital:
        return fitted.custodial_band_marital[(band, marital)]
    return fitted.custodial_overall


def enumerated_joinable_keys(
    father_links_child: pd.DataFrame,
) -> frozenset[tuple[int, int]]:
    """Return enumerated ``(father, birth-year)`` keys.

    Copied from ``household_composition_sim_v7.py:223-239``.
    """
    frame = father_links_child[["parent_person_id", "birth_year"]].copy()
    frame["parent_person_id"] = frame["parent_person_id"].astype("int64")
    frame["birth_year"] = frame["birth_year"].astype("int64")
    return frozenset(map(tuple, frame.drop_duplicates().to_numpy().tolist()))


def episode_length_hist(
    father_id: np.ndarray,
    child_key: np.ndarray,
    year: np.ndarray,
    coresident: np.ndarray,
) -> tuple[dict[str, float], float, int]:
    """Return the maximal-run episode distribution.

    Copied from ``household_composition_sim_v7.py:245-297``.
    """
    n = len(father_id)
    empty = ({bucket: 0.0 for bucket in EPISODE_BUCKETS}, 0.0, 0)
    if n == 0:
        return empty
    frame = pd.DataFrame(
        {
            "fid": np.asarray(father_id, dtype=np.int64),
            "ck": np.asarray(child_key, dtype=np.int64),
            "year": np.asarray(year, dtype=np.int64),
            "cor": np.asarray(coresident, dtype=bool),
        }
    )
    frame = frame.groupby(["fid", "ck", "year"], as_index=False)["cor"].max()
    frame = frame.sort_values(["fid", "ck", "year"], kind="stable")
    n = len(frame)
    father = frame["fid"].to_numpy()
    child = frame["ck"].to_numpy()
    core = frame["cor"].to_numpy()
    new_pair = np.empty(n, dtype=bool)
    new_pair[0] = True
    new_pair[1:] = (father[1:] != father[:-1]) | (child[1:] != child[:-1])
    previous = np.empty(n, dtype=bool)
    previous[0] = False
    previous[1:] = core[:-1]
    start = core & (new_pair | ~previous)
    if not start.any():
        return empty
    episode_id = np.cumsum(start) - 1
    lengths = np.bincount(episode_id[core])
    lengths = lengths[lengths > 0]
    total = int(lengths.size)
    if total == 0:
        return empty
    return (
        {
            "1": float(np.sum(lengths == 1) / total),
            "2": float(np.sum(lengths == 2) / total),
            "3": float(np.sum(lengths == 3) / total),
            "4+": float(np.sum(lengths >= 4) / total),
        },
        float(lengths.mean()),
        total,
    )


def reference_linked_episode_stats(
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    father_waves: pd.DataFrame,
) -> tuple[dict[str, float], float, int]:
    """Return train reference episode statistics.

    Copied from ``household_composition_sim_v7.py:303-346``.
    """
    links = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ].copy()
    links["parent_person_id"] = links["parent_person_id"].astype("int64")
    links["child_person_id"] = links["child_person_id"].astype("int64")
    links["birth_year"] = links["birth_year"].astype("int64")
    waves = father_waves[["person_id", "year"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    reference = links.merge(waves, on="parent_person_id", how="inner")
    reference["child_age"] = reference["year"] - reference["birth_year"]
    reference = reference[
        (reference["child_age"] >= 0)
        & (reference["child_age"] <= SPELL_CHILD_MAX_AGE)
    ]
    pairs = parent_pairs[
        ["parent_person_id", "child_person_id", "year"]
    ].assign(_cor=True)
    reference = reference.merge(
        pairs,
        on=["parent_person_id", "child_person_id", "year"],
        how="left",
    )
    reference["coresident"] = reference["_cor"].fillna(False).astype(bool)
    return episode_length_hist(
        reference["parent_person_id"].to_numpy(np.int64),
        reference["birth_year"].to_numpy(np.int64),
        reference["year"].to_numpy(np.int64),
        reference["coresident"].to_numpy(bool),
    )


def simulate_linked_episode_coresidence(
    father_id: np.ndarray,
    birth_id: np.ndarray,
    probability: np.ndarray,
    persistence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the persistent, sibling-synchronized frailty mixture.

    Copied from ``household_composition_sim_v7.py:352-386``.
    """
    n = len(probability)
    if n == 0:
        return np.zeros(0, dtype=bool)
    unique_father, father_inverse = np.unique(
        np.asarray(father_id, dtype=np.int64), return_inverse=True
    )
    unique_birth, birth_inverse = np.unique(
        np.asarray(birth_id, dtype=np.int64), return_inverse=True
    )
    father_frailty = rng.random(len(unique_father))
    follows = rng.random(len(unique_birth)) < persistence
    idiosyncratic = rng.random(n)
    uniform = np.where(
        follows[birth_inverse], father_frailty[father_inverse], idiosyncratic
    )
    return uniform < np.asarray(probability, dtype=np.float64)


def linked_exposure_frame(
    linked_births: pd.DataFrame,
    father_waves: pd.DataFrame,
    marital: pd.DataFrame,
    fitted: Any,
    joinable_keys: frozenset[tuple[int, int]],
) -> pd.DataFrame:
    """Build the linked father-wave exposure in frozen row order.

    Copied from ``household_composition_sim_v7.py:389-459``.
    """
    waves = father_waves.reset_index(drop=True)
    waves = waves.assign(_row=np.arange(len(waves), dtype=np.int64))
    waves = waves[["person_id", "year", "_row"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    links = linked_births.reset_index(drop=True)
    links = links.assign(_birth_id=np.arange(len(links), dtype=np.int64))
    exposure = links.merge(waves, on="parent_person_id", how="inner")
    if not len(exposure):
        return exposure.assign(
            child_age=np.array([], dtype=np.int64),
            child_band=np.array([], dtype=object),
            marital=np.array([], dtype=object),
            joinable=np.array([], dtype=bool),
            p_c=np.array([], dtype=np.float64),
        )
    exposure["child_age"] = exposure["year"] - exposure["birth_year"]
    exposure = exposure[
        (exposure["child_age"] >= 0)
        & (exposure["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    exposure["child_band"] = exposure["child_age"].map(child_band)
    exposure = exposure[exposure["child_band"].notna()]
    if not len(exposure):
        return exposure.assign(
            marital=np.array([], dtype=object),
            joinable=np.array([], dtype=bool),
            p_c=np.array([], dtype=np.float64),
        )
    exposure = exposure.merge(
        marital.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    exposure["marital"] = exposure["marital"].fillna(NOT_MARRIED)
    exposure = exposure.sort_values(["parent_person_id", "birth_year", "_row"])
    exposure = exposure.reset_index(drop=True)
    index = pd.MultiIndex.from_arrays(
        [exposure["parent_person_id"], exposure["birth_year"]]
    )
    exposure["joinable"] = np.asarray(index.isin(joinable_keys), dtype=bool)
    exposure["p_c"] = np.array(
        [
            custodial_probability(
                fitted, int(age), era_of_year(int(year)), str(state)
            )
            for age, year, state in zip(
                exposure["child_age"].to_numpy(),
                exposure["year"].to_numpy(),
                exposure["marital"].to_numpy(),
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    return exposure


def expected_episode_mean_at_persistence(
    exposure_joinable: pd.DataFrame,
    persistence: float,
    rng: np.random.Generator,
) -> float:
    """Return simulated minor-child episode mean for ``persistence``.

    Copied from ``household_composition_sim_v7.py:462-490``.
    """
    exposure = exposure_joinable.sort_values(
        ["_birth_id", "year"]
    ).reset_index(drop=True)
    coresident = simulate_linked_episode_coresidence(
        exposure["parent_person_id"].to_numpy(np.int64),
        exposure["_birth_id"].to_numpy(np.int64),
        exposure["p_c"].to_numpy(np.float64),
        persistence,
        rng,
    )
    minor = exposure["child_age"].to_numpy() <= SPELL_CHILD_MAX_AGE
    _distribution, mean, _count = episode_length_hist(
        exposure["parent_person_id"].to_numpy(np.int64)[minor],
        exposure["birth_year"].to_numpy(np.int64)[minor],
        exposure["year"].to_numpy(np.int64)[minor],
        coresident[minor],
    )
    return mean


def fit_linked_episode_persistence(
    person_waves: pd.DataFrame,
    fitted: Any,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    joinable_keys: frozenset[tuple[int, int]],
    train_ids: set[int],
    *,
    fit_seed: int = DELTA_STREAM_TAG_V7,
    n_bisect: int = 24,
) -> tuple[float, dict[str, Any]]:
    """Fit the linked episode persistence fraction.

    Copied from ``household_composition_sim_v7.py:493-570``.
    """
    train_pw = person_waves[person_waves["person_id"].isin(train_ids)]
    father_waves = train_pw[["person_id", "year"]]
    ref_dist, ref_mean, ref_n = reference_linked_episode_stats(
        father_links_child, parent_pairs, father_waves
    )
    links = fitted.father_links[["parent_person_id", "birth_year"]].copy()
    links["parent_person_id"] = links["parent_person_id"].astype("int64")
    links["birth_year"] = links["birth_year"].astype("int64")
    links = links[links["parent_person_id"].isin(train_ids)]
    exposure = linked_exposure_frame(
        links, father_waves, marital_by_year, fitted, joinable_keys
    )
    exposure_joinable = exposure[exposure["joinable"].to_numpy()].copy()

    def simulated_mean(rho: float) -> float:
        return expected_episode_mean_at_persistence(
            exposure_joinable, rho, np.random.default_rng([fit_seed, 0])
        )

    mean_lo = simulated_mean(0.0)
    mean_hi = simulated_mean(1.0)
    if ref_mean <= mean_lo:
        rho = 0.0
    elif ref_mean >= mean_hi:
        rho = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if simulated_mean(mid) < ref_mean:
                lo = mid
            else:
                hi = mid
        rho = 0.5 * (lo + hi)
    achieved = simulated_mean(rho)
    return rho, {
        "target_reference_episode_mean_train": ref_mean,
        "target_reference_episode_distribution_train": ref_dist,
        "target_reference_n_episodes_train": ref_n,
        "candidate6_independent_episode_mean_train": mean_lo,
        "fully_persistent_episode_mean_train": mean_hi,
        "fitted_persistence_rho": rho,
        "achieved_episode_mean_at_rho_train": achieved,
        "n_bisect_iterations": n_bisect,
        "spell_child_max_age": SPELL_CHILD_MAX_AGE,
        "note": (
            "rho is the fraction of linked children that follow the "
            "persistent, sibling-synchronized frailty episode (vs the "
            "candidate-6 independent per-wave draw); fitted on train side B "
            "so the simulated episode-length mean matches the reference "
            "(~5.93 waves) while the per-wave custodial marginal is "
            "preserved exactly."
        ),
    }


def custodial_linked_child_counts(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    fitted: Any,
    episode_rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw enumeration-conditioned, episode-persistent linked children.

    Copied from ``household_composition_sim_v7.py:674-791``.
    """
    counts = np.zeros(len(side_a_pw), dtype=np.int64)
    diag = {
        "n_linked_child_coresident_wave_units": 0,
        "nonjoinable_share_by_band": {},
        "marginal_preservation_by_band": {},
        "marginal_preservation_max_abs_dev": 0.0,
        "sim_episode_length_distribution": {
            bucket: 0.0 for bucket in EPISODE_BUCKETS
        },
        "sim_episode_mean_length": 0.0,
        "sim_n_episodes": 0,
        "persistence_rho": float(fitted.linked_episode_persistence),
    }
    if not len(linked_births):
        return counts, diag
    exposure = linked_exposure_frame(
        linked_births,
        side_a_pw,
        marital_sim,
        fitted,
        fitted.joinable_keys,
    )
    if not len(exposure):
        return counts, diag
    joinable = exposure["joinable"].to_numpy(dtype=bool)
    bands = exposure["child_band"].to_numpy(dtype=object)
    for band in CHILD_BAND_LABELS:
        mask = bands == band
        n_rows = int(mask.sum())
        n_nonjoinable = int((mask & ~joinable).sum())
        diag["nonjoinable_share_by_band"][band] = {
            "n_exposure_rows": n_rows,
            "n_nonjoinable": n_nonjoinable,
            "share": (round(n_nonjoinable / n_rows, 6) if n_rows else 0.0),
        }
    joinable_exposure = (
        exposure[joinable]
        .sort_values(["_birth_id", "year"])
        .reset_index(drop=True)
    )
    coresident = simulate_linked_episode_coresidence(
        joinable_exposure["parent_person_id"].to_numpy(np.int64),
        joinable_exposure["_birth_id"].to_numpy(np.int64),
        joinable_exposure["p_c"].to_numpy(np.float64),
        float(fitted.linked_episode_persistence),
        episode_rng,
    )
    rows = joinable_exposure["_row"].to_numpy()
    np.add.at(counts, rows[coresident], 1)
    diag["n_linked_child_coresident_wave_units"] = int(coresident.sum())
    joinable_bands = joinable_exposure["child_band"].to_numpy(dtype=object)
    probability = joinable_exposure["p_c"].to_numpy(dtype=np.float64)
    max_deviation = 0.0
    for band in CHILD_BAND_LABELS:
        mask = joinable_bands == band
        if not mask.any():
            continue
        target = float(probability[mask].mean())
        simulated = float(coresident[mask].mean())
        deviation = abs(simulated - target)
        max_deviation = max(max_deviation, deviation)
        diag["marginal_preservation_by_band"][band] = {
            "target_mean_custody_prob": target,
            "sim_coresident_share": simulated,
            "abs_deviation": deviation,
            "n_joinable_exposure_rows": int(mask.sum()),
        }
    diag["marginal_preservation_max_abs_dev"] = max_deviation
    minor = joinable_exposure["child_age"].to_numpy() <= SPELL_CHILD_MAX_AGE
    distribution, mean, n_episodes = episode_length_hist(
        joinable_exposure["parent_person_id"].to_numpy(np.int64)[minor],
        joinable_exposure["birth_year"].to_numpy(np.int64)[minor],
        joinable_exposure["year"].to_numpy(np.int64)[minor],
        coresident[minor],
    )
    diag["sim_episode_length_distribution"] = distribution
    diag["sim_episode_mean_length"] = mean
    diag["sim_n_episodes"] = n_episodes
    return counts, diag
