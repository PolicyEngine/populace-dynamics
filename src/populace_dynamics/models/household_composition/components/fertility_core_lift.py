"""Cohort-scoped fertility-core lift and retention/link closure.

Completed-size fits, Q14 anchors, retention fitting, and application helpers are
copied from ``household_composition_sim_v8.py:111-144,150-271,277-426,
530-695,802-968``.  The scoped write gate is copied from
``household_composition_sim_v9.py:70-110,151-256``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models.household_composition.base_simulator import (
    compose_base,
)
from populace_dynamics.models.household_composition.common import (
    weighted_share,
)
from populace_dynamics.models.household_composition.components.child_attribution import (
    CHILD_CORESIDENCE_MAX_AGE,
    NOT_MARRIED,
    child_band,
    custodial_probability,
)

SIZE_BUCKETS = ("0", "1", "2", "3", "4+")
COMPOSITION_BANDS = tuple(
    hc.band_label(lo, hi) for lo, hi in hc.COMPOSITION_AGE_BANDS
)
RETENTION_EXIT_CELLS = (
    "coresident_child.65-74|male",
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
LINK_COVERAGE_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
)
FERTILITY_LIFT_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
DEFICIT_COHORTS = frozenset(
    (cell.split(".", 1)[1].split("|")[0], cell.split("|")[1])
    for cell in FERTILITY_LIFT_CELLS
)
N_FIT_DRAWS = 6
DELTA_STREAM_TAG_V8 = 0xC8


def size_bucket(counts: np.ndarray) -> np.ndarray:
    """Map counts to the 0/1/2/3/4+ completed-size buckets.

    Copied from ``household_composition_sim_v8.py:150-153``.
    """
    values = np.asarray(counts, dtype=np.int64)
    return np.where(values >= 4, "4+", values.astype(str).astype(object))


def cell_of(cell: str) -> tuple[str, str]:
    """Split a child-cell key into band and sex.

    Copied from ``household_composition_sim_v8.py:156-160``.
    """
    tail = cell.split(".", 1)[1]
    band, sex = tail.split("|")
    return band, sex


def train_completed_size(
    parent_pairs: pd.DataFrame, train_ids: set[int]
) -> dict[int, int]:
    """Return maximum observed own-child count per train parent.

    Copied from ``household_composition_sim_v8.py:163-178``.
    """
    pairs = parent_pairs[parent_pairs["parent_person_id"].isin(train_ids)]
    if not len(pairs):
        return {}
    per_wave = pairs.groupby(["parent_person_id", "year"], sort=False).size()
    per_parent = per_wave.groupby("parent_person_id").max()
    return {int(key): int(value) for key, value in per_parent.items()}


def sim_completed_size_row(
    person_id_row: np.ndarray, child_counts_row: np.ndarray
) -> np.ndarray:
    """Broadcast each simulated person's maximum child count to their rows.

    Copied from ``household_composition_sim_v8.py:181-190``.
    """
    frame = pd.DataFrame({"pid": person_id_row, "cc": child_counts_row})
    maximum = frame.groupby("pid")["cc"].transform("max")
    return maximum.to_numpy(dtype=np.int64)


def size_dist(sizes: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Return a weighted completed-size distribution.

    Copied from ``household_composition_sim_v8.py:193-206``.
    """
    weights = np.asarray(weight, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return {bucket: 0.0 for bucket in SIZE_BUCKETS}
    counts = np.asarray(sizes, dtype=np.int64)
    return {
        "0": float(weights[counts == 0].sum() / total),
        "1": float(weights[counts == 1].sum() / total),
        "2": float(weights[counts == 2].sum() / total),
        "3": float(weights[counts == 3].sum() / total),
        "4+": float(weights[counts >= 4].sum() / total),
    }


def completed_size_dist_by_cell(
    hh: hc.HouseholdCompositionPanel,
    size_map: dict[int, int],
    train_ids: set[int],
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, float]]:
    """Fit train completed-size distributions by cohort and sex.

    Copied from ``household_composition_sim_v8.py:209-238``.
    """
    pw = hh.person_waves[hh.person_waves["person_id"].isin(train_ids)]
    pw = pw[pw["band"].notna()]
    size_all = (
        pw["person_id"]
        .map(lambda person: size_map.get(int(person), 0))
        .to_numpy(dtype=np.int64)
    )
    weight_all = pw["weight"].to_numpy(np.float64)
    distribution_all = size_dist(size_all, weight_all)
    by_cell: dict[tuple[str, str], dict[str, float]] = {}
    band_array = pw["band"].to_numpy(dtype=object)
    sex_array = pw["sex"].to_numpy()
    for band in COMPOSITION_BANDS:
        for sex in ("male", "female"):
            mask = (band_array == band) & (sex_array == sex)
            if not mask.any():
                by_cell[(band, sex)] = {bucket: 0.0 for bucket in SIZE_BUCKETS}
                continue
            by_cell[(band, sex)] = size_dist(size_all[mask], weight_all[mask])
    return by_cell, distribution_all


def cell_completed_size_dk(
    pw_cell: pd.DataFrame, size_map: dict[int, int]
) -> tuple[dict[str, float], dict[str, float], float]:
    """Return train D[S], K[S], and marginal rate for a cell.

    Copied from ``household_composition_sim_v8.py:241-271``.
    """
    if not len(pw_cell):
        zeros = {bucket: 0.0 for bucket in SIZE_BUCKETS}
        return zeros, zeros, 0.0
    sizes = (
        pw_cell["person_id"]
        .map(lambda person: size_map.get(int(person), 0))
        .to_numpy(dtype=np.int64)
    )
    buckets = size_bucket(sizes)
    weight = pw_cell["weight"].to_numpy(np.float64)
    coresident = pw_cell["coresident_child"].to_numpy(bool)
    total = float(weight.sum())
    distribution: dict[str, float] = {}
    kernel: dict[str, float] = {}
    for bucket in SIZE_BUCKETS:
        mask = buckets == bucket
        bucket_weight = float(weight[mask].sum())
        distribution[bucket] = bucket_weight / total if total > 0 else 0.0
        kernel[bucket] = (
            float((weight[mask] * coresident[mask]).sum() / bucket_weight)
            if bucket_weight > 0
            else 0.0
        )
    full = sum(distribution[b] * kernel[b] for b in SIZE_BUCKETS)
    return distribution, kernel, float(full)


def q14_linked_reference_cell(
    hh: hc.HouseholdCompositionPanel,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    fitted: Any,
    train_ids: set[int],
    cell: str,
) -> dict[str, float]:
    """Return deterministic linked-child anchors for an older-male cell.

    Copied from ``household_composition_sim_v8.py:277-426``.
    """
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(train_ids)][
        [
            "person_id",
            "year",
            "age",
            "band",
            "sex",
            "weight",
            "coresident_child",
        ]
    ].copy()
    links = fitted.father_links[["parent_person_id", "birth_year"]].copy()
    links["parent_person_id"] = links["parent_person_id"].astype("int64")
    links["birth_year"] = links["birth_year"].astype("int64")
    linked_father_ids = set(
        int(value) for value in links["parent_person_id"].unique()
    )
    joinable_keys = fitted.joinable_keys
    joinable_links = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ].copy()
    joinable_links["parent_person_id"] = joinable_links[
        "parent_person_id"
    ].astype("int64")
    joinable_links["child_person_id"] = joinable_links[
        "child_person_id"
    ].astype("int64")
    joinable_links["birth_year"] = joinable_links["birth_year"].astype("int64")
    father_waves = pw_b[["person_id", "year", "band", "sex", "weight"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    exposure = links.merge(father_waves, on="parent_person_id", how="inner")
    exposure["child_age"] = exposure["year"] - exposure["birth_year"]
    exposure = exposure[
        (exposure["child_age"] >= 0)
        & (exposure["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    exposure["child_band"] = exposure["child_age"].map(child_band)
    exposure = exposure[exposure["child_band"].notna()].copy()
    exposure = exposure.merge(
        marital_by_year.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    exposure["marital"] = exposure["marital"].fillna(NOT_MARRIED)
    index = pd.MultiIndex.from_arrays(
        [exposure["parent_person_id"], exposure["birth_year"]]
    )
    exposure["joinable"] = np.asarray(index.isin(joinable_keys), dtype=bool)
    probability = np.array(
        [
            custodial_probability(
                fitted, int(age), _era(int(year)), str(state)
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
    log_no_joinable = np.where(
        exposure["joinable"].to_numpy(),
        np.log1p(-np.clip(probability, 0.0, 1.0 - 1e-15)),
        0.0,
    )
    exposure["_logno_j"] = log_no_joinable
    grouped = exposure.groupby(["parent_person_id", "year"], sort=False)
    father_agg = grouped.agg(logno_j=("_logno_j", "sum")).reset_index()
    father_agg["a_refexp_j"] = 1.0 - np.exp(father_agg["logno_j"].to_numpy())
    index2 = pd.MultiIndex.from_arrays(
        [joinable_links["parent_person_id"], joinable_links["birth_year"]]
    )
    restricted_links = joinable_links[
        np.asarray(index2.isin(joinable_keys), dtype=bool)
    ]
    linked_core = parent_pairs.merge(
        restricted_links,
        on=["parent_person_id", "child_person_id"],
        how="inner",
    )
    linked_core["child_age"] = linked_core["year"] - linked_core["birth_year"]
    linked_core = linked_core[
        (linked_core["child_age"] >= 0)
        & (linked_core["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ]
    core_father_wave = (
        linked_core.groupby(["parent_person_id", "year"])
        .size()
        .rename("n_cor_j")
        .reset_index()
    )
    pw_b = pw_b.assign(
        _linked=pw_b["person_id"].isin(linked_father_ids),
        _male=(pw_b["sex"] == "male"),
    )
    band, _sex = cell_of(cell)
    cell_mask = (pw_b["band"] == band) & pw_b["_male"]
    cell_weight = pw_b.loc[cell_mask, "weight"].to_numpy(np.float64)
    total = float(cell_weight.sum())
    any_child = pw_b.loc[cell_mask, "coresident_child"].to_numpy(bool)
    is_linked = pw_b.loc[cell_mask, "_linked"].to_numpy(bool)
    reference_full = (
        float((cell_weight * any_child).sum() / total) if total > 0 else 0.0
    )
    linked_any = (
        float((cell_weight * (any_child & is_linked)).sum() / total)
        if total > 0
        else 0.0
    )
    linked = (
        pw_b.loc[cell_mask & pw_b["_linked"], ["person_id", "year", "weight"]]
        .merge(
            father_agg[["parent_person_id", "year", "a_refexp_j"]].rename(
                columns={"parent_person_id": "person_id"}
            ),
            on=["person_id", "year"],
            how="left",
        )
        .merge(
            core_father_wave.rename(columns={"parent_person_id": "person_id"}),
            on=["person_id", "year"],
            how="left",
        )
    )
    linked["a_refexp_j"] = linked["a_refexp_j"].fillna(0.0)
    linked["n_cor_j"] = linked["n_cor_j"].fillna(0).astype("int64")
    linked_weight = linked["weight"].to_numpy(np.float64)
    analytic = linked["a_refexp_j"].to_numpy(np.float64)
    n_coresident = linked["n_cor_j"].to_numpy(np.int64)
    analytic_joinable = (
        float((linked_weight * analytic).sum() / total) if total > 0 else 0.0
    )
    observed_joinable = (
        float((linked_weight * (n_coresident > 0)).sum() / total)
        if total > 0
        else 0.0
    )
    return {
        "reference_full_rate": reference_full,
        "linked_any": linked_any,
        "s_joinable_restricted": observed_joinable,
        "a_refexp_joinable": analytic_joinable,
        "link_coverage": observed_joinable - linked_any,
        "v7_interaction": analytic_joinable - observed_joinable,
    }


def _era(year: int) -> str:
    if year <= 1996:
        return "pre-1997"
    if year <= 2009:
        return "1997-2009"
    return "2010-2023"


def oaxaca_terms(
    cell_bucket: np.ndarray,
    cell_core: np.ndarray,
    cell_weight: np.ndarray,
    train_distribution: dict[str, float],
    train_kernel: dict[str, float],
) -> tuple[float, float, float, dict[str, float], dict[str, float]]:
    """Return the exact Oaxaca endowment/coefficient telescope.

    Copied from ``household_composition_sim_v8.py:670-695``.
    """
    weight = np.asarray(cell_weight, dtype=np.float64)
    total = float(weight.sum())
    sim_distribution: dict[str, float] = {}
    sim_kernel: dict[str, float] = {}
    for bucket in SIZE_BUCKETS:
        mask = cell_bucket == bucket
        bucket_weight = float(weight[mask].sum())
        sim_distribution[bucket] = bucket_weight / total if total > 0 else 0.0
        sim_kernel[bucket] = (
            float((weight[mask] * cell_core[mask]).sum() / bucket_weight)
            if bucket_weight > 0
            else 0.0
        )
    sim_full = sum(sim_distribution[b] * sim_kernel[b] for b in SIZE_BUCKETS)
    endowment = sum(
        (sim_distribution[b] - train_distribution.get(b, 0.0)) * sim_kernel[b]
        for b in SIZE_BUCKETS
    )
    coefficient = sum(
        train_distribution.get(b, 0.0)
        * (sim_kernel[b] - train_kernel.get(b, 0.0))
        for b in SIZE_BUCKETS
    )
    return (
        float(endowment),
        float(coefficient),
        float(sim_full),
        sim_distribution,
        sim_kernel,
    )


def fit_retention_link(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    fitted: Any,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    size_map: dict[int, int],
    train_ids: set[int],
    *,
    n_fit_draws: int = N_FIT_DRAWS,
    fit_seed_base: int = DELTA_STREAM_TAG_V8,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit the band-signed exit-origin and link-coverage closures.

    Copied from ``household_composition_sim_v8.py:530-667``.
    """
    all_cells = tuple(
        dict.fromkeys(RETENTION_EXIT_CELLS + LINK_COVERAGE_CELLS)
    )
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(train_ids)]
    pw_b = pw_b[pw_b["band"].notna()]
    train_dk: dict[str, Any] = {}
    for cell in all_cells:
        band, sex = cell_of(cell)
        sub = pw_b[(pw_b["band"] == band) & (pw_b["sex"] == sex)][
            ["person_id", "weight", "coresident_child"]
        ]
        distribution, kernel, full = cell_completed_size_dk(sub, size_map)
        train_dk[cell] = {
            "d_train": distribution,
            "k_train": kernel,
            "ref_full": full,
        }
    linked_ref = {
        cell: q14_linked_reference_cell(
            hh,
            father_links_child,
            parent_pairs,
            marital_by_year,
            fitted,
            train_ids,
            cell,
        )
        for cell in LINK_COVERAGE_CELLS
    }
    endowment_acc = {cell: [] for cell in all_cells}
    coefficient_acc = {cell: [] for cell in all_cells}
    sim_full_acc = {cell: [] for cell in all_cells}
    for draw in range(n_fit_draws):
        composition = compose_base(
            hh, mpanel, fitted, train_ids, fit_seed_base + draw
        )
        completed = sim_completed_size_row(
            composition["person_id"], composition["child_counts"]
        )
        buckets = size_bucket(completed)
        for cell in all_cells:
            band, sex = cell_of(cell)
            mask = (composition["band"] == band) & (composition["sex"] == sex)
            endowment, coefficient, sim_full, _dist, _kernel = oaxaca_terms(
                buckets[mask],
                composition["coresident_child"][mask],
                composition["weight"][mask],
                train_dk[cell]["d_train"],
                train_dk[cell]["k_train"],
            )
            endowment_acc[cell].append(endowment)
            coefficient_acc[cell].append(coefficient)
            sim_full_acc[cell].append(sim_full)
    shifts: dict[str, float] = {}
    diagnostic_cells: dict[str, Any] = {}
    for cell in all_cells:
        endowment = float(np.mean(endowment_acc[cell]))
        coefficient = float(np.mean(coefficient_acc[cell]))
        sim_full = float(np.mean(sim_full_acc[cell]))
        reference_full = train_dk[cell]["ref_full"]
        if cell in LINK_COVERAGE_CELLS:
            link = linked_ref[cell]["link_coverage"]
            v7 = linked_ref[cell]["v7_interaction"]
        else:
            link = 0.0
            v7 = 0.0
        exit_origin = coefficient - link - v7
        shift = 0.0
        if cell in RETENTION_EXIT_CELLS:
            shift += -exit_origin
        if cell in LINK_COVERAGE_CELLS:
            shift += -link
        shifts[cell] = float(shift)
        diagnostic_cells[cell] = {
            "sim_full_train": sim_full,
            "reference_full_train": reference_full,
            "cell_miss": sim_full - reference_full,
            "fertility_origin": endowment,
            "coefficient_kernel_shift": coefficient,
            "link_coverage": link,
            "v7_persistence_enumeration_interaction": v7,
            "exit_origin": exit_origin,
            "applied_shift": float(shift),
            "closes": (
                ["exit_origin", "link_coverage"]
                if cell in RETENTION_EXIT_CELLS and cell in LINK_COVERAGE_CELLS
                else (
                    ["exit_origin"]
                    if cell in RETENTION_EXIT_CELLS
                    else ["link_coverage"]
                )
            ),
        }
    return shifts, {"n_fit_draws": n_fit_draws, "per_cell": diagnostic_cells}


def apply_scoped_lift(
    person_id: np.ndarray,
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    child_counts: np.ndarray,
    coresident_child: np.ndarray,
    hh_size: np.ndarray,
    completed_size_dist_train: dict[tuple[str, str], dict[str, float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the candidate-9 write-gated completed-size swap.

    Copied from ``household_composition_sim_v9.py:151-256``.
    """
    simulated_size = sim_completed_size_row(person_id, child_counts)
    simulated_bucket = size_bucket(simulated_size)
    core_new = np.asarray(coresident_child, dtype=bool).copy()
    hh_new = np.asarray(hh_size, dtype=np.int64).copy()
    diag_cells: dict[str, Any] = {}
    for band_label in COMPOSITION_BANDS:
        for sex_label in ("male", "female"):
            cell = (band_label, sex_label)
            mask = (band == band_label) & (sex == sex_label)
            if not mask.any():
                continue
            index = np.flatnonzero(mask)
            train_distribution = completed_size_dist_train.get(cell)
            if train_distribution is None:
                continue
            probabilities = np.array(
                [
                    train_distribution.get(bucket, 0.0)
                    for bucket in SIZE_BUCKETS
                ],
                dtype=np.float64,
            )
            total = probabilities.sum()
            if total <= 0:
                continue
            probabilities = probabilities / total
            target_index = rng.choice(
                len(SIZE_BUCKETS), size=len(index), p=probabilities
            )
            target_bucket = np.array(SIZE_BUCKETS, dtype=object)[target_index]
            cell_weight = np.asarray(weight, dtype=np.float64)[index]
            cell_core = np.asarray(coresident_child, dtype=bool)[index]
            cell_hh = np.asarray(hh_size, dtype=np.int64)[index]
            cell_bucket = simulated_bucket[index]
            simulated_kernel: dict[str, float] = {}
            hh_pool: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
            for bucket in SIZE_BUCKETS:
                bucket_mask = cell_bucket == bucket
                bucket_weight = float(cell_weight[bucket_mask].sum())
                simulated_kernel[bucket] = (
                    float(
                        (
                            cell_weight[bucket_mask] * cell_core[bucket_mask]
                        ).sum()
                        / bucket_weight
                    )
                    if bucket_weight > 0
                    else 0.0
                )
                if bucket_mask.any() and bucket_weight > 0:
                    hh_pool[bucket] = (
                        cell_hh[bucket_mask],
                        cell_weight[bucket_mask] / bucket_weight,
                    )
                else:
                    hh_pool[bucket] = (cell_hh, None)
            kernel_row = np.array(
                [simulated_kernel[bucket] for bucket in target_bucket],
                dtype=np.float64,
            )
            uniform = rng.random(len(index))
            in_scope = cell in DEFICIT_COHORTS
            if in_scope:
                core_new[index] = uniform < kernel_row
            for bucket in SIZE_BUCKETS:
                selected = np.flatnonzero(target_bucket == bucket)
                if not len(selected):
                    continue
                values, weight_probability = hh_pool[bucket]
                if weight_probability is None or not len(values):
                    continue
                pick = rng.choice(
                    len(values), size=len(selected), p=weight_probability
                )
                if in_scope:
                    hh_new[index[selected]] = values[pick]
            diag_cells[f"coresident_child.{band_label}|{sex_label}"] = {
                "in_scope": bool(in_scope),
                "sim_completed_size_dist": size_dist(
                    simulated_size[index], cell_weight
                ),
                "train_completed_size_dist": train_distribution,
                "k_sim_given_size": simulated_kernel,
            }
    return (
        core_new,
        hh_new,
        {
            "per_cell": diag_cells,
            "scope_cells": list(FERTILITY_LIFT_CELLS),
        },
    )


def conditional_flip(
    values: np.ndarray,
    weight: np.ndarray,
    subset: np.ndarray,
    target_shift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Realize a weighted additive cell shift.

    Copied from ``household_composition_sim_v8.py:802-843``.
    """
    out = np.asarray(values, dtype=bool).copy()
    if target_shift == 0.0 or not subset.any():
        return out
    weights = np.asarray(weight, dtype=np.float64)
    total = float(weights[subset].sum())
    if total <= 0:
        return out
    need = target_shift * total
    if target_shift > 0:
        candidates = subset & (~out)
        candidate_weight = float(weights[candidates].sum())
        if candidate_weight <= 0:
            return out
        probability = min(max(need / candidate_weight, 0.0), 1.0)
        uniform = rng.random(len(out))
        flip = candidates & (uniform < probability)
        out[flip] = True
    else:
        candidates = subset & out
        candidate_weight = float(weights[candidates].sum())
        if candidate_weight <= 0:
            return out
        probability = min(max((-need) / candidate_weight, 0.0), 1.0)
        uniform = rng.random(len(out))
        flip = candidates & (uniform < probability)
        out[flip] = False
    return out


def apply_retention_link(
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    coresident_child: np.ndarray,
    shifts: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the fitted retention/link shifts in insertion order.

    Copied from ``household_composition_sim_v8.py:938-968``.
    """
    core_new = np.asarray(coresident_child, dtype=bool).copy()
    diag: dict[str, Any] = {}
    for cell, shift in shifts.items():
        band_label, sex_label = cell_of(cell)
        subset = (band == band_label) & (sex == sex_label)
        before = weighted_share(weight, core_new & subset, subset)
        core_new = conditional_flip(core_new, weight, subset, shift, rng)
        after = weighted_share(weight, core_new & subset, subset)
        diag[cell] = {
            "target_shift": float(shift),
            "rate_before": before,
            "rate_after": after,
            "realized_shift": after - before,
        }
    return core_new, diag
