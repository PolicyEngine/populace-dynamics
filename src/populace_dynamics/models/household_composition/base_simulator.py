"""Candidate-7 composition core used by fitting and candidate-9 simulation.

``compose_base`` is a flattened copy of
``household_composition_sim_v8.py:1025-1305``.  Its carried C1 state helper
copies the RNG-consuming prefix of
``household_composition_sim.py:645-747``.  Retired streams and discarded
leave-year draws remain explicit because they are part of candidate 9's RNG
topology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models.family_transitions.components.fertility import (
    build_fertility_lookup,
)
from populace_dynamics.models.household_composition.common import (
    coresident_child_counts,
    evolve_absorbing_exit,
    evolve_two_state,
    padded_person_matrices,
)
from populace_dynamics.models.household_composition.components.child_attribution import (
    DELTA_STREAM_TAG_V7,
    custodial_linked_child_counts,
)
from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
    COHAB_OVERLAY_LIFT,
)
from populace_dynamics.models.household_composition.components.household_size import (
    compose_states,
)
from populace_dynamics.models.household_composition.components.marital_core_adapter import (
    paternal_births,
    simulated_marital_binary,
    spouse_from_marital,
)
from populace_dynamics.models.household_composition.components.multigenerational_occupancy import (
    DELTA_STREAM_TAG_V5,
    GRANDCHILD_LO,
)
from populace_dynamics.models.household_composition.components.nonfamily_bridge import (
    sample_nonfamily_count,
)
from populace_dynamics.models.household_composition.components.parental_home_exit import (
    DELTA_STREAM_TAG_V6,
    child_leave_years,
    child_leave_years_refit,
)


def _carried_c1_states(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    fitted: Any,
    holdout_ids: set[int],
    draw_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C1 spouse, parent, and multigen states with exact C1 RNG use.

    Copied from ``household_composition_sim.py:645-747``.  Child-count and
    panel construction after the occupancy draws are omitted because candidate
    9 reads only these three states and those omitted operations consume no RNG.
    """
    occupancy_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, 0xB2B])
    )
    sim_panel, sim_births = ft.simulate(
        mpanel, holdout_ids, fitted.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    side_a_pw = (
        hh.person_waves[hh.person_waves["person_id"].isin(holdout_ids)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )
    spouse = spouse_from_marital(side_a_pw, sim_years)
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    lookup, decade_map = build_fertility_lookup(
        fitted.family_transitions.fertility
    )
    paternal = paternal_births(
        sim_years,
        mpanel.attrs[mpanel.attrs["person_id"].isin(holdout_ids)],
        fitted.male_gap,
        lookup,
        decade_map,
        occupancy_rng,
    )
    all_births = pd.concat([maternal, paternal], ignore_index=True)
    all_births = all_births[all_births["parent_person_id"].isin(holdout_ids)]
    child_leave_years(all_births, fitted.parental_exit, occupancy_rng)

    matrices = padded_person_matrices(side_a_pw)
    pw = matrices["pw"]
    row_of = matrices["row_of"]
    n_persons = matrices["n_persons"]
    max_waves = matrices["max_waves"]
    valid = row_of >= 0
    safe_row = np.where(valid, row_of, 0)
    age_mat = pw["age"].to_numpy()[safe_row]
    sex_mat = pw["sex"].to_numpy()[safe_row]
    is_male_mat = (sex_mat == "male").astype(np.float64)
    band_mat = pw["band"].to_numpy(dtype=object)[safe_row]
    obs_parent = pw["coresident_parent"].to_numpy(dtype=bool)[safe_row] & valid
    obs_multigen = pw["multigen"].to_numpy(dtype=bool)[safe_row] & valid
    exit_prob = fitted.parental_exit.predict(
        age_mat.reshape(-1), is_male_mat.reshape(-1)
    ).reshape(n_persons, max_waves)
    parent_state = evolve_absorbing_exit(
        valid, obs_parent, exit_prob, occupancy_rng
    )
    entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    exitm_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in fitted.multigen_entry.items():
        mask = (band_mat == band) & (sex_mat == sex)
        entry_prob[mask] = rate
    for (band, sex), rate in fitted.multigen_exit.items():
        mask = (band_mat == band) & (sex_mat == sex)
        exitm_prob[mask] = rate
    multigen_state = evolve_two_state(
        valid, obs_multigen, entry_prob, exitm_prob, occupancy_rng
    )
    parent_row = np.zeros(len(pw), dtype=bool)
    multigen_row = np.zeros(len(pw), dtype=bool)
    parent_row[row_of[valid]] = parent_state[valid]
    multigen_row[row_of[valid]] = multigen_state[valid]
    return spouse, parent_row, multigen_row


def compose_base(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    fitted: Any,
    holdout_ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Compose candidate 7 exactly, before the candidate-8/9 deltas.

    Copied from ``household_composition_sim_v8.py:1025-1305``.
    """
    legal_spouse, c1_parent, c1_multigen = _carried_c1_states(
        hh, mpanel, fitted, holdout_ids, draw_seed
    )
    side_a_pw = (
        hh.person_waves[hh.person_waves["person_id"].isin(holdout_ids)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )
    side_a_pw = side_a_pw.merge(
        fitted.cohab_flag, on=["person_id", "year"], how="left"
    )
    side_a_pw["cohabiting"] = (
        side_a_pw["cohabiting"].fillna(False).astype(bool)
    )

    delta_ss = np.random.SeedSequence([draw_seed, 0xC2])
    child_ss, cohab_ss = delta_ss.spawn(2)
    child_rng = np.random.default_rng(child_ss)
    cohab_rng = np.random.default_rng(cohab_ss)

    matrices = padded_person_matrices(side_a_pw)
    pw = matrices["pw"]
    row_of = matrices["row_of"]
    n_persons = matrices["n_persons"]
    max_waves = matrices["max_waves"]
    valid = row_of >= 0
    safe_row = np.where(valid, row_of, 0)
    age_mat = pw["age"].to_numpy()[safe_row]
    sex_mat = pw["sex"].to_numpy()[safe_row]
    band_mat = pw["band"].to_numpy(dtype=object)[safe_row]
    obs_cohab = pw["cohabiting"].to_numpy(dtype=bool)[safe_row] & valid
    cohab_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    cohab_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in fitted.cohab_entry.items():
        cohab_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in fitted.cohab_exit.items():
        cohab_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (age, sex), rate in fitted.cohab_entry_age.items():
        cohab_entry_prob[(age_mat == age) & (sex_mat == sex)] = rate
    for (age, sex), rate in fitted.cohab_exit_age.items():
        cohab_exit_prob[(age_mat == age) & (sex_mat == sex)] = rate
    female_mat = sex_mat == "female"
    for age, rate in fitted.cohab_entry_age_female.items():
        cohab_entry_prob[(age_mat == age) & female_mat] = rate
    for age, rate in fitted.cohab_exit_age_female.items():
        cohab_exit_prob[(age_mat == age) & female_mat] = rate
    cohab_state = evolve_two_state(
        valid,
        obs_cohab,
        cohab_entry_prob,
        cohab_exit_prob,
        cohab_rng,
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]

    c4_ss = np.random.SeedSequence([draw_seed, 0xC4])
    legal_residual_ss, _retired_nonfamily2_ss = c4_ss.spawn(2)
    legal_residual_rng = np.random.default_rng(legal_residual_ss)
    lr_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_marginal_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in fitted.legal_residual_entry.items():
        mask = (band_mat == band) & (sex_mat == sex)
        lr_exit_prob[mask] = fitted.legal_residual_exit[(band, sex)]
        lr_entry_prob[mask] = rate
        lr_marginal_prob[mask] = fitted.legal_residual_marginal[(band, sex)]
    lr_initial = np.zeros((n_persons, max_waves), dtype=bool)
    lr_initial[:, 0] = (
        legal_residual_rng.random(n_persons) < lr_marginal_prob[:, 0]
    ) & valid[:, 0]
    lr_state = evolve_two_state(
        valid,
        lr_initial,
        lr_entry_prob,
        lr_exit_prob,
        legal_residual_rng,
    )
    lr_row = np.zeros(len(pw), dtype=bool)
    lr_row[row_of[valid]] = lr_state[valid]
    spouse = legal_spouse | cohab_row | lr_row

    sim_panel, sim_births = ft.simulate(
        mpanel, holdout_ids, fitted.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"
    men_a = mpanel.attrs[
        (mpanel.attrs["sex"] == "male")
        & (mpanel.attrs["person_id"].isin(holdout_ids))
    ]
    men_ids = set(int(value) for value in men_a["person_id"])
    linked_ids = (
        set(int(value) for value in fitted.father_links["parent_person_id"])
        & men_ids
    )
    paternal_linked = fitted.father_links[
        fitted.father_links["parent_person_id"].isin(linked_ids)
    ][["parent_person_id", "birth_year"]].copy()
    paternal_linked["parent_person_id"] = paternal_linked[
        "parent_person_id"
    ].astype("int64")
    paternal_linked["birth_year"] = paternal_linked["birth_year"].astype(
        "int64"
    )
    paternal_linked["_source"] = "linked"
    unlinked_men = men_a[~men_a["person_id"].isin(linked_ids)]
    lookup, decade_map = build_fertility_lookup(
        fitted.family_transitions.fertility
    )
    paternal_shadow = paternal_births(
        sim_years,
        unlinked_men,
        fitted.male_gap,
        lookup,
        decade_map,
        child_rng,
    )
    paternal_shadow["_source"] = "shadow"
    all_births = pd.concat(
        [maternal, paternal_linked, paternal_shadow], ignore_index=True
    )
    all_births = all_births[all_births["parent_person_id"].isin(holdout_ids)]
    base_leaves = child_leave_years(
        all_births, fitted.parental_exit, child_rng
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    c3_ss = np.random.SeedSequence([draw_seed, 0xC3])
    _retired_custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)
    c5_ss = np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)
    maternal_leave_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V6])
    )
    episode_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V7])
    )

    maternal_births = all_births[all_births["_source"] == "maternal"]
    maternal_leaves = child_leave_years_refit(
        maternal_births,
        fitted.parental_exit,
        fitted.child_exit_single_year,
        maternal_leave_rng,
    )
    nonlinked_leaves = pd.concat(
        [maternal_leaves, shadow_leaves], ignore_index=True
    )
    child_counts_nonlinked = coresident_child_counts(
        nonlinked_leaves, side_a_pw
    )
    marital_sim = simulated_marital_binary(sim_years, side_a_pw)
    child_counts_linked, linked_diag = custodial_linked_child_counts(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        fitted,
        episode_rng,
    )
    child_counts = child_counts_nonlinked + child_counts_linked
    coresident_child, grandchild_composed, _discarded_hh_default = (
        compose_states(spouse, c1_parent, c1_multigen, child_counts, 2)
    )

    bands_row = pw["band"].to_numpy(dtype=object)
    sexes_row = pw["sex"].to_numpy()
    ages_row = pw["age"].to_numpy()
    two_share = np.full(
        len(pw), fitted.parent_count_two_pooled, dtype=np.float64
    )
    for (band, sex), share in fitted.parent_count_two_share.items():
        two_share[(bands_row == band) & (sexes_row == sex)] = share
    parent_uniform = parentcount_rng.random(len(pw))
    parent_count_ego = np.where(parent_uniform < two_share, 2, 1).astype(
        np.int64
    )
    n_parents_ego = np.where(c1_parent, parent_count_ego, 0).astype(np.int64)
    hh_size_base = (
        1
        + spouse.astype(np.int64)
        + child_counts.astype(np.int64)
        + n_parents_ego
    ).astype(np.int64)

    is_55_row = ages_row >= GRANDCHILD_LO
    coupled_probability = np.zeros(len(pw), dtype=np.float64)
    for (sex, multigen), rate in fitted.coupling_child_pooled.items():
        coupled_probability[
            is_55_row & (sexes_row == sex) & (c1_multigen == multigen)
        ] = rate
    for (
        band,
        sex,
        multigen,
    ), rate in fitted.coupling_child_given_multigen.items():
        mask = (
            is_55_row
            & (bands_row == band)
            & (sexes_row == sex)
            & (c1_multigen == multigen)
        )
        coupled_probability[mask] = rate
    coupled_uniform = coupling_rng.random(len(pw))
    coupled_child = coupled_uniform < coupled_probability
    grandchild_coupled = c1_multigen & coupled_child & (~c1_parent)
    grandchild_final = np.where(
        is_55_row, grandchild_coupled, grandchild_composed
    )

    obs_skipgen = (
        pw["coresident_grandchild"].to_numpy(dtype=bool)[safe_row]
        & ~pw["multigen"].to_numpy(dtype=bool)[safe_row]
        & valid
    )
    skip_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    skip_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in fitted.skipgen_entry.items():
        skip_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in fitted.skipgen_exit.items():
        skip_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for ((lo, hi), sex), rate in fitted.skipgen_entry_age.items():
        mask = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sex)
        skip_entry_prob[mask] = rate
    for ((lo, hi), sex), rate in fitted.skipgen_exit_age.items():
        mask = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sex)
        skip_exit_prob[mask] = rate
    skipgen_state = evolve_two_state(
        valid, obs_skipgen, skip_entry_prob, skip_exit_prob, skipgen_rng
    )
    skipgen_row = np.zeros(len(pw), dtype=bool)
    skipgen_row[row_of[valid]] = skipgen_state[valid]
    coresident_grandchild = grandchild_final | skipgen_row

    nonfamily_count = sample_nonfamily_count(
        pw, fitted.nonfamily_count_by_core, nonfamily_rng, hh_size_base
    )
    hh_size = (hh_size_base + nonfamily_count).astype(np.int64)
    return {
        "pw": pw,
        "person_id": pw["person_id"].to_numpy(np.int64),
        "band": pw["band"].to_numpy(dtype=object),
        "sex": pw["sex"].to_numpy(),
        "age": pw["age"].to_numpy(np.int64),
        "weight": pw["weight"].to_numpy(np.float64),
        "spouse": spouse,
        "coresident_parent": c1_parent,
        "multigen": c1_multigen,
        "coresident_child": coresident_child,
        "coresident_grandchild": coresident_grandchild,
        "child_counts": child_counts.astype(np.int64),
        "hh_size": hh_size,
        "linked_diagnostics": linked_diag,
        "cohab_overlay_lift": COHAB_OVERLAY_LIFT,
    }
