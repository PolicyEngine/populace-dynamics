"""Engine-side candidate-9 composition with authoritative marital state.

Candidate 9's certified standalone simulator runs candidate 16 twice.  M6 has
already resolved marital state at step 3, so this module ports the candidate-9
composition path and replaces both internal simulations with a
:class:`~populace_dynamics.engine.marital.MaritalStepResult`.  Every random
generator is supplied by the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
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
    custodial_linked_child_counts,
)
from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
    apply_overlay_lift,
)
from populace_dynamics.models.household_composition.components.fertility_core_lift import (
    apply_retention_link,
    apply_scoped_lift,
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
    GRANDCHILD_LO,
)
from populace_dynamics.models.household_composition.components.nonfamily_bridge import (
    sample_nonfamily_count,
)
from populace_dynamics.models.household_composition.components.parental_home_exit import (
    child_leave_years,
    child_leave_years_refit,
)
from populace_dynamics.models.household_composition.fitted import (
    FittedHouseholdComposition,
)

from .marital import MaritalStepResult
from .rng import ProjectionModule, ProjectionRNGRegistry
from .steps import FertilityDraws


@dataclass(frozen=True)
class CompositionRngs:
    """Every candidate-9 projection stream, named by its consumer."""

    occupancy: np.random.Generator
    child: np.random.Generator
    cohabitation: np.random.Generator
    legal_residual: np.random.Generator
    nonfamily: np.random.Generator
    skip_generation: np.random.Generator
    multigenerational_coupling: np.random.Generator
    parent_count: np.random.Generator
    maternal_leave: np.random.Generator
    linked_episode: np.random.Generator
    fertility_lift: np.random.Generator
    retention: np.random.Generator
    cohabitation_lift: np.random.Generator


def composition_rngs_from_registry(
    registry: ProjectionRNGRegistry, period: int
) -> CompositionRngs:
    """Resolve candidate-9's named streams from the M6 registry.

    At period zero this reproduces the certified component tag/spawn topology;
    later periods retain those tags below a distinct period/module stream.
    """
    module = ProjectionModule.HOUSEHOLD_COMPOSITION
    child = registry.tagged_child_generator
    tagged = registry.tagged_generator
    return CompositionRngs(
        occupancy=tagged(period, module, 0xB2B),
        child=child(period, module, 0xC2, 0),
        cohabitation=child(period, module, 0xC2, 1),
        legal_residual=child(period, module, 0xC4, 0),
        nonfamily=child(period, module, 0xC3, 1),
        skip_generation=child(period, module, 0xC3, 2),
        multigenerational_coupling=child(period, module, 0xC5, 0),
        parent_count=child(period, module, 0xC5, 1),
        maternal_leave=tagged(period, module, 0xC6),
        linked_episode=tagged(period, module, 0xC7),
        fertility_lift=child(period, module, 0xC8, 0),
        retention=child(period, module, 0xC8, 1),
        cohabitation_lift=child(period, module, 0xC8, 2),
    )


@dataclass(frozen=True)
class CompositionDiagnostics:
    """Latent marital-consuming channels aligned to output person-waves."""

    weight: np.ndarray
    legal_core: np.ndarray
    cohabitation_state: np.ndarray
    cohabitation_increment: np.ndarray
    legal_residual_state: np.ndarray
    legal_residual_increment: np.ndarray
    final_spouse: np.ndarray
    coresident_parent: np.ndarray
    multigen: np.ndarray
    coresident_child: np.ndarray
    coresident_grandchild: np.ndarray
    household_size: np.ndarray
    model_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RecertificationCell:
    """One pre-named injected-vs-internal distributional margin."""

    channel_set: str
    cell: str
    injected_mean: float
    internal_mean: float
    absolute_delta: float
    sigma_of_mean_difference: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class RecertificationResult:
    """The complete targeted candidate-9 transfer check."""

    sigma_multiplier: float
    cells: tuple[RecertificationCell, ...]

    @property
    def passed(self) -> bool:
        return all(cell.passed for cell in self.cells)


RECERTIFICATION_CHANNEL_SETS: dict[str, tuple[str, ...]] = {
    "cohabitation": (
        "cohabitation_state",
        "cohabitation_increment",
    ),
    "legal_spouse_residual": (
        "legal_core",
        "legal_residual_state",
        "legal_residual_increment",
        "final_spouse",
    ),
    "occupancy": (
        "coresident_parent",
        "multigen",
        "coresident_child",
        "coresident_grandchild",
    ),
    "household_size": (
        "household_size.1",
        "household_size.2",
        "household_size.3",
        "household_size.4",
        "household_size.5+",
    ),
}


def _attach_cohabitation_seed(
    person_waves: pd.DataFrame, fitted: FittedHouseholdComposition
) -> pd.DataFrame:
    """Prefer the harness's realized-anchor state when it is injected.

    The certified standalone path still obtains cohabitation from its fitted
    sparse flag.  The M6 panel builder supplies a realized anchor value on
    every support row; merging the cutoff flag over it would both discard that
    registered initial condition and create ambiguous ``_x``/``_y`` columns.
    """
    out = person_waves.copy()
    if "cohabiting" not in out:
        out = out.merge(
            fitted.cohab_flag, on=["person_id", "year"], how="left"
        )
    out["cohabiting"] = out["cohabiting"].fillna(False).astype(bool)
    return out


def _carried_c1_states_injected(
    hh: hc.HouseholdCompositionPanel,
    fitted: FittedHouseholdComposition,
    holdout_ids: set[int],
    marital: MaritalStepResult,
    occupancy_rng: np.random.Generator,
    fertility: FertilityDraws | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate-1 carried states with step-3 state replacing ft.simulate."""
    sim_years = marital.sim_years
    side_a_pw = (
        hh.person_waves[hh.person_waves["person_id"].isin(holdout_ids)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )
    spouse = spouse_from_marital(side_a_pw, sim_years)
    maternal_source = (
        marital.births if fertility is None else fertility.maternal
    )
    maternal = maternal_source[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    if fertility is None:
        lookup, decade_map = build_fertility_lookup(
            fitted.family_transitions.fertility
        )
        paternal = paternal_births(
            sim_years,
            marital.panel.attrs,
            fitted.male_gap,
            lookup,
            decade_map,
            occupancy_rng,
        )
    else:
        paternal = fertility.paternal[
            ["parent_person_id", "birth_year"]
        ].copy()
    all_births = pd.concat([maternal, paternal], ignore_index=True)
    all_births = all_births[all_births["parent_person_id"].isin(holdout_ids)]
    # Candidate 9 retains these otherwise discarded draws.
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
        entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in fitted.multigen_exit.items():
        exitm_prob[(band_mat == band) & (sex_mat == sex)] = rate
    multigen_state = evolve_two_state(
        valid, obs_multigen, entry_prob, exitm_prob, occupancy_rng
    )
    parent_row = np.zeros(len(pw), dtype=bool)
    multigen_row = np.zeros(len(pw), dtype=bool)
    parent_row[row_of[valid]] = parent_state[valid]
    multigen_row[row_of[valid]] = multigen_state[valid]
    return spouse, parent_row, multigen_row


def _compose_base_injected(
    hh: hc.HouseholdCompositionPanel,
    fitted: FittedHouseholdComposition,
    holdout_ids: set[int],
    marital: MaritalStepResult,
    rngs: CompositionRngs,
    fertility: FertilityDraws | None,
) -> dict[str, Any]:
    legal_spouse, c1_parent, c1_multigen = _carried_c1_states_injected(
        hh,
        fitted,
        holdout_ids,
        marital,
        rngs.occupancy,
        fertility,
    )
    side_a_pw = (
        hh.person_waves[hh.person_waves["person_id"].isin(holdout_ids)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )
    side_a_pw = _attach_cohabitation_seed(side_a_pw, fitted)

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
        rngs.cohabitation,
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]

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
        rngs.legal_residual.random(n_persons) < lr_marginal_prob[:, 0]
    ) & valid[:, 0]
    lr_state = evolve_two_state(
        valid,
        lr_initial,
        lr_entry_prob,
        lr_exit_prob,
        rngs.legal_residual,
    )
    lr_row = np.zeros(len(pw), dtype=bool)
    lr_row[row_of[valid]] = lr_state[valid]
    spouse = legal_spouse | cohab_row | lr_row

    sim_years = marital.sim_years
    maternal_source = (
        marital.births if fertility is None else fertility.maternal
    )
    maternal = maternal_source[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"
    men_a = marital.panel.attrs[marital.panel.attrs["sex"] == "male"]
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
    if fertility is None:
        lookup, decade_map = build_fertility_lookup(
            fitted.family_transitions.fertility
        )
        paternal_shadow = paternal_births(
            sim_years,
            unlinked_men,
            fitted.male_gap,
            lookup,
            decade_map,
            rngs.child,
        )
    else:
        paternal_shadow = fertility.paternal[
            fertility.paternal["parent_person_id"].isin(
                set(unlinked_men["person_id"])
            )
        ][["parent_person_id", "birth_year"]].copy()
    paternal_shadow["_source"] = "shadow"
    all_births = pd.concat(
        [maternal, paternal_linked, paternal_shadow], ignore_index=True
    )
    all_births = all_births[all_births["parent_person_id"].isin(holdout_ids)]
    base_leaves = child_leave_years(
        all_births, fitted.parental_exit, rngs.child
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    maternal_births = all_births[all_births["_source"] == "maternal"]
    maternal_leaves = child_leave_years_refit(
        maternal_births,
        fitted.parental_exit,
        fitted.child_exit_single_year,
        rngs.maternal_leave,
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
        rngs.linked_episode,
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
    parent_uniform = rngs.parent_count.random(len(pw))
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
    coupled_uniform = rngs.multigenerational_coupling.random(len(pw))
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
        valid,
        obs_skipgen,
        skip_entry_prob,
        skip_exit_prob,
        rngs.skip_generation,
    )
    skipgen_row = np.zeros(len(pw), dtype=bool)
    skipgen_row[row_of[valid]] = skipgen_state[valid]
    coresident_grandchild = grandchild_final | skipgen_row

    nonfamily_count = sample_nonfamily_count(
        pw,
        fitted.nonfamily_count_by_core,
        rngs.nonfamily,
        hh_size_base,
    )
    hh_size = (hh_size_base + nonfamily_count).astype(np.int64)
    return {
        "pw": pw,
        "person_id": pw["person_id"].to_numpy(np.int64),
        "band": bands_row,
        "sex": sexes_row,
        "weight": pw["weight"].to_numpy(np.float64),
        "legal_core": legal_spouse,
        "cohabitation_state": cohab_row,
        "cohabitation_increment": cohab_row & (~legal_spouse),
        "legal_residual_state": lr_row,
        "legal_residual_increment": lr_row & (~legal_spouse) & (~cohab_row),
        "spouse": spouse,
        "coresident_parent": c1_parent,
        "multigen": c1_multigen,
        "coresident_child": coresident_child,
        "coresident_grandchild": coresident_grandchild,
        "child_counts": child_counts.astype(np.int64),
        "hh_size": hh_size,
        "linked_diagnostics": linked_diag,
    }


def simulate_candidate9_injected(
    hh: hc.HouseholdCompositionPanel,
    fitted: FittedHouseholdComposition,
    holdout_ids: set[int],
    marital: MaritalStepResult,
    rngs: CompositionRngs,
    fertility: FertilityDraws | None = None,
) -> tuple[hc.HouseholdCompositionPanel, CompositionDiagnostics]:
    """Simulate candidate 9 using step-3 state and engine generators."""
    composition = _compose_base_injected(
        hh, fitted, holdout_ids, marital, rngs, fertility
    )
    pw = composition["pw"]
    band = composition["band"]
    sex = composition["sex"]
    weight = composition["weight"]

    core_after_lift, hh_after_lift, fertility_diag = apply_scoped_lift(
        composition["person_id"],
        band,
        sex,
        weight,
        composition["child_counts"],
        composition["coresident_child"],
        composition["hh_size"],
        fitted.completed_size_dist_train,
        rngs.fertility_lift,
    )
    core_after_retention, retention_diag = apply_retention_link(
        band,
        sex,
        weight,
        core_after_lift,
        fitted.retention_link_shift,
        rngs.retention,
    )
    spouse_after_lift, cohab_diag = apply_overlay_lift(
        band,
        sex,
        weight,
        composition["spouse"],
        fitted.cohab_overlay_lift,
        rngs.cohabitation_lift,
    )

    sim_pw = pw.copy()
    sim_pw["coresident_spouse"] = spouse_after_lift
    sim_pw["coresident_parent"] = composition["coresident_parent"]
    sim_pw["coresident_child"] = core_after_retention
    sim_pw["coresident_grandchild"] = composition["coresident_grandchild"]
    sim_pw["multigen"] = composition["multigen"]
    sim_pw["hh_size"] = hh_after_lift
    sim_pw = sim_pw.drop(
        columns=[
            "has_next",
            "next_coresident_parent",
            "next_coresident_spouse",
            "next_multigen",
            "cohabiting",
        ]
    )
    sim_pw = hc._add_transitions(sim_pw)
    attrs = sim_pw[["person_id"]].drop_duplicates().reset_index(drop=True)
    panel = hc.HouseholdCompositionPanel(person_waves=sim_pw, attrs=attrs)
    diagnostics = CompositionDiagnostics(
        weight=weight,
        legal_core=composition["legal_core"],
        cohabitation_state=composition["cohabitation_state"],
        cohabitation_increment=composition["cohabitation_increment"],
        legal_residual_state=composition["legal_residual_state"],
        legal_residual_increment=composition["legal_residual_increment"],
        final_spouse=spouse_after_lift,
        coresident_parent=composition["coresident_parent"],
        multigen=composition["multigen"],
        coresident_child=core_after_retention,
        coresident_grandchild=composition["coresident_grandchild"],
        household_size=hh_after_lift,
        model_diagnostics={
            "linked_child": composition["linked_diagnostics"],
            "scoped_fertility_core_lift": fertility_diag,
            "retention_link_refit": retention_diag,
            "cohab_overlay_lift": cohab_diag,
        },
    )
    return panel, diagnostics


def simulate_candidate9_internal_reference(
    hh: hc.HouseholdCompositionPanel,
    mpanel: Any,
    fitted: FittedHouseholdComposition,
    holdout_ids: set[int],
    draw_seed: int,
) -> tuple[hc.HouseholdCompositionPanel, CompositionDiagnostics]:
    """Expose candidate-9's internal-marital reference for re-certification.

    This is not a projection path.  It deliberately invokes the frozen
    candidate-16 simulator, then measures the same latent output channels as
    the injected adapter so the targeted transfer test can compare the two
    distributions on one complete surface.
    """
    if draw_seed < 5200:
        raise ValueError("internal reference draw_seed must be at least 5200")
    sim_panel, births = ft.simulate(
        mpanel, holdout_ids, fitted.family_transitions, draw_seed
    )
    marital = MaritalStepResult(
        sim_years=sim_panel.person_years,
        births=births,
        panel=sim_panel,
    )
    registry = ProjectionRNGRegistry(
        draw_index=draw_seed - 5200,
        n_periods=0,
    )
    return simulate_candidate9_injected(
        hh,
        fitted,
        holdout_ids,
        marital,
        composition_rngs_from_registry(registry, 0),
    )


def composition_channel_moments(
    diagnostics: CompositionDiagnostics,
) -> dict[str, float]:
    """Return the pre-named weighted margins used for re-certification."""
    weight = np.asarray(diagnostics.weight, dtype=np.float64)
    total = float(weight.sum())
    if total <= 0:
        raise ValueError("composition diagnostics have no positive weight")

    def share(values: np.ndarray) -> float:
        return float(
            (weight * np.asarray(values, dtype=np.float64)).sum() / total
        )

    moments = {
        name: share(getattr(diagnostics, name))
        for name in (
            "cohabitation_state",
            "cohabitation_increment",
            "legal_core",
            "legal_residual_state",
            "legal_residual_increment",
            "final_spouse",
            "coresident_parent",
            "multigen",
            "coresident_child",
            "coresident_grandchild",
        )
    }
    size = np.asarray(diagnostics.household_size, dtype=np.int64)
    for value in (1, 2, 3, 4):
        moments[f"household_size.{value}"] = share(size == value)
    moments["household_size.5+"] = share(size >= 5)
    return moments


def check_candidate9_recertification(
    injected: Sequence[CompositionDiagnostics],
    internal: Sequence[CompositionDiagnostics],
    *,
    sigma_multiplier: float = 3.0,
) -> RecertificationResult:
    """Check every named channel at three standard errors of the mean delta.

    This is deliberately strict and has no tolerance floor.  A zero-variance
    channel must match exactly.  Any failed channel raises immediately because
    the verified design sends a failed targeted check to fuller re-ceremony.
    """
    if len(injected) != len(internal) or len(injected) < 2:
        raise ValueError(
            "re-certification requires equally sized ensembles of at least "
            "two draws"
        )
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be positive")
    injected_moments = [composition_channel_moments(row) for row in injected]
    internal_moments = [composition_channel_moments(row) for row in internal]
    expected_cells = {
        cell
        for cells in RECERTIFICATION_CHANNEL_SETS.values()
        for cell in cells
    }
    if set(injected_moments[0]) != expected_cells:
        raise RuntimeError("re-certification moment surface is incomplete")

    cells: list[RecertificationCell] = []
    failures: list[str] = []
    n = len(injected_moments)
    for channel_set, names in RECERTIFICATION_CHANNEL_SETS.items():
        for name in names:
            inj = np.array([row[name] for row in injected_moments])
            ref = np.array([row[name] for row in internal_moments])
            injected_mean = float(inj.mean())
            internal_mean = float(ref.mean())
            delta = abs(injected_mean - internal_mean)
            sigma = float(np.sqrt(inj.var(ddof=1) / n + ref.var(ddof=1) / n))
            tolerance = float(sigma_multiplier * sigma)
            passed = bool(delta <= tolerance)
            cells.append(
                RecertificationCell(
                    channel_set=channel_set,
                    cell=name,
                    injected_mean=injected_mean,
                    internal_mean=internal_mean,
                    absolute_delta=delta,
                    sigma_of_mean_difference=sigma,
                    tolerance=tolerance,
                    passed=passed,
                )
            )
            if not passed:
                failures.append(
                    f"{channel_set}/{name}: delta={delta:.8g} > "
                    f"{sigma_multiplier:g}sigma={tolerance:.8g}"
                )
    result = RecertificationResult(
        sigma_multiplier=float(sigma_multiplier), cells=tuple(cells)
    )
    if failures:
        raise AssertionError(
            "candidate-9 injected-state re-certification failed; fuller "
            "re-ceremony required: " + "; ".join(failures)
        )
    return result
