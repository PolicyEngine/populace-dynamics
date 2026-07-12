"""Flattened candidate-9 registry simulator.

The final C8 substream spawn, delta order, and panel build are copied from
``household_composition_sim_v9.py:262-347``.  The candidate-7 composition is
the flattened copy in :mod:`.base_simulator`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models.household_composition.base_simulator import (
    compose_base,
)
from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
    apply_overlay_lift,
)
from populace_dynamics.models.household_composition.components.fertility_core_lift import (
    DELTA_STREAM_TAG_V8,
    apply_retention_link,
    apply_scoped_lift,
)
from populace_dynamics.models.household_composition.fitted import (
    FittedHouseholdComposition,
)


def simulate_with_diagnostics(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    fitted: FittedHouseholdComposition,
    holdout_ids: set[int],
    draw_seed: int,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-9 draw and return diagnostics.

    Copied from ``household_composition_sim_v9.py:262-347``.
    """
    composition = compose_base(hh, mpanel, fitted, holdout_ids, draw_seed)
    pw = composition["pw"]
    band = composition["band"]
    sex = composition["sex"]
    weight = composition["weight"]

    c8_sequence = np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V8])
    fertility_sequence, retention_sequence, cohab_sequence = c8_sequence.spawn(
        3
    )
    fertility_rng = np.random.default_rng(fertility_sequence)
    retention_rng = np.random.default_rng(retention_sequence)
    cohab_rng = np.random.default_rng(cohab_sequence)

    core_after_lift, hh_after_lift, fertility_diag = apply_scoped_lift(
        composition["person_id"],
        band,
        sex,
        weight,
        composition["child_counts"],
        composition["coresident_child"],
        composition["hh_size"],
        fitted.completed_size_dist_train,
        fertility_rng,
    )
    core_after_retention, retention_diag = apply_retention_link(
        band,
        sex,
        weight,
        core_after_lift,
        fitted.retention_link_shift,
        retention_rng,
    )
    spouse_after_lift, cohab_diag = apply_overlay_lift(
        band,
        sex,
        weight,
        composition["spouse"],
        fitted.cohab_overlay_lift,
        cohab_rng,
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
    return panel, {
        "linked_persistence_rho": float(fitted.linked_episode_persistence),
        "scoped_fertility_core_lift": fertility_diag,
        "retention_link_refit": retention_diag,
        "cohab_overlay_lift": cohab_diag,
        "delta_stream_tag_v8": DELTA_STREAM_TAG_V8,
    }


def simulate(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    fitted: FittedHouseholdComposition,
    holdout_ids: set[int],
    draw_seed: int,
) -> hc.HouseholdCompositionPanel:
    """Simulate one registry-driven candidate-9 household draw.

    The public adapter wraps the copied candidate-9 draw at
    ``household_composition_sim_v9.py:262-347``.
    """
    panel, _diagnostics = simulate_with_diagnostics(
        hh, mpanel, fitted, holdout_ids, draw_seed
    )
    return panel
