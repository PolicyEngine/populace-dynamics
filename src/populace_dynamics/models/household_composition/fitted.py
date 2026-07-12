"""Flat fitted bundle selected by the candidate-9 component registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models.household_composition.components.parental_home_exit import (
    ParentalExitModel,
)


@dataclass(frozen=True)
class FittedHouseholdComposition:
    """All effective candidate-9 fits, without candidate-ladder nesting.

    The field set resolves the nested bundles declared in
    ``household_composition_sim.py:166-176`` and
    ``household_composition_sim_v2.py:91-110`` through
    ``household_composition_sim_v8.py:432-455`` into named component outputs.
    """

    family_transitions: ft.FittedFamilyTransitions
    male_gap: float
    parental_exit: ParentalExitModel
    child_exit_single_year: dict[tuple[int, str], float]
    multigen_entry: dict[tuple[str, str], float]
    multigen_exit: dict[tuple[str, str], float]
    coupling_child_given_multigen: dict[tuple[str, str, bool], float]
    coupling_child_pooled: dict[tuple[str, bool], float]
    cohab_flag: pd.DataFrame
    cohab_entry: dict[tuple[str, str], float]
    cohab_exit: dict[tuple[str, str], float]
    cohab_entry_age: dict[tuple[int, str], float]
    cohab_exit_age: dict[tuple[int, str], float]
    cohab_entry_age_female: dict[int, float]
    cohab_exit_age_female: dict[int, float]
    cohab_overlay_lift: float
    legal_residual_entry: dict[tuple[str, str], float]
    legal_residual_exit: dict[tuple[str, str], float]
    legal_residual_marginal: dict[tuple[str, str], float]
    legal_residual_target: dict[tuple[str, str], float]
    father_links: pd.DataFrame
    custodial_era: dict[tuple[int, str, str], float]
    custodial_age_marital: dict[tuple[int, str], float]
    custodial_band_marital: dict[tuple[str, str], float]
    custodial_overall: float
    custodial_child_record: dict[tuple[str, str], float]
    joinable_keys: frozenset[tuple[int, int]]
    linked_episode_persistence: float
    skipgen_entry: dict[tuple[str, str], float]
    skipgen_exit: dict[tuple[str, str], float]
    skipgen_entry_age: dict[tuple[tuple[int, int], str], float]
    skipgen_exit_age: dict[tuple[tuple[int, int], str], float]
    nonfamily_count_by_core: dict[int, tuple[np.ndarray, np.ndarray]]
    parent_count_two_share: dict[tuple[str, str], float]
    parent_count_two_pooled: float
    completed_size_dist_train: dict[tuple[str, str], dict[str, float]]
    completed_size_dist_train_all: dict[str, float]
    retention_link_shift: dict[str, float]
    delta3_fit: dict[str, Any] = field(default_factory=dict)
    component_meta: dict[str, Any] = field(default_factory=dict)
    implementation_ids: dict[str, str] = field(default_factory=dict)
