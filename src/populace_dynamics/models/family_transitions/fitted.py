"""Explicit fitted bundle for the candidate-16 family-transition model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from populace_dynamics.models.family_transitions.components.first_marriage import (
    FirstMarriageModel,
)
from populace_dynamics.models.family_transitions.components.initial_states import (
    ObservedInitialStates,
)
from populace_dynamics.models.family_transitions.components.widowhood import (
    WidowhoodModel,
)

__all__ = ["FittedFamilyTransitions"]


@dataclass(frozen=True)
class FittedFamilyTransitions:
    """Every resolved candidate-16 component with no dynamic attributes.

    This replaces the progressively mutated ``Components`` object ultimately
    assembled in ``scripts/run_gate2_candidate16.py:645-713``.
    """

    first_marriage: FirstMarriageModel
    divorce: np.ndarray
    widowhood: WidowhoodModel
    remarriage: dict[tuple[int, int, str, str], float]
    fertility: dict[tuple[int, int, int], float]
    initial_states: ObservedInitialStates
    spousal_age_gaps: dict[str, dict[int, np.ndarray]]
    implementation_ids: dict[str, str]
