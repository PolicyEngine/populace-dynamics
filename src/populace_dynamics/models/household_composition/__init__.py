"""Flattened household-composition component registry."""

from populace_dynamics.models.household_composition.data import load_sources
from populace_dynamics.models.household_composition.fitted import (
    FittedHouseholdComposition,
)
from populace_dynamics.models.household_composition.registry import (
    CANDIDATE_9,
    REGISTRY,
    FitContext,
    fit_context_from_sources,
)
from populace_dynamics.models.household_composition.simulator import (
    simulate,
    simulate_with_diagnostics,
)

__all__ = [
    "CANDIDATE_9",
    "REGISTRY",
    "FitContext",
    "FittedHouseholdComposition",
    "fit_context_from_sources",
    "load_sources",
    "simulate",
    "simulate_with_diagnostics",
]
