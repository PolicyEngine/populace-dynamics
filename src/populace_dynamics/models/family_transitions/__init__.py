"""Flattened family-transition component registry."""

from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)
from populace_dynamics.models.family_transitions.registry import (
    CANDIDATE_16,
    REGISTRY,
    CandidateSpec,
    ComponentRef,
    ComponentRegistry,
    FitContext,
)
from populace_dynamics.models.family_transitions.simulator import simulate

__all__ = [
    "CANDIDATE_16",
    "REGISTRY",
    "CandidateSpec",
    "ComponentRef",
    "ComponentRegistry",
    "FitContext",
    "FittedFamilyTransitions",
    "simulate",
]
