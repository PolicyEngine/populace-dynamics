"""M6 annual projection engine and cutoff-refit adapters."""

from populace_dynamics.engine.assembly import (
    CertifiedEngineInputs,
    assemble_period_modules,
)
from populace_dynamics.engine.composition import (
    CompositionDiagnostics,
    CompositionRngs,
    RecertificationResult,
    check_candidate9_recertification,
    composition_rngs_from_registry,
    simulate_candidate9_injected,
    simulate_candidate9_internal_reference,
)
from populace_dynamics.engine.disability import simulate_reproduction
from populace_dynamics.engine.earnings_domain import (
    EarningsDomainAdapter,
    earnings_domain_person_ids,
)
from populace_dynamics.engine.forward_earnings import (
    ForwardEarningsGenerator,
    fit_forward_earnings,
)
from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodContext,
    PeriodModules,
    ProjectionEngine,
    ProjectionResult,
    SyntheticPersonIdAllocator,
)
from populace_dynamics.engine.marital import (
    simulate_marital_step,
    simulate_maternal_births,
)
from populace_dynamics.engine.refit import (
    BOUNDARY_YEAR,
    M6RefitBundle,
    M6RefitInputs,
    prepare_m6_preflight_context,
    refit_m6_components,
)
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.engine.steps import (
    AgeSexMortalityModel,
    ClaimingSchedule,
    FertilityDraws,
)
from populace_dynamics.engine.support import (
    EvaluationMode,
    PresenceBasis,
    StartWaveWeightSnapshot,
    prepare_evaluation_support,
)

__all__ = [
    "BOUNDARY_YEAR",
    "AgeSexMortalityModel",
    "CertifiedEngineInputs",
    "ClaimingSchedule",
    "CompositionDiagnostics",
    "CompositionRngs",
    "EvaluationMode",
    "EarningsDomainAdapter",
    "FertilityDraws",
    "ForwardEarningsGenerator",
    "M6RefitBundle",
    "M6RefitInputs",
    "MaritalStepResult",
    "PeriodContext",
    "PeriodModules",
    "PresenceBasis",
    "ProjectionEngine",
    "ProjectionModule",
    "ProjectionRNGRegistry",
    "ProjectionResult",
    "RecertificationResult",
    "StartWaveWeightSnapshot",
    "SyntheticPersonIdAllocator",
    "assemble_period_modules",
    "check_candidate9_recertification",
    "composition_rngs_from_registry",
    "earnings_domain_person_ids",
    "prepare_evaluation_support",
    "prepare_m6_preflight_context",
    "fit_forward_earnings",
    "refit_m6_components",
    "simulate_candidate9_injected",
    "simulate_candidate9_internal_reference",
    "simulate_marital_step",
    "simulate_maternal_births",
    "simulate_reproduction",
]
