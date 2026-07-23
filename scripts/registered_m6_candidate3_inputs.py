"""Production input-plan adapter for the registered M6 candidate-3 run.

Candidate 3 changes only the forward-earnings refresh operation.  All
train-only readers, field-date fences, SSA identity checks, and deferred
full-window construction therefore reuse the candidate-2 production factory
without modification.  This adapter preserves those exact objects while
binding the candidate-3 runner's explicit plan type.
"""

from __future__ import annotations

import registered_m6_candidate2_inputs as inherited

from populace_dynamics.harness.m6_candidate3_runner import (
    M6Candidate3InputPlan,
)


def build_input_plan() -> M6Candidate3InputPlan:
    """Wrap the inherited frozen input plan without triggering its callback."""
    plan = inherited.build_input_plan()
    return M6Candidate3InputPlan(
        fit_inputs=plan.fit_inputs,
        load_full_inputs=plan.load_full_inputs,
    )
