"""Reader-free tests for the candidate-3 production input-plan adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from populace_dynamics.harness.m6_candidate2_runner import (
    M6Candidate2InputPlan,
)
from populace_dynamics.harness.m6_candidate3_runner import (
    M6Candidate3InputPlan,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import registered_m6_candidate3_inputs as candidate3_factory  # noqa: E402


def test_candidate3_adapter_preserves_the_frozen_candidate2_plan(monkeypatch):
    fit_inputs = object()
    calls = []

    def load_full_inputs():
        calls.append("load")
        return object()

    inherited = M6Candidate2InputPlan(
        fit_inputs=fit_inputs,
        load_full_inputs=load_full_inputs,
    )
    monkeypatch.setattr(
        candidate3_factory.inherited,
        "build_input_plan",
        lambda: inherited,
    )

    plan = candidate3_factory.build_input_plan()

    assert isinstance(plan, M6Candidate3InputPlan)
    assert plan.fit_inputs is fit_inputs
    assert plan.load_full_inputs is load_full_inputs
    assert calls == []
