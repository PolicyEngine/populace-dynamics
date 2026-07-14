"""Synthetic tests for the two M6 pre-scoring checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from populace_dynamics.engine.composition import (
    CompositionDiagnostics,
    RecertificationResult,
)
from populace_dynamics.harness import m6_preflight


def _diagnostics(value: bool) -> CompositionDiagnostics:
    binary = np.asarray([value, not value], dtype=bool)
    return CompositionDiagnostics(
        weight=np.ones(2),
        legal_core=binary,
        cohabitation_state=binary,
        cohabitation_increment=binary,
        legal_residual_state=binary,
        legal_residual_increment=binary,
        final_spouse=binary,
        coresident_parent=binary,
        multigen=binary,
        coresident_child=binary,
        coresident_grandchild=binary,
        household_size=np.asarray([1, 5]),
        model_diagnostics={},
    )


def test_candidate9_preflight_uses_period_zero_for_both_paths(monkeypatch):
    marital_periods = []
    composition_periods = []
    internal_seeds = []

    def fake_marital(*args, main_rng, gap_rng):
        del args
        marital_periods.append((main_rng.random(), gap_rng.random()))
        return object()

    def fake_rngs(registry, period):
        del registry
        composition_periods.append(period)
        return object()

    def fake_injected(*args):
        del args
        return object(), _diagnostics(True)

    def fake_internal(*args):
        internal_seeds.append(args[-1])
        return object(), _diagnostics(True)

    monkeypatch.setattr(m6_preflight, "simulate_marital_step", fake_marital)
    monkeypatch.setattr(
        m6_preflight, "composition_rngs_from_registry", fake_rngs
    )
    monkeypatch.setattr(
        m6_preflight, "simulate_candidate9_injected", fake_injected
    )
    monkeypatch.setattr(
        m6_preflight,
        "simulate_candidate9_internal_reference",
        fake_internal,
    )

    result = m6_preflight.run_candidate9_recertification(
        m6_preflight.Candidate9PreflightInputs(
            marital_panel=object(),  # type: ignore[arg-type]
            household_panel=object(),  # type: ignore[arg-type]
            holdout_ids={1},
            family=object(),
            modifier=object(),
            permanent_axis=object(),
            household=object(),
        ),
        draw_indices=(0, 1),
    )

    assert isinstance(result, RecertificationResult)
    assert result.passed
    assert composition_periods == [0, 0]
    assert internal_seeds == [5200, 5201]
    assert len(marital_periods) == 2


class _ExternalGate:
    """The draw_sign TEST SEAM double (design amendment 3c: must be REJECTED)."""

    def draw_sign(self, current_level, target_age, uniforms):
        del current_level, target_age
        return (uniforms < 0.5).astype(np.int64)


class _CertifiedInnerGate:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, values):
        return np.tile(
            np.asarray([[0.5, 0.5]], dtype=np.float64), (len(values), 1)
        )


class _CertifiedModel:
    columns = ("earnings", "age_tp2")
    gate = _CertifiedInnerGate()


class _CertifiedGate:
    """A _target_models reconstruction double (the CERTIFIED branch)."""

    def __init__(self):
        self._target_models = {"earnings_tp2": _CertifiedModel()}


def test_sign_path_records_certified_branch_on_synthetic_probe():
    record = m6_preflight.verify_external_sign_path(
        SimpleNamespace(
            shared_gate=_CertifiedGate(), zero_anchor_gate=_CertifiedGate()
        )
    )

    assert record.branch == "certified_target_models_reconstruction"
    assert record.gates_checked == ("shared_gate", "zero_anchor_gate")
    assert record.probe_rows == 2
    # uniforms [0.25, 0.75] against cumsum [0.5, 1.0] -> classes 0, 1.
    assert record.output_signs == (0, 1, 0, 1)


def test_sign_path_rejects_draw_sign_test_seam():
    with pytest.raises(RuntimeError, match="_target_models"):
        m6_preflight.verify_external_sign_path(
            SimpleNamespace(shared_gate=_ExternalGate(), zero_anchor_gate=None)
        )


def test_sign_path_rejects_gate_without_reconstruction():
    bare = SimpleNamespace()
    with pytest.raises(RuntimeError, match="_target_models"):
        m6_preflight.verify_external_sign_path(
            SimpleNamespace(shared_gate=bare, zero_anchor_gate=None)
        )
