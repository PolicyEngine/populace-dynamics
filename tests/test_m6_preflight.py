"""Synthetic tests for the two M6 pre-scoring checks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.engine.composition import (
    CompositionDiagnostics,
    RecertificationCell,
    RecertificationResult,
)
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
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


# ---------------------------------------------------------------------------
# Pre-flight 1 fertility-wiring regression fixtures (issue #42, reg-4 abort).
#
# The certified re-certification aborts when the injected arm carries a
# different maternal-birth line than the internal reference.  These fixtures
# make the injected arm's household-composition channels a *function of the
# maternal births the fertility argument supplies*, so a fixture where the
# injected arm loses its maternal line breaks re-certification -- unlike the
# pre-existing symmetric fixture whose two arms returned identical diagnostics
# regardless of fertility and so passed even with the wiring defect present.
# ---------------------------------------------------------------------------

_PREFLIGHT_PERSONS = 12
# Distinct per-draw maternal counts give the ensemble non-zero variance so the
# 3-sigma band is exercised (not a degenerate zero-variance exact match).
_MATERNAL_COUNTS = (6, 7, 5)


def _diag_from_maternal(
    n_maternal: int, n_persons: int = _PREFLIGHT_PERSONS
) -> CompositionDiagnostics:
    """Build composition diagnostics whose child/size channels track births.

    ``coresident_child`` (and therefore the household-size mix) rises with the
    number of maternal births; every other channel is a fixed spine so only
    the fertility-driven channels can diverge between arms.
    """
    idx = np.arange(n_persons)
    coresident_child = idx < n_maternal
    household_size = np.where(coresident_child, 3, 2).astype(np.int64)
    spine = idx % 2 == 0
    off = np.zeros(n_persons, dtype=bool)
    return CompositionDiagnostics(
        weight=np.ones(n_persons),
        legal_core=spine,
        cohabitation_state=spine,
        cohabitation_increment=off,
        legal_residual_state=spine,
        legal_residual_increment=off,
        final_spouse=spine,
        coresident_parent=off,
        multigen=off,
        coresident_child=coresident_child,
        coresident_grandchild=off,
        household_size=household_size,
        model_diagnostics={},
    )


def _maternal_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_person_id": np.arange(n_rows, dtype=np.int64),
            "birth_year": np.full(n_rows, 2000, dtype=np.int64),
        }
    )


def _preflight_inputs(**overrides):
    base = dict(
        marital_panel=object(),
        household_panel=object(),
        holdout_ids={1, 2, 3},
        family=object(),
        modifier=object(),
        permanent_axis=object(),
        household=SimpleNamespace(male_gap=-2.375),
    )
    base.update(overrides)
    return m6_preflight.Candidate9PreflightInputs(**base)  # type: ignore[arg-type]


def test_candidate9_preflight_uses_period_zero_for_both_paths(monkeypatch):
    marital_periods = []
    composition_periods = []
    internal_seeds = []

    def fake_marital(*args, main_rng, gap_rng):
        del args
        marital_periods.append((main_rng.random(), gap_rng.random()))
        return SimpleNamespace(births=_maternal_frame(0))

    def fake_fertility(marital, components, holdout_ids, male_gap, rng):
        del marital, components, holdout_ids, male_gap
        rng.random()  # consume the (period 0, FERTILITY) stream
        return SimpleNamespace(
            maternal=_maternal_frame(1), paternal=_maternal_frame(0)
        )

    def fake_rngs(registry, period):
        del registry
        composition_periods.append(period)
        return object()

    def fake_injected(*args, fertility=None):
        del args
        assert fertility is not None
        return object(), _diagnostics(True)

    registered_family = object()

    def fake_internal(*args, registered_family=None):
        internal_seeds.append((args[-1], registered_family))
        return object(), _diagnostics(True)

    monkeypatch.setattr(m6_preflight, "simulate_marital_step", fake_marital)
    monkeypatch.setattr(m6_preflight, "simulate_fertility", fake_fertility)
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
        _preflight_inputs(
            registered_reference_family=registered_family,
        ),
        draw_indices=(0, 1),
    )

    assert isinstance(result, RecertificationResult)
    assert result.passed
    assert composition_periods == [0, 0]
    assert internal_seeds == [
        (5200, registered_family),
        (5201, registered_family),
    ]
    assert len(marital_periods) == 2


def test_candidate1_success_payload_bytes_remain_unchanged():
    result = RecertificationResult(
        sigma_multiplier=3.0,
        cells=(
            RecertificationCell(
                channel_set="cohabitation",
                cell="cohabitation_state",
                injected_mean=0.25,
                internal_mean=0.25,
                absolute_delta=0.0,
                sigma_of_mean_difference=0.01,
                tolerance=0.03,
                passed=True,
            ),
        ),
    )

    payload = m6_preflight.recertification_payload(result)
    actual = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert actual == (
        b'{"cells":[{"absolute_delta":0.0,'
        b'"cell":"cohabitation_state",'
        b'"channel_set":"cohabitation",'
        b'"injected_mean":0.25,"internal_mean":0.25,'
        b'"passed":true,"sigma_of_mean_difference":0.01,'
        b'"tolerance":0.03}],"passed":true,'
        b'"sigma_multiplier":3.0}'
    )
    assert "signed_delta" not in payload["cells"][0]


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


def _install_recertification_fakes(
    monkeypatch,
    *,
    fake_fertility,
    injected_from_fertility,
    internal_counts,
):
    """Wire the pre-flight seams for a fertility-sensitive re-certification.

    ``injected_from_fertility`` maps the fertility passed to the injected core
    (and the injected marital births) to a maternal count; the internal arm
    always carries ``internal_counts`` maternal births per draw.
    """
    internal_counter = {"i": 0}

    def fake_marital(*args, main_rng, gap_rng):
        del args, main_rng, gap_rng
        return SimpleNamespace(births=_maternal_frame(0))

    def fake_injected(hh, fitted, ids, marital, rngs, fertility=None):
        del hh, fitted, ids, rngs
        return object(), _diag_from_maternal(
            injected_from_fertility(fertility, marital)
        )

    def fake_internal(*args, registered_family=None):
        del args, registered_family
        count = internal_counts[internal_counter["i"]]
        internal_counter["i"] += 1
        return object(), _diag_from_maternal(count)

    monkeypatch.setattr(m6_preflight, "simulate_marital_step", fake_marital)
    monkeypatch.setattr(m6_preflight, "simulate_fertility", fake_fertility)
    monkeypatch.setattr(
        m6_preflight,
        "composition_rngs_from_registry",
        lambda registry, period: object(),
    )
    monkeypatch.setattr(
        m6_preflight, "simulate_candidate9_injected", fake_injected
    )
    monkeypatch.setattr(
        m6_preflight,
        "simulate_candidate9_internal_reference",
        fake_internal,
    )


def test_preflight_injected_arm_carries_pinned_maternal_line(monkeypatch):
    """The patched wiring feeds the injected arm a non-empty maternal line.

    Discriminating fixture: the injected arm's household-composition channels
    are a function of the maternal births supplied via ``fertility=``.  When
    the pinned fertility carries the same maternal line the internal reference
    generates inline, the two arms re-certify.  (The pre-existing symmetric
    fixture could not detect the wiring defect because both arms returned the
    same diagnostics regardless of fertility.)
    """
    draws = (0, 1, 2)
    fertility_maternal_seen = []
    fert_counter = {"i": 0}

    def fake_fertility(marital, components, holdout_ids, male_gap, rng):
        del marital, components, holdout_ids, male_gap, rng
        count = _MATERNAL_COUNTS[fert_counter["i"]]
        fert_counter["i"] += 1
        return SimpleNamespace(
            maternal=_maternal_frame(count), paternal=_maternal_frame(0)
        )

    def injected_from_fertility(fertility, marital):
        assert fertility is not None
        rows = len(fertility.maternal)
        fertility_maternal_seen.append(rows)
        return rows

    _install_recertification_fakes(
        monkeypatch,
        fake_fertility=fake_fertility,
        injected_from_fertility=injected_from_fertility,
        internal_counts=_MATERNAL_COUNTS,
    )

    result = m6_preflight.run_candidate9_recertification(
        _preflight_inputs(), draw_indices=draws
    )

    assert result.passed
    # The injected arm consumed a non-empty maternal line on every draw --
    # exactly the counts the internal reference carries, so the arms match.
    assert fertility_maternal_seen == list(_MATERNAL_COUNTS[: len(draws)])
    assert all(rows > 0 for rows in fertility_maternal_seen)


def test_preflight_without_pinned_maternal_line_fails_recertification(
    monkeypatch,
):
    """A real fertility-injection omission still fails the repaired check.

    The fertility draw and internal arm carry identical nonempty maternal
    lines.  The injected test double deliberately discards that supplied draw
    and follows the ``fertility=None`` branch, isolating a wiring defect rather
    than a law or realization difference.  Re-certification must still abort.
    """
    draws = (0, 1, 2)
    fert_counter = {"i": 0}

    def fake_fertility(marital, components, holdout_ids, male_gap, rng):
        del marital, components, holdout_ids, male_gap, rng
        count = _MATERNAL_COUNTS[fert_counter["i"]]
        fert_counter["i"] += 1
        return SimpleNamespace(
            maternal=_maternal_frame(count), paternal=_maternal_frame(0)
        )

    def injected_from_fertility(fertility, marital):
        assert fertility is not None
        assert len(fertility.maternal) > 0
        # Synthetic defect: ignore the supplied draw and take the None branch.
        return len(marital.births)

    _install_recertification_fakes(
        monkeypatch,
        fake_fertility=fake_fertility,
        injected_from_fertility=injected_from_fertility,
        internal_counts=_MATERNAL_COUNTS,
    )

    with pytest.raises(AssertionError) as excinfo:
        m6_preflight.run_candidate9_recertification(
            _preflight_inputs(registered_reference_family=object()),
            draw_indices=draws,
        )

    message = str(excinfo.value)
    assert "re-certification failed" in message
    assert "coresident_child" in message
    assert "household_size" in message


def test_preflight_supplies_every_pinned_conditioning_input(monkeypatch):
    """Every section-2.8.2-pinned injected-core input is supplied by name.

    Mirrors engine/assembly.py:394-408.  A future omission -- dropping
    ``fertility=``, changing the ``male_gap`` source, the fertility component,
    the holdout ids, the period-0 composition RNG, or the (period 0, FERTILITY)
    address -- fails one of these named assertions.
    """
    male_gap_value = -2.375
    family_sentinel = object()
    household_sentinel = SimpleNamespace(male_gap=male_gap_value)
    holdout = {4, 5, 6}
    inputs = _preflight_inputs(
        family=family_sentinel,
        household=household_sentinel,
        holdout_ids=holdout,
    )

    generator_calls = []
    original_generator = ProjectionRNGRegistry.generator

    def spy_generator(self, period, module):
        generator_calls.append((period, module))
        return original_generator(self, period, module)

    monkeypatch.setattr(ProjectionRNGRegistry, "generator", spy_generator)

    fertility_calls = []
    fertility_sentinel = SimpleNamespace(
        maternal=_maternal_frame(6), paternal=_maternal_frame(0)
    )

    def fake_fertility(marital, components, holdout_ids, male_gap, rng):
        fertility_calls.append(
            SimpleNamespace(
                marital=marital,
                components=components,
                holdout_ids=set(holdout_ids),
                male_gap=male_gap,
                rng=rng,
            )
        )
        return fertility_sentinel

    marital_sentinel = SimpleNamespace(births=_maternal_frame(0))

    def fake_marital(*args, main_rng, gap_rng):
        del args, main_rng, gap_rng
        return marital_sentinel

    composition_periods = []
    rngs_sentinel = object()

    def fake_comp_rngs(registry, period):
        del registry
        composition_periods.append(period)
        return rngs_sentinel

    injected_calls = []

    def fake_injected(hh, fitted, ids, marital, rngs, fertility=None):
        injected_calls.append(
            SimpleNamespace(
                hh=hh,
                fitted=fitted,
                ids=set(ids),
                marital=marital,
                rngs=rngs,
                fertility=fertility,
            )
        )
        return object(), _diag_from_maternal(6)

    def fake_internal(*args):
        del args
        return object(), _diag_from_maternal(6)

    monkeypatch.setattr(m6_preflight, "simulate_marital_step", fake_marital)
    monkeypatch.setattr(m6_preflight, "simulate_fertility", fake_fertility)
    monkeypatch.setattr(
        m6_preflight, "composition_rngs_from_registry", fake_comp_rngs
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
        inputs, draw_indices=(0, 1)
    )
    assert result.passed

    # Fertility built once per draw with the assembly-mirrored derivations.
    assert len(fertility_calls) == 2
    for call in fertility_calls:
        assert call.marital is marital_sentinel  # injected step-3 state
        assert call.components is family_sentinel  # inputs.family (C16 fit)
        assert call.holdout_ids == holdout  # inputs.holdout_ids
        assert (
            call.male_gap == male_gap_value
        )  # float(inputs.household.male_gap)

    # The injected household core receives the pinned fertility and the single
    # period-0 CompositionRngs.
    assert len(injected_calls) == 2
    for call in injected_calls:
        assert call.fertility is fertility_sentinel
        assert call.rngs is rngs_sentinel
    assert composition_periods == [0, 0]

    # Fertility drawn on the pinned (period 0, FERTILITY) address; the marital
    # core keeps its own pinned (period 0, MARITAL_CORE) address.
    assert (0, ProjectionModule.FERTILITY) in generator_calls
    assert (0, ProjectionModule.MARITAL_CORE) in generator_calls
