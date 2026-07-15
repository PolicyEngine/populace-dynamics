"""Candidate-blind pre-flights for the M6 scored-run harness.

Neither pre-flight reads a holdout cell.  The first compares two simulation
paths on the cutoff-refitted native panels; the second exercises the fitted
earnings participation gate on a synthetic probe and records the selected
implementation branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from populace_dynamics.data import household_composition, transitions
from populace_dynamics.engine.composition import (
    RecertificationResult,
    check_candidate9_recertification,
    composition_rngs_from_registry,
    simulate_candidate9_injected,
    simulate_candidate9_internal_reference,
)
from populace_dynamics.engine.forward_earnings import _gate_sign_draw
from populace_dynamics.engine.marital import simulate_marital_step
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.engine.steps import simulate_fertility

DRAW_COUNT = 20


@dataclass(frozen=True)
class Candidate9PreflightInputs:
    """Native panels and cutoff-refitted objects used by pre-flight 1."""

    marital_panel: transitions.MaritalPanel
    household_panel: household_composition.HouseholdCompositionPanel
    holdout_ids: set[int]
    family: Any
    modifier: Any
    permanent_axis: Any
    household: Any


@dataclass(frozen=True)
class SignPathRecord:
    """Machine-readable evidence for the certified participation branch."""

    branch: str
    gates_checked: tuple[str, ...]
    probe_rows: int
    output_signs: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_candidate9_recertification(
    inputs: Candidate9PreflightInputs,
    *,
    draw_indices: tuple[int, ...] = tuple(range(DRAW_COUNT)),
) -> RecertificationResult:
    """Run the injected-vs-internal candidate-9 margin check.

    A failing margin raises inside :func:`check_candidate9_recertification`.
    The caller must run this before any scored projection or one-shot write.
    """
    if len(draw_indices) < 2 or len(set(draw_indices)) != len(draw_indices):
        raise ValueError(
            "pre-flight 1 needs at least two distinct draw indices"
        )
    injected = []
    internal = []
    for draw_index in draw_indices:
        if draw_index < 0:
            raise ValueError("draw indices must be non-negative")
        registry = ProjectionRNGRegistry(draw_index=draw_index, n_periods=0)
        marital = simulate_marital_step(
            inputs.marital_panel,
            set(inputs.holdout_ids),
            inputs.family,
            inputs.modifier,
            inputs.permanent_axis,
            main_rng=registry.generator(0, ProjectionModule.MARITAL_CORE),
            gap_rng=registry.child_generator(
                0, ProjectionModule.MARITAL_CORE, 1
            ),
        )
        # Section 2.8.2 pins the injected household core's conditioning
        # fertility to ``registry.generator(0, FERTILITY)``; the certified
        # projection honors this at ``engine/assembly.py:394-408``.  Build it
        # the same way so the injected arm carries the maternal line the
        # internal reference generates inline through ``ft.simulate`` -- with
        # ``fertility=None`` the injected arm dropped the entire maternal
        # line (~24k births/draw) while the internal reference kept it, an
        # asymmetry that aborted seven household-composition channels at
        # 7-118x the 3-sigma tolerance (issue #42 forensics 4973982118).
        #
        # Argument derivations mirror ``assembly.py`` exactly: ``marital`` is
        # the injected step-3 state, ``inputs.family`` is the authoritative
        # candidate-16 fit, ``holdout_ids`` is this universe's train ids, and
        # ``male_gap`` is ``float(inputs.household.male_gap)`` -- assembly's
        # ``inputs.male_gap`` is likewise ``float(household_fit.male_gap)``
        # (assembly.py:150,160).  The pre-flight deliberately does not rebind
        # the candidate-9 household core to the family fit, so its own fitted
        # ``male_gap`` is the value the internal reference's paternal shadow
        # also consumes (``engine/composition.py`` fertility=None branch),
        # keeping the two arms male_gap-symmetric.
        household_fertility = simulate_fertility(
            marital,
            inputs.family,
            set(inputs.holdout_ids),
            float(inputs.household.male_gap),
            registry.generator(0, ProjectionModule.FERTILITY),
        )
        _panel, injected_diagnostics = simulate_candidate9_injected(
            inputs.household_panel,
            inputs.household,
            set(inputs.holdout_ids),
            marital,
            composition_rngs_from_registry(registry, 0),
            fertility=household_fertility,
        )
        _reference_panel, internal_diagnostics = (
            simulate_candidate9_internal_reference(
                inputs.household_panel,
                inputs.marital_panel,
                inputs.household,
                set(inputs.holdout_ids),
                5200 + draw_index,
            )
        )
        injected.append(injected_diagnostics)
        internal.append(internal_diagnostics)
    return check_candidate9_recertification(injected, internal)


def recertification_payload(
    result: RecertificationResult,
) -> dict[str, Any]:
    """Convert pre-flight 1 evidence to the scored-run artifact schema."""
    return {
        "passed": result.passed,
        "sigma_multiplier": result.sigma_multiplier,
        "cells": [asdict(cell) for cell in result.cells],
    }


def verify_external_sign_path(generator: Any) -> SignPathRecord:
    """Exercise and record the externally-driven earnings sign-gate branch.

    The probe is synthetic and deliberately bypasses the earnings frame.  Its
    sole purpose is to prove that each fitted participation gate deploys the
    CERTIFIED ``_target_models`` reconstruction (the branch every real
    ``FittedRegimeGatedQRF`` executes, section 2.8.6 as corrected by design
    amendment 3c) rather than the ``draw_sign`` test seam, which exists only
    on injected doubles.
    """
    named_gates = [("shared_gate", getattr(generator, "shared_gate", None))]
    zero_gate = getattr(generator, "zero_anchor_gate", None)
    if zero_gate is not None:
        named_gates.append(("zero_anchor_gate", zero_gate))

    current_level = np.asarray([0.0, 25_000.0], dtype=np.float64)
    target_age = np.asarray([40.0, 50.0], dtype=np.float64)
    uniforms = np.asarray([0.25, 0.75], dtype=np.float64)
    outputs: list[int] = []
    checked: list[str] = []
    for name, gate in named_gates:
        if (
            gate is None
            or callable(getattr(gate, "draw_sign", None))
            or getattr(gate, "_target_models", None) is None
        ):
            raise RuntimeError(
                f"{name} does not deploy the certified _target_models"
                " reconstruction (or carries the draw_sign test seam)"
            )
        signs = np.asarray(
            _gate_sign_draw(gate, current_level, target_age, uniforms),
            dtype=np.int64,
        )
        if signs.shape != current_level.shape:
            raise RuntimeError(f"{name} returned the wrong probe shape")
        checked.append(name)
        outputs.extend(int(value) for value in signs)
    return SignPathRecord(
        branch="certified_target_models_reconstruction",
        gates_checked=tuple(checked),
        probe_rows=len(current_level),
        output_signs=tuple(outputs),
    )
