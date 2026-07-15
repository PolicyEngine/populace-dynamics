#!/usr/bin/env python3
"""Run the registered M6 candidate-1 temporal-holdout gate exactly once.

This script is deliberately not self-starting: a fresh issue-42 registration
and an explicit input factory are required.  The factory must return
``M6HarnessInputs`` and is responsible for supplying the pre-2015 SSA,
claiming, and mortality references required by ``load_m6_inputs``.  That keeps
the runner from silently selecting a new external vintage.

Environment prerequisites (the scored run, not the build/test lane).  The refit
imports the **fitting stack** -- ``populace-fit`` (with ``populace-frame``),
installed editable from ``~/PolicyEngine/populace/packages``.  ``populace-fit``
pins scikit-learn ``<1.9`` and cannot coexist with the base ``.venv``, so the
scored run uses a **dedicated** venv (the ``.venv-gate`` / W1 two-env split: a
base env for build/test, a separate fitting env for the run).  The certified
SSA-parameter vintage must also be importable: ``policyengine-us==1.752.2`` with
``POPULACE_DYNAMICS_PE_US_DIR`` pointed at the directory containing
``policyengine_us/``.  The first crash of the third registration was exactly a
fitting-stack import miss (grading #42 comment 4972045579): its single-venv env
named policyengine-us but not populace-fit.  ``build_realized_population`` runs
before the QRF refit, so it needs only the base install; the fitting stack is
required from the earnings/QRF phase onward.

Example after fresh registration::

    .venv-wt/bin/python scripts/run_gate_m6_candidate1.py \
      --registration-id 1234567890 \
      --input-factory registered_m6_inputs:build_inputs

The build/test lane must never execute this command against real data.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from populace_dynamics.harness.m6_inputs import M6HarnessInputs
from populace_dynamics.harness.m6_runner import (
    DEFAULT_OUTPUT,
    execute_registered_m6_run,
    guard_registered_m6_run,
)

ROOT = Path(__file__).resolve().parents[1]


def _input_factory(specification: str) -> Callable[[], M6HarnessInputs]:
    """Resolve an explicit ``module:callable`` without choosing input data."""
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--input-factory must have the form module:callable")
    module = importlib.import_module(module_name)
    factory: Any = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"input factory {specification!r} is not callable")
    return factory


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "ONE-SHOT M6 scored run. Requires a fresh #42 registration and "
            "an explicit vintaged-input factory."
        )
    )
    command.add_argument(
        "--registration-id",
        required=True,
        help="Fresh issue #42 comment id or full issue-comment URL.",
    )
    command.add_argument(
        "--input-factory",
        required=True,
        help=(
            "Import path module:callable returning M6HarnessInputs; the "
            "callable must explicitly bind all external vintages."
        ),
    )
    command.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Fresh exclusive artifact path (default: "
            f"{DEFAULT_OUTPUT.as_posix()})."
        ),
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Governance checks precede even importing the external input factory, so
    # a stale registration or occupied one-shot path cannot touch PSID.
    guard_registered_m6_run(
        registration_id=args.registration_id,
        output=args.out,
        root=ROOT,
    )
    factory = _input_factory(args.input_factory)
    inputs = factory()
    if not isinstance(inputs, M6HarnessInputs):
        raise TypeError("input factory must return M6HarnessInputs")
    artifact = execute_registered_m6_run(
        inputs,
        registration_id=args.registration_id,
        output=args.out,
        root=ROOT,
    )
    verdict = "PASS" if artifact["verdict"]["pass"] else "FAIL"
    print(
        f"M6 candidate 1 {verdict}: valid={artifact['verdict']['valid']} "
        f"out={args.out}. Certifies nothing about mortality drift. "
        "Gated earnings use an M6-first-certified forward law; no gate_1 "
        "backward-law certificate transfers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
