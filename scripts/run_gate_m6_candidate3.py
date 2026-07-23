#!/usr/bin/env python3
"""Run the hash-bound registered M6 candidate-3 gate exactly once.

The command has no candidate selector: candidate number 3 and both model specs
come from their registries.  Governance and hash guards run before the explicit
input factory is imported, so a dirty tree, occupied output, or pin mismatch
cannot read PSID data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from populace_dynamics.harness.m6_candidate3_runner import (
    DEFAULT_OUTPUT,
    M6Candidate3DesignedAbort,
    M6Candidate3InputPlan,
    execute_registered_m6_candidate3_run,
    guard_registered_m6_candidate3_run,
)

ROOT = Path(__file__).resolve().parents[1]


def _input_factory(specification: str) -> Callable[[], M6Candidate3InputPlan]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--input-factory must have the form module:callable")
    module = importlib.import_module(module_name)
    source_name = getattr(module, "__file__", None)
    if not isinstance(source_name, str):
        raise RuntimeError("input factory module has no source file")
    source = Path(source_name).resolve()
    try:
        relative = source.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(
            "input factory module is outside the guarded source tree"
        ) from error
    tracked = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError(
            "input factory module is not tracked by the guarded source commit"
        )
    factory: Any = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"input factory {specification!r} is not callable")
    return factory


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "ONE-SHOT hash-bound M6 candidate-3 run. Requires a fresh #42 "
            "registration and an explicit vintaged-input factory."
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
            "Import path module:callable returning an "
            "M6Candidate3InputPlan whose fit-only "
            "inputs bind all external vintages and whose full-input loader "
            "is not invoked until the selected-C fit preflight passes."
        ),
    )
    command.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Exclusive artifact path; only "
            f"{DEFAULT_OUTPUT.as_posix()} is accepted."
        ),
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    preparation = guard_registered_m6_candidate3_run(
        registration_id=args.registration_id,
        output=args.out,
        root=ROOT,
    )
    factory = _input_factory(args.input_factory)
    input_plan = factory()
    if not isinstance(input_plan, M6Candidate3InputPlan):
        raise TypeError(
            "input factory must return M6Candidate3InputPlan; eager "
            "M6HarnessInputs are prohibited"
        )
    try:
        artifact = execute_registered_m6_candidate3_run(
            input_plan,
            registration_id=args.registration_id,
            output=args.out,
            root=ROOT,
            preparation=preparation,
        )
    except M6Candidate3DesignedAbort as error:
        print(
            json.dumps(
                error.report,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 2
    verdict = artifact["verdict"]["status"]
    print(
        f"M6 candidate 3 {verdict}: valid={artifact['verdict']['valid']} "
        f"out={args.out}. Certifies nothing about mortality drift. "
        "Gated earnings use the registered candidate-3 correlated-refresh law."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
