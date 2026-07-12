"""Replay candidate 9 through the flattened household-composition registry."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from populace_dynamics.artifacts import write_new
from populace_dynamics.models.household_composition.compatibility import (
    build_compatibility_certificate,
)
from populace_dynamics.models.household_composition.evaluation import (
    evaluate_candidate9,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "gate2b_registry_certificate_v1.json"


def main() -> None:
    """Evaluate the registry and exclusively create its compatibility record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New certificate path (must not already exist).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="lbfgs failed to converge")
    evaluation = evaluate_candidate9(ROOT, verbose=not args.quiet)
    certificate = build_compatibility_certificate(
        ROOT,
        evaluation,
        runner_path=Path(__file__).resolve(),
    )
    write_new(args.output, certificate)
    print(
        f"wrote {args.output}; certificate_pass="
        f"{certificate['certificate_pass']}"
    )
    if not certificate["certificate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
