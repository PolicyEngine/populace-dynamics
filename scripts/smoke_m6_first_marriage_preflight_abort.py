#!/usr/bin/env python3
"""Exercise the real first-marriage preflight before any holdout truth load.

This synthetic fit deliberately receives one LBFGS iteration.  The real sibling
component therefore returns an ineligible convergence certificate, and its real
registered validator must raise before the poisoned full-input callback runs.
The script opens no staged data and constructs no holdout truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from populace_dynamics.harness.m6_runner import (
    continue_m6_after_fit_preflight,
)
from populace_dynamics.models.family_transitions.components.first_marriage_support_aware import (
    GRADIENT_TOL,
    FirstMarriagePreflightAbort,
    fit_support_aware_first_marriage,
    validate_support_aware_first_marriage_fit,
)


@dataclass(frozen=True)
class SyntheticFirstMarriageFitInputs:
    """Named synthetic fit state consumed by the real component callback."""

    train_rows: pd.DataFrame
    event_years: frozenset[tuple[int, int]]
    c: float
    max_iter: int


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    people = (
        (1, "female", 1970),
        (2, "female", 1980),
        (3, "male", 1970),
        (4, "male", 1980),
    )
    for person_id, sex, birth_decade in people:
        for offset, year in enumerate((2005, 2006)):
            rows.append(
                {
                    "person_id": person_id,
                    "year": year,
                    "age": 20 + person_id + offset,
                    "sex": sex,
                    "weight": float(person_id),
                    "birth_decade": birth_decade,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    events: list[str] = []
    calls = {
        "holdout_truth_loader": 0,
        "holdout_truth_builder": 0,
        "downstream_continuation": 0,
        "projection": 0,
        "score": 0,
        "candidate_artifact_write": 0,
    }
    fit_inputs = SyntheticFirstMarriageFitInputs(
        train_rows=_synthetic_rows(),
        event_years=frozenset({(1, 2006), (4, 2006)}),
        c=0.01,
        max_iter=1,
    )
    model = None

    def fit(observed: SyntheticFirstMarriageFitInputs) -> Any:
        nonlocal model
        if not isinstance(observed, SyntheticFirstMarriageFitInputs):
            raise TypeError("unexpected synthetic fit-input type")
        events.append("fit")
        model = fit_support_aware_first_marriage(
            observed.train_rows,
            set(observed.event_years),
            c=observed.c,
            max_iter=observed.max_iter,
        )
        return model

    def preflight(model: Any) -> None:
        events.append("preflight")
        validate_support_aware_first_marriage_fit(model)

    def truth_builder() -> Any:
        calls["holdout_truth_builder"] += 1
        raise AssertionError("holdout truth builder must remain unreachable")

    def load_full_inputs() -> Any:
        calls["holdout_truth_loader"] += 1
        return truth_builder()

    def poisoned_continuation(_result: Any) -> Any:
        calls["downstream_continuation"] += 1
        calls["projection"] += 1
        calls["score"] += 1
        calls["candidate_artifact_write"] += 1
        raise AssertionError("downstream continuation must remain unreachable")

    message = None
    try:
        continue_m6_after_fit_preflight(
            fit_inputs,  # type: ignore[arg-type]
            fit=fit,  # type: ignore[arg-type]
            preflight=preflight,
            load_full_inputs=load_full_inputs,
            continuation=poisoned_continuation,
        )
    except FirstMarriagePreflightAbort as error:
        message = str(error)
    else:
        raise AssertionError("synthetic failing fit did not abort")

    if model is None or model.fit_audit is None or message is None:
        raise AssertionError("synthetic abort did not retain its fit audit")
    if events != ["fit", "preflight"] or any(calls.values()):
        raise AssertionError(
            f"preflight ordering fence failed: events={events}, calls={calls}"
        )

    audit = model.fit_audit
    payload = {
        "schema": "m6.first_marriage.lazy_preflight_abort.synthetic.v1",
        "status": "EXPECTED_PREFLIGHT_ABORT",
        "synthetic_only": True,
        "staged_data_opened": False,
        "synthetic_fit_input": {
            "type": SyntheticFirstMarriageFitInputs.__name__,
            "n_rows": len(fit_inputs.train_rows),
            "n_events": len(fit_inputs.event_years),
            "consumed_by_real_fit_callback": True,
        },
        "fit_certificate": {
            "c": audit.c,
            "n_train_rows": audit.n_train_rows,
            "n_train_events": audit.n_train_events,
            "solver_success": audit.solver_success,
            "n_iter": audit.n_iter,
            "max_iter": audit.max_iter,
            "max_iter_reached": audit.max_iter_reached,
            "warning_count": audit.warning_count,
            "gradient_above_threshold": (
                audit.gradient_inf_norm > GRADIENT_TOL
            ),
            "eligible": audit.eligible,
            "eligibility_failures": list(audit.eligibility_failures),
            "support_sha256": audit.checksums["support_sha256"],
        },
        "abort": {
            "exception": FirstMarriagePreflightAbort.__name__,
            "message": message,
            "propagated_unchanged": True,
        },
        "ordering": events,
        "fences": calls,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
