#!/usr/bin/env python3.13
"""Select candidate-2's first-marriage ridge strength on train data only.

The default, no-argument command is the one real selector.  It performs the
27 registered pseudo-boundary fits (three boundaries by nine ``C`` values),
then one final ``<=2014`` fit when a registerable ``C`` exists.  It never
imports the M6 scorer, reads ``gates.yaml`` or ``runs/``, or writes a file.
Progress goes to stderr and the sole stdout value is one strict-JSON ledger.

``--synthetic-smoke`` exercises the same fitting, selection, publication, and
reduction functions on a small deterministic fixture.  The fixture contains
an empty sex-by-cohort cell, a positive-exposure zero-event cell, and a second
all-ineligible case that must publish
``NO_REGISTERABLE_FIRST_MARRIAGE_FIT`` rather than raise.

The real selector is deliberately unusable outside the numeric identity that
was frozen for the M6 remarriage round-5/6 selection convention: CPython
3.13.12 and NumPy 2.4.2.  The identity check runs before staged data loading or
any selection fit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.models.family_transitions.components.first_marriage_support_aware import (
    C_GRID,
    GRADIENT_TOL,
    MAX_INFORMATION_YEAR,
    MAX_ITER,
    PSEUDO_BOUNDARIES,
    SOLVER_FTOL,
    SOLVER_TOL,
    FirstMarriagePreflightAbort,
    SupportAwareFirstMarriageModel,
    fit_support_aware_first_marriage,
    recompute_support_aware_first_marriage_checksums,
    validate_support_aware_first_marriage_fit,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_SCHEMA = "m6.first_marriage.c_selection.full.v1"
REDUCED_SCHEMA = "m6.first_marriage.c_selection.findings.v1"
SMOKE_SCHEMA = "m6.first_marriage.c_selection.synthetic_smoke.v1"
NO_REGISTERABLE = "NO_REGISTERABLE_FIRST_MARRIAGE_FIT"
SELECTION_COMPLETE = "SELECTION_COMPLETE"
TIE_TOLERANCE = 1e-12
EXPECTED_RUNTIME_IDENTITY = {
    "python_implementation": "CPython",
    "python_version": "3.13.12",
    "numpy_version": "2.4.2",
}
EVALUATION_WINDOWS = {
    boundary: tuple(range(boundary + 1, boundary + 5))
    for boundary in PSEUDO_BOUNDARIES
}
CALENDAR_YEAR_MULTIPLICITY = {
    year: sum(year in years for years in EVALUATION_WINDOWS.values())
    for year in range(min(PSEUDO_BOUNDARIES) + 1, MAX_INFORMATION_YEAR + 1)
}
FRAME_COLUMNS = (
    "person_id",
    "year",
    "required_interview_year",
    "age",
    "sex",
    "weight",
    "birth_decade",
    "event",
)
FIT_COLUMNS = (
    "person_id",
    "year",
    "age",
    "sex",
    "weight",
    "birth_decade",
)
FitFunction = Callable[..., SupportAwareFirstMarriageModel]


class ProtocolAbort(RuntimeError):
    """A runtime, information-boundary, or ledger-shape failure."""


class RuntimeIdentityAbort(ProtocolAbort):
    """The process does not have the frozen numeric runtime identity."""


@dataclass(frozen=True)
class BoundaryFrames:
    """Canonical fit and pseudo-holdout frames for one boundary."""

    fit: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class SelectionFrames:
    """All train-only frames consumed by the selector."""

    boundaries: Mapping[int, BoundaryFrames]
    final: pd.DataFrame
    source_audit: Mapping[str, Any]
    mode: str


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool | np.bool_) and missing:
        return None
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _strict_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        _plain(value),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_audit(
    *,
    observed_identity: Mapping[str, str] | None = None,
    dont_write_bytecode: bool | None = None,
) -> dict[str, Any]:
    """Abort unless the round-5/6 numeric identity is exact."""
    observed = dict(
        observed_identity
        if observed_identity is not None
        else {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        }
    )
    if observed != EXPECTED_RUNTIME_IDENTITY:
        raise RuntimeIdentityAbort(
            "ROOT_VALIDATION_MISMATCH: numeric runtime "
            f"{observed} differs from {EXPECTED_RUNTIME_IDENTITY}"
        )
    bytecode_disabled = (
        sys.dont_write_bytecode
        if dont_write_bytecode is None
        else bool(dont_write_bytecode)
    )
    if not bytecode_disabled:
        raise RuntimeIdentityAbort(
            "PYTHONDONTWRITEBYTECODE=1 is required before selection"
        )
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name.startswith("populace_dynamics.harness.m6_scoring")
        or name.startswith("populace_dynamics.harness.m6_runner")
    )
    if forbidden_modules:
        raise ProtocolAbort(
            f"forbidden M6 scoring modules imported: {forbidden_modules}"
        )
    dependencies: dict[str, str | None] = {}
    for distribution in ("pandas", "scipy", "scikit-learn"):
        try:
            dependencies[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    return {
        **observed,
        "python_executable": sys.executable,
        "dont_write_bytecode": bytecode_disabled,
        "dependencies": dependencies,
        "forbidden_m6_modules_imported": forbidden_modules,
        "identity_checked_before_data_load_and_selection": True,
    }


def _assert_protocol_constants() -> None:
    if C_GRID != (
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    ):
        raise ProtocolAbort("first-marriage C grid drifted")
    if PSEUDO_BOUNDARIES != (2006, 2008, 2010):
        raise ProtocolAbort("first-marriage pseudo-boundaries drifted")
    if MAX_INFORMATION_YEAR != 2014:
        raise ProtocolAbort("first-marriage information boundary drifted")
    if (
        MAX_ITER != 10_000
        or SOLVER_TOL != 1e-8
        or SOLVER_FTOL != float(np.finfo(np.float64).eps)
    ):
        raise ProtocolAbort("first-marriage LBFGS contract drifted")
    if GRADIENT_TOL != 1e-6 or TIE_TOLERANCE != 1e-12:
        raise ProtocolAbort("first-marriage numeric certificate drifted")
    expected_multiplicity = {
        2007: 1,
        2008: 1,
        2009: 2,
        2010: 2,
        2011: 2,
        2012: 2,
        2013: 1,
        2014: 1,
    }
    if CALENDAR_YEAR_MULTIPLICITY != expected_multiplicity:
        raise ProtocolAbort("pseudo-window calendar multiplicity drifted")


def _repository_freeze() -> dict[str, Any]:
    """Require a clean commit and bind every selection implementation byte."""
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProtocolAbort(
            "real selection requires a committed clean freeze; status is:\n"
            f"{status}"
        )
    paths = (
        "scripts/select_m6_first_marriage_c.py",
        (
            "src/populace_dynamics/models/family_transitions/components/"
            "first_marriage_support_aware.py"
        ),
        "src/populace_dynamics/models/family_transitions/registry.py",
        "src/populace_dynamics/engine/refit.py",
        "docs/design/m6_candidate2_program.md",
    )
    blobs: dict[str, str] = {}
    sha256: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        head_blob = _git("rev-parse", f"HEAD:{relative}")
        if _git("hash-object", relative) != head_blob:
            raise ProtocolAbort(f"working bytes differ from HEAD: {relative}")
        blobs[relative] = head_blob
        sha256[relative] = _sha256_file(path)
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "worktree_clean": True,
        "frozen_blob_sha1": blobs,
        "frozen_file_sha256": sha256,
        "source_tree_sha1": _git("rev-parse", "HEAD:src/populace_dynamics"),
    }


def _canonical_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(FRAME_COLUMNS) - set(frame.columns))
    if missing:
        raise ProtocolAbort(f"{label} is missing columns {missing}")
    out = frame.loc[:, list(FRAME_COLUMNS)].copy()
    for column in (
        "person_id",
        "year",
        "required_interview_year",
        "age",
        "weight",
        "birth_decade",
    ):
        numeric = pd.to_numeric(out[column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        ):
            raise ProtocolAbort(f"{label}.{column} must be finite")
        out[column] = numeric
    if not out["sex"].isin(("female", "male")).all():
        raise ProtocolAbort(f"{label}.sex contains an unsupported value")
    if (out["weight"] < 0).any():
        raise ProtocolAbort(f"{label}.weight contains a negative value")
    if not out["event"].isin((False, True, 0, 1)).all():
        raise ProtocolAbort(f"{label}.event is not binary")
    out["person_id"] = out["person_id"].astype(np.int64)
    out["year"] = out["year"].astype(np.int64)
    out["required_interview_year"] = out["required_interview_year"].astype(
        np.int64
    )
    out["birth_decade"] = out["birth_decade"].astype(np.int64)
    out["event"] = out["event"].astype(bool)
    out = out.sort_values(["person_id", "year"], kind="stable").reset_index(
        drop=True
    )
    if out.duplicated(["person_id", "year"]).any():
        raise ProtocolAbort(f"{label} contains duplicate person-year rows")
    return out


def _frame_checksum(frame: pd.DataFrame) -> str:
    selected = frame.loc[:, list(FRAME_COLUMNS)]
    hashed = pd.util.hash_pandas_object(
        selected, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return _sha256_bytes(hashed.tobytes())


def _selected_frame_checksum(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ProtocolAbort(f"{label} checksum is missing columns {missing}")
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(
        list(columns), kind="stable", na_position="last"
    ).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(
        selected, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return _sha256_bytes(hashed.tobytes())


def _max_numeric(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.max()) if len(values) else None


def _frame_audit(
    frame: pd.DataFrame,
    *,
    label: str,
    year_minimum: int | None = None,
    year_maximum: int,
    interview_maximum: int,
) -> dict[str, Any]:
    """Enforce the field-aware date ceiling and return its proof."""
    years = frame["year"].to_numpy(dtype=np.int64)
    interviews = frame["required_interview_year"].to_numpy(dtype=np.int64)
    if years.size and int(years.max()) > year_maximum:
        raise ProtocolAbort(
            f"{label}.year reaches {int(years.max())}, beyond {year_maximum}"
        )
    if (
        years.size
        and year_minimum is not None
        and int(years.min()) < year_minimum
    ):
        raise ProtocolAbort(
            f"{label}.year reaches {int(years.min())}, before {year_minimum}"
        )
    if interviews.size and int(interviews.max()) > interview_maximum:
        raise ProtocolAbort(
            f"{label}.required_interview_year reaches "
            f"{int(interviews.max())}, beyond {interview_maximum}"
        )
    positive = frame["weight"].to_numpy(dtype=np.float64) > 0
    event = frame["event"].to_numpy(dtype=bool)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    return {
        "label": label,
        "n_rows": int(len(frame)),
        "positive_weight_rows": int(positive.sum()),
        "event_rows": int(event.sum()),
        "positive_weight_event_rows": int((positive & event).sum()),
        "row_weight": float(weight.sum()),
        "event_weight": float(np.dot(weight, event.astype(np.float64))),
        "year_min": int(years.min()) if years.size else None,
        "year_max": int(years.max()) if years.size else None,
        "required_interview_year_min": (
            int(interviews.min()) if interviews.size else None
        ),
        "required_interview_year_max": (
            int(interviews.max()) if interviews.size else None
        ),
        "asserted_year_minimum": year_minimum,
        "asserted_year_maximum": year_maximum,
        "asserted_interview_maximum": interview_maximum,
        "field_aware_boundary_pass": True,
        "canonical_person_year_order": True,
        "frame_sha256": _frame_checksum(frame),
    }


def _support_record(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame[frame["weight"] > 0]
    if positive.empty:
        return {
            "n_rows": 0,
            "row_weight": 0.0,
            "n_events": 0,
            "event_weight": 0.0,
            "age_min": None,
            "age_max": None,
        }
    weight = positive["weight"].to_numpy(dtype=np.float64)
    event = positive["event"].to_numpy(dtype=np.float64)
    age = positive["age"].to_numpy(dtype=np.float64)
    return {
        "n_rows": int(len(positive)),
        "row_weight": float(weight.sum()),
        "n_events": int(event.sum()),
        "event_weight": float(np.dot(weight, event)),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
    }


def _frame_support(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame[frame["weight"] > 0]
    cohorts = tuple(
        sorted(int(value) for value in positive["birth_decade"].unique())
    )
    return {
        "positive_weight_only": True,
        "sex": {
            sex: _support_record(positive[positive["sex"] == sex])
            for sex in ("female", "male")
        },
        "pooled_cohort": {
            str(cohort): _support_record(
                positive[positive["birth_decade"] == cohort]
            )
            for cohort in cohorts
        },
        "sex_by_cohort": {
            f"{sex}|{cohort}": _support_record(
                positive[
                    (positive["sex"] == sex)
                    & (positive["birth_decade"] == cohort)
                ]
            )
            for sex in ("female", "male")
            for cohort in cohorts
        },
    }


def _event_years(frame: pd.DataFrame) -> set[tuple[int, int]]:
    events = frame[frame["event"]]
    return {
        (int(person), int(year))
        for person, year in zip(
            events["person_id"], events["year"], strict=True
        )
    }


def _validate_all_frames(
    frames: SelectionFrames,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every working frame before the first fit is attempted."""
    if set(frames.boundaries) != set(PSEUDO_BOUNDARIES):
        raise ProtocolAbort("selection boundary frame keys drifted")
    boundaries: dict[str, Any] = {}
    for boundary in PSEUDO_BOUNDARIES:
        pair = frames.boundaries[boundary]
        expected_years = EVALUATION_WINDOWS[boundary]
        fit_audit = _frame_audit(
            pair.fit,
            label=f"boundary_{boundary}.fit",
            year_maximum=boundary,
            interview_maximum=boundary,
        )
        evaluation_audit = _frame_audit(
            pair.evaluation,
            label=f"boundary_{boundary}.evaluation",
            year_minimum=min(expected_years),
            year_maximum=max(expected_years),
            interview_maximum=MAX_INFORMATION_YEAR,
        )
        observed_years = set(
            int(value) for value in pair.evaluation["year"].unique()
        )
        extra_years = sorted(observed_years - set(expected_years))
        if extra_years:
            raise ProtocolAbort(
                f"boundary {boundary} evaluation has extra years {extra_years}"
            )
        boundaries[str(boundary)] = {
            "fit_frame": fit_audit,
            "evaluation_frame": evaluation_audit,
            "intended_evaluation_years": list(expected_years),
            "observed_evaluation_years": sorted(observed_years),
            "calendar_rows": {
                str(year): int(pair.evaluation["year"].eq(year).sum())
                for year in expected_years
            },
        }
    final_audit = _frame_audit(
        frames.final,
        label="final_2014.fit",
        year_maximum=MAX_INFORMATION_YEAR,
        interview_maximum=MAX_INFORMATION_YEAR,
    )
    return boundaries, final_audit


def _empty_evaluation_record(
    frame: pd.DataFrame, reason: str
) -> dict[str, Any]:
    return {
        "attempted": False,
        "eligible": False,
        "failure": reason,
        "n_rows": int(len(frame)),
        "n_events": int(frame["event"].sum()),
        "row_weight": float(frame["weight"].sum()),
        "event_weight": float(
            np.dot(
                frame["weight"].to_numpy(dtype=np.float64),
                frame["event"].to_numpy(dtype=np.float64),
            )
        ),
        "weighted_deviance_numerator": None,
        "weighted_mean_bernoulli_deviance": None,
        "linear_predictor_min": None,
        "linear_predictor_max": None,
        "probability_min": None,
        "probability_max": None,
        "prediction_sha256": None,
    }


def _evaluate_model(
    model: SupportAwareFirstMarriageModel, frame: pd.DataFrame
) -> dict[str, Any]:
    if frame.empty:
        return _empty_evaluation_record(frame, "empty_evaluation_frame")
    positive = frame[frame["weight"] > 0]
    if positive.empty:
        return _empty_evaluation_record(
            frame, "no_positive_weight_evaluation_rows"
        )
    age = positive["age"].to_numpy(dtype=np.float64)
    male = positive["sex"].to_numpy() == "male"
    decade = positive["birth_decade"].to_numpy(dtype=np.int64)
    try:
        linear = model.linear_predictor(age, male, decade)
        probability = model.predict(age, male, decade)
    except (FloatingPointError, ValueError) as error:
        return _empty_evaluation_record(
            frame, f"{type(error).__name__}: {error}"
        )
    if not np.isfinite(linear).all():
        return _empty_evaluation_record(
            frame, "nonfinite_evaluation_linear_predictor"
        )
    if not np.isfinite(probability).all():
        return _empty_evaluation_record(
            frame, "nonfinite_evaluation_probability"
        )
    if not ((probability > 0.0) & (probability < 1.0)).all():
        return _empty_evaluation_record(
            frame, "evaluation_probability_outside_strict_unit_interval"
        )
    weight = positive["weight"].to_numpy(dtype=np.float64)
    event = positive["event"].to_numpy(dtype=np.float64)
    exposure = float(weight.sum())
    deviance_numerator = float(
        -2.0
        * np.sum(
            weight
            * (
                event * np.log(probability)
                + (1.0 - event) * np.log1p(-probability)
            )
        )
    )
    deviance = deviance_numerator / exposure
    prediction_hash = _sha256_bytes(
        np.ascontiguousarray(probability, dtype="<f8").tobytes()
    )
    return {
        "attempted": True,
        "eligible": bool(math.isfinite(deviance)),
        "failure": None if math.isfinite(deviance) else "nonfinite_deviance",
        "n_rows": int(len(positive)),
        "n_events": int(event.sum()),
        "row_weight": exposure,
        "event_weight": float(np.dot(weight, event)),
        "weighted_deviance_numerator": deviance_numerator,
        "weighted_mean_bernoulli_deviance": deviance,
        "linear_predictor_min": float(linear.min()),
        "linear_predictor_max": float(linear.max()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "prediction_sha256": prediction_hash,
    }


def _fit_rung(
    fit_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    c: float,
    fit_function: FitFunction,
) -> tuple[dict[str, Any], SupportAwareFirstMarriageModel | None]:
    input_support = _frame_support(fit_frame)
    try:
        model = fit_function(
            fit_frame.loc[:, list(FIT_COLUMNS)],
            _event_years(fit_frame),
            c=c,
            max_iter=MAX_ITER,
            tol=SOLVER_TOL,
        )
    except (FloatingPointError, ValueError) as error:
        exception = {
            "type": type(error).__name__,
            "message": str(error),
        }
        return (
            {
                "c": float(c),
                "eligible": False,
                "eligibility_failures": ["fit_exception"],
                "fit_exception": exception,
                "input_support": input_support,
                "fit_audit": None,
                "evaluation": _empty_evaluation_record(
                    evaluation_frame,
                    f"fit_exception: {type(error).__name__}",
                ),
            },
            None,
        )
    if model.fit_audit is None:
        raise ProtocolAbort("support-aware fit returned no fit audit")
    fit_audit = model.fit_audit.canonical_dict()
    if float(fit_audit["c"]) != float(c):
        raise ProtocolAbort("support-aware fit audit recorded the wrong C")
    if int(fit_audit["max_iter"]) != MAX_ITER:
        raise ProtocolAbort("support-aware fit max_iter drifted")
    evaluation = _evaluate_model(model, evaluation_frame)
    failures = list(fit_audit["eligibility_failures"])
    if not evaluation["eligible"]:
        failures.append(str(evaluation["failure"]))
    eligible = bool(fit_audit["eligible"] and evaluation["eligible"])
    return (
        {
            "c": float(c),
            "eligible": eligible,
            "eligibility_failures": list(dict.fromkeys(failures)),
            "fit_exception": None,
            "input_support": input_support,
            "fit_audit": fit_audit,
            "evaluation": evaluation,
        },
        model,
    )


def _selection_from_boundaries(
    boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for index, c in enumerate(C_GRID):
        rungs = [
            boundaries[str(boundary)]["rungs"][index]
            for boundary in PSEUDO_BOUNDARIES
        ]
        eligible = all(bool(rung["eligible"]) for rung in rungs)
        deviances = {
            str(boundary): rung["evaluation"][
                "weighted_mean_bernoulli_deviance"
            ]
            for boundary, rung in zip(PSEUDO_BOUNDARIES, rungs, strict=True)
        }
        mean_deviance = (
            float(sum(float(value) for value in deviances.values()) / 3.0)
            if eligible
            else None
        )
        candidates.append(
            {
                "c": float(c),
                "eligible_at_all_boundaries": eligible,
                "boundary_deviances": deviances,
                "equal_boundary_mean_deviance": mean_deviance,
            }
        )
    eligible_candidates = [
        record for record in candidates if record["eligible_at_all_boundaries"]
    ]
    if not eligible_candidates:
        minimum = None
        tie_set: list[float] = []
        selected = None
        reason = "all C values are ineligible at one or more boundaries"
    else:
        minimum = min(
            float(record["equal_boundary_mean_deviance"])
            for record in eligible_candidates
        )
        tie_set = sorted(
            float(record["c"])
            for record in eligible_candidates
            if abs(float(record["equal_boundary_mean_deviance"]) - minimum)
            <= TIE_TOLERANCE
        )
        selected = tie_set[0]
        reason = (
            "lowest equal-boundary mean deviance; values within 1e-12 "
            "tie and the smaller C wins"
        )
    return {
        "candidates": candidates,
        "eligible_cs": [float(record["c"]) for record in eligible_candidates],
        "minimum_mean_deviance": minimum,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_set": tie_set,
        "selected_c": selected,
        "reason": reason,
        "candidate_1_score_consulted": False,
        "post_2014_row_entered_numerical_selection": False,
        "substitution_or_reselection": False,
    }


def _hazard_table(
    model: SupportAwareFirstMarriageModel,
) -> list[dict[str, Any]]:
    cohorts = tuple(sorted(set(model.cohort_levels) | {1990}))
    targets: list[tuple[float, bool, int]] = []
    for sex in ("female", "male"):
        for cohort in cohorts:
            for age in range(18, 30):
                targets.append((float(age), sex == "male", cohort))
    age = np.asarray([row[0] for row in targets], dtype=np.float64)
    is_male = np.asarray([row[1] for row in targets], dtype=bool)
    decade = np.asarray([row[2] for row in targets], dtype=np.int64)
    diagnostics = model.transport_diagnostics(age, is_male, decade)
    records = diagnostics.canonical_records()
    linear = model.linear_predictor(age, is_male, decade)
    probability = model.predict(age, is_male, decade)
    if not np.isfinite(linear).all() or not (
        np.isfinite(probability).all()
        and ((probability > 0.0) & (probability < 1.0)).all()
    ):
        raise FirstMarriagePreflightAbort(
            "final hazard table is outside the strict numeric preflight"
        )
    return [
        {
            **record,
            "linear_predictor": float(linear_value),
            "probability": float(probability_value),
        }
        for record, linear_value, probability_value in zip(
            records, linear, probability, strict=True
        )
    ]


def _unattempted_final(
    frame_audit: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "attempted": False,
        "c": None,
        "eligible": False,
        "reason": reason,
        "fit_exception": None,
        "input_frame": dict(frame_audit),
        "input_support": None,
        "fit_audit": None,
        "hazard_table_ages_18_29": [],
        "preflight": {
            "passed": False,
            "message": "not attempted",
            "expected_checksum_replay": False,
            "recomputed_checksums": None,
        },
    }


def _fit_final(
    frame: pd.DataFrame,
    frame_audit: Mapping[str, Any],
    *,
    c: float,
    fit_function: FitFunction,
) -> tuple[dict[str, Any], bool]:
    support = _frame_support(frame)
    try:
        model = fit_function(
            frame.loc[:, list(FIT_COLUMNS)],
            _event_years(frame),
            c=c,
            max_iter=MAX_ITER,
            tol=SOLVER_TOL,
        )
    except (FloatingPointError, ValueError) as error:
        return (
            {
                "attempted": True,
                "c": float(c),
                "eligible": False,
                "reason": "final_fit_exception",
                "fit_exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "input_frame": dict(frame_audit),
                "input_support": support,
                "fit_audit": None,
                "hazard_table_ages_18_29": [],
                "preflight": {
                    "passed": False,
                    "message": f"{type(error).__name__}: {error}",
                    "expected_checksum_replay": False,
                    "recomputed_checksums": None,
                },
            },
            False,
        )
    if model.fit_audit is None:
        raise ProtocolAbort("final fit returned no fit audit")
    fit_audit = model.fit_audit.canonical_dict()
    replayed_checksums: dict[str, str] | None = None
    try:
        replayed_checksums = dict(
            recompute_support_aware_first_marriage_checksums(model)
        )
        validate_support_aware_first_marriage_fit(
            model,
            expected_checksums=dict(model.fit_audit.checksums),
        )
        hazards = _hazard_table(model)
    except (FirstMarriagePreflightAbort, ValueError) as error:
        return (
            {
                "attempted": True,
                "c": float(c),
                "eligible": False,
                "reason": "final_fit_preflight_failed",
                "fit_exception": None,
                "input_frame": dict(frame_audit),
                "input_support": support,
                "fit_audit": fit_audit,
                "hazard_table_ages_18_29": [],
                "preflight": {
                    "passed": False,
                    "message": str(error),
                    "expected_checksum_replay": False,
                    "recomputed_checksums": replayed_checksums,
                },
            },
            False,
        )
    return (
        {
            "attempted": True,
            "c": float(c),
            "eligible": True,
            "reason": "selected C produced an eligible final <=2014 fit",
            "fit_exception": None,
            "input_frame": dict(frame_audit),
            "input_support": support,
            "fit_audit": fit_audit,
            "hazard_table_ages_18_29": hazards,
            "preflight": {
                "passed": True,
                "message": "all convergence, numeric, and checksum checks pass",
                "expected_checksum_replay": True,
                "recomputed_checksums": replayed_checksums,
            },
        },
        True,
    )


def _protocol_ledger() -> dict[str, Any]:
    return {
        "maximum_information_year": MAX_INFORMATION_YEAR,
        "c_grid": list(C_GRID),
        "pseudo_boundaries": list(PSEUDO_BOUNDARIES),
        "evaluation_windows": {
            str(boundary): list(years)
            for boundary, years in EVALUATION_WINDOWS.items()
        },
        "calendar_year_multiplicity": {
            str(year): count
            for year, count in CALENDAR_YEAR_MULTIPLICITY.items()
        },
        "pseudo_fit_count": len(C_GRID) * len(PSEUDO_BOUNDARIES),
        "eligible_final_fit_count": 1,
        "solver": "scipy.optimize.minimize L-BFGS-B",
        "max_iter": MAX_ITER,
        "tol": SOLVER_TOL,
        "lbfgs_projected_gradient_gtol": SOLVER_TOL,
        "lbfgs_function_reduction_ftol": SOLVER_FTOL,
        "independent_gradient_inf_norm_maximum": GRADIENT_TOL,
        "tie_tolerance": TIE_TOLERANCE,
        "boundary_aggregation": "arithmetic mean of three boundary means",
        "evaluation_deviance": (
            "raw-positive-F6-weighted mean of -2 Bernoulli log likelihood"
        ),
        "canonical_order": ["person_id", "year"],
        "initialization": "all-zero intercept and coefficient vector",
        "randomness_used": False,
        "stochastic_selector_seed": None,
        "first_marriage_section_4_supplies_stochastic_seed": False,
        "earnings_section_6_fit_seed_5200_borrowed": False,
        "earnings_section_6_draw_seeds_6200_6219_borrowed": False,
        "gate_seed_contract_recorded_not_consumed": {
            "gate_seeds": list(range(5)),
            "gate_draw_seeds": list(range(5200, 5220)),
            "consumed": False,
        },
        "candidate_1_score_reads": False,
        "post_2014_rows_enter_fit_or_evaluation_frames": False,
        "m6_scorer_calls": False,
        "gates_yaml_reads": False,
        "runs_reads_or_writes": False,
        "no_registerable_is_publishable": True,
    }


def run_selection(
    frames: SelectionFrames,
    *,
    runtime: Mapping[str, Any],
    freeze: Mapping[str, Any],
    fit_function: FitFunction = fit_support_aware_first_marriage,
) -> dict[str, Any]:
    """Run the complete selector on already constructed train-only frames."""
    _assert_protocol_constants()
    frame_audits, final_frame_audit = _validate_all_frames(frames)
    boundaries: dict[str, Any] = {}
    pseudo_attempts = 0
    for boundary in PSEUDO_BOUNDARIES:
        _progress(f"boundary {boundary}: fitting nine C rungs")
        pair = frames.boundaries[boundary]
        rungs: list[dict[str, Any]] = []
        for c in C_GRID:
            pseudo_attempts += 1
            rung, _model = _fit_rung(
                pair.fit,
                pair.evaluation,
                c=c,
                fit_function=fit_function,
            )
            rungs.append(rung)
        boundaries[str(boundary)] = {
            **frame_audits[str(boundary)],
            "rungs": rungs,
        }
    if pseudo_attempts != 27:
        raise ProtocolAbort(
            f"selector attempted {pseudo_attempts}, not 27 fits"
        )
    selection = _selection_from_boundaries(boundaries)
    selected_c = selection["selected_c"]
    if selected_c is None:
        final_fit = _unattempted_final(
            final_frame_audit, "all pseudo-boundary C values are ineligible"
        )
        status = NO_REGISTERABLE
        final_attempts = 0
    else:
        _progress(f"final <=2014 fit: selected C={selected_c}")
        final_fit, final_eligible = _fit_final(
            frames.final,
            final_frame_audit,
            c=float(selected_c),
            fit_function=fit_function,
        )
        final_attempts = 1
        status = SELECTION_COMPLETE if final_eligible else NO_REGISTERABLE
    ledger = {
        "schema": RAW_SCHEMA,
        "status": status,
        "authority": {
            "program": "docs/design/m6_candidate2_program.md#4",
            "program_merge": "051b4494ecce9345da14d68488bb2833ed476d22",
            "registration_precondition": "section 9.3 box 2",
            "draft_pre_freeze_selector": True,
        },
        "freeze": dict(freeze),
        "runtime": dict(runtime),
        "protocol": _protocol_ledger(),
        "source_audit": dict(frames.source_audit),
        "information_contact": {
            "train_outcomes_through_2014_contacted": True,
            "pseudo_holdout_is_within_train_information": True,
            "raw_retrospective_post_2014_source_values_may_be_read": (
                frames.mode == "real"
            ),
            "post_2014_holdout_truth_table_contacted": False,
            "post_2014_row_entered_fit_or_evaluation_frame": False,
            "candidate_1_or_candidate_2_score_contacted": False,
        },
        "boundaries": boundaries,
        "selection": selection,
        "final_fit": final_fit,
        "fit_counts": {
            "pseudo_boundary_attempts": pseudo_attempts,
            "expected_pseudo_boundary_attempts": 27,
            "final_attempts": final_attempts,
            "total_attempts": pseudo_attempts + final_attempts,
        },
        "registration_disposition": {
            "selected_c": selected_c,
            "registerable_c": (
                selected_c if status == SELECTION_COMPLETE else None
            ),
            "status": status,
            "registration_may_proceed_from_this_ledger": (
                status == SELECTION_COMPLETE
            ),
            "substitution_or_reselection": False,
        },
        "publication": {
            "stdout_documents": 1,
            "strict_json_allow_nan_false": True,
            "all_27_rungs_retained": True,
            "publish_regardless_of_outcome": True,
            "writes_files": False,
            "no_op_is_publishable": True,
        },
    }
    validate_complete_ledger(ledger)
    _strict_json(ledger)
    return ledger


def _require_mapping(
    value: Any, expected_keys: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolAbort(f"{label} must be a JSON object")
    observed = set(value)
    if observed != expected_keys:
        raise ProtocolAbort(
            f"{label} keys differ: "
            f"missing={sorted(expected_keys - observed)}, "
            f"extra={sorted(observed - expected_keys)}"
        )
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolAbort(f"{label} must be a JSON array")
    return value


def _validate_support(value: Any, label: str) -> Mapping[str, Any]:
    support = _require_mapping(
        value,
        {"positive_weight_only", "sex", "pooled_cohort", "sex_by_cohort"},
        label,
    )
    sex = _require_mapping(support["sex"], {"female", "male"}, f"{label}.sex")
    pooled = support["pooled_cohort"]
    sex_cohort = support["sex_by_cohort"]
    if not isinstance(pooled, Mapping) or not isinstance(sex_cohort, Mapping):
        raise ProtocolAbort(f"{label} cohort supports must be JSON objects")
    record_keys = {
        "n_rows",
        "row_weight",
        "n_events",
        "event_weight",
        "age_min",
        "age_max",
    }
    for group_name, group in (
        [(f"sex.{key}", item) for key, item in sex.items()]
        + [(f"pooled_cohort.{key}", item) for key, item in pooled.items()]
        + [(f"sex_by_cohort.{key}", item) for key, item in sex_cohort.items()]
    ):
        _require_mapping(group, record_keys, f"{label}.{group_name}")
    expected_sex_cohort = {
        f"{sex_name}|{cohort}"
        for sex_name in ("female", "male")
        for cohort in pooled
    }
    if set(sex_cohort) != expected_sex_cohort:
        raise ProtocolAbort(
            f"{label}.sex_by_cohort does not publish the full Cartesian grid"
        )
    return support


def _validate_fit_audit(value: Any, label: str) -> Mapping[str, Any]:
    audit_keys = {
        "c",
        "n_input_rows",
        "n_train_rows",
        "n_train_events",
        "n_features",
        "solver_success",
        "solver_status",
        "solver_message",
        "n_iter",
        "max_iter",
        "max_iter_reached",
        "solver_gtol",
        "solver_ftol",
        "warning_count",
        "warning_messages",
        "convergence_warning_count",
        "convergence_warning_messages",
        "objective_value",
        "gradient_inf_norm",
        "intercept",
        "coefficients",
        "design_finite",
        "coefficients_finite",
        "linear_predictor_finite",
        "linear_predictor_min",
        "linear_predictor_max",
        "probabilities_finite",
        "probabilities_strict_unit_interval",
        "probability_min",
        "probability_max",
        "eligible",
        "eligibility_failures",
        "checksums",
        "support",
    }
    audit = _require_mapping(value, audit_keys, label)
    _require_list(audit["warning_messages"], f"{label}.warning_messages")
    _require_list(
        audit["convergence_warning_messages"],
        f"{label}.convergence_warning_messages",
    )
    _require_list(audit["coefficients"], f"{label}.coefficients")
    _require_list(
        audit["eligibility_failures"], f"{label}.eligibility_failures"
    )
    _require_mapping(
        audit["checksums"],
        {
            "canonical_rows_sha256",
            "design_matrix_sha256",
            "support_sha256",
            "standardization_sha256",
            "normalized_weight_sha256",
            "coefficient_sha256",
            "selected_c_sha256",
        },
        f"{label}.checksums",
    )
    _validate_support(audit["support"], f"{label}.support")
    if float(audit["solver_gtol"]) != SOLVER_TOL:
        raise ProtocolAbort(f"{label}.solver_gtol drifted")
    if float(audit["solver_ftol"]) != SOLVER_FTOL:
        raise ProtocolAbort(f"{label}.solver_ftol drifted")
    if int(audit["max_iter"]) != MAX_ITER:
        raise ProtocolAbort(f"{label}.max_iter drifted")
    if int(audit["warning_count"]) != len(audit["warning_messages"]):
        raise ProtocolAbort(f"{label}.warning_count does not reconcile")
    if int(audit["convergence_warning_count"]) != len(
        audit["convergence_warning_messages"]
    ):
        raise ProtocolAbort(
            f"{label}.convergence_warning_count does not reconcile"
        )
    return audit


def _validate_frame_audit(value: Any, label: str) -> Mapping[str, Any]:
    return _require_mapping(
        value,
        {
            "label",
            "n_rows",
            "positive_weight_rows",
            "event_rows",
            "positive_weight_event_rows",
            "row_weight",
            "event_weight",
            "year_min",
            "year_max",
            "required_interview_year_min",
            "required_interview_year_max",
            "asserted_year_minimum",
            "asserted_year_maximum",
            "asserted_interview_maximum",
            "field_aware_boundary_pass",
            "canonical_person_year_order",
            "frame_sha256",
        },
        label,
    )


def _validate_complete_ledger(ledger: Mapping[str, Any]) -> None:
    """Implementation for the public malformed-ledger-safe validator."""
    required_top = {
        "schema",
        "status",
        "authority",
        "freeze",
        "runtime",
        "protocol",
        "source_audit",
        "information_contact",
        "boundaries",
        "selection",
        "final_fit",
        "fit_counts",
        "registration_disposition",
        "publication",
    }
    ledger = _require_mapping(ledger, required_top, "full-ledger top-level")
    if ledger["schema"] != RAW_SCHEMA:
        raise ProtocolAbort("unexpected first-marriage full-ledger schema")
    authority = _require_mapping(
        ledger["authority"],
        {
            "program",
            "program_merge",
            "registration_precondition",
            "draft_pre_freeze_selector",
        },
        "authority",
    )
    if authority != {
        "program": "docs/design/m6_candidate2_program.md#4",
        "program_merge": "051b4494ecce9345da14d68488bb2833ed476d22",
        "registration_precondition": "section 9.3 box 2",
        "draft_pre_freeze_selector": True,
    }:
        raise ProtocolAbort("authority values drifted")
    if not isinstance(ledger["freeze"], Mapping):
        raise ProtocolAbort("freeze must be a JSON object")
    if not isinstance(ledger["runtime"], Mapping):
        raise ProtocolAbort("runtime must be a JSON object")
    if not isinstance(ledger["source_audit"], Mapping):
        raise ProtocolAbort("source_audit must be a JSON object")
    if _canonical_bytes(ledger["protocol"]) != _canonical_bytes(
        _protocol_ledger()
    ):
        raise ProtocolAbort("protocol ledger differs from frozen constants")
    information_contact = _require_mapping(
        ledger["information_contact"],
        {
            "train_outcomes_through_2014_contacted",
            "pseudo_holdout_is_within_train_information",
            "raw_retrospective_post_2014_source_values_may_be_read",
            "post_2014_holdout_truth_table_contacted",
            "post_2014_row_entered_fit_or_evaluation_frame",
            "candidate_1_or_candidate_2_score_contacted",
        },
        "information_contact",
    )
    if information_contact != {
        "train_outcomes_through_2014_contacted": True,
        "pseudo_holdout_is_within_train_information": True,
        "raw_retrospective_post_2014_source_values_may_be_read": (
            ledger["source_audit"].get("mode") == "real_train_only_psid"
        ),
        "post_2014_holdout_truth_table_contacted": False,
        "post_2014_row_entered_fit_or_evaluation_frame": False,
        "candidate_1_or_candidate_2_score_contacted": False,
    }:
        raise ProtocolAbort("information-contact values drifted")
    boundaries = ledger["boundaries"]
    expected_boundary_keys = {str(value) for value in PSEUDO_BOUNDARIES}
    boundaries = _require_mapping(
        boundaries, expected_boundary_keys, "full-ledger boundaries"
    )
    rung_keys = {
        "c",
        "eligible",
        "eligibility_failures",
        "fit_exception",
        "input_support",
        "fit_audit",
        "evaluation",
    }
    boundary_keys = {
        "fit_frame",
        "evaluation_frame",
        "intended_evaluation_years",
        "observed_evaluation_years",
        "calendar_rows",
        "rungs",
    }
    evaluation_keys = {
        "attempted",
        "eligible",
        "failure",
        "n_rows",
        "n_events",
        "row_weight",
        "event_weight",
        "weighted_deviance_numerator",
        "weighted_mean_bernoulli_deviance",
        "linear_predictor_min",
        "linear_predictor_max",
        "probability_min",
        "probability_max",
        "prediction_sha256",
    }
    for boundary in PSEUDO_BOUNDARIES:
        record = _require_mapping(
            boundaries[str(boundary)],
            boundary_keys,
            f"boundary {boundary}",
        )
        _validate_frame_audit(
            record["fit_frame"], f"boundary {boundary}.fit_frame"
        )
        _validate_frame_audit(
            record["evaluation_frame"],
            f"boundary {boundary}.evaluation_frame",
        )
        intended = _require_list(
            record["intended_evaluation_years"],
            f"boundary {boundary}.intended_evaluation_years",
        )
        observed = _require_list(
            record["observed_evaluation_years"],
            f"boundary {boundary}.observed_evaluation_years",
        )
        if intended != list(EVALUATION_WINDOWS[boundary]):
            raise ProtocolAbort(f"boundary {boundary} intended years drifted")
        if any(year not in intended for year in observed):
            raise ProtocolAbort(f"boundary {boundary} observed years drifted")
        calendar_rows = _require_mapping(
            record["calendar_rows"],
            {str(year) for year in EVALUATION_WINDOWS[boundary]},
            f"boundary {boundary}.calendar_rows",
        )
        if any(int(count) < 0 for count in calendar_rows.values()):
            raise ProtocolAbort(f"boundary {boundary} has negative row count")
        rungs = _require_list(record["rungs"], f"boundary {boundary}.rungs")
        if len(rungs) != len(C_GRID):
            raise ProtocolAbort(
                f"boundary {boundary} does not retain nine rungs"
            )
        for index, rung_value in enumerate(rungs):
            rung = _require_mapping(
                rung_value,
                rung_keys,
                f"boundary {boundary}.rungs[{index}]",
            )
            if float(rung["c"]) != C_GRID[index]:
                raise ProtocolAbort(f"boundary {boundary} C order drifted")
            _require_list(
                rung["eligibility_failures"],
                f"boundary {boundary}.rungs[{index}].eligibility_failures",
            )
            if rung["fit_exception"] is not None:
                _require_mapping(
                    rung["fit_exception"],
                    {"type", "message"},
                    f"boundary {boundary}.rungs[{index}].fit_exception",
                )
            support = _validate_support(
                rung["input_support"],
                f"boundary {boundary}.rungs[{index}].input_support",
            )
            evaluation = _require_mapping(
                rung["evaluation"],
                evaluation_keys,
                f"boundary {boundary}.rungs[{index}].evaluation",
            )
            if (
                evaluation["eligible"]
                and evaluation["weighted_mean_bernoulli_deviance"] is None
            ):
                raise ProtocolAbort("eligible evaluation lacks deviance")
            if rung["fit_audit"] is not None:
                fit_audit = _validate_fit_audit(
                    rung["fit_audit"],
                    f"boundary {boundary}.rungs[{index}].fit_audit",
                )
                if float(fit_audit["c"]) != float(rung["c"]):
                    raise ProtocolAbort("rung and fit-audit C differ")
                if _canonical_bytes(fit_audit["support"]) != _canonical_bytes(
                    support
                ):
                    raise ProtocolAbort(
                        "component and selector support ledgers do not match"
                    )
                expected_eligible = bool(
                    fit_audit["eligible"] and evaluation["eligible"]
                )
                if bool(rung["eligible"]) != expected_eligible:
                    raise ProtocolAbort(
                        "rung eligibility differs from fit/evaluation audits"
                    )
            elif rung["fit_exception"] is None:
                raise ProtocolAbort(
                    "rung without fit audit must retain a fit exception"
                )
            elif rung["eligible"]:
                raise ProtocolAbort("fit-exception rung cannot be eligible")
    selection = _require_mapping(
        ledger["selection"],
        {
            "candidates",
            "eligible_cs",
            "minimum_mean_deviance",
            "tie_tolerance",
            "tie_set",
            "selected_c",
            "reason",
            "candidate_1_score_consulted",
            "post_2014_row_entered_numerical_selection",
            "substitution_or_reselection",
        },
        "selection",
    )
    candidates = _require_list(selection["candidates"], "selection.candidates")
    if len(candidates) != len(C_GRID):
        raise ProtocolAbort(
            "selection does not retain nine candidate summaries"
        )
    for index, candidate in enumerate(candidates):
        candidate = _require_mapping(
            candidate,
            {
                "c",
                "eligible_at_all_boundaries",
                "boundary_deviances",
                "equal_boundary_mean_deviance",
            },
            f"selection.candidates[{index}]",
        )
        if float(candidate["c"]) != C_GRID[index]:
            raise ProtocolAbort("selection candidate C order drifted")
        _require_mapping(
            candidate["boundary_deviances"],
            expected_boundary_keys,
            f"selection.candidates[{index}].boundary_deviances",
        )
    _require_list(selection["eligible_cs"], "selection.eligible_cs")
    _require_list(selection["tie_set"], "selection.tie_set")
    recomputed = _selection_from_boundaries(boundaries)
    if _canonical_bytes(recomputed) != _canonical_bytes(selection):
        raise ProtocolAbort(
            "published selection differs from all-rung reduction"
        )
    final_fit = _require_mapping(
        ledger["final_fit"],
        {
            "attempted",
            "c",
            "eligible",
            "reason",
            "fit_exception",
            "input_frame",
            "input_support",
            "fit_audit",
            "hazard_table_ages_18_29",
            "preflight",
        },
        "final_fit",
    )
    _validate_frame_audit(final_fit["input_frame"], "final_fit.input_frame")
    if final_fit["fit_exception"] is not None:
        _require_mapping(
            final_fit["fit_exception"],
            {"type", "message"},
            "final_fit.fit_exception",
        )
    if final_fit["input_support"] is not None:
        final_support = _validate_support(
            final_fit["input_support"], "final_fit.input_support"
        )
    else:
        final_support = None
    if final_fit["fit_audit"] is not None:
        final_audit = _validate_fit_audit(
            final_fit["fit_audit"], "final_fit.fit_audit"
        )
        if final_support is None or _canonical_bytes(
            final_audit["support"]
        ) != _canonical_bytes(final_support):
            raise ProtocolAbort("final fit and input supports differ")
        if final_fit["c"] != final_audit["c"]:
            raise ProtocolAbort("final fit and fit-audit C differ")
    hazards = _require_list(
        final_fit["hazard_table_ages_18_29"],
        "final_fit.hazard_table_ages_18_29",
    )
    hazard_keys = {
        "sex",
        "target_birth_decade",
        "mapped_birth_decade",
        "age",
        "global_age_evaluated",
        "cohort_age_evaluated",
        "global_boundary_evaluated",
        "cohort_boundary_evaluated",
        "linear_predictor",
        "probability",
    }
    for index, hazard in enumerate(hazards):
        _require_mapping(
            hazard, hazard_keys, f"final_fit.hazard_table[{index}]"
        )
    preflight = _require_mapping(
        final_fit["preflight"],
        {
            "passed",
            "message",
            "expected_checksum_replay",
            "recomputed_checksums",
        },
        "final_fit.preflight",
    )
    if preflight["recomputed_checksums"] is not None:
        replayed = _require_mapping(
            preflight["recomputed_checksums"],
            {
                "canonical_rows_sha256",
                "design_matrix_sha256",
                "support_sha256",
                "standardization_sha256",
                "normalized_weight_sha256",
                "coefficient_sha256",
                "selected_c_sha256",
            },
            "final_fit.preflight.recomputed_checksums",
        )
        if final_fit["fit_audit"] is None or _canonical_bytes(
            replayed
        ) != _canonical_bytes(final_fit["fit_audit"]["checksums"]):
            raise ProtocolAbort("final recomputed checksums differ from audit")
    counts = _require_mapping(
        ledger["fit_counts"],
        {
            "pseudo_boundary_attempts",
            "expected_pseudo_boundary_attempts",
            "final_attempts",
            "total_attempts",
        },
        "fit_counts",
    )
    if int(counts["pseudo_boundary_attempts"]) != 27:
        raise ProtocolAbort("full ledger does not record 27 pseudo attempts")
    if int(counts["expected_pseudo_boundary_attempts"]) != 27:
        raise ProtocolAbort("expected pseudo attempt count drifted")
    if int(counts["total_attempts"]) != 27 + int(counts["final_attempts"]):
        raise ProtocolAbort("total fit attempt count does not reconcile")
    disposition = _require_mapping(
        ledger["registration_disposition"],
        {
            "selected_c",
            "registerable_c",
            "status",
            "registration_may_proceed_from_this_ledger",
            "substitution_or_reselection",
        },
        "registration_disposition",
    )
    publication = _require_mapping(
        ledger["publication"],
        {
            "stdout_documents",
            "strict_json_allow_nan_false",
            "all_27_rungs_retained",
            "publish_regardless_of_outcome",
            "writes_files",
            "no_op_is_publishable",
        },
        "publication",
    )
    if publication != {
        "stdout_documents": 1,
        "strict_json_allow_nan_false": True,
        "all_27_rungs_retained": True,
        "publish_regardless_of_outcome": True,
        "writes_files": False,
        "no_op_is_publishable": True,
    }:
        raise ProtocolAbort("publication contract values drifted")
    status = ledger["status"]
    if status not in (SELECTION_COMPLETE, NO_REGISTERABLE):
        raise ProtocolAbort(f"unknown selection status {status!r}")
    if disposition["status"] != status:
        raise ProtocolAbort("registration disposition status differs")
    if disposition["selected_c"] != selection["selected_c"]:
        raise ProtocolAbort("registration and selection selected C differ")
    expected_final_attempts = 1 if selection["selected_c"] is not None else 0
    if int(counts["final_attempts"]) != expected_final_attempts:
        raise ProtocolAbort(
            "final attempt count differs from selected-C state"
        )
    if bool(final_fit["attempted"]) != bool(expected_final_attempts):
        raise ProtocolAbort(
            "final attempted flag differs from selected-C state"
        )
    if final_fit["c"] != selection["selected_c"]:
        raise ProtocolAbort("final fit C differs from selected C")
    if status == SELECTION_COMPLETE:
        if not final_fit["eligible"] or not final_fit["preflight"]["passed"]:
            raise ProtocolAbort("complete selection has ineligible final fit")
        if (
            final_fit["preflight"]["expected_checksum_replay"] is not True
            or final_fit["preflight"]["recomputed_checksums"] is None
        ):
            raise ProtocolAbort("complete selection lacks checksum replay")
        if int(counts["final_attempts"]) != 1:
            raise ProtocolAbort("complete selection lacks its one final fit")
        if disposition["registerable_c"] != selection["selected_c"]:
            raise ProtocolAbort("complete selection registerable C differs")
        if (
            disposition["registration_may_proceed_from_this_ledger"]
            is not True
        ):
            raise ProtocolAbort(
                "complete selection disposition cannot proceed"
            )
    elif disposition["registerable_c"] is not None:
        raise ProtocolAbort(
            "no-registerable ledger publishes a registerable C"
        )
    elif disposition["registration_may_proceed_from_this_ledger"] is not False:
        raise ProtocolAbort("no-registerable disposition may not proceed")
    if disposition["substitution_or_reselection"] is not False:
        raise ProtocolAbort("substitution or reselection is forbidden")


def validate_complete_ledger(ledger: Mapping[str, Any]) -> None:
    """Reject malformed, incomplete, extra, or stale ledger content."""
    try:
        _validate_complete_ledger(ledger)
    except ProtocolAbort:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ProtocolAbort(
            f"malformed nested full ledger: {type(error).__name__}: {error}"
        ) from error


def reduce_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a validated full ledger while retaining null/zero support cells."""
    validate_complete_ledger(ledger)
    boundaries: dict[str, Any] = {}
    for boundary in PSEUDO_BOUNDARIES:
        source = ledger["boundaries"][str(boundary)]
        boundaries[str(boundary)] = {
            "fit_frame": source["fit_frame"],
            "evaluation_frame": source["evaluation_frame"],
            "calendar_rows": source["calendar_rows"],
            "rungs": [
                {
                    "c": rung["c"],
                    "eligible": rung["eligible"],
                    "eligibility_failures": rung["eligibility_failures"],
                    "input_support": rung["input_support"],
                    "fit_eligible": (
                        rung["fit_audit"]["eligible"]
                        if rung["fit_audit"] is not None
                        else False
                    ),
                    "fit_checksums": (
                        rung["fit_audit"]["checksums"]
                        if rung["fit_audit"] is not None
                        else None
                    ),
                    "deviance": rung["evaluation"][
                        "weighted_mean_bernoulli_deviance"
                    ],
                }
                for rung in source["rungs"]
            ],
        }
    reduced = {
        "schema": REDUCED_SCHEMA,
        "source_schema": RAW_SCHEMA,
        "source_ledger_sha256": _sha256_bytes(_canonical_bytes(ledger)),
        "status": ledger["status"],
        "calendar_year_multiplicity": ledger["protocol"][
            "calendar_year_multiplicity"
        ],
        "boundaries": boundaries,
        "selection": ledger["selection"],
        "final_fit": ledger["final_fit"],
        "fit_counts": ledger["fit_counts"],
        "registration_disposition": ledger["registration_disposition"],
    }
    _strict_json(reduced)
    return reduced


def _native_family_context(
    raw_demo: pd.DataFrame,
    raw_marriages: pd.DataFrame,
    birth_records: pd.DataFrame,
    raw_deaths: pd.DataFrame,
) -> tuple[Any, dict[str, Any]]:
    """Build the native full-history family context before field truncation.

    This intentionally matches the registered M6 household-source choreography:
    the marital panel, marriage-order map, and latest-positive panel weights are
    constructed from the full native retrospective sources.  Only train-ID
    eligibility is identified from positive demographic rows at or before
    ``T*``.  ``_truncate_family_context`` is the sole operation that removes
    post-boundary information from a consumed fit context.
    """
    from populace_dynamics.data import transitions
    from populace_dynamics.models.family_transitions import registry
    from populace_dynamics.models.family_transitions.common import (
        marriage_order_map,
    )

    if not birth_records.empty:
        raise ProtocolAbort(
            "first-marriage selector must not load or consume birth history"
        )
    positive_weight = raw_demo[raw_demo["weight"] > 0]
    person_weight = (
        positive_weight.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    native_panel = transitions.build_marital_panel(
        raw_marriages,
        raw_deaths,
        person_weight,
    )
    order_map = marriage_order_map(raw_marriages)
    preboundary_demo = raw_demo[
        (raw_demo["period"] <= MAX_INFORMATION_YEAR) & (raw_demo["weight"] > 0)
    ]
    preboundary_demo_ids = {
        int(value) for value in preboundary_demo["person_id"].unique()
    }
    train_ids = frozenset(
        preboundary_demo_ids
        & {int(value) for value in native_panel.attrs["person_id"].unique()}
    )
    context = registry.FitContext(
        panel=native_panel,
        demographic_panel=raw_demo,
        marriage_records=raw_marriages,
        birth_records=birth_records,
        marriage_order_map=order_map,
        train_ids=train_ids,
    )
    return context, {
        "construction": (
            "full native retrospective panel/order map first; field-aware "
            "context truncation second"
        ),
        "matches_registered_household_source_load_order": True,
        "native_marriage_records_pretruncated": False,
        "native_death_records_presanitized": False,
        "native_demographic_rows_pretruncated": False,
        "latest_positive_weights_use_full_native_demographic_panel": True,
        "train_ids_use_positive_demographic_rows_through_2014": True,
        "latest_positive_weight_persons": int(len(person_weight)),
        "preboundary_positive_demo_persons": int(len(preboundary_demo_ids)),
        "family_train_ids": int(len(train_ids)),
        "native_marital_person_years": int(len(native_panel.person_years)),
        "native_marital_events": int(len(native_panel.events)),
    }


def _context_first_marriage_frame(context: Any, label: str) -> pd.DataFrame:
    from populace_dynamics.engine import refit

    person_years = context.panel.person_years[
        context.panel.person_years["person_id"].isin(context.train_ids)
        & context.panel.person_years["marital_state"].eq("never_married")
    ].copy()
    if refit.REQUIRED_INTERVIEW_COLUMN not in person_years:
        raise ProtocolAbort(f"{label} lacks required interview provenance")
    attrs = context.panel.attrs.set_index("person_id")
    birth_decade = (attrs["birth_year"] // 10 * 10).astype("int64")
    person_years["birth_decade"] = person_years["person_id"].map(birth_decade)
    events = context.panel.events[
        context.panel.events["person_id"].isin(context.train_ids)
        & context.panel.events["transition"].eq("first_marriage")
    ]
    event_keys = {
        (int(person), int(year))
        for person, year in zip(
            events["person_id"], events["year"], strict=True
        )
    }
    person_years["event"] = [
        (int(person), int(year)) in event_keys
        for person, year in zip(
            person_years["person_id"], person_years["year"], strict=True
        )
    ]
    risk_keys = set(
        zip(
            person_years["person_id"].astype(int),
            person_years["year"].astype(int),
            strict=True,
        )
    )
    unmatched = event_keys - risk_keys
    if unmatched:
        raise ProtocolAbort(
            f"{label} has first-marriage events outside the risk frame"
        )
    person_years = person_years.rename(
        columns={refit.REQUIRED_INTERVIEW_COLUMN: "required_interview_year"}
    )
    return _canonical_frame(person_years, label)


def _assert_context_boundary(
    context: Any, boundary: int, label: str
) -> dict[str, Any]:
    """Assert every date-bearing source field in a fitted context."""
    from populace_dynamics.engine import refit

    person_years = context.panel.person_years
    events = context.panel.events
    for frame_name, frame in (
        ("panel_person_years", person_years),
        ("panel_events", events),
    ):
        for required in ("year", refit.REQUIRED_INTERVIEW_COLUMN):
            if required not in frame:
                raise ProtocolAbort(f"{label}.{frame_name} lacks {required}")
            numeric = pd.to_numeric(frame[required], errors="coerce")
            if numeric.isna().any():
                raise ProtocolAbort(
                    f"{label}.{frame_name}.{required} has missing provenance"
                )

    actual_marriages = context.marriage_records
    if "is_marriage" in actual_marriages:
        actual_marriages = actual_marriages[
            actual_marriages["is_marriage"].fillna(False)
        ]
    actual_births = context.birth_records
    if "is_event" in actual_births:
        actual_births = actual_births[actual_births["is_event"].fillna(False)]
    frames_and_fields = (
        ("panel_person_years", person_years, "year"),
        (
            "panel_person_years",
            person_years,
            refit.REQUIRED_INTERVIEW_COLUMN,
        ),
        ("panel_events", events, "year"),
        ("panel_events", events, refit.REQUIRED_INTERVIEW_COLUMN),
        ("panel_attrs", context.panel.attrs, "start_exposure_year"),
        ("panel_attrs", context.panel.attrs, "censor_year"),
        ("demographic_panel", context.demographic_panel, "period"),
        ("actual_marriages", actual_marriages, "start_year"),
        ("actual_marriages", actual_marriages, "end_year"),
        ("actual_marriages", actual_marriages, "separation_year"),
        (
            "actual_marriages",
            actual_marriages,
            "most_recent_report_year",
        ),
        ("actual_births", actual_births, "birth_year"),
        (
            "actual_births",
            actual_births,
            "most_recent_child_report_year",
        ),
        ("marriage_order_map", context.marriage_order_map, "start_year"),
    )
    maxima: dict[str, int | None] = {}
    for frame_name, frame, column in frames_and_fields:
        if column not in frame:
            continue
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ProtocolAbort(
                f"{label}.{frame_name}.{column} has a nonnumeric date"
            )
        observed = numeric.dropna()
        maximum = int(observed.max()) if len(observed) else None
        if maximum is not None and maximum > boundary:
            raise ProtocolAbort(
                f"{label}.{frame_name}.{column} reaches {maximum}, "
                f"beyond {boundary}"
            )
        maxima[f"{frame_name}.{column}"] = maximum
    return {
        "boundary": boundary,
        "field_maxima": maxima,
        "every_relevant_context_date_field_asserted": True,
        "person_year_and_event_interview_provenance_nonmissing": True,
    }


def _load_real_selection_frames() -> SelectionFrames:
    """Build field-bounded frames from native retrospective fit inputs."""
    from populace_dynamics.data import deaths, marriage, panels
    from populace_dynamics.engine import refit

    _progress("loading staged PSID first-marriage fit sources (read-only)")
    raw_demo = panels.demographic_panel()
    raw_marriages = marriage.marriage_history()
    raw_deaths = deaths.read_death_records()
    context, native_context_audit = _native_family_context(
        raw_demo,
        raw_marriages,
        pd.DataFrame(),
        raw_deaths,
    )
    final_context = refit._truncate_family_context(
        context, MAX_INFORMATION_YEAR
    )
    demo = final_context.demographic_panel
    marriages = final_context.marriage_records
    actual_marriages = marriages[marriages["is_marriage"].fillna(False)]
    final_context_audit = _assert_context_boundary(
        final_context, MAX_INFORMATION_YEAR, "final_context"
    )
    final_frame = _context_first_marriage_frame(
        final_context, "final_2014.fit"
    )
    boundary_frames: dict[int, BoundaryFrames] = {}
    context_boundary_audits: dict[str, Any] = {}
    for boundary in PSEUDO_BOUNDARIES:
        truncated = refit._truncate_family_context(context, boundary)
        context_boundary_audits[str(boundary)] = _assert_context_boundary(
            truncated, boundary, f"boundary_{boundary}_context"
        )
        fit_frame = _context_first_marriage_frame(
            truncated, f"boundary_{boundary}.fit"
        )
        years = EVALUATION_WINDOWS[boundary]
        evaluation = final_frame[final_frame["year"].isin(years)].copy()
        evaluation = _canonical_frame(
            evaluation, f"boundary_{boundary}.evaluation"
        )
        boundary_frames[boundary] = BoundaryFrames(
            fit=fit_frame,
            evaluation=evaluation,
        )
    source_audit = {
        "mode": "real_train_only_psid",
        "raw_source_is_retrospective_product": True,
        "raw_retrospective_sources_read_before_boundary_filter": True,
        "raw_post_2014_values_may_be_read_for_native_panel_construction": True,
        "retrospective_post_2014_report_products_can_establish_pre_2015_history": (
            True
        ),
        "selection_frames_contain_post_2014_date_values": False,
        "native_panel_built_before_field_aware_context_truncation": True,
        "maximum_information_year": MAX_INFORMATION_YEAR,
        "calendar_2014_rows_require_interview_by_2014": True,
        "birth_history_loaded": False,
        "holdout_truth_loaded_or_built": False,
        "candidate_artifact_loaded": False,
        "native_context_construction": native_context_audit,
        "context_boundary_audits": {
            **context_boundary_audits,
            str(MAX_INFORMATION_YEAR): final_context_audit,
        },
        "raw_rows": {
            "demographic": int(len(raw_demo)),
            "marriage_records": int(len(raw_marriages)),
            "deaths": int(len(raw_deaths)),
        },
        "raw_retrospective_value_counts": {
            "demographic_rows_after_2014": int(
                (
                    pd.to_numeric(raw_demo["period"], errors="coerce") > 2014
                ).sum()
            ),
            "marriage_records_reported_after_2014": int(
                (
                    pd.to_numeric(
                        raw_marriages["most_recent_report_year"],
                        errors="coerce",
                    )
                    > 2014
                ).sum()
            ),
            "pre_2015_marriages_reported_after_2014": int(
                (
                    raw_marriages["is_marriage"].fillna(False)
                    & (
                        pd.to_numeric(
                            raw_marriages["start_year"], errors="coerce"
                        )
                        <= 2014
                    )
                    & (
                        pd.to_numeric(
                            raw_marriages["most_recent_report_year"],
                            errors="coerce",
                        )
                        > 2014
                    )
                ).sum()
            ),
            "death_years_after_2014": int(
                (
                    pd.to_numeric(raw_deaths["death_year"], errors="coerce")
                    > 2014
                ).sum()
            ),
        },
        "raw_max_year": {
            "demographic_period": _max_numeric(raw_demo, "period"),
            "marriage_start": _max_numeric(raw_marriages, "start_year"),
            "marriage_report": _max_numeric(
                raw_marriages, "most_recent_report_year"
            ),
            "death": _max_numeric(raw_deaths, "death_year"),
            "native_marital_person_year": _max_numeric(
                context.panel.person_years, "year"
            ),
            "native_marital_event": _max_numeric(context.panel.events, "year"),
        },
        "sanitized_rows": {
            "demographic": int(len(demo)),
            "marriage_records": int(len(marriages)),
            "actual_marriages": int(len(actual_marriages)),
            "marital_person_years": int(len(final_context.panel.person_years)),
            "marital_events": int(len(final_context.panel.events)),
            "first_marriage_risk": int(len(final_frame)),
        },
        "sanitized_max_year": {
            "demographic_period": _max_numeric(demo, "period"),
            "actual_marriage_start": _max_numeric(
                actual_marriages, "start_year"
            ),
            "actual_marriage_report": _max_numeric(
                actual_marriages, "most_recent_report_year"
            ),
            "marital_person_year": _max_numeric(
                final_context.panel.person_years, "year"
            ),
            "marital_person_year_required_interview": _max_numeric(
                final_context.panel.person_years,
                refit.REQUIRED_INTERVIEW_COLUMN,
            ),
            "marital_event": _max_numeric(final_context.panel.events, "year"),
            "marital_event_required_interview": _max_numeric(
                final_context.panel.events,
                refit.REQUIRED_INTERVIEW_COLUMN,
            ),
        },
        "checksums": {
            "raw_demographic": _selected_frame_checksum(
                raw_demo,
                ("person_id", "period", "weight", "interview"),
                "raw_demographic",
            ),
            "raw_marriage_records": _selected_frame_checksum(
                raw_marriages,
                (
                    "person_id",
                    "start_year",
                    "end_year",
                    "how_ended",
                    "most_recent_report_year",
                    "is_marriage",
                ),
                "raw_marriage_records",
            ),
            "raw_deaths": _selected_frame_checksum(
                raw_deaths,
                ("person_id", "death_year"),
                "raw_deaths",
            ),
            "demographic": _selected_frame_checksum(
                demo,
                ("person_id", "period", "weight", "interview"),
                "demographic",
            ),
            "marriage_records": _selected_frame_checksum(
                marriages,
                (
                    "person_id",
                    "start_year",
                    "end_year",
                    "how_ended",
                    "most_recent_report_year",
                    "is_marriage",
                ),
                "marriage_records",
            ),
            "native_marital_person_years": _selected_frame_checksum(
                context.panel.person_years,
                ("person_id", "year", "marital_state", "weight"),
                "native_marital_person_years",
            ),
            "native_marital_events": _selected_frame_checksum(
                context.panel.events,
                ("person_id", "year", "transition", "weight"),
                "native_marital_events",
            ),
            "marital_person_years": _selected_frame_checksum(
                final_context.panel.person_years,
                (
                    "person_id",
                    "year",
                    refit.REQUIRED_INTERVIEW_COLUMN,
                    "marital_state",
                    "weight",
                ),
                "marital_person_years",
            ),
            "marital_events": _selected_frame_checksum(
                final_context.panel.events,
                (
                    "person_id",
                    "year",
                    refit.REQUIRED_INTERVIEW_COLUMN,
                    "transition",
                    "weight",
                ),
                "marital_events",
            ),
            "final_first_marriage_frame": _frame_checksum(final_frame),
        },
        "train_id_count": int(len(context.train_ids)),
    }
    return SelectionFrames(
        boundaries=boundary_frames,
        final=final_frame,
        source_audit=source_audit,
        mode="real",
    )


def _synthetic_frames(*, all_events_zero: bool = False) -> SelectionFrames:
    """Build deterministic frames with empty and zero-event support cells."""
    rows: list[dict[str, Any]] = []
    person_id = 1
    for year in range(1998, MAX_INFORMATION_YEAR + 1):
        for cohort in (1970, 1980, 1990):
            sexes = ("male",) if cohort == 1990 else ("female", "male")
            for sex in sexes:
                for replicate in range(8):
                    birth_year = cohort + replicate
                    age = year - birth_year
                    if age < 15:
                        continue
                    event = (
                        year + replicate + cohort // 10 + (sex == "male") * 3
                    ) % 13 == 0
                    if sex == "female" and cohort == 1980:
                        event = False
                    if all_events_zero:
                        event = False
                    rows.append(
                        {
                            "person_id": person_id,
                            "year": year,
                            "required_interview_year": year,
                            "age": age,
                            "sex": sex,
                            "weight": 0.75 + 0.1 * (replicate % 4),
                            "birth_decade": cohort,
                            "event": event,
                        }
                    )
                    person_id += 1
    full = _canonical_frame(pd.DataFrame(rows), "synthetic.final")
    boundaries = {
        boundary: BoundaryFrames(
            fit=_canonical_frame(
                full[
                    (full["year"] <= boundary)
                    & (full["required_interview_year"] <= boundary)
                ],
                f"synthetic.boundary_{boundary}.fit",
            ),
            evaluation=_canonical_frame(
                full[full["year"].isin(EVALUATION_WINDOWS[boundary])],
                f"synthetic.boundary_{boundary}.evaluation",
            ),
        )
        for boundary in PSEUDO_BOUNDARIES
    }
    return SelectionFrames(
        boundaries=boundaries,
        final=full,
        source_audit={
            "mode": "synthetic",
            "all_events_zero": all_events_zero,
            "empty_sex_by_cohort_cell": "female|1990",
            "zero_event_positive_exposure_cell": "female|1980",
            "post_2014_rows": 0,
            "holdout_truth_loaded_or_built": False,
        },
        mode="synthetic",
    )


def _synthetic_smoke(runtime: Mapping[str, Any]) -> dict[str, Any]:
    synthetic_freeze = {
        "synthetic_only": True,
        "repository_clean_freeze_required": False,
    }
    happy = run_selection(
        _synthetic_frames(),
        runtime=runtime,
        freeze=synthetic_freeze,
    )
    no_op = run_selection(
        _synthetic_frames(all_events_zero=True),
        runtime=runtime,
        freeze=synthetic_freeze,
    )
    happy_round_trip = json.loads(_strict_json(happy))
    no_op_round_trip = json.loads(_strict_json(no_op))
    happy_reduced = reduce_ledger(happy_round_trip)
    no_op_reduced = reduce_ledger(no_op_round_trip)
    empty_cell = happy["final_fit"]["fit_audit"]["support"]["sex_by_cohort"][
        "female|1990"
    ]
    zero_event_cell = happy["final_fit"]["fit_audit"]["support"][
        "sex_by_cohort"
    ]["female|1980"]
    if empty_cell["n_rows"] != 0 or empty_cell["age_min"] is not None:
        raise ProtocolAbort(
            "synthetic empty cell was not published explicitly"
        )
    if zero_event_cell["n_rows"] <= 0 or zero_event_cell["n_events"] != 0:
        raise ProtocolAbort("synthetic zero-event cell was not preserved")
    if happy["fit_counts"] != {
        "pseudo_boundary_attempts": 27,
        "expected_pseudo_boundary_attempts": 27,
        "final_attempts": 1,
        "total_attempts": 28,
    }:
        raise ProtocolAbort("happy synthetic did not execute 28 fits")
    if no_op["status"] != NO_REGISTERABLE:
        raise ProtocolAbort("all-ineligible synthetic did not publish no-op")
    if (
        no_op["fit_counts"]["pseudo_boundary_attempts"] != 27
        or no_op["fit_counts"]["final_attempts"] != 0
    ):
        raise ProtocolAbort("no-op synthetic fit counts drifted")
    return {
        "schema": SMOKE_SCHEMA,
        "status": "PASS",
        "runtime": dict(runtime),
        "happy_28_fit_case": {
            "status": happy["status"],
            "fit_counts": happy["fit_counts"],
            "selection": happy["selection"],
            "empty_sex_by_cohort_cell": empty_cell,
            "zero_event_positive_exposure_cell": zero_event_cell,
            "full_ledger_sha256": _sha256_bytes(_canonical_bytes(happy)),
            "reduced_ledger_sha256": _sha256_bytes(
                _canonical_bytes(happy_reduced)
            ),
            "publication_and_reduction_round_trip": True,
        },
        "all_ineligible_no_op_case": {
            "status": no_op["status"],
            "fit_counts": no_op["fit_counts"],
            "selection": no_op["selection"],
            "registration_disposition": no_op["registration_disposition"],
            "all_27_rungs_retained": all(
                len(no_op["boundaries"][str(boundary)]["rungs"]) == 9
                for boundary in PSEUDO_BOUNDARIES
            ),
            "full_ledger_sha256": _sha256_bytes(_canonical_bytes(no_op)),
            "reduced_ledger_sha256": _sha256_bytes(
                _canonical_bytes(no_op_reduced)
            ),
            "publication_and_reduction_round_trip": True,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="run deterministic synthetic publication/reduction checks",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime = _runtime_audit()
    _assert_protocol_constants()
    if args.synthetic_smoke:
        print(_strict_json(_synthetic_smoke(runtime)))
        return 0
    freeze = _repository_freeze()
    frames = _load_real_selection_frames()
    ledger = run_selection(frames, runtime=runtime, freeze=freeze)
    print(_strict_json(ledger))
    _progress(f"selection disposition: {ledger['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
