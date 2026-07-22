"""Support-aware first-marriage sibling for the M6 candidate-2 program.

This module deliberately does not modify or inherit runtime state from the
frozen candidate-16 implementation.  It reuses only that implementation's
public restricted-cubic-spline basis and knot tuple.  The fitted law follows
``docs/design/m6_candidate2_program.md`` section 4: the global age curve is
clipped to sex-specific positive-weight support, the shared cohort deviation
is clipped to pooled-cohort positive-weight support, and unseen cohorts use
the nearest fitted decade.

Fit failures are data, not exceptions, for the train-only selector.  A solver
attempt therefore returns a model carrying a complete :class:`FirstMarriageFitAudit`
even when the attempt is ineligible.  The separate validation function turns
an ineligible selected/full fit into the designed preflight abort.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, minimize
from scipy.special import expit

from populace_dynamics.models.family_transitions.components.first_marriage import (
    SPLINE_KNOTS,
    ncs_basis,
)

__all__ = [
    "C_GRID",
    "GRADIENT_TOL",
    "MAX_INFORMATION_YEAR",
    "MAX_ITER",
    "PSEUDO_BOUNDARIES",
    "SOLVER_FTOL",
    "SOLVER_TOL",
    "FirstMarriageFitAudit",
    "FirstMarriagePreflightAbort",
    "FirstMarriageTransportDiagnostics",
    "SupportAwareFirstMarriageModel",
    "fit_support_aware_first_marriage",
    "recompute_support_aware_first_marriage_checksums",
    "validate_support_aware_first_marriage_fit",
]

MAX_INFORMATION_YEAR = 2014
C_GRID: tuple[float, ...] = (
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)
PSEUDO_BOUNDARIES: tuple[int, ...] = (2006, 2008, 2010)
MAX_ITER = 10_000
SOLVER_TOL = 1e-8
GRADIENT_TOL = 1e-6
SOLVER_FTOL = float(np.finfo(np.float64).eps)

_REQUIRED_COLUMNS = (
    "person_id",
    "year",
    "age",
    "sex",
    "weight",
    "birth_decade",
)
_CHECKSUM_KEYS = (
    "canonical_rows_sha256",
    "design_matrix_sha256",
    "support_sha256",
    "standardization_sha256",
    "normalized_weight_sha256",
    "coefficient_sha256",
    "selected_c_sha256",
)


class FirstMarriagePreflightAbort(RuntimeError):
    """The selected support-aware fit is not eligible for projection."""


def _read_only_array(values: Any, *, dtype: Any) -> np.ndarray:
    """Return an owning immutable array for fitted state and diagnostics."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    """Recursively make JSON-like audit payloads mutation-resistant."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class FirstMarriageTransportDiagnostics:
    """Canonical support-extension coordinates for ordered target rows."""

    age: np.ndarray
    is_male: np.ndarray
    target_birth_decade: np.ndarray
    mapped_birth_decade: np.ndarray
    mapped_cohort_index: np.ndarray
    global_age_evaluated: np.ndarray
    cohort_age_evaluated: np.ndarray
    global_boundary_evaluated: np.ndarray
    cohort_boundary_evaluated: np.ndarray

    def __post_init__(self) -> None:
        fields = {
            "age": (self.age, np.float64),
            "is_male": (self.is_male, np.bool_),
            "target_birth_decade": (self.target_birth_decade, np.int64),
            "mapped_birth_decade": (self.mapped_birth_decade, np.int64),
            "mapped_cohort_index": (self.mapped_cohort_index, np.int64),
            "global_age_evaluated": (
                self.global_age_evaluated,
                np.float64,
            ),
            "cohort_age_evaluated": (
                self.cohort_age_evaluated,
                np.float64,
            ),
            "global_boundary_evaluated": (
                self.global_boundary_evaluated,
                np.bool_,
            ),
            "cohort_boundary_evaluated": (
                self.cohort_boundary_evaluated,
                np.bool_,
            ),
        }
        for name, (values, dtype) in fields.items():
            object.__setattr__(
                self,
                name,
                _read_only_array(values, dtype=dtype),
            )

    def canonical_records(self) -> list[dict[str, Any]]:
        """Return strict-JSON-ready records in the original target order."""
        return [
            {
                "sex": "male" if bool(is_male) else "female",
                "target_birth_decade": int(target_decade),
                "mapped_birth_decade": int(mapped_decade),
                "age": float(age),
                "global_age_evaluated": float(global_age),
                "cohort_age_evaluated": float(cohort_age),
                "global_boundary_evaluated": bool(global_boundary),
                "cohort_boundary_evaluated": bool(cohort_boundary),
            }
            for (
                age,
                is_male,
                target_decade,
                mapped_decade,
                global_age,
                cohort_age,
                global_boundary,
                cohort_boundary,
            ) in zip(
                self.age,
                self.is_male,
                self.target_birth_decade,
                self.mapped_birth_decade,
                self.global_age_evaluated,
                self.cohort_age_evaluated,
                self.global_boundary_evaluated,
                self.cohort_boundary_evaluated,
                strict=True,
            )
        ]


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _combined_array_sha256(
    arrays: tuple[tuple[str, np.ndarray, str], ...],
) -> str:
    digest = hashlib.sha256()
    for label, values, dtype in arrays:
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(_array_sha256(values, dtype=dtype).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
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
    return value


@dataclass(frozen=True)
class FirstMarriageFitAudit:
    """Complete convergence, support, and identity record for one fit."""

    c: float
    n_input_rows: int
    n_train_rows: int
    n_train_events: int
    n_features: int
    solver_success: bool
    solver_status: int | None
    solver_message: str
    n_iter: int
    max_iter: int
    max_iter_reached: bool
    solver_gtol: float
    solver_ftol: float
    warning_count: int
    warning_messages: tuple[str, ...]
    convergence_warning_count: int
    convergence_warning_messages: tuple[str, ...]
    objective_value: float
    gradient_inf_norm: float
    intercept: float
    coefficients: tuple[float, ...]
    design_finite: bool
    coefficients_finite: bool
    linear_predictor_finite: bool
    linear_predictor_min: float
    linear_predictor_max: float
    probabilities_finite: bool
    probabilities_strict_unit_interval: bool
    probability_min: float
    probability_max: float
    eligible: bool
    eligibility_failures: tuple[str, ...]
    checksums: Mapping[str, str]
    support: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checksums", _deep_freeze(self.checksums))
        object.__setattr__(self, "support", _deep_freeze(self.support))

    def __reduce__(self):
        return (
            type(self),
            (
                self.c,
                self.n_input_rows,
                self.n_train_rows,
                self.n_train_events,
                self.n_features,
                self.solver_success,
                self.solver_status,
                self.solver_message,
                self.n_iter,
                self.max_iter,
                self.max_iter_reached,
                self.solver_gtol,
                self.solver_ftol,
                self.warning_count,
                self.warning_messages,
                self.convergence_warning_count,
                self.convergence_warning_messages,
                self.objective_value,
                self.gradient_inf_norm,
                self.intercept,
                self.coefficients,
                self.design_finite,
                self.coefficients_finite,
                self.linear_predictor_finite,
                self.linear_predictor_min,
                self.linear_predictor_max,
                self.probabilities_finite,
                self.probabilities_strict_unit_interval,
                self.probability_min,
                self.probability_max,
                self.eligible,
                self.eligibility_failures,
                dict(self.checksums),
                _plain(dict(self.support)),
            ),
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-ready selector/preflight ledger record."""
        payload = {
            "c": self.c,
            "n_input_rows": self.n_input_rows,
            "n_train_rows": self.n_train_rows,
            "n_train_events": self.n_train_events,
            "n_features": self.n_features,
            "solver_success": self.solver_success,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "n_iter": self.n_iter,
            "max_iter": self.max_iter,
            "max_iter_reached": self.max_iter_reached,
            "solver_gtol": self.solver_gtol,
            "solver_ftol": self.solver_ftol,
            "warning_count": self.warning_count,
            "warning_messages": list(self.warning_messages),
            "convergence_warning_count": self.convergence_warning_count,
            "convergence_warning_messages": list(
                self.convergence_warning_messages
            ),
            "objective_value": self.objective_value,
            "gradient_inf_norm": self.gradient_inf_norm,
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "design_finite": self.design_finite,
            "coefficients_finite": self.coefficients_finite,
            "linear_predictor_finite": self.linear_predictor_finite,
            "linear_predictor_min": self.linear_predictor_min,
            "linear_predictor_max": self.linear_predictor_max,
            "probabilities_finite": self.probabilities_finite,
            "probabilities_strict_unit_interval": (
                self.probabilities_strict_unit_interval
            ),
            "probability_min": self.probability_min,
            "probability_max": self.probability_max,
            "eligible": self.eligible,
            "eligibility_failures": list(self.eligibility_failures),
            "checksums": dict(self.checksums),
            "support": _plain(dict(self.support)),
        }
        return _plain(payload)


@dataclass(frozen=True)
class _FirstMarriageFitReplay:
    """Compact canonical fit rows retained for independent preflight replay."""

    person_id: np.ndarray
    year: np.ndarray
    age: np.ndarray
    is_male: np.ndarray
    weight: np.ndarray
    birth_decade: np.ndarray
    outcomes: np.ndarray
    n_input_rows: int
    c: float
    max_iter: int
    solver_gtol: float
    solver_ftol: float

    def __post_init__(self) -> None:
        fields = {
            "person_id": (self.person_id, np.int64),
            "year": (self.year, np.int64),
            "age": (self.age, np.float64),
            "is_male": (self.is_male, np.bool_),
            "weight": (self.weight, np.float64),
            "birth_decade": (self.birth_decade, np.int64),
            "outcomes": (self.outcomes, np.float64),
        }
        lengths = {len(values) for values, _dtype in fields.values()}
        if len(lengths) != 1:
            raise ValueError("first-marriage replay arrays differ in length")
        for name, (values, dtype) in fields.items():
            object.__setattr__(
                self,
                name,
                _read_only_array(values, dtype=dtype),
            )


def _nearest_levels(
    decade: np.ndarray, cohort_levels: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Map to a fitted cohort; exact ties resolve to the older decade."""
    levels = np.asarray(cohort_levels, dtype=np.int64)
    target = np.asarray(decade, dtype=np.int64)
    right = np.searchsorted(levels, target, side="left")
    left_index = np.clip(right - 1, 0, len(levels) - 1)
    right_index = np.clip(right, 0, len(levels) - 1)
    left_distance = np.abs(target - levels[left_index])
    right_distance = np.abs(levels[right_index] - target)
    use_right = right_distance < left_distance
    indices = np.where(use_right, right_index, left_index)
    return levels[indices], indices.astype(np.int64, copy=False)


@dataclass(frozen=True, eq=False)
class SupportAwareFirstMarriageModel:
    """Regularized first-marriage hazard with boundary-flat transport."""

    intercept: float
    coefficients: np.ndarray
    cohort_levels: tuple[int, ...]
    knots: tuple[float, ...]
    col_mean: np.ndarray
    col_sd: np.ndarray
    sex_age_min: np.ndarray
    sex_age_max: np.ndarray
    cohort_age_min: np.ndarray
    cohort_age_max: np.ndarray
    fit_audit: FirstMarriageFitAudit | None
    fit_replay: _FirstMarriageFitReplay | None = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        arrays = {
            "coefficients": (self.coefficients, np.float64),
            "col_mean": (self.col_mean, np.float64),
            "col_sd": (self.col_sd, np.float64),
            "sex_age_min": (self.sex_age_min, np.float64),
            "sex_age_max": (self.sex_age_max, np.float64),
            "cohort_age_min": (self.cohort_age_min, np.float64),
            "cohort_age_max": (self.cohort_age_max, np.float64),
        }
        for name, (values, dtype) in arrays.items():
            object.__setattr__(
                self,
                name,
                _read_only_array(values, dtype=dtype),
            )

    def transport_diagnostics(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> FirstMarriageTransportDiagnostics:
        """Resolve the registered boundary-flat transport law for targets."""
        age_values = np.asarray(age, dtype=np.float64)
        male = np.asarray(is_male, dtype=bool)
        decade_values = np.asarray(decade, dtype=np.int64)
        if not (age_values.shape == male.shape == decade_values.shape):
            raise ValueError("age, is_male, and decade must have equal shapes")
        if age_values.ndim != 1:
            raise ValueError("first-marriage prediction arrays must be 1-D")
        if not np.isfinite(age_values).all():
            raise ValueError("first-marriage prediction age must be finite")

        sex_index = male.astype(np.int64)
        mapped_decade, cohort_index = _nearest_levels(
            decade_values, self.cohort_levels
        )
        global_age = np.clip(
            age_values,
            self.sex_age_min[sex_index],
            self.sex_age_max[sex_index],
        )
        cohort_age = np.clip(
            age_values,
            self.cohort_age_min[cohort_index],
            self.cohort_age_max[cohort_index],
        )
        return FirstMarriageTransportDiagnostics(
            age=age_values,
            is_male=male,
            target_birth_decade=decade_values,
            mapped_birth_decade=mapped_decade,
            mapped_cohort_index=cohort_index,
            global_age_evaluated=global_age,
            cohort_age_evaluated=cohort_age,
            global_boundary_evaluated=global_age != age_values,
            cohort_boundary_evaluated=cohort_age != age_values,
        )

    def _raw_design(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> np.ndarray:
        diagnostics = self.transport_diagnostics(age, is_male, decade)
        if diagnostics.age.size == 0:
            return np.empty((0, len(self.col_mean)), dtype=np.float64)
        male = diagnostics.is_male
        mapped_decade = diagnostics.mapped_birth_decade
        global_age = diagnostics.global_age_evaluated
        cohort_age = diagnostics.cohort_age_evaluated
        global_spline = ncs_basis(global_age, self.knots)
        cohort_spline = ncs_basis(cohort_age, self.knots)
        male_column = male.astype(np.float64).reshape(-1, 1)
        parts = [
            global_spline,
            male_column,
            global_spline * male_column,
        ]
        if len(self.cohort_levels) > 1:
            dummies = np.column_stack(
                [mapped_decade == level for level in self.cohort_levels[1:]]
            ).astype(np.float64)
            parts.append(dummies)
            for column in range(cohort_spline.shape[1]):
                parts.append(cohort_spline[:, [column]] * dummies)
        return np.column_stack(parts)

    def _design(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> np.ndarray:
        raw = self._raw_design(age, is_male, decade)
        return (raw - self.col_mean) / self.col_sd

    def raw_design(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> np.ndarray:
        """Return the frozen pre-standardization support-aware design."""
        return self._raw_design(age, is_male, decade)

    def design(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> np.ndarray:
        """Return the standardized support-aware design."""
        return self._design(age, is_male, decade)

    def audit_dict(self) -> dict[str, Any]:
        """Return the complete strict-JSON-ready fit audit."""
        if self.fit_audit is None:
            raise ValueError(
                "support-aware first-marriage fit audit is absent"
            )
        return self.fit_audit.canonical_dict()

    def linear_predictor(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> np.ndarray:
        """Return the fitted logit on the support-aware design."""
        if np.asarray(age).size == 0:
            return np.zeros(0, dtype=np.float64)
        return self.intercept + self._design(age, is_male, decade).dot(
            self.coefficients
        )

    def predict(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        decade: np.ndarray,
    ) -> np.ndarray:
        """Return probabilities without masking exact-zero/one failures."""
        if np.asarray(age).size == 0:
            return np.zeros(0, dtype=np.float64)
        return expit(self.linear_predictor(age, is_male, decade))


def _support_record(
    frame: pd.DataFrame, outcomes: np.ndarray
) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_rows": 0,
            "row_weight": 0.0,
            "n_events": 0,
            "event_weight": 0.0,
            "age_min": None,
            "age_max": None,
        }
    positions = frame["_fit_position"].to_numpy(dtype=np.int64)
    event = outcomes[positions]
    weight = frame["weight"].to_numpy(dtype=np.float64)
    age = frame["age"].to_numpy(dtype=np.float64)
    return {
        "n_rows": int(len(frame)),
        "row_weight": float(weight.sum()),
        "n_events": int(event.sum()),
        "event_weight": float(np.dot(weight, event)),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
    }


def _support_ledger(
    frame: pd.DataFrame,
    outcomes: np.ndarray,
    cohort_levels: tuple[int, ...],
) -> dict[str, Any]:
    sexes = ("female", "male")
    return {
        "positive_weight_only": True,
        "sex": {
            sex: _support_record(frame[frame["sex"] == sex], outcomes)
            for sex in sexes
        },
        "pooled_cohort": {
            str(level): _support_record(
                frame[frame["birth_decade"] == level], outcomes
            )
            for level in cohort_levels
        },
        "sex_by_cohort": {
            f"{sex}|{level}": _support_record(
                frame[
                    (frame["sex"] == sex) & (frame["birth_decade"] == level)
                ],
                outcomes,
            )
            for sex in sexes
            for level in cohort_levels
        },
    }


def _prepare_fit_rows(
    train_py: pd.DataFrame,
    event_years: set[tuple[int, int]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    missing = sorted(set(_REQUIRED_COLUMNS) - set(train_py.columns))
    if missing:
        raise ValueError(f"first-marriage fit frame missing columns {missing}")
    frame = train_py.loc[:, list(_REQUIRED_COLUMNS)].copy()
    numeric_columns = ("person_id", "year", "age", "weight", "birth_decade")
    for column in numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        ):
            raise ValueError(f"first-marriage fit {column} must be finite")
        frame[column] = numeric
    if (frame["weight"] < 0).any():
        raise ValueError("first-marriage fit weights must be nonnegative")
    frame = frame[frame["weight"] > 0].copy()
    if frame.empty:
        raise ValueError("first-marriage fit has no positive-weight rows")
    if not frame["sex"].isin(("female", "male")).all():
        raise ValueError("first-marriage fit sex must be female or male")
    if set(frame["sex"].unique()) != {"female", "male"}:
        raise ValueError("first-marriage fit requires both sex support rows")
    frame["person_id"] = frame["person_id"].astype(np.int64)
    frame["year"] = frame["year"].astype(np.int64)
    frame["birth_decade"] = frame["birth_decade"].astype(np.int64)
    frame = frame.sort_values(
        ["person_id", "year"], kind="stable"
    ).reset_index(drop=True)
    if frame.duplicated(["person_id", "year"]).any():
        raise ValueError("first-marriage fit has duplicate person-year rows")
    frame["_fit_position"] = np.arange(len(frame), dtype=np.int64)
    outcomes = np.fromiter(
        (
            (int(person_id), int(year)) in event_years
            for person_id, year in zip(
                frame["person_id"].to_numpy(),
                frame["year"].to_numpy(),
                strict=True,
            )
        ),
        dtype=np.float64,
        count=len(frame),
    )
    if np.unique(outcomes).size != 2:
        raise ValueError(
            "first-marriage fit requires event and non-event rows"
        )
    weight = frame["weight"].to_numpy(dtype=np.float64)
    normalized_weight = len(frame) * weight / weight.sum()
    return frame, outcomes, normalized_weight


def _objective_and_gradient(
    parameters: np.ndarray,
    design: np.ndarray,
    outcomes: np.ndarray,
    normalized_weight: np.ndarray,
    c: float,
) -> tuple[float, np.ndarray]:
    intercept = parameters[0]
    coefficients = parameters[1:]
    linear = intercept + design.dot(coefficients)
    weighted_loss = normalized_weight * (
        np.logaddexp(0.0, linear) - outcomes * linear
    )
    n_rows = len(outcomes)
    objective = float(
        weighted_loss.sum() / n_rows
        + coefficients.dot(coefficients) / (2.0 * c * n_rows)
    )
    residual = normalized_weight * (expit(linear) - outcomes)
    gradient = np.empty_like(parameters)
    gradient[0] = residual.sum() / n_rows
    gradient[1:] = design.T.dot(residual) / n_rows + coefficients / (
        c * n_rows
    )
    return objective, gradient


def _is_convergence_warning(record: warnings.WarningMessage) -> bool:
    """Classify only optimizer/convergence warnings as fit-ineligible."""
    if issubclass(record.category, OptimizeWarning):
        return True
    if record.category.__name__ == "ConvergenceWarning":
        return True
    message = str(record.message).lower()
    return any(
        marker in message
        for marker in (
            "failed to converge",
            "convergence failed",
            "did not converge",
            "iteration limit",
            "maximum number of iterations",
            "abnormal termination",
        )
    )


def fit_support_aware_first_marriage(
    train_py: pd.DataFrame,
    event_years: set[tuple[int, int]],
    *,
    c: float,
    max_iter: int = MAX_ITER,
    tol: float = SOLVER_TOL,
) -> SupportAwareFirstMarriageModel:
    """Fit one deterministic support-aware L2-logit attempt.

    The explicit objective is the normalized-F6-weighted mean Bernoulli loss
    plus ``||theta||^2 / (2 C n)``.  The intercept is not penalized.  Rows are
    sorted canonically by ``(person_id, year)`` before support, design,
    standardization, checksums, or optimization are computed.
    """
    if not math.isfinite(c) or c <= 0:
        raise ValueError("first-marriage C must be finite and positive")
    if max_iter <= 0:
        raise ValueError("first-marriage max_iter must be positive")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("first-marriage tol must be finite and positive")

    frame, outcomes, normalized_weight = _prepare_fit_rows(
        train_py, event_years
    )
    age = frame["age"].to_numpy(dtype=np.float64)
    is_male = frame["sex"].to_numpy() == "male"
    decade = frame["birth_decade"].to_numpy(dtype=np.int64)
    cohort_levels = tuple(sorted(int(value) for value in np.unique(decade)))
    support = _support_ledger(frame, outcomes, cohort_levels)
    sex_age_min = np.asarray(
        [support["sex"][sex]["age_min"] for sex in ("female", "male")],
        dtype=np.float64,
    )
    sex_age_max = np.asarray(
        [support["sex"][sex]["age_max"] for sex in ("female", "male")],
        dtype=np.float64,
    )
    cohort_age_min = np.asarray(
        [
            support["pooled_cohort"][str(level)]["age_min"]
            for level in cohort_levels
        ],
        dtype=np.float64,
    )
    cohort_age_max = np.asarray(
        [
            support["pooled_cohort"][str(level)]["age_max"]
            for level in cohort_levels
        ],
        dtype=np.float64,
    )

    placeholder = SupportAwareFirstMarriageModel(
        intercept=0.0,
        coefficients=np.zeros(1, dtype=np.float64),
        cohort_levels=cohort_levels,
        knots=SPLINE_KNOTS,
        col_mean=np.zeros(1, dtype=np.float64),
        col_sd=np.ones(1, dtype=np.float64),
        sex_age_min=sex_age_min,
        sex_age_max=sex_age_max,
        cohort_age_min=cohort_age_min,
        cohort_age_max=cohort_age_max,
        fit_audit=None,
        fit_replay=None,
    )
    raw = placeholder._raw_design(age, is_male, decade)
    design_finite = bool(np.isfinite(raw).all())
    if not design_finite:
        raise ValueError("first-marriage raw design is non-finite")
    col_mean = raw.mean(axis=0)
    col_sd = raw.std(axis=0, ddof=0)
    col_sd = np.where(col_sd > 0, col_sd, 1.0)
    design = (raw - col_mean) / col_sd
    design_finite = bool(np.isfinite(design).all())
    if not design_finite:
        raise ValueError("first-marriage standardized design is non-finite")

    initial = np.zeros(design.shape[1] + 1, dtype=np.float64)
    caught: list[warnings.WarningMessage]
    optimizer_error = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = minimize(
                _objective_and_gradient,
                initial,
                args=(design, outcomes, normalized_weight, float(c)),
                method="L-BFGS-B",
                jac=True,
                # The registered ``tol`` is the projected-gradient tolerance.
                # A looser function-reduction exit can otherwise report
                # success above the independently certified gradient bound.
                options={
                    "maxiter": int(max_iter),
                    "gtol": float(tol),
                    "ftol": SOLVER_FTOL,
                },
            )
            parameters = np.asarray(result.x, dtype=np.float64)
            solver_success = bool(result.success)
            solver_status = int(result.status)
            solver_message = str(result.message)
            n_iter = int(result.nit)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            parameters = initial
            solver_success = False
            solver_status = None
            solver_message = f"{type(error).__name__}: {error}"
            optimizer_error = solver_message
            n_iter = 0

    warning_messages = tuple(
        f"{record.category.__name__}: {record.message}" for record in caught
    )
    convergence_warning_messages = tuple(
        f"{record.category.__name__}: {record.message}"
        for record in caught
        if _is_convergence_warning(record)
    )
    objective_value, independent_gradient = _objective_and_gradient(
        parameters,
        design,
        outcomes,
        normalized_weight,
        float(c),
    )
    gradient_inf_norm = float(np.max(np.abs(independent_gradient)))
    coefficients_finite = bool(np.isfinite(parameters).all())
    linear = parameters[0] + design.dot(parameters[1:])
    probability = expit(linear)
    linear_finite = bool(np.isfinite(linear).all())
    probability_finite = bool(np.isfinite(probability).all())
    probability_strict = bool(
        probability_finite
        and np.all(probability > 0.0)
        and np.all(probability < 1.0)
    )
    max_iter_reached = n_iter >= max_iter
    checksums = {
        "canonical_rows_sha256": _combined_array_sha256(
            (
                (
                    "person_id",
                    frame["person_id"].to_numpy(dtype=np.int64),
                    "<i8",
                ),
                ("year", frame["year"].to_numpy(dtype=np.int64), "<i8"),
                ("outcome", outcomes, "<f8"),
            )
        ),
        "design_matrix_sha256": _array_sha256(design),
        "support_sha256": _json_sha256(support),
        "standardization_sha256": _combined_array_sha256(
            (
                ("column_mean", col_mean, "<f8"),
                ("column_sd", col_sd, "<f8"),
            )
        ),
        "normalized_weight_sha256": _array_sha256(normalized_weight),
        "coefficient_sha256": _array_sha256(parameters),
        "selected_c_sha256": _json_sha256({"selected_c": float(c)}),
    }
    eligibility_failures: list[str] = []
    if not solver_success:
        eligibility_failures.append("solver_unsuccessful")
    if optimizer_error:
        eligibility_failures.append("optimizer_exception")
    if convergence_warning_messages:
        eligibility_failures.append("convergence_warning_emitted")
    if max_iter_reached:
        eligibility_failures.append("max_iter_reached")
    if not math.isfinite(gradient_inf_norm):
        eligibility_failures.append("gradient_nonfinite")
    elif gradient_inf_norm > GRADIENT_TOL:
        eligibility_failures.append("gradient_above_threshold")
    if not design_finite:
        eligibility_failures.append("design_nonfinite")
    if not coefficients_finite:
        eligibility_failures.append("coefficients_nonfinite")
    if not linear_finite:
        eligibility_failures.append("linear_predictor_nonfinite")
    if not probability_finite:
        eligibility_failures.append("probabilities_nonfinite")
    elif not probability_strict:
        eligibility_failures.append("probabilities_not_strict_unit_interval")
    eligible = not eligibility_failures
    audit = FirstMarriageFitAudit(
        c=float(c),
        n_input_rows=int(len(train_py)),
        n_train_rows=int(len(frame)),
        n_train_events=int(outcomes.sum()),
        n_features=int(design.shape[1]),
        solver_success=solver_success,
        solver_status=solver_status,
        solver_message=solver_message,
        n_iter=n_iter,
        max_iter=int(max_iter),
        max_iter_reached=max_iter_reached,
        solver_gtol=float(tol),
        solver_ftol=SOLVER_FTOL,
        warning_count=len(warning_messages),
        warning_messages=warning_messages,
        convergence_warning_count=len(convergence_warning_messages),
        convergence_warning_messages=convergence_warning_messages,
        objective_value=objective_value,
        gradient_inf_norm=gradient_inf_norm,
        intercept=float(parameters[0]),
        coefficients=tuple(float(value) for value in parameters[1:]),
        design_finite=design_finite,
        coefficients_finite=coefficients_finite,
        linear_predictor_finite=linear_finite,
        linear_predictor_min=float(np.min(linear)),
        linear_predictor_max=float(np.max(linear)),
        probabilities_finite=probability_finite,
        probabilities_strict_unit_interval=probability_strict,
        probability_min=float(np.min(probability)),
        probability_max=float(np.max(probability)),
        eligible=eligible,
        eligibility_failures=tuple(eligibility_failures),
        checksums=checksums,
        support=support,
    )
    return SupportAwareFirstMarriageModel(
        intercept=float(parameters[0]),
        coefficients=parameters[1:].copy(),
        cohort_levels=cohort_levels,
        knots=SPLINE_KNOTS,
        col_mean=col_mean,
        col_sd=col_sd,
        sex_age_min=sex_age_min,
        sex_age_max=sex_age_max,
        cohort_age_min=cohort_age_min,
        cohort_age_max=cohort_age_max,
        fit_audit=audit,
        fit_replay=_FirstMarriageFitReplay(
            person_id=frame["person_id"].to_numpy(dtype=np.int64),
            year=frame["year"].to_numpy(dtype=np.int64),
            age=age,
            is_male=is_male,
            weight=frame["weight"].to_numpy(dtype=np.float64),
            birth_decade=decade,
            outcomes=outcomes,
            n_input_rows=int(len(train_py)),
            c=float(c),
            max_iter=int(max_iter),
            solver_gtol=float(tol),
            solver_ftol=SOLVER_FTOL,
        ),
    )


def _replayed_fit_state(
    model: SupportAwareFirstMarriageModel,
) -> dict[str, Any]:
    replay = model.fit_replay
    if replay is None:
        raise ValueError("compact fit replay is absent")
    frame = pd.DataFrame(
        {
            "person_id": replay.person_id,
            "year": replay.year,
            "age": replay.age,
            "sex": np.where(replay.is_male, "male", "female"),
            "weight": replay.weight,
            "birth_decade": replay.birth_decade,
            "_fit_position": np.arange(len(replay.year), dtype=np.int64),
        }
    )
    cohort_levels = tuple(
        sorted(int(value) for value in np.unique(replay.birth_decade))
    )
    support = _support_ledger(frame, replay.outcomes, cohort_levels)
    normalized_weight = (
        len(replay.weight) * replay.weight / replay.weight.sum()
    )
    design = model.design(
        replay.age,
        replay.is_male,
        replay.birth_decade,
    )
    parameters = np.concatenate(
        (
            np.asarray([model.intercept], dtype=np.float64),
            np.asarray(model.coefficients, dtype=np.float64),
        )
    )
    objective, gradient = _objective_and_gradient(
        parameters,
        design,
        replay.outcomes,
        normalized_weight,
        replay.c,
    )
    linear = parameters[0] + design.dot(parameters[1:])
    probability = expit(linear)
    checksums = {
        "canonical_rows_sha256": _combined_array_sha256(
            (
                ("person_id", replay.person_id, "<i8"),
                ("year", replay.year, "<i8"),
                ("outcome", replay.outcomes, "<f8"),
            )
        ),
        "design_matrix_sha256": _array_sha256(design),
        "support_sha256": _json_sha256(support),
        "standardization_sha256": _combined_array_sha256(
            (
                ("column_mean", model.col_mean, "<f8"),
                ("column_sd", model.col_sd, "<f8"),
            )
        ),
        "normalized_weight_sha256": _array_sha256(normalized_weight),
        "coefficient_sha256": _array_sha256(parameters),
        "selected_c_sha256": _json_sha256({"selected_c": replay.c}),
    }
    return {
        "replay": replay,
        "cohort_levels": cohort_levels,
        "support": support,
        "normalized_weight": normalized_weight,
        "design": design,
        "parameters": parameters,
        "objective": objective,
        "gradient": gradient,
        "linear": linear,
        "probability": probability,
        "checksums": checksums,
    }


def recompute_support_aware_first_marriage_checksums(
    model: SupportAwareFirstMarriageModel,
) -> Mapping[str, str]:
    """Recompute every registered checksum from live state and replay rows."""
    return MappingProxyType(dict(_replayed_fit_state(model)["checksums"]))


def _float64_equal(left: float, right: float) -> bool:
    return (
        np.asarray([left], dtype="<f8").tobytes()
        == np.asarray([right], dtype="<f8").tobytes()
    )


def validate_support_aware_first_marriage_fit(
    model: SupportAwareFirstMarriageModel,
    *,
    expected_checksums: Mapping[str, str] | None = None,
) -> None:
    """Raise unless the live selected fit and independent replay reproduce."""
    audit = model.fit_audit
    if audit is None:
        raise FirstMarriagePreflightAbort(
            "NO_REGISTERABLE_FIRST_MARRIAGE_FIT: fit audit is absent"
        )
    failures: list[str] = []
    try:
        state = _replayed_fit_state(model)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        state = None
        failures.append(f"fit replay failed: {type(error).__name__}: {error}")

    if not audit.solver_success:
        failures.append("solver termination was unsuccessful")
    if audit.max_iter_reached:
        failures.append("max_iter was reached")
    if audit.convergence_warning_count:
        failures.append("fit emitted a convergence warning")
    if not math.isfinite(audit.gradient_inf_norm):
        failures.append("gradient infinity norm is non-finite")
    elif audit.gradient_inf_norm > GRADIENT_TOL:
        failures.append("gradient infinity norm exceeds 1e-6")
    if not audit.design_finite:
        failures.append("design is non-finite")
    if not audit.coefficients_finite:
        failures.append("coefficients are non-finite")
    if not audit.linear_predictor_finite:
        failures.append("linear predictor is non-finite")
    if not audit.probabilities_strict_unit_interval:
        failures.append("probabilities are not strictly between zero and one")
    if not audit.eligible:
        failures.append("fit audit is ineligible")

    if audit.warning_count != len(audit.warning_messages):
        failures.append("warning count differs from warning messages")
    if audit.convergence_warning_count != len(
        audit.convergence_warning_messages
    ):
        failures.append(
            "convergence-warning count differs from warning messages"
        )
    if any(
        message not in audit.warning_messages
        for message in audit.convergence_warning_messages
    ):
        failures.append("convergence warnings are absent from all warnings")
    if audit.max_iter_reached != (audit.n_iter >= audit.max_iter):
        failures.append("max-iteration certificate does not reproduce")

    if state is not None:
        replay: _FirstMarriageFitReplay = state["replay"]
        design = state["design"]
        parameters = state["parameters"]
        gradient = state["gradient"]
        linear = state["linear"]
        probability = state["probability"]
        support = state["support"]
        recomputed_checksums = state["checksums"]
        gradient_inf_norm = float(np.max(np.abs(gradient)))
        design_finite = bool(np.isfinite(design).all())
        coefficients_finite = bool(np.isfinite(parameters).all())
        linear_finite = bool(np.isfinite(linear).all())
        probability_finite = bool(np.isfinite(probability).all())
        probability_strict = bool(
            probability_finite
            and np.all(probability > 0.0)
            and np.all(probability < 1.0)
        )

        if audit.n_input_rows != replay.n_input_rows:
            failures.append("input-row count does not replay")
        if audit.n_train_rows != len(replay.year):
            failures.append("training-row count does not replay")
        if audit.n_train_events != int(replay.outcomes.sum()):
            failures.append("training-event count does not replay")
        if audit.n_features != design.shape[1]:
            failures.append("feature count does not replay")
        if audit.c != replay.c:
            failures.append("selected C differs from replay")
        if audit.max_iter != replay.max_iter:
            failures.append("max_iter differs from replay")
        if audit.solver_gtol != replay.solver_gtol:
            failures.append("solver gtol differs from replay")
        if audit.solver_ftol != replay.solver_ftol:
            failures.append("solver ftol differs from replay")
        if replay.solver_ftol != SOLVER_FTOL:
            failures.append("solver ftol differs from float64 epsilon")
        if model.cohort_levels != state["cohort_levels"]:
            failures.append("cohort levels differ from fit rows")
        if tuple(model.knots) != tuple(SPLINE_KNOTS):
            failures.append("spline knots differ from registered knots")

        expected_sex_min = np.asarray(
            [support["sex"][sex]["age_min"] for sex in ("female", "male")],
            dtype=np.float64,
        )
        expected_sex_max = np.asarray(
            [support["sex"][sex]["age_max"] for sex in ("female", "male")],
            dtype=np.float64,
        )
        expected_cohort_min = np.asarray(
            [
                support["pooled_cohort"][str(level)]["age_min"]
                for level in state["cohort_levels"]
            ],
            dtype=np.float64,
        )
        expected_cohort_max = np.asarray(
            [
                support["pooled_cohort"][str(level)]["age_max"]
                for level in state["cohort_levels"]
            ],
            dtype=np.float64,
        )
        for label, observed, expected in (
            ("sex support minima", model.sex_age_min, expected_sex_min),
            ("sex support maxima", model.sex_age_max, expected_sex_max),
            (
                "cohort support minima",
                model.cohort_age_min,
                expected_cohort_min,
            ),
            (
                "cohort support maxima",
                model.cohort_age_max,
                expected_cohort_max,
            ),
        ):
            if not np.array_equal(observed, expected):
                failures.append(f"{label} differ from fit rows")

        if not _float64_equal(audit.intercept, model.intercept):
            failures.append("audit intercept differs from fitted state")
        if not np.array_equal(
            np.asarray(audit.coefficients, dtype=np.float64),
            model.coefficients,
        ):
            failures.append("audit coefficients differ from fitted state")
        if not _float64_equal(audit.objective_value, state["objective"]):
            failures.append("objective does not replay")
        if not _float64_equal(audit.gradient_inf_norm, gradient_inf_norm):
            failures.append("gradient infinity norm does not replay")
        if audit.design_finite != design_finite:
            failures.append("design-finite certificate does not replay")
        if audit.coefficients_finite != coefficients_finite:
            failures.append("coefficient-finite certificate does not replay")
        if audit.linear_predictor_finite != linear_finite:
            failures.append("linear-finite certificate does not replay")
        if audit.probabilities_finite != probability_finite:
            failures.append("probability-finite certificate does not replay")
        if audit.probabilities_strict_unit_interval != probability_strict:
            failures.append("strict-probability certificate does not replay")
        for label, observed, expected in (
            (
                "linear minimum",
                audit.linear_predictor_min,
                float(linear.min()),
            ),
            (
                "linear maximum",
                audit.linear_predictor_max,
                float(linear.max()),
            ),
            (
                "probability minimum",
                audit.probability_min,
                float(probability.min()),
            ),
            (
                "probability maximum",
                audit.probability_max,
                float(probability.max()),
            ),
        ):
            if not _float64_equal(observed, expected):
                failures.append(f"{label} does not replay")
        if _plain(audit.support) != _plain(support):
            failures.append("support ledger does not replay")

        recorded_keys = set(audit.checksums)
        if recorded_keys != set(_CHECKSUM_KEYS):
            failures.append(
                "fit-audit checksum keys differ: "
                f"missing={sorted(set(_CHECKSUM_KEYS) - recorded_keys)}, "
                f"extra={sorted(recorded_keys - set(_CHECKSUM_KEYS))}"
            )
        for key in _CHECKSUM_KEYS:
            if audit.checksums.get(key) != recomputed_checksums[key]:
                failures.append(f"fit-audit {key} does not replay")

        expected_eligibility_failures: list[str] = []
        if not audit.solver_success:
            expected_eligibility_failures.append("solver_unsuccessful")
        if audit.solver_status is None and not audit.solver_success:
            expected_eligibility_failures.append("optimizer_exception")
        if audit.convergence_warning_messages:
            expected_eligibility_failures.append("convergence_warning_emitted")
        if audit.n_iter >= audit.max_iter:
            expected_eligibility_failures.append("max_iter_reached")
        if not math.isfinite(gradient_inf_norm):
            expected_eligibility_failures.append("gradient_nonfinite")
        elif gradient_inf_norm > GRADIENT_TOL:
            expected_eligibility_failures.append("gradient_above_threshold")
        if not design_finite:
            expected_eligibility_failures.append("design_nonfinite")
        if not coefficients_finite:
            expected_eligibility_failures.append("coefficients_nonfinite")
        if not linear_finite:
            expected_eligibility_failures.append("linear_predictor_nonfinite")
        if not probability_finite:
            expected_eligibility_failures.append("probabilities_nonfinite")
        elif not probability_strict:
            expected_eligibility_failures.append(
                "probabilities_not_strict_unit_interval"
            )
        if tuple(expected_eligibility_failures) != audit.eligibility_failures:
            failures.append("eligibility failures do not replay")
        if audit.eligible != (not expected_eligibility_failures):
            failures.append("fit eligibility does not replay")

        if expected_checksums is not None:
            missing = sorted(set(_CHECKSUM_KEYS) - set(expected_checksums))
            extra = sorted(set(expected_checksums) - set(_CHECKSUM_KEYS))
            if missing or extra:
                failures.append(
                    "registered checksum keys differ: "
                    f"missing={missing}, extra={extra}"
                )
            for key in _CHECKSUM_KEYS:
                if key in expected_checksums and (
                    expected_checksums[key] != recomputed_checksums[key]
                ):
                    failures.append(f"registered {key} does not reproduce")
    elif expected_checksums is not None:
        failures.append("registered checksums cannot be replayed")

    if failures:
        detail = "; ".join(dict.fromkeys(failures))
        raise FirstMarriagePreflightAbort(
            f"NO_REGISTERABLE_FIRST_MARRIAGE_FIT: {detail}"
        )
