"""Support and weight contracts for M6 evaluation and forward projection.

The temporal-holdout path is deliberately a different object from an open
forward projection.  A gated run may condition on realized PSID wave presence,
but it must apply that condition symmetrically to the projected, realized, and
floor inputs.  A forward run cannot observe future presence and therefore
rejects it instead of silently inheriting the gated support.

The gated weight is likewise explicit: callers supply the realized boundary
snapshot, and that person-constant weight replaces every later wave weight on
all three sides.  This module never guesses which survey wave represents a
non-interview boundary year.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

__all__ = [
    "EvaluationMode",
    "PresenceBasis",
    "PreparedSupport",
    "StartWaveWeightSnapshot",
    "WidowhoodMode",
    "prepare_evaluation_support",
]


class EvaluationMode(str, Enum):
    """The two structurally distinct M6 execution modes."""

    GATED_REALIZED = "gated_realized"
    FORWARD = "forward"


class PresenceBasis(str, Enum):
    """How realized presence defines the eligible gated support."""

    START_OF_INTERVAL = "start_of_interval"
    EXACT_WAVE = "exact_wave"


class WidowhoodMode(str, Enum):
    """Widowhood treatment disclosed by each execution mode."""

    EXOGENOUS_CERTIFIED_HAZARD = "exogenous_certified_hazard"
    ENDOGENOUS_RECONCILIATION_REQUIRED = "endogenous_reconciliation_required"


@dataclass(frozen=True)
class StartWaveWeightSnapshot:
    """Caller-supplied realized-boundary weights for the gated closed panel.

    ``frame`` must contain exactly one positive finite weight per person.  It
    intentionally has no period-selection behavior: resolving the survey wave
    that represents ``boundary_period`` is an upstream data decision, not a
    fallback this contract may make silently.
    """

    boundary_period: int
    person_ids: tuple[object, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        boundary_period: int,
        person_column: str = "person_id",
        weight_column: str = "weight",
    ) -> StartWaveWeightSnapshot:
        """Validate and freeze one caller-resolved boundary snapshot."""
        _require_columns(
            frame, (person_column, weight_column), "weight snapshot"
        )
        snapshot = frame[[person_column, weight_column]].copy()
        if snapshot[person_column].isna().any():
            raise ValueError("weight snapshot contains a missing person id")
        duplicated = snapshot[person_column].duplicated(keep=False)
        if duplicated.any():
            duplicate_ids = snapshot.loc[duplicated, person_column].tolist()
            raise ValueError(
                "weight snapshot must have one row per person; duplicate "
                f"ids: {duplicate_ids}"
            )
        try:
            weights = snapshot[weight_column].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "weight snapshot weights must be numeric"
            ) from error
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(
                "weight snapshot weights must be positive and finite"
            )
        return cls(
            boundary_period=int(boundary_period),
            person_ids=tuple(snapshot[person_column].tolist()),
            weights=tuple(float(weight) for weight in weights),
        )

    def apply(
        self,
        frame: pd.DataFrame,
        *,
        person_column: str = "person_id",
        weight_column: str = "weight",
    ) -> pd.DataFrame:
        """Return ``frame`` with the boundary weight fixed on every row."""
        _require_columns(frame, (person_column,), "evaluation frame")
        weight_by_person = pd.Series(
            self.weights,
            index=pd.Index(self.person_ids),
            dtype=np.float64,
        )
        mapped = frame[person_column].map(weight_by_person)
        missing = mapped.isna()
        if missing.any():
            missing_ids = (
                frame.loc[missing, person_column].drop_duplicates().tolist()
            )
            raise ValueError(
                "boundary weight snapshot does not cover evaluation person(s): "
                f"{missing_ids}"
            )
        out = frame.copy()
        out[weight_column] = mapped.to_numpy(dtype=np.float64)
        return out


@dataclass(frozen=True)
class PreparedSupport:
    """Mode-explicit support returned to the evaluator or forward engine."""

    mode: EvaluationMode
    projection: pd.DataFrame
    truth: pd.DataFrame | None
    floor: pd.DataFrame | None
    presence_basis: PresenceBasis | None
    boundary_period: int | None
    widowhood_mode: WidowhoodMode
    structural_delta: str | None


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing column(s) {missing}")


def _presence_keys(
    realized_presence: pd.DataFrame,
    *,
    person_column: str,
    period_column: str,
) -> pd.MultiIndex:
    _require_columns(
        realized_presence,
        (person_column, period_column),
        "realized presence",
    )
    keys = realized_presence[[person_column, period_column]]
    if keys.isna().any(axis=None):
        raise ValueError("realized presence contains a missing support key")
    if keys.duplicated().any():
        raise ValueError(
            "realized presence must contain one row per person-period key"
        )
    return pd.MultiIndex.from_frame(keys)


def _condition_on_presence(
    frame: pd.DataFrame,
    keys: pd.MultiIndex,
    *,
    person_column: str,
    period_column: str,
) -> pd.DataFrame:
    _require_columns(
        frame,
        (person_column, period_column),
        "evaluation frame",
    )
    frame_keys = pd.MultiIndex.from_frame(
        frame[[person_column, period_column]]
    )
    return frame.loc[frame_keys.isin(keys)].reset_index(drop=True)


def _conditioned_keys(
    frame: pd.DataFrame,
    *,
    person_column: str,
    period_column: str,
    label: str,
) -> pd.MultiIndex:
    keys = pd.MultiIndex.from_frame(frame[[person_column, period_column]])
    if keys.has_duplicates:
        raise ValueError(f"{label} has duplicate person-period support")
    return keys


def prepare_evaluation_support(
    projection: pd.DataFrame,
    *,
    mode: EvaluationMode,
    truth: pd.DataFrame | None = None,
    floor: pd.DataFrame | None = None,
    realized_presence: pd.DataFrame | None = None,
    presence_basis: PresenceBasis = PresenceBasis.EXACT_WAVE,
    start_weights: StartWaveWeightSnapshot | None = None,
    person_column: str = "person_id",
    period_column: str = "period",
    weight_column: str = "weight",
) -> PreparedSupport:
    """Prepare either symmetric gated support or unconditioned forward support.

    For ``START_OF_INTERVAL``, ``period_column`` is the interval's origin
    period; the function intentionally does not condition on end presence.
    For ``EXACT_WAVE``, it is the observed state-wave period.  Both policies
    use the same key filter, while the enum keeps their distinct estimands
    explicit in the returned record.

    A gated call requires all realized inputs and the boundary-weight snapshot.
    A forward call accepts only the projected panel and rejects realized future
    information and truth/floor comparisons.
    """
    mode = EvaluationMode(mode)
    if mode is EvaluationMode.FORWARD:
        forbidden = {
            "truth": truth,
            "floor": floor,
            "realized_presence": realized_presence,
            "start_weights": start_weights,
        }
        supplied = [
            name for name, value in forbidden.items() if value is not None
        ]
        if supplied:
            raise ValueError(
                "forward mode cannot consume realized holdout inputs: "
                f"{supplied}"
            )
        return PreparedSupport(
            mode=mode,
            projection=projection.copy(),
            truth=None,
            floor=None,
            presence_basis=None,
            boundary_period=None,
            widowhood_mode=WidowhoodMode.ENDOGENOUS_RECONCILIATION_REQUIRED,
            structural_delta=(
                "forward support cannot condition on realized future wave "
                "presence; endogenous widowhood reconciliation remains a "
                "successor-gate requirement"
            ),
        )

    missing = [
        name
        for name, value in (
            ("truth", truth),
            ("floor", floor),
            ("realized_presence", realized_presence),
            ("start_weights", start_weights),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"gated-realized mode requires {missing}")

    # Narrow Optional types after the explicit contract check above.
    assert truth is not None
    assert floor is not None
    assert realized_presence is not None
    assert start_weights is not None
    presence_basis = PresenceBasis(presence_basis)
    keys = _presence_keys(
        realized_presence,
        person_column=person_column,
        period_column=period_column,
    )
    conditioned = [
        _condition_on_presence(
            frame,
            keys,
            person_column=person_column,
            period_column=period_column,
        )
        for frame in (projection, truth, floor)
    ]
    projection_keys = _conditioned_keys(
        conditioned[0],
        person_column=person_column,
        period_column=period_column,
        label="conditioned projection",
    )
    truth_keys = _conditioned_keys(
        conditioned[1],
        person_column=person_column,
        period_column=period_column,
        label="conditioned truth",
    )
    if set(projection_keys) != set(truth_keys):
        raise ValueError(
            "symmetric presence-conditioning requires identical projection "
            "and truth person-period support"
        )
    weighted = [
        start_weights.apply(
            frame,
            person_column=person_column,
            weight_column=weight_column,
        )
        for frame in conditioned
    ]
    return PreparedSupport(
        mode=mode,
        projection=weighted[0],
        truth=weighted[1],
        floor=weighted[2],
        presence_basis=presence_basis,
        boundary_period=start_weights.boundary_period,
        widowhood_mode=WidowhoodMode.EXOGENOUS_CERTIFIED_HAZARD,
        structural_delta=None,
    )
