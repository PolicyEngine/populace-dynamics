"""Longitudinal views: score candidate panels in trajectory space.

Extends the population-view harness (adapted from
PolicyEngine/imputation-paper) with time. A panel is a view of the
same latent population whose projection lands in trajectory space: a
:class:`PanelView` reshapes person-period records into fixed windows —
the value columns at each offset plus window-start covariates — so the
geometry blocks (energy distance, PRDC, weighted C2ST, the tail block)
run unchanged on ``(y_t, ..., y_{t+h}, covariates)`` tuples under one
weight per trajectory window.

Two properties carry over from the cross-sectional harness by
construction. Holdout windows split by person, never by row, so no
person contributes to both sides. And PRDC coverage is invariant to
candidate reweighting, which makes it the guard against the
trajectory-reweighting failure mode the design paper warns about
(reweighting to hit future cross-sections can select entire
correlated life courses; weights cannot fake support).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from populace_dynamics.harness import views

__all__ = [
    "PanelView",
    "project_panel",
    "panel_scorecard",
    "noise_floor",
    "split_panel_by_person",
]


@dataclass(frozen=True)
class PanelView:
    """One panel's view of the population, in trajectory space.

    Attributes:
        name: Scorecard key (e.g. ``"psid_earnings_pairs"``).
        id_column: Person identifier; windows never span persons.
        period_column: Integer period (year or wave index).
        value_columns: Time-varying variables projected at every
            window offset.
        weight_column: Weight column; the window carries the weight
            observed at its first period (one weight per trajectory).
        covariate_columns: Columns projected at the window start only
            (age, cohort, or other slowly varying covariates).
        window: Number of observations per trajectory window
            (``window=2`` scores one-step dynamics).
        period_step: Spacing between consecutive observations
            (``2`` for biennial panels such as post-1997 PSID).
        target_columns: Value columns whose final-offset dimension
            receives the tail-sensitive block; defaults to all value
            columns.
    """

    name: str
    id_column: str
    period_column: str
    value_columns: tuple[str, ...]
    weight_column: str
    covariate_columns: tuple[str, ...] = ()
    window: int = 2
    period_step: int = 1
    target_columns: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be at least 2, got {self.window}.")
        if not self.value_columns:
            raise ValueError("value_columns must be non-empty.")
        if not self.target_columns:
            object.__setattr__(
                self, "target_columns", tuple(self.value_columns)
            )
        unknown = set(self.target_columns) - set(self.value_columns)
        if unknown:
            raise ValueError(
                f"target_columns {sorted(unknown)} are not in "
                "value_columns."
            )

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Projected dimension names, in point-matrix order."""
        names = [
            f"{col}_t{k}"
            for k in range(self.window)
            for col in self.value_columns
        ]
        names.extend(self.covariate_columns)
        return tuple(names)

    @property
    def target_dims(self) -> dict[str, int]:
        """Final-offset dimensions of the target columns."""
        names = self.dimension_names
        return {
            f"{col}_t{self.window - 1}": names.index(
                f"{col}_t{self.window - 1}"
            )
            for col in self.target_columns
        }


def project_panel(
    table: pd.DataFrame, view: PanelView
) -> tuple[np.ndarray, np.ndarray]:
    """Project a person-period table into trajectory-window space.

    A window starts at every observed person-period whose next
    ``window - 1`` observations exist at exactly ``period_step``
    spacing; gaps in observation break windows, which is how entry
    and exit enter the projection. Returns ``(points, weights)``
    with columns ordered as :attr:`PanelView.dimension_names` and one
    weight per window, taken at the window's first period.
    """
    required = [
        view.id_column,
        view.period_column,
        view.weight_column,
        *view.value_columns,
        *view.covariate_columns,
    ]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise ValueError(f"Panel projection is missing column(s) {missing}.")
    non_numeric = [
        c
        for c in (*view.value_columns, *view.covariate_columns)
        if not pd.api.types.is_numeric_dtype(table[c])
    ]
    if non_numeric:
        raise ValueError(
            f"Panel column(s) {non_numeric} are not numeric; encode "
            "them before projection."
        )
    base = table[required].copy()
    merged = base.rename(columns={c: f"{c}_t0" for c in view.value_columns})
    for k in range(1, view.window):
        ahead = table[
            [view.id_column, view.period_column, *view.value_columns]
        ].copy()
        ahead[view.period_column] = (
            ahead[view.period_column] - k * view.period_step
        )
        ahead = ahead.rename(
            columns={c: f"{c}_t{k}" for c in view.value_columns}
        )
        merged = merged.merge(
            ahead, on=[view.id_column, view.period_column], how="inner"
        )
    columns = list(view.dimension_names)
    points = merged.loc[:, columns].to_numpy(dtype=np.float64)
    weights = merged[view.weight_column].to_numpy(dtype=np.float64)
    return points, weights


def panel_scorecard(
    candidate: pd.DataFrame,
    holdout: pd.DataFrame,
    view: PanelView,
    *,
    k: int = 5,
    max_points: int = 2048,
    seed: int = 0,
) -> dict[str, float]:
    """Score a candidate panel against a held-out panel on one view.

    Both tables are person-period frames carrying the view's columns.
    The holdout must not have been used upstream in fitting or
    calibration. Returns the geometry-block scores of
    :func:`populace_dynamics.harness.views.score_view`, with the
    tail-sensitive block on the final-offset target dimensions.
    """
    candidate_points, candidate_weights = project_panel(candidate, view)
    holdout_points, holdout_weights = project_panel(holdout, view)
    return views.score_view(
        candidate_points,
        candidate_weights,
        holdout_points,
        holdout_weights,
        k=k,
        max_points=max_points,
        seed=seed,
        target_dims=view.target_dims,
    )


def split_panel_by_person(
    table: pd.DataFrame,
    id_column: str,
    *,
    fraction: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a panel into two person-disjoint parts.

    Splitting by person, not by row, keeps every window of a person
    on one side, so nothing a window touches leaks across the split.
    """
    ids = np.sort(table[id_column].unique())
    rng = np.random.default_rng(seed)
    picked = rng.random(len(ids)) < fraction
    left_ids = set(ids[picked])
    left = table[table[id_column].isin(left_ids)]
    right = table[~table[id_column].isin(left_ids)]
    return left, right


def noise_floor(
    holdout: pd.DataFrame,
    view: PanelView,
    *,
    k: int = 5,
    max_points: int = 2048,
    seed: int = 0,
) -> dict[str, float]:
    """Sampling-noise floor: half the holdout scored against half.

    Mirrors the same-survey reference of the cross-sectional harness:
    two person-disjoint halves of one panel differ only by sampling
    noise, so their scorecard is the floor a candidate is judged
    against rather than an absolute zero.
    """
    left, right = split_panel_by_person(
        holdout, view.id_column, fraction=0.5, seed=seed
    )
    return panel_scorecard(
        left, right, view, k=k, max_points=max_points, seed=seed
    )
