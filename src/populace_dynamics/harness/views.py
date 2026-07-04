"""The population-view harness: score a candidate population against survey holdouts.

Adapted from PolicyEngine/imputation-paper
(src/imputation_paper/experiments/views.py), the population-view harness
described in that paper; extended here for longitudinal views.

The formal picture: one latent population distribution produces individuals;
each survey is a *view* of it -- a variable subset, a sampling design, a
measurement idiom -- and a survey's holdout is a weighted sample from that
view. A candidate synthetic population (any generator's weighted file) is
evaluated by projecting it through each view and comparing the projection to
that view's holdout *in the view's own variable space*. No cross-survey
consistency is ever required -- the candidate is only asked to explain each
view in its own idiom -- which is what makes the harness usable even though
surveys disagree with each other.

Per view the scorecard carries three complementary axes:

* :func:`~populace_dynamics.harness.metrics.energy_distance` -- strictly
  proper joint score (weight-sensitive block);
* :func:`~populace_dynamics.harness.metrics.prdc` -- support/coverage
  geometry; coverage is invariant to any reweighting of the candidate
  (calibration-blind block) and is the anti-mode-collapse axis;
* :func:`~populace_dynamics.harness.metrics.classifier_two_sample_auc` --
  the omnibus distinguishability check.

Sample-geometry metrics under a subsample cap are weak exactly where economic
variables live: deep in a heavy right tail. A candidate whose imputed wealth
q99 is twice the holdout's can tie on energy distance because the discrepant
mass is a sliver of standardized pairwise space. Views therefore carry a
fourth, tail-sensitive block on the *imputed target* columns: per-target
weighted Wasserstein-1 (scaled by the holdout's weighted sd) and weighted
q90/q99 ratios (candidate over holdout; 1 is perfect).

The holdouts must never have been used upstream (fitting or calibration), so
the harness is the stack's non-self-referential test surface. Design emulation
is by weighting in this version: a view compares *weighted* measures on both
sides; reproducing a survey's record-level selection mechanism is future work
and is documented as such in the paper.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.harness import metrics

__all__ = ["SurveyView", "project_view", "score_view", "harness_scorecard"]


@dataclass(frozen=True)
class SurveyView:
    """One survey's view of the population.

    Attributes:
        name: The survey's short name (e.g. ``"scf"``); the scorecard key.
        columns: The variables this view observes -- the projection. Every
            column must be numeric in any table the view projects.
        weight_column: The weight column carried by tables seen through this
            view (the survey's weights on its holdout; the candidate supplies
            its own weight column separately).
        target_columns: The subset of ``columns`` that were *imputed* (the
            rest are carried real data). The tail-sensitive block scores these
            columns marginally; empty means no tail block.
    """

    name: str
    columns: tuple[str, ...]
    weight_column: str
    target_columns: tuple[str, ...] = ()


def project_view(
    table: pd.DataFrame,
    columns: Iterable[str],
    weight_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a weighted table onto a view's variable space.

    Args:
        table: The table to project (candidate or holdout).
        columns: The view's variables; must all be numeric columns of ``table``.
        weight_column: The weight column of ``table``.

    Returns:
        ``(points, weights)``: float64 points of shape ``(n, d)`` and the
        aligned weight vector.

    Raises:
        ValueError: If a view column or the weight column is missing, or a
            view column is non-numeric. Messages name the culprits.
    """
    columns = list(columns)
    missing = [c for c in (*columns, weight_column) if c not in table.columns]
    if missing:
        raise ValueError(
            f"View projection is missing column(s) {missing}; the table has "
            f"{len(table.columns)} columns."
        )
    non_numeric = [
        c for c in columns if not pd.api.types.is_numeric_dtype(table[c])
    ]
    if non_numeric:
        raise ValueError(
            f"View column(s) {non_numeric} are not numeric; encode them "
            "before projection."
        )
    points = table.loc[:, columns].to_numpy(dtype=np.float64)
    weights = table[weight_column].to_numpy(dtype=np.float64)
    return points, weights


def score_view(
    candidate_points: np.ndarray,
    candidate_weights: np.ndarray,
    holdout_points: np.ndarray,
    holdout_weights: np.ndarray,
    *,
    k: int = 5,
    max_points: int = 2048,
    seed: int = 0,
    target_dims: dict[str, int] | None = None,
) -> dict[str, float]:
    """Score one projected candidate against one view's holdout.

    Args:
        candidate_points: Candidate projection, shape ``(m, d)``.
        candidate_weights: Candidate weights.
        holdout_points: Holdout projection, shape ``(n, d)``.
        holdout_weights: Holdout (survey) weights.
        k: PRDC neighbour rank.
        max_points: Pairwise-metric size cap (seeded resample above it).
        seed: Seed for resampling and the C2ST folds.
        target_dims: Imputed-column name -> dimension index; each gets the
            tail-sensitive block (``w1_over_sd``, ``q90_ratio``,
            ``q99_ratio``), computed on the full weighted samples (no
            subsample cap), because the tail is exactly what caps blur.

    Returns:
        Metric name -> value: ``energy_distance``, ``c2st_auc``, the four
        ``prdc_*`` components, and per-target tail metrics
        ``{w1_over_sd,q90_ratio,q99_ratio}.<target>``.
    """
    scores: dict[str, float] = {
        "energy_distance": metrics.energy_distance(
            candidate_points,
            holdout_points,
            imputed_weights=candidate_weights,
            holdout_weights=holdout_weights,
            max_points=max_points,
            seed=seed,
        ),
        "c2st_auc": metrics.classifier_two_sample_auc(
            holdout_points,
            candidate_points,
            real_weights=holdout_weights,
            synthetic_weights=candidate_weights,
            seed=seed,
        ),
    }
    prdc_scores = metrics.prdc(
        holdout_points,
        candidate_points,
        k=k,
        real_weights=holdout_weights,
        synthetic_weights=candidate_weights,
        max_points=max_points,
        seed=seed,
    )
    scores.update(
        {f"prdc_{name}": value for name, value in prdc_scores.items()}
    )

    for name, dim in (target_dims or {}).items():
        candidate_col = candidate_points[:, dim]
        holdout_col = holdout_points[:, dim]
        moments = metrics._weighted_moments(
            holdout_col[:, None], holdout_weights
        )
        holdout_sd = float(moments[1][0])
        w1 = metrics.weighted_wasserstein1(
            candidate_col,
            holdout_col,
            imputed_weights=candidate_weights,
            donor_weights=holdout_weights,
        )
        scores[f"w1_over_sd.{name}"] = (
            w1 / holdout_sd if holdout_sd > 0 else w1
        )
        quantiles = np.array([0.90, 0.99])
        candidate_q = metrics._weighted_quantile(
            candidate_col, candidate_weights, quantiles
        )
        holdout_q = metrics._weighted_quantile(
            holdout_col, holdout_weights, quantiles
        )
        for level, c_q, h_q in zip(
            ("q90", "q99"), candidate_q, holdout_q, strict=True
        ):
            scores[f"{level}_ratio.{name}"] = (
                float(c_q / h_q) if h_q != 0 else np.nan
            )
    return scores


def harness_scorecard(
    candidate: pd.DataFrame,
    candidate_weight_column: str,
    views: Iterable[SurveyView],
    holdouts: Mapping[str, pd.DataFrame],
    *,
    k: int = 5,
    max_points: int = 2048,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Score a candidate population against every view's holdout.

    Args:
        candidate: The candidate weighted population file. It must carry every
            view's columns (that is what makes it a *population* claim) plus
            ``candidate_weight_column``.
        candidate_weight_column: The candidate's weight column.
        views: The survey views to score against.
        holdouts: View name -> holdout table (carrying that view's columns and
            its ``weight_column``). Holdout rows must not have been used
            upstream in fitting or calibration.
        k: PRDC neighbour rank.
        max_points: Pairwise-metric size cap.
        seed: Seed for resampling and C2ST folds.

    Returns:
        Long-format rows ``{"view", "metric", "value"}`` -- one per view and
        metric, matching the sweep's ``metrics_long.csv`` idiom (the caller
        adds method/seed context).

    Raises:
        KeyError: If a view has no holdout table.
        ValueError: If a projection fails (missing/non-numeric columns).
    """
    rows: list[dict[str, Any]] = []
    for view in views:
        if view.name not in holdouts:
            raise KeyError(
                f"No holdout table for view {view.name!r}; have {sorted(holdouts)}."
            )
        candidate_points, candidate_weights = project_view(
            candidate, view.columns, candidate_weight_column
        )
        holdout_points, holdout_weights = project_view(
            holdouts[view.name], view.columns, view.weight_column
        )
        target_dims = {
            name: list(view.columns).index(name)
            for name in view.target_columns
        }
        scores = score_view(
            candidate_points,
            candidate_weights,
            holdout_points,
            holdout_weights,
            k=k,
            max_points=max_points,
            seed=seed,
            target_dims=target_dims,
        )
        rows.extend(
            {"view": view.name, "metric": metric, "value": value}
            for metric, value in scores.items()
        )
    return rows
