"""The population-view harness on a known latent population.

Adapted from PolicyEngine/imputation-paper (tests/test_views.py), the
population-view harness described in that paper; extended here for
longitudinal views.

The setup mirrors the paper's formal picture at toy scale: one latent
population over (a, b, c); a "cps" view observes (a, b), an "scf" view
observes (a, c). A faithful candidate (an independent draw from the same
population) must beat a mode-collapsed candidate on every view, and the
scorecard plumbing must be shaped like the sweep's long rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.harness.views import (
    SurveyView,
    harness_scorecard,
    project_view,
)

VIEWS = (
    SurveyView(name="cps", columns=("a", "b"), weight_column="weight"),
    SurveyView(name="scf", columns=("a", "c"), weight_column="weight"),
)


def _population(seed: int, n: int = 700) -> pd.DataFrame:
    """A draw from the latent toy population over (a, b, c), weighted."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    b = 0.8 * a + rng.normal(0.0, 0.6, n)
    c = -0.5 * a + rng.normal(0.0, 0.8, n)
    weight = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "weight": weight})


def test_harness_prefers_faithful_candidate_on_every_view() -> None:
    """Faithful draw beats modal collapse on energy and coverage, per view."""
    holdouts = {"cps": _population(seed=1), "scf": _population(seed=2)}
    faithful = _population(seed=3)
    modal = faithful.copy()
    for column in ("a", "b", "c"):
        modal[column] = float(faithful[column].median())

    def by_key(rows):
        return {(r["view"], r["metric"]): r["value"] for r in rows}

    good = by_key(
        harness_scorecard(faithful, "weight", VIEWS, holdouts, seed=0)
    )
    bad = by_key(harness_scorecard(modal, "weight", VIEWS, holdouts, seed=0))

    for view in ("cps", "scf"):
        assert good[(view, "energy_distance")] < bad[(view, "energy_distance")]
        assert good[(view, "prdc_coverage")] > 0.6
        assert bad[(view, "prdc_coverage")] < 0.1
        # The faithful candidate is near-indistinguishable; the collapsed one
        # is trivially separable.
        assert abs(good[(view, "c2st_auc")] - 0.5) < 0.15
        assert bad[(view, "c2st_auc")] > 0.9


def test_scorecard_rows_are_long_format_and_complete() -> None:
    """One row per view x metric, with the sweep's row schema."""
    holdouts = {"cps": _population(seed=4), "scf": _population(seed=5)}
    rows = harness_scorecard(_population(seed=6), "weight", VIEWS, holdouts)
    assert {tuple(sorted(r)) for r in rows} == {("metric", "value", "view")}
    per_view = {view.name for view in VIEWS}
    assert {r["view"] for r in rows} == per_view
    metrics_seen = {r["metric"] for r in rows if r["view"] == "cps"}
    assert {
        "energy_distance",
        "c2st_auc",
        "prdc_precision",
        "prdc_recall",
        "prdc_density",
        "prdc_coverage",
    } == metrics_seen


def test_projection_validates_columns() -> None:
    """Missing or non-numeric view columns are refused, named."""
    table = _population(seed=7).assign(label=lambda df: df["a"].astype(str))
    with pytest.raises(ValueError, match="missing"):
        project_view(table, ["a", "ghost"], "weight")
    with pytest.raises(ValueError, match="not numeric"):
        project_view(table, ["a", "label"], "weight")
    with pytest.raises(KeyError, match="No holdout"):
        harness_scorecard(table, "weight", VIEWS, {"cps": table})


def test_tail_block_scores_imputed_targets() -> None:
    """Views with target_columns emit tail metrics that see tail inflation.

    A candidate whose target tail is inflated 3x must show a q99_ratio near 3
    even if its body matches -- the failure mode sample-geometry metrics blur.
    """
    holdout = _population(seed=8, n=4000)
    inflated = _population(seed=9, n=4000)
    # Inflate the top decile of `c` only: the body is untouched.
    top = inflated["c"] > inflated["c"].quantile(0.9)
    inflated.loc[top, "c"] = inflated.loc[top, "c"] * 3.0

    view = SurveyView(
        name="scf",
        columns=("a", "c"),
        weight_column="weight",
        target_columns=("c",),
    )
    rows = harness_scorecard(
        inflated, "weight", [view], {"scf": holdout}, seed=0
    )
    by_metric = {r["metric"]: r["value"] for r in rows}
    assert {"w1_over_sd.c", "q90_ratio.c", "q99_ratio.c"} <= set(by_metric)
    assert by_metric["q99_ratio.c"] > 1.5
    assert by_metric["w1_over_sd.c"] > 0.0

    faithful = _population(seed=10, n=4000)
    faithful_rows = harness_scorecard(
        faithful, "weight", [view], {"scf": holdout}, seed=0
    )
    faithful_by_metric = {r["metric"]: r["value"] for r in faithful_rows}
    assert 0.8 < faithful_by_metric["q99_ratio.c"] < 1.25
    assert faithful_by_metric["w1_over_sd.c"] < by_metric["w1_over_sd.c"]
