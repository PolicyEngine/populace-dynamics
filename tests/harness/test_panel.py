"""Tests for longitudinal views (PanelView projection and scoring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.harness import panel


def _panel_frame(
    n_ids: int = 300,
    n_periods: int = 8,
    phi: float = 0.7,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(n_ids):
        log_y = rng.normal(10.0, 1.0)
        birth_year = int(rng.integers(1950, 1990))
        for period in range(n_periods):
            log_y = (
                phi * (log_y - 10.0)
                + 10.0
                + rng.normal(0.0, np.sqrt(1 - phi**2))
            )
            rows.append(
                {
                    "person_id": person,
                    "period": 2000 + period,
                    "earnings": np.exp(log_y),
                    "age": 2000 + period - birth_year,
                    "weight": rng.uniform(0.5, 2.0),
                }
            )
    return pd.DataFrame(rows)


VIEW = panel.PanelView(
    name="test_pairs",
    id_column="person_id",
    period_column="period",
    value_columns=("earnings",),
    covariate_columns=("age",),
    weight_column="weight",
    window=2,
)


def test_dimension_names_and_target_dims():
    view = panel.PanelView(
        name="v",
        id_column="i",
        period_column="p",
        value_columns=("a", "b"),
        covariate_columns=("age",),
        weight_column="w",
        window=3,
    )
    assert view.dimension_names == (
        "a_t0",
        "b_t0",
        "a_t1",
        "b_t1",
        "a_t2",
        "b_t2",
        "age",
    )
    assert view.target_dims == {"a_t2": 4, "b_t2": 5}


def test_window_below_two_rejected():
    with pytest.raises(ValueError, match="window"):
        panel.PanelView(
            name="v",
            id_column="i",
            period_column="p",
            value_columns=("a",),
            weight_column="w",
            window=1,
        )


def test_projection_shapes_and_weight_at_start():
    frame = _panel_frame(n_ids=5, n_periods=4)
    points, weights = panel.project_panel(frame, VIEW)
    # 4 periods -> 3 windows per person, 5 persons.
    assert points.shape == (15, 3)
    start = frame[frame.period < 2003].sort_values(["person_id", "period"])
    assert np.allclose(np.sort(weights), np.sort(start.weight))


def test_projection_breaks_windows_at_gaps():
    frame = _panel_frame(n_ids=1, n_periods=6)
    gapped = frame[frame.period != 2002]
    points, _ = panel.project_panel(gapped, VIEW)
    # Periods 2000,2001,2003,2004,2005: pairs (2000,2001),
    # (2003,2004), (2004,2005) — the gap kills two pairs.
    assert points.shape[0] == 3


def test_projection_respects_period_step():
    frame = _panel_frame(n_ids=3, n_periods=6)
    biennial = frame[frame.period % 2 == 0]
    view = panel.PanelView(
        name="biennial",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        weight_column="weight",
        window=2,
        period_step=2,
    )
    points, _ = panel.project_panel(biennial, view)
    assert points.shape[0] == 3 * 2  # 3 waves -> 2 pairs per person
    annual = panel.PanelView(
        name="annual",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        weight_column="weight",
        window=2,
        period_step=1,
    )
    points_annual, _ = panel.project_panel(biennial, annual)
    assert points_annual.shape[0] == 0


def test_projection_missing_column_raises():
    frame = _panel_frame(n_ids=2, n_periods=3).drop(columns=["age"])
    with pytest.raises(ValueError, match="missing column"):
        panel.project_panel(frame, VIEW)


def test_split_panel_by_person_is_disjoint_and_exhaustive():
    frame = _panel_frame(n_ids=100)
    left, right = panel.split_panel_by_person(frame, "person_id", seed=1)
    left_ids = set(left.person_id)
    right_ids = set(right.person_id)
    assert left_ids.isdisjoint(right_ids)
    assert left_ids | right_ids == set(frame.person_id)


def test_scorecard_same_process_near_floor():
    holdout = _panel_frame(n_ids=400, seed=2)
    candidate = _panel_frame(n_ids=400, seed=3)
    scores = panel.panel_scorecard(candidate, holdout, VIEW, seed=0)
    floor = panel.noise_floor(holdout, VIEW, seed=0)
    assert 0.35 < scores["c2st_auc"] < 0.65
    assert scores["prdc_coverage"] > 0.8
    assert scores["energy_distance"] < floor["energy_distance"] + 0.1
    for name in VIEW.target_dims:
        assert scores[f"q99_ratio.{name}"] == pytest.approx(1.0, abs=0.35)


def test_scorecard_detects_broken_dynamics():
    holdout = _panel_frame(n_ids=400, phi=0.9, seed=4)
    shuffled = _panel_frame(n_ids=400, phi=0.0, seed=5)
    good = _panel_frame(n_ids=400, phi=0.9, seed=6)
    scores_bad = panel.panel_scorecard(shuffled, holdout, VIEW, seed=0)
    scores_good = panel.panel_scorecard(good, holdout, VIEW, seed=0)
    # Marginals match by construction; only the joint separates them.
    assert scores_bad["energy_distance"] > scores_good["energy_distance"]
    assert scores_bad["c2st_auc"] > scores_good["c2st_auc"]
