"""Proofs for the candidate-3 train-only F1 mechanism diagnostic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m6_c3_f1_mechanism_diagnostic as diagnostic  # noqa: E402

from populace_dynamics.engine.forward_earnings import (  # noqa: E402
    CellMarginal,
    ForwardEarningsGenerator,
    ProjectedWageIndex,
)


def _projected_index() -> ProjectedWageIndex:
    slope = math.log(1.1) / 2.0
    intercept = math.log(100.0) - slope * 2010
    return ProjectedWageIndex(
        actual={2010: 100.0},
        intercept=intercept,
        slope=slope,
        boundary_year=2010,
    )


def _generator(index: object) -> ForwardEarningsGenerator:
    cell = CellMarginal(
        p0=0.0,
        wtil=np.asarray([0.2, 0.5, 0.8]),
        yval=np.asarray([1.0, 2.0, 4.0]),
        n_pos=3,
        w_total=3.0,
    )
    return ForwardEarningsGenerator(
        shared_gate=object(),
        zero_anchor_gate=None,
        marginals={bin_index: cell for bin_index in range(8)},
        pools={},
        wage_index=index,
        u_w_by_person={},
        realized_earn_2014_by_person={},
        realized_earn_2012_by_person={},
        boundary_year=2010,
    )


def _frame(generator: ForwardEarningsGenerator) -> pd.DataFrame:
    people = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    ages = {
        2010: np.asarray([30.0, 35.0, 40.0, 50.0, 55.0, 60.0]),
        2012: np.asarray([32.0, 37.0, 42.0, 52.0, 57.0, 62.0]),
        2014: np.asarray([34.0, 39.0, 44.0, 54.0, 59.0, 64.0]),
    }
    ranks = {
        2010: np.asarray([0.2, 0.5, 0.8, 0.8, 0.5, 0.2]),
        2012: np.asarray([0.5, 0.8, 0.2, 0.2, 0.8, 0.5]),
        2014: np.asarray([0.8, 0.2, 0.5, 0.5, 0.2, 0.8]),
    }
    weights = np.asarray([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    cohorts = np.asarray(["prime"] * 3 + ["older"] * 3)
    frames = []
    for year in diagnostic.REFERENCE_WINDOW:
        frames.append(
            pd.DataFrame(
                {
                    "person_id": people,
                    "period": year,
                    "earnings": generator.rank_to_level(
                        ranks[year], ages[year], year
                    ),
                    "age": ages[year],
                    "weight": weights,
                    "cohort": cohorts,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_signed_decomposition_closes_exactly():
    split = diagnostic.decompose_gap(
        truth_value=0.25,
        candidate_value=0.10,
        realized_index_value=0.20,
    )

    assert split["gap_truth_minus_candidate"] == pytest.approx(0.15)
    assert split["index_explained_component"] == pytest.approx(0.10)
    assert split["residual_conditional_on_index"] == pytest.approx(0.05)
    assert split["index_explained_percent"] == pytest.approx(200.0 / 3.0)
    assert split["residual_percent"] == pytest.approx(100.0 / 3.0)
    assert split["closure_error"] == pytest.approx(0.0)


def test_index_only_difference_is_fully_index_explained():
    candidate_generator = _generator(_projected_index())
    realized_generator = diagnostic.with_realized_index_path(
        candidate_generator,
        realized_nawi={2012: 120.0, 2014: 150.0},
    )
    probe_rank = np.asarray([0.2, 0.5, 0.8])
    probe_age = np.asarray([32.0, 37.0, 42.0])
    realized_level = realized_generator.rank_to_level(
        probe_rank, probe_age, 2012
    )
    assert not np.array_equal(
        realized_level,
        candidate_generator.rank_to_level(probe_rank, probe_age, 2012),
    )
    assert np.allclose(
        realized_generator.level_to_rank(realized_level, probe_age, 2012),
        probe_rank,
    )
    assert realized_generator.wage_index.normalization_index(2012) == 120.0
    candidate = diagnostic.reduce_diagnostic_moments(
        _frame(candidate_generator)
    )
    counterfactual = diagnostic.reduce_diagnostic_moments(
        _frame(realized_generator)
    )
    truth = counterfactual

    assert (
        candidate["primary_f1"]["window_aggregate"]
        != counterfactual["primary_f1"]["window_aggregate"]
    )
    for key in ("window_aggregate", "2012", "2014"):
        candidate_value = (
            candidate["primary_f1"][key]
            if key == "window_aggregate"
            else candidate["primary_f1"]["by_reference_year"][key]
        )
        realized_value = (
            counterfactual["primary_f1"][key]
            if key == "window_aggregate"
            else counterfactual["primary_f1"]["by_reference_year"][key]
        )
        truth_value = (
            truth["primary_f1"][key]
            if key == "window_aggregate"
            else truth["primary_f1"]["by_reference_year"][key]
        )
        split = diagnostic.decompose_gap(
            truth_value=truth_value,
            candidate_value=candidate_value,
            realized_index_value=realized_value,
        )
        assert split["gap_truth_minus_candidate"] != 0.0
        assert split["index_explained_percent"] == pytest.approx(100.0)
        assert split["residual_percent"] == pytest.approx(0.0)


def test_identical_indexes_have_zero_index_explained_component():
    projected = _projected_index()
    candidate_generator = _generator(projected)
    identical_generator = diagnostic.with_realized_index_path(
        candidate_generator,
        realized_nawi={
            2012: projected.projected(2012),
            2014: projected.projected(2014),
        },
    )
    candidate_frame = _frame(candidate_generator)
    identical_frame = _frame(identical_generator)
    assert np.array_equal(
        candidate_frame["earnings"].to_numpy(),
        identical_frame["earnings"].to_numpy(),
    )

    truth_frame = identical_frame.copy()
    truth_frame.loc[truth_frame["period"] == 2014, "earnings"] *= 1.25
    candidate = diagnostic.reduce_diagnostic_moments(candidate_frame)
    counterfactual = diagnostic.reduce_diagnostic_moments(identical_frame)
    truth = diagnostic.reduce_diagnostic_moments(truth_frame)
    split = diagnostic.decompose_gap(
        truth_value=truth["primary_f1"]["window_aggregate"],
        candidate_value=candidate["primary_f1"]["window_aggregate"],
        realized_index_value=counterfactual["primary_f1"]["window_aggregate"],
    )

    assert split["gap_truth_minus_candidate"] != 0.0
    assert split["index_explained_component"] == pytest.approx(0.0)
    assert split["index_explained_percent"] == pytest.approx(0.0)
    assert split["residual_percent"] == pytest.approx(100.0)
