"""Synthetic unit tests for M6 support and fixed-boundary weights."""

from __future__ import annotations

import pandas as pd
import pytest

from populace_dynamics.engine.support import (
    EvaluationMode,
    PresenceBasis,
    StartWaveWeightSnapshot,
    WidowhoodMode,
    prepare_evaluation_support,
)


def _side(weight_offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3],
            "period": [2015, 2016, 2015, 2016, 2015],
            "weight": [
                1.0 + weight_offset,
                2.0 + weight_offset,
                3.0 + weight_offset,
                4.0 + weight_offset,
                5.0 + weight_offset,
            ],
            "event": [False, True, True, False, True],
        }
    )


def _snapshot() -> StartWaveWeightSnapshot:
    return StartWaveWeightSnapshot.from_frame(
        pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "weight": [100.0, 200.0, 300.0],
            }
        ),
        boundary_period=2014,
    )


@pytest.mark.parametrize(
    "basis",
    [PresenceBasis.START_OF_INTERVAL, PresenceBasis.EXACT_WAVE],
)
def test_gated_support_conditions_all_sides_and_fixes_boundary_weight(basis):
    presence = pd.DataFrame(
        {
            "person_id": [1, 2, 2],
            "period": [2015, 2015, 2016],
        }
    )
    prepared = prepare_evaluation_support(
        _side(0.0),
        mode=EvaluationMode.GATED_REALIZED,
        truth=_side(10.0),
        floor=_side(20.0),
        realized_presence=presence,
        presence_basis=basis,
        start_weights=_snapshot(),
    )

    expected_keys = [(1, 2015), (2, 2015), (2, 2016)]
    for frame in (prepared.projection, prepared.truth, prepared.floor):
        assert frame is not None
        assert list(
            frame[["person_id", "period"]].itertuples(index=False, name=None)
        ) == (expected_keys)
        assert frame["weight"].tolist() == [100.0, 200.0, 200.0]
    assert prepared.presence_basis is basis
    assert prepared.boundary_period == 2014
    assert prepared.widowhood_mode is WidowhoodMode.EXOGENOUS_CERTIFIED_HAZARD
    assert prepared.structural_delta is None


def test_start_of_interval_does_not_require_end_presence():
    intervals = pd.DataFrame(
        {
            "person_id": [1, 2],
            "period": [2015, 2015],
            "end_period": [2016, 2016],
            "weight": [1.0, 1.0],
        }
    )
    start_presence = intervals[["person_id", "period"]].copy()
    prepared = prepare_evaluation_support(
        intervals,
        mode=EvaluationMode.GATED_REALIZED,
        truth=intervals,
        floor=intervals,
        realized_presence=start_presence,
        presence_basis=PresenceBasis.START_OF_INTERVAL,
        start_weights=_snapshot(),
    )
    assert len(prepared.projection) == 2


def test_forward_mode_rejects_realized_future_information():
    projection = _side(0.0)
    presence = projection[["person_id", "period"]]
    with pytest.raises(ValueError, match="forward mode cannot consume"):
        prepare_evaluation_support(
            projection,
            mode=EvaluationMode.FORWARD,
            realized_presence=presence,
        )

    prepared = prepare_evaluation_support(
        projection,
        mode=EvaluationMode.FORWARD,
    )
    pd.testing.assert_frame_equal(prepared.projection, projection)
    assert prepared.truth is None
    assert prepared.floor is None
    assert prepared.presence_basis is None
    assert prepared.widowhood_mode is (
        WidowhoodMode.ENDOGENOUS_RECONCILIATION_REQUIRED
    )
    assert "cannot condition" in prepared.structural_delta


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"person_id": [1, 1], "weight": [1.0, 2.0]}),
            "one row per person",
        ),
        (
            pd.DataFrame({"person_id": [1], "weight": [0.0]}),
            "positive and finite",
        ),
    ],
)
def test_boundary_weight_snapshot_fails_loudly_on_invalid_input(
    frame, message
):
    with pytest.raises(ValueError, match=message):
        StartWaveWeightSnapshot.from_frame(frame, boundary_period=2014)


def test_gated_support_requires_weight_coverage_for_every_present_person():
    projection = pd.DataFrame(
        {"person_id": [99], "period": [2015], "weight": [1.0]}
    )
    presence = projection[["person_id", "period"]]
    with pytest.raises(ValueError, match="does not cover"):
        prepare_evaluation_support(
            projection,
            mode=EvaluationMode.GATED_REALIZED,
            truth=projection,
            floor=projection,
            realized_presence=presence,
            start_weights=_snapshot(),
        )


def test_gated_support_rejects_asymmetric_projection_and_truth_keys():
    truth = _side(0.0)
    projection = truth.iloc[1:].reset_index(drop=True)
    presence = truth[["person_id", "period"]]
    with pytest.raises(ValueError, match="identical projection and truth"):
        prepare_evaluation_support(
            projection,
            mode=EvaluationMode.GATED_REALIZED,
            truth=truth,
            floor=truth,
            realized_presence=presence,
            start_weights=_snapshot(),
        )
