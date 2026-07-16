"""Synthetic tests for M6 projected-support preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import disability, transitions
from populace_dynamics.engine.support import (
    EvaluationMode,
    PresenceBasis,
    StartWaveWeightSnapshot,
)
from populace_dynamics.harness.m6_cells import disability_pairs
from populace_dynamics.harness.m6_projection import (
    prepare_gated_realized_support,
    prepare_projected_disability,
    prepare_projected_marital,
    project_earnings_on_realized_support,
)


class _CertifiedEarnings:
    realized_earn_2014_by_person = {1: 100.0}
    realized_earn_2012_by_person = {1: 90.0}
    u_w_by_person = {1: 0.2}

    def __init__(self):
        self.seen_columns = []
        self.seen_ages = []

    def materialize_initial_frame(self, frame):
        out = frame.copy()
        out["u_w"] = 0.2
        out["realized_earn_2014"] = 100.0
        out["realized_earn_2012"] = 90.0
        out["earnings"] = 100.0
        out["gen_earn_w2"] = 100.0
        out["gen_earn_w4"] = 90.0
        return out

    def generate(self, frame, year, rng):
        self.seen_columns.append(tuple(frame.columns))
        self.seen_ages.append((year, frame["age"].tolist()))
        if year % 2:
            return frame["earnings"].to_numpy(float)
        return np.asarray([100.0 + rng.random()])


def test_earnings_reproduction_uses_truth_support_and_fences_realized_nawi():
    generator = _CertifiedEarnings()
    initial = pd.DataFrame(
        {
            "person_id": [1, 2],
            "year": [2014, 2014],
            "age": [31, 40],
            "sex": ["female", "male"],
            "earnings_domain": [True, False],
            "nawi": [object(), object()],
        }
    )
    truth = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "period": [2014, 2016, 2018],
            "earnings": [100.0, 110.0, 120.0],
            "age": [30, 32, 34],
            "weight": [2.0, 2.0, 2.0],
            "cohort": ["prime", "prime", "prime"],
        }
    )

    projected = project_earnings_on_realized_support(
        initial_slice=initial,
        truth_support=truth,
        generator=generator,
        domain_person_ids={1},
        all_person_ids={1, 2},
        draw_index=0,
    )

    assert projected[["person_id", "period"]].values.tolist() == [
        [1, 2014],
        [1, 2016],
        [1, 2018],
    ]
    assert all("nawi" not in columns for columns in generator.seen_columns)
    assert generator.seen_ages == [
        (2015, [32]),
        (2016, [33]),
        (2017, [34]),
        (2018, [35]),
    ]
    prepared = prepare_gated_realized_support(
        projected,
        truth,
        realized_presence=truth[["person_id", "period"]],
        start_weights=StartWaveWeightSnapshot.from_frame(
            pd.DataFrame({"person_id": [1], "weight": [2.0]}),
            boundary_period=2014,
        ),
        presence_basis=PresenceBasis.EXACT_WAVE,
        period_column="period",
    )
    assert prepared.mode is EvaluationMode.GATED_REALIZED
    assert prepared.presence_basis is PresenceBasis.EXACT_WAVE
    assert prepared.projection["weight"].tolist() == [2.0, 2.0, 2.0]


def test_projected_flow_preparation_applies_gated_windows_and_presence():
    person_years = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 1, 2],
            "year": [2015, 2016, 2019, 2020, 2015],
            "age": [20, 21, 24, 25, 30],
            "sex": ["female", "female", "female", "female", "male"],
            "weight": [9.0] * 5,
            "marital_state": ["never_married"] * 5,
        }
    )
    events = person_years.iloc[[1, 2, 3]].copy()
    events["transition"] = "first_marriage"
    panel = transitions.MaritalPanel(
        person_years, events, pd.DataFrame({"person_id": [1, 2]})
    )
    anchor = pd.DataFrame(
        {
            "person_id": [1, 2],
            "weight": [1.0, 2.0],
            "household_id": [10, 20],
        }
    )
    projected_events, projected_py = prepare_projected_marital(
        panel,
        anchor,
        {2015: {1}, 2017: set(), 2019: {1}},
    )

    assert projected_events["year"].tolist() == [2016, 2019]
    assert projected_py["person_id"].unique().tolist() == [1]
    assert projected_py["weight"].unique().tolist() == [1.0]
    marital_support = prepare_gated_realized_support(
        projected_py,
        projected_py.assign(weight=99.0),
        realized_presence=projected_py[["person_id", "year"]],
        start_weights=StartWaveWeightSnapshot.from_frame(
            anchor[["person_id", "weight"]], boundary_period=2014
        ),
        presence_basis=PresenceBasis.START_OF_INTERVAL,
        period_column="year",
    )
    assert marital_support.projection["weight"].unique().tolist() == [1.0]
    assert marital_support.truth is not None
    assert marital_support.truth["weight"].unique().tolist() == [1.0]

    dis_py = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 1],
            "period": [2015, 2017, 2019, 2021],
            "age": [30, 32, 34, 36],
            "sex": ["female"] * 4,
            "weight": [99.0] * 4,
            "disabled": [False, True, False, True],
        }
    )
    dis_panel = disability.DisabilityPanel(dis_py, pd.DataFrame())
    pairs = prepare_projected_disability(dis_panel, anchor.iloc[[0]])

    assert pairs["start_wave"].tolist() == [2015, 2017]
    assert pairs["weight"].tolist() == [1.0, 1.0]


def test_disability_projection_and_truth_share_the_scored_universe():
    rows = []
    cases = (
        (1, 18, "female", False),
        (2, 30, "male", False),
        (3, 70, "female", True),
        (4, 40, "uncoded", False),
    )
    for person_id, age, sex, retired in cases:
        rows.extend(
            [
                {
                    "person_id": person_id,
                    "period": period,
                    "age": age + offset,
                    "disabled": bool(offset),
                    "retired": retired,
                    "sex": sex,
                }
                for period, offset in ((2015, 0), (2017, 2))
            ]
        )
    projected_panel = disability.DisabilityPanel(
        person_years=pd.DataFrame(rows),
        pairs=pd.DataFrame(),
    )
    status = projected_panel.person_years.drop(columns="sex")
    death_records = pd.DataFrame(
        {
            "person_id": [case[0] for case in cases],
            "sex": [case[2] for case in cases],
        }
    )
    anchor = pd.DataFrame(
        {
            "person_id": [case[0] for case in cases],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )

    truth = disability_pairs(status, death_records, anchor)
    projected = prepare_projected_disability(projected_panel, anchor)

    assert truth[["person_id", "start_wave"]].values.tolist() == [[2, 2015]]
    assert projected[["person_id", "start_wave"]].values.tolist() == [
        [2, 2015]
    ]
    legacy_projected = pd.concat(
        [
            projected,
            pd.DataFrame(
                {
                    "person_id": [1, 3, 4],
                    "sex": ["female", "female", "uncoded"],
                    "age": [18, 70, 40],
                    "start_wave": [2015, 2015, 2015],
                    "from_disabled": [False, False, False],
                    "to_disabled": [True, True, True],
                    "weight": [1.0, 3.0, 4.0],
                    "band": [None, None, "40-49"],
                }
            ),
        ],
        ignore_index=True,
    )
    presence = pd.DataFrame(
        {"person_id": [1, 2, 3, 4], "start_wave": [2015] * 4}
    )
    weights = StartWaveWeightSnapshot.from_frame(anchor, boundary_period=2014)
    with pytest.raises(
        ValueError,
        match=(
            "symmetric presence-conditioning requires identical projection "
            "and truth person-period support"
        ),
    ):
        prepare_gated_realized_support(
            legacy_projected,
            truth,
            realized_presence=presence,
            start_weights=weights,
            presence_basis=PresenceBasis.START_OF_INTERVAL,
            period_column="start_wave",
        )
    prepared = prepare_gated_realized_support(
        projected,
        truth,
        realized_presence=presence,
        start_weights=weights,
        presence_basis=PresenceBasis.START_OF_INTERVAL,
        period_column="start_wave",
    )
    assert prepared.projection[
        ["person_id", "start_wave"]
    ].values.tolist() == [[2, 2015]]
    assert prepared.truth is not None
    assert prepared.truth[["person_id", "start_wave"]].values.tolist() == [
        [2, 2015]
    ]
