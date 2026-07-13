"""Synthetic unit tests for the injected-generator M4 engine adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import disability
from populace_dynamics.engine.disability import simulate_reproduction
from populace_dynamics.engine.support import StartWaveWeightSnapshot
from populace_dynamics.models import disability_hazard_sim as m4


def _panel() -> disability.DisabilityPanel:
    person_years = pd.DataFrame(
        {
            "person_id": [2, 1, 1, 2, 1, 2, 3],
            "period": [2015, 2015, 2017, 2017, 2021, 2021, 2015],
            "age": [61, 45, 47, 63, 51, 67, 30],
            "sex": pd.array(
                ["male", "female", "female", "male", "female", "male", "male"],
                dtype="string",
            ),
            "weight": [20.0, 10.0, 11.0, 21.0, 12.0, 22.0, 30.0],
            "status_code": [1, 1, 5, 5, 1, 4, 1],
            "disabled": [False, False, True, True, False, False, False],
            "retired": [False, False, False, False, False, True, False],
        }
    )
    return disability.DisabilityPanel(
        person_years=person_years,
        pairs=disability.build_transition_pairs(person_years),
    )


def _model() -> m4.DisabilityHazardModel:
    shape = (1 + len(disability.DI_AGE_BANDS), len(disability.SEXES))
    incidence = np.zeros(shape, dtype=np.float64)
    recovery = np.zeros(shape, dtype=np.float64)
    prevalence = np.zeros(shape, dtype=np.float64)
    incidence[1:, :] = np.array([0.25, 0.55])
    recovery[1:, :] = np.array([0.35, 0.65])
    prevalence[1:, :] = np.array([0.45, 0.75])
    return m4.DisabilityHazardModel(
        incidence=incidence,
        recovery=recovery,
        prevalence0=prevalence,
        exit_retirement_share=np.array([0.4, 0.8]),
        n_train_persons=100,
        train_moments_note="synthetic unit fixture",
    )


def _start_weights() -> StartWaveWeightSnapshot:
    return StartWaveWeightSnapshot.from_frame(
        pd.DataFrame({"person_id": [1, 2, 3], "weight": [10.0, 20.0, 30.0]}),
        boundary_period=2014,
    )


def test_injected_generator_adapter_is_exactly_the_frozen_seed_path():
    panel = _panel()
    model = _model()
    holdout = {1, 2}
    seed = 5207

    fixed_person_years = _start_weights().apply(panel.person_years)
    fixed_panel = disability.DisabilityPanel(
        person_years=fixed_person_years,
        pairs=disability.build_transition_pairs(fixed_person_years),
    )
    expected = m4.simulate_draw(fixed_panel, model, holdout, seed)
    actual = simulate_reproduction(
        panel,
        model,
        holdout,
        np.random.default_rng(seed),
        start_weights=_start_weights(),
    )

    pd.testing.assert_frame_equal(
        actual.person_years.drop(columns="di_converted"),
        expected.person_years,
    )
    assert not actual.person_years["di_converted"].any()
    pd.testing.assert_frame_equal(actual.pairs, expected.pairs)


def test_reproduction_preserves_realized_support_and_start_wave_weights():
    panel = _panel()
    simulated = simulate_reproduction(
        panel,
        _model(),
        {1, 2},
        np.random.default_rng(42),
        start_weights=_start_weights(),
    )
    expected = _start_weights().apply(
        panel.person_years[panel.person_years["person_id"].isin({1, 2})]
    )
    expected = expected.sort_values(
        ["person_id", "period"], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        simulated.person_years[
            ["person_id", "period", "age", "sex", "weight"]
        ],
        expected[["person_id", "period", "age", "sex", "weight"]],
    )
    assert set(simulated.person_years["person_id"]) == {1, 2}


def test_post_gap_wave_reinitializes_from_prevalence():
    panel = _panel()
    shape = (1 + len(disability.DI_AGE_BANDS), len(disability.SEXES))
    model = m4.DisabilityHazardModel(
        incidence=np.zeros(shape),
        recovery=np.zeros(shape),
        prevalence0=np.ones(shape),
        exit_retirement_share=np.zeros(len(disability.SEXES)),
        n_train_persons=1,
        train_moments_note="forced initial disability",
    )
    simulated = simulate_reproduction(
        panel,
        model,
        {1},
        np.random.default_rng(0),
        start_weights=_start_weights(),
    ).person_years

    # 2015 initializes disabled; 2017 carries it; the >2-year gap to 2021
    # starts a new episode and initializes disabled again.
    assert simulated["period"].tolist() == [2015, 2017, 2021]
    assert simulated["disabled"].tolist() == [True, True, True]


def test_disability_exit_to_retirement_is_marked_as_di_conversion():
    panel = _panel()
    shape = (1 + len(disability.DI_AGE_BANDS), len(disability.SEXES))
    model = m4.DisabilityHazardModel(
        incidence=np.zeros(shape),
        recovery=np.ones(shape),
        prevalence0=np.ones(shape),
        exit_retirement_share=np.ones(len(disability.SEXES)),
        n_train_persons=1,
        train_moments_note="forced conversion",
    )
    simulated = simulate_reproduction(
        panel,
        model,
        {2},
        np.random.default_rng(0),
        start_weights=_start_weights(),
    ).person_years

    assert simulated["period"].tolist() == [2015, 2017, 2021]
    assert simulated["di_converted"].tolist() == [False, True, False]


def test_adapter_requires_an_injected_generator():
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        simulate_reproduction(
            _panel(),
            _model(),
            {1},
            5200,  # type: ignore[arg-type]
            start_weights=_start_weights(),
        )
