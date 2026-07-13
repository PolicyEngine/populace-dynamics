"""Synthetic unit coverage for M6's injected candidate-16 step."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from populace_dynamics.data import transitions
from populace_dynamics.engine.marital import simulate_marital_step
from populace_dynamics.engine.steps import simulate_fertility
from populace_dynamics.models.couple_formation_sim_v2 import (
    FirstMarriageEarningsModifier,
)
from populace_dynamics.models.family_transitions.components.initial_states import (
    EntryWidowedCells,
    ObservedInitialStates,
    SupportComposition,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)


class _ConstantFirstMarriage:
    def predict(self, age, is_male, birth_decade):
        del is_male, birth_decade
        return np.full(len(age), 0.28)


def _panel() -> transitions.MaritalPanel:
    attrs = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "birth_year": [1980, 1982, 1978, 1981, 1983, 1979],
            "sex": ["female", "male", "female", "male", "female", "male"],
            "weight": [1.0, 1.5, 0.8, 1.2, 0.9, 1.1],
            "start_exposure_year": [2010] * 6,
            "censor_year": [2015] * 6,
            "n_marriages": [0.0] * 6,
        }
    )
    rows = []
    for row in attrs.itertuples(index=False):
        for year in range(row.start_exposure_year, row.censor_year + 1):
            rows.append(
                {
                    "person_id": row.person_id,
                    "year": year,
                    "age": year - row.birth_year,
                    "sex": row.sex,
                    "weight": row.weight,
                    "marital_state": "never_married",
                    "marriage_duration": pd.NA,
                    "years_since_dissolution": pd.NA,
                }
            )
    person_years = pd.DataFrame(rows)
    person_years["marriage_duration"] = pd.array(
        person_years["marriage_duration"], dtype="Int64"
    )
    person_years["years_since_dissolution"] = pd.array(
        person_years["years_since_dissolution"], dtype="Int64"
    )
    events = pd.DataFrame(
        columns=[
            "person_id",
            "year",
            "age",
            "sex",
            "weight",
            "transition",
            "marriage_duration",
            "years_since_dissolution",
            "origin",
        ]
    )
    return transitions.MaritalPanel(person_years, events, attrs)


def _components() -> FittedFamilyTransitions:
    empty_entry = EntryWidowedCells(
        key_sorted=np.array([], dtype=np.int64),
        years_since_dissolution_sorted=np.array([], dtype=np.int64),
        person_id_sorted=np.array([], dtype=np.int64),
        weight_sorted=np.array([], dtype=np.float64),
    )
    initial = ObservedInitialStates(
        marriage_residual_by_person={},
        entry_widowed=empty_entry,
        support=SupportComposition(
            pd.Series(0, index=pd.Index(range(1, 7), name="person_id"))
        ),
    )
    fertility = {
        (age, parity, decade): 0.08
        for age in range(15, 50)
        for parity in range(4)
        for decade in (1970, 1980)
    }
    gaps = {
        sex: {band: np.array([0], dtype=np.int64) for band in range(4)}
        for sex in ("female", "male")
    }
    return FittedFamilyTransitions(
        first_marriage=_ConstantFirstMarriage(),
        divorce=np.zeros((4, 2)),
        widowhood=SimpleNamespace(lookup=np.zeros((7, 2, 2))),
        remarriage={},
        fertility=fertility,
        initial_states=initial,
        spousal_age_gaps=gaps,
        implementation_ids={},
    )


def _neutral_modifier() -> FirstMarriageEarningsModifier:
    ones = np.ones((2, 4, 3))
    phi = np.full((2, 4, 3), 1.0 / 3.0)
    return FirstMarriageEarningsModifier(
        m_norm=ones.copy(),
        m_raw=ones.copy(),
        m_shrunk=ones.copy(),
        phi_cert=phi,
        z_norm=np.ones((2, 4)),
        n_events=np.ones((2, 4, 3), dtype=int),
        alpha=8.0,
        meta={},
    )


def test_injected_generators_replay_the_step3_marital_path():
    panel = _panel()
    components = _components()
    seed = 5207
    main_rng = np.random.default_rng(seed)
    gap_seed = main_rng.bit_generator.seed_seq.spawn(1)[0]
    result = simulate_marital_step(
        panel,
        set(range(1, 7)),
        components,
        _neutral_modifier(),
        SimpleNamespace(
            earn={person: float(person) for person in range(1, 7)},
            cuts=(2.5, 4.5),
        ),
        main_rng=main_rng,
        gap_rng=np.random.default_rng(gap_seed),
    )

    replay_main = np.random.default_rng(seed)
    replay_gap_seed = replay_main.bit_generator.seed_seq.spawn(1)[0]
    replay = simulate_marital_step(
        panel,
        set(range(1, 7)),
        components,
        _neutral_modifier(),
        SimpleNamespace(
            earn={person: float(person) for person in range(1, 7)},
            cuts=(2.5, 4.5),
        ),
        main_rng=replay_main,
        gap_rng=np.random.default_rng(replay_gap_seed),
    )
    pd.testing.assert_frame_equal(
        result.panel.person_years, replay.panel.person_years
    )
    pd.testing.assert_frame_equal(result.panel.events, replay.panel.events)
    assert result.births.empty
    assert result.sim_years is result.panel.person_years
    assert result.modifier_check["constraint_holds"] is True
    assert set(result.exposure["person_id"]) == set(range(1, 7))


def test_step4_fertility_reads_authoritative_married_state_for_fathers():
    panel = _panel()
    components = _components()
    result = simulate_marital_step(
        panel,
        set(range(1, 7)),
        components,
        _neutral_modifier(),
        SimpleNamespace(
            earn={person: float(person) for person in range(1, 7)},
            cuts=(2.5, 4.5),
        ),
        main_rng=np.random.default_rng(3),
        gap_rng=np.random.default_rng(4),
    )
    certain_fertility = {key: 1.0 for key in components.fertility}
    draws = simulate_fertility(
        result,
        replace(components, fertility=certain_fertility),
        set(range(1, 7)),
        -2.0,
        np.random.default_rng(5),
    )
    married_men = result.sim_years[
        (result.sim_years["sex"] == "male")
        & (result.sim_years["marital_state"] == "married")
    ]
    expected_keys = set(
        married_men[["person_id", "year"]].itertuples(index=False, name=None)
    )
    paternal_keys = set(
        draws.paternal[["parent_person_id", "birth_year"]].itertuples(
            index=False, name=None
        )
    )
    assert paternal_keys
    assert paternal_keys <= expected_keys


def test_gate2c_modifier_is_weighted_view_not_state_mutation():
    panel = _panel()
    components = _components()
    result = simulate_marital_step(
        panel,
        set(range(1, 7)),
        components,
        _neutral_modifier(),
        SimpleNamespace(
            earn={person: float(person) for person in range(1, 7)},
            cuts=(2.5, 4.5),
        ),
        main_rng=np.random.default_rng(5211),
        gap_rng=np.random.default_rng(9211),
    )

    before = result.panel.person_years[
        ["person_id", "year", "marital_state"]
    ].copy()
    after = result.sim_years[["person_id", "year", "marital_state"]]
    pd.testing.assert_frame_equal(before, after)
    assert list(result.weighted_events.columns) == [
        "person_id",
        "age",
        "sex",
        "weight",
        "transition",
        "tercile",
        "fm_band",
    ]
