"""Synthetic unit coverage for M6 candidate-9 state injection."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.engine.composition import (
    RECERTIFICATION_CHANNEL_SETS,
    CompositionDiagnostics,
    CompositionRngs,
    check_candidate9_recertification,
    composition_rngs_from_registry,
    simulate_candidate9_injected,
    simulate_candidate9_internal_reference,
)
from populace_dynamics.engine.marital import (
    MaritalStepResult,
    simulate_marital_step,
)
from populace_dynamics.engine.rng import ProjectionRNGRegistry
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models.family_transitions.components.initial_states import (
    SupportComposition,
)
from tests.test_m6_engine_marital import _components, _neutral_modifier


class _NoExit:
    def predict(self, age, is_male):
        del is_male
        return np.zeros(len(age))


def _household_panel(n_people: int) -> hc.HouseholdCompositionPanel:
    rows = []
    for person_id in range(1, n_people + 1):
        for year in range(2015, 2020):
            rows.append(
                {
                    "person_id": person_id,
                    "year": year,
                    "age": 30 + year - 2015,
                    "band": "25-34",
                    "sex": "male",
                    "weight": 1.0 + (person_id % 7) / 10,
                    "hh_size": 1,
                    "coresident_spouse": False,
                    "coresident_parent": False,
                    "coresident_child": False,
                    "coresident_grandchild": False,
                    "multigen": False,
                }
            )
    person_waves = hc._add_transitions(pd.DataFrame(rows))
    attrs = pd.DataFrame({"person_id": range(1, n_people + 1)})
    return hc.HouseholdCompositionPanel(person_waves, attrs)


def _fitted(family_transitions=None):
    key = ("25-34", "male")
    fertility = {
        (age, parity, 1980): 0.16
        for age in range(15, 50)
        for parity in range(4)
    }
    nonfamily = {
        size: (np.array([0], dtype=np.int64), np.array([1.0]))
        for size in range(1, 6)
    }
    return SimpleNamespace(
        family_transitions=(
            SimpleNamespace(fertility=fertility)
            if family_transitions is None
            else family_transitions
        ),
        male_gap=-2.0,
        parental_exit=_NoExit(),
        child_exit_single_year={},
        multigen_entry={key: 0.0},
        multigen_exit={key: 0.0},
        coupling_child_given_multigen={},
        coupling_child_pooled={},
        cohab_flag=pd.DataFrame(
            {
                "person_id": pd.Series(dtype="int64"),
                "year": pd.Series(dtype="int64"),
                "cohabiting": pd.Series(dtype="bool"),
            }
        ),
        cohab_entry={key: 0.22},
        cohab_exit={key: 0.35},
        cohab_entry_age={},
        cohab_exit_age={},
        cohab_entry_age_female={},
        cohab_exit_age_female={},
        cohab_overlay_lift=0.0,
        legal_residual_entry={key: 0.08},
        legal_residual_exit={key: 0.50},
        legal_residual_marginal={key: 0.10},
        legal_residual_target={key: 0.10},
        father_links=pd.DataFrame(
            {
                "parent_person_id": pd.Series(dtype="int64"),
                "birth_year": pd.Series(dtype="int64"),
            }
        ),
        custodial_era={},
        custodial_age_marital={},
        custodial_band_marital={},
        custodial_overall=0.0,
        custodial_child_record={},
        joinable_keys=frozenset(),
        linked_episode_persistence=0.0,
        skipgen_entry={key: 0.0},
        skipgen_exit={key: 0.0},
        skipgen_entry_age={},
        skipgen_exit_age={},
        nonfamily_count_by_core=nonfamily,
        parent_count_two_share={key: 0.0},
        parent_count_two_pooled=0.0,
        completed_size_dist_train={},
        completed_size_dist_train_all={},
        retention_link_shift={},
    )


def _marital_result(
    n_people: int, rng: np.random.Generator
) -> MaritalStepResult:
    attrs = pd.DataFrame(
        {
            "person_id": range(1, n_people + 1),
            "birth_year": [1985] * n_people,
            "sex": ["male"] * n_people,
            "weight": [1.0] * n_people,
            "start_exposure_year": [2015] * n_people,
            "censor_year": [2019] * n_people,
            "n_marriages": [0.0] * n_people,
        }
    )
    rows = []
    for person_id in range(1, n_people + 1):
        for year in range(2015, 2020):
            rows.append(
                {
                    "person_id": person_id,
                    "year": year,
                    "age": year - 1985,
                    "sex": "male",
                    "weight": 1.0,
                    "marital_state": (
                        "married" if rng.random() < 0.56 else "never_married"
                    ),
                    "marriage_duration": pd.NA,
                    "years_since_dissolution": pd.NA,
                }
            )
    sim_years = pd.DataFrame(rows)
    empty_events = pd.DataFrame(
        columns=[
            "person_id",
            "year",
            "age",
            "sex",
            "weight",
            "transition",
        ]
    )
    panel = transitions.MaritalPanel(sim_years, empty_events, attrs)
    births = pd.DataFrame(
        {
            "parent_person_id": pd.Series(dtype="int64"),
            "birth_year": pd.Series(dtype="Int64"),
            "birth_order": pd.Series(dtype="Int64"),
            "record_type": pd.Series(dtype="string"),
            "is_event": pd.Series(dtype="bool"),
        }
    )
    return MaritalStepResult(
        panel=panel,
        sim_years=sim_years,
        births=births,
        weighted_events=pd.DataFrame(),
        exposure=pd.DataFrame(),
        modifier_check={"constraint_holds": True},
    )


def _rngs(seed: int) -> CompositionRngs:
    streams = np.random.SeedSequence(seed).spawn(13)
    generators = [np.random.default_rng(stream) for stream in streams]
    return CompositionRngs(*generators)


def test_period_zero_composition_streams_match_candidate9_tags():
    actual = composition_rngs_from_registry(
        ProjectionRNGRegistry(draw_index=0, n_periods=1), 0
    )
    c2_child, c2_cohab = np.random.SeedSequence([5200, 0xC2]).spawn(2)
    c5_coupling, c5_parent = np.random.SeedSequence([5200, 0xC5]).spawn(2)
    assert np.array_equal(
        actual.occupancy.random(5),
        np.random.default_rng(np.random.SeedSequence([5200, 0xB2B])).random(5),
    )
    assert np.array_equal(
        actual.child.random(5), np.random.default_rng(c2_child).random(5)
    )
    assert np.array_equal(
        actual.cohabitation.random(5),
        np.random.default_rng(c2_cohab).random(5),
    )
    assert np.array_equal(
        actual.multigenerational_coupling.random(5),
        np.random.default_rng(c5_coupling).random(5),
    )
    assert np.array_equal(
        actual.parent_count.random(5),
        np.random.default_rng(c5_parent).random(5),
    )
    assert np.array_equal(
        actual.linked_episode.random(5),
        np.random.default_rng(np.random.SeedSequence([5200, 0xC7])).random(5),
    )


def test_candidate9_injection_bypasses_both_internal_marital_runs(monkeypatch):
    n_people = 80
    hh = _household_panel(n_people)
    marital = _marital_result(n_people, np.random.default_rng(5100))

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("candidate-9 attempted an internal ft.simulate")

    monkeypatch.setattr(ft, "simulate", forbidden)
    panel, diagnostics = simulate_candidate9_injected(
        hh,
        _fitted(),
        set(range(1, n_people + 1)),
        marital,
        _rngs(5200),
    )

    assert len(panel.person_waves) == n_people * 5
    assert diagnostics.legal_core.any()
    assert diagnostics.cohabitation_increment.any()
    assert diagnostics.legal_residual_increment.any()
    assert diagnostics.coresident_child.any()
    assert np.array_equal(
        panel.person_waves["hh_size"].to_numpy(),
        diagnostics.household_size,
    )


def test_targeted_recertification_covers_every_marital_consuming_channel():
    n_people = 600
    hh = _household_panel(n_people)
    family = _components()
    support = SupportComposition(
        pd.Series(0, index=pd.Index(range(1, n_people + 1), name="person_id"))
    )
    family = replace(
        family,
        initial_states=replace(family.initial_states, support=support),
    )
    fitted = _fitted(family)
    source = _marital_result(n_people, np.random.default_rng(4100)).panel
    ids = set(range(1, n_people + 1))
    axis = SimpleNamespace(
        earn={person_id: float(person_id) for person_id in ids},
        cuts=(n_people / 3, 2 * n_people / 3),
    )
    internal: list[CompositionDiagnostics] = []
    injected: list[CompositionDiagnostics] = []
    for draw_seed in range(5200, 5220):
        _panel_a, diag_a = simulate_candidate9_internal_reference(
            hh,
            source,
            fitted,
            ids,
            draw_seed,
        )
        marital = simulate_marital_step(
            source,
            ids,
            family,
            _neutral_modifier(),
            axis,
            main_rng=np.random.default_rng([draw_seed, 0xA1]),
            gap_rng=np.random.default_rng([draw_seed, 0xA2]),
        )
        _panel_b, diag_b = simulate_candidate9_injected(
            hh,
            fitted,
            ids,
            marital,
            _rngs(draw_seed + 20_000),
        )
        internal.append(diag_a)
        injected.append(diag_b)

    result = check_candidate9_recertification(injected, internal)
    assert result.passed is True
    assert {cell.channel_set for cell in result.cells} == set(
        RECERTIFICATION_CHANNEL_SETS
    )
    assert {cell.cell for cell in result.cells} == {
        cell
        for cells in RECERTIFICATION_CHANNEL_SETS.values()
        for cell in cells
    }
    assert all(
        cell.tolerance == 3.0 * cell.sigma_of_mean_difference
        for cell in result.cells
    )


def test_targeted_recertification_fails_loudly_on_a_channel_shift():
    n = 100
    base = CompositionDiagnostics(
        weight=np.ones(n),
        legal_core=np.zeros(n, dtype=bool),
        cohabitation_state=np.zeros(n, dtype=bool),
        cohabitation_increment=np.zeros(n, dtype=bool),
        legal_residual_state=np.zeros(n, dtype=bool),
        legal_residual_increment=np.zeros(n, dtype=bool),
        final_spouse=np.zeros(n, dtype=bool),
        coresident_parent=np.zeros(n, dtype=bool),
        multigen=np.zeros(n, dtype=bool),
        coresident_child=np.zeros(n, dtype=bool),
        coresident_grandchild=np.zeros(n, dtype=bool),
        household_size=np.ones(n, dtype=np.int64),
        model_diagnostics={},
    )
    shifted = replace(
        base,
        cohabitation_increment=np.ones(n, dtype=bool),
    )
    with pytest.raises(AssertionError, match="fuller re-ceremony required"):
        check_candidate9_recertification([shifted, shifted], [base, base])
