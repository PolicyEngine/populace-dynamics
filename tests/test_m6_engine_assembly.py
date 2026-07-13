"""Synthetic integration test for the real fitted-object assembly seam."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from populace_dynamics.data import disability, household_composition
from populace_dynamics.engine import assembly
from populace_dynamics.engine.assembly import (
    CertifiedEngineInputs,
    assemble_period_modules,
)
from populace_dynamics.engine.composition import CompositionDiagnostics
from populace_dynamics.engine.loop import MaritalStepResult, ProjectionEngine
from populace_dynamics.engine.steps import (
    AgeSexMortalityModel,
    ClaimingSchedule,
    FertilityDraws,
)
from populace_dynamics.engine.support import StartWaveWeightSnapshot


class _Earnings:
    def generate(self, frame, year, rng):
        del year
        return rng.uniform(1_000, 2_000, len(frame))


def test_assembly_wires_refitted_objects_and_step4_births(monkeypatch):
    calls = []
    family = SimpleNamespace(name="same-cutoff-fit")
    household_fit = SimpleNamespace(
        family_transitions=family,
        male_gap=-2.0,
    )
    empty_births = pd.DataFrame(
        {
            "parent_person_id": pd.Series(dtype="int64"),
            "birth_year": pd.Series(dtype="Int64"),
        }
    )
    native_marital = SimpleNamespace(
        attrs=pd.DataFrame({"person_id": [1]}),
        person_years=pd.DataFrame(),
    )

    def fake_marital(panel, ids, fitted, modifier, axis, **rngs):
        del panel, fitted, modifier, axis, rngs
        calls.append("marital")
        years = pd.DataFrame(
            {
                "person_id": [1],
                "year": [2015],
                "marital_state": ["married"],
            }
        )
        return MaritalStepResult(
            sim_years=years,
            births=empty_births,
            panel=native_marital,
        )

    def fake_fertility(frame, context, marital, rng, *, birth_store, **kwargs):
        del marital, rng, kwargs
        calls.append("fertility")
        birth_store[context.year] = FertilityDraws(
            maternal=empty_births,
            paternal=pd.DataFrame(
                {"parent_person_id": [1], "birth_year": [2015]}
            ),
        )
        return frame

    def fake_disability(
        panel,
        model,
        ids,
        rng,
        *,
        start_weights,
        rng_by_period,
    ):
        del model, ids, rng, start_weights, rng_by_period
        calls.append("disability")
        assert panel.person_years["period"].tolist() == [2015]
        person_years = pd.DataFrame(
            {
                "person_id": [1],
                "period": [2015],
                "age": [61],
                "sex": ["female"],
                "weight": [10.0],
                "status_code": [1],
                "disabled": [False],
                "retired": [False],
                "di_converted": [True],
            }
        )
        return disability.DisabilityPanel(
            person_years, disability.build_transition_pairs(person_years)
        )

    def fake_composition(hh, fitted, ids, marital, rngs, *, fertility):
        del hh, fitted, ids, marital, rngs
        calls.append("composition")
        assert fertility.paternal["parent_person_id"].tolist() == [1]
        person_waves = pd.DataFrame(
            {
                "person_id": [1],
                "year": [2015],
                "coresident_spouse": [True],
                "coresident_parent": [False],
                "coresident_child": [False],
                "coresident_grandchild": [False],
                "multigen": [False],
                "hh_size": [2],
            }
        )
        panel = household_composition.HouseholdCompositionPanel(
            person_waves, pd.DataFrame({"person_id": [1]})
        )
        diagnostics = CompositionDiagnostics(
            weight=np.ones(1),
            legal_core=np.ones(1, dtype=bool),
            cohabitation_state=np.zeros(1, dtype=bool),
            cohabitation_increment=np.zeros(1, dtype=bool),
            legal_residual_state=np.zeros(1, dtype=bool),
            legal_residual_increment=np.zeros(1, dtype=bool),
            final_spouse=np.ones(1, dtype=bool),
            coresident_parent=np.zeros(1, dtype=bool),
            multigen=np.zeros(1, dtype=bool),
            coresident_child=np.zeros(1, dtype=bool),
            coresident_grandchild=np.zeros(1, dtype=bool),
            household_size=np.array([2]),
            model_diagnostics={},
        )
        return panel, diagnostics

    monkeypatch.setattr(assembly, "simulate_marital_step", fake_marital)
    monkeypatch.setattr(assembly, "apply_fertility", fake_fertility)
    monkeypatch.setattr(assembly, "simulate_reproduction", fake_disability)
    monkeypatch.setattr(
        assembly, "simulate_candidate9_injected", fake_composition
    )

    mortality = AgeSexMortalityModel(
        bands=((0, 120),),
        probability={("0+", "female"): 0.0, ("0+", "male"): 0.0},
    )
    disability_panel = disability.DisabilityPanel(
        person_years=pd.DataFrame(
            {
                "person_id": [1, 1],
                "period": [2015, 2017],
                "age": [61, 63],
                "sex": ["female", "female"],
                "weight": [99.0, 99.0],
                "status_code": [1, 1],
                "disabled": [False, False],
                "retired": [False, False],
            }
        ),
        pairs=pd.DataFrame(),
    )
    inputs = CertifiedEngineInputs(
        family=family,
        modifier=object(),
        permanent_axis=object(),
        household=household_fit,
        mortality=mortality,
        disability=object(),
        claiming=ClaimingSchedule({("female", 2014): {62: 1.0}}),
        earnings=_Earnings(),
        marital_panel_builder=lambda frame, context: (object(), {1}),
        household_panel_builder=lambda frame, context: (object(), {1}),
        disability_panel=disability_panel,
        disability_ids={1},
        start_weights=StartWaveWeightSnapshot.from_frame(
            pd.DataFrame({"person_id": [1], "weight": [10.0]}),
            boundary_period=2014,
        ),
        male_gap=-2.0,
    )
    initial = pd.DataFrame(
        {
            "person_id": [1],
            "year": [2014],
            "age": [60],
            "sex": ["female"],
            "weight": [10.0],
        }
    )
    result = ProjectionEngine(assemble_period_modules(inputs)).project(
        initial,
        end_year=2015,
        draw_index=0,
        metadata={
            "nawi_by_year": {2015: 50_000.0},
            "wage_base_by_year": {2015: 120_000.0},
        },
    )
    assert calls == ["marital", "fertility", "disability", "composition"]
    final = result.slices[-1]
    assert final.loc[0, "coresident_spouse"]
    assert final.loc[0, "hh_size"] == 2
    assert final.loc[0, "marital_state"] == "married"
    assert 1_000 <= final.loc[0, "earnings"] <= 2_000
    assert final.loc[0, "claim_age"] == 62
    assert final.loc[0, "claimed"]
