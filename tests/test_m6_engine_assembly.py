"""Synthetic integration test for the real fitted-object assembly seam."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from populace_dynamics.data import disability, household_composition
from populace_dynamics.engine import assembly
from populace_dynamics.engine.assembly import (
    M6_DRAW_OUTPUTS_KEY,
    CertifiedEngineInputs,
    assemble_period_modules,
)
from populace_dynamics.engine.composition import CompositionDiagnostics
from populace_dynamics.engine.loop import MaritalStepResult, ProjectionEngine
from populace_dynamics.engine.refit import M6RefitBundle
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


class _InitializingEarnings(_Earnings):
    def materialize_initial_frame(self, frame):
        out = frame.copy()
        out["earnings_state_initialized"] = True
        return out


@dataclass(frozen=True)
class _FamilyFit:
    name: str
    spousal_age_gaps: dict


@dataclass(frozen=True)
class _HouseholdFit:
    family_transitions: object
    male_gap: float


def _minimal_refit_bundle(earnings):
    family = _FamilyFit(
        name="authoritative-cutoff-fit",
        spousal_age_gaps={"male": {0: np.asarray([-4.0, -2.0])}},
    )
    embedded = _FamilyFit(
        name="candidate9-internal-cutoff-fit",
        spousal_age_gaps={},
    )
    household = _HouseholdFit(
        family_transitions=embedded,
        male_gap=-2.0,
    )
    return M6RefitBundle(
        boundary_year=2014,
        family=SimpleNamespace(fitted=family),
        household=SimpleNamespace(fitted=household),
        earnings=earnings,
        modifier=SimpleNamespace(modifier=object(), axis=object()),
        disability=object(),
        claiming_pmfs={("female", 2014): {62: 1.0}},
    )


def _minimal_bundle_binding_kwargs():
    mortality = AgeSexMortalityModel(
        bands=((0, 120),),
        probability={
            ("0+", "female"): 0.0,
            ("0+", "male"): 0.0,
        },
    )
    panel = disability.DisabilityPanel(
        person_years=pd.DataFrame(),
        pairs=pd.DataFrame(),
    )
    weights = StartWaveWeightSnapshot.from_frame(
        pd.DataFrame({"person_id": [1], "weight": [1.0]}),
        boundary_period=2014,
    )
    return {
        "mortality": mortality,
        "marital_panel_builder": lambda frame, context: (object(), {1}),
        "household_panel_builder": lambda frame, context: (object(), {1}),
        "disability_panel": panel,
        "disability_ids": {1},
        "start_weights": weights,
    }


def test_refit_bundle_uses_default_earnings_and_its_initializer():
    default = _InitializingEarnings()
    bundle = _minimal_refit_bundle(SimpleNamespace(generator=default))

    inputs = CertifiedEngineInputs.from_refit_bundle(
        bundle, **_minimal_bundle_binding_kwargs()
    )
    original_household = bundle.household.fitted
    assert inputs.household is not original_household
    assert inputs.household.family_transitions is bundle.family.fitted
    assert original_household.family_transitions is not bundle.family.fitted
    assert inputs.male_gap == -3.0
    modules = assemble_period_modules(inputs)
    frame = pd.DataFrame({"person_id": [1], "year": [2014]})

    assert inputs.earnings is default
    initialized = (
        ProjectionEngine(modules)
        .project(frame, end_year=2014, draw_index=0)
        .slices[0]
    )
    assert initialized["earnings_state_initialized"].tolist() == [True]


def test_refit_bundle_external_earnings_overrides_default_and_initializer():
    default = _InitializingEarnings()
    external = _Earnings()
    bundle = _minimal_refit_bundle(SimpleNamespace(generator=default))

    inputs = CertifiedEngineInputs.from_refit_bundle(
        bundle,
        earnings=external,
        **_minimal_bundle_binding_kwargs(),
    )
    modules = assemble_period_modules(inputs)
    frame = pd.DataFrame({"person_id": [1], "year": [2014]})

    assert inputs.earnings is external
    initialized = (
        ProjectionEngine(modules)
        .project(frame, end_year=2014, draw_index=0)
        .slices[0]
    )
    pd.testing.assert_frame_equal(initialized, frame)


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

    def fake_fertility(
        frame,
        context,
        marital,
        rng,
        *,
        birth_store,
        roster_absent_births,
        **kwargs,
    ):
        del marital, rng, kwargs
        calls.append("fertility")
        birth_store[context.year] = FertilityDraws(
            maternal=empty_births,
            paternal=pd.DataFrame(
                {"parent_person_id": [1], "birth_year": [2015]}
            ),
        )
        roster_absent_births[context.year] = {
            "dropped_parent_ids": frozenset({100 + context.draw_index}),
            "dropped_count": 1,
        }
        return frame

    def fake_household_fertility(marital, components, ids, male_gap, rng):
        del marital, components, ids, male_gap, rng
        calls.append("household_fertility")
        return FertilityDraws(
            maternal=empty_births,
            paternal=pd.DataFrame(
                {"parent_person_id": [1], "birth_year": [2015]}
            ),
        )

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
    monkeypatch.setattr(
        assembly, "simulate_fertility", fake_household_fertility
    )
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
    engine = ProjectionEngine(assemble_period_modules(inputs))
    draw0_outputs = {}
    result = engine.project(
        initial,
        end_year=2016,
        draw_index=0,
        metadata={
            "nawi_by_year": {2015: 50_000.0, 2016: 51_000.0},
            "wage_base_by_year": {2015: 120_000.0, 2016: 121_000.0},
            M6_DRAW_OUTPUTS_KEY: draw0_outputs,
        },
    )
    assert calls == [
        "marital",
        "fertility",
        "disability",
        "household_fertility",
        "composition",
        "fertility",
    ]
    final = result.slices[-1]
    assert final.loc[0, "coresident_spouse"]
    assert final.loc[0, "hh_size"] == 2
    assert final.loc[0, "marital_state"] == "married"
    assert 1_000 <= final.loc[0, "earnings"] <= 2_000
    assert final.loc[0, "claim_age"] == 62
    assert final.loc[0, "claimed"]

    # A module assembly may be reused, but every projection invocation must
    # rebuild the once-per-draw cores and publish them to that draw's collector.
    draw1_outputs = {}
    engine.project(
        initial,
        end_year=2016,
        draw_index=1,
        metadata={
            "nawi_by_year": {2015: 50_000.0, 2016: 51_000.0},
            "wage_base_by_year": {2015: 120_000.0, 2016: 121_000.0},
            M6_DRAW_OUTPUTS_KEY: draw1_outputs,
        },
    )
    assert (
        calls
        == [
            "marital",
            "fertility",
            "disability",
            "household_fertility",
            "composition",
            "fertility",
        ]
        * 2
    )
    assert draw0_outputs["marital"] is not draw1_outputs["marital"]
    assert draw0_outputs["roster_absent_births"] == {
        2015: {
            "dropped_parent_ids": frozenset({100}),
            "dropped_count": 1,
        },
        2016: {
            "dropped_parent_ids": frozenset({100}),
            "dropped_count": 1,
        },
    }
    assert draw1_outputs["roster_absent_births"][2015] == {
        "dropped_parent_ids": frozenset({101}),
        "dropped_count": 1,
    }
    assert draw0_outputs["disability"] is not draw1_outputs["disability"]
    assert draw0_outputs["household"] is not draw1_outputs["household"]
