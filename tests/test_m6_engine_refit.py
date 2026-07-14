"""Synthetic unit tests for M6's <=T* refit boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import populace_dynamics.engine.refit as refit_module
from populace_dynamics import claiming
from populace_dynamics.data import disability, transitions
from populace_dynamics.data import household_composition as hc_data
from populace_dynamics.engine.refit import (
    M6RefitBundle,
    M6RefitInputs,
    claiming_pmfs_from_reference,
    fit_mortality_model,
    prepare_m6_preflight_context,
    prepare_mortality_refit_inputs,
    refit_disability,
    refit_earnings_chained_generator,
    refit_family_transitions,
    refit_first_marriage_modifier,
    refit_household_composition,
    refit_m6_components,
    truncate_estimation_frame,
)
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition as hc
from populace_dynamics.ss.params import SSAParameters


class _RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def fit(self, spec, context):
        self.calls.append((spec, context))
        return {"new_fit": len(self.calls)}


class _RecordingQRF:
    def __init__(self, owner: _RecordingQRFFactory, seed: int) -> None:
        self.owner = owner
        self.seed = seed

    def fit(self, frame, **kwargs):
        call = {
            "seed": self.seed,
            "frame": frame.copy(),
            "kwargs": kwargs,
        }
        self.owner.fit_calls.append(call)
        return call


class _RecordingQRFFactory:
    def __init__(self) -> None:
        self.factory_seeds: list[int] = []
        self.fit_calls: list[dict[str, Any]] = []

    def __call__(self, *, seed: int):
        self.factory_seeds.append(seed)
        return _RecordingQRF(self, seed)


def _marital_panel() -> transitions.MaritalPanel:
    person_years = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "year": [2012, 2014, 2015, 2013, 2015],
            "required_interview_year": [2013, 2015, 2015, 2013, 2015],
            "age": [30, 32, 33, 40, 42],
            "sex": ["female", "female", "female", "male", "male"],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0],
            "marital_state": [
                "never_married",
                "never_married",
                "married",
                "married",
                "married",
            ],
            "marriage_duration": pd.array(
                [pd.NA, pd.NA, 0, 5, 7], dtype="Int64"
            ),
            "years_since_dissolution": pd.array([pd.NA] * 5, dtype="Int64"),
        }
    )
    events = pd.DataFrame(
        {
            "person_id": [1, 2],
            "year": [2014, 2013],
            "required_interview_year": [2015, 2013],
            "age": [32, 40],
            "sex": ["female", "male"],
            "weight": [1.0, 2.0],
            "transition": ["first_marriage", "divorce"],
            "marriage_duration": pd.array([pd.NA, 5], dtype="Int64"),
            "years_since_dissolution": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "origin": pd.array([pd.NA, pd.NA], dtype="string"),
        }
    )
    attrs = pd.DataFrame(
        {
            "person_id": [1, 2],
            "birth_year": [1982.0, 1973.0],
            "censor_year": [2023.0, 2023.0],
            "weight": [1.0, 2.0],
            "n_marriages": [2.0, 1.0],
        }
    )
    return transitions.MaritalPanel(person_years, events, attrs)


def _marriage_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "sex": ["female", "female", "male"],
            "birth_year": pd.array([1982, 1982, 1973], dtype="Int64"),
            "start_year": pd.array([2012, 2018, 2010], dtype="Int64"),
            "required_interview_year": pd.array(
                [2013, 2019, 2011], dtype="Int64"
            ),
            "end_year": pd.array([2018, pd.NA, 2016], dtype="Int64"),
            "separation_year": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "most_recent_report_year": pd.array(
                [2023, 2023, 2023], dtype="Int64"
            ),
            "n_marriages": pd.array([2, 2, 1], dtype="Int64"),
            "is_marriage": [True, True, True],
            "spouse_person_id": pd.array([2, 2, 1], dtype="Int64"),
        }
    )


def _birth_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_person_id": [1, 1],
            "birth_year": pd.array([2010, 2017], dtype="Int64"),
            "required_interview_year": pd.array([2011, 2017], dtype="Int64"),
            "is_event": [True, True],
        }
    )


def _family_context() -> ft.FitContext:
    return ft.FitContext(
        panel=_marital_panel(),
        demographic_panel=pd.DataFrame(
            {
                "person_id": [1, 1, 2, 2],
                "period": [2013, 2015, 2013, 2015],
            }
        ),
        marriage_records=_marriage_records(),
        birth_records=_birth_records(),
        marriage_order_map=pd.DataFrame(
            {
                "person_id": [1, 1, 2],
                "start_year": [2012, 2018, 2010],
                "order": [1, 2, 1],
            }
        ),
        train_ids=frozenset({1, 2}),
    )


def test_conservative_biennial_boundary_excludes_2014_event_seen_in_2015():
    frame = pd.DataFrame(
        {
            "event_year": [2013, 2014, 2015],
            "required_interview_year": [2013, 2015, 2015],
            "value": [1, 2, 3],
        }
    )
    got = truncate_estimation_frame(
        frame,
        boundary_year=2014,
        year_column="event_year",
        flow=True,
    )
    assert got["value"].tolist() == [1]

    with pytest.raises(ValueError, match="required_interview_year"):
        truncate_estimation_frame(
            frame.drop(columns="required_interview_year"),
            boundary_year=2014,
            year_column="event_year",
            flow=True,
        )


def test_family_registry_keeps_spec_hash_and_receives_only_truncated_context():
    registry = _RecordingRegistry()
    result = refit_family_transitions(
        _family_context(), boundary_year=2014, registry=registry
    )
    spec, context = registry.calls[0]

    assert spec is ft.CANDIDATE_16
    assert result.spec_sha256 == ft.CANDIDATE_16.sha256
    assert result.spec_sha256 == (
        "6d4d2b2beadc87d17404a3deb64a272c2456d7471b3ad6f1cef779d807765aa1"
    )
    assert context.panel.person_years["year"].max() == 2013
    assert context.panel.events["year"].tolist() == [2013]
    assert context.demographic_panel["period"].max() == 2013
    assert context.marriage_records["start_year"].max() == 2012
    assert context.birth_records["birth_year"].max() == 2010
    assert context.panel.attrs["censor_year"].max() == 2014
    assert not result.provenance.certified_full_window_artifacts_read
    assert not result.provenance.certified_full_window_artifacts_written


def _household_context() -> hc.FitContext:
    person_waves = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "year": [2011, 2013, 2015, 2013, 2015],
            "age": [29, 31, 33, 40, 42],
            "band": ["25-34", "25-34", "25-34", "35-44", "35-44"],
            "sex": ["female", "female", "female", "male", "male"],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0],
            "hh_size": [2, 2, 3, 1, 2],
            "coresident_spouse": [False, True, True, False, True],
            "coresident_parent": [True, False, False, False, False],
            "coresident_child": [False, False, True, False, False],
            "coresident_grandchild": [False] * 5,
            "multigen": [False, False, True, False, False],
            "has_next": [True, True, False, True, False],
            "next_coresident_parent": [False, False, np.nan, False, np.nan],
            "next_coresident_spouse": [True, True, np.nan, True, np.nan],
            "next_multigen": [False, True, np.nan, False, np.nan],
        }
    )
    panel = hc_data.HouseholdCompositionPanel(
        person_waves=person_waves,
        attrs=pd.DataFrame({"person_id": [1, 2]}),
    )
    year_frame = lambda name: pd.DataFrame(  # noqa: E731
        {"person_id": [1, 1], "year": [2013, 2015], name: [1, 2]}
    )
    return hc.FitContext(
        hh=panel,
        mpanel=_marital_panel(),
        demographic_panel=_family_context().demographic_panel,
        marriage_records=_marriage_records(),
        birth_records=_birth_records(),
        marriage_order_map=_family_context().marriage_order_map,
        relationship_map=pd.DataFrame(
            {
                "interview_year": [2013, 2015],
                "ego_rel_to_alter": [10, 10],
            }
        ),
        train_ids=frozenset({1, 2}),
        father_links_child=pd.DataFrame(
            {
                "parent_person_id": [1, 1],
                "child_person_id": [10, 11],
                "birth_year": [2010, 2017],
            }
        ),
        parent_pairs=year_frame("child_person_id"),
        marital_by_year=year_frame("marital"),
        family_unit_sizes=year_frame("family_unit_size"),
        legal_spouse_flag=year_frame("legal_spouse_obs"),
        child_record_exposure=year_frame("child_age"),
        parent_counts=year_frame("n_parent_links"),
    )


def test_household_registry_rebuilds_next_wave_after_cut_and_keeps_spec():
    registry = _RecordingRegistry()
    result = refit_household_composition(
        _household_context(), boundary_year=2014, registry=registry
    )
    spec, context = registry.calls[0]

    assert spec is hc.CANDIDATE_9
    assert result.spec_sha256 == hc.CANDIDATE_9.sha256
    assert result.spec_sha256 == (
        "6137d921032c49ccd71c2302418759868689dac731706f1093a21716080800ab"
    )
    waves = context.hh.person_waves
    assert waves["year"].max() == 2013
    person_one = waves[waves["person_id"] == 1]
    assert person_one["has_next"].tolist() == [True, False]
    assert person_one["next_coresident_spouse"].tolist() == [True, np.nan]
    assert context.relationship_map["interview_year"].tolist() == [2013]
    assert context.parent_pairs["year"].tolist() == [2013]


def test_preflight_context_is_the_exact_holdout_blind_household_input():
    context = _household_context()
    inputs = SimpleNamespace(household_context=context)

    prepared = prepare_m6_preflight_context(inputs)  # type: ignore[arg-type]

    assert prepared.hh.person_waves["year"].max() == 2013
    assert prepared.mpanel.person_years["year"].max() <= 2014
    assert prepared.demographic_panel["period"].max() == 2013
    assert prepared.relationship_map["interview_year"].max() == 2013
    assert prepared.train_ids == frozenset({1, 2})


def _earnings_panel() -> pd.DataFrame:
    rows = []
    for bin_index in range(8):
        for offset in range(3):
            person_id = 10 * bin_index + offset + 1
            for period in range(2004, 2017, 2):
                earnings = float(
                    1_000 * (bin_index + 1)
                    + 50 * offset
                    + 10 * (period - 2004)
                )
                if offset == 0 and period == 2014:
                    earnings = 0.0
                if period == 2016:
                    earnings = 999_999.0
                rows.append(
                    {
                        "person_id": person_id,
                        "period": period,
                        "earnings": earnings,
                        "age": 27 + 5 * bin_index,
                        "weight": float(offset + 1),
                    }
                )
    return pd.DataFrame(rows)


def _earnings_nawi() -> dict[int, float]:
    return {year: 100.0 * 1.02 ** (year - 2004) for year in range(2004, 2017)}


def test_earnings_refit_fits_both_qrf_gates_on_only_preboundary_pairs():
    factory = _RecordingQRFFactory()
    result = refit_earnings_chained_generator(
        _earnings_panel(),
        _earnings_nawi(),
        seed=7,
        boundary_year=2014,
        qrf_factory=factory,
    )

    assert factory.factory_seeds == [7, 7]
    assert len(factory.fit_calls) == 2
    shared, zero = factory.fit_calls
    assert shared["frame"]["period_tp2"].max() == 2014
    assert zero["frame"]["person_id"].unique().tolist() == [
        1,
        11,
        21,
        31,
        41,
        51,
        61,
        71,
    ]
    assert result.anchors.set_index("person_id").loc[1, "period"] == 2014
    assert result.anchors.set_index("person_id").loc[1, "earnings"] == 0
    assert result.estimation_panel["period"].max() == 2014
    assert result.n_zero_anchor_pairs == 40
    for call in factory.fit_calls:
        assert call["kwargs"] == {
            "predictors": ["earnings", "age_tp2"],
            "targets": ["earnings_tp2"],
            "weights": "weight_tp2",
        }


def test_disability_refit_rebuilds_pairs_after_interview_aware_cut():
    person_years = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "period": [2011, 2013, 2014],
            "required_interview_year": [2011, 2013, 2015],
            "age": [40, 42, 43],
            "sex": ["female"] * 3,
            "weight": [1.0] * 3,
            "status_code": [1, 5, 5],
            "disabled": [False, True, True],
            "retired": [False, False, False],
        }
    )
    panel = disability.DisabilityPanel(
        person_years=person_years,
        pairs=disability.build_transition_pairs(person_years),
    )
    seen = {}

    def fitter(truncated, ids):
        seen["panel"] = truncated
        seen["ids"] = ids
        return "m4-refit"

    assert refit_disability(panel, fitter=fitter) == "m4-refit"
    assert seen["panel"].person_years["period"].tolist() == [2011, 2013]
    assert len(seen["panel"].pairs) == 1
    assert seen["ids"] == {1}


def _claim_reference(supplement_year: int) -> claiming.ClaimAgeReference:
    categories = {
        "age62": 40.0,
        "age63": 10.0,
        "age64": 10.0,
        "age65": 10.0,
        "age66": 10.0,
        "disability_conversion": 10.0,
        "age67_69": 6.0,
        "age70plus": 4.0,
    }
    row = {
        "number_thousands": 1,
        "average_age": 64.0,
        "raw": {},
        "categories": categories,
        "fra_at": {"share": 0.0, "at_age": 66},
    }
    return claiming.ClaimAgeReference(
        schema_version="ssa_claim_ages.v1",
        table="synthetic",
        supplement_year=supplement_year,
        raw_columns=(),
        collapsed_categories=tuple(categories),
        provenance={},
        validation={},
        fra_schedule={},
        _data={"female": {"2014": row}, "male": {"2014": row}},
    )


def test_claiming_and_mortality_reject_post_boundary_external_vintages():
    with pytest.raises(ValueError, match="post-T"):
        claiming_pmfs_from_reference(_claim_reference(2023))
    pmfs = claiming_pmfs_from_reference(_claim_reference(2014))
    assert set(pmfs) == {("female", 2014), ("male", 2014)}
    assert sum(pmfs[("female", 2014)].values()) == pytest.approx(1.0)

    exposure = pd.DataFrame(
        {
            "event_year": [2013, 2014],
            "required_interview_year": [2013, 2015],
            "death": [0, 1],
        }
    )
    rates = pd.DataFrame({"age": [40], "qx": [0.01]})
    with pytest.raises(ValueError, match="post-T"):
        prepare_mortality_refit_inputs(
            exposure, rates, external_vintage_year=2023
        )
    prepared = prepare_mortality_refit_inputs(
        exposure, rates, external_vintage_year=2014
    )
    assert prepared.exposure["event_year"].tolist() == [2013]


def test_mortality_refit_applies_psid_band_ratio_to_admissible_external_rates():
    exposure = pd.DataFrame(
        {
            "event_year": [2013, 2013, 2013, 2013],
            "required_interview_year": [2013] * 4,
            "age_band": ["0+"] * 4,
            "sex": ["female", "female", "male", "male"],
            "start_weight": [1.0, 1.0, 1.0, 1.0],
            "exposure": [1.0, 1.0, 1.0, 1.0],
            "death": [1.0, 0.0, 1.0, 0.0],
        }
    )
    external = pd.DataFrame(
        {
            "lower_age": [0, 0],
            "upper_age": [120, 120],
            "age_band": ["0+", "0+"],
            "sex": ["female", "male"],
            "central_rate": [0.25, 0.40],
        }
    )
    prepared = prepare_mortality_refit_inputs(
        exposure, external, external_vintage_year=2014
    )
    model = fit_mortality_model(prepared)
    expected = float(-np.expm1(-0.5))
    assert model.probability[("0+", "female")] == pytest.approx(expected)
    assert model.probability[("0+", "male")] == pytest.approx(expected)


def test_complete_refit_entrypoint_invokes_every_composed_object(monkeypatch):
    calls = []
    modifier_calls = []
    family = SimpleNamespace(fitted="family-fit")
    household = SimpleNamespace(fitted="household-fit")

    monkeypatch.setattr(
        refit_module,
        "refit_family_transitions",
        lambda *args, **kwargs: calls.append("family") or family,
    )
    monkeypatch.setattr(
        refit_module,
        "refit_household_composition",
        lambda *args, **kwargs: calls.append("household") or household,
    )
    monkeypatch.setattr(
        refit_module,
        "refit_earnings_chained_generator",
        lambda *args, **kwargs: calls.append("earnings") or "earnings-fit",
    )

    def record_modifier(*args, **kwargs):
        calls.append("modifier")
        modifier_calls.append((args, kwargs))
        return "modifier-fit"

    monkeypatch.setattr(
        refit_module, "refit_first_marriage_modifier", record_modifier
    )
    monkeypatch.setattr(
        refit_module,
        "refit_disability",
        lambda *args, **kwargs: calls.append("disability") or "disability-fit",
    )
    monkeypatch.setattr(
        refit_module,
        "claiming_pmfs_from_reference",
        lambda *args, **kwargs: calls.append("claiming")
        or {("female", 2014): {}},
    )
    monkeypatch.setattr(
        refit_module,
        "prepare_mortality_refit_inputs",
        lambda *args, **kwargs: calls.append("mortality") or "mortality-fit",
    )
    inputs = M6RefitInputs(
        family_context=object(),  # type: ignore[arg-type]
        household_context=object(),  # type: ignore[arg-type]
        earnings_panel=pd.DataFrame(),
        earnings_seed=0,
        modifier_marital_panel=object(),  # type: ignore[arg-type]
        modifier_interview_years=pd.Series([2013, 2015]),
        modifier_marriage_records=pd.DataFrame(),
        modifier_person_weight=pd.Series(dtype=float),
        ssa_params=SimpleNamespace(nawi={}),
        ssa_params_vintage=2014,
        modifier_train_ids=set(),
        disability_panel=object(),  # type: ignore[arg-type]
        disability_train_ids=set(),
        claiming_reference=object(),  # type: ignore[arg-type]
        mortality_exposure=pd.DataFrame(),
        mortality_external_rates=pd.DataFrame(),
        mortality_external_vintage=2014,
    )
    bundle = refit_m6_components(inputs)
    assert calls == [
        "family",
        "household",
        "earnings",
        "modifier",
        "disability",
        "claiming",
        "mortality",
    ]
    assert bundle.family is family
    assert bundle.household is household
    assert bundle.earnings == "earnings-fit"
    assert bundle.modifier == "modifier-fit"
    assert bundle.disability == "disability-fit"
    assert bundle.mortality == "mortality-fit"
    assert modifier_calls[0][1]["interview_years"].tolist() == [2013, 2015]


class _ConstantFirstMarriage:
    def predict(self, age, is_male, decade):
        return np.full(len(age), 0.1, dtype=np.float64)


def _ssa_params() -> SSAParameters:
    years = list(range(2000, 2017))
    return SSAParameters(
        nawi={year: 100.0 + year for year in years},
        wage_base={1900: 1_000_000.0},
        pia_factors=(0.9, 0.32, 0.15),
        fra_months_by_birth_year=[(1900, 804)],
        early_monthly_rates=(0.0, 0.0),
        early_first_bracket_months=36,
        pe_us_revision="synthetic-2014-vintage",
    )


def _modifier_panel() -> transitions.MaritalPanel:
    rows = []
    for person_id, sex in ((1, "female"), (2, "male")):
        for year in range(2006, 2015, 2):
            rows.append(
                {
                    "person_id": person_id,
                    "year": year,
                    "required_interview_year": year,
                    "age": year - 1950,
                    "sex": sex,
                    "weight": 1.0,
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
        {
            "person_id": [1],
            "year": [2012],
            "required_interview_year": [2013],
            "age": [62],
            "sex": ["female"],
            "weight": [1.0],
            "transition": ["first_marriage"],
        }
    )
    attrs = pd.DataFrame(
        {
            "person_id": [1, 2],
            "birth_year": [1950.0, 1950.0],
            "censor_year": [2014.0, 2014.0],
            "weight": [1.0, 1.0],
            "n_marriages": [1.0, 0.0],
        }
    )
    return transitions.MaritalPanel(person_years, events, attrs)


def _modifier_earnings() -> pd.DataFrame:
    rows = []
    for person_id, scale in ((1, 1.0), (2, 2.0)):
        for period in (*range(2006, 2015, 2), 2016):
            rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "earnings": scale * (10_000 + 100 * period),
                    "age": period - 1950,
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_gate2c_modifier_axis_is_refit_from_truncated_earnings():
    family = SimpleNamespace(first_marriage=_ConstantFirstMarriage())
    marriages = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["female", "male"],
            "birth_year": pd.array([1950, 1950], dtype="Int64"),
            "start_year": pd.array([2012, 2010], dtype="Int64"),
            "required_interview_year": pd.array([2013, 2011], dtype="Int64"),
            "end_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "separation_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "most_recent_report_year": pd.array([2014, 2014], dtype="Int64"),
            "n_marriages": pd.array([1, 1], dtype="Int64"),
            "is_marriage": [True, True],
        }
    )
    result = refit_first_marriage_modifier(
        family,  # type: ignore[arg-type]
        _modifier_panel(),
        _modifier_earnings(),
        marriages,
        pd.Series({1: 1.0, 2: 1.0}),
        _ssa_params(),
        params_vintage=2014,
        train_ids={1, 2},
    )
    truncated = _modifier_earnings().query("period <= 2014")
    assert result.provenance.n_rows["earnings"] == len(truncated)
    assert result.provenance.max_year["earnings"] == 2014
    assert set(result.axis.earn) == {1, 2}
    assert result.modifier.constraint_max_abs_dev() < 1e-12

    with pytest.raises(ValueError, match="post-T"):
        refit_first_marriage_modifier(
            family,  # type: ignore[arg-type]
            _modifier_panel(),
            _modifier_earnings(),
            marriages,
            pd.Series({1: 1.0, 2: 1.0}),
            _ssa_params(),
            params_vintage=2015,
            train_ids={1, 2},
        )


def test_gate2c_modifier_accepts_native_panel_with_interview_calendar():
    family = SimpleNamespace(first_marriage=_ConstantFirstMarriage())
    source = _modifier_panel()
    post_boundary_event = source.events.iloc[[0]].copy()
    post_boundary_event["person_id"] = 2
    post_boundary_event["year"] = 2014
    post_boundary_event["age"] = 64
    post_boundary_event["sex"] = "male"
    post_boundary_event["required_interview_year"] = 2015
    native_panel = transitions.MaritalPanel(
        person_years=source.person_years.drop(
            columns="required_interview_year"
        ),
        events=pd.concat(
            [source.events, post_boundary_event], ignore_index=True
        ).drop(columns="required_interview_year"),
        attrs=source.attrs,
    )
    marriages = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["female", "male"],
            "birth_year": pd.array([1950, 1950], dtype="Int64"),
            "start_year": pd.array([2012, 2010], dtype="Int64"),
            "required_interview_year": pd.array([2013, 2011], dtype="Int64"),
            "end_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "separation_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "most_recent_report_year": pd.array([2014, 2014], dtype="Int64"),
            "n_marriages": pd.array([1, 1], dtype="Int64"),
            "is_marriage": [True, True],
        }
    )

    result = refit_first_marriage_modifier(
        family,  # type: ignore[arg-type]
        native_panel,
        _modifier_earnings(),
        marriages,
        pd.Series({1: 1.0, 2: 1.0}),
        _ssa_params(),
        params_vintage=2014,
        train_ids={1, 2},
        interview_years=pd.Series([2007, 2009, 2011, 2013, 2015]),
    )

    assert result.provenance.n_rows["marital_person_years"] == 8
    assert result.provenance.n_rows["marital_events"] == 1
    assert result.provenance.max_year["marital"] == 2012


def test_refit_entry_points_do_not_use_filesystem_artifacts(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("refit attempted filesystem IO")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    factory = _RecordingQRFFactory()
    result = refit_earnings_chained_generator(
        _earnings_panel(),
        _earnings_nawi(),
        seed=1,
        qrf_factory=factory,
    )
    assert result.provenance.certified_full_window_artifacts_read is False


def test_bundle_reports_frozen_registry_specs_separately_from_new_fits():
    family_registry = _RecordingRegistry()
    household_registry = _RecordingRegistry()
    family = refit_family_transitions(
        _family_context(), registry=family_registry
    )
    household = refit_household_composition(
        _household_context(), registry=household_registry
    )
    bundle = M6RefitBundle(
        boundary_year=2014,
        family=family,
        household=household,
    )
    assert bundle.registry_spec_sha256s == {
        "family_transitions": ft.CANDIDATE_16.sha256,
        "household_composition": hc.CANDIDATE_9.sha256,
    }
    assert family.fitted == {"new_fit": 1}
    assert household.fitted == {"new_fit": 1}
