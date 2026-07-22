"""Reader-free tests for the production M6 candidate-2 input plan."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from populace_dynamics.data import family
from populace_dynamics.harness.m6_candidate2_runner import (
    M6Candidate2InputPlan,
)
from populace_dynamics.harness.m6_inputs import M6HarnessInputs

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import registered_m6_candidate2_inputs as candidate2_factory  # noqa: E402
import registered_m6_inputs as candidate1_factory  # noqa: E402


def _full_inputs(refit_inputs: object) -> M6HarnessInputs:
    return M6HarnessInputs(
        refit_inputs=refit_inputs,
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance=object(),
    )


def test_candidate2_plan_defers_every_full_window_reader(monkeypatch):
    events: list[str] = []
    raw = object()
    fit_inputs = object()
    independently_assembled_fit = object()
    assembled = _full_inputs(independently_assembled_fit)

    monkeypatch.setattr(
        candidate2_factory,
        "load_train_only_raw_inputs",
        lambda: events.append("load_train_only") or raw,
    )

    def assemble_fit(observed, **kwargs):
        assert observed is raw
        assert kwargs == {"boundary_year": 2014, "earnings_seed": 5200}
        events.append("assemble_fit_inputs")
        return fit_inputs

    def build_full():
        events.append("candidate1_full_window_factory")
        return assembled

    monkeypatch.setattr(
        candidate2_factory, "assemble_m6_fit_inputs", assemble_fit
    )
    monkeypatch.setattr(candidate1_factory, "build_inputs", build_full)

    plan = candidate2_factory.build_input_plan()

    assert isinstance(plan, M6Candidate2InputPlan)
    assert plan.fit_inputs is fit_inputs
    assert events == ["load_train_only", "assemble_fit_inputs"]

    full_inputs = plan.load_full_inputs()
    assert events == [
        "load_train_only",
        "assemble_fit_inputs",
        "candidate1_full_window_factory",
    ]
    assert full_inputs.refit_inputs is fit_inputs
    assert full_inputs.truth is assembled.truth

    assert plan.load_full_inputs() is full_inputs
    assert events.count("candidate1_full_window_factory") == 1


def test_train_loader_field_caps_scored_sources_and_masks_future_deaths(
    monkeypatch,
):
    events: list[str] = []
    params = object()
    claiming_reference = object()
    demographic = pd.DataFrame(
        {
            "person_id": [1],
            "period": [2013],
            "age": [40],
            "weight": [1.0],
            "interview": [10],
        }
    )
    deaths = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["female", "male"],
            "death_year": pd.array([2012, 2020], dtype="Int64"),
        }
    )
    household_sources = {"bounded": True}
    earnings = pd.DataFrame(
        {
            "person_id": [1],
            "period": [2014],
            "earnings": [1.0],
            "age": [40],
            "weight": [1.0],
        }
    )
    disability = pd.DataFrame(
        {
            "person_id": [1],
            "period": [2013],
            "status_code": [1],
        }
    )
    external_rates = pd.DataFrame({"central_rate": [0.1]})
    exposure = pd.DataFrame(
        {
            "event_year": [2013],
            "required_interview_year": [2013],
        }
    )

    monkeypatch.setattr(
        candidate1_factory,
        "assert_pe_us_version",
        lambda: events.append("version"),
    )
    monkeypatch.setattr(
        candidate1_factory,
        "assert_pe_us_param_dir",
        lambda: events.append("param_dir"),
    )
    monkeypatch.setattr(
        candidate1_factory,
        "boundary_ssa_parameters",
        lambda: events.append("params") or params,
    )
    monkeypatch.setattr(
        candidate1_factory,
        "load_claiming_reference",
        lambda: events.append("claiming") or claiming_reference,
    )

    def load_demographic(**kwargs):
        assert kwargs == {"max_period": 2014}
        events.append("bounded_demographic")
        return demographic

    monkeypatch.setattr(
        candidate2_factory.panels,
        "demographic_panel",
        load_demographic,
    )
    monkeypatch.setattr(
        candidate2_factory.deaths,
        "read_death_records",
        lambda: events.append("retrospective_deaths") or deaths,
    )

    def load_household(
        observed_demo,
        observed_deaths,
        retrospective_deaths,
    ):
        assert observed_demo is demographic
        assert observed_deaths.loc[0, "death_year"] == 2012
        assert pd.isna(observed_deaths.loc[1, "death_year"])
        assert retrospective_deaths is deaths
        assert retrospective_deaths.loc[1, "death_year"] == 2020
        events.append("bounded_household")
        return household_sources

    monkeypatch.setattr(
        candidate2_factory,
        "_load_train_household_sources",
        load_household,
    )

    def load_earnings(*, waves):
        assert tuple(waves) == candidate2_factory.TRAIN_EARNINGS_WAVES
        assert max(wave - 1 for wave in waves) == 2014
        events.append("bounded_earnings")
        return earnings

    monkeypatch.setattr(
        candidate2_factory.family,
        "family_earnings_panel",
        load_earnings,
    )

    def load_disability(**kwargs):
        assert kwargs == {"max_period": 2014}
        events.append("bounded_disability")
        return disability

    monkeypatch.setattr(
        candidate2_factory.disability,
        "read_disability_status",
        load_disability,
    )
    monkeypatch.setattr(
        candidate1_factory,
        "nchs_2010_external_rates",
        lambda: events.append("external_rates") or external_rates,
    )

    def mortality_adapter(observed_demo, observed_deaths):
        assert observed_demo is demographic
        assert pd.isna(observed_deaths.loc[1, "death_year"])
        events.append("bounded_mortality")
        return exposure

    monkeypatch.setattr(
        candidate1_factory,
        "mortality_exposure_adapter",
        mortality_adapter,
    )

    def pad(observed_rates, observed_exposure, **kwargs):
        assert observed_rates is external_rates
        assert observed_exposure is exposure
        assert kwargs == {"boundary_year": 2014}
        events.append("mortality_pad")
        return external_rates, exposure

    monkeypatch.setattr(
        candidate1_factory,
        "_pad_below_25_projection_coverage",
        pad,
    )
    monkeypatch.setattr(
        candidate1_factory,
        "build_inputs",
        lambda: (_ for _ in ()).throw(
            AssertionError("full-window factory reached during plan build")
        ),
    )

    raw = candidate2_factory.load_train_only_raw_inputs()

    assert raw.household_sources is household_sources
    assert raw.earnings_panel is earnings
    assert raw.disability_status is disability
    assert raw.ssa_params is params
    assert raw.claiming_reference is claiming_reference
    assert raw.death_records["death_year"].tolist()[0] == 2012
    assert pd.isna(raw.death_records["death_year"].tolist()[1])
    assert events == [
        "version",
        "param_dir",
        "params",
        "claiming",
        "bounded_demographic",
        "retrospective_deaths",
        "bounded_household",
        "bounded_earnings",
        "bounded_disability",
        "external_rates",
        "bounded_mortality",
        "mortality_pad",
    ]


def test_household_loader_caps_relationships_then_uses_shared_truncator(
    monkeypatch,
):
    demographic = pd.DataFrame(
        {
            "person_id": [1],
            "period": [2013],
            "weight": [1.0],
        }
    )
    deaths = pd.DataFrame({"person_id": [1], "sex": ["female"]})
    retrospective_deaths = deaths.assign(death_year=2020)
    relationship = pd.DataFrame(
        {
            "interview_year": [2013],
            "ego_person_id": [1],
            "ego_rel_to_alter": [10],
        }
    )
    person_waves = pd.DataFrame({"person_id": [1], "year": [2013]})
    marriage_records = object()
    birth_records = object()
    marital_panel = object()
    empty = pd.DataFrame({"person_id": [], "year": []})
    calls = []

    def relationship_map(**kwargs):
        calls.append(("relationship", kwargs))
        return relationship

    monkeypatch.setattr(
        candidate2_factory.relmap,
        "relationship_map",
        relationship_map,
    )
    monkeypatch.setattr(
        candidate2_factory.household_composition,
        "household_roster",
        lambda observed: calls.append(("roster", observed)) or object(),
    )
    monkeypatch.setattr(
        candidate2_factory.household_composition,
        "join_demographics",
        lambda _roster, observed_demo, observed_deaths: (
            calls.append(("join", observed_demo, observed_deaths))
            or person_waves
        ),
    )
    monkeypatch.setattr(
        candidate2_factory.marriage,
        "marriage_history",
        lambda: marriage_records,
    )
    monkeypatch.setattr(
        candidate2_factory.births,
        "birth_history",
        lambda: birth_records,
    )
    monkeypatch.setattr(
        candidate2_factory.transitions,
        "build_marital_panel",
        lambda records, observed_deaths, _weight: (
            calls.append(("marital", records, observed_deaths))
            or marital_panel
        ),
    )
    monkeypatch.setattr(
        candidate2_factory.hc_data,
        "marriage_order_map",
        lambda _records: empty,
    )
    for name in (
        "father_link_births_with_child",
        "parent_child_coresidence_pairs",
        "father_marital_by_year",
        "family_unit_sizes",
        "legal_spouse_flag",
        "parent_link_counts",
    ):
        monkeypatch.setattr(candidate2_factory, name, lambda *_args: empty)
    monkeypatch.setattr(
        candidate2_factory,
        "build_child_record_exposure",
        lambda *_args: empty,
    )

    def fit_context(sources, train_ids):
        assert sources["mh"] is marriage_records
        assert sources["bh"] is birth_records
        assert train_ids == {1}
        calls.append(("context", sources))
        return object()

    monkeypatch.setattr(
        candidate2_factory.hc_registry,
        "fit_context_from_sources",
        fit_context,
    )
    bounded = SimpleNamespace(
        hh=object(),
        mpanel=object(),
        demographic_panel=demographic,
        marriage_records=object(),
        birth_records=object(),
        relationship_map=relationship,
        marriage_order_map=empty,
        father_links_child=empty,
        parent_pairs=empty,
        marital_by_year=empty,
        family_unit_sizes=empty,
        legal_spouse_flag=empty,
        parent_counts=empty,
        child_record_exposure=empty,
    )

    def truncate(context, boundary):
        assert boundary == 2014
        calls.append(("truncate", context))
        return bounded

    monkeypatch.setattr(
        candidate2_factory.refit_engine,
        "_truncate_household_context",
        truncate,
    )

    observed = candidate2_factory._load_train_household_sources(
        demographic,
        deaths,
        retrospective_deaths,
    )

    assert calls[0] == (
        "relationship",
        {"waves": (2013,), "chunksize": 250_000},
    )
    marital_calls = [call for call in calls if call[0] == "marital"]
    assert len(marital_calls) == 1
    assert marital_calls[0][1] is marriage_records
    assert marital_calls[0][2] is retrospective_deaths
    assert calls[-1][0] == "truncate"
    assert observed["demo"] is demographic
    assert observed["rel_map"] is relationship


@pytest.mark.parametrize(
    ("column", "value", "label"),
    [
        ("period", 2015, "demographic"),
        ("period", 2016, "earnings"),
        ("period", 2023, "disability"),
        ("interview_year", 2015, "relationship"),
        ("event_year", 2015, "mortality"),
    ],
)
def test_train_boundary_guard_rejects_future_truth(column, value, label):
    with pytest.raises(RuntimeError, match="post-2014 truth"):
        candidate2_factory._assert_bounded(
            pd.DataFrame({column: [value]}),
            column,
            label,
        )


def test_family_earnings_reader_caps_its_demographic_source(monkeypatch):
    calls = []
    demo = pd.DataFrame(
        {
            "person_id": [1],
            "period": [2015],
            "interview": [10],
            "relationship": [10],
            "age": [40],
            "weight": [1.0],
        }
    )
    labor = pd.DataFrame(
        {
            "interview": [10],
            "head_labor": [100.0],
            "spouse_labor": [0.0],
            "head_acc": [0],
            "spouse_acc": [0],
        }
    )

    def demographic_panel(**kwargs):
        calls.append(kwargs)
        return demo

    monkeypatch.setattr(family.panels, "demographic_panel", demographic_panel)
    monkeypatch.setattr(
        family,
        "read_family_labor",
        lambda wave, **_kwargs: labor,
    )

    observed = family.family_earnings_panel(waves=(2015,))

    assert calls == [{"data_dir": None, "max_period": 2015}]
    assert observed["period"].tolist() == [2014]
