"""Synthetic, reader-free tests for the M6 production input boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics import claiming
from populace_dynamics.data import household_composition as household_data
from populace_dynamics.data import transitions
from populace_dynamics.engine.panel_builders import PanelBuilderInputs
from populace_dynamics.harness import m6_inputs
from populace_dynamics.harness.m6_inputs import (
    M6RawInputs,
    assemble_m6_fit_inputs,
    assemble_m6_inputs,
    load_m6_raw_inputs,
)


def _claim_reference(year: int = 2014) -> claiming.ClaimAgeReference:
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
        supplement_year=year,
        raw_columns=(),
        collapsed_categories=tuple(categories),
        provenance={},
        validation={},
        fra_schedule={},
        _data={"female": {"2014": row}, "male": {"2014": row}},
    )


def _demographic_panel() -> pd.DataFrame:
    rows = []
    for person_id, _sex, birth, first_wave in (
        (1, "female", 1980, 2013),
        (2, "male", 1970, 2013),
        (3, "female", 1990, 2017),
    ):
        for wave in (2013, 2015, 2017, 2019, 2021, 2023):
            if wave < first_wave:
                continue
            rows.append(
                {
                    "person_id": person_id,
                    "period": wave,
                    "age": wave - birth,
                    "sequence": 1,
                    "relationship": 10,
                    "weight": float(person_id if wave <= 2014 else 90 + wave),
                    "interview": person_id * 100 + wave,
                }
            )
    return pd.DataFrame(rows)


def _marital_panel() -> transitions.MaritalPanel:
    rows = []
    for person_id, sex, birth in (
        (1, "female", 1980),
        (2, "male", 1970),
        (3, "female", 1990),
    ):
        for year in range(2013, 2023):
            rows.append(
                {
                    "person_id": person_id,
                    "year": year,
                    "age": year - birth,
                    "sex": sex,
                    "weight": float(person_id),
                    "marital_state": (
                        "never_married" if person_id != 2 else "married"
                    ),
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
            "year": [2016],
            "age": [36],
            "sex": ["female"],
            "weight": [1.0],
            "transition": ["first_marriage"],
            "marriage_duration": pd.array([pd.NA], dtype="Int64"),
            "years_since_dissolution": pd.array([pd.NA], dtype="Int64"),
            "origin": pd.array([pd.NA], dtype="string"),
        }
    )
    attrs = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "sex": ["female", "male", "female"],
            "birth_year": [1980.0, 1970.0, 1990.0],
            "most_recent_report_year": [2023.0] * 3,
            "n_marriages": [0.0, 1.0, 0.0],
            "death_year": pd.array([pd.NA] * 3, dtype="Int64"),
            "censor_year": [2023.0] * 3,
            "weight": [1.0, 2.0, 3.0],
            "start_exposure_year": [1995.0, 1985.0, 2005.0],
        }
    )
    return transitions.MaritalPanel(person_years, events, attrs)


def _household_panel(
    demo: pd.DataFrame,
) -> household_data.HouseholdCompositionPanel:
    rows = []
    for row in demo.itertuples(index=False):
        if row.age < household_data.START_AGE:
            continue
        rows.append(
            {
                "person_id": row.person_id,
                "year": row.period,
                "age": row.age,
                "band": household_data._band_of(row.age),
                "sex": "female" if row.person_id != 2 else "male",
                "weight": row.weight,
                "hh_size": 1,
                "coresident_parent": False,
                "coresident_spouse": False,
                "coresident_child": False,
                "coresident_grandchild": False,
                "multigen": False,
            }
        )
    person_waves = household_data._add_transitions(pd.DataFrame(rows))
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    return household_data.HouseholdCompositionPanel(person_waves, attrs)


def _disability_status() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "period": [2013, 2015, 2017, 2013, 2015, 2017, 2017, 2019],
            "age": [33, 35, 37, 43, 45, 47, 27, 29],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0],
            "status_code": [1, 1, 5, 1, 5, 1, 1, 1],
            "disabled": [False, False, True, False, True, False, False, False],
            "retired": [False] * 8,
        }
    )


def _household_sources() -> dict[str, object]:
    demo = _demographic_panel()
    marital = _marital_panel()
    empty_year = pd.DataFrame({"person_id": [], "year": []})
    relationship = pd.DataFrame(
        {
            "ego_rel_to_alter": pd.Series(dtype="int64"),
            "interview_year": pd.Series(dtype="int64"),
            "ego_person_id": pd.Series(dtype="int64"),
        }
    )
    return {
        "hh": _household_panel(demo),
        "mpanel": marital,
        "demo": demo,
        "mh": pd.DataFrame({"person_id": [1, 2, 3]}),
        "bh": pd.DataFrame({"birth_year": [], "person_id": []}),
        "rel_map": relationship,
        "order_map": pd.DataFrame(
            {"person_id": [], "start_year": [], "order": []}
        ),
        "father_links_child": pd.DataFrame(
            {"person_id": [], "birth_year": []}
        ),
        "parent_pairs": empty_year.copy(),
        "marital_by_year": empty_year.copy(),
        "fu_sizes": empty_year.copy(),
        "legal_flag": empty_year.copy(),
        "parent_counts": empty_year.copy(),
        "child_record_expo": empty_year.copy(),
    }


def _raw() -> M6RawInputs:
    death_records = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "sex": ["female", "male", "female"],
            "death_year": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
        }
    )
    earnings = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "period": [2014, 2016, 2018, 2014, 2016, 2018, 2016, 2018],
            "earnings": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 5.0, 6.0],
            "age": [34, 36, 38, 44, 46, 48, 26, 28],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    mortality_exposure = pd.DataFrame(
        {
            "event_year": [2013, 2015],
            "required_interview_year": [2013, 2015],
            "age_band": ["35-44", "35-44"],
            "sex": ["female", "female"],
            "start_weight": [1.0, 1.0],
            "exposure": [1.0, 1.0],
            "death": [0.0, 0.0],
        }
    )
    rates = pd.DataFrame(
        {
            "lower_age": [35],
            "upper_age": [44],
            "age_band": ["35-44"],
            "sex": ["female"],
            "central_rate": [0.01],
        }
    )
    return M6RawInputs(
        household_sources=_household_sources(),
        death_records=death_records,
        earnings_panel=earnings,
        disability_status=_disability_status(),
        ssa_params=SimpleNamespace(
            nawi={year: float(year) for year in range(2005, 2015)},
            pe_us_revision="synthetic-pre-2015",
        ),
        ssa_params_vintage=2014,
        claiming_reference=_claim_reference(),
        mortality_exposure=mortality_exposure,
        mortality_external_rates=rates,
        mortality_external_vintage=2010,
    )


@pytest.fixture
def certified_marital_builder(monkeypatch):
    captured = {}

    def build(
        cls,
        *,
        anchor,
        marriage_records,
        death_records,
        household,
        cohabitation,
        projection_end_year=2022,
    ):
        captured.update(
            anchor=anchor,
            marriage_records=marriage_records,
            death_records=death_records,
            household=household,
            cohabitation=cohabitation,
        )
        return PanelBuilderInputs(
            anchor=anchor.copy(),
            marital=_marital_panel(),
            household=household,
            cohabitation=cohabitation,
            projection_end_year=projection_end_year,
        )

    monkeypatch.setattr(
        PanelBuilderInputs,
        "from_realized_histories",
        classmethod(build),
    )
    return captured


def test_assembly_builds_every_runner_input_without_a_reader_call(
    certified_marital_builder, monkeypatch
):
    for reader in (
        (m6_inputs.hc_data, "load_sources"),
        (m6_inputs.deaths, "read_death_records"),
        (m6_inputs.family, "family_earnings_panel"),
        (m6_inputs.disability, "read_disability_status"),
    ):
        monkeypatch.setattr(
            *reader,
            lambda: (_ for _ in ()).throw(AssertionError("reader called")),
        )

    raw = _raw()
    bundle = assemble_m6_inputs(raw, earnings_seed=5217)

    assert bundle.refit_inputs.earnings_seed == 5217
    assert bundle.refit_inputs.earnings_panel is raw.earnings_panel
    assert bundle.refit_inputs.family_context.train_ids == frozenset({1, 2})
    assert bundle.refit_inputs.household_context.train_ids == frozenset({1, 2})
    assert bundle.refit_inputs.disability_train_ids == {1, 2}
    assert bundle.refit_inputs.modifier_person_weight.to_dict() == {
        1: 1.0,
        2: 2.0,
    }
    assert bundle.truth.anchor["person_id"].tolist() == [1, 2, 3]
    assert set(bundle.truth.presence[2015]) == {1, 2}
    assert set(bundle.truth.earnings["person_id"]) == {1, 2, 3}
    assert len(bundle.truth.marital_events) == 1
    assert (
        certified_marital_builder["marriage_records"]
        is raw.household_sources["mh"]
    )
    assert bundle.provenance.external_vintages == {
        "ssa_parameters": 2014,
        "claiming_reference": 2014,
        "mortality_reference": 2010,
    }
    artifact = bundle.provenance.to_artifact()
    assert artifact["boundary_year"] == 2014
    assert artifact["certified_full_window_artifacts_read"] is False
    assert artifact["certified_full_window_artifacts_written"] is False


def test_fit_only_assembly_never_invokes_a_holdout_truth_builder(monkeypatch):
    touched = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        touched.append("truth")
        raise AssertionError("holdout truth builder called")

    for name in (
        "build_anchor_frame",
        "presence_by_wave",
        "mortality_slices",
        "marital_tables_from_panel",
        "disability_pairs",
        "earnings_frame",
    ):
        monkeypatch.setattr(m6_inputs, name, forbidden)
    monkeypatch.setattr(
        PanelBuilderInputs,
        "from_realized_histories",
        classmethod(forbidden),
    )

    fit_inputs = assemble_m6_fit_inputs(_raw(), earnings_seed=5217)

    assert touched == []
    assert fit_inputs.earnings_seed == 5217
    assert fit_inputs.family_context.train_ids == frozenset({1, 2})
    assert fit_inputs.household_context.train_ids == frozenset({1, 2})
    assert fit_inputs.disability_train_ids == {1, 2}


def _assert_same_structure(left, right, *, path):
    if left is right:
        return
    assert type(left) is type(right), path
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, obj=path)
        return
    if isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, obj=path)
        return
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
        return
    if is_dataclass(left):
        for field in fields(left):
            _assert_same_structure(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(left, Mapping):
        assert tuple(left) == tuple(right), path
        for key in left:
            _assert_same_structure(
                left[key], right[key], path=f"{path}[{key!r}]"
            )
        return
    if isinstance(left, tuple | list):
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _assert_same_structure(
                left_item, right_item, path=f"{path}[{index}]"
            )
        return
    if isinstance(left, set | frozenset):
        assert left == right, path
        return
    if hasattr(left, "__dict__"):
        _assert_same_structure(
            vars(left), vars(right), path=f"{path}.__dict__"
        )
        return
    assert left == right, path


def test_fit_only_and_legacy_assembly_match_every_refit_input_field(
    certified_marital_builder,
):
    del certified_marital_builder
    raw = _raw()
    fit_only = assemble_m6_fit_inputs(raw, earnings_seed=5217)
    legacy = assemble_m6_inputs(raw, earnings_seed=5217).refit_inputs

    compared = []
    for field in fields(fit_only):
        compared.append(field.name)
        _assert_same_structure(
            getattr(fit_only, field.name),
            getattr(legacy, field.name),
            path=f"M6RefitInputs.{field.name}",
        )
    assert compared == [field.name for field in fields(legacy)]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"ssa_params_vintage": 2015}, "post-T"),
        ({"claiming_reference": _claim_reference(2023)}, "post-T"),
        ({"mortality_external_vintage": 2015}, "post-T"),
    ],
)
def test_assembly_rejects_every_post_boundary_external_vintage(
    certified_marital_builder, change, match
):
    del certified_marital_builder
    with pytest.raises(ValueError, match=match):
        assemble_m6_inputs(replace(_raw(), **change))


def test_assembly_rejects_unregistered_boundary(certified_marital_builder):
    del certified_marital_builder
    with pytest.raises(ValueError, match="requires boundary_year=2014"):
        assemble_m6_inputs(_raw(), boundary_year=2015)


def test_production_loader_uses_only_certified_readers_and_caller_externals(
    monkeypatch,
):
    raw = _raw()
    calls = []
    monkeypatch.setattr(
        m6_inputs.hc_data,
        "load_sources",
        lambda: calls.append("household") or raw.household_sources,
    )
    monkeypatch.setattr(
        m6_inputs.deaths,
        "read_death_records",
        lambda: calls.append("deaths") or raw.death_records,
    )
    monkeypatch.setattr(
        m6_inputs.family,
        "family_earnings_panel",
        lambda: calls.append("earnings") or raw.earnings_panel,
    )
    monkeypatch.setattr(
        m6_inputs.disability,
        "read_disability_status",
        lambda: calls.append("disability") or raw.disability_status,
    )

    loaded = load_m6_raw_inputs(
        ssa_params=raw.ssa_params,
        ssa_params_vintage=raw.ssa_params_vintage,
        claiming_reference=raw.claiming_reference,
        mortality_exposure=raw.mortality_exposure,
        mortality_external_rates=raw.mortality_external_rates,
        mortality_external_vintage=raw.mortality_external_vintage,
    )

    assert calls == ["household", "deaths", "earnings", "disability"]
    assert loaded.household_sources is raw.household_sources
    assert loaded.ssa_params is raw.ssa_params
    assert loaded.claiming_reference is raw.claiming_reference
    assert loaded.mortality_external_rates is raw.mortality_external_rates
