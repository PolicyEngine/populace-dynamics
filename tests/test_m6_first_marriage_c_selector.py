"""Synthetic full-pipeline guards for the train-only first-marriage selector."""

from __future__ import annotations

import copy
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import select_m6_first_marriage_c as selector  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_ledgers() -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = {"synthetic_test_runtime": True}
    freeze = {"synthetic_test_freeze": True}
    happy = selector.run_selection(
        selector._synthetic_frames(), runtime=runtime, freeze=freeze
    )
    no_op = selector.run_selection(
        selector._synthetic_frames(all_events_zero=True),
        runtime=runtime,
        freeze=freeze,
    )
    return happy, no_op


def test_protocol_and_runtime_identity_are_exact() -> None:
    selector._assert_protocol_constants()
    assert selector.EXPECTED_RUNTIME_IDENTITY == {
        "python_implementation": "CPython",
        "python_version": "3.13.12",
        "numpy_version": "2.4.2",
    }
    protocol = selector._protocol_ledger()
    assert protocol["pseudo_fit_count"] == 27
    assert protocol["eligible_final_fit_count"] == 1
    assert protocol["max_iter"] == 10_000
    assert protocol["lbfgs_projected_gradient_gtol"] == 1e-8
    assert protocol["lbfgs_function_reduction_ftol"] == selector.SOLVER_FTOL
    assert protocol["stochastic_selector_seed"] is None
    assert protocol["randomness_used"] is False
    assert protocol["earnings_section_6_fit_seed_5200_borrowed"] is False
    assert protocol["post_2014_rows_enter_fit_or_evaluation_frames"] is False
    assert (
        protocol["gate_seed_contract_recorded_not_consumed"]["consumed"]
        is False
    )
    assert protocol["calendar_year_multiplicity"] == {
        "2007": 1,
        "2008": 1,
        "2009": 2,
        "2010": 2,
        "2011": 2,
        "2012": 2,
        "2013": 1,
        "2014": 1,
    }


def test_runtime_mismatch_aborts_before_repository_or_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = {"repository": False, "data": False}

    def repository_sentinel() -> dict[str, Any]:
        touched["repository"] = True
        return {}

    def data_sentinel() -> selector.SelectionFrames:
        touched["data"] = True
        raise AssertionError("data loader must not run")

    monkeypatch.setattr(platform, "python_version", lambda: "0.0.0")
    monkeypatch.setattr(selector, "_repository_freeze", repository_sentinel)
    monkeypatch.setattr(selector, "_load_real_selection_frames", data_sentinel)
    monkeypatch.setattr(sys, "argv", ["select_m6_first_marriage_c.py"])
    with pytest.raises(selector.RuntimeIdentityAbort, match="numeric runtime"):
        selector.main()
    assert touched == {"repository": False, "data": False}


def test_native_panel_then_field_truncate_preserves_late_reported_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from populace_dynamics.data import (
        births,
        panels,
        transitions,
    )
    from populace_dynamics.data import (
        deaths as death_data,
    )
    from populace_dynamics.data import (
        marriage as marriage_data,
    )
    from populace_dynamics.engine import refit

    columns = [
        "person_id",
        "sex",
        "birth_year",
        "birth_month",
        "marriage_order",
        "spouse_person_id",
        "start_year",
        "start_month",
        "end_year",
        "end_month",
        "separation_year",
        "separation_month",
        "how_ended",
        "last_known_status",
        "most_recent_report_year",
        "n_marriages",
        "n_records",
        "is_marriage",
    ]
    marriages = pd.DataFrame(
        [
            {
                "person_id": 1,
                "sex": "female",
                "birth_year": 1980,
                "marriage_order": 1,
                "start_year": 2010,
                "how_ended": "intact",
                "last_known_status": "married",
                "most_recent_report_year": 2017,
                "n_marriages": 1,
                "n_records": 1,
                "is_marriage": True,
            },
            {
                "person_id": 2,
                "sex": "male",
                "birth_year": 1982,
                "how_ended": "never_married",
                "last_known_status": "never_married",
                "most_recent_report_year": 2013,
                "n_marriages": 0,
                "n_records": 1,
                "is_marriage": False,
            },
        ],
        columns=columns,
    )
    for column in (
        "birth_year",
        "marriage_order",
        "spouse_person_id",
        "start_year",
        "end_year",
        "separation_year",
        "most_recent_report_year",
        "n_marriages",
    ):
        marriages[column] = marriages[column].astype("Int64")
    for column in ("sex", "how_ended", "last_known_status"):
        marriages[column] = marriages[column].astype("string")
    marriages["is_marriage"] = marriages["is_marriage"].astype(bool)
    demo_rows = []
    for person_id in (1, 2):
        for period, weight in (
            (2005, 1.0),
            (2007, 1.05),
            (2009, 1.1),
            (2011, 1.2),
            (2013, 1.3),
            (2017, 9.0),
        ):
            demo_rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "weight": weight,
                    "interview": person_id * 100 + period,
                }
            )
    demo = pd.DataFrame(demo_rows)
    deaths = pd.DataFrame(
        {
            "person_id": [1, 2],
            "death_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
        }
    )

    native, construction = selector._native_family_context(
        demo, marriages, pd.DataFrame(), deaths
    )
    assert construction["native_marriage_records_pretruncated"] is False
    native_event = native.panel.events[
        native.panel.events["person_id"].eq(1)
        & native.panel.events["transition"].eq("first_marriage")
    ]
    assert native_event["year"].tolist() == [2010]

    truncated = refit._truncate_family_context(native, 2014)
    selector._assert_context_boundary(truncated, 2014, "late_report_fixture")
    retained_event = truncated.panel.events[
        truncated.panel.events["person_id"].eq(1)
        & truncated.panel.events["transition"].eq("first_marriage")
    ]
    assert retained_event["year"].tolist() == [2010]
    assert retained_event["required_interview_year"].tolist() == [2011.0]
    risk = selector._context_first_marriage_frame(
        truncated, "late_report_fixture.fit"
    )
    event_risk = risk[risk["person_id"].eq(1) & risk["year"].eq(2010)]
    assert event_risk["event"].tolist() == [True]
    assert event_risk["weight"].tolist() == [1.3]

    pretruncated_records = refit._truncate_marriage_records(marriages, 2014)
    assert 1 not in set(pretruncated_records["person_id"])
    full_weight = (
        demo[demo["weight"] > 0]
        .sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    pretruncated_panel = transitions.build_marital_panel(
        pretruncated_records, deaths, full_weight
    )
    assert 1 not in set(pretruncated_panel.attrs["person_id"])
    assert not pretruncated_panel.events["person_id"].eq(1).any()

    monkeypatch.setattr(panels, "demographic_panel", lambda: demo.copy())
    monkeypatch.setattr(
        marriage_data, "marriage_history", lambda: marriages.copy()
    )
    monkeypatch.setattr(
        death_data, "read_death_records", lambda: deaths.copy()
    )

    def forbidden_birth_read() -> pd.DataFrame:
        raise AssertionError("first-marriage selector loaded birth history")

    monkeypatch.setattr(births, "birth_history", forbidden_birth_read)
    loaded = selector._load_real_selection_frames()
    loaded_event_risk = loaded.final[
        loaded.final["person_id"].eq(1) & loaded.final["year"].eq(2010)
    ]
    assert loaded_event_risk["event"].tolist() == [True]
    assert loaded.source_audit["birth_history_loaded"] is False
    assert (
        loaded.source_audit[
            "raw_post_2014_values_may_be_read_for_native_panel_construction"
        ]
        is True
    )
    assert (
        loaded.source_audit["selection_frames_contain_post_2014_date_values"]
        is False
    )
    assert (
        loaded.source_audit[
            "native_panel_built_before_field_aware_context_truncation"
        ]
        is True
    )
    assert (
        loaded.source_audit["raw_retrospective_value_counts"][
            "pre_2015_marriages_reported_after_2014"
        ]
        == 1
    )


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("final", "year"),
        ("final", "required_interview_year"),
        ("boundary_fit", "required_interview_year"),
        ("boundary_evaluation", "required_interview_year"),
    ],
)
def test_field_aware_2015_poison_aborts_before_any_fit(
    location: str, field: str
) -> None:
    frames = selector._synthetic_frames()
    boundaries = dict(frames.boundaries)
    poisoned_final = frames.final
    if location == "final":
        poisoned_final = frames.final.copy()
        poisoned_final.loc[poisoned_final.index[0], field] = 2015
    elif location == "boundary_fit":
        pair = boundaries[2006]
        fit = pair.fit.copy()
        fit.loc[fit.index[0], field] = 2015
        boundaries[2006] = selector.BoundaryFrames(
            fit=fit, evaluation=pair.evaluation
        )
    else:
        pair = boundaries[2006]
        evaluation = pair.evaluation.copy()
        evaluation.loc[evaluation.index[0], field] = 2015
        boundaries[2006] = selector.BoundaryFrames(
            fit=pair.fit, evaluation=evaluation
        )
    poisoned = selector.SelectionFrames(
        boundaries=boundaries,
        final=poisoned_final,
        source_audit=frames.source_audit,
        mode=frames.mode,
    )
    fit_calls = 0

    def fit_sentinel(*args: Any, **kwargs: Any) -> Any:
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fit must not run after a boundary poison")

    with pytest.raises(selector.ProtocolAbort, match="beyond"):
        selector.run_selection(
            poisoned,
            runtime={},
            freeze={},
            fit_function=fit_sentinel,
        )
    assert fit_calls == 0


def test_happy_synthetic_executes_27_plus_one_full_pipeline(
    synthetic_ledgers: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    happy, _ = synthetic_ledgers
    assert happy["status"] == selector.SELECTION_COMPLETE
    assert happy["information_contact"] == {
        "train_outcomes_through_2014_contacted": True,
        "pseudo_holdout_is_within_train_information": True,
        "raw_retrospective_post_2014_source_values_may_be_read": False,
        "post_2014_holdout_truth_table_contacted": False,
        "post_2014_row_entered_fit_or_evaluation_frame": False,
        "candidate_1_or_candidate_2_score_contacted": False,
    }
    assert (
        happy["selection"]["post_2014_row_entered_numerical_selection"]
        is False
    )
    assert happy["fit_counts"] == {
        "pseudo_boundary_attempts": 27,
        "expected_pseudo_boundary_attempts": 27,
        "final_attempts": 1,
        "total_attempts": 28,
    }
    assert happy["selection"]["selected_c"] in selector.C_GRID
    assert (
        happy["registration_disposition"]["registerable_c"]
        == happy["selection"]["selected_c"]
    )
    assert happy["final_fit"]["preflight"]["expected_checksum_replay"] is True
    assert (
        happy["final_fit"]["preflight"]["recomputed_checksums"]
        == happy["final_fit"]["fit_audit"]["checksums"]
    )
    assert all(
        len(happy["boundaries"][str(boundary)]["rungs"]) == 9
        for boundary in selector.PSEUDO_BOUNDARIES
    )
    for boundary in selector.PSEUDO_BOUNDARIES:
        record = happy["boundaries"][str(boundary)]
        assert record["fit_frame"]["year_max"] <= boundary
        assert record["fit_frame"]["required_interview_year_max"] <= boundary
        assert record["evaluation_frame"]["year_max"] <= 2014
        assert (
            record["evaluation_frame"]["required_interview_year_max"] <= 2014
        )
        for rung in record["rungs"]:
            audit = rung["fit_audit"]
            assert audit["max_iter"] == 10_000
            assert audit["solver_gtol"] == 1e-8
            assert audit["solver_ftol"] == selector.SOLVER_FTOL
            if rung["eligible"]:
                assert audit["solver_success"] is True
                assert audit["convergence_warning_count"] == 0
                assert audit["gradient_inf_norm"] <= selector.GRADIENT_TOL
    female_1990 = [
        row
        for row in happy["final_fit"]["hazard_table_ages_18_29"]
        if row["sex"] == "female"
        and row["target_birth_decade"] == 1990
        and row["age"] >= 23
    ]
    assert [row["age"] for row in female_1990] == list(range(23, 30))
    strict = json.dumps(happy, sort_keys=True, allow_nan=False)
    selector.validate_complete_ledger(json.loads(strict))


def test_empty_and_zero_event_cells_survive_publication_and_reduction(
    synthetic_ledgers: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    happy, _ = synthetic_ledgers
    support = happy["final_fit"]["fit_audit"]["support"]["sex_by_cohort"]
    assert support["female|1990"] == {
        "n_rows": 0,
        "row_weight": 0.0,
        "n_events": 0,
        "event_weight": 0.0,
        "age_min": None,
        "age_max": None,
    }
    assert support["female|1980"]["n_rows"] > 0
    assert support["female|1980"]["n_events"] == 0
    assert support["female|1980"]["event_weight"] == 0.0
    reduced = selector.reduce_ledger(json.loads(selector._strict_json(happy)))
    reduced_support = reduced["final_fit"]["fit_audit"]["support"][
        "sex_by_cohort"
    ]
    assert reduced_support["female|1990"] == support["female|1990"]
    assert reduced_support["female|1980"] == support["female|1980"]
    assert reduced["selection"] == happy["selection"]
    json.dumps(reduced, sort_keys=True, allow_nan=False)


def test_all_ineligible_case_publishes_complete_no_op_ledger(
    synthetic_ledgers: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, no_op = synthetic_ledgers
    assert no_op["status"] == selector.NO_REGISTERABLE
    assert no_op["selection"]["selected_c"] is None
    assert no_op["selection"]["eligible_cs"] == []
    assert no_op["fit_counts"] == {
        "pseudo_boundary_attempts": 27,
        "expected_pseudo_boundary_attempts": 27,
        "final_attempts": 0,
        "total_attempts": 27,
    }
    assert no_op["final_fit"]["attempted"] is False
    assert no_op["registration_disposition"]["registerable_c"] is None
    rungs = [
        rung
        for boundary in selector.PSEUDO_BOUNDARIES
        for rung in no_op["boundaries"][str(boundary)]["rungs"]
    ]
    assert len(rungs) == 27
    assert all(not rung["eligible"] for rung in rungs)
    assert all(rung["fit_exception"]["type"] == "ValueError" for rung in rungs)
    parsed = json.loads(selector._strict_json(no_op))
    reduced = selector.reduce_ledger(parsed)
    assert reduced["status"] == selector.NO_REGISTERABLE
    assert (
        sum(
            len(reduced["boundaries"][str(boundary)]["rungs"])
            for boundary in selector.PSEUDO_BOUNDARIES
        )
        == 27
    )


def test_tie_within_tolerance_chooses_smaller_c() -> None:
    boundaries: dict[str, Any] = {}
    for boundary in selector.PSEUDO_BOUNDARIES:
        rungs = []
        for index, c in enumerate(selector.C_GRID):
            eligible = index in (0, 1)
            deviance = (
                1.0 + selector.TIE_TOLERANCE / 2.0
                if index == 0
                else 1.0 if index == 1 else None
            )
            rungs.append(
                {
                    "c": c,
                    "eligible": eligible,
                    "evaluation": {
                        "weighted_mean_bernoulli_deviance": deviance
                    },
                }
            )
        boundaries[str(boundary)] = {"rungs": rungs}
    selection = selector._selection_from_boundaries(boundaries)
    assert selection["tie_set"] == [
        selector.C_GRID[0],
        selector.C_GRID[1],
    ]
    assert selection["selected_c"] == selector.C_GRID[0]


def test_complete_ledger_validator_rejects_extra_keys_and_stale_selection(
    synthetic_ledgers: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    happy, _ = synthetic_ledgers
    extra = copy.deepcopy(happy)
    extra["unexpected"] = True
    with pytest.raises(selector.ProtocolAbort, match="top-level keys differ"):
        selector.validate_complete_ledger(extra)

    extra_rung = copy.deepcopy(happy)
    extra_rung["boundaries"]["2006"]["rungs"][0]["unexpected"] = True
    with pytest.raises(
        selector.ProtocolAbort, match=r"rungs\[0\] keys differ"
    ):
        selector.validate_complete_ledger(extra_rung)

    missing_nested = copy.deepcopy(happy)
    del missing_nested["boundaries"]["2008"]["rungs"][2]["evaluation"][
        "prediction_sha256"
    ]
    with pytest.raises(selector.ProtocolAbort, match="evaluation keys differ"):
        selector.validate_complete_ledger(missing_nested)

    stale = copy.deepcopy(happy)
    stale["selection"]["selected_c"] = selector.C_GRID[-1]
    with pytest.raises(selector.ProtocolAbort, match="all-rung reduction"):
        selector.validate_complete_ledger(stale)
