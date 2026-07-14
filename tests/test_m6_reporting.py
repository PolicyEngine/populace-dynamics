"""Synthetic-only tests for the M6 report-only phase."""

from __future__ import annotations

import pandas as pd
import pytest

from populace_dynamics.harness.m6_reporting import (
    NOT_CERTIFIED_ORDER,
    assemble_report_only_payload,
    build_alignment_displacement,
    build_entrant_diagnostics,
    build_mortality_anchor_disclosure,
    build_not_certified_surface,
    build_redrawn_seed_comparison,
    build_shock_window_diagnostics,
)


def test_shock_diagnostic_is_dual_axis_report_only():
    diagnostic = build_shock_window_diagnostics(
        flow_projected={"first_marriage.18-29|female": {"rate": 0.08}},
        flow_truth={"first_marriage.18-29|female": {"rate": 0.10}},
        earnings_projected={"earn_p50.prime": {"value": 90.0}},
        earnings_truth={"earn_p50.prime": {"value": 100.0}},
    )

    assert diagnostic["gated"] is False
    assert diagnostic["machine_reason"] == (
        "exogenous_shock_outside_model_class"
    )
    assert diagnostic["axes"]["flows"] == {
        "axis": "event_year",
        "years": [2020, 2021],
    }
    assert diagnostic["axes"]["earnings"]["years"] == [2020, 2022]
    assert diagnostic["axes"]["earnings"]["unobserved_reference_years"] == [
        2021
    ]
    assert diagnostic["flow_cells"][0]["absolute_gap"] == pytest.approx(0.02)
    assert diagnostic["earnings_cells"][0][
        "absolute_log_ratio"
    ] == pytest.approx(abs(__import__("math").log(0.9)))


def test_not_certified_order_starts_with_mortality_and_names_survivorship():
    surface = build_not_certified_surface(
        {"mortality_drift": {"maximum_gap": 0.3}}
    )

    assert tuple(item["margin"] for item in surface) == NOT_CERTIFIED_ORDER
    assert surface[0]["margin"] == "mortality_drift"
    assert surface[0]["measurement"] == {"maximum_gap": 0.3}
    assert NOT_CERTIFIED_ORDER[:9] == (
        "mortality_drift",
        "widowhood",
        "shock_window_2020_2022",
        "entrants_open_panel",
        "autocorrelation_lag5",
        "forward_projection_2100_extrapolation",
        "stock_margins",
        "remarriage_above_pooled_working_age",
        "forward_earnings_survivorship",
    )
    assert all(item["gated"] is False for item in surface)


def test_entrant_counts_keep_family_b_and_earnings_openers_distinct():
    diagnostic = build_entrant_diagnostics(
        synthetic_births={2020: 2, 2021: 3},
        immigrant_cohorts=4,
        later_earnings_entrants={2017: 6, 2019: 7},
        marked_no_earnings_state=15,
        synthetic_person_ids=9,
    )

    family_b = diagnostic["family_b_open_additions"]
    earnings = diagnostic["earnings_module_open_additions"]
    assert family_b["synthetic_births"]["total"] == 5
    assert family_b["immigrant_cohorts"]["total"] == 4
    assert family_b["synthetic_person_ids"]["total"] == 9
    assert earnings["later_earnings_entrants"]["total"] == 13
    assert earnings["classification"] == (
        "closed_panel_for_flows_open_addition_for_earnings_only"
    )
    assert set(diagnostic["bridge_requirements"]) == {
        "synthetic_births",
        "immigrant_cohorts",
    }
    assert diagnostic["gated"] is False


def test_alignment_displacement_reports_year_maxima_but_scores_unaligned():
    before = pd.DataFrame(
        {
            "person_id": [1, 2, 1],
            "year": [2020, 2020, 2021],
            "earnings": [10.0, 20.0, 15.0],
            "weight": [1.0, 1.0, 2.0],
        }
    )
    after = pd.DataFrame(
        {
            "person_id": [1, 2, 1],
            "year": [2020, 2020, 2021],
            "earnings": [12.0, 19.0, 18.0],
            "weight": [1.1, 0.8, 2.0],
        }
    )

    diagnostic = build_alignment_displacement(
        before,
        after,
        value_columns=("earnings", "weight"),
    )

    assert diagnostic["scored_path"] == "unaligned"
    assert diagnostic["alignment_path"] == "report_only"
    assert diagnostic["maximum_alignment_displacement"] == 3.0
    by_year = {
        record["year"]: record for record in diagnostic["per_year_maximum"]
    }
    assert by_year[2020]["maximum"] == 2.0
    assert by_year[2020]["by_field"]["earnings"] == 2.0
    assert by_year[2020]["by_field"]["weight"] == pytest.approx(0.2)
    assert by_year[2021] == {
        "year": 2021,
        "maximum": 3.0,
        "by_field": {"earnings": 3.0, "weight": 0.0},
    }
    assert diagnostic["n_intervened_rows"] == 3


def test_alignment_absence_is_explicit_and_never_fabricated_as_zero():
    diagnostic = build_alignment_displacement(None, None)

    assert diagnostic["status"] == "not_computed"
    assert diagnostic["maximum_alignment_displacement"] is None
    assert diagnostic["n_intervened_rows"] is None


def test_redrawn_seed_gap_is_explicit_when_successor_machinery_is_absent():
    diagnostic = build_redrawn_seed_comparison(
        realized_seed_cells={"cell": {"rate": 0.1}}
    )

    assert diagnostic["status"] == "unavailable"
    assert diagnostic["reason"] == (
        "successor_forward_seed_machinery_out_of_scope"
    )
    assert diagnostic["redrawn_seed_cells"] is None
    assert diagnostic["comparison"] is None
    assert diagnostic["pass"] is None
    assert diagnostic["margin"] == {
        "name": None,
        "bound": None,
        "status": "unresolved",
        "unresolved_reason": "pre_named_margin_absent_from_ratified_spec",
        "gated": False,
        "successor_flip_seed": True,
    }


def test_redrawn_seed_comparison_remains_non_gating_when_supplied():
    diagnostic = build_redrawn_seed_comparison(
        realized_seed_cells={"cell": {"rate": 0.1}},
        redrawn_seed_cells={"cell": {"rate": 0.12}},
        margin_name="registered_successor_initialization_margin",
        margin_bound=0.25,
    )

    assert diagnostic["status"] == "computed"
    assert diagnostic["comparison"][0]["absolute_gap"] == pytest.approx(0.02)
    assert diagnostic["margin"]["status"] == "supplied"
    assert diagnostic["pass"] is None
    assert diagnostic["gated"] is False


def test_mortality_anchor_names_nchs_calibration_circularity():
    diagnostic = build_mortality_anchor_disclosure(
        {"2020|female": {"projected": 0.01, "anchor": 0.011}}
    )

    assert diagnostic["deliverable"] == (
        "ssa_nchs_life_table_mortality_anchor"
    )
    assert diagnostic["required_before_m7_lock_flip"] is True
    assert "NCHS x PSID-band anchored" in diagnostic["circularity_disclosure"]
    assert diagnostic["external_level_log_ratio_gated"] is False


def test_phase_assembly_rejects_any_component_marked_gated():
    shock = build_shock_window_diagnostics()
    not_certified = build_not_certified_surface()
    entrants = build_entrant_diagnostics(
        synthetic_births=0,
        immigrant_cohorts=0,
        later_earnings_entrants=0,
        marked_no_earnings_state=0,
    )
    alignment = build_alignment_displacement(None, None)
    redrawn = build_redrawn_seed_comparison(realized_seed_cells={})
    mortality = build_mortality_anchor_disclosure()

    payload = assemble_report_only_payload(
        shock_window=shock,
        not_certified=not_certified,
        entrants=entrants,
        alignment=alignment,
        redrawn_seed=redrawn,
        mortality_anchor=mortality,
    )
    assert payload["changes_gate_verdict"] is False
    assert payload["publishes_regardless"] is True

    bad = dict(shock, gated=True)
    with pytest.raises(ValueError, match="gated=False"):
        assemble_report_only_payload(
            shock_window=bad,
            not_certified=not_certified,
            entrants=entrants,
            alignment=alignment,
            redrawn_seed=redrawn,
            mortality_anchor=mortality,
        )
