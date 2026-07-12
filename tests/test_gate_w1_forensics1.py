"""Tests for W1 forensics 1 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the
forensics-1 artifact (``runs/gate_w1_forensics1_v1.json``) and the committed
candidate-1 gate artifact (``runs/gate_w1_candidate1_v1.json``); they never
rerun the diagnostic and need no PSID or frame, so they run in CI. The
PSID/frame-bound bit-identity rebuilds live in
``test_gate_w1_forensics1_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_forensics1_v1.json"
CANDIDATE1 = ROOT / "runs" / "gate_w1_candidate1_v1.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _c1() -> dict:
    return json.loads(CANDIDATE1.read_text())


# -- provenance / registration --------------------------------------------
def test_reported_not_gated():
    a = _artifact()
    assert a["reported_not_gated"] is True
    assert a["schema_version"] == "gate_w1_forensics1.v1"
    assert a["gate"] == "gate_w1"


def test_registration_pointer_is_the_frozen_spec():
    a = _artifact()
    assert a["registration"]["comment_id"] == "4951218279"
    assert a["registration"]["issue"] == 42
    assert a["grading_pointer"] == "4951216895"
    assert a["candidate1_pointer"] == "4950931131"


def test_all_five_questions_present():
    a = _artifact()
    for key in (
        "q1_marital_equilibration",
        "q2_participation_boundary",
        "q3_household_scope",
        "q4_di_level_bridge",
        "q5_tail_upper_read",
    ):
        assert key in a


# -- reconciliations at machine epsilon -----------------------------------
def test_instrumentation_bit_identity_is_exact():
    r = _artifact()["reconciliations"]
    assert r["q1_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["q2_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["q5_positive_year_panel_bit_identical"] is True
    assert r["q5_upper_read_quartet_bit_identical"] is True


def test_reconciliations_are_identities_at_machine_epsilon():
    r = _artifact()["reconciliations"]
    eps = r["float64_machine_epsilon"]
    # the test bar is tighter than the artifact-build 64*eps bar.
    assert r["q1_decomposition_max_abs_remainder"] <= 16 * eps
    assert r["q3_decomposition_max_abs_remainder"] <= 16 * eps
    assert r["q4_decomposition_max_abs_remainder"] <= 16 * eps
    assert r["q3_reference_moment_max_dev"] <= 16 * eps
    assert r["all_identity_reconciliations_at_machine_epsilon"] is True


# -- Q1: the exposure + hazard components telescope to the deficit ---------
def test_q1_decomposition_telescopes():
    q1 = _artifact()["q1_marital_equilibration"]
    assert q1["instrumentation_fidelity"]["bit_identical"] is True
    for cell in q1["per_band_sex"].values():
        total = cell["total_deficit_O_minus_S"]
        recon = (
            cell["component_b_exposure_window"]
            + cell["component_c_hazard_residual"]
        )
        assert abs(total - recon) < 1e-12
        assert (
            abs(cell["observed_init_O"] - cell["synthetic_equilibration_S"])
            - abs(total)
            < 1e-12
        )


def test_q1_hazard_residual_dominates_conformant_path():
    # the honest refinement: extending exposure barely helps; the
    # hazard-level residual dominates -> initialization is the lever.
    f = _artifact()["q1_marital_equilibration"]["finding"]
    assert f["hazard_residual_dominates_conformant_path"] is True
    assert (
        f["mean_abs_component_c_hazard_residual"]
        > f["mean_abs_component_b_exposure"]
    )


# -- Q2: which treatment clears which boundary cell -----------------------
def test_q2_nearest_bin_clears_nothing_extension_clears_most():
    q2 = _artifact()["q2_participation_boundary"]
    assert q2["instrumentation_fidelity"]["bit_identical"] is True
    tally = q2["cells_cleared_tally"]
    # (a) nearest-bin extrapolation clears no boundary cell; (c) frame's own
    # ages clear all (the identity); (b) clears most (a strict majority).
    assert tally["a_nearest_bin"] == 0
    assert tally["c_frame_ages"] == q2["n_scored"]
    assert tally["b_boundary_extension"] > q2["n_scored"] / 2


# -- Q3: scope + composition telescope; scope is NOT the whole miss --------
def test_q3_scope_plus_composition_telescopes():
    q3 = _artifact()["q3_household_scope"]
    assert q3["reference_moment_fidelity"]["bit_identical"] is True
    for cell in q3["per_cell"].values():
        total = cell["total_miss_D_minus_A"]
        recon = (
            cell["scope_component_U_minus_A"]
            + cell["composition_residual_D_minus_U"]
        )
        assert abs(total - recon) < 1e-12
    assert q3["resolution"]["scope_is_whole_miss"] is False
    assert q3["abs_composition_total"] > 0.1  # a material residual remains


# -- Q4: the gate-design determination + concept-delta dominance -----------
def test_q4_is_a_gate_design_finding():
    q4 = _artifact()["q4_di_level_bridge"]
    det = q4["gate_design_determination"]
    assert det["is_gate_design_finding"] is True
    assert det["insured_denominator_available"] is False
    assert q4["concept_delta_dominant_share"] > 0.5


def test_q4_awards_flow_composition_sums_to_100():
    q4 = _artifact()["q4_di_level_bridge"]
    assert abs(q4["awards_flow_composition_sums_to"] - 100.0) < 1e-6
    for band in q4["per_band"].values():
        gap = band["total_gap_anchor_minus_deployed"]
        recon = (
            band["m4_shape_component_flow_minus_deployed"]
            + band["duration_concept_flow_to_stock"]
        )
        assert abs(gap - recon) < 1e-9


def test_q4_duration_concept_dominates_the_60_fra_gap():
    band = _artifact()["q4_di_level_bridge"]["per_band"]["60-fra"]
    assert abs(band["duration_concept_flow_to_stock"]) > abs(
        band["m4_shape_component_flow_minus_deployed"]
    )


# -- Q5: the single most consequential number -----------------------------
def test_q5_c1_non_reversal_is_robust():
    q5 = _artifact()["q5_tail_upper_read"]
    fid = q5["instrumentation_fidelity"]
    assert fid["positive_year_panel_bit_identical_vs_committed"] is True
    assert fid["upper_read_F4_quartet_bit_identical_vs_committed"] is True
    ur, cor = q5["upper_read"], q5["corrected_tail"]
    # neither tail reverses C1; PPI stays below NRA in both.
    assert ur["c1_reversed"] is False
    assert cor["c1_reversed"] is False
    assert ur["ppi_savings_abs"] < ur["nra_savings_abs"]
    assert cor["ppi_savings_abs"] < cor["nra_savings_abs"]
    # the correction moves PPI in the CONSERVATIVE direction (down / lighter).
    assert cor["tail_lighter_than_upper_read"] is True
    assert cor["ppi_savings_abs"] <= ur["ppi_savings_abs"]
    assert q5["c1_robustness_answer"]["answer_non_reversal_is_robust"] is True


def test_q5_upper_read_matches_committed_candidate1():
    q5 = _artifact()["q5_tail_upper_read"]["upper_read"]
    c1 = _c1()["family_c"]
    assert (
        q5["frac_payroll_above_wage_base"]
        == c1["frac_payroll_above_wage_base"]
    )
    assert q5["c1_order"] == c1["fingerprints"]["c1"]["deployed_order"]


def test_q5_c2_committed_swap_holds_both_tails():
    q5 = _artifact()["q5_tail_upper_read"]
    # the committed C2 swap (elimination outranks +2pp) is realised in both.
    assert q5["upper_read"]["c2_swap_realised"] is True
    assert q5["corrected_tail"]["c2_swap_realised"] is True


# -- generic guards (the forensics convention) ----------------------------
def test_all_identity_remainder_fields_are_negligible():
    a = _artifact()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.endswith("remainder") and isinstance(v, int | float):
                    assert abs(v) < 1e-6, k
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(a)


def test_finite_or_null_floats_only():
    text = ARTIFACT.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
