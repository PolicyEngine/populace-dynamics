"""Tests for W1 forensics 2 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the
forensics-2 artifact (``runs/gate_w1_forensics2_v1.json``), the committed
candidate-2 gate artifact (``runs/gate_w1_candidate2_v1.json``), and the
forensics-1 artifact (``runs/gate_w1_forensics1_v1.json``); they never rerun
the diagnostic and need no PSID or frame, so they run in CI. The PSID/frame-
bound bit-identity rebuilds live in
``test_gate_w1_forensics2_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_forensics2_v1.json"
CANDIDATE2 = ROOT / "runs" / "gate_w1_candidate2_v1.json"
FORENSICS1 = ROOT / "runs" / "gate_w1_forensics1_v1.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# -- provenance / registration --------------------------------------------
def test_reported_not_gated():
    a = _artifact()
    assert a["reported_not_gated"] is True
    assert a["schema_version"] == "gate_w1_forensics2.v1"
    assert a["gate"] == "gate_w1"


def test_registration_pointer_is_the_frozen_spec():
    a = _artifact()
    assert a["registration"]["comment_id"] == "4953088871"
    assert a["registration"]["issue"] == 42
    assert a["grading_pointer"] == "4953064479"
    assert a["candidate2_pointer"] == "4952253568"
    assert a["forensics1_pointer"] == "4951218279"


def test_all_four_questions_present():
    a = _artifact()
    for key in (
        "q6_marital_calibration_frame",
        "q7_coresident_parent_fertility",
        "q8_interior_sex_covariate",
        "q9_concept_cells",
    ):
        assert key in a


# -- reconciliations at machine epsilon -----------------------------------
def test_instrumentation_bit_identity_is_exact():
    r = _artifact()["reconciliations"]
    assert r["q6_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["q7_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["q8_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["all_identity_reconciliations_at_machine_epsilon"] is True


def test_reconciliations_are_identities_at_machine_epsilon():
    r = _artifact()["reconciliations"]
    eps = r["float64_machine_epsilon"]
    # the test bar is tighter than the artifact-build 64*eps bar.
    assert r["q6_decomposition_max_abs_remainder"] <= 16 * eps
    assert r["q7_partition_max_abs_deviation_from_one"] <= 16 * eps


# -- Q6: entry-level + hazard telescope; the refined 65+ channel -----------
def test_q6_decomposition_telescopes():
    q6 = _artifact()["q6_marital_calibration_frame"]
    assert q6["instrumentation_fidelity"]["bit_identical"] is True
    pbs = q6["per_band_sex"]
    # the decomposition must be POPULATED (guard against a vacuous pass): the
    # gated married cells at both boundary bands are present.
    assert len(pbs) >= 8
    for band_sex in ("25-34|male", "25-34|female", "65+|male", "65+|female"):
        assert band_sex in pbs
    for cell in pbs.values():
        total = cell["total_miss_D_minus_A"]
        recon = (
            cell["component_a_entry_level_L_minus_A"]
            + cell["component_b_hazard_evolution_D_minus_L"]
        )
        assert abs(total - recon) < 1e-12


def test_q6_entry_level_dominant_at_25_34():
    q6 = _artifact()["q6_marital_calibration_frame"]
    f = q6["finding"]
    assert f["entry_level_dominant_at_25_34"] is True
    assert any(bs.startswith("25-34") for bs in q6["per_band_sex"])
    for band_sex, cell in q6["per_band_sex"].items():
        if band_sex.startswith("25-34"):
            # the 25-34 overshoot is entry-level, not hazard.
            assert cell["dominant_component"] == "entry_level"
            assert cell["component_a_entry_level_L_minus_A"] > 0  # overshoot


def test_q6_65plus_channel_is_divorce_not_widowhood():
    # the honest refinement: the pre-registered widowhood channel is NOT
    # realized; the deployed over-accumulates DIVORCE and under-widows.
    q6 = _artifact()["q6_marital_calibration_frame"]
    f = q6["finding"]
    assert f["realized_65plus_channel"] == "divorce"
    assert f["widowhood_channel_realized_65plus"] is False
    for sex in ("female", "male"):
        ch = q6["dissolution_channel_65plus"][sex]
        assert ch["divorced_excess"] > ch["widowed_excess"]


def test_q6_top_code_gradient_missed():
    # the frame's within-65+ widowhood rises with age (survivor selection);
    # the deployed stays low at the 85 top-code.
    tc = _artifact()["q6_marital_calibration_frame"]["top_code_85"]
    assert 0.0 < tc["frac_65plus_at_top_code_85"] < 0.25
    top = tc["by_age_slice"]["85_topcode"]
    assert top["frame_widowed"] > top["deployed_widowed"]


def test_q6_adjudication_permits_cps_anchored_entry():
    adj = _artifact()["q6_marital_calibration_frame"]["contract_adjudication"]
    assert "CONTRACT-PERMITTED" in adj["determination"]
    assert "BACK-SOLVING" in adj["determination"]  # the prohibited inverse-map


# -- Q7: the five hh_size cells partition; joint feasibility ---------------
def test_q7_partition_and_bit_identity():
    q7 = _artifact()["q7_coresident_parent_fertility"]
    assert q7["instrumentation_fidelity"]["bit_identical"] is True
    for block in q7["scoring"].values():
        assert abs(block["sum_of_shares"] - 1.0) < 1e-12


def test_q7_joint_moves_every_cell_toward_frame_but_insufficient():
    q7 = _artifact()["q7_coresident_parent_fertility"]
    jf = q7["joint_feasibility"]
    # every hh_size cell moves toward the frame under the joint...
    assert jf["all_cells_closed_toward_frame"] is True
    assert jf["size1_materially_improved"] is True
    # ...but the pre-registered "PROVES 3/4/5+" is REFUTED: only size-2 clears.
    assert q7["finding"]["pre_registration_proves_3_4_5plus"] is False
    assert jf["large_sizes_3_4_5plus_all_clear"] is False
    assert jf["joint_cells_cleared"] == ["2"]


def test_q7_coresident_parent_is_the_size1_lever():
    jf = _artifact()["q7_coresident_parent_fertility"]["joint_feasibility"]
    # lever (a) alone cuts size-1 below the baseline.
    assert jf["lever_a_size1"] < jf["size1_baseline"]
    # the fuller window lifts minors per adult materially.
    assert jf["minors_per_adult_fuller15"] > jf["minors_per_adult_base25"]


# -- Q8: the interior sex covariate clears >=3/4 with no collateral --------
def test_q8_clears_at_least_three_of_four_no_collateral():
    q8 = _artifact()["q8_interior_sex_covariate"]
    assert q8["instrumentation_fidelity"]["bit_identical"] is True
    assert q8["n_target_cells"] == 4
    assert q8["n_target_cells_clear"] >= 3
    assert q8["finding"]["clears_at_least_3_of_4"] is True
    assert q8["collateral_flips"] == []
    assert q8["finding"]["no_collateral"] is True


def test_q8_target_cell_scores_within_tolerance():
    q8 = _artifact()["q8_interior_sex_covariate"]
    cleared = sum(1 for c in q8["target_cells"].values() if c["sexcov_clears"])
    assert cleared == q8["n_target_cells_clear"]
    for cell in q8["target_cells"].values():
        if cell["sexcov_clears"]:
            assert cell["sexcov_score"] <= cell["tolerance"]


# -- Q9: the amendment-2 concept evidence ---------------------------------
def test_q9_18_24_concept_gap_exceeds_15pp():
    cg = _artifact()["q9_concept_cells"]["concept_gap_18_24_participation"]
    assert cg["pooled_gap_pp"] >= 15.0
    assert cg["exceeds_15pp_amendment_threshold"] is True
    # the PSID head/spouse universe participates far above the CPS all-person
    # frame -- a population-concept delta, not a fit-support gap.
    assert (
        cg["psid_head_spouse_universe"]["pooled"]
        > cg["cps_all_person_frame"]["pooled"]
    )


def test_q9_c1_non_reversal_is_robust_and_consolidated():
    c1 = _artifact()["q9_concept_cells"]["c1_binary"]
    ana = c1["forensics1_analytic"]
    assert ana["non_reversal_is_robust"] is True
    # PPI never overtakes NRA under either tail.
    assert ana["upper_read_ppi_vs_nra"][0] < ana["upper_read_ppi_vs_nra"][1]
    assert (
        ana["corrected_tail_ppi_vs_nra"][0]
        < ana["corrected_tail_ppi_vs_nra"][1]
    )
    assert c1["candidate2_empirical"]["required_swap_realised"] is False


def test_q9_matches_forensics1_c1_numbers():
    c1 = _artifact()["q9_concept_cells"]["c1_binary"]["forensics1_analytic"]
    f1 = json.loads(FORENSICS1.read_text())["q5_tail_upper_read"]
    assert c1["upper_read_ppi_vs_nra"] == [
        f1["upper_read"]["ppi_savings_abs"],
        f1["upper_read"]["nra_savings_abs"],
    ]


# -- design implications label the four findings --------------------------
def test_candidate3_design_implications_present():
    impl = _artifact()["candidate3_design_implications"]
    assert set(impl) == {"q6", "q7", "q8", "q9"}
    assert "DIVORCE" in impl["q6"]  # the refined 65+ channel
    assert "AMENDMENT-2" in impl["q9"]


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
