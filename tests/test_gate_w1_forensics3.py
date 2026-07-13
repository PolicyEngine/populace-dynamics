"""Tests for W1 forensics 3 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the
forensics-3 artifact (``runs/gate_w1_forensics3_v1.json``), the committed
candidate gate artifacts (``runs/gate_w1_candidate{1,2,3}_v1.json``), and the
committed #117 encoding (``runs/m2_pseudo_projection_v1.json``); they never
rerun the diagnostic and need no PSID or frame, so they run in CI. The
PSID/frame-bound bit-identity rebuilds live in
``test_gate_w1_forensics3_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_forensics3_v1.json"
CANDIDATE3 = ROOT / "runs" / "gate_w1_candidate3_v1.json"
M2 = ROOT / "runs" / "m2_pseudo_projection_v1.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# -- provenance / registration --------------------------------------------
def test_reported_not_gated():
    a = _artifact()
    assert a["reported_not_gated"] is True
    assert a["schema_version"] == "gate_w1_forensics3.v1"
    assert a["gate"] == "gate_w1"


def test_registration_pointer_is_the_frozen_spec():
    a = _artifact()
    assert a["registration"]["comment_id"] == "4959668253"
    assert a["registration"]["issue"] == 42
    assert a["c3_grading_pointer"] == "4959658059"
    assert a["c3_registration_pointer"] == "4959017270"


def test_both_questions_present():
    a = _artifact()
    assert "q10_cap150k_adjacency" in a
    assert "q11_hhsize_residual" in a


# -- instrumentation bit-identity (both questions) ------------------------
def test_instrumentation_bit_identity_is_exact():
    r = _artifact()["reconciliations"]
    assert r["q10_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["q11_instrumentation_bit_identity_max_dev"] == 0.0
    assert r["all_instrumentation_reconciled"] is True


def test_q11_partition_sums_to_one():
    q11 = _artifact()["q11_hhsize_residual"]
    assert q11["partition_max_abs_deviation_from_one"] <= 1e-9
    for cfg in q11["configs"].values():
        s = sum(cfg["shares"].values())
        assert abs(s - 1.0) < 1e-9


# ==========================================================================
# Q10 -- cap_150k adjacency.
# ==========================================================================
def test_q10_rerun_reproduces_committed_cube_deltas_exactly():
    # MUTATION-CHECK: the re-run must reproduce ALL FOUR committed c2
    # exhaustion deltas (a wrong provision key / empty join finds 0 and fails).
    q10 = _artifact()["q10_cap150k_adjacency"]
    fid = q10["instrumentation_fidelity"]
    committed = json.loads(CANDIDATE3.read_text())["family_c"]["fingerprints"][
        "c2"
    ]["provision_deltas"]["our_exhaustion_deltas"]
    rerun = fid["rerun_exhaustion_deltas"]
    assert (
        set(rerun)
        == set(committed)
        == {
            "cap_150k",
            "elimination",
            "payroll_plus_1pp",
            "payroll_plus_2pp",
        }
    )
    for prov in committed:
        assert abs(rerun[prov] - committed[prov]) == 0.0
    assert fid["bit_identical"] is True


def test_q10_gap_decomposition_telescopes_and_a_dominates():
    gd = _artifact()["q10_cap150k_adjacency"]["gap_decomposition"]
    # MUTATION-CHECK: the three anchor points come straight from the committed
    # sources (Smith year-delta = 1, the m2/deployed cap deltas).
    assert gd["smith_cap_150k_years"] == 1.0
    m2_cap = next(
        p["exhaustion_delta_years"]
        for p in json.loads(M2.read_text())["provisions"]
        if p["provision"] == "cap_150k"
    )
    assert abs(gd["psid_anchor_frame_cap_150k_years"] - m2_cap) < 1e-9
    # telescopes: (a) + (b+c) == the total gap.
    recon = (
        gd["component_a_frame_above_cap_share_years"]
        + gd["component_bc_encoding_config_plus_vintage_years"]
    )
    assert abs(gd["gap_total_deployed_minus_smith"] - recon) < 1e-9
    # (a) the frame above-cap share dominates the gap.
    assert gd["dominant_component"] == "a_frame_above_cap_share"
    assert gd["share_a_frame_above_cap"] > 0.9


def test_q10_revenue_component_identities():
    # the EXACT ledger identities: the payroll increments MUST be the rate
    # increments (PV(payroll)==PV(B)); a broken frame join breaks this.
    rc = _artifact()["q10_cap150k_adjacency"]["revenue_components"]
    for frame in ("psid_anchor_frame", "deployed_representative_frame"):
        comp = rc[frame]
        assert comp["p1_delta_is_1pp"] is True
        assert comp["p2_delta_is_2pp"] is True
        assert abs(comp["d_bal_p1"] - 0.01) <= 1e-9
        assert abs(comp["d_bal_p2"] - 0.02) <= 1e-9
    # the compression correction raised A/B (non-vacuous: the two frames differ).
    assert (
        rc["deployed_representative_frame"]["A_over_B"]
        > rc["psid_anchor_frame"]["A_over_B"] + 0.05
    )
    assert rc["A_over_B_moved_by_correction"] is True


def test_q10_entailment_holds_on_deployed_frame():
    en = _artifact()["q10_cap150k_adjacency"]["entailment"]
    assert en["entailment_holds"] is True
    # the deployed frame ranks cap_150k SECOND (not last), breaking the Smith
    # 4-element adjacency.
    assert en["deployed_frame_order"][1] == "cap_150k"
    assert en["deployed_cap_rank"] == 2
    assert en["deployed_breaks_adjacency"] is True
    # the certified swap breakeven is the exact 2pp/rate constant.
    assert abs(en["c2_breakeven_A_over_B"] - 0.02 / 0.124) < 1e-12
    # the Smith adjacency window (at f_psid) is narrow.
    lo, hi = en["smith_adjacency_window_A_over_B_at_f_psid"]
    assert lo < hi
    assert en["smith_adjacency_window_width_at_f_psid"] < 0.05


def test_q10_construction_enumerates_empty():
    ca = _artifact()["q10_cap150k_adjacency"]["construction_attempt"]
    assert ca["restoration_exists"] is False
    # MUTATION-CHECK: the lever enumeration is populated and every lever is
    # dispositioned not-permitted (a vacuous empty list would pass silently).
    assert len(ca["levers_enumerated"]) >= 4
    assert all(lev["permitted"] is False for lev in ca["levers_enumerated"])


def test_q10_pair_scoped_respec_realised_three_of_three():
    rs = _artifact()["q10_cap150k_adjacency"]["pair_scoped_respec"]
    assert rs["realised_3_of_3"] is True
    assert set(rs["swap_realised_by_candidate"]) == {"c1", "c2", "c3"}
    assert all(rs["swap_realised_by_candidate"].values())
    assert rs["anchor_supported"] is True
    assert (
        rs[
            "any_published_anchor_supports_full_4_element_on_representative"
            "_frame"
        ]
        is False
    )


def test_q10_adjudication_three_subquestions():
    adj = _artifact()["q10_cap150k_adjacency"]["adjudication"]
    assert adj["q1_permitted_lever_restores_adjacency"]["answer"] is False
    assert adj["q2_cap_150k_concept_mismatched_entailment"]["answer"] is True
    assert "pair-scoped" in adj["q3_pair_scoped_respec"]["answer"]


# ==========================================================================
# Q11 -- hh_size residual.
# ==========================================================================
def test_q11_configs_present_and_toggle_structure():
    # MUTATION-CHECK: the four lever-3 toggle configs are present and the
    # baseline (no lever 3) has NO terminal coresident parent.
    q11 = _artifact()["q11_hhsize_residual"]
    assert set(q11["configs"]) == {
        "c0_none",
        "cor_only",
        "fer_only",
        "joint_c3",
    }
    assert q11["configs"]["c0_none"]["terminal_coresident_parent_rate"] == 0.0
    assert q11["configs"]["cor_only"]["terminal_coresident_parent_rate"] > 0.0


def test_q11_channel_decomposition_telescopes():
    # MUTATION-CHECK: per-cell channels reconstruct the total lever-3 effect,
    # and all five hh_size cells are present (a wrong prefix finds none).
    pc = _artifact()["q11_hhsize_residual"]["per_cell_attribution"]
    assert set(pc) == {"1", "2", "3", "4", "5plus"}
    for cell in pc.values():
        recon = (
            cell["coresidence_main_effect"]
            + cell["fertility_main_effect"]
            + cell["marital_core_interaction"]
        )
        assert abs(cell["total_lever3_effect"] - recon) < 1e-9
        # residual + joint == frame (definition check).
        assert (
            abs(
                cell["residual_after_c3_rate_a_minus_joint"]
                + cell["joint_c3"]
                - cell["rate_a_frame"]
            )
            < 1e-9
        )


def test_q11_hh_size_matches_committed_cube_frame_rates():
    # MUTATION-CHECK: the frame rate_a per cell equals the committed c3 cube's
    # rate_a (guards against reading the wrong cell).
    pc = _artifact()["q11_hhsize_residual"]["per_cell_attribution"]
    pc0 = json.loads(CANDIDATE3.read_text())["family_a"]["per_seed"][0][
        "per_cell"
    ]
    for c in ("1", "2", "3", "4", "5plus"):
        assert (
            abs(pc[c]["rate_a_frame"] - pc0[f"hh_size_share.{c}"]["rate_a"])
            < 1e-9
        )


def test_q11_channel_ownership_mixed_but_structured():
    # HONEST refinement: coresidence does NOT own >50% of every failing cell.
    # It owns size-3 and the size-1<->size-3 mirror; fertility loads
    # size-4/5plus. Structured, not featureless.
    q11 = _artifact()["q11_hhsize_residual"]
    f = q11["finding"]
    assert f["coresidence_dominant_all_failing_cells"] is False
    assert f["attribution"] == "mixed_but_structured"
    co = q11["channel_ownership"]
    # coresidence owns size-3; fertility owns the large-size deficits.
    assert "3" in co["coresidence_dominant_cells"]
    assert "4" in co["fertility_dominant_cells"]
    assert "5plus" in co["fertility_dominant_cells"]
    # the two ownership sets partition the failing quad (mutation-check: no
    # cell is claimed by both / neither).
    assert set(co["coresidence_dominant_cells"]) | set(
        co["fertility_dominant_cells"]
    ) == {"1", "3", "4", "5plus"}
    assert (
        set(co["coresidence_dominant_cells"])
        & set(co["fertility_dominant_cells"])
        == set()
    )
    # coresidence dominates size-3 by more than half; fertility dominates
    # size-5plus (coresidence share well below half there).
    assert (
        q11["per_cell_attribution"]["3"]["dominant_channel"] == "coresidence"
    )
    assert (
        q11["per_cell_attribution"]["5plus"]["dominant_channel"] == "fertility"
    )
    assert co["coresidence_share_of_moved_by_cell"]["3"] > 0.5
    assert co["coresidence_share_of_moved_by_cell"]["5plus"] < 0.5


def test_q11_mirror_structure():
    m = _artifact()["q11_hhsize_residual"]["mirror_structure"]
    # size-1 EXCESS mirrors size-3+ DEFICIT (the coresidence signature).
    assert m["size1_c3_excess_vs_frame"] > 0
    assert m["size3plus_c3_deficit_vs_frame"] < 0
    # partition: they balance (up to size-2's small excess).
    assert abs(m["excess_deficit_balance"]) < 0.06
    # the coresidence channel moves size-1 DOWN and size-3+ UP (mirror).
    assert m["coresidence_moves_size1"] < 0
    assert m["coresidence_moves_size3plus"] > 0
    assert m["coresidence_is_mirror_structured"] is True


def test_q11_candidate_independence_verdict():
    ci = _artifact()["q11_hhsize_residual"]["candidate_independence"]
    assert ci["verdict"] in ("levers_exhausted", "untried_lever")
    # MUTATION-CHECK: the exact failing quad is recorded with its seed-pass
    # counts (1/3/5plus fail 5/5, size-4 fails 4/5).
    quad = ci["exact_quad_failing_cells"]
    assert set(quad) == {"1", "3", "4", "5plus"}
    assert quad["1"]["n_seed_pass"] == 0
    assert quad["3"]["n_seed_pass"] == 0
    assert quad["5plus"]["n_seed_pass"] == 0
    assert quad["4"]["n_seed_pass"] == 1
    # the roster ceiling probe was run.
    assert "untried_full_age_roster_probe" in ci


# -- expectations vs findings (the registration's graded probabilities) ----
def test_expectations_vs_findings_present():
    e = _artifact()["expectations_vs_findings"]
    assert e["q10"]["registered"]["entailment_holds_and_a_dominant"] == 0.6
    assert (
        e["q11"]["registered"]["coresidence_dominant_gt_half_with_mirror"]
        == 0.7
    )
    assert e["q10"]["found"]["entailment_holds"] is True


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
