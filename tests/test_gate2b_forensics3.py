"""Tests for gate-2b forensics 3 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the forensics-3
artifact (``runs/gate2b_forensics3_v1.json``), the committed candidate-5 gate
artifact (``runs/gate2b_hazard_v5.json``) and ``gates.yaml``; they never rerun
the diagnostic and need no PSID, so they run in CI. They audit that every
headline recomputes from the stored per-seed values and that each of the three
endgame decompositions reconciles to (machine-)zero: the Q8 four-channel
attribution partition + the 0-4 basis inversion + the 45-54|female
retention-vs-aging-out split, the Q9 legal-core/cohabitation-overlay/legal-
residual component partition + the female-band residual enumeration, and the
Q10 (core size, non-core count) train joint + the honest-joint counterfactual.

The PSID reproduction pins (rebuild the train-side inputs live and match seed 0's
deterministic decompositions and the instrumented-draw fidelity to float
precision, skipped when the PSID relationship matrix is absent) live in
``tests/test_gate2b_forensics3_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics3_v1.json"
CANDIDATE5 = ROOT / "runs" / "gate2b_hazard_v5.json"

RECON_ATOL = 1e-9
EPS_ATOL = 1e-12
CHILD_CHANNELS = ("maternal", "linked_married", "linked_not_married", "shadow")
Q8_CELLS = (
    "coresident_child.15-24|male",
    "coresident_child.35-44|male",
    "coresident_child.45-54|female",
)
Q9_CELL = "coresident_spouse.25-34|female"
FEMALE_BANDS = ("15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2b_forensics3.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4945702151")
    assert a["registration_pointer"] == "4945702151"
    assert a["grading_pointer"] == "4945697846"
    for block in (
        "question_8_child_cell_triage",
        "question_9_spouse_25_34_female_decomposition",
        "question_10_hh_size_3_5plus_joint_constraint",
    ):
        assert block in a
        assert a[block]["finding"]
        assert a[block]["question"]
        assert a[block]["method"]


def test_protocol_is_train_side_only_and_holdout_untouched():
    p = _artifact()["protocol"]
    assert p["train_side_only"] is True
    assert "never" in p["outer_holdout_contact"].lower()
    assert "re-simulated" in p["outer_holdout_contact"].lower()
    assert p["gate_seeds"] == [0, 1, 2, 3, 4]
    assert p["n_draws"] == 20
    assert "side B only" in p["no_holdout_tuning_surface"]
    assert "household_composition_sim_v5" in p["fit_simulate_machinery"]
    # The forensics-1 spouse splitter and forensics-2 custody bases are reused.
    assert "spouse_concept_codes" in p["reused_machinery"]
    assert "q5_custodial_selection" in p["reused_machinery"]


def test_instrumentation_is_bit_identical_to_committed_draw():
    fid = _artifact()["protocol"]["instrumentation_fidelity"]
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v5"] == 0.0
    assert fid["child_channel_additivity_residual"] == 0
    assert fid["linked_marital_split_additivity_residual"] == 0


def test_reconciliations_are_identities_at_machine_epsilon():
    r = _artifact()["reconciliations"]
    eps = r["float64_machine_epsilon"]
    # Bit-identity and the integer channel additivities are EXACTLY zero.
    assert r["instrumentation_bit_identity_max_rate_deviation"] == 0.0
    assert r["child_channel_additivity_residual"] == 0
    assert r["linked_marital_split_additivity_residual"] == 0
    # The additive-share remainders sit at machine epsilon (~eps/2, one ULP).
    assert r["q8_channel_reconciliation_max_abs_remainder"] <= eps
    assert r["q9_spouse_component_reconciliation_max_abs_remainder"] <= eps
    assert r["all_identity_reconciliations_at_machine_epsilon"] is True


def test_no_gate_verdict_written_and_candidate5_named():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2b_pass" not in a
    assert a["candidate5_artifact"] == "runs/gate2b_hazard_v5.json"
    assert "candidate 5" in a["candidate_under_diagnosis"]
    assert a["candidate_6_implications"]


def test_per_seed_covers_the_five_gate_seeds():
    per_seed = _artifact()["per_seed"]
    assert sorted(s["seed"] for s in per_seed) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# Q8 -- child cell triage
# --------------------------------------------------------------------------
def test_q8_registered_cells_present_and_flagged():
    q8 = _artifact()["question_8_child_cell_triage"]
    assert tuple(q8["registered_cells"]) == Q8_CELLS
    for cell in Q8_CELLS:
        assert q8["per_cell"][cell]["is_registered_q8_cell"] is True


def test_q8_channels_partition_each_cell_rate():
    q8 = _artifact()["question_8_child_cell_triage"]
    for _cell, rec in q8["per_cell"].items():
        chans = rec["channel_contributions"]
        assert set(chans) == set(CHILD_CHANNELS)
        assert rec["sim_full_rate_train"] == pytest.approx(
            sum(chans.values()), abs=RECON_ATOL
        )
        assert abs(rec["channel_reconciliation_remainder"]) < EPS_ATOL
        assert rec["miss_sim_minus_reference"] == pytest.approx(
            rec["sim_full_rate_train"] - rec["reference_rate_train"],
            abs=RECON_ATOL,
        )


def test_q8_15_24_male_dominated_by_linked_channel():
    # Young fathers coreside with their (0-4) linked children; the shadow
    # (unlinked-man) channel is near-zero and there is no maternal mass.
    rec = _artifact()["question_8_child_cell_triage"]["per_cell"][
        "coresident_child.15-24|male"
    ]
    ch = rec["channel_contributions"]
    assert ch["maternal"] == 0.0
    linked = ch["linked_married"] + ch["linked_not_married"]
    assert linked > 5 * ch["shadow"]
    assert rec["miss_sim_minus_reference"] > 0.0  # over-produced


def test_q8_45_54_female_is_entirely_maternal_and_over_produced():
    q8 = _artifact()["question_8_child_cell_triage"]
    rec = q8["per_cell"]["coresident_child.45-54|female"]
    ch = rec["channel_contributions"]
    assert ch["linked_married"] == 0.0
    assert ch["linked_not_married"] == 0.0
    assert ch["shadow"] == 0.0
    assert ch["maternal"] == pytest.approx(
        rec["sim_full_rate_train"], abs=RECON_ATOL
    )
    assert rec["miss_sim_minus_reference"] > 0.0


def test_q8_0_4_basis_is_a_denominator_artifact_via_sign_inversion():
    # The registered adjudication: child-record HIGHER than observable at 0-4
    # but LOWER at school ages -- a sign inversion that marks the 0-4 gap as a
    # selective-enumeration denominator artifact of the child-record join.
    adj = _artifact()["question_8_child_cell_triage"][
        "adjudication_0_4_basis_15_24_male"
    ]
    assert adj["child_record_higher_at_0_4_confirmed"] is True
    assert adj["child_record_lower_at_school_ages_5_17"] is True
    assert adj["sign_inverts_between_0_4_and_school_ages"] is True
    assert adj["not_married_0_4"]["child_record_minus_observable"] > 0.0
    assert adj["not_married_5_12"]["child_record_minus_observable"] < 0.0
    assert adj["not_married_13_17"]["child_record_minus_observable"] < 0.0
    assert "observable" in adj["verdict"].lower()


def test_q8_45_54_female_is_aging_out_not_retention_shortfall():
    adj = _artifact()["question_8_child_cell_triage"][
        "adjudication_retention_vs_aging_out_45_54_female"
    ]
    assert adj["cell_is_over_produced"] is True
    ent = adj["cell_is_entirely_maternal_channel"]
    assert ent["non_maternal_contribution"] == 0.0
    # The over-production concentrates in adult (18+) children, not minors.
    assert (
        adj["adult_present_gap_sim_minus_reference"]
        > adj["minor_present_gap_sim_minus_reference"]
    )
    # The reference coupling at 45-54 is weak -- far below the ~5x at 55+ the
    # delta-1 coupling targeted -- so it must NOT extend downward.
    lift = adj["reference_coupling_signature"][
        "reference_coupling_lift_ratio_joint_over_product"
    ]
    assert lift < 3.0


# --------------------------------------------------------------------------
# Q9 -- spouse 25-34|female
# --------------------------------------------------------------------------
def test_q9_female_band_components_reconcile():
    q9 = _artifact()["question_9_spouse_25_34_female_decomposition"]
    assert q9["registered_cell"] == Q9_CELL
    for _cell, rec in q9["per_female_band"].items():
        assert rec["sim_full_rate_train"] == pytest.approx(
            rec["sim_legal_core"]
            + rec["sim_cohab_overlay_contribution"]
            + rec["sim_legal_residual_overlay_contribution"],
            abs=RECON_ATOL,
        )
        assert (
            abs(rec["reconciliation_remainder_full_vs_components"]) < EPS_ATOL
        )
        assert rec["residual_miss_full_minus_reference"] == pytest.approx(
            rec["sim_full_rate_train"] - rec["reference_full_rate_train"],
            abs=RECON_ATOL,
        )


def test_q9_female_residual_enumeration_covers_all_bands():
    q9 = _artifact()["question_9_spouse_25_34_female_decomposition"]
    enum = q9["female_band_residual_enumeration"]
    assert set(enum) == {f"coresident_spouse.{b}|female" for b in FEMALE_BANDS}
    for _cell, rec in enum.items():
        assert rec["direction"] in ("under", "over")
        assert (rec["residual_miss"] < 0) == (rec["direction"] == "under")


def test_q9_25_34_female_is_overlay_shortfall_not_under_produced_legal():
    # The prior-contradicting headline: the legal core is NOT the shortfall
    # (legal gap >= 0 or small); the cohabitation overlay under-supplies.
    q9 = _artifact()["question_9_spouse_25_34_female_decomposition"]
    assert q9["target_cell_classification"] == "overlay_cohabitation_shortfall"
    assert q9["target_cell_overlay_gap"] < 0.0
    assert abs(q9["target_cell_overlay_gap"]) >= 2.0 * abs(
        q9["target_cell_legal_gap"]
    )
    tgt = q9["per_female_band"][Q9_CELL]
    assert tgt["residual_direction"] == "under"
    # The c4 legal top-up left 25-34|female essentially untreated.
    assert tgt["legal_residual_overlay_treated_this_band"] is False


def test_q9_registration_cited_prior_note_present():
    q9 = _artifact()["question_9_spouse_25_34_female_decomposition"]
    note = q9["registration_cited_prior_note"]
    assert "0.031" in note
    assert "65-74|female" in note


# --------------------------------------------------------------------------
# Q10 -- the hh_size 3<->5+ joint constraint
# --------------------------------------------------------------------------
def test_q10_registered_cells_and_distributions_sum_to_one():
    q10 = _artifact()["question_10_hh_size_3_5plus_joint_constraint"]
    assert tuple(q10["registered_cells"]) == (
        "hh_size.3",
        "hh_size.4",
        "hh_size.5+",
    )
    for key in (
        "reference_hh_size_distribution",
        "sim_c5_hh_size_distribution",
    ):
        assert sum(q10[key].values()) == pytest.approx(1.0, abs=1e-3)


def test_q10_honest_joint_counterfactual_structure_and_sanity():
    q10 = _artifact()["question_10_hh_size_3_5plus_joint_constraint"]
    cf = q10["honest_joint_counterfactual"]
    # The counterfactual hh distribution sums to ~1 and covers the 3 cells.
    assert sum(cf["implied_hh_from_sim_core"].values()) == pytest.approx(
        1.0, abs=1e-3
    )
    assert set(cf["per_cell"]) == {"hh_size.3", "hh_size.4", "hh_size.5+"}
    # The reference-core sanity convolution reproduces the reference hh_size up
    # to the small CORE_SIZE_CAP pooling remainder.
    assert cf["reference_core_sanity_max_abs_dev"] < 5e-3
    for _cell, rec in cf["per_cell"].items():
        assert rec["counterfactual_signed_logratio"] == pytest.approx(
            np.log(
                rec["counterfactual_honest_joint_on_sim_core"]
                / rec["reference_train"]
            ),
            abs=RECON_ATOL,
        )


def test_q10_c5_fails_size3_over_and_size5plus_under():
    # The trade-off direction: c5 over-produces size-3 and under-produces
    # size-5+ (the two hh_size mechanisms trading off).
    q10 = _artifact()["question_10_hh_size_3_5plus_joint_constraint"]
    cc = q10["honest_joint_counterfactual"]["per_cell"]
    assert cc["hh_size.3"]["sim_c5_signed_logratio"] > 0.0
    assert cc["hh_size.5+"]["sim_c5_signed_logratio"] < 0.0


def test_q10_core_under_produces_large_families():
    # The upstream (non-binding) signal: the sim core under-produces large
    # cores, traceable to a 3+-own-child fertility deficit.
    q10 = _artifact()["question_10_hh_size_3_5plus_joint_constraint"]
    assert q10["core_5plus_deficit_sim_minus_reference"] < 0.0
    sim = q10["sim_c5_child_count_distribution"]
    ref = q10["reference_own_child_count_distribution"]
    sim_3plus = sim["3"] + sim["4+"]
    ref_3plus = ref["3"] + ref["4+"]
    assert sim_3plus < ref_3plus


# --------------------------------------------------------------------------
# Cross-cutting recomputation + global guards
# --------------------------------------------------------------------------
def test_q8_full_rate_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    for cell in Q8_CELLS:
        recomputed = np.mean([s["q8_sim_full_mean"][cell] for s in per_seed])
        assert a["question_8_child_cell_triage"]["per_cell"][cell][
            "sim_full_rate_train"
        ] == pytest.approx(float(recomputed), abs=RECON_ATOL)


def test_q10_sim_hh3_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    tot = np.mean([s["q10_sim_hh_distribution"]["3"] for s in per_seed])
    assert a["question_10_hh_size_3_5plus_joint_constraint"][
        "sim_c5_hh_size_distribution"
    ]["3"] == pytest.approx(float(tot), abs=RECON_ATOL)


def test_all_identity_remainder_fields_are_negligible():
    # Every reconciliation IDENTITY remainder is machine-negligible. The Q10
    # honest-joint "pooling_remainder" is deliberately excluded: it is NOT an
    # identity but a CORE_SIZE_CAP=5 pooling approximation (cores 6-8 sharing
    # the core-5 conditional), documented as such and bounded (< 5e-3) by
    # test_q10_honest_joint_counterfactual_structure_and_sanity.
    a = _artifact()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "pooling_remainder" in k:
                    continue
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


def test_holdout_committed_reads_candidate5_only():
    # Q8/Q9/Q10 cite the committed candidate-5 side-A cells (never
    # re-simulated); the tolerances match the locked contract.
    a = _artifact()
    c5 = json.loads(CANDIDATE5.read_text())
    q8_cell = a["question_8_child_cell_triage"]["per_cell"][
        "coresident_child.15-24|male"
    ]["holdout_committed_candidate5"]
    q9_cell = a["question_9_spouse_25_34_female_decomposition"][
        "holdout_committed_candidate5"
    ]
    q10_cells = a["question_10_hh_size_3_5plus_joint_constraint"][
        "holdout_committed_candidate5"
    ]
    committed_c = c5["per_seed"][0]["gated_cells"][
        "coresident_child.15-24|male"
    ]["tolerance"]
    committed_s = c5["per_seed"][0]["gated_cells"][Q9_CELL]["tolerance"]
    committed_h = c5["per_seed"][0]["gated_cells"]["hh_size.5+"]["tolerance"]
    assert q8_cell["tolerance"] == pytest.approx(committed_c)
    assert q9_cell["tolerance"] == pytest.approx(committed_s)
    assert q10_cells["hh_size.5+"]["tolerance"] == pytest.approx(committed_h)
