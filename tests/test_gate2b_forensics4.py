"""Tests for gate-2b forensics 4 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the forensics-4
artifact (``runs/gate2b_forensics4_v1.json``), the committed candidate-6 gate
artifact (``runs/gate2b_hazard_v6.json``) and the forensics-3 artifact; they
never rerun the diagnostic and need no PSID, so they run in CI. They audit that
every headline recomputes from the stored per-seed values and that the three
frozen questions hold: Q11's five-channel telescoping decomposition of the
linked-father overshoot reconciles to the cell miss at machine epsilon with the
EXISTENCE channel a structural zero; Q12's delta-3-off component replay is inert
at the target and the spouse cell is inherited-fragile; and Q13's marginal seed
is systematic structure, not draw noise.

The PSID reproduction pins (rebuild seed 0's train-side inputs live and match the
recorded decompositions + the instrumented-draw fidelity to float precision,
skipped when the PSID relationship matrix is absent) live in
``tests/test_gate2b_forensics4_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics4_v1.json"
CANDIDATE6 = ROOT / "runs" / "gate2b_hazard_v6.json"

RECON_ATOL = 1e-9
EPS_ATOL = 1e-12
Q11_CELLS = (
    "coresident_child.25-34|male",
    "coresident_child.35-44|male",
)
Q12_CELL = "coresident_spouse.25-34|female"
Q13_CELL = "hh_size.5+"
Q13_MARGINAL_SEED = 3
CHANNELS = (
    "existence_identical_exposure",
    "spell_length",
    "marital_state_joint",
    "unenumerated_nonjoinable_supply",
    "supply_residual_nonlinked_coresidence",
    "shadow_unlinked_channel",
    "monte_carlo_finite_draw_gap",
)
LINKED_SUB_CHANNELS = (
    "spell_length",
    "marital_state_joint",
    "unenumerated_nonjoinable_supply",
    "supply_residual_nonlinked_coresidence",
)
EPISODE_BUCKETS = ("1", "2", "3", "4+")


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2b_forensics4.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4947226688")
    assert a["registration_pointer"] == "4947226688"
    assert a["grading_pointer"] == "4947225286"
    for block in (
        "question_11_linked_father_child_supply",
        "question_12_spouse_25_34_female_movement",
        "question_13_hh_size_5plus_marginal_seed",
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
    assert "household_composition_sim_v6" in p["fit_simulate_machinery"]
    # Q11 does NOT relitigate custody -- supply and spells only.
    assert "does NOT" in p["reused_machinery"]
    assert "custodial_prob_v6" in p["reused_machinery"]


def test_instrumentation_is_bit_identical_to_committed_draw():
    fid = _artifact()["protocol"]["instrumentation_fidelity"]
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v6"] == 0.0
    assert fid["child_channel_additivity_residual"] == 0
    assert fid["linked_marital_split_additivity_residual"] == 0


def test_reconciliations_are_identities_at_machine_epsilon():
    r = _artifact()["reconciliations"]
    eps = r["float64_machine_epsilon"]
    assert r["instrumentation_bit_identity_max_rate_deviation"] == 0.0
    assert r["child_channel_additivity_residual"] == 0
    assert r["linked_marital_split_additivity_residual"] == 0
    # The five-channel telescoping reconciliation of the cell miss is an
    # identity up to float summation order (a few ULP of float64).
    assert r["q11_cell_miss_reconciliation_max_abs_remainder"] <= 16 * eps
    assert r["all_identity_reconciliations_at_machine_epsilon"] is True


def test_no_gate_verdict_written_and_candidate6_named():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2b_pass" not in a
    assert a["candidate6_artifact"] == "runs/gate2b_hazard_v6.json"
    assert "candidate 6" in a["candidate_under_diagnosis"]
    assert a["candidate_7_implications"]


def test_per_seed_covers_the_five_gate_seeds():
    per_seed = _artifact()["per_seed"]
    assert sorted(s["seed"] for s in per_seed) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# Q11 -- linked-father child supply (existence vs spell)
# --------------------------------------------------------------------------
def test_q11_registered_cells_present():
    q11 = _artifact()["question_11_linked_father_child_supply"]
    assert tuple(q11["registered_cells"]) == Q11_CELLS
    assert set(q11["per_cell"]) == set(Q11_CELLS)


def test_q11_existence_is_a_structural_zero_by_construction():
    # The cah85 linked exposure is the same data on both sides and the sim
    # re-labels states on the same father-wave grid, so the exposed-linked-child
    # counts are identical -- the existence channel is EXACTLY zero.
    q11 = _artifact()["question_11_linked_father_child_supply"]
    for cell in Q11_CELLS:
        rec = q11["per_cell"][cell]
        # The load-bearing claim: the existence channel is an EXACT structural
        # zero (same model.father_links exposure data, same father-wave grid).
        assert rec["channels"]["existence_identical_exposure"] == 0.0
        exist = rec["existence_distributions"]
        # The exposed-linked-child distributions essentially coincide (both over
        # the same committed exposure; a tiny boundary difference is possible
        # between the sim linked mask and the reference linked membership).
        assert exist["sim_exposed_linked_count"] == pytest.approx(
            exist["reference_exposed_linked_count"], abs=0.03
        )


def test_q11_channels_reconcile_to_cell_miss_exactly():
    q11 = _artifact()["question_11_linked_father_child_supply"]
    for cell in Q11_CELLS:
        rec = q11["per_cell"][cell]
        ch = rec["channels"]
        assert set(ch) == set(CHANNELS)
        # The cell miss is the sim-minus-reference male full rate.
        assert rec["cell_miss_sim_minus_reference"] == pytest.approx(
            rec["sim_male_full_rate_train"]
            - rec["reference_male_full_rate_train"],
            abs=RECON_ATOL,
        )
        # The six channels (existence == 0) reconstruct the miss.
        assert rec["channel_reconstruction_sum"] == pytest.approx(
            sum(ch.values()), abs=RECON_ATOL
        )
        assert rec["channel_reconstruction_sum"] == pytest.approx(
            rec["cell_miss_sim_minus_reference"], abs=RECON_ATOL
        )
        assert abs(rec["reconciliation_remainder"]) < 1e-9
    assert q11["reconciliation_max_abs_remainder"] < 1e-9


def test_q11_both_cells_over_produce_and_shadow_is_a_named_channel():
    q11 = _artifact()["question_11_linked_father_child_supply"]
    for cell in Q11_CELLS:
        rec = q11["per_cell"][cell]
        # Both male cells over-produce (positive miss).
        assert rec["cell_miss_sim_minus_reference"] > 0.0
        # The dominant LINKED sub-channel is one of the occupancy channels; the
        # shadow (unlinked paternal) channel is surfaced separately.
        dom = q11["dominant_linked_channel_per_cell"][cell]
        assert dom["dominant_linked_channel"] in LINKED_SUB_CHANNELS
        assert "shadow_channel" in dom
        assert "linked_channel_total" in dom


def test_q11_alignment_uses_committed_father_links_basis():
    # The decomposition must target the COMMITTED mechanism: the sim draws over
    # model.father_links (father_link_births), NOT the enumerated-only subset.
    a = _artifact()
    ad = a["q11_alignment_decision"]
    assert "model.father_links" in ad["resolution"]
    assert "father_link_births" in ad["issue"]
    q11 = a["question_11_linked_father_child_supply"]
    # The unenumerated non-joinable supply channel is a real, non-negative
    # over-attribution -- the alignment fix made visible.
    for cell in Q11_CELLS:
        ch = q11["per_cell"][cell]["channels"]
        assert "unenumerated_nonjoinable_supply" in ch
        assert ch["unenumerated_nonjoinable_supply"] >= 0.0


def test_q11_spell_signature_sim_episodes_shorter_than_reference():
    # The occupancy-vs-episode signature: the faithful per-wave probability
    # applied as an INDEPENDENT per-wave occupancy fragments coresidence into
    # shorter, more numerous spells than the observed contiguous episodes.
    q11 = _artifact()["question_11_linked_father_child_supply"]
    epi = q11["episode_length_distributions"]
    for side in ("sim", "reference"):
        dist = epi[side]["distribution"]
        assert set(dist) == set(EPISODE_BUCKETS)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
    assert (
        epi["sim"]["mean_episode_length"]
        < epi["reference"]["mean_episode_length"]
    )
    # Sim over-weights single-wave spells; reference over-weights 4+.
    assert (
        epi["sim"]["distribution"]["1"] > epi["reference"]["distribution"]["1"]
    )
    assert (
        epi["sim"]["distribution"]["4+"]
        < epi["reference"]["distribution"]["4+"]
    )


def test_q11_spell_channel_present_and_flagged_per_cell():
    # The spell-length channel (the registration's named mechanism) is measured
    # per cell with a boolean flag for whether it is the largest linked channel;
    # the data decides which channel dominates (not hardcoded).
    q11 = _artifact()["question_11_linked_father_child_supply"]
    for cell in Q11_CELLS:
        dom = q11["dominant_linked_channel_per_cell"][cell]
        assert isinstance(dom["spell_is_largest_linked_channel"], bool)
        assert "spell_length" in q11["per_cell"][cell]["channels"]


def test_q11_finding_mentions_faithful_custody_not_relitigated():
    q11 = _artifact()["question_11_linked_father_child_supply"]
    assert "custodial_prob_v6" in q11["finding"]
    assert "not re-estimated" in q11["finding"]
    assert "model.father_links" in q11["finding"]


# --------------------------------------------------------------------------
# Q12 -- spouse 25-34|female movement attribution
# --------------------------------------------------------------------------
def test_q12_delta3_is_inert_at_the_target_cell():
    q12 = _artifact()["question_12_spouse_25_34_female_movement"]
    assert q12["registered_cell"] == Q12_CELL
    # Disabling delta 3 (the female cohab override) does not move the 25-34
    # female full spouse rate materially.
    assert abs(q12["delta3_target_effect_full"]) < 1e-6
    assert q12["delta3_inert_at_target"] is True
    band = q12["per_female_band_component_replay"]["25-34"]
    assert band["delta3_effect_full"] == pytest.approx(
        q12["delta3_target_effect_full"], abs=EPS_ATOL
    )
    # Delta 3 DOES carry new structure at 35-44|female (larger than at 25-34).
    band44 = q12["per_female_band_component_replay"]["35-44"]
    assert abs(band44["delta3_effect_full"]) > abs(band["delta3_effect_full"])


def test_q12_component_replay_partitions_the_spouse_rate():
    q12 = _artifact()["question_12_spouse_25_34_female_movement"]
    for _b, rec in q12["per_female_band_component_replay"].items():
        # legal + cohab-overlay + lr-overlay reconstruct the full c6 rate.
        assert rec["c6_full"] == pytest.approx(
            rec["c6_legal"]
            + rec["c6_cohab_overlay"]
            + rec["c6_legal_residual_overlay"],
            abs=RECON_ATOL,
        )


def test_q12_stability_is_fragile_inherited_not_a_c6_interaction():
    q12 = _artifact()["question_12_spouse_25_34_female_movement"]
    assert (
        q12["stability_verdict"]
        == "fragile_marginal_inherited_not_c6_interaction"
    )
    # Thin margin: the committed holdout seed-mean clears by a small amount and
    # at least one split seed exceeds tolerance.
    assert q12["holdout_n_seeds_over_tolerance"] >= 1
    committed = q12["holdout_committed_candidate6"]
    assert committed["seed_mean_score"] < committed["tolerance"]
    assert q12["holdout_seed_mean_margin_under_tolerance"] == pytest.approx(
        committed["tolerance"] - committed["seed_mean_score"], abs=RECON_ATOL
    )


# --------------------------------------------------------------------------
# Q13 -- hh_size.5+ marginal seed (noise vs structure)
# --------------------------------------------------------------------------
def test_q13_marginal_seed_is_structure_not_noise():
    q13 = _artifact()["question_13_hh_size_5plus_marginal_seed"]
    assert q13["registered_cell"] == Q13_CELL
    assert q13["marginal_seed"] == Q13_MARGINAL_SEED
    disp = q13["holdout_marginal_seed_dispersion"]
    # Systematic tilt: every per-draw rate on the same (under) side, the spread
    # does NOT straddle zero, and the seed-mean sits well past the tolerance.
    assert disp["all_draws_same_sign"] is True
    assert disp["per_draw_spread_straddles_zero"] is False
    assert disp["n_draws_over_tolerance_over_side"] == 0
    assert disp["standard_errors_past_line"] >= 2.0
    assert q13["verdict_structure_not_noise"] is True


def test_q13_component_is_the_upstream_large_family_core_deficit():
    q13 = _artifact()["question_13_hh_size_5plus_marginal_seed"]
    assert q13["component"] == "upstream_core_large_family_fertility_deficit"
    corr = q13["train_side_corroboration"]
    # The sim core-5+ under-produces the reference on average.
    assert corr["mean_core5plus_deficit_sim_minus_reference"] < 0.0
    for rec in corr["per_seed"]:
        assert rec["core5plus_deficit_sim_minus_ref"] == pytest.approx(
            rec["sim_core5plus"] - rec["reference_core5plus"], abs=RECON_ATOL
        )


def test_q13_marginal_seed_dispersion_recomputes_from_committed_holdout():
    # The dispersion headline is a faithful read of the committed candidate-6
    # holdout per-draw rates for the marginal seed (side A never re-simulated).
    a = _artifact()
    q13 = a["question_13_hh_size_5plus_marginal_seed"]
    disp = q13["holdout_marginal_seed_dispersion"]
    c6 = json.loads(CANDIDATE6.read_text())
    seed3 = next(s for s in c6["per_seed"] if s["seed"] == Q13_MARGINAL_SEED)[
        "gated_cells"
    ][Q13_CELL]
    rate_a = float(seed3["rate_a"])
    lns = [math.log(float(r) / rate_a) for r in seed3["per_draw_rate"]]
    assert disp["rate_a"] == pytest.approx(rate_a, abs=EPS_ATOL)
    assert disp["per_draw_signed_logratio_min"] == pytest.approx(
        min(lns), abs=RECON_ATOL
    )
    assert disp["per_draw_signed_logratio_max"] == pytest.approx(
        max(lns), abs=RECON_ATOL
    )
    assert disp["tolerance"] == pytest.approx(float(seed3["tolerance"]))


# --------------------------------------------------------------------------
# Cross-cutting recomputation + global guards
# --------------------------------------------------------------------------
def test_q11_sim_male_full_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    for cell in Q11_CELLS:
        recomputed = np.mean(
            [s["q11_sim"][cell]["male_full"] for s in per_seed]
        )
        assert a["question_11_linked_father_child_supply"]["per_cell"][cell][
            "sim_male_full_rate_train"
        ] == pytest.approx(float(recomputed), abs=RECON_ATOL)


def test_q13_reference_hh5plus_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    tot = np.mean([s["q13_reference_hh5plus"] for s in per_seed])
    corr = a["question_13_hh_size_5plus_marginal_seed"][
        "train_side_corroboration"
    ]["per_seed"]
    per_seed_ref = np.mean([r["reference_hh5plus"] for r in corr])
    assert float(per_seed_ref) == pytest.approx(float(tot), abs=RECON_ATOL)


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


def test_holdout_committed_reads_candidate6_only():
    # Q11/Q12/Q13 cite the committed candidate-6 side-A cells (never
    # re-simulated); the tolerances match the locked contract.
    a = _artifact()
    c6 = json.loads(CANDIDATE6.read_text())
    q11_cell = a["question_11_linked_father_child_supply"]["per_cell"][
        "coresident_child.35-44|male"
    ]["holdout_committed_candidate6"]
    q12_cell = a["question_12_spouse_25_34_female_movement"][
        "holdout_committed_candidate6"
    ]
    q13_cell = a["question_13_hh_size_5plus_marginal_seed"][
        "holdout_committed_candidate6"
    ]
    g0 = c6["per_seed"][0]["gated_cells"]
    assert q11_cell["tolerance"] == pytest.approx(
        float(g0["coresident_child.35-44|male"]["tolerance"])
    )
    assert q12_cell["tolerance"] == pytest.approx(
        float(g0[Q12_CELL]["tolerance"])
    )
    assert q13_cell["tolerance"] == pytest.approx(
        float(g0[Q13_CELL]["tolerance"])
    )
