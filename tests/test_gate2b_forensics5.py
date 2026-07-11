"""Tests for gate-2b forensics 5 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the forensics-5
artifact (``runs/gate2b_forensics5_v1.json``) and the committed candidate-7 gate
artifact (``runs/gate2b_hazard_v7.json``); they never rerun the diagnostic and
need no PSID, so they run in CI. They audit that every headline recomputes from
the stored per-seed values and that the three frozen questions hold: Q14's
four-channel (fertility / exit / link-coverage / v7-interaction) Oaxaca-Blinder
decomposition of the older-parent supply miss reconciles to the cell miss at
machine epsilon with fertility-origin the completed-family-size shortfall; Q15's
single-lever convergence PROVES for hh_size.5+ and 55-64|male but NOT for
65-74|male (the diagnostic qualifying the registration); and Q16's cohabitation-
overlay lift clears the fragile spouse cell without collateral.

The PSID reproduction pins (rebuild seed 0's train-side inputs live and match the
recorded decompositions + the instrumented-draw fidelity to float precision,
skipped when the PSID relationship matrix is absent) live in
``tests/test_gate2b_forensics5_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics5_v1.json"
CANDIDATE7 = ROOT / "runs" / "gate2b_hazard_v7.json"

RECON_ATOL = 1e-9
EPS_ATOL = 1e-12
Q14_MALE_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
)
Q14_FEMALE_CELLS = (
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
Q14_CELLS = Q14_MALE_CELLS + Q14_FEMALE_CELLS
CHANNELS = (
    "fertility_origin",
    "exit_origin",
    "link_coverage",
    "v7_persistence_enumeration_interaction",
)
Q16_CELL = "coresident_spouse.25-34|female"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2b_forensics5.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4948430423")
    assert a["registration_pointer"] == "4948430423"
    assert a["grading_pointer"] == "4948429354"
    for block in (
        "question_14_older_parent_adult_child_supply",
        "question_15_single_lever_convergence",
        "question_16_fragile_spouse_cell",
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
    assert "household_composition_sim_v7" in p["fit_simulate_machinery"]
    assert "does NOT" in p["reused_machinery"]
    assert "custodial_prob_v6" in p["reused_machinery"]


def test_instrumentation_is_bit_identical_to_committed_draw():
    fid = _artifact()["protocol"]["instrumentation_fidelity"]
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v7"] == 0.0
    assert fid["child_channel_additivity_residual"] == 0
    assert fid["linked_marital_split_additivity_residual"] == 0


def test_reconciliations_are_identities_at_machine_epsilon():
    r = _artifact()["reconciliations"]
    eps = r["float64_machine_epsilon"]
    assert r["instrumentation_bit_identity_max_rate_deviation"] == 0.0
    assert r["child_channel_additivity_residual"] == 0
    assert r["linked_marital_split_additivity_residual"] == 0
    # The four-channel Oaxaca reconciliation of the cell miss is an identity up
    # to float summation order (a few ULP of float64).
    assert r["q14_cell_miss_reconciliation_max_abs_remainder"] <= 16 * eps
    # The lever's reference-core sanity convolution reproduces the reference.
    assert r["q15_reference_core_sanity_max_abs_dev"] <= 16 * eps
    assert r["all_identity_reconciliations_at_machine_epsilon"] is True


def test_no_gate_verdict_written_and_candidate7_named():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2b_pass" not in a
    assert a["candidate7_artifact"] == "runs/gate2b_hazard_v7.json"
    assert "candidate 7" in a["candidate_under_diagnosis"]
    assert a["candidate_8_implications"]


def test_per_seed_covers_the_five_gate_seeds():
    per_seed = _artifact()["per_seed"]
    assert sorted(s["seed"] for s in per_seed) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# Q14 -- older-parent adult-child supply
# --------------------------------------------------------------------------
def test_q14_registered_cells_present():
    q14 = _artifact()["question_14_older_parent_adult_child_supply"]
    assert tuple(q14["registered_cells"]) == Q14_CELLS
    assert set(q14["per_cell"]) == set(Q14_CELLS)


def test_q14_four_channels_reconcile_to_cell_miss_exactly():
    q14 = _artifact()["question_14_older_parent_adult_child_supply"]
    for cell in Q14_CELLS:
        rec = q14["per_cell"][cell]
        ch = rec["channels"]
        assert set(ch) == set(CHANNELS)
        assert rec["cell_miss_sim_minus_reference"] == pytest.approx(
            rec["sim_full_rate_train"] - rec["reference_full_rate_train"],
            abs=RECON_ATOL,
        )
        assert rec["channel_reconstruction_sum"] == pytest.approx(
            sum(ch.values()), abs=RECON_ATOL
        )
        assert rec["channel_reconstruction_sum"] == pytest.approx(
            rec["cell_miss_sim_minus_reference"], abs=RECON_ATOL
        )
        assert abs(rec["reconciliation_remainder"]) < 1e-9
        # The exact two-term Oaxaca: endowment + coefficient == cell miss.
        ox = rec["oaxaca_two_term"]
        assert ox["endowment_fertility_origin"] + ox[
            "coefficient_kernel_shift"
        ] == pytest.approx(
            rec["cell_miss_sim_minus_reference"], abs=RECON_ATOL
        )
        # The endowment IS the fertility-origin channel.
        assert ox["endowment_fertility_origin"] == pytest.approx(
            ch["fertility_origin"], abs=EPS_ATOL
        )
    assert q14["reconciliation_max_abs_remainder"] < 1e-9


def test_q14_female_cells_have_no_linked_channels():
    # The maternal female cells have no paternal-linked channel: link-coverage
    # and the v7 persistence/enumeration interaction are exact structural zeros.
    q14 = _artifact()["question_14_older_parent_adult_child_supply"]
    for cell in Q14_FEMALE_CELLS:
        ch = q14["per_cell"][cell]["channels"]
        assert ch["link_coverage"] == 0.0
        assert ch["v7_persistence_enumeration_interaction"] == 0.0


def test_q14_completed_family_size_3plus_deficit_at_every_cohort():
    # The registration's fertility signal: the sim under-produces 3+-coresident-
    # child families at every older cohort (the parent side of the forensics-3/4
    # large-family deficit).
    q14 = _artifact()["question_14_older_parent_adult_child_supply"]
    for cell in Q14_CELLS:
        cfs = q14["per_cell"][cell]["completed_family_size"]
        assert cfs["three_plus_deficit_sim_minus_train"] < 0.0
        assert cfs["sim_3plus_share"] == pytest.approx(
            cfs["sim_distribution"]["3"] + cfs["sim_distribution"]["4+"],
            abs=RECON_ATOL,
        )


def test_q14_fertility_dominant_at_55_but_not_at_65_male():
    # The diagnostic beats the registration's 'fertility-origin dominant at
    # 65-74|male' prior: fertility dominates at 55-64|male (younger adult
    # children still coreside) but NOT at 65-74|male, where the endowment is
    # attenuated by the near-zero adult-child kernel and exit/coverage co-lead.
    q14 = _artifact()["question_14_older_parent_adult_child_supply"]
    c55 = q14["per_cell"]["coresident_child.55-64|male"]
    c65 = q14["per_cell"]["coresident_child.65-74|male"]
    assert c55["dominant_channel"] == "fertility_origin"
    assert c65["dominant_channel"] != "fertility_origin"
    # At 65-74|male the fertility channel is smaller in magnitude than the
    # combined exit + link-coverage.
    ch65 = c65["channels"]
    assert abs(ch65["fertility_origin"]) < abs(ch65["exit_origin"]) + abs(
        ch65["link_coverage"]
    )


def test_q14_maternal_45_54_female_overproduces_via_retention():
    # 45-54|female OVER-produces: a fertility deficit more than offset by a
    # positive coresidence-given-size shift (the sim retains mid-life mothers'
    # children too long).
    q14 = _artifact()["question_14_older_parent_adult_child_supply"]
    rec = q14["per_cell"]["coresident_child.45-54|female"]
    assert rec["cell_miss_sim_minus_reference"] > 0.0
    assert rec["channels"]["fertility_origin"] < 0.0
    assert rec["channels"]["exit_origin"] > 0.0


# --------------------------------------------------------------------------
# Q15 -- the single-lever convergence proof
# --------------------------------------------------------------------------
def test_q15_convergence_verdict_structure():
    q15 = _artifact()["question_15_single_lever_convergence"]
    v = q15["convergence_verdict"]
    # hh_size.5+ and 55-64|male converge under the fertility lever.
    assert v["hh_size_5plus_proves"] is True
    assert v["older_male_cells_prove"]["coresident_child.55-64|male"] is True
    # 65-74|male does NOT converge on fertility alone.
    assert v["older_male_cells_prove"]["coresident_child.65-74|male"] is False
    assert v["older_male_all_prove"] is False
    # The trade cells and cleared cells stay in tolerance under the lever.
    assert v["hh_size_3_4_trade_holds"] is True
    assert v["cleared_child_cells_hold"] is True
    # So the single lever does NOT prove FULL convergence.
    assert v["single_lever_proves_full_convergence"] is False


def test_q15_lever_counterfactual_is_sim_minus_fertility_origin():
    # The older-cell counterfactual under the lever equals the sim minus the
    # Q14 fertility-origin endowment (closing exactly the fertility channel).
    a = _artifact()
    q14 = a["question_14_older_parent_adult_child_supply"]["per_cell"]
    q15 = a["question_15_single_lever_convergence"]["older_male_supply_cells"]
    per_seed = a["per_seed"]
    for cell in Q14_MALE_CELLS:
        for rec in q15[cell]["per_seed"]:
            seed = rec["seed"]
            s = next(x for x in per_seed if x["seed"] == seed)
            expect = (
                s["q14_sim"][cell]["sim_full"]
                - s["q14_sim"][cell]["endowment_fertility_origin"]
            )
            assert rec["counterfactual_lever"] == pytest.approx(
                expect, abs=RECON_ATOL
            )
        # The cell's holdout-failing seeds match the committed candidate-7 read.
        assert q15[cell]["holdout_committed_candidate7"]["tolerance"] == (
            q14[cell]["holdout_committed_candidate7"]["tolerance"]
        )


def test_q15_hh5plus_lever_clears_its_failing_seeds():
    q15 = _artifact()["question_15_single_lever_convergence"]
    hh5 = q15["hh_size_cells"]["hh_size.5+"]
    # hh_size.5+ fails seeds 3 and 4 on the committed holdout.
    assert set(hh5["holdout_committed_candidate7"]["failing_seeds"]) == {3, 4}
    assert hh5["counterfactual_clears_holdout_failing_seeds"] is True
    # And every per-seed counterfactual sits within tolerance.
    for rec in hh5["per_seed"]:
        assert rec["counterfactual_within_tolerance"] is True


def test_q15_reference_core_sanity_is_machine_epsilon():
    q15 = _artifact()["question_15_single_lever_convergence"]
    assert q15["reference_core_sanity_max_abs_dev"] < 1e-9


# --------------------------------------------------------------------------
# Q16 -- fragile spouse cell
# --------------------------------------------------------------------------
def test_q16_lift_clears_without_collateral():
    q16 = _artifact()["question_16_fragile_spouse_cell"]
    assert q16["registered_cell"] == Q16_CELL
    assert q16["overlay_shortfall_applied"] == 0.045
    assert q16["lift_clears_holdout_failing_seeds"] is True
    assert q16["no_collateral_spouse_cell_moves_out"] is True
    # The cell's measured 2/5 fragility on the committed holdout.
    committed = q16["holdout_committed_candidate7"]
    assert len(committed["failing_seeds"]) == 2
    assert committed["n_seeds_pass"] == 3


def test_q16_lift_recomputes_and_raises_every_seed():
    q16 = _artifact()["question_16_fragile_spouse_cell"]
    lift = q16["overlay_shortfall_applied"]
    for rec in q16["per_seed"]:
        expect = rec["sim_full"] + lift * (1.0 - rec["sim_full"])
        assert rec["lifted_full"] == pytest.approx(expect, abs=RECON_ATOL)
        assert rec["lifted_full"] > rec["sim_full"]


def test_q16_collateral_cells_stay_in_tolerance():
    q16 = _artifact()["question_16_fragile_spouse_cell"]
    assert q16["collateral_cells"]
    for _cell, rec in q16["collateral_cells"].items():
        assert rec["within_tolerance"] is True


# --------------------------------------------------------------------------
# Cross-cutting recomputation + global guards
# --------------------------------------------------------------------------
def test_q14_sim_full_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    q14 = a["question_14_older_parent_adult_child_supply"]["per_cell"]
    for cell in Q14_CELLS:
        recomputed = np.mean(
            [s["q14_sim"][cell]["sim_full"] for s in per_seed]
        )
        assert q14[cell]["sim_full_rate_train"] == pytest.approx(
            float(recomputed), abs=RECON_ATOL
        )


def test_q15_hh5plus_sim_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    hh5 = a["question_15_single_lever_convergence"]["hh_size_cells"][
        "hh_size.5+"
    ]
    recomputed = np.mean([s["q15_hh_sim"]["5+"]["sim"] for s in per_seed])
    assert hh5["seed_mean_sim"] == pytest.approx(
        float(recomputed), abs=RECON_ATOL
    )


def test_holdout_committed_reads_candidate7_only():
    # Q14/Q15/Q16 cite the committed candidate-7 side-A cells (never
    # re-simulated); the tolerances match the locked contract.
    a = _artifact()
    c7 = json.loads(CANDIDATE7.read_text())
    g0 = c7["per_seed"][0]["gated_cells"]
    q14_cell = a["question_14_older_parent_adult_child_supply"]["per_cell"][
        "coresident_child.65-74|male"
    ]["holdout_committed_candidate7"]
    q16_cell = a["question_16_fragile_spouse_cell"][
        "holdout_committed_candidate7"
    ]
    assert q14_cell["tolerance"] == pytest.approx(
        float(g0["coresident_child.65-74|male"]["tolerance"])
    )
    assert q16_cell["tolerance"] == pytest.approx(
        float(g0[Q16_CELL]["tolerance"])
    )


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
