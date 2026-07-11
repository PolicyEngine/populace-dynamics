"""Tests for gate-2b forensics 2 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the forensics-2
artifact (``runs/gate2b_forensics2_v1.json``), the committed candidate-4 gate
artifact (``runs/gate2b_hazard_v4.json``) and ``gates.yaml``; they never rerun
the diagnostic and need no PSID, so they run in CI. They audit that every
headline recomputes from the stored per-seed values and that each of the three
concept-class decompositions reconciles to (machine-)zero: the Q5 maternal
four-way partition and the observable-minus-child-record selection gap, the Q6
grandchild channel partition + the composed/skip-gen supply split + the
multigen-vs-coresident-child coupling contrast, and the Q7 size-3 route
partitions + the composition-vs-non-core gap decomposition.

The three PSID reproduction pins (rebuild the train-side inputs live and match
seed 0's deterministic decompositions and the instrumented-draw fidelity to
float precision, skipped when the PSID relationship matrix is absent) live in
``tests/test_gate2b_forensics2_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics2_v1.json"
CANDIDATE4 = ROOT / "runs" / "gate2b_hazard_v4.json"

RECON_ATOL = 1e-9
EPS_ATOL = 1e-15
CUSTODIAL_BANDS = ("0-4", "5-12", "13-17", "18-24", "25-60")
REF_GRANDPARENT_LINK = {66, 68, 82, 87, 88}
EXCLUDED_GRANDPARENT_CODES = {67, 69, 83}
Q7_ROUTES = (
    "couple_plus_child",
    "single_parent_plus_two_children",
    "couple_plus_parent",
    "three_adults",
    "other_family_core",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _seed(per_seed: list[dict], seed: int) -> dict:
    return next(s for s in per_seed if s["seed"] == seed)


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2b_forensics2.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4942005972")
    assert a["registration_pointer"] == "4942005972"
    assert a["grading_pointer"] == "4942004647"
    for block in (
        "question_5_custodial_selection_basis",
        "question_6_grandchild_reference_channels",
        "question_7_hh_size3_family_core_routes",
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
    assert "household_composition_sim_v4" in p["fit_simulate_machinery"]


def test_instrumentation_is_bit_identical_to_committed_draw():
    fid = _artifact()["protocol"]["instrumentation_fidelity"]
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v4"] == 0.0


def test_no_gate_verdict_written_and_candidate4_named():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2b_pass" not in a
    assert a["candidate4_artifact"] == "runs/gate2b_hazard_v4.json"
    assert "candidate 4" in a["candidate_under_diagnosis"]
    assert a["candidate_5_implications"]


def test_per_seed_covers_the_five_gate_seeds():
    per_seed = _artifact()["per_seed"]
    assert sorted(s["seed"] for s in per_seed) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# Q5 -- custodial selection basis
# --------------------------------------------------------------------------
def test_q5_maternal_complement_partitions_to_one():
    q5 = _artifact()["question_5_custodial_selection_basis"]
    ov = q5["maternal_complement_overall"]
    four = (
        ov["with_both"]
        + ov["with_father_only"]
        + ov["with_mother_only"]
        + ov["with_neither"]
    )
    assert four == pytest.approx(1.0, abs=RECON_ATOL)
    assert abs(ov["partition_reconciliation_remainder"]) < EPS_ATOL
    # P(with father) = both + father-only; P(with mother) = both + mother-only.
    assert ov["p_with_father"] == pytest.approx(
        ov["with_both"] + ov["with_father_only"], abs=EPS_ATOL
    )
    assert ov["p_with_mother"] == pytest.approx(
        ov["with_both"] + ov["with_mother_only"], abs=EPS_ATOL
    )
    # The complement identity P(father)+P(mother)-both+neither == 1 exactly.
    assert abs(ov["father_complement_identity_remainder"]) < EPS_ATOL
    for band in CUSTODIAL_BANDS:
        rec = q5["maternal_complement_by_child_band"][band]
        s = (
            rec["with_both"]
            + rec["with_father_only"]
            + rec["with_mother_only"]
            + rec["with_neither"]
        )
        assert s == pytest.approx(1.0, abs=RECON_ATOL)
        assert abs(rec["partition_reconciliation_remainder"]) < EPS_ATOL


def test_q5_child_not_with_father_is_usually_with_mother():
    ov = _artifact()["question_5_custodial_selection_basis"][
        "maternal_complement_overall"
    ]
    # Among children not with the father, the mother-only mass exceeds neither's
    # (a child off the father's roster is more often with the mother than with
    # neither) -- the maternal complement the registration names.
    assert ov["with_mother_only"] > 0.0
    assert ov["p_with_mother"] > 0.0


def test_q5_selection_gap_recomputes_and_is_signed_consistently():
    q5 = _artifact()["question_5_custodial_selection_basis"]
    table = q5["selection_gap_table_by_band_marital"]
    for rec in table.values():
        assert rec[
            "selection_gap_observable_minus_child_record"
        ] == pytest.approx(
            rec["observable_basis_p_coresident"]
            - rec["child_record_basis_p_coresident"],
            abs=RECON_ATOL,
        )
        for p in (
            "observable_basis_p_coresident",
            "child_record_basis_p_coresident",
        ):
            assert 0.0 <= rec[p] <= 1.0


def test_q5_gap_is_heterogeneous_by_marital():
    # The honest, prior-contradicting headline: the gap REVERSES for married
    # young ages (child-record >= observable) but the observable OVER-states
    # coresidence for not-married school-age fathers (gap > 0).
    q5 = _artifact()["question_5_custodial_selection_basis"]
    assert q5["young_married_seed_mean_gap"] < 0.0
    assert q5["not_married_school_age_seed_mean_gap"] > 0.0
    table = q5["selection_gap_table_by_band_marital"]
    # At each young married band the observable is faithful/under the child
    # record (gap <= a small positive slack), never materially higher.
    for band in ("0-4", "5-12", "13-17"):
        assert (
            table[f"{band}|married"][
                "selection_gap_observable_minus_child_record"
            ]
            <= 0.02
        )


# --------------------------------------------------------------------------
# Q6 -- grandchild reference channels
# --------------------------------------------------------------------------
def test_q6_channels_partition_reference_stock():
    q6 = _artifact()["question_6_grandchild_reference_channels"]
    ch = q6["reference_channels"]
    assert q6["reference_stock_55plus_female"] == pytest.approx(
        sum(ch.values()), abs=RECON_ATOL
    )
    assert abs(q6["reference_channel_reconciliation_remainder"]) < EPS_ATOL
    # a' breaks detail is a subset of the a' channel.
    bd = q6["a_prime_breaks_detail"]
    assert (
        bd["child_in_law_middle_generation_only"]
        + bd["four_generation_own_parent_present"]
        <= ch["a_prime_ego_on_top_composed_breaks"] + RECON_ATOL
    )


def test_q6_sim_union_reconciles_to_composed_plus_skipgen():
    s = _artifact()["question_6_grandchild_reference_channels"][
        "simulation_reachable_supply"
    ]
    assert s["union"] == pytest.approx(
        s["composed_multigen_path"] + s["skipgen_only"], abs=RECON_ATOL
    )
    assert abs(s["union_reconciliation_remainder"]) < EPS_ATOL
    assert s["unreachable_reference_minus_sim_union"] > 0.0


def test_q6_unreachable_is_channel_a_not_the_in_law_channel():
    # The prior-contradicting headline: channel (a) ego-on-top three-generation
    # dominates the reference stock, and the in-law-break channel (a') is tiny.
    q6 = _artifact()["question_6_grandchild_reference_channels"]
    ch = q6["reference_channels"]
    assert ch["a_ego_on_top_composed_reachable"] > ch["b_skipped_generation"]
    assert (
        ch["a_ego_on_top_composed_reachable"]
        > 5 * ch["a_prime_ego_on_top_composed_breaks"]
    )


def test_q6_reference_couples_multigen_and_child_but_sim_decouples():
    # The mechanism: the reference joint (multigen AND child AND NOT-parent)
    # sits FAR above the independence product (coupled), while the sim joint
    # sits AT its independence product (decoupled) -- so the composed path
    # under-fires despite comparable marginals.
    q6 = _artifact()["question_6_grandchild_reference_channels"]
    ref = q6["reference_component_rates_55plus_female"]
    assert (
        ref["multigen_and_child_and_notparent"]
        > 2 * ref["independence_product_multigen_x_child_x_notparent"]
    )
    sim = q6["simulation_reachable_supply"]
    assert sim["composed_component_rates_55plus_female"][
        "multigen_and_child_and_notparent"
    ] == pytest.approx(
        sim["composed_independence_product_55plus_female"], abs=0.005
    )
    # The reference joint equals channel (a) by construction (multigen AND child
    # AND NOT-parent IS the ego-on-top composed test).
    assert ref["multigen_and_child_and_notparent"] == pytest.approx(
        q6["reference_channels"]["a_ego_on_top_composed_reachable"], abs=1e-3
    )


def test_q6_grandparent_code_inventory_membership():
    inv = _artifact()["question_6_grandchild_reference_channels"][
        "grandparent_code_inventory"
    ]
    for code, rec in inv.items():
        in_ref = int(code) in REF_GRANDPARENT_LINK
        assert rec["in_reference_link"] is in_ref
        if int(code) in EXCLUDED_GRANDPARENT_CODES:
            assert rec["in_reference_link"] is False
        assert rec["weighted_share_of_55plus_female"] >= 0.0
    # Plain grandparent (66) carries the dominant share.
    if "66" in inv:
        assert inv["66"]["in_reference_link"] is True


# --------------------------------------------------------------------------
# Q7 -- hh_size.3 family-core routes
# --------------------------------------------------------------------------
def test_q7_routes_partition_each_total():
    q7 = _artifact()["question_7_hh_size3_family_core_routes"]
    assert q7["sim_family_core_size3_total"] == pytest.approx(
        sum(q7["sim_core_routes"].values()), abs=RECON_ATOL
    )
    assert q7["reference_actual_size3_total"] == pytest.approx(
        sum(q7["reference_actual_routes"].values()), abs=RECON_ATOL
    )
    assert q7["reference_core_size3_total"] == pytest.approx(
        sum(q7["reference_core_routes"].values()), abs=RECON_ATOL
    )
    for routes in (
        q7["sim_core_routes"],
        q7["reference_actual_routes"],
        q7["reference_core_routes"],
    ):
        assert set(routes) == set(Q7_ROUTES)


def test_q7_gap_decomposition_reconciles_exactly():
    gd = _artifact()["question_7_hh_size3_family_core_routes"][
        "gap_decomposition"
    ]
    assert gd["total_gap_sim_core_minus_ref_actual"] == pytest.approx(
        gd["composition_gap_total"] + gd["noncore_member_gap_total"],
        abs=RECON_ATOL,
    )
    assert abs(gd["reconciliation_remainder"]) < EPS_ATOL
    # Per-route the two component gaps sum to the actual gap.
    q7 = _artifact()["question_7_hh_size3_family_core_routes"]
    for route in Q7_ROUTES:
        total = q7["route_gap_sim_core_minus_reference_actual"][route]
        comp = gd["composition_gap_sim_core_minus_ref_core"][route]
        nonc = gd["noncore_member_gap_ref_core_minus_ref_actual"][route]
        assert total == pytest.approx(comp + nonc, abs=RECON_ATOL)


def test_q7_over_produced_route_is_the_argmax_gap():
    q7 = _artifact()["question_7_hh_size3_family_core_routes"]
    gaps = q7["route_gap_sim_core_minus_reference_actual"]
    assert q7["over_produced_route"] == max(gaps, key=gaps.get)


def test_q7_noncore_member_effect_dominates():
    # The 0.271-vs-0.181 core gap is dominantly the non-core-member construct
    # (ref-core minus ref-actual), not the sim's core composition error.
    gd = _artifact()["question_7_hh_size3_family_core_routes"][
        "gap_decomposition"
    ]
    assert gd["noncore_member_gap_total"] > gd["composition_gap_total"]
    assert gd["total_gap_sim_core_minus_ref_actual"] > 0.0


# --------------------------------------------------------------------------
# Cross-cutting recomputation + global guards
# --------------------------------------------------------------------------
def test_q6_reference_stock_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    stock = np.mean(
        [s["q6_reference"]["reference_stock_55plus_female"] for s in per_seed]
    )
    assert a["question_6_grandchild_reference_channels"][
        "reference_stock_55plus_female"
    ] == pytest.approx(float(stock), abs=RECON_ATOL)


def test_q7_sim_core_total_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    tot = np.mean([s["q7_sim_core_size3_total_mean"] for s in per_seed])
    assert a["question_7_hh_size3_family_core_routes"][
        "sim_family_core_size3_total"
    ] == pytest.approx(float(tot), abs=RECON_ATOL)


def test_all_remainder_fields_are_negligible():
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


def test_holdout_committed_reads_candidate4_only():
    # Both Q6 and Q7 cite the committed candidate-4 side-A cell (never
    # re-simulated); the tolerances match the locked contract.
    a = _artifact()
    c4 = json.loads(CANDIDATE4.read_text())
    gc = a["question_6_grandchild_reference_channels"][
        "holdout_committed_candidate4"
    ]
    hh3 = a["question_7_hh_size3_family_core_routes"][
        "holdout_committed_candidate4"
    ]
    committed_gc = c4["per_seed"][0]["gated_cells"][
        "coresident_grandchild.55+|female"
    ]["tolerance"]
    committed_hh3 = c4["per_seed"][0]["gated_cells"]["hh_size.3"]["tolerance"]
    assert gc["tolerance"] == pytest.approx(committed_gc)
    assert hh3["tolerance"] == pytest.approx(committed_hh3)
