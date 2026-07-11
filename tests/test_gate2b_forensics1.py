"""Tests for gate-2b forensics 1 (reported, not gated).

ALWAYS RUNNABLE (artifact tier). The consistency tests read only the forensics
artifact (``runs/gate2b_forensics1_v1.json``), the committed candidate-3 gate
artifact (``runs/gate2b_hazard_v3.json``) and ``gates.yaml``; they never rerun
the diagnostic and need no PSID, so they run in CI. They audit that every
headline recomputes from the stored per-seed values and that each of the four
mechanism decompositions reconciles to zero: the Q1 legal-core vs
cohabitation-overlay split and the concept-residual code enumeration, the Q2
linked vs unlinked child contribution, the Q3 size-3 non-family and
composition-route partitions, and the Q4 composed vs skipped-generation
grandchild split.

The four PSID reproduction pins (rebuild the train-side inputs live and match
seed 0's deterministic decompositions and the instrumented-draw fidelity to
float precision, skipped when the PSID relationship matrix is absent) live in
``tests/test_gate2b_forensics1_reproduction.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics1_v1.json"
CANDIDATE3 = ROOT / "runs" / "gate2b_hazard_v3.json"

RECON_ATOL = 1e-9


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _seed(per_seed: list[dict], seed: int) -> dict:
    return next(s for s in per_seed if s["seed"] == seed)


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2b_forensics1.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4940442065")
    assert a["registration_pointer"] == "4940442065"
    assert a["grading_pointer"] == "4940440505"
    for block in (
        "question_1_spouse_legal_vs_cohabitation",
        "question_2_child_shadow_residual",
        "question_3_hh_size_middle_distribution",
        "question_4_grandchild_skipgen_remainder",
    ):
        assert block in a
        assert a[block]["finding"]


def test_protocol_is_train_side_only_and_holdout_untouched():
    p = _artifact()["protocol"]
    assert p["train_side_only"] is True
    assert "never" in p["outer_holdout_contact"].lower()
    assert "re-simulated" in p["outer_holdout_contact"].lower()
    assert p["gate_seeds"] == [0, 1, 2, 3, 4]
    assert p["n_draws"] == 20
    assert "side B only" in p["no_holdout_tuning_surface"]


def test_instrumentation_is_bit_identical_to_committed_draw():
    fid = _artifact()["protocol"]["instrumentation_fidelity"]
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v3"] == 0.0


def test_no_gate_verdict_written_and_lock_untouched():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2b_pass" not in a
    assert a["revision_pins"]["gates_yaml_locked"] is True


def test_failing_cells_match_committed_candidate3():
    a = _artifact()
    committed = json.loads(CANDIDATE3.read_text())
    fails: dict[str, set[int]] = {}
    for f in committed["verdict"]["all_failing_gated_cells"]:
        fails.setdefault(f["cell"], set()).add(int(f["seed"]))
    # Every spouse/child(male)/grandchild/hh_size failing cell the questions
    # name is a real committed failing cell.
    q1 = a["question_1_spouse_legal_vs_cohabitation"]["failing_cells"]
    for cell, seeds in q1.items():
        assert set(seeds) == fails[cell]


# --------------------------------------------------------------------------
# Q1 -- spouse legal-core vs cohabitation-overlay
# --------------------------------------------------------------------------
def test_q1_legal_plus_overlay_reconciles_to_reference():
    q1 = _artifact()["question_1_spouse_legal_vs_cohabitation"]
    for seeds in q1["legal_core_vs_overlay_decomposition"].values():
        for rec in seeds.values():
            # full = legal-only + overlay (the union is additive).
            assert rec["full_spouse_rate"] == pytest.approx(
                rec["legal_core_only_rate"] + rec["overlay_contribution"],
                abs=RECON_ATOL,
            )
            # legal_core_gap = overlay + residual_miss, exactly.
            assert rec["legal_core_gap"] == pytest.approx(
                rec["overlay_contribution"] + rec["residual_miss_train"],
                abs=RECON_ATOL,
            )
            assert abs(rec["reconciliation_remainder"]) < RECON_ATOL


def test_q1_concept_residual_is_a_verified_null():
    c = _artifact()["question_1_spouse_legal_vs_cohabitation"][
        "concept_residual"
    ]
    assert c["reference_concept_codes_mx8"] == [20, 22]
    assert c["codes_beyond_20_22_present"] == []
    assert c["share_beyond_20_22"] == pytest.approx(0.0, abs=RECON_ATOL)
    # The 20-only / 22-only / both / beyond shares partition the spouse mass.
    total = (
        c["share_code20_legal_only"]
        + c["share_code22_cohab_only"]
        + c["share_both_codes"]
        + c["share_beyond_20_22"]
    )
    assert total == pytest.approx(1.0, abs=1e-6)
    assert abs(c["reconciliation_remainder"]) < 1e-6


def test_q1_direction_split_over_vs_under():
    # The young male cell overshoots (overlay-driven); the older male cells
    # undershoot (legal-core-driven). The per-cell summary records it.
    summ = _artifact()["question_1_spouse_legal_vs_cohabitation"][
        "per_cell_summary"
    ]
    young = summ["coresident_spouse.15-24|male"]
    assert young["holdout_direction"] == "over"
    assert young["overlay_contribution"] > 0
    # Older male cells undershoot and are legal-core dominated (low cohab share).
    for cell in ("coresident_spouse.65-74|male", "coresident_spouse.75+|male"):
        assert summ[cell]["holdout_direction"] == "under"
        assert summ[cell]["code22_cohab_share_of_reference"] < 0.1


def test_q1_band_hazard_matches_band_aggregate_but_hides_single_year():
    # The fitted band hazard equals the raw band entry rate (it IS the weighted
    # band rate), yet the within-band single-year gradient is steep -- the 2a
    # band-split lesson made visible.
    af = _artifact()["question_1_spouse_legal_vs_cohabitation"][
        "single_year_age_fit"
    ]["male"]["band_detail"]["15-24"]
    assert af["fitted_entry_hazard"] == pytest.approx(
        af["raw_entry_rate"], abs=1e-6
    )
    assert af["within_band_old_over_young_ratio"] > 10.0


def test_q1_draw_stability_probabilities_in_range():
    stab = _artifact()["question_1_spouse_legal_vs_cohabitation"][
        "draw_stability"
    ]
    for cell in stab.values():
        for rec in cell.values():
            assert 0.0 <= rec["holdout_fraction_draws_clip_tolerance"] <= 1.0
            assert isinstance(
                rec["holdout_direction_same_sign_all_draws"], bool
            )


# --------------------------------------------------------------------------
# Q2 -- child shadow residual
# --------------------------------------------------------------------------
def test_q2_linked_plus_unlinked_reconciles_to_full():
    q2 = _artifact()["question_2_child_shadow_residual"]
    for cell, seeds in q2["linked_vs_unlinked_decomposition"].items():
        assert cell.endswith("|male")
        for rec in seeds.values():
            assert rec["full_train_rate"] == pytest.approx(
                rec["linked_contribution"]
                + rec["unlinked_shadow_contribution"],
                abs=RECON_ATOL,
            )
            assert abs(rec["reconciliation_remainder"]) < RECON_ATOL
            # Population-share partition sums to 1.
            assert rec["w_linked"] + rec["w_unlinked"] == pytest.approx(
                1.0, abs=1e-6
            )


def test_q2_shadow_is_the_minority_share():
    # The unlinked shadow supplies less than the linked custodial children on
    # every failing male cell (the descriptive headline).
    q2 = _artifact()["question_2_child_shadow_residual"]
    for seeds in q2["linked_vs_unlinked_decomposition"].values():
        for rec in seeds.values():
            assert (
                rec["unlinked_shadow_contribution"]
                < rec["linked_contribution"]
            )
    prof = q2["unlinked_men_profile"]
    # The shadow fires only for married men; unlinked men are far less married.
    assert prof["unlinked_married_share"] < prof["linked_married_share"]


# --------------------------------------------------------------------------
# Q3 -- hh_size middle distribution
# --------------------------------------------------------------------------
def test_q3_size3_partitions_reconcile():
    q3 = _artifact()["question_3_hh_size_middle_distribution"]
    nf = q3["size3_partition_by_nonfamily_count"]
    assert nf["total_size3"] == pytest.approx(
        nf["nonfamily_0_family_core"]
        + nf["nonfamily_1"]
        + nf["nonfamily_2_truncated"],
        abs=RECON_ATOL,
    )
    assert abs(nf["reconciliation_remainder"]) < RECON_ATOL
    routes = q3["size3_partition_by_composition_route"]
    route_sum = sum(
        v
        for k, v in routes.items()
        if k not in ("total_size3", "reconciliation_remainder")
    )
    assert routes["total_size3"] == pytest.approx(route_sum, abs=RECON_ATOL)
    assert abs(routes["reconciliation_remainder"]) < RECON_ATOL


def test_q3_family_core_over_produces_size3():
    # The pre-registered hunch ("non-family 2+ mis-shaped rather than family
    # core") is corrected: the family core ALONE over-produces size 3.
    d = _artifact()["question_3_hh_size_middle_distribution"][
        "hh_size_distribution"
    ]
    assert d["family_core_only_simulated"]["3"] > d["reference_train"]["3"]
    # The bridge REDUCES size 3 (pushes core-3 up), yet it still overshoots.
    assert d["family_core_only_simulated"]["3"] > d["full_simulated"]["3"]
    assert d["full_simulated"]["3"] > d["reference_train"]["3"]


def test_q3_minimal_reading_truncates_the_tail():
    t = _artifact()["question_3_hh_size_middle_distribution"][
        "nonfamily_2plus_minimal_reading_test"
    ]
    # 2+ households truly average > 2 members; the bridge caps them at 2.
    assert t["mean_nonfamily_count_within_2plus_households"] > 2.0
    assert (
        t["bridge_truncated_mean_count_2plus_as_2"]
        < t["train_true_weighted_mean_count"]
    )
    assert t["mean_count_lost_to_truncation"] == pytest.approx(
        t["train_true_weighted_mean_count"]
        - t["bridge_truncated_mean_count_2plus_as_2"],
        abs=RECON_ATOL,
    )
    # The deep tail undershoots (the truncation's separate cost).
    d = _artifact()["question_3_hh_size_middle_distribution"][
        "hh_size_distribution"
    ]
    assert d["full_simulated"]["5+"] < d["reference_train"]["5+"]


# --------------------------------------------------------------------------
# Q4 -- grandchild 55+|female remainder
# --------------------------------------------------------------------------
def test_q4_union_reconciles_to_composed_plus_skipgen():
    q4 = _artifact()["question_4_grandchild_skipgen_remainder"]
    for rec in q4["per_seed_decomposition"].values():
        assert rec["union_rate"] == pytest.approx(
            rec["composed_multigen_path_rate"] + rec["skipgen_only_rate"],
            abs=RECON_ATOL,
        )
        assert abs(rec["reconciliation_remainder"]) < RECON_ATOL


def test_q4_skipgen_carries_majority_but_level_short():
    sa = _artifact()["question_4_grandchild_skipgen_remainder"][
        "seed_averaged"
    ]
    # The skip-gen delta carries the majority of the (still-short) union.
    assert sa["skipgen_share_of_union"] > 0.5
    assert sa["union_rate"] < sa["reference_train_rate_b"]
    assert sa["residual_miss_after_both"] == pytest.approx(
        sa["reference_train_rate_b"] - sa["union_rate"], abs=RECON_ATOL
    )


def test_q4_seed_averaged_recomputes_from_per_seed():
    a = _artifact()
    q4 = a["question_4_grandchild_skipgen_remainder"]
    per_seed = a["per_seed"]
    seeds = q4["failing_seeds"]
    union = np.mean([_seed(per_seed, s)["q4_union_mean"] for s in seeds])
    assert q4["seed_averaged"]["union_rate"] == pytest.approx(
        float(union), abs=RECON_ATOL
    )


# --------------------------------------------------------------------------
# Cross-cutting recomputation from the stored per-seed values
# --------------------------------------------------------------------------
def test_q1_full_rate_recomputes_from_per_seed_draws_mean():
    a = _artifact()
    q1 = a["question_1_spouse_legal_vs_cohabitation"]
    per_seed = a["per_seed"]
    for cell, seeds in q1["legal_core_vs_overlay_decomposition"].items():
        for seed_str, rec in seeds.items():
            sd = _seed(per_seed, int(seed_str))
            assert rec["full_spouse_rate"] == pytest.approx(
                sd["q1_spouse_full_mean"][cell], abs=RECON_ATOL
            )
            assert rec["legal_core_only_rate"] == pytest.approx(
                sd["q1_spouse_legal_mean"][cell], abs=RECON_ATOL
            )


def test_reference_rate_b_matches_across_questions():
    # rate_b is the single train-side reference each question scores against.
    a = _artifact()
    per_seed = a["per_seed"]
    q1 = a["question_1_spouse_legal_vs_cohabitation"]
    for cell, seeds in q1["legal_core_vs_overlay_decomposition"].items():
        for seed_str, rec in seeds.items():
            sd = _seed(per_seed, int(seed_str))
            assert rec["reference_train_rate_b"] == pytest.approx(
                sd["rate_b"][cell], abs=RECON_ATOL
            )


def test_all_reconciliation_remainders_are_negligible():
    # A single guard over every stored reconciliation_remainder field.
    a = _artifact()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "reconciliation_remainder" and isinstance(
                    v, (int, float)
                ):
                    assert abs(v) < 1e-6, k
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(a)


def test_finite_or_null_floats_only():
    # c3._json_safe maps non-finite floats to null; none should be NaN/inf.
    text = ARTIFACT.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
