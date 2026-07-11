"""Tests for the gate-2b candidate-5 one-shot scored run.

Candidate 5 (issue #42 comment 4945159933) is candidate 4 with EXACTLY THREE
frozen deltas, each designed against a corrected mechanism the gate-2b
forensics-2 decomposition (``runs/gate2b_forensics2_v1.json``) measured:

* **delta 1 -- multigen--adult-child coupling** (``coresident_grandchild``
  55+|female): for 55+ egos, replace the independent coresident-own-adult-child
  input to the composed grandchild with a train-fitted
  ``P(child | multigen, band, sex)`` on the simulated multigen occupancy; the
  multigen MARGINAL is unchanged (load-bearing spec constraint);
* **delta 2 -- not-married custodial correction** (``coresident_child`` male):
  for NOT-married linked fathers, swap the observable-basis custodial
  probability for the child-record-basis rate; the young-married gate untouched;
* **delta 3 -- bridge reach + parent_count composition** (``hh_size``): re-fit
  the non-family bridge conditional on core size and draw the coresident-parent
  count (1 vs 2) from the train composition instead of the fixed 2.

Everything candidate 4 cleared or carried -- coresident_parent, multigen (stock
+ transitions), parental_home_exit AND coresident_spouse -- is carried
byte-faithfully: their per-seed scores are IDENTICAL to candidate 4 to bit
precision. The one-shot outcome is pinned below from the committed artifact
``runs/gate2b_hazard_v5.json``. Always runnable: inspects the committed
artifact, binds the [20, 46, 5] cube, the byte-identical carried check, the
multigen-marginal-unchanged check, the c1->c5 progression, the coupling /
gap-closure checks, and the carried-blocker analysis. The reproduction pin
lives in ``tests/test_gate2b_candidate5_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v5.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
CANDIDATE3_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v3.json"
CANDIDATE4_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v4.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4945159933"
CANDIDATE4_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)
CARRIED_FAMILY_CELL_PREFIXES = (
    "coresident_parent.",
    "coresident_spouse.",
    "multigen.",
    "parental_home_",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _floor() -> dict:
    return json.loads(FLOOR.read_text())


def _gate2b_thresholds() -> dict:
    gates = yaml.safe_load(GATES.read_text())
    return gates["gates"]["gate_2"]["gate_2b"]["thresholds"]


def _gate2b_tolerances() -> dict[str, float]:
    tol: dict[str, float] = {}
    for view in _gate2b_thresholds()["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


# --------------------------------------------------------------------------
# Presence, registration, deltas
# --------------------------------------------------------------------------
def test_artifact_present_and_identity():
    a = _artifact()
    assert a["schema_version"] == "gate2b_hazard.v5"
    assert a["run"] == "gate2b_hazard_v5"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 5"
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["candidate4_registration_pointer"] == "4941160621"
    assert a["forensics_artifact"] == "runs/gate2b_forensics2_v1.json"
    assert a["forensics_pointer"] == "4942005972"
    assert a["grading_pointer"] == "4945156926"


def test_three_deltas_declared_and_mapped():
    a = _artifact()
    deltas = a["deltas_vs_candidate_4"]
    assert len(deltas) == 3
    assert any("coupling" in d for d in deltas)
    assert any("not-married" in d and "custodial" in d for d in deltas)
    assert any("bridge reach" in d and "parent_count" in d for d in deltas)
    mapping = a["per_delta_target_family"]
    assert (
        mapping["delta_1_multigen_child_coupling"] == "coresident_grandchild"
    )
    assert mapping["delta_2_not_married_custodial"] == "coresident_child"
    assert mapping["delta_3_bridge_reach_and_parent_count"] == "hh_size"


def test_one_shot_verdict_pinned_fail_0_of_5():
    """The committed one-shot outcome: gate FAIL, 0 of 5 seeds pass. The
    coupling clears grandchild 55+|female on all seeds, but the male
    coresident_child overshoot does not drain (married-father dominated, the
    not-married correction reaches only the not-married subset) and two carried
    female cells cap the other seeds."""
    v = _artifact()["verdict"]
    assert v["gate_2b_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert v["n_gate_seeds"] == 5
    assert all(p is False for p in v["seed_pass"].values())


def test_per_seed_gated_pass_counts_pinned():
    """Committed per-seed gated pass counts (39/39/40/40/41)."""
    by_seed = {s["seed"]: s for s in _artifact()["per_seed"]}
    assert {s: by_seed[s]["n_gated_pass"] for s in GATE_SEEDS} == {
        0: 39,
        1: 39,
        2: 40,
        3: 40,
        4: 41,
    }


def test_carried_blockers_cap_four_seeds():
    """Four of five seeds are capped by a carried cell byte-identical to
    candidate 4 (coresident_child.45-54|female on seeds 0/2;
    coresident_spouse.25-34|female on seeds 1/3); only seed 4 is unblocked."""
    bl = _artifact()["carried_blocker_analysis"]
    assert bl["n_seeds_capped_by_carried_cell"] == 4
    assert bl["max_attainable_seeds_given_carried_blockers"] == 1


def test_grandchild_55plus_female_clears_all_seeds():
    """The delta-1 coupling clears the modal residual on every seed."""
    a = _artifact()
    cell = "coresident_grandchild.55+|female"
    passes = sum(s["gated_cells"][cell]["pass"] for s in a["per_seed"])
    assert passes == 5


def test_one_shot_and_forecast_recorded_not_graded():
    a = _artifact()
    assert "REGARDLESS of verdict" in a["one_shot"]
    f = a["pre_registered_forecast"]
    assert f["p_gate_pass_4_of_5"] == "0.40-0.55"
    assert f["deliberately_not_majority_side"] is True
    assert "does NOT grade" in f["grading_note"]


def test_spec_resolution_notes_present():
    notes = _artifact()["spec_resolution_notes"]
    for key in (
        "rng_byte_identical_carried_families",
        "delta_1_multigen_child_coupling",
        "delta_2_not_married_custodial",
        "delta_3_bridge_reach_and_parent_count",
        "carried_families_byte_faithful",
        "multigen_marginal_unchanged",
        "observed_initial_states_are_the_holdout_persons_own",
    ):
        assert key in notes and notes[key]
    assert "MARGINAL is UNCHANGED" in notes["delta_1_multigen_child_coupling"]
    assert "MARRIED fathers keep" in notes["delta_2_not_married_custodial"]
    assert "core size" in notes["delta_3_bridge_reach_and_parent_count"]


# --------------------------------------------------------------------------
# Precheck + cell set + undefined rule
# --------------------------------------------------------------------------
def test_precheck_reproduced_exactly():
    p = _artifact()["precheck"]
    assert p["all_reproduced_exactly"] is True
    assert p["reference_moments_max_abs_deviation"] == 0.0
    assert p["rate_a_max_abs_deviation"] == 0.0
    assert p["holdout_sha256_all_match"] is True


def test_gated_cells_match_floor_gate_partition():
    a = _artifact()
    gated = set(_floor()["gate_partition"]["gate_eligible"])
    assert len(gated) == N_GATED
    assert set(_gate2b_tolerances()) == gated
    for s in a["per_seed"]:
        assert set(s["gated_cells"]) == gated


def test_report_only_cells_match_gates_yaml_and_never_gate():
    a = _artifact()
    report_only = set(_gate2b_thresholds()["report_only"])
    for s in a["per_seed"]:
        assert set(s["report_only_cells"]) == report_only
        for rec in s["report_only_cells"].values():
            assert rec["gated"] is False


def test_undefined_draw_rule_not_triggered_and_run_valid():
    u = _artifact()["fresh_run_artifact_schema"]["undefined_draw_rule"]
    assert u["required"] is True
    assert u["run_invalidated"] is False
    assert u["n_undefined_gated_draws"] == 0
    for s in _artifact()["per_seed"]:
        assert s["undefined_gated_draws"] == []
        for rec in s["gated_cells"].values():
            assert rec["n_draws_defined"] == N_DRAWS


# --------------------------------------------------------------------------
# Fresh-run artifact-schema conformance ([20, 46, 5])
# --------------------------------------------------------------------------
def test_per_draw_per_cell_rates_shape_and_index():
    pc = _artifact()["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    assert pc["shape"] == [N_DRAWS, N_GATED, len(GATE_SEEDS)]
    assert pc["cell_index"] == sorted(_gate2b_tolerances())
    assert pc["seed_index"] == GATE_SEEDS
    assert pc["k_index_draw_seeds"] == [
        DRAW_SEED_BASE + k for k in range(N_DRAWS)
    ]
    rates = pc["rates"]
    assert len(rates) == N_DRAWS
    assert all(len(r) == N_GATED for r in rates)
    assert all(len(c) == len(GATE_SEEDS) for r in rates for c in r)


def test_per_draw_cube_matches_per_seed_records():
    a = _artifact()
    pc = a["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    ci, si = pc["cell_index"], pc["seed_index"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for k in range(N_DRAWS):
        for c_idx, cell in enumerate(ci):
            for s_idx, seed in enumerate(si):
                cube = pc["rates"][k][c_idx][s_idx]
                stored = by_seed[seed]["gated_cells"][cell]["per_draw_rate"][k]
                assert cube == pytest.approx(stored, abs=1e-15)


def test_rbar_recomputes_from_per_draw_rates_and_scores():
    for s in _artifact()["per_seed"]:
        for rec in s["gated_cells"].values():
            rates = rec["per_draw_rate"]
            assert len(rates) == N_DRAWS
            rbar = float(np.mean(rates))
            assert rbar == pytest.approx(rec["rbar"], abs=1e-12)
            assert rec["r_candidate"] == pytest.approx(rec["rbar"], abs=1e-15)
            rate_a = rec["rate_a"]
            if rbar > 0 and rate_a > 0:
                expected = abs(math.log(rbar / rate_a))
                assert rec["score"] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# Verdict recompute from the cells
# --------------------------------------------------------------------------
def test_every_gated_pass_recomputes_from_score():
    for s in _artifact()["per_seed"]:
        for rec in s["gated_cells"].values():
            assert rec["pass"] == (rec["score"] <= rec["tolerance"])


def test_seed_pass_recomputes_from_all_gated_cells():
    for s in _artifact()["per_seed"]:
        n_pass = sum(rec["pass"] for rec in s["gated_cells"].values())
        assert n_pass == s["n_gated_pass"]
        assert s["seed_pass"] == (n_pass == N_GATED)


def test_verdict_recomputes_from_seed_conjunction():
    a = _artifact()
    n_pass = sum(s["seed_pass"] for s in a["per_seed"])
    assert a["verdict"]["n_seeds_pass"] == n_pass
    assert a["verdict"]["gate_2b_pass"] == (n_pass >= 4)
    assert a["verdict"]["n_gate_seeds"] == 5
    assert a["verdict"]["n_gated_cells"] == N_GATED


def test_stored_tolerances_match_locked_gates_yaml():
    tol = _gate2b_tolerances()
    for s in _artifact()["per_seed"]:
        for cell, rec in s["gated_cells"].items():
            assert rec["tolerance"] == pytest.approx(tol[cell], abs=1e-12)


# --------------------------------------------------------------------------
# Byte-identical carried families (incl. spouse) + multigen marginal + cleared
# --------------------------------------------------------------------------
def test_cleared_families_still_clear():
    chk = _artifact()["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    assert set(chk["families"]) == set(CANDIDATE4_CLEARED_FAMILIES)
    for fam in CANDIDATE4_CLEARED_FAMILIES:
        assert chk["detail"][fam]["candidate5_pass_rate"] == 1.0
        assert chk["detail"][fam]["still_clears"] is True


def test_carried_family_scores_byte_identical_to_candidate_4():
    """Every carried cell's per-seed gated score equals candidate 4's, cell for
    cell and seed for seed. coresident_spouse IS carried in candidate 5."""
    a = _artifact()
    byt = a["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["byte_identical"] is True
    assert byt["max_abs_score_deviation_vs_candidate4"] == 0.0
    c4 = json.loads(CANDIDATE4_ARTIFACT.read_text())
    by_seed_5 = {s["seed"]: s for s in a["per_seed"]}
    by_seed_4 = {s["seed"]: s for s in c4["per_seed"]}
    carried_cells = [
        c
        for c in _gate2b_tolerances()
        if c.startswith(CARRIED_FAMILY_CELL_PREFIXES)
        or c in ("multigen_entry", "multigen_exit")
    ]
    assert set(carried_cells) == set(byt["carried_cells"])
    # coresident_spouse IS now in the carried set (unlike candidate 4).
    assert any(c.startswith("coresident_spouse.") for c in carried_cells)
    for seed in GATE_SEEDS:
        for cell in carried_cells:
            s5 = by_seed_5[seed]["gated_cells"][cell]["score"]
            s4 = by_seed_4[seed]["gated_cells"][cell]["score"]
            assert s5 == pytest.approx(s4, abs=1e-12), (seed, cell)


def test_multigen_marginal_unchanged_vs_candidate_4():
    """The delta-1 load-bearing constraint: every multigen stock and transition
    cell score is byte-identical to candidate 4 (the coupling reads the multigen
    state but never changes the marginal)."""
    a = _artifact()
    mg = a["comparison_across_candidates"]["multigen_marginal_unchanged_check"]
    assert mg["marginal_unchanged"] is True
    assert mg["max_abs_score_deviation_vs_candidate4"] == 0.0
    expected = sorted(
        c
        for c in _gate2b_tolerances()
        if c.startswith("multigen.")
        or c in ("multigen_entry", "multigen_exit")
    )
    assert sorted(mg["multigen_cells"]) == expected
    for cell, dev in mg["per_cell_max_abs_score_deviation"].items():
        assert dev == 0.0, cell


# --------------------------------------------------------------------------
# c1 -> c5 progression + per-delta effects
# --------------------------------------------------------------------------
def test_progression_recomputes_c1_to_c5():
    a = _artifact()
    prog = a["comparison_across_candidates"]["per_family_progression"]
    decomp = a["per_family_decomposition"]
    arts = {
        "candidate1_pass_rate": CANDIDATE1_ARTIFACT,
        "candidate2_pass_rate": CANDIDATE2_ARTIFACT,
        "candidate3_pass_rate": CANDIDATE3_ARTIFACT,
        "candidate4_pass_rate": CANDIDATE4_ARTIFACT,
    }
    loaded = {
        k: json.loads(p.read_text())["per_family_decomposition"]
        for k, p in arts.items()
    }
    for fam, block in prog.items():
        assert (
            block["candidate5_pass_rate"] == decomp[fam]["cell_seed_pass_rate"]
        )
        for k, d in loaded.items():
            assert block[k] == d[fam]["cell_seed_pass_rate"]
        expected = round(
            block["candidate5_pass_rate"] - block["candidate4_pass_rate"], 4
        )
        assert block["delta_c4_to_c5"] == pytest.approx(expected, abs=1e-9)


def test_coupling_lifts_grandchild_family_over_c4():
    """Delta 1 lifts the coresident_grandchild family pass rate above candidate
    4 (the 55+|female cell moves toward the reference via the coupling)."""
    prog = _artifact()["comparison_across_candidates"][
        "per_family_progression"
    ]
    assert prog["coresident_grandchild"]["delta_c4_to_c5"] > 0
    # No carried family regresses.
    for fam in (
        "coresident_spouse",
        "coresident_parent",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
    ):
        assert prog[fam]["delta_c4_to_c5"] == 0.0


def test_per_family_decomposition_covers_all_gated_cells():
    a = _artifact()
    decomp = a["per_family_decomposition"]
    covered: set[str] = set()
    for fam in decomp.values():
        covered.update(fam["cells"])
        assert fam["mechanism"]
    assert covered == set(_gate2b_tolerances())


# --------------------------------------------------------------------------
# Coupling (Q6) + gap-closure (Q7) checks + delta stats
# --------------------------------------------------------------------------
def test_coupling_check_lifts_joint_above_independence_product():
    """The delta-1 fit lifts the composed 55+|female joint from the candidate-4
    independence product toward the reference joint (~0.0384)."""
    d1 = _artifact()["coupling_and_gap_checks"]["checks"]["delta_1_coupling"]
    assert d1["coupling_lifts_joint_toward_reference"] is True
    assert (
        d1["implied_fit_joint_multigen_x_p_child_given_multigen"]
        > d1["reference_independence_product"]
    )
    # The fitted P(child | multigen=True) is high (the coupling).
    assert d1["fitted_p_child_given_multigen_true_female"] > 0.7
    assert d1[
        "reference_joint_multigen_and_child_and_notparent"
    ] == pytest.approx(0.038416868577178784, abs=1e-9)


def test_gap_closure_check_records_the_forensics_split():
    d3 = _artifact()["coupling_and_gap_checks"]["checks"][
        "delta_3_gap_closure"
    ]
    # The 0.088 gap splits into ~0.051 non-core + ~0.037 composition.
    assert d3["noncore_member_gap_total"] == pytest.approx(0.0512, abs=1e-3)
    assert d3["composition_gap_total"] == pytest.approx(0.0366, abs=1e-3)
    # Core-3 carries the highest non-core-member incidence (the bridge reach).
    inc = d3["fitted_p_noncore_member_present_by_core"]
    assert inc["3"] == max(inc.values())


def test_delta_stats_recorded_for_all_three_deltas():
    for s in _artifact()["per_seed"]:
        ds = s["delta_stats"]
        assert set(ds) == {
            "delta_1_multigen_child_coupling",
            "delta_2_not_married_custodial",
            "delta_3_bridge_reach_and_parent_count",
        }
        d1 = ds["delta_1_multigen_child_coupling"]
        assert d1["n_coupled_grandchild_waves_simulated"] > 0
        # The coupling lifts the sim joint above its independence product.
        assert (
            d1["sim_gc55f_joint_mg_child_notparent"]
            > d1["sim_gc55f_independence_product"]
        )
        assert d1["sim_gc55f_union"] > 0
        d2 = ds["delta_2_not_married_custodial"]
        assert d2["n_linked_child_coresident_wave_units"] > 0
        assert len(d2["not_married_child_record_by_band"]) == 5
        d3 = ds["delta_3_bridge_reach_and_parent_count"]
        # The parent-count composition draws a mix (not the fixed 2).
        assert 1.0 < d3["sim_mean_n_parents_among_coresident_parent"] < 2.0
        assert d3["n_dense_core_band_sex_cells"] > 0


def test_model_records_delta_module_and_windows():
    a = _artifact()
    assert a["model"]["family_transitions_spec"] == "candidate16_registry_v1"
    assert (
        a["model"]["delta_module"]
        == "populace_dynamics.models.household_composition_sim_v5"
    )
    assert (
        a["model"]["candidate4_module"]
        == "populace_dynamics.models.household_composition_sim_v4"
    )
    assert a["model"]["grandchild_coupling_age_lo"] == 55
    assert a["model"]["core_size_cap"] == 5
    assert a["model"]["delta_stream_tag_v5"] == 0xC5
    assert a["model"]["coupling_age_bands_55plus"] == [
        [55, 64],
        [65, 74],
        [75, 120],
    ]


# --------------------------------------------------------------------------
# Carried-blocker analysis (the honest 'incomplete mechanism list' record)
# --------------------------------------------------------------------------
def test_carried_blocker_analysis_classifies_every_failing_seed():
    a = _artifact()
    bl = a["carried_blocker_analysis"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for seed_str, rec in bl["per_seed"].items():
        seed = int(seed_str)
        fails = [
            c
            for c in by_seed[seed]["gated_cells"]
            if not by_seed[seed]["gated_cells"][c]["pass"]
        ]
        # Every failing cell is classified exactly once.
        assert set(rec["carried_blockers"]) | set(
            rec["delta_target_fails"]
        ) == set(fails)
        assert not (
            set(rec["carried_blockers"]) & set(rec["delta_target_fails"])
        )
        assert rec["n_fail"] == len(fails)
        assert rec["seed_capped_by_carried_cell"] == bool(
            rec["carried_blockers"]
        )
    assert bl["max_attainable_seeds_given_carried_blockers"] == (
        5 - bl["n_seeds_capped_by_carried_cell"]
    )


def test_carried_blockers_are_byte_identical_to_candidate_4():
    """A cell flagged a carried blocker must indeed be byte-identical to
    candidate 4 (so it caps the seed regardless of the three deltas)."""
    a = _artifact()
    c4 = json.loads(CANDIDATE4_ARTIFACT.read_text())
    by_seed_5 = {s["seed"]: s for s in a["per_seed"]}
    by_seed_4 = {s["seed"]: s for s in c4["per_seed"]}
    for seed_str, rec in a["carried_blocker_analysis"]["per_seed"].items():
        seed = int(seed_str)
        for cell in rec["carried_blockers"]:
            s5 = by_seed_5[seed]["gated_cells"][cell]["score"]
            s4 = by_seed_4[seed]["gated_cells"][cell]["score"]
            assert s5 == pytest.approx(s4, abs=1e-12), (seed, cell)
