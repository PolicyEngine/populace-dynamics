"""Tests for the gate-2b candidate-6 one-shot scored run.

Candidate 6 (issue #42 comment 4946285556) is candidate 5 with EXACTLY FOUR
frozen deltas, each designed against a graded gate-2b forensics-3 finding
(``runs/gate2b_forensics3_v1.json``, grading 4946281888):

* **delta 1 -- 0-4 basis revert** (``coresident_child`` 15-24|male): revert the
  not-married custodial swap to the observable basis at child ages 0-4 (keep the
  child-record swap at 5-17);
* **delta 2 -- adult-child exit timing** (``coresident_child`` 35-44|male,
  45-54|female): re-fit the single-year 18-30 child-age home-exit hazard and
  override the maternal own-birth leave; the linked-married side is candidate 4's
  single-year observable declining married custodial prob UNCHANGED (a hard
  leave would double-count the aging-out); the multigen coupling stays at 55+;
* **delta 3 -- female cohabitation lift at 25-34** (``coresident_spouse``
  25-34|female): re-fit the FEMALE single-year cohabitation entry/exit over
  25-44; the legal top-up is NOT applied;
* **delta 4 -- count-conditional bridge** (``hh_size`` 3/4/5+): draw the FULL
  non-core count from the train joint P(count | capped core).

Everything candidate 5 cleared or carried -- coresident_parent, multigen (stock
+ transitions), parental_home_exit AND the 55+ coupled grandchild -- is carried
byte-faithfully. The one-shot outcome is pinned below from the committed
artifact ``runs/gate2b_hazard_v6.json``. Always runnable. The reproduction pin
lives in ``tests/test_gate2b_candidate6_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v5.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"
FORENSICS = ROOT / "runs" / "gate2b_forensics3_v1.json"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4946285556"

#: Pinned one-shot outcome (from the committed artifact runs/gate2b_hazard_v6).
PINNED_N_SEEDS_PASS = 0
PINNED_PER_SEED_GATED_PASS = [42, 42, 42, 42, 43]

STRICT_CARRIED_PREFIXES = (
    "coresident_parent.",
    "multigen.",
    "parental_home_",
)
DELTA_MODULE = "populace_dynamics.models.household_composition_sim_v6"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


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
    assert a["schema_version"] == "gate2b_hazard.v6"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 6"
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["candidate5_registration_pointer"] == "4945159933"
    assert a["grading_pointer"] == "4946281888"
    assert a["forensics_artifact"] == "runs/gate2b_forensics3_v1.json"
    assert a["model"]["delta_module"] == DELTA_MODULE


def test_four_deltas_declared_and_mapped():
    a = _artifact()
    assert len(a["deltas_vs_candidate_5"]) == 4
    mapping = a["per_delta_target_family"]
    assert mapping["delta_1_zero_four_revert"] == "coresident_child"
    assert mapping["delta_2_adult_child_exit_timing"] == "coresident_child"
    assert mapping["delta_3_female_cohab_lift"] == "coresident_spouse"
    assert mapping["delta_4_count_conditional_bridge"] == "hh_size"


def test_windows_and_stream_tags_recorded():
    m = _artifact()["model"]
    assert m["grandchild_coupling_age_lo"] == 55  # coupling stays 55+
    assert m["custodial_revert_band"] == [0, 4]
    assert m["child_exit_refit_range"] == [18, 30]
    assert m["cohab_female_refit_range"] == [25, 44]
    assert m["delta_stream_tag_v5"] == 0xC5
    assert m["delta_stream_tag_v6"] == 0xC6


def test_one_shot_and_forecast_recorded_not_graded():
    a = _artifact()
    assert "REGARDLESS of verdict" in a["one_shot"]
    f = a["pre_registered_forecast"]
    assert f["p_gate_pass_4_of_5"] == "0.55-0.70"
    assert "does NOT grade" in f["grading_note"]


def test_spec_resolution_notes_present():
    notes = _artifact()["spec_resolution_notes"]
    for key in (
        "rng_byte_identical_carried_families",
        "delta_1_zero_four_revert",
        "delta_2_adult_child_exit_timing",
        "delta_3_female_cohab_lift",
        "delta_4_count_conditional_bridge",
        "multigen_marginal_unchanged",
        "no_coupling_extension_below_55",
    ):
        assert key in notes and len(notes[key]) > 80


# --------------------------------------------------------------------------
# Verdict + per-seed pins
# --------------------------------------------------------------------------
def test_one_shot_verdict_pinned():
    v = _artifact()["verdict"]
    assert v["n_seeds_pass"] == PINNED_N_SEEDS_PASS
    assert v["gate_2b_pass"] == (PINNED_N_SEEDS_PASS >= 4)
    assert v["n_gate_seeds"] == 5
    assert v["n_gated_cells"] == N_GATED


def test_per_seed_gated_pass_counts_pinned():
    a = _artifact()
    got = [
        {s["seed"]: s["n_gated_pass"] for s in a["per_seed"]}[i]
        for i in GATE_SEEDS
    ]
    assert got == PINNED_PER_SEED_GATED_PASS


def test_precheck_reproduced_exactly():
    pc = _artifact()["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["holdout_sha256_all_match"] is True
    assert pc["reference_moments_max_abs_deviation"] <= 1e-12
    assert pc["rate_a_max_abs_deviation"] <= 1e-12


def test_gated_cells_match_floor_gate_partition():
    floor = json.loads(FLOOR.read_text())
    gated = set(floor["gate_partition"]["gate_eligible"])
    assert gated == set(_gate2b_tolerances())
    assert len(gated) == N_GATED


def test_undefined_draw_rule_not_triggered_and_run_valid():
    u = _artifact()["fresh_run_artifact_schema"]["undefined_draw_rule"]
    assert u["n_undefined_gated_draws"] == 0
    assert u["run_invalidated"] is False


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


def test_rbar_and_score_recompute_from_per_draw_rates():
    for s in _artifact()["per_seed"]:
        for rec in s["gated_cells"].values():
            rates = rec["per_draw_rate"]
            assert len(rates) == N_DRAWS
            rbar = float(np.mean(rates))
            assert rbar == pytest.approx(rec["rbar"], abs=1e-12)
            assert rec["r_candidate"] == pytest.approx(rec["rbar"], abs=1e-15)
            if rbar > 0 and rec["rate_a"] > 0:
                expected = abs(math.log(rbar / rec["rate_a"]))
                assert rec["score"] == pytest.approx(expected, abs=1e-12)


def test_every_gated_pass_recomputes_and_seed_pass():
    for s in _artifact()["per_seed"]:
        for rec in s["gated_cells"].values():
            assert rec["pass"] == (rec["score"] <= rec["tolerance"])
        n_pass = sum(rec["pass"] for rec in s["gated_cells"].values())
        assert n_pass == s["n_gated_pass"]
        assert s["seed_pass"] == (n_pass == N_GATED)


def test_verdict_recomputes_from_seed_conjunction():
    a = _artifact()
    n_pass = sum(s["seed_pass"] for s in a["per_seed"])
    assert a["verdict"]["n_seeds_pass"] == n_pass
    assert a["verdict"]["gate_2b_pass"] == (n_pass >= 4)


def test_stored_tolerances_match_locked_gates_yaml():
    tol = _gate2b_tolerances()
    for s in _artifact()["per_seed"]:
        for cell, rec in s["gated_cells"].items():
            assert rec["tolerance"] == pytest.approx(tol[cell], abs=1e-12)


# --------------------------------------------------------------------------
# Byte-identical carried families + multigen marginal + cleared + grandchild
# --------------------------------------------------------------------------
def test_strict_carried_family_scores_byte_identical_to_candidate_5():
    a = _artifact()
    byt = a["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["byte_identical"] is True
    assert byt["max_abs_score_deviation_vs_candidate5"] == 0.0
    c5 = json.loads(CANDIDATE5_ARTIFACT.read_text())
    by6 = {s["seed"]: s for s in a["per_seed"]}
    by5 = {s["seed"]: s for s in c5["per_seed"]}
    carried = [
        c
        for c in _gate2b_tolerances()
        if c.startswith(STRICT_CARRIED_PREFIXES)
        or c in ("multigen_entry", "multigen_exit")
    ]
    assert set(carried) == set(byt["strict_carried_cells"])
    for seed in GATE_SEEDS:
        for cell in carried:
            s6 = by6[seed]["gated_cells"][cell]["score"]
            s5 = by5[seed]["gated_cells"][cell]["score"]
            assert s6 == pytest.approx(s5, abs=1e-12), (seed, cell)


def test_grandchild_55plus_byte_identical_to_candidate_5():
    byt = _artifact()["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["coresident_grandchild_55plus_byte_identical"] is True
    assert (
        byt["coresident_grandchild_55plus_female_max_abs_score_deviation"]
        == 0.0
    )


def test_multigen_marginal_unchanged_vs_candidate_5():
    mg = _artifact()["comparison_across_candidates"][
        "multigen_marginal_unchanged_check"
    ]
    assert mg["marginal_unchanged"] is True
    assert mg["max_abs_score_deviation_vs_candidate5"] == 0.0
    for dev in mg["per_cell_max_abs_score_deviation"].values():
        assert dev == 0.0


def test_cleared_families_still_clear_incl_grandchild_one():
    chk = _artifact()["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    assert "coresident_grandchild" in chk["families"]
    for fam, d in chk["detail"].items():
        assert d["candidate6_pass_rate"] == 1.0, fam


def test_progression_recomputes_c1_to_c6():
    a = _artifact()
    prog = a["comparison_across_candidates"]["per_family_progression"]
    # every gated family carries a c1..c6 pass-rate progression.
    for row in prog.values():
        for n in (1, 2, 3, 4, 5, 6):
            assert f"candidate{n}_pass_rate" in row
    # hh_size improves under the count-conditional bridge (delta 4).
    hh = prog["hh_size"]
    assert hh["candidate6_pass_rate"] >= hh["candidate5_pass_rate"]


# --------------------------------------------------------------------------
# The four deltas' fit-vs-raw checks (vs forensics-3)
# --------------------------------------------------------------------------
def test_delta_1_revert_basis_recorded():
    d1 = _artifact()["c6_delta_checks"]["checks"]["delta_1_zero_four_revert"]
    assert d1["revert_band"] == [0, 4]
    # forensics-3: child-record HIGHER than observable at 0-4 (the artifact).
    assert (
        d1["forensics_child_record_0_4_nm"] > d1["forensics_observable_0_4_nm"]
    )
    assert d1["sign_inversion_confirmed"] is True


def test_delta_2_no_coupling_extension_below_55():
    d2 = _artifact()["c6_delta_checks"]["checks"][
        "delta_2_adult_child_exit_timing"
    ]
    assert d2["coupling_stays_at_55_plus"] is True
    assert d2["no_coupling_extension_to_45_54"] is True
    assert d2["linked_married_35_44m_is_supply_deferred_to_c7"] is True
    # the 45-54|female reference coupling is weak (far below the ~5x at 55+).
    assert d2["forensics_45_54f_reference_coupling_lift_ratio"] < 2.0
    # every draw records the invariant that no coupling fires below age 55.
    for s in _artifact()["per_seed"]:
        assert (
            s["delta_stats"]["delta_2_adult_child_exit_timing"][
                "max_p_coupled_below_55"
            ]
            == 0.0
        )


def test_delta_3_female_overlay_shortfall_recorded():
    d3 = _artifact()["c6_delta_checks"]["checks"]["delta_3_female_cohab_lift"]
    assert d3["forensics_target_classification"] == (
        "overlay_cohabitation_shortfall"
    )
    assert d3["forensics_overlay_gap_25_34f"] < 0  # under-supplied overlay
    assert d3["forensics_legal_gap_25_34f"] > 0  # legal core already over
    assert d3["legal_top_up_applied"] is False


def test_delta_4_reproduces_q10_feasibility():
    d4 = _artifact()["c6_delta_checks"]["checks"][
        "delta_4_count_conditional_bridge"
    ]
    rep = d4["reproduces_feasibility_0_1887_0_1709_0_1303"]
    assert rep["all_three_clear_on_committed_c5_core"] is True
    # the convolution on the committed candidate-5 core reproduces ~0.1887/
    # 0.1709/0.1303 (within the per-seed train-conditional deviation).
    mine = d4["implied_hh_from_committed_c5_core_MINE"]
    assert mine["3"] == pytest.approx(0.1887, abs=0.003)
    assert mine["4"] == pytest.approx(0.1709, abs=0.003)
    assert mine["5+"] == pytest.approx(0.1303, abs=0.003)


def test_delta_stats_recorded_for_all_four_deltas():
    for s in _artifact()["per_seed"]:
        ds = s["delta_stats"]
        assert set(ds) == {
            "delta_1_zero_four_revert",
            "delta_2_adult_child_exit_timing",
            "delta_3_female_cohab_lift",
            "delta_4_count_conditional_bridge",
        }


def test_carried_blocker_analysis_flags_inert_spouse_cell():
    a = _artifact()
    blk = a["carried_blocker_analysis"]
    cell = "coresident_spouse.25-34|female"
    # on any seed where the inert spouse cell fails, it is recorded as a carried
    # blocker (the delta-3 refit coincides with candidate 4 there).
    for s in a["per_seed"]:
        rec = blk["per_seed"][str(s["seed"])]
        if not s["gated_cells"][cell]["pass"]:
            assert cell in rec["carried_blockers"]
    # its per-seed score is essentially unchanged from candidate 5 (the female
    # single-year refit coincides with candidate 4 at 25-34; a negligible
    # ~2.6e-7 cohab-persistence propagation from the 35-44 refit never flips
    # its pass/fail).
    c5 = json.loads(CANDIDATE5_ARTIFACT.read_text())
    by5 = {s["seed"]: s for s in c5["per_seed"]}
    by6 = {s["seed"]: s for s in a["per_seed"]}
    for seed in GATE_SEEDS:
        s6 = by6[seed]["gated_cells"][cell]
        s5 = by5[seed]["gated_cells"][cell]["score"]
        assert s6["score"] == pytest.approx(s5, abs=1e-5)
        assert s6["pass"] == by5[seed]["gated_cells"][cell]["pass"]
