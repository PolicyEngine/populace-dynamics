"""Tests for the gate-2b candidate-7 one-shot scored run.

Candidate 7 (issue #42 comment 4948186843) is candidate 6 with EXACTLY TWO
frozen deltas, each designed against a graded gate-2b forensics-4 finding
(``runs/gate2b_forensics4_v1.json``, grading 4948183531):

* **delta 1 -- enumeration conditioning** (``coresident_child`` 25-34|male,
  35-44|male): restrict the paternal-linked coresidence draw to ENUMERATED
  children (the joinable ``(parent, birth_year)`` keys); the non-joinable
  biological children (25.8% of linked exposure) cannot be drawn coresident,
  removing the dominant ``unenumerated_nonjoinable_supply`` channel;
* **delta 2 -- episode persistence** (``coresident_child`` 25-34|male,
  35-44|male): replace the independent per-wave occupancy with a correlated
  entry/persist/exit process fitted to the train episode-length mean (~5.93
  waves), CONSTRAINED to preserve the per-wave custodial marginal by band --
  removing the ``spell_length`` fragmentation channel.

Everything candidate 6 cleared or carried -- coresident_parent, coresident_spouse
(ALL bands, INCLUDING the fragile 25-34|female cell carried UNTOUCHED), multigen
(stock + transitions), parental_home_exit AND the 55+ coupled grandchild -- is
carried byte-faithfully. The one-shot outcome is pinned below from the committed
artifact ``runs/gate2b_hazard_v7.json``. Always runnable. The reproduction pin
lives in ``tests/test_gate2b_candidate7_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v6.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"
FORENSICS = ROOT / "runs" / "gate2b_forensics4_v1.json"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4948186843"

#: Pinned one-shot outcome (from the committed artifact runs/gate2b_hazard_v7).
PINNED_N_SEEDS_PASS = 0
PINNED_PER_SEED_GATED_PASS = [44, 43, 44, 42, 42]

#: Both registered target cells clear on every gate seed (the c6 universal
#: blockers -- both failed all 5 seeds in candidate 6).
TARGET_CELLS = (
    "coresident_child.25-34|male",
    "coresident_child.35-44|male",
)
FRAGILE_SPOUSE_CELL = "coresident_spouse.25-34|female"
STRICT_CARRIED_PREFIXES = (
    "coresident_parent.",
    "coresident_spouse.",
    "multigen.",
    "parental_home_",
)
DELTA_MODULE = "populace_dynamics.models.household_composition_sim_v7"


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
    assert a["schema_version"] == "gate2b_hazard.v7"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 7"
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["candidate6_registration_pointer"] == "4946285556"
    assert a["grading_pointer"] == "4948183531"
    assert a["forensics_artifact"] == "runs/gate2b_forensics4_v1.json"
    assert a["model"]["delta_module"] == DELTA_MODULE


def test_two_deltas_declared_and_mapped():
    a = _artifact()
    assert len(a["deltas_vs_candidate_6"]) == 2
    mapping = a["per_delta_target_family"]
    assert mapping["delta_1_enumeration_conditioning"] == "coresident_child"
    assert mapping["delta_2_episode_persistence"] == "coresident_child"


def test_windows_and_stream_tags_recorded():
    m = _artifact()["model"]
    assert m["grandchild_coupling_age_lo"] == 55  # coupling stays 55+
    assert m["custodial_revert_band"] == [0, 4]  # carried
    assert m["spell_child_max_age"] == 17
    assert m["delta_stream_tag_v6"] == 0xC6  # carried
    assert m["delta_stream_tag_v7"] == 0xC7  # new, isolated


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
        "delta_1_enumeration_conditioning",
        "delta_2_episode_persistence",
        "shadow_channel_named_residual",
        "fragile_spouse_cell_carried_untouched",
        "hh_size_5plus_structural_seed",
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
# Byte-identical carried families vs candidate 6 (incl. the fragile cell)
# --------------------------------------------------------------------------
def test_strict_carried_family_scores_byte_identical_to_candidate_6():
    a = _artifact()
    byt = a["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["byte_identical"] is True
    assert byt["max_abs_score_deviation_vs_candidate6"] == 0.0
    c6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by7 = {s["seed"]: s for s in a["per_seed"]}
    by6 = {s["seed"]: s for s in c6["per_seed"]}
    carried = [
        c
        for c in _gate2b_tolerances()
        if c.startswith(STRICT_CARRIED_PREFIXES)
        or c in ("multigen_entry", "multigen_exit")
    ]
    assert set(carried) == set(byt["strict_carried_cells"])
    for seed in GATE_SEEDS:
        for cell in carried:
            s7 = by7[seed]["gated_cells"][cell]["score"]
            s6 = by6[seed]["gated_cells"][cell]["score"]
            assert s7 == pytest.approx(s6, abs=1e-12), (seed, cell)


def test_fragile_spouse_cell_carried_untouched_byte_identical():
    a = _artifact()
    fr = a["comparison_across_candidates"][
        "fragile_spouse_cell_carried_untouched"
    ]
    assert fr["cell"] == FRAGILE_SPOUSE_CELL
    assert fr["byte_identical"] is True
    assert fr["max_abs_score_deviation_vs_candidate6"] == 0.0
    # its 2/5 split-seed fragility is on the record (forensics-4 Q12).
    assert fr["forensics4_n_split_seeds_over_tolerance"] == 2
    c6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by7 = {s["seed"]: s for s in a["per_seed"]}
    by6 = {s["seed"]: s for s in c6["per_seed"]}
    for seed in GATE_SEEDS:
        s7 = by7[seed]["gated_cells"][FRAGILE_SPOUSE_CELL]
        s6 = by6[seed]["gated_cells"][FRAGILE_SPOUSE_CELL]
        assert s7["score"] == pytest.approx(s6["score"], abs=1e-12)
        assert s7["pass"] == s6["pass"]


def test_grandchild_55plus_byte_identical_to_candidate_6():
    byt = _artifact()["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["coresident_grandchild_55plus_byte_identical"] is True
    assert (
        byt["coresident_grandchild_55plus_female_max_abs_score_deviation"]
        == 0.0
    )


def test_multigen_marginal_unchanged_vs_candidate_6():
    mg = _artifact()["comparison_across_candidates"][
        "multigen_marginal_unchanged_check"
    ]
    assert mg["marginal_unchanged"] is True
    assert mg["max_abs_score_deviation_vs_candidate6"] == 0.0
    for dev in mg["per_cell_max_abs_score_deviation"].values():
        assert dev == 0.0


def test_cleared_families_still_clear_incl_grandchild_one():
    chk = _artifact()["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    assert "coresident_grandchild" in chk["families"]
    for fam, dtl in chk["detail"].items():
        assert dtl["candidate7_pass_rate"] == 1.0, fam


def test_progression_recomputes_c1_to_c7():
    a = _artifact()
    prog = a["comparison_across_candidates"]["per_family_progression"]
    for row in prog.values():
        for n in (1, 2, 3, 4, 5, 6):
            assert f"candidate{n}_pass_rate" in row
        assert "candidate7_pass_rate" in row
    # coresident_child improves under the two deltas (best on the ladder).
    child = prog["coresident_child"]
    assert child["candidate7_pass_rate"] >= child["candidate6_pass_rate"]
    assert child["delta_c6_to_c7"] > 0


# --------------------------------------------------------------------------
# The two registered target cells clear on every gate seed
# --------------------------------------------------------------------------
def test_both_target_cells_clear_on_every_seed():
    a = _artifact()
    for s in a["per_seed"]:
        for cell in TARGET_CELLS:
            assert s["gated_cells"][cell]["pass"], (s["seed"], cell)


def test_target_cells_are_c6_universal_blockers_now_fixed():
    # candidate 6 failed both target cells on ALL 5 seeds; candidate 7 clears
    # them on all 5 -- the two exactly-measured levers doing their job.
    c6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    for s in c6["per_seed"]:
        for cell in TARGET_CELLS:
            assert not s["gated_cells"][cell]["pass"], (s["seed"], cell)


# --------------------------------------------------------------------------
# Delta 1 + delta 2 fit-vs-raw checks (vs forensics-4)
# --------------------------------------------------------------------------
def test_delta1_nonjoinable_share_recorded_and_matches_forensics():
    d1 = _artifact()["c7_delta_checks"]["checks"][
        "delta_1_enumeration_conditioning"
    ]
    # 25.8% of linked exposure is non-joinable (9,500 of 36,887).
    assert d1["nonjoinable_share"] == pytest.approx(0.258, abs=0.02)
    assert d1["forensics_measured_nonjoinable_share"] == pytest.approx(
        9500 / 36887, abs=1e-6
    )
    assert d1["removes_channel"] == "unenumerated_nonjoinable_supply"


def test_delta2_episode_fit_target_and_rho():
    d2 = _artifact()["c7_delta_checks"]["checks"][
        "delta_2_episode_persistence"
    ]
    assert 0.0 < d2["fitted_persistence_rho"] <= 1.0
    # candidate-6 fragmented mean ~3.57; reference ~5.93; the train fit hits it.
    assert d2["forensics_sim_v6_episode_mean"] == pytest.approx(3.57, abs=0.1)
    assert d2["forensics_reference_episode_mean"] == pytest.approx(
        5.93, abs=0.1
    )
    assert d2["achieved_episode_mean_at_rho_train"] == pytest.approx(
        d2["target_reference_episode_mean_train"], abs=0.35
    )
    assert d2["removes_channel"] == "spell_length"


def test_channel_removal_arithmetic_reproduces_registration():
    arith = _artifact()["c7_delta_checks"]["checks"][
        "channel_removal_arithmetic"
    ]
    # removing the two measured channels takes 25-34|male to ~+0.004 and
    # 35-44|male to ~+0.020 (the registration's channel arithmetic).
    a25 = arith["coresident_child.25-34|male"]
    a35 = arith["coresident_child.35-44|male"]
    assert a25["predicted_residual_after_two_deltas"] == pytest.approx(
        0.004, abs=0.003
    )
    assert a35["predicted_residual_after_two_deltas"] == pytest.approx(
        0.020, abs=0.003
    )
    # the arithmetic is raw_miss minus (unenumerated + spell).
    for a in (a25, a35):
        expect = a["raw_cell_miss_candidate6"] - (
            a["unenumerated_nonjoinable_channel"] + a["spell_length_channel"]
        )
        assert a["predicted_residual_after_two_deltas"] == pytest.approx(
            expect, abs=1e-6
        )


def test_shadow_channel_named_residual_untouched():
    sh = _artifact()["c7_delta_checks"]["checks"][
        "shadow_channel_untouched_named_residual"
    ]
    # the shadow channel is positive and deliberately untouched.
    assert sh["coresident_child.25-34|male"] > 0
    assert sh["coresident_child.35-44|male"] > 0


# --------------------------------------------------------------------------
# Delta-2 marginal-preservation check + non-joinable share per band per seed
# --------------------------------------------------------------------------
def test_marginal_preservation_check_holds_every_seed():
    for s in _artifact()["per_seed"]:
        mp = s["marginal_preservation_check"]
        # the mixture preserves the per-wave marginal exactly in expectation;
        # the per-draw deviation is Monte-Carlo only (small).
        assert mp["mean_max_abs_dev_over_draws"] < 0.03, s["seed"]
        for band, rec in mp["per_band"].items():
            assert rec["mean_abs_deviation"] < 0.02, (s["seed"], band)


def test_nonjoinable_share_recorded_per_band_every_seed():
    for s in _artifact()["per_seed"]:
        band_share = s["delta_stats"]["delta_1_enumeration_conditioning"][
            "sim_nonjoinable_share_by_band"
        ]
        assert set(band_share) == {
            "0-4",
            "5-12",
            "13-17",
            "18-24",
            "25-60",
        }
        # the adult (25-60) band carries the highest non-joinable share.
        assert band_share["25-60"] > band_share["0-4"]


def test_episode_fit_vs_raw_recorded_every_seed():
    for s in _artifact()["per_seed"]:
        ef = s["episode_length_fit_vs_raw"]
        assert ef["raw_candidate6_sim_episode_mean"] == pytest.approx(
            3.57, abs=0.1
        )
        assert ef["reference_episode_mean"] == pytest.approx(5.93, abs=0.1)
        # the holdout simulated episode mean is lifted well above the c6 raw.
        assert ef["sim_holdout_episode_mean_over_draws"] > 4.5


# --------------------------------------------------------------------------
# hh_size carry (moves via the delta-1 linked-count reduction) + blockers
# --------------------------------------------------------------------------
def test_hh_size_carry_check_recorded():
    hh = _artifact()["comparison_across_candidates"]["hh_size_carry_check"]
    assert set(hh["cells"]) == {"hh_size.3", "hh_size.4"}
    for dtl in hh["detail"].values():
        assert "max_abs_score_deviation_vs_candidate6" in dtl
        assert "candidate7_pass_all_seeds" in dtl


def test_carried_blocker_analysis_flags_fragile_spouse_cap():
    blk = _artifact()["carried_blocker_analysis"]
    # the fragile spouse cell caps the seeds where it exceeds tolerance.
    for s in _artifact()["per_seed"]:
        rec = blk["per_seed"][str(s["seed"])]
        if not s["gated_cells"][FRAGILE_SPOUSE_CELL]["pass"]:
            assert rec["fragile_spouse_cell_caps_seed"] is True
            assert FRAGILE_SPOUSE_CELL in rec["carried_blockers"]
