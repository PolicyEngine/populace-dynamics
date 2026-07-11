"""Tests for the gate-2b candidate-8 one-shot scored run.

Candidate 8 (issue #42 comment 4948604739) is candidate 7 with EXACTLY THREE
frozen deltas, each designed against a graded gate-2b forensics-5 finding
(``runs/gate2b_forensics5_v1.json``, grading 4948603337):

* **delta 1 -- fertility-core lift** (``hh_size.5+`` /
  ``coresident_child.55-64|male``): swap the sim completed-family-size
  distribution to the train ``D_train[S]`` per (band, sex), holding the sim's
  own kernels (the Q15 analytic application; hh_size.5+ 0.128 -> 0.144,
  55-64|male 0.213 -> 0.255);
* **delta 2 -- cohabitation-overlay lift at 25-34|female**: lift the
  currently-non-spouse mass by the measured -0.045 overlay shortfall (Bernoulli
  superposition; 0.588 -> 0.606), age-band-specific;
* **delta 3 -- band-signed adult-child retention refit at parent 45+ +
  link-coverage**: close the Q14 EXIT-ORIGIN channel band-signed (lift 65-74
  under-retention, reduce 45-54|female over-retention) and the LINK-COVERAGE
  channel (-0.020 55-64|male, -0.016 65-74|male); the v7 interaction is the
  named residual.

Everything candidate 7 cleared or carried -- coresident_parent, coresident_spouse
(every band EXCEPT the delta-2 lifted 25-34|female), multigen (stock +
transitions), parental_home_exit AND coresident_grandchild -- is carried
byte-faithfully. The one-shot outcome is pinned below from the committed artifact
``runs/gate2b_hazard_v8.json``. Always runnable. The reproduction pin lives in
``tests/test_gate2b_candidate8_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v7.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"
FORENSICS = ROOT / "runs" / "gate2b_forensics5_v1.json"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4948604739"
EXACT_ATOL = 1e-12

#: Pinned one-shot outcome (from the committed artifact runs/gate2b_hazard_v8).
PINNED_N_SEEDS_PASS = 1
PINNED_PER_SEED_GATED_PASS = [43, 43, 45, 44, 46]

#: The three registered delta-target cells that WON on the holdout (the two
#: proven levers plus the band-signed retention refit).
WON_TARGET_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
    "coresident_child.45-54|female",
    "coresident_spouse.25-34|female",
    "hh_size.5+",
)
#: The delta-2 lifted cell (was the candidate-7 fragile carry).
LIFTED_SPOUSE_CELL = "coresident_spouse.25-34|female"
STRICT_CARRIED_PREFIXES = (
    "coresident_parent.",
    "multigen.",
    "parental_home_",
    "coresident_grandchild.",
)


@pytest.fixture(scope="module")
def art():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def c7():
    return json.loads(CANDIDATE7_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def forensics():
    return json.loads(FORENSICS.read_text())


# --------------------------------------------------------------------------
# Identity, deltas, protocol
# --------------------------------------------------------------------------
def test_artifact_present_and_identity(art):
    assert art["schema_version"] == "gate2b_hazard.v8"
    assert art["gate"] == "gate_2b"
    assert art["candidate"] == "candidate 8"
    assert art["registration_pointer"] == REGISTRATION_POINTER
    assert art["forensics_artifact"] == "runs/gate2b_forensics5_v1.json"
    assert art["grading_pointer"] == "4948603337"


def test_three_deltas_declared_and_mapped(art):
    assert len(art["deltas_vs_candidate_7"]) == 3
    tgt = art["per_delta_target_family"]
    assert tgt["delta_1_fertility_core_lift"] == [
        "coresident_child",
        "hh_size",
    ]
    assert tgt["delta_2_cohab_overlay_lift"] == ["coresident_spouse"]
    assert tgt["delta_3_retention_link_refit"] == ["coresident_child"]


def test_stream_tags_and_model_recorded(art):
    m = art["model"]
    assert m["delta_stream_tag_v7"] == 0xC7  # carried
    assert m["delta_stream_tag_v8"] == 0xC8  # new, isolated
    assert m["cohab_overlay_lift"] == 0.045
    assert m["cohab_overlay_lift_band"] == "25-34"
    assert m["size_buckets"] == ["0", "1", "2", "3", "4+"]
    assert "coresident_child.65-74|male" in m["retention_exit_cells"]
    assert "coresident_child.45-54|female" in m["retention_exit_cells"]
    assert set(m["link_coverage_cells"]) == {
        "coresident_child.55-64|male",
        "coresident_child.65-74|male",
    }


def test_one_shot_and_forecast_recorded_not_graded(art):
    assert "4948604739" in art["one_shot"]
    fc = art["pre_registered_forecast"]
    assert fc["p_gate_pass_4_of_5"] == "0.60-0.75"
    assert "does NOT grade" in fc["grading_note"]


def test_spec_resolution_notes_present(art):
    notes = art["spec_resolution_notes"]
    for key in (
        "rng_byte_identical_carried_families",
        "delta_1_fertility_core_lift",
        "delta_2_cohab_overlay_lift",
        "delta_3_retention_link_refit",
        "v7_interaction_named_residual",
    ):
        assert key in notes and len(notes[key]) > 40


# --------------------------------------------------------------------------
# Verdict + per-seed conjunction (pinned)
# --------------------------------------------------------------------------
def test_one_shot_verdict_pinned(art):
    v = art["verdict"]
    assert v["gate_2b_pass"] is False
    assert v["n_seeds_pass"] == PINNED_N_SEEDS_PASS
    assert v["n_gated_cells"] == N_GATED
    # only seed 4 passes.
    assert v["seed_pass"] == {
        "0": False,
        "1": False,
        "2": False,
        "3": False,
        "4": True,
    }


def test_per_seed_gated_pass_counts_pinned(art):
    counts = [s["n_gated_pass"] for s in art["per_seed"]]
    assert counts == PINNED_PER_SEED_GATED_PASS
    for s in art["per_seed"]:
        assert s["n_gated"] == N_GATED
        assert s["seed_pass"] == (s["n_gated_pass"] == N_GATED)


def test_precheck_reproduced_exactly(art):
    pc = art["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["reference_moments_max_abs_deviation"] <= EXACT_ATOL
    assert pc["rate_a_max_abs_deviation"] <= EXACT_ATOL
    assert pc["holdout_sha256_all_match"] is True


# --------------------------------------------------------------------------
# Locked contract + fresh-run schema
# --------------------------------------------------------------------------
def test_gated_cells_match_floor_gate_partition(art):
    floor = json.loads(FLOOR.read_text())
    gated = set(floor["gate_partition"]["gate_eligible"])
    scored = set(art["per_seed"][0]["gated_cells"].keys())
    assert scored == gated
    assert len(scored) == N_GATED


def test_undefined_draw_rule_not_triggered_and_run_valid(art):
    u = art["fresh_run_artifact_schema"]["undefined_draw_rule"]
    assert u["run_invalidated"] is False
    assert u["n_undefined_gated_draws"] == 0


def test_per_draw_per_cell_rates_shape_and_index(art):
    cube = art["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    assert cube["shape"] == [N_DRAWS, N_GATED, len(GATE_SEEDS)]
    assert cube["seed_index"] == GATE_SEEDS
    assert cube["k_index_draw_seeds"] == [
        DRAW_SEED_BASE + k for k in range(N_DRAWS)
    ]
    assert len(cube["rates"]) == N_DRAWS
    assert len(cube["rates"][0]) == N_GATED
    assert len(cube["rates"][0][0]) == len(GATE_SEEDS)


def test_rbar_and_score_recompute_from_per_draw_rates(art):
    cube = art["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    cells = cube["cell_index"]
    by_seed = {s["seed"]: s for s in art["per_seed"]}
    for si, seed in enumerate(cube["seed_index"]):
        for ci, cell in enumerate(cells):
            rates = [cube["rates"][k][ci][si] for k in range(N_DRAWS)]
            rbar = sum(rates) / N_DRAWS
            rec = by_seed[seed]["gated_cells"][cell]
            assert abs(rbar - rec["rbar"]) <= 1e-9, (cell, seed)
            rate_a = rec["rate_a"]
            if rbar > 0 and rate_a > 0:
                score = abs(math.log(rbar / rate_a))
                assert abs(score - rec["score"]) <= 1e-9, (cell, seed)
            assert rec["pass"] == (rec["score"] <= rec["tolerance"])


def test_verdict_recomputes_from_seed_conjunction(art):
    n_pass = sum(1 for s in art["per_seed"] if s["seed_pass"])
    assert n_pass == art["verdict"]["n_seeds_pass"]
    assert art["verdict"]["gate_2b_pass"] == (n_pass >= 4)


def test_stored_tolerances_match_locked_gates_yaml(art):
    gates = yaml.safe_load(GATES.read_text())
    th = gates["gates"]["gate_2"]["gate_2b"]["thresholds"]
    tol = {}
    for view in th["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    assert len(tol) == N_GATED
    g0 = art["per_seed"][0]["gated_cells"]
    for cell, t in tol.items():
        assert abs(g0[cell]["tolerance"] - t) <= EXACT_ATOL, cell


# --------------------------------------------------------------------------
# Byte-identical carries vs candidate 7 (spouse EXCLUDES 25-34|female)
# --------------------------------------------------------------------------
def test_strict_carried_family_scores_byte_identical_to_candidate_7(art, c7):
    chk = art["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert chk["byte_identical"] is True
    assert chk["max_abs_score_deviation_vs_candidate7"] <= EXACT_ATOL
    # spot-check directly: carried cells equal candidate 7 to bit precision.
    c8_by_seed = {s["seed"]: s for s in art["per_seed"]}
    c7_by_seed = {s["seed"]: s for s in c7["per_seed"]}
    for seed in GATE_SEEDS:
        for cell in c8_by_seed[seed]["gated_cells"]:
            fam_carried = cell.startswith(STRICT_CARRIED_PREFIXES) or cell in (
                "multigen_entry",
                "multigen_exit",
            )
            spouse_carried = (
                cell.startswith("coresident_spouse.")
                and cell != LIFTED_SPOUSE_CELL
            )
            if fam_carried or spouse_carried:
                a = c8_by_seed[seed]["gated_cells"][cell]["score"]
                b = c7_by_seed[seed]["gated_cells"][cell]["score"]
                assert abs(a - b) <= EXACT_ATOL, (cell, seed)


def test_multigen_marginal_unchanged_vs_candidate_7(art):
    chk = art["comparison_across_candidates"][
        "multigen_marginal_unchanged_check"
    ]
    assert chk["marginal_unchanged"] is True
    assert chk["max_abs_score_deviation_vs_candidate7"] <= EXACT_ATOL


def test_cleared_families_still_clear(art):
    chk = art["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    for fam in (
        "coresident_parent",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
        "coresident_grandchild",
    ):
        assert chk["detail"][fam]["still_clears"] is True


def test_no_seed_capped_by_a_carried_blocker(art):
    """Every failing cell is a delta-target family (coresident_child); no
    byte-identical carry fails, so the deltas own the outcome."""
    blocker = art["carried_blocker_analysis"]
    assert blocker["n_seeds_capped_by_carried_cell"] == 0
    for _seed, rec in blocker["per_seed"].items():
        assert rec["carried_blockers"] == []


# --------------------------------------------------------------------------
# Progression c1 -> c8
# --------------------------------------------------------------------------
def test_progression_recomputes_c1_to_c8(art):
    comp = art["comparison_across_candidates"]
    verdicts = comp["candidate_verdicts"]
    for n in (1, 2, 3, 4, 5, 6, 7):
        assert str(n) in verdicts or n in verdicts
    # candidate 8 is the first non-zero gate-2b seed count (c1-c7 all 0/5).
    for n, v in verdicts.items():
        assert v["n_seeds_pass"] == 0, n
    assert art["verdict"]["n_seeds_pass"] == 1


# --------------------------------------------------------------------------
# Delta 2: the fragile spouse cell is LIFTED (Q16 sign test)
# --------------------------------------------------------------------------
def test_lifted_spouse_cell_delta2_clears(art):
    lifted = art["comparison_across_candidates"]["lifted_spouse_cell_delta2"]
    assert lifted["cell"] == LIFTED_SPOUSE_CELL
    # candidate 7 carried it fragile (3/5); delta 2 lifts it to 5/5.
    assert lifted["candidate7_n_seeds_pass"] == 3
    assert lifted["candidate8_n_seeds_pass"] == 5


# --------------------------------------------------------------------------
# The registered delta targets WON; the fertility lift's collateral landed on
# the cleared middle cohorts (the honest one-shot decomposition).
# --------------------------------------------------------------------------
def test_won_target_cells_pass_every_seed(art):
    for s in art["per_seed"]:
        for cell in WON_TARGET_CELLS:
            assert s["gated_cells"][cell]["pass"], (cell, s["seed"])


def test_hard_65_74_male_cell_now_clears(art, c7):
    """coresident_child.65-74|male failed all 5 seeds in candidate 7 (0/5); the
    fertility lift + delta-3 exit/link closure clears it on every seed."""
    for s in c7["per_seed"]:
        assert not s["gated_cells"]["coresident_child.65-74|male"]["pass"]
    for s in art["per_seed"]:
        assert s["gated_cells"]["coresident_child.65-74|male"]["pass"]


def test_all_failing_cells_are_fertility_lift_collateral(art):
    """The only failing cells are the cleared middle cohorts the global
    fertility lift overshoots on the holdout -- 35-44|male/female, 45-54|male,
    55-64|female -- not the delta targets or the carries."""
    fails = {f["cell"] for f in art["verdict"]["all_failing_gated_cells"]}
    expected_collateral = {
        "coresident_child.35-44|male",
        "coresident_child.35-44|female",
        "coresident_child.45-54|male",
        "coresident_child.55-64|female",
    }
    assert fails <= expected_collateral, fails
    # every failing cell is a coresident_child cell (the delta-1 target family).
    assert all(c.startswith("coresident_child.") for c in fails)


def test_hh_size_family_fully_clears(art, c7):
    """Delta 1 lifts hh_size.5+ into tolerance; the hh_size family clears 5/5
    (candidate 7 failed hh_size.5+ on seeds 3 and 4)."""
    dec7 = c7["per_family_decomposition"]["hh_size"]["cell_seed_pass_rate"]
    dec8 = art["per_family_decomposition"]["hh_size"]["cell_seed_pass_rate"]
    assert dec7 < 1.0
    assert dec8 == 1.0


# --------------------------------------------------------------------------
# Delta fit-vs-raw reproduction checks (Q14 / Q15 / Q16)
# --------------------------------------------------------------------------
def test_delta_checks_reproduce_q15_q16_headlines(art):
    ck = art["c8_delta_checks"]["checks"]
    hh5 = ck["delta_1_fertility_core_lift"]["reproduction_hh_size_5plus"]
    assert hh5["headline"] == "0.128 -> 0.144 (vs reference 0.139)"
    assert abs(hh5["seed_mean_sim"] - 0.1272) < 0.005
    assert abs(hh5["seed_mean_counterfactual_lever"] - 0.1443) < 0.005
    m5564 = ck["delta_1_fertility_core_lift"][
        "reproduction_coresident_child_55_64_male"
    ]
    assert abs(m5564["seed_mean_sim"] - 0.2128) < 0.005
    assert abs(m5564["seed_mean_counterfactual_lever"] - 0.2547) < 0.005
    q16 = ck["delta_2_cohab_overlay_lift"]["reproduction_spouse_25_34_female"]
    assert q16["headline"] == "0.588 -> 0.606 (vs reference 0.621)"


def test_delta3_band_sign_fit_vs_raw_recorded(art):
    d3 = art["c8_delta_checks"]["checks"]["delta_3_retention_link_refit"]
    bs = d3["band_sign_fit_vs_raw"]
    # 45-54|female over-retains (+exit) -> reduce; the others lift (-exit).
    assert bs["coresident_child.45-54|female"]["exit_sign"] == (
        "reduce_over_retention"
    )
    assert bs["coresident_child.65-74|male"]["exit_sign"] == (
        "lift_under_retention"
    )
    assert bs["coresident_child.65-74|female"]["exit_sign"] == (
        "lift_under_retention"
    )
    # link-coverage closes at both older-male bands (negative channel).
    for cell in ("coresident_child.55-64|male", "coresident_child.65-74|male"):
        assert d3["link_coverage_share"][cell]["link_coverage_channel"] < 0.0


def test_v7_interaction_named_residual_untouched(art):
    resid = art["c8_delta_checks"]["checks"]["named_residual_v7_interaction"]
    assert "coresident_child.55-64|male" in resid
    assert "coresident_child.65-74|male" in resid
    assert "candidate-9" in resid["note"]


# --------------------------------------------------------------------------
# Dispersion disclosure (report-only)
# --------------------------------------------------------------------------
def test_dispersion_disclosure_present(art):
    disp = art["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    assert disp["report_only"] is True
    assert disp["gated"] is False
    assert len(disp["cells"]) == N_GATED
