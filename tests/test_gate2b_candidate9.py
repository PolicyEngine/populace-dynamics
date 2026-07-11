"""Tests for the gate-2b candidate-9 one-shot scored run.

Candidate 9 (issue #42 comment 4948839837) is candidate 8 with EXACTLY ONE
frozen delta against the graded candidate-8 collateral (grading 4948838962): the
delta-1 completed-fertility swap is CONFINED to the forensics-5 deficit cohorts x
sex -- ``{55-64, 65-74} x male`` and ``{45-54, 65-74} x female`` -- with every
other cohort reverting to the sim's own distribution (candidate-7 behavior).
Everything else in candidate 8 (the delta-2 cohab-overlay lift, the delta-3
band-signed retention + link-coverage refit and every carried family) is carried
BYTE-FAITHFULLY.

The one-shot outcome is pinned below from the committed artifact
``runs/gate2b_hazard_v9.json``: the gate PASSES 4/5 (the sole miss is
``hh_size.5+`` on seed 3, the priced modal residual the registration named).
Always runnable. The reproduction pin lives in
``tests/test_gate2b_candidate9_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v9.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v7.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"
FORENSICS = ROOT / "runs" / "gate2b_forensics5_v1.json"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4948839837"
EXACT_ATOL = 1e-12

#: Pinned one-shot outcome (from the committed artifact runs/gate2b_hazard_v9).
PINNED_N_SEEDS_PASS = 4
PINNED_GATE_PASS = True
PINNED_PER_SEED_GATED_PASS = [46, 46, 46, 45, 46]
#: The sole failing (cell, seed): the priced modal residual.
PINNED_SOLE_FAILING_CELL = "hh_size.5+"
PINNED_SOLE_FAILING_SEED = 3

#: The four deficit cohorts the fertility lift is confined to (byte-identical to
#: candidate 8).
SCOPE_CHILD_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
#: The four candidate-8 collateral cells (revert to candidate 7, their cleared
#: state).
COLLATERAL_CELLS = (
    "coresident_child.35-44|male",
    "coresident_child.35-44|female",
    "coresident_child.45-54|male",
    "coresident_child.55-64|female",
)
STRICT_CARRIED_PREFIXES = (
    "coresident_parent.",
    "multigen.",
    "parental_home_",
    "coresident_grandchild.",
    "coresident_spouse.",
)


@pytest.fixture(scope="module")
def art():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def c8():
    return json.loads(CANDIDATE8_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def c7():
    return json.loads(CANDIDATE7_ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Identity, the one delta, scope + collateral mapping
# --------------------------------------------------------------------------
def test_artifact_present_and_identity(art):
    assert art["schema_version"] == "gate2b_hazard.v9"
    assert art["gate"] == "gate_2b"
    assert art["candidate"] == "candidate 9"
    assert art["registration_pointer"] == REGISTRATION_POINTER
    assert art["candidate8_registration_pointer"] == "4948604739"
    assert art["candidate8_grading_pointer"] == "4948838962"
    assert art["forensics_artifact"] == "runs/gate2b_forensics5_v1.json"


def test_the_one_delta_declared_and_mapped(art):
    delta = art["delta_vs_candidate_8"]
    assert "confined to the forensics-5 deficit cohorts" in delta
    assert "candidate-7 behavior" in delta
    # the two carried deltas are declared byte-faithful.
    assert len(art["deltas_carried_from_candidate_8"]) == 2
    tgt = art["per_delta_target_family"]
    assert tgt["delta_1_fertility_core_lift_SCOPED"] == [
        "coresident_child",
        "hh_size",
    ]


def test_model_records_scope_and_reverted_cells(art):
    m = art["model"]
    assert m["delta_stream_tag_v7"] == 0xC7  # carried
    assert m["delta_stream_tag_v8"] == 0xC8  # unchanged tag (identical to c8)
    assert set(m["fertility_lift_scope_cells"]) == set(SCOPE_CHILD_CELLS)
    assert set(m["candidate8_collateral_cells"]) == set(COLLATERAL_CELLS)
    # the collateral cells are a subset of the reverted cells.
    assert set(COLLATERAL_CELLS) <= set(m["reverted_child_cells"])
    # scope and reverted are disjoint and partition the gated child cells.
    assert not (set(m["fertility_lift_scope_cells"]) & set(COLLATERAL_CELLS))


def test_one_shot_and_forecast_recorded_not_graded(art):
    assert "4948839837" in art["one_shot"]
    assert "independent verification" in art["one_shot"]
    fc = art["pre_registered_forecast"]
    assert fc["p_gate_pass_4_of_5"] == "0.50-0.70"
    assert "does NOT grade" in fc["grading_note"]
    # the modal residual is named as hh_size.5+.
    assert "hh_size.5+" in fc["modal_outcome_if_fail"]


def test_spec_resolution_notes_present(art):
    notes = art["spec_resolution_notes"]
    for key in (
        "the_one_delta_vs_candidate8",
        "deficit_cohorts_lift_unchanged",
        "non_deficit_cohorts_revert_to_candidate7",
        "carried_deltas_byte_identical",
        "hh_size_5plus_priced_uncertainty",
        "pre_run_analytic_check",
    ):
        assert key in notes and len(notes[key]) > 40


# --------------------------------------------------------------------------
# Verdict + per-seed conjunction (pinned): the gate PASSES 4/5
# --------------------------------------------------------------------------
def test_one_shot_verdict_pinned_passes_4_of_5(art):
    v = art["verdict"]
    assert v["gate_2b_pass"] is PINNED_GATE_PASS
    assert v["n_seeds_pass"] == PINNED_N_SEEDS_PASS
    assert v["n_gated_cells"] == N_GATED
    assert v["seed_pass"] == {
        "0": True,
        "1": True,
        "2": True,
        "3": False,
        "4": True,
    }


def test_per_seed_gated_pass_counts_pinned(art):
    counts = [s["n_gated_pass"] for s in art["per_seed"]]
    assert counts == PINNED_PER_SEED_GATED_PASS
    for s in art["per_seed"]:
        assert s["n_gated"] == N_GATED
        assert s["seed_pass"] == (s["n_gated_pass"] == N_GATED)


def test_sole_failing_cell_is_hh_size_5plus_seed_3(art):
    """The only miss is hh_size.5+ on seed 3 -- the priced modal residual."""
    fails = art["verdict"]["all_failing_gated_cells"]
    assert len(fails) == 1
    assert fails[0]["cell"] == PINNED_SOLE_FAILING_CELL
    assert fails[0]["seed"] == PINNED_SOLE_FAILING_SEED
    assert fails[0]["family"] == "hh_size"


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
# Byte-identity vs candidate 8 (carries + spouse + deficit scope cells)
# --------------------------------------------------------------------------
def test_byte_identical_vs_candidate8(art, c8):
    chk = art["comparison_across_candidates"]["byte_identical_vs_candidate8"]
    assert chk["byte_identical"] is True
    assert chk["max_abs_score_deviation_vs_candidate8"] <= EXACT_ATOL
    # spot-check directly: carried families, all spouse AND the deficit scope
    # child cells equal candidate 8 to bit precision.
    c9_by_seed = {s["seed"]: s for s in art["per_seed"]}
    c8_by_seed = {s["seed"]: s for s in c8["per_seed"]}
    for seed in GATE_SEEDS:
        for cell in c9_by_seed[seed]["gated_cells"]:
            carried = cell.startswith(STRICT_CARRIED_PREFIXES) or cell in (
                "multigen_entry",
                "multigen_exit",
            )
            if carried or cell in SCOPE_CHILD_CELLS:
                a = c9_by_seed[seed]["gated_cells"][cell]["score"]
                b = c8_by_seed[seed]["gated_cells"][cell]["score"]
                assert abs(a - b) <= EXACT_ATOL, (cell, seed)


def test_deficit_scope_cells_byte_identical_to_candidate8(art):
    chk = art["comparison_across_candidates"]["byte_identical_vs_candidate8"]
    dev = chk["scope_child_cells_max_dev_vs_candidate8"]
    for cell in SCOPE_CHILD_CELLS:
        assert dev[cell] <= EXACT_ATOL, cell


def test_reverted_child_cells_byte_identical_to_candidate7(art, c7):
    chk = art["comparison_across_candidates"][
        "reverted_child_cells_vs_candidate7"
    ]
    assert chk["byte_identical_to_candidate7"] is True
    assert chk["max_abs_score_deviation_vs_candidate7"] <= EXACT_ATOL
    # the four candidate-8 collateral cells are among the reverted cells and
    # MOVED vs candidate 8 (the scope change's point).
    for cell in COLLATERAL_CELLS:
        assert cell in chk["cells"]
        assert chk["moved_vs_candidate8"][cell] > 0.0
    # spot-check directly vs candidate 7.
    c9_by_seed = {s["seed"]: s for s in art["per_seed"]}
    c7_by_seed = {s["seed"]: s for s in c7["per_seed"]}
    for seed in GATE_SEEDS:
        for cell in chk["cells"]:
            a = c9_by_seed[seed]["gated_cells"][cell]["score"]
            b = c7_by_seed[seed]["gated_cells"][cell]["score"]
            assert abs(a - b) <= EXACT_ATOL, (cell, seed)


def test_cleared_families_still_clear(art):
    chk = art["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    for fam in (
        "coresident_parent",
        "coresident_spouse",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
        "coresident_grandchild",
    ):
        assert chk["detail"][fam]["still_clears"] is True


def test_no_seed_capped_by_a_carried_blocker(art):
    """The gate loss (seed 3) is on the change surface (hh_size.5+), not any
    cell byte-identical to candidate 8."""
    blocker = art["carried_blocker_analysis"]
    assert blocker["n_seeds_capped_by_carried_cell"] == 0
    assert blocker["max_attainable_seeds_given_carried_blockers"] == 5
    for _seed, rec in blocker["per_seed"].items():
        assert rec["carried_blockers"] == []


# --------------------------------------------------------------------------
# Progression c1 -> c9 (the ladder: 0..0, c8=1, c9=4)
# --------------------------------------------------------------------------
def test_progression_c1_to_c9(art):
    comp = art["comparison_across_candidates"]
    verdicts = comp["candidate_verdicts"]
    for n in range(1, 8):
        v = verdicts.get(str(n), verdicts.get(n))
        assert v["n_seeds_pass"] == 0, n
    c8v = verdicts.get("8", verdicts.get(8))
    assert c8v["n_seeds_pass"] == 1
    assert art["verdict"]["n_seeds_pass"] == 4
    assert art["verdict"]["gate_2b_pass"] is True


def test_collateral_recovery_child_family_clears(art, c8):
    """The scoping recovers the coresident_child family to 1.0 (candidate 8's
    global lift capped it at 0.85 via the four collateral cells)."""
    dec8 = c8["per_family_decomposition"]["coresident_child"][
        "cell_seed_pass_rate"
    ]
    dec9 = art["per_family_decomposition"]["coresident_child"][
        "cell_seed_pass_rate"
    ]
    assert dec8 < 1.0
    assert dec9 == 1.0
    # every collateral cell passes every seed under the scoped lift.
    for s in art["per_seed"]:
        for cell in COLLATERAL_CELLS:
            assert s["gated_cells"][cell]["pass"], (cell, s["seed"])


def test_deficit_male_cells_hold_their_clears(art):
    """55-64|male and 65-74|male hold their candidate-8 clears (deficit cohorts;
    lift unchanged)."""
    for s in art["per_seed"]:
        for cell in (
            "coresident_child.55-64|male",
            "coresident_child.65-74|male",
        ):
            assert s["gated_cells"][cell]["pass"], (cell, s["seed"])


# --------------------------------------------------------------------------
# The PRE-RUN train-side analytic check (registration 4948839837)
# --------------------------------------------------------------------------
def test_pre_run_analytic_check_present_and_shaped(art):
    ac = art["pre_run_analytic_check"]
    assert ac is not None
    for cell in (
        "hh_size.5+",
        "coresident_child.55-64|male",
        "coresident_child.65-74|male",
    ) + COLLATERAL_CELLS:
        assert cell in ac["cells"]


def test_analytic_check_prices_hh_size_5plus(art):
    """The scoped lift forgoes the middle-cohort share of candidate 8's
    hh_size.5+ lift; the analytic check records that priced share."""
    hp = art["pre_run_analytic_check"]["hh_size_5plus_priced"]
    # candidate 8's global lift ~0.017; the scoped lift delivers only a small
    # slice; the middle cohorts carried the large majority (~0.87).
    assert hp["global_lift_candidate8"] > hp["scoped_lift_candidate9"] > 0.0
    assert 0.75 < hp["middle_cohort_share_of_lift"] < 0.95
    # the priced cell is predicted to clear on the seed-mean (barely -- it is
    # the closest-to-tolerance clearing cell).
    hh5 = art["pre_run_analytic_check"]["cells"]["hh_size.5+"]
    assert hh5["predicted_within_tolerance"] is True
    assert hh5["predicted_score"] < hh5["tolerance"]


def test_analytic_check_predicts_collateral_reverts_to_cleared(art):
    """Each collateral cell's scoped counterfactual is its sim rate (reverted);
    all four are predicted to clear."""
    cells = art["pre_run_analytic_check"]["cells"]
    for cell in COLLATERAL_CELLS:
        c = cells[cell]
        assert c["in_scope"] is False
        assert c["reverts_to_candidate7"] is True
        # scoped counterfactual == sim (no lift), predicted within tolerance.
        assert abs(c["scoped_counterfactual"] - c["sim_train"]) <= 1e-9
        assert c["predicted_within_tolerance"] is True


def test_analytic_check_scope_cells_keep_global_counterfactual(art):
    """The deficit scope cells' scoped counterfactual equals the global one
    (they are in scope): 55-64|male holds; 65-74|male's fertility-only cf
    under-predicts (its clear is delta-3, not delta-1)."""
    cells = art["pre_run_analytic_check"]["cells"]
    for cell in ("coresident_child.55-64|male", "coresident_child.65-74|male"):
        c = cells[cell]
        assert c["in_scope"] is True
        assert (
            abs(c["scoped_counterfactual"] - c["global_counterfactual"])
            <= 1e-9
        )
    # 55-64|male predicted to hold on fertility alone.
    assert (
        cells["coresident_child.55-64|male"]["predicted_within_tolerance"]
        is True
    )
    # 65-74|male's fertility-only counterfactual does NOT reach tolerance -- its
    # realized clear comes from the carried delta-3 exit/link closure.
    assert (
        cells["coresident_child.65-74|male"]["predicted_within_tolerance"]
        is False
    )


def test_analytic_check_vs_realized_recorded(art):
    """The pre-run prediction sits beside the realized holdout: the collateral
    cells and 55-64|male match; hh_size.5+'s magnitude prediction lands within
    ~0.001 of the realized seed-mean (it correctly priced the marginal cell).
    """
    avr = art["analytic_check_vs_realized"]["cells"]
    for cell in COLLATERAL_CELLS + ("coresident_child.55-64|male",):
        assert avr[cell]["prediction_matches_realized"] is True
        assert avr[cell]["realized_n_seeds_pass"] == 5
    hh5 = avr["hh_size.5+"]
    # magnitude agreement train->holdout.
    assert (
        abs(
            hh5["predicted_scoped_counterfactual_train"]
            - hh5["realized_seed_mean_rbar_holdout"]
        )
        < 0.005
    )
    # realized 4/5 (seed 3 missed): the priced modal residual.
    assert hh5["realized_n_seeds_pass"] == 4
    # 65-74|male realized clears 5/5 despite the fertility-only prediction.
    assert avr["coresident_child.65-74|male"]["realized_n_seeds_pass"] == 5


# --------------------------------------------------------------------------
# Carried delta fit-vs-raw checks reproduce candidate 8 / forensics-5
# --------------------------------------------------------------------------
def test_carried_delta_checks_reproduce_q15_q16_headlines(art):
    ck = art["c9_delta_checks"]["checks"]
    hh5 = ck["delta_1_fertility_core_lift"]["reproduction_hh_size_5plus"]
    assert hh5["headline"] == "0.128 -> 0.144 (vs reference 0.139)"
    q16 = ck["delta_2_cohab_overlay_lift"]["reproduction_spouse_25_34_female"]
    assert q16["headline"] == "0.588 -> 0.606 (vs reference 0.621)"


def test_carried_delta3_band_signs_recorded(art):
    d3 = art["c9_delta_checks"]["checks"]["delta_3_retention_link_refit"]
    bs = d3["band_sign_fit_vs_raw"]
    assert bs["coresident_child.45-54|female"]["exit_sign"] == (
        "reduce_over_retention"
    )
    assert bs["coresident_child.65-74|male"]["exit_sign"] == (
        "lift_under_retention"
    )


# --------------------------------------------------------------------------
# Dispersion disclosure (report-only)
# --------------------------------------------------------------------------
def test_dispersion_disclosure_present(art):
    disp = art["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    assert disp["report_only"] is True
    assert disp["gated"] is False
    assert len(disp["cells"]) == N_GATED
