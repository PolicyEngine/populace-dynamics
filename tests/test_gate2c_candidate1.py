"""Tests for the gate-2c candidate-1 one-shot scored run.

Candidate 1 (issue #42 comment 4950250151) is the first candidate against the
LOCKED gate-2c contract: couple formation composed from the certified
tranche-2a components plus train-fitted joints (the assortative kernel, the
around-event shift kernels, the shared-earnings cells).

The one-shot outcome is pinned below from the committed artifact
``runs/gate2c_hazard_v1.json``: the gate does NOT pass -- 1 of 5 seeds pass
(only seed 2), bound entirely by the ``first_marriage_by_earnings`` family
(the earnings-conditional marriage-hazard composition class the registration
forecast). Every other family clears every cell on every seed. Always
runnable; the PSID / pe-us reproduction pin lives in
``tests/test_gate2c_candidate1_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2c_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2c_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 27
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4950250151"
EXACT_ATOL = 1e-12

#: Pinned one-shot outcome (from the committed artifact).
PINNED_N_SEEDS_PASS = 1
PINNED_GATE_PASS = False
PINNED_SEED_PASS = {"0": False, "1": False, "2": True, "3": False, "4": False}
PINNED_PER_SEED_GATED_PASS = [25, 26, 27, 26, 26]
#: The five failing (cell, seed) pairs -- all in first_marriage_by_earnings.
PINNED_FAILING_CELL_SEEDS = {
    ("first_marriage_by_earnings.t1.18-24|female", 0),
    ("first_marriage_by_earnings.t1.18-24|female", 4),
    ("first_marriage_by_earnings.t3.18-24|female", 1),
    ("first_marriage_by_earnings.t3.25-34|male", 0),
    ("first_marriage_by_earnings.t3.25-34|male", 3),
}
#: Pinned per-family cell-seed pass rates.
PINNED_FAMILY_PASS_RATE = {
    "assort_mating": 1.0,
    "first_marriage_by_earnings": 0.875,
    "remarriage_by_earnings": 1.0,
    "earnings_around_marriage": 1.0,
    "earnings_around_divorce": 1.0,
    "shared_earnings_ratio": 1.0,
}


@pytest.fixture(scope="module")
def art():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def floor():
    return json.loads(FLOOR.read_text())


@pytest.fixture(scope="module")
def locked_tolerances():
    gates = yaml.safe_load(GATES.read_text())
    th = gates["gates"]["gate_2"]["gate_2c"]["thresholds"]
    tol = {}
    for view in th["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


# --------------------------------------------------------------------------
# Identity + registration
# --------------------------------------------------------------------------
def test_artifact_present_and_identity(art):
    assert art["schema_version"] == "gate2c_hazard.v1"
    assert art["gate"] == "gate_2c"
    assert art["candidate"] == "candidate 1"
    assert art["registration_pointer"] == REGISTRATION_POINTER
    assert REGISTRATION_POINTER in art["spec_registration"]
    assert REGISTRATION_POINTER in art["one_shot"]
    assert "publishes regardless" in art["one_shot"]


def test_five_components_declared(art):
    comps = art["model"]["five_components"]
    for key in (
        "1_marriage_events",
        "2_who_marries_whom",
        "3_spouse_age",
        "4_event_window_dynamics",
        "5_shared_earnings_cells",
    ):
        assert key in comps and len(comps[key]) > 10
    assert art["model"]["n_deciles"] == 10
    assert art["model"]["kernel_smoothing_alpha"] > 0
    assert len(art["model"]["certified_spec_sha256"]) == 64


def test_forecast_recorded_not_graded(art):
    fc = art["pre_registered_forecast"]
    assert fc["p_gate_pass_4_of_5"] == "0.10-0.25"
    assert len(fc["modal_failure_classes_in_order"]) == 3
    assert "does NOT grade" in fc["grading_note"]
    assert "forensics-1" in fc["primary_value_if_fail"]


def test_spec_resolution_notes_present(art):
    notes = art["spec_resolution_notes"]
    for key in (
        "five_components_source",
        "assortative_kernel",
        "directed_both_orientation_emission",
        "committed_cut_provenance",
        "event_window_support_and_detrend",
        "reference_moments_reused_verbatim",
        "shared_earnings_ratio_is_a_per_input_shape_moment",
        "rng_topology",
        "spouse_age_inert_for_gated_cells",
    ):
        assert key in notes and len(notes[key]) > 40


# --------------------------------------------------------------------------
# Verdict + per-seed conjunction (pinned): the gate does NOT pass (1/5)
# --------------------------------------------------------------------------
def test_one_shot_verdict_pinned_fails_1_of_5(art):
    v = art["verdict"]
    assert v["gate_2c_pass"] is PINNED_GATE_PASS
    assert v["n_seeds_pass"] == PINNED_N_SEEDS_PASS
    assert v["n_gated_cells"] == N_GATED
    assert v["seed_pass"] == PINNED_SEED_PASS


def test_per_seed_gated_pass_counts_pinned(art):
    counts = [s["n_gated_pass"] for s in art["per_seed"]]
    assert counts == PINNED_PER_SEED_GATED_PASS
    for s in art["per_seed"]:
        assert s["n_gated"] == N_GATED
        assert s["seed_pass"] == (s["n_gated_pass"] == N_GATED)


def test_all_failures_are_first_marriage_family(art):
    fails = art["verdict"]["all_failing_gated_cells"]
    got = {(f["cell"], f["seed"]) for f in fails}
    assert got == PINNED_FAILING_CELL_SEEDS
    assert all(f["family"] == "first_marriage_by_earnings" for f in fails)
    # every failure is marginal (just over tolerance).
    assert all(1.0 < f["score_over_tolerance"] < 1.5 for f in fails)


def test_verdict_recomputes_from_seed_conjunction(art):
    n_pass = sum(1 for s in art["per_seed"] if s["seed_pass"])
    assert n_pass == art["verdict"]["n_seeds_pass"]
    assert art["verdict"]["gate_2c_pass"] == (n_pass >= 4)
    # each seed_pass is exactly the all-gated conjunction.
    for s in art["per_seed"]:
        conj = all(r["pass"] for r in s["gated_cells"].values())
        assert s["seed_pass"] == conj


# --------------------------------------------------------------------------
# Locked contract + fresh-run schema
# --------------------------------------------------------------------------
def test_gated_cells_match_floor_gate_partition(art, floor):
    gated = set(floor["gate_partition"]["gate_eligible"])
    scored = set(art["per_seed"][0]["gated_cells"].keys())
    assert scored == gated
    assert len(scored) == N_GATED


def test_stored_tolerances_match_locked_gates_yaml(art, locked_tolerances):
    assert len(locked_tolerances) == N_GATED
    g0 = art["per_seed"][0]["gated_cells"]
    for cell, t in locked_tolerances.items():
        assert abs(g0[cell]["tolerance"] - t) <= EXACT_ATOL, cell


def test_tolerances_match_floor_draft_thresholds(art, floor):
    check = art["protocol"]["tolerance_cross_check_vs_floor"]
    assert check["tolerances_match_floor_draft_thresholds"] is True
    assert check["tolerance_cells_equal_floor_gate_eligible"] is True
    assert check["k"] == 4 and check["rounding"] == 3
    # independently: locked tolerance == round(floor mean + 4*sd, 3).
    nf = floor["noise_floor_seeds_0_99"]
    g0 = art["per_seed"][0]["gated_cells"]
    for cell in g0:
        expect = round(nf[cell]["mean"] + 4 * nf[cell]["sd"], 3)
        assert abs(g0[cell]["tolerance"] - expect) <= EXACT_ATOL, cell


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


def test_dispersion_disclosure_present(art):
    disp = art["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    assert disp["report_only"] is True
    assert disp["gated"] is False
    assert len(disp["cells"]) == N_GATED * len(GATE_SEEDS)


# --------------------------------------------------------------------------
# Per-family decomposition (the run's forensic value)
# --------------------------------------------------------------------------
def test_per_family_decomposition_pinned(art):
    dec = art["per_family_decomposition"]
    assert set(dec) == set(PINNED_FAMILY_PASS_RATE)
    for fam, rate in PINNED_FAMILY_PASS_RATE.items():
        assert abs(dec[fam]["cell_seed_pass_rate"] - rate) <= 1e-9, fam
        assert dec[fam]["mechanism"]


def test_sole_binding_family_is_first_marriage(art):
    dec = art["per_family_decomposition"]
    below = [f for f, d in dec.items() if d["cell_seed_pass_rate"] < 1.0]
    assert below == ["first_marriage_by_earnings"]
    # the forecast named this class second; the diagonal (assort_mating) held.
    assert dec["assort_mating"]["cell_seed_pass_rate"] == 1.0


def test_decomposition_family_pass_counts_consistent(art):
    dec = art["per_family_decomposition"]
    for d in dec.values():
        assert d["n_cell_seed"] == d["n_cells"] * len(GATE_SEEDS)
        rate = d["n_cell_seed_pass"] / d["n_cell_seed"]
        assert abs(rate - d["cell_seed_pass_rate"]) <= 1e-9


# --------------------------------------------------------------------------
# Precheck (hard stop) reproduced the floor exactly
# --------------------------------------------------------------------------
def test_precheck_reproduced_exactly(art):
    pc = art["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["reference_moments_max_abs_deviation"] <= EXACT_ATOL
    assert pc["rate_a_max_abs_deviation"] <= EXACT_ATOL
    assert pc["holdout_sha256_all_match"] is True


def test_protocol_estimator_and_pass_rule(art):
    p = art["protocol"]
    assert p["option"] == "a"
    assert p["estimator"] == "mean_over_K20_draws"
    assert p["n_draws"] == N_DRAWS
    assert "5200 + k" in p["draw_rng_rule"]
    assert "4 of 5" in p["pass_rule"]
    assert "component_id" in p["split"]


def test_registration_pointer_recorded_in_pins(art):
    pins = art["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2c_hazard.v1"
    assert len(pins["certified_spec_sha256"]) == 64
    assert len(pins["floor_run_sha256"]) == 64
