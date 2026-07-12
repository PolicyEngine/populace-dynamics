"""Tests for the gate-2c candidate-2 one-shot scored run.

Candidate 2 (issue #42 comment 4950370498) is ONE targeted delta against
candidate 1: first-marriage timing gains earnings-axis conditioning via a
train-fitted multiplicative hazard modifier ``m(tercile | age band, sex)``
composed onto the certified 2a first-marriage hazard, normalized so the
age x sex timing marginal is preserved. Everything else is byte-carried from
candidate 1.

The one-shot outcome is pinned below from the committed artifact
``runs/gate2c_hazard_v2.json``: the gate PASSES -- 4 of 5 seeds pass (only
seed 2 fails, on two marginal ``first_marriage_by_earnings`` cells). The delta
resolves all five candidate-1 misses; the carried families are byte-identical
to candidate 1. Always runnable; the PSID / pe-us reproduction pin lives in
``tests/test_gate2c_candidate2_reproduction.py``; the modifier unit math in
``tests/test_couple_formation_v2.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2c_hazard_v2.json"
C1_ARTIFACT = ROOT / "runs" / "gate2c_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2c_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 27
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4950370498"
EXACT_ATOL = 1e-12

#: Pinned one-shot outcome (from the committed artifact): the gate PASSES 4/5.
PINNED_N_SEEDS_PASS = 4
PINNED_GATE_PASS = True
PINNED_SEED_PASS = {"0": True, "1": True, "2": False, "3": True, "4": True}
PINNED_PER_SEED_GATED_PASS = [27, 27, 25, 27, 27]
#: The two failing (cell, seed) pairs -- both seed 2, both first_marriage.
PINNED_FAILING_CELL_SEEDS = {
    ("first_marriage_by_earnings.t2.18-24|female", 2),
    ("first_marriage_by_earnings.t3.18-24|female", 2),
}
#: Per-family cell-seed pass rates (only first_marriage < 1.0 now: 38/40).
PINNED_FAMILY_PASS_RATE = {
    "assort_mating": 1.0,
    "first_marriage_by_earnings": 0.95,
    "remarriage_by_earnings": 1.0,
    "earnings_around_marriage": 1.0,
    "earnings_around_divorce": 1.0,
    "shared_earnings_ratio": 1.0,
}
#: The five candidate-1 misses the delta resolves.
PINNED_C1_MISSES_RESOLVED = 5
CARRIED_FAMILIES = {
    "assort_mating",
    "remarriage_by_earnings",
    "earnings_around_marriage",
    "earnings_around_divorce",
    "shared_earnings_ratio",
}


@pytest.fixture(scope="module")
def art():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def c1():
    return json.loads(C1_ARTIFACT.read_text())


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
    assert art["schema_version"] == "gate2c_hazard.v2"
    assert art["gate"] == "gate_2c"
    assert art["candidate"] == "candidate 2"
    assert art["registration_pointer"] == REGISTRATION_POINTER
    assert REGISTRATION_POINTER in art["spec_registration"]
    assert REGISTRATION_POINTER in art["one_shot"]
    assert "publishes regardless" in art["one_shot"]


def test_one_delta_declared(art):
    model = art["model"]
    assert model["module"].endswith("couple_formation_sim_v2")
    assert model["base_module"].endswith("couple_formation_sim_v1")
    assert "earnings-conditioned first-marriage" in model["one_delta"]
    delta = model["delta_modifier"]
    assert delta["conditioning"] == "m(tercile | age band, sex)"
    assert "residual" in delta["form"]
    assert delta["shrinkage_alpha"] > 0
    assert "sum_t m * phi_cert = 1" in delta["normalization"]
    assert delta["gated_marginal_bands"] == ["18-24", "25-34"]
    # the certified core is pinned unchanged.
    assert len(model["certified_spec_sha256"]) == 64
    assert model["n_deciles"] == 10


def test_forecast_recorded_not_graded(art):
    fc = art["pre_registered_forecast"]
    assert fc["p_gate_pass_4_of_5"] == "0.45-0.65"
    assert len(fc["named_expectations"]) == 4
    assert "does NOT grade" in fc["grading_note"]
    assert "verification round" in fc["if_pass"]


def test_spec_resolution_notes_present(art):
    notes = art["spec_resolution_notes"]
    # candidate-1 resolutions carried verbatim ...
    for key in (
        "five_components_source",
        "assortative_kernel",
        "directed_both_orientation_emission",
        "committed_cut_provenance",
        "event_window_support_and_detrend",
        "reference_moments_reused_verbatim",
        "spouse_age_inert_for_gated_cells",
        "rng_topology",
    ):
        assert key in notes and len(notes[key]) > 40
    # ... plus the candidate-2 delta resolutions.
    for key in (
        "delta_earnings_conditioned_first_marriage",
        "delta_modifier_normalization_and_marginal_preservation",
        "delta_modifier_shrinkage",
        "delta_byte_carry",
    ):
        assert key in notes and len(notes[key]) > 40


# --------------------------------------------------------------------------
# Verdict (pinned): the gate PASSES 4/5
# --------------------------------------------------------------------------
def test_one_shot_verdict_pinned_passes_4_of_5(art):
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


def test_all_failures_are_marginal_first_marriage(art):
    fails = art["verdict"]["all_failing_gated_cells"]
    got = {(f["cell"], f["seed"]) for f in fails}
    assert got == PINNED_FAILING_CELL_SEEDS
    assert all(f["family"] == "first_marriage_by_earnings" for f in fails)
    # every remaining failure is marginal (just over tolerance).
    assert all(1.0 < f["score_over_tolerance"] < 1.1 for f in fails)


def test_verdict_recomputes_from_seed_conjunction(art):
    n_pass = sum(1 for s in art["per_seed"] if s["seed_pass"])
    assert n_pass == art["verdict"]["n_seeds_pass"]
    assert art["verdict"]["gate_2c_pass"] == (n_pass >= 4)
    for s in art["per_seed"]:
        conj = all(r["pass"] for r in s["gated_cells"].values())
        assert s["seed_pass"] == conj


# --------------------------------------------------------------------------
# THE DELTA: byte-carry regression + marginal preservation + fit-vs-raw
# --------------------------------------------------------------------------
def test_byte_carry_regression_carried_families_identical(art):
    b = art["byte_carry_regression"]
    assert b["carried_byte_identical"] is True
    assert b["carried_max_abs_rate_deviation"] == 0.0
    assert set(b["carried_families"]) == CARRIED_FAMILIES
    assert b["delta_family"] == "first_marriage_by_earnings"
    # every carried family deviated by exactly 0.0; only the delta moved.
    pf = b["per_family_max_abs_rate_deviation"]
    for fam in CARRIED_FAMILIES:
        assert pf[fam] == 0.0, fam
    assert pf["first_marriage_by_earnings"] > 0.0
    assert b["delta_family_moved"] is True
    assert b["c1_registration_pointer"] == "4950250151"
    assert len(b["c1_artifact_sha256"]) == 64


def test_byte_carry_regression_comparison_coverage(art):
    b = art["byte_carry_regression"]
    # 25 carried cells x 5 seeds = 125 cell-seeds; x 20 draws = 2500.
    assert b["n_carried_cells_per_seed_compared"] == 125
    assert b["n_carried_per_draw_comparisons"] == 2500
    # 24 first_marriage cells x 5 seeds, all moved.
    assert b["delta_cell_seeds_total"] == 120
    assert b["delta_cell_seeds_moved"] == 120


def test_marginal_preservation_constraint_holds_every_seed(art):
    for s in art["per_seed"]:
        mg = s["marginal_preservation"]
        assert mg["constraint_holds_all_draws"] is True
        assert mg["constraint_max_abs_dev_from_one_over_draws"] <= 1e-9
        # the GATED-band timing marginal barely moves (Monte Carlo only).
        assert (
            mg["realized_pooled_band_hazard_max_abs_ln_gated_bands_over_draws"]
            < 0.05
        )


def test_modifier_fit_vs_raw_recorded_and_normalized(art):
    fvr = art["per_seed"][0]["fm_modifier_fit_vs_raw"]
    assert fvr["alpha"] == art["model"]["delta_modifier"]["shrinkage_alpha"]
    assert fvr["constraint_max_abs_dev_from_one"] <= 1e-9
    # 2 sexes x 4 bands x 3 terciles = 24 cells recorded, each with the fit
    # (m_raw) and the applied (m_norm) modifier.
    assert len(fvr["cells"]) == 24
    for cell in fvr["cells"].values():
        assert cell["m_raw"] > 0
        assert cell["m_norm"] > 0
        assert 0.0 <= cell["phi_cert"] <= 1.0
    # phi_cert sums to 1 over terciles within each (band, sex).
    from collections import defaultdict

    band_sum = defaultdict(float)
    for name, cell in fvr["cells"].items():
        band = name.split(".", 1)[1]  # {band}|{sex}
        band_sum[band] += cell["phi_cert"]
    for band, total in band_sum.items():
        assert abs(total - 1.0) <= 1e-9, band


def test_modifier_normalization_constraint_by_hand(art):
    """sum_t m_norm * phi_cert = 1 per (band, sex), independently."""
    from collections import defaultdict

    fvr = art["per_seed"][0]["fm_modifier_fit_vs_raw"]
    acc = defaultdict(float)
    for name, cell in fvr["cells"].items():
        band = name.split(".", 1)[1]
        acc[band] += cell["m_norm"] * cell["phi_cert"]
    for band, total in acc.items():
        # bands with certified mass integrate to 1; empty bands to 0.
        assert abs(total - 1.0) <= 1e-9 or total == 0.0, band


# --------------------------------------------------------------------------
# c1 -> c2 progression (the run's headline)
# --------------------------------------------------------------------------
def test_c1_to_c2_progression(art):
    p = art["c1_to_c2_progression"]
    assert p["c1_n_seeds_pass"] == 1
    assert p["c2_n_seeds_pass"] == PINNED_N_SEEDS_PASS
    assert p["c1_gate_pass"] is False
    # all five candidate-1 misses now pass.
    assert p["n_c1_misses_resolved"] == PINNED_C1_MISSES_RESOLVED
    assert p["n_c1_misses_total"] == PINNED_C1_MISSES_RESOLVED
    for r in p["c1_misses_now"]:
        assert r["now_passes"] is True
        assert r["c2_score"] < r["c1_score"]
    # the only newly-failing cells are the two seed-2 marginal misses.
    nf = {(r["cell"], r["seed"]) for r in p["newly_failing_cells"]}
    assert nf == PINNED_FAILING_CELL_SEEDS


def test_progression_consistent_with_c1_artifact(art, c1):
    p = art["c1_to_c2_progression"]
    assert p["c1_n_seeds_pass"] == c1["verdict"]["n_seeds_pass"]
    c1_by_seed = {s["seed"]: s for s in c1["per_seed"]}
    for seed_s, row in p["per_seed_gated_pass"].items():
        seed = int(seed_s)
        assert row["c1_n_gated_pass"] == c1_by_seed[seed]["n_gated_pass"]


# --------------------------------------------------------------------------
# Locked contract + fresh-run schema (carried invariants)
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


def test_delta_cube_equals_c1_cube_times_modifier(art, c1):
    """The gated first_marriage cube = c1 cube; other cells identical."""
    cube = art["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    c1cube = c1["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    assert cube["cell_index"] == c1cube["cell_index"]
    for ci, cell in enumerate(cube["cell_index"]):
        fam = cell.split(".")[0]
        for k in range(N_DRAWS):
            for si in range(len(GATE_SEEDS)):
                v2 = cube["rates"][k][ci][si]
                v1 = c1cube["rates"][k][ci][si]
                if fam == "first_marriage_by_earnings":
                    continue  # moved (tested elsewhere)
                assert v2 == v1, (cell, k, si)


# --------------------------------------------------------------------------
# Per-family decomposition + precheck
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
    assert dec["assort_mating"]["cell_seed_pass_rate"] == 1.0


def test_precheck_reproduced_exactly(art):
    pc = art["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["reference_moments_max_abs_deviation"] <= EXACT_ATOL
    assert pc["rate_a_max_abs_deviation"] <= EXACT_ATOL
    assert pc["holdout_sha256_all_match"] is True


def test_protocol_estimator_and_pass_rule(art):
    p = art["protocol"]
    assert p["estimator"] == "mean_over_K20_draws"
    assert p["n_draws"] == N_DRAWS
    assert "5200 + k" in p["draw_rng_rule"]
    assert "4 of 5" in p["pass_rule"]
    assert "component_id" in p["split"]


def test_registration_pins_recorded(art):
    pins = art["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2c_hazard.v2"
    assert len(pins["certified_spec_sha256"]) == 64
    assert len(pins["floor_run_sha256"]) == 64
    assert len(pins["c1_artifact_sha256"]) == 64
