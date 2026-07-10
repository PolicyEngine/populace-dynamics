"""Tests for the gate-2b candidate-1 one-shot scored run.

Candidate 1 is the FIRST pre-registered gate-2b candidate (issue #42 comment
4938726107): a six-component structural household-composition generator
composed from the certified tranche-2a marital core. The one-shot outcome
(published REGARDLESS of verdict) is pinned below from the committed artifact
``runs/gate2b_hazard_v1.json``: **FAIL 0/5** -- the forecast's failure-likely
class, whose per-family decomposition is the primary value (what candidate 2
targets).

This module is always runnable: it inspects the committed artifact, binds the
stored tolerances to the ratified floor and the locked gates.yaml block, proves
the [20, 46, 5] per-draw cube reproduces every score, and unit-tests the pure
components 5-6 composition and the parental-exit spline basis. The
PSID-dependent reproduction pin lives in
``tests/test_gate2b_candidate1_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from populace_dynamics.models import household_composition_sim as hcs

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4938726107"


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
# Presence, registration, one-shot verdict pin
# --------------------------------------------------------------------------
def test_artifact_present_and_identity():
    a = _artifact()
    assert a["schema_version"] == "gate2b_hazard.v1"
    assert a["run"] == "gate2b_hazard_v1"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 1"
    assert a["registration_pointer"] == REGISTRATION_POINTER


def test_one_shot_verdict_pinned_fail_0_of_5():
    """The committed one-shot outcome: gate FAIL, 0 of 5 seeds pass."""
    v = _artifact()["verdict"]
    assert v["gate_2b_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert v["n_gate_seeds"] == 5
    assert v["n_gated_cells"] == N_GATED
    assert all(p is False for p in v["seed_pass"].values())


def test_forecast_recorded_and_not_graded_here():
    f = _artifact()["pre_registered_forecast"]
    assert f["p_gate_pass_4_of_5"] == "0.15-0.30"
    assert "grading_note" in f
    assert "orchestrator" in f["grading_note"]


def test_spec_resolution_notes_present():
    notes = _artifact()["spec_resolution_notes"]
    for key in (
        "rng_two_stream",
        "observed_initial_states_are_the_holdout_persons_own",
        "coresident_children_from_certified_kernel",
        "household_size_composition",
        "coresident_grandchild_composed_only",
    ):
        assert key in notes and notes[key]


# --------------------------------------------------------------------------
# Precheck: the committed artifact records a bit-exact floor reproduction
# --------------------------------------------------------------------------
def test_precheck_reproduced_exactly():
    p = _artifact()["precheck"]
    assert p["all_reproduced_exactly"] is True
    assert p["reference_moments_max_abs_deviation"] == 0.0
    assert p["rate_a_max_abs_deviation"] == 0.0
    assert p["holdout_sha256_all_match"] is True


# --------------------------------------------------------------------------
# Cell set: 46 gated == floor gate_partition; 47 report-only
# --------------------------------------------------------------------------
def test_gated_cells_match_floor_gate_partition():
    a = _artifact()
    floor = _floor()
    gated = set(floor["gate_partition"]["gate_eligible"])
    assert len(gated) == N_GATED
    assert set(_gate2b_tolerances()) == gated
    for s in a["per_seed"]:
        assert set(s["gated_cells"]) == gated


def test_report_only_cells_match_gates_yaml_and_never_gate():
    a = _artifact()
    report_only = set(_gate2b_thresholds()["report_only"])
    assert len(report_only) == 47
    for s in a["per_seed"]:
        assert set(s["report_only_cells"]) == report_only
        for rec in s["report_only_cells"].values():
            assert rec["gated"] is False


# --------------------------------------------------------------------------
# Fresh-run artifact-schema conformance ([20, 46, 5] + undefined + dispersion)
# --------------------------------------------------------------------------
def test_per_draw_per_cell_rates_shape_and_index():
    pc = _artifact()["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    assert pc["shape"] == [N_DRAWS, N_GATED, len(GATE_SEEDS)]
    assert pc["cell_index"] == sorted(_gate2b_tolerances())
    assert len(pc["cell_index"]) == N_GATED
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


def test_undefined_draw_rule_not_triggered_and_run_valid():
    u = _artifact()["fresh_run_artifact_schema"]["undefined_draw_rule"]
    assert u["required"] is True
    assert u["pre_specified"] is True
    assert u["run_invalidated"] is False
    assert u["n_undefined_gated_draws"] == 0
    for s in _artifact()["per_seed"]:
        assert s["undefined_gated_draws"] == []
        for rec in s["gated_cells"].values():
            assert rec["n_draws_defined"] == N_DRAWS


def test_per_draw_dispersion_disclosure_report_only():
    a = _artifact()
    d = a["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    assert d["gated"] is False
    assert d["report_only"] is True
    assert sorted(d["cells"]) == sorted(_gate2b_tolerances())
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    cell = "hh_size.5+"
    for seed in GATE_SEEDS:
        rates = by_seed[seed]["gated_cells"][cell]["per_draw_rate"]
        expected = float(np.std(rates, ddof=1))
        stored = d["cells"][cell]["per_seed"][str(seed)]["per_draw_rate_sd"]
        assert stored == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# Seed conjunction + verdict recompute from the cells
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


def test_all_failing_gated_cells_are_real_failures():
    a = _artifact()
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for f in a["verdict"]["all_failing_gated_cells"]:
        rec = by_seed[f["seed"]]["gated_cells"][f["cell"]]
        assert rec["pass"] is False
        assert f["score"] == rec["score"]


# --------------------------------------------------------------------------
# Tolerance binding: locked gates.yaml == round(floor mean + 4*sd, 3) == stored
# --------------------------------------------------------------------------
def test_stored_tolerances_match_locked_gates_yaml():
    tol = _gate2b_tolerances()
    for s in _artifact()["per_seed"]:
        for cell, rec in s["gated_cells"].items():
            assert rec["tolerance"] == pytest.approx(tol[cell], abs=1e-12)


def test_tolerances_bind_to_ratified_floor():
    """Each locked tolerance is round(floor mean + 4*sd, 3) -- bound, and a
    decisive floor perturbation would move it (not a coincidence)."""
    tol = _gate2b_tolerances()
    floor = _floor()["noise_floor_seeds_0_99"]
    for cell, t in tol.items():
        mean, sd = floor[cell]["mean"], floor[cell]["sd"]
        assert t == round(mean + 4 * sd, 3), cell
        assert round(mean + 0.05 + 4 * sd, 3) != t, cell


def test_per_family_decomposition_covers_all_gated_cells():
    a = _artifact()
    decomp = a["per_family_decomposition"]
    covered: set[str] = set()
    for fam in decomp.values():
        covered.update(fam["cells"])
    assert covered == set(_gate2b_tolerances())
    # Every family carries a stated mechanism.
    for fam in decomp.values():
        assert fam["mechanism"]


# --------------------------------------------------------------------------
# Derivations: components 5-6 composition (pure) + the exit spline basis
# --------------------------------------------------------------------------
def test_compose_states_household_size_and_flags():
    spouse = np.array([True, False, False, True])
    parent = np.array([False, True, False, False])
    multigen = np.array([True, True, False, True])
    child = np.array([2, 0, 0, 1])
    cc, gc, hh = hcs.compose_states(spouse, parent, multigen, child, 2)
    # coresident_child = any child.
    assert cc.tolist() == [True, False, False, True]
    # hh_size = 1 + spouse + children + (2 if parent).
    assert hh.tolist() == [1 + 1 + 2, 1 + 0 + 2, 1, 1 + 1 + 1]
    # grandchild = multigen AND child AND NOT parent.
    assert gc.tolist() == [True, False, False, True]


def test_compose_states_grandchild_requires_all_three_conditions():
    # multigen but no child -> no grandchild; parent present -> no grandchild.
    cc, gc, hh = hcs.compose_states(
        np.array([False, False]),
        np.array([False, True]),  # second has a coresident parent
        np.array([True, True]),
        np.array([0, 3]),
        2,
    )
    assert gc.tolist() == [False, False]


def test_parent_count_zero_when_no_coresident_parent():
    _, _, hh = hcs.compose_states(
        np.array([False]),
        np.array([False]),
        np.array([False]),
        np.array([0]),
        2,
    )
    assert hh.tolist() == [1]


def test_restricted_cubic_basis_shape_and_boundary_linearity():
    knots = hcs.PARENTAL_EXIT_KNOTS
    x = np.array([16.0, 20.0, 30.0, 60.0, 80.0])
    b = hcs.restricted_cubic_basis(x, knots)
    # K knots -> K-1 columns; column 0 is the linear term.
    assert b.shape == (5, len(knots) - 1)
    assert np.allclose(b[:, 0], x)
    # Beyond the last knot the nonlinear terms are linear in x, so second
    # differences of each nonlinear column vanish on an equally spaced grid.
    xr = np.array([50.0, 60.0, 70.0, 80.0])
    br = hcs.restricted_cubic_basis(xr, knots)
    for col in range(1, br.shape[1]):
        second_diff = np.diff(br[:, col], n=2)
        assert np.allclose(second_diff, 0.0, atol=1e-9)


def test_model_records_certified_registry_spec():
    a = _artifact()
    assert a["model"]["family_transitions_spec"] == "candidate16_registry_v1"
    # The per-seed component meta pins the same certified spec sha.
    sha = a["model"]["family_transitions_spec_sha256"]
    for s in a["per_seed"]:
        assert s["component_meta"]["family_transitions_spec_sha256"] == sha
