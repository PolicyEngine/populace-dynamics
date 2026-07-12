"""Always-runnable pins for the M4 candidate-1 scored run.

Reads only the committed artifact ``runs/gate_m4_hazard_v1.json``, the
frozen floor ``runs/m4_gate_floors_v1.json`` and ``gates.yaml`` -- no PSID
data -- so it runs in CI. It pins the one-shot verdict, recomputes the
gate arithmetic from the committed ``[20, 8, 5]`` cube (nothing on
trust), and binds every gated tolerance / anchor sd candidate -> floor ->
locked gate_m4.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_m4_hazard_v1.json"
FLOOR = ROOT / "runs" / "m4_gate_floors_v1.json"
GATES = ROOT / "gates.yaml"

REGISTRATION_POINTER = "4950931158"
FLOW_K, STOCK_K, ROUNDING = 3, 4, 3
MARGIN_K = 3
T_MAX = math.log(1.5)

INTERNAL_CELLS = [
    "incidence.40-49|female",
    "incidence.50-59|female",
    "incidence.50-59|male",
    "prevalence.50-59|female",
    "prevalence.50-59|male",
    "recovery.50-59|female",
    "recovery.60-66|female",
    "recovery.60-66|male",
]
ANCHOR_CELLS = [
    "conversion_exit.retirement_dominant|female",
    "conversion_exit.retirement_dominant|male",
    "prevalence_ageshape.comonotone|female",
    "prevalence_ageshape.comonotone|male",
]

# The published one-shot verdict (publishes regardless; this is the record).
PINNED_GATE_PASS = True
PINNED_N_SEEDS_PASS = 5
PINNED_SEED_PASS = {"0": True, "1": True, "2": True, "3": True, "4": True}
PINNED_INTERNAL_PASS = {s: 8 for s in PINNED_SEED_PASS}
PINNED_ANCHOR_PASS = {s: 4 for s in PINNED_SEED_PASS}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def floor() -> dict:
    return json.loads(FLOOR.read_text())


@pytest.fixture(scope="module")
def gate_m4() -> dict:
    return yaml.safe_load(GATES.read_text())["gates"]["gate_m4"]


def _cell_k(cell: str) -> int:
    return STOCK_K if cell.split(".", 1)[0] == "prevalence" else FLOW_K


# --------------------------------------------------------------------------
# Identity + registration
# --------------------------------------------------------------------------
def test_identity(artifact: dict) -> None:
    assert artifact["schema_version"] == "gate_m4_hazard.v1"
    assert artifact["run"] == "gate_m4_hazard_v1"
    assert artifact["gate"] == "gate_m4"
    assert artifact["candidate"] == "candidate 1"
    assert artifact["registration_pointer"] == REGISTRATION_POINTER


def test_registration_pointer_embedded(artifact: dict) -> None:
    assert REGISTRATION_POINTER in artifact["spec_registration"]
    assert REGISTRATION_POINTER in artifact["one_shot"]
    assert "no holdout tuning" in artifact["one_shot"]


def test_cell_orders_match_floor_partition(
    artifact: dict, floor: dict
) -> None:
    part = floor["gate_partition"]
    assert artifact["internal_cell_order"] == sorted(
        part["internal_gate_eligible"]
    )
    assert artifact["anchor_cell_order"] == sorted(
        part["anchor_gate_eligible"]
    )
    assert artifact["internal_cell_order"] == INTERNAL_CELLS
    assert artifact["anchor_cell_order"] == ANCHOR_CELLS


# --------------------------------------------------------------------------
# The pinned one-shot verdict
# --------------------------------------------------------------------------
def test_verdict_pins(artifact: dict) -> None:
    v = artifact["verdict"]
    assert v["gate_m4_pass"] is PINNED_GATE_PASS
    assert v["n_seeds_pass"] == PINNED_N_SEEDS_PASS
    assert v["seed_pass"] == PINNED_SEED_PASS
    assert v["n_gated_cells"] == 12
    assert v["n_gated_internal"] == 8
    assert v["n_gated_anchor"] == 4
    assert v["all_failing_gated_cells"] == []


def test_internal_anchor_decomposition_pins(artifact: dict) -> None:
    dec = artifact["verdict"]["internal_vs_anchor_decomposition"]
    assert dec["internal"]["per_seed_n_pass"] == PINNED_INTERNAL_PASS
    assert dec["anchor"]["per_seed_n_pass"] == PINNED_ANCHOR_PASS
    assert dec["internal"]["all_failing_internal_cells"] == []
    assert dec["anchor"]["all_failing_anchor_cells"] == []


def test_gate_conjunction_recomputes(artifact: dict) -> None:
    """The >=4-of-5 rule recomputes from the per-seed booleans."""
    per_seed = {s["seed"]: s for s in artifact["per_seed"]}
    n_pass = 0
    for seed in range(5):
        s = per_seed[seed]
        internal_ok = all(c["pass"] for c in s["internal_cells"].values())
        anchor_ok = all(c["pass"] for c in s["anchor_cells"].values())
        seed_pass = internal_ok and anchor_ok
        assert seed_pass == s["seed_pass"]
        n_pass += int(seed_pass)
    assert n_pass == artifact["verdict"]["n_seeds_pass"]
    assert (n_pass >= 4) == artifact["verdict"]["gate_m4_pass"]


# --------------------------------------------------------------------------
# The [20, 8, 5] cube recomputes rbar / score / pass -- nothing on trust
# --------------------------------------------------------------------------
def test_cube_shape(artifact: dict) -> None:
    cube_block = artifact["per_draw_per_cell_rates"]
    assert cube_block["shape"] == [20, 8, 5]
    cube = cube_block["cube"]
    assert len(cube) == 20
    assert all(len(plane) == 8 for plane in cube)
    assert all(len(row) == 5 for plane in cube for row in plane)
    assert cube_block["axis_order"]["cells"] == INTERNAL_CELLS
    assert cube_block["axis_order"]["seeds"] == [0, 1, 2, 3, 4]
    assert cube_block["axis_order"]["draws"] == [5200 + k for k in range(20)]


def test_rbar_and_score_recompute_from_cube(artifact: dict) -> None:
    cube = artifact["per_draw_per_cell_rates"]["cube"]
    per_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for ci, cell in enumerate(INTERNAL_CELLS):
        for si, seed in enumerate(range(5)):
            draws = [cube[k][ci][si] for k in range(20)]
            rec = per_seed[seed]["internal_cells"][cell]
            # cube plane == the per-seed per_draw_rate
            assert draws == pytest.approx(rec["per_draw_rate"], abs=1e-12)
            rbar = sum(draws) / len(draws)
            assert rbar == pytest.approx(rec["rbar"], abs=1e-9)
            rate_a = rec["rate_a"]
            score = abs(math.log(rbar / rate_a)) if rbar > 0 else math.inf
            assert score == pytest.approx(rec["score"], abs=1e-9)
            assert rec["pass"] == (score <= rec["tolerance"])


def test_no_undefined_draws(artifact: dict) -> None:
    for s in artifact["per_seed"]:
        assert s["undefined_internal_draws"] == []
        assert s["undefined_conversion_anchor_draws"] == []


# --------------------------------------------------------------------------
# Tolerance binding: candidate == floor-derived == locked gate_m4
# --------------------------------------------------------------------------
def _locked_tolerances(gate_m4: dict) -> dict[str, float]:
    views = gate_m4["thresholds"]["internal_surface"]["views"]
    tol: dict[str, float] = {}
    tol.update(views["internal_flow_hazards"]["tolerances"])
    tol.update(views["internal_prevalence_stock"]["tolerances"])
    return tol


def test_internal_tolerances_bind_floor_and_gates(
    artifact: dict, floor: dict, gate_m4: dict
) -> None:
    noise = floor["internal_noise_floor"]["noise_floor_seeds_0_99"]
    locked = _locked_tolerances(gate_m4)
    per_seed0 = artifact["per_seed"][0]["internal_cells"]
    for cell in INTERNAL_CELLS:
        k = _cell_k(cell)
        derived = min(
            round(noise[cell]["mean"] + k * noise[cell]["sd"], ROUNDING),
            T_MAX,
        )
        # candidate tolerance == floor-derived == locked gate value
        assert per_seed0[cell]["tolerance"] == pytest.approx(
            derived, abs=1e-12
        )
        assert derived == pytest.approx(locked[cell], abs=1e-12)


def test_tolerance_is_consistent_across_seeds(artifact: dict) -> None:
    ref = artifact["per_seed"][0]["internal_cells"]
    for s in artifact["per_seed"]:
        for cell in INTERNAL_CELLS:
            assert (
                s["internal_cells"][cell]["tolerance"]
                == ref[cell]["tolerance"]
            )
            assert s["internal_cells"][cell]["k"] == _cell_k(cell)


# --------------------------------------------------------------------------
# Anchor binding: margin >= 3 sigma, sd == floor == locked gate_m4
# --------------------------------------------------------------------------
def _locked_anchor_sd(gate_m4: dict) -> dict[str, float]:
    cells = gate_m4["thresholds"]["anchor_surface"]["cells"]
    return {c: cells[c]["real_half_split_sd"] for c in cells}


def test_anchor_sd_binds_floor_and_gates(
    artifact: dict, floor: dict, gate_m4: dict
) -> None:
    locked = _locked_anchor_sd(gate_m4)
    ac = floor["anchor_checks"]
    for cell in ANCHOR_CELLS:
        hsf = ac[cell]["half_split_floor"]
        floor_sd = (hsf["min_gap"] if "min_gap" in hsf else hsf["share"])["sd"]
        for s in artifact["per_seed"]:
            rec = s["anchor_cells"][cell]
            assert rec["real_half_split_sd"] == pytest.approx(
                floor_sd, abs=1e-15
            )
            assert rec["real_half_split_sd"] == pytest.approx(
                locked[cell], abs=1e-15
            )
            assert rec["threshold"] == pytest.approx(
                MARGIN_K * floor_sd, abs=1e-15
            )


def test_anchor_margin_pass_recomputes(artifact: dict) -> None:
    for s in artifact["per_seed"]:
        for cell in ANCHOR_CELLS:
            rec = s["anchor_cells"][cell]
            sigma = rec["invariant_value"] / rec["real_half_split_sd"]
            assert sigma == pytest.approx(rec["margin_sigma_units"], abs=1e-9)
            assert rec["pass"] == (sigma >= MARGIN_K)
            # published run passes every anchor on every seed
            assert rec["pass"] is True


# --------------------------------------------------------------------------
# Floor provenance
# --------------------------------------------------------------------------
def test_floor_run_sha256_pinned(artifact: dict) -> None:
    committed = hashlib.sha256(FLOOR.read_bytes()).hexdigest()
    assert artifact["data"]["floor_run_sha256"] == committed
    assert artifact["revision_pins"]["floor_run_sha256"] == committed


def test_panel_scale(artifact: dict) -> None:
    d = artifact["data"]
    assert d["n_persons"] == 41345
    assert d["n_person_years"] == 284932
