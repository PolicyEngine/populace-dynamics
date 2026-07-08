"""Tests for the gate-2 candidate-1 (run 1) pre-registered run.

Candidate 1 is the first pre-registered gate-2 candidate: a five-component
family-transition simulator (logistic first-marriage hazard; empirical
divorce / remarriage / fertility tables; mortality-composed widowhood)
scored under the LOCKED gate-2 protocol (gates.yaml ``gate_2``, ratified
PR #79 + flip #81). Frozen spec: issue #42 comment 4910914098.

Two tiers, mirroring the gate-1 candidate tests:

* the always-runnable consistency suite (touches only the committed
  artifact and ``gates.yaml``): schema and spec URL, the bit-exact
  reproduction precheck attestation, every stored gated-cell pass
  recomputes from its own score against its stored (locked) tolerance,
  the stored tolerances equal the locked gates.yaml, each seed's pass
  recomputes from the 46 gated cells, the verdict recomputes from the
  seed conjunction, the 16 report-only cells are present but never gate,
  and the forecast / registration / revision pins are carried;
* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  history files are absent) reruns seed 0 end-to-end through the runner
  and pins the committed seed-0 block to float precision -- the
  simulation is deterministic given the registered RNG rule
  ``numpy.random.default_rng(4200 + seed)``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4910914098"
)
GATE_SEEDS = [0, 1, 2, 3, 4]
N_GATED = 46
N_REPORT_ONLY = 16


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate2_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_2"]["thresholds"]


def _gate2_tolerances() -> dict[str, float]:
    tol: dict[str, float] = {}
    for view in _gate2_thresholds()["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate1 as runner

    return runner


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and exposes the frozen dials."""
    runner = _import_runner()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.SIM_SEED_BASE == 4200
    assert runner.SPLINE_KNOTS == (20.0, 25.0, 30.0, 40.0)
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v1"
    assert "4910914098" in runner.SPEC_REGISTRATION


def test_ncs_basis_is_natural_cubic_spline():
    """3 columns for 4 knots; the nonlinear terms are linear past the ends."""
    import numpy as np

    runner = _import_runner()
    knots = (20.0, 25.0, 30.0, 40.0)
    x = np.array([15.0, 20.0, 30.0, 45.0, 60.0])
    basis = runner.ncs_basis(x, knots)
    assert basis.shape == (5, 3)
    # Column 0 is the linear term.
    assert basis[:, 0] == pytest.approx(x)
    # The nonlinear terms are exactly 0 at/below the first knot (natural
    # spline is linear there, absorbed by the linear column).
    assert basis[1, 1:] == pytest.approx([0.0, 0.0])
    assert basis[0, 1:] == pytest.approx([0.0, 0.0])
    # Beyond the last knot the second difference is constant (linearity):
    hi = runner.ncs_basis(np.array([45.0, 55.0, 65.0]), knots)[:, 1]
    assert (hi[2] - hi[1]) == pytest.approx(hi[1] - hi[0], rel=1e-9)


# --------------------------------------------------------------------------
# Artifact presence, spec, lock (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v1"
    assert a["run"] == "gate2_hazard_v1"
    assert a["gate"] == "gate_2"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert "4910914098" in a["spec_registration"]
    assert a["forecast_pointer"]["registration"] == SPEC_URL


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.60-0.70"
    assert fc["registration"] == SPEC_URL
    assert "stock" in fc["modal_failure"].lower()
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
    }


def test_precheck_reproduced_exactly():
    """The scoring path reproduced the committed floor bit-for-bit."""
    pc = _artifact()["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["reference_moments_exact"] is True
    assert pc["rate_a_exact"] is True
    assert pc["holdout_sha256_all_match"] is True
    assert pc["reference_moments_max_abs_deviation"] == 0.0
    assert pc["rate_a_max_abs_deviation"] == 0.0
    # Every gate seed's holdout sha256 matched the committed floor.
    for row in pc["per_seed"]:
        assert row["holdout_sha256_match"] is True


# --------------------------------------------------------------------------
# Tolerances match the locked gates.yaml (always runnable)
# --------------------------------------------------------------------------
def test_stored_tolerances_match_locked_gates_yaml():
    a = _artifact()
    locked = _gate2_tolerances()
    assert len(locked) == N_GATED
    # The gated cell set matches the floor's committed gate_partition.
    floor = json.loads(FLOOR.read_text())
    assert set(locked) == set(floor["gate_partition"]["gate_eligible"])
    for seed in a["per_seed"]:
        stored = {k: v["tolerance"] for k, v in seed["gated_cells"].items()}
        assert set(stored) == set(locked)
        for cell, value in stored.items():
            assert value == pytest.approx(locked[cell], abs=0)


def test_report_only_cells_match_gates_yaml_and_never_gate():
    a = _artifact()
    locked_report = set(_gate2_thresholds()["report_only"])
    locked_tol = set(_gate2_tolerances())
    assert len(locked_report) == N_REPORT_ONLY
    assert locked_report.isdisjoint(locked_tol)
    for seed in a["per_seed"]:
        report = seed["report_only_cells"]
        assert set(report) == locked_report
        for cell, rec in report.items():
            assert rec["gated"] is False
            # Report-only cells carry a score but no gating tolerance/pass.
            assert "tolerance" not in rec
            assert "pass" not in rec
            assert cell not in seed["gated_cells"]


# --------------------------------------------------------------------------
# Every stored pass recomputes from its score + tolerance (always runnable)
# --------------------------------------------------------------------------
def test_every_gated_pass_recomputes_from_score():
    a = _artifact()
    for seed in a["per_seed"]:
        n_pass = 0
        for cell, rec in seed["gated_cells"].items():
            recomputed = rec["score"] <= rec["tolerance"]
            assert recomputed == rec["pass"], (
                f"seed {seed['seed']} {cell}: stored={rec['pass']} "
                f"recomputed={recomputed}"
            )
            # The score is the absolute log rate ratio (or non-finite when a
            # candidate rate is zero).
            if rec["r_candidate"] > 0 and rec["rate_a"] > 0:
                expected = abs(math.log(rec["r_candidate"] / rec["rate_a"]))
                assert rec["score"] == pytest.approx(expected, abs=1e-12)
            n_pass += rec["pass"]
        assert seed["n_gated"] == N_GATED
        assert seed["n_gated_pass"] == n_pass
        assert seed["n_gated_fail"] == N_GATED - n_pass


def test_seed_pass_recomputes_from_all_gated_cells():
    a = _artifact()
    for seed in a["per_seed"]:
        all_pass = all(rec["pass"] for rec in seed["gated_cells"].values())
        assert seed["seed_pass"] == all_pass
        assert seed["seed_pass"] == (seed["n_gated_pass"] == N_GATED)


def test_verdict_recomputes_from_seed_conjunction():
    a = _artifact()
    v = a["verdict"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    n_pass = 0
    for row in a["seed_conjunction"]:
        s = by_seed[row["seed"]]
        assert row["seed_pass"] == s["seed_pass"]
        assert row["n_gated_pass"] == s["n_gated_pass"]
        assert row["n_gated_fail"] == s["n_gated_fail"]
        n_pass += row["seed_pass"]
    assert v["n_seeds_pass"] == n_pass
    assert v["n_gate_seeds"] == len(GATE_SEEDS)
    assert v["gate_2_pass"] == (n_pass >= 4)
    assert v["seed_pass"] == {
        str(s): by_seed[s]["seed_pass"] for s in GATE_SEEDS
    }


def test_verdict_per_block_counts_consistent():
    """Every per-family per-seed pass count recomputes from the gated cells."""
    a = _artifact()
    v = a["verdict"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    all_block_cells: set[str] = set()
    for _fam, blk in v["per_block"].items():
        all_block_cells |= set(blk["cells"])
        for seed_key, d in blk["per_seed_pass"].items():
            seed = by_seed[int(seed_key)]
            recomputed = sum(
                seed["gated_cells"][c]["pass"] for c in blk["cells"]
            )
            assert d["n_pass"] == recomputed
            assert d["n_cells"] == len(blk["cells"])
    # Every gated cell belongs to exactly one family block.
    assert all_block_cells == set(_gate2_tolerances())


def test_all_failing_gated_cells_are_real_failures():
    a = _artifact()
    v = a["verdict"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    listed = {(f["cell"], f["seed"]) for f in v["all_failing_gated_cells"]}
    actual = {
        (cell, s["seed"])
        for s in a["per_seed"]
        for cell, rec in s["gated_cells"].items()
        if not rec["pass"]
    }
    assert listed == actual
    for f in v["all_failing_gated_cells"]:
        rec = by_seed[f["seed"]]["gated_cells"][f["cell"]]
        assert f["score"] == pytest.approx(rec["score"], abs=0)
        assert f["tolerance"] == pytest.approx(rec["tolerance"], abs=0)
        assert rec["score"] > rec["tolerance"]


def test_modal_failure_block_recomputes():
    a = _artifact()
    v = a["verdict"]
    modal = a["modal_failure_materialized"]
    fails = {f["cell"]: [] for f in v["all_failing_gated_cells"]}
    for f in v["all_failing_gated_cells"]:
        fails[f["cell"]].append(f["seed"])
    stock_failed = any(c.startswith("share_") for c in fails)
    c1980s_failed = "ever_married_by_40.c1980s" in fails
    ad_failed = "remarriage.after_divorce" in fails
    assert modal["stock_occupancy_failed"] == stock_failed
    assert modal["c1980s_failed"] == c1980s_failed
    assert modal["remarriage_after_divorce_failed"] == ad_failed
    assert modal["any_materialized"] == (
        stock_failed or c1980s_failed or ad_failed
    )


def test_revision_pins_record_sklearn_version():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v1"
    assert pins["sklearn_version"].startswith("1.9")
    assert "numpy_version" in pins
    assert "pandas_version" in pins
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


# --------------------------------------------------------------------------
# Live seed-0 reproduction (needs the staged PSID history files)
# --------------------------------------------------------------------------
@needs_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 end-to-end and match the committed seed-0 block.

    The registered RNG rule (default_rng(4200 + seed)) plus the fixed
    person ordering makes the simulation deterministic, so every stored
    candidate rate, score, and pass must reproduce to float precision.
    """
    runner = _import_runner()
    from populace_dynamics.data import marriage

    thresholds = runner.load_gate2_thresholds()
    tol = runner.gated_tolerances(thresholds)
    report_only = list(thresholds["report_only"])
    floor = json.loads(FLOOR.read_text())

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, fert, _ = runner.g2f.load_panels()
    order_map = runner._order_map(mh)

    result = runner.score_seed(
        0,
        panel,
        fert,
        demo,
        dr,
        mh,
        bh,
        order_map,
        floor,
        tol,
        report_only,
        False,
    )
    committed = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    assert result["n_gated_pass"] == committed["n_gated_pass"]
    assert result["seed_pass"] == committed["seed_pass"]
    for cell, rec in committed["gated_cells"].items():
        got = result["gated_cells"][cell]
        assert got["r_candidate"] == pytest.approx(
            rec["r_candidate"], abs=1e-12
        ), cell
        assert got["rate_a"] == pytest.approx(rec["rate_a"], abs=0), cell
        assert got["score"] == pytest.approx(rec["score"], abs=1e-12), cell
        assert got["pass"] == rec["pass"], cell
    for cell, rec in committed["report_only_cells"].items():
        got = result["report_only_cells"][cell]
        assert got["r_candidate"] == pytest.approx(
            rec["r_candidate"], abs=1e-12
        ), cell
        assert got["score"] == pytest.approx(rec["score"], abs=1e-12), cell
    meta = result["component_meta"]
    cmeta = committed["component_meta"]
    assert (
        meta["first_marriage_converged"] == cmeta["first_marriage_converged"]
    )
    assert meta["first_marriage_lbfgs_n_iter"] == (
        cmeta["first_marriage_lbfgs_n_iter"]
    )
