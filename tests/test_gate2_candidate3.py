"""Tests for the gate-2 candidate-3 (run 1) pre-registered run.

Candidate 3 is the third pre-registered gate-2 candidate: candidate 2's
five-component family-transition simulator with EXACTLY two named deltas --
(1) the spouse-death mortality table's smoothing law becomes exposure-
weighted shrinkage toward the pooled band x sex rate
(``cell_rate = (Wd + K * pooled_rate(band, sex)) / (We + K)``, K = 500
weighted person-years a-priori), replacing candidate 2's add-one-at-mean-
weight; and (2) the first-marriage spline gains a knot at 22, becoming
20/22/25/30/40. Everything else is byte-identical to candidate 2 -- the
simulation is candidate 2's ``simulate_holdout`` reused unchanged. Frozen
spec: issue #42 comment 4911357564 (candidate 2's spec, comment 4911167286,
with the two deltas).

Three tiers:

* the always-runnable consistency suite (touches only the committed
  artifacts and ``gates.yaml``): schema and spec URLs, the two recorded
  deltas, the bit-exact reproduction precheck attestation, every stored
  gated-cell pass recomputes from its score against its stored (locked)
  tolerance, the stored tolerances equal the locked gates.yaml, each seed's
  pass recomputes from the 46 gated cells, the verdict recomputes from the
  seed conjunction, the report-only cells never gate, the registered modal
  (``share_widowed.75+|female``) and the candidate-2 killer movement
  recompute, and the forecast / registration / revision pins are carried;
* structural delta checks that need no data: delta 2 records the shrinkage
  smoothing law and its K, delta 1 adds exactly one spline basis column to
  candidate 2's first-marriage design (interactions unchanged), the
  simulation is candidate 2's object (reused unchanged), and the fertility
  component is byte-identical to candidate 1 across every seed (RNG-isolated
  from the marital process, so the two deltas cannot perturb it -- a direct
  check of the "everything else byte-identical" claim);
* :func:`test_seed0_reproduces_committed_artifact` and the live delta-2 /
  fertility checks (skipped when the PSID history files are absent) rerun
  seed 0 end-to-end through the candidate-3 runner and pin the committed
  seed-0 block to float precision, confirm the shrinkage target is
  candidate 1's pooled table, and confirm fertility reproduces candidate 1
  bit-for-bit.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_hazard_v3.json"
ARTIFACT_C1 = ROOT / "runs" / "gate2_hazard_v1.json"
ARTIFACT_C2 = ROOT / "runs" / "gate2_hazard_v2.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911357564"
)
SPEC_URL_C2 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911167286"
)
SPEC_URL_C1 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4910914098"
)
GATE_SEEDS = [0, 1, 2, 3, 4]
N_GATED = 46
N_REPORT_ONLY = 16

#: The registered modal failure (comment 4911357564).
MODAL_CELL = "share_widowed.75+|female"
#: The candidate-2 killers the delta-2 smoothing fix un-explodes (each
#: failed all five seeds under candidate 2's add-one convention).
C2_CASCADE_CELLS = (
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
    "share_divorced.45-54|female",
    "share_divorced.55-64|female",
    "widowhood.45+|male",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c1() -> dict:
    return json.loads(ARTIFACT_C1.read_text())


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
    import run_gate2_candidate3 as runner

    return runner


def _import_c1():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate1 as c1

    return c1


def _import_c2():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate2 as c2

    return c2


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses candidate 1's dials.

    The frozen dials are IMPORTED from candidate 1, so they are identical by
    construction (any drift would be a divergence from the frozen spec). The
    two delta constants -- the knot-at-22 spline and the K = 500 shrinkage
    prior -- are the only estimation dials that move.
    """
    runner = _import_runner()
    c1 = _import_c1()
    c2 = _import_c2()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.GATE_SEEDS is c1.GATE_SEEDS
    assert runner.SIM_SEED_BASE == 4200
    assert runner.SIM_SEED_BASE == c1.SIM_SEED_BASE
    # DELTA 1: the spline gains a knot at 22 (candidate 2 used 20/25/30/40).
    assert runner.SPLINE_KNOTS_C3 == (20.0, 22.0, 25.0, 30.0, 40.0)
    assert runner.SPLINE_KNOTS_C2 == (20.0, 25.0, 30.0, 40.0)
    assert runner.SPLINE_KNOTS_C2 == c2.SPLINE_KNOTS
    added = set(runner.SPLINE_KNOTS_C3) - set(runner.SPLINE_KNOTS_C2)
    assert added == {22.0}
    # DELTA 2: the shrinkage prior strength, a-priori 500 weighted PY.
    assert runner.K_MORT_PRIOR == 500.0
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v3"
    assert "4911357564" in runner.SPEC_REGISTRATION
    assert "4911167286" in runner.CANDIDATE2_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert len(runner.DELTAS_VS_CANDIDATE2) == 2


def test_simulation_is_candidate2_reused_unchanged():
    """No delta touches the simulation: it is candidate 2's exact object.

    Delta 2 preserves candidate 2's ``"period|band|sex"`` mortality-table
    structure (only the per-cell smoothing changes), and delta 1 changes only
    the fitted first-marriage model behind the unchanged ``predict``
    interface -- so the annual simulation loop, its RNG rule, and its lookups
    are candidate 2's, reused by identity (the strongest possible
    "byte-identical simulation" guarantee).
    """
    runner = _import_runner()
    c2 = _import_c2()
    assert runner.simulate_holdout is c2.simulate_holdout
    assert runner._build_sim_lookups is c2._build_sim_lookups
    assert runner._widow_probs is c2._widow_probs
    assert runner._period_index is c2._period_index
    # The first-marriage design class is candidate 2's (delta 1 changes the
    # knots passed to it, not its interaction structure).
    assert runner.FirstMarriageModelC3 is c2.FirstMarriageModelC2


# --------------------------------------------------------------------------
# Delta 1: first-marriage design gains exactly one spline column (knot 22)
# --------------------------------------------------------------------------
def test_delta1_first_marriage_adds_knot_at_22():
    """The candidate-3 design = candidate 2's design + one spline column.

    Adding the knot at 22 turns the restricted cubic spline from 3 basis
    columns (4 knots) into 4 (5 knots); this propagates through the spline,
    the age-spline x sex block, and the age-spline x cohort block, but the
    INTERACTION STRUCTURE is unchanged (age-spline x sex + age-spline x
    cohort + sex). The delta is exactly the added knot.
    """
    import numpy as np

    runner = _import_runner()
    c1 = _import_c1()
    c2 = _import_c2()
    knots2 = (20.0, 25.0, 30.0, 40.0)
    knots3 = (20.0, 22.0, 25.0, 30.0, 40.0)
    cohorts = [1940, 1950, 1960]
    common = dict(
        clf=None,
        cohort_levels=cohorts,
        col_mean=np.zeros(1),
        col_sd=np.ones(1),
        n_train_rows=0,
        n_train_events=0,
        n_iter=0,
        converged=False,
    )
    m2 = c2.FirstMarriageModelC2(knots=knots2, **common)
    m3 = runner.FirstMarriageModelC3(knots=knots3, **common)
    age = np.array([18.0, 21.0, 23.0, 27.0, 33.0, 45.0, 55.0])
    is_male = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    decade = np.array([1940, 1950, 1960, 1940, 1950, 1960, 1940])
    d2 = m2._raw_design(age, is_male, decade)
    d3 = m3._raw_design(age, is_male, decade)
    sp2 = c1.ncs_basis(age, knots2)
    sp3 = c1.ncs_basis(age, knots3)
    assert sp2.shape[1] == 3 and sp3.shape[1] == 4
    # c2: [sp(3), sex(1), sp x sex(3), cohort(2), sp x cohort(6)] = 15.
    # c3: [sp(4), sex(1), sp x sex(4), cohort(2), sp x cohort(8)] = 19.
    assert d2.shape[1] == 15
    assert d3.shape[1] == 19
    male = is_male.reshape(-1, 1)
    dmat = np.column_stack(
        [(decade == 1950).astype(float), (decade == 1960).astype(float)]
    )
    # Candidate 3 layout: same blocks, one wider spline.
    assert d3[:, :4] == pytest.approx(sp3)  # spline (4 cols)
    assert d3[:, 4:5] == pytest.approx(male)  # sex
    assert d3[:, 5:9] == pytest.approx(sp3 * male)  # age-spline x sex (4)
    assert d3[:, 9:11] == pytest.approx(dmat)  # cohort dummies (2)
    exp_cohort = np.column_stack([sp3[:, [c]] * dmat for c in range(4)])
    assert d3[:, 11:19] == pytest.approx(exp_cohort)  # age-spline x cohort (8)
    # The sex-interaction block is zero for females, nonzero for males.
    assert d3[0, 5:9] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert np.any(d3[1, 5:9] != 0.0)


# --------------------------------------------------------------------------
# Delta 2: mortality smoothing law is the pooled-rate shrinkage
# --------------------------------------------------------------------------
def test_delta2_mortality_smoothing_recorded():
    """Every seed records the shrinkage smoothing law and its K."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert meta["mortality_stratification"] == "decade_period x band x sex"
        assert (
            meta["mortality_smoothing"]
            == "exposure_weighted_shrinkage_toward_pooled_band_sex_rate"
        )
        assert meta["mortality_prior_strength_K"] == 500.0
        # More than one decade-period is identified on the train complement.
        assert len(meta["mortality_periods"]) >= 2
        assert meta["first_marriage_knots"] == [20.0, 22.0, 25.0, 30.0, 40.0]
        assert "sex" in meta["first_marriage_design"]


# --------------------------------------------------------------------------
# Artifact presence, spec, lock (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v3"
    assert a["run"] == "gate2_hazard_v3"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 3"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_deltas_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert "4911357564" in a["spec_registration"]
    assert a["candidate2_registration"] == SPEC_URL_C2
    assert "4911167286" in a["candidate2_registration"]
    assert a["candidate1_registration"] == SPEC_URL_C1
    # Exactly two named deltas: mortality shrinkage, first-marriage knot.
    deltas = a["deltas_vs_candidate2"]
    assert len(deltas) == 2
    joined = " ".join(deltas).lower()
    assert "shrinkage" in joined
    assert "pooled" in joined
    assert "20/22/25/30/40" in joined
    assert a["forecast_pointer"]["registration"] == SPEC_URL
    assert a["forecast_pointer"]["candidate2_registration"] == SPEC_URL_C2
    assert a["forecast_pointer"]["candidate1_registration"] == SPEC_URL_C1


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.40-0.50"
    assert fc["registration"] == SPEC_URL
    assert MODAL_CELL in fc["modal_failure"]
    assert len(fc["deltas_vs_candidate2"]) == 2
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
    }
    # The two deltas surface in the model component descriptions.
    assert "20/22/25/30/40" in model["components"]["first_marriage"]
    assert "delta 1" in model["components"]["first_marriage"].lower()
    wid = model["components"]["widowhood"].lower()
    assert "shrinkage" in wid
    assert "delta 2" in wid
    assert "pooled" in wid
    assert len(model["deltas_vs_candidate2"]) == 2


def test_precheck_reproduced_exactly():
    """The scoring path reproduced the committed floor bit-for-bit."""
    pc = _artifact()["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["reference_moments_exact"] is True
    assert pc["rate_a_exact"] is True
    assert pc["holdout_sha256_all_match"] is True
    assert pc["reference_moments_max_abs_deviation"] == 0.0
    assert pc["rate_a_max_abs_deviation"] == 0.0
    for row in pc["per_seed"]:
        assert row["holdout_sha256_match"] is True


# --------------------------------------------------------------------------
# Tolerances match the locked gates.yaml (always runnable)
# --------------------------------------------------------------------------
def test_stored_tolerances_match_locked_gates_yaml():
    a = _artifact()
    locked = _gate2_tolerances()
    assert len(locked) == N_GATED
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


# --------------------------------------------------------------------------
# The verdict, the registered modal, and the cascade movement (the finding)
# --------------------------------------------------------------------------
def test_verdict_is_fail_zero_of_five():
    """The pre-registered outcome: FAIL, 0/5 seeds pass (published)."""
    v = _artifact()["verdict"]
    assert v["gate_2_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert all(v["seed_pass"][str(s)] is False for s in GATE_SEEDS)


def test_registered_modal_failure_materialized():
    """share_widowed.75+|female failed every seed (the registered modal)."""
    a = _artifact()
    modal = a["modal_failure_materialized"]
    assert modal["modal_cell"] == MODAL_CELL
    assert modal["modal_failed"] is True
    assert modal["any_materialized"] is True
    assert modal["modal_failed_seeds"] == GATE_SEEDS
    # Cross-check against the raw per-seed pass flags.
    for s in a["per_seed"]:
        assert s["gated_cells"][MODAL_CELL]["pass"] is False


def test_modal_block_and_c2_killer_movement_recompute():
    """The modal / secondary / c2-killer tracks recompute from the cells."""
    a = _artifact()
    v = a["verdict"]
    modal = a["modal_failure_materialized"]
    fails: dict[str, list[int]] = {}
    for f in v["all_failing_gated_cells"]:
        fails.setdefault(f["cell"], []).append(f["seed"])

    def check_track(cell: str, track: dict) -> None:
        for s in a["per_seed"]:
            assert track["per_seed_score"][str(s["seed"])] == pytest.approx(
                s["gated_cells"][cell]["score"], abs=0
            )
            assert (
                track["per_seed_pass"][str(s["seed"])]
                == s["gated_cells"][cell]["pass"]
            )
        assert track["failed_seeds"] == sorted(fails.get(cell, []))

    check_track(MODAL_CELL, modal["modal_track"])
    for cell, track in modal["secondary_track"].items():
        check_track(cell, track)
    for cell, track in modal["candidate2_killer_movement"].items():
        check_track(cell, track)


def test_delta2_un_explodes_the_candidate2_cascade():
    """Delta 2 returns the c2 cascade cells to passing on every seed.

    Candidate 2's add-one convention exploded these cells (each failed all
    five seeds): the remarriage bands, the divorced stocks, and the male
    widowhood hazard. Candidate 3's shrinkage toward the pooled rate returns
    them under tolerance on every seed -- the pre-registered "un-explode the
    cascade" prediction, checked directly from the committed cells.
    """
    a = _artifact()
    for cell in C2_CASCADE_CELLS:
        for s in a["per_seed"]:
            rec = s["gated_cells"][cell]
            assert rec["pass"] is True, (cell, s["seed"], rec["score"])


def test_delta1_clears_young_age_first_marriage_clips():
    """The knot at 22 clears first_marriage.18-24 on every seed, both sexes.

    Candidate 1 failed first_marriage.18-24|female on all five seeds;
    candidate 2's age x sex fix cut that to three but introduced three
    |male clips against the tight 0.119 tolerance. The knot at 22 clears
    both on every seed.
    """
    a = _artifact()
    for cell in ("first_marriage.18-24|female", "first_marriage.18-24|male"):
        for s in a["per_seed"]:
            assert s["gated_cells"][cell]["pass"] is True, (cell, s["seed"])


def test_seed0_pins():
    """Pin the committed seed-0 headline outcome (recomputable snapshot)."""
    a = _artifact()
    s0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    assert s0["n_gated_pass"] == 40
    assert s0["seed_pass"] is False
    assert s0["gated_cells"][MODAL_CELL]["pass"] is False
    assert s0["component_meta"]["first_marriage_knots"] == [
        20.0,
        22.0,
        25.0,
        30.0,
        40.0,
    ]


def test_revision_pins_record_runner_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v3"
    assert pins["sklearn_version"].startswith("1.9")
    assert "numpy_version" in pins
    assert "pandas_version" in pins
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    # Provenance that candidate 3 was built on candidates 1 and 2's runners.
    assert pins["candidate1_runner"] == "scripts/run_gate2_candidate1.py"
    assert pins["candidate2_runner"] == "scripts/run_gate2_candidate2.py"
    assert len(pins["candidate1_runner_sha256"]) == 64
    assert len(pins["candidate2_runner_sha256"]) == 64


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate2_registration"] == SPEC_URL_C2
    assert fp["candidate1_registration"] == SPEC_URL_C1
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


# --------------------------------------------------------------------------
# Fertility is byte-identical to candidate 1 (the delta cannot touch it)
# --------------------------------------------------------------------------
def test_fertility_cells_bit_identical_to_candidate1():
    """Fertility is RNG-isolated from the marital process.

    The simulation draws fertility from a separate, demographically-sized
    uniform block each year (``rng.random(n_fertile)``), and fertility is
    modelled independent of marital state, so the two deltas -- which touch
    only first marriage and widowhood -- cannot perturb any fertility
    outcome. Every asfr / completed_fertility cell (gated and report-only)
    must therefore reproduce candidate 1's committed value to bit precision
    across every seed. This directly checks the "everything else
    byte-identical" claim for the RNG-isolated component.
    """
    a3 = _artifact()
    a1 = _artifact_c1()
    by1 = {s["seed"]: s for s in a1["per_seed"]}
    by3 = {s["seed"]: s for s in a3["per_seed"]}
    for seed in GATE_SEEDS:
        s1, s3 = by1[seed], by3[seed]
        checked = 0
        for cell, rec in s3["gated_cells"].items():
            if _is_fertility(cell):
                r1 = s1["gated_cells"][cell]
                assert rec["r_candidate"] == pytest.approx(
                    r1["r_candidate"], abs=1e-12
                ), (seed, cell)
                assert rec["score"] == pytest.approx(r1["score"], abs=1e-12), (
                    seed,
                    cell,
                )
                checked += 1
        for cell, rec in s3["report_only_cells"].items():
            if _is_fertility(cell):
                r1 = s1["report_only_cells"][cell]
                assert rec["r_candidate"] == pytest.approx(
                    r1["r_candidate"], abs=1e-12
                ), (seed, cell)
                assert rec["score"] == pytest.approx(r1["score"], abs=1e-12), (
                    seed,
                    cell,
                )
                checked += 1
        # 11 gated (6 asfr + 5 completed_fertility) + 1 report (asfr.45-49).
        assert checked == 12


# --------------------------------------------------------------------------
# Live seed-0 reproduction (needs the staged PSID history files)
# --------------------------------------------------------------------------
@needs_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 end-to-end and match the committed seed-0 block.

    The registered RNG rule (default_rng(4200 + seed)) plus the fixed person
    ordering and the deterministic (fixed max_iter) logistic fit make the
    candidate-3 simulation deterministic, so every stored candidate rate,
    score, and pass must reproduce to float precision.
    """
    runner = _import_runner()
    from populace_dynamics.data import marriage

    thresholds = runner.c1.load_gate2_thresholds()
    tol = runner.c1.gated_tolerances(thresholds)
    report_only = list(thresholds["report_only"])
    floor = json.loads(FLOOR.read_text())

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, fert, _ = runner.g2f.load_panels()
    order_map = runner.c1._order_map(mh)

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
    assert meta["first_marriage_knots"] == [20.0, 22.0, 25.0, 30.0, 40.0]
    # Delta 2: the smoothing law and its K are recorded.
    assert (
        meta["mortality_smoothing"]
        == "exposure_weighted_shrinkage_toward_pooled_band_sex_rate"
    )
    assert meta["mortality_prior_strength_K"] == 500.0
    assert len(meta["mortality_periods"]) >= 2


@needs_psid
def test_seed0_fertility_matches_candidate1_live():
    """Live check that seed-0 fertility reproduces candidate 1 bit-for-bit.

    Runs the candidate-3 seed-0 simulation and confirms the RNG-isolated
    fertility cells equal candidate 1's committed seed-0 values -- the
    generative check behind :func:`test_fertility_cells_bit_identical`.
    """
    runner = _import_runner()
    from populace_dynamics.data import marriage

    thresholds = runner.c1.load_gate2_thresholds()
    tol = runner.c1.gated_tolerances(thresholds)
    report_only = list(thresholds["report_only"])
    floor = json.loads(FLOOR.read_text())

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, fert, _ = runner.g2f.load_panels()
    order_map = runner.c1._order_map(mh)

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
    c1_seed0 = next(s for s in _artifact_c1()["per_seed"] if s["seed"] == 0)
    for cell, rec in result["gated_cells"].items():
        if _is_fertility(cell):
            assert rec["r_candidate"] == pytest.approx(
                c1_seed0["gated_cells"][cell]["r_candidate"], abs=1e-12
            ), cell
            assert rec["score"] == pytest.approx(
                c1_seed0["gated_cells"][cell]["score"], abs=1e-12
            ), cell


@needs_psid
def test_seed0_delta2_shrinks_toward_candidate1_pooled_table():
    """Delta 2's shrinkage target is candidate 1's pooled band x sex table.

    On seed 0's train complement, refit the candidate-3 period mortality
    table and confirm (i) the non-delta components are byte-identical to
    candidate 2, (ii) the shrinkage formula is exact, and (iii) a thin
    period x band x sex cell collapses to candidate 1's pooled band x sex
    rate (``build_mortality_floors.weighted_hazards``) -- the literal
    "thin cells collapse to candidate 1's pooled behavior".
    """
    import numpy as np

    runner = _import_runner()
    c2 = _import_c2()
    from populace_dynamics.data import marriage
    from populace_dynamics.harness import panel as hpanel

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, _fert, _ = runner.g2f.load_panels()
    order_map = runner.c1._order_map(mh)

    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())
    comp3 = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    comp2 = c2.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # Non-delta components byte-identical to candidate 2.
    assert np.array_equal(comp3.divorce, comp2.divorce)
    assert comp3.remarriage == comp2.remarriage
    assert comp3.fertility == comp2.fertility
    assert comp3.gap_by_sex == comp2.gap_by_sex

    # Recompute the shrinkage table independently; the target is candidate
    # 1's pooled weighted_hazards table.
    mort = runner.mort
    slices = mort.build_exposure_slices(demo, dr)
    slices = slices[slices["person_id"].isin(ids_b)].copy()
    pooled = {k: v["psid_m"] for k, v in mort.weighted_hazards(slices).items()}
    slices["period"] = (slices["start_wave"] // 10 * 10).astype(np.int64)
    slices["we"] = slices["weight"] * slices["exposure"]
    slices["wd"] = slices["weight"] * slices["death"]
    g = slices.groupby(["period", "band", "sex"], observed=True).agg(
        we=("we", "sum"), wd=("wd", "sum")
    )
    thin_key = None
    thin_we = None
    for (p, b, s), r in g.iterrows():
        key = f"{int(p)}|{b}|{s}"
        exp = (float(r.wd) + 500.0 * pooled[f"{b}|{s}"]) / (
            float(r.we) + 500.0
        )
        assert comp3.mortality[key] == pytest.approx(exp, abs=1e-15), key
        if thin_we is None or float(r.we) < thin_we:
            thin_we, thin_key = float(r.we), (key, f"{b}|{s}")
    # The thinnest cell sits within 15% of candidate 1's pooled rate
    # (add-one would have blown it toward 0.5); shrinkage keeps it pooled.
    key, bs = thin_key
    assert comp3.mortality[key] == pytest.approx(pooled[bs], rel=0.15)
