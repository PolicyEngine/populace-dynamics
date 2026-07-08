"""Tests for the gate-2 candidate-2 (run 1) pre-registered run.

Candidate 2 is the second pre-registered gate-2 candidate: candidate 1's
five-component family-transition simulator with EXACTLY two named deltas --
(1) the first-marriage logistic hazard gains an age-spline x sex
interaction, and (2) the widowhood composition's spouse-death table becomes
decade-period x age band x sex (add-one smoothed). Everything else is
byte-identical to candidate 1. Frozen spec: issue #42 comment 4911167286
(candidate 1's spec, comment 4910914098, with the two deltas).

Three tiers:

* the always-runnable consistency suite (touches only the committed
  artifacts and ``gates.yaml``): schema and spec URLs, the two recorded
  deltas, the bit-exact reproduction precheck attestation, every stored
  gated-cell pass recomputes from its score against its stored (locked)
  tolerance, the stored tolerances equal the locked gates.yaml, each seed's
  pass recomputes from the 46 gated cells, the verdict recomputes from the
  seed conjunction, the report-only cells never gate, and the forecast /
  registration / revision pins are carried;
* structural delta checks that need no data: delta 1 adds exactly the
  age-spline x sex block to candidate 1's first-marriage design, and the
  fertility component is byte-identical to candidate 1 across every seed
  (it is RNG-isolated from the marital process, so the two deltas cannot
  perturb it -- a direct check of the "everything else byte-identical"
  claim);
* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  history files are absent) reruns seed 0 end-to-end through the candidate-2
  runner and pins the committed seed-0 block to float precision.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_hazard_v2.json"
ARTIFACT_C1 = ROOT / "runs" / "gate2_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
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
    import run_gate2_candidate2 as runner

    return runner


def _import_c1():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate1 as c1

    return c1


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses candidate 1's dials.

    The frozen dials are IMPORTED from candidate 1, so they are identical by
    construction (any drift would be a divergence from the frozen spec).
    """
    runner = _import_runner()
    c1 = _import_c1()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.GATE_SEEDS is c1.GATE_SEEDS
    assert runner.SIM_SEED_BASE == 4200
    assert runner.SIM_SEED_BASE == c1.SIM_SEED_BASE
    assert runner.SPLINE_KNOTS == (20.0, 25.0, 30.0, 40.0)
    assert runner.SPLINE_KNOTS == c1.SPLINE_KNOTS
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v2"
    assert "4911167286" in runner.SPEC_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert len(runner.DELTAS_VS_CANDIDATE1) == 2


# --------------------------------------------------------------------------
# Delta 1: first-marriage design gains exactly the age-spline x sex block
# --------------------------------------------------------------------------
def test_delta1_first_marriage_adds_age_sex_interaction():
    """The candidate-2 design = candidate 1's design + age-spline x sex."""
    import numpy as np

    runner = _import_runner()
    c1 = _import_c1()
    knots = (20.0, 25.0, 30.0, 40.0)
    cohorts = [1940, 1950, 1960]
    common = dict(
        clf=None,
        cohort_levels=cohorts,
        knots=knots,
        col_mean=np.zeros(1),
        col_sd=np.ones(1),
        n_train_rows=0,
        n_train_events=0,
        n_iter=0,
        converged=False,
    )
    m2 = runner.FirstMarriageModelC2(**common)
    m1 = c1.FirstMarriageModel(**common)
    age = np.array([18.0, 22.0, 30.0, 45.0, 55.0])
    is_male = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    decade = np.array([1940, 1950, 1960, 1940, 1960])
    d1 = m1._raw_design(age, is_male, decade)
    d2 = m2._raw_design(age, is_male, decade)
    # Candidate 2 has exactly 3 more design columns (age-spline x sex).
    assert d2.shape[1] == d1.shape[1] + 3
    spline = c1.ncs_basis(age, knots)
    male = is_male.reshape(-1, 1)
    # Layout: [spline(3), sex(1), spline x sex(3), cohort + age x cohort].
    assert d2[:, :4] == pytest.approx(d1[:, :4])  # spline + sex unchanged
    assert d2[:, 4:7] == pytest.approx(spline * male)  # the new block
    assert d2[:, 7:] == pytest.approx(d1[:, 4:])  # cohort block unchanged
    # The new block is zero for females (reference sex) and nonzero for males.
    assert d2[0, 4:7] == pytest.approx([0.0, 0.0, 0.0])  # female row
    assert np.any(d2[1, 4:7] != 0.0)  # male row


# --------------------------------------------------------------------------
# Artifact presence, spec, lock (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v2"
    assert a["run"] == "gate2_hazard_v2"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 2"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_deltas_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert "4911167286" in a["spec_registration"]
    assert a["candidate1_registration"] == SPEC_URL_C1
    assert "4910914098" in a["candidate1_registration"]
    # Exactly two named deltas, one on first marriage, one on widowhood.
    deltas = a["deltas_vs_candidate1"]
    assert len(deltas) == 2
    joined = " ".join(deltas).lower()
    assert "age-spline x sex" in joined or "age spline x sex" in joined
    assert "decade-period" in joined
    assert a["forecast_pointer"]["registration"] == SPEC_URL
    assert a["forecast_pointer"]["candidate1_registration"] == SPEC_URL_C1


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.55-0.65"
    assert fc["registration"] == SPEC_URL
    assert "male" in fc["modal_failure"].lower()
    assert "occupancy" in fc["modal_failure"].lower()
    assert len(fc["deltas_vs_candidate1"]) == 2
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
    assert "sex" in model["components"]["first_marriage"].lower()
    assert "delta 1" in model["components"]["first_marriage"].lower()
    assert "decade-period" in model["components"]["widowhood"].lower()
    assert "delta 2" in model["components"]["widowhood"].lower()
    assert len(model["deltas_vs_candidate1"]) == 2


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


def test_modal_failure_block_recomputes():
    """The registered c2 modal (male ever-married occupancy) recomputes."""
    a = _artifact()
    v = a["verdict"]
    modal = a["modal_failure_materialized"]
    fails: dict[str, list[int]] = {}
    for f in v["all_failing_gated_cells"]:
        fails.setdefault(f["cell"], []).append(f["seed"])
    male_occ = {
        c: sorted(fails[c])
        for c in ("ever_married_by_40|male", "ever_married_by_60|male")
        if c in fails
    }
    assert modal["male_ever_married_occupancy_failed"] == bool(male_occ)
    assert modal["male_ever_married_occupancy_seeds"] == male_occ
    assert modal["any_materialized"] == bool(male_occ)
    # The candidate-1 killer movement block tracks every targeted cell.
    for cell in (
        "first_marriage.18-24|female",
        "share_widowed.75+|female",
        "share_widowed.65-74|female",
        "widowhood.45-64|female",
    ):
        track = modal["candidate1_killer_movement"][cell]
        for s in a["per_seed"]:
            assert track["per_seed_score"][str(s["seed"])] == pytest.approx(
                s["gated_cells"][cell]["score"], abs=0
            )
            assert (
                track["per_seed_pass"][str(s["seed"])]
                == s["gated_cells"][cell]["pass"]
            )


def test_revision_pins_record_sklearn_version():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v2"
    assert pins["sklearn_version"].startswith("1.9")
    assert "numpy_version" in pins
    assert "pandas_version" in pins
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    # Provenance that candidate 2 was built on candidate 1's runner.
    assert pins["candidate1_runner"] == "scripts/run_gate2_candidate1.py"
    assert len(pins["candidate1_runner_sha256"]) == 64


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate1_registration"] == SPEC_URL_C1
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


def test_mortality_is_period_stratified():
    """Delta 2 is recorded in every seed's component meta."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert meta["mortality_stratification"] == "decade_period x band x sex"
        # More than one decade-period is identified on the train complement.
        assert len(meta["mortality_periods"]) >= 2
        assert "first_marriage_design" in meta
        assert "sex" in meta["first_marriage_design"]


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
    a2 = _artifact()
    a1 = _artifact_c1()
    by1 = {s["seed"]: s for s in a1["per_seed"]}
    by2 = {s["seed"]: s for s in a2["per_seed"]}
    for seed in GATE_SEEDS:
        s1, s2 = by1[seed], by2[seed]
        checked = 0
        for cell, rec in s2["gated_cells"].items():
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
        for cell, rec in s2["report_only_cells"].items():
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
    ordering makes the candidate-2 simulation deterministic, so every stored
    candidate rate, score, and pass must reproduce to float precision.
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
    # Delta 2: seed 0 identifies several decade-periods of spouse mortality.
    assert meta["mortality_stratification"] == "decade_period x band x sex"
    assert len(meta["mortality_periods"]) >= 2


@needs_psid
def test_seed0_fertility_matches_candidate1_live():
    """Live check that seed-0 fertility reproduces candidate 1 bit-for-bit.

    Runs the candidate-2 seed-0 simulation and confirms the RNG-isolated
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
            assert rec["score"] == pytest.approx(
                c1_seed0["gated_cells"][cell]["score"], abs=1e-12
            ), cell
