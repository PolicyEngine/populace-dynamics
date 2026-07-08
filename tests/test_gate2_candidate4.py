"""Tests for the gate-2 candidate-4 (run 1) pre-registered run.

Candidate 4 is the fourth pre-registered gate-2 candidate: candidate 3's
five-component family-transition simulator with EXACTLY two named deltas --
(1) the spouse-death hazard drops candidate 3's decade-period shrinkage
table ENTIRELY and models the period effect parametrically, ``rate(band,
sex, year) = pooled(band, sex) * exp(beta_sex * (year - 1995))`` with
``pooled`` = candidate 1's pooled band x sex rate (the 1995 anchor) and
``beta_sex`` a single per-sex log-linear period slope from a weighted
Poisson GLM on the train ``(band x start_wave)`` death cells with age-band
fixed effects; and (2) the simulated spouse's age gap is DRAWN from the
train sex-specific empirical 1-year-binned gap distribution (the same record
selection whose mean candidates 1-3 used) instead of imputed at the mean,
via an RNG stream spawned from the registered ``default_rng(4200 + seed)``.
Everything else is byte-identical to candidate 3 -- in particular the
first-marriage hazard (candidate 3's knot-at-22 spline, reused by identity),
and the fertility subprocess (RNG-isolated: the gap draw is spawned off the
registered stream and cannot perturb the per-year uniform blocks, so
fertility reproduces candidate 1 bit-for-bit). Frozen spec: issue #42
comment 4911532899 (candidate 3's spec, comment 4911357564, with the two
deltas).

Three tiers:

* the always-runnable consistency suite (touches only the committed
  artifacts and ``gates.yaml``): schema and spec URLs, the two recorded
  deltas, the bit-exact reproduction precheck attestation, every stored
  gated-cell pass recomputes from its score against its stored (locked)
  tolerance, the stored tolerances equal the locked gates.yaml, each seed's
  pass recomputes from the 46 gated cells, the verdict recomputes from the
  seed conjunction, the report-only cells never gate, the registered modal
  (``mean_lifetime_marriages|male``), the widowed-stock cluster, the decider
  analysis and the candidate-3 movement all recompute, and the forecast /
  registration / revision pins are carried;
* structural delta checks: delta 1 records the parametric trend and its
  per-sex betas (and carries no stale decade-period / shrinkage-K keys),
  delta 2 records the empirical gap-distribution draw, the first-marriage
  fitter is candidate 3's object (reused by identity -- neither delta touches
  it), the RNG-isolation mechanism (``SeedSequence.spawn`` leaves the parent
  bit stream byte-identical) holds as a direct numpy unit check, and the
  fertility component is byte-identical to candidate 1 across every seed;
* :func:`test_seed0_reproduces_committed_artifact` and the live delta checks
  (skipped when the PSID history files are absent) rerun seed 0 end-to-end
  through the candidate-4 runner and pin the committed seed-0 block to float
  precision, confirm the delta-1 anchor is candidate 1's pooled table and the
  betas reproduce, confirm the non-delta components are byte-identical to
  candidate 3, and confirm fertility reproduces candidate 1 bit-for-bit.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_hazard_v4.json"
ARTIFACT_C1 = ROOT / "runs" / "gate2_hazard_v1.json"
ARTIFACT_C3 = ROOT / "runs" / "gate2_hazard_v3.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911532899"
)
SPEC_URL_C3 = (
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

#: The registered modal failure (comment 4911532899).
MODAL_CELL = "mean_lifetime_marriages|male"
#: The female widowed-stock cluster that was candidate 3's isolated failure
#: (each failed all five seeds under candidate 3).
WIDOWED_STOCK_CLUSTER = (
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c1() -> dict:
    return json.loads(ARTIFACT_C1.read_text())


def _artifact_c3() -> dict:
    return json.loads(ARTIFACT_C3.read_text())


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
    import run_gate2_candidate4 as runner

    return runner


def _import_c1():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate1 as c1

    return c1


def _import_c3():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate3 as c3

    return c3


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses candidate 1's dials.

    The frozen dials are IMPORTED from candidate 1, so they are identical by
    construction. The delta-1 anchor year (1995) and the reused candidate-3
    first-marriage fitter are the provenance of "candidate 3 verbatim except
    the two deltas".
    """
    runner = _import_runner()
    c1 = _import_c1()
    c3 = _import_c3()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.GATE_SEEDS is c1.GATE_SEEDS
    assert runner.SIM_SEED_BASE == 4200
    assert runner.SIM_SEED_BASE == c1.SIM_SEED_BASE
    # DELTA 1: parametric trend anchored at 1995.
    assert runner.TREND_ANCHOR_YEAR == 1995.0
    # First marriage is candidate 3's, reused unchanged (knot-at-22 spline).
    assert runner.SPLINE_KNOTS_C3 == (20.0, 22.0, 25.0, 30.0, 40.0)
    assert runner.SPLINE_KNOTS_C3 is c3.SPLINE_KNOTS_C3
    assert runner.fit_first_marriage is c3.fit_first_marriage
    assert runner.FirstMarriageModelC4 is c3.FirstMarriageModelC3
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v4"
    assert "4911532899" in runner.SPEC_REGISTRATION
    assert "4911357564" in runner.CANDIDATE3_REGISTRATION
    assert "4911167286" in runner.CANDIDATE2_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert len(runner.DELTAS_VS_CANDIDATE3) == 2


def test_first_marriage_reused_from_candidate3_by_identity():
    """Neither delta touches first marriage: it is candidate 3's exact object.

    The candidate-4 first-marriage fitter and design class ARE candidate 3's
    (the knot-at-22 20/22/25/30/40 spline with the age-spline x sex +
    age-spline x cohort design), reused by identity -- the strongest possible
    "first marriage byte-identical to candidate 3" guarantee.
    """
    runner = _import_runner()
    c3 = _import_c3()
    assert runner.fit_first_marriage is c3.fit_first_marriage
    assert runner.FirstMarriageModelC4 is c3.FirstMarriageModelC3


# --------------------------------------------------------------------------
# Delta 1: parametric per-sex log-linear mortality trend
# --------------------------------------------------------------------------
def test_delta1_mortality_trend_recorded():
    """Every seed records the parametric trend and its per-sex betas.

    Candidate 4 drops candidate 3's decade-period shrinkage table entirely,
    so the artifact must carry NO stale period / shrinkage-K keys and instead
    the pooled-anchor + per-sex log-linear trend and its fitted betas.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert meta["mortality_stratification"] == (
            "pooled band x sex (candidate 1 anchor) x per-sex log-linear "
            "period trend"
        )
        assert meta["mortality_trend"] == (
            "rate(band, sex, year) = pooled(band, sex) * "
            "exp(beta_sex * (year - 1995))"
        )
        assert meta["mortality_trend_anchor_year"] == 1995.0
        assert "Poisson" in meta["mortality_trend_estimator"]
        assert "fixed effects" in meta["mortality_trend_estimator"]
        # 7 mortality bands x 2 sexes = 14 pooled anchor cells.
        assert meta["mortality_cells"] == 14
        beta = meta["mortality_beta_by_sex"]
        assert set(beta) == {"female", "male"}
        assert isinstance(beta["female"], float)
        assert isinstance(beta["male"], float)
        # Candidate 3's decade-period / shrinkage keys are GONE, and the
        # smoothing field is explicitly "none (parametric ...)".
        assert "mortality_periods" not in meta
        assert "mortality_prior_strength_K" not in meta
        assert meta["mortality_smoothing"].startswith("none")
        assert "parametric" in meta["mortality_smoothing"]
        # First marriage is candidate 3's knot-at-22 spline (retained).
        assert meta["first_marriage_knots"] == [20.0, 22.0, 25.0, 30.0, 40.0]
        # Per-sex trend diagnostics recorded and (weighted) MLE converged.
        diag = meta["mortality_trend_diagnostics"]
        for sex in ("female", "male"):
            assert diag[sex]["converged"] is True
            assert diag[sex]["beta"] == pytest.approx(beta[sex], abs=0)


# --------------------------------------------------------------------------
# Delta 2: empirical spousal-age-gap distribution draw
# --------------------------------------------------------------------------
def test_delta2_gap_distribution_recorded():
    """Every seed records the empirical gap-distribution draw + its summary.

    Delta 2 replaces the train MEAN spousal age gap with a per-person draw
    from the train sex-specific empirical 1-year-binned distribution, via an
    RNG stream spawned from the registered generator; the mean is retained in
    ``gap_by_sex`` for provenance.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        imp = meta["spousal_gap_imputation"]
        assert "empirical" in imp
        assert "1-year bins" in imp
        assert "spawned" in imp
        summ = meta["spousal_gap_dist_summary"]
        assert set(summ) == {"female", "male"}
        for sex in ("female", "male"):
            assert summ[sex]["n"] > 0
            assert summ[sex]["n_bins"] > 1  # a real distribution, not a point
            # The distribution mean equals candidate 1's retained mean gap.
            assert summ[sex]["mean"] == pytest.approx(
                meta["gap_by_sex"][sex], abs=1e-9
            )


def test_rng_isolation_spawn_leaves_parent_stream_identical():
    """The load-bearing delta-2 mechanism: spawn does not advance the parent.

    Fertility byte-identity rests on drawing the spousal-gap from a stream
    SPAWNED off the registered ``default_rng(4200 + seed)`` -- the spawn must
    not perturb the parent's bit stream, so the per-year ``rng.random`` blocks
    (and thus the RNG-isolated fertility subprocess) are unchanged. A direct
    numpy unit check of exactly the runner's construction.
    """
    import numpy as np

    ref = np.random.default_rng(4242).random(16)
    rng = np.random.default_rng(4242)
    gap_seed_seq = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_seq)
    _ = gap_rng.choice(np.arange(-30, 31), size=5000)  # heavy gap draws
    after = rng.random(16)
    assert np.array_equal(ref, after)


# --------------------------------------------------------------------------
# Artifact presence, spec, lock (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v4"
    assert a["run"] == "gate2_hazard_v4"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 4"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_deltas_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert "4911532899" in a["spec_registration"]
    assert a["candidate3_registration"] == SPEC_URL_C3
    assert a["candidate2_registration"] == SPEC_URL_C2
    assert a["candidate1_registration"] == SPEC_URL_C1
    # Exactly two named deltas: parametric mortality trend, gap distribution.
    deltas = a["deltas_vs_candidate3"]
    assert len(deltas) == 2
    joined = " ".join(deltas).lower()
    assert "poisson" in joined
    assert "exp(beta_sex" in joined
    assert "1995" in joined
    assert "distribution" in joined
    assert "1-year bins" in joined
    assert "spawned" in joined
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate3_registration"] == SPEC_URL_C3
    assert fp["candidate1_registration"] == SPEC_URL_C1


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.45-0.55"
    assert fc["registration"] == SPEC_URL
    assert MODAL_CELL in fc["modal_failure"]
    # The pre-registered diagnostic escape-hatch is recorded (so it cannot be
    # a post-hoc rescue).
    assert "diagnostic" in fc["diagnostic_flag"].lower()
    assert len(fc["deltas_vs_candidate3"]) == 2
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
    wid = model["components"]["widowhood"].lower()
    assert "delta 1" in wid
    assert "poisson" in wid
    assert "1995" in wid
    assert "delta 2" in wid  # the gap draw is named in the widowhood block
    assert len(model["deltas_vs_candidate3"]) == 2


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
# The verdict, the registered modal, and the cluster movement (the finding)
# --------------------------------------------------------------------------
def test_verdict_is_fail_zero_of_five():
    """The pre-registered outcome: FAIL, 0/5 seeds pass (published)."""
    v = _artifact()["verdict"]
    assert v["gate_2_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert all(v["seed_pass"][str(s)] is False for s in GATE_SEEDS)


def test_registered_modal_materialized_four_seeds():
    """mean_lifetime_marriages|male failed seeds 0,1,3,4 (the modal residual).

    The registered modal persisted through candidate 4 essentially unmoved --
    the widowed-pool recomposition did not lift it off its 0.047 tolerance.
    """
    a = _artifact()
    modal = a["modal_failure_materialized"]
    assert modal["modal_cell"] == MODAL_CELL
    assert modal["modal_failed"] is True
    assert modal["modal_failed_seeds"] == [0, 1, 3, 4]
    for s in a["per_seed"]:
        expect = s["seed"] in (0, 1, 3, 4)
        assert s["gated_cells"][MODAL_CELL]["pass"] is (not expect)


def test_widowed_stock_65_74_female_fully_fixed():
    """Delta 1's headline: share_widowed.65-74|female passes ALL five seeds.

    Candidate 3 failed this cell on all five seeds (the isolated widowed-stock
    cluster); the parametric per-sex mortality trend returns it under
    tolerance on every seed -- the pre-registered "the trend attacks the stock
    drift's mechanism" prediction, checked directly from the committed cells.
    """
    a = _artifact()
    cell = "share_widowed.65-74|female"
    for s in a["per_seed"]:
        assert s["gated_cells"][cell]["pass"] is True, (s["seed"],)


def test_decider_is_broader_than_modal_and_cluster():
    """The verdict was decided diffusely, not by the modal or cluster alone.

    Even forgiving BOTH the registered modal and the entire widowed-stock
    cluster leaves zero passing seeds: the residual is spread across
    mean_lifetime_marriages (both sexes), the surviving 75+ female stock, and
    the inherited fertility / remarriage footnote cells. So no single
    mechanism holds the gate -- the honest finding.
    """
    a = _artifact()
    dec = a["modal_failure_materialized"]["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == 0
    assert dec["n_seeds_pass_if_modal_forgiven"] == 0
    assert dec["n_seeds_pass_if_cluster_forgiven"] == 0
    assert dec["n_seeds_pass_if_both_forgiven"] == 0
    assert "broader" in dec["decider"]


def test_modal_and_cluster_tracks_recompute():
    """The modal / cluster / candidate-3 residual tracks recompute."""
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
    for cell, track in modal["widowed_stock_cluster_track"].items():
        check_track(cell, track)
    for cell, track in modal["candidate3_residual_track"].items():
        check_track(cell, track)
    assert set(modal["widowed_stock_cluster"]) == set(WIDOWED_STOCK_CLUSTER)


def test_candidate3_comparison_movement_recomputes():
    """The vs-candidate-3 movement block recomputes from both artifacts.

    share_widowed.65-74|female moves 0/5 -> 5/5; mean_lifetime_marriages|male
    stays 1/5 (the modal residual). Both recompute from the committed
    candidate-3 and candidate-4 per-seed cells.
    """
    a = _artifact()
    cmp = a["candidate3_comparison"]
    assert cmp["available"] is True
    a3 = _artifact_c3()
    by3 = {s["seed"]: s for s in a3["per_seed"]}
    by4 = {s["seed"]: s for s in a["per_seed"]}
    for cell, d in cmp["cells"].items():
        c3_pass = sum(by3[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c4_pass = sum(by4[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate3_n_seeds_pass"] == c3_pass
        assert d["candidate4_n_seeds_pass"] == c4_pass
        for s in GATE_SEEDS:
            assert d["candidate3_per_seed_score"][str(s)] == pytest.approx(
                by3[s]["gated_cells"][cell]["score"], abs=0
            )
            assert d["candidate4_per_seed_score"][str(s)] == pytest.approx(
                by4[s]["gated_cells"][cell]["score"], abs=0
            )
    # The headline movement, pinned.
    assert (
        cmp["cells"]["share_widowed.65-74|female"]["candidate3_n_seeds_pass"]
        == 0
    )
    assert (
        cmp["cells"]["share_widowed.65-74|female"]["candidate4_n_seeds_pass"]
        == 5
    )
    assert cmp["cells"][MODAL_CELL]["candidate3_n_seeds_pass"] == 1
    assert cmp["cells"][MODAL_CELL]["candidate4_n_seeds_pass"] == 1


def test_seed0_pins():
    """Pin the committed seed-0 headline outcome (recomputable snapshot)."""
    a = _artifact()
    s0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    assert s0["n_gated_pass"] == 44
    assert s0["seed_pass"] is False
    fails = {c for c, r in s0["gated_cells"].items() if not r["pass"]}
    assert fails == {"asfr.20-24", "mean_lifetime_marriages|male"}
    assert s0["component_meta"]["first_marriage_knots"] == [
        20.0,
        22.0,
        25.0,
        30.0,
        40.0,
    ]
    assert set(s0["component_meta"]["mortality_beta_by_sex"]) == {
        "female",
        "male",
    }


def test_revision_pins_record_runner_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v4"
    assert pins["sklearn_version"].startswith("1.9")
    assert "numpy_version" in pins
    assert "pandas_version" in pins
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    # Provenance that candidate 4 was built on candidates 1, 2, and 3.
    for n in (1, 2, 3):
        assert pins[f"candidate{n}_runner"] == (
            f"scripts/run_gate2_candidate{n}.py"
        )
        assert len(pins[f"candidate{n}_runner_sha256"]) == 64


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate3_registration"] == SPEC_URL_C3
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


# --------------------------------------------------------------------------
# Fertility is byte-identical to candidate 1 (the delta cannot touch it)
# --------------------------------------------------------------------------
def test_fertility_cells_bit_identical_to_candidate1():
    """Fertility is RNG-isolated: the delta-2 gap draw cannot perturb it.

    The simulation draws fertility from a separate, demographically-sized
    uniform block each year (``rng.random(n_fertile)``) and models fertility
    independent of marital state; the delta-2 spousal-gap draw comes from a
    stream SPAWNED off the registered generator, which does not advance the
    registered bit stream, so every ``asfr`` / ``completed_fertility`` cell
    (gated and report-only) reproduces candidate 1's committed value to bit
    precision across every seed -- the "everything else byte-identical" claim
    for the RNG-isolated component, and the test-pinned fertility byte-identity
    vs v1.
    """
    a4 = _artifact()
    a1 = _artifact_c1()
    by1 = {s["seed"]: s for s in a1["per_seed"]}
    by4 = {s["seed"]: s for s in a4["per_seed"]}
    for seed in GATE_SEEDS:
        s1, s4 = by1[seed], by4[seed]
        checked = 0
        for cell, rec in s4["gated_cells"].items():
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
        for cell, rec in s4["report_only_cells"].items():
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
    ordering, the deterministic logistic fit, and the deterministic Poisson
    IRLS make the candidate-4 simulation deterministic, so every stored
    candidate rate, score, pass, and fitted beta must reproduce to float
    precision.
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
    assert meta["first_marriage_knots"] == [20.0, 22.0, 25.0, 30.0, 40.0]
    # Delta 1: the fitted per-sex betas reproduce.
    for sex in ("female", "male"):
        assert meta["mortality_beta_by_sex"][sex] == pytest.approx(
            cmeta["mortality_beta_by_sex"][sex], abs=1e-12
        )


@needs_psid
def test_seed0_delta1_anchor_is_candidate1_pooled_table():
    """Delta 1's anchor is candidate 1's pooled band x sex table.

    On seed 0's train complement, refit the candidate-4 components and confirm
    (i) the mortality anchor is exactly candidate 1's pooled weighted_hazards
    table (``build_mortality_floors.weighted_hazards`` psid_m), (ii) the
    non-delta components (divorce, remarriage, fertility, the spousal-gap
    MEAN, and first marriage) are byte-identical to candidate 3, and (iii) the
    empirical gap distribution's mean equals the retained mean gap.
    """
    import numpy as np

    runner = _import_runner()
    c3 = _import_c3()
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
    comp4 = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    comp3 = c3.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # (i) Anchor = candidate 1's pooled weighted_hazards table.
    mort = runner.mort
    slices = mort.build_exposure_slices(demo, dr)
    slices = slices[slices["person_id"].isin(ids_b)]
    pooled = {k: v["psid_m"] for k, v in mort.weighted_hazards(slices).items()}
    assert comp4.mortality == pooled

    # (ii) Non-delta components byte-identical to candidate 3.
    assert np.array_equal(comp4.divorce, comp3.divorce)
    assert comp4.remarriage == comp3.remarriage
    assert comp4.fertility == comp3.fertility
    assert comp4.gap_by_sex == comp3.gap_by_sex
    # First marriage: candidate 3's knots, retained.
    assert comp4.first_marriage.knots == (20.0, 22.0, 25.0, 30.0, 40.0)
    assert comp4.first_marriage.knots == comp3.first_marriage.knots

    # (iii) The empirical gap distribution mean equals the retained mean.
    for sex in ("female", "male"):
        assert float(comp4.gap_dist_by_sex[sex].mean()) == pytest.approx(
            comp4.gap_by_sex[sex], abs=1e-9
        )


@needs_psid
def test_seed0_fertility_matches_candidate1_live():
    """Live check that seed-0 fertility reproduces candidate 1 bit-for-bit.

    Runs the candidate-4 seed-0 simulation and confirms the RNG-isolated
    fertility cells equal candidate 1's committed seed-0 values -- the
    generative check behind :func:`test_fertility_cells_bit_identical`, and a
    direct end-to-end confirmation that the delta-2 spawned gap-draw stream
    does not perturb the registered fertility subprocess.
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
