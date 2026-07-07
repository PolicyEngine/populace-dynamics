"""Tests for the gate-1 candidate-9 calibrated-memory run (amended gate).

Candidate 9 is the FIRST run scored under the amended gate (PR #57/#59): the
runs-view c2st is demoted to reported-not-gated, a gated benefit-space block
folds into the geometry verdict, and the gate needs >= 4/5 geometry AND
>= 4/5 battery AND the pooled Q0 band. The tests mirror the candidate-7/8
tests, adapted for the two registered changes and the amended scoring:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent, and ``importorskip("populace.fit")`` because the
  participation gates need it) reruns seed 0 through the candidate-9 u_w
  decomposition + donor pools + SMM-lambda calibration + zero-anchor regime +
  backward k-NN generation and pins the committed artifact's seed-0 lambda,
  geometry, battery, and Q0 benefit values to float precision. It is run live
  in the dedicated gate venv before the artifact is committed.
* :func:`test_donor_pools_and_blend_on_synthetic_pool` (always runnable; no
  PSID, importorskip populace because the u_w decomposition imports the
  candidate-3 -> candidate-2 -> populace.fit chain) drives the pool
  construction with the carried ``u_w`` / ``anchor_zero`` fields, the
  zero-anchor re-entry pool, the pinned tie-break sort, and the donor-blend
  coordinate.
* :func:`test_knn_draw_weighted_and_tie_break` (always runnable) pins the
  k-NN weighted single-record draw (imported byte-for-byte from candidate 7).
* :func:`test_lambda_grid_and_smm_tie_break` (always runnable) checks the
  frozen lambda grid and the SMM tie-break (smaller lambda wins on equal
  SSE) on a synthetic ladder objective.
* :func:`test_zero_anchor_regime_routes_restricted_pool` (importorskip
  populace) checks that a zero-anchor target draws re-entry innovations from
  the zero-anchor-restricted pool while positive-anchor targets do not.
* The always-runnable consistency tests touch only the committed artifact and
  ``gates.yaml``: the schema and spec URL, the exact battery-reference
  reproduction, every reported geometry / battery / benefit-space pass
  recomputes from its own stored score against its stored (locked) threshold,
  the zero-persistence identity, the AMENDED verdict recomputation (geometry
  conjoins the per-seed benefit-space metrics; the gate conjoins the pooled
  Q0 gate), the two registered changes, the lambda-per-seed record, and the
  reported diagnostics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v3.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4898825218"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate9 as runner

    return runner


def _synthetic_train() -> pd.DataFrame:
    """A hand-built train panel spanning the biennial reference years.

    Enough persons across enough of the 13 biennial periods (1998-2022) that
    every within-person lag k = 1..5 has pairs, so the u_w z-panel
    decomposition has finite moments. Includes zero-anchor persons (positive
    history, zero at their last observed period) so the zero-anchor re-entry
    pool is non-empty.
    """
    rng = np.random.default_rng(9)
    periods = list(range(1998, 2023, 2))  # 13 biennial periods
    rows = []
    for pid in range(1, 121):
        start = int(rng.integers(0, 7))
        span = int(rng.integers(6, 13 - start + 1))
        perm = rng.normal(10.6, 0.5)
        for k in range(span):
            p = periods[start + k]
            age = min(59, 30 + (start + k) * 2 + int(rng.integers(0, 3)))
            earn = float(np.exp(perm + rng.normal(0, 0.3)))
            rows.append((pid, p, earn, age, 1.0 + (pid % 3)))
        if pid % 5 == 0:  # zero-anchor persons
            last = rows[-1]
            rows[-1] = (last[0], last[1], 0.0, last[3], last[4])
    return pd.DataFrame(
        rows, columns=["person_id", "period", "earnings", "age", "weight"]
    )


# --------------------------------------------------------------------------
# Reproduction (needs the staged PSID family files AND populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 and match the committed artifact to float precision."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    runner = _import_runner()
    import build_downstream_relevance as ds

    from populace_dynamics.ss.params import load_ssa_parameters

    artifact = _artifact()
    seed0 = next(s for s in artifact["per_seed"] if s["seed"] == 0)
    thresholds = runner.load_gate1_thresholds()
    views_cfg = thresholds["views"]
    benefit_metrics_cfg = thresholds["benefit_space"]["metrics"]
    battery_reference = json.loads(
        (runner.ROOT / runner.BATTERY_REFERENCE_RUN).read_text()
    )["battery_reference"]
    panel = runner.load_filtered_panel()
    all_anchor = runner.anchor_rows(panel)
    params = load_ssa_parameters()
    cutpoints = ds.anchor_quintile_cutpoints(all_anchor)
    view_specs = {
        "psid_family_earnings_pairs": runner.build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": runner.build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }
    result = runner.run_seed(
        0,
        panel,
        all_anchor,
        view_specs,
        views_cfg,
        battery_reference,
        battery_tol,
        benefit_metrics_cfg,
        params,
        cutpoints,
        False,
    )
    # The SMM chooses lambda deterministically from the seed.
    assert result["lambda"] == seed0["lambda"]
    for view, block in seed0["geometry"].items():
        for metric, stored in block["scores"].items():
            assert result["geometry"][view]["scores"][metric] == pytest.approx(
                stored, abs=1e-12
            ), f"{view}.{metric}"
    for stat, stored in seed0["battery_values"].items():
        assert result["battery_values"][stat] == pytest.approx(
            stored, abs=1e-12
        ), stat
    # Deterministic given the split: pool sizes and the u_w decomposition.
    for key in ("n_pairs", "n_triples", "n_reentry", "n_reentry_q0"):
        assert result["pools"][key] == seed0["pools"][key], key
    assert result["uw_fit"]["rho"] == pytest.approx(
        seed0["uw_fit"]["rho"], abs=0
    )
    assert result["uw_fit"]["sigma_hat_w"] == pytest.approx(
        seed0["uw_fit"]["sigma_hat_w"], abs=1e-12
    )
    # The gated Q0 benefit statistic reproduces to float precision.
    q0_stored = seed0["benefit_space"]["by_anchor_quintile"]["quintiles"][
        "Q0"
    ]["distribution"]["mean"]["pct_diff"]
    q0_got = result["benefit_space"]["by_anchor_quintile"]["quintiles"]["Q0"][
        "distribution"
    ]["mean"]["pct_diff"]
    assert q0_got == pytest.approx(q0_stored, abs=1e-9)


# --------------------------------------------------------------------------
# Donor pools + blend on a hand-built panel (importorskip populace)
# --------------------------------------------------------------------------
def test_donor_pools_and_blend_on_synthetic_pool():
    """u_w-carrying pools, the zero-anchor re-entry pool, and the blend."""
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    runner = _import_runner()
    import run_gate1_candidate8 as c8

    panel = _synthetic_train()
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(panel)
    uw = c8.build_donor_uw(panel, marginals)
    pools = runner.build_donor_pools(
        panel, all_anchor, marginals, uw["u_w_of_person"]
    )

    # Every record carries u_A, u_w, anchor_zero, weight.
    for name in ("pairs", "triples", "reentry"):
        p = pools[name]
        n = len(p["u_prev"])
        for field in ("u_A", "u_w", "anchor_zero", "weight"):
            assert len(p[field]) == n, f"{name}.{field}"
        assert np.all(np.isfinite(p["u_w"]))
        assert np.all((p["u_w"] >= 0.0) & (p["u_w"] <= 1.0))
        # Pinned in ascending (person_id, period_prev) order.
        key = p["person_id"].astype(np.int64) * 100000 + p[
            "period_prev"
        ].astype(np.int64)
        assert np.all(np.diff(key) >= 0), f"{name} not in pinned order"

    # u_w is constant within a person.
    for name in ("pairs", "triples", "reentry"):
        p = pools[name]
        df = pd.DataFrame({"pid": p["person_id"], "uw": p["u_w"]})
        for _, g in df.groupby("pid"):
            assert g["uw"].nunique() == 1

    # The zero-anchor re-entry pool is the subset whose donor anchor is zero.
    zero_anchor_pids = set(
        int(p) for p in all_anchor[all_anchor.earnings == 0].person_id
    )
    req0 = pools["reentry_q0"]
    assert np.all(req0["anchor_zero"])
    for pid in req0["person_id"]:
        assert int(pid) in zero_anchor_pids
    assert pools["n_reentry_q0"] == len(req0["u_prev"])
    assert len(req0["u_prev"]) <= pools["n_reentry"]
    # The synthetic panel has zero-anchor persons -> the pool is non-empty.
    assert pools["n_reentry_q0"] > 0

    # The blend: lambda=0 -> donor u_A; lambda=1 -> donor u_w; monotone.
    u_w = pools["pairs"]["u_w"]
    u_A = pools["pairs"]["u_A"]
    assert np.allclose(runner._donor_blend(u_w, u_A, 0.0), u_A)
    assert np.allclose(runner._donor_blend(u_w, u_A, 1.0), u_w)
    mid = runner._donor_blend(u_w, u_A, 0.5)
    assert np.allclose(mid, 0.5 * u_w + 0.5 * u_A)


def test_knn_draw_weighted_and_tie_break():
    """The k-NN draw (candidate 7's, imported) takes k nearest, draws by weight."""
    runner = _import_runner()

    dist = np.array([[0.20, 0.40, 0.05, 0.30]], dtype=np.float64)
    weight = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    u_prev = np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float64)
    drawn, kth = runner._knn_draw(dist, weight, u_prev, np.array([0.0]))
    assert float(drawn[0]) == pytest.approx(0.33, abs=0)
    assert float(kth[0]) == pytest.approx(0.40, abs=0)

    weight_w = np.array([1.0, 1.0, 100.0, 1.0], dtype=np.float64)
    drawn_w, _ = runner._knn_draw(dist, weight_w, u_prev, np.array([0.5]))
    assert float(drawn_w[0]) == pytest.approx(0.33, abs=0)

    dist_tie = np.array([[0.10, 0.10, 0.05, 0.30]], dtype=np.float64)
    drawn_t, _ = runner._knn_draw(dist_tie, weight, u_prev, np.array([0.34]))
    assert float(drawn_t[0]) == pytest.approx(0.11, abs=0)


def test_lambda_grid_and_smm_tie_break():
    """The frozen lambda grid and the SMM tie-break (smaller lambda wins).

    Drives :func:`calibrate_lambda` indirectly is heavy (needs populace); the
    tie-break rule itself is a pure "strict-<" scan over the grid, checked
    here by replicating the selection convention on a synthetic SSE ladder
    with an exact tie between two lambdas -- the smaller must win.
    """
    runner = _import_runner()
    # Frozen grid: 11 points {0, 0.1, ..., 1.0}.
    assert runner.LAMBDA_GRID == tuple(round(0.1 * i, 1) for i in range(11))
    assert runner.SMM_SUBSAMPLE == 2000
    assert runner.SMM_LAGS == (1, 2, 5)

    # Selection convention: strict "<" keeps the FIRST (smallest) lambda on a
    # tie. Replicate the scan on a synthetic SSE vector with a tie at the two
    # smallest-SSE lambdas (indices 3 and 5, both 0.10).
    sse = [0.9, 0.8, 0.7, 0.10, 0.6, 0.10, 0.5, 0.4, 0.3, 0.2, 0.15]
    best_idx = None
    best = np.inf
    for i, s in enumerate(sse):
        if s < best:
            best = s
            best_idx = i
    assert best_idx == 3  # the smaller lambda of the tied pair
    assert runner.LAMBDA_GRID[best_idx] == pytest.approx(0.3)


def test_zero_anchor_regime_routes_restricted_pool():
    """A zero-anchor target's re-entry draws use the restricted pool.

    Drives :func:`generate_candidate` on a small panel with a stubbed
    always-positive participation gate and checks, from the diagnostics, that
    zero-anchor holdout persons route re-entry draws through the zero-anchor
    re-entry pool (n_reentry_draws_q0 > 0) while some re-entry draws remain in
    the full pool (positive-anchor persons).
    """
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    runner = _import_runner()
    import run_gate1_candidate8 as c8

    train = _synthetic_train()
    rng = np.random.default_rng(3)
    periods = list(range(1998, 2023, 2))
    hrows = []
    for pid in range(1001, 1017):
        span = int(rng.integers(6, 11))
        start = int(rng.integers(0, 13 - span + 1))
        perm = rng.normal(10.6, 0.5)
        for k in range(span):
            p = periods[start + k]
            age = min(59, 30 + (start + k) * 2)
            earn = float(np.exp(perm + rng.normal(0, 0.3)))
            hrows.append((pid, p, earn, age, 1.0))
        # Force an interior zero (a re-entry step) and a zero anchor for some.
        if pid % 2 == 0 and len(hrows) >= 3:
            mid = hrows[-2]
            hrows[-2] = (mid[0], mid[1], 0.0, mid[3], mid[4])
        if pid % 3 == 0:
            last = hrows[-1]
            hrows[-1] = (last[0], last[1], 0.0, last[3], last[4])
    holdout = pd.DataFrame(
        hrows, columns=["person_id", "period", "earnings", "age", "weight"]
    )
    panel = pd.concat([train, holdout], ignore_index=True)
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(train)
    uw = c8.build_donor_uw(train, marginals)
    pools = runner.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )

    orig = runner._gate_sign_draw
    runner._gate_sign_draw = lambda fitted, nl, age, u: np.ones(
        len(nl), dtype=int
    )
    try:
        # Both gates stubbed identically (always positive) so every step
        # generates and the zero-anchor branch is exercised.
        _, diag = runner.generate_candidate(
            holdout,
            all_anchor,
            marginals,
            object(),
            object(),
            pools,
            0.5,
            0,
        )
    finally:
        runner._gate_sign_draw = orig

    za = diag["zero_anchor_reentry"]
    assert diag["n_zero_anchor_holdout_persons"] > 0
    # Some re-entry draws routed through the zero-anchor-restricted pool.
    assert za["n_reentry_draws_q0"] > 0
    assert za["n_reentry_draws_q0"] <= za["n_reentry_draws_total"]
    assert 0.0 <= za["q0_share_of_reentry_draws"] <= 1.0


# --------------------------------------------------------------------------
# Always-runnable consistency tests (artifact + gates.yaml only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_rank_knn.v3"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen-spec issue-comment URL is carried in the artifact."""
    assert _artifact()["spec_registration"] == SPEC_URL
    # Base and u_w registrations recorded for provenance.
    assert "4896132094" in _artifact()["base_registration"]
    assert "4897723604" in _artifact()["uw_registration"]


def test_two_changes_declared():
    """Candidate 9 declares exactly the two registered changes."""
    model = _artifact()["model"]
    assert "calibration" in model
    # The only calibration is the train-side SMM for lambda.
    assert "SMM for lambda" in model["calibration"]
    assert "change_1_donor_blend" in model
    assert "change_2_zero_anchor_participation_regime" in model
    # Candidate 9 does NOT adopt candidate 8's attachment distance.
    c2 = model["change_2_zero_anchor_participation_regime"]
    assert "not_candidate_8_attachment" in c2
    # The blend replaces the third distance term.
    knn = model["knn"]
    assert "lambda*u_w" in knn["distance_pairs"]
    assert knn["k"] == 25
    assert knn["weights"] == {"w_next": 1.0, "w_next2": 0.5, "w_anchor": 0.25}


def test_lambda_grid_recorded_and_per_seed():
    """The frozen lambda grid and per-seed chosen lambda are recorded."""
    artifact = _artifact()
    grid = artifact["model"]["change_1_donor_blend"]["lambda_grid"]
    assert grid == [round(0.1 * i, 1) for i in range(11)]
    lam_by_seed = artifact["lambda_by_seed"]
    for s in artifact["per_seed"]:
        lam = s["lambda"]
        assert lam in grid
        assert lam_by_seed[str(s["seed"])] == lam
        # Each seed carries the SMM target/grid ladders.
        cal = s["lambda_calibration"]
        assert set(cal["target_autocorr"]) == {"lag1", "lag2", "lag5"}
        assert len(cal["grid_trace"]) == 11
        assert cal["lambda"] == lam
        # The chosen lambda is the strict-min-SSE grid point (smaller on tie).
        finite = [
            (g["lambda"], g["sse"])
            for g in cal["grid_trace"]
            if g["sse"] is not None
        ]
        best_lam, best_sse = min(finite, key=lambda t: (t[1], t[0]))
        assert best_lam == pytest.approx(lam)


def test_battery_reference_reproduced_exactly_in_artifact():
    """The stored reproduction block must attest exact float matches."""
    repro = _artifact()["battery_reference_reproduction"]
    assert repro["all_committed_values_reproduced_exactly"] is True
    committed = json.loads(
        (ROOT / _artifact()["battery_reference_run"]).read_text()
    )["battery_reference"]
    for name, chk in repro["checks"].items():
        assert chk["exact_float_match"] is True
        assert chk["committed"] == pytest.approx(committed[name], abs=0)
        assert chk["recomputed"] == pytest.approx(committed[name], abs=0)


def test_stored_thresholds_match_locked_gates_yaml():
    """Every stored geometry threshold equals the locked one."""
    artifact = _artifact()
    views_cfg = _gate1_thresholds()["views"]
    for seed in artifact["per_seed"]:
        for vname, view in seed["geometry"].items():
            stored = dict(view["thresholds"])
            for metric in views_cfg[vname].get("reported_not_gated", []):
                stored.pop(f"{metric}_max", None)
                stored.pop(f"{metric}_range", None)
                stored.pop(f"{metric}_min", None)
            assert stored == views_cfg[vname]["geometry"]

    battery_tol = {
        k[: -len("_tolerance")]: v
        for k, v in _gate1_thresholds()["battery"].items()
        if k.endswith("_tolerance")
    }
    for seed in artifact["per_seed"]:
        for stat, chk in seed["battery_checks"].items():
            assert chk["tolerance"] == pytest.approx(battery_tol[stat], abs=0)


def _recompute_geometry_pass(check: dict) -> bool:
    comp = check["comparison"]
    score = check["score"]
    thr = check["threshold"]
    if comp == "<=":
        return score <= thr
    if comp == ">=":
        return score >= thr
    if comp == "in":
        lo, hi = thr
        return (score >= lo) and (score <= hi)
    raise AssertionError(f"unknown comparison {comp!r}")


def test_every_geometry_threshold_pass_recomputes_from_stored_score():
    """Recompute each locked-geometry pass/fail from its stored score."""
    artifact = _artifact()
    for seed in artifact["per_seed"]:
        thresholds_pass = True
        for vname, view in seed["geometry"].items():
            view_pass = True
            for tname, chk in view["checks"].items():
                recomputed = _recompute_geometry_pass(chk)
                assert recomputed == chk["pass"], (
                    f"seed {seed['seed']} {vname}.{tname}: "
                    f"stored pass={chk['pass']} recomputed={recomputed}"
                )
                view_pass = view_pass and chk["pass"]
            assert view["view_pass"] == view_pass
            thresholds_pass = thresholds_pass and view_pass
        assert seed["geometry_thresholds_pass"] == thresholds_pass


def test_benefit_space_per_seed_recomputes_from_stored_values():
    """Recompute each per-seed benefit-space gate from its stored value."""
    artifact = _artifact()
    bmetrics = _gate1_thresholds()["benefit_space"]["metrics"]
    mean_thr = float(bmetrics["abs_mean_pct_diff_max"]["value"])
    med_thr = float(bmetrics["abs_median_pct_diff_max"]["value"])
    dec_thr = float(bmetrics["decile_pct_diff_max"]["value"])
    dec_gated = bmetrics["decile_pct_diff_max"]["deciles_gated"]
    ks_thr = float(bmetrics["weighted_ks_max"]["value"])
    for seed in artifact["per_seed"]:
        checks = seed.get("benefit_space_checks")
        if checks is None:
            pytest.skip("benefit-space block absent (SSA oracle unavailable)")
        # Each stored gate threshold equals the locked one.
        assert checks["abs_mean_pct_diff"]["threshold"] == pytest.approx(
            mean_thr, abs=0
        )
        assert checks["abs_median_pct_diff"]["threshold"] == pytest.approx(
            med_thr, abs=0
        )
        assert checks["weighted_ks"]["threshold"] == pytest.approx(
            ks_thr, abs=0
        )
        for dkey in dec_gated:
            assert checks[f"decile_{dkey}_pct_diff"][
                "threshold"
            ] == pytest.approx(dec_thr, abs=0)
        # Recompute each pass from the stored value.
        recomputed_all = True
        for name, chk in checks.items():
            val = chk["value"]
            thr = chk["threshold"]
            if val is None:
                recomputed = False
            elif chk["comparison"] == "|.| <=":
                recomputed = abs(float(val)) <= thr
            else:
                recomputed = float(val) <= thr
            assert recomputed == chk["pass"], (
                f"seed {seed['seed']} benefit_space {name}: "
                f"stored={chk['pass']} recomputed={recomputed}"
            )
            recomputed_all = recomputed_all and chk["pass"]
        assert seed["benefit_space_seed_pass"] == recomputed_all


def test_amended_geometry_verdict_conjoins_benefit_space():
    """Each seed's amended geometry verdict = thresholds AND benefit-space."""
    for seed in _artifact()["per_seed"]:
        bs_pass = seed.get("benefit_space_seed_pass")
        if bs_pass is None:
            # Without the benefit block the geometry verdict is thresholds
            # only (the amended gate is not evaluable; run() refuses to
            # publish in that case, so this branch is defensive).
            assert seed["geometry_pass"] == seed["geometry_thresholds_pass"]
        else:
            assert seed["geometry_pass"] == (
                seed["geometry_thresholds_pass"] and bs_pass
            )


def test_every_battery_pass_recomputes_from_stored_value():
    """Recompute each battery pass/fail from stored value/ref/tolerance."""
    artifact = _artifact()
    for seed in artifact["per_seed"]:
        seed_battery_pass = True
        for stat, chk in seed["battery_checks"].items():
            deviation = abs(chk["value"] - chk["reference"])
            assert deviation == pytest.approx(chk["deviation"], abs=1e-12)
            recomputed = deviation <= chk["tolerance"]
            assert recomputed == chk["pass"], (
                f"seed {seed['seed']} battery {stat}: "
                f"stored pass={chk['pass']} recomputed={recomputed}"
            )
            seed_battery_pass = seed_battery_pass and chk["pass"]
        assert seed["battery_pass"] == seed_battery_pass


def test_battery_references_match_committed_floor():
    """Every stored battery reference equals the committed floor value."""
    artifact = _artifact()
    committed = json.loads(
        (ROOT / artifact["battery_reference_run"]).read_text()
    )["battery_reference"]
    alias = {"mobility_diagonal": "mobility_diagonal_mean"}
    for seed in artifact["per_seed"]:
        for stat, chk in seed["battery_checks"].items():
            ref_key = alias.get(stat, stat)
            assert chk["reference"] == pytest.approx(committed[ref_key], abs=0)


def test_zero_persistence_identity_holds_in_every_seed():
    """The lock pins zero_persistence == 1 - exit_rate."""
    for seed in _artifact()["per_seed"]:
        zp = seed["battery_values"]["zero_persistence"]
        ex = seed["battery_values"]["exit_rate"]
        assert zp == pytest.approx(1.0 - ex, abs=1e-12)


def test_pooled_q0_gate_recomputes_from_per_seed():
    """The pooled Q0 gate recomputes from the per-seed Q0 means."""
    artifact = _artifact()
    pooled = artifact["benefit_space_gated"]["pooled_q0_gate"]
    thr = float(
        _gate1_thresholds()["benefit_space"]["metrics"][
            "abs_q0_mean_pct_diff_max"
        ]["value"]
    )
    if pooled["pooled_q0_mean_pct_diff"] is None:
        pytest.skip("pooled Q0 unavailable (SSA oracle absent)")
    assert pooled["threshold"] == pytest.approx(thr, abs=0)
    # Recompute the pooled mean from the per-seed Q0 means.
    vals = [
        r["q0_mean_pct_diff"]
        for r in pooled["per_seed_q0"]
        if r["q0_mean_pct_diff"] is not None
    ]
    recomputed = float(np.mean(vals)) if vals else None
    assert recomputed == pytest.approx(
        pooled["pooled_q0_mean_pct_diff"], abs=1e-9
    )
    assert pooled["pooled_q0_pass"] == (abs(recomputed) <= thr)
    # And each per-seed Q0 mean equals the benefit block's Q0 mean %.
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for r in pooled["per_seed_q0"]:
        s = by_seed[r["seed"]]
        if "benefit_space" not in s:
            continue
        q0 = s["benefit_space"]["by_anchor_quintile"]["quintiles"].get(
            "Q0", {}
        )
        stored = (
            q0["distribution"]["mean"]["pct_diff"]
            if q0.get("n_persons")
            else None
        )
        assert r["q0_mean_pct_diff"] == (
            pytest.approx(stored) if stored is not None else None
        )


def test_amended_verdict_recomputes_from_seed_conjunction():
    """The AMENDED gate verdict recomputes from the seed table + pooled Q0.

    Amended rule: gate passes iff >= 4/5 seeds pass geometry (locked
    thresholds AND per-seed benefit-space), >= 4/5 seeds pass battery, AND
    the pooled Q0 gate holds.
    """
    artifact = _artifact()
    table = artifact["seed_conjunction"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in table:
        assert row["geometry_pass"] == by_seed[row["seed"]]["geometry_pass"]
        assert row["battery_pass"] == by_seed[row["seed"]]["battery_pass"]
        assert row["lambda"] == by_seed[row["seed"]]["lambda"]

    n_geo = sum(1 for r in table if r["geometry_pass"])
    n_bat = sum(1 for r in table if r["battery_pass"])
    pooled_q0_pass = artifact["benefit_space_gated"]["pooled_q0_gate"][
        "pooled_q0_pass"
    ]
    verdict = artifact["verdict"]
    assert verdict["n_geometry_pass"] == n_geo
    assert verdict["n_battery_pass"] == n_bat
    assert verdict["geometry_gate_pass"] == (n_geo >= 4)
    assert verdict["battery_gate_pass"] == (n_bat >= 4)
    assert verdict["pooled_q0_pass"] == pooled_q0_pass
    assert verdict["gate_1_pass"] == (
        (n_geo >= 4) and (n_bat >= 4) and pooled_q0_pass
    )


def test_q0_participation_diagnostics_present():
    """The Q0 participation diagnostics (generated vs real) are reported."""
    q0p = _artifact()["q0_participation_diagnostics"]
    assert "per_seed" in q0p
    for row in q0p["per_seed"]:
        gen = row["generated"]
        real = row["real"]
        assert 0.0 <= gen["all_zero_share"] <= 1.0
        assert 0.0 <= real["all_zero_share"] <= 1.0
        assert gen["mean_positive_periods"] >= 0.0
        assert real["mean_positive_periods"] >= 0.0


def test_knn_diagnostics_reported_not_gated():
    """Reported-not-gated: usage shares, zero-anchor re-entry split, corners."""
    artifact = _artifact()
    for seed in artifact["knn_context"]["per_seed"]:
        usage = seed["triple_pair_usage"]
        assert usage["n_triple_draws"] >= 0
        assert usage["n_pair_draws"] >= 0
        assert usage["n_reentry_draws"] >= 0
        assert 0.0 <= usage["triple_share_of_positive"] <= 1.0

        za = seed["zero_anchor_reentry"]
        assert za["n_reentry_draws_q0"] <= za["n_reentry_draws_total"]
        assert 0.0 <= za["q0_share_of_reentry_draws"] <= 1.0

        for block in seed["drawn_corner_mass_by_anchor_quintile"].values():
            assert block["n"] >= 0
            assert 0.0 <= block["bottom_share"] <= 1.0
            assert 0.0 <= block["top_share"] <= 1.0

        nd = seed["neighbor_distance_distribution"]
        if nd:
            pcts = [nd[f"p{p}"] for p in (0, 10, 25, 50, 75, 90, 100)]
            assert all(v >= 0 for v in pcts)
            assert pcts == sorted(pcts), "neighbor-distance pcts not monotone"


def test_donor_reuse_reported_not_gated():
    """Reported-not-gated: donor-reuse record counts are non-negative."""
    for seed in _artifact()["knn_context"]["per_seed"]:
        reuse = seed["donor_reuse"]
        assert reuse["n_pair_records"] > 0
        assert reuse["n_triple_records"] > 0
        assert reuse["n_reentry_records"] > 0
        assert reuse["n_reentry_records_q0"] >= 0
        for key in (
            "pair_draws_per_record",
            "triple_draws_per_record",
            "reentry_draws_per_record",
        ):
            assert reuse[key] >= 0.0


def test_uw_fit_reported_per_seed():
    """Each seed carries the u_w decomposition (reported)."""
    for seed in _artifact()["knn_context"]["per_seed"]:
        fit = seed["uw_fit"]
        assert fit["sigma_hat_w"] >= 0.0
        assert 0.0 <= fit["implied_perm_share"] <= 1.0
        u = fit["u_w_distribution_positive_obs"]
        assert 0.0 <= u["min"] <= u["median"] <= u["max"] <= 1.0


def test_candidate_panel_pin_metadata_consistent():
    """Each seed's window counts are positive and pairs >= runs."""
    for seed in _artifact()["per_seed"]:
        assert seed["n_persons"] > 0
        assert seed["n_person_periods"] >= seed["n_persons"]
        assert set(seed["n_windows"]) == {
            "psid_family_earnings_pairs",
            "psid_family_earnings_runs",
        }
        for vname, n in seed["n_windows"].items():
            assert n > 0, f"{vname} has no windows"
        assert (
            seed["n_windows"]["psid_family_earnings_pairs"]
            >= seed["n_windows"]["psid_family_earnings_runs"]
        )
