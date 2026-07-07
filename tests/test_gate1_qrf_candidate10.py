"""Tests for the gate-1 candidate-10 inner-validated-composition run.

Candidate 10 is the inner sweep's V1-lam0.1 variant at OUTER scale, scored
under the amended gate (the same conjunction run 11 / candidate 9 used): the
runs-view c2st is demoted to reported-not-gated, a gated benefit-space block
folds into the geometry verdict, and the gate needs >= 4/5 geometry AND >= 4/5
battery AND the pooled Q0 band. It differs from candidate 9 in three ways --
lambda is the FIXED constant 0.1 (no SMM calibration), the re-entry pool is
FULL (no zero-anchor restriction), and Q0 targets are memory-EXEMPT -- and the
tests target exactly those differences plus the amended scoring:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent, and ``importorskip("populace.fit")`` because the
  participation gates need it) reruns seed 0 through the candidate-10 u_w
  decomposition + donor pools + fixed-lambda blend + zero-anchor regime +
  backward k-NN generation and pins the committed artifact's seed-0 lambda,
  geometry, battery, and Q0 benefit values to float precision. It is run live
  in the dedicated gate venv before the artifact is committed.
* :func:`test_generation_matches_inner_sweep_v1lam01` (importorskip populace)
  proves candidate 10's ``generate_candidate`` is byte-for-byte the inner
  sweep's ``generate_variant`` at ``memory_mode="lambda_blend"``, ``lam=0.1``,
  ``use_zero_anchor_gate=True`` when both use candidate 7's two-element
  substream seeding -- the registration's "mirror its generation code paths
  exactly" claim, checked mechanically.
* :func:`test_q0_exempt_and_full_reentry_pool` (importorskip populace) checks
  the two removed poisons: Q0 targets never touch the blend (their draw is
  identical whether lambda is 0.1 or 0) and no draw is routed through the
  zero-anchor-restricted re-entry pool (n_reentry_draws_q0 == 0).
* :func:`test_donor_pools_and_blend_on_synthetic_pool` (importorskip populace)
  drives the pool construction with the carried ``u_w`` / ``anchor_zero``
  fields, the pinned tie-break sort, and the donor-blend coordinate.
* :func:`test_knn_draw_weighted_and_tie_break` (always runnable) pins the k-NN
  weighted single-record draw (imported byte-for-byte from candidate 7).
* The always-runnable consistency tests touch only the committed artifact and
  ``gates.yaml``: the schema and spec URL, the exact battery-reference
  reproduction, every reported geometry / battery / benefit-space pass
  recomputes from its own stored score against its stored (locked) threshold,
  the zero-persistence identity, the AMENDED verdict recomputation (geometry
  conjoins the per-seed benefit-space metrics; the gate conjoins the pooled Q0
  gate), the fixed-lambda record (every seed lambda == 0.1, no calibration
  block), the Q0-participation diagnostics, and the reported diagnostics.
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
ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v4.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4902561460"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate10 as runner

    return runner


def _synthetic_train() -> pd.DataFrame:
    """A hand-built train panel spanning the biennial reference years.

    Enough persons across enough of the 13 biennial periods (1998-2022) that
    every within-person lag k = 1..5 has pairs (so the u_w z-panel
    decomposition has finite moments), including zero-anchor persons (positive
    history, zero at their last observed period) so the zero-anchor gate and
    the anchor-zero pool slice are non-empty. Byte-for-byte the candidate-9
    test's synthetic panel.
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


def _synthetic_holdout() -> pd.DataFrame:
    """A hand-built holdout panel with interior zeros and zero anchors.

    Disjoint person ids from the train panel; some persons have an interior
    zero (a re-entry step) and some a zero anchor, so both the re-entry branch
    and the zero-anchor participation regime are exercised.
    """
    rng = np.random.default_rng(3)
    periods = list(range(1998, 2023, 2))
    hrows = []
    for pid in range(1001, 1041):
        span = int(rng.integers(6, 11))
        start = int(rng.integers(0, 13 - span + 1))
        perm = rng.normal(10.6, 0.5)
        for k in range(span):
            p = periods[start + k]
            age = min(59, 30 + (start + k) * 2)
            earn = float(np.exp(perm + rng.normal(0, 0.3)))
            hrows.append((pid, p, earn, age, 1.0))
        if pid % 2 == 0 and len(hrows) >= 3:
            mid = hrows[-2]
            hrows[-2] = (mid[0], mid[1], 0.0, mid[3], mid[4])
        if pid % 3 == 0:
            last = hrows[-1]
            hrows[-1] = (last[0], last[1], 0.0, last[3], last[4])
    return pd.DataFrame(
        hrows, columns=["person_id", "period", "earnings", "age", "weight"]
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
    # Lambda is the FIXED constant 0.1 (no calibration).
    assert result["lambda"] == 0.1
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
# Generation equivalence to the inner sweep's V1-lam0.1 (importorskip populace)
# --------------------------------------------------------------------------
def test_generation_matches_inner_sweep_v1lam01():
    """c10.generate_candidate == sweep V1-lam0.1 under candidate-7 seeding.

    The registration pins candidate 10 as the inner sweep's V1-lam0.1 variant
    at outer scale, mirroring its generation code paths exactly. Here we drive
    the inner sweep's ``generate_variant`` with candidate 7's two-element
    substream seeding (patching its ``_substream``) so the RNG draws match
    candidate 10's, and require the two candidate panels to be BYTE-identical
    -- proving the algorithmic path (the Q0-exempt non-Q0/Q0 split, the
    full-pool re-entry, the fixed draw order, the fixed-lambda blend) is the
    sweep's exactly, differing only in the substream derivation the
    registration allows.
    """
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    runner = _import_runner()
    import run_gate1_candidate8 as c8
    import run_gate1_candidate9 as c9
    import run_inner_sweep as sweep

    train = _synthetic_train()
    holdout = _synthetic_holdout()
    panel = pd.concat([train, holdout], ignore_index=True)
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(train)
    uw = c8.build_donor_uw(train, marginals)
    pools = c9.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )
    fitted_shared, _ = c9.fit_participation_gate(train, 0)
    fitted_zero, _ = c9.fit_zero_anchor_participation_gate(
        train, all_anchor, 0
    )

    # Stub the participation sign-draw identically on BOTH modules: on the
    # small synthetic panel the RegimeGatedQRF learns no zero gate, so
    # _gate_sign_draw's classifier is None. Stubbing it to always-positive on
    # both candidate 10 and the sweep exercises every generation branch and
    # keeps the two paths on the same participation law, so the comparison
    # isolates the DRAW logic (which is what "mirror the code paths" means).
    def _always_pos(fitted, nl, age, u):  # noqa: ANN001
        return np.ones(len(nl), dtype=int)

    orig_c10_gate = runner._gate_sign_draw
    orig_sw_gate = sweep._gate_sign_draw
    orig_ss = sweep._substream

    def _c7_seed(outer_seed, variant_code, label):  # noqa: ANN001
        code = sweep.SUBSTREAM_CODES[label]
        return np.random.default_rng(
            np.random.SeedSequence([int(outer_seed), code])
        )

    runner._gate_sign_draw = _always_pos
    try:
        cand_c10, _ = runner.generate_candidate(
            holdout,
            all_anchor,
            marginals,
            fitted_shared,
            fitted_zero,
            pools,
            0,
        )
    finally:
        runner._gate_sign_draw = orig_c10_gate

    sweep._substream = _c7_seed
    sweep._gate_sign_draw = _always_pos
    try:
        cand_sw, _ = sweep.generate_variant(
            holdout,
            all_anchor,
            marginals,
            fitted_shared,
            fitted_zero,
            pools,
            "lambda_blend",
            0,
            999,  # variant_code ignored by the patched substream
            lam=0.1,
            uw_of_person=uw["u_w_of_person"],
            uw_fit=uw["fit"],
            pooled_z_mean=uw["pooled_z_mean"],
            sigma_hat_w=uw["sigma_hat_w"],
            use_zero_anchor_gate=True,
        )
    finally:
        sweep._substream = orig_ss
        sweep._gate_sign_draw = orig_sw_gate

    a = cand_c10.sort_values(["person_id", "period"]).reset_index(drop=True)
    b = cand_sw.sort_values(["person_id", "period"]).reset_index(drop=True)
    assert a["person_id"].equals(b["person_id"])
    assert a["period"].equals(b["period"])
    assert np.array_equal(
        a["earnings"].to_numpy(), b["earnings"].to_numpy()
    ), "candidate 10 generation diverged from the inner sweep's V1-lam0.1"


def test_q0_exempt_and_full_reentry_pool():
    """Q0 targets are memory-exempt and no draw uses the restricted pool.

    Two checks of the removed poisons:

    * **Q0 exemption**: generating at lambda = 0.1 vs lambda = 0 must give the
      SAME earnings for every zero-anchor holdout person (their third distance
      term is the anchor rank in both cases; the blend never touches them),
      while at least one positive-anchor person differs (the blend does touch
      them). Comparison uses the shared candidate-7 substreams so only the
      lambda differs.
    * **Full re-entry pool**: the diagnostics report zero re-entry draws routed
      through the zero-anchor-restricted pool (candidate 10 has no restriction;
      the full pool serves every target).
    """
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    runner = _import_runner()
    import run_gate1_candidate8 as c8
    import run_gate1_candidate9 as c9

    train = _synthetic_train()
    holdout = _synthetic_holdout()
    panel = pd.concat([train, holdout], ignore_index=True)
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(train)
    uw = c8.build_donor_uw(train, marginals)
    pools = c9.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )
    fitted_shared, _ = c9.fit_participation_gate(train, 0)
    fitted_zero, _ = c9.fit_zero_anchor_participation_gate(
        train, all_anchor, 0
    )

    # Stub the participation sign-draw to always-positive (the synthetic panel
    # learns no zero gate) so both generations run every branch under the same
    # participation law and the only difference is lambda.
    def _always_pos(fitted, nl, age, u):  # noqa: ANN001
        return np.ones(len(nl), dtype=int)

    orig_gate = runner._gate_sign_draw
    runner._gate_sign_draw = _always_pos
    orig_lam = runner.LAMBDA_FIXED
    try:
        # lambda = 0.1 (candidate 10 as-is).
        cand_01, diag_01 = runner.generate_candidate(
            holdout,
            all_anchor,
            marginals,
            fitted_shared,
            fitted_zero,
            pools,
            0,
        )
        # lambda = 0 (temporarily override the fixed constant): all third
        # terms collapse to the anchor rank, so NO target sees the blend.
        runner.LAMBDA_FIXED = 0.0
        cand_00, _ = runner.generate_candidate(
            holdout,
            all_anchor,
            marginals,
            fitted_shared,
            fitted_zero,
            pools,
            0,
        )
    finally:
        runner.LAMBDA_FIXED = orig_lam
        runner._gate_sign_draw = orig_gate

    a = cand_01.sort_values(["person_id", "period"]).reset_index(drop=True)
    b = cand_00.sort_values(["person_id", "period"]).reset_index(drop=True)
    zero_ids = set(
        int(p) for p in all_anchor[all_anchor.earnings == 0].person_id
    )
    is_q0 = a["person_id"].isin(zero_ids).to_numpy()
    ea = a["earnings"].to_numpy()
    eb = b["earnings"].to_numpy()
    # Q0 persons: identical at lambda 0.1 and lambda 0 (memory-exempt).
    assert np.array_equal(
        ea[is_q0], eb[is_q0]
    ), "Q0 targets are not memory-exempt (lambda changed their earnings)"
    # Positive-anchor persons: at least one differs (the blend touches them).
    assert not np.array_equal(
        ea[~is_q0], eb[~is_q0]
    ), "the fixed-lambda blend had no effect on any non-Q0 target"

    # Full re-entry pool: no draw routed through the restricted pool.
    za = diag_01["zero_anchor_reentry"]
    assert za["n_reentry_draws_q0"] == 0
    assert diag_01["reentry_pool"] == "full (no zero-anchor restriction)"
    assert diag_01["q0_memory_exempt"] is True
    assert diag_01["lambda"] == 0.1


def test_donor_pools_and_blend_on_synthetic_pool():
    """u_w-carrying pools, the pinned tie-break, and the blend coordinate."""
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    runner = _import_runner()
    import run_gate1_candidate8 as c8
    import run_gate1_candidate9 as c9

    panel = _synthetic_train()
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(panel)
    uw = c8.build_donor_uw(panel, marginals)
    pools = c9.build_donor_pools(
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

    # The blend (imported from candidate 9): lambda=0 -> donor u_A; lambda=1 ->
    # donor u_w; and candidate 10's fixed lambda = 0.1 is the 0.9/0.1 mix.
    u_w = pools["pairs"]["u_w"]
    u_A = pools["pairs"]["u_A"]
    assert np.allclose(c9._donor_blend(u_w, u_A, 0.0), u_A)
    assert np.allclose(c9._donor_blend(u_w, u_A, 1.0), u_w)
    mid = c9._donor_blend(u_w, u_A, 0.1)
    assert np.allclose(mid, 0.1 * u_w + 0.9 * u_A)
    assert runner.LAMBDA_FIXED == 0.1


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


# --------------------------------------------------------------------------
# Always-runnable consistency tests (artifact + gates.yaml only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_rank_knn.v4"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen-spec issue-comment URL is carried in the artifact."""
    assert _artifact()["spec_registration"] == SPEC_URL
    # Base / u_w / candidate-9 / forecast registrations recorded for provenance.
    assert "4896132094" in _artifact()["base_registration"]
    assert "4897723604" in _artifact()["uw_registration"]
    assert "4898825218" in _artifact()["c9_registration"]
    assert "4902561584" in _artifact()["forecast_registration"]


def test_fixed_lambda_no_calibration():
    """Candidate 10 declares a FIXED lambda = 0.1 and NO calibration stage."""
    artifact = _artifact()
    model = artifact["model"]
    # The only "calibration" statement is that there is none in this run.
    assert "none in this run" in model["calibration"]
    assert "lambda_calibration" not in artifact["knn_context"]["per_seed"][0]
    # Every seed carries the fixed lambda 0.1.
    lam_by_seed = artifact["lambda_by_seed"]
    for s in artifact["per_seed"]:
        assert s["lambda"] == 0.1
        assert lam_by_seed[str(s["seed"])] == 0.1
        assert "lambda_calibration" not in s
    # The knn context declares the fixed lambda.
    assert artifact["knn_context"]["lambda"] == 0.1
    assert artifact["knn_context"]["lambda_fixed"] is True
    # The blend replaces the non-Q0 third distance term; Q0 stays anchor-only.
    knn = model["knn"]
    assert "0.1*u_w" in knn["distance_pairs_nonq0"]
    assert knn["distance_pairs_q0"] == "|u_next - v1| + 0.25|u_A - a|"
    assert knn["k"] == 25
    assert knn["weights"] == {"w_next": 1.0, "w_next2": 0.5, "w_anchor": 0.25}


def test_two_changes_declared():
    """Candidate 10 declares its two changes to candidate 9 (poisons removed)."""
    model = _artifact()["model"]
    assert "change_1_fixed_donor_blend" in model
    assert "change_2_zero_anchor_participation_regime" in model
    c1 = model["change_1_fixed_donor_blend"]
    assert c1["lambda"] == 0.1
    assert c1["lambda_fixed"] is True
    c2 = model["change_2_zero_anchor_participation_regime"]
    # Full re-entry pool (no restriction) and Q0 memory-exemption declared.
    assert "NO" in c2["reentry_pool"] and "restriction" in c2["reentry_pool"]
    assert "exempt" in c2["q0_memory_exempt"]
    # The inner-validation provenance is recorded.
    iv = model["inner_validation"]
    assert "V1-lam0.1" in iv["this_candidate_is"]
    assert iv["sweep_artifact"] == "runs/inner_sweep_v1.json"


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
            demoted = views_cfg[vname].get(
                "reported_not_gated", []
            ) + views_cfg[vname].get("per_seed_rule_superseded", [])
            for metric in demoted:
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
    thresholds AND per-seed benefit-space), >= 4/5 seeds pass battery, AND the
    pooled Q0 gate holds.
    """
    artifact = _artifact()
    table = artifact["seed_conjunction"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in table:
        assert row["geometry_pass"] == by_seed[row["seed"]]["geometry_pass"]
        assert row["battery_pass"] == by_seed[row["seed"]]["battery_pass"]
        assert row["lambda"] == by_seed[row["seed"]]["lambda"]
        assert row["lambda"] == 0.1

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
    """Reported-not-gated: usage shares, corners, neighbor distances."""
    artifact = _artifact()
    for seed in artifact["knn_context"]["per_seed"]:
        usage = seed["triple_pair_usage"]
        assert usage["n_triple_draws"] >= 0
        assert usage["n_pair_draws"] >= 0
        assert usage["n_reentry_draws"] >= 0
        assert 0.0 <= usage["triple_share_of_positive"] <= 1.0

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
