"""Tests for the gate-1 candidate-8 permanent-rank donor-matching run.

Mirrors the candidate-7 tests (``test_gate1_qrf_candidate7``), adapted for
the two registered substitutions:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent, and ``importorskip("populace.fit")`` because the
  participation gate needs it) reruns seed 0 through the candidate-8 u_w
  decomposition + donor pools + backward k-NN generation and pins the
  committed artifact's seed-0 geometry and battery values to float
  precision. It is run live in the dedicated gate venv before the artifact
  is committed.
* :func:`test_donor_pools_and_uw_on_synthetic_pool` (always runnable; no
  PSID, no populace-fit) drives the u_w z-panel decomposition, the
  pair/triple/re-entry pool construction with the carried substitution
  fields, the Q0-restricted pools, and the pinned tie-break sort on a
  hand-built multi-period train panel.
* :func:`test_knn_draw_weighted_and_tie_break` (always runnable) pins the
  k-NN weighted single-record draw (imported byte-for-byte from candidate
  7) and its k nearest selection with the record-order tie-break.
* :func:`test_zero_anchor_distance_uses_attachment` (always runnable)
  checks that a zero-anchor target routes to the Q0-restricted pools with
  the attachment (age / observed-span) distance and never touches u_w,
  while a positive-anchor target uses the full pools with the u_w distance.
* The always-runnable consistency tests touch only the committed artifact
  and ``gates.yaml``: the schema and spec URL, the exact battery-reference
  reproduction, every reported pass/fail recomputes from its own stored
  score against its stored (locked) threshold, the zero-persistence
  identity, the verdict recomputation from the seed-conjunction table, the
  two-substitution constants, the u_w fit / Q0 pool sizes, and the
  reported-not-gated benefit-space block.
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
ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4897723604"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate8 as runner

    return runner


def _synthetic_train() -> pd.DataFrame:
    """A hand-built train panel spanning the biennial reference years.

    Enough persons observed across enough of the 13 biennial periods
    (1998-2022) that every within-person lag k = 1..5 has pairs, so the u_w
    z-panel decomposition (candidate 3's stage-1 grid-rho NNLS) has finite
    moments to fit. Includes zero-anchor persons (positive history, zero at
    their last observed period) so the Q0-restricted pools are non-empty.
    """
    rng = np.random.default_rng(7)
    periods = list(range(1998, 2023, 2))  # 13 biennial periods
    rows = []
    for pid in range(1, 81):
        start = int(rng.integers(0, 7))
        span = int(rng.integers(6, 13 - start + 1))
        perm = rng.normal(10.6, 0.5)
        for k in range(span):
            p = periods[start + k]
            age = min(59, 30 + (start + k) * 2 + int(rng.integers(0, 3)))
            earn = float(np.exp(perm + rng.normal(0, 0.3)))
            rows.append((pid, p, earn, age, 1.0 + (pid % 3)))
        if pid % 6 == 0:  # zero-anchor persons
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

    artifact = _artifact()
    seed0 = next(s for s in artifact["per_seed"] if s["seed"] == 0)
    thresholds = runner.load_gate1_thresholds()
    views_cfg = thresholds["views"]
    battery_reference = json.loads(
        (runner.ROOT / runner.BATTERY_REFERENCE_RUN).read_text()
    )["battery_reference"]
    panel = runner.load_filtered_panel()
    all_anchor = runner.anchor_rows(panel)
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
    # Reproduce the GATED path (geometry + battery) without the benefit-space
    # oracle; the gate verdict never depends on the SSA oracle being present.
    result = runner.run_seed(
        0,
        panel,
        all_anchor,
        view_specs,
        views_cfg,
        battery_reference,
        battery_tol,
        None,
        None,
        False,
    )
    for view, block in seed0["geometry"].items():
        for metric, stored in block["scores"].items():
            assert result["geometry"][view]["scores"][metric] == pytest.approx(
                stored, abs=1e-12
            ), f"{view}.{metric}"
    for stat, stored in seed0["battery_values"].items():
        assert result["battery_values"][stat] == pytest.approx(
            stored, abs=1e-12
        ), stat
    # Deterministic given the split: full and Q0-restricted pool sizes and
    # the u_w decomposition reproduce exactly.
    for key in (
        "n_pairs",
        "n_triples",
        "n_reentry",
        "n_pairs_q0",
        "n_triples_q0",
        "n_reentry_q0",
    ):
        assert result["pools"][key] == seed0["pools"][key], key
    assert result["uw_fit"]["rho"] == pytest.approx(
        seed0["uw_fit"]["rho"], abs=0
    )
    assert result["uw_fit"]["sigma_hat_w"] == pytest.approx(
        seed0["uw_fit"]["sigma_hat_w"], abs=1e-12
    )


# --------------------------------------------------------------------------
# Donor pools + u_w on a hand-built panel (always runnable; no populace-fit)
# --------------------------------------------------------------------------
def test_donor_pools_and_uw_on_synthetic_pool():
    """u_w decomposition, pool construction, and the Q0 restriction.

    The u_w decomposition reuses candidate 3's stage-1 machinery, which the
    runner imports lazily (candidate 3 -> candidate 2 -> populace.fit at top
    level); so exercising ``build_donor_uw`` needs populace importable.
    ``importorskip`` keeps this green under the repo .venv (no populace) and
    runs it live in the dedicated gate venv.
    """
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    runner = _import_runner()
    panel = _synthetic_train()
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(panel)

    uw = runner.build_donor_uw(panel, marginals)
    # The z-panel decomposition produced finite variances and a rho on the
    # locked candidate-3 grid.
    fit = uw["fit"]
    assert fit["sigma2_perm"] >= 0.0
    assert fit["sigma_hat_w"] == pytest.approx(
        float(np.sqrt(fit["sigma2_perm"])), abs=1e-12
    )
    # rho sits on candidate 3's frozen grid (carried in the fit block).
    assert fit["rho_grid"][0] <= fit["rho"] <= fit["rho_grid"][-1]
    assert fit["gamma_lags_biennial"] == [0, 1, 2, 3, 4, 5]
    # u_w is a per-person rank in [0, 1] for every train person.
    for v in uw["u_w_of_person"].values():
        assert 0.0 <= v <= 1.0

    pools = runner.build_donor_pools(
        panel, all_anchor, marginals, uw["u_w_of_person"]
    )
    # Every record carries u_A, u_w, age_prev, n_obs, anchor_zero, weight.
    for name in ("pairs", "triples", "reentry"):
        p = pools[name]
        n = len(p["u_prev"])
        for field in (
            "u_A",
            "u_w",
            "age_prev",
            "n_obs",
            "anchor_zero",
            "weight",
        ):
            assert len(p[field]) == n, f"{name}.{field}"
        assert np.all(np.isfinite(p["u_w"]))
        assert np.all((p["u_w"] >= 0.0) & (p["u_w"] <= 1.0))
        # Pinned in ascending (person_id, period_prev) order.
        key = p["person_id"].astype(np.int64) * 100000 + p[
            "period_prev"
        ].astype(np.int64)
        assert np.all(np.diff(key) >= 0), f"{name} not in pinned order"

    # u_w is constant within a person (a person-level quantity).
    for name in ("pairs", "triples", "reentry"):
        p = pools[name]
        df = pd.DataFrame({"pid": p["person_id"], "uw": p["u_w"]})
        for _, g in df.groupby("pid"):
            assert g["uw"].nunique() == 1

    # n_obs equals the person's observed-period count in the panel.
    depth = panel.groupby("person_id")["period"].size()
    for pid, nobs in zip(
        pools["pairs"]["person_id"], pools["pairs"]["n_obs"], strict=True
    ):
        assert int(nobs) == int(depth[int(pid)])

    # Q0-restricted pools are the subset whose donor person's anchor is zero.
    zero_anchor_pids = set(
        int(p) for p in all_anchor[all_anchor.earnings == 0].person_id
    )
    for base, q0 in (
        ("pairs", "pairs_q0"),
        ("triples", "triples_q0"),
        ("reentry", "reentry_q0"),
    ):
        pq = pools[q0]
        assert np.all(pq["anchor_zero"])
        for pid in pq["person_id"]:
            assert int(pid) in zero_anchor_pids
        # Size recorded consistently.
        assert pools[f"n_{q0}"] == len(pq["u_prev"])
        # A subset of the full pool.
        assert len(pq["u_prev"]) <= pools[f"n_{base}"]
    # The synthetic panel has zero-anchor persons, so the Q0 re-entry pool is
    # non-empty (each zero-anchor person's last-period drop is a re-entry).
    assert pools["n_reentry_q0"] > 0


def test_knn_draw_weighted_and_tie_break():
    """The k-NN draw (candidate 7's, imported) takes k nearest, draws by weight."""
    runner = _import_runner()

    dist = np.array([[0.20, 0.40, 0.05, 0.30]], dtype=np.float64)
    weight = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    u_prev = np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float64)
    drawn, kth = runner._knn_draw(dist, weight, u_prev, np.array([0.0]))
    # Selected order by (distance, index): [2, 0, 3, 1]; u_draw=0 -> donor 2.
    assert float(drawn[0]) == pytest.approx(0.33, abs=0)
    assert float(kth[0]) == pytest.approx(0.40, abs=0)

    weight_w = np.array([1.0, 1.0, 100.0, 1.0], dtype=np.float64)
    drawn_w, _ = runner._knn_draw(dist, weight_w, u_prev, np.array([0.5]))
    assert float(drawn_w[0]) == pytest.approx(0.33, abs=0)

    dist_tie = np.array([[0.10, 0.10, 0.05, 0.30]], dtype=np.float64)
    drawn_t, _ = runner._knn_draw(dist_tie, weight, u_prev, np.array([0.34]))
    assert float(drawn_t[0]) == pytest.approx(0.11, abs=0)


def test_zero_anchor_distance_uses_attachment():
    """A zero-anchor target routes to the Q0 pools with the attachment distance.

    Drives :func:`run_gate1_candidate8.generate_candidate` on a small panel
    with a stubbed always-positive participation gate and checks, from the
    diagnostics, that (a) the zero-anchor holdout persons produce draws
    routed through the Q0-restricted pools, and (b) positive-anchor persons
    produce draws NOT routed through the Q0 pools (they used the u_w
    distance on the full pools). Also confirms the attachment-scale
    constants are the registered range widths.
    """
    runner = _import_runner()
    assert runner.ZERO_ANCHOR_AGE_SCALE == 40.0
    assert runner.ZERO_ANCHOR_NOBS_SCALE == 13.0

    # Building the pools calls build_donor_uw (candidate 3's decomposition,
    # imported lazily via candidate 2 -> populace.fit); skip without populace.
    pytest.importorskip(
        "populace",
        reason="u_w decomposition imports candidate 3 -> candidate 2 -> "
        "populace.fit",
    )
    train = _synthetic_train()
    # A holdout with BOTH a zero-anchor person and positive-anchor persons.
    rng = np.random.default_rng(3)
    periods = list(range(1998, 2023, 2))
    hrows = []
    for pid in range(1001, 1013):
        span = int(rng.integers(6, 11))
        start = int(rng.integers(0, 13 - span + 1))
        perm = rng.normal(10.6, 0.5)
        for k in range(span):
            p = periods[start + k]
            age = min(59, 30 + (start + k) * 2)
            earn = float(np.exp(perm + rng.normal(0, 0.3)))
            hrows.append((pid, p, earn, age, 1.0))
        if pid % 3 == 0:  # some zero-anchor holdout persons
            last = hrows[-1]
            hrows[-1] = (last[0], last[1], 0.0, last[3], last[4])
    holdout = pd.DataFrame(
        hrows, columns=["person_id", "period", "earnings", "age", "weight"]
    )
    panel = pd.concat([train, holdout], ignore_index=True)
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(train)
    uw = runner.build_donor_uw(train, marginals)
    pools = runner.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )

    # Stub the participation gate to always draw positive so every step
    # generates a magnitude (exercises every branch deterministically).
    orig = runner._gate_sign_draw
    runner._gate_sign_draw = lambda fitted, nl, age, u: np.ones(
        len(nl), dtype=int
    )
    try:
        _, diag = runner.generate_candidate(
            holdout, all_anchor, marginals, object(), pools, 0
        )
    finally:
        runner._gate_sign_draw = orig

    za = diag["zero_anchor_draw_split"]
    n_zero = diag["n_zero_anchor_holdout_persons"]
    assert n_zero > 0, "test panel should have zero-anchor holdout persons"
    # Zero-anchor persons produced draws through the Q0-restricted pools.
    assert za["n_positive_draws_q0"] > 0
    # ... but not ALL positive draws are Q0 (positive-anchor persons exist).
    assert za["n_positive_draws_q0"] < za["n_positive_draws_total"]
    assert 0.0 < za["q0_share_of_positive_draws"] < 1.0


# --------------------------------------------------------------------------
# Always-runnable consistency tests (artifact + gates.yaml only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_rank_knn.v2"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen-spec issue-comment URL is carried in the artifact."""
    assert _artifact()["spec_registration"] == SPEC_URL


def test_two_substitutions_declared():
    """Candidate 8 declares exactly the two registered substitutions."""
    artifact = _artifact()
    assert "calibration" in artifact["model"]
    assert artifact["model"]["calibration"].startswith("none")
    # No SMM (5b) / kernel (6); donor_pools + knn as candidate 7, plus the
    # two substitution blocks.
    assert "smm" not in artifact["model"]
    assert "kernel" not in artifact["model"]
    assert "donor_pools" in artifact["model"]
    assert "knn" in artifact["model"]
    assert "substitution_1_donor_permanent_rank" in artifact["model"]
    assert "substitution_2_zero_anchor_conditioning" in artifact["model"]
    # Base registration is candidate 7.
    assert "4896132094" in artifact["base_registration"]


def test_substitution_constants_recorded():
    """The frozen k, distance weights, and attachment scales are recorded."""
    model = _artifact()["model"]
    knn = model["knn"]
    assert knn["k"] == 25
    assert knn["weights"] == {"w_next": 1.0, "w_next2": 0.5, "w_anchor": 0.25}
    scales = model["substitution_2_zero_anchor_conditioning"]["scales"]
    assert scales["age"] == 40.0
    assert scales["n_observed_periods"] == 13.0


def test_uw_fit_and_q0_pools_reported_per_seed():
    """Each seed carries the u_w decomposition and the Q0 pool sizes."""
    for seed in _artifact()["knn_context"]["per_seed"]:
        fit = seed["uw_fit"]
        assert fit["sigma_hat_w"] >= 0.0
        assert 0.0 <= fit["implied_perm_share"] <= 1.0
        u = fit["u_w_distribution_positive_obs"]
        assert 0.0 <= u["min"] <= u["median"] <= u["max"] <= 1.0
        for key in ("n_pairs_q0", "n_triples_q0", "n_reentry_q0"):
            assert seed[key] >= 0
            assert seed[key] <= seed[key.replace("_q0", "")]


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
            # Ratified amendments may demote a metric after this run
            # published (gates.yaml amendment_history); the stored
            # thresholds remain the correct record of the gate AS RUN.
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


def test_every_geometry_pass_recomputes_from_stored_score():
    """Recompute each geometry pass/fail from its stored score+threshold."""
    artifact = _artifact()
    for seed in artifact["per_seed"]:
        seed_geometry_pass = True
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
            seed_geometry_pass = seed_geometry_pass and view_pass
        assert seed["geometry_pass"] == seed_geometry_pass


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


def test_verdict_recomputes_from_seed_conjunction():
    """The gate verdict recomputes from the seed-conjunction table."""
    artifact = _artifact()
    table = artifact["seed_conjunction"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in table:
        assert row["geometry_pass"] == by_seed[row["seed"]]["geometry_pass"]
        assert row["battery_pass"] == by_seed[row["seed"]]["battery_pass"]

    n_geo = sum(1 for r in table if r["geometry_pass"])
    n_bat = sum(1 for r in table if r["battery_pass"])
    verdict = artifact["verdict"]
    assert verdict["n_geometry_pass"] == n_geo
    assert verdict["n_battery_pass"] == n_bat
    assert verdict["geometry_gate_pass"] == (n_geo >= 4)
    assert verdict["battery_gate_pass"] == (n_bat >= 4)
    assert verdict["gate_1_pass"] == ((n_geo >= 4) and (n_bat >= 4))


def test_knn_diagnostics_reported_not_gated():
    """Reported-not-gated: usage shares, Q0 split, corner masses in range."""
    artifact = _artifact()
    for seed in artifact["knn_context"]["per_seed"]:
        usage = seed["triple_pair_usage"]
        assert usage["n_triple_draws"] >= 0
        assert usage["n_pair_draws"] >= 0
        assert usage["n_reentry_draws"] >= 0
        assert 0.0 <= usage["triple_share_of_positive"] <= 1.0

        za = seed["zero_anchor_draw_split"]
        assert za["n_positive_draws_q0"] <= za["n_positive_draws_total"]
        assert 0.0 <= za["q0_share_of_positive_draws"] <= 1.0

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
            "n_pair_records_q0",
            "n_triple_records_q0",
            "n_reentry_records_q0",
        ):
            assert reuse[key] >= 0
        for key in (
            "pair_draws_per_record",
            "triple_draws_per_record",
            "reentry_draws_per_record",
        ):
            assert reuse[key] >= 0.0


def test_benefit_space_reported_not_gated():
    """The reported-not-gated benefit-space block is present and well-formed.

    Reads no gate. When the SSA oracle was available at run time the block
    carries the pooled PIA-proxy distribution gaps, the person-level block,
    and the by-anchor-quintile concentration with Q0 called out; the
    candidate-7 reference numbers are recorded for comparison.
    """
    bs = _artifact()["benefit_space_reported_not_gated"]
    if not bs.get("available"):
        pytest.skip("benefit-space block absent (SSA oracle unavailable)")
    assert bs["reported_not_gated"] is True
    dist = bs["distribution"]
    for key in ("mean_pct_diff", "median_pct_diff", "ks_distance"):
        assert dist[key]["n_seeds"] >= 1
    # Q0 is present and its stats are finite numbers.
    q0 = bs["by_anchor_quintile"]["Q0"]
    assert q0["n_seeds_present"] >= 1
    assert isinstance(q0["mean_pct_diff"], float)
    # The candidate-7 comparison numbers are recorded.
    c7 = bs["candidate7_reference"]
    assert c7["Q0_mean_pct_diff"] == pytest.approx(9.295292319680367)


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
