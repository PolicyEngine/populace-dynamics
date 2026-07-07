"""Tests for the inner-validation harness and the candidate-10 design sweep.

The inner harness (:mod:`inner_validation`) mirrors the amended gate on INNER
splits carved from each outer seed's TRAIN complement; the sweep
(:mod:`run_inner_sweep`) scores V0 / V0-shared / V1 / V2 / V3 on it. This work
is REPORTED-NOT-GATED and touches NO outer holdout data. The tests split into:

* always-runnable pure-logic tests (no PSID, no populace-fit): the inner-seed
  offset and split-side convention, the margin arithmetic, the amended-gate
  verdict recomputation, the V2 persistent-latent posterior algebra, the
  variant catalog, and the Q0-exemption distance construction on a
  hand-built pool;
* a seed-0 harness reproduction pin (skipped without the staged PSID family
  files, and ``importorskip("populace.fit")`` for the participation gates)
  that reruns the sweep's seed-0 V0 fit + generation + scoring and pins it to
  the committed artifact to float precision, plus the harness's non-contact
  guarantee on real data;
* artifact-consistency tests (touch only the committed sweep artifact and
  ``gates.yaml``): the schema, the reported-not-gated / no-outer-contact
  flags, the amended-gate mirror, every ranking-table margin recomputed from
  the stored per-seed scorecards, and the inner-gate verdicts recomputed from
  the seed tables.
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
SCRIPTS = ROOT / "scripts"
ARTIFACT = ROOT / "runs" / "inner_sweep_v1.json"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _import_harness():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import inner_validation as iv

    return iv


def _import_sweep():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_inner_sweep as sw

    return sw


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _synthetic_panel(n_persons: int = 200, seed: int = 7) -> pd.DataFrame:
    """A hand-built biennial panel spanning the reference years 1998-2022.

    Enough persons across enough periods that the outer 0.2 split leaves a
    train complement large enough to carve a non-empty inner pair, every
    within-person lag has pairs, and there are zero-anchor persons (positive
    history, zero at their last observed period).
    """
    rng = np.random.default_rng(seed)
    periods = list(range(1998, 2023, 2))  # 13 biennial periods
    rows = []
    for pid in range(1, n_persons + 1):
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
# Pure-logic: the inner split convention (no PSID, no populace-fit)
# --------------------------------------------------------------------------
def test_inner_seed_offset():
    iv = _import_harness()
    assert iv.INNER_HOLDOUT_FRACTION == 0.25
    assert iv.INNER_SEED_OFFSET == 1000
    for s in range(5):
        assert iv.inner_seed(s) == 1000 + s


def test_inner_split_disjoint_and_covers_train_and_avoids_outer():
    """The inner pair is disjoint, covers TRAIN, and avoids the outer holdout.

    Runs on the synthetic panel (no PSID), so the whole non-contact guarantee
    is checked in CI. The assertions live inside inner_split; this test drives
    it on every seed and re-verifies from the returned frames.
    """
    iv = _import_harness()
    panel = _synthetic_panel()
    for s in range(5):
        inner_holdout, inner_train, outer_holdout = iv.inner_split(panel, s)
        ih = set(inner_holdout.person_id)
        it = set(inner_train.person_id)
        oh = set(outer_holdout.person_id)
        # The outer holdout is the drawn 20%; TRAIN is the complement.
        assert ih | it == (set(panel.person_id) - oh)
        assert ih.isdisjoint(it)
        assert ih.isdisjoint(oh)
        assert it.isdisjoint(oh)
        # The inner holdout is the DRAWN 25% of TRAIN (the smaller side).
        assert len(ih) < len(it)


def test_inner_split_side_matches_split_panel_by_person():
    """inner_holdout is exactly split_panel_by_person's LEFT (drawn) side."""
    iv = _import_harness()
    from run_gate1_baseline import split_holdout_train

    from populace_dynamics.harness.panel import (
        split_panel_by_person,
    )

    panel = _synthetic_panel()
    _, train = split_holdout_train(panel, 2)
    left, right = split_panel_by_person(
        train, "person_id", fraction=0.25, seed=1002
    )
    inner_holdout, inner_train, _ = iv.inner_split(panel, 2)
    assert set(inner_holdout.person_id) == set(left.person_id)
    assert set(inner_train.person_id) == set(right.person_id)


# --------------------------------------------------------------------------
# Pure-logic: margin arithmetic and the amended-gate verdict
# --------------------------------------------------------------------------
def _fake_seed_result(
    *,
    pairs_c2st: float,
    ac10: float,
    q0_mean_pct: float,
    seed: int = 0,
) -> dict:
    """A minimal per-seed scorecard shaped like score_inner_pair's output.

    Carries just enough for the margin and verdict helpers: one gated pairs
    geometry check (C2ST), one battery check (10-year rung, whose reference
    and tolerance are the committed ones), and a benefit_space block with a Q0
    mean %. Everything passes by construction unless the passed values push a
    metric out of band.
    """
    c2st_pass = pairs_c2st <= 0.53
    ref10 = 0.538879427219318
    dev10 = abs(ac10 - ref10)
    bat_pass = dev10 <= 0.07
    return {
        "seed": seed,
        "geometry": {
            "psid_family_earnings_pairs": {
                "scores": {"c2st_auc": pairs_c2st},
                "checks": {
                    "c2st_auc_max": {
                        "metric": "c2st_auc",
                        "score": pairs_c2st,
                        "threshold": 0.53,
                        "comparison": "<=",
                        "pass": bool(c2st_pass),
                    }
                },
                "view_pass": bool(c2st_pass),
            },
            "psid_family_earnings_runs": {
                "scores": {"prdc_coverage": 0.95},
                "checks": {
                    "prdc_coverage_min": {
                        "metric": "prdc_coverage",
                        "score": 0.95,
                        "threshold": 0.90,
                        "comparison": ">=",
                        "pass": True,
                    }
                },
                "view_pass": True,
            },
        },
        "geometry_thresholds_pass": bool(c2st_pass),
        "geometry_pass": bool(c2st_pass),
        "battery_values": {"autocorr_log_10yr": ac10},
        "battery_checks": {
            "autocorr_log_10yr": {
                "value": ac10,
                "reference": ref10,
                "tolerance": 0.07,
                "deviation": dev10,
                "pass": bool(bat_pass),
            }
        },
        "battery_pass": bool(bat_pass),
        "benefit_space": {
            "by_anchor_quintile": {
                "quintiles": {
                    "Q0": {
                        "n_persons": 900,
                        "distribution": {"mean": {"pct_diff": q0_mean_pct}},
                    }
                }
            }
        },
        "benefit_space_seed_pass": True,
    }


def test_geometry_margin_sign_and_magnitude():
    """A passing max-metric has a positive margin = threshold - score."""
    iv = _import_harness()
    s = _fake_seed_result(pairs_c2st=0.52, ac10=0.54, q0_mean_pct=2.0)
    gm = iv.geometry_margins(s)
    key = "psid_family_earnings_pairs.c2st_auc_max"
    assert gm[key]["margin"] == pytest.approx(0.53 - 0.52, abs=1e-12)
    assert gm[key]["pass"] is True
    # A failing max-metric has a negative margin.
    s2 = _fake_seed_result(pairs_c2st=0.55, ac10=0.54, q0_mean_pct=2.0)
    gm2 = iv.geometry_margins(s2)
    assert gm2[key]["margin"] == pytest.approx(0.53 - 0.55, abs=1e-12)
    assert gm2[key]["pass"] is False


def test_battery_margin_is_tolerance_minus_deviation():
    iv = _import_harness()
    s = _fake_seed_result(pairs_c2st=0.52, ac10=0.50, q0_mean_pct=2.0)
    bm = iv.battery_margins(s)
    m = bm["autocorr_log_10yr"]
    assert m["margin"] == pytest.approx(0.07 - m["deviation"], abs=1e-12)


def test_inner_gate_verdict_matches_amended_rule():
    """>=4/5 geometry AND >=4/5 battery AND pooled Q0 <= 5 (abs)."""
    iv = _import_harness()
    metrics_cfg = _gate1_thresholds()["benefit_space"]["metrics"]
    # 5 seeds all passing geometry+battery, pooled Q0 = mean(2,2,2,2,2) = 2.
    per_seed = [
        _fake_seed_result(pairs_c2st=0.52, ac10=0.54, q0_mean_pct=2.0, seed=s)
        for s in range(5)
    ]
    v = iv.inner_gate_verdict(per_seed, metrics_cfg)
    assert v["n_geometry_pass"] == 5
    assert v["n_battery_pass"] == 5
    assert v["pooled_q0_mean_pct_diff"] == pytest.approx(2.0)
    assert v["pooled_q0_pass"] is True
    assert v["inner_gate_pass"] is True

    # One seed fails battery (4/5 still passes), pooled Q0 pushed to 6 -> fail.
    per_seed2 = [
        _fake_seed_result(pairs_c2st=0.52, ac10=0.54, q0_mean_pct=6.0, seed=s)
        for s in range(4)
    ] + [
        _fake_seed_result(pairs_c2st=0.52, ac10=0.20, q0_mean_pct=6.0, seed=4)
    ]
    v2 = iv.inner_gate_verdict(per_seed2, metrics_cfg)
    assert v2["n_battery_pass"] == 4
    assert v2["battery_gate_pass"] is True
    assert v2["pooled_q0_mean_pct_diff"] == pytest.approx(6.0)
    assert v2["pooled_q0_pass"] is False
    assert v2["inner_gate_pass"] is False  # pooled Q0 fails

    # Two seeds fail geometry -> geometry gate fails even if Q0/battery pass.
    per_seed3 = [
        _fake_seed_result(pairs_c2st=0.55, ac10=0.54, q0_mean_pct=2.0, seed=s)
        for s in range(2)
    ] + [
        _fake_seed_result(pairs_c2st=0.52, ac10=0.54, q0_mean_pct=2.0, seed=s)
        for s in range(2, 5)
    ]
    v3 = iv.inner_gate_verdict(per_seed3, metrics_cfg)
    assert v3["n_geometry_pass"] == 3
    assert v3["geometry_gate_pass"] is False
    assert v3["inner_gate_pass"] is False


# --------------------------------------------------------------------------
# Pure-logic: the V2 persistent-latent posterior algebra
# --------------------------------------------------------------------------
def test_draw_wcarry_posterior_mean_and_variance():
    """V2's w-draw matches the candidate-3 single-obs normal-normal posterior.

    posterior mean = (sigma2_perm / gamma_0) * (z_A - pooled_z_mean);
    posterior var  = sigma2_perm * (1 - sigma2_perm / gamma_0). Drawn once per
    person from N(mean, var). With a fixed RNG the draw equals mean + sd *
    standard_normal, so we reconstruct it exactly.
    """
    sw = _import_sweep()
    from scipy.stats import norm

    s2p, s2t, s2n = 0.5, 0.3, 0.2
    gamma_0 = s2p + s2t + s2n
    pooled = 0.1
    uw_fit = {
        "sigma2_perm": s2p,
        "sigma2_trans": s2t,
        "sigma2_noise": s2n,
    }
    ids = np.array([3, 1, 2], dtype=np.int64)
    # Anchor ranks -> z_A = Phi^-1(rank).
    anchor_rank = {1: 0.3, 2: 0.6, 3: 0.8}
    rng = np.random.default_rng(123)
    got = sw.draw_wcarry(ids, anchor_rank, uw_fit, pooled, rng)

    # Reconstruct: the function sorts ids, builds z_A - pooled per sorted id,
    # mean = shrink*z, draw = mean + sd*standard_normal(size).
    shrink = s2p / gamma_0
    post_sd = float(np.sqrt(s2p * (1.0 - shrink)))
    ids_sorted = np.sort(ids)
    z = np.array([norm.ppf(anchor_rank[int(p)]) - pooled for p in ids_sorted])
    rng2 = np.random.default_rng(123)
    noise = rng2.standard_normal(ids_sorted.size)
    expect = {
        int(p): float(shrink * z[i] + post_sd * noise[i])
        for i, p in enumerate(ids_sorted)
    }
    assert got == pytest.approx(expect, abs=1e-12)


def test_draw_wcarry_degenerate_permanent_variance():
    """Zero permanent variance -> zero shrink and zero posterior sd."""
    sw = _import_sweep()

    uw_fit = {"sigma2_perm": 0.0, "sigma2_trans": 0.5, "sigma2_noise": 0.5}
    ids = np.array([1, 2], dtype=np.int64)
    anchor_rank = {1: 0.4, 2: 0.7}
    rng = np.random.default_rng(1)
    got = sw.draw_wcarry(ids, anchor_rank, uw_fit, 0.0, rng)
    # shrink=0, post_sd=0 -> every w is exactly 0.
    assert all(v == pytest.approx(0.0, abs=0) for v in got.values())


# --------------------------------------------------------------------------
# Pure-logic: the variant catalog and RNG stream separation
# --------------------------------------------------------------------------
def test_variant_catalog_is_the_banked_findings_base():
    """Every variant keeps the zero-anchor refit except V0-shared."""
    sw = _import_sweep()
    catalog = sw.variant_catalog(sw.V1_LAMBDAS)
    names = [v["name"] for v in catalog]
    assert names == ["V0", "V0-shared", "V1-lam0.1", "V1-lam0.2", "V2", "V3"]
    by = {v["name"]: v for v in catalog}
    # V0-shared is the ONLY one without the zero-anchor refit.
    assert by["V0-shared"]["use_zero_anchor_gate"] is False
    for n in ("V0", "V1-lam0.1", "V1-lam0.2", "V2", "V3"):
        assert by[n]["use_zero_anchor_gate"] is True
    # Memory modes.
    assert by["V0"]["memory_mode"] == "none"
    assert by["V0-shared"]["memory_mode"] == "none"
    assert by["V1-lam0.1"]["memory_mode"] == "lambda_blend"
    assert by["V1-lam0.1"]["lambda"] == 0.1
    assert by["V1-lam0.2"]["lambda"] == 0.2
    assert by["V2"]["memory_mode"] == "wcarry"
    assert by["V3"]["memory_mode"] == "running_mean"


def test_substream_labels_are_reproducible_and_distinct():
    """Each (variant, label) draws its own reproducible stream."""
    sw = _import_sweep()
    a = sw._substream(0, sw.VARIANT_CODES["V0"], "gate").random(5)
    b = sw._substream(0, sw.VARIANT_CODES["V0"], "gate").random(5)
    assert np.allclose(a, b)  # reproducible
    c = sw._substream(0, sw.VARIANT_CODES["V1-lam0.1"], "gate").random(5)
    assert not np.allclose(a, c)  # distinct variant -> distinct stream
    d = sw._substream(0, sw.VARIANT_CODES["V0"], "donor-draw").random(5)
    assert not np.allclose(a, d)  # distinct label -> distinct stream


# --------------------------------------------------------------------------
# Pure-logic: the Q0 exemption in the transition distance
# --------------------------------------------------------------------------
def test_q0_exemption_in_transition_draw():
    """Q0 rows match on |u_A - a|; non-Q0 rows match on |d_third - third|.

    Drives :func:`run_inner_sweep._transition_draw` on a two-row query (one
    Q0, one non-Q0) against a tiny donor pool where the anchor coordinate and
    the per-variant third coordinate point at DIFFERENT donors, so the drawn
    ``u_prev`` reveals which coordinate each row used.
    """
    sw = _import_sweep()

    # A pool of 50 donors in TWO clusters of 25 (>= k = 25). The k-NN draw
    # takes the k = 25 nearest, so a query whose distance strongly prefers one
    # cluster draws entirely from that cluster; giving each cluster a single
    # shared u_prev makes the draw deterministic regardless of the weighted
    # pick. u_next is a constant (first distance term cancels).
    n = 25
    d_u_next = np.full(2 * n, 0.5)
    # Cluster 0 (donors 0..24): third=0.9, u_A=0.1, u_prev=0.11.
    # Cluster 1 (donors 25..49): third=0.1, u_A=0.9, u_prev=0.99.
    d_third = np.concatenate([np.full(n, 0.9), np.full(n, 0.1)])
    d_u_A = np.concatenate([np.full(n, 0.1), np.full(n, 0.9)])
    d_w = np.ones(2 * n)
    d_u_prev = np.concatenate([np.full(n, 0.11), np.full(n, 0.99)])

    # Row 0: non-Q0, third_target=0.9 -> the k nearest by |third - 0.9| are
    #   cluster 0 -> u_prev 0.11 (matched on the per-variant third coordinate).
    # Row 1: Q0, a=0.9 -> the k nearest by |u_A - 0.9| are cluster 1 ->
    #   u_prev 0.99 (matched on the ANCHOR coordinate; memory-exempt).
    idx = np.array([0, 1])
    za_local = np.array([False, True])  # row 1 is Q0
    v1 = np.array([0.5, 0.5])
    third_target = np.array([0.9, 0.0])  # row1's third_target unused (Q0)
    a_local = np.array([0.0, 0.9])  # row0's a unused (non-Q0 uses third)
    u_prev_local = np.zeros(2)
    rng = np.random.default_rng(0)

    sw._transition_draw(
        idx,
        za_local,
        v1,
        None,
        third_target,
        a_local,
        None,
        d_u_next,
        None,
        d_third,
        d_u_A,
        d_w,
        d_u_prev,
        "lambda_blend",
        rng,
        u_prev_local,
        [],
        triple=False,
    )
    # Row 0 (non-Q0) matched the third coordinate 0.9 -> cluster 0 -> 0.11.
    assert u_prev_local[0] == pytest.approx(0.11, abs=0)
    # Row 1 (Q0) matched the anchor coordinate 0.9 -> cluster 1 -> 0.99
    # (the memory exemption: Q0 uses |u_A - a|, NOT the per-variant third).
    assert u_prev_local[1] == pytest.approx(0.99, abs=0)


def test_running_mean_adds_fourth_axis_only_for_nonq0():
    """V3's fourth axis (u_next vs running mean) shifts only the non-Q0 draw.

    Two donor clusters of 25 (>= k). The first three distance terms are made
    a constant across both clusters (identical u_next, identical anchor), so
    ONLY V3's fourth axis (|u_next(donor) - running_mean|) can separate them --
    and it applies to the non-Q0 row but NOT the Q0 row (exempt). The clusters
    carry distinct u_prev so the draw reveals which cluster won.
    """
    sw = _import_sweep()

    n = 25
    # Cluster 0: u_next=0.2, u_prev=0.10. Cluster 1: u_next=0.8, u_prev=0.90.
    d_u_next = np.concatenate([np.full(n, 0.2), np.full(n, 0.8)])
    d_u_A = np.full(2 * n, 0.5)  # anchor identical -> third term constant
    d_w = np.ones(2 * n)
    d_u_prev = np.concatenate([np.full(n, 0.10), np.full(n, 0.90)])

    idx = np.array([0, 1])
    za_local = np.array([False, True])
    # Non-Q0 row: v1=0.5 is equidistant from both clusters' u_next, so its
    #   winner is decided by the fourth axis alone.
    # Q0 row: v1=0.2 matches cluster 0's u_next EXACTLY (first term), so it
    #   picks cluster 0 WITHOUT any fourth axis -- and its run_target=0.8
    #   (which would pull to cluster 1) must be ignored.
    v1 = np.array([0.5, 0.2])
    third_target = np.array([0.5, 0.5])
    a_local = np.array([0.5, 0.5])
    run_target = np.array([0.2, 0.8])
    u_prev_local = np.zeros(2)
    rng = np.random.default_rng(0)

    sw._transition_draw(
        idx,
        za_local,
        v1,
        None,
        third_target,
        a_local,
        run_target,
        d_u_next,
        None,
        d_u_A,  # d_third == d_u_A for V3 (third term unchanged)
        d_u_A,
        d_w,
        d_u_prev,
        "running_mean",
        rng,
        u_prev_local,
        [],
        triple=False,
    )
    # Non-Q0 row: |u_next - v1| ties both clusters; the fourth axis
    # |u_next - 0.2| favours cluster 0 -> u_prev 0.10.
    assert u_prev_local[0] == pytest.approx(0.10, abs=0)
    # Q0 row: NO fourth axis. |u_next - v1| ties both clusters and the anchor
    # term is constant, so the k = 25 nearest are cluster 0 by donor order
    # (lexsort tie-break on donor index) -> u_prev 0.10. Critically it did NOT
    # use run_target=0.8 (which would have pulled it to cluster 1's 0.90).
    assert u_prev_local[1] == pytest.approx(0.10, abs=0)


# --------------------------------------------------------------------------
# Seed-0 harness reproduction (needs PSID family files AND populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_v0_reproduces_committed_scorecard():
    """Rerun seed-0 V0 through the sweep and match the artifact to precision.

    The sweep is deterministic given the inner split and the outer seed, so
    seed-0 V0's inner-holdout geometry scores, battery values, and Q0 benefit
    mean reproduce exactly. Run live in the dedicated gate venv before the
    artifact is committed.
    """
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (the sweep uses a dedicated venv)",
    )
    if not ARTIFACT.exists():
        pytest.skip("inner_sweep_v1.json not committed yet")
    iv = _import_harness()
    sw = _import_sweep()
    from run_gate1_baseline import load_filtered_panel
    from run_gate1_candidate5b import anchor_rows

    artifact = _artifact()
    v0_seed0 = next(
        s for s in artifact["per_variant_per_seed"]["V0"] if s["seed"] == 0
    )

    gate_cfg = iv.load_amended_gate_config()
    view_specs = iv.build_inner_view_specs()
    panel = load_filtered_panel()
    full_anchor = anchor_rows(panel)
    params, _ = sw._load_benefit_oracle()
    assert params is not None, "SSA oracle must load for the reproduction pin"

    inner_holdout, inner_train, _ = iv.inner_split(panel, 0)
    train_persons = set(
        inner_holdout.person_id.tolist() + inner_train.person_id.tolist()
    )
    train_anchor = full_anchor[
        full_anchor.person_id.isin(train_persons)
    ].reset_index(drop=True)
    cutpoints = iv.inner_anchor_cutpoints(train_anchor)

    fit = sw.fit_inner_seed(inner_train, full_anchor, 0)
    candidate, _ = sw.generate_variant(
        inner_holdout,
        full_anchor,
        fit["marginals"],
        fit["fitted_shared"],
        fit["fitted_zero"],
        fit["pools"],
        "none",
        0,
        sw.VARIANT_CODES["V0"],
        use_zero_anchor_gate=True,
    )
    scorecard = iv.score_inner_pair(
        0,
        inner_holdout,
        candidate,
        train_anchor,
        gate_cfg,
        view_specs,
        params,
        cutpoints,
    )

    for view, block in v0_seed0["geometry"].items():
        for metric, stored in block["scores"].items():
            assert scorecard["geometry"][view]["scores"][
                metric
            ] == pytest.approx(stored, abs=1e-9), f"{view}.{metric}"
    for stat, stored in v0_seed0["battery_values"].items():
        assert scorecard["battery_values"][stat] == pytest.approx(
            stored, abs=1e-9
        ), stat
    q0_stored = v0_seed0["benefit_space"]["by_anchor_quintile"]["quintiles"][
        "Q0"
    ]["distribution"]["mean"]["pct_diff"]
    q0_got = scorecard["benefit_space"]["by_anchor_quintile"]["quintiles"][
        "Q0"
    ]["distribution"]["mean"]["pct_diff"]
    assert q0_got == pytest.approx(q0_stored, abs=1e-9)


@needs_real_family
def test_inner_split_never_touches_outer_holdout_on_real_data():
    """The non-contact guarantee holds on the real filtered panel too."""
    iv = _import_harness()
    from run_gate1_baseline import load_filtered_panel

    panel = load_filtered_panel()
    for s in iv.SEEDS:
        inner_holdout, inner_train, outer_holdout = iv.inner_split(panel, s)
        oh = set(outer_holdout.person_id)
        assert set(inner_holdout.person_id).isdisjoint(oh)
        assert set(inner_train.person_id).isdisjoint(oh)


# --------------------------------------------------------------------------
# Artifact consistency (committed sweep artifact + gates.yaml only)
# --------------------------------------------------------------------------
def _require_artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("inner_sweep_v1.json not committed yet")
    return _artifact()


def test_artifact_schema_and_reported_not_gated():
    artifact = _require_artifact()
    assert artifact["schema_version"] == "inner_sweep.v1"
    assert artifact["reported_not_gated"] is True
    assert "no_outer_holdout_contact" in artifact
    assert "inner_scale_caveat" in artifact
    assert "fraction=0.25" in artifact["inner_split"]["inner_split"]
    assert "1000+s" in artifact["inner_split"]["inner_split"]


def test_artifact_variants_are_the_five_designs():
    artifact = _require_artifact()
    names = set(artifact["variants"])
    assert names == {"V0", "V0-shared", "V1-lam0.1", "V1-lam0.2", "V2", "V3"}
    # V0-shared is the refit-off control.
    assert artifact["variants"]["V0-shared"]["use_zero_anchor_gate"] is False
    assert artifact["variants"]["V0"]["use_zero_anchor_gate"] is True


def test_ranking_verdicts_recompute_from_per_seed_scorecards():
    """Each ranking row's pass-counts recompute from the stored scorecards."""
    artifact = _require_artifact()
    for row in artifact["ranking"]["table"]:
        name = row["variant"]
        per_seed = artifact["per_variant_per_seed"][name]
        n_geo = sum(1 for s in per_seed if s["geometry_pass"])
        n_bat = sum(1 for s in per_seed if s["battery_pass"])
        assert row["n_geometry_pass"] == n_geo
        assert row["n_battery_pass"] == n_bat
        assert row["geometry_gate_pass"] == (n_geo >= 4)
        assert row["battery_gate_pass"] == (n_bat >= 4)
        # clears_all iff geometry AND battery AND pooled Q0 all pass.
        assert row["clears_all_on_ge4_seeds"] == (
            (n_geo >= 4) and (n_bat >= 4) and row["pooled_q0_pass"]
        )


def test_pooled_q0_recomputes_from_per_seed_q0_means():
    """Each variant's pooled Q0 = mean of the per-seed Q0 means."""
    artifact = _require_artifact()
    thr = float(
        _gate1_thresholds()["benefit_space"]["metrics"][
            "abs_q0_mean_pct_diff_max"
        ]["value"]
    )
    for name, rep in artifact["variants"].items():
        verdict = rep["inner_gate_verdict"]
        per_seed = artifact["per_variant_per_seed"][name]
        q0s = []
        for s in per_seed:
            bs = s.get("benefit_space")
            if bs is None:
                continue
            q0 = bs["by_anchor_quintile"]["quintiles"].get("Q0", {})
            if q0.get("n_persons", 0) > 0:
                q0s.append(q0["distribution"]["mean"]["pct_diff"])
        pooled = float(np.mean(q0s)) if q0s else None
        if pooled is None:
            assert verdict["pooled_q0_mean_pct_diff"] is None
        else:
            assert verdict["pooled_q0_mean_pct_diff"] == pytest.approx(
                pooled, abs=1e-9
            )
            assert verdict["pooled_q0_pass"] == (abs(pooled) <= thr)


def test_focal_margins_recompute_from_stored_values():
    """The ranking's focal margins recompute from the stored scorecards.

    Pairs C2ST min_margin = min over seeds of (0.53 - pairs_c2st_score); the
    10-year rung min_margin = min over seeds of (0.07 - |ac10 - ref|).
    """
    artifact = _require_artifact()
    ref10 = 0.538879427219318
    for row in artifact["ranking"]["table"]:
        name = row["variant"]
        per_seed = artifact["per_variant_per_seed"][name]
        c2sts = [
            s["geometry"]["psid_family_earnings_pairs"]["scores"]["c2st_auc"]
            for s in per_seed
        ]
        pc_min = min(0.53 - c for c in c2sts)
        assert row["focal_margins"]["pairs_c2st"][
            "min_margin"
        ] == pytest.approx(pc_min, abs=1e-9)
        ac10s = [s["battery_values"]["autocorr_log_10yr"] for s in per_seed]
        ac_min = min(0.07 - abs(a - ref10) for a in ac10s)
        assert row["focal_margins"]["battery_10yr"][
            "min_margin"
        ] == pytest.approx(ac_min, abs=1e-9)


def test_refit_effect_isolates_v0_vs_v0shared():
    """The refit-effect block reports V0 vs V0-shared pooled Q0."""
    artifact = _require_artifact()
    eff = artifact["refit_effect_v0_vs_v0shared"]
    if not eff.get("available"):
        pytest.skip("refit-effect block unavailable")
    pq = eff["pooled_q0_mean_pct_diff"]
    if pq["V0"] is not None and pq["V0-shared"] is not None:
        assert pq["refit_effect"] == pytest.approx(
            pq["V0"] - pq["V0-shared"], abs=1e-9
        )


def test_every_variant_seed_has_both_views_and_no_outer_contact_flag():
    """Every variant x seed scorecard has both views' windows and Q0 exempt."""
    artifact = _require_artifact()
    for per_seed in artifact["per_variant_per_seed"].values():
        assert len(per_seed) == len(artifact["seeds"])
        for s in per_seed:
            assert set(s["n_windows"]) == {
                "psid_family_earnings_pairs",
                "psid_family_earnings_runs",
            }
            # pairs windows >= runs windows (window-2 vs window-3).
            assert (
                s["n_windows"]["psid_family_earnings_pairs"]
                >= s["n_windows"]["psid_family_earnings_runs"]
            )
            diag = s["generation_diagnostics"]
            assert diag["n_nonq0_memory_terms"] >= 0
            # The re-entry pool is the FULL pool (no restriction): the
            # diagnostic does not carry a q0-restricted re-entry count.
            assert "n_reentry_draws" in diag
