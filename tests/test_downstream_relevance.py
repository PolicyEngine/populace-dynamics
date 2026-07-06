"""Tests for the downstream-relevance (candidate 7 in benefit space) artifact.

Two tiers, mirroring the gate runs' test structure:

* Always-runnable internal-consistency tests (no PSID, no populace-fit) that
  touch only the committed artifact plus the pure helper functions in
  :mod:`scripts.build_downstream_relevance`: the schema is sane and marked
  reported-not-gated; every stored distribution gap RECOMPUTES from its own
  stored per-side statistics; every context verdict recomputes from its
  stored candidate/noise/criterion values; and the weighted-statistic
  helpers reproduce closed-form answers on hand-built inputs.
* A seed-0 reproduction pin (skipped when the PSID family files are absent,
  and ``importorskip('populace.fit')`` because the candidate-7 participation
  gate needs it) that reruns the seed-0 candidate and noise measurements
  through the build machinery and pins the committed artifact's seed-0
  numbers to float precision. It is run live in the dedicated gate venv
  before the artifact is committed.

The build script is a REPORTED-NOT-GATED downstream analysis: it reads no
gate and changes no gate. These tests assert that framing is recorded and
that the artifact's arithmetic is internally reproducible, which is the bar
its own numbers must clear before they could inform a future amendment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "downstream_relevance_c7_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_downstream_relevance as builder

    return builder


# --------------------------------------------------------------------------
# Schema and reported-not-gated framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "downstream_relevance_c7.v1"
    assert art["run"] == "downstream_relevance_c7_v1"
    # The load-bearing framing: reported, not gated, with a purpose line.
    assert art["reported_not_gated"] is True
    assert isinstance(art["purpose"], str) and art["purpose"].strip()
    assert "changes no gate" in art["purpose"]
    # The proxy-not-full-415(b) disclaimer must be present.
    assert "STATUTE-SHAPED PROXY" in art["not_full_415b"]
    assert "415(b)" in art["not_full_415b"]
    # Oracle revision is pinned.
    assert art["oracle"]["pe_us_revision"]
    assert art["revision_pins"]["pe_us_revision"] == (
        art["oracle"]["pe_us_revision"]
    )
    # Candidate-7 artifact referenced by content hash.
    ref = art["candidate7_reference"]
    assert ref["run"] == "gate1_rank_knn_v1"
    assert len(ref["artifact_sha256"]) == 64


def test_candidate7_reference_hash_matches_committed_artifact():
    """The referenced candidate-7 hash matches the committed artifact bytes."""
    import hashlib

    art = _artifact()
    committed = (ROOT / "runs" / "gate1_rank_knn_v1.json").read_bytes()
    assert (
        art["candidate7_reference"]["artifact_sha256"]
        == hashlib.sha256(committed).hexdigest()
    )


def test_success_criterion_recorded():
    art = _artifact()
    assert art["success_criterion"]["pct"] == 5.0
    assert "within 5 percent" in art["success_criterion"]["text"]
    assert art["context"]["summary"]["success_criterion_pct"] == 5.0


def test_seeds_present_candidate_and_noise():
    art = _artifact()
    seeds = [0, 1, 2, 3, 4]
    assert [s["seed"] for s in art["candidate_per_seed"]] == seeds
    assert [s["seed"] for s in art["noise_per_seed"]] == seeds
    # Noise rows are real-vs-real: person-level and by-quintile are n/a.
    for s in art["noise_per_seed"]:
        assert s["person_level"] == "n/a (disjoint persons)"
        assert s["by_anchor_quintile"] == "n/a (disjoint persons)"


# --------------------------------------------------------------------------
# Every stored gap recomputes from its stored per-side statistics
# --------------------------------------------------------------------------
def test_percent_diffs_recompute_from_stored_sides():
    """mean/median/decile % diffs recompute from stored candidate & real."""
    art = _artifact()

    def check_block(dist: dict) -> None:
        for key in ("mean", "median"):
            cand = dist[key]["candidate"]
            real = dist[key]["real"]
            stored = dist[key]["pct_diff"]
            if real == 0.0:
                assert stored is None
            else:
                assert stored == pytest.approx(
                    100.0 * (cand - real) / real, rel=1e-9, abs=1e-9
                ), key
        for dkey, cell in dist["deciles"].items():
            cand = cell["candidate"]
            real = cell["real"]
            stored = cell["pct_diff"]
            if real == 0.0:
                assert stored is None, dkey
            else:
                assert stored == pytest.approx(
                    100.0 * (cand - real) / real, rel=1e-9, abs=1e-9
                ), dkey

    for s in art["candidate_per_seed"]:
        check_block(s["distribution"])
    for s in art["noise_per_seed"]:
        check_block(s["distribution"])
    # Every by-quintile sub-block too.
    for s in art["candidate_per_seed"]:
        for qblk in s["by_anchor_quintile"]["quintiles"].values():
            if qblk.get("n_persons", 0) > 0:
                check_block(qblk["distribution"])


def test_gini_difference_recomputes():
    """Stored Gini difference equals candidate minus real Gini everywhere."""
    art = _artifact()
    for s in art["candidate_per_seed"] + art["noise_per_seed"]:
        g = s["distribution"]["gini"]
        assert g["difference"] == pytest.approx(
            g["candidate"] - g["real"], rel=1e-9, abs=1e-12
        )


def test_pooled_means_recompute_from_per_seed():
    """Pooled distribution/person-level means equal the per-seed averages."""
    art = _artifact()

    # Candidate pooled mean %-diff = mean of per-seed mean %-diffs.
    per_seed_means = [
        s["distribution"]["mean"]["pct_diff"]
        for s in art["candidate_per_seed"]
    ]
    assert art["candidate_pooled"]["distribution"]["mean_pct_diff"][
        "mean"
    ] == pytest.approx(float(np.mean(per_seed_means)), rel=1e-9)

    # Candidate pooled KS mean.
    per_seed_ks = [
        s["distribution"]["ks_distance"] for s in art["candidate_per_seed"]
    ]
    assert art["candidate_pooled"]["distribution"]["ks_distance"][
        "mean"
    ] == pytest.approx(float(np.mean(per_seed_ks)), rel=1e-9)

    # Candidate pooled person-level MAE mean.
    per_seed_mae = [
        s["person_level"]["weighted_mae"] for s in art["candidate_per_seed"]
    ]
    assert art["candidate_pooled"]["person_level"]["weighted_mae"][
        "mean"
    ] == pytest.approx(float(np.mean(per_seed_mae)), rel=1e-9)

    # Noise pooled median %-diff mean.
    per_seed_noise_med = [
        s["distribution"]["median"]["pct_diff"] for s in art["noise_per_seed"]
    ]
    assert art["noise_pooled"]["distribution"]["median_pct_diff"][
        "mean"
    ] == pytest.approx(float(np.mean(per_seed_noise_med)), rel=1e-9)


def test_context_verdicts_recompute():
    """Context within-noise and within-5% flags recompute from stored values."""
    art = _artifact()
    ctx = art["context"]
    crit = 5.0

    def check_place(row: dict) -> None:
        cand_abs = abs(row["candidate_pct"])
        noise_abs = abs(row["noise_pct"])
        assert row["candidate_abs_pct"] == pytest.approx(cand_abs, abs=1e-12)
        assert row["noise_abs_pct"] == pytest.approx(noise_abs, abs=1e-12)
        assert row["within_noise_anchor"] == (cand_abs <= noise_abs)
        assert row["within_5pct_criterion"] == (cand_abs <= crit)

    check_place(ctx["mean_pct_diff"])
    check_place(ctx["median_pct_diff"])
    for row in ctx["deciles"].values():
        check_place(row)

    # KS within-noise: candidate <= noise (a distance, not a percent).
    assert ctx["ks_distance"]["within_noise_anchor"] == (
        ctx["ks_distance"]["candidate"] <= ctx["ks_distance"]["noise"]
    )

    # Summary conjunctions recompute.
    pct_rows = [ctx["mean_pct_diff"], ctx["median_pct_diff"]] + list(
        ctx["deciles"].values()
    )
    dist_rows = pct_rows + [ctx["ks_distance"], ctx["gini_difference"]]
    assert ctx["summary"]["all_distribution_gaps_within_noise_anchor"] == all(
        r["within_noise_anchor"] for r in dist_rows
    )
    assert ctx["summary"]["all_percent_gaps_within_5pct_criterion"] == all(
        r["within_5pct_criterion"] for r in pct_rows
    )


def test_person_level_shares_in_range():
    """Weighted within-tolerance shares are valid probabilities, w5 <= w10."""
    art = _artifact()
    for s in art["candidate_per_seed"]:
        pl = s["person_level"]
        w5 = pl["weighted_share_within_5pct"]
        w10 = pl["weighted_share_within_10pct"]
        assert 0.0 <= w5 <= 1.0
        assert 0.0 <= w10 <= 1.0
        assert w5 <= w10 + 1e-12
        assert pl["weighted_mae"] >= 0.0
        assert pl["weighted_rmse"] >= pl["weighted_mae"] - 1e-9


# --------------------------------------------------------------------------
# Weighted-statistic helpers reproduce closed-form answers
# --------------------------------------------------------------------------
def test_weighted_quantile_matches_hazen_when_equal_weights():
    builder = _import_builder()
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    w = np.ones(50)
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    got = builder._weighted_quantile(v, w, q)
    # The midpoint plotting position (cum - 0.5 w)/total is, at equal
    # weights, exactly numpy's Hazen method ((i - 0.5)/n) -- the same
    # convention the harness metric uses.
    expect = np.quantile(v, q, method="hazen")
    assert np.allclose(got, expect, atol=1e-12)


def test_weighted_mean_matches_numpy_average():
    builder = _import_builder()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 2.0, 4.0])
    assert builder._weighted_mean(v, w) == pytest.approx(
        float(np.average(v, weights=w))
    )


def test_weighted_gini_known_values():
    builder = _import_builder()
    # A perfectly equal distribution has Gini 0.
    v = np.array([5.0, 5.0, 5.0, 5.0])
    w = np.ones(4)
    assert builder._weighted_gini(v, w) == pytest.approx(0.0, abs=1e-9)
    # Two equal-weight points {0, 1}: Lorenz area = 1/8, Gini = 1 - 2/8 = 0.5.
    v2 = np.array([0.0, 1.0])
    w2 = np.array([1.0, 1.0])
    assert builder._weighted_gini(v2, w2) == pytest.approx(0.5, abs=1e-9)
    # Weights scale-invariant: doubling weights leaves Gini unchanged.
    v3 = np.array([1.0, 2.0, 3.0, 10.0])
    w3 = np.array([2.0, 1.0, 3.0, 1.0])
    assert builder._weighted_gini(v3, w3) == pytest.approx(
        builder._weighted_gini(v3, 2.0 * w3), abs=1e-12
    )


def test_weighted_ks_identical_and_shifted():
    builder = _import_builder()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    # Identical distributions -> KS 0.
    assert builder._weighted_ks(v, w, v, w) == pytest.approx(0.0, abs=1e-12)
    # Disjoint supports (all a < all b) -> KS 1.
    a = np.array([0.0, 0.1])
    b = np.array([10.0, 11.0])
    assert builder._weighted_ks(a, np.ones(2), b, np.ones(2)) == pytest.approx(
        1.0, abs=1e-12
    )
    # Half the mass shifted out: two points {0,0} vs {0,1} -> KS 0.5.
    a2 = np.array([0.0, 0.0])
    b2 = np.array([0.0, 1.0])
    assert builder._weighted_ks(
        a2, np.ones(2), b2, np.ones(2)
    ) == pytest.approx(0.5, abs=1e-12)


def test_person_pia_proxy_zero_and_monotone():
    """The proxy is 0 for no positive obs and monotone in earnings.

    Uses a minimal parameter stand-in (flat NAWI, a generous wage base, and
    the statutory 2022 bend points / factors) so the test exercises the
    proxy assembly (positive filter, indexing, top-N, divisor) driving the
    REAL 415(a)/415(g) ``pia`` -- no pe-us checkout required.
    """
    builder = _import_builder()

    class _FakeParams:
        nawi = {y: 1.0 for y in range(1998, 2023, 2)}
        pia_factors = (0.9, 0.32, 0.15)

        def wage_base_for(self, year):  # generous cap (no clipping here)
            return 1e12

        def bend_points(self, year):  # statutory 2022 bend points
            return (1024.0, 6172.0)

    fake = _FakeParams()
    periods = np.array([1998, 2000, 2002], dtype=int)
    zero = builder.person_pia_proxy(periods, np.array([0.0, 0.0, 0.0]), fake)
    assert zero == 0.0
    low = builder.person_pia_proxy(
        periods, np.array([10_000.0, 10_000.0, 10_000.0]), fake
    )
    high = builder.person_pia_proxy(
        periods, np.array([60_000.0, 60_000.0, 60_000.0]), fake
    )
    assert high > low > 0.0
    # A person with more positive years and higher earnings ranks higher.
    assert (
        builder.person_pia_proxy(
            periods, np.array([80_000.0, 80_000.0, 80_000.0]), fake
        )
        > high
    )


def test_assign_quintiles_zero_to_q0_positives_by_quartile():
    """Q0 is exactly the zero-anchor persons; positives split by quartile."""
    import pandas as pd

    builder = _import_builder()
    anchor = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            # Two zero-anchor persons -> Q0; four positives across quartiles.
            "earnings": [0.0, 0.0, 10_000.0, 30_000.0, 60_000.0, 120_000.0],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    cuts = builder.anchor_quintile_cutpoints(anchor)
    # Positive-anchor quartile cuts sit strictly inside the positive support.
    assert cuts[0] > 0.0
    q = builder._assign_quintiles(anchor, anchor["person_id"].to_numpy(), cuts)
    assert q[1] == 0 and q[2] == 0  # zero-anchor -> Q0
    assert q[3] == 1  # smallest positive -> Q1
    assert q[6] == builder.N_QUINTILES - 1  # largest positive -> Q4
    # Every positive person lands in Q1..Q4 (never Q0).
    for pid in (3, 4, 5, 6):
        assert 1 <= q[pid] <= builder.N_QUINTILES - 1


def test_by_quintile_pooled_block_present_and_sane():
    """The pooled by-quintile block covers Q0..Q4 with valid shares."""
    art = _artifact()
    pooled = art["candidate_pooled"]["by_anchor_quintile"]
    for q in range(5):
        blk = pooled[f"Q{q}"]
        if blk.get("n_seeds_present", 0) == 0:
            continue
        assert 0.0 <= blk["weighted_share_within_5pct"] <= 1.0
        assert 0.0 <= blk["weighted_share_within_10pct"] <= 1.0
        assert blk["weighted_mae"] >= 0.0


# --------------------------------------------------------------------------
# Seed-0 reproduction pin (needs PSID + populace-fit; run in the gate venv)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun the seed-0 candidate and noise blocks; match to float precision."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    builder = _import_builder()

    art = _artifact()
    cand0 = next(s for s in art["candidate_per_seed"] if s["seed"] == 0)
    noise0 = next(s for s in art["noise_per_seed"] if s["seed"] == 0)

    params = builder.load_ssa_parameters()
    # The committed artifact pins the pe-us revision it was built against.
    assert params.pe_us_revision == art["oracle"]["pe_us_revision"], (
        "policyengine-us checkout revision differs from the pinned build; "
        "set POPULACE_DYNAMICS_PE_US_DIR to the pinned checkout"
    )
    panel = builder.load_filtered_panel()
    all_anchor = builder.anchor_rows(panel)
    cutpoints = builder.anchor_quintile_cutpoints(all_anchor)

    got_cand = builder.measure_seed_candidate(
        0, panel, all_anchor, params, cutpoints, False
    )
    got_noise = builder.measure_seed_noise(0, panel, all_anchor, params, False)

    # Distribution block (candidate): mean/median/decile values, Gini, KS.
    def assert_distribution(got: dict, ref: dict) -> None:
        for key in ("mean", "median"):
            assert got[key]["candidate"] == pytest.approx(
                ref[key]["candidate"], abs=1e-9
            ), key
            assert got[key]["real"] == pytest.approx(
                ref[key]["real"], abs=1e-9
            ), key
        for dkey in ref["deciles"]:
            assert got["deciles"][dkey]["candidate"] == pytest.approx(
                ref["deciles"][dkey]["candidate"], abs=1e-9
            ), dkey
            assert got["deciles"][dkey]["real"] == pytest.approx(
                ref["deciles"][dkey]["real"], abs=1e-9
            ), dkey
        assert got["gini"]["candidate"] == pytest.approx(
            ref["gini"]["candidate"], abs=1e-9
        )
        assert got["ks_distance"] == pytest.approx(
            ref["ks_distance"], abs=1e-9
        )

    assert_distribution(got_cand["distribution"], cand0["distribution"])
    assert_distribution(got_noise["distribution"], noise0["distribution"])

    # Person-level block (candidate).
    for key in (
        "weighted_mae",
        "weighted_rmse",
        "weighted_share_within_5pct",
        "weighted_share_within_10pct",
    ):
        assert got_cand["person_level"][key] == pytest.approx(
            cand0["person_level"][key], abs=1e-9
        ), key

    # Candidate-7 pool sizes reproduce exactly (deterministic given split).
    assert got_cand["pools"] == cand0["pools"]
    assert got_cand["n_persons"] == cand0["n_persons"]
    assert got_noise["n_persons_side_a"] == noise0["n_persons_side_a"]
    assert got_noise["n_persons_side_b"] == noise0["n_persons_side_b"]
