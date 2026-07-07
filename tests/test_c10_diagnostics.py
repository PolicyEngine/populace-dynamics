"""Tests for the candidate-10 diagnostics (seed extension + forensics).

REPORTED-NOT-GATED. The consistency tests here read only the diagnostics
artifact (``runs/c10_diagnostics_v1.json``), the committed gate and floor
artifacts, and ``gates.yaml``; they never rerun the gate and need no PSID or
populace-fit, so they run in CI. They audit that every stored number is
internally consistent with the claims the artifact makes -- the 20-seed
distribution and the clip inference recompute from the per-seed values, the
committed seeds 0-4 candidate C2ST equals the committed gate artifact, the
floor provenance is self-consistent, and the forensics' mirrored classifier
equals the gate's C2ST bit-for-bit and matches the committed gate value per
seed.

One reproduction pin (``test_seed5_reproduces_recorded_values``) reruns the
diagnostic-1 generation live at seed 5 (skipped when the PSID family files
are absent, and ``importorskip("populace.fit")`` because the participation
gates need it) and matches the recorded fresh candidate and floor C2ST to
float precision, so the fresh seed-extension numbers are pinned exactly as
the candidate-10 run's own seed-0 reproduction pins the gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "c10_diagnostics_v1.json"
GATE_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v4.json"
FLOOR_ARTIFACT = ROOT / "runs" / "noise_floor_psid_family_ctx20_9822.json"
SCRIPTS = ROOT / "scripts"
PAIRS = "psid_family_earnings_pairs"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _d1() -> dict:
    return _artifact()["diagnostic_1_seed_extension"]


def _d2() -> dict:
    return _artifact()["diagnostic_2_microtexture_forensics"]


def _gate_pairs_c2st() -> dict[int, float]:
    gate = json.loads(GATE_ARTIFACT.read_text())
    return {
        int(s["seed"]): float(s["geometry"][PAIRS]["scores"]["c2st_auc"])
        for s in gate["per_seed"]
    }


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "c10_diagnostics.v1"
    assert a["reported_not_gated"] is True
    assert "candidate 10" in a["candidate"]
    assert "diagnostic_1_seed_extension" in a
    assert "diagnostic_2_microtexture_forensics" in a
    assert "verdict" in a


def test_verdict_pulls_consistent_numbers_from_blocks():
    """The top-level verdict's key numbers match the two diagnostic blocks."""
    v = _artifact()["verdict"]
    assert v["reading"] == "noise"
    d1 = _d1()
    k1 = v["diagnostic_1_key"]
    assert k1["candidate_mean_c2st"] == pytest.approx(
        d1["candidate_c2st_distribution"]["mean"]
    )
    assert k1["margin_below_line_in_se"] == pytest.approx(
        d1["clip_inference"]["margin_below_line_in_se"]
    )
    assert (
        k1["n_seeds_over_threshold"]
        == d1["clip_inference"]["n_seeds_over_threshold"]
    )
    assert k1["excess_over_floor_mean"] == pytest.approx(
        d1["excess_over_floor_distribution"]["mean"]
    )
    k2 = v["diagnostic_2_key"]
    con = _d2()["contrast_failing_vs_passing"]
    assert (
        k2["top_signal_feature"]
        == con["signal_ranking_by_separation"][0]["feature"]
    )
    assert k2["gate_c2st_failing_minus_passing"] == pytest.approx(
        con["gate_c2st_failing_minus_passing"]
    )
    # The verdict records the analytical judgments the numbers support.
    assert k2["signal_is_same_across_seeds"] is True
    assert k2["coherent_failing_vs_passing_signal"] is False


# --------------------------------------------------------------------------
# Diagnostic 1: seed-extension noise measurement
# --------------------------------------------------------------------------
def test_d1_threshold_matches_locked_gates_yaml():
    """The focal threshold equals the locked pairs-view c2st_auc_max."""
    import yaml

    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    view = gates["gates"]["gate_1"]["thresholds"]["views"][PAIRS]
    # Amendment 2 (2026-07-07) superseded the per-seed c2st_auc_max
    # with the 20-seed mean rule at the SAME ratified 0.53 line; the
    # diagnostics artifact predates the flip and measured that line.
    thr = view["c2st_mean_rule"]["value_max"]
    assert "c2st_auc" in view["per_seed_rule_superseded"]
    assert _d1()["pairs_c2st_threshold"] == pytest.approx(thr, abs=0)


def test_d1_distribution_recomputes_from_per_seed():
    """Candidate/floor/excess distribution stats recompute from per-seed."""
    d1 = _d1()
    cand = [p["candidate_pairs_c2st"] for p in d1["per_seed"]]
    floor = [p["floor_pairs_c2st"] for p in d1["per_seed"]]
    excess = [p["excess_over_floor"] for p in d1["per_seed"]]
    cd = d1["candidate_c2st_distribution"]
    fd = d1["floor_c2st_distribution"]
    ed = d1["excess_over_floor_distribution"]
    assert cd["n"] == len(cand)
    assert cd["mean"] == pytest.approx(float(np.mean(cand)))
    assert cd["sd"] == pytest.approx(float(np.std(cand, ddof=1)))
    assert fd["mean"] == pytest.approx(float(np.mean(floor)))
    assert ed["mean"] == pytest.approx(float(np.mean(excess)))
    assert ed["mean"] == pytest.approx(cd["mean"] - fd["mean"])


def test_d1_excess_is_candidate_minus_floor():
    for p in _d1()["per_seed"]:
        assert p["excess_over_floor"] == pytest.approx(
            p["candidate_pairs_c2st"] - p["floor_pairs_c2st"], abs=1e-12
        )
        assert p["candidate_over_threshold"] == (
            p["candidate_pairs_c2st"] > _d1()["pairs_c2st_threshold"]
        )


def test_d1_n_over_threshold_recomputes():
    d1 = _d1()
    thr = d1["pairs_c2st_threshold"]
    n_over = sum(1 for p in d1["per_seed"] if p["candidate_pairs_c2st"] > thr)
    assert d1["n_candidate_over_threshold"] == n_over
    assert d1["clip_inference"]["n_seeds_over_threshold"] == n_over


def test_d1_clip_inference_recomputes():
    """The noise inference recomputes from the stored mean/sd (norm+binom)."""
    inf = _d1()["clip_inference"]
    mean = inf["mean"]
    sd = inf["sd"]
    n = inf["n_seeds"]
    thr = inf["threshold"]

    # Distribution moments match the candidate distribution.
    cand = [p["candidate_pairs_c2st"] for p in _d1()["per_seed"]]
    assert mean == pytest.approx(float(np.mean(cand)))
    assert sd == pytest.approx(float(np.std(cand, ddof=1)))

    # Parametric single-seed clip prob and the binomial gate outcomes.
    p = float(stats.norm.sf((thr - mean) / sd))
    assert inf["single_seed_clip_prob_parametric"] == pytest.approx(p)
    assert inf["run12_outcome_prob_ge2_of_5_clip_parametric"] == pytest.approx(
        float(stats.binom.sf(1, 5, p))
    )
    assert inf["prob_le1_of_5_clip_parametric"] == pytest.approx(
        float(stats.binom.cdf(1, 5, p))
    )
    # The mean's margin below the line, in standard errors of the mean.
    se = sd / np.sqrt(n)
    assert inf["se_mean"] == pytest.approx(se)
    assert inf["margin_below_line_in_se"] == pytest.approx((thr - mean) / se)
    assert inf["mean_below_threshold"] is (mean < thr)

    # Empirical (distribution-free) clip fraction and its binomial.
    pe = inf["n_seeds_over_threshold"] / n
    assert inf["single_seed_clip_prob_empirical"] == pytest.approx(pe)
    assert inf["run12_outcome_prob_ge2_of_5_clip_empirical"] == pytest.approx(
        float(stats.binom.sf(1, 5, pe))
    )


def test_d1_committed_candidate_seeds_match_gate_artifact():
    """Seeds 0-4 candidate C2ST equals the committed gate artifact."""
    gate = _gate_pairs_c2st()
    by_seed = {p["seed"]: p for p in _d1()["per_seed"]}
    for s in (0, 1, 2, 3, 4):
        assert by_seed[s]["candidate_pairs_c2st"] == pytest.approx(
            gate[s], abs=1e-12
        )
        assert by_seed[s]["source"].startswith("committed")


def test_d1_floor_provenance_internally_consistent():
    """The floor provenance delta = committed - recomputed, self-contained.

    The floor is recomputed on the current panel because the committed
    floor artifact no longer reproduces (a pre-existing drift filed
    separately). This checks the recorded provenance is arithmetically
    consistent and that the recomputed floor equals the per-seed floor.
    """
    fp = _d1()["floor_provenance"]
    committed = fp["committed_floor_seeds_0_4"]
    recomputed = fp["recomputed_floor_seeds_0_4"]
    delta = fp["seed_0_4_committed_minus_recomputed"]
    for s, dv in delta.items():
        assert dv == pytest.approx(committed[s] - recomputed[s], abs=1e-12)
    by_seed = {str(p["seed"]): p for p in _d1()["per_seed"]}
    for s, rv in recomputed.items():
        assert rv == pytest.approx(by_seed[s]["floor_pairs_c2st"], abs=1e-12)
    # The committed-floor snapshot matches the committed floor artifact.
    committed_art = json.loads(FLOOR_ARTIFACT.read_text())
    vals = committed_art["noise_floor_seeds_0_4"]["c2st_auc"]["values"]
    for i, v in enumerate(vals):
        assert committed[str(i)] == pytest.approx(v, abs=1e-12)
    # The classifier-version provenance is recorded (the floor is a
    # scikit-learn-version-sensitive C2ST; recompute venv vs floor venv).
    assert fp["recompute_venv_sklearn"]
    assert "1.9.0" in fp["committed_floor_venv_sklearn"]


def test_d1_seeds_are_contiguous_from_zero():
    seeds = _d1()["seeds"]
    assert seeds == sorted(seeds)
    assert seeds[:5] == [0, 1, 2, 3, 4]
    assert seeds == list(range(seeds[0], seeds[-1] + 1))


# --------------------------------------------------------------------------
# Diagnostic 2: microtexture forensics
# --------------------------------------------------------------------------
def test_d2_reported_not_gated_no_holdout_contact():
    d2 = _d2()
    assert d2["reported_not_gated"] is True
    assert d2["holdout_contact"] is False
    assert d2["view"] == PAIRS
    assert d2["failing_seeds"] == [0, 1]
    assert d2["passing_seeds"] == [2, 4]
    assert {s["seed"] for s in d2["per_seed"]} == {0, 1, 2, 4}


def test_d2_mirror_equals_harness_every_seed():
    """The mirrored classifier reproduces the harness C2ST bit-for-bit."""
    for s in _d2()["per_seed"]:
        sa = s["sanity_anchor"]
        assert sa["mirror_equals_harness"] is True
        assert sa["mirror_c2st_vs_holdout"] == pytest.approx(
            sa["harness_c2st_vs_holdout"], abs=1e-12
        )


def test_d2_gate_c2st_matches_committed_gate_artifact():
    """Each seed's harness-c2st-vs-holdout equals the committed gate value.

    score_view scores C2ST as classifier_two_sample_auc(holdout, candidate),
    so the sanity anchor's harness value IS the gate number -- tying the
    forensic regeneration to the committed run 12.
    """
    gate = _gate_pairs_c2st()
    by_seed = {s["seed"]: s for s in _d2()["per_seed"]}
    for s in (0, 1, 2, 4):
        assert by_seed[s]["sanity_anchor"][
            "harness_c2st_vs_holdout"
        ] == pytest.approx(gate[s], abs=1e-9)
    # The two failing seeds are over 0.53; the two passing seeds are under.
    for s in (0, 1):
        assert by_seed[s]["is_failing"] is True
        assert gate[s] > 0.53
    for s in (2, 4):
        assert by_seed[s]["is_failing"] is False
        assert gate[s] < 0.53


def test_d2_marginal_and_pair_auc_band():
    """Marginal/pair AUCs sit between the train-halves floor and full AUC."""
    slack = 0.02
    for s in _d2()["per_seed"]:
        fa = s["feature_attribution"]
        floor = s["noise_floor_vs_train_halves"]
        full = fa["full_auc"]
        for name, auc in {
            **fa["marginal_auc"],
            **fa["pairwise_auc"],
        }.items():
            assert auc <= full + slack, f"seed {s['seed']} {name} > full"
            assert auc >= floor - slack, f"seed {s['seed']} {name} < floor"


def test_d2_train_windows_are_whole_dollar():
    """Real PSID train earnings are integer-valued (the lattice baseline)."""
    for s in _d2()["per_seed"]:
        rn = s["distributional"]["round_number"]
        assert rn["train"]["non_integer_share"] == pytest.approx(0.0, abs=1e-6)


def test_d2_signal_ranking_sorted_and_recomputes():
    """Signal ranking is sorted by separation and recomputes from per-seed."""
    c = _d2()["contrast_failing_vs_passing"]
    ranking = c["signal_ranking_by_separation"]
    seps = [r["separation_over_half"] for r in ranking]
    assert seps == sorted(seps, reverse=True)
    by_seed = {s["seed"]: s for s in _d2()["per_seed"]}
    for r in ranking:
        assert r["separation_over_half"] == pytest.approx(
            r["all_seed_marginal_auc"] - 0.5
        )
        vals = [
            by_seed[s]["feature_attribution"]["marginal_auc"][r["feature"]]
            for s in (0, 1, 2, 4)
        ]
        assert r["all_seed_marginal_auc"] == pytest.approx(
            float(np.mean(vals))
        )


def test_d2_contrast_failing_vs_passing_recomputes():
    """The failing/passing means recompute from the per-seed values."""
    c = _d2()["contrast_failing_vs_passing"]
    by_seed = {s["seed"]: s for s in _d2()["per_seed"]}
    sig = c["scalar_signals"]["gate_c2st_vs_holdout"]
    fmean = float(
        np.mean(
            [
                by_seed[s]["sanity_anchor"]["harness_c2st_vs_holdout"]
                for s in (0, 1)
            ]
        )
    )
    pmean = float(
        np.mean(
            [
                by_seed[s]["sanity_anchor"]["harness_c2st_vs_holdout"]
                for s in (2, 4)
            ]
        )
    )
    assert sig["failing_mean"] == pytest.approx(fmean)
    assert sig["passing_mean"] == pytest.approx(pmean)
    assert sig["failing_minus_passing"] == pytest.approx(fmean - pmean)
    assert c["gate_c2st_failing_minus_passing"] == pytest.approx(fmean - pmean)


# --------------------------------------------------------------------------
# Reproduction pin (needs the staged PSID family files AND populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed5_reproduces_recorded_values():
    """Rerun the seed-5 generation and match the recorded fresh values."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import c10_seed_extension as d1mod

    panel = d1mod.c10.load_filtered_panel()
    all_anchor = d1mod.c10.anchor_rows(panel)
    pview = d1mod.build_pairs_view()
    fview = d1mod.build_floor_view()
    cand = d1mod.candidate_pairs_c2st(5, panel, all_anchor, pview)
    floor = d1mod.ctx20_floor_pairs_c2st(5, panel, fview)

    by_seed = {p["seed"]: p for p in _d1()["per_seed"]}
    assert cand["pairs_c2st"] == pytest.approx(
        by_seed[5]["candidate_pairs_c2st"], abs=1e-12
    )
    assert floor["floor_c2st"] == pytest.approx(
        by_seed[5]["floor_pairs_c2st"], abs=1e-12
    )
