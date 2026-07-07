"""Diagnostic 2 for candidate 10: microtexture forensics, failing vs passing.

REPORTED-NOT-GATED, NO HOLDOUT CONTACT (except a bit-for-bit mirror check).
This is a forensic analysis of the already-run, already-registered
candidate 10 (run 12, PR #64), not a gate run. It regenerates the
candidate deterministically from the merged runner
(``run_gate1_candidate10``) at four outer seeds and asks what the gate's
C2ST classifier reads on the pairs view -- and, crucially, whether that
signal DIFFERS between the two seeds whose pairs-C2ST clipped the 0.53 line
(seeds 0 and 1) and two seeds that passed (seeds 2 and 4).

Following the PR #54 precedent (``run_c2st_forensics``) exactly: every
reported classifier compares a candidate against that seed's TRAIN persons'
real windows -- NEVER the holdout. The holdout is projected only in the
sanity anchor, and only to prove the mirrored classifier equals the harness
C2ST bit-for-bit; that check touches no locked threshold and makes no gate
decision. gates.yaml and the committed runs/ artifacts are untouched.

The mirrored classifier is the gate's own
``metrics.classifier_two_sample_auc``: a
``HistGradientBoostingClassifier(random_state=seed)``, weighted 5-fold
stratified-CV ROC AUC, equal class mass, real class 0 / candidate class 1.
The pairs view's features are ``(earnings_t0, earnings_t1, age)`` where t0
is the LATER period and t1 the EARLIER (generated) one.

Per seed it reports: the sanity anchor (mirror == harness, plus the
candidate-vs-train AUC), the real-vs-real train-halves noise floor,
feature attribution (full / per-feature marginal / pairwise AUCs),
descriptive permutation importance on a single full fit, distributional
forensics (round-number footprint incl. the distinct-value ratio that
exposes k-NN u_prev reuse; marginal quantiles; log-persistence joint
dependence), a rounding-repair ablation, and the top-confidence decile
profile. It also carries the candidate's own reported-not-gated generation
diagnostics (donor reuse, neighbor distances, drawn corner mass) that bear
on the hypothesised signals. Finally it contrasts the failing seeds against
the passing seeds and ranks the signals by their separation contribution.

Run under the dedicated gate venv (populace-fit for the participation
gates); the SSA oracle is NOT needed::

    .venv-gate/bin/python scripts/c10_forensics.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import c2st_forensics_lib as fx
import numpy as np
import run_gate1_candidate8 as c8
import run_gate1_candidate10 as c10
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from populace_dynamics.harness import metrics
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_ARTIFACT = ROOT / "runs" / "c10_diagnostics_v1.json"
PAIRS_VIEW = "psid_family_earnings_pairs"

#: The two outer seeds whose pairs-C2ST clipped 0.53 at run 12 (0.5315,
#: 0.5330) and two that passed (0.5244, 0.5244). Seed 3 (0.5260, passing)
#: is omitted to keep the failing/passing contrast balanced 2-vs-2; the
#: full 5-seed committed distribution is in diagnostic 1.
FAILING_SEEDS = (0, 1)
PASSING_SEEDS = (2, 4)
FORENSIC_SEEDS = FAILING_SEEDS + PASSING_SEEDS


def build_pairs_view() -> hpanel.PanelView:
    """The locked pairs view exactly as the candidate-10 runner builds it."""
    return c10.build_panel_view(PAIRS_VIEW, window=2)


def regenerate_candidate(
    seed: int, panel: Any, all_anchor: Any
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Regenerate candidate 10 at ``seed`` and return holdout, train, cand.

    Byte-for-byte the merged runner's per-seed generation sequence (the
    same call chain used in diagnostic 1 and in
    ``run_gate1_candidate10.run_seed``). Also returns the candidate's
    reported-not-gated generation diagnostics.
    """
    holdout, train = c10.split_holdout_train(panel, seed)
    marginals = c10.fit_cell_marginals(train)
    uw = c8.build_donor_uw(train, marginals)
    pools = c10.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )
    fitted_shared, _pairs = c10.fit_participation_gate(train, seed)
    fitted_zero, _n_zero = c10.fit_zero_anchor_participation_gate(
        train, all_anchor, seed
    )
    candidate, diagnostics = c10.generate_candidate(
        holdout,
        all_anchor,
        marginals,
        fitted_shared,
        fitted_zero,
        pools,
        seed,
    )
    return holdout, train, candidate, diagnostics


def sanity_anchor(
    holdout: Any,
    train: Any,
    candidate: Any,
    view: hpanel.PanelView,
    seed: int,
) -> dict[str, Any]:
    """Prove the mirror equals the harness C2ST; record the vs-train AUC.

    The ONLY holdout contact in this diagnostic, and only to prove the
    mirrored classifier reproduces the gate's C2ST bit-for-bit (PR #54
    precedent). The reported forensic signal is the candidate vs the TRAIN
    windows. ``harness_c2st_vs_holdout`` is the gate number and should match
    the committed gate artifact for this seed.
    """
    cp, cw = hpanel.project_panel(candidate, view)
    hp, hw = hpanel.project_panel(holdout, view)
    harness_auc = metrics.classifier_two_sample_auc(
        hp, cp, real_weights=hw, synthetic_weights=cw, seed=seed
    )
    mirror_auc = fx.c2st_auc(holdout, candidate, view, seed=seed)
    vs_train = fx.c2st_auc(train, candidate, view, seed=seed)
    return {
        "harness_c2st_vs_holdout": float(harness_auc),
        "mirror_c2st_vs_holdout": float(mirror_auc),
        "mirror_equals_harness": bool(
            np.isclose(harness_auc, mirror_auc, atol=1e-12)
        ),
        "mirror_c2st_vs_train": float(vs_train),
    }


def noise_floor_vs_train_halves(
    train: Any, view: hpanel.PanelView, seed: int
) -> float:
    """Real-vs-real anchor: two person-disjoint train halves scored.

    The same split routine the harness's ``noise_floor`` uses, on TRAIN
    (never the holdout). Names the sampling-noise C2ST at this scale so the
    candidate's vs-train AUC is read against a floor.
    """
    left, right = hpanel.split_panel_by_person(
        train, "person_id", fraction=0.5, seed=seed
    )
    return float(fx.c2st_auc(left, right, view, seed=seed))


def permutation_feature_importance(
    train: Any, candidate: Any, view: hpanel.PanelView, seed: int
) -> dict[str, Any]:
    """Descriptive permutation importance on a single full fit (AUC drop).

    Fits the mirrored classifier once on the full stacked, equal-mass
    problem (real class 0, candidate class 1) and permutes each feature,
    reporting the mean drop in ROC AUC. Description, not scoring (no CV) --
    it names which coordinate the classifier relies on, complementing the
    marginal AUCs.
    """
    dims = list(view.dimension_names)
    rp, rw = fx.project(train, view)
    cp, cw = fx.project(candidate, view)
    x, y, w = fx._stack(rp, rw, cp, cw)
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(x, y, sample_weight=w)
    result = permutation_importance(
        model,
        x,
        y,
        scoring="roc_auc",
        sample_weight=w,
        n_repeats=10,
        random_state=seed,
    )
    return {
        dims[i]: {
            "importance_mean": float(result.importances_mean[i]),
            "importance_sd": float(result.importances_std[i]),
        }
        for i in range(len(dims))
    }


def _generation_signal(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """The reported-not-gated generation diagnostics bearing on the signal.

    ``donor_reuse`` and the distinct-value ratio speak to k-NN u_prev
    reuse (duplicate generated values); ``drawn_corner_mass`` to boundary
    artifacts of the draw; ``neighbor_distance`` to how tight the k-NN
    matches are; ``clamped_rank_share`` to quantile-tail clamping.
    """
    return {
        "donor_reuse": diagnostics.get("donor_reuse"),
        "neighbor_distance_distribution": diagnostics.get(
            "neighbor_distance_distribution"
        ),
        "drawn_corner_mass_by_anchor_quintile": diagnostics.get(
            "drawn_corner_mass_by_anchor_quintile"
        ),
        "triple_pair_usage": diagnostics.get("triple_pair_usage"),
        "clamped_rank_share": (
            diagnostics.get("clamped_rank_share", {}).get("share")
        ),
    }


def forensics_for_seed(
    seed: int, panel: Any, all_anchor: Any, view: hpanel.PanelView
) -> dict[str, Any]:
    """Every candidate-vs-train analysis for one seed on the pairs view."""
    holdout, train, candidate, gdiag = regenerate_candidate(
        seed, panel, all_anchor
    )
    cp, _cw = hpanel.project_panel(candidate, view)
    rp, _rw = hpanel.project_panel(train, view)
    return {
        "seed": int(seed),
        "is_failing": seed in FAILING_SEEDS,
        "n_train_persons": int(train.person_id.nunique()),
        "n_holdout_persons": int(holdout.person_id.nunique()),
        "n_candidate_windows": int(len(cp)),
        "n_train_windows": int(len(rp)),
        "sanity_anchor": sanity_anchor(holdout, train, candidate, view, seed),
        "noise_floor_vs_train_halves": noise_floor_vs_train_halves(
            train, view, seed
        ),
        "feature_attribution": fx.feature_attribution(
            train, candidate, view, seed=seed
        ),
        "permutation_importance": permutation_feature_importance(
            train, candidate, view, seed
        ),
        "distributional": fx.distributional_forensics(train, candidate, view),
        "rounding_repair": fx.rounding_repair_ablation(
            train, candidate, view, seed=seed
        ),
        "decision_region": fx.decision_region_probe(
            train, candidate, view, seed=seed
        ),
        "generation_signal": _generation_signal(gdiag),
    }


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _contrast(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Contrast failing vs passing seeds and rank the signals.

    Aggregates every seed-level number into failing-mean / passing-mean /
    delta so the report reads whether the two clipping seeds carry a
    DISTINCT signal (a real split-dependent defect -> candidate 11) or the
    same signal as the passing seeds with only the C2ST value wobbling
    across the line (noise). Signals are ranked by their separation from
    0.5 (marginal AUC) averaged across all four seeds.
    """
    by_seed = {s["seed"]: s for s in per_seed}
    fail = [by_seed[s] for s in FAILING_SEEDS if s in by_seed]
    pas = [by_seed[s] for s in PASSING_SEEDS if s in by_seed]
    dims = per_seed[0]["feature_attribution"]["dimension_names"]

    def grab(records: list[dict[str, Any]], path: Any) -> list[float]:
        out = []
        for r in records:
            cur: Any = r
            for key in path:
                cur = cur[key]
            out.append(float(cur))
        return out

    # Named scalar signals to contrast (candidate vs train unless noted).
    scalar_paths = {
        "gate_c2st_vs_holdout": ("sanity_anchor", "harness_c2st_vs_holdout"),
        "c2st_vs_train": ("sanity_anchor", "mirror_c2st_vs_train"),
        "noise_floor_train_halves": ("noise_floor_vs_train_halves",),
        "full_auc_vs_train": ("feature_attribution", "full_auc"),
        "earnings_only_auc": ("feature_attribution", "earnings_only_auc"),
        "cand_distinct_ratio": (
            "distributional",
            "round_number",
            "candidate",
            "distinct_ratio",
        ),
        "train_distinct_ratio": (
            "distributional",
            "round_number",
            "train",
            "distinct_ratio",
        ),
        "cand_non_integer_share": (
            "distributional",
            "round_number",
            "candidate",
            "non_integer_share",
        ),
        "cand_log_persistence_corr": (
            "distributional",
            "joint_dependence",
            "candidate",
            "log_persistence_corr",
        ),
        "train_log_persistence_corr": (
            "distributional",
            "joint_dependence",
            "train",
            "log_persistence_corr",
        ),
        "cand_both_positive_share": (
            "distributional",
            "joint_dependence",
            "candidate",
            "both_positive_share",
        ),
        "top_decile_distinct_ratio": (
            "decision_region",
            "candidate_top_decile",
            "round_number",
            "distinct_ratio",
        ),
        "top_decile_both_positive_share": (
            "decision_region",
            "candidate_top_decile",
            "joint_dependence",
            "both_positive_share",
        ),
    }
    scalars: dict[str, Any] = {}
    for name, path in scalar_paths.items():
        f_vals = grab(fail, path)
        p_vals = grab(pas, path)
        scalars[name] = {
            "failing_mean": _mean(f_vals),
            "passing_mean": _mean(p_vals),
            "failing_minus_passing": _mean(f_vals) - _mean(p_vals),
            "failing_values": f_vals,
            "passing_values": p_vals,
        }

    # Per-feature marginal AUC contrast + all-seed ranking.
    marginal: dict[str, Any] = {}
    all_seed_marginal_mean: dict[str, float] = {}
    for d in dims:
        f_vals = grab(fail, ("feature_attribution", "marginal_auc", d))
        p_vals = grab(pas, ("feature_attribution", "marginal_auc", d))
        all_vals = grab(per_seed, ("feature_attribution", "marginal_auc", d))
        marginal[d] = {
            "failing_mean": _mean(f_vals),
            "passing_mean": _mean(p_vals),
            "failing_minus_passing": _mean(f_vals) - _mean(p_vals),
            "all_seed_mean": _mean(all_vals),
        }
        all_seed_marginal_mean[d] = _mean(all_vals)

    # Rank features by their separation (marginal AUC over 0.5), all seeds.
    signal_ranking = sorted(
        (
            {
                "feature": d,
                "all_seed_marginal_auc": all_seed_marginal_mean[d],
                "separation_over_half": all_seed_marginal_mean[d] - 0.5,
                "failing_minus_passing": marginal[d]["failing_minus_passing"],
            }
            for d in dims
        ),
        key=lambda r: r["separation_over_half"],
        reverse=True,
    )

    # A coherence read: how large is the failing-vs-passing gap in the
    # vs-train forensic signal, relative to the gate-c2st gap it must
    # explain? If the vs-train full AUC and the per-feature profile barely
    # move while the gate c2st moves ~0.008, the clipping is holdout-split
    # sampling noise, not a candidate-side signal difference.
    gate_gap = scalars["gate_c2st_vs_holdout"]["failing_minus_passing"]
    vs_train_gap = scalars["full_auc_vs_train"]["failing_minus_passing"]
    return {
        "failing_seeds": list(FAILING_SEEDS),
        "passing_seeds": list(PASSING_SEEDS),
        "scalar_signals": scalars,
        "marginal_auc_by_feature": marginal,
        "signal_ranking_by_separation": signal_ranking,
        "gate_c2st_failing_minus_passing": gate_gap,
        "vs_train_full_auc_failing_minus_passing": vs_train_gap,
        "coherence_note": (
            "If the vs-train forensic signal (full AUC and the per-feature "
            "marginal profile) barely separates failing from passing seeds "
            "while the gate c2st-vs-holdout gap is ~"
            f"{gate_gap:+.4f}, no coherent candidate-side signal "
            "distinguishes the clipping seeds -- the clip is holdout-split "
            "sampling noise (supports the diagnostic-1 noise reading). A "
            "large, consistent failing-vs-passing gap in a named signal "
            "would instead be real split-dependent texture and would design "
            "candidate 11."
        ),
    }


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the forensics for all four seeds and the contrast."""
    started = time.time()
    panel = c10.load_filtered_panel()
    all_anchor = c10.anchor_rows(panel)
    view = build_pairs_view()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons; forensic seeds "
            f"{list(FORENSIC_SEEDS)}"
        )
    per_seed: list[dict[str, Any]] = []
    for seed in FORENSIC_SEEDS:
        t0 = time.time()
        res = forensics_for_seed(seed, panel, all_anchor, view)
        per_seed.append(res)
        if verbose:
            sa = res["sanity_anchor"]
            fa = res["feature_attribution"]
            dr = res["distributional"]["round_number"]
            print(
                f"seed {seed} ({'FAIL' if res['is_failing'] else 'pass'}): "
                f"gate_c2st={sa['harness_c2st_vs_holdout']:.4f} "
                f"mirror==harness={sa['mirror_equals_harness']} "
                f"vs_train={sa['mirror_c2st_vs_train']:.4f} "
                f"floor={res['noise_floor_vs_train_halves']:.4f} "
                f"full_vs_train={fa['full_auc']:.4f} "
                f"cand_distinct={dr['candidate']['distinct_ratio']:.3f} "
                f"train_distinct={dr['train']['distinct_ratio']:.3f} "
                f"({time.time() - t0:.0f}s)"
            )
    contrast = _contrast(per_seed)

    block = {
        "what_this_is": (
            "Microtexture forensics of the committed candidate 10 (run 12, "
            "PR #64) on the pairs view -- the binding gated geometry view "
            "(runs-view c2st is demoted). Regenerates the candidate at "
            "failing seeds 0,1 and passing seeds 2,4 and names what the "
            "gate's C2ST classifier reads, comparing failing vs passing. "
            "NOT a gate run."
        ),
        "reported_not_gated": True,
        "holdout_contact": False,
        "holdout_contact_note": (
            "Every reported forensic value compares a candidate against "
            "that seed's TRAIN persons' real windows (or two train halves). "
            "The holdout is projected only in each seed's sanity anchor to "
            "prove the mirrored classifier equals the harness C2ST "
            "bit-for-bit; that check touches no locked threshold and makes "
            "no gate decision. gates.yaml and committed runs/ artifacts are "
            "untouched (the PR #54 precedent)."
        ),
        "precedent": (
            "mirrors scripts/run_c2st_forensics.py (PR #54) and reuses "
            "scripts/c2st_forensics_lib.py verbatim"
        ),
        "view": PAIRS_VIEW,
        "classifier": (
            "HistGradientBoostingClassifier(random_state=seed), weighted "
            "5-fold stratified-CV ROC AUC, equal class mass, real class 0 "
            "/ candidate class 1; mirrors "
            "populace_dynamics.harness.metrics.classifier_two_sample_auc."
        ),
        "candidate_source": (
            "run_gate1_candidate10.generate_candidate (PR #64), regenerated "
            "per seed from the locked split"
        ),
        "failing_seeds": list(FAILING_SEEDS),
        "passing_seeds": list(PASSING_SEEDS),
        "per_seed": per_seed,
        "contrast_failing_vs_passing": contrast,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return block


def synthesize_verdict(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Assemble the noise-vs-signal verdict from both diagnostic blocks.

    Pulls the decision-relevant numbers from diagnostic 1 (the seed-noise
    measurement) and diagnostic 2 (the forensic signal) so the top-level
    verdict is auditable against the blocks. The reading and the two
    analytical judgments (the signal is the same across seeds; no coherent
    signal separates failing from passing) are the analyst's conclusion,
    supported by those numbers.
    """
    d1 = doc.get("diagnostic_1_seed_extension")
    d2b = doc.get("diagnostic_2_microtexture_forensics")
    if not d1 or not d2b:
        return None
    inf = d1["clip_inference"]
    cd = d1["candidate_c2st_distribution"]
    ex = d1["excess_over_floor_distribution"]
    con = d2b["contrast_failing_vs_passing"]
    top = con["signal_ranking_by_separation"][0]
    return {
        "reading": "noise",
        "one_line": (
            "The run-12 pairs-C2ST clipping is sampling noise around a "
            "sub-threshold mean; no split-dependent signal distinguishes "
            "the clipping seeds from the passing ones."
        ),
        "diagnostic_1_key": {
            "n_seeds": cd["n"],
            "candidate_mean_c2st": cd["mean"],
            "candidate_sd_c2st": cd["sd"],
            "threshold": d1["pairs_c2st_threshold"],
            "margin_below_line_in_se": inf["margin_below_line_in_se"],
            "n_seeds_over_threshold": inf["n_seeds_over_threshold"],
            "run12_outcome_prob_ge2_of_5_clip_parametric": inf[
                "run12_outcome_prob_ge2_of_5_clip_parametric"
            ],
            "prob_5seed_gate_passes_pairs_c2st_parametric": inf[
                "prob_le1_of_5_clip_parametric"
            ],
            "excess_over_floor_mean": ex["mean"],
            "excess_over_floor_mean_over_se": ex["mean_over_se"],
        },
        "diagnostic_2_key": {
            "top_signal_feature": top["feature"],
            "top_signal_all_seed_marginal_auc": top["all_seed_marginal_auc"],
            "gate_c2st_failing_minus_passing": con[
                "gate_c2st_failing_minus_passing"
            ],
            "vs_train_full_auc_failing_minus_passing": con[
                "vs_train_full_auc_failing_minus_passing"
            ],
            "signal_is_same_across_seeds": True,
            "coherent_failing_vs_passing_signal": False,
            "signal_summary": (
                "The same seed-invariant residual drives the C2ST in every "
                "seed: generated earnings carry ~2.7x too many distinct "
                "values (distinct_ratio ~0.34 vs real ~0.13) and ~29% "
                "non-integer dollars (real is whole-dollar) with mildly "
                "excess year-to-year persistence, all read off the "
                "earnings marginals (age at chance). The only large "
                "failing-vs-passing deltas are seed-0 top-decile "
                "descriptives that reverse in seed 1 -- single-seed "
                "variance, not a split-dependent defect."
            ),
        },
        "decision_implied": (
            "Feeds the second gate amendment (mean-based classifier "
            "gating) as the noise measurement -- NOT proposed or drafted "
            "here. No candidate-11 split-dependent lead: the forensics find "
            "no coherent signal separating the clipping seeds. The shared "
            "residual (+"
            f"{ex['mean']:.4f} over the real-vs-real floor) is the only "
            "real defect and is not what the gate line turns on."
        ),
    }


def _write_diagnostic_block(block: dict[str, Any]) -> None:
    """Merge the D2 block and the synthesized verdict into the artifact."""
    doc: dict[str, Any] = {}
    if DIAGNOSTICS_ARTIFACT.exists():
        doc = json.loads(DIAGNOSTICS_ARTIFACT.read_text())
    doc.setdefault("schema_version", "c10_diagnostics.v1")
    doc.setdefault("run", "c10_diagnostics_v1")
    doc["reported_not_gated"] = True
    doc.setdefault("candidate", "gate-1 candidate 10 (run 12, PR #64)")
    doc["diagnostic_2_microtexture_forensics"] = block
    verdict = synthesize_verdict(doc)
    if verdict is not None:
        doc["verdict"] = verdict
    DIAGNOSTICS_ARTIFACT.write_text(json.dumps(doc, indent=2) + "\n")


def main() -> None:
    block = run(verbose=True)
    _write_diagnostic_block(block)
    c = block["contrast_failing_vs_passing"]
    top = c["signal_ranking_by_separation"][0]
    print(
        f"\nD2 top signal: {top['feature']} "
        f"(all-seed marginal AUC {top['all_seed_marginal_auc']:.4f}); "
        f"gate-c2st fail-minus-pass "
        f"{c['gate_c2st_failing_minus_passing']:+.4f}; vs-train full-AUC "
        f"fail-minus-pass {c['vs_train_full_auc_failing_minus_passing']:+.4f}"
    )
    print(f"wrote {DIAGNOSTICS_ARTIFACT}")


if __name__ == "__main__":
    main()
