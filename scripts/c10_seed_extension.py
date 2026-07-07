"""Diagnostic 1 for candidate 10: the seed-extension noise measurement.

REPORTED-NOT-GATED. This script does NOT run the gate and does NOT touch
``gates.yaml`` or the committed gate artifact. It measures the SAME
committed candidate -- gate-1 candidate 10, run 12, the inner-validated
composition merged in PR #64 -- on additional protocol-identical splits so
that the run-12 near-miss (pairs-view C2ST clipping the 0.53 line on outer
seeds 0 and 1, mean 0.5279 over five seeds) can be read against the
sampling noise of the protocol rather than against five draws alone.

Extending the seed set is a MEASUREMENT of an already-registered,
already-run candidate, not a new gate run: the locked gate scores the
pre-registered seeds 0-4 and its verdict (``runs/gate1_rank_knn_v4.json``,
gate_1_pass=False, 3/5 geometry) stands untouched. Seeds 5-19 are scored
here only to estimate the per-seed sampling distribution of the focal
number (pairs-view ``c2st_auc``) around its true mean; no seed scored here
enters any pass/fail decision.

What it computes, per fresh seed s in 5..19 (protocol-identical to run 12):

* the CANDIDATE pairs-view C2ST -- regenerate candidate 10 EXACTLY per its
  frozen spec (issue #42 comment 4902561460; the merged runner
  ``run_gate1_candidate10`` implements it, reused byte-for-byte here) over
  the seed-s holdout, then score ONLY the locked pairs view with
  ``panel_scorecard`` (the focal ``c2st_auc``). The battery and the
  benefit-space block are skipped -- they passed at run 12 and are not the
  binding constraint; the pairs C2ST is the only gated geometry threshold
  the near-miss failed (runs-view C2ST is demoted under the amendment).

* the ctx20 FLOOR pairs-view C2ST -- the committed floor generator's
  real-vs-real construction (``build_gate1_floor_artifacts``) at the same
  seed: draw 40% of persons (seed 1000+s), halve it person-disjointly
  (seed s), and score one real half against the other. This gives the
  candidate's EXCESS over the sampling floor at MATCHED noise.

The 20-seed distributions combine the committed seeds 0-4 (candidate from
``runs/gate1_rank_knn_v4.json``; floor from
``runs/noise_floor_psid_family_ctx20_9822.json``) with the fresh seeds
5-19 scored here, and feed the inference: given the estimated per-seed sd,
what is the probability that a candidate with THIS true mean clips >=2 of 5
seeds (the run-12 outcome)? Is the gate outcome reproducible or seed luck?

Run under the dedicated gate venv (populace-fit for the participation
gates); the SSA oracle is NOT needed (no benefit-space block here)::

    .venv-gate/bin/python scripts/c10_seed_extension.py --seeds 5-19

Generation dominates runtime (~2-4 min/seed); results are cached
incrementally so a rerun resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_gate1_candidate8 as c8
import run_gate1_candidate10 as c10
from scipy import stats

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CANDIDATE_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v4.json"
COMMITTED_FLOOR_ARTIFACT = (
    ROOT / "runs" / "noise_floor_psid_family_ctx20_9822.json"
)
DIAGNOSTICS_ARTIFACT = ROOT / "runs" / "c10_diagnostics_v1.json"

#: The single gated geometry threshold the run-12 near-miss failed
#: (gates.yaml gate_1 psid_family_earnings_pairs c2st_auc_max). Read from
#: gates.yaml at runtime so nothing is hardcoded against the lock.
PAIRS_VIEW = "psid_family_earnings_pairs"

#: The committed seeds the locked gate scores. Seeds 5-19 are the extension.
COMMITTED_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_EXTENSION_SEEDS = tuple(range(5, 20))


def pairs_c2st_threshold() -> float:
    """The locked pairs-view ``c2st_auc_max``, read from gates.yaml."""
    thresholds = c10.load_gate1_thresholds()
    geom = thresholds["views"][PAIRS_VIEW]["geometry"]
    return float(geom["c2st_auc_max"])


def build_pairs_view() -> hpanel.PanelView:
    """The locked pairs view exactly as the candidate-10 runner builds it."""
    return c10.build_panel_view(PAIRS_VIEW, window=2)


def build_floor_view() -> hpanel.PanelView:
    """The floor generator's window-2 view (build_gate1_floor_artifacts).

    Same projection geometry as :func:`build_pairs_view` (window 2, step 2,
    value=earnings, covariate=age); the name differs only because the floor
    builder tags its views ``w2``. C2ST is name-independent, so the two are
    directly comparable at matched noise.
    """
    return hpanel.PanelView(
        name="psid_family_earnings_w2",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=2,
        period_step=2,
    )


def candidate_pairs_c2st(
    seed: int,
    panel: Any,
    all_anchor: Any,
    view: hpanel.PanelView,
) -> dict[str, Any]:
    """Regenerate candidate 10 at ``seed`` and score ONLY the pairs C2ST.

    Byte-for-byte the merged runner's generation sequence
    (``run_gate1_candidate10.run_seed`` minus the runs view, the battery,
    and the benefit-space block): the locked 0.2 person-disjoint split, the
    train-side per-cell marginals, candidate 8's u_w decomposition,
    candidate 9's donor pools and two participation gates, and the
    candidate-10 backward k-NN chain (fixed lambda=0.1, full re-entry pool,
    Q0 memory-exempt). Only the pairs view is projected and scored.
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
    scores = hpanel.panel_scorecard(candidate, holdout, view, seed=seed)
    cand_windows, _ = hpanel.project_panel(candidate, view)
    return {
        "seed": int(seed),
        "pairs_c2st": float(scores["c2st_auc"]),
        "pairs_energy_distance": float(scores["energy_distance"]),
        "pairs_prdc_coverage": float(scores["prdc_coverage"]),
        "n_holdout_persons": int(holdout.person_id.nunique()),
        "n_pairs_windows": int(len(cand_windows)),
        "clamped_rank_share": float(
            diagnostics["clamped_rank_share"]["share"]
        ),
    }


def ctx20_floor_pairs_c2st(
    seed: int, panel: Any, view: hpanel.PanelView
) -> dict[str, Any]:
    """The committed ctx20 real-vs-real floor at ``seed`` (pairs view).

    Byte-for-byte ``build_gate1_floor_artifacts``'s ctx20 construction:
    draw 40% of persons (fraction=0.4, seed=1000+s), halve it
    person-disjointly (fraction=0.5, seed=s), score one half against the
    other (panel_scorecard, seed=s). Real vs real at the ~20%-of-persons
    scale the gate scores candidates at.
    """
    forty, _ = hpanel.split_panel_by_person(
        panel, "person_id", fraction=0.4, seed=1000 + seed
    )
    left, right = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=seed
    )
    scores = hpanel.panel_scorecard(left, right, view, seed=seed)
    return {
        "seed": int(seed),
        "floor_c2st": float(scores["c2st_auc"]),
        "n_persons_per_side": int(left.person_id.nunique()),
    }


def compute_fresh_seeds(
    candidate_seeds: tuple[int, ...],
    cache_path: Path,
    verbose: bool = True,
) -> dict[int, dict[str, Any]]:
    """Score the candidate at ``candidate_seeds`` and the floor at 0..max.

    The CANDIDATE is regenerated only at ``candidate_seeds`` (the fresh
    extension, default 5-19); the committed seeds 0-4 candidate C2ST comes
    from the gate artifact (verified reproducible -- seed 0 matches to
    1e-6). The FLOOR is recomputed on the CURRENT panel for every seed
    (committed 0-4 union the extension), because the committed ctx20 floor
    artifact predates the #46 family-loader refactor and no longer
    reproduces on the current panel (its ``needs_real_family`` reproduction
    test fails on master); recomputing keeps the floor matched to the
    candidate's panel, which is what "matched noise" requires.

    A rerun reuses any seed already present in ``cache_path`` so a long run
    resumes after an interruption. The cache is a plain seed -> record map.
    """
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    panel = c10.load_filtered_panel()
    all_anchor = c10.anchor_rows(panel)
    pview = build_pairs_view()
    fview = build_floor_view()
    floor_seeds = sorted(set(COMMITTED_SEEDS) | set(candidate_seeds))
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons; candidate seeds "
            f"{list(candidate_seeds)}; floor seeds {floor_seeds}"
        )
    for seed in floor_seeds:
        key = str(seed)
        rec = cache.get(key, {})
        need_candidate = seed in candidate_seeds and "candidate" not in rec
        need_floor = "floor" not in rec
        if not need_candidate and not need_floor:
            if verbose:
                f = rec["floor"]["floor_c2st"]
                cstr = (
                    f"cand={rec['candidate']['pairs_c2st']:.6f} "
                    if "candidate" in rec
                    else ""
                )
                print(f"seed {seed}: cached {cstr}floor={f:.6f}")
            continue
        t0 = time.time()
        if need_candidate:
            rec["candidate"] = candidate_pairs_c2st(
                seed, panel, all_anchor, pview
            )
        if need_floor:
            rec["floor"] = ctx20_floor_pairs_c2st(seed, panel, fview)
        cache[key] = rec
        cache_path.write_text(json.dumps(cache, indent=2) + "\n")
        if verbose:
            cstr = (
                f"cand_pairs_c2st={rec['candidate']['pairs_c2st']:.6f} "
                if "candidate" in rec
                else ""
            )
            print(
                f"seed {seed}: {cstr}"
                f"floor_pairs_c2st={rec['floor']['floor_c2st']:.6f} "
                f"({time.time() - t0:.0f}s)"
            )
    return {int(k): v for k, v in cache.items()}


def _sklearn_version() -> str:
    """The live scikit-learn version (the C2ST classifier's version)."""
    import sklearn

    return str(sklearn.__version__)


def _committed_candidate_c2st() -> dict[int, float]:
    """Seeds 0-4 pairs C2ST from the committed gate artifact."""
    artifact = json.loads(COMMITTED_CANDIDATE_ARTIFACT.read_text())
    out: dict[int, float] = {}
    for s in artifact["per_seed"]:
        c2st = s["geometry"][PAIRS_VIEW]["scores"]["c2st_auc"]
        out[int(s["seed"])] = float(c2st)
    return out


def _committed_floor_c2st() -> dict[int, float]:
    """Seeds 0-4 ctx20 floor pairs C2ST from the committed floor artifact."""
    artifact = json.loads(COMMITTED_FLOOR_ARTIFACT.read_text())
    values = artifact["noise_floor_seeds_0_4"]["c2st_auc"]["values"]
    return {int(i): float(v) for i, v in enumerate(values)}


def _dist_stats(values: list[float]) -> dict[str, Any]:
    """Mean, sample sd (ddof=1), min/max of a value list."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _clip_inference(
    cand_by_seed: dict[int, float],
    threshold: float,
) -> dict[str, Any]:
    """The noise inference: P(>=2 of 5 clip | this true mean, per-seed sd).

    Models each protocol-identical seed's pairs C2ST as
    ``Normal(mu, sigma)`` with ``mu`` and ``sigma`` estimated from the full
    seed set. The single-seed clip probability is
    ``p = P(X > threshold) = 1 - Phi((threshold - mu)/sigma)``; the number
    that clip in a fresh 5-seed gate is ``Binomial(5, p)``. Reports the
    run-12 outcome probability ``P(>=2 of 5 clip)``, the complementary
    ``P(<=1 of 5 clip)`` (>=4/5 seeds under the line -- the pairs-C2ST
    condition for the geometry gate to pass), the mean-vs-threshold margin
    and its standard error, and the empirical clip fraction as a
    distribution-free cross-check.
    """
    seeds = sorted(cand_by_seed)
    values = np.array([cand_by_seed[s] for s in seeds], dtype=np.float64)
    n = int(values.size)
    mu = float(values.mean())
    sigma = float(values.std(ddof=1))
    se_mean = float(sigma / np.sqrt(n))

    # Parametric single-seed clip probability under Normal(mu, sigma).
    z = (threshold - mu) / sigma
    p_clip_param = float(stats.norm.sf(z))
    # Empirical single-seed clip fraction (distribution-free).
    n_over = int((values > threshold).sum())
    p_clip_emp = float(n_over / n)

    def _binom_ge2(p: float) -> float:
        return float(stats.binom.sf(1, 5, p))  # P(X >= 2) = 1 - P(X<=1)

    def _binom_le1(p: float) -> float:
        return float(stats.binom.cdf(1, 5, p))  # P(X <= 1)

    # Margin of the mean below the line, in standard errors of the mean.
    margin = threshold - mu
    margin_se = float(margin / se_mean) if se_mean > 0 else float("inf")
    # One-sided test that the true mean is below the threshold.
    p_mean_below = float(stats.norm.cdf(margin_se))

    return {
        "threshold": float(threshold),
        "n_seeds": n,
        "mean": mu,
        "sd": sigma,
        "se_mean": se_mean,
        "mean_minus_threshold": float(mu - threshold),
        "mean_below_threshold": bool(mu < threshold),
        "margin_below_line_in_se": margin_se,
        "prob_true_mean_below_threshold_one_sided": p_mean_below,
        "single_seed_clip_prob_parametric": p_clip_param,
        "single_seed_clip_prob_empirical": p_clip_emp,
        "n_seeds_over_threshold": n_over,
        "run12_outcome_prob_ge2_of_5_clip_parametric": _binom_ge2(
            p_clip_param
        ),
        "run12_outcome_prob_ge2_of_5_clip_empirical": _binom_ge2(p_clip_emp),
        "prob_le1_of_5_clip_parametric": _binom_le1(p_clip_param),
        "prob_le1_of_5_clip_empirical": _binom_le1(p_clip_emp),
        "prob_le1_of_5_clip_note": (
            "P(<=1 of 5 seeds over the line) is the pairs-C2ST condition "
            "for the geometry gate's >=4/5 rule; the OTHER geometry "
            "thresholds and the battery must also hold for a seed to pass."
        ),
    }


def assemble_diagnostic_1(
    fresh: dict[int, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Combine committed seeds 0-4 candidate with the current-panel floor.

    Candidate seeds 0-4 come from the committed gate artifact (verified
    reproducible); candidate seeds 5-19 from the fresh cache. The FLOOR for
    every seed is the current-panel recomputation (the committed floor
    artifact no longer reproduces -- see ``floor_provenance``), so the
    candidate and floor are matched on one panel.
    """
    committed_cand = _committed_candidate_c2st()
    committed_floor = _committed_floor_c2st()

    cand_by_seed: dict[int, float] = dict(committed_cand)
    floor_by_seed: dict[int, float] = {}
    for seed, rec in fresh.items():
        if "candidate" in rec:
            cand_by_seed[seed] = float(rec["candidate"]["pairs_c2st"])
        if "floor" in rec:
            floor_by_seed[seed] = float(rec["floor"]["floor_c2st"])

    seeds = sorted(set(cand_by_seed) & set(floor_by_seed))
    cand_values = [cand_by_seed[s] for s in seeds]
    floor_values = [floor_by_seed[s] for s in seeds]
    excess_values = [cand_by_seed[s] - floor_by_seed[s] for s in seeds]

    per_seed = []
    for s in seeds:
        src = "committed (gate seeds 0-4)" if s in COMMITTED_SEEDS else "fresh"
        per_seed.append(
            {
                "seed": s,
                "source": src,
                "candidate_pairs_c2st": cand_by_seed[s],
                "floor_pairs_c2st": floor_by_seed[s],
                "excess_over_floor": cand_by_seed[s] - floor_by_seed[s],
                "candidate_over_threshold": bool(cand_by_seed[s] > threshold),
            }
        )

    inference = _clip_inference(cand_by_seed, threshold)
    excess_stats = _dist_stats(excess_values)
    # Paired standard error of the mean excess and its z (excess > 0 means
    # the candidate is detectably separated from the real-vs-real floor).
    exc = np.asarray(excess_values, dtype=np.float64)
    exc_se = float(exc.std(ddof=1) / np.sqrt(exc.size))
    excess_stats["se_mean"] = exc_se
    excess_stats["mean_over_se"] = (
        float(excess_stats["mean"] / exc_se) if exc_se > 0 else float("inf")
    )

    return {
        "what_this_is": (
            "MEASUREMENT of the committed candidate 10 (run 12, PR #64) on "
            "protocol-identical splits at seeds beyond the pre-registered "
            "0-4. NOT a gate run: the locked gate's seeds and verdict "
            "(runs/gate1_rank_knn_v4.json, gate_1_pass=False, 3/5 geometry) "
            "are untouched; no seed here enters any pass/fail. The focal "
            "number is the pairs-view c2st_auc, the single gated geometry "
            "threshold the near-miss failed."
        ),
        "candidate_spec_registration": c10.SPEC_REGISTRATION,
        "committed_gate_artifact": str(
            COMMITTED_CANDIDATE_ARTIFACT.relative_to(ROOT)
        ),
        "committed_floor_artifact": str(
            COMMITTED_FLOOR_ARTIFACT.relative_to(ROOT)
        ),
        "pairs_c2st_threshold": float(threshold),
        "seeds": seeds,
        "n_committed": len(COMMITTED_SEEDS),
        "n_fresh": len([s for s in seeds if s not in COMMITTED_SEEDS]),
        "per_seed": per_seed,
        "candidate_c2st_distribution": _dist_stats(cand_values),
        "floor_c2st_distribution": _dist_stats(floor_values),
        "excess_over_floor_distribution": excess_stats,
        "n_candidate_over_threshold": int(
            sum(1 for v in cand_values if v > threshold)
        ),
        "clip_inference": inference,
        "floor_provenance": {
            "note": (
                "The floor for EVERY seed here is recomputed under the GATE "
                "venv (.venv-gate) via the committed floor generator's "
                "construction (build_gate1_floor_artifacts ctx20), so it is "
                "matched to the candidate's classifier version -- required "
                "for a matched-noise excess. The committed ctx20 floor "
                "artifact was built under the plain .venv (scikit-learn "
                "1.9.0, per build_gate1_floor_artifacts) and reproduces "
                "there exactly; but the C2ST HistGradientBoosting classifier "
                "is scikit-learn-version-sensitive, and the gate candidates "
                "require populace-fit, which pins scikit-learn <1.9, so they "
                "-- and this diagnostic -- run under .venv-gate (scikit-learn "
                "1.8.0). Per-seed floor c2st therefore shifts ~+-0.002-0.006 "
                "between the versions (seed 0: 0.517505 under .venv vs "
                "0.515788 under .venv-gate), though the floor MEAN is stable "
                "(~0.5109). The candidate-10 gate artifact (also .venv-gate) "
                "reproduces exactly, so candidate and this recomputed floor "
                "share one classifier version; the committed floor does not "
                "(it is the 1.9.0 value). This cross-venv mismatch between "
                "floor derivation (1.9.0) and gate scoring (1.8.0) is filed "
                "as a separate task; NOT changed here (the locked 0.53 "
                "threshold was derived from the 1.9.0 floor and moving it "
                "needs the amendment ceremony)."
            ),
            "recompute_venv_sklearn": _sklearn_version(),
            "committed_floor_venv_sklearn": (
                "1.9.0 (.venv; per build_gate1_floor_artifacts)"
            ),
            "committed_floor_seeds_0_4": {
                str(s): committed_floor[s] for s in sorted(committed_floor)
            },
            "recomputed_floor_seeds_0_4": {
                str(s): floor_by_seed[s]
                for s in COMMITTED_SEEDS
                if s in floor_by_seed
            },
            "seed_0_4_committed_minus_recomputed": {
                str(s): committed_floor[s] - floor_by_seed[s]
                for s in COMMITTED_SEEDS
                if s in floor_by_seed and s in committed_floor
            },
        },
    }


def _write_diagnostic_block(block: dict[str, Any]) -> None:
    """Merge the D1 block into runs/c10_diagnostics_v1.json."""
    doc: dict[str, Any] = {}
    if DIAGNOSTICS_ARTIFACT.exists():
        doc = json.loads(DIAGNOSTICS_ARTIFACT.read_text())
    doc.setdefault("schema_version", "c10_diagnostics.v1")
    doc.setdefault("run", "c10_diagnostics_v1")
    doc["reported_not_gated"] = True
    doc["candidate"] = "gate-1 candidate 10 (run 12, PR #64)"
    doc["diagnostic_1_seed_extension"] = block
    DIAGNOSTICS_ARTIFACT.write_text(json.dumps(doc, indent=2) + "\n")


def _parse_seeds(spec: str) -> tuple[int, ...]:
    """Parse ``5-19`` or ``5,6,7`` or ``5-9,12`` into a seed tuple."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="5-19",
        help="Fresh seeds to score (e.g. '5-19', '5-14', '5,6,7').",
    )
    parser.add_argument(
        "--cache",
        default=str(
            Path.home() / ".claude-worktrees" / "_c10_seed_ext_cache.json"
        ),
        help="Incremental per-seed cache path (outside runs/).",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Skip generation; assemble the D1 block from the cache.",
    )
    args = parser.parse_args()

    threshold = pairs_c2st_threshold()
    cache_path = Path(args.cache)
    seeds = _parse_seeds(args.seeds)

    if args.assemble_only:
        cache = json.loads(cache_path.read_text())
        fresh = {int(k): v for k, v in cache.items()}
    else:
        fresh = compute_fresh_seeds(seeds, cache_path, verbose=True)
    # Restrict to the committed seeds (floor only; candidate from the
    # artifact) plus the requested fresh seeds, so a partial (5-14) run
    # assembles honestly and a stale cache cannot leak extra seeds.
    relevant = set(COMMITTED_SEEDS) | set(seeds)
    fresh = {s: rec for s, rec in fresh.items() if s in relevant}

    block = assemble_diagnostic_1(fresh, threshold)
    _write_diagnostic_block(block)

    d = block["candidate_c2st_distribution"]
    f = block["floor_c2st_distribution"]
    inf = block["clip_inference"]
    print(
        f"\nD1 {d['n']}-seed candidate pairs-c2st: mean={d['mean']:.4f} "
        f"sd={d['sd']:.4f} n_over_{threshold}="
        f"{block['n_candidate_over_threshold']}"
    )
    print(
        f"D1 floor pairs-c2st: mean={f['mean']:.4f} sd={f['sd']:.4f}; "
        f"excess mean={block['excess_over_floor_distribution']['mean']:+.4f}"
    )
    print(
        f"D1 inference: single-seed clip p={inf['single_seed_clip_prob_parametric']:.3f}"
        f" -> P(>=2/5 clip)={inf['run12_outcome_prob_ge2_of_5_clip_parametric']:.3f}"
        f"; mean below line by {inf['margin_below_line_in_se']:.2f} SE"
    )
    print(f"wrote {DIAGNOSTICS_ARTIFACT}")


if __name__ == "__main__":
    main()
