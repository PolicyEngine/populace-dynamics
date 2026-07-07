"""Gate-1 candidate 11 (run 13): candidate 10's spec, re-registered one-shot
under ratified amendment 2.

The THIRTEENTH pre-registered model run of PolicyEngine/populace-dynamics.
Its specification is BYTE-IDENTICAL to candidate 10 (registration comment
4902561460; ``spec_registration`` in ``runs/gate1_rank_knn_v4.json``): the
k-NN conditional rank bootstrap (k=25, distance weights 1/0.5/0.25) with
two-step-plus-anchor memory, a FIXED lambda = 0.1 calibrated blend of the
donor coordinates (Q0-exempt), and the zero-anchor participation regime refit
(no re-entry-pool restriction). No modeling constant moves; every dial is
exactly as inner-validated for candidate 10.

Why a re-registration. Amendment 2 (proposal + both referee rounds on PR #67;
ratification merge 4e06e244; flipped live in PR #69) is now live in
``gates.yaml``. Its standing ``amendment_rules.no_self_rescue`` means candidate
10's committed run-12 verdict (FAIL) does not change; a pass must come from a
FRESH registration executed one-shot under the amended gate. This runner is
that registration, pinned before the run in issue #42's candidate-11 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4905323933).

What amendment 2 changed (and this runner scores). The pairs-view ``c2st_auc``
gating moved from per-seed on the five locked seeds (0.53 each, folded into the
4-of-5 conjunction) to:

* a 20-seed MEAN rule -- the mean of pairs-view ``c2st_auc`` over pre-registered
  seeds 0-19 must be ``<= 0.53`` (``thresholds.views.psid_family_earnings_pairs
  .c2st_mean_rule.value_max``); and
* a per-seed catastrophe CAP -- no one of the 20 seeds may exceed ``0.554``
  (``.c2st_per_seed_cap.value_max``, = the version-matched ctx20 floor mean +
  8 sd under sklearn 1.8.0).

The per-seed pairs ``c2st_auc`` line is retired from the per-seed geometry
conjunction (``.per_seed_rule_superseded``); every OTHER per-seed geometry
threshold (energy_distance, prdc_coverage, q99/q90 tails, w1_over_sd on pairs;
prdc_coverage on runs), the battery, and the benefit-space block keep their
existing per-seed form on the five locked seeds under the 4-of-5 rule, and the
pooled-Q0 gate is unchanged. All thresholds are read from ``gates.yaml`` at
runtime; nothing is hardcoded against the lock.

Amended pass rule (``thresholds.protocol.pass_rule``), the full conjunction:
``gate_1_pass`` iff
  (a) >= 4/5 locked seeds pass geometry (every locked per-seed pairs+runs
      geometry threshold in the gates.yaml geometry blocks -- pairs c2st NOT
      among them -- AND every per-seed benefit-space metric), AND
  (b) >= 4/5 locked seeds pass battery, AND
  (c) the pooled Q0 gate holds (|pooled-mean Q0 %| <= 5), AND
  (d) the 20-seed pairs c2st MEAN <= 0.53, AND
  (e) the 20-seed pairs c2st per-seed MAX <= 0.554.

Runner mechanics. The candidate-10 machinery is reused EXACTLY:

* seeds 0-4 are scored by :func:`run_gate1_candidate10.run_seed` verbatim (the
  filter-first load, the person-disjoint 0.2 split, all geometry views, the
  battery on the candidate panel vs the committed reference, the gated
  benefit-space block, the pooled-Q0 diagnostic, and the per-seed storage --
  byte-for-byte identical in shape to ``runs/gate1_rank_knn_v4.json``'s
  ``per_seed``). Because ``run_seed`` reads the pairs geometry block from the
  live (amended) ``gates.yaml``, which no longer carries ``c2st_auc_max``, the
  per-seed pairs geometry check naturally excludes c2st -- while
  ``panel_scorecard`` still computes the ``c2st_auc`` SCORE into ``scores``.
* seeds 5-19 generate the candidate and score ONLY the pairs-view ``c2st_auc``,
  via :func:`c10_seed_extension.candidate_pairs_c2st` (byte-for-byte the
  20-seed extension's scoring path).
* the battery-reference bit-exact precheck is
  :func:`run_gate1_candidate10.reproduce_battery_reference` unchanged (hard-stop
  on any mismatch, before any candidate is scored).

Determinism and the one-shot rule. Same seeds, same data, same pinned
environment (sklearn 1.8.0 per the ratified ``amendment_rules
.classifier_version_pin``; the pe-us pin unchanged). The 20 per-seed pairs c2st
scores for this exact spec are already public in ``runs/c10_diagnostics_v1.json``
(mean 0.5234, max 0.5330), and the locked-five blocks are deterministic
re-scores of run 12; scores should reproduce run 12 and the diagnostics
bit-exactly. The artifact carries a REPRODUCTION block comparing every score
against those committed baselines. Per the registration, ANY deviation is a
FINDING, not noise: this run does not re-run to remove deviations; it publishes
them, prominently, in the artifact and the PR body.

Environment. The participation gates are ``RegimeGatedQRF`` sign gates and need
populace-fit; the benefit-space block needs the SSA oracle
(``POPULACE_DYNAMICS_PE_US_DIR`` -> the pinned policyengine-us checkout). Run
the full gate from the repository root with the PSID family files staged, using
the DEDICATED gate venv (populace-fit pins scikit-learn < 1.9)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_gate1_candidate11.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import c10_seed_extension as c10ext
import numpy as np

# The candidate-10 runner supplies the ENTIRE scoring machinery byte-for-byte
# (it in turn imports the merged baseline runner, candidate 5b/7/8/9, and the
# inner sweep). The 20-seed extension supplies the pairs-only scoring path.
# Both import cleanly under the repo .venv (they defer every populace-fit
# import), so the artifact-only tests need no gate venv.
import run_gate1_candidate10 as c10

ROOT = c10.ROOT
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_knn_v5.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank_knn_v5"
RUN_NAME = "gate1_rank_knn_v5"

#: This run's frozen-spec registration (candidate 11 = candidate 10's spec,
#: re-registered under amendment 2). Pinned before the run.
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4905323933"
)
#: The candidate-10 registration whose spec this run re-registers byte-for-byte.
CANDIDATE10_SPEC_REGISTRATION = c10.SPEC_REGISTRATION
#: Provenance of the reused machinery (inherited from candidate 10).
BASE_REGISTRATION = c10.BASE_REGISTRATION
UW_REGISTRATION = c10.UW_REGISTRATION
C9_REGISTRATION = c10.C9_REGISTRATION

#: Reproduction baselines: the committed run-12 gate artifact (locked-five
#: re-score reference) and the committed 20-seed diagnostics (pairs-c2st
#: reference for every seed 0-19).
RUN12_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v4.json"
DIAGNOSTICS_ARTIFACT = ROOT / "runs" / "c10_diagnostics_v1.json"

PAIRS_VIEW = "psid_family_earnings_pairs"

#: The pre-registered locked seeds (full scoring) and the extension seeds
#: (pairs c2st only). Their union is the 20-seed set the mean rule + cap score.
LOCKED_SEEDS = tuple(c10.SEEDS)
EXTENSION_SEEDS = tuple(range(5, 20))
ALL_SEEDS = tuple(range(20))

#: Float-equality tolerance for the reproduction attestation (bit-exact).
EXACT_ATOL = 1e-12

#: Incremental per-seed cache (OUTSIDE runs/), so a crash can be fixed and
#: relaunched WITHOUT re-scoring any already-scored seed -- honouring the
#: one-shot rule (no scored output influences a fix; a resume reuses the
#: verbatim scored output). Never committed.
DEFAULT_CACHE = Path.home() / ".claude-worktrees" / "_c11_run13_cache.json"


def _json_default(o: Any) -> Any:
    """Numpy-aware JSON encoder for the per-seed cache (float-exact)."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, default=_json_default) + "\n"
    )


# --------------------------------------------------------------------------
# Reproduction helpers (compare freshly-scored values to committed baselines)
# --------------------------------------------------------------------------
def _abs_dev(a: float | None, b: float | None) -> float:
    """Absolute deviation with None handling (a missing value is a finding)."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(float(a) - float(b))


def _block_deviation(
    mine: dict[str, Any], ref: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """Max abs deviation over the union of keys, plus the per-key deviations."""
    keys = sorted(set(mine) | set(ref))
    devs = {k: _abs_dev(mine.get(k), ref.get(k)) for k in keys}
    mx = max(devs.values()) if devs else 0.0
    return float(mx), devs


def _benefit_metric_values(seed_result: dict[str, Any]) -> dict[str, float]:
    """The per-seed benefit metrics compared for reproduction.

    The gated per-seed benefit-space check VALUES (abs mean %, abs median %,
    each gated decile d3-d9 %, weighted KS) plus the headline Q0 subgroup mean
    PIA-proxy % gap -- the numbers the amendment's benefit block turns on.
    """
    out: dict[str, float] = {}
    checks = seed_result.get("benefit_space_checks") or {}
    for name, chk in checks.items():
        out[name] = chk.get("value")
    bs = seed_result.get("benefit_space")
    if bs is not None:
        q0 = bs["by_anchor_quintile"]["quintiles"].get("Q0", {})
        out["q0_mean_pct_diff"] = (
            q0["distribution"]["mean"]["pct_diff"]
            if q0.get("n_persons")
            else None
        )
    return out


def _geometry_score_values(seed_result: dict[str, Any]) -> dict[str, float]:
    """Flatten every geometry SCORE across both locked views (view.metric)."""
    out: dict[str, float] = {}
    for vname, block in seed_result["geometry"].items():
        for metric, score in block["scores"].items():
            out[f"{vname}.{metric}"] = score
    return out


def build_reproduction_block(
    per_seed_locked: list[dict[str, Any]],
    pairs_c2st_by_seed: dict[int, float],
    pooled_q0: dict[str, Any],
) -> dict[str, Any]:
    """Compare every freshly-scored value to its committed baseline.

    Two comparisons, each recording per-item deviations, a max abs deviation,
    and a bit-exact (``<= 1e-12``) attestation:

    * pairs-view c2st for ALL 20 seeds vs
      ``runs/c10_diagnostics_v1.json`` ``diagnostic_1_seed_extension.per_seed
      [].candidate_pairs_c2st``;
    * each locked seed's battery values, benefit metrics, and geometry scores
      vs the matching seed in ``runs/gate1_rank_knn_v4.json`` (plus the pooled
      Q0 mean gap vs run 12's verdict).

    Per the registration, deviations are FINDINGS, recorded not suppressed.
    """
    diag = json.loads(DIAGNOSTICS_ARTIFACT.read_text())
    diag_by_seed = {
        int(r["seed"]): float(r["candidate_pairs_c2st"])
        for r in diag["diagnostic_1_seed_extension"]["per_seed"]
    }
    pairs_per_seed = []
    pairs_max = 0.0
    for s in ALL_SEEDS:
        this = float(pairs_c2st_by_seed[s])
        comm = diag_by_seed[s]
        dev = abs(this - comm)
        pairs_max = max(pairs_max, dev)
        pairs_per_seed.append(
            {
                "seed": s,
                "source": (
                    "locked_full_scoring"
                    if s in LOCKED_SEEDS
                    else "extension_pairs_only"
                ),
                "this_run": this,
                "committed": comm,
                "abs_deviation": dev,
            }
        )
    pairs_block = {
        "baseline": "runs/c10_diagnostics_v1.json"
        " :: diagnostic_1_seed_extension.per_seed[].candidate_pairs_c2st",
        "per_seed": pairs_per_seed,
        "max_abs_deviation": float(pairs_max),
        "exact_match": bool(pairs_max <= EXACT_ATOL),
    }

    run12 = json.loads(RUN12_ARTIFACT.read_text())
    run12_by_seed = {s["seed"]: s for s in run12["per_seed"]}
    locked_per_seed = []
    bat_max = ben_max = geo_max = 0.0
    for res in per_seed_locked:
        s = res["seed"]
        ref = run12_by_seed[s]
        b_mx, _ = _block_deviation(
            res["battery_values"], ref["battery_values"]
        )
        n_mx, _ = _block_deviation(
            _benefit_metric_values(res), _benefit_metric_values(ref)
        )
        g_mx, _ = _block_deviation(
            _geometry_score_values(res), _geometry_score_values(ref)
        )
        bat_max = max(bat_max, b_mx)
        ben_max = max(ben_max, n_mx)
        geo_max = max(geo_max, g_mx)
        locked_per_seed.append(
            {
                "seed": s,
                "battery": {
                    "max_abs_deviation": b_mx,
                    "exact_match": bool(b_mx <= EXACT_ATOL),
                },
                "benefit_metrics": {
                    "max_abs_deviation": n_mx,
                    "exact_match": bool(n_mx <= EXACT_ATOL),
                },
                "geometry_scores": {
                    "max_abs_deviation": g_mx,
                    "exact_match": bool(g_mx <= EXACT_ATOL),
                },
            }
        )

    # Pooled Q0 mean gap vs run 12 (the locked-five deterministic re-score).
    my_pooled = pooled_q0.get("pooled_q0_mean_pct_diff")
    ref_pooled = run12["verdict"].get("pooled_q0_mean_pct_diff")
    pooled_dev = _abs_dev(my_pooled, ref_pooled)
    pooled_block = {
        "this_run": my_pooled,
        "committed": ref_pooled,
        "abs_deviation": pooled_dev,
        "exact_match": bool(pooled_dev <= EXACT_ATOL),
    }

    locked_block = {
        "baseline": "runs/gate1_rank_knn_v4.json :: per_seed[]",
        "per_seed": locked_per_seed,
        "battery_max_abs_deviation": float(bat_max),
        "benefit_metrics_max_abs_deviation": float(ben_max),
        "geometry_scores_max_abs_deviation": float(geo_max),
        "pooled_q0": pooled_block,
        "exact_match": bool(
            bat_max <= EXACT_ATOL
            and ben_max <= EXACT_ATOL
            and geo_max <= EXACT_ATOL
            and pooled_dev <= EXACT_ATOL
        ),
    }

    all_exact = bool(
        pairs_block["exact_match"] and locked_block["exact_match"]
    )
    return {
        "one_shot_rule": (
            "Same seeds, same data, same pinned environment (sklearn 1.8.0; "
            "pe-us pin unchanged): scores should reproduce run 12 and the "
            "20-seed diagnostics bit-exactly. ANY deviation is a FINDING, "
            "recorded here and in the PR body, NOT noise and NOT re-run away "
            "(registration: one shot). exact_match asserts every deviation "
            "<= 1e-12."
        ),
        "baselines": {
            "run12_artifact": "runs/gate1_rank_knn_v4.json",
            "diagnostics_artifact": "runs/c10_diagnostics_v1.json",
        },
        "pairs_c2st_20_seed": pairs_block,
        "locked_seed_blocks": locked_block,
        "all_exact_match": all_exact,
    }


# --------------------------------------------------------------------------
# Per-seed scoring (cached)
# --------------------------------------------------------------------------
def _score_locked_seed(
    seed: int,
    cache: dict[str, Any],
    cache_path: Path,
    panel: Any,
    all_anchor: Any,
    view_specs: dict[str, Any],
    views_cfg: dict[str, Any],
    battery_reference: dict[str, float],
    battery_tol: dict[str, float],
    benefit_metrics_cfg: dict[str, Any],
    benefit_params: Any,
    benefit_cutpoints: Any,
    verbose: bool,
) -> dict[str, Any]:
    """Full candidate-10 scoring for one locked seed (cached, resumable)."""
    key = f"locked_{seed}"
    if key in cache:
        if verbose:
            print(f"seed {seed}: cached (locked full scoring)")
        return cache[key]
    result = c10.run_seed(
        seed,
        panel,
        all_anchor,
        view_specs,
        views_cfg,
        battery_reference,
        battery_tol,
        benefit_metrics_cfg,
        benefit_params,
        benefit_cutpoints,
        verbose,
    )
    cache[key] = json.loads(json.dumps(result, default=_json_default))
    _save_cache(cache_path, cache)
    return cache[key]


def _score_extension_seed(
    seed: int,
    cache: dict[str, Any],
    cache_path: Path,
    panel: Any,
    all_anchor: Any,
    pairs_view: Any,
    verbose: bool,
) -> dict[str, Any]:
    """Pairs-only candidate c2st for one extension seed (cached, resumable)."""
    key = f"ext_{seed}"
    if key in cache:
        if verbose:
            print(
                f"seed {seed}: cached "
                f"(pairs c2st={cache[key]['pairs_c2st']:.6f})"
            )
        return cache[key]
    t0 = time.time()
    rec = c10ext.candidate_pairs_c2st(seed, panel, all_anchor, pairs_view)
    cache[key] = json.loads(json.dumps(rec, default=_json_default))
    _save_cache(cache_path, cache)
    if verbose:
        print(
            f"seed {seed}: pairs c2st={rec['pairs_c2st']:.6f} "
            f"({time.time() - t0:.0f}s)"
        )
    return cache[key]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-11 run (amendment 2)."""
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = _load_cache(cache_path)

    thresholds = c10.load_gate1_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_1 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    views_cfg = thresholds["views"]
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }
    benefit_cfg = thresholds.get("benefit_space")
    if benefit_cfg is None:
        raise RuntimeError(
            "gates.yaml gate_1 thresholds carry no benefit_space block; the "
            "amended gate cannot be scored."
        )
    benefit_metrics_cfg = benefit_cfg["metrics"]

    # The amended pairs-view classifier rules, read from gates.yaml at runtime.
    pairs_cfg = views_cfg[PAIRS_VIEW]
    mean_rule_cfg = pairs_cfg.get("c2st_mean_rule")
    cap_cfg = pairs_cfg.get("c2st_per_seed_cap")
    if mean_rule_cfg is None or cap_cfg is None:
        raise RuntimeError(
            "gates.yaml pairs view carries no c2st_mean_rule / "
            "c2st_per_seed_cap; expected ratified amendment 2 (PR #67/#69) "
            "to be live."
        )
    mean_rule_max = float(mean_rule_cfg["value_max"])
    cap_max = float(cap_cfg["value_max"])
    mean_rule_seed_set = [int(s) for s in mean_rule_cfg["seed_set"]]
    if sorted(mean_rule_seed_set) != list(ALL_SEEDS):
        raise RuntimeError(
            "c2st_mean_rule.seed_set is not the pre-registered 0-19; refusing "
            f"to score a mismatched seed set (got {sorted(mean_rule_seed_set)})."
        )

    battery_ref_artifact = json.loads(
        (ROOT / c10.BATTERY_REFERENCE_RUN).read_text()
    )
    battery_reference = battery_ref_artifact["battery_reference"]

    panel = c10.load_filtered_panel()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    # Battery-reference bit-exact precheck, identical to the c10 runner: the
    # battery code path must reproduce every committed reference value to float
    # precision before ANY candidate is scored. Hard-stop on mismatch.
    repro = c10.reproduce_battery_reference(panel)
    if verbose:
        print(
            "battery_reference reproduced exactly: "
            f"{repro['all_committed_values_reproduced_exactly']}"
        )
    if not repro["all_committed_values_reproduced_exactly"]:
        raise RuntimeError(
            "Battery code path does not reproduce the committed "
            "battery_reference to float precision; refusing to proceed."
        )

    all_anchor = c10.anchor_rows(panel)

    # The amended gate scores a GATED benefit-space block; the SSA oracle must
    # be present. Refuse to publish a verdict without it.
    benefit_params, benefit_cutpoints = c10._load_benefit_oracle()
    if benefit_params is None:
        raise RuntimeError(
            "The amended gate scores a GATED benefit-space block, but the SSA "
            "oracle did not load (set POPULACE_DYNAMICS_PE_US_DIR to the "
            "pinned policyengine-us checkout and rerun)."
        )
    if verbose:
        print(
            "benefit-space oracle (GATED): pe_us_revision="
            f"{benefit_params.pe_us_revision}"
        )

    view_specs = {
        "psid_family_earnings_pairs": c10.build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": c10.build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }
    pairs_view = c10ext.build_pairs_view()

    # --- seeds 0-4: full candidate-10 scoring (byte-for-byte) ---
    per_seed: list[dict[str, Any]] = []
    for seed in LOCKED_SEEDS:
        per_seed.append(
            _score_locked_seed(
                seed,
                cache,
                cache_path,
                panel,
                all_anchor,
                view_specs,
                views_cfg,
                battery_reference,
                battery_tol,
                benefit_metrics_cfg,
                benefit_params,
                benefit_cutpoints,
                verbose,
            )
        )

    # --- seeds 5-19: pairs-view c2st only (mirror the 20-seed extension) ---
    ext_records: dict[int, dict[str, Any]] = {}
    for seed in EXTENSION_SEEDS:
        ext_records[seed] = _score_extension_seed(
            seed,
            cache,
            cache_path,
            panel,
            all_anchor,
            pairs_view,
            verbose,
        )

    # --- the 20-seed pairs c2st vector (locked from full scoring; extension
    #     from pairs-only), in seed order 0..19 ---
    pairs_c2st_by_seed: dict[int, float] = {}
    for res in per_seed:
        pairs_c2st_by_seed[res["seed"]] = float(
            res["geometry"][PAIRS_VIEW]["scores"]["c2st_auc"]
        )
    for seed, rec in ext_records.items():
        pairs_c2st_by_seed[seed] = float(rec["pairs_c2st"])
    pairs_vec = np.array(
        [pairs_c2st_by_seed[s] for s in ALL_SEEDS], dtype=np.float64
    )
    c2st_mean_20 = float(pairs_vec.mean())
    c2st_max_20 = float(pairs_vec.max())

    # --- the five amended sub-verdicts (each with score + threshold) ---
    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    pooled_q0 = c10.check_pooled_q0(per_seed, benefit_metrics_cfg)
    pooled_q0_pass = bool(pooled_q0["pooled_q0_pass"])
    mean_rule_pass = c2st_mean_20 <= mean_rule_max
    cap_pass = c2st_max_20 <= cap_max
    gate_pass = bool(
        geometry_gate_pass
        and battery_gate_pass
        and pooled_q0_pass
        and mean_rule_pass
        and cap_pass
    )

    sub_verdicts = {
        "a_geometry_4_of_5": {
            "score": f"{n_geo}/5",
            "n_pass": n_geo,
            "n_seeds": 5,
            "threshold": ">= 4/5",
            "pass": bool(geometry_gate_pass),
            "note": (
                "locked per-seed pairs+runs geometry thresholds (pairs c2st "
                "NOT among them, per per_seed_rule_superseded) AND per-seed "
                "benefit-space metrics"
            ),
        },
        "b_battery_4_of_5": {
            "score": f"{n_bat}/5",
            "n_pass": n_bat,
            "n_seeds": 5,
            "threshold": ">= 4/5",
            "pass": bool(battery_gate_pass),
        },
        "c_pooled_q0": {
            "score": pooled_q0["pooled_q0_mean_pct_diff"],
            "abs_score": pooled_q0["abs_pooled_q0_mean_pct_diff"],
            "threshold": pooled_q0["threshold"],
            "comparison": "|.| <=",
            "pass": pooled_q0_pass,
        },
        "d_c2st_mean_rule": {
            "score": c2st_mean_20,
            "threshold": mean_rule_max,
            "comparison": "<=",
            "seed_set": list(ALL_SEEDS),
            "n_seeds": len(ALL_SEEDS),
            "pass": bool(mean_rule_pass),
        },
        "e_c2st_per_seed_cap": {
            "score": c2st_max_20,
            "threshold": cap_max,
            "comparison": "<= (per-seed max over 20)",
            "n_seeds": len(ALL_SEEDS),
            "pass": bool(cap_pass),
        },
    }

    benefit_pooled = c10._pool_benefit_space(per_seed)
    q0_participation = c10._q0_participation_diagnostics(per_seed, panel)

    reproduction = build_reproduction_block(
        per_seed, pairs_c2st_by_seed, pooled_q0
    )
    if verbose and not reproduction["all_exact_match"]:
        print(
            "REPRODUCTION FINDING: a value deviated from the committed "
            "baselines; see the reproduction block (published, not re-run)."
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_1",
        "gate_variant": (
            "amendment 2 (PR #67 ratified 4e06e244, flipped live #69): pairs-"
            "view c2st_auc gating moved from per-seed on the five locked seeds "
            "to the 20-seed MEAN rule (mean over seeds 0-19 <= 0.53) plus a "
            "version-matched per-seed CAP (each of 20 seeds <= 0.554); the "
            "per-seed pairs c2st line is retired from the geometry conjunction "
            "(per_seed_rule_superseded). Everything else per-seed on the five "
            "locked seeds is unchanged: >= 4/5 geometry (other pairs+runs "
            "thresholds AND per-seed benefit-space) AND >= 4/5 battery AND "
            "pooled Q0, PLUS the 20-seed mean rule AND per-seed cap. Spec "
            "byte-identical to candidate 10; no modeling constant moved."
        ),
        "spec_registration": SPEC_REGISTRATION,
        "candidate10_spec_registration": CANDIDATE10_SPEC_REGISTRATION,
        "base_registration": BASE_REGISTRATION,
        "uw_registration": UW_REGISTRATION,
        "c9_registration": C9_REGISTRATION,
        "pre_registered_forecast": {
            "p_pass": 0.97,
            "registration": SPEC_REGISTRATION,
            "note": (
                "residual mass entirely on execution/environment error "
                "(runner defect, environment drift, artifact-schema "
                "mismatch), not on the statistics, which are known: the 20 "
                "per-seed pairs c2st are public in runs/c10_diagnostics_v1"
                ".json (mean 0.5234 <= 0.53, max 0.5330 <= 0.554); the "
                "locked-five blocks are deterministic re-scores of run 12 "
                "(battery 4/5, benefit-space 5/5, pooled Q0 +0.0376%, other "
                "pairs geometry 5/5, runs coverage 5/5). Modal failure mode "
                "if it fails: a reproduction mismatch, itself published as a "
                "finding."
            ),
        },
        "changes": (
            "NONE to the model: the candidate is byte-identical to candidate "
            "10 (the inner-validated composition -- k-NN conditional rank "
            "bootstrap, k=25, 1/0.5/0.25 lag weights, FIXED lambda=0.1 donor "
            "blend for non-Q0 targets, zero-anchor participation regime with "
            "a full re-entry pool and Q0 memory-exemption). The ONLY change "
            "is the gate estimator (amendment 2): pairs-view c2st_auc is "
            "scored as a 20-seed mean plus a per-seed cap instead of per-seed "
            "on the five locked seeds. spec_registration is a re-registration "
            "of candidate 10's spec under the amended gate."
        ),
        "description": (
            "Gate-1 candidate 11 (run 13): candidate 10's inner-validated "
            "composition re-registered one-shot under ratified amendment 2. "
            "Seeds 0-4 are scored by run_gate1_candidate10.run_seed verbatim "
            "(all geometry views, battery, gated benefit-space, pooled Q0; "
            "per-seed storage byte-identical in shape to gate1_rank_knn_v4). "
            "Because run_seed reads the amended gates.yaml pairs geometry "
            "block, which no longer carries c2st_auc_max, the per-seed pairs "
            "geometry check excludes c2st while panel_scorecard still computes "
            "the c2st_auc score. Seeds 5-19 generate the candidate and score "
            "ONLY the pairs-view c2st_auc (c10_seed_extension.candidate_pairs"
            "_c2st, byte-for-byte the 20-seed measurement's path). The verdict "
            "reads the amended pass_rule from gates.yaml at runtime: >= 4/5 "
            "geometry AND >= 4/5 battery AND pooled Q0 AND 20-seed mean <= "
            "0.53 AND per-seed max <= 0.554. A reproduction block compares "
            "every score to the committed baselines (run 12 for the locked "
            "five; the 20-seed diagnostics for pairs c2st); any deviation is "
            "published as a finding, not re-run away."
        ),
        "model": _model_block(),
        "protocol": {
            "filter": (
                f"age {c10.AGE_MIN}-{c10.AGE_MAX}, reference years "
                f"{c10.PERIOD_MIN}-{c10.PERIOD_MAX}, positive weights "
                "(applied before the split)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel, 'person_id', fraction=0.2, seed=s); the drawn 20% is "
                "the holdout, the complement is the training set (imported "
                "from the baseline runner)"
            ),
            "locked_seeds": list(LOCKED_SEEDS),
            "extension_seeds": list(EXTENSION_SEEDS),
            "mean_rule_seed_set": list(ALL_SEEDS),
            "views": {
                "psid_family_earnings_pairs": {"window": 2, "period_step": 2},
                "psid_family_earnings_runs": {"window": 3, "period_step": 2},
            },
            "scoring": (
                "seeds 0-4: panel_scorecard per locked view + battery + gated "
                "benefit-space (run_gate1_candidate10.run_seed); seeds 5-19: "
                "pairs-view c2st_auc only (c10_seed_extension"
                ".candidate_pairs_c2st)"
            ),
            "pass_rule": (
                "AMENDED (gates.yaml, ratified 2026-07-06 and 2026-07-07): a "
                "locked seed passes geometry iff every locked per-seed pairs+"
                "runs geometry threshold holds (pairs c2st_auc NOT among them; "
                "see per_seed_rule_superseded) AND every per-seed benefit-"
                "space metric holds; a locked seed passes battery iff every "
                "locked tolerance holds. gate_1_pass iff >= 4/5 geometry AND "
                ">= 4/5 battery AND the pooled Q0 gate (|pooled-mean Q0 %| <= "
                "5) AND the pairs-view c2st_mean_rule (mean over seeds 0-19 "
                "<= 0.53) AND the c2st_per_seed_cap (each of the 20 seeds <= "
                "0.554)."
            ),
        },
        "battery_reference_reproduction": repro,
        "battery_reference_run": c10.BATTERY_REFERENCE_RUN,
        "per_seed": per_seed,
        "pairs_c2st_seed_set": {
            "note": (
                "the 20-seed pairs-view c2st_auc vector the amended mean rule "
                "and per-seed cap score. Seeds 0-4 from the full locked "
                "scoring (per_seed[].geometry.psid_family_earnings_pairs"
                ".scores.c2st_auc); seeds 5-19 from the pairs-only extension."
            ),
            "seed_set": list(ALL_SEEDS),
            "mean_rule_value_max": mean_rule_max,
            "per_seed_cap_value_max": cap_max,
            "per_seed": [
                {
                    "seed": s,
                    "c2st_auc": pairs_c2st_by_seed[s],
                    "source": (
                        "locked_full_scoring"
                        if s in LOCKED_SEEDS
                        else "extension_pairs_only"
                    ),
                }
                for s in ALL_SEEDS
            ],
            "mean": c2st_mean_20,
            "max": c2st_max_20,
            "n_seeds": len(ALL_SEEDS),
        },
        "seed_conjunction": [
            {
                "seed": s["seed"],
                "lambda": s["lambda"],
                "geometry_thresholds_pass": s["geometry_thresholds_pass"],
                "benefit_space_seed_pass": s.get("benefit_space_seed_pass"),
                "geometry_pass": s["geometry_pass"],
                "battery_pass": s["battery_pass"],
                "pairs_c2st": pairs_c2st_by_seed[s["seed"]],
            }
            for s in per_seed
        ],
        "lambda_by_seed": {str(s["seed"]): s["lambda"] for s in per_seed},
        "knn_context": _knn_context_block(per_seed),
        "benefit_space_gated": {
            "note": (
                "GATED under amendment 1. Per-seed metrics fold into the "
                "geometry verdict; the pooled Q0 gate is a standalone gate "
                "condition."
            ),
            "metrics_config": benefit_metrics_cfg,
            "pooled_q0_gate": pooled_q0,
            "pooled": benefit_pooled,
            "per_seed_checks": [
                {
                    "seed": s["seed"],
                    "benefit_space_seed_pass": s.get(
                        "benefit_space_seed_pass"
                    ),
                    "checks": s.get("benefit_space_checks"),
                }
                for s in per_seed
            ],
        },
        "q0_participation_diagnostics": q0_participation,
        "reproduction": reproduction,
        "verdict": {
            "n_seeds_locked": len(LOCKED_SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "pooled_q0_pass": pooled_q0_pass,
            "pooled_q0_mean_pct_diff": pooled_q0["pooled_q0_mean_pct_diff"],
            "c2st_mean_20": c2st_mean_20,
            "c2st_mean_rule_value_max": mean_rule_max,
            "c2st_mean_rule_pass": bool(mean_rule_pass),
            "c2st_max_20": c2st_max_20,
            "c2st_per_seed_cap_value_max": cap_max,
            "c2st_per_seed_cap_pass": bool(cap_pass),
            "sub_verdicts": sub_verdicts,
            "gate_1_pass": gate_pass,
            "rule": (
                ">= 4/5 seeds geometry (locked pairs+runs geometry thresholds, "
                "pairs c2st excluded, AND per-seed benefit-space) AND >= 4/5 "
                "seeds battery AND pooled Q0 gate AND 20-seed pairs c2st mean "
                "<= 0.53 AND 20-seed pairs c2st per-seed max <= 0.554"
            ),
            "reproduction_all_exact_match": reproduction["all_exact_match"],
        },
        "revision_pins": _revision_pins(benefit_params),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5, "
            f"pooled_Q0={v['pooled_q0_mean_pct_diff']:+.4f} "
            f"pass={pooled_q0_pass}; "
            f"c2st mean_20={c2st_mean_20:.4f}<={mean_rule_max:.2f} "
            f"pass={mean_rule_pass}; "
            f"c2st max_20={c2st_max_20:.4f}<={cap_max:.3f} "
            f"pass={cap_pass})"
        )
        print(
            "reproduction all_exact_match="
            f"{reproduction['all_exact_match']} "
            f"(pairs max dev={reproduction['pairs_c2st_20_seed']['max_abs_deviation']:.2e}, "
            f"locked battery/benefit/geometry max dev="
            f"{reproduction['locked_seed_blocks']['battery_max_abs_deviation']:.2e}/"
            f"{reproduction['locked_seed_blocks']['benefit_metrics_max_abs_deviation']:.2e}/"
            f"{reproduction['locked_seed_blocks']['geometry_scores_max_abs_deviation']:.2e})"
        )
    return artifact


def _model_block() -> dict[str, Any]:
    """The candidate-11 model block (byte-identical spec to candidate 10)."""
    return {
        "class": (
            "k-NN conditional rank bootstrap with a FIXED anchor/permanent-"
            "rank donor blend (lambda = 0.1) for non-Q0 targets and a zero-"
            "anchor participation regime with a full (unrestricted) re-entry "
            "pool and Q0 memory-exemption -- byte-identical to candidate 10"
        ),
        "spec_identical_to_candidate_10": True,
        "modeling_constants_changed": "none",
        "stochastic": True,
        "populace_fit_used": True,
        "calibration": (
            "none in this run: lambda is the FIXED constant 0.1 (the inner-"
            "sweep winner), not chosen against any score. Byte-identical to "
            "candidate 10; no dial moves under amendment 2."
        ),
        "inner_validation": {
            "this_candidate_is": (
                "the inner sweep's V1-lam0.1 variant at OUTER scale "
                "(candidate 10's spec, unchanged)"
            ),
            "sweep_artifact": "runs/inner_sweep_v1.json",
            "harness": "scripts/inner_validation.py",
        },
        "change_1_fixed_donor_blend": {
            "third_distance_term_nonq0": (
                "|0.1*u_w(donor) + 0.9*u_A(donor) - u_A(target)| at weight "
                "0.25 (transitions) / bare (re-entry); the TARGET side stays "
                "u_A, exactly as candidate 7/8/9/10"
            ),
            "third_distance_term_q0": (
                "|u_A(donor) - u_A(target)| (candidate 7 verbatim; Q0 targets "
                "are memory-exempt, no blend)"
            ),
            "lambda": c10.LAMBDA_FIXED,
            "lambda_fixed": True,
        },
        "change_2_zero_anchor_participation_regime": {
            "participation_gate": (
                "a RegimeGatedQRF sign gate refit ONLY on train pairs whose "
                "person has zero anchor earnings (same features, same "
                "populace-fit defaults; candidate 9's component 2, kept "
                "verbatim by candidate 10)"
            ),
            "reentry_pool": (
                "the FULL re-entry pool serves every target -- NO zero-anchor "
                "restriction"
            ),
            "q0_memory_exempt": (
                "Q0 (zero-anchor) targets are exempt from every memory "
                "coordinate: their third distance term stays candidate 7's "
                "|u_A(donor) - u_A(target)|"
            ),
        },
        "knn": {
            "k": c10.K_NEIGHBORS,
            "distance_pairs_nonq0": (
                "|u_next - v1| + 0.25|0.1*u_w + 0.9*u_A - a|"
            ),
            "distance_pairs_q0": "|u_next - v1| + 0.25|u_A - a|",
            "weights": {
                "w_next": c10.W_NEXT,
                "w_next2": c10.W_NEXT2,
                "w_anchor": c10.W_ANCHOR,
            },
            "draw": (
                "one record drawn with probability proportional to its weight "
                "among the k=25 nearest (seeded substream); byte-for-byte "
                "candidate 7's _knn_draw"
            ),
        },
    }


def _knn_context_block(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Reported-not-gated per-seed diagnostics (same shape as v4)."""
    return {
        "note": (
            "Reported-not-gated per seed: the u_w decomposition, neighbor-"
            "distance distribution, triple-vs-pair usage share, donor-record "
            "reuse, drawn corner mass by anchor quintile, and the clamped-rank "
            "share. lambda is the fixed constant 0.1 (no calibration). None of "
            "these enters the pass/fail beyond the amended gate's named blocks."
        ),
        "lambda": c10.LAMBDA_FIXED,
        "lambda_fixed": True,
        "per_seed": [
            {
                "seed": s["seed"],
                "lambda": s["lambda"],
                "n_pairs": s["pools"]["n_pairs"],
                "n_triples": s["pools"]["n_triples"],
                "n_reentry": s["pools"]["n_reentry"],
                "n_reentry_q0_built_unused": s["pools"]["n_reentry_q0"],
                "n_zero_anchor_train_pairs": s["n_zero_anchor_train_pairs"],
                "uw_fit": s["uw_fit"],
                "neighbor_distance_distribution": s["generation_diagnostics"][
                    "neighbor_distance_distribution"
                ],
                "triple_pair_usage": s["generation_diagnostics"][
                    "triple_pair_usage"
                ],
                "donor_reuse": s["generation_diagnostics"]["donor_reuse"],
                "drawn_corner_mass_by_anchor_quintile": s[
                    "generation_diagnostics"
                ]["drawn_corner_mass_by_anchor_quintile"],
                "anchor_rank_distribution": s["generation_diagnostics"][
                    "anchor_rank_distribution"
                ],
                "cell_count_distribution": s["generation_diagnostics"][
                    "cell_count_distribution"
                ],
                "clamped_rank_share": s["generation_diagnostics"][
                    "clamped_rank_share"
                ]["share"],
            }
            for s in per_seed
        ],
    }


def _revision_pins(benefit_params: Any) -> dict[str, Any]:
    """Repo/populace SHAs, schema version, pe-us pin, and the sklearn version.

    The sklearn version is RECORDED per the ratified
    ``amendment_rules.classifier_version_pin``: the C2ST HistGradientBoosting
    classifier is scikit-learn-version-sensitive, so every gate-run artifact
    records the version that scored it (must match the floor derivation, 1.8.0).
    """
    import sklearn

    pins = c10._revision_pins(benefit_params)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    pins["gates_yaml_locked"] = True
    pins["sklearn_version"] = str(sklearn.__version__)
    return pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    artifact = run(verbose=True, cache_path=Path(args.cache))
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
