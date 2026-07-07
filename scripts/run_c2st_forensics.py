"""C2ST forensics: what the gate classifier reads at 0.547.

REPORTED-NOT-GATED, NO HOLDOUT CONTACT. This is a forensic analysis of an
already-run pair of gate-1 candidates, not a gate run. It regenerates each
candidate's seed-0 panel deterministically from the merged runners
(``run_gate1_candidate5a2`` for segment splicing, PR #51;
``run_gate1_candidate6`` for the anchored empirical rank-transition kernel,
PR #53), then compares each candidate against the seed-0 TRAIN persons'
real windows only -- never the holdout. The locked gate and the committed
``runs/`` artifacts are untouched.

Both candidates land pairs-view C2ST ~= 0.547 against the holdout (splice
0.5456, kernel 0.5470; the runs-view kernel is higher at 0.5984). The
question this answers: is that a single shared residual signal, and what
coordinate does the classifier actually read? The analysis:

1. Sanity anchor -- the mirrored classifier reproduces the harness C2ST
   exactly against the holdout (bit-for-bit), then re-scores vs train.
2. Feature attribution -- full / per-feature marginal / pairwise AUCs.
3. Distributional forensics -- round-number footprint, marginal quantiles,
   joint (log-persistence) dependence; a rounding-repair ablation.
4. Decision-region probe -- the top-confidence candidate windows profiled.
5. Shared-signal test -- splice-vs-kernel C2ST and the real-vs-real
   (train-half vs train-half) noise floor.

Run under the gate venv (needs populace-fit for the kernel's regime gate;
the splice needs no fit but imports cleanly there too):

    .venv-gate/bin/python scripts/run_c2st_forensics.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import c2st_forensics_lib as fx
import numpy as np
import run_gate1_baseline as B
import run_gate1_candidate5a2 as SPLICE
import run_gate1_candidate6 as KERNEL

from populace_dynamics.harness import metrics
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "c2st_forensics_v1.json"
ARTIFACT_SCHEMA_VERSION = "c2st_forensics.v1"
SEED = 0

SPEC = (
    "https://github.com/PolicyEngine/populace-dynamics/pull/51 (splicing) "
    "and https://github.com/PolicyEngine/populace-dynamics/pull/53 (kernel)"
)


def _build_panels() -> dict[str, Any]:
    """Regenerate the seed-0 candidates and the seed-0 train real panel.

    Deterministic: the split is ``split_panel_by_person(..., seed=0)`` (the
    locked protocol split), and both generators are seeded off the gate
    seed. The splice needs no model; the kernel fits its per-cell
    marginals, counts its kernel on the train pairs, and fits the
    participation regime gate (populace-fit). The holdout is never returned
    or scored against.
    """
    panel = B.load_filtered_panel()
    all_anchor = SPLICE.anchor_rows(panel)
    holdout, train = B.split_holdout_train(panel, SEED)

    splice_cand, _ = SPLICE.generate_candidate(holdout, train, all_anchor)

    marginals = KERNEL.fit_cell_marginals(train)
    kernel_counts = KERNEL.estimate_kernel(train, all_anchor, marginals)
    fitted, _pairs = KERNEL.fit_participation_gate(train, SEED)
    kernel_cand, _ = KERNEL.generate_candidate(
        holdout, all_anchor, marginals, fitted, kernel_counts, SEED
    )

    return {
        "panel": panel,
        "holdout": holdout,
        "train": train,
        "splice": splice_cand,
        "kernel": kernel_cand,
    }


def _sanity_anchor(panels: dict[str, Any], view: hpanel.PanelView) -> dict:
    """Confirm the mirror == the harness C2ST, and record vs-train values.

    Against the HOLDOUT the mirrored classifier must equal the harness's
    ``classifier_two_sample_auc`` to float precision (this is the only
    place the holdout is read, and only to prove the mirror is exact -- it
    is not a gate decision and contacts no locked threshold). The reported
    forensic value is the candidate vs the TRAIN windows.
    """
    out: dict[str, Any] = {}
    for name in ("splice", "kernel"):
        cand = panels[name]
        cp, cw = hpanel.project_panel(cand, view)
        hp, hw = hpanel.project_panel(panels["holdout"], view)
        harness_auc = metrics.classifier_two_sample_auc(
            hp, cp, real_weights=hw, synthetic_weights=cw, seed=SEED
        )
        mirror_auc = fx.c2st_auc(panels["holdout"], cand, view, seed=SEED)
        vs_train = fx.c2st_auc(panels["train"], cand, view, seed=SEED)
        out[name] = {
            "harness_c2st_vs_holdout": float(harness_auc),
            "mirror_c2st_vs_holdout": float(mirror_auc),
            "mirror_equals_harness": bool(
                np.isclose(harness_auc, mirror_auc, atol=1e-12)
            ),
            "mirror_c2st_vs_train": float(vs_train),
        }
    return out


def _analyze_view(
    panels: dict[str, Any], view: hpanel.PanelView, view_name: str
) -> dict[str, Any]:
    """Run every candidate-vs-train analysis on one view."""
    train = panels["train"]
    result: dict[str, Any] = {
        "view": view_name,
        "dimension_names": list(view.dimension_names),
        "sanity_anchor": _sanity_anchor(panels, view),
        "shared_signal": fx.shared_signal_test(
            panels["splice"], panels["kernel"], train, view, seed=SEED
        ),
        "candidates": {},
    }
    for name in ("splice", "kernel"):
        cand = panels[name]
        result["candidates"][name] = {
            "feature_attribution": fx.feature_attribution(
                train, cand, view, seed=SEED
            ),
            "distributional": fx.distributional_forensics(train, cand, view),
            "rounding_repair": fx.rounding_repair_ablation(
                train, cand, view, seed=SEED
            ),
            "decision_region": fx.decision_region_probe(
                train, cand, view, seed=SEED
            ),
        }
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full forensic analysis on both locked views."""
    started = time.time()
    if verbose:
        print("regenerating seed-0 candidates + train panel ...")
    panels = _build_panels()
    if verbose:
        print(
            f"  train persons={panels['train'].person_id.nunique()} "
            f"splice rows={len(panels['splice'])} "
            f"kernel rows={len(panels['kernel'])}"
        )

    views = {
        "psid_family_earnings_pairs": B.build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": B.build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }
    by_view: dict[str, Any] = {}
    for vname, view in views.items():
        if verbose:
            print(f"analyzing view {vname} ...")
        by_view[vname] = _analyze_view(panels, view, vname)
        if verbose:
            sa = by_view[vname]["sanity_anchor"]
            ss = by_view[vname]["shared_signal"]
            print(
                "  mirror==harness: "
                f"splice={sa['splice']['mirror_equals_harness']} "
                f"kernel={sa['kernel']['mirror_equals_harness']}  "
                f"splice_vs_kernel={ss['splice_vs_kernel']:.4f} "
                f"noise_floor={ss['noise_floor']:.4f}"
            )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "c2st_forensics_v1",
        "reported_not_gated": True,
        "holdout_contact": False,
        "holdout_contact_note": (
            "Every reported forensic value compares a candidate against the "
            "seed-0 TRAIN persons' real windows (or one candidate against "
            "another, or two train halves). The holdout is projected only in "
            "the sanity anchor to prove the mirrored classifier equals the "
            "harness C2ST bit-for-bit; that check touches no locked "
            "threshold and makes no gate decision. gates.yaml and committed "
            "runs/ artifacts are untouched."
        ),
        "seed": SEED,
        "spec": SPEC,
        "candidate_sources": {
            "splice": "run_gate1_candidate5a2.generate_candidate (PR #51)",
            "kernel": "run_gate1_candidate6.generate_candidate (PR #53)",
        },
        "classifier": (
            "HistGradientBoostingClassifier(random_state=0), weighted "
            "5-fold stratified-CV ROC AUC, equal class mass, real class 0 / "
            "candidate class 1 (harness order); mirrors "
            "populace_dynamics.harness.metrics.classifier_two_sample_auc."
        ),
        "published_c2st": {
            "splice": {
                "psid_family_earnings_pairs": 0.5456,
                "psid_family_earnings_runs": 0.5657,
            },
            "kernel": {
                "psid_family_earnings_pairs": 0.5470,
                "psid_family_earnings_runs": 0.5984,
            },
            "gate_thresholds": {
                "psid_family_earnings_pairs_c2st_max": 0.53,
                "psid_family_earnings_runs_c2st_max": 0.54,
            },
        },
        "by_view": by_view,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
