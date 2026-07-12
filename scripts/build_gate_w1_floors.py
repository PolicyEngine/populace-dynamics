"""Build the W1 transport-gate floors evidence base (three target families).

PRE-LOCK EVIDENCE, NOT A GATE RUN. Step 1 of the gate-W1 lock ceremony --
the exact analogue, one MILESTONE up, of what PR #79 built for tranche 2a,
PR #118 for gate 2b, and PR #124 for gate 2c. The design is issue #151
(``W1 transport gate: design``); the milestone is #113 M5 and the seam is
#100 W1.

W1 certifies the TRANSPORT: generators ESTIMATED on PSID and DEPLOYED on the
certified populace CPS person records must produce distributions matching
targets OBSERVABLE WITHOUT PANEL DATA. Three families, three pricings:

* **A -- CPS-observable cross-sectional joints** (FLOOR-priced). The
  earnings x age x sex distribution, marital composition, and household
  composition the certified file already pins; a faithful transport
  reproduces them as its deployed cross-section. Priced by a 100-seed
  HOUSEHOLD-DISJOINT half-split noise floor on the file's own weighted
  moments (the deployment frame's sampling-noise floor). Statistic:
  ``|ln(rate_a/rate_b)|``, tolerance ``round(mean + 4*sd, 3)`` capped at
  ``T_max = ln(1.5)`` -- identical to gate-2a/2b/2c. HOUSEHOLD-disjoint (not
  merely person-disjoint) because CPS is a household-cluster sample and the
  composition moments are household-level (the 2c "couple/person-disjoint
  where units correlate" lesson).

* **B -- SSA administrative anchors** (ANCHOR-priced, point values). The
  deployed + simulated benefits must land on admin margins PSID never sees:
  the claim-age distribution (Supplement 6.B5.1), DI prevalence age
  composition (DI ASR 2023 Table 19), and the benefit-level distribution
  (6.B4/6.B3). Point values have no sampling floor; each carries a NAMED,
  MACHINE-DERIVABLE vintage/measurement tolerance from the anchor's OWN
  published series (detrended residual sd + measurement floor). Benefit
  LEVELS are report-only (they depend on the not-yet-deployed transported
  AIME).

* **C -- the two ordinal compression fingerprints** (BINARY, no floor). On
  the PSID observed-career frame two reform orderings came out wrong
  (compressed careers): PPI<->NRA (#115 T2 / #117 F4) and elimination<->+2pp
  (#117 F2). A certified representative frame must REVERSE both to the anchor
  orderings, and nothing else. Ordinal -> a binary check against the
  committed before/after orderings.

It reads no gate and changes no gate, and writes NO ``gates.yaml`` block:
gate_w1 does not yet exist as a stub (the proposal is in #151); only the
ratifying flip inserts it. This artifact feeds the ceremony (referee ->
fixes -> verification -> ratifying flip).

Run from the repository root; the certified frame is resolved by its pinned
revision + sha256 (:data:`deployment_frame.CERTIFIED_PIN`) from the Hugging
Face cache (public repo). The env used for the committed build is recorded in
``revision_pins`` (policyengine.py .venv: pandas/pytables/huggingface_hub;
populace_dynamics on PYTHONPATH). It needs no populace-fit (real-vs-real
only) and no PSID (family A is CPS-only; the PSID-estimated generators are the
eventual CANDIDATE, not this floor)::

    PYTHONPATH=src \\
      ~/PolicyEngine/policyengine.py/.venv/bin/python \\
      scripts/build_gate_w1_floors.py
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate_w1_floors_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate_w1_floors.v1"

CLAIM_AGE_ANCHOR = (
    ROOT / "data" / "external" / "ssa_claim_ages_2023supplement.json"
)
DI_ANCHOR = ROOT / "data" / "external" / "di_asr_2023" / "tables.json"

# --- floor / partition constants (the gate-2a/2b/2c precedent, verbatim) ---
SEEDS = tuple(range(100))
GATE_SEEDS = (0, 1, 2, 3, 4)
#: HOUSEHOLD-disjoint split: the CPS cluster is the household, so a
#: household-disjoint split is person-disjoint AND keeps every household on
#: one side (the 2c "units correlate" lesson).
SPLIT_COLUMN = "household_id"
SPLIT_FRACTION = 0.5
MIN_EVENTS_FOR_GATE = 20
DRAFT_K = 4
DRAFT_ROUNDING = 3
T_MAX = math.log(1.5)
T_MAX_SOURCE = "ln(1.5)"
#: The eventual W1 candidate scores with the ratified mean-over-K=20-draws
#: estimator (tranche-2a amendment 1) FROM THE START; a distinct draw stream.
CANDIDATE_DRAWS = 20
DRAW_STREAM_BASE = 9100
CANDIDATE_DRAW_STREAM = (
    f"numpy.random.default_rng({DRAW_STREAM_BASE} + k), k=0..K-1"
)
#: gate-W1 declares NO family-A coverage-recovery aggregates (the 2c stance);
#: sparse marital/household cells stay report-only by POWER, not masking.
AGGREGATIONS: dict[str, list[str]] = {}

# --- family-B vintage-tolerance rule (machine-derivable) ---
#: Recent published window used to measure vintage dispersion.
VINTAGE_WINDOW_YEARS = 10
#: tolerance_pp = round(K_VINTAGE * detrended_residual_sd + MEASUREMENT_PP, 2).
K_VINTAGE = 2.0
#: Half the 0.1 pp published rounding unit.
MEASUREMENT_PP = 0.05
#: A family-B category gates only if its vintage tolerance is tight enough to
#: reject a materially wrong margin (the anchor-space analogue of T_max).
T_MAX_PP = 3.0


# --------------------------------------------------------------------------
# Family A -- the CPS-observable joints floor
# --------------------------------------------------------------------------
def measure_seed_halfsplit(seed: int, persons: Any) -> dict[str, Any]:
    """One seed: split 50/50 HOUSEHOLD-DISJOINT, all cells per half.

    The split object is ``household_id`` (fix, the 2c couple-disjoint lesson):
    every person of a household lands on one side, so the two halves are
    genuinely independent CPS clusters and the symmetric half-vs-half ratio
    ``|ln(r_A/r_B)|`` is the null a faithful transport realises.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        persons, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
    )
    cells_a = dfm.reference_moments(side_a, weighted=True)
    cells_b = dfm.reference_moments(side_b, weighted=True)
    cells: dict[str, Any] = {}
    for key in set(cells_a) | set(cells_b):
        a = cells_a.get(key)
        b = cells_b.get(key)
        ra = a["rate"] if a else 0.0
        rb = b["rate"] if b else 0.0
        na = a["n_events"] if a else 0
        nb = b["n_events"] if b else 0
        defined = a is not None and b is not None and ra > 0 and rb > 0
        cells[key] = {
            "rate_a": float(ra),
            "rate_b": float(rb),
            "n_events_a": int(na),
            "n_events_b": int(nb),
            "log_ratio_abs": (
                float(abs(np.log(ra / rb))) if defined else None
            ),
            "pct_diff_abs": (
                float(abs(ra - rb) / rb * 100.0) if defined else None
            ),
        }
    return {
        "seed": seed,
        "n_households_side_a": int(side_a["household_id"].nunique()),
        "n_households_side_b": int(side_b["household_id"].nunique()),
        "n_persons_side_a": int(len(side_a)),
        "n_persons_side_b": int(len(side_b)),
        "cells": cells,
    }


def _floor_summary(log_ratios: list[float], pct_diffs: list[float]) -> dict:
    arr = np.array(log_ratios, dtype=np.float64)
    pct = np.array(pct_diffs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
        "realized_sigma": float(np.sqrt(np.mean(arr**2))),
        "values": [float(v) for v in arr],
        "pct_diff_abs": {
            "mean": float(pct.mean()),
            "sd": float(pct.std(ddof=1)) if pct.size > 1 else 0.0,
            "min": float(pct.min()),
            "max": float(pct.max()),
            "n_seeds": int(pct.size),
        },
    }


def pool_internal_floor(
    per_seed: list[dict[str, Any]], cell_keys: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    noise_floor: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    for key in cell_keys:
        log_ratios = [s["cells"][key]["log_ratio_abs"] for s in per_seed]
        pct_diffs = [s["cells"][key]["pct_diff_abs"] for s in per_seed]
        min_events = min(
            min(s["cells"][key]["n_events_a"], s["cells"][key]["n_events_b"])
            for s in per_seed
        )
        defined_seeds = sum(v is not None for v in log_ratios)
        stability[key] = {
            "defined_seeds": defined_seeds,
            "n_seeds": len(per_seed),
            "min_events_either_half": int(min_events),
        }
        if defined_seeds == len(per_seed):
            noise_floor[key] = _floor_summary(
                [float(v) for v in log_ratios],
                [float(v) for v in pct_diffs],
            )
    return noise_floor, stability


def raw_tolerances(noise_floor: dict[str, Any]) -> dict[str, float]:
    return {
        key: round(block["mean"] + DRAFT_K * block["sd"], DRAFT_ROUNDING)
        for key, block in noise_floor.items()
    }


def _events_ok(stab: dict[str, Any]) -> bool:
    return (
        stab["defined_seeds"] == stab["n_seeds"]
        and stab["min_events_either_half"] >= MIN_EVENTS_FOR_GATE
    )


def _floor_max_covered(key: str, noise_floor: dict, tolerances: dict) -> bool:
    """The tolerance covers the committed floor's OWN worst seed.

    The heavy-tailed-deployment-frame guard (a W1 refinement absent in 2a/2b/
    2c, whose PSID floors were light-tailed): the sparse-refit weights
    concentrate, so a handful of cells have a half-split |ln ratio| whose max
    over the 100 seeds exceeds mean + 4*sd. Requiring the tolerance to cover
    that max guarantees a faithful candidate -- and the training copy -- pass
    every gated cell (the K=20 mean is TIGHTER than a single half-split, so
    this is conservative for the candidate). A cell that fails it is
    report-only by POWER (heavy-tailed floor), not masking.
    """
    return key in noise_floor and noise_floor[key]["max"] <= tolerances[key]


def _passes(
    key: str, stability: dict, tolerances: dict, noise_floor: dict
) -> bool:
    return (
        key in tolerances
        and _events_ok(stability[key])
        and tolerances[key] <= T_MAX
        and _floor_max_covered(key, noise_floor, tolerances)
    )


def _demote_reason(
    key: str, stability: dict, tolerances: dict, noise_floor: dict
) -> str:
    stab = stability[key]
    if stab["defined_seeds"] != stab["n_seeds"]:
        return "undefined_on_some_seed"
    if stab["min_events_either_half"] < MIN_EVENTS_FOR_GATE:
        return "below_20_events"
    if key not in tolerances:
        return "no_floor"
    if tolerances[key] > T_MAX:
        return "tolerance_above_t_max"
    if not _floor_max_covered(key, noise_floor, tolerances):
        return "floor_max_exceeds_tolerance"
    return "gate_eligible"


def partition_cells(
    stability: dict[str, Any],
    tolerances: dict[str, float],
    noise_floor: dict[str, Any],
) -> tuple[set[str], set[str], dict[str, str]]:
    """gate-eligible / report-only partition (the gate-2c logic + the W1
    heavy-tail guard).

    A cell gates iff defined on every seed, >=20 events on the weaker half of
    the worst seed, stabilised tolerance <= T_max, the tolerance covers the
    committed floor's own max (the W1 heavy-tail guard), AND it is not
    superseded by a gating aggregate. gate-W1 declares NO coverage-recovery
    aggregates, so the supersession clause is inert; the machinery is kept
    identical to 2a/2b/2c.
    """
    member_of: dict[str, str] = {}
    for agg, members in AGGREGATIONS.items():
        for m in members:
            member_of[m] = agg

    gated: set[str] = set()
    report: set[str] = set()
    reasons: dict[str, str] = {}

    for agg in AGGREGATIONS:
        if _passes(agg, stability, tolerances, noise_floor):
            gated.add(agg)
            reasons[agg] = "gated_aggregate"
        else:
            report.add(agg)
            reasons[agg] = "aggregate_" + _demote_reason(
                agg, stability, tolerances, noise_floor
            )

    for member, agg in member_of.items():
        if agg in gated:
            report.add(member)
            reasons[member] = f"superseded_by:{agg}"
        elif _passes(member, stability, tolerances, noise_floor):
            gated.add(member)
            reasons[member] = "gated_per_age"
        else:
            report.add(member)
            reasons[member] = _demote_reason(
                member, stability, tolerances, noise_floor
            )

    for key in stability:
        if key in AGGREGATIONS or key in member_of:
            continue
        if _passes(key, stability, tolerances, noise_floor):
            gated.add(key)
            reasons[key] = "gated"
        else:
            report.add(key)
            reasons[key] = _demote_reason(
                key, stability, tolerances, noise_floor
            )
    return gated, report, reasons


def draft_thresholds(
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(gated):
        block = noise_floor[key]
        tol = tolerances[key]
        sigma = block["realized_sigma"]
        out[key] = {
            "log_ratio_abs_max": tol,
            "derivation": {
                "floor_mean": block["mean"],
                "floor_sd": block["sd"],
                "k": DRAFT_K,
                "rounding": DRAFT_ROUNDING,
            },
            "realized_sigma": sigma,
            "tolerance_sigma_units": (
                round(tol / sigma, 3) if sigma > 0 else None
            ),
        }
    return out


def annotate_stability(
    stability: dict[str, Any],
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
    reasons: dict[str, str],
) -> None:
    for key, stab in stability.items():
        tol = tolerances.get(key)
        sigma = noise_floor.get(key, {}).get("realized_sigma")
        floor_max = noise_floor.get(key, {}).get("max")
        stab["tolerance"] = tol
        stab["realized_sigma"] = sigma
        stab["floor_max"] = floor_max
        stab["floor_max_over_tolerance"] = (
            round(floor_max / tol, 3)
            if floor_max is not None and tol
            else None
        )
        stab["tolerance_sigma_units"] = (
            round(tol / sigma, 3) if tol is not None and sigma else None
        )
        stab["gate_eligible"] = key in gated
        stab["report_reason"] = reasons.get(key)


def holdout_id_commitment(persons: Any) -> dict[str, Any]:
    """sha256 of each gate seed's holdout (side A) HOUSEHOLD ids."""
    per_seed = []
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            persons, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
        )
        ids = sorted(int(x) for x in side_a["household_id"].unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        per_seed.append(
            {
                "seed": seed,
                "n_holdout_households": len(ids),
                "holdout_household_id_sha256": digest,
            }
        )
    return {
        "gate_seeds": list(GATE_SEEDS),
        "id_column": SPLIT_COLUMN,
        "split_unit": "household",
        "fraction": SPLIT_FRACTION,
        "numpy_generator": (
            "populace_dynamics.harness.panel.split_panel_by_person on "
            "household_id: side A (holdout) = households h of "
            "np.sort(unique household_id) with "
            "np.random.default_rng(seed).random(n_households)[h] < 0.5. Every "
            "person of a household shares its id, so the holdout is a "
            "household-coherent CPS-cluster set; the sha256 is over the "
            "holdout household ids."
        ),
        "per_seed": per_seed,
    }


def training_copy_check(
    per_seed: list[dict[str, Any]],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    worst_ratio = 0.0
    worst_cell = None
    per_seed_pass = []
    for s in per_seed[: len(GATE_SEEDS)]:
        seed_ok = True
        for key in gated:
            score = s["cells"][key]["log_ratio_abs"]
            if score is None:
                continue
            ratio = score / tolerances[key]
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_cell = (key, s["seed"])
            if score > tolerances[key]:
                seed_ok = False
        per_seed_pass.append(bool(seed_ok))
    n_pass = sum(per_seed_pass)
    return {
        "candidate": "verbatim copy of the train half (side B rates)",
        "score_identity": "|ln(rate_B / rate_A)| == the committed floor",
        "gate_seeds": list(GATE_SEEDS),
        "seed_pass": per_seed_pass,
        "n_seeds_pass": n_pass,
        "passes_4_of_5": n_pass >= 4,
        "max_score_over_tolerance": round(worst_ratio, 4),
        "max_ratio_cell": worst_cell,
        "interpretation": (
            "A train-copy passes at ~the noise floor, as any moment gate's "
            "copy must; the memorisation defence is procedural (registration "
            "+ holdout exclusion + no_self_rescue), NOT the cell set. What the "
            "family-A cells DO catch is a transport that DISTORTS the deployed "
            "cross-section: a generator whose deployed earnings x age x sex, "
            "marital, or household composition departs from the certified "
            "frame's own beyond the CPS sampling floor."
        ),
    }


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def faithful_candidate_oc(
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    p_seed = 1.0
    for key in sorted(gated):
        sigma = noise_floor[key]["realized_sigma"]
        tol = tolerances[key]
        p = 2.0 * _normal_cdf(tol / sigma) - 1.0 if sigma > 0 else 1.0
        per_cell[key] = {
            "realized_sigma": sigma,
            "cell_pass_prob": round(p, 6),
        }
        p_seed *= p
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return {
        "method": (
            "Independence-approx normal OC (draw-noise-free half-normal "
            "basis): a faithful candidate's per-cell score ~ "
            "half-normal(realized_sigma); the mean-over-K=20 estimator SHARES "
            "that basis, so this OC is ACHIEVABLE (a single-draw estimator "
            "would make it UNACHIEVABLE -- tranche-2a amendment-1 finding 1)."
        ),
        "n_gated_cells": len(gated),
        "p_seed_pass": round(p_seed, 4),
        "p_gate_pass_4_of_5": round(p_gate, 4),
        "per_cell": per_cell,
    }


def protocol_block(oc: dict[str, Any], n_gated: int) -> dict[str, Any]:
    return {
        "option": (
            "a -- the gate-1 / tranche-2a/2b/2c mirror: per-seed "
            "HOUSEHOLD-DISJOINT holdout with train-complement fitting, for "
            "which the half-vs-half floor is exactly the null. The 4-of-5 seed "
            "conjunction is operative."
        ),
        "estimator": (
            "mean-over-K=20-draws (tranche 2a amendment 1, ratified 2026-07-08 "
            "PR 96), adopted FROM THE START of this W1 build. rbar_candidate,s "
            "is the MEAN over K=20 pre-registered deployment draws "
            f"(numpy default_rng({DRAW_STREAM_BASE} + k), k=0..19; a stream "
            "DISTINCT from the split seeds), scored ONCE per cell as "
            "|ln(rbar / rate_a)| -- NOT the mean of the per-draw |ln| scores."
        ),
        "candidate_draws": CANDIDATE_DRAWS,
        "candidate_draw_stream": CANDIDATE_DRAW_STREAM,
        "statistic": (
            "|ln(rbar_candidate,s / rate_a,s)| per cell: rbar the K=20-draw "
            "mean of the deployed cell rate on seed s's holdout households, "
            "rate_a the holdout half's own weighted rate. Symmetric, "
            "scale-free -- the gate-2a/2b/2c statistic."
        ),
        "varies_per_seed": (
            "the 50/50 household-disjoint split (holdout = side A of "
            "split_panel_by_person(household_id, seed=s)), the candidate's fit "
            "complement, and its K=20 deployed draws of the holdout."
        ),
        "scored_against": (
            "side A's own weighted rate (rate_a), committed per gate seed in "
            "noise_floor_per_seed. The half-vs-half floor |ln(rate_A/rate_B)| "
            "is exactly the null a faithful transport realises."
        ),
        "pass_rule": (
            "Seed-level conjunction. Seed s passes iff every gate-eligible "
            "family-A cell's |ln(rbar/rate_a)| <= its derived tolerance at "
            "seed s; the gate passes iff >=4 of 5 gate seeds pass. Report-only "
            "family-A cells, all family-B anchors, and the family-C "
            "fingerprints are published on their own rules."
        ),
        "faithful_candidate_oc": {
            "n_gated_cells": oc["n_gated_cells"],
            "p_seed_pass": oc["p_seed_pass"],
            "p_gate_pass_4_of_5": oc["p_gate_pass_4_of_5"],
            "basis_note": (
                "DRAW-NOISE-FREE half-normal basis; ACHIEVABLE under the "
                "K=20 mean, UNACHIEVABLE under a single draw. Bound to "
                "faithful_candidate_oc by tests/test_gate_w1_derivations.py."
            ),
        },
        "fresh_run_artifact_schema": {
            "applies_to": (
                "a fresh W1 candidate one-shot run registered AFTER the "
                "gate_w1 lock; NEVER this floor build (real-vs-real, no draws)."
            ),
            "per_draw_per_cell_rates": {
                "required": True,
                "shape": [CANDIDATE_DRAWS, n_gated, len(GATE_SEEDS)],
                "shape_dims": "K_draws x gated_cells x gate_seeds",
                "rule": (
                    "commit every draw's per-cell deployed rate r[k, cell, s] "
                    f"for all K={CANDIDATE_DRAWS} draws, all {n_gated} gated "
                    "cells, all 5 gate seeds, so rbar and |ln(rbar/rate_a)| "
                    "recompute cell-by-cell."
                ),
            },
            "undefined_draw_rule": {
                "required": True,
                "pre_specified": True,
                "rule": (
                    "if any gated cell's deployed rate is UNDEFINED on any "
                    "draw, the RUN IS INVALIDATED and must be re-registered; "
                    "no draw is dropped, skipped, or re-rolled."
                ),
            },
            "per_draw_dispersion_disclosure": {
                "required": True,
                "gated": False,
                "report_only": True,
                "note": (
                    "REPORT-ONLY: the sd across the K=20 draws per gated cell "
                    "per seed, so a referee sees whether a passing mean "
                    "conceals a wild draw."
                ),
            },
        },
    }


# --------------------------------------------------------------------------
# Family B -- SSA administrative anchors + vintage/measurement tolerances
# --------------------------------------------------------------------------
def _detrended_residual_sd(years: list[int], values: list[float]) -> float:
    """Residual sd of an OLS linear fit -- the vintage NOISE net of trend."""
    x = np.array(years, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    if x.size < 3:
        return float(np.std(y, ddof=1)) if x.size > 1 else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return float(np.sqrt(np.sum(resid**2) / (x.size - 2)))


def _linear_slope(years: list[int], values: list[float]) -> float:
    if len(years) < 2:
        return 0.0
    slope, _ = np.polyfit(
        np.array(years, dtype=np.float64),
        np.array(values, dtype=np.float64),
        1,
    )
    return float(slope)


def _vintage_tolerance(
    years: list[int], values: list[float]
) -> dict[str, Any]:
    resid_sd = _detrended_residual_sd(years, values)
    tol = round(K_VINTAGE * resid_sd + MEASUREMENT_PP, 2)
    return {
        "tolerance_pp": tol,
        "detrended_residual_sd_pp": round(resid_sd, 3),
        "trend_pp_per_year": round(_linear_slope(years, values), 3),
        "window_years": [years[0], years[-1]],
        "n_vintages": len(years),
        "gate_eligible": tol <= T_MAX_PP,
        "rule": (
            f"round({K_VINTAGE} * detrended_residual_sd + {MEASUREMENT_PP}, 2)"
            f"; gated iff <= T_max_pp = {T_MAX_PP}"
        ),
    }


def claim_age_anchor() -> dict[str, Any]:
    """6.B5.1 claim-age distribution: 2022 anchor + vintage tolerances."""
    doc = json.loads(CLAIM_AGE_ANCHOR.read_text())
    cats = doc["column_schema"]["collapsed_categories"]
    out: dict[str, Any] = {}
    for sex in ("male", "female"):
        series = doc["data"][sex]
        years = sorted(int(y) for y in series)
        window = years[-VINTAGE_WINDOW_YEARS:]
        latest = str(years[-1])
        for cat in cats:
            key = f"claim_age.{cat}|{sex}"
            anchor_val = series[latest]["categories"][cat]
            vals = [series[str(y)]["categories"][cat] for y in window]
            tol = _vintage_tolerance(window, vals)
            out[key] = {
                "anchor_pp": anchor_val,
                "anchor_year": years[-1],
                "units": "percentage points of the claim-age distribution",
                **tol,
            }
    return out


def _parse_di_table19_all_workers() -> dict[int, dict[str, float]]:
    """The 'All disabled workers' panel of DI ASR Table 19 (year -> shares)."""
    doc = json.loads(DI_ANCHOR.read_text())
    tsv = doc["Table 19"]["tsv"].splitlines()
    bands = [
        "under30",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-fra",
    ]
    out: dict[int, dict[str, float]] = {}
    panel = None
    for line in tsv:
        low = line.lower()
        if "all disabled workers" in low:
            panel = "all"
            continue
        if low.strip().startswith("men") or low.strip().startswith("women"):
            panel = "other"
            continue
        parts = line.split("\t")
        if panel != "all" or len(parts) < 12:
            continue
        try:
            year = int(parts[0].strip())
        except ValueError:
            continue
        # parts: year, number, total(100.0), 8 bands, avg_age
        try:
            shares = [float(parts[3 + i]) for i in range(8)]
        except ValueError:
            continue
        out[year] = dict(zip(bands, shares, strict=False))
    return out


def di_prevalence_anchor() -> dict[str, Any]:
    """DI ASR Table 19 age composition: 2023 anchor + vintage tolerances."""
    series = _parse_di_table19_all_workers()
    years = sorted(series)
    window = years[-VINTAGE_WINDOW_YEARS:]
    latest = years[-1]
    out: dict[str, Any] = {}
    for band in series[latest]:
        key = f"di_prevalence.{band}"
        vals = [series[y][band] for y in window]
        tol = _vintage_tolerance(window, vals)
        out[key] = {
            "anchor_pp": series[latest][band],
            "anchor_year": latest,
            "units": (
                "percentage points of the disabled-worker age distribution "
                "(All disabled workers panel)"
            ),
            **tol,
        }
    return out


def benefit_level_anchor() -> dict[str, Any]:
    """6.B4 PIA + 6.B3 MOB level anchors -- REPORT-ONLY (depend on the
    not-yet-deployed transported AIME; single 2022 vintage, no series)."""
    return {
        "status": "report_only",
        "reason": "depends_on_transported_aime_levels; single_vintage_no_series",
        "units": "2022 dollars",
        "anchors": {
            "avg_pia_all": 1984.09,
            "avg_pia_men": 2230.79,
            "avg_pia_women": 1733.94,
            "avg_monthly_benefit_all": 1908.86,
            "avg_monthly_benefit_men": 2131.04,
            "avg_monthly_benefit_women": 1683.57,
            "avg_monthly_benefit_with_reduction": 1529.83,
            "avg_monthly_benefit_without_reduction": 2308.28,
        },
        "source": "SSA Statistical Supplement 2023, Tables 6.B4 / 6.B3 (2022 awards)",
        "provenance": "dynasim-refs/ssa_supplement_2023_6b.txt (headline rows)",
        "note": (
            "Levels are report-only in v1: they depend on the transported AIME "
            "the W1 candidate deploys (not yet built), and only a single 2022 "
            "vintage is staged, so no machine vintage sd is derivable. Recorded "
            "with an uprating-band context tolerance a fixes round can gate "
            "once the AIME transport lands."
        ),
        "uprating_context_tolerance_pct": 6.0,
    }


def family_b_block() -> dict[str, Any]:
    claim = claim_age_anchor()
    di = di_prevalence_anchor()
    bene = benefit_level_anchor()
    gated = sorted(
        [k for k, v in claim.items() if v["gate_eligible"]]
        + [k for k, v in di.items() if v["gate_eligible"]]
    )
    report = sorted(
        [k for k, v in claim.items() if not v["gate_eligible"]]
        + [k for k, v in di.items() if not v["gate_eligible"]]
        + ["benefit_level.report_only"]
    )
    return {
        "pricing": "anchor (point values + named machine-derivable tolerances)",
        "tolerance_rule": (
            "vintage/measurement: tolerance_pp = round(K_VINTAGE * "
            "detrended_residual_sd + MEASUREMENT_PP, 2) over the last "
            f"{VINTAGE_WINDOW_YEARS} published vintages; gated iff <= T_max_pp "
            f"= {T_MAX_PP}. Detrended so the FRA-transition / DI-ageing TREND "
            "is disclosed (trend_pp_per_year) and the tolerance prices the "
            "year-to-year NOISE, not the trend."
        ),
        "knobs": {
            "vintage_window_years": VINTAGE_WINDOW_YEARS,
            "k_vintage": K_VINTAGE,
            "measurement_pp": MEASUREMENT_PP,
            "t_max_pp": T_MAX_PP,
        },
        "claim_age": claim,
        "di_prevalence": di,
        "benefit_level": bene,
        "gate_partition": {
            "gate_eligible": gated,
            "report_only": report,
            "n_gate_eligible": len(gated),
            "n_report_only": len(report),
        },
        "sources": {
            "claim_age": "data/external/ssa_claim_ages_2023supplement.json (6.B5.1)",
            "di_prevalence": "data/external/di_asr_2023/tables.json (Table 19)",
            "benefit_level": "dynasim-refs/ssa_supplement_2023_6b.txt (6.B4/6.B3)",
        },
        "note": (
            "The deployed + simulated benefits must land on these admin "
            "margins PSID never sees. Point values -> no sampling floor; the "
            "tolerance is the anchor's own vintage noise. Benefit LEVELS are "
            "report-only until the transported AIME is deployed."
        ),
    }


# --------------------------------------------------------------------------
# Family C -- the two ordinal compression fingerprints (binary)
# --------------------------------------------------------------------------
def family_c_block() -> dict[str, Any]:
    """The two pre-committed before/after fingerprint orderings (#113).

    Binary: on the certified representative frame both PSID-frame orderings
    must REVERSE to the anchor orderings (nothing else). No floor.
    """
    c1 = {
        "id": "ppi_nra",
        "name": "PPI<->NRA at the PIA bends",
        "source_before": "runs/replication_cost_ordering_v1.json T2 / runs/m2_pseudo_projection_v1.json F4",
        "anchor": "Mermin Table 1 (75-yr payroll savings ordering)",
        "psid_frame_order": [
            "price_indexing",
            "nra_raised_to_70",
            "progressive_price_indexing",
            "reduced_cola",
        ],
        "required_representative_order": [
            "price_indexing",
            "progressive_price_indexing",
            "nra_raised_to_70",
            "reduced_cola",
        ],
        "swap_pair": ["nra_raised_to_70", "progressive_price_indexing"],
        "required_reversal": "progressive_price_indexing outranks nra_raised_to_70",
        "kendall_tau_before": round(2.0 / 3.0, 6),
        "kendall_tau_after_required": 1.0,
        "mechanism": (
            "a representative frame carries more AIME above the second PIA "
            "bend, where progressive price indexing bites, so PPI's savings "
            "exceed NRA->70's -- reversing the compressed-career order."
        ),
    }
    c2 = {
        "id": "elimination_plus2pp",
        "name": "elimination<->+2pp at the taxable maximum",
        "source_before": "runs/m2_pseudo_projection_v1.json F2",
        "anchor": "Smith (2015) solvency-year deltas",
        "psid_frame_order": [
            "payroll_plus_2pp",
            "elimination",
            "payroll_plus_1pp",
            "cap_150k",
        ],
        "required_representative_order": [
            "elimination",
            "payroll_plus_2pp",
            "payroll_plus_1pp",
            "cap_150k",
        ],
        "swap_pair": ["payroll_plus_2pp", "elimination"],
        "required_reversal": "elimination outranks payroll_plus_2pp",
        "kendall_tau_before": round(2.0 / 3.0, 6),
        "kendall_tau_after_required": 1.0,
        "mechanism": (
            "a representative frame carries >16.1% of payroll above the wage "
            "base (the break-even), so taxable-max elimination's revenue gain "
            "exceeds +2pp's -- reversing the compressed-tail order."
        ),
    }
    return {
        "pricing": "binary ordinal (committed anchor orderings; no floor)",
        "pre_committed_in": "#113 (M5 row + the two compression fingerprints)",
        "check": (
            "on the certified representative frame BOTH fingerprints reverse "
            "to required_representative_order (Kendall tau -> 1.0 vs the "
            "anchor), and no OTHER adjacency changes. If transport is real "
            "both reverse; if cosmetic they do not."
        ),
        "fingerprints": {"c1": c1, "c2": c2},
        "gate_partition": {
            "gate_eligible": [
                "fingerprint.ppi_nra",
                "fingerprint.elimination_plus2pp",
            ],
            "report_only": [],
            "n_gate_eligible": 2,
            "n_report_only": 0,
        },
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def _git_sha(rev: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", rev], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _resolve_certified_h5() -> str:
    """Resolve the pinned certified h5 from the Hugging Face cache/download."""
    from huggingface_hub import hf_hub_download

    pin = dfm.CERTIFIED_PIN
    return hf_hub_download(
        repo_id=pin["hf_repo_id"],
        filename=pin["hf_filename"],
        repo_type=pin["hf_repo_type"],
        revision=pin["revision"],
    )


def run(verbose: bool = False) -> dict[str, Any]:
    t0 = time.time()

    # ---- Family A: load the certified frame and build the floor ----
    h5_path = _resolve_certified_h5()
    sha = hashlib.sha256()
    with open(h5_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    file_sha = sha.hexdigest()
    if file_sha != dfm.CERTIFIED_PIN["artifact_sha256"]:
        raise ValueError(
            "certified frame sha256 mismatch: got "
            f"{file_sha}, pinned {dfm.CERTIFIED_PIN['artifact_sha256']}"
        )

    persons, populated = dfm.load_certified_persons(h5_path)
    ref_weighted = dfm.reference_moments(persons, weighted=True)
    ref_unweighted = dfm.reference_moments(persons, weighted=False)
    cell_keys = sorted(ref_weighted)

    per_seed = []
    for seed in SEEDS:
        per_seed.append(measure_seed_halfsplit(seed, persons))
        if verbose and seed % 20 == 0:
            print(f"  seed {seed} done ({time.time() - t0:.1f}s)")

    noise_floor, stability = pool_internal_floor(per_seed, cell_keys)
    tolerances = raw_tolerances(noise_floor)
    gated, report, reasons = partition_cells(
        stability, tolerances, noise_floor
    )
    annotate_stability(stability, noise_floor, tolerances, gated, reasons)
    drafts = draft_thresholds(noise_floor, tolerances, gated)
    oc = faithful_candidate_oc(noise_floor, tolerances, gated)
    copy_check = training_copy_check(per_seed, tolerances, gated)
    holdout_ids = holdout_id_commitment(persons)

    reference_moments_out = {
        key: {
            "rate": ref_weighted[key]["rate"],
            "rate_unweighted": ref_unweighted[key]["rate"],
            "num_wt": ref_weighted[key]["num_wt"],
            "den_wt": ref_weighted[key]["den_wt"],
            "n_events": ref_weighted[key]["n_events"],
        }
        for key in cell_keys
    }
    per_seed_stored = per_seed[: len(GATE_SEEDS)]

    # weight-concentration disclosure (the sparse-file WART)
    w = persons["weight"].to_numpy()
    kish_n_eff = float(w.sum() ** 2 / np.sum(w**2))

    family_b = family_b_block()
    family_c = family_c_block()

    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate_w1_floors_v1",
        "reported_anchor_not_gated": True,
        "referee_round": "PR (draft -- 2b/2c round lessons pre-empted)",
        "component": (
            "W1 representative-frame transport gate (M5): CPS-observable "
            "cross-sectional joints (family A, floor-priced), SSA "
            "administrative margins of the deployed+simulated benefits "
            "(family B, anchor-priced), and the two ordinal compression "
            "fingerprints that must reverse to the anchor orderings (family C, "
            "binary)."
        ),
        "purpose": (
            "Gate-W1 pre-lock evidence (v1): step 1 of the lock ceremony "
            "(#151 design; #113 M5; #100 W1). Reads no gate, changes no gate, "
            "writes NO gates.yaml block -- the gate_w1 stub is proposed in "
            "#151 and inserted only by the ratifying flip. It certifies the "
            "TRANSPORT of the PSID-estimated generators onto the certified "
            "populace frame, NOT the dynamics (2a/2b/2c/M4-certified) nor the "
            "projection (M6+)."
        ),
        "design_issue": "#151",
        "roadmap_issue": "#113",
        "seam_issue": "#100",
        "holdout_basis": [
            "populace_us_2024_certified",
            "ssa_supplement_6b",
            "di_asr_2023",
            "mermin_smith_committed_orderings",
        ],
        "ceremony": {
            "milestone": "M5_representative_frame_transport",
            "gates_yaml_stub": "none yet (proposed gate_w1 in #151)",
            "step": (
                "1 of the lock ceremony (floors -> referee -> fixes -> "
                "verification -> ratifying flip). Never marked ready/merged "
                "by its author."
            ),
            "gates_yaml_untouched": True,
            "mirrors": "runs/gate2c_floors_v1.json (2c) + gate2b_floors_v1.json",
        },
        "estimand": {
            "deployment_frame": dfm.CERTIFIED_PIN,
            "population": "US resident persons, reference period 2024",
            "n_persons": int(len(persons)),
            "n_households": int(persons["household_id"].nunique()),
            "total_weight": float(w.sum()),
            "populated_source_fractions": {
                k: round(v, 4) for k, v in populated.items()
            },
            "claim": (
                "Family A gates EXACTLY the CPS-observable cross-sectional "
                "cells the certified L0-sparse-refit file carries with "
                "verified non-zero support; NOT an uncalibrated CPS extract "
                "and NOT the PSID estimation sample. Per "
                "governance.description_claims_exactly_the_scored_surface."
            ),
        },
        "family_a": {
            "pricing": "100-seed household-disjoint half-split noise floor",
            "cell_families": dfm.cell_families(),
            "statistic": (
                "|ln(rate_a/rate_b)| on the file's own weighted moments; "
                "tolerance round(mean + 4*sd, 3) capped at T_max = ln(1.5)."
            ),
            "split_note": (
                "HOUSEHOLD-disjoint (not merely person-disjoint): CPS is a "
                "household-cluster sample and the composition moments are "
                "household-level, so the honest null keeps every household on "
                "one side (the 2c 'couple/person-disjoint where units "
                "correlate' lesson)."
            ),
            "scope_note": (
                "Family A is INTERNAL transport consistency to the certified "
                "frame -- the deployed cross-section must equal the file's "
                "own within its sampling floor. The frame's own fidelity to "
                "Census/CPS margins is populace-certified, NOT re-litigated "
                "here."
            ),
        },
        "reference_moments": reference_moments_out,
        "internal_noise_floor": {
            "method": (
                "half_vs_half -- two household-disjoint halves of the "
                "certified frame scored against each other; the sampling-noise "
                "floor of the DEPLOYMENT frame (not PSID)."
            ),
            "split_unit": "household",
            "split_fraction": SPLIT_FRACTION,
            "floor_seeds": list(SEEDS),
            "gate_seeds": list(GATE_SEEDS),
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
            "t_max": T_MAX,
            "t_max_source": T_MAX_SOURCE,
        },
        "protocol": protocol_block(oc, len(gated)),
        "knobs": {
            "earn_bands": [list(b) for b in dfm.EARN_BANDS],
            "dispersion_bands": [list(b) for b in dfm.DISPERSION_BANDS],
            "adult_bands": [list(b) for b in dfm.ADULT_BANDS],
            "profile_ref_band": list(dfm.PROFILE_REF_BAND),
            "spouse_present_codes": list(dfm.SPOUSE_PRESENT_CODES),
            "draw_stream_base": DRAW_STREAM_BASE,
            "split_fraction": SPLIT_FRACTION,
            "note": (
                "Load-bearing knobs pinned against the module so a mutation "
                "is caught WITHOUT the certified-frame reproduction."
            ),
        },
        "weight_concentration": {
            "household_weight_max": float(
                persons.groupby("household_id")["weight"].first().max()
            ),
            "household_weight_mean": float(
                persons.groupby("household_id")["weight"].first().mean()
            ),
            "kish_effective_n_persons": round(kish_n_eff, 1),
            "note": (
                "The L0-sparse-refit file concentrates weight (a few "
                "households carry very large weight), so the effective sample "
                "is far below the nominal 166,302 persons. The "
                "household-disjoint split keeps each heavy household intact on "
                "one side, so a few cells have a HEAVY-TAILED half-split floor "
                "whose max over the 100 seeds exceeds mean+4*sd; the W1 "
                "heavy-tail guard (report_reason floor_max_exceeds_tolerance) "
                "demotes exactly those, guaranteeing a faithful candidate and "
                "the training copy pass every gated cell. The honest, "
                "self-correcting behaviour, disclosed here."
            ),
        },
        "aggregations": {
            agg: {"members": list(members), "gated": agg in gated}
            for agg, members in AGGREGATIONS.items()
        },
        "aggregations_note": (
            "EMPTY by design: gate-W1 declares no family-A coverage-recovery "
            "aggregates (the 2c stance). Sparse marital/household cells stay "
            "report-only by POWER, not masking; a fixes round may add a pooled "
            "aggregate only where it is standalone-gateable."
        ),
        "cell_stability": stability,
        "gate_partition": {
            "gate_eligible": sorted(gated),
            "report_only": sorted(report),
            "n_gate_eligible": len(gated),
            "n_report_only": len(report),
        },
        "noise_floor_seeds_0_99": noise_floor,
        "noise_floor_per_seed": per_seed_stored,
        "holdout_ids": holdout_ids,
        "training_copy_check": copy_check,
        "faithful_candidate_oc": oc,
        "draft_thresholds": {
            "k": DRAFT_K,
            "rounding": DRAFT_ROUNDING,
            "t_max": T_MAX,
            "statistic": "abs log ratio of the weighted cell rate",
            "note": (
                "DRAFT GATE-W1 family-A tolerances -- PRE-LOCK, NOT RATIFIED "
                "(v1, step 1 of the lock ceremony). NOT yet mirrored into "
                "gates.yaml; the ratifying flip inserts a gate_w1 block. "
                "tolerance == round(floor mean + 4*sd, 3), capped at "
                "T_max = ln(1.5)."
            ),
            "cells": drafts,
        },
        "family_b": family_b,
        "family_c": family_c,
        "warts": WARTS,
        "draft_thresholds_note": (
            "DRAFT GATE-W1 VALIDATION EVIDENCE -- PRE-LOCK, NOT RATIFIED (v1, "
            "step 1 of the gate-W1 lock ceremony; the 2b/2c round lessons "
            "pre-empted). The lock ceremony (referee -> fixes -> verification "
            "-> ratifying flip) controls the lock; this is a DRAFT, never "
            "marked ready or merged by its author."
        ),
        "revision_pins": {
            "base_sha": _git_sha("HEAD"),
            "origin_master_sha": _git_sha("origin/master"),
            "certified_artifact_sha256": dfm.CERTIFIED_PIN["artifact_sha256"],
            "certified_revision": dfm.CERTIFIED_PIN["revision"],
            "pe_us_version": dfm.CERTIFIED_PIN["model_version"],
            "build_env": (
                "policyengine.py .venv (pandas 2.3.2 / pytables 3.11.1 / "
                "huggingface_hub 0.34.4) with populace_dynamics on PYTHONPATH; "
                "the certified pin comes from pe.us.model.release_bundle "
                "(policyengine 4.18.8), resolved by revision+sha256."
            ),
            "build_commit_note": (
                "gate-w1-floors v1: certified_artifact_sha256 pins the "
                "deployment frame; no PSID and no populace-fit."
            ),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return artifact


WARTS = [
    {
        "id": "sparse_frame_zeroes_untargeted_inputs",
        "class": "estimand honesty (2c fix B / description-claims-exactly)",
        "wart": (
            "The certified default is the L0-sparse-refit-57k file, which "
            "zeroes UNTARGETED inputs. Family A builds moments only on columns "
            "verified populated (age, is_female, employment_income_before_lsr, "
            "A_MARITL, household membership; assert_columns_populated fails "
            "loudly otherwise). Cells on a zeroed column are OUT of scope by "
            "construction, not silently gated."
        ),
    },
    {
        "id": "weight_concentration",
        "class": "floor honesty (heavy-tail weights)",
        "wart": (
            "Household weights are highly concentrated (max ~433k vs mean "
            "~2.3k), so the Kish effective n is far below the nominal sample. "
            "The household-disjoint split keeps heavy households intact, so a "
            "few cells have a heavy-tailed half-split floor. Unlike the "
            "light-tailed PSID floors of 2a/2b/2c, a mean+4*sd tolerance does "
            "not always cover the worst seed, so a NEW gate-eligibility "
            "criterion (the heavy-tail guard: tolerance >= floor max over the "
            "100 seeds, report_reason floor_max_exceeds_tolerance) demotes "
            "exactly those cells; k stays 4 (the precedent). This guarantees "
            "the training copy passes 5/5 at <=1x (as in 2a/2b/2c). Disclosed "
            "in weight_concentration + the partition reasons."
        ),
    },
    {
        "id": "family_a_is_internal_consistency_not_census_fidelity",
        "class": "scope honesty",
        "wart": (
            "Family A checks the deployed cross-section against the certified "
            "FRAME'S OWN moments, not against Census/CPS external truth. The "
            "frame's fidelity to admin margins is populace's calibration "
            "certification, OUTSIDE W1. W1-A is transport-fidelity to the "
            "deployment frame."
        ),
    },
    {
        "id": "family_b_non_stationary_anchors",
        "class": "anchor honesty (vintage non-stationarity)",
        "wart": (
            "The claim-age distribution (FRA transition) and DI age "
            "composition (workforce ageing) TREND. The vintage tolerance is "
            "the DETRENDED residual sd (year-to-year noise), and the trend is "
            "disclosed per category (trend_pp_per_year); categories whose "
            "detrended tolerance still exceeds T_max_pp demote to report-only. "
            "No trend is hidden inside a wide tolerance."
        ),
    },
    {
        "id": "family_b_benefit_levels_report_only",
        "class": "holdout discipline",
        "wart": (
            "Benefit LEVELS (6.B4 PIA / 6.B3 MOB) are report-only: they depend "
            "on the transported AIME the W1 candidate deploys (not yet built), "
            "and only a 2022 vintage is staged (no machine vintage sd). "
            "Recorded with an uprating-band context, gate-able by a fixes "
            "round once the AIME transport lands."
        ),
    },
    {
        "id": "family_c_records_not_runs_the_reversal",
        "class": "ceremony scope",
        "wart": (
            "This floor RECORDS the two committed before/after orderings; it "
            "does NOT run them on a representative frame (the transport "
            "generator does not exist yet -- it is the eventual candidate). "
            "The binary reversal is checked at candidate time; here it is the "
            "pre-committed anchor the gate will bind."
        ),
    },
    {
        "id": "di_prevalence_rate_denominator_absent",
        "class": "anchor coverage",
        "wart": (
            "Family B anchors the DI age COMPOSITION (Table 19), not the "
            "prevalence RATE: the insured denominator (Supplement 4.C2) is not "
            "staged (#123 wanted-list). The composition is gated; the rate is "
            "deferred, named, not fabricated."
        ),
    },
    {
        "id": "earnings_concept_before_lsr",
        "class": "concept alignment",
        "wart": (
            "Family-A earnings = employment + self-employment labor income "
            "BEFORE the labor-supply response (the certified input columns), "
            "the CPS-observable analog of the gate-1 PSID labor-income "
            "concept. Not the post-LSR modelled earnings; disclosed."
        ),
    },
]


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
