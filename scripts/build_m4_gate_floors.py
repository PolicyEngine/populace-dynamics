"""Build the M4 disability gate floors: the anchor + holdout evidence base
for the DI gate ceremony (roadmap #113, M4). CEREMONY STEP 1.

REPORTED ANCHOR, NOT A GATE RUN. Like the gate-2/2b/2c floors this reads no
gate and decides nothing on its own; it is committed evidence pinned by
reproduction tests. It is the floors + anchor basis the DI gate would derive
pre-registered thresholds from. ``gates.yaml`` is UNTOUCHED: the ``gate_m4``
stub-flip text lives in ``draft_thresholds_note`` and the PR body, for the
referee rounds and the eventual maintainer ratification to insert -- exactly
as gate-2c's step-1 floor left the ``gate_2c`` flip to a later step.

Why ANCHOR-BASED (gate-1 external-check style), not HOLDOUT-only (gate-2)
----------------------------------------------------------------------
The PSID ``EMPLOYMENT STATUS == 5`` self-report differs from an SSA DI award
by seven named concept deltas (``concept_deltas``): self-report vs
adjudication, all-adults vs insured-workers, no severity screen, huge
recovery churn, a different conversion denominator, biennial censoring, and
period pooling. So the gate must NOT gate a LEVEL match to any SSA DI rate --
that would "reject reality". Gated cells are defined only on surfaces where a
concept bridge is defensible, and every level a delta blocks stays
REPORT-ONLY with the delta named (never calibrated). The four surfaces:

(a) **DI->retirement conversion vs Table 50.** Among non-death exits from
    self-reported disability at the FRA-crossing ages, the weighted share
    that goes to ``retired`` -- the PSID analog of Table 50's FRA-conversion
    share of worker terminations. The concept-robust invariant is DOMINANCE
    (retirement is the majority non-death exit), which Table 50 supports
    (FRA-conversion 57.8% >> medical-cessation 12.6% of worker terminations;
    82.0% of the non-death conversion+cessation pair). GATED as a dominance
    margin, per sex. The 6.B5.1 level ratio and the sex-ordering are
    REPORT-ONLY (denominator delta / half-split fragility).

(b) **Prevalence AGE-SHAPE vs Table 19.** Per sex, the disabled-prevalence
    distribution across age bands. LEVELS are not bridgeable (self-report
    captures milder/younger limitation: young bands carry ~5x the Table 19
    share), but the co-monotone RISE across the four working-age bands and
    the 50-59 working-age peak ARE shared with the Table 19 stock (rank
    agreement 1.0 over the bridged bands; the 50-59 share even matches in
    level). GATED as a co-monotone/peak ordinal invariant, per sex. The
    60-FRA band and the per-band levels are REPORT-ONLY (the relabel-to-
    retired + stock-aging delta, and the severity/population deltas).

(c) **Termination-rate TREND vs Table 49.** Table 49's worker termination
    rate (rising 90->107 per 1,000 over 2015-2023; a secular fall then rise
    since 1960). REPORT-ONLY: the PSID hazards pool covered waves 1982-2023
    (period-pooling delta), so there is no year-indexed PSID analog to gate a
    trend against. Recorded as context, never gated.

(d) **PSID-internal held-out hazard stability.** The 100-seed person-disjoint
    half-split ``|ln(r_A / r_B)|`` floor on the incidence/recovery cells --
    the gate-2 machinery. This gates candidate REPRODUCTION of the PSID
    self-report hazards (not a DI-level claim) within the real-vs-real noise.
    At the inherited derivation (round(mean + k*sd) capped at ln(1.5)) with
    ``k=3`` (the range m4_disability_v1's proposed_thresholds_note itself
    proposed), the well-powered prime-age/older cells gate and the noisy
    young/onset cells demote (tolerance_above_t_max) -- an honest partition,
    no dead-zone wide tolerances. ``k=4`` (the gate-2c inheritance) empties
    the surface; the k-sensitivity is disclosed (``warts``).

Run from the repository root with the PSID individual file staged::

    .venv/bin/python scripts/build_m4_gate_floors.py

It needs no populace-fit and no policyengine-us checkout (real-vs-real and
real-vs-archived-SSA only), so its derivations reproduce in CI.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics import artifacts, claiming
from populace_dynamics.data import deaths, disability
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m4_gate_floors_v1.json"
ANCHOR_TABLES_REL = "data/external/di_asr_2023/tables.json"
ANCHOR_TABLES_PATH = ROOT / ANCHOR_TABLES_REL
SCHEMA_VERSION = "m4_gate_floors.v1"
RUN = "m4_gate_floors_v1"

#: 100-seed floor (the gate-2b/2c standard, adopted from the START), scored on
#: 5 gate seeds.
SEEDS: tuple[int, ...] = tuple(range(100))
GATE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SPLIT_FRACTION = 0.5
SPLIT_COLUMN = "person_id"

#: Minimum transition events (weaker half, worst seed) for a cell to gate --
#: the transition analog of the mortality floor's 20-death reliability floor.
MIN_EVENTS_FOR_GATE = 20

#: Tolerance derivation: round(floor_mean + DRAFT_K*sd, DRAFT_ROUNDING),
#: power-capped at T_MAX. DRAFT_K = 3 is the top of the [2, 3] range
#: m4_disability_v1.json's proposed_thresholds_note pre-registered for this
#: ceremony (gate-2c used 4 on its less-noisy couple statistics; k is a DRAFT
#: for the referee rounds to fix). MARGIN_K is the same k for the ordinal
#: anchor-cell margins.
DRAFT_K = 3
DRAFT_ROUNDING = 3
MARGIN_K = 3

#: Power cap: a cell whose stabilised |ln ratio| tolerance exceeds a 1.5x
#: rate ratio is too noisy at half-panel to gate -- demoted to report-only,
#: NOT gated with a meaningless-wide tolerance (the no-dead-zone rule).
T_MAX = math.log(1.5)
T_MAX_SOURCE = "ln(1.5)"

#: The candidate estimator a scored DI model would use: mean over K=20
#: pre-registered draws (the ratified tranche-2a amendment-1 estimator,
#: adopted from the START so the single-draw dead zone the 2b/2c round-1
#: referee flagged never appears).
CANDIDATE_DRAWS = 20
DRAW_STREAM_BASE = 4100
CANDIDATE_DRAW_STREAM = (
    "numpy.random.default_rng(4100 + k), k=0..K-1 (distinct from the split "
    "seeds 0..99)"
)

#: Only the transition-hazard families are candidates for INTERNAL (holdout)
#: level-gating (surface d). Prevalence and conversion cells are report-only
#: at the level (the concept deltas block a level gate) and are bridged to the
#: SSA anchors by SHAPE / DOMINANCE in the anchor_checks block instead.
FAMILY_INTERNAL_GATED = frozenset({"incidence", "recovery"})

#: The prevalence age-shape bridge is defined on the four working-age bands;
#: the 60-66 band is report-only (the relabel-to-retired + stock-aging delta).
BRIDGED_BANDS = ("20-29", "30-39", "40-49", "50-59")
PEAK_BAND = "50-59"
ALL_BANDS = ("20-29", "30-39", "40-49", "50-59", "60-66")

#: FRA-crossing window for the conversion-exit statistic (the module's
#: conversion analog window).
EXIT_WINDOW = disability.CONVERSION_WINDOW  # (60, 67)


def _family(key: str) -> str:
    return key.split(".", 1)[0]


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank_order(values: list[float]) -> tuple[int, ...]:
    """Ascending rank position of each element (ties by index) -- a
    level-invariant fingerprint of a shape vector, for rank agreement."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return tuple(ranks)


# --------------------------------------------------------------------------
# Anchor-check candidate statistics (the surfaces the module + tables bridge)
# --------------------------------------------------------------------------
def prevalence_shares(
    moments: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Per sex, the disabled-prevalence share across the five age bands
    (weighted disabled person-years in the band / the sex total)."""
    out: dict[str, np.ndarray] = {}
    for sex in disability.SEXES:
        nw = np.array(
            [moments[f"prevalence.{b}|{sex}"]["num_wt"] for b in ALL_BANDS],
            dtype=np.float64,
        )
        total = nw.sum()
        out[sex] = nw / total if total > 0 else np.zeros_like(nw)
    return out


def conversion_exit_shares(
    panel: disability.DisabilityPanel, person_ids: set[int] | None = None
) -> dict[str, dict[str, float]]:
    """Per sex: among NON-DEATH exits from self-reported disability in the
    FRA-crossing window, the weighted share transitioning to ``retired``.

    A death between waves leaves no ``w'`` observation (right-censored, no
    pair), so this is the retirement share among the OBSERVABLE (retirement
    vs return-to-work) disability exits -- the PSID analog of Table 50's
    FRA-conversion share among the non-death worker terminations. The
    censoring of death is a named delta, not a silent omission.
    """
    pairs = panel.pairs
    if person_ids is not None:
        pairs = pairs[pairs["person_id"].isin(person_ids)]
    lo, hi = EXIT_WINDOW
    exits = pairs[
        pairs["from_disabled"]
        & (~pairs["to_disabled"])
        & (pairs["age"] >= lo)
        & (pairs["age"] <= hi)
    ]
    out: dict[str, dict[str, float]] = {}
    for sex in disability.SEXES:
        g = exits[exits["sex"] == sex]
        w = g["weight"].to_numpy(dtype=float)
        tot = float(w.sum())
        ret_mask = g["to_retired"].to_numpy(dtype=bool)
        ret = float(w[ret_mask].sum())
        out[sex] = {
            "share": ret / tot if tot > 0 else 0.0,
            "n_exits": int(len(g)),
            "n_retired": int(ret_mask.sum()),
        }
    return out


# --------------------------------------------------------------------------
# One-seed half-split: internal cells + anchor half-statistics
# --------------------------------------------------------------------------
def measure_seed_halfsplit(
    seed: int, panel: disability.DisabilityPanel
) -> dict[str, Any]:
    """Split persons 50/50 person-disjoint; all cells + anchor stats per half.

    Cell floor statistic is ``|ln(r_A / r_B)|`` (``None`` when either half has
    a zero rate). The anchor half-statistics (prevalence shares side A;
    conversion-exit share both sides) drive the ordinal anchor-cell stability.
    """
    ids_frame = panel.person_years[["person_id"]].drop_duplicates()
    left, right = hpanel.split_panel_by_person(
        ids_frame, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
    )
    ids_a = set(left["person_id"])
    ids_b = set(right["person_id"])
    mom_a = disability.reference_moments(panel, ids_a)
    mom_b = disability.reference_moments(panel, ids_b)

    cells: dict[str, Any] = {}
    for key in mom_a:
        r_a, r_b = mom_a[key]["rate"], mom_b[key]["rate"]
        defined = r_a > 0 and r_b > 0
        cells[key] = {
            "rate_a": float(r_a),
            "rate_b": float(r_b),
            "n_events_a": int(mom_a[key]["n_events"]),
            "n_events_b": int(mom_b[key]["n_events"]),
            "log_ratio_abs": (
                float(abs(np.log(r_a / r_b))) if defined else None
            ),
            "pct_diff_abs": (
                float(abs(r_a - r_b) / r_b * 100.0) if defined else None
            ),
        }

    prev_a = prevalence_shares(mom_a)
    ce_a = conversion_exit_shares(panel, ids_a)
    ce_b = conversion_exit_shares(panel, ids_b)
    anchor: dict[str, Any] = {}
    for sex in disability.SEXES:
        shares = prev_a[sex]
        gaps = [
            float(shares[i + 1] - shares[i])
            for i in range(len(BRIDGED_BANDS) - 1)
        ]
        sa, sb = ce_a[sex]["share"], ce_b[sex]["share"]
        anchor[sex] = {
            "prev_min_gap": min(gaps) if gaps else 0.0,
            "prev_peak_is_5059": bool(
                int(np.argmax(shares[: len(BRIDGED_BANDS)]))
                == BRIDGED_BANDS.index(PEAK_BAND)
            ),
            "conv_exit_share_a": float(sa),
            "conv_exit_share_b": float(sb),
            "conv_exit_dominant_both": bool(sa > 0.5 and sb > 0.5),
            "conv_exit_logratio_abs": (
                float(abs(math.log(sa / sb))) if sa > 0 and sb > 0 else None
            ),
        }
    # conversion-analog sex ordering (the reported, fragile cell)
    anchor["sexorder_male_gt_female_a"] = bool(
        mom_a["conversion.retired_from_disabled|male"]["rate"]
        > mom_a["conversion.retired_from_disabled|female"]["rate"]
    )
    return {
        "seed": seed,
        "n_persons_side_a": len(ids_a),
        "n_persons_side_b": len(ids_b),
        "cells": cells,
        "anchor": anchor,
    }


# --------------------------------------------------------------------------
# Internal (holdout) floor pooling + partition -- the gate-2 machinery
# --------------------------------------------------------------------------
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
    """Across-seed floor per cell + the raw stability data (pre power-cap)."""
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
            "family": _family(key),
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
    """round(mean + DRAFT_K*sd, DRAFT_ROUNDING) for every floored cell."""
    return {
        key: round(block["mean"] + DRAFT_K * block["sd"], DRAFT_ROUNDING)
        for key, block in noise_floor.items()
    }


def _events_ok(stab: dict[str, Any]) -> bool:
    return (
        stab["defined_seeds"] == stab["n_seeds"]
        and stab["min_events_either_half"] >= MIN_EVENTS_FOR_GATE
    )


def _demote_reason(
    key: str, stability: dict, tolerances: dict[str, float]
) -> str:
    """The machine reason a cell is report-only (surface d, family-aware)."""
    stab = stability[key]
    if stab["family"] not in FAMILY_INTERNAL_GATED:
        # Prevalence/conversion levels are bridged by the anchor SHAPE cells,
        # never by an internal level floor (the concept deltas block it).
        return "level_bridged_via_anchor_shape"
    if stab["defined_seeds"] != stab["n_seeds"]:
        return "undefined_on_some_seed"
    if stab["min_events_either_half"] < MIN_EVENTS_FOR_GATE:
        return "below_20_events"
    if key not in tolerances:
        return "no_floor"
    if tolerances[key] > T_MAX:
        return "tolerance_above_t_max"
    return "gate_eligible"


def partition_cells(
    stability: dict[str, Any], tolerances: dict[str, float]
) -> tuple[set[str], set[str], dict[str, str]]:
    """The internal (surface d) gate-eligible / report-only partition.

    A transition-hazard cell gates iff it is defined on every seed, carries
    >=20 events on the weaker half of the worst seed, AND its stabilised
    tolerance <= T_max. Prevalence / conversion cells are never internally
    gated -- their reason is ``level_bridged_via_anchor_shape`` and they are
    gated (or not) as ordinal anchor cells instead.
    """
    gated: set[str] = set()
    report: set[str] = set()
    reasons: dict[str, str] = {}
    for key in stability:
        reason = _demote_reason(key, stability, tolerances)
        reasons[key] = reason
        if reason == "gate_eligible":
            gated.add(key)
        else:
            report.add(key)
    return gated, report, reasons


def draft_thresholds(
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    """DRAFT per-cell |ln ratio| tolerances for the GATED internal cells."""
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
    """Fold the tolerance, sigma, gate flag and report reason into stability."""
    for key, stab in stability.items():
        tol = tolerances.get(key)
        sigma = noise_floor.get(key, {}).get("realized_sigma")
        stab["tolerance"] = tol
        stab["realized_sigma"] = sigma
        stab["tolerance_sigma_units"] = (
            round(tol / sigma, 3) if tol is not None and sigma else None
        )
        stab["gate_eligible"] = key in gated
        stab["report_reason"] = reasons.get(key)


def k_sensitivity(noise_floor: dict[str, Any]) -> dict[str, Any]:
    """How many transition cells gate at k in {2, 3, 4} -- the disclosure
    that k=4 (the gate-2c inheritance) empties the internal surface while
    k=3 (this DRAFT, the m4_disability_v1 proposal) gates the well-powered
    prime-age/older cells."""
    out: dict[str, Any] = {}
    for k in (2, 3, 4):
        elig = sorted(
            key
            for key, b in noise_floor.items()
            if _family(key) in FAMILY_INTERNAL_GATED
            and round(b["mean"] + k * b["sd"], DRAFT_ROUNDING) <= T_MAX
        )
        out[str(k)] = {"n_gate_eligible": len(elig), "cells": elig}
    return out


# --------------------------------------------------------------------------
# In-repo SSA anchor tables (sha-pinned) and their extracted values
# --------------------------------------------------------------------------
def _t19_within_sex_shares(tables: dict) -> dict[str, dict[str, Any]]:
    """Table 19 (2023) collapsed to the five DI_AGE_BANDS, per sex, as a
    share of the sex total (the administrative prevalence age-shape)."""
    tsv = tables["Table 19"]["tsv"]
    section = None
    out: dict[str, dict[str, Any]] = {}
    for line in tsv.split("\n"):
        parts = line.split("\t")
        marker = parts[1] if len(parts) > 1 else ""
        if marker == "Men":
            section = "male"
            continue
        if marker == "Women":
            section = "female"
            continue
        if parts[0] == "2023" and section:
            # parts: Year Number 100.0 U30 30-34 35-39 40-44 45-49 50-54
            #        55-59 60-FRA AvgAge
            pct = [float(x) for x in parts[3:11]]
            collapsed = [
                pct[0],  # 20-29 (Under 30)
                pct[1] + pct[2],  # 30-39
                pct[3] + pct[4],  # 40-49
                pct[5] + pct[6],  # 50-59
                pct[7],  # 60-66 (60-FRA)
            ]
            total = sum(collapsed)
            out[section] = {
                "pct_by_band": dict(zip(ALL_BANDS, collapsed, strict=True)),
                "share_by_band": dict(
                    zip(
                        ALL_BANDS,
                        [c / total for c in collapsed],
                        strict=True,
                    )
                ),
            }
    return out


def _t50_worker_terminations(tables: dict) -> dict[str, Any]:
    """Table 50 worker-column termination reasons + the non-death
    FRA-conversion share (the anchor for surface a)."""
    tsv = tables["Table 50"]["tsv"]
    want = {
        "Total": "total",
        "Death of beneficiary": "death",
        "FRA by disabled worker": "fra_conversion",
        "Does not meet medical standards b": "medical_cessation",
    }
    vals: dict[str, int] = {}
    for line in tsv.split("\n"):
        parts = line.split("\t")
        if parts and parts[0] in want:
            # Worker column is the 3rd cell (All beneficiaries, Workers, ...)
            vals[want[parts[0]]] = int(parts[2].replace(",", ""))
    conv = vals["fra_conversion"]
    cess = vals["medical_cessation"]
    return {
        "worker_terminations_total": vals["total"],
        "death": vals["death"],
        "fra_conversion": conv,
        "medical_cessation": cess,
        "fra_conversion_share_of_terminations": conv / vals["total"],
        "fra_conversion_share_of_nondeath_exits": conv / (conv + cess),
    }


def _t49_worker_rate_trend(tables: dict) -> dict[str, Any]:
    """Table 49 worker termination rate per 1,000, series + recent trend."""
    tsv = tables["Table 49"]["tsv"]
    rate_by_year: dict[int, int] = {}
    for line in tsv.split("\n"):
        parts = line.split("\t")
        if parts and parts[0].isdigit() and len(parts) >= 5:
            try:
                rate_by_year[int(parts[0])] = int(parts[4])
            except ValueError:
                continue
    recent_years = [y for y in range(2015, 2024) if y in rate_by_year]
    recent = [rate_by_year[y] for y in recent_years]
    slope = float(np.polyfit(recent_years, recent, 1)[0])
    return {
        "worker_rate_2015": rate_by_year.get(2015),
        "worker_rate_2023": rate_by_year.get(2023),
        "worker_rate_2003_trough": rate_by_year.get(2003),
        "worker_rate_1960": rate_by_year.get(1960),
        "recent_trend_2015_2023_slope_per_1000_per_yr": round(slope, 3),
        "recent_trend_direction": "rising" if slope > 0 else "falling",
    }


def anchor_tables_block() -> dict[str, Any]:
    """The sha-pinned in-repo DI ASR 2023 tables + extracted anchor values."""
    tables = json.loads(ANCHOR_TABLES_PATH.read_text())
    return {
        "source": (
            "SSA Disability Insurance Annual Statistical Report 2023 "
            "(Tables 19, 35, 36, 49, 50), extracted in-browser to "
            f"{ANCHOR_TABLES_REL} (see its provenance.md)"
        ),
        "path": ANCHOR_TABLES_REL,
        "sha256": _sha256_bytes(ANCHOR_TABLES_PATH),
        "n_bytes": ANCHOR_TABLES_PATH.stat().st_size,
        "t19_prevalence_age_shape_2023": _t19_within_sex_shares(tables),
        "t50_worker_terminations_2023": _t50_worker_terminations(tables),
        "t49_worker_termination_rate": _t49_worker_rate_trend(tables),
    }


# --------------------------------------------------------------------------
# Anchor structural (ordinal) gated cells -- surfaces (a) and (b)
# --------------------------------------------------------------------------
def _anchor_stat_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
    }


def prevalence_ageshape_checks(
    moments: dict[str, dict[str, float]],
    per_seed: list[dict[str, Any]],
    anchor_tables: dict[str, Any],
) -> dict[str, Any]:
    """Surface (b): per-sex co-monotone-rise + 50-59-peak vs Table 19.

    Bounded/ordinal statistic (unit-honest -- NOT a |ln ratio| gate): the
    gate rule is that the invariant holds on the full sample AND on all 100
    person-disjoint halves AND rank-agrees with the Table 19 age-shape over
    the bridged bands. The 60-66 band and the per-band levels are report-only.
    """
    full = prevalence_shares(moments)
    t19 = anchor_tables["t19_prevalence_age_shape_2023"]
    checks: dict[str, Any] = {}
    for sex in disability.SEXES:
        shares = full[sex]
        gaps = [
            float(shares[i + 1] - shares[i])
            for i in range(len(BRIDGED_BANDS) - 1)
        ]
        min_gap = min(gaps)
        rises = all(g > 0 for g in gaps)
        peak = int(np.argmax(shares[: len(BRIDGED_BANDS)])) == (
            BRIDGED_BANDS.index(PEAK_BAND)
        )
        # Table 19 age-shape (bridged bands) ranks
        t19_bridged = [t19[sex]["share_by_band"][b] for b in BRIDGED_BANDS]
        psid_bridged = list(shares[: len(BRIDGED_BANDS)])
        rank_match = _rank_order(psid_bridged) == _rank_order(t19_bridged)
        t19_rises = all(
            t19_bridged[i + 1] > t19_bridged[i]
            for i in range(len(t19_bridged) - 1)
        )
        # 100-half stability of the min-gap + peak
        half_min_gaps = [s["anchor"][sex]["prev_min_gap"] for s in per_seed]
        peak_halves = sum(
            s["anchor"][sex]["prev_peak_is_5059"] for s in per_seed
        )
        gap_floor = _anchor_stat_summary(half_min_gaps)
        sd = gap_floor["sd"]
        margin_sigma = round(min_gap / sd, 3) if sd > 0 else None
        holds_all_halves = all(g > 0 for g in half_min_gaps) and (
            peak_halves == len(per_seed)
        )
        gate_rule = (
            rises
            and peak
            and rank_match
            and t19_rises
            and holds_all_halves
            and (margin_sigma is not None and margin_sigma >= MARGIN_K)
        )
        checks[f"prevalence_ageshape.comonotone|{sex}"] = {
            "statistic": (
                "disabled-prevalence share strictly rises across "
                "20-29<30-39<40-49<50-59 and peaks at 50-59, co-monotone "
                "with the Table 19 age-shape over the bridged bands"
            ),
            "gated": bool(gate_rule),
            "psid_share_by_band": dict(
                zip(ALL_BANDS, [float(x) for x in shares], strict=True)
            ),
            "psid_min_adjacent_gap": min_gap,
            "psid_working_age_peak": (
                PEAK_BAND
                if peak
                else ALL_BANDS[int(np.argmax(shares[: len(BRIDGED_BANDS)]))]
            ),
            "psid_strictly_rises": bool(rises),
            "t19_share_by_band": t19[sex]["share_by_band"],
            "t19_strictly_rises_bridged": bool(t19_rises),
            "rank_agreement_over_bridged_bands": bool(rank_match),
            "half_split_floor": {
                "min_gap": gap_floor,
                "holds_on_all_halves": bool(holds_all_halves),
                "peak_is_5059_on_n_halves": int(peak_halves),
            },
            "margin_sigma_units": margin_sigma,
            "gate_rule": (
                "strict rise + 50-59 peak + Table-19 rank agreement + holds "
                f"on all {len(per_seed)} halves + margin >= {MARGIN_K} sigma"
            ),
            "anchor_source": "Table 19 (prevalence by sex x age, 2023)",
            "concept_bridge": (
                "LEVELS are not bridgeable (self-report captures milder / "
                "younger limitation: the 20-29 PSID share is ~5x the Table 19 "
                "share; deltas 1-2). The co-monotone RISE and 50-59 peak are "
                "level-invariant, so the severity/population deltas do not "
                "touch them; the 50-59 share happens to match in level too. "
                "The 60-FRA band is REPORT-ONLY (delta 3: PSID disabled "
                "relabel to retired near FRA, while the DI stock ages into "
                "60-FRA)."
            ),
            "report_only_companions": {
                f"prevalence_level.{b}|{sex}": (
                    "self_report_level_delta (deltas 1-2; only the shape "
                    "bridges)"
                )
                for b in ALL_BANDS
            },
        }
    return checks


def conversion_exit_checks(
    panel: disability.DisabilityPanel,
    per_seed: list[dict[str, Any]],
    anchor_tables: dict[str, Any],
) -> dict[str, Any]:
    """Surface (a): per-sex retirement-conversion dominance among non-death
    FRA-window disability exits vs Table 50.

    Bounded/ordinal statistic (unit-honest): the gate rule is that retirement
    is the majority non-death exit on the full sample AND on all 100 halves,
    co-moving with Table 50 (where FRA-conversion is the dominant non-death
    worker-termination reason). The 6.B5.1 level ratio and the sex-ordering
    are report-only.
    """
    full = conversion_exit_shares(panel)
    t50 = anchor_tables["t50_worker_terminations_2023"]
    t50_share = t50["fra_conversion_share_of_nondeath_exits"]
    t50_dominant = t50_share > 0.5
    checks: dict[str, Any] = {}
    for sex in disability.SEXES:
        share = full[sex]["share"]
        margin = share - 0.5
        half_shares = [s["anchor"][sex]["conv_exit_share_a"] for s in per_seed]
        dominant_halves = sum(
            s["anchor"][sex]["conv_exit_dominant_both"] for s in per_seed
        )
        share_floor = _anchor_stat_summary(half_shares)
        sd = share_floor["sd"]
        margin_sigma = round(margin / sd, 3) if sd > 0 else None
        holds_all_halves = dominant_halves == len(per_seed)
        gate_rule = (
            share > 0.5
            and t50_dominant
            and holds_all_halves
            and (margin_sigma is not None and margin_sigma >= MARGIN_K)
            and full[sex]["n_exits"] >= MIN_EVENTS_FOR_GATE
        )
        checks[f"conversion_exit.retirement_dominant|{sex}"] = {
            "statistic": (
                "weighted share of non-death disability exits at ages "
                f"{EXIT_WINDOW[0]}-{EXIT_WINDOW[1]} that go to retired "
                "(vs return-to-work)"
            ),
            "gated": bool(gate_rule),
            "psid_retirement_exit_share": share,
            "psid_dominance_margin": margin,
            "psid_n_exits": full[sex]["n_exits"],
            "psid_n_retired": full[sex]["n_retired"],
            "t50_nondeath_fra_conversion_share": t50_share,
            "t50_is_dominant": bool(t50_dominant),
            "half_split_floor": {
                "share": share_floor,
                "dominant_on_n_halves": int(dominant_halves),
                "holds_on_all_halves": bool(holds_all_halves),
            },
            "margin_sigma_units": margin_sigma,
            "gate_rule": (
                "retirement share > 0.5 + Table-50 also dominant + dominant "
                f"on all {len(per_seed)} halves + margin >= {MARGIN_K} sigma "
                f"+ >= {MIN_EVENTS_FOR_GATE} exits"
            ),
            "anchor_source": "Table 50 (worker terminations by reason, 2023)",
            "concept_bridge": (
                "The exit-reason DOMINANCE is concept-robust: near FRA, "
                "retirement is the majority non-death exit from disability in "
                "both the PSID self-report and Table 50 (FRA-conversion "
                f"{t50['fra_conversion']:,} vs medical-cessation "
                f"{t50['medical_cessation']:,} worker terminations). Deaths "
                "are right-censored in the PSID pairs (no w' observation), so "
                "this is the retirement-vs-return-to-work split among the "
                "OBSERVABLE exits -- a named censoring delta, matched by "
                "excluding death from the Table 50 denominator. LEVELS "
                "(6.B5.1 ratio) and the sex-ordering are report-only."
            ),
        }
    return checks


def conversion_reports(
    moments: dict[str, dict[str, float]],
    per_seed: list[dict[str, Any]],
    claim_ref: claiming.ClaimAgeReference,
) -> dict[str, Any]:
    """Report-only conversion cells: the 6.B5.1 level ratio (denominator
    delta) and the sex-ordering (half-split fragile)."""
    admin = {
        sex: float(
            np.mean(
                [
                    claim_ref.row(sex, y)["categories"][
                        "disability_conversion"
                    ]
                    for y in claim_ref.years()
                ]
            )
        )
        for sex in disability.SEXES
    }
    sexorder_halves = sum(
        s["anchor"]["sexorder_male_gt_female_a"] for s in per_seed
    )
    by_sex = {}
    for sex in disability.SEXES:
        psid = moments[f"conversion.retired_from_disabled|{sex}"]["rate"] * 100
        by_sex[sex] = {
            "psid_conversion_analog_pct": round(psid, 3),
            "admin_6b51_mean_pct": round(admin[sex], 3),
            "ratio_psid_to_admin": round(psid / admin[sex], 4),
        }
    return {
        "gated": False,
        "report_only": True,
        "level_ratio": {
            "note": (
                "conversion_denominator_delta (delta 4): 6.B5.1 is "
                "conversions / retired-worker AWARDS; the PSID analog is "
                "disabled->retired / all self-reported retirement entries. "
                "Different denominators -> the ~0.27-0.32 ratio is REPORTED, "
                "never calibrated."
            ),
            "by_sex": by_sex,
        },
        "sex_ordering": {
            "psid_male_gt_female_full_sample": bool(
                moments["conversion.retired_from_disabled|male"]["rate"]
                > moments["conversion.retired_from_disabled|female"]["rate"]
            ),
            "admin_6b51_male_gt_female": bool(admin["male"] > admin["female"]),
            "psid_male_gt_female_on_n_halves": int(sexorder_halves),
            "n_halves": len(per_seed),
            "note": (
                "The PSID conversion-analog sex-ordering (male>female) matches "
                "the 6.B5.1 admin ordering on the full sample but flips on "
                f"{len(per_seed) - sexorder_halves}/{len(per_seed)} halves -- "
                "REPORT-ONLY (half_split_fragile), not a gated co-movement."
            ),
        },
    }


def termination_trend_report(anchor_tables: dict[str, Any]) -> dict[str, Any]:
    """Report-only surface (c): Table 49 worker termination-rate trend, with
    the period-pooling delta that blocks a PSID trend gate."""
    t49 = anchor_tables["t49_worker_termination_rate"]
    return {
        "gated": False,
        "report_only": True,
        "table_49_worker_rate": t49,
        "report_reason": "no_pooled_year_series",
        "note": (
            "Table 49's worker termination rate rises "
            f"{t49['worker_rate_2015']}->{t49['worker_rate_2023']} per 1,000 "
            "over 2015-2023 (a secular fall from "
            f"{t49['worker_rate_1960']} in 1960 to a "
            f"{t49['worker_rate_2003_trough']} trough in 2003, then a rise). "
            "The PSID hazards POOL covered waves 1982-2023 (period-pooling "
            "delta 6), so there is no year-indexed PSID analog to gate this "
            "trend against. Recorded as context, never gated; a windowed "
            "year-indexed PSID hazard is future work (see wanted_ssa_tables)."
        ),
    }


# --------------------------------------------------------------------------
# Candidate operating characteristic (internal cells) -- the gate-2 mirror
# --------------------------------------------------------------------------
def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def training_copy_check(
    per_seed: list[dict[str, Any]],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    """A train-half copy scored under option (a): it passes at ~the floor."""
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
                worst_cell = [key, s["seed"]]
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
            "+ holdout exclusion), NOT the cell set. What the internal cells "
            "catch is a candidate that mis-reproduces the PSID incidence / "
            "recovery hazard beyond real-vs-real noise."
        ),
    }


def faithful_candidate_oc(
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    """The 4-of-5 operating characteristic for a faithful candidate."""
    per_cell = {}
    p_seed = 1.0
    for key in sorted(gated):
        sigma = noise_floor[key]["realized_sigma"]
        tol = tolerances[key]
        p = (2.0 * _normal_cdf(tol / sigma) - 1.0) if sigma > 0 else 1.0
        per_cell[key] = {
            "tolerance": tol,
            "realized_sigma": sigma,
            "cell_pass_prob": round(p, 6),
        }
        p_seed *= p
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return {
        "method": (
            "Independence-approx normal OC. Per gated internal cell a "
            "faithful candidate's score ~ half-normal(realized_sigma); cell "
            "pass = 2*Phi(tolerance/sigma)-1. Seed pass = product over gated "
            "cells; gate = P(>=4 of 5) = p^5 + 5 p^4 (1-p). The K=20 "
            "estimator shares this draw-noise-free basis."
        ),
        "n_gated_internal_cells": len(gated),
        "p_seed_pass": round(p_seed, 4),
        "p_gate_pass_4_of_5": round(p_gate, 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# Concept deltas, wanted tables (statuses updated), warts, note
# --------------------------------------------------------------------------
def concept_deltas() -> list[dict[str, str]]:
    """Reuse the seven named deltas from the M4 foundation builder."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import build_disability_floors as foundation

    return foundation.concept_deltas()


def wanted_ssa_tables() -> list[dict[str, Any]]:
    """The SSA series a level-anchored M4 gate needs. Tables 19/49/50 are now
    IN HAND (di_asr_2023); the Trustees rate surfaces + insured denominator
    remain WANTED."""
    return [
        {
            "series": "SSA DI Annual Statistical Report 2023",
            "table": "Table 19 (disabled workers by sex x age)",
            "needed_for": "the PREVALENCE age-shape anchor (surface b)",
            "status": "IN HAND (di_asr_2023; gated in prevalence_ageshape)",
        },
        {
            "series": "SSA DI Annual Statistical Report 2023",
            "table": "Table 50 (worker terminations by reason)",
            "needed_for": "the CONVERSION-exit dominance anchor (surface a)",
            "status": "IN HAND (di_asr_2023; gated in conversion_exit)",
        },
        {
            "series": "SSA DI Annual Statistical Report 2023",
            "table": "Table 49 (worker termination rate, 1960-2023)",
            "needed_for": "the termination-rate TREND context (surface c)",
            "status": "IN HAND (di_asr_2023; report-only, period-pooling)",
        },
        {
            "series": "SSA DI Annual Statistical Report 2023",
            "table": "Tables 35/36 (disabled-worker awards by age x sex)",
            "needed_for": (
                "an INCIDENCE (onset) age-shape anchor -- awards are a flow, "
                "the counterpart to the incidence hazard"
            ),
            "status": "IN HAND (di_asr_2023; future incidence-shape surface)",
        },
        {
            "series": "OASDI Trustees Report",
            "table": "Tables V.C5 / V.C6 (incidence + termination rates)",
            "needed_for": "the M6 projection assumption-layer rate surfaces",
            "status": "WANTED",
        },
        {
            "series": "SSA Annual Statistical Supplement",
            "table": "Table 4.C2 (disability-insured workers by age x sex)",
            "needed_for": "the insured incidence-rate denominator",
            "status": "WANTED",
        },
        {
            "series": "SSA Annual Statistical Supplement",
            "table": "Table 6.B5.1 (disability-conversion column)",
            "needed_for": "the conversion level context (report-only)",
            "status": "IN HAND (archived claim-age reference)",
        },
    ]


def protocol_block(
    n_gated_internal: int, n_gated_anchor: int
) -> dict[str, Any]:
    """The candidate scoring protocol a future DI model would be held to --
    the ratified K=20 mean-over-draws estimator, adopted from the START."""
    return {
        "candidate_estimator": (
            "mean-over-K=20-draws (tranche-2a amendment 1, ratified "
            "2026-07-08). Adopted from the START so the single-draw dead zone "
            "the 2b/2c round-1 referees flagged never appears."
        ),
        "candidate_draws": CANDIDATE_DRAWS,
        "candidate_draw_stream": CANDIDATE_DRAW_STREAM,
        "draw_stream_base": DRAW_STREAM_BASE,
        "internal_statistic": (
            "per gated internal cell, |ln(rbar_candidate,s / r_holdout,s)| "
            "<= the committed tolerance, where rbar is the mean over the 20 "
            "draws of the cell RATE (not the mean of per-draw scores)"
        ),
        "anchor_statistic": (
            "per gated anchor cell, the ordinal invariant (co-monotone "
            "prevalence rise + 50-59 peak vs Table 19; retirement-dominant "
            "non-death FRA exit vs Table 50) holds on the candidate's "
            "simulated holdout, as it does on the real full sample and all "
            "100 halves"
        ),
        "conjunction": (
            "seed passes iff every gated cell holds; the gate passes iff "
            ">= 4 of 5 gate seeds pass"
        ),
        "n_gated_internal_cells": n_gated_internal,
        "n_gated_anchor_cells": n_gated_anchor,
        "note": (
            "No candidate is scored HERE: this is the reported floor + anchor "
            "basis. The protocol is recorded so the standard exists before "
            "any scored use (the pre-registration discipline)."
        ),
    }


def warts(
    k_sens: dict[str, Any],
    gated_internal: set[str],
    anchor_checks: dict[str, Any],
) -> list[dict[str, str]]:
    """Honest limitations a referee should see up front."""
    return [
        {
            "wart": "k=4 empties the internal surface",
            "detail": (
                "At the gate-2c inherited k=4, "
                f"{k_sens['4']['n_gate_eligible']} transition cells gate "
                "(all demote, tolerance_above_t_max): the PSID self-report "
                "hazards are too noisy at half-panel to pin to within 1.5x. "
                "This DRAFT uses k=3 (the top of the [2,3] range "
                "m4_disability_v1's proposed_thresholds_note pre-registered), "
                f"at which {k_sens['3']['n_gate_eligible']} well-powered "
                "prime-age/older cells gate. k stays a DRAFT for the referee "
                "rounds; the k-sensitivity is committed (k_sensitivity)."
            ),
        },
        {
            "wart": "the gate leans on the anchor cells",
            "detail": (
                f"{len(gated_internal)} internal (holdout) level cells gate "
                f"vs {sum(c['gated'] for c in anchor_checks.values())} anchor "
                "structural cells; the binding power is mostly the "
                "concept-bridged SHAPE/DOMINANCE anchors, not internal "
                "levels. That is by design (the concept deltas block most "
                "level gates), but a referee should weigh that the DI gate is "
                "predominantly an ordinal-anchor gate."
            ),
        },
        {
            "wart": "half-samples overlap across seeds",
            "detail": (
                "The 100 per-seed side-A statistics are drawn from "
                "overlapping halves, so their sd underestimates independent "
                "sampling noise (the gate-2/2c floor convention). The "
                "primary no-dead-zone evidence for the anchor cells is the "
                "non-parametric 'holds on all 100 halves', not the sd."
            ),
        },
        {
            "wart": "death is censored in the conversion-exit statistic",
            "detail": (
                "A death between waves forms no pair, so the conversion-exit "
                "share is retirement vs return-to-work among OBSERVABLE "
                "exits; the Table 50 anchor is matched by excluding death "
                "from its denominator. The full three-way (death) exit "
                "composition is not identified on the biennial PSID grid."
            ),
        },
        {
            "wart": "single-era anchor vs pooled hazards",
            "detail": (
                "Tables 19/49/50 are 2023; the PSID hazards pool 1982-2023. "
                "DI prevalence/incidence have strong secular trends (delta "
                "6), so the age-SHAPE bridge assumes the shape is more "
                "period-stable than the levels -- defensible for the rising "
                "limb and the 50-59 peak, disclosed for the rest."
            ),
        },
    ]


def _gate_m4_stub_proposal(
    gated_internal: set[str],
    anchor_checks: dict[str, Any],
) -> str:
    gated_anchor = sorted(k for k, v in anchor_checks.items() if v["gated"])
    return (
        "PROPOSED gate_m4 STUB (for gates.yaml -- NOT written here; the flip "
        "is a later ceremony step, gates.yaml stays UNTOUCHED):\n"
        "  gate_m4:\n"
        "    status: unlocked\n"
        "    locked: false\n"
        "    tranche_id: m4_disability\n"
        "    holdout_basis: [ind2023er_employment_status, di_asr_2023]\n"
        "    kind: anchor_based   # gate-1 external-check style, not gate-2 "
        "holdout-only\n"
        "    statistic: >-\n"
        "      per internal incidence/recovery cell, |ln(rbar_candidate / "
        "r_PSID)| <= round(floor_mean + k*sd, 3) capped at ln(1.5), k DRAFT "
        "3; per anchor cell, the ordinal invariant (co-monotone prevalence "
        "rise + 50-59 peak vs Table 19; retirement-dominant non-death FRA "
        "exit vs Table 50) holding on the full sample and all 100 halves.\n"
        "    candidate_estimator: mean-over-K=20-draws (rng 4100+k)\n"
        "    conjunction: all gated cells per seed AND >=4 of 5 gate seeds\n"
        f"    gated_internal_cells: {sorted(gated_internal)}\n"
        f"    gated_anchor_cells: {gated_anchor}\n"
        "    external_anchor: bundled (di_asr_2023, sha-pinned); the DI "
        "concept deltas keep every un-bridgeable LEVEL report-only."
    )


def draft_thresholds_note(
    gated_internal: set[str],
    report_internal: set[str],
    anchor_checks: dict[str, Any],
    k_sens: dict[str, Any],
) -> str:
    gated_anchor = sorted(k for k, v in anchor_checks.items() if v["gated"])
    return (
        "DRAFT M4 DISABILITY GATE EVIDENCE -- CEREMONY STEP 1, NOT RATIFIED "
        "(v1). Floors + anchor basis for the DI gate (roadmap #113 M4). It "
        "changes no locked value, no model reads it, and it writes NO "
        "gates.yaml block: gates.yaml stays UNTOUCHED and the gate_m4 "
        "stub-flip (proposal -> referee rounds -> fixes -> verification -> "
        "maintainer ratify) inserts thresholds later. The k value (3) is a "
        "DRAFT for the ceremony to fix.\n\n"
        "DESIGN: ANCHOR-BASED (gate-1 external-check style), not holdout-only "
        "(gate-2). The PSID self-report is NOT the SSA DI program (seven named "
        "concept_deltas), so no SSA DI LEVEL is gated -- that would reject "
        "reality. Gated cells live only on concept-bridgeable surfaces:\n"
        "  (a) conversion_exit vs Table 50: retirement is the majority "
        "non-death disability exit near FRA (dominance, per sex);\n"
        "  (b) prevalence_ageshape vs Table 19: co-monotone rise + 50-59 peak "
        "(ordinal, per sex);\n"
        "  (d) internal incidence/recovery: |ln(candidate/PSID)| within the "
        "100-seed half-split floor (holdout reproduction).\n"
        "Report-only (delta named, never calibrated): (c) the Table 49 "
        "termination-rate trend (period-pooling); the 6.B5.1 conversion level "
        "and sex-ordering; all prevalence LEVELS and the 60-FRA band; the "
        "noisy transition cells above the ln(1.5) cap.\n\n"
        "STATISTIC + INTERNAL FLOOR. Per age band x sex the weighted "
        "per-interval PSID transition rate and a candidate's same-band rate; "
        "discrepancy scored as |ln(r_candidate / r_PSID)|. The person-disjoint "
        "50/50 half-split |ln ratio| floor (seeds 0-99) gives round(mean + "
        f"{DRAFT_K}*sd, {DRAFT_ROUNDING}) tolerances, power-capped at "
        f"{T_MAX_SOURCE}. At k=4 the internal surface is EMPTY "
        f"({k_sens['4']['n_gate_eligible']} cells); at this DRAFT k=3 it is "
        f"{sorted(gated_internal)} (report-only: {sorted(report_internal)}).\n\n"
        "ANCHOR CELLS (bounded/ordinal -- unit-honest, NOT |ln ratio| gates). "
        f"Gated: {gated_anchor}. Each holds on the full sample AND all 100 "
        f"person-disjoint halves AND rank/dominance-agrees with the sha-pinned "
        "in-repo Table 19 / Table 50, with the margin in >= "
        f"{MARGIN_K} sigma. A label swap (male<->female shape, or "
        "retirement<->work exit) inverts them -- caught in CI with no PSID.\n\n"
        + _gate_m4_stub_proposal(gated_internal, anchor_checks)
        + "\n\nRATIFICATION requires the full lock ceremony: this floor -> "
        "adversarial referee rounds -> verification -> maintainer ratify by "
        "merge, exactly as gate-2c step 1 preceded the gate_2c flip."
    )


# --------------------------------------------------------------------------
# Provenance + driver
# --------------------------------------------------------------------------
def _rev(cwd: Path, ref: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", ref], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _merge_base(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "merge-base", "HEAD", "origin/master"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def run(verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    assert "populace.fit" not in sys.modules

    status = disability.read_disability_status()
    death_records = deaths.read_death_records()
    panel = disability.build_disability_panel(status, death_records)
    coding = disability.verify_employment_status_codes()
    moments = disability.reference_moments(panel)
    keys = sorted(moments)

    per_seed = [measure_seed_halfsplit(s, panel) for s in SEEDS]
    noise_floor, stability = pool_internal_floor(per_seed, keys)
    tolerances = raw_tolerances(noise_floor)
    gated, report, reasons = partition_cells(stability, tolerances)
    annotate_stability(stability, noise_floor, tolerances, gated, reasons)
    thresholds = draft_thresholds(noise_floor, tolerances, gated)
    k_sens = k_sensitivity(noise_floor)

    anchor_tables = anchor_tables_block()
    prev_checks = prevalence_ageshape_checks(moments, per_seed, anchor_tables)
    conv_checks = conversion_exit_checks(panel, per_seed, anchor_tables)
    anchor_checks = {**prev_checks, **conv_checks}
    claim_ref = claiming.load_claim_age_reference()
    conv_reports = conversion_reports(moments, per_seed, claim_ref)
    trend_report = termination_trend_report(anchor_tables)

    copy_check = training_copy_check(per_seed, tolerances, gated)
    oc = faithful_candidate_oc(noise_floor, tolerances, gated)

    gated_anchor = {k for k, v in anchor_checks.items() if v["gated"]}
    all_gated = sorted(gated) + sorted(gated_anchor)
    all_report = sorted(report) + sorted(set(anchor_checks) - gated_anchor)

    n_persons = int(panel.person_years["person_id"].nunique())
    waves = sorted(int(w) for w in status["period"].unique())
    if verbose:
        print(
            f"panel: {len(panel.person_years)} py, {n_persons} persons, "
            f"{len(panel.pairs)} pairs"
        )
        print(f"internal gated (k={DRAFT_K}): {sorted(gated)}")
        print(f"anchor gated: {sorted(gated_anchor)}")
        print(
            "k-sensitivity: "
            + ", ".join(
                f"k={k}:{v['n_gate_eligible']}" for k, v in k_sens.items()
            )
        )

    # Store per_seed for the 5 gate seeds only (bound artifact size).
    per_seed_stored = [
        {
            "seed": s["seed"],
            "n_persons_side_a": s["n_persons_side_a"],
            "n_persons_side_b": s["n_persons_side_b"],
            "cells": s["cells"],
            "anchor": s["anchor"],
        }
        for s in per_seed
        if s["seed"] in GATE_SEEDS
    ]

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "reported_anchor_not_gated": True,
        "referee_round": "PR (draft -- ceremony step 1; 2b/2c lessons preempted)",
        "component": (
            "disability: DI incidence/recovery/prevalence + DI->retirement "
            "conversion gate floors (roadmap #113, M4)"
        ),
        "purpose": (
            "Ceremony step 1 for the DI gate: the anchor + holdout evidence "
            "the gate_m4 thresholds derive from. ANCHOR-BASED (gate-1 "
            "external-check style): concept-bridged SHAPE/DOMINANCE cells "
            "gated against the sha-pinned in-repo SSA DI ASR 2023 tables "
            "(19/49/50), plus the gate-2 100-seed half-split floor on the "
            "incidence/recovery hazards. Every SSA DI LEVEL the seven concept "
            "deltas block stays report-only, never calibrated. Reads no gate "
            "and changes no gate; gates.yaml is UNTOUCHED. See "
            "draft_thresholds_note (NOT RATIFIED)."
        ),
        "holdout_basis": [
            "ind2023er_employment_status (PSID self-report)",
            "di_asr_2023 (in-repo SSA DI ASR tables, sha-pinned)",
        ],
        "ceremony": {
            "tranche": "m4_disability",
            "gates_yaml_stub": "gate_m4 (PROPOSED; not yet in gates.yaml)",
            "step": (
                "1 of the lock ceremony: the 100-seed floor + the "
                "concept-bridged anchor cells. Referee rounds, verification "
                "and the gates.yaml flip come AFTER."
            ),
            "gates_yaml_untouched": True,
            "mirrors": "runs/gate2c_floors_v1.json (schema) + m4_disability_v1.json (moments)",
            "lessons_preempted": (
                "K=20 mean-over-draws estimator from the START; estimand "
                "named honestly (self-report != DI, seven deltas); machine "
                "reasons on every report-only cell; no dead-zone wide "
                "tolerances (ln(1.5) cap demotes noisy cells); bounded "
                "statistics (prevalence shape, conversion-exit share) gated "
                "as ordinal invariants, never as |ln ratio| floors "
                "(unit-correlation-honest)."
            ),
        },
        "data": {
            "psid_micro_basis": (
                "populace_dynamics.data.disability.read_disability_status -- "
                "cross-year individual-file EMPLOYMENT STATUS (code 5 = "
                "'Permanently disabled'), a SELF-REPORTED labor-force status, "
                "joined to person-constant sex (ER32000)"
            ),
            "n_persons": n_persons,
            "n_person_years": int(len(panel.person_years)),
            "n_transition_pairs": int(len(panel.pairs)),
            "waves_covered": waves,
            "employment_status_code_verification": {
                "n_waves_verified": len(coding),
                "example_2023": coding.get(2023),
            },
        },
        "gate_design": {
            "kind": "anchor_based",
            "contrast": (
                "gate-1 external checks anchor to admin series; gate-2/2b/2c "
                "are holdout-only. M4 is anchor-based: the DI concept deltas "
                "forbid a self-report level gate, so the gated surfaces are "
                "concept-bridged SHAPE/DOMINANCE vs the SSA tables, with the "
                "gate-2 half-split floor reused for hazard REPRODUCTION only."
            ),
            "surfaces": {
                "a_conversion_exit_vs_t50": "GATED (dominance, per sex)",
                "b_prevalence_ageshape_vs_t19": "GATED (co-monotone+peak, per sex)",
                "c_termination_trend_vs_t49": "REPORT-ONLY (period-pooling)",
                "d_internal_hazard_floor": (
                    f"GATED at k={DRAFT_K} ({len(gated)} cells); the rest "
                    "report-only (tolerance_above_t_max)"
                ),
            },
        },
        "reference_moments": moments,
        "anchor_tables": anchor_tables,
        "anchor_checks": anchor_checks,
        "conversion_reports": conv_reports,
        "termination_trend_report": trend_report,
        "internal_noise_floor": {
            "method": (
                "person-disjoint 50/50 half-split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction=0.5, seeds 0-99) of the same band x sex reference "
                "moments; floor statistic is |ln(r_A / r_B)|"
            ),
            "split_unit": "person",
            "split_fraction": SPLIT_FRACTION,
            "floor_seeds": list(SEEDS),
            "gate_seeds": list(GATE_SEEDS),
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
            "t_max": T_MAX,
            "t_max_source": T_MAX_SOURCE,
            "noise_floor_seeds_0_99": noise_floor,
            "cell_stability": stability,
            "k_sensitivity": k_sens,
            "per_seed": per_seed_stored,
        },
        "gate_partition": {
            "gate_eligible": all_gated,
            "report_only": all_report,
            "n_gate_eligible": len(all_gated),
            "n_report_only": len(all_report),
            "internal_gate_eligible": sorted(gated),
            "anchor_gate_eligible": sorted(gated_anchor),
        },
        "draft_thresholds": {
            "k": DRAFT_K,
            "rounding": DRAFT_ROUNDING,
            "t_max": T_MAX,
            "margin_k": MARGIN_K,
            "statistic": "log_ratio_abs_max (internal); ordinal margin (anchor)",
            "cells": thresholds,
            "note": (
                f"DRAFT internal |ln ratio| tolerances = round(mean + "
                f"{DRAFT_K}*sd, {DRAFT_ROUNDING}) capped at {T_MAX_SOURCE}; "
                "anchor cells gate on the ordinal invariant + 100-half "
                "stability + >= margin_k sigma (see anchor_checks)."
            ),
        },
        "protocol": protocol_block(len(gated), len(gated_anchor)),
        "training_copy_check": copy_check,
        "faithful_candidate_oc": oc,
        "concept_deltas": concept_deltas(),
        "wanted_ssa_tables": wanted_ssa_tables(),
        "warts": warts(k_sens, gated, anchor_checks),
        "draft_thresholds_note": draft_thresholds_note(
            gated, report, anchor_checks, k_sens
        ),
        "revision_pins": {
            "base_sha": _rev(ROOT, "HEAD"),
            "origin_master_sha": _rev(ROOT, "origin/master"),
            "merge_base_sha": _merge_base(ROOT),
            "claim_age_reference_schema": claim_ref.schema_version,
            "anchor_tables_sha256": anchor_tables["sha256"],
            "artifact_schema_version": SCHEMA_VERSION,
        },
        "build": {
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "built_by": "scripts/build_m4_gate_floors.py",
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    artifacts.write_new(ARTIFACT_PATH, artifact)
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
