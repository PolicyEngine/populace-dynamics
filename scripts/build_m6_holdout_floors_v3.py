"""gate_m6 floors v3: complete the pinned marital coarsening ladder's rung
enumeration (referee amendment 2 + the orchestrator's v3 adjudication).

v3 is NOT a second design pass. It corrects an INCOMPLETE enumeration in v2's
marital coarsening ladder: v2 pinned a symmetric age-2 rung ([18-44],[45-120])
but never enumerated the ASYMMETRIC age-2 rung ([18-64],[65+]) -- a "merge the
lower bands wide, isolate the elderly tail" shape the MORTALITY ladder itself
already used ([25-54],[55-74],[75-84] and [25-64],[65-84]). The adversarial
referee's independent sweep of every contiguous adjacent-band pooling found that
this one un-enumerated rung is exactly where `remarriage` clears
(remarriage.18-64: tolerance 0.403 <= ln(1.5), 143 weaker-half events, 100/100
seeds, household-disjoint), while it clears NOTHING at any rung v2 enumerated.
So v2's "the holdout is too thin for remarriage even pooled" was true only
relative to the enumerated rungs, not of the data. Completing the ladder to
enumerate the SAME rung shapes the mortality ladder used -- uniformly per
transition type as originally pinned -- is fidelity to the adjudicated criteria.

The completion is a SINGLE machinery change: MARITAL_LADDER gains the asymmetric
age-2 rung ([18-64],[65+]) after the symmetric age-2 rung (ordered by
coarseness: among the equal-band-count 2-band rungs, the symmetric split whose
widest merged group spans 2 raw bands precedes the elderly-isolating split whose
widest merged group spans 3). Everything else -- every invariant, both ladder
rules, the pinned earnings order, the OC machinery -- is byte-for-byte v2. The
minimal-rung rule then adopts, uniformly:
  * first_marriage -> age_x_sex   (unchanged; gates first_marriage.18-29|female)
  * divorce        -> sex_pooled_age2 (unchanged; gates divorce.18-44)
  * remarriage     -> sex_pooled_age2p (NEW; gates remarriage.18-64)
  * widowhood      -> none cleared at ANY rung (best 0.474, fully pooled -- the
                      sweep now shows it evaluated at the asymmetric rung too)
  * death          -> none cleared (mortality ladder untouched)
The extra 5th gated FLOW cell lowers the combined p_gate at every earnings-prune
step, so the SAME pinned weakest-first earnings ladder is FORCED exactly one
more prune (earn_dlog_mean.older, the next non-last-in-concept cell), landing at
11 cells (5 flow + 6 earnings). The outcome is reported by the arithmetic.

1. COARSENING LADDER (flows). Per transition type (each marital transition,
   mortality, disability incidence/recovery), pool ADJACENT strata minimally in
   the pinned order sex-pool -> age-pool-adjacent, climbing until a cell's
   tolerance <= ln(1.5) AND >= 20 events in the weaker half. The minimal rung
   that yields >= 1 clearing cell is adopted uniformly for that transition type;
   the clearing cells gate, the rest are report-only. Mortality 85+ stays
   attrition_confounded_truth report-only regardless (pre-lock demotion).

2. EARNINGS DECOMPOUNDING LADDER. Prune gated earnings cells weakest-power-first
   (largest v1 tolerance/sigma ratio first, a pinned deterministic order from
   the v1 floors), recomputing the COMBINED faithful p_gate after each prune,
   stopping at the FIRST point p_gate >= 0.90, never pruning below >= 1 gated
   cell per concept family (log-quantiles, dispersion, mobility, zero-rate,
   autocorrelation, change-mean). This retains the LARGEST surface meeting the
   power floor under the ladder -- maximum falsifiability subject to power.

Invariants held FIXED (touching any = STOP): T*=2014 (no window extension this
round), lag-5 report-only (horizon), FLOW k=3, ln(1.5) cap, 4-of-5 conjunction,
0.90 p_gate floor. Applied ONCE; the honest outcome is reported either way.

Frozen once written (artifacts.write_new). v1 AND v2 are neither read-mutated
nor rewritten -- v1 is read only for the pinned pruning order + the earnings
floor (identical in v1/v2); v2 is cited as immediate lineage, not read for any
choice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics import artifacts
from populace_dynamics.data import deaths, disability, panels


def _load_v1():
    """Load the frozen v1 builder as a module (shared loaders + floor + OC)."""
    path = Path(__file__).resolve().parent / "build_m6_holdout_floors.py"
    spec = importlib.util.spec_from_file_location(
        "build_m6_holdout_floors", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load_v1()

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m6_holdout_floors_v3.json"
V1_PATH = ROOT / "runs" / "m6_holdout_floors_v1.json"
V1_SHA = "16c28d8cd9095e5233ab224c659c8d5b9eb1621099e2524455a3a8ff8e88d318"
V2_PATH = ROOT / "runs" / "m6_holdout_floors_v2.json"
V2_SHA = "3f273d474692917b01055f85830cb982dfbe9e63070581c99975aa799759b9a0"
SCHEMA_VERSION = "m6_holdout_floors.v3"
RUN = "m6_holdout_floors_v3"

T_MAX = v1.T_MAX
K_FLOW = v1.K_FLOW
MIN_EVENTS = v1.MIN_EVENTS_FOR_GATE
WEAK_POWER_P_GATE_FLOOR = v1.WEAK_POWER_P_GATE_FLOOR
MIN_GATED_FLOW_CELLS = v1.MIN_GATED_CELLS_FOR_POWER
METRIC_CAP = v1.METRIC_CAP

# --------------------------------------------------------------------------
# The pinned coarsening ladder per family: (label, bands, sex_pool). Rung 0 is
# the v1 granularity; each later rung pools ADJACENT strata minimally, sex
# first then age. Mortality omits 85+ (it stays report-only regardless).
#
# v3 completes the MARITAL ladder's rung enumeration: the asymmetric age-2 rung
# (`sex_pooled_age2p`, [18-64],[65+]) is the "merge the lower bands wide, isolate
# the elderly tail" shape the MORTALITY ladder already used (its age3/age2 rungs)
# but the marital ladder had omitted. It is ordered AFTER the symmetric age-2
# rung: both partition into 2 bands, but the symmetric split's widest merged
# group spans 2 raw bands while the elderly-isolating split's spans 3, so the
# symmetric rung is the less coarse (explored first) under the same minimal-rung
# rule. This is enumeration completeness, not a new design choice -- the rung
# shape, the acceptance conditions, and the uniform-per-type application are all
# as originally pinned.
# --------------------------------------------------------------------------
MARITAL_LADDER = [
    ("age_x_sex", [(18, 29), (30, 44), (45, 64), (65, 120)], False),
    ("sex_pooled", [(18, 29), (30, 44), (45, 64), (65, 120)], True),
    ("sex_pooled_age2", [(18, 44), (45, 120)], True),
    ("sex_pooled_age2p", [(18, 64), (65, 120)], True),
    ("sex_pooled_age1", [(18, 120)], True),
]
MORTALITY_LADDER = [
    (
        "age_x_sex",
        [(25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 84)],
        False,
    ),
    (
        "sex_pooled",
        [(25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 84)],
        True,
    ),
    ("sex_pooled_age3", [(25, 54), (55, 74), (75, 84)], True),
    ("sex_pooled_age2", [(25, 64), (65, 84)], True),
    ("sex_pooled_age1", [(25, 84)], True),
]
DISABILITY_LADDER = [
    ("age_x_sex", [(20, 29), (30, 39), (40, 49), (50, 59), (60, 66)], False),
    ("sex_pooled", [(20, 29), (30, 39), (40, 49), (50, 59), (60, 66)], True),
    ("sex_pooled_age3", [(20, 39), (40, 59), (60, 66)], True),
    ("sex_pooled_age2", [(20, 49), (50, 66)], True),
    ("sex_pooled_age1", [(20, 66)], True),
]
MARITAL_TRANSITIONS = ("first_marriage", "divorce", "widowhood", "remarriage")

CONCEPT_FAMILY = {
    "earn_p10": "log_quantiles",
    "earn_p50": "log_quantiles",
    "earn_p90": "log_quantiles",
    "earn_dlog_sd": "dispersion",
    "earn_mob_h1_diag": "mobility",
    "earn_mob_h2_diag": "mobility",
    "earn_zero_rate": "zero_rate",
    "earn_autocorr_lag1": "autocorrelation",
    "earn_autocorr_lag2": "autocorrelation",
    "earn_dlog_mean": "change_mean",
}


def _concept(cell: str) -> str:
    head = cell.split(".")[0]
    return CONCEPT_FAMILY[head]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratum(frame: pd.DataFrame, bands, sex_pool: bool) -> pd.DataFrame:
    f = frame.copy()
    f["band2"] = f["age"].map(lambda a: v1.band_of(a, tuple(bands)))
    f = f[f["band2"].notna()]
    if sex_pool:
        f["stratum"] = f["band2"]
    else:
        f["stratum"] = f["band2"].astype(str) + "|" + f["sex"].astype(str)
    return f


# --------------------------------------------------------------------------
# Re-banded flow cells for one floor half.
# --------------------------------------------------------------------------
def mort_cells(sl: pd.DataFrame, bands, sex_pool: bool) -> dict[str, dict]:
    f = _stratum(sl, bands, sex_pool)
    g = (
        f.assign(we=f.weight * f.exposure, wd=f.weight * f.death)
        .groupby("stratum", observed=True)
        .agg(
            we=("we", "sum"),
            wd=("wd", "sum"),
            nd=("death", "sum"),
            na=("death", "size"),
        )
    )
    return {
        f"death.{s}": {
            "rate": v1._rate(r["wd"], r["we"]),
            "n_events": int(round(r["nd"])),
            "n_at_risk": int(r["na"]),
        }
        for s, r in g.iterrows()
    }


def marital_cells(
    ev: pd.DataFrame, py: pd.DataFrame, trans: str, bands, sex_pool: bool
) -> dict[str, dict]:
    states = v1.MARITAL_AT_RISK[trans]
    te = _stratum(ev[ev.transition == trans], bands, sex_pool)
    atr = _stratum(py[py.marital_state.isin(states)], bands, sex_pool)
    num = te.groupby("stratum", observed=True).agg(
        numw=("weight", "sum"), ne=("weight", "size")
    )
    den = atr.groupby("stratum", observed=True).agg(
        denw=("weight", "sum"), na=("weight", "size")
    )
    joined = num.join(den, how="outer")
    out: dict[str, dict] = {}
    for s, r in joined.iterrows():
        numw = 0.0 if pd.isna(r["numw"]) else float(r["numw"])
        denw = 0.0 if pd.isna(r["denw"]) else float(r["denw"])
        n_ev = 0 if pd.isna(r["ne"]) else int(r["ne"])
        n_at = 0 if pd.isna(r["na"]) else int(r["na"])
        out[f"{trans}.{s}"] = {
            "rate": v1._rate(numw, denw),
            "n_events": n_ev,
            "n_at_risk": n_at,
        }
    return out


def disab_cells(
    pairs: pd.DataFrame, kind: str, bands, sex_pool: bool
) -> dict[str, dict]:
    if kind == "incidence":
        sub = pairs[~pairs.from_disabled].copy()
        sub["_ev"] = sub.to_disabled.astype(float)
    else:
        sub = pairs[pairs.from_disabled].copy()
        sub["_ev"] = (~sub.to_disabled).astype(float)
    f = _stratum(sub, bands, sex_pool)
    g = (
        f.assign(w=f.weight, wev=f.weight * f._ev)
        .groupby("stratum", observed=True)
        .agg(
            denw=("w", "sum"),
            numw=("wev", "sum"),
            na=("w", "size"),
            ne=("_ev", "sum"),
        )
    )
    return {
        f"{kind}.{s}": {
            "rate": v1._rate(r["numw"], r["denw"]),
            "n_events": int(round(r["ne"])),
            "n_at_risk": int(r["na"]),
        }
        for s, r in g.iterrows()
    }


def _cleared(floor: dict, key: str) -> tuple[bool, float]:
    fl = floor[key]
    tol = v1._tol(fl["mean"], fl["sd"], K_FLOW)
    ok = (
        fl["n_defined_seeds"] == len(v1.FLOOR_SEEDS)
        and fl["min_events_weaker_half"] >= MIN_EVENTS
        and tol <= T_MAX + 1e-12
    )
    return ok, tol


def climb_ladder(
    anchor, compute_factory, ladder, transition: str
) -> dict[str, Any]:
    """Climb the pinned ladder for one transition type; adopt the MINIMAL rung
    that yields >= 1 clearing cell. compute_factory(bands, sex_pool) -> a
    compute(ids) closure; the split unit follows the family."""
    split_col = v1.split_col_for_family(
        "marital"
        if transition in MARITAL_TRANSITIONS
        else ("mortality" if transition == "death" else "disability")
    )
    steps: list[dict[str, Any]] = []
    adopted = None
    for label, bands, sex_pool in ladder:
        compute = compute_factory(bands, sex_pool)
        floor, per_seed = v1.run_floor(anchor, compute, split_col)
        ref = compute(set(anchor.person_id.to_numpy().tolist()))
        cleared = []
        cell_rows = {}
        for key in sorted(floor):
            ok, tol = _cleared(floor, key)
            cell_rows[key] = {
                "tolerance": tol,
                "realized_sigma": floor[key]["realized_sigma"],
                "min_events_weaker_half": floor[key]["min_events_weaker_half"],
                "n_events_full": ref.get(key, {}).get("n_events", 0),
                "clears": bool(ok),
            }
            if ok:
                cleared.append(key)
        steps.append(
            {
                "rung": label,
                "bands": [list(b) for b in bands],
                "sex_pool": sex_pool,
                "cells": cell_rows,
                "n_cleared": len(cleared),
            }
        )
        if cleared and adopted is None:
            adopted = {
                "rung": label,
                "bands": [list(b) for b in bands],
                "sex_pool": sex_pool,
                "floor": floor,
                "per_seed": per_seed,
                "gated": sorted(cleared),
                "report_only": sorted(k for k in floor if k not in cleared),
                "tolerances": {
                    k: v1._tol(floor[k]["mean"], floor[k]["sd"], K_FLOW)
                    for k in floor
                },
            }
            break
    return {"transition": transition, "steps": steps, "adopted": adopted}


def build_v3(verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    demo = panels.demographic_panel()
    death_records = deaths.read_death_records()
    status = disability.read_disability_status()
    # earnings floor is read from v1 (frozen); no raw earnings load needed here.

    anchor = v1.build_anchor_frame(demo)
    present = v1.presence_by_wave(demo)

    sl = v1.mortality_slices(demo, death_records, anchor)
    ev, py = v1.marital_tables(death_records, anchor, present)
    dpairs = v1.disability_pairs(status, death_records, anchor)

    sl_g = sl[sl.window == "gated"]
    ev_g = ev[ev.window == "gated"]
    py_g = py[py.window == "gated"]
    dp_g = dpairs[dpairs.window == "gated"]

    # ---- Ladder 1: coarsen each flow transition type ----
    ladders: dict[str, Any] = {}

    def mort_factory(bands, sex_pool):
        # 85+ stays report-only: it is never in the pooled ladder bands.
        return lambda ids: mort_cells(
            sl_g[sl_g.person_id.isin(ids)], bands, sex_pool
        )

    if verbose:
        print("ladder: mortality...")
    ladders["death"] = climb_ladder(
        anchor, mort_factory, MORTALITY_LADDER, "death"
    )
    for trans in MARITAL_TRANSITIONS:
        if verbose:
            print(f"ladder: {trans}...")

        def mar_factory(bands, sex_pool, _t=trans):
            return lambda ids: marital_cells(
                ev_g[ev_g.person_id.isin(ids)],
                py_g[py_g.person_id.isin(ids)],
                _t,
                bands,
                sex_pool,
            )

        ladders[trans] = climb_ladder(
            anchor, mar_factory, MARITAL_LADDER, trans
        )
    for kind in ("incidence", "recovery"):
        if verbose:
            print(f"ladder: {kind}...")

        def dis_factory(bands, sex_pool, _k=kind):
            return lambda ids: disab_cells(
                dp_g[dp_g.person_id.isin(ids)], _k, bands, sex_pool
            )

        ladders[kind] = climb_ladder(
            anchor, dis_factory, DISABILITY_LADDER, kind
        )

    # assemble the coarsened flow gated surface
    flow_floor: dict[str, dict] = {}
    flow_tol: dict[str, float] = {}
    flow_gated: list[str] = []
    flow_report: dict[str, str] = {}
    flow_family: dict[str, str] = {}
    adopted_configs: dict[str, dict] = (
        {}
    )  # transition -> config for diagnostic
    for trans, rec in ladders.items():
        fam = (
            "marital"
            if trans in MARITAL_TRANSITIONS
            else ("mortality" if trans == "death" else "disability")
        )
        adopted = rec["adopted"]
        if adopted is None:
            # nothing cleared at any rung; the finest rung's cells are report-
            # only on power (record the last step's floor for provenance)
            last = rec["steps"][-1]
            for key in last["cells"]:
                flow_report[key] = "tolerance_above_power_cap_all_rungs"
                flow_family[key] = fam
            continue
        for key in adopted["floor"]:
            flow_floor[key] = adopted["floor"][key]
            flow_tol[key] = adopted["tolerances"][key]
            flow_family[key] = fam
        flow_gated.extend(adopted["gated"])
        for key in adopted["report_only"]:
            flow_report[key] = "below_cap_or_events_at_adopted_rung"
        adopted_configs[trans] = {
            "family": fam,
            "bands": adopted["bands"],
            "sex_pool": adopted["sex_pool"],
            "gated": adopted["gated"],
        }
    # mortality 85+ stays report-only (attrition), recorded explicitly
    flow_report["death.85+|female"] = "attrition_confounded_truth"
    flow_report["death.85+|male"] = "attrition_confounded_truth"
    flow_family["death.85+|female"] = "mortality"
    flow_family["death.85+|male"] = "mortality"

    # ---- Ladder 2: earnings decompounding from the v1 floors ----
    v1_art = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v1_floor = v1_art["floor"]["cells"]
    v1_tol = v1_art["tolerances"]
    v1_earn_gated = [
        k
        for k in v1_art["partition"]["gated"]
        if v1_art["cell_meta"][k]["family"] == "earnings"
    ]

    # pinned order: largest tolerance/sigma (weakest power) first
    def _ratio(k: str) -> float:
        return v1_tol[k] / v1_floor[k]["realized_sigma"]

    pinned_order = sorted(v1_earn_gated, key=lambda k: (-_ratio(k), k))

    def combined_p_gate(earn_cells: list[str]) -> tuple[float, float]:
        cells = flow_gated + earn_cells
        floor_all = {**flow_floor, **{k: v1_floor[k] for k in earn_cells}}
        tol_all = {**flow_tol, **{k: v1_tol[k] for k in earn_cells}}
        oc = v1.oc_4of5(floor_all, tol_all, cells)
        return oc["p_seed_pass"], oc["p_gate_pass_4_of_5"]

    retained = list(v1_earn_gated)
    prune_log: list[dict[str, Any]] = []
    _, p_gate = combined_p_gate(retained)
    for cell in pinned_order:
        if p_gate >= WEAK_POWER_P_GATE_FLOOR:
            break
        concept = _concept(cell)
        n_in_concept = sum(1 for c in retained if _concept(c) == concept)
        if n_in_concept <= 1:
            prune_log.append(
                {
                    "cell": cell,
                    "tol_over_sigma": round(_ratio(cell), 4),
                    "action": "kept_last_in_concept",
                    "concept": concept,
                }
            )
            continue
        retained.remove(cell)
        _, p_gate = combined_p_gate(retained)
        prune_log.append(
            {
                "cell": cell,
                "tol_over_sigma": round(_ratio(cell), 4),
                "action": "pruned",
                "concept": concept,
                "combined_p_gate_after": p_gate,
            }
        )

    earn_gated = sorted(retained)
    earn_pruned = sorted(set(v1_earn_gated) - set(retained))

    # ---- combined surface, OC, weak-power check ----
    gated = sorted(flow_gated + earn_gated)
    floor_all = {**flow_floor, **{k: v1_floor[k] for k in earn_gated}}
    tol_all = {**flow_tol, **{k: v1_tol[k] for k in earn_gated}}
    meta = {k: {"family": flow_family[k]} for k in flow_floor}
    for k in earn_gated:
        meta[k] = {"family": "earnings"}

    flow_gated_sorted = sorted(flow_gated)
    oc_combined = v1.oc_4of5(floor_all, tol_all, gated)
    oc_flows = v1.oc_4of5(flow_floor, flow_tol, flow_gated_sorted)
    oc_earn = v1.oc_4of5(
        {k: v1_floor[k] for k in earn_gated},
        {k: v1_tol[k] for k in earn_gated},
        earn_gated,
    )

    n_flow_gated = len(flow_gated)
    at_cap = [
        k
        for k in gated
        if abs(tol_all[k] - T_MAX) <= 1e-9
        and meta[k]["family"] in ("mortality", "marital", "disability")
    ]
    clears_lower = oc_combined["p_gate_pass_4_of_5"] >= WEAK_POWER_P_GATE_FLOOR
    clears_flow_power = n_flow_gated >= MIN_GATED_FLOW_CELLS
    # v3 restores v1's THIRD vacuity guard, which v2 dropped from the
    # conjunction: not every gated tolerance is pinned at its cap. n_at_cap = 0
    # here, so it has no effect on the verdict -- but the clears-conjunction is
    # registry-faithful to v1's three-guard structure again.
    clears_not_all_capped = len(at_cap) < len(gated)
    clears = bool(clears_lower and clears_flow_power and clears_not_all_capped)

    if verbose:
        print(
            f"v3 gated={len(gated)} (flow={n_flow_gated} "
            f"earn={len(earn_gated)}) p_gate_combined="
            f"{oc_combined['p_gate_pass_4_of_5']} clears={clears}"
        )

    # split-width diagnostic: person- vs household-disjoint floor widths on the
    # adopted coarsened household-split (marital + mortality) flow cells -- the
    # ceremony watch-item re-checked at the redesigned granularity.
    diag = _split_diag(anchor, adopted_configs, flow_floor, sl_g, ev_g, py_g)

    art: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "reported_truth_side_only": True,
        "no_projection_no_candidate": True,
        "candidate_blind": (
            "Every choice -- each pooling rung (including the completed "
            "asymmetric age-2 marital rung) and each earnings prune -- is "
            "determined ONLY by truth-side power arithmetic (floor sigmas, "
            "weaker-half event counts, and v1 tolerance/sigma ratios). Nothing "
            "here references a projection engine, a candidate, or what any "
            "model would find easy or hard. The one vocabulary freedom the "
            "referee flagged (the un-enumerated rung) SHRANK the gated surface "
            "by forcing a further earnings prune -- the opposite direction from "
            "candidate-favoring."
        ),
        "supersedes_note": (
            "v3 supersedes v2 by COMPLETING the marital coarsening ladder's "
            "rung enumeration (referee amendment 2 + the orchestrator's v3 "
            "adjudication): v2's marital ladder enumerated a symmetric age-2 "
            "rung but not the asymmetric age-2 rung ([18-64],[65+]) that the "
            "mortality ladder's own merge vocabulary already used -- the single "
            "rung where remarriage clears. v1 (the frozen PAUSE evidence) and "
            "v2 (the frozen prior surface) are BOTH neither mutated nor "
            "rewritten; v1 is read only for the pinned earnings pruning order "
            "and the earnings floor (identical in v1/v2), and v2 is cited as "
            "immediate lineage, read for no choice."
        ),
        "v1_pause_evidence": {
            "run": "m6_holdout_floors_v1",
            "sha256": V1_SHA,
            "committed_sha256": _sha256(V1_PATH),
            "v1_combined_p_gate": v1_art["oc_before_lock"]["p_gate_combined"],
            "v1_n_gated_flow_cells": v1_art["oc_before_lock"][
                "n_gated_flow_cells"
            ],
        },
        "v2_lineage": {
            "run": "m6_holdout_floors_v2",
            "sha256": V2_SHA,
            "committed_sha256": _sha256(V2_PATH),
            "v2_n_gated": 11,
            "v2_n_gated_flow_cells": 4,
            "v2_combined_p_gate": 0.9067,
            "note": (
                "v2 cleared the weak-power threshold with an INCOMPLETE marital "
                "ladder enumeration (symmetric age-2 only). v3 completes the "
                "enumeration; v2 is frozen as lineage evidence and is not read "
                "for any v3 choice."
            ),
        },
        "invariants_held_fixed": {
            "boundary_T_star": 2014,
            "no_window_extension_this_round": True,
            "lag5_report_only": True,
            "flow_k": K_FLOW,
            "t_max": T_MAX,
            "t_max_source": "ln(1.5)",
            "conjunction": "4 of 5 seeds",
            "weak_power_p_gate_floor": WEAK_POWER_P_GATE_FLOOR,
        },
        "coarsening_ladder": {
            "rule": (
                "per transition type, pool ADJACENT strata minimally in the "
                "pinned order sex-pool -> age-pool-adjacent; adopt the MINIMAL "
                "rung with >= 1 cell clearing tolerance <= ln(1.5) AND >= 20 "
                "weaker-half events; mortality 85+ stays report-only "
                "(attrition_confounded_truth) at every rung."
            ),
            "marital_ladder_completion": {
                "change": (
                    "v3 adds the asymmetric age-2 marital rung sex_pooled_age2p "
                    "([18-64],[65+]) after the symmetric age-2 rung -- the "
                    "elderly-isolating merge shape the mortality ladder already "
                    "used, uniformly per marital transition type as originally "
                    "pinned."
                ),
                "marital_rungs_enumerated": [
                    label for label, _bands, _sp in MARITAL_LADDER
                ],
                "sweep_dispositions": {
                    "first_marriage": (
                        "adopts age_x_sex (unchanged); clears at rung 0"
                    ),
                    "divorce": (
                        "adopts sex_pooled_age2 (unchanged); the new "
                        "asymmetric rung is later in the ladder and never "
                        "reached"
                    ),
                    "remarriage": (
                        "adopts sex_pooled_age2p (NEW); clears ONLY at the "
                        "asymmetric age-2 rung -- nothing clears at any rung v2 "
                        "enumerated"
                    ),
                    "widowhood": (
                        "clears NOTHING at any rung, including the new "
                        "asymmetric rung -- best tolerance ~0.474 (fully "
                        "pooled), robustly above the ln(1.5) cap"
                    ),
                },
            },
            "ladders": {t: _ladder_summary(r) for t, r in ladders.items()},
        },
        "earnings_decompounding_ladder": {
            "rule": (
                "prune gated earnings cells weakest-power-first (largest v1 "
                "tolerance/sigma first), recomputing combined p_gate after each "
                "prune, stopping at the FIRST p_gate >= 0.90, never below >= 1 "
                "gated cell per concept family -- the LARGEST surface meeting "
                "the power floor under the ladder (maximum falsifiability "
                "subject to power)."
            ),
            "pinned_order": [
                {
                    "cell": k,
                    "tol_over_sigma": round(_ratio(k), 4),
                    "concept": _concept(k),
                }
                for k in pinned_order
            ],
            "prune_log": prune_log,
            "concept_families": sorted(set(CONCEPT_FAMILY.values())),
            "retained": earn_gated,
            "pruned": earn_pruned,
            "retained_by_concept": {
                c: sorted(k for k in earn_gated if _concept(k) == c)
                for c in sorted(set(_concept(k) for k in earn_gated))
            },
        },
        "presence_conditioning": v1_art["presence_conditioning"],
        "f6_weight_rule": v1_art["f6_weight_rule"],
        "household_id_weight_carriage_rule": v1_art[
            "household_id_weight_carriage_rule"
        ],
        "split_units": v1_art["split_units"],
        "design_pins": v1_art["design_pins"],
        "data": v1_art["data"],
        "floor": {
            "method": v1_art["floor"]["method"],
            "cells": floor_all,
        },
        "metric_caps": v1_art["metric_caps"],
        "cell_meta": meta,
        "partition": {
            "gated": gated,
            "report_only": flow_report,
            "n_gated": len(gated),
            "n_gated_flow_cells": n_flow_gated,
            "n_gated_earnings_cells": len(earn_gated),
            "by_family_gated": {
                fam: sorted(k for k in gated if meta[k]["family"] == fam)
                for fam in sorted({meta[k]["family"] for k in gated})
            },
        },
        "tolerances": {k: tol_all[k] for k in gated},
        "faithful_candidate_oc": {
            "combined": oc_combined,
            "family_a_flows": oc_flows,
            "earnings_subfamily": oc_earn,
            "p_seed_pass": oc_combined["p_seed_pass"],
            "p_gate_pass_4_of_5": oc_combined["p_gate_pass_4_of_5"],
        },
        "oc_before_lock": {
            "weak_power_p_gate_floor": WEAK_POWER_P_GATE_FLOOR,
            "min_gated_flow_cells": MIN_GATED_FLOW_CELLS,
            "n_gated_cells": len(gated),
            "n_gated_flow_cells": n_flow_gated,
            "n_gated_earnings_cells": len(earn_gated),
            "n_gated_tolerances_at_cap": len(at_cap),
            "p_gate_combined": oc_combined["p_gate_pass_4_of_5"],
            "p_gate_flows": oc_flows["p_gate_pass_4_of_5"],
            "p_gate_earnings": oc_earn["p_gate_pass_4_of_5"],
            "clears_lower_bound_not_unpassable": bool(clears_lower),
            "clears_flow_surface_power": bool(clears_flow_power),
            "clears_not_all_tolerances_capped": bool(clears_not_all_capped),
            "clears_weak_power_threshold": clears,
            "ceremony_may_proceed": clears,
            "outcome": (
                "surface clears; ceremony may proceed to lock"
                if clears
                else "surface does NOT clear; window extension is a separate "
                "adjudication (re-derives design decisions 1-2)"
            ),
        },
        "split_width_diagnostic": diag,
        "revision_pins": {"artifact_schema_version": SCHEMA_VERSION},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return art


def _ladder_summary(rec: dict[str, Any]) -> dict[str, Any]:
    adopted = rec["adopted"]
    return {
        "steps": [
            {
                "rung": s["rung"],
                "sex_pool": s["sex_pool"],
                "bands": s["bands"],
                "n_cleared": s["n_cleared"],
                "cells": {
                    k: {
                        "tolerance": v["tolerance"],
                        "min_events_weaker_half": v["min_events_weaker_half"],
                        "n_events_full": v["n_events_full"],
                        "clears": v["clears"],
                    }
                    for k, v in s["cells"].items()
                },
            }
            for s in rec["steps"]
        ],
        "adopted_rung": None if adopted is None else adopted["rung"],
        "gated": [] if adopted is None else adopted["gated"],
    }


def _split_diag(
    anchor, adopted_configs, flow_floor, sl_g, ev_g, py_g
) -> dict[str, Any]:
    """Person- vs household-disjoint width on the adopted household-split flow
    cells: re-run the SAME rung under a person-disjoint split and compare the
    realized sigma to the household-disjoint one used for the gate. A ratio ~1
    confirms the T*-anchored household-ID rule does not bias the redesigned
    family-A floor."""
    per_cell: dict[str, Any] = {}
    ratios: list[float] = []
    for trans, cfg in adopted_configs.items():
        if cfg["family"] not in ("marital", "mortality"):
            continue
        bands, sex_pool = cfg["bands"], cfg["sex_pool"]
        if cfg["family"] == "mortality":

            def compute(ids, _b=bands, _s=sex_pool):
                return mort_cells(sl_g[sl_g.person_id.isin(ids)], _b, _s)

        else:

            def compute(ids, _t=trans, _b=bands, _s=sex_pool):
                return marital_cells(
                    ev_g[ev_g.person_id.isin(ids)],
                    py_g[py_g.person_id.isin(ids)],
                    _t,
                    _b,
                    _s,
                )

        person_floor, _ = v1.run_floor(anchor, compute, "person_id")
        for key in cfg["gated"]:
            hh = flow_floor[key]["realized_sigma"]
            pp = person_floor.get(key, {}).get("realized_sigma", 0.0)
            ratio = float(hh / pp) if pp > 0 else None
            per_cell[key] = {
                "household_sigma": round(hh, 6),
                "person_sigma": round(pp, 6),
                "household_over_person": (
                    None if ratio is None else round(ratio, 4)
                ),
            }
            if ratio is not None:
                ratios.append(ratio)
    arr = np.array(ratios) if ratios else np.array([1.0])
    return {
        "note": (
            "person- vs household-disjoint realized_sigma on the adopted "
            "coarsened household-split (marital + mortality) flow cells; ~1 "
            "confirms the T*-anchored household-ID rule does not bias the "
            "redesigned family-A floor (the sec. 9 finding-13 watch-item)."
        ),
        "n_household_gated_flow_cells": len(per_cell),
        "mean_household_over_person": round(float(arr.mean()), 4),
        "median_household_over_person": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "per_cell": per_cell,
    }


def main() -> None:
    art = build_v3(verbose=True)
    artifacts.write_new(ARTIFACT_PATH, art, sidecar=True)
    print(f"wrote {ARTIFACT_PATH}")
    print(f"sha256 {_sha256(ARTIFACT_PATH)}")


if __name__ == "__main__":
    main()
