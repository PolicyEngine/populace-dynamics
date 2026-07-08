"""Build the gate-2 family-transition floors: moments, floors, anchors.

PRE-LOCK EVIDENCE, NOT A GATE RUN. The analogue of what preceded gate 1's
lock, one gate down: the committed reference moments, the person-disjoint
half-split noise floor (the sd basis the DRAFT gate-2 thresholds derive
from), and the honest external anchors for the family-transition
statistics a survivor/spousal/caregiver reform is scored on. It reads no
gate and, on its own, changes nothing; the DRAFT thresholds it feeds live
in ``gates.yaml`` under ``gate_2`` with ``locked: false`` and
``status: draft_pending_referee_round``. The lock ceremony (floors ->
thresholds with machine-bound derivations -> adversarial referee round ->
verification -> maintainer ratification) comes AFTER, exactly as gate 1's
did.

Round-1 referee amendments (this v2 build, PR #79 comment 4910467957):

* the floor is measured on **100 person-disjoint split seeds** (not 5), so
  each per-cell tolerance ``round(mean + k*sd, 3)`` is a STABLE ~3.2-sigma
  bound rather than a 5-draw estimator that realises anywhere from ~0.9 to
  ~5.6 sigma. Each cell records its realised noise scale (finding 2);
* the ONE coherent scoring protocol is option (a) of finding 1 -- per-seed
  refit on the train complement, simulate the seed's holdout persons, score
  ``|ln(r_cand / r_holdout)|`` against the holdout half's own empirical
  rate, for which the half-vs-half floor is exactly the null;
* a pre-stated power cap ``T_max = ln(1.5)`` demotes cells whose stabilised
  tolerance is too loose to reject a bad model, with coverage recovered by
  pre-registered aggregate bands (finding 3);
* joint/sequence statistics -- origin-split remarriage, cohort ever-married
  and dissolved-state stock shares -- bind the structure a banded marginal
  gate misses (finding 4);
* the external anchors are concept-decomposed (marriage/divorce person-vs-
  couple x denominator) and period-matched (annual NCHS ASFR), finding 6.

Six marital/fertility statistic families, all from the retrospective PSID
history files via :mod:`populace_dynamics.data.transitions`:

1. first-marriage hazard by age band x sex (+ the 35+ aggregate),
2. divorce hazard by marriage-duration band,
3. widowhood incidence by age band x sex (+ 45+ male / 45-64 female
   aggregates),
4. remarriage hazard by years-since-dissolution, and by origin state (after
   divorce / after widowhood, widow(er) under/over 60),
5. occupancy: share ever married by 40 / 60 x sex, mean lifetime marriages,
   cohort ever-married-by-40, dissolved-state stock shares by age x sex,
6. fertility: age-specific birth rates + completed fertility by cohort.

Run from the repository root with the PSID history files staged::

    .venv/bin/python scripts/build_gate2_floors.py

It needs no populace-fit (real-vs-real / real-vs-external only).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics.data import (
    births,
    deaths,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_floors_v2.json"
ASFR_PATH = ROOT / "data" / "external" / "nchs_asfr_2024.json"
MD_PATH = ROOT / "data" / "external" / "nchs_marriage_divorce_rates_2023.json"
ARTIFACT_SCHEMA_VERSION = "gate2_floors.v2"

#: Floor seeds: 100 person-disjoint splits stabilise the tolerance
#: estimator (round-1 finding 2). ``round(mean + k*sd)`` over 5 draws of a
#: half-normal realised 0.9-5.6 sigma; over 100 it is a stable ~3.2 sigma.
SEEDS = tuple(range(100))
#: The pinned gate-run seeds: a candidate is scored on these five (the
#: per-seed holdout reference rates live in ``noise_floor_per_seed``), with
#: the 4-of-5 conjunction, exactly as gate 1 gates on five seeds while its
#: floor is measured at deployment scale.
GATE_SEEDS = (0, 1, 2, 3, 4)
#: Minimum events (weaker half, worst of the 100 seeds) for a cell to be
#: gate-eligible. 20 is NCHS's own reliability floor.
MIN_EVENTS_FOR_GATE = 20
#: DRAFT threshold multiplier: |ln ratio| tolerance = round(mean + k*sd).
#: ~4 sigma on the STABILISED floor, the gate-1 c2st precedent (k=4.2) +
#: the 4-of-5-seed rule. DRAFT -- the referee round sets the final k.
DRAFT_K = 4
DRAFT_ROUNDING = 3
#: Power cap (round-1 finding 3): a cell is gate-eligible only if its
#: stabilised tolerance is tight enough to reject a materially wrong model.
#: T_max = ln(1.5) -- a gated cell accepts at most a 1.5x rate error. Fixed
#: a priori, BEFORE the stabilised tolerances are seen; which cells the cap
#: demotes is then derived from the floor.
T_MAX = math.log(1.5)
T_MAX_SOURCE = "ln(1.5)"
#: Calendar-year floor for the "recent" external-anchor window.
RECENT_START_YEAR = 2010

#: Pre-registered coverage-recovery aggregates (finding 3): when a per-age
#: hazard cell fails the power cap, the maintainer's a-priori aggregate
#: band pools the demoted cells' exposure into one gate-eligible cell.
#: Every per-age member an aggregate GATES is demoted to report-only
#: (superseded); the bands are fixed before the numbers are seen.
AGGREGATIONS: dict[str, tuple[str, ...]] = {
    "widowhood.45+|male": (
        "widowhood.45-54|male",
        "widowhood.55-64|male",
        "widowhood.65-74|male",
        "widowhood.75+|male",
    ),
    "widowhood.45-64|female": (
        "widowhood.45-54|female",
        "widowhood.55-64|female",
    ),
    "first_marriage.35+|female": (
        "first_marriage.35-44|female",
        "first_marriage.45+|female",
    ),
    "first_marriage.35+|male": (
        "first_marriage.35-44|male",
        "first_marriage.45+|male",
    ),
}

#: Marriage/divorce anchor concept decomposition (finding 6a). PSID counts
#: PERSONS transitioning (two spouses per event); NCHS counts COUPLES -- an
#: exact factor of 2. The PSID crude-equivalent denominator is person-years
#: age 15+, ~82% of the population (US Census), so it runs 1/0.82 ~ 1.22x
#: hotter than the NCHS per-total-population rate. Concept factor ~ 2.44;
#: the residual after dividing it out is the real PSID/NCHS agreement.
PERSON_TO_COUPLE_FACTOR = 2.0
POP_15PLUS_SHARE = 0.82  # US Census: ~82% of the US population is aged 15+.


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------
def load_panels() -> (
    tuple[transitions.MaritalPanel, transitions.FertilityPanel, dict[str, Any]]
):
    """Load the PSID history files and build the marital + fertility panels."""
    mh = marriage.marriage_history()
    dr = deaths.read_death_records()
    bh = births.birth_history()
    demo = panels.demographic_panel()
    demo_pos = demo[demo.weight > 0]
    person_weight = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    n_marriage_persons = int(mh.person_id.nunique())
    panel = transitions.build_marital_panel(mh, dr, person_weight)
    fert = transitions.build_fertility_panel(panel, bh)
    n_kept = int(panel.attrs.person_id.nunique())
    data_meta = {
        "marriage_history_persons": n_marriage_persons,
        "panel_persons_weighted": n_kept,
        "panel_retained_share": round(n_kept / n_marriage_persons, 4),
        "n_person_years": int(len(panel.person_years)),
        "person_year_calendar_range": [
            int(panel.person_years.year.min()),
            int(panel.person_years.year.max()),
        ],
        "n_transition_events": int(len(panel.events)),
        "transition_event_counts": {
            k: int(v)
            for k, v in panel.events.transition.value_counts().items()
        },
        "n_woman_years_15_49": int(len(fert.woman_years)),
        "n_maternal_births": int(len(fert.births)),
        "n_completed_fertility_women": int(len(fert.completed)),
        "death_records_exact_year": int((dr.death_status == "exact").sum()),
    }
    return panel, fert, data_meta


# --------------------------------------------------------------------------
# Internal noise floor (person-disjoint 50/50 half-split, 100 seeds)
# --------------------------------------------------------------------------
def measure_seed_halfsplit(
    seed: int,
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
) -> dict[str, Any]:
    """One seed: split persons 50/50, all reference-moment cells per half.

    Returns per cell the two halves' rates, the minimum event count, the
    absolute log rate ratio ``|ln(r_A / r_B)|`` and the absolute percent
    gap -- undefined (``None``) when either half's rate is zero
    (denominator-fragile). Side A is the drawn (holdout) half, side B the
    train complement; under the gate protocol a candidate refit on side B
    is scored against side A's empirical rate, for which the symmetric
    half-vs-half ``|ln(r_A/r_B)|`` is exactly the null.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(side_a.person_id)
    ids_b = set(side_b.person_id)
    cells_a = transitions.reference_moments(panel, fert, ids_a, weighted=True)
    cells_b = transitions.reference_moments(panel, fert, ids_b, weighted=True)

    cells: dict[str, Any] = {}
    for key in cells_a:
        ra = cells_a[key]["rate"]
        rb = cells_b[key]["rate"]
        na = cells_a[key]["n_events"]
        nb = cells_b[key]["n_events"]
        defined = ra > 0 and rb > 0
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
        "n_persons_side_a": int(side_a.person_id.nunique()),
        "n_persons_side_b": int(side_b.person_id.nunique()),
        "cells": cells,
    }


def _floor_summary(log_ratios: list[float], pct_diffs: list[float]) -> dict:
    """The pooled floor block for a cell defined on every seed.

    ``values`` are the per-seed ``|log rate ratio|`` (the null draws the
    tolerance derives from); mean/sd recompute from them. ``realized_sigma``
    is the root-mean-square of ``values`` -- the noise scale tau of the
    half-vs-half comparison (= the sd of the SIGNED log ratio, since the
    person split is exchangeable and mean-zero). The stabilised tolerance
    then realises ``tolerance / realized_sigma`` sigma of measured noise,
    the round-1 finding-2 diagnostic.
    """
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
            "values": [float(v) for v in pct],
        },
    }


def pool_internal_floor(
    per_seed: list[dict[str, Any]], cell_keys: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Across-seed floor per cell + the raw stability data (pre power-cap).

    ``noise_floor[key]`` follows the committed-floor convention (mean/sd of
    ``|log rate ratio|`` over the 100 seeds, plus the realised-sigma
    diagnostic and the nested percent-gap block). Only cells defined on
    every seed carry a floor block. ``stability[key]`` records the defined-
    seed count and the minimum event count -- the events half of the
    gate-eligibility rule; the power-cap half is applied in
    :func:`partition_cells` once the tolerances exist.
    """
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
    """The round(mean + k*sd, rounding) tolerance for every floored cell."""
    return {
        key: round(block["mean"] + DRAFT_K * block["sd"], DRAFT_ROUNDING)
        for key, block in noise_floor.items()
    }


def _events_ok(stab: dict[str, Any]) -> bool:
    return (
        stab["defined_seeds"] == stab["n_seeds"]
        and stab["min_events_either_half"] >= MIN_EVENTS_FOR_GATE
    )


def _passes(key: str, stability: dict, tolerances: dict) -> bool:
    """Gate-eligible on its own merits: defined + >=20 events + tol<=T_max."""
    return (
        key in tolerances
        and _events_ok(stability[key])
        and tolerances[key] <= T_MAX
    )


def _demote_reason(key: str, stability: dict, tolerances: dict) -> str:
    stab = stability[key]
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
    """The gate-eligible / report-only partition (round-1 finding 3).

    A cell gates iff it is defined on every seed, carries >=20 events on the
    weaker half of the worst seed, AND its stabilised tolerance <= T_max,
    AND it is not superseded by a gating aggregate. Where per-age cells fail
    the cap, the pre-registered :data:`AGGREGATIONS` recover the coverage;
    every per-age member of a GATING aggregate becomes report-only. Purely
    derived from the floor -- no cell is hand-picked.
    """
    member_of: dict[str, str] = {}
    for agg, members in AGGREGATIONS.items():
        for m in members:
            member_of[m] = agg

    gated: set[str] = set()
    report: set[str] = set()
    reasons: dict[str, str] = {}

    # 1) aggregates decide first.
    for agg in AGGREGATIONS:
        if _passes(agg, stability, tolerances):
            gated.add(agg)
            reasons[agg] = "gated_aggregate"
        else:
            report.add(agg)
            reasons[agg] = "aggregate_" + _demote_reason(
                agg, stability, tolerances
            )

    # 2) per-age members: superseded by a gating aggregate, else on merit.
    for member, agg in member_of.items():
        if agg in gated:
            report.add(member)
            reasons[member] = f"superseded_by:{agg}"
        elif _passes(member, stability, tolerances):
            gated.add(member)
            reasons[member] = "gated_per_age"
        else:
            report.add(member)
            reasons[member] = _demote_reason(member, stability, tolerances)

    # 3) every other cell on its own merit.
    for key in stability:
        if key in AGGREGATIONS or key in member_of:
            continue
        if _passes(key, stability, tolerances):
            gated.add(key)
            reasons[key] = "gated"
        else:
            report.add(key)
            reasons[key] = _demote_reason(key, stability, tolerances)
    return gated, report, reasons


def draft_thresholds(
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    """DRAFT per-cell |ln ratio| tolerances for the GATED cells, with the
    machine-checkable derivation and the realised-sigma diagnostic."""
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
    """Fold the tolerance, realised sigma, gate flag and report reason into
    ``stability`` in place, so the partition is fully machine-visible."""
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


# --------------------------------------------------------------------------
# Holdout-id commitment (per-seed reproducible splits, finding 5)
# --------------------------------------------------------------------------
def holdout_id_commitment(panel: transitions.MaritalPanel) -> dict[str, Any]:
    """The sha256 of each gate seed's holdout (side A) person-ids.

    Commits the per-seed holdout set so a candidate cannot pick its split:
    the holdout is derived from the seed by the pinned numpy Generator, and
    its sorted person-id list is hashed here.
    """
    per_seed = []
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        per_seed.append(
            {
                "seed": seed,
                "n_holdout_persons": len(ids),
                "holdout_person_id_sha256": digest,
            }
        )
    return {
        "gate_seeds": list(GATE_SEEDS),
        "id_column": "person_id",
        "fraction": 0.5,
        "numpy_generator": (
            "populace_dynamics.harness.panel.split_panel_by_person: side A "
            "(holdout) = persons i of np.sort(unique person_id) with "
            "np.random.default_rng(seed).random(n_persons)[i] < 0.5"
        ),
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------
# Training-copy check + faithful-candidate operating characteristic
# --------------------------------------------------------------------------
def training_copy_check(
    per_seed: list[dict[str, Any]],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    """A train-half copy scored under option (a): does it pass, and how far?

    Under the protocol a candidate is fit on side B (train) and scored
    against side A (holdout); a verbatim train-copy therefore EMITS side
    B's rates, so its per-cell score is exactly ``|ln(rate_B / rate_A)|`` =
    the committed floor value for that seed. It passes at the noise floor --
    as ANY moment gate's copy must -- because a model reproducing the
    population moments should pass. The added structural cells do NOT and
    cannot change this (a copy reproduces every moment, structural ones
    included); the memorisation defence is procedural (finding 5), not the
    cell set. Reported as such.
    """
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
            "A train-copy passes at ~the noise floor (max score well below "
            "1x tolerance), as any moment gate's copy must. The added "
            "structural cells do not fix this -- a copy reproduces every "
            "moment; the memorisation defence is procedural (registration + "
            "holdout exclusion + no_self_rescue, finding 5), NOT the cell "
            "set. What the added cells DO catch is non-copy structural "
            "failure (origin-blind / sex-blind / time-homogeneous)."
        ),
    }


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def faithful_candidate_oc(
    noise_floor: dict[str, Any],
    tolerances: dict[str, float],
    gated: set[str],
) -> dict[str, Any]:
    """The operating characteristic of the 4-of-5 conjunction for a faithful
    candidate (round-1 finding 2, recomputed on the stabilised floor).

    Under option (a) a faithful candidate's per-cell score is distributed
    like the half-vs-half floor: half-normal with scale
    ``realized_sigma``. Per-cell pass probability is therefore
    ``2*Phi(tol/sigma) - 1``; a seed passes iff every gated cell holds
    (product, independence approximation); the gate passes iff >=4 of 5
    seeds pass. The design intends ~4 sigma -- this table shows the
    stabilised floor delivers it, where the committed 5-seed floor did not.
    """
    per_cell = {}
    p_seed = 1.0
    for key in sorted(gated):
        sigma = noise_floor[key]["realized_sigma"]
        tol = tolerances[key]
        p = (2.0 * _normal_cdf(tol / sigma) - 1.0) if sigma > 0 else 1.0
        per_cell[key] = {
            "tolerance": tol,
            "realized_sigma": sigma,
            "tolerance_sigma_units": (
                round(tol / sigma, 3) if sigma > 0 else None
            ),
            "cell_pass_prob": round(p, 6),
        }
        p_seed *= p
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return {
        "method": (
            "Independence-approx normal OC. Per gated cell a faithful "
            "candidate's score ~ half-normal(realized_sigma); cell pass = "
            "2*Phi(tolerance/sigma)-1. Seed pass = product over gated "
            "cells; gate = P(>=4 of 5 seeds) = p^5 + 5 p^4 (1-p)."
        ),
        "n_gated_cells": len(gated),
        "p_seed_pass": round(p_seed, 4),
        "p_gate_pass_4_of_5": round(p_gate, 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# External anchors
# --------------------------------------------------------------------------
def _psid_asfr_window(
    fert: transitions.FertilityPanel, *, start_year: int | None
) -> dict[str, dict[str, float]]:
    """PSID ASFR per 1,000 women by band for a calendar-year window."""
    wy = fert.woman_years
    bi = fert.births
    if start_year is not None:
        wy = wy[wy.year >= start_year]
        bi = bi[bi.year >= start_year]
    out: dict[str, dict[str, float]] = {}
    for lo, hi in transitions.ASFR_AGE_BANDS:
        band = transitions.band_label(lo, hi)
        w = wy[(wy.age >= lo) & (wy.age <= hi)]
        b = bi[(bi.mother_age >= lo) & (bi.mother_age <= hi)]
        den = float(w.weight.sum())
        num = float(b.weight.sum())
        out[band] = {
            "psid_asfr_per_1000": (num / den * 1000.0) if den > 0 else 0.0,
            "n_births": int(len(b)),
        }
    return out


def _asfr_period_matched(
    fert: transitions.FertilityPanel, nchs_annual: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Exposure-weighted, period-matched PSID/NCHS ASFR ratios (finding 6b).

    Each PSID woman-year in calendar year y is matched to that year's own
    national ASFR, so the ratio is not confounded by the secular fertility
    decline (the pooled-window-vs-single-vintage defect). Restricted to the
    calendar years the annual NCHS series and the PSID panel share.
    """
    annual_years = sorted(int(y) for y in nchs_annual)
    matched_years = [y for y in annual_years if y <= transitions.MAX_YEAR]
    wy = fert.woman_years[fert.woman_years.year.isin(matched_years)]
    bi = fert.births[fert.births.year.isin(matched_years)]

    by_band: dict[str, Any] = {}
    ratios: list[float] = []
    for lo, hi in transitions.ASFR_AGE_BANDS:
        band = transitions.band_label(lo, hi)
        w = wy[(wy.age >= lo) & (wy.age <= hi)]
        b = bi[(bi.mother_age >= lo) & (bi.mother_age <= hi)]
        exposure_by_year = w.groupby("year").weight.sum()
        psid_births = float(b.weight.sum())
        psid_wy = float(w.weight.sum())
        expected = 0.0
        for year, exposure in exposure_by_year.items():
            national = nchs_annual[str(int(year))][band]
            expected += float(exposure) * national / 1000.0
        ratio = (psid_births / expected) if expected > 0 else None
        if ratio is not None and psid_births > 0:
            ratios.append(ratio)
        by_band[band] = {
            "psid_asfr_per_1000": (
                psid_births / psid_wy * 1000.0 if psid_wy > 0 else 0.0
            ),
            "exposure_weighted_nchs_asfr_per_1000": (
                expected / psid_wy * 1000.0 if psid_wy > 0 else 0.0
            ),
            "period_matched_ratio": ratio,
            "n_births": int(len(b)),
        }
    return {
        "matched_years": matched_years,
        "method": (
            "observed PSID births / births expected under each woman-year's "
            "own calendar-year national ASFR (exposure-weighted, "
            "period-matched)"
        ),
        "by_band": by_band,
        "ratio_summary": {
            "n_bands": len(ratios),
            "median_ratio": float(np.median(ratios)) if ratios else None,
            "min_ratio": float(np.min(ratios)) if ratios else None,
            "max_ratio": float(np.max(ratios)) if ratios else None,
        },
    }


def asfr_anchor(fert: transitions.FertilityPanel) -> dict[str, Any]:
    """PSID vs NCHS age-specific fertility rates.

    The period-matched block (finding 6b) is the honest headline: each PSID
    woman-year weighed against its own year's national rate. The single-
    vintage 'all'/'recent' windows are kept for continuity but labelled as
    vintage-confounded, since they compare a pooled PSID window against the
    2024 column across a steep decline.
    """
    nchs = json.loads(ASFR_PATH.read_text())
    nchs_asfr = nchs["tables"][str(nchs["vintage_year"])]
    nchs_annual = nchs["annual"]["tables"]
    sha = hashlib.sha256(ASFR_PATH.read_bytes()).hexdigest()

    windows: dict[str, Any] = {}
    for name, start in (("all", None), ("recent", RECENT_START_YEAR)):
        psid = _psid_asfr_window(fert, start_year=start)
        by_band: dict[str, Any] = {}
        ratios: list[float] = []
        for lo, hi in transitions.ASFR_AGE_BANDS:
            band = transitions.band_label(lo, hi)
            p = psid[band]["psid_asfr_per_1000"]
            n = nchs_asfr[band]
            ratio = (p / n) if n > 0 else None
            if ratio is not None and psid[band]["n_births"] > 0:
                ratios.append(ratio)
            by_band[band] = {
                "psid_asfr_per_1000": p,
                "nchs_asfr_per_1000": n,
                "ratio": ratio,
                "n_births": psid[band]["n_births"],
            }
        windows[name] = {
            "start_year_min": start,
            "vintage_confounded": True,
            "by_band": by_band,
            "ratio_summary": {
                "n_bands": len(ratios),
                "median_ratio": float(np.median(ratios)) if ratios else None,
                "min_ratio": float(np.min(ratios)) if ratios else None,
                "max_ratio": float(np.max(ratios)) if ratios else None,
            },
        }

    period_matched = _asfr_period_matched(fert, nchs_annual)
    return {
        "nchs_reference_file": str(ASFR_PATH.relative_to(ROOT)),
        "nchs_vintage_year": nchs["vintage_year"],
        "nchs_citation": nchs["report"]["nvsr_citation"],
        "nchs_reference_sha256": sha,
        "concept_note": (
            "PSID age-specific fertility rate (weighted maternal births per "
            "1,000 woman-years in the band) vs the NCHS ASFR. The "
            "PERIOD-MATCHED block is the honest anchor: each PSID woman-year "
            "is weighed against its own calendar year's national ASFR (from "
            "the annual NCHS series), so the ratio is not confounded by the "
            "secular fertility decline. The single-vintage 'all'/'recent' "
            "windows compare a pooled PSID window against the 2024 column "
            "across a steep decline and are labelled vintage_confounded; "
            "they are kept for continuity only. Ratios REPORTED, not tuned."
        ),
        "period_matched": period_matched,
        "windows": windows,
    }


def _psid_crude_rates(
    panel: transitions.MaritalPanel, *, start_year: int | None
) -> dict[str, float]:
    """PSID crude-equivalent marriage & divorce rates per 1,000 person-years.

    Marriages = first-marriage + remarriage events; divorces = divorce
    events; denominator = all weighted person-years (age 15+, every marital
    state). A crude-equivalent to the NCHS per-1,000-total-population rate
    with the denominator on marriageable-age person-years, not total pop.
    """
    py = panel.person_years
    ev = panel.events
    if start_year is not None:
        py = py[py.year >= start_year]
        ev = ev[ev.year >= start_year]
    py_wt = float(py.weight.sum())
    marriages = ev[ev.transition.isin(["first_marriage", "remarriage"])]
    divorces = ev[ev.transition == "divorce"]
    return {
        "person_years_weighted": py_wt,
        "marriage_rate_per_1000": (
            float(marriages.weight.sum() / py_wt * 1000.0) if py_wt else 0.0
        ),
        "divorce_rate_per_1000": (
            float(divorces.weight.sum() / py_wt * 1000.0) if py_wt else 0.0
        ),
        "n_marriage_events": int(len(marriages)),
        "n_divorce_events": int(len(divorces)),
    }


def marriage_divorce_anchor(panel: transitions.MaritalPanel) -> dict[str, Any]:
    """PSID crude-equivalent vs NCHS national marriage/divorce rates.

    Concept-decomposed (finding 6a): the raw PSID/NCHS ratio is the product
    of a person-vs-couple factor (x2 exactly) and a denominator factor
    (~1.22, the 15+ population share), so the RESIDUAL after dividing the
    ~2.44 concept factor out is the real agreement -- ~1.0 on the recent
    window (a near-bullseye), with the 'all'-window residual a labelled
    historical-period effect, not an unexplained excess.
    """
    nchs = json.loads(MD_PATH.read_text())
    year = str(nchs["latest_year"])
    nchs_m = nchs["tables"]["marriage"][year]["rate_per_1000"]
    nchs_d = nchs["tables"]["divorce"][year]["rate_per_1000"]
    sha = hashlib.sha256(MD_PATH.read_bytes()).hexdigest()

    denominator_factor = 1.0 / POP_15PLUS_SHARE
    concept_factor = PERSON_TO_COUPLE_FACTOR * denominator_factor

    windows: dict[str, Any] = {}
    for name, start in (("all", None), ("recent", RECENT_START_YEAR)):
        psid = _psid_crude_rates(panel, start_year=start)
        m_ratio = psid["marriage_rate_per_1000"] / nchs_m if nchs_m else None
        d_ratio = psid["divorce_rate_per_1000"] / nchs_d if nchs_d else None
        windows[name] = {
            "start_year_min": start,
            "psid_marriage_rate_per_1000_py15plus": psid[
                "marriage_rate_per_1000"
            ],
            "nchs_marriage_rate_per_1000_totalpop": nchs_m,
            "marriage_ratio": m_ratio,
            "marriage_residual_after_concept": (
                round(m_ratio / concept_factor, 3)
                if m_ratio is not None
                else None
            ),
            "psid_divorce_rate_per_1000_py15plus": psid[
                "divorce_rate_per_1000"
            ],
            "nchs_divorce_rate_per_1000_totalpop": nchs_d,
            "divorce_ratio": d_ratio,
            "divorce_residual_after_concept": (
                round(d_ratio / concept_factor, 3)
                if d_ratio is not None
                else None
            ),
            "n_marriage_events": psid["n_marriage_events"],
            "n_divorce_events": psid["n_divorce_events"],
        }
    return {
        "nchs_reference_file": str(MD_PATH.relative_to(ROOT)),
        "nchs_latest_year": nchs["latest_year"],
        "nchs_citation": nchs["report"]["nvss_citation"],
        "nchs_reference_sha256": sha,
        "concept_decomposition": {
            "person_to_couple_factor": PERSON_TO_COUPLE_FACTOR,
            "pop_15plus_share": POP_15PLUS_SHARE,
            "denominator_factor": round(denominator_factor, 4),
            "concept_factor": round(concept_factor, 4),
            "note": (
                "PSID counts PERSONS transitioning (two spouses per "
                "marriage / divorce); NCHS counts COUPLES -- a factor of 2 "
                "exactly. The PSID crude-equivalent denominator is "
                "person-years age 15+ (~82% of the population, US Census), "
                "so it runs ~1.22x hotter than the NCHS per-total-population "
                "rate. Concept factor ~2.44; the residual after dividing it "
                "out is the real PSID/NCHS agreement: ~1.0 on the recent "
                "window. The 'all'-window residual (~1.4-1.7) is a labelled "
                "historical-period effect (deeper history married more), not "
                "an unexplained excess."
            ),
        },
        "concept_delta_note": (
            "LOOSE order-of-magnitude anchor, REPORTED not gated. See "
            "concept_decomposition: the raw ~2.4-2.5x recent ratio is the "
            "person-vs-couple x denominator concept factor (~2.44), leaving "
            "a residual ~1.0 -- the anchor is a near-bullseye once "
            "concept-aligned. The gate-2 external check on this pair is a "
            "shape/direction report, not a level gate."
        ),
        "windows": windows,
    }


# --------------------------------------------------------------------------
# Draft-thresholds note (pre-lock; feeds gates.yaml gate_2)
# --------------------------------------------------------------------------
def draft_thresholds_note(
    gated: set[str],
    report: set[str],
    asfr: dict[str, Any],
    oc: dict[str, Any],
) -> str:
    recent = asfr["period_matched"]["ratio_summary"]["median_ratio"]
    return (
        "DRAFT GATE-2 VALIDATION EVIDENCE -- PRE-LOCK, NOT RATIFIED "
        "(v2, round-1 referee amendments applied).\n\n"
        "Pre-lock evidence package for the gate-2 family-transition "
        "statistics (the analogue of what preceded gate 1's lock, one gate "
        "down). It changes no locked value and no model reads it. The DRAFT "
        "thresholds it feeds live in gates.yaml under gate_2 with "
        "locked: false and status: draft_pending_referee_round. "
        "Ratification requires the full lock ceremony: floors -> thresholds "
        "with machine-bound derivations -> an adversarial referee round -> "
        "verification -> maintainer ratification by merge. The k value "
        f"({DRAFT_K}) is a DRAFT proposal (gate-1 c2st precedent k=4.2 + the "
        "4-of-5-seed rule), for the ceremony to fix.\n\n"
        "PROTOCOL (finding 1, option a -- gate-1 mirror). Per gate seed s a "
        "candidate is refit on the seed-s TRAIN complement (side B; the "
        "seed-s holdout persons are excluded from all fitting), simulates "
        "the holdout persons' family histories (simulation seed s), and "
        "each cell is scored |ln(r_candidate,s / r_holdout,s)| against the "
        "seed-s HOLDOUT half's own empirical rate (side A, committed per "
        "seed in noise_floor_per_seed as rate_a). The symmetric half-vs-"
        "half floor |ln(rate_A/rate_B)| is then exactly the null. Seed s "
        "passes iff every gate-eligible cell holds at s; the gate passes "
        "iff >=4 of 5 seeds pass.\n\n"
        "INTERNAL FLOOR (finding 2). The person-disjoint 50/50 half-split "
        "|log rate ratio| floor over 100 split seeds (noise_floor_seeds_0_"
        "99). Each DRAFT per-cell tolerance is "
        f"round(mean + {DRAFT_K}*sd, {DRAFT_ROUNDING}); on 100 seeds the "
        "estimator is stable (~3.2 sigma of the cell's own realized_sigma), "
        "where 5 seeds realised 0.9-5.6 sigma. The faithful-candidate "
        f"operating characteristic recomputes P(seed) = {oc['p_seed_pass']} "
        f"and P(gate >=4/5) = {oc['p_gate_pass_4_of_5']} over "
        f"{oc['n_gated_cells']} gated cells (faithful_candidate_oc).\n\n"
        "POWER CAP (finding 3). A cell is gate-eligible only if its "
        f"stabilised tolerance <= T_max = {T_MAX_SOURCE} (a gated cell "
        "accepts at most a 1.5x rate error) AND it carries >=20 events on "
        "the weaker half. Per-age cells that fail the cap are demoted to "
        "report-only and their coverage recovered by the pre-registered "
        "aggregates (widowhood.45+|male, widowhood.45-64|female, "
        "first_marriage.35+|sex). The partition is derived from "
        "cell_stability, never hand-picked.\n\n"
        "JOINT / SEQUENCE STATISTICS (finding 4). Origin-split remarriage "
        "(after divorce vs after widowhood; widow(er) under/over 60), "
        "cohort ever-married-by-40 (the secular decline), and dissolved-"
        "state stock shares by age x sex bind the cross-transition, cohort "
        "and stock structure a banded-marginal gate misses. Disclosure: the "
        "gate certifies POPULATION MOMENTS, not novelty -- a verbatim "
        "training-copy passes at the noise floor by design (training_copy_"
        "check); the memorisation defence is procedural (registration + "
        "holdout exclusion + no_self_rescue), not the cell set.\n\n"
        "EXTERNAL ANCHOR (finding 6). NCHS ASFR is now PERIOD-MATCHED "
        "(each PSID woman-year weighed against its own calendar year's "
        "national rate; recent-band median PSID/NCHS ratio "
        f"~{recent:.2f}), not the vintage-confounded pooled-vs-2024 window. "
        "The marriage/divorce anchor is concept-decomposed: the raw ~2.44x "
        "ratio is person-vs-couple (x2) x denominator (~1.22), leaving a "
        "recent residual ~1.0. Both are SHAPE/RATIO REPORTS, never level "
        "gates.\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY. Gated cells: "
        f"{sorted(gated)}. Report-only: {sorted(report)}.\n\n"
        "GOVERNANCE (finding 5). Registration on issue #42, one-shot outer "
        "runs, amendments only by public proposal + referee round, "
        "no_self_rescue and the version pin inherited from gate 1; the "
        "per-seed holdout ids are committed (holdout_ids); every gated "
        "statistic is weighted by the person-constant most-recent-positive "
        "PSID weight (no unweighted gated statistic).\n\n"
        "BASELINE CONVENTION. These transitions feed benefit levels through "
        "household composition and survivorship; a scored reform states its "
        "scheduled/payable baseline. This evidence fixes the component's "
        "estimation/validation standard only."
    )


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _sha_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    assert "populace.fit" not in sys.modules

    panel, fert, data_meta = load_panels()
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons, "
            f"{data_meta['n_transition_events']} events"
        )

    ref_w = transitions.reference_moments(panel, fert, weighted=True)
    ref_u = transitions.reference_moments(panel, fert, weighted=False)
    cell_keys = list(ref_w)
    reference_moments = {
        key: {
            "rate": ref_w[key]["rate"],
            "rate_unweighted": ref_u[key]["rate"],
            "num_wt": ref_w[key]["num_wt"],
            "den_wt": ref_w[key]["den_wt"],
            "n_events": ref_w[key]["n_events"],
        }
        for key in cell_keys
    }

    per_seed = [measure_seed_halfsplit(s, panel, fert) for s in SEEDS]
    noise_floor, stability = pool_internal_floor(per_seed, cell_keys)
    tolerances = raw_tolerances(noise_floor)
    gated, report, reasons = partition_cells(stability, tolerances)
    drafts = draft_thresholds(noise_floor, tolerances, gated)
    annotate_stability(stability, noise_floor, tolerances, gated, reasons)

    holdout_ids = holdout_id_commitment(panel)
    copy_check = training_copy_check(per_seed, tolerances, gated)
    oc = faithful_candidate_oc(noise_floor, tolerances, gated)

    asfr = asfr_anchor(fert)
    md = marriage_divorce_anchor(panel)

    if verbose:
        print(
            f"cells: {len(cell_keys)} ({len(gated)} gated, "
            f"{len(report)} report-only); draft thresholds: {len(drafts)}"
        )
        print(
            f"faithful OC: P(seed)={oc['p_seed_pass']} "
            f"P(gate>=4/5)={oc['p_gate_pass_4_of_5']}; "
            f"train-copy passes 4/5: {copy_check['passes_4_of_5']} "
            f"(max {copy_check['max_score_over_tolerance']}x tol)"
        )
        print(
            "period-matched ASFR median PSID/NCHS ratio: "
            f"{asfr['period_matched']['ratio_summary']['median_ratio']:.3f}"
        )

    # Only the gate seeds are stored in full per-seed detail (reproduction +
    # the gate's per-seed holdout reference rates + the train-copy check);
    # the pooled floor's `values` arrays carry all 100 seeds' floor draws.
    per_seed_stored = per_seed[: len(GATE_SEEDS)]

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate2_floors_v2",
        "supersedes": "gate2_floors_v1 (round-1 referee record, retained)",
        "referee_round": "PR #79 comment 4910467957",
        "reported_anchor_not_gated": True,
        "component": (
            "family transitions (gate 2 pre-lock; marriage / divorce / "
            "widowhood / remarriage hazards + occupancy + fertility)"
        ),
        "purpose": (
            "Gate-2 pre-lock evidence (v2, round-1 amendments): committed "
            "reference moments, the 100-seed person-disjoint half-split "
            "noise floor (the sd basis the DRAFT gate-2 thresholds derive "
            "from), the coherent option-(a) scoring protocol, the power-cap "
            "partition, the joint/sequence statistics, and the concept-"
            "decomposed / period-matched external anchors. Reads no gate and "
            "changes no gate on its own; the DRAFT thresholds live in "
            "gates.yaml gate_2 (locked: false, status: "
            "draft_pending_referee_round). The lock ceremony comes after. "
            "See draft_thresholds_note (NOT RATIFIED)."
        ),
        "holdout_basis": ["mh85_23", "cah85_23", "MX23REL"],
        "data": {
            "marriage_history": (
                "populace_dynamics.data.marriage.marriage_history (mh85_23; "
                "retrospective per-marriage records + never-married "
                "placeholders)"
            ),
            "childbirth_history": (
                "populace_dynamics.data.births.birth_history (cah85_23; "
                "retrospective per-child records)"
            ),
            "deaths_and_sex": (
                "populace_dynamics.data.deaths.read_death_records (ind2023er "
                "ER32000 sex, ER32050 exact year of death) -- own-death "
                "censoring of the at-risk window"
            ),
            "weights": (
                "person-constant most-recent positive cross-sectional weight "
                "from populace_dynamics.data.panels.demographic_panel; "
                "persons with no positive PSID weight are excluded from the "
                "weighted moments (repo weight>0 convention). Every gated "
                "statistic is weighted; there is no unweighted gated "
                "statistic. The weighted/unweighted gap is material on "
                "several cells (finding 5), so the weighting choice is "
                "load-bearing and pinned here, not inherited implicitly."
            ),
            **data_meta,
        },
        "panel_construction": {
            "unit": "annual person-year",
            "start_age": transitions.START_AGE,
            "max_year": transitions.MAX_YEAR,
            "rule": (
                "Reconstruct marital state annually from each person's "
                "ordered marriage episodes (never_married -> married -> "
                "divorced/widowed/separated), from age START_AGE through "
                "censor_year = min(most_recent_report_year, exact death "
                "year, MAX_YEAR). State at person-year t is the state "
                "entering t (a transition in year t is the event, and t "
                "carries the pre-transition at-risk state -- discrete-time "
                "hazard, merge_asof allow_exact_matches=False)."
            ),
            "widowhood_source": (
                "the marriage file's how_ended=widowhood (directly dated at "
                "the marriage end year); deaths supply own-death censoring "
                "of continued exposure"
            ),
            "remarriage_origin": (
                "origin state attributed from the dissolved marriage's "
                "prev_how_ended (not the state-entering-year marital_state), "
                "so same-year dissolve-remarry events keep their true "
                "divorced/widowed origin; post-separation starts are "
                "excluded from the numerator to keep it state-consistent "
                "with the divorced/widowed person-year denominator"
            ),
            "coverage_caveats": [
                "retrospective histories are LEFT-TRUNCATED: reconstructed "
                "young-adult person-years for a person first seen at an old "
                "age enter only if they survived and were sampled -- pooled "
                "deep-history young-age rates are selected (reported, not "
                "corrected)",
                "the 'all' external-anchor window pools all reconstructed "
                "calendar years against a single recent national year (a "
                "named period delta); the period-matched anchor removes it",
                "person-constant weight from the last positive PSID wave; "
                "retrospective person-years before that wave carry it",
            ],
        },
        "statistic_families": {
            "first_marriage_hazard": "by age band x sex (+ 35+ aggregate)",
            "divorce_hazard": "by marriage-duration band",
            "widowhood_incidence": (
                "by age band x sex (+ 45+ male / 45-64 female aggregates)"
            ),
            "remarriage_hazard": (
                "by years-since-dissolution; by origin state (after divorce "
                "/ after widowhood; widow(er) under/over 60)"
            ),
            "occupancy": (
                "share ever married by 40 / 60 x sex; mean lifetime "
                "marriages x sex; cohort ever-married-by-40; dissolved-state "
                "stock shares by age x sex"
            ),
            "fertility": (
                "age-specific birth rate by 5-year band; completed "
                "fertility by decade birth cohort"
            ),
        },
        "reference_moments": reference_moments,
        "external_anchor": {"asfr": asfr, "marriage_divorce": md},
        "internal_noise_floor": {
            "method": (
                "person-disjoint 50/50 half-split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction=0.5, seeds 0-99) of every reference-moment cell; "
                "the floor statistic is |ln(rate_A / rate_B)| between two "
                "independent real halves -- the sd basis the DRAFT gate-2 "
                "thresholds derive from as round(mean + k*sd)"
            ),
            "floor_seeds": list(SEEDS),
            "gate_seeds": list(GATE_SEEDS),
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
            "t_max": T_MAX,
            "t_max_source": T_MAX_SOURCE,
        },
        "protocol": {
            "option": "a",
            "description": (
                "Per gate seed s: refit the candidate on the seed-s train "
                "complement (side B), simulate the seed-s holdout persons "
                "(side A, simulation seed s), score each cell "
                "|ln(r_candidate,s / r_holdout,s)| against side A's own "
                "empirical rate (rate_a). The half-vs-half floor is the "
                "null. Seed passes iff every gated cell holds; gate passes "
                "iff >=4 of 5 gate seeds pass."
            ),
            "varies_per_seed": (
                "the 50/50 person split (holdout = side A of "
                "split_panel_by_person(seed=s)); the candidate's fit "
                "complement and simulated holdout"
            ),
            "candidate_emits": (
                "each cell's rate from its simulated histories of the seed-s "
                "holdout persons, same annual person-year construction, "
                "same weight"
            ),
            "scored_against": (
                "the seed-s holdout half's empirical rate (rate_a in "
                "noise_floor_per_seed)"
            ),
            "conjunction": "all gated cells per seed AND >=4 of 5 gate seeds",
        },
        "aggregations": {
            agg: {
                "members": list(members),
                "gated": agg in gated,
            }
            for agg, members in AGGREGATIONS.items()
        },
        "noise_floor_seeds_0_99": noise_floor,
        "cell_stability": stability,
        "gate_partition": {
            "gate_eligible": sorted(gated),
            "report_only": sorted(report),
            "n_gate_eligible": len(gated),
            "n_report_only": len(report),
        },
        "noise_floor_per_seed": per_seed_stored,
        "holdout_ids": holdout_ids,
        "training_copy_check": copy_check,
        "faithful_candidate_oc": oc,
        "draft_thresholds": {
            "k": DRAFT_K,
            "rounding": DRAFT_ROUNDING,
            "t_max": T_MAX,
            "statistic": "log_ratio_abs_max",
            "note": (
                "DRAFT per-cell |ln ratio| tolerances = round(floor mean + "
                f"{DRAFT_K}*sd, {DRAFT_ROUNDING}); gated cells only "
                "(defined on all seeds AND >=20 events AND tolerance <= "
                "T_max AND not superseded by an aggregate). Mirrored into "
                "gates.yaml gate_2 (locked: false), machine-bound by "
                "tests/test_gates_derivations.py."
            ),
            "cells": drafts,
        },
        "draft_thresholds_note": draft_thresholds_note(
            gated, report, asfr, oc
        ),
        "revision_pins": {
            "populace_dynamics_sha": _git_sha(ROOT),
            "nchs_asfr_sha256": _sha_of_file(ASFR_PATH),
            "nchs_marriage_divorce_sha256": _sha_of_file(MD_PATH),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
