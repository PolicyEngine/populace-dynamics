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

Five marital-transition statistic families plus fertility, all built from
the retrospective PSID history files via
:mod:`populace_dynamics.data.transitions`:

1. first-marriage hazard by age band x sex,
2. divorce hazard by marriage-duration band,
3. widowhood incidence by age band x sex,
4. remarriage hazard by years-since-dissolution band,
5. occupancy: share ever married by 40 / 60 x sex, mean lifetime marriages,
6. fertility: age-specific birth rates + completed fertility by cohort.

Three products:

(a) **Reference moments.** Weighted PSID rates per cell (the committed
    battery-analogue targets a candidate reproduces), with the unweighted
    rate reported alongside so the weighted/unweighted gap is visible.
(b) **Internal noise floor.** A person-disjoint 50/50 half-split
    (``split_panel_by_person``, seeds 0-4) of every cell; the floor
    statistic is ``|ln(rate_A / rate_B)|`` between two independent real
    halves -- the sd basis the DRAFT per-cell tolerances derive from as
    ``round(mean + k*sd)`` (the ``tests/test_gates_derivations.py``
    convention). Cells with fewer than 20 events on the weaker half of the
    worst seed are report-only (NCHS's reliability floor), the analog of
    the PIA floor's d1/d2.
(c) **External anchors.** PSID age-specific fertility rates vs the NCHS
    2024 ASFR, and PSID crude-equivalent marriage/divorce rates vs the
    NCHS national crude rates -- reported as ratios with the coverage and
    period-concept deltas stated, never tuned (PSID retrospective family
    files have known coverage caveats).

Run from the repository root with the PSID history files staged::

    .venv/bin/python scripts/build_gate2_floors.py

It needs no populace-fit (real-vs-real / real-vs-external only).
"""

from __future__ import annotations

import hashlib
import json
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
ARTIFACT_PATH = ROOT / "runs" / "gate2_floors_v1.json"
ASFR_PATH = ROOT / "data" / "external" / "nchs_asfr_2024.json"
MD_PATH = ROOT / "data" / "external" / "nchs_marriage_divorce_rates_2023.json"
ARTIFACT_SCHEMA_VERSION = "gate2_floors.v1"

SEEDS = (0, 1, 2, 3, 4)
#: Minimum events (weaker half, worst seed) for a cell to be gate-eligible.
#: 20 is NCHS's own reliability floor (rates on < 20 events are flagged
#: unreliable); below it a cell is report-only, the analog of the PIA
#: floor's denominator-fragile d1/d2.
MIN_EVENTS_FOR_GATE = 20
#: DRAFT threshold multiplier: |ln ratio| tolerance = round(mean + k*sd).
#: ~4 sigma, consistent with the gate-1 c2st precedent (k=4.2) and the
#: 4-of-5-seed rule. DRAFT -- the referee round sets the final k.
DRAFT_K = 4
DRAFT_ROUNDING = 3
#: Calendar-year floor for the "recent" external-anchor window (brackets
#: the deep-history period effect against the ~2023/2024 national rates).
RECENT_START_YEAR = 2010


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
# Internal noise floor (person-disjoint 50/50 half-split, seeds 0-4)
# --------------------------------------------------------------------------
def _summary(values: list[float]) -> dict[str, Any]:
    """Mean/sd/min/max and raw values (float64, ddof=1), the floor convention
    shared with the other ``runs/`` artifacts."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
        "values": [float(v) for v in arr],
    }


def measure_seed_halfsplit(
    seed: int,
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
) -> dict[str, Any]:
    """One seed: split persons 50/50, all reference-moment cells per half.

    Returns per cell the two halves' rates, the minimum event count, the
    absolute log rate ratio ``|ln(r_A / r_B)|`` and the absolute percent
    gap -- undefined (``None``) when either half's rate is zero
    (denominator-fragile).
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


def pool_internal_floor(
    per_seed: list[dict[str, Any]], cell_keys: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Across-seed floor per cell + the gate-eligible/report-only partition.

    ``noise_floor_seeds_0_4[key]`` follows the committed-floor convention
    (mean/sd/values of ``|log rate ratio|`` plus the nested percent-gap
    block), so the DRAFT thresholds derive ``round(mean + k*sd)`` from it
    exactly as ``test_gates_derivations`` binds. Only cells defined on
    every seed carry a floor block. ``cell_stability[key]`` records the
    defined-seed count and the minimum event count -- the machine-visible
    evidence for the gate-eligible vs report-only split.
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
            "gate_eligible": (
                defined_seeds == len(per_seed)
                and min_events >= MIN_EVENTS_FOR_GATE
            ),
        }
        if defined_seeds == len(per_seed):
            block = _summary([float(v) for v in log_ratios])
            block["pct_diff_abs"] = _summary([float(v) for v in pct_diffs])
            noise_floor[key] = block
    return noise_floor, stability


def draft_thresholds(
    noise_floor: dict[str, Any], stability: dict[str, Any]
) -> dict[str, Any]:
    """DRAFT per-cell |ln ratio| tolerances = round(mean + k*sd) for the
    gate-eligible cells, with the machine-checkable derivation recorded."""
    out: dict[str, Any] = {}
    for key, stab in stability.items():
        if not stab["gate_eligible"]:
            continue
        mean = noise_floor[key]["mean"]
        sd = noise_floor[key]["sd"]
        out[key] = {
            "log_ratio_abs_max": round(mean + DRAFT_K * sd, DRAFT_ROUNDING),
            "derivation": {
                "floor_mean": mean,
                "floor_sd": sd,
                "k": DRAFT_K,
                "rounding": DRAFT_ROUNDING,
            },
        }
    return out


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


def asfr_anchor(fert: transitions.FertilityPanel) -> dict[str, Any]:
    """PSID vs NCHS 2024 age-specific fertility rates (all + recent)."""
    nchs = json.loads(ASFR_PATH.read_text())
    nchs_asfr = nchs["tables"][str(nchs["vintage_year"])]
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
            "by_band": by_band,
            "ratio_summary": {
                "n_bands": len(ratios),
                "median_ratio": float(np.median(ratios)) if ratios else None,
                "min_ratio": float(np.min(ratios)) if ratios else None,
                "max_ratio": float(np.max(ratios)) if ratios else None,
            },
        }
    return {
        "nchs_reference_file": str(ASFR_PATH.relative_to(ROOT)),
        "nchs_vintage_year": nchs["vintage_year"],
        "nchs_citation": nchs["report"]["nvsr_citation"],
        "nchs_reference_sha256": sha,
        "concept_note": (
            "PSID age-specific fertility rate (weighted maternal births per "
            "1,000 woman-years in the band) vs the NCHS 2024 published "
            "ASFR. Concept-aligned (both births per 1,000 women by age); "
            "the 'all' window pools PSID births across all reconstructed "
            "calendar years (a named period delta -- earlier decades peak "
            "younger and higher), the 'recent' window (calendar year >= "
            f"{RECENT_START_YEAR}) brackets it. Ratios REPORTED, not tuned."
        ),
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
    """PSID crude-equivalent vs NCHS national marriage/divorce rates."""
    nchs = json.loads(MD_PATH.read_text())
    year = str(nchs["latest_year"])
    nchs_m = nchs["tables"]["marriage"][year]["rate_per_1000"]
    nchs_d = nchs["tables"]["divorce"][year]["rate_per_1000"]
    sha = hashlib.sha256(MD_PATH.read_bytes()).hexdigest()

    windows: dict[str, Any] = {}
    for name, start in (("all", None), ("recent", RECENT_START_YEAR)):
        psid = _psid_crude_rates(panel, start_year=start)
        windows[name] = {
            "start_year_min": start,
            "psid_marriage_rate_per_1000_py15plus": psid[
                "marriage_rate_per_1000"
            ],
            "nchs_marriage_rate_per_1000_totalpop": nchs_m,
            "marriage_ratio": (
                psid["marriage_rate_per_1000"] / nchs_m if nchs_m else None
            ),
            "psid_divorce_rate_per_1000_py15plus": psid[
                "divorce_rate_per_1000"
            ],
            "nchs_divorce_rate_per_1000_totalpop": nchs_d,
            "divorce_ratio": (
                psid["divorce_rate_per_1000"] / nchs_d if nchs_d else None
            ),
            "n_marriage_events": psid["n_marriage_events"],
            "n_divorce_events": psid["n_divorce_events"],
        }
    return {
        "nchs_reference_file": str(MD_PATH.relative_to(ROOT)),
        "nchs_latest_year": nchs["latest_year"],
        "nchs_citation": nchs["report"]["nvss_citation"],
        "nchs_reference_sha256": sha,
        "concept_delta_note": (
            "LOOSE order-of-magnitude anchor only. The NCHS rate is per "
            "1,000 TOTAL population (all ages); the PSID crude-equivalent "
            "is per 1,000 person-years age 15+ (marriageable), so PSID "
            "runs higher by construction (a smaller, older denominator). "
            "The 'all' window also pools reconstructed historical years "
            "(higher marriage rates) against the 2023 provisional rate. "
            "Ratios REPORTED with the concept delta stated, never tuned; "
            "the gate-2 external check on this pair is a shape/direction "
            "report, not a level gate."
        ),
        "windows": windows,
    }


# --------------------------------------------------------------------------
# Draft-thresholds note (pre-lock; feeds gates.yaml gate_2)
# --------------------------------------------------------------------------
def draft_thresholds_note(
    stability: dict[str, Any], asfr: dict[str, Any]
) -> str:
    gate_eligible = sorted(
        k for k, v in stability.items() if v["gate_eligible"]
    )
    report_only = sorted(
        k for k, v in stability.items() if not v["gate_eligible"]
    )
    recent = asfr["windows"]["recent"]["ratio_summary"]["median_ratio"]
    return (
        "DRAFT GATE-2 VALIDATION EVIDENCE -- PRE-LOCK, NOT RATIFIED.\n\n"
        "This is the pre-lock evidence package for the gate-2 family-"
        "transition statistics (the analogue of what preceded gate 1's "
        "lock). It changes no locked value and no model reads it. The "
        "DRAFT thresholds it feeds live in gates.yaml under gate_2 with "
        "locked: false and status: draft_pending_referee_round. "
        "Ratification requires the full lock ceremony: these floors -> "
        "thresholds with machine-bound derivations -> an adversarial "
        "referee round -> verification -> maintainer ratification by "
        f"merge. The k value ({DRAFT_K}) is a DRAFT proposal (consistent "
        "with the gate-1 c2st precedent k=4.2 and the 4-of-5-seed rule), "
        "for the ceremony to fix.\n\n"
        "STATISTIC. Per cell, the weighted PSID rate (first-marriage / "
        "divorce / widowhood / remarriage hazard; ever-married share; "
        "mean lifetime marriages; age-specific fertility rate; completed "
        "fertility) from the retrospective annual person-year panel, and a "
        "candidate's rate built the same way. The discrepancy is scored as "
        "the absolute log rate ratio |ln(r_candidate / r_PSID)| per cell "
        "(symmetric, scale-free).\n\n"
        "INTERNAL FLOOR. The person-disjoint 50/50 half-split "
        "|log rate ratio| floor in noise_floor_seeds_0_4 (mean/sd across "
        "seeds 0-4). Each DRAFT per-cell tolerance is "
        f"round(mean + {DRAFT_K}*sd, {DRAFT_ROUNDING}) at the shared "
        "derivation convention (tests/test_gates_derivations.py).\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY CELLS. Gate only cells defined on "
        "every seed AND carrying at least 20 events on the weaker half of "
        "the worst seed -- NCHS's own reliability floor (rates on < 20 "
        f"events are unreliable): {gate_eligible}. Report-only "
        f"(denominator-fragile, the analog of the PIA floor's d1/d2): "
        f"{report_only}. Derived from cell_stability, not hand-picked.\n\n"
        "EXTERNAL ANCHOR. NCHS 2024 ASFR (concept-aligned) and the NCHS "
        "national marriage/divorce crude rates (loose order-of-magnitude). "
        "PSID retrospective family files have known coverage caveats "
        "(left-truncated retrospective histories; a pooled-period vs "
        f"recent-year delta -- recent-window median PSID/NCHS ASFR ratio "
        f"~{recent:.2f}), so the external check is a SHAPE/RATIO REPORT, "
        "not a level gate: (i) the ASFR is a unimodal age hump in both; "
        "(ii) the marriage/divorce crude-equivalent is the same order of "
        "magnitude; (iii) every PSID/NCHS ratio is reported per band with "
        "its coverage/period delta named. A candidate that reproduces "
        "PSID's own rates inherits these documented offsets; the anchor "
        "certifies shape, not level.\n\n"
        "BASELINE CONVENTION. These transitions feed benefit levels "
        "through household composition and survivorship; a scored reform "
        "states its scheduled/payable baseline. This evidence fixes the "
        "component's estimation/validation standard only."
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
    drafts = draft_thresholds(noise_floor, stability)

    asfr = asfr_anchor(fert)
    md = marriage_divorce_anchor(panel)

    if verbose:
        n_gate = sum(v["gate_eligible"] for v in stability.values())
        print(
            f"cells: {len(cell_keys)} ({n_gate} gate-eligible, "
            f"{len(cell_keys) - n_gate} report-only); "
            f"draft thresholds: {len(drafts)}"
        )
        print(
            "recent ASFR median PSID/NCHS ratio: "
            f"{asfr['windows']['recent']['ratio_summary']['median_ratio']:.3f}"
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate2_floors_v1",
        "reported_anchor_not_gated": True,
        "component": (
            "family transitions (gate 2 pre-lock; marriage / divorce / "
            "widowhood / remarriage hazards + occupancy + fertility)"
        ),
        "purpose": (
            "Gate-2 pre-lock evidence: committed reference moments, the "
            "person-disjoint half-split noise floor (the sd basis the DRAFT "
            "gate-2 thresholds derive from), and honest external anchors "
            "for the family-transition statistics. Reads no gate and "
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
                "weighted moments (repo weight>0 convention)"
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
            "coverage_caveats": [
                "retrospective histories are LEFT-TRUNCATED: reconstructed "
                "young-adult person-years for a person first seen at an old "
                "age enter only if they survived and were sampled -- pooled "
                "deep-history young-age rates are selected (reported, not "
                "corrected)",
                "the 'all' external-anchor window pools all reconstructed "
                "calendar years against a single recent national year (a "
                "named period delta); a 'recent' window brackets it",
                "person-constant weight from the last positive PSID wave; "
                "retrospective person-years before that wave carry it",
            ],
        },
        "statistic_families": {
            "first_marriage_hazard": "by age band x sex",
            "divorce_hazard": "by marriage-duration band",
            "widowhood_incidence": "by age band x sex",
            "remarriage_hazard": "by years-since-dissolution band",
            "occupancy": (
                "share ever married by 40 / 60 x sex; mean lifetime "
                "marriages among ever-married x sex"
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
                "fraction=0.5, seeds 0-4) of every reference-moment cell; "
                "the floor statistic is |ln(rate_A / rate_B)| between two "
                "independent real halves -- the sd basis the DRAFT gate-2 "
                "thresholds derive from as round(mean + k*sd)"
            ),
            "seeds": list(SEEDS),
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
        },
        "noise_floor_seeds_0_4": noise_floor,
        "cell_stability": stability,
        "noise_floor_per_seed": per_seed,
        "draft_thresholds": {
            "k": DRAFT_K,
            "rounding": DRAFT_ROUNDING,
            "statistic": "log_ratio_abs_max",
            "note": (
                "DRAFT per-cell |ln ratio| tolerances = round(floor mean + "
                f"{DRAFT_K}*sd, {DRAFT_ROUNDING}); gate-eligible cells only. "
                "These are mirrored into gates.yaml gate_2 (locked: false) "
                "and machine-bound by tests/test_gates_derivations.py."
            ),
            "cells": drafts,
        },
        "draft_thresholds_note": draft_thresholds_note(stability, asfr),
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
