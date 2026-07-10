"""Build the gate-2b household-composition floors: moments + noise floor.

PRE-LOCK EVIDENCE, NOT A GATE RUN. The first step of the gate-2b lock
ceremony (``gates.yaml`` gate_2b, ``lock_ceremony.exists: false``,
``required_before_any_2b_pass``: *the SAME ceremony 2a followed -- a
pre-registered gate on a 100-seed split noise floor over MX23REL
household-composition transitions*). It is the exact analogue, one tranche
over, of what ``scripts/build_gate2_floors.py`` built for tranche 2a: the
committed reference moments, the person-disjoint 100-seed half-split noise
floor (the sd basis the DRAFT tolerances derive from), the coherent
option-(a) scoring protocol, the T_max = ln(1.5) power-cap partition with
pre-registered aggregate supersession, the training-copy disclosure and
the faithful-candidate operating characteristic.

It reads no gate and changes no gate. Unlike the 2a build it writes NO
``gates.yaml`` block: gate_2b already exists in ``gates.yaml`` as an
unlocked stub, and the lock-ceremony flip (proposal -> referee -> fixes
-> verification -> ratify) inserts the thresholds later. This artifact
only feeds that ceremony.

Household-composition statistic families, all from the MX23REL person-pair
relationship matrix joined to the demographic panel via
:mod:`populace_dynamics.data.household_composition`:

1. coresident-spouse share by age band x sex,
2. coresident-parent share by age band x sex (+ 45+ aggregate),
3. coresident-child share by age band x sex,
4. coresident-grandchild share by age band x sex (+ 55+ aggregate),
5. multigenerational-household share by age band x sex (+ 55+ aggregate),
6. household-size share distribution (person level).

Run from the repository root with the PSID products staged::

    .venv/bin/python scripts/build_gate2b_floors.py

It needs no populace-fit (real-vs-real only).
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

from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_floors_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_floors.v1"

#: Floor seeds: 100 person-disjoint splits stabilise the tolerance
#: estimator (tranche-2a round-1 finding 2), where 5 draws of a half-normal
#: realise anywhere from ~0.9 to ~5.6 sigma.
SEEDS = tuple(range(100))
#: The pinned gate-run seeds; a candidate is scored on these five with the
#: 4-of-5 conjunction, exactly as tranche 2a (its per-seed holdout
#: reference rates live in ``noise_floor_per_seed``).
GATE_SEEDS = (0, 1, 2, 3, 4)
#: Minimum events (weaker half, worst of the 100 seeds) for a cell to be
#: gate-eligible. 20 is NCHS's own reliability floor, inherited from 2a.
MIN_EVENTS_FOR_GATE = 20
#: DRAFT threshold multiplier: |ln ratio| tolerance = round(mean + k*sd).
#: ~4 sigma on the stabilised floor (the gate-1 c2st precedent k=4.2 + the
#: 4-of-5-seed rule). DRAFT -- the referee round sets the final k.
DRAFT_K = 4
DRAFT_ROUNDING = 3
#: Power cap (tranche-2a finding 3): a cell is gate-eligible only if its
#: stabilised tolerance is tight enough to reject a materially wrong model.
#: T_max = ln(1.5) -- a gated cell accepts at most a 1.5x rate error. Fixed
#: a priori; which cells the cap demotes is then derived from the floor.
T_MAX = math.log(1.5)
T_MAX_SOURCE = "ln(1.5)"

#: Pre-registered coverage-recovery aggregates (the household analogue of
#: tranche 2a's widowhood.45+/45-64), imported from the moment module so
#: the pooled cells and the supersession map cannot drift apart. Fixed a
#: priori (older-age tail of the moderate-rate families).
AGGREGATIONS: dict[str, list[str]] = hc.aggregation_members()


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------
def load_panel() -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Load MX23REL + demographics and build the household panel."""
    panel = hc.build_household_panel()
    pw = panel.person_waves
    data_meta = {
        "n_person_waves": int(len(pw)),
        "n_persons": int(pw["person_id"].nunique()),
        "wave_range": [int(pw["year"].min()), int(pw["year"].max())],
        "mean_hh_size_person_weighted": round(
            float((pw["hh_size"] * pw["weight"]).sum() / pw["weight"].sum()),
            4,
        ),
        "weighted_person_waves": round(float(pw["weight"].sum()), 1),
        "sex_counts": {
            str(k): int(v) for k, v in pw["sex"].value_counts().items()
        },
    }
    return panel, data_meta


# --------------------------------------------------------------------------
# Internal noise floor (person-disjoint 50/50 half-split, 100 seeds)
# --------------------------------------------------------------------------
def measure_seed_halfsplit(
    seed: int, panel: hc.HouseholdCompositionPanel
) -> dict[str, Any]:
    """One seed: split persons 50/50, all reference-moment cells per half.

    Returns per cell the two halves' rates, the minimum event count, the
    absolute log rate ratio ``|ln(r_A / r_B)|`` and the absolute percent
    gap -- undefined (``None``) when either half's rate is zero. Side A is
    the drawn (holdout) half, side B the train complement; under the gate
    protocol a candidate refit on side B is scored against side A's
    empirical rate, for which the symmetric half-vs-half ratio is the null.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(side_a.person_id)
    ids_b = set(side_b.person_id)
    cells_a = hc.reference_moments(panel, ids_a, weighted=True)
    cells_b = hc.reference_moments(panel, ids_b, weighted=True)

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
    is the root-mean-square of ``values`` -- the noise scale of the
    half-vs-half comparison. The stabilised tolerance then realises
    ``tolerance / realized_sigma`` sigma of measured noise.
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
    seed count and the minimum event count; the power-cap half of the
    gate-eligibility rule is applied in :func:`partition_cells`.
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
    """The gate-eligible / report-only partition (tranche-2a finding 3).

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
# Holdout-id commitment (per-seed reproducible splits)
# --------------------------------------------------------------------------
def holdout_id_commitment(
    panel: hc.HouseholdCompositionPanel,
) -> dict[str, Any]:
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
    the committed floor value for that seed. It passes at the noise floor,
    as ANY moment gate's copy must -- because a model reproducing the
    population moments should pass. The memorisation defence is procedural
    (registration + holdout exclusion + no_self_rescue), not the cell set.
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
            "1x tolerance), as any moment gate's copy must. The composition "
            "cells do not fix this -- a copy reproduces every moment; the "
            "memorisation defence is procedural (registration + holdout "
            "exclusion + no_self_rescue), NOT the cell set. What the cells "
            "DO catch is non-copy structural failure (a household model that "
            "misplaces who lives with whom by age, sex or generation)."
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
    candidate (tranche-2a finding 2, recomputed on the stabilised floor).

    Under option (a) a faithful candidate's per-cell score is distributed
    like the half-vs-half floor: half-normal with scale
    ``realized_sigma``. Per-cell pass probability is ``2*Phi(tol/sigma) -
    1``; a seed passes iff every gated cell holds (product, independence
    approximation); the gate passes iff >=4 of 5 seeds pass.
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
# External anchor: the honest gap (declared, not fabricated)
# --------------------------------------------------------------------------
def external_anchor_gap() -> dict[str, Any]:
    """Declare the external-anchor gap this pre-lock evidence leaves open.

    Tranche 2a shipped concept-decomposed NCHS anchors (ASFR,
    marriage/divorce). No concept-aligned external household-composition
    source is bundled here: the sole basis is the internal real-vs-real
    half-split floor. Naming the gap (rather than fabricating a level
    anchor) is the honest move; the referee round should add a
    concept-decomposed ACS/CPS anchor before ratification.
    """
    return {
        "status": "none_bundled",
        "reported_anchor_not_gated": True,
        "note": (
            "No external household-composition anchor is bundled with this "
            "pre-lock evidence. The gate-2b floor is measured entirely on "
            "the INTERNAL real-vs-real person-disjoint half-split; the "
            "reference LEVELS (coresidence and multigenerational shares, "
            "household-size distribution) are NOT yet cross-checked against "
            "an external national source. This is a KNOWN GAP relative to "
            "tranche 2a (which shipped concept-decomposed NCHS ASFR and "
            "marriage/divorce anchors). It is declared, not fabricated: a "
            "level anchor would carry a real concept delta (below), so the "
            "lock ceremony's referee round should add a concept-decomposed "
            "ACS/CPS anchor -- a SHAPE/RATIO report, not a level gate -- "
            "before ratification."
        ),
        "concept_delta": (
            "The PSID FAMILY UNIT is a wider, differently-bounded concept "
            "than the Census HOUSEHOLD (PSID splits subfamilies and applies "
            "its own cohabitor/roomer rules), so PSID household-size and "
            "coresidence shares are not level-comparable to ACS/CPS "
            "household tabulations without a concept decomposition -- the "
            "household analogue of the person-vs-couple x denominator "
            "decomposition tranche 2a applied to the crude marriage rate."
        ),
        "candidate_sources": [
            "ACS household-type / relationship-to-householder tabulations",
            "CPS ASEC family/subfamily and coresidence tables",
            "Census multigenerational-household estimates (B11017 / P2)",
            "ACS PUMS for age x sex x relationship coresidence shares",
        ],
    }


# --------------------------------------------------------------------------
# Draft-thresholds note (pre-lock; feeds the gate_2b ceremony, NOT gates.yaml)
# --------------------------------------------------------------------------
def draft_thresholds_note(
    gated: set[str], report: set[str], oc: dict[str, Any]
) -> str:
    return (
        "DRAFT GATE-2B VALIDATION EVIDENCE -- PRE-LOCK, NOT RATIFIED "
        "(v1, the first step of the gate-2b lock ceremony).\n\n"
        "Pre-lock evidence package for the gate-2b household-composition "
        "statistics (who lives with whom; gates.yaml gate_2b holdout_basis "
        "MX23REL), the exact analogue of what preceded tranche 2a's lock "
        "and one tranche over. It changes no locked value and no model "
        "reads it. Unlike the 2a build it writes NO gates.yaml block: "
        "gate_2b already exists as an unlocked stub, and the lock-ceremony "
        "flip (proposal -> referee -> fixes -> verification -> ratify) "
        "inserts the thresholds later. The k value "
        f"({DRAFT_K}) is a DRAFT proposal, for the ceremony to fix.\n\n"
        "PROTOCOL (option a -- the 2a mirror). Per gate seed s a candidate "
        "is refit on the seed-s TRAIN complement (side B; the seed-s "
        "holdout persons are excluded from all fitting), simulates the "
        "holdout persons' households (simulation seed s), and each cell is "
        "scored |ln(r_candidate,s / r_holdout,s)| against the seed-s "
        "HOLDOUT half's own empirical rate (side A, committed per seed in "
        "noise_floor_per_seed as rate_a). The symmetric half-vs-half floor "
        "|ln(rate_A/rate_B)| is then exactly the null. Seed s passes iff "
        "every gate-eligible cell holds at s; the gate passes iff >=4 of 5 "
        "seeds pass.\n\n"
        "INTERNAL FLOOR. The person-disjoint 50/50 half-split |log rate "
        "ratio| floor over 100 split seeds (noise_floor_seeds_0_99). Each "
        f"DRAFT per-cell tolerance is round(mean + {DRAFT_K}*sd, "
        f"{DRAFT_ROUNDING}); on 100 seeds the estimator is stable. The "
        "faithful-candidate operating characteristic recomputes "
        f"P(seed) = {oc['p_seed_pass']} and P(gate >=4/5) = "
        f"{oc['p_gate_pass_4_of_5']} over {oc['n_gated_cells']} gated cells "
        "(faithful_candidate_oc).\n\n"
        "POWER CAP. A cell is gate-eligible only if its stabilised "
        f"tolerance <= T_max = {T_MAX_SOURCE} (a gated cell accepts at most "
        "a 1.5x rate error) AND it carries >=20 events on the weaker half. "
        "The moderate-rate families' sparse older-age tail (coresident "
        "parent 45+, coresident grandchild 55+, multigen 55+) fails the cap "
        "per-band and is recovered by the pre-registered aggregates; the "
        "partition is derived from cell_stability, never hand-picked.\n\n"
        "COORDINATE SEMANTICS. ego_rel_to_alter (MX8) is EGO's relation to "
        "alter, so an ego coded 'child' HAS a coresident parent and an ego "
        "coded 'parent' HAS a coresident child; the person-wave age "
        "gradients in the committed floor are the check (coresident_parent "
        "peaks at 15-24, coresident_child at 25-44).\n\n"
        "EXTERNAL ANCHOR. NONE bundled (external_anchor.status = "
        "none_bundled) -- a KNOWN GAP vs 2a, declared not fabricated. The "
        "PSID family unit is a wider concept than the Census household, so "
        "a level anchor needs a concept decomposition; the referee round "
        "should add a concept-decomposed ACS/CPS shape/ratio report.\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY. Gated cells: "
        f"{sorted(gated)}. Report-only: {sorted(report)}.\n\n"
        "GOVERNANCE. One-shot outer runs, amendments only by public "
        "proposal + referee round, no_self_rescue and the version pin "
        "inherited from gate 1; the per-seed holdout ids are committed "
        "(holdout_ids); every gated statistic is weighted by that wave's "
        "cross-sectional PSID weight (no unweighted gated statistic).\n\n"
        "BASELINE CONVENTION. These household-composition statistics feed "
        "benefit levels through who is a spouse/dependent/coresident; this "
        "evidence fixes the component's estimation/validation standard "
        "only, not a scored reform."
    )


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
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

    panel, data_meta = load_panel()
    if verbose:
        print(
            f"panel: {data_meta['n_person_waves']} person-waves, "
            f"{data_meta['n_persons']} persons, "
            f"{data_meta['wave_range']} waves"
        )

    ref_w = hc.reference_moments(panel, weighted=True)
    ref_u = hc.reference_moments(panel, weighted=False)
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

    per_seed = [measure_seed_halfsplit(s, panel) for s in SEEDS]
    noise_floor, stability = pool_internal_floor(per_seed, cell_keys)
    tolerances = raw_tolerances(noise_floor)
    gated, report, reasons = partition_cells(stability, tolerances)
    drafts = draft_thresholds(noise_floor, tolerances, gated)
    annotate_stability(stability, noise_floor, tolerances, gated, reasons)

    holdout_ids = holdout_id_commitment(panel)
    copy_check = training_copy_check(per_seed, tolerances, gated)
    oc = faithful_candidate_oc(noise_floor, tolerances, gated)

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

    per_seed_stored = per_seed[: len(GATE_SEEDS)]

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate2b_floors_v1",
        "reported_anchor_not_gated": True,
        "component": (
            "household / relationship composition (gate 2b pre-lock; "
            "who-lives-with-whom coresidence + multigenerational + "
            "household-size shares from MX23REL)"
        ),
        "purpose": (
            "Gate-2b pre-lock evidence (v1): committed household-composition "
            "reference moments, the 100-seed person-disjoint half-split "
            "noise floor (the sd basis the DRAFT tolerances derive from), "
            "the coherent option-(a) scoring protocol, and the power-cap "
            "partition -- the first step of the gate_2b lock ceremony "
            "(gates.yaml gate_2b.lock_ceremony.required_before_any_2b_pass). "
            "Reads no gate and changes no gate on its own; writes NO "
            "gates.yaml block (the ceremony flip inserts the thresholds "
            "later). See draft_thresholds_note (NOT RATIFIED)."
        ),
        "holdout_basis": ["MX23REL"],
        "ceremony": {
            "tranche": "2b_relationship_household",
            "gates_yaml_stub": "gate_2b (status: unlocked, locked: false)",
            "step": (
                "1 of the lock ceremony: the 100-seed split noise floor "
                "over MX23REL household-composition transitions "
                "(proposal); referee -> fixes -> verification -> ratify "
                "follow, and only the ratifying flip touches gates.yaml"
            ),
            "gates_yaml_untouched": True,
            "mirrors_tranche_2a": "runs/gate2_floors_v2.json (PR #79)",
        },
        "data": {
            "relationship_matrix": (
                "populace_dynamics.data.relmap.relationship_map (MX23REL; "
                "1968-2023 Family Relationship Matrix, ordered within-family "
                "person pairs per wave). Household composition (who lives "
                "with whom) is read from ego_rel_to_alter (MX8, ego's "
                "relation to alter) and ego_rel_to_rp (MX7)."
            ),
            "demographics": (
                "populace_dynamics.data.panels.demographic_panel "
                "(ind2023er): each person-wave's contemporaneous age and "
                "cross-sectional individual weight, joined on "
                "(person_id, interview_year)"
            ),
            "sex": (
                "populace_dynamics.data.deaths.read_death_records "
                "(ind2023er ER32000), person-constant"
            ),
            "weights": (
                "each person-wave carries its WAVE cross-sectional "
                "individual weight (the contemporaneous weight for a "
                "cross-sectional composition statistic, unlike tranche 2a's "
                "person-constant most-recent weight for retrospective "
                "histories). Person-waves with no positive wave weight, no "
                "joined age, or na sex are dropped (repo weight>0 "
                "convention). Every gated statistic is weighted; the "
                "unweighted rate is reported alongside."
            ),
            **data_meta,
        },
        "panel_construction": {
            "unit": "person-wave (one row per enumerated person per wave)",
            "start_age": hc.START_AGE,
            "max_age": hc.MAX_AGE,
            "split_unit": "person_id (every wave of a person on one side)",
            "coresidence_semantics": (
                "ego_rel_to_alter (MX8) is EGO's relation to alter: ego "
                "coded child (30/33/35/38) -> coresident_parent; ego coded "
                "parent (50/53/55/56) -> coresident_child; ego coded "
                "spouse/partner (20/22) -> coresident_spouse; ego coded "
                "grandparent (66/67/68/69/82/87/88) -> coresident_grandchild"
            ),
            "multigen_rule": (
                "3+ lineal generations in the family unit (generation span "
                ">= 2 over LINEAL kin only; collateral relatives excluded). "
                "The pre-1983 abbreviated relmap frame has no grandparent "
                "code, so a household whose oldest member is the RP's "
                "grandparent is under-detected there (documented era "
                "asymmetry)."
            ),
            "coverage_caveats": [
                "the scored unit is the PERSON-WAVE, correlated within "
                "person; the person-disjoint split keeps a person's waves "
                "together so the floor reflects person-level resampling, but "
                "n_events counts person-waves, not independent persons "
                "(same property as tranche 2a's person-years)",
                "the PSID FAMILY UNIT is not the Census HOUSEHOLD; "
                "household-size and coresidence shares carry that concept "
                "delta (see external_anchor.concept_delta)",
                "person-waves with no positive wave weight (~17%: pre-1997 "
                "Latino-oversample integration years, institutionalised, "
                "zero-weight movers) are dropped from the weighted moments",
            ],
        },
        "statistic_families": {
            "coresident_spouse": "coresident spouse/partner share by age "
            "band x sex",
            "coresident_parent": "share living with a parent by age band x "
            "sex (+ 45+ aggregate)",
            "coresident_child": "share living with a child by age band x sex",
            "coresident_grandchild": "share living with a grandchild by age "
            "band x sex (+ 55+ aggregate)",
            "multigen": "3+-generation household share by age band x sex "
            "(+ 55+ aggregate)",
            "hh_size": "person-level household-size share distribution "
            "(1..4 and open 5+)",
        },
        "reference_moments": reference_moments,
        "external_anchor": external_anchor_gap(),
        "internal_noise_floor": {
            "method": (
                "person-disjoint 50/50 half-split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction=0.5, seeds 0-99) of every reference-moment cell; "
                "the floor statistic is |ln(rate_A / rate_B)| between two "
                "independent real halves -- the sd basis the DRAFT gate-2b "
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
                "complement (side B), simulate the seed-s holdout persons' "
                "households (side A, simulation seed s), score each cell "
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
                "each cell's rate from its simulated households of the "
                "seed-s holdout persons, same person-wave construction, "
                "same wave weight"
            ),
            "scored_against": (
                "the seed-s holdout half's empirical rate (rate_a in "
                "noise_floor_per_seed)"
            ),
            "conjunction": "all gated cells per seed AND >=4 of 5 gate seeds",
        },
        "aggregations": {
            agg: {"members": list(members), "gated": agg in gated}
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
                f"{DRAFT_K}*sd, {DRAFT_ROUNDING}); gated cells only (defined "
                "on all seeds AND >=20 events AND tolerance <= T_max AND not "
                "superseded by an aggregate). NOT yet mirrored into "
                "gates.yaml -- the gate_2b lock-ceremony flip inserts them "
                "after the referee + verification rounds."
            ),
            "cells": drafts,
        },
        "draft_thresholds_note": draft_thresholds_note(gated, report, oc),
        "revision_pins": {
            "populace_dynamics_sha": _git_sha(ROOT),
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
