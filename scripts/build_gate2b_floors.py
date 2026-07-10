"""Build the gate-2b household-composition floors: moments + noise floor.

PRE-LOCK EVIDENCE, NOT A GATE RUN. The first step of the gate-2b lock
ceremony (``gates.yaml`` gate_2b, ``lock_ceremony.exists: false``,
``required_before_any_2b_pass``: *the SAME ceremony 2a followed -- a
pre-registered gate on a 100-seed split noise floor over MX23REL
household-composition transitions*). It is the exact analogue, one tranche
over, of what ``scripts/build_gate2_floors.py`` built for tranche 2a: the
committed reference moments, the person-disjoint 100-seed half-split noise
floor (the sd basis the DRAFT tolerances derive from), the ratified 2a
mean-over-K=20-draws scoring protocol (amendment 1), the T_max = ln(1.5)
power-cap partition with pre-registered aggregate supersession, the
training-copy disclosure and the faithful-candidate operating
characteristic.

It reads no gate and changes no gate. Unlike the 2a build it writes NO
``gates.yaml`` block: gate_2b already exists in ``gates.yaml`` as an
unlocked stub, and the lock-ceremony flip (proposal -> referee -> fixes
-> verification -> ratify) inserts the thresholds later. This artifact
only feeds that ceremony.

Household-composition statistic families, all from the MX23REL person-pair
relationship matrix joined to the demographic panel via
:mod:`populace_dynamics.data.household_composition`:

STOCKS (point-in-time shares):
1. coresident-spouse share by age band x sex,
2. coresident-parent (lives with a parent) share by band x sex (+ 45+ agg),
3. coresident-child (lives with a child) share by band x sex,
4. coresident-grandchild share by band x sex (+ 55+ agg),
5. multigenerational (>=3 distinct generations, B11017) by band x sex
   (+ 65+ agg),
6. household-size share distribution (person level).

TRANSITIONS (wave-to-wave, the tranche name):
7. parental-home exit 15-34 by sex,
8. spousal-coresidence loss 55+ by sex,
9. multigenerational entry / exit (pooled).

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

from populace_dynamics.data import deaths, panels, relmap
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

#: Candidate-scoring estimator (tranche 2a amendment 1, ratified
#: 2026-07-08): the candidate cell rate is the MEAN over K pre-registered
#: simulation draws (numpy default_rng(5200 + k)), a stream DISTINCT from
#: the split seeds, scored once as |ln(rbar / rate_a)| (NOT the mean of the
#: per-draw scores). The floor and the tolerances are draw-noise-free; the
#: K-draw mean shares that basis, so the faithful OC is achievable. This
#: floor build measures the real-vs-real null (no simulation); the K=20
#: estimator is the contract a future candidate run scores under.
CANDIDATE_DRAWS = 20
CANDIDATE_DRAW_STREAM = "numpy.random.default_rng(5200 + k), k=0..K-1"

#: Report-only era windows for the pooled-cross-section mixture disclosure.
ERA_WINDOWS: tuple[tuple[int, int, str], ...] = (
    (1969, 1996, "1969-1996"),
    (1997, 2009, "1997-2009"),
    (2010, 2023, "2010-2023"),
)

#: Pre-registered coverage-recovery aggregates (the household analogue of
#: tranche 2a's widowhood.45+/45-64), imported from the moment module so
#: the pooled cells and the supersession map cannot drift apart. Fixed a
#: priori BEFORE the corrected-multigen rebuild (referee round-1 finding
#: 6b: the multigen pool is 65+ so multigen.55-64 can gate standalone).
AGGREGATIONS: dict[str, list[str]] = hc.aggregation_members()


# --------------------------------------------------------------------------
# Panel assembly + estimand disclosure
# --------------------------------------------------------------------------
def _era_weight_shares(pw: Any) -> list[dict[str, Any]]:
    """Share of total pooled weight in each calendar era (fix D)."""
    total = float(pw["weight"].sum())
    out = []
    for lo, hi, label in (
        (1969, 1979, "1969-1979"),
        (1980, 1989, "1980-1989"),
        (1990, 1996, "1990-1996"),
        (1997, 1999, "1997-1999"),
        (2000, 2009, "2000-2009"),
        (2010, 2023, "2010-2023"),
    ):
        w = float(pw[(pw["year"] >= lo) & (pw["year"] <= hi)]["weight"].sum())
        out.append(
            {
                "era": label,
                "weight_share_pct": round(100.0 * w / total, 2),
            }
        )
    return out


def _era_slices(pw: Any) -> dict[str, Any]:
    """Per-era pooled family rates (REPORT-ONLY): expose the certified
    1997-2023 mixture run over run (fix D). Not floored, not gated."""
    families = list(hc.CORESIDENCE_LINKS) + ["multigen"]
    slices: dict[str, Any] = {}
    for lo, hi, label in ERA_WINDOWS:
        sub = pw[(pw["year"] >= lo) & (pw["year"] <= hi)]
        w = float(sub["weight"].sum())
        rates = {}
        for flag in families:
            num = float(sub.loc[sub[flag], "weight"].sum())
            rates[flag] = round(num / w, 4) if w > 0 else None
        rates["mean_hh_size"] = (
            round(float((sub["hh_size"] * sub["weight"]).sum() / w), 4)
            if w > 0
            else None
        )
        slices[label] = {
            "n_person_waves": int(len(sub)),
            "weight_share_pct": round(
                100.0 * w / float(pw["weight"].sum()), 2
            ),
            "pooled_rates": rates,
        }
    slices["note"] = (
        "REPORT-ONLY, never gated. Per-era pooled weighted family rates so "
        "the certified pooled estimand (effectively 1997-2023) is visible: "
        "a candidate matching only the most recent era can be read against "
        "the pooled cell it is actually scored on."
    )
    return slices


def characterize_zero_weight_drop(
    roster: Any, demo: Any, sex: Any
) -> dict[str, Any]:
    """Characterize the weight>0 drop (referee finding 6a) in the artifact.

    Re-joins the roster to demographics WITHOUT the weight filter, then
    compares the dropped (zero/absent wave-weight) person-waves to the kept
    ones: their era concentration and their (unweighted) composition.
    """
    demo_j = demo[["person_id", "period", "age", "weight"]].rename(
        columns={"period": "interview_year"}
    )
    merged = roster.merge(
        demo_j, on=["person_id", "interview_year"], how="left"
    ).merge(sex[["person_id", "sex"]], on="person_id", how="left")
    base = merged[merged["age"].notna() & merged["sex"].isin(hc.SEXES)].copy()
    base["age"] = base["age"].astype(int)
    base = base[(base["age"] >= hc.START_AGE) & (base["age"] <= hc.MAX_AGE)]
    positive = base["weight"].fillna(0) > 0
    kept = base[positive]
    dropped = base[~positive]
    n_drop = int(len(dropped))
    in_9096 = int(
        len(
            dropped[
                (dropped["interview_year"] >= 1990)
                & (dropped["interview_year"] <= 1996)
            ]
        )
    )
    return {
        "base_person_waves": int(len(base)),
        "dropped_person_waves": n_drop,
        "dropped_share_pct": round(100.0 * n_drop / len(base), 2),
        "dropped_share_in_1990_1996_pct": (
            round(100.0 * in_9096 / n_drop, 1) if n_drop else None
        ),
        "composition_unweighted_kept_vs_dropped": {
            flag: {
                "kept": round(float(kept[flag].mean()), 3),
                "dropped": round(float(dropped[flag].mean()), 3),
            }
            for flag in ("coresident_spouse", "coresident_parent")
        },
        "note": (
            "The weight>0 drop is not composition-neutral: dropped "
            "person-waves (54% in the 1990-1996 Latino-oversample "
            "integration window) are less likely to have a coresident "
            "spouse and more likely to live with a parent than kept ones. "
            "This is fine for the weighted estimand (a zero-weight case "
            "represents nobody) but is disclosed here, not just caveated."
        ),
    }


def load_panel() -> (
    tuple[hc.HouseholdCompositionPanel, dict[str, Any], dict[str, Any]]
):
    """Load MX23REL + demographics, build the panel, characterize the drop."""
    rel_map = relmap.relationship_map()
    demo = panels.demographic_panel()
    sex = deaths.read_death_records()
    roster = hc.household_roster(rel_map)
    person_waves = hc.join_demographics(roster, demo, sex)
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    panel = hc.HouseholdCompositionPanel(
        person_waves=person_waves, attrs=attrs
    )
    pw = person_waves
    post97 = float(pw[pw["year"] >= 1997]["weight"].sum())
    data_meta = {
        "n_person_waves": int(len(pw)),
        "n_persons": int(pw["person_id"].nunique()),
        "wave_span": [int(pw["year"].min()), int(pw["year"].max())],
        "post_1997_weight_share_pct": round(
            100.0 * post97 / float(pw["weight"].sum()), 2
        ),
        "mean_hh_size_person_weighted": round(
            float((pw["hh_size"] * pw["weight"]).sum() / pw["weight"].sum()),
            4,
        ),
        "weighted_person_waves": round(float(pw["weight"].sum()), 1),
        "sex_counts": {
            str(k): int(v) for k, v in pw["sex"].value_counts().items()
        },
        "era_weight_shares": _era_weight_shares(pw),
    }
    drop = characterize_zero_weight_drop(roster, demo, sex)
    return panel, data_meta, drop


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
            "misplaces who lives with whom by age, sex or generation, or "
            "the wave-to-wave transition rates between those states)."
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
            "cells; gate = P(>=4 of 5 seeds) = p^5 + 5 p^4 (1-p). The K=20 "
            "estimator shares this draw-noise-free basis (see "
            "protocol.basis_note)."
        ),
        "n_gated_cells": len(gated),
        "p_seed_pass": round(p_seed, 4),
        "p_gate_pass_4_of_5": round(p_gate, 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# Protocol: the ratified 2a K=20 estimator (fix A)
# --------------------------------------------------------------------------
def _basis_note(oc: dict[str, Any]) -> str:
    return (
        "The faithful-candidate OC (p_seed_pass "
        f"{oc['p_seed_pass']}, p_gate_pass_4_of_5 "
        f"{oc['p_gate_pass_4_of_5']}) is computed on a DRAW-NOISE-FREE "
        "half-normal basis: a faithful candidate's per-cell score ~ "
        "half-normal(realized_sigma) -- floor sigmas only, no "
        "simulation-draw term. The mean-over-K=20-draws estimator SHARES "
        "that basis (the 20-draw mean rate approximates the draw-noise-free "
        "expected rate, shrinking the residual simulation term ~1/sqrt(K)), "
        "so the OC is achievable. Under the SUPERSEDED single-draw "
        "estimator it was UNACHIEVABLE (tranche 2a amendment-1 finding 1): "
        "one simulation replicate injects ~one extra floor-sigma per cell "
        "(score ~sqrt(1.5) x realized_sigma), dropping a faithful candidate "
        "to roughly P(seed) 0.67 / P(gate>=4/5) 0.46 on this 46-gated-cell "
        "floor -- a coin flip, not the 3.2-sigma design. Numbers unchanged; "
        "the estimator now shares the OC's derivation basis."
    )


def _fresh_run_artifact_schema(n_gated: int) -> dict[str, Any]:
    """The candidate-run contract under the K=20 estimator (2a amendment 1),
    inherited here so a future gate_2b candidate run is auditable."""
    return {
        "applies_to": (
            "a fresh candidate one-shot run registered AFTER a future "
            "gate_2b lock; NEVER this floor build, which measures the "
            "real-vs-real half-vs-half null (no simulation, no draws)."
        ),
        "per_draw_per_cell_rates": {
            "required": True,
            "shape": [CANDIDATE_DRAWS, n_gated, len(GATE_SEEDS)],
            "shape_dims": "K_draws x gated_cells x gate_seeds",
            "rule": (
                "commit every draw's per-cell rate r[k, cell, s] for all "
                "K=20 draws (numpy default_rng(5200 + k), k=0..19), all "
                f"{n_gated} gated cells, and all 5 gate seeds -- NOT only "
                "the 20-draw means. So rbar_candidate,s = mean over k "
                "recomputes cell-by-cell and |ln(rbar / rate_a)| is "
                "independently auditable, not taken on trust."
            ),
        },
        "undefined_draw_rule": {
            "required": True,
            "pre_specified": True,
            "rule": (
                "if any gated cell's rate is UNDEFINED on any draw (empty "
                "simulated denominator), the RUN IS INVALIDATED and must be "
                "re-registered and re-run. No draw may be dropped, skipped, "
                "substituted, or re-rolled; rbar is always the mean over "
                "ALL K registered draws."
            ),
            "rationale": (
                "silently dropping an undefined draw reintroduces the "
                "post-hoc draw selection the mean-over-all-K rule "
                "forecloses; a smaller outcome-selected set is not the "
                "pre-registered mean-of-20."
            ),
        },
        "per_draw_dispersion_disclosure": {
            "required": True,
            "gated": False,
            "report_only": True,
            "commit": {
                "per_cell_per_draw_sd": (
                    "the sd across the K=20 draws of each gated cell's rate, "
                    "per gated cell per gate seed -- the per-draw "
                    "over-dispersion the 20-draw mean hides."
                ),
                "max_per_draw_abs_ln_per_cell": (
                    "the maximum over the K=20 draws of |ln(r[k, cell, s] / "
                    "rate_a,s)| per gated cell per gate seed -- the worst "
                    "single-draw excursion, reported alongside rbar."
                ),
            },
            "note": (
                "REPORT-ONLY: no dispersion cap gates the run; the "
                "disclosure lets a referee see whether a passing 20-draw "
                "mean conceals a wild individual draw."
            ),
        },
    }


def protocol_block(oc: dict[str, Any], n_gated: int) -> dict[str, Any]:
    """The scoring protocol: the ratified 2a mean-over-K=20-draws estimator
    (amendment 1, 2026-07-08), replacing the superseded single-draw text."""
    return {
        "option": "a",
        "estimator": (
            "mean-over-K=20-draws (tranche 2a amendment 1, ratified "
            "2026-07-08, PR 96). The candidate cell rate rbar_candidate,s "
            "is the MEAN over K=20 pre-registered simulation draws (numpy "
            "default_rng(5200 + k), k=0..19; a stream DISTINCT from the "
            "split seeds), scored ONCE per cell as |ln(rbar_candidate,s / "
            "rate_a,s)| -- NOT the mean of the per-draw |ln| scores. This "
            "2b build re-uses the ratified 2a estimator verbatim, replacing "
            "the superseded single-draw replicate (finding 1)."
        ),
        "candidate_draws": CANDIDATE_DRAWS,
        "candidate_draw_stream": CANDIDATE_DRAW_STREAM,
        "statistic": (
            "|ln(rbar_candidate,s / rate_a,s)|, rbar = mean over K=20 "
            "draws of the simulated cell rate; symmetric and scale-free"
        ),
        "description": (
            "Per gate seed s: refit the candidate on the seed-s train "
            "complement (side B; the seed-s holdout persons excluded from "
            "all fitting), SIMULATE side A's persons' households at K=20 "
            "pre-registered draws (default_rng(5200 + k)), form the 20-draw "
            "mean cell rate rbar_candidate,s, and score |ln(rbar / rate_a)| "
            "against side A's own empirical rate. The real-vs-real "
            "half-vs-half floor |ln(rate_A/rate_B)| is the null. Seed s "
            "passes iff every gated cell holds; the gate passes iff >=4 of "
            "5 gate seeds pass."
        ),
        "varies_per_seed": (
            "the 50/50 person split (holdout = side A of "
            "split_panel_by_person(seed=s)); the candidate's fit complement "
            "and its K=20 simulated draws of the holdout"
        ),
        "scored_against": (
            "the seed-s holdout half's empirical rate (rate_a in "
            "noise_floor_per_seed)"
        ),
        "conjunction": "all gated cells per seed AND >=4 of 5 gate seeds",
        "amended": (
            "the estimator is tranche 2a amendment 1 (amendment_history "
            "entry 1, ratified 2026-07-08 by merge of PR 96): the candidate "
            "statistic moved from one frozen simulation replicate to the "
            "mean over K=20 draws at BYTE-IDENTICAL tolerances. This 2b "
            "floor inherits it so the disclosed OC is achievable."
        ),
        "basis_note": _basis_note(oc),
        "fresh_run_artifact_schema": _fresh_run_artifact_schema(n_gated),
    }


# --------------------------------------------------------------------------
# External anchor: the honest gap (declared, not fabricated) -- fix F
# --------------------------------------------------------------------------
def external_anchor_gap() -> dict[str, Any]:
    """Declare the external-anchor gap and the pre-ratification commitment.

    Tranche 2a shipped concept-decomposed NCHS anchors (ASFR,
    marriage/divorce) IN its floor PR. No concept-aligned external
    household-composition source is bundled here: the sole basis is the
    internal real-vs-real half-split floor. Naming the gap (rather than
    fabricating a level anchor) is the honest move. Per referee finding 5,
    the anchor MUST be bundled before the ratifying flip -- a later
    ceremony step than this floor -- and its concept bridge is now
    buildable because fixes B/C make ``multigen`` the clean Census B11017
    concept.
    """
    return {
        "status": "none_bundled",
        "reported_anchor_not_gated": True,
        "required_before_ratifying_flip": True,
        "note": (
            "No external household-composition anchor is bundled with this "
            "STEP-1 floor. The reference LEVELS (coresidence, "
            "multigenerational, household-size, and transition rates) are "
            "cross-checked only against themselves (real-vs-real "
            "half-split). This is a KNOWN GAP vs tranche 2a (which shipped "
            "concept-decomposed NCHS anchors in its floor PR). It is "
            "declared, not fabricated. Referee finding 5 (round 1) requires "
            "the concept-decomposed ACS/CPS shape/ratio report -- "
            "sha-pinned like 2a's NCHS files, reported-not-gated -- to "
            "PRECEDE the ratifying flip (a later ceremony step than this "
            "floor). It is now buildable: fixes B/C make multigen the "
            "Census B11017 concept, so a B11017 comparison no longer embeds "
            "the 14.7% spurious-cohabitor / 30.8% skipped-generation "
            "components a span>=2 flag carried."
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
        "multigen_bridge": (
            "multigen is now >=3 distinct lineal generations present "
            "(Census B11017), so it maps to ACS table B11017 directly; the "
            "committed span>=2 predecessor would have embedded a 14.7% "
            "first-year-cohabitor artifact (finding 2) and 30.8% "
            "skipped-generation households B11017 excludes."
        ),
        "candidate_sources": [
            "Census/ACS table B11017 (multigenerational households)",
            "ACS S1101 / B11016 (household size and type)",
            "CPS ASEC family/subfamily and coresidence tables",
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
        "(v1, the first step of the gate-2b lock ceremony; referee round-1 "
        "fixes A-H applied).\n\n"
        "Pre-lock evidence package for the gate-2b household-composition "
        "statistics (who lives with whom, and how it changes wave to wave; "
        "gates.yaml gate_2b holdout_basis MX23REL), the analogue of what "
        "preceded tranche 2a's lock, one tranche over. It changes no locked "
        "value and no model reads it. Unlike the 2a build it writes NO "
        "gates.yaml block: gate_2b already exists as an unlocked stub, and "
        "the lock-ceremony flip (proposal -> referee -> fixes -> "
        "verification -> ratify) inserts the thresholds later. The k value "
        f"({DRAFT_K}) is a DRAFT proposal, for the ceremony to fix.\n\n"
        "PROTOCOL (option a, the ratified 2a estimator -- finding 1 fix). "
        "Per gate seed s a candidate is refit on the seed-s TRAIN "
        "complement (side B; the seed-s holdout persons excluded from all "
        "fitting), SIMULATES the holdout persons' households at K=20 "
        "pre-registered draws (numpy default_rng(5200 + k), a stream "
        "distinct from the split seeds), and each cell is scored ONCE as "
        "|ln(rbar_candidate,s / rate_a,s)| where rbar is the mean over the "
        "20 draws of the cell RATE (NOT the mean of the per-draw |ln| "
        "scores), against the seed-s HOLDOUT half's own empirical rate "
        "(rate_a). The symmetric half-vs-half floor is the null. Seed "
        "passes iff every gate-eligible cell holds; the gate passes iff "
        ">=4 of 5 seeds pass. The single-draw text the round-1 referee "
        "flagged is replaced by this ratified mean-over-K=20 estimator; see "
        "protocol.basis_note.\n\n"
        "INTERNAL FLOOR. The person-disjoint 50/50 half-split |log rate "
        "ratio| floor over 100 split seeds (noise_floor_seeds_0_99). Each "
        f"DRAFT per-cell tolerance is round(mean + {DRAFT_K}*sd, "
        f"{DRAFT_ROUNDING}); on 100 seeds the estimator is stable. The "
        "faithful-candidate operating characteristic recomputes "
        f"P(seed) = {oc['p_seed_pass']} and P(gate >=4/5) = "
        f"{oc['p_gate_pass_4_of_5']} over {oc['n_gated_cells']} gated cells "
        "(faithful_candidate_oc), draw-noise-free, achievable under the "
        "K=20 mean.\n\n"
        "POWER CAP. A cell is gate-eligible only if its stabilised "
        f"tolerance <= T_max = {T_MAX_SOURCE} (a gated cell accepts at most "
        "a 1.5x rate error) AND it carries >=20 events on the weaker half. "
        "The moderate-rate families' sparse older-age tail (coresident "
        "parent 45+, coresident grandchild 55+, multigen 65+) fails the cap "
        "per-band and is recovered by the pre-registered aggregates; the "
        "partition is derived from cell_stability, never hand-picked.\n\n"
        "ESTIMAND (finding 3 fix). The pooled weighted moment is "
        "wave-weighted PSID cross-sections, EFFECTIVELY 1997-2023: raw PSID "
        "weights are population-scaled only from 1997, so 1997+ carries "
        "99.81% of the pooled weight and 1969-1996 carries 0.19%. It is "
        "NOT a 1969-2023 average; named per "
        "description_claims_exactly_the_scored_surface. See "
        "era_weight_shares and the report-only era_slices.\n\n"
        "TRANSITIONS (finding 4 fix). The tranche name is 'transitions'; "
        "wave-to-wave families (parental-home exit, spousal-coresidence "
        "loss, multigen entry/exit) are priced by the identical 100-seed "
        "machinery, linking a person-wave to the next observed wave at a "
        "1-2 year gap (biennial post-1997), gated where the cap admits.\n\n"
        "SEMANTICS (finding 2 fix). Coresidence reads ego_rel_to_alter "
        "(MX8; ego coded child -> coresident parent) and multigen reads "
        "ego_rel_to_rp (MX7, a DIFFERENT frame: code 88 is "
        "first_year_cohabitor / gen 0, not the MX8 step-great-grandparent). "
        "multigen is >=3 distinct generations (Census B11017), not span>=2 "
        "-- the corrected map drops the 14.7% spurious-cohabitor and 30.8% "
        "skipped-generation flags.\n\n"
        "EXTERNAL ANCHOR. NONE bundled (external_anchor.status = "
        "none_bundled) -- a KNOWN GAP vs 2a; declared not fabricated, and "
        "REQUIRED before the ratifying flip (finding 5), now buildable "
        "against Census B11017.\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY. Gated cells: "
        f"{sorted(gated)}. Report-only: {sorted(report)}.\n\n"
        "GOVERNANCE. One-shot outer runs, amendments only by public "
        "proposal + referee round, no_self_rescue and the version pin "
        "inherited from gate 1; the per-seed holdout ids are committed "
        "(holdout_ids); every gated statistic is weighted by that wave's "
        "individual weight (no unweighted gated statistic). BASELINE: these "
        "statistics feed benefit levels through who is a "
        "spouse/dependent/coresident; this evidence fixes the component's "
        "estimation/validation standard only, not a scored reform."
    )


# --------------------------------------------------------------------------
# Provenance
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


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    assert "populace.fit" not in sys.modules

    panel, data_meta, zero_weight_drop = load_panel()
    if verbose:
        print(
            f"panel: {data_meta['n_person_waves']} person-waves, "
            f"{data_meta['n_persons']} persons, "
            f"{data_meta['post_1997_weight_share_pct']}% post-1997 weight"
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
        "referee_round": "PR #118 round-1 (fixes A-H applied)",
        "component": (
            "household / relationship composition (gate 2b pre-lock; "
            "who-lives-with-whom coresidence + multigenerational + "
            "household-size STOCKS and wave-to-wave TRANSITIONS from "
            "MX23REL)"
        ),
        "purpose": (
            "Gate-2b pre-lock evidence (v1, round-1 fixes A-H): committed "
            "household-composition reference moments (stocks + "
            "transitions), the 100-seed person-disjoint half-split noise "
            "floor (the sd basis the DRAFT tolerances derive from), the "
            "ratified 2a mean-over-K=20-draws scoring protocol, and the "
            "power-cap partition -- the first step of the gate_2b lock "
            "ceremony (gates.yaml gate_2b.lock_ceremony."
            "required_before_any_2b_pass). Reads no gate and changes no "
            "gate on its own; writes NO gates.yaml block (the ceremony flip "
            "inserts the thresholds later). See draft_thresholds_note (NOT "
            "RATIFIED)."
        ),
        "holdout_basis": ["MX23REL"],
        "ceremony": {
            "tranche": "2b_relationship_household",
            "gates_yaml_stub": "gate_2b (status: unlocked, locked: false)",
            "step": (
                "1 of the lock ceremony: the 100-seed split noise floor "
                "over MX23REL household-composition stocks + transitions "
                "(proposal); referee -> fixes -> verification -> ratify "
                "follow, and only the ratifying flip touches gates.yaml"
            ),
            "gates_yaml_untouched": True,
            "mirrors_tranche_2a": "runs/gate2_floors_v2.json (PR #79)",
            "round_1_fixes": (
                "A ratified K=20 estimator; B/C corrected multigen "
                "(MX7 code 88 -> gen 0; >=3 distinct generations); D "
                "estimand named effectively-1997-2023 + era disclosure; E "
                "transition families added; F anchor required-before-flip; "
                "G multigen aggregate re-scoped to 65+ + drop "
                "characterized; H record + pin corrections"
            ),
        },
        "data": {
            "relationship_matrix": (
                "populace_dynamics.data.relmap.relationship_map (MX23REL; "
                "1968-2023 Family Relationship Matrix, ordered within-family "
                "person pairs per wave). Coresidence reads ego_rel_to_alter "
                "(MX8); generation (multigen) reads ego_rel_to_rp (MX7) -- "
                "two DIFFERENT frames."
            ),
            "demographics": (
                "populace_dynamics.data.panels.demographic_panel "
                "(ind2023er): each person-wave's contemporaneous age and "
                "individual weight, joined on (person_id, interview_year)"
            ),
            "sex": (
                "populace_dynamics.data.deaths.read_death_records "
                "(ind2023er ER32000), person-constant"
            ),
            "weights": (
                "each person-wave carries its WAVE individual weight "
                "(cross-sectional INDIVIDUAL WEIGHT for most waves; for "
                "1993-1996 the resolved series is CORE INDIVIDUAL "
                "LONGITUDINAL WEIGHT, ~0.05% of pooled weight -- corrected "
                "from the round-1 'cross-sectional' claim). Raw PSID weights "
                "are population-scaled only from 1997, so the pooled "
                "weighted estimand is effectively 1997-2023 (see estimand "
                "and era_weight_shares). Person-waves with no positive wave "
                "weight, no joined age, or na sex are dropped (see "
                "zero_weight_drop). Every gated statistic is weighted; the "
                "unweighted rate is reported alongside."
            ),
            "estimand": (
                "wave-weighted pooled PSID cross-sections, EFFECTIVELY "
                "1997-2023: 1997+ carries 99.81% of the pooled weight, "
                "1969-1996 carries 0.19%. The pooled weighted moment is a "
                "recent-cross-section estimand, NOT a 1969-2023 average -- "
                "named per governance.amendment_rules."
                "description_claims_exactly_the_scored_surface (finding 3)."
            ),
            **data_meta,
        },
        "zero_weight_drop": zero_weight_drop,
        "era_slices": _era_slices(panel.person_waves),
        "panel_construction": {
            "unit": "person-wave (one row per enumerated person per wave)",
            "start_age": hc.START_AGE,
            "max_age": hc.MAX_AGE,
            "split_unit": (
                "person_id (every wave AND transition of a person on one "
                "side of the half-split)"
            ),
            "coresidence_semantics": (
                "ego_rel_to_alter (MX8) is EGO's relation to alter: ego "
                "coded child (30/33/35/38) -> coresident_parent; parent "
                "(50/53/55/56) -> coresident_child; spouse/partner (20/22) "
                "-> coresident_spouse; grandparent (66/68/82/87/88) -> "
                "coresident_grandchild. Each set is own + step + social + "
                "foster kin and EXCLUDES in-law-by-marriage links "
                "(37/57/67/69) -- one inclusion rule across the four "
                "families (finding 8 iv)."
            ),
            "multigen_rule": (
                "THREE OR MORE distinct lineal generations present in the "
                "family unit (Census B11017), counted over lineal kin only "
                "(collateral relatives excluded). Skipped-generation "
                "households (grandparent + grandchild, no middle) are TWO "
                "generations and are NOT multigenerational. Generation "
                "offsets read ego_rel_to_rp (MX7), a DIFFERENT frame from "
                "MX8: code 88 is first_year_cohabitor (gen 0), NOT the MX8 "
                "'step great-grandparent' (finding 2). Pre-1983 the "
                "abbreviated MX7 frame has no grandparent code (documented "
                "era asymmetry; ~99.8% of weight is post-1997)."
            ),
            "transition_convention": (
                "a transition links a person-wave to the NEXT observed wave "
                "at a 1-2 year gap (MAX_TRANSITION_GAP_YEARS=2: the biennial "
                "post-1997 cadence, 1-year pre-1997); attrition gaps > 2 "
                "years are not linked. The transition rate is per "
                "wave-interval (NOT annualised); ~99.8% of weight is "
                "post-1997, so it is effectively a biennial rate."
            ),
            "coverage_caveats": [
                "the scored unit is the PERSON-WAVE, correlated within "
                "person; the person-disjoint split keeps a person's waves "
                "(and transitions) together so the floor reflects "
                "person-level resampling (referee finding 7 verified this "
                "via person-block bootstrap: 2*se_block ~ realized_sigma), "
                "but n_events counts person-waves, not independent persons",
                "the PSID FAMILY UNIT is not the Census HOUSEHOLD; "
                "household-size and coresidence shares carry that concept "
                "delta (see external_anchor.concept_delta)",
                "person-waves with no positive wave weight (~17.8%) are "
                "dropped; the drop is not composition-neutral (see "
                "zero_weight_drop)",
                "boundary fragility: a few cells' tolerances sit near "
                "T_max=ln(1.5) and could cross under re-measure (referee "
                "finding 7c); the lock freezes the committed partition",
            ],
        },
        "statistic_families": {
            "coresident_spouse": (
                "STOCK: coresident spouse/partner share by age band x sex"
            ),
            "coresident_parent": (
                "STOCK: share living with a parent by age band x sex "
                "(+ 45+ aggregate)"
            ),
            "coresident_child": (
                "STOCK: share living with a child by age band x sex"
            ),
            "coresident_grandchild": (
                "STOCK: share living with a grandchild by age band x sex "
                "(+ 55+ aggregate)"
            ),
            "multigen": (
                "STOCK: >=3-distinct-generation (Census B11017) household "
                "share by age band x sex (+ 65+ aggregate)"
            ),
            "hh_size": (
                "STOCK: person-level household-size share distribution "
                "(1..4 and open 5+)"
            ),
            "parental_home_exit": (
                "TRANSITION: share leaving a coresident parent by the next "
                "wave, 15-34 by sex"
            ),
            "spousal_loss": (
                "TRANSITION: share losing a coresident spouse by the next "
                "wave, 55+ by sex"
            ),
            "multigen_entry_exit": (
                "TRANSITION: entry into / exit from a 3-generation "
                "household (pooled over age x sex)"
            ),
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
        "protocol": protocol_block(oc, len(gated)),
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
            "base_sha": _merge_base(ROOT),
            "origin_master_sha": _rev(ROOT, "origin/master"),
            "build_commit_note": (
                "The build script + module that produced this artifact live "
                "on branch gate2b-floors and are committed TOGETHER with "
                "this file; a git commit cannot embed its own hash "
                "(chicken-and-egg), so no field can name the build commit. "
                "base_sha pins the origin/master commit the branch extends. "
                "To reproduce: check out branch gate2b-floors HEAD (NOT "
                "base_sha, which predates the gate-2b code) and run "
                ".venv/bin/python scripts/build_gate2b_floors.py -- it "
                "reproduces this artifact bit-identically except "
                "elapsed_seconds and the two sha pins."
            ),
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
