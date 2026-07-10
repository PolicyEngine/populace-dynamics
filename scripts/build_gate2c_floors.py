"""Build the gate-2c marriage x earnings floors: moments + noise floor.

PRE-LOCK EVIDENCE, NOT A GATE RUN. The first step of the gate-2c lock
ceremony (``gates.yaml`` gate_2c, ``lock_ceremony.exists: false``,
``required_before_any_2c_pass``: *same ceremony as 2a/2b -- a
pre-registered gate on an assortative marriage x earnings floor, an
adversarial referee round, a verification round, and a ratifying merge*).
It is the analogue, one tranche over, of ``scripts/build_gate2_floors.py``
(tranche 2a) and ``scripts/build_gate2b_floors.py`` (gate 2b): the
committed reference moments, the person-disjoint 100-seed half-split noise
floor (the sd basis the DRAFT tolerances derive from), the ratified 2a
mean-over-K=20-draws scoring protocol (amendment 1), the T_max = ln(1.5)
power-cap partition, the training-copy disclosure and the
faithful-candidate operating characteristic.

It reads no gate and changes no gate, and writes NO ``gates.yaml`` block:
gate_2c already exists in ``gates.yaml`` as an unlocked stub, and the
lock-ceremony flip (proposal -> referee -> fixes -> verification -> ratify)
inserts the thresholds later. This artifact only feeds that ceremony.

Marriage x earnings JOINT statistic families (see
:mod:`populace_dynamics.data.couple_earnings`):

1. assortative mating -- own-AIME-tercile x spouse-AIME-tercile contingency
   shares over directed couples (the within-couple AIME rank correlation is
   reported alongside, REPORT-ONLY);
2. marriage hazards conditional on own earnings -- first-marriage and
   remarriage rates by AIME tercile x age band x sex;
3. earnings dynamics around marital events -- median post/pre earnings-ratio
   around marriage and divorce, by sex;
4. shared-earnings distribution shape -- combined-AIME quintile cutpoint
   ratios.

Run from the repository root with the PSID products staged and the pe-us
checkout available (AIME needs the certified NAWI series)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv/bin/python scripts/build_gate2c_floors.py

It needs no populace-fit (real-vs-real only).
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics import artifacts
from populace_dynamics.data import couple_earnings as ce
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2c_floors_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2c_floors.v1"

#: Floor seeds: 100 person-disjoint splits stabilise the tolerance
#: estimator (tranche-2a round-1 finding 2).
SEEDS = tuple(range(100))
#: The pinned gate-run seeds; a candidate is scored on these five with the
#: 4-of-5 conjunction, exactly as 2a / 2b.
GATE_SEEDS = (0, 1, 2, 3, 4)
#: Minimum events (weaker half, worst of the 100 seeds) for a cell to be
#: gate-eligible. 20 is NCHS's own reliability floor, inherited from 2a.
MIN_EVENTS_FOR_GATE = 20
#: DRAFT threshold multiplier: |ln ratio| tolerance = round(mean + k*sd).
#: ~4 sigma on the stabilised floor. DRAFT -- the referee round sets k.
DRAFT_K = 4
DRAFT_ROUNDING = 3
#: Power cap (tranche-2a finding 3): a gated cell accepts at most a 1.5x
#: rate error. Fixed a priori; which cells demote is derived from the floor.
T_MAX = math.log(1.5)
T_MAX_SOURCE = "ln(1.5)"

#: Candidate-scoring estimator (tranche 2a amendment 1, ratified
#: 2026-07-08): the candidate cell rate is the MEAN over K pre-registered
#: simulation draws (numpy default_rng(5200 + k)), a stream DISTINCT from
#: the split seeds, scored once as |ln(rbar / rate_a)|. The floor and the
#: tolerances are draw-noise-free; the K-draw mean shares that basis. This
#: build measures the real-vs-real null (no simulation); the K=20 estimator
#: is the contract a future candidate run scores under.
CANDIDATE_DRAWS = 20
CANDIDATE_DRAW_STREAM = "numpy.random.default_rng(5200 + k), k=0..K-1"

#: No coverage-recovery aggregates for gate-2c (imported from the moment
#: module, which returns an empty map -- see its docstring and fix G below).
AGGREGATIONS: dict[str, list[str]] = ce.aggregation_members()


# --------------------------------------------------------------------------
# Panel assembly + estimand / selection disclosure
# --------------------------------------------------------------------------
def _native(obj: Any) -> Any:
    """Recursively coerce numpy scalars to JSON-native python scalars."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _marriage_decade_slices(panel: ce.CoupleEarningsPanel) -> dict[str, Any]:
    """Per-marriage-decade REPORT-ONLY slices: couple counts, the diagonal
    (concordant-tercile) share, and the Spearman rho -- so the selected
    marriage-era mixture is visible run over run (the 2c analogue of gate
    2b's era_slices; finding 3 / estimand honesty)."""
    couples = panel.couples
    total_w = float(couples["weight"].sum())
    corr = ce.assortative_correlation(panel)["by_marriage_decade"]
    slices: dict[str, Any] = {}
    for decade in ce.MARRIAGE_DECADE_BANDS:
        sub = couples[couples["start_decade"] == decade]
        w = float(sub["weight"].sum())
        diag = float(
            sub.loc[
                sub["own_tercile"] == sub["spouse_tercile"], "weight"
            ].sum()
        )
        slices[str(decade)] = {
            "n_directed_couples": int(len(sub)),
            "weight_share_pct": (
                round(100.0 * w / total_w, 2) if total_w else None
            ),
            "concordant_tercile_share": round(diag / w, 4) if w else None,
            "spearman_aime": corr[str(decade)]["spearman_aime"],
        }
    slices["note"] = (
        "REPORT-ONLY, never gated. Per-marriage-decade couple counts, the "
        "weighted concordant-tercile (diagonal) share, and the within-couple "
        "AIME Spearman rho, so the SELECTED marriage-era mixture the pooled "
        "contingency certifies is visible: a candidate matching only one era "
        "can be read against the pooled cells it is scored on."
    )
    return slices


def _birth_cohort_disclosure(
    panel: ce.CoupleEarningsPanel,
) -> dict[str, Any]:
    """The AIME-computable birth-cohort concentration behind the couples."""
    couples = panel.couples
    # start_year - own age is not carried; disclose the marriage-decade
    # span and the couple selection instead (cohort is bounded by the NAWI
    # indexing window, disclosed in selection_estimand).
    decades = couples["start_decade"]
    return {
        "marriage_start_year_range": (
            [
                int(couples["start_year"].min()),
                int(couples["start_year"].max()),
            ]
            if len(couples)
            else None
        ),
        "marriage_decade_counts": {
            str(int(d)): int(n)
            for d, n in decades.value_counts().sort_index().items()
        },
    }


# --------------------------------------------------------------------------
# Internal noise floor (person-disjoint 50/50 half-split, 100 seeds)
# --------------------------------------------------------------------------
def measure_seed_halfsplit(
    seed: int, panel: ce.CoupleEarningsPanel
) -> dict[str, Any]:
    """One seed: split persons 50/50 (by ego person_id), all cells per half.

    Side A (the drawn half) is seed s's HOLDOUT; side B the train
    complement. Under the gate protocol a candidate refit on side B is
    scored against side A's empirical rate, for which the symmetric
    half-vs-half ratio |ln(r_A / r_B)| is the null.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(side_a.person_id)
    ids_b = set(side_b.person_id)
    cells_a = ce.reference_moments(panel, ids_a, weighted=True)
    cells_b = ce.reference_moments(panel, ids_b, weighted=True)

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
    """The pooled floor block for a cell defined on every seed."""
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
    AND it is not superseded by a gating aggregate. Gate-2c declares NO
    coverage-recovery aggregates (:data:`AGGREGATIONS` is empty; see the
    moment-module docstring and fix G), so the third clause is inert here --
    every cell is judged on its own merit. The supersession machinery is
    kept intact so the derivation is identical to 2a / 2b.
    """
    member_of: dict[str, str] = {}
    for agg, members in AGGREGATIONS.items():
        for m in members:
            member_of[m] = agg

    gated: set[str] = set()
    report: set[str] = set()
    reasons: dict[str, str] = {}

    # 1) aggregates decide first (none for 2c).
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
    """DRAFT per-cell |ln ratio| tolerances for the GATED cells."""
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
    """Fold the tolerance, sigma, gate flag and report reason into
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
def holdout_id_commitment(panel: ce.CoupleEarningsPanel) -> dict[str, Any]:
    """The sha256 of each gate seed's holdout (side A) ego person-ids."""
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
    against side A (holdout); a verbatim train-copy EMITS side B's rates, so
    its per-cell score is exactly ``|ln(rate_B / rate_A)|`` = the committed
    floor value for that seed. It passes at the noise floor, as ANY moment
    gate's copy must. The memorisation defence is procedural (registration +
    holdout exclusion + no_self_rescue), not the cell set.
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
            "A train-copy passes at ~the noise floor (max score below 1x "
            "tolerance), as any moment gate's copy must. The marriage x "
            "earnings cells do not fix this -- a copy reproduces every "
            "moment; the memorisation defence is procedural (registration + "
            "holdout exclusion + no_self_rescue), NOT the cell set. What the "
            "cells DO catch is non-copy structural failure: a couple model "
            "that mismatches who marries whom by earnings tercile, the "
            "earnings-conditional marriage timing, the around-event earnings "
            "dynamics, or the shared-earnings distribution shape."
        ),
    }


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
            "2*Phi(tolerance/sigma)-1. Seed pass = product over gated cells; "
            "gate = P(>=4 of 5 seeds) = p^5 + 5 p^4 (1-p). The K=20 "
            "estimator shares this draw-noise-free basis (see "
            "protocol.basis_note)."
        ),
        "n_gated_cells": len(gated),
        "p_seed_pass": round(p_seed, 4),
        "p_gate_pass_4_of_5": round(p_gate, 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# Protocol: the ratified 2a K=20 estimator (used from the start; fix A)
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
        "so the OC is achievable. Under a SINGLE-draw estimator it would be "
        "UNACHIEVABLE (tranche 2a amendment-1 finding 1): one simulation "
        "replicate injects ~one extra floor-sigma per cell (score "
        "~sqrt(1.5) x realized_sigma), roughly halving P(seed). This 2c "
        "build adopts the ratified K=20 estimator FROM THE START, so the "
        "single-draw defect the 2b round-1 referee flagged never enters."
    )


def _fresh_run_artifact_schema(n_gated: int) -> dict[str, Any]:
    """The candidate-run contract under the K=20 estimator (2a amendment 1),
    inherited so a future gate_2c candidate run is auditable."""
    return {
        "applies_to": (
            "a fresh candidate one-shot run registered AFTER a future "
            "gate_2c lock; NEVER this floor build, which measures the "
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
    (amendment 1, 2026-07-08), adopted from the start (fix A pre-empted)."""
    return {
        "option": "a",
        "estimator": (
            "mean-over-K=20-draws (tranche 2a amendment 1, ratified "
            "2026-07-08, PR 96). The candidate cell rate rbar_candidate,s is "
            "the MEAN over K=20 pre-registered simulation draws (numpy "
            "default_rng(5200 + k), k=0..19; a stream DISTINCT from the "
            "split seeds), scored ONCE per cell as |ln(rbar_candidate,s / "
            "rate_a,s)| -- NOT the mean of the per-draw |ln| scores. This 2c "
            "build uses the ratified 2a estimator FROM THE START; the "
            "superseded single-draw replicate the 2b round-1 referee flagged "
            "(finding 1) never enters."
        ),
        "candidate_draws": CANDIDATE_DRAWS,
        "candidate_draw_stream": CANDIDATE_DRAW_STREAM,
        "statistic": (
            "|ln(rbar_candidate,s / rate_a,s)|, rbar = mean over K=20 draws "
            "of the simulated cell rate; symmetric and scale-free"
        ),
        "description": (
            "Per gate seed s: refit the candidate on the seed-s train "
            "complement (side B; the seed-s holdout persons excluded from "
            "all fitting), SIMULATE side A's couples / marital events / "
            "earnings at K=20 pre-registered draws (default_rng(5200 + k)), "
            "form the 20-draw mean cell rate rbar_candidate,s, and score "
            "|ln(rbar / rate_a)| against side A's own empirical rate. The "
            "real-vs-real half-vs-half floor |ln(rate_A/rate_B)| is the "
            "null. Seed s passes iff every gated cell holds; the gate passes "
            "iff >=4 of 5 gate seeds pass."
        ),
        "varies_per_seed": (
            "the 50/50 person split by ego person_id (holdout = side A of "
            "split_panel_by_person(seed=s)); the candidate's fit complement "
            "and its K=20 simulated draws of the holdout"
        ),
        "scored_against": (
            "the seed-s holdout half's empirical rate (rate_a in "
            "noise_floor_per_seed)"
        ),
        "conjunction": "all gated cells per seed AND >=4 of 5 gate seeds",
        "adopted": (
            "the estimator is tranche 2a amendment 1 (amendment_history "
            "entry 1, ratified 2026-07-08 by merge of PR 96), adopted here "
            "at draft time so the disclosed OC is achievable and the "
            "single-draw defect never enters (unlike the 2b draft, which "
            "shipped the superseded text and the referee flagged it)."
        ),
        "basis_note": _basis_note(oc),
        "fresh_run_artifact_schema": _fresh_run_artifact_schema(n_gated),
    }


# --------------------------------------------------------------------------
# External anchor: the honest gap (declared, not fabricated) -- fix F
# --------------------------------------------------------------------------
def external_anchor_gap() -> dict[str, Any]:
    """Declare the external-anchor gap and the pre-ratification commitment."""
    return {
        "status": "none_bundled",
        "reported_anchor_not_gated": True,
        "required_before_ratifying_flip": True,
        "note": (
            "No external marriage x earnings anchor is bundled with this "
            "STEP-1 floor. The reference moments (assortative-mating "
            "contingency, earnings-conditional marriage hazards, "
            "around-event earnings dynamics, shared-earnings shape) are "
            "cross-checked only against themselves (real-vs-real "
            "half-split). This is a KNOWN GAP vs tranche 2a (which shipped "
            "concept-decomposed NCHS anchors in its floor PR). It is "
            "declared, not fabricated. As with gate 2b, the "
            "concept-decomposed external report (candidate sources below) "
            "must PRECEDE the ratifying flip -- a later ceremony step than "
            "this floor."
        ),
        "concept_delta": (
            "The couple earnings measure is a PSID-observed PARTIAL-CAREER "
            "AIME proxy (family-file labor income, reference years "
            "1993-2022, top-35 of the observed years) -- not the true SSA "
            "covered-earnings AIME, so LEVELS are lower bounds and are not "
            "level-comparable to SSA/OCACT tabulations without a "
            "truncation decomposition. The gated moments use terciles, "
            "within-couple ranks and cutpoint RATIOS, which are far more "
            "robust to that truncation than the levels."
        ),
        "candidate_sources": [
            "SSA/OCACT MINT model spousal/dual-entitlement tabulations",
            "Census/ACS PUMS spouse-earnings correlation by cohort",
            "CPS ASEC married-couple earnings-rank contingency tables",
            "Schwartz & Mare (2005) / Eika-Mogstad-Zafar assortative-mating "
            "series (educational + earnings sorting)",
        ],
    }


# --------------------------------------------------------------------------
# Draft-thresholds note (pre-lock; feeds the gate_2c ceremony, NOT gates.yaml)
# --------------------------------------------------------------------------
def draft_thresholds_note(
    gated: set[str], report: set[str], oc: dict[str, Any], meta: dict[str, Any]
) -> str:
    cov_j = meta.get("join_coverage_both_over_joinable")
    cov_e = meta.get("join_coverage_both_over_ego_supply")
    return (
        "DRAFT GATE-2C VALIDATION EVIDENCE -- PRE-LOCK, NOT RATIFIED "
        "(v1, the first step of the gate-2c lock ceremony; the 2b round-1 "
        "referee's lessons pre-empted, not repeated).\n\n"
        "Pre-lock evidence package for the gate-2c marriage x earnings JOINT "
        "statistics (who marries whom, and what they earn; gates.yaml "
        "gate_2c: mh85_23 marital histories crossed with the "
        "gate-1-certified earnings histories), the analogue of what "
        "preceded tranche 2a's and gate 2b's lock, one tranche over. It "
        "changes no locked value and no model reads it. It writes NO "
        "gates.yaml block: gate_2c already exists as an unlocked stub, and "
        "the lock-ceremony flip (proposal -> referee -> fixes -> "
        "verification -> ratify) inserts the thresholds later. The k value "
        f"({DRAFT_K}) is a DRAFT proposal, for the ceremony to fix.\n\n"
        "PROTOCOL (option a, the ratified 2a K=20 estimator -- adopted from "
        "the START). Per gate seed s a candidate is refit on the seed-s "
        "TRAIN complement (side B), SIMULATES the holdout couples/events at "
        "K=20 pre-registered draws (numpy default_rng(5200 + k), a stream "
        "distinct from the split seeds), and each cell is scored ONCE as "
        "|ln(rbar_candidate,s / rate_a,s)| where rbar is the mean over the "
        "20 draws of the cell RATE (NOT the mean of the per-draw |ln| "
        "scores), against the seed-s HOLDOUT half's own empirical rate. The "
        "symmetric half-vs-half floor is the null. Seed passes iff every "
        "gate-eligible cell holds; the gate passes iff >=4 of 5 seeds pass. "
        "The single-draw text the 2b round-1 referee flagged (finding 1) is "
        "pre-empted; see protocol.basis_note.\n\n"
        "INTERNAL FLOOR. The person-disjoint 50/50 half-split (by ego "
        "person_id) |log rate ratio| floor over 100 split seeds. Each DRAFT "
        f"per-cell tolerance is round(mean + {DRAFT_K}*sd, {DRAFT_ROUNDING}); "
        "on 100 seeds the estimator is stable. The faithful-candidate "
        f"operating characteristic recomputes P(seed) = {oc['p_seed_pass']} "
        f"and P(gate >=4/5) = {oc['p_gate_pass_4_of_5']} over "
        f"{oc['n_gated_cells']} gated cells, draw-noise-free, achievable "
        "under the K=20 mean.\n\n"
        "POWER CAP. A cell is gate-eligible only if its stabilised tolerance "
        f"<= T_max = {T_MAX_SOURCE} (a gated cell accepts at most a 1.5x "
        "rate error) AND it carries >=20 events on the weaker half. The "
        "sparse late-first-marriage-by-earnings cells (35-44 / 45+ terciles) "
        "and the noisiest median cells fail the cap and stay report-only; "
        "the partition is derived from cell_stability, never hand-picked. "
        "There are NO coverage-recovery aggregates (finding-6b-aware; see "
        "aggregations and the moment-module docstring).\n\n"
        "ESTIMAND / SELECTION (finding 3 fix -- named, not implied). The "
        "gated moments describe the SELECTED universe of PSID couples with a "
        "computable partial-career AIME proxy for BOTH partners: "
        f"{meta.get('n_directed_couples')} directed couples "
        f"({meta.get('n_unordered_couple_pairs')} unordered pairs) from "
        f"{meta.get('n_marriage_episodes_joinable')} joinable marriage "
        f"records -- join coverage {cov_j} of joinable records, {cov_e} of "
        "AIME-computable egos. It is NOT the marriage x earnings joint of "
        "all PSID marriages; the couples concentrate in the birth cohorts "
        "whose age-60 indexing year lands in the NAWI series and whose "
        "careers overlap 1993-2022. See selection_estimand, "
        "marriage_decade_slices and the WARTS.\n\n"
        "SEMANTICS. AIME is the PSID-observed partial-career proxy (fix: "
        "levels are lower bounds; terciles/ranks/cutpoint-ratios are the "
        "gated quantities, robust to truncation). Terciles are fixed on the "
        "full AIME supply and applied to every half-split. The contingency "
        "over directed couples is symmetric by construction; the "
        "within-couple Spearman rho is REPORT-ONLY (a correlation is not a "
        "scale-free |ln ratio|).\n\n"
        "EXTERNAL ANCHOR. NONE bundled (external_anchor.status = "
        "none_bundled) -- a KNOWN GAP vs 2a; declared not fabricated, and "
        "REQUIRED before the ratifying flip (finding 5).\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY. Gated cells: "
        f"{sorted(gated)}. Report-only: {sorted(report)}.\n\n"
        "GOVERNANCE. One-shot outer runs, amendments only by public proposal "
        "+ referee round, no_self_rescue and the version pin inherited from "
        "gate 1; the per-seed holdout ids are committed (holdout_ids); every "
        "gated statistic is weighted by the person-constant most-recent "
        "positive PSID weight (no unweighted gated statistic). The AIME "
        "chain pins pe_us_revision (classifier_version_pin). BASELINE: these "
        "statistics feed spousal / survivor benefit LEVELS through who "
        "marries whom by earnings; this evidence fixes the component's "
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

    panel = ce.build_couple_panel()
    meta = panel.meta
    if verbose:
        print(
            f"couples: {meta['n_directed_couples']} directed "
            f"({meta['n_unordered_couple_pairs']} pairs); "
            f"AIME supply {meta['n_aime_supply_persons']}; "
            f"split persons {meta['n_split_persons']}"
        )

    ref_w = ce.reference_moments(panel, weighted=True)
    ref_u = ce.reference_moments(panel, weighted=False)
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
    correlation = ce.assortative_correlation(panel)

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
        "run": "gate2c_floors_v1",
        "reported_anchor_not_gated": True,
        "referee_round": "PR (draft -- 2b round-1 lessons pre-empted)",
        "component": (
            "marriage x earnings joint (gate 2c pre-lock; who-marries-whom "
            "by earnings: assortative-mating contingency + "
            "earnings-conditional marriage hazards + around-event earnings "
            "dynamics + shared-earnings shape, from mh85_23 x the "
            "gate-1-certified earnings histories)"
        ),
        "purpose": (
            "Gate-2c pre-lock evidence (v1): committed marriage x earnings "
            "joint reference moments, the 100-seed person-disjoint "
            "half-split noise floor (the sd basis the DRAFT tolerances "
            "derive from), the ratified 2a mean-over-K=20-draws scoring "
            "protocol adopted from the start, and the power-cap partition -- "
            "the first step of the gate_2c lock ceremony (gates.yaml "
            "gate_2c.lock_ceremony.required_before_any_2c_pass). Reads no "
            "gate and changes no gate on its own; writes NO gates.yaml block "
            "(the ceremony flip inserts the thresholds later). See "
            "draft_thresholds_note (NOT RATIFIED)."
        ),
        "holdout_basis": ["mh85_23", "family_earnings_panel_gate1_certified"],
        "ceremony": {
            "tranche": "2c_marriage_earnings_joint",
            "gates_yaml_stub": "gate_2c (status: unlocked, locked: false)",
            "step": (
                "1 of the lock ceremony: the 100-seed split noise floor over "
                "the marriage x earnings joint (proposal); referee -> fixes "
                "-> verification -> ratify follow, and only the ratifying "
                "flip touches gates.yaml"
            ),
            "gates_yaml_untouched": True,
            "mirrors": "runs/gate2_floors_v2.json (2a) + gate2b_floors_v1.json",
            "round_1_lessons_preempted": (
                "K=20 estimator adopted from the start (not the superseded "
                "single draw); estimand named as the SELECTED couple "
                "universe with join-coverage + marriage-decade disclosure; "
                "machine-recorded gated/report-only reasons; no perturbation "
                "dead zones in the binding tests; an always-runnable "
                "label-swap catch; raw MH7/MH8/MH12 mappings pre-verified "
                "against the staged formats (joinable-spouse count 32,522 "
                "reproduces the marriage-module figure)"
            ),
        },
        "data": {
            "marriage_history": (
                "populace_dynamics.data.marriage.marriage_history (mh85_23 "
                "Marriage History File): per-person marriages with joinable "
                "spouse_person_id (MH7 in 1-9308, MH8 in 1-399), start/end "
                "years and how_ended -- all decoded from the documented code "
                "sets and label-verified at read"
            ),
            "earnings": (
                "populace_dynamics.data.family.family_earnings_panel "
                "(head/spouse family-file labor income, reference years "
                "1993-2022) -> populace_dynamics.ss.benefits.aime: the "
                "gate-1-certified AIME chain (NAWI-indexed top-35), a "
                "PSID-observed PARTIAL-CAREER proxy (levels are lower "
                "bounds; ranks/terciles/ratios are the gated quantities)"
            ),
            "weights": (
                "each person carries the person-constant MOST-RECENT-POSITIVE "
                "PSID cross-sectional weight "
                "(populace_dynamics.data.panels.demographic_panel), exactly "
                "tranche 2a's weight_definition. No unweighted gated "
                "statistic; the unweighted rate is reported alongside."
            ),
            "estimand": (
                "the marriage x earnings joint of the SELECTED universe of "
                "PSID couples with a computable partial-career AIME for both "
                "partners -- NOT all PSID marriages. Named per "
                "governance.amendment_rules."
                "description_claims_exactly_the_scored_surface (2b finding "
                "3). See selection_estimand and marriage_decade_slices."
            ),
            **_native(meta),
        },
        "selection_estimand": {
            "universe": (
                "directed couples = marriage-history records with a joinable "
                "PSID sample-member spouse where BOTH partners carry a "
                "computable AIME proxy (age-60 indexing year within the NAWI "
                f"series {meta['nawi_year_range']}, and >= "
                f"{ce.MIN_EARNINGS_YEARS} observed positive-earnings years)"
            ),
            "join_coverage_both_over_joinable": meta[
                "join_coverage_both_over_joinable"
            ],
            "join_coverage_both_over_ego_supply": meta[
                "join_coverage_both_over_ego_supply"
            ],
            "aime_is_partial_career_proxy": True,
            "min_earnings_years": ce.MIN_EARNINGS_YEARS,
            "note": (
                "The AIME-computable requirement selects birth cohorts whose "
                "age-60 year lands in the NAWI window and whose careers "
                "overlap the 1993-2022 family-earnings panel; the couples "
                "therefore concentrate in older marriage decades. This "
                "selection is the 2c analogue of gate 2b's era-weighting "
                "disclosure -- named, not implied."
            ),
        },
        "birth_cohort_disclosure": _birth_cohort_disclosure(panel),
        "marriage_decade_slices": _marriage_decade_slices(panel),
        "assortative_correlation_report_only": correlation,
        "panel_construction": {
            "couple_unit": (
                "directed couple = one marriage-history record (ego -> "
                "spouse); a two-sample-member marriage yields both directed "
                "records, so the tercile contingency is symmetric"
            ),
            "split_unit": (
                "ego person_id (every couple, marital event and earnings "
                "window of an ego lands on one side of the half-split)"
            ),
            "couple_correlation_wart": (
                "the split is person-disjoint by EGO but NOT couple-disjoint: "
                "a couple whose two members split to opposite sides "
                "contributes one directed record to EACH half, so the halves "
                "share couples. This makes the halves marginally MORE alike, "
                "so the |ln ratio| floor is if anything conservative "
                "(tighter tolerances, harder gate) -- disclosed, not hidden."
            ),
            "aime_semantics": (
                "AIME = NAWI-indexed top-35 of the PSID-observed family-file "
                "earnings (1993-2022), a PARTIAL-CAREER proxy; terciles are "
                "fixed on the full AIME supply and applied to every "
                "half-split so the categories do not shift; the earnings "
                "measure LEVELS are lower bounds (truncation), the gated "
                "quantities are terciles / ranks / cutpoint ratios"
            ),
            "hazard_stratifier_caveat": (
                "the marriage-hazard 'own earnings tercile' is the person's "
                "FIXED career-AIME tercile, not contemporaneous earnings, so "
                "it summarises earnings CAPACITY (with the usual "
                "marriage<->earnings reverse-causality caveat), used as a "
                "stable stratum for the age-band hazard"
            ),
            "event_window_convention": (
                "the around-event earnings ratio is (mean post-window "
                f"earnings) / (mean pre-window earnings) over a +/-"
                f"{ce.EVENT_WINDOW_YEARS}-year window; an event contributes "
                "only where the person has >=1 observed positive-earnings "
                "year in BOTH windows -- an earnings-support restriction that "
                "selects events overlapping 1993-2022"
            ),
            "coverage_caveats": [
                "the earnings source is family-file HEAD/SPOUSE labor income "
                "only, so a person never observed as head or spouse has no "
                "AIME and cannot enter a couple",
                "the AIME proxy is a truncated partial career (1993-2022 "
                "observed years), so its LEVELS understate the true SSA AIME; "
                "the gated moments use terciles / ranks / ratios",
                "the person-disjoint split is by ego, not couple-disjoint "
                "(couple_correlation_wart)",
                "boundary fragility: a few cells' tolerances sit near "
                "T_max=ln(1.5) and could cross under re-measure; the lock "
                "freezes the committed partition",
            ],
        },
        "statistic_families": {
            "assort_mating": (
                "own-AIME-tercile x spouse-AIME-tercile contingency share "
                "over directed couples (3x3)"
            ),
            "first_marriage_by_earnings": (
                "first-marriage hazard by AIME tercile x age band x sex "
                "(never-married person-years at risk)"
            ),
            "remarriage_by_earnings": (
                "remarriage hazard by AIME tercile x sex "
                "(post-dissolution person-years at risk)"
            ),
            "earnings_around_marriage": (
                "weighted median (post/pre) earnings ratio around marriage, "
                "by sex + all"
            ),
            "earnings_around_divorce": (
                "weighted median (post/pre) earnings ratio around divorce, "
                "by sex + all"
            ),
            "shared_earnings_ratio": (
                "adjacent-quintile cutpoint ratios of the couple's combined "
                "AIME (scale-free distribution shape)"
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
                "independent real halves -- the sd basis the DRAFT gate-2c "
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
        "aggregations_note": (
            "EMPTY by design: gate-2c declares no coverage-recovery "
            "aggregates. Pooling the sparse late-first-marriage age bands "
            "would mask a standalone-gateable 35-44 cell (2b round-1 finding "
            "6b) or pool across the earnings terciles the tranche exists to "
            "resolve, so the sparse cells are honestly report-only."
        ),
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
                "on all seeds AND >=20 events AND tolerance <= T_max). NOT "
                "yet mirrored into gates.yaml -- the gate_2c lock-ceremony "
                "flip inserts them after the referee + verification rounds."
            ),
            "cells": drafts,
        },
        "draft_thresholds_note": draft_thresholds_note(
            gated, report, oc, meta
        ),
        "revision_pins": {
            "base_sha": _merge_base(ROOT),
            "origin_master_sha": _rev(ROOT, "origin/master"),
            "pe_us_revision": meta.get("pe_us_revision"),
            "build_commit_note": (
                "The build script + module that produced this artifact live "
                "on branch gate2c-floors and are committed TOGETHER with "
                "this file; a git commit cannot embed its own hash "
                "(chicken-and-egg), so no field names the build commit. "
                "base_sha pins the origin/master commit the branch extends; "
                "pe_us_revision pins the SSA-parameter checkout the AIME "
                "chain used (classifier_version_pin). To reproduce: check "
                "out branch gate2c-floors HEAD (NOT base_sha) with the same "
                "pe_us_revision and run "
                ".venv/bin/python scripts/build_gate2c_floors.py -- it "
                "reproduces this artifact bit-identically except "
                "elapsed_seconds and the two sha pins."
            ),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return _native(artifact)


def main() -> None:
    artifact = run(verbose=True)
    artifacts.write_new(ARTIFACT_PATH, artifact)
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
