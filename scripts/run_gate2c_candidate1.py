"""Gate-2c candidate 1 one-shot scored run (registration 4950250151).

FROZEN spec: issue #42 comment 4950250151. Couple formation composed from
certified components plus train-fitted joints
(:mod:`populace_dynamics.models.couple_formation_sim_v1`), scored against the
LOCKED gate-2c contract (``gates.yaml`` gate_2.gate_2c, LOCKED 2026-07-10)
and the frozen floor ``runs/gate2c_floors_v1.json``:

* 27 gated marriage x earnings joint cells; the ``|ln(rbar / rate_a)|``
  statistic; ``rbar`` = MEAN over K=20 pre-registered simulation draws
  (``numpy.random.default_rng(5200 + k)``, k=0..19), scored ONCE per cell
  (NOT the mean of the per-draw scores);
* per-seed COUPLE-DISJOINT holdout, split by ``attrs.component_id`` with
  ``default_rng(seed)`` (side A = holdout / simulated, side B = train / fit);
* seed passes iff every gated cell holds; the gate passes iff >= 4 of 5
  gate seeds pass. One-shot; publishes regardless (no holdout tuning).

The scored tolerances are read from the LOCKED ``gates.yaml`` block and
cross-checked against the frozen floor's ``draft_thresholds`` (both are
``round(floor mean + 4*sd, 3)`` capped at ``ln(1.5)``). Run:

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_gate2c_candidate1.py \
        --out runs/gate2c_hazard_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from populace_dynamics import artifacts
from populace_dynamics.data import (
    births,
    deaths,
    family,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.data import (
    couple_earnings as ce,
)
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import couple_formation_sim_v1 as cfs
from populace_dynamics.models.family_transitions.common import (
    marriage_order_map,
)
from populace_dynamics.ss.params import load_ssa_parameters

ROOT = Path(__file__).resolve().parents[1]
FLOOR_RUN = ROOT / "runs" / "gate2c_floors_v1.json"
GATES = ROOT / "gates.yaml"
DEFAULT_OUT = ROOT / "runs" / "gate2c_hazard_v1.json"

SCHEMA_VERSION = "gate2c_hazard.v1"
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
#: Provenance-only outer seed base (the superseded single-draw estimator's
#: stream; NOT consumed for scoring, per the locked protocol / the 2a
#: convention default_rng(4200 + seed)).
SIM_SEED_PROVENANCE_BASE = 4200
SPLIT_COLUMN = "component_id"
SPLIT_FRACTION = 0.5
EXACT_ATOL = 1e-12
N_GATED = 27

REGISTRATION_POINTER = "4950250151"
SPEC_REGISTRATION = (
    "issue #42 comment 4950250151 -- Gate-2c candidate 1 registration: "
    "couple formation from certified components (FROZEN spec, one-shot)."
)

#: The registration's pre-registered forecast (recorded, never graded here).
PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.10-0.25",
    "basis": (
        "a first candidate against a joint-distribution contract; the 2a and "
        "2b ladders both opened 0/5"
    ),
    "modal_failure_classes_in_order": [
        "the assortative contingency diagonal (a decile-conditional kernel "
        "may under-concentrate the strong per-year diagonal delta~1.37)",
        "the earnings-conditional marriage-hazard cells (composition between "
        "the certified marital core and the earnings conditioning)",
        "the event-window cells (shift-kernel support)",
    ],
    "primary_value_if_fail": (
        "the per-family decomposition that seeds forensics-1 of this ladder"
    ),
    "grading_note": (
        "this run RECORDS its verdict and forecast; it does NOT grade itself. "
        "Grading is a separate ceremony step."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "five_components_source": (
        "component 1 (marriage/remarriage events + timing) = the certified "
        "tranche-2a registry CANDIDATE_16 via REGISTRY.fit refit on side B "
        "and simulate(mpanel, holdout_ids, components, 5200+k); component 3 "
        "(spouse age) = the same fit's spousal_age_gaps via draw_spousal_gaps "
        "(the 2a machinery's existing convention). Components 2 (assortative "
        "kernel), 4 (event-window shift kernel) and 5 (shared-earnings cells) "
        "are layered on the certified simulated events."
    ),
    "assortative_kernel": (
        "P(spouse earnings-decile | own earnings-decile, own age band, own "
        "sex) fit on side-B directed couples on the committed decile axis, "
        "hierarchically add-alpha smoothed with backoff "
        "(global -> sex -> sex x own_decile -> full) so every conditioning "
        "cell is well-defined despite sparse per-cell couple counts. The own "
        "age band reuses the certified first-marriage hazard bands."
    ),
    "directed_both_orientation_emission": (
        "candidate_construction.couple_emission pin: each simulated marriage "
        "of a holdout supply ego emits BOTH directed records (ego -> drawn "
        "spouse AND drawn spouse -> ego), weighted by the same ego couple "
        "weight, so the tercile contingency is exactly symmetrized-directed "
        "(a single-orientation emission is NON-CONFORMANT)."
    ),
    "committed_cut_provenance": (
        "candidate_construction.cut_provenance pin: the tercile cut levels, "
        "the earnings-decile edges and within-decile spouse-value pools, and "
        "the placebo drift deflators are FIXED on the full real earnings "
        "supply and applied to every seed / draw -- never recomputed on "
        "simulated output. Only the conditional kernels and the tranche-2a "
        "components are train-fitted (side B); holdout couples are excluded "
        "from all fitting."
    ),
    "event_window_support_and_detrend": (
        "the emitted event-window ratio is a raw (nominal) draw from the "
        "train around-event post/pre ratio pool for (event_type, sex), "
        "weighted by the ego demographic weight (fix F); the locked cell "
        "detrends by the COMMITTED placebo deflator (fix E), so the gate "
        "certifies the event increment. The train pool is the supported "
        "windows (>=1 observed positive year in both +/-3y sides), so the "
        "support restriction rides in the kernel."
    ),
    "reference_moments_reused_verbatim": (
        "the simulated frames are built to the CoupleEarningsPanel schema and "
        "scored through couple_earnings.reference_moments VERBATIM with the "
        "committed tercile cuts and placebo deflators, so each candidate cell "
        "is the identical statistic the floor measured on the real half. "
        "reference_moments is called with person_ids=None because the frames "
        "are already restricted to the seed's holdout egos and the "
        "both-orientation mirror records carry synthetic (non-holdout) ego "
        "ids that an id filter would drop."
    ),
    "shared_earnings_ratio_is_a_per_input_shape_moment": (
        "shared_earnings_ratio recomputes the couple combined-axis adjacent "
        "quintile-cutpoint RATIOS on each input (the floor does the same on "
        "each real half); the cut_provenance frozen categories are the "
        "tercile cuts and placebo deflators, which reference_moments takes "
        "from the committed panel. The quintile ratio is a scale-free "
        "distribution-shape moment measured on the candidate's own couples."
    ),
    "rng_topology": (
        "split seed = the raw gate seed (default_rng(seed) inside "
        "split_panel_by_person); draw stream = default_rng(5200 + k), k=0..19 "
        "(consumed by the certified simulate); the candidate joints draw from "
        "four independent children of SeedSequence([5200+k, 0x2C1]) (spouse "
        "decile, spouse value, spouse age gap, event window). default_rng("
        "4200 + seed) is recorded as provenance only, not consumed."
    ),
    "spouse_age_inert_for_gated_cells": (
        "component 3 (spouse age from the certified age-gap distributions) is "
        "drawn and recorded but reads into no gated cell; it is included to "
        "honor the five-component composition, mirroring the certified core "
        "where the age gap is retained for RNG topology."
    ),
}

#: Family -> a mechanism note when the family does not fully clear (evidence
#: -driven; filled from the realized decomposition).
_FAMILY_MECHANISMS = {
    "assort_mating": (
        "who-marries-whom by earnings tercile: the train-fitted "
        "decile-conditional assortative kernel emitted in both directions"
    ),
    "first_marriage_by_earnings": (
        "earnings-tercile-conditional first-marriage timing: the certified "
        "marital core conditions on age x sex x cohort, NOT the earnings "
        "axis, so a tercile cell is the age/cohort-hazard composition over "
        "that tercile's age/cohort mix"
    ),
    "remarriage_by_earnings": (
        "earnings-tercile-conditional remarriage timing: same "
        "core-vs-earnings composition as first marriage, over "
        "post-dissolution person-years"
    ),
    "earnings_around_marriage": (
        "around-marriage earnings dynamics from the train shift kernel, "
        "detrended by the committed placebo deflator"
    ),
    "earnings_around_divorce": (
        "around-divorce earnings dynamics from the train shift kernel, "
        "detrended by the committed placebo deflator"
    ),
    "shared_earnings_ratio": (
        "couple combined-axis distribution shape from the drawn spouse axis "
        "values plus the real ego axis"
    ),
}


# --------------------------------------------------------------------------
# JSON coercion
# --------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_of(cell: str) -> str:
    return cell.split(".")[0]


# --------------------------------------------------------------------------
# Contract + data loading
# --------------------------------------------------------------------------
def load_locked_tolerances() -> tuple[dict[str, float], list[str]]:
    """The 27 locked gate-2c tolerances + the report-only cells, from the
    LOCKED gates.yaml block."""
    gates = yaml.safe_load(GATES.read_text())
    th = gates["gates"]["gate_2"]["gate_2c"]["thresholds"]
    if not th.get("locked"):
        raise RuntimeError("gate_2c thresholds are not locked")
    tol: dict[str, float] = {}
    for view in th["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    if len(tol) != N_GATED:
        raise RuntimeError(f"expected {N_GATED} gated cells, got {len(tol)}")
    return tol, list(th["report_only"])


def _cross_check_tolerances(tol: dict[str, float], floor: dict) -> dict:
    """Confirm the locked tolerances equal the frozen floor's
    draft_thresholds (round(mean + 4*sd, 3), capped at ln(1.5))."""
    cells = floor["draft_thresholds"]["cells"]
    max_dev = 0.0
    for cell, t in tol.items():
        max_dev = max(max_dev, abs(t - cells[cell]["log_ratio_abs_max"]))
    gated = set(floor["gate_partition"]["gate_eligible"])
    return {
        "tolerances_match_floor_draft_thresholds": max_dev <= EXACT_ATOL,
        "max_abs_tolerance_deviation": max_dev,
        "tolerance_cells_equal_floor_gate_eligible": set(tol) == gated,
        "k": floor["draft_thresholds"]["k"],
        "rounding": floor["draft_thresholds"]["rounding"],
        "t_max": floor["draft_thresholds"]["t_max"],
    }


def load_all() -> dict[str, Any]:
    """Load the couple panel, the certified marital panel + fit sources, and
    the committed earnings axis -- once, shared across all seeds."""
    params = load_ssa_parameters()
    marriage_records = marriage.marriage_history()
    death_records = deaths.read_death_records()
    birth_records = births.birth_history()
    demo = panels.demographic_panel()
    demo_pos = demo[demo["weight"] > 0]
    person_weight = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    earnings_panel = family.family_earnings_panel()
    ce_panel = ce.build_couple_panel(
        params=params,
        marriage_records=marriage_records,
        earnings_panel=earnings_panel,
        death_records=death_records,
        person_weight=person_weight,
    )
    mpanel = transitions.build_marital_panel(
        marriage_records, death_records, person_weight
    )
    order_map = marriage_order_map(marriage_records)
    axis = cfs.build_committed_axis(
        ce_panel,
        earnings_panel=earnings_panel,
        marriage_records=marriage_records,
        params=params,
        person_weight=person_weight,
    )
    return {
        "params": params,
        "ce_panel": ce_panel,
        "mpanel": mpanel,
        "demographic_panel": demo,
        "marriage_records": marriage_records,
        "birth_records": birth_records,
        "order_map": order_map,
        "axis": axis,
    }


# --------------------------------------------------------------------------
# Precheck (hard stop): reproduce the floor exactly
# --------------------------------------------------------------------------
def run_precheck(ce_panel: ce.CoupleEarningsPanel, floor: dict) -> dict:
    ref = ce.reference_moments(ce_panel, weighted=True)
    committed = floor["reference_moments"]
    ref_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )

    committed_ho = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    rate_a_dev = 0.0
    sha_all = True
    per_seed = {s["seed"]: s for s in floor["noise_floor_per_seed"]}
    gated = set(floor["gate_partition"]["gate_eligible"])
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            ce_panel.attrs, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        sha_all = sha_all and (
            digest == committed_ho[seed]["holdout_person_id_sha256"]
        )
        cells_a = ce.reference_moments(ce_panel, set(ids), weighted=True)
        for key in gated:
            rate_a_dev = max(
                rate_a_dev,
                abs(
                    cells_a[key]["rate"]
                    - per_seed[seed]["cells"][key]["rate_a"]
                ),
            )
    result = {
        "all_reproduced_exactly": (
            ref_dev <= EXACT_ATOL and rate_a_dev <= EXACT_ATOL and sha_all
        ),
        "reference_moments_max_abs_deviation": ref_dev,
        "rate_a_max_abs_deviation": rate_a_dev,
        "holdout_sha256_all_match": sha_all,
        "atol": EXACT_ATOL,
    }
    if not result["all_reproduced_exactly"]:
        raise RuntimeError(f"precheck failed (hard stop): {result}")
    return result


# --------------------------------------------------------------------------
# Score one seed: fit on side B, K=20 draws on side A, mean-of-K estimator
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    data: dict[str, Any],
    floor: dict,
    tol: dict[str, float],
    report_only: list[str],
) -> dict[str, Any]:
    started = time.time()
    ce_panel = data["ce_panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        ce_panel.attrs, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = cfs.fit_couple_model_v1(
        ce_panel,
        data["mpanel"],
        demographic_panel=data["demographic_panel"],
        marriage_records=data["marriage_records"],
        birth_records=data["birth_records"],
        marriage_order_map=data["order_map"],
        axis=data["axis"],
        train_ids=ids_b,
    )

    draw_seeds = [DRAW_SEED_BASE + k for k in range(N_DRAWS)]
    committed_cells = {s["seed"]: s for s in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]

    all_keys = list(tol) + list(report_only)
    per_draw_rate: dict[str, list[float]] = {k: [] for k in all_keys}
    per_draw_den: dict[str, list[float]] = {k: [] for k in all_keys}
    per_draw_nev: dict[str, list[int]] = {k: [] for k in all_keys}
    sim_diagnostics: list[dict[str, Any]] = []

    for draw_seed in draw_seeds:
        sim_panel, diag = cfs.simulate_draw_v1(
            ce_panel, data["mpanel"], model, data["axis"], ids_a, draw_seed
        )
        cand = ce.reference_moments(sim_panel, weighted=True)
        for key in all_keys:
            cell = cand[key]
            per_draw_rate[key].append(float(cell["rate"]))
            per_draw_den[key].append(float(cell["den_wt"]))
            per_draw_nev[key].append(int(cell["n_events"]))
        sim_diagnostics.append(diag)

    def _score(key: str, gated: bool) -> dict[str, Any]:
        rates = np.asarray(per_draw_rate[key], dtype=float)
        dens = np.asarray(per_draw_den[key], dtype=float)
        rate_a = float(committed_cells[key]["rate_a"])
        rbar = float(rates.mean())
        n_defined = int((dens > 0).sum())
        undefined = [draw_seeds[k] for k in range(N_DRAWS) if dens[k] <= 0.0]
        if rbar > 0 and rate_a > 0:
            score = abs(math.log(rbar / rate_a))
        else:
            score = math.inf
        per_draw_ln = [
            abs(math.log(r / rate_a)) if r > 0 and rate_a > 0 else math.inf
            for r in rates
        ]
        rec = {
            "rbar": rbar,
            "rate_a": rate_a,
            "score": score,
            "n_events_candidate_mean": float(np.mean(per_draw_nev[key])),
            "per_draw_rate": [float(r) for r in rates],
            "per_draw_rate_sd": float(rates.std(ddof=1)),
            "max_per_draw_abs_ln": (max(per_draw_ln) if per_draw_ln else None),
            "n_draws_defined": n_defined,
            "undefined_draw_seeds": undefined,
        }
        if gated:
            rec["tolerance"] = tol[key]
            rec["pass"] = bool(score <= tol[key])
        return rec

    gated_cells = {k: _score(k, True) for k in tol}
    report_only_cells = {k: _score(k, False) for k in report_only}

    n_gated_pass = sum(1 for r in gated_cells.values() if r["pass"])
    undefined_gated_draws = [
        {"cell": k, "draw_seeds": r["undefined_draw_seeds"]}
        for k, r in gated_cells.items()
        if r["undefined_draw_seeds"]
    ]
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "estimator": "mean_over_K20_draws",
        "draw_seeds": draw_seeds,
        "sim_seed_single_draw_provenance": SIM_SEED_PROVENANCE_BASE + seed,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(n_gated_pass == len(tol)),
        "gated_cells": gated_cells,
        "report_only_cells": report_only_cells,
        "undefined_gated_draws": undefined_gated_draws,
        "fit_meta": model.meta,
        "sim_diagnostics_mean": _mean_diag(sim_diagnostics),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def _mean_diag(diags: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "n_marriages",
        "n_directed_couples",
        "n_marital_events",
        "n_marital_exposure_person_years",
        "n_event_windows",
        "spouse_age_gap_mean",
        "spouse_age_mean",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in diags if d.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


# --------------------------------------------------------------------------
# Assemble the verdict + decomposition + fresh-run schema
# --------------------------------------------------------------------------
def build_verdict(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    seed_pass = {str(s["seed"]): bool(s["seed_pass"]) for s in per_seed}
    n_pass = sum(seed_pass.values())
    failing = []
    for s in per_seed:
        for cell, rec in s["gated_cells"].items():
            if not rec["pass"]:
                failing.append(
                    {
                        "cell": cell,
                        "seed": s["seed"],
                        "family": _family_of(cell),
                        "score": rec["score"],
                        "tolerance": rec["tolerance"],
                        "score_over_tolerance": (
                            rec["score"] / rec["tolerance"]
                            if math.isfinite(rec["score"])
                            else None
                        ),
                    }
                )
    return {
        "n_gate_seeds": len(per_seed),
        "n_gated_cells": len(tol),
        "seed_pass": seed_pass,
        "n_seeds_pass": n_pass,
        "gate_2c_pass": bool(n_pass >= 4),
        "rule": (
            "seed passes iff every gated cell |ln(rbar/rate_a)| <= tolerance; "
            "gate passes iff >= 4 of 5 gate seeds pass"
        ),
        "all_failing_gated_cells": sorted(
            failing, key=lambda r: (r["seed"], r["cell"])
        ),
    }


def per_family_decomposition(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    families: dict[str, list[str]] = {}
    for cell in tol:
        families.setdefault(_family_of(cell), []).append(cell)

    out: dict[str, Any] = {}
    for fam, cells in families.items():
        n_cell_seed = len(cells) * len(per_seed)
        n_pass = 0
        per_seed_passed = {}
        cell_abs_ln: dict[str, list[float]] = {c: [] for c in cells}
        for s in per_seed:
            passed = 0
            for c in cells:
                rec = s["gated_cells"][c]
                if rec["pass"]:
                    n_pass += 1
                    passed += 1
                if math.isfinite(rec["score"]):
                    cell_abs_ln[c].append(rec["score"])
            per_seed_passed[str(s["seed"])] = passed
        # worst cell = highest mean score / tolerance across seeds
        worst_cell = None
        worst_ratio = -1.0
        worst_mean_abs_ln = None
        for c in cells:
            if not cell_abs_ln[c]:
                continue
            mean_abs = float(np.mean(cell_abs_ln[c]))
            ratio = mean_abs / tol[c]
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_cell = c
                worst_mean_abs_ln = mean_abs
        cell_seed_pass_rate = n_pass / n_cell_seed if n_cell_seed else None
        out[fam] = {
            "n_cells": len(cells),
            "cells": sorted(cells),
            "n_cell_seed": n_cell_seed,
            "n_cell_seed_pass": n_pass,
            "cell_seed_pass_rate": cell_seed_pass_rate,
            "per_seed_cells_passed": per_seed_passed,
            "worst_cell": worst_cell,
            "worst_cell_mean_abs_ln": worst_mean_abs_ln,
            "worst_cell_tolerance": tol[worst_cell] if worst_cell else None,
            "worst_cell_mean_ln_over_tol": (
                round(worst_ratio, 4) if worst_ratio >= 0 else None
            ),
            "mechanism": _FAMILY_MECHANISMS[fam],
        }
    return out


def fresh_run_artifact_schema(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    cell_index = sorted(tol)
    seed_index = [s["seed"] for s in per_seed]
    by_seed = {s["seed"]: s for s in per_seed}
    rates = [
        [
            [
                by_seed[s]["gated_cells"][c]["per_draw_rate"][k]
                for s in seed_index
            ]
            for c in cell_index
        ]
        for k in range(N_DRAWS)
    ]
    n_undefined = 0
    undefined = []
    for s in per_seed:
        for cell, rec in s["gated_cells"].items():
            if rec["undefined_draw_seeds"]:
                n_undefined += len(rec["undefined_draw_seeds"])
                undefined.append(
                    {
                        "cell": cell,
                        "seed": s["seed"],
                        "draw_seeds": rec["undefined_draw_seeds"],
                    }
                )
    dispersion = {}
    for s in per_seed:
        for cell, rec in s["gated_cells"].items():
            dispersion[f"{cell}|seed{s['seed']}"] = {
                "per_draw_rate_sd": rec["per_draw_rate_sd"],
                "max_per_draw_abs_ln": rec["max_per_draw_abs_ln"],
                "rbar": rec["rbar"],
            }
    return {
        "per_draw_per_cell_rates": {
            "required": True,
            "shape": [N_DRAWS, len(cell_index), len(seed_index)],
            "shape_dims": "K_draws x gated_cells x gate_seeds",
            "k_index_draw_seeds": [DRAW_SEED_BASE + k for k in range(N_DRAWS)],
            "cell_index": cell_index,
            "seed_index": seed_index,
            "rates": rates,
            "note": (
                "r[k][cell][seed]; rbar_candidate,s = mean over k; "
                "|ln(rbar / rate_a)| recomputes cell-by-cell"
            ),
        },
        "undefined_draw_rule": {
            "required": True,
            "pre_specified": True,
            "n_undefined_gated_draws": n_undefined,
            "undefined_gated_draws": undefined,
            "run_invalidated": bool(n_undefined > 0),
            "rule": (
                "if any gated cell's rate is undefined on any draw (empty "
                "simulated denominator) the run is invalidated; no draw is "
                "dropped, substituted, or re-rolled"
            ),
        },
        "per_draw_dispersion_disclosure": {
            "required": True,
            "gated": False,
            "report_only": True,
            "note": (
                "report-only: no dispersion cap gates the run; the disclosure "
                "shows whether a passing 20-draw mean conceals a wild draw"
            ),
            "cells": dispersion,
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(out_path: Path, verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    tol, report_only = load_locked_tolerances()
    floor = json.loads(FLOOR_RUN.read_text())
    tolerance_check = _cross_check_tolerances(tol, floor)
    if not tolerance_check["tolerances_match_floor_draft_thresholds"]:
        raise RuntimeError(f"tolerance cross-check failed: {tolerance_check}")
    if not tolerance_check["tolerance_cells_equal_floor_gate_eligible"]:
        raise RuntimeError("gated cell set != floor gate_eligible")

    if verbose:
        print(f"loading data ({len(tol)} gated cells)...")
    data = load_all()

    if verbose:
        print("precheck (hard stop: reproduce the floor)...")
    precheck = run_precheck(data["ce_panel"], floor)

    per_seed = []
    for seed in GATE_SEEDS:
        if verbose:
            print(f"scoring seed {seed} (fit side B, K={N_DRAWS} draws)...")
        rec = score_seed(seed, data, floor, tol, report_only)
        per_seed.append(rec)
        if verbose:
            print(
                f"  seed {seed}: {rec['n_gated_pass']}/{len(tol)} gated pass, "
                f"seed_pass={rec['seed_pass']} ({rec['elapsed_seconds']}s)"
            )

    verdict = build_verdict(per_seed, tol)
    decomposition = per_family_decomposition(per_seed, tol)
    fresh_schema = fresh_run_artifact_schema(per_seed, tol)

    if fresh_schema["undefined_draw_rule"]["run_invalidated"]:
        raise RuntimeError(
            "run invalidated: a gated cell was undefined on some draw "
            f"({fresh_schema['undefined_draw_rule']['n_undefined_gated_draws']}"
            " undefined gated draws)"
        )

    meta = data["ce_panel"].meta
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": "gate2c_hazard_v1",
        "gate": "gate_2c",
        "candidate": "candidate 1",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "one_shot": (
            "one-shot scored run registered on issue #42 comment "
            f"{REGISTRATION_POINTER}; independent verification by rerun; "
            "publishes regardless; no holdout tuning"
        ),
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "module": "populace_dynamics.models.couple_formation_sim_v1",
            "five_components": {
                "1_marriage_events": (
                    "certified tranche-2a registry CANDIDATE_16 (refit side B,"
                    " simulate side A at 5200+k)"
                ),
                "2_who_marries_whom": (
                    "train-fitted P(spouse decile | own decile, age band, "
                    "sex), directed both-orientation emission"
                ),
                "3_spouse_age": (
                    "certified spousal_age_gaps + draw_spousal_gaps "
                    "(recorded; not gated)"
                ),
                "4_event_window_dynamics": (
                    "train around-event post/pre ratio shift kernels, "
                    "placebo-detrended by the committed deflator"
                ),
                "5_shared_earnings_cells": (
                    "combined-axis cutpoint ratios per the locked cell "
                    "definitions"
                ),
            },
            "kernel_smoothing_alpha": cfs.KERNEL_SMOOTHING_ALPHA,
            "n_deciles": cfs.N_DECILES,
            "certified_spec_sha256": cfs.CERTIFIED_SPEC.sha256,
            "certified_component_implementation_ids": dict(
                per_seed[0]["fit_meta"]["component_implementation_ids"]
            ),
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "outer_seed_provenance_only": (
                "numpy.random.default_rng(4200 + seed) recorded, not consumed"
            ),
            "split": (
                "COUPLE-DISJOINT 50/50 by attrs.component_id, "
                "default_rng(seed); side A = holdout, side B = train"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "statistic": (
                "|ln(rbar_candidate,s / rate_a,s)|, rbar the 20-draw mean "
                "rate, scored once (NOT the mean of per-draw scores)"
            ),
            "pass_rule": (
                "seed passes iff all 27 gated cells hold; gate passes iff "
                ">= 4 of 5 seeds pass"
            ),
            "tolerance_source": "gates.yaml gate_2.gate_2c (LOCKED)",
            "tolerance_cross_check_vs_floor": tolerance_check,
        },
        "fresh_run_artifact_schema": fresh_schema,
        "data": {
            "holdout_basis": floor["holdout_basis"],
            "floor_run": "runs/gate2c_floors_v1.json",
            "floor_run_sha256": _sha256_file(FLOOR_RUN),
            "n_directed_couples": meta["n_directed_couples"],
            "n_earnings_supply_persons": meta["n_earnings_supply_persons"],
            "pe_us_revision": meta.get("pe_us_revision"),
        },
        "precheck": precheck,
        "per_seed": per_seed,
        "verdict": verdict,
        "per_family_decomposition": decomposition,
        "revision_pins": {
            "artifact_schema_version": SCHEMA_VERSION,
            "pe_us_revision": meta.get("pe_us_revision"),
            "certified_spec_sha256": cfs.CERTIFIED_SPEC.sha256,
            "floor_run_sha256": _sha256_file(FLOOR_RUN),
            "base_sha": floor.get("revision_pins", {}).get(
                "origin_master_sha"
            ),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    artifact = _json_safe(artifact)
    artifacts.write_new(out_path, artifact)
    if verbose:
        v = verdict
        print(
            f"\nGATE-2C candidate 1: {v['n_seeds_pass']}/5 seeds pass "
            f"(gate_2c_pass={v['gate_2c_pass']}); wrote {out_path} "
            f"({artifact['elapsed_seconds']}s)"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
