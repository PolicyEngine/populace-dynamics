"""Gate-2b candidate 9: the cohort-scoped fertility lift.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b; 46 gated household/relationship cells, K=20 mean-over-draws,
per-seed train/holdout protocol). Registered on issue #42 comment 4948839837
BEFORE this run; published REGARDLESS of verdict. No holdout tuning.

Candidate 9 is candidate 8 (registration 4948604739; PR #147) with EXACTLY ONE
frozen delta against the graded candidate-8 collateral (grading 4948838962): the
delta-1 completed-fertility swap is CONFINED to the forensics-5-measured deficit
cohorts x sex -- ``{55-64, 65-74} x male`` and ``{45-54, 65-74} x female`` --
with every OTHER cohort retaining the sim's own completed-family-size
distribution (candidate-7 behavior). Everything else in candidate 8 is carried
BYTE-FAITHFULLY: the delta-2 cohabitation-overlay lift, the delta-3 band-signed
retention refit + link-coverage inclusion, and every carried candidate-7 family.

The scope change is a WRITE GATE on candidate 8's own fertility lift (see
``household_composition_sim_v9.apply_scoped_fertility_core_lift``): the composed
frame, the isolated ``SeedSequence([draw_seed, 0xC8])`` and every per-cohort
draw are reproduced bit-for-bit from candidate 8, so (i) the deficit cohorts get
candidate 8's lift UNCHANGED (byte-identical), (ii) every non-deficit cohort
reverts to candidate 7 byte-identically -- the four candidate-8 collateral cells
(35-44|m/f, 45-54|m, 55-64|f) return to their cleared state -- and (iii) delta 2,
delta 3 and every carry are byte-identical to candidate 8.

The artifact records the PRE-RUN train-side analytic check (registration
4948839837): the scoped lift's implied counterfactual for ``hh_size.5+`` (the
priced aggregate), the two held deficit-male cells and the four collateral
cells, computed BEFORE the scored holdout run.

Estimator, artifact schema, undefined-draw rule and protocol are the locked
contract. Artifact ``runs/gate2b_hazard_v9.json``; ``artifacts.write_new``
refuses to overwrite. Reproduce with ``.venv-gate`` and the PSID products staged
(``POPULACE_DYNAMICS_PSID_DIR``; ``POPULACE_DYNAMICS_PE_US_DIR``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from populace_dynamics import artifacts
from populace_dynamics.data import (
    births,
    deaths,
    marriage,
    panels,
    relmap,
    transitions,
)
from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition_sim_v2 as hcs2
from populace_dynamics.models import household_composition_sim_v3 as hcs3
from populace_dynamics.models import household_composition_sim_v8 as hcs8
from populace_dynamics.models import household_composition_sim_v9 as hcs9

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v9.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v9"
RUN_NAME = "gate2b_hazard_v9"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS_RUN = ROOT / "runs" / "gate2b_forensics5_v1.json"
CANDIDATE_ARTIFACTS = {
    n: ROOT / "runs" / f"gate2b_hazard_v{n}.json" for n in range(1, 9)
}
REGISTRATION_POINTER = "4948839837"
CANDIDATE8_REGISTRATION_POINTER = "4948604739"
CANDIDATE8_GRADING_POINTER = "4948838962"  # c9 designs against this
FORENSICS_REGISTRATION_POINTER = "4948430423"
FORENSICS_GRADING_POINTER = "4948603337"
SPEC_REGISTRATION = (
    "issue #42 comment 4948839837: gate-2b candidate 9, the cohort-scoped "
    "fertility lift -- the delta-1 completed-fertility swap confined to the "
    "forensics-5 deficit cohorts ({55-64,65-74} x male, {45-54,65-74} x "
    "female); all other cohorts revert to the sim's own distribution (c7 "
    "behavior); everything else in candidate 8 carried byte-faithfully"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12
FIT_VS_RAW_SEED = 0

#: Families carried BYTE-IDENTICAL to candidate 8 (and candidate 7): no
#: candidate-9 change touches their streams. Includes ALL of coresident_spouse
#: (the delta-2 25-34|female lift is itself carried byte-faithfully from c8).
CANDIDATE8_STRICT_CARRIED_FAMILIES = (
    "coresident_parent",
    "coresident_spouse",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_grandchild",
)
CANDIDATE8_CLEARED_FAMILIES = (
    "coresident_parent",
    "coresident_spouse",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_grandchild",
    "hh_size",
)
MULTIGEN_MARGINAL_CELLS_PREFIX = ("multigen.",)
MULTIGEN_MARGINAL_CELLS_EXACT = ("multigen_entry", "multigen_exit")

#: The deficit-scope coresident_child cells: candidate 8's fertility lift is kept
#: UNCHANGED on these (byte-identical to c8, delta 1 + delta 3).
SCOPE_CHILD_CELLS = tuple(hcs9.FERTILITY_LIFT_CELLS)
#: The non-deficit coresident_child cells: revert to candidate 7 byte-identically
#: (the four candidate-8 collateral cells plus the younger reverted cohorts).
REVERTED_CHILD_CELLS = tuple(hcs9.REVERTED_CHILD_CELLS)
COLLATERAL_CELLS = tuple(hcs9.COLLATERAL_CELLS)

PER_DELTA_TARGET_FAMILY = {
    "delta_1_fertility_core_lift_SCOPED": ["coresident_child", "hh_size"],
    "delta_2_cohab_overlay_lift_carried": ["coresident_spouse"],
    "delta_3_retention_link_refit_carried": ["coresident_child"],
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.50-0.70",
    "majority_side": True,
    "wide_on_purpose": (
        "the four collateral cells should revert to their cleared state and "
        "the deficit-cohort clears should hold (their lift is unchanged), but "
        "hh_size.5+ is the priced uncertainty: candidate 8's GLOBAL lift "
        "delivered its clearance, and if a material share came through "
        "middle-cohort large families the scoped lift under-delivers it"
    ),
    "named_expectations": [
        "the four candidate-8 collateral cells clear (35-44|male, 35-44|"
        "female, 45-54|male, 55-64|female revert to candidate 7)",
        "coresident_child.55-64|male holds its clear (deficit cohort; lift "
        "unchanged from candidate 8)",
        "coresident_child.65-74|male holds its clear (deficit cohort; delta-1 "
        "+ delta-3 unchanged from candidate 8)",
        "coresident_spouse.25-34|female and every carried family hold "
        "byte-identically (delta 2 + carries unchanged from candidate 8)",
        "modal residual = hh_size.5+ (the scoped lift's priced uncertainty)",
    ],
    "modal_outcome_if_fail": (
        "hh_size.5+ on 1-2 seeds, everything else clean -- if it fails, "
        "candidate 10 needs the middle-cohort share of the hh_size.5+ mass "
        "measured (a one-question forensics)"
    ),
    "named_residual_risk": (
        "hh_size.5+ under-delivery if the middle cohorts contributed a "
        "material share of candidate 8's global large-family lift; the "
        "pre-run analytic check records middle_cohort_share_of_lift"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT grade "
        "it. Per the standing addendum, any pass faces independent "
        "verification before entering the record."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "the_one_delta_vs_candidate8": (
        "Candidate 9's SOLE change against candidate 8 is the confinement of "
        "the delta-1 completed-fertility swap to the forensics-5-measured "
        "deficit cohorts x sex (55-64|male, 65-74|male, 45-54|female, "
        "65-74|female). It is realized as a WRITE GATE on candidate 8's "
        "apply_fertility_core_lift: every per-cohort random draw (target-bucket "
        "resample, coresidence re-emission uniforms, hh_size pool picks) is "
        "drawn for ALL composition cohorts in candidate 8's exact order and "
        "count, so the RNG is consumed identically; only the assignment back "
        "into coresident_child / hh_size is confined to the deficit cohorts."
    ),
    "deficit_cohorts_lift_unchanged": (
        "The deficit cohorts get candidate 8's lift UNCHANGED -- byte-identical "
        "coresident_child and hh_size on every draw and seed, because their "
        "draws and their writes are candidate 8's. Delta 3 (band-signed "
        "retention + link-coverage) operates only on the deficit cohorts "
        "(RETENTION_EXIT_CELLS + LINK_COVERAGE_CELLS), whose delta-1 output is "
        "candidate 8's, so the deficit cells' final coresident_child is "
        "byte-identical to candidate 8."
    ),
    "non_deficit_cohorts_revert_to_candidate7": (
        "Every non-deficit composition cohort reverts to the candidate-7 "
        "(unlifted) composition byte-identically: the four candidate-8 "
        "collateral cells (35-44|male, 35-44|female, 45-54|male, 55-64|female) "
        "return to their candidate-7 cleared state -- the mechanism the "
        "candidate-8 grading localized (the global lift over-read a train-side "
        "proof as holdout-transportable) -- as do the younger reverted child "
        "cohorts (15-24, 25-34 both sexes)."
    ),
    "carried_deltas_byte_identical": (
        "Delta 2 (cohabitation-overlay lift at 25-34|female) and every carried "
        "candidate-7 family (coresident_parent, coresident_spouse, multigen "
        "stock + transitions, parental_home_exit, coresident_grandchild) are "
        "byte-identical to candidate 8: their 0xC8 substreams are spawned "
        "independently of the fertility stream, and the carries come from the "
        "candidate-7 composition unchanged."
    ),
    "hh_size_5plus_priced_uncertainty": (
        "hh_size.5+ is the priced uncertainty. Candidate 8's GLOBAL lift raised "
        "it 0.127 -> 0.144 (cleared 5/5); the scoped lift delivers only the "
        "deficit cohorts' share of that mass. The pre-run analytic check "
        "decomposes candidate 8's hh_size.5+ lift into the deficit-cohort "
        "(scoped) contribution and the middle-cohort share the scoping forgoes, "
        "predicting the scoped counterfactual before the holdout is scored."
    ),
    "pre_run_analytic_check": (
        "Recorded BEFORE the scored holdout run (train side B, deterministic): "
        "the scoped lift's implied Q15 law-of-total-probability counterfactual "
        "for hh_size.5+, 55-64|male, 65-74|male and the four collateral cells, "
        "scored against the observed side-B reference at the locked tolerance. "
        "The run proceeds REGARDLESS of what the check predicts."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "VERBATIM from candidate 8 / candidate 7: each simulated side-A holdout "
        "person is seeded from their OWN observed state at their first 2b wave, "
        "then evolved with train-fitted hazards; no parameter is estimated from "
        "side A. The candidate-8 fitted structures (D_train per cohort, the "
        "delta-3 channels, the delta-2 lift) are reused verbatim, fitted on "
        "side B only (no holdout tuning surface)."
    ),
    "gates_yaml_path": (
        "The locked block is gates.yaml gates.gate_2.gate_2b.thresholds; its 46 "
        "gated tolerances and the floor gate_partition are read at runtime, "
        "never hardcoded (candidate convention)."
    ),
}


def load_gate2b_thresholds() -> dict[str, Any]:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_2"]["gate_2b"]["thresholds"]


def gated_tolerances(thresholds: dict[str, Any]) -> dict[str, float]:
    tol: dict[str, float] = {}
    for view in thresholds["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _order_map(mh_records: pd.DataFrame) -> pd.DataFrame:
    ep = marriage.marriage_episodes(mh_records)
    ep = ep[ep["start_year"].notna()].copy()
    ep["start_year"] = ep["start_year"].astype("int64")
    ep = ep.sort_values(["person_id", "start_year"])
    ep["order"] = ep.groupby("person_id").cumcount() + 1
    return ep[["person_id", "start_year", "order"]].drop_duplicates(
        ["person_id", "start_year"]
    )


def load_all() -> dict[str, Any]:
    rel_map = relmap.relationship_map()
    demo = panels.demographic_panel()
    sex = deaths.read_death_records()
    roster = hc.household_roster(rel_map)
    person_waves = hc.join_demographics(roster, demo, sex)
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    hh = hc.HouseholdCompositionPanel(person_waves=person_waves, attrs=attrs)

    mh = marriage.marriage_history()
    bh = births.birth_history()
    demo_pos = demo[demo.weight > 0]
    person_weight = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    mpanel = transitions.build_marital_panel(mh, sex, person_weight)
    order_map = _order_map(mh)

    father_links_child = hcs3.father_link_births_with_child(bh)
    parent_pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    fu_sizes = hcs3.family_unit_sizes(rel_map)
    from populace_dynamics.models import household_composition_sim_v4 as hcs4
    from populace_dynamics.models import household_composition_sim_v5 as hcs5

    legal_flag = hcs4.legal_spouse_flag(rel_map)
    parent_counts = hcs5.parent_link_counts(rel_map)
    marital_by_year = hcs3._father_marital_by_year(mpanel)
    child_record_expo = hcs5.build_child_record_exposure(
        father_links_child, parent_pairs, marital_by_year, demo, rel_map
    )
    return {
        "hh": hh,
        "mpanel": mpanel,
        "demo": demo,
        "mh": mh,
        "bh": bh,
        "rel_map": rel_map,
        "order_map": order_map,
        "father_links_child": father_links_child,
        "parent_pairs": parent_pairs,
        "marital_by_year": marital_by_year,
        "fu_sizes": fu_sizes,
        "legal_flag": legal_flag,
        "parent_counts": parent_counts,
        "child_record_expo": child_record_expo,
    }


# --------------------------------------------------------------------------
# Precheck: reproduce the committed floor bit-for-bit (candidate mirror)
# --------------------------------------------------------------------------
def run_precheck(
    hh: hc.HouseholdCompositionPanel, floor: dict[str, Any]
) -> dict[str, Any]:
    ref_w = hc.reference_moments(hh, weighted=True)
    committed_ref = floor["reference_moments"]
    ref_max = max(
        abs(ref_w[k]["rate"] - committed_ref[k]["rate"]) for k in committed_ref
    )
    committed_ho = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    committed_ns = {p["seed"]: p for p in floor["noise_floor_per_seed"]}
    per_seed = []
    rate_a_max = 0.0
    sha_all_ok = True
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            hh.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        sha_ok = digest == committed_ho[seed]["holdout_person_id_sha256"]
        sha_all_ok = sha_all_ok and sha_ok
        cells_a = hc.reference_moments(hh, set(ids), weighted=True)
        committed_cells = committed_ns[seed]["cells"]
        seed_dev = max(
            abs(cells_a[k]["rate"] - committed_cells[k]["rate_a"])
            for k in committed_cells
        )
        rate_a_max = max(rate_a_max, seed_dev)
        per_seed.append(
            {
                "seed": seed,
                "holdout_sha256_match": bool(sha_ok),
                "n_holdout": len(ids),
                "rate_a_max_abs_deviation": float(seed_dev),
            }
        )
    ok = bool(
        ref_max <= EXACT_ATOL and rate_a_max <= EXACT_ATOL and sha_all_ok
    )
    return {
        "note": (
            "Hard-stop precheck (candidate mirror): the scoring path must "
            "reproduce every committed reference moment and per-gate-seed "
            "rate_a bit-for-bit and each holdout-id sha256 before simulating."
        ),
        "reference_moments_max_abs_deviation": float(ref_max),
        "n_reference_cells": len(committed_ref),
        "per_seed": per_seed,
        "rate_a_max_abs_deviation": float(rate_a_max),
        "holdout_sha256_all_match": bool(sha_all_ok),
        "all_reproduced_exactly": ok,
    }


# --------------------------------------------------------------------------
# Per-seed scoring (mean over K=20 draws)
# --------------------------------------------------------------------------
def _delta_stats(
    model: hcs8.HouseholdCompositionModelV8, dmean, dmean_dict
) -> dict:
    fert = model.meta["fertility_core_lift"]
    d3 = model.delta3_fit["per_cell"]
    return {
        "delta_1_fertility_core_lift_SCOPED": {
            "scope_cells": list(hcs9.FERTILITY_LIFT_CELLS),
            "completed_size_dist_train_all": (
                fert["completed_size_dist_train_all"]
            ),
            "note": (
                "The train D_train is fit for every cohort (candidate 8's fit "
                "verbatim); the swap is APPLIED only to the deficit cohorts."
            ),
        },
        "delta_2_cohab_overlay_lift_carried": {
            "band": hcs8.COHAB_OVERLAY_LIFT_BAND,
            "overlay_shortfall": model.cohab_overlay_lift,
            "mean_realized_lift": dmean_dict("cohab_overlay_lift").get(
                "realized_lift"
            ),
        },
        "delta_3_retention_link_refit_carried": {
            "retention_link_shift": model.retention_link_shift,
            "per_cell_channels": {
                cell: {
                    "exit_origin": d3[cell]["exit_origin"],
                    "link_coverage": d3[cell]["link_coverage"],
                    "v7_persistence_enumeration_interaction": d3[cell][
                        "v7_persistence_enumeration_interaction"
                    ],
                    "applied_shift": d3[cell]["applied_shift"],
                    "closes": d3[cell]["closes"],
                }
                for cell in d3
            },
        },
    }


def score_seed(
    seed: int,
    data: dict[str, Any],
    floor: dict[str, Any],
    tol: dict[str, float],
    report_only: list[str],
    forensics: dict[str, Any],
    verbose: bool,
    compute_fit_checks: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    hh = data["hh"]
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs9.fit_household_model_v9(
        hh,
        data["mpanel"],
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b,
        father_links_child=data["father_links_child"],
        parent_pairs=data["parent_pairs"],
        marital_by_year=data["marital_by_year"],
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    coverage = hcs2.father_link_coverage(
        data["mpanel"], model.father_links, ids_a
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]
    all_cells = sorted(set(tol) | set(report_only))
    draw_seeds = [DRAW_SEED_BASE + k for k in range(N_DRAWS)]
    per_draw_rate: dict[str, list[float]] = {c: [] for c in all_cells}
    per_draw_den: dict[str, list[float]] = {c: [] for c in all_cells}
    per_draw_nev: dict[str, list[int]] = {c: [] for c in all_cells}
    draw_diagnostics: list[dict[str, Any]] = []
    for draw_seed in draw_seeds:
        sim_panel, diag = hcs9.simulate_draw_v9(
            hh, data["mpanel"], model, ids_a, draw_seed
        )
        draw_diagnostics.append(diag)
        cand = hc.reference_moments(sim_panel, ids_a, weighted=True)
        for c in all_cells:
            cell = cand[c]
            per_draw_rate[c].append(float(cell["rate"]))
            per_draw_den[c].append(float(cell.get("den_wt", 0.0)))
            per_draw_nev[c].append(int(cell.get("n_events", 0)))

    undefined: list[dict[str, Any]] = []
    for c in sorted(tol):
        for k in range(N_DRAWS):
            if per_draw_den[c][k] <= 0.0:
                undefined.append(
                    {"cell": c, "draw_k": k, "draw_seed": draw_seeds[k]}
                )

    def score_cell(key: str) -> dict[str, Any]:
        rate_a = float(committed_cells[key]["rate_a"])
        rates = np.asarray(per_draw_rate[key], dtype=np.float64)
        rbar = float(rates.mean())
        s = (
            float(abs(math.log(rbar / rate_a)))
            if rbar > 0 and rate_a > 0
            else float("inf")
        )
        sd = float(rates.std(ddof=1)) if rates.size > 1 else 0.0
        if rate_a > 0:
            per_draw_abs_ln = [
                (abs(math.log(r / rate_a)) if r > 0 else float("inf"))
                for r in rates
            ]
            finite = [x for x in per_draw_abs_ln if math.isfinite(x)]
            max_abs_ln = float(max(finite)) if finite else None
        else:
            max_abs_ln = None
        return {
            "r_candidate": rbar,
            "rbar": rbar,
            "rate_a": rate_a,
            "n_events_candidate": int(np.mean(per_draw_nev[key])),
            "log_ratio_abs": s if math.isfinite(s) else None,
            "score": s,
            "per_draw_rate": [float(r) for r in rates],
            "per_draw_rate_sd": sd,
            "max_per_draw_abs_ln": max_abs_ln,
            "n_draws_defined": int((np.asarray(per_draw_den[key]) > 0).sum()),
        }

    gated_cells: dict[str, Any] = {}
    n_gated_pass = 0
    for key in sorted(tol):
        rec = score_cell(key)
        rec["tolerance"] = float(tol[key])
        rec["pass"] = bool(rec["score"] <= tol[key])
        n_gated_pass += rec["pass"]
        gated_cells[key] = rec

    report_cells: dict[str, Any] = {}
    for key in sorted(report_only):
        rec = score_cell(key)
        rec["gated"] = False
        report_cells[key] = rec

    seed_pass = n_gated_pass == len(tol)
    elapsed = round(time.time() - t0, 1)

    def _dmean(key: str) -> float:
        return float(np.mean([d[key] for d in draw_diagnostics]))

    def _dmean_dict(key: str) -> dict[str, Any]:
        vals = [d[key] for d in draw_diagnostics]
        out: dict[str, Any] = {}
        for k in vals[0]:
            if isinstance(vals[0][k], int | float) and not isinstance(
                vals[0][k], bool
            ):
                out[k] = float(np.mean([v[k] for v in vals]))
            else:
                out[k] = vals[0][k]
        return out

    delta_stats = _delta_stats(model, _dmean, _dmean_dict)
    delta_realized = _delta_realized_block(draw_diagnostics)
    coverage = {
        **coverage,
        "linked_episode_persistence_rho": float(
            model.linked_episode_persistence
        ),
    }
    fit_checks = None
    analytic_check = None
    if compute_fit_checks:
        fit_checks = hcs8.c8_delta_checks(model, forensics)
        reference_b = hc.reference_moments(hh, ids_b, weighted=True)
        analytic_check = hcs9.scoped_lift_analytic_check(
            model, hh, data["mpanel"], ids_b, reference_b, tol
        )

    if verbose:
        fails = [k for k, v in gated_cells.items() if not v["pass"]]
        print(
            f"seed {seed}: {n_gated_pass}/{len(tol)} gated pass "
            f"(seed_pass={seed_pass}); K={N_DRAWS}; "
            f"undefined={len(undefined)}; fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "estimator": "mean_over_K20_draws",
        "draw_seeds": draw_seeds,
        "sim_seed_single_draw_provenance": SIM_SEED_PROVENANCE_BASE + seed,
        "linked_episode_persistence_rho": float(
            model.linked_episode_persistence
        ),
        "component_meta": {
            "grandchild_coupling_age_lo": hcs8.GRANDCHILD_LO,
            "core_size_cap": hcs8.CORE_SIZE_CAP,
            "delta_stream_tag_v7": hcs8.DELTA_STREAM_TAG_V7,
            "delta_stream_tag_v8": hcs8.DELTA_STREAM_TAG_V8,
            "n_fit_draws": hcs8.N_FIT_DRAWS,
            "fertility_lift_scope_cells": list(hcs9.FERTILITY_LIFT_CELLS),
        },
        "father_link_coverage": coverage,
        "delta_stats": delta_stats,
        "delta_realized_shifts": delta_realized,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "undefined_gated_draws": undefined,
        "c9_delta_checks": fit_checks,
        "pre_run_analytic_check": analytic_check,
        "elapsed_seconds": elapsed,
    }


def _delta_realized_block(
    draw_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """The realized per-draw delta shifts, aggregated over the K draws."""
    cohab = [d["cohab_overlay_lift"] for d in draw_diagnostics]
    retention_cells = list(draw_diagnostics[0]["retention_link_refit"].keys())
    retention: dict[str, Any] = {}
    for cell in retention_cells:
        targets = [
            d["retention_link_refit"][cell]["target_shift"]
            for d in draw_diagnostics
        ]
        realized = [
            d["retention_link_refit"][cell]["realized_shift"]
            for d in draw_diagnostics
        ]
        retention[cell] = {
            "target_shift": float(np.mean(targets)),
            "mean_realized_shift": float(np.mean(realized)),
        }
    return {
        "note": (
            "The realized per-draw delta shifts (mean over K=20 draws): the "
            "carried delta-2 cohabitation-overlay lift and the carried delta-3 "
            "band-signed retention + link-coverage additive closures. Both are "
            "byte-identical to candidate 8 (independent 0xC8 substreams)."
        ),
        "delta_2_cohab_overlay_realized_lift": float(
            np.mean([c["realized_lift"] for c in cohab])
        ),
        "delta_2_cohab_overlay_target": float(cohab[0]["lift"]),
        "delta_3_retention_link_per_cell": retention,
    }


# --------------------------------------------------------------------------
# Fresh-run artifact-schema blocks ([20, 46, 5]; undefined; dispersion)
# --------------------------------------------------------------------------
def _per_draw_cube(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    cell_index = sorted(tol)
    seed_index = [s["seed"] for s in per_seed]
    by_seed = {s["seed"]: s for s in per_seed}
    rates = [
        [
            [
                float(by_seed[s]["gated_cells"][c]["per_draw_rate"][k])
                for s in seed_index
            ]
            for c in cell_index
        ]
        for k in range(N_DRAWS)
    ]
    return {
        "required": True,
        "shape": [N_DRAWS, len(cell_index), len(seed_index)],
        "shape_dims": "K_draws x gated_cells x gate_seeds",
        "k_index_draw_seeds": [DRAW_SEED_BASE + k for k in range(N_DRAWS)],
        "cell_index": cell_index,
        "seed_index": seed_index,
        "rates": rates,
        "note": (
            "r[k][cell][seed]; rbar_candidate,s = mean over k of "
            "r[k, cell, s]; the certified score is |ln(rbar / rate_a,s)|, "
            "recomputable cell-by-cell from this cube."
        ),
    }


def _undefined_block(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    all_undefined: list[dict[str, Any]] = []
    for s in per_seed:
        for u in s["undefined_gated_draws"]:
            all_undefined.append({"seed": s["seed"], **u})
    return {
        "required": True,
        "pre_specified": True,
        "n_undefined_gated_draws": len(all_undefined),
        "undefined_gated_draws": all_undefined,
        "run_invalidated": bool(all_undefined),
        "rule": (
            "if any gated cell's rate is UNDEFINED on any draw (empty "
            "simulated denominator) the run is invalidated; no draw is "
            "dropped, substituted or re-rolled."
        ),
    }


def _dispersion_block(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    for c in sorted(tol):
        per_cell[c] = {
            "per_seed": {
                s["seed"]: {
                    "per_draw_rate_sd": s["gated_cells"][c][
                        "per_draw_rate_sd"
                    ],
                    "max_per_draw_abs_ln": s["gated_cells"][c][
                        "max_per_draw_abs_ln"
                    ],
                    "rbar": s["gated_cells"][c]["rbar"],
                }
                for s in per_seed
            }
        }
    return {
        "required": True,
        "gated": False,
        "report_only": True,
        "note": (
            "REPORT-ONLY dispersion: per gated cell per seed, the sd across "
            "the K=20 draws and the worst single-draw |ln(r/rate_a)|. No "
            "dispersion cap gates the run."
        ),
        "cells": per_cell,
    }


# --------------------------------------------------------------------------
# Verdict + per-family decomposition
# --------------------------------------------------------------------------
_FAMILY_OF = [
    ("coresident_spouse", lambda k: k.startswith("coresident_spouse.")),
    ("coresident_parent", lambda k: k.startswith("coresident_parent.")),
    ("coresident_child", lambda k: k.startswith("coresident_child.")),
    (
        "coresident_grandchild",
        lambda k: k.startswith("coresident_grandchild."),
    ),
    ("multigen_stock", lambda k: k.startswith("multigen.")),
    (
        "multigen_transition",
        lambda k: k in ("multigen_entry", "multigen_exit"),
    ),
    ("parental_home_exit", lambda k: k.startswith("parental_home_exit.")),
    ("hh_size", lambda k: k.startswith("hh_size.")),
]


def _family_of(cell: str) -> str:
    for fam, pred in _FAMILY_OF:
        if pred(cell):
            return fam
    return "other"


def build_verdict(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    seed_pass = {s["seed"]: s["seed_pass"] for s in per_seed}
    n_seeds_pass = sum(seed_pass.values())
    gate_pass = n_seeds_pass >= 4
    all_failing = [
        {
            "cell": c,
            "seed": s["seed"],
            "family": _family_of(c),
            "score": s["gated_cells"][c]["score"],
            "tolerance": s["gated_cells"][c]["tolerance"],
            "r_candidate": s["gated_cells"][c]["r_candidate"],
            "rate_a": s["gated_cells"][c]["rate_a"],
        }
        for s in per_seed
        for c in sorted(tol)
        if not s["gated_cells"][c]["pass"]
    ]
    return {
        "n_gate_seeds": len(per_seed),
        "n_gated_cells": len(tol),
        "seed_pass": seed_pass,
        "n_seeds_pass": n_seeds_pass,
        "gate_2b_pass": bool(gate_pass),
        "rule": (
            "A seed passes iff every one of the 46 gated cells holds "
            "(|ln(rbar / rate_a)| <= locked tolerance); the gate passes iff "
            ">= 4 of the 5 gate seeds pass."
        ),
        "all_failing_gated_cells": all_failing,
    }


_MECHANISMS = {
    "coresident_spouse": (
        "CARRIED candidate 8 byte-identically (legal registry UNION "
        "cohabitation overlay UNION legal-residual overlay, with the delta-2 "
        "-0.045 cohab-overlay lift at 25-34|female carried unchanged)."
    ),
    "coresident_parent": (
        "CARRIED candidate 1 logistic exit hazard (byte-identical to c8)."
    ),
    "coresident_child": (
        "candidate-8 composition with the delta-1 fertility-core lift CONFINED "
        "to the deficit cohorts (55-64|male, 65-74|male, 45-54|female, "
        "65-74|female; byte-identical to c8 there via delta-1 + delta-3), and "
        "every other cohort reverted to candidate 7 byte-identically."
    ),
    "coresident_grandchild": ("CARRIED candidate 8 (byte-identical)."),
    "multigen_stock": (
        "CARRIED candidate 1 initial state + train band x sex entry/exit "
        "(byte-identical to c8)."
    ),
    "multigen_transition": (
        "CARRIED candidate 1 pooled entry/exit rates (byte-identical to c8)."
    ),
    "parental_home_exit": ("CARRIED candidate 1 (byte-identical to c8)."),
    "hh_size": (
        "composed ego-centric family unit + the CARRIED count-conditional "
        "non-family bridge, then the delta-1 fertility-core hh_size resample "
        "CONFINED to the deficit cohorts (the priced hh_size.5+ aggregate now "
        "carries only the deficit cohorts' share of candidate 8's lift)."
    ),
}


def per_family_decomposition(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    families = [fam for fam, _ in _FAMILY_OF]
    out: dict[str, Any] = {}
    for fam in families:
        cells = sorted(c for c in tol if _family_of(c) == fam)
        if not cells:
            continue
        n_cell_seed = len(cells) * len(per_seed)
        n_pass = sum(
            s["gated_cells"][c]["pass"] for s in per_seed for c in cells
        )
        worst_cell = None
        worst_ratio = -1.0
        worst_mean_ln = None
        worst_tol = None
        for c in cells:
            scores = [s["gated_cells"][c]["score"] for s in per_seed]
            finite = [x for x in scores if math.isfinite(x)]
            mean_ln = float(np.mean(finite)) if finite else float("inf")
            ratio = mean_ln / tol[c] if tol[c] > 0 else float("inf")
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_cell = c
                worst_mean_ln = mean_ln
                worst_tol = tol[c]
        seed_pass_counts = {
            s["seed"]: int(sum(s["gated_cells"][c]["pass"] for c in cells))
            for s in per_seed
        }
        out[fam] = {
            "n_cells": len(cells),
            "cells": cells,
            "cell_seed_pass_rate": round(n_pass / n_cell_seed, 4),
            "n_cell_seed_pass": int(n_pass),
            "n_cell_seed": int(n_cell_seed),
            "per_seed_cells_passed": seed_pass_counts,
            "worst_cell": worst_cell,
            "worst_cell_mean_abs_ln": (
                round(worst_mean_ln, 4)
                if worst_mean_ln is not None and math.isfinite(worst_mean_ln)
                else None
            ),
            "worst_cell_tolerance": worst_tol,
            "worst_cell_mean_ln_over_tol": (
                round(worst_ratio, 3) if math.isfinite(worst_ratio) else None
            ),
            "mechanism": _MECHANISMS.get(fam, ""),
        }
    return out


def report_only_summary(
    per_seed: list[dict[str, Any]], report_only: list[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cell in sorted(report_only):
        scores = [s["report_only_cells"][cell]["score"] for s in per_seed]
        finite = [x for x in scores if x is not None and math.isfinite(x)]
        out[cell] = {
            "per_seed_score": {
                s["seed"]: s["report_only_cells"][cell]["score"]
                for s in per_seed
            },
            "mean_score": (float(np.mean(finite)) if finite else None),
            "max_score": (float(np.max(finite)) if finite else None),
        }
    return {
        "note": (
            "The report-only cells (below the 20-event floor, above the T_max "
            "power cap, or superseded by a gating aggregate). Same "
            "|ln(rbar / rate_a)| statistic; never gated."
        ),
        "cells": out,
    }


# --------------------------------------------------------------------------
# Candidate 1 -> ... -> 9 progression + regression vs c8 + byte carry
# --------------------------------------------------------------------------
def comparison_across_candidates(
    decomposition: dict[str, Any],
    per_seed: list[dict[str, Any]],
    tol: dict[str, float],
) -> dict[str, Any]:
    cand = {
        n: json.loads(p.read_text()) for n, p in CANDIDATE_ARTIFACTS.items()
    }
    decomp = {n: cand[n]["per_family_decomposition"] for n in cand}
    per_family: dict[str, Any] = {}
    for fam, d9 in decomposition.items():
        r8 = decomp[8].get(fam, {}).get("cell_seed_pass_rate")
        per_family[fam] = {
            f"candidate{n}_pass_rate": decomp[n]
            .get(fam, {})
            .get("cell_seed_pass_rate")
            for n in cand
        }
        per_family[fam]["candidate9_pass_rate"] = d9["cell_seed_pass_rate"]
        per_family[fam]["delta_c8_to_c9"] = (
            round(d9["cell_seed_pass_rate"] - r8, 4)
            if r8 is not None
            else None
        )
        per_family[fam]["candidate8_worst_cell"] = (
            decomp[8].get(fam, {}).get("worst_cell")
        )
        per_family[fam]["candidate9_worst_cell"] = d9["worst_cell"]
        per_family[fam]["candidate9_worst_mean_abs_ln"] = d9[
            "worst_cell_mean_abs_ln"
        ]

    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE8_CLEARED_FAMILIES:
        r8 = decomp[8].get(fam, {}).get("cell_seed_pass_rate")
        r9 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r9 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate8_pass_rate": r8,
            "candidate9_pass_rate": r9,
            "still_clears": bool(holds),
        }

    c7_by_seed = {s["seed"]: s for s in cand[7]["per_seed"]}
    c8_by_seed = {s["seed"]: s for s in cand[8]["per_seed"]}
    c9_by_seed = {s["seed"]: s for s in per_seed}

    def _max_dev(ref_by_seed: dict[int, Any], cells: list[str]) -> float:
        d = 0.0
        for seed in (s["seed"] for s in per_seed):
            for cell in cells:
                a = c9_by_seed[seed]["gated_cells"][cell]["score"]
                b = ref_by_seed[seed]["gated_cells"][cell]["score"]
                if math.isfinite(a) and math.isfinite(b):
                    d = max(d, abs(a - b))
        return d

    # (1) Cells byte-identical to candidate 8: all carries + all spouse (the
    # delta-2 lift carried) + the deficit scope child cells (delta-1 + delta-3
    # unchanged there).
    byte_identical_c8_cells = sorted(
        c
        for c in tol
        if (
            c.startswith(
                (
                    "coresident_parent.",
                    "multigen.",
                    "parental_home_",
                    "coresident_grandchild.",
                    "coresident_spouse.",
                )
            )
            or c in ("multigen_entry", "multigen_exit")
            or c in SCOPE_CHILD_CELLS
        )
    )
    max_c8_dev = _max_dev(c8_by_seed, byte_identical_c8_cells)

    # (2) The reverted child cells: byte-identical to candidate 7 (revert) and
    # MOVED vs candidate 8 (the scope change).
    max_c7_revert_dev = _max_dev(c7_by_seed, list(REVERTED_CHILD_CELLS))
    reverted_moved_vs_c8 = {
        cell: _max_dev(c8_by_seed, [cell]) for cell in REVERTED_CHILD_CELLS
    }

    # (3) The deficit scope child cells: byte-identical to candidate 8.
    scope_child_dev_vs_c8 = {
        cell: _max_dev(c8_by_seed, [cell]) for cell in SCOPE_CHILD_CELLS
    }

    # Per-cell c8 vs c9 pass movement for the child + hh_size families.
    child_hh_cells = sorted(
        c
        for c in tol
        if c.startswith("coresident_child.") or c.startswith("hh_size.")
    )
    child_hh_movement = {}
    for cell in child_hh_cells:
        child_hh_movement[cell] = {
            "scope_status": (
                "deficit_scope"
                if cell in SCOPE_CHILD_CELLS
                else (
                    "reverted_to_c7"
                    if cell in REVERTED_CHILD_CELLS
                    else "hh_size_scoped_resample"
                )
            ),
            "is_candidate8_collateral": cell in COLLATERAL_CELLS,
            "candidate8_per_seed_pass": {
                seed: c8_by_seed[seed]["gated_cells"][cell]["pass"]
                for seed in (s["seed"] for s in per_seed)
            },
            "candidate9_per_seed_pass": {
                seed: c9_by_seed[seed]["gated_cells"][cell]["pass"]
                for seed in (s["seed"] for s in per_seed)
            },
        }

    return {
        "candidate_artifacts": {
            n: f"runs/gate2b_hazard_v{n}.json" for n in cand
        },
        "candidate_verdicts": {
            n: {
                "gate_2b_pass": cand[n]["verdict"]["gate_2b_pass"],
                "n_seeds_pass": cand[n]["verdict"]["n_seeds_pass"],
            }
            for n in cand
        },
        "per_family_progression": per_family,
        "cleared_family_regression_check": {
            "families": list(CANDIDATE8_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-8 cleared families stay cleared. The carried "
                "families (parent / spouse / multigen / parental-home / "
                "grandchild) are byte-identical to candidate 8; hh_size is "
                "re-checked because the scoped fertility resample changes its "
                "aggregate (the priced hh_size.5+ cell)."
            ),
        },
        "byte_identical_vs_candidate8": {
            "cells": byte_identical_c8_cells,
            "max_abs_score_deviation_vs_candidate8": max_c8_dev,
            "byte_identical": bool(max_c8_dev <= EXACT_ATOL),
            "scope_child_cells_max_dev_vs_candidate8": scope_child_dev_vs_c8,
            "note": (
                "Every carried family, every coresident_spouse band (incl. the "
                "delta-2 25-34|female lift) AND the four deficit scope child "
                "cells are byte-identical to candidate 8 to bit precision -- "
                "candidate 9 changes ONLY which cohorts the delta-1 swap "
                "writes, and the deficit cohorts' draws and writes are "
                "candidate 8's."
            ),
        },
        "reverted_child_cells_vs_candidate7": {
            "cells": list(REVERTED_CHILD_CELLS),
            "max_abs_score_deviation_vs_candidate7": max_c7_revert_dev,
            "byte_identical_to_candidate7": bool(
                max_c7_revert_dev <= EXACT_ATOL
            ),
            "moved_vs_candidate8": reverted_moved_vs_c8,
            "collateral_cells": list(COLLATERAL_CELLS),
            "note": (
                "The non-deficit coresident_child cells revert to candidate 7 "
                "byte-identically (max score deviation vs c7 at bit precision) "
                "and MOVE vs candidate 8 -- the four candidate-8 collateral "
                "cells (35-44|m/f, 45-54|m, 55-64|f) return to their cleared "
                "candidate-7 state, the scope change's whole point."
            ),
        },
        "child_hh_size_movement_vs_candidate8": {
            "cells": child_hh_cells,
            "detail": child_hh_movement,
            "note": (
                "Per-cell candidate-8 vs candidate-9 pass status for the "
                "coresident_child + hh_size families: deficit scope cells "
                "unchanged, reverted cells back to candidate 7, hh_size "
                "resampled at deficit-cohort scope."
            ),
        },
    }


def carried_blocker_analysis(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Classify each seed's failing cells as scope-target or carried-blocker.

    A cell byte-identical to candidate 8 that fails caps the seed regardless of
    the scope change. For candidate 9 the deficit scope child cells and hh_size
    are the change surface; every other family (incl. all coresident_spouse) is
    a carried blocker if it fails.
    """
    strict_carried_families = {
        "coresident_parent",
        "coresident_spouse",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
        "coresident_grandchild",
    }
    per_seed_out: dict[int, Any] = {}
    for s in per_seed:
        fails = [c for c in sorted(tol) if not s["gated_cells"][c]["pass"]]
        carried_blockers = []
        change_surface = []
        for c in fails:
            fam = _family_of(c)
            is_carried = fam in strict_carried_families or (
                fam == "coresident_child" and c in SCOPE_CHILD_CELLS
            )
            # The deficit scope child cells are byte-identical to c8 (carried
            # effectively); the reverted child cells + hh_size are the surface.
            (carried_blockers if is_carried else change_surface).append(c)
        per_seed_out[s["seed"]] = {
            "n_fail": len(fails),
            "carried_blockers": carried_blockers,
            "scope_change_surface_fails": change_surface,
            "seed_capped_by_carried_cell": bool(carried_blockers),
        }
    n_capped = sum(
        1 for v in per_seed_out.values() if v["seed_capped_by_carried_cell"]
    )
    return {
        "per_seed": per_seed_out,
        "n_seeds_capped_by_carried_cell": n_capped,
        "max_attainable_seeds_given_carried_blockers": len(per_seed)
        - n_capped,
        "note": (
            "A cell byte-identical to candidate 8 (every carry, all "
            "coresident_spouse, and the four deficit scope child cells) that "
            "fails caps the seed regardless of the candidate-9 scope change. "
            "The change surface is the reverted child cells + hh_size."
        ),
    }


def analytic_check_vs_realized(
    analytic: dict[str, Any],
    per_seed: list[dict[str, Any]],
    tol: dict[str, float],
) -> dict[str, Any]:
    """Compare the pre-run train-side prediction to the realized holdout."""
    by_seed = {s["seed"]: s for s in per_seed}
    cells: dict[str, Any] = {}
    for cell in hcs9.ANALYTIC_CHECK_CELLS:
        pred = analytic["cells"][cell]
        rbars = [
            by_seed[s["seed"]]["gated_cells"][cell]["rbar"] for s in per_seed
        ]
        passes = {
            s["seed"]: by_seed[s["seed"]]["gated_cells"][cell]["pass"]
            for s in per_seed
        }
        n_pass = sum(1 for v in passes.values() if v)
        cells[cell] = {
            "predicted_scoped_counterfactual_train": pred[
                "scoped_counterfactual"
            ],
            "predicted_within_tolerance": pred["predicted_within_tolerance"],
            "realized_seed_mean_rbar_holdout": float(np.mean(rbars)),
            "realized_per_seed_pass": passes,
            "realized_n_seeds_pass": n_pass,
            "prediction_matches_realized": (
                pred["predicted_within_tolerance"] == (n_pass == len(per_seed))
            ),
        }
    return {
        "note": (
            "The pre-run train-side analytic prediction (scoped counterfactual "
            "vs observed side-B reference) beside the realized holdout outcome "
            "(seed-mean rbar vs rate_a). The prediction is train-side; the "
            "realization is the scored holdout. hh_size.5+ is the priced cell."
        ),
        "hh_size_5plus_priced": analytic["hh_size_5plus_priced"],
        "cells": cells,
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _sha_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _merge_base() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "merge-base", "HEAD", "origin/master"],
                cwd=ROOT,
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
def run(
    verbose: bool = True,
    seeds: tuple[int, ...] = GATE_SEEDS,
    n_draws: int = N_DRAWS,
    artifact_path: Path = ARTIFACT_PATH,
    cache_dir: Path | None = None,
    score_only: bool = False,
) -> dict[str, Any]:
    started = time.time()
    global N_DRAWS
    N_DRAWS = n_draws

    thresholds = load_gate2b_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2b thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    tol = gated_tolerances(thresholds)
    if len(tol) != 46:
        raise RuntimeError(
            f"expected 46 gated tolerances, got {len(tol)} from gates.yaml."
        )
    report_only = list(thresholds["report_only"])

    floor = json.loads(FLOOR_RUN.read_text())
    forensics = json.loads(FORENSICS_RUN.read_text())
    gated_set = set(floor["gate_partition"]["gate_eligible"])
    if set(tol) != gated_set:
        raise RuntimeError(
            "gates.yaml gated tolerances do not match the floor's "
            "gate_partition; refusing to score a mismatched cell set."
        )

    data = load_all()
    hh = data["hh"]
    if verbose:
        print(
            f"panel: {len(hh.person_waves)} person-waves, "
            f"{hh.person_waves.person_id.nunique()} persons; "
            f"estimator: mean over K={N_DRAWS} draws (5200 + k)"
        )

    precheck = run_precheck(hh, floor)
    if verbose:
        print(
            "precheck all_reproduced_exactly="
            f"{precheck['all_reproduced_exactly']}"
        )
    if not precheck["all_reproduced_exactly"]:
        raise RuntimeError(
            "Scoring path does not reproduce the committed gate-2b floor to "
            "bit precision; refusing to proceed."
        )

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []
    fit_checks = None
    analytic_check = None
    for seed in seeds:
        cache_path = (
            (cache_dir / f"seed_{seed}.json")
            if cache_dir is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            rec = json.loads(cache_path.read_text())
            if verbose:
                print(f"seed {seed}: loaded from cache {cache_path.name}")
        else:
            rec = score_seed(
                seed,
                data,
                floor,
                tol,
                report_only,
                forensics,
                verbose,
                compute_fit_checks=(seed == FIT_VS_RAW_SEED),
            )
            if cache_path is not None:
                cache_path.write_text(json.dumps(_json_safe(rec)))
        if rec.get("c9_delta_checks") is not None:
            fit_checks = rec["c9_delta_checks"]
        if rec.get("pre_run_analytic_check") is not None:
            analytic_check = rec["pre_run_analytic_check"]
        rec.pop("c9_delta_checks", None)
        rec.pop("pre_run_analytic_check", None)
        per_seed.append(rec)

    if score_only:
        if verbose:
            print(
                f"score-only: cached {len(per_seed)} seed(s) "
                f"{[s['seed'] for s in per_seed]}; no assembly."
            )
        return {"score_only": True, "seeds": [s["seed"] for s in per_seed]}

    per_draw_cube = _per_draw_cube(per_seed, tol)
    undefined_block = _undefined_block(per_seed)
    dispersion_block = _dispersion_block(per_seed, tol)
    if undefined_block["run_invalidated"]:
        raise RuntimeError(
            "RUN INVALIDATED (undefined_draw_rule): "
            f"{undefined_block['n_undefined_gated_draws']} undefined gated "
            "cell draw(s); the run must be re-registered and re-run."
        )

    verdict = build_verdict(per_seed, tol)
    decomposition = per_family_decomposition(per_seed, tol)
    comparison = comparison_across_candidates(decomposition, per_seed, tol)
    blocker = carried_blocker_analysis(per_seed, tol)
    report_block = report_only_summary(per_seed, report_only)
    avr = (
        analytic_check_vs_realized(analytic_check, per_seed, tol)
        if analytic_check is not None
        else None
    )
    seed_conjunction = [
        {
            "seed": s["seed"],
            "n_gated_pass": s["n_gated_pass"],
            "n_gated_fail": s["n_gated_fail"],
            "seed_pass": s["seed_pass"],
        }
        for s in per_seed
    ]

    if verbose:
        print(
            f"VERDICT: gate_2b_pass={verdict['gate_2b_pass']} "
            f"({verdict['n_seeds_pass']}/{len(per_seed)} seeds)"
        )
        for fam, d in decomposition.items():
            c = comparison["per_family_progression"][fam]
            print(
                f"  {fam:22s} pass {d['cell_seed_pass_rate']:.2f} "
                f"(c8 {c['candidate8_pass_rate']}, d_c8c9 "
                f"{c['delta_c8_to_c9']}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} tol={d['worst_cell_tolerance']}"
            )
        byt = comparison["byte_identical_vs_candidate8"]
        rev = comparison["reverted_child_cells_vs_candidate7"]
        print(
            "  byte-identical vs c8="
            f"{byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate8']:.2e}); "
            "reverted==c7="
            f"{rev['byte_identical_to_candidate7']} (max dev "
            f"{rev['max_abs_score_deviation_vs_candidate7']:.2e})"
        )
        if avr is not None:
            hp = avr["hh_size_5plus_priced"]
            print(
                "  analytic hh_size.5+ scoped_lift="
                f"{hp['scoped_lift_candidate9']:.4f} of global "
                f"{hp['global_lift_candidate8']:.4f} "
                f"(middle share {hp['middle_cohort_share_of_lift']})"
            )
        print(
            "  carried blockers cap "
            f"{blocker['n_seeds_capped_by_carried_cell']} seed(s); "
            f"max attainable = "
            f"{blocker['max_attainable_seeds_given_carried_blockers']}/5"
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "candidate": "candidate 9",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate8_registration_pointer": CANDIDATE8_REGISTRATION_POINTER,
        "candidate8_grading_pointer": CANDIDATE8_GRADING_POINTER,
        "grading_pointer": CANDIDATE8_GRADING_POINTER,
        "forensics_registration_pointer": FORENSICS_REGISTRATION_POINTER,
        "forensics_grading_pointer": FORENSICS_GRADING_POINTER,
        "forensics_artifact": "runs/gate2b_forensics5_v1.json",
        "one_shot": (
            "Registered on issue #42 comment 4948839837 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact. Per the standing addendum, any "
            "pass faces independent verification before entering the record."
        ),
        "delta_vs_candidate_8": (
            "ONE delta: the delta-1 completed-fertility swap is confined to the "
            "forensics-5 deficit cohorts x sex ({55-64,65-74} x male, "
            "{45-54,65-74} x female); every other cohort reverts to the sim's "
            "own completed-family-size distribution (candidate-7 behavior). "
            "Delta 2 (cohab-overlay lift), delta 3 (band-signed retention + "
            "link-coverage) and every carried family are carried byte-faithfully "
            "from candidate 8."
        ),
        "deltas_carried_from_candidate_8": [
            "delta 2 (carried): cohabitation-overlay lift at 25-34|female "
            "(Bernoulli superposition new = old + 0.045 * (1 - old)); "
            "byte-identical to candidate 8",
            "delta 3 (carried): band-signed adult-child retention refit at "
            "parent 45+ + link-coverage inclusion; byte-identical to candidate "
            "8 (operates only on the deficit cohorts, whose delta-1 output is "
            "candidate 8's)",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 8's generator and fitted bundle REUSED verbatim; "
                "the SOLE candidate-9 change is a simulate-time write gate "
                "confining the delta-1 fertility-core swap to the deficit "
                "cohorts (household_composition_sim_v9."
                "apply_scoped_fertility_core_lift). Every per-cohort random "
                "draw is candidate 8's, so the deficit cohorts are "
                "byte-identical to candidate 8 and every other cohort reverts "
                "to candidate 7."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate8_module": (
                "populace_dynamics.models.household_composition_sim_v8"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v9"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "grandchild_coupling_age_lo": hcs8.GRANDCHILD_LO,
            "core_size_cap": hcs8.CORE_SIZE_CAP,
            "size_buckets": list(hcs8.SIZE_BUCKETS),
            "cohab_overlay_lift_band": hcs8.COHAB_OVERLAY_LIFT_BAND,
            "cohab_overlay_lift": hcs8.COHAB_OVERLAY_LIFT,
            "retention_exit_cells": list(hcs8.RETENTION_EXIT_CELLS),
            "link_coverage_cells": list(hcs8.LINK_COVERAGE_CELLS),
            "fertility_lift_scope_cells": list(hcs9.FERTILITY_LIFT_CELLS),
            "reverted_child_cells": list(hcs9.REVERTED_CHILD_CELLS),
            "candidate8_collateral_cells": list(hcs9.COLLATERAL_CELLS),
            "n_fit_draws": hcs8.N_FIT_DRAWS,
            "n_analytic_draws": hcs9.N_ANALYTIC_DRAWS,
            "delta_stream_tag_v7": hcs8.DELTA_STREAM_TAG_V7,
            "delta_stream_tag_v8": hcs8.DELTA_STREAM_TAG_V8,
            "components": [
                "coresident_spouse<-CARRIED candidate 8 byte-identical (incl. "
                "the delta-2 25-34|female cohab-overlay lift)",
                "coresident_parent<-CARRIED candidate 1 (byte-identical to c8)",
                "multigen<-CARRIED candidate 8 (byte-identical)",
                "coresident_child<-candidate-8 fertility lift CONFINED to the "
                "deficit cohorts (byte-identical to c8 there) + every other "
                "cohort reverted to candidate 7 (0xC8)",
                "coresident_grandchild<-CARRIED candidate 8 (byte-identical)",
                "hh_size<-candidate-8 composition + delta-1 fertility-core "
                "hh_size resample CONFINED to the deficit cohorts (0xC8)",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from the candidate-8 (== candidate-7) "
                "streams UNCHANGED (0xB2B / 0xC2 / 0xC3 / 0xC4 / 0xC5 / 0xC6 / "
                "0xC7); the three candidate-8 deltas draw from "
                "SeedSequence([5200+k, 0xC8]) exactly as candidate 8, and the "
                "candidate-9 scope gate confines only the delta-1 WRITE to the "
                "deficit cohorts (rng consumed identically to candidate 8)"
            ),
            "gate_seeds": list(seeds),
            "statistic": (
                "|ln(rbar_candidate,s / rate_a,s)|, rbar the 20-draw mean "
                "rate, scored once (NOT the mean of per-draw scores)"
            ),
            "pass_rule": (
                "seed passes iff all 46 gated cells hold; gate passes iff "
                ">=4 of 5 seeds pass"
            ),
        },
        "fresh_run_artifact_schema": {
            "per_draw_per_cell_rates": per_draw_cube,
            "undefined_draw_rule": undefined_block,
            "per_draw_dispersion_disclosure": dispersion_block,
        },
        "pre_run_analytic_check": analytic_check,
        "analytic_check_vs_realized": avr,
        "c9_delta_checks": {
            "computed_on_seed": FIT_VS_RAW_SEED,
            "note": (
                "The carried deltas' fit-vs-raw checks vs the forensics-5 "
                "measured quantities (fit seed's train side B), unchanged from "
                "candidate 8 because delta 2, delta 3 and the per-cohort "
                "D_train are all carried verbatim: the Q15 fertility-lever "
                "headline (hh_size.5+ 0.128 -> 0.144; 55-64|male 0.213 -> "
                "0.255), the Q16 cohab-overlay lift (0.588 -> 0.606), and the "
                "Q14 band-signed EXIT-ORIGIN + LINK-COVERAGE channels."
            ),
            "checks": fit_checks,
        },
        "data": {
            "holdout_basis": ["MX23REL"],
            "paternal_link_basis": ["cah85_23"],
            "n_person_waves": int(len(hh.person_waves)),
            "n_persons": int(hh.person_waves.person_id.nunique()),
            "floor_run": "runs/gate2b_floors_v1.json",
            "floor_run_sha256": _sha_of_file(FLOOR_RUN),
            "forensics_run_sha256": _sha_of_file(FORENSICS_RUN),
        },
        "precheck": precheck,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "verdict": verdict,
        "per_family_decomposition": decomposition,
        "comparison_across_candidates": comparison,
        "carried_blocker_analysis": blocker,
        "report_only": report_block,
        "revision_pins": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "head_sha": _git_sha(),
            "base_sha": _merge_base(),
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "candidate8_artifact_sha256": _sha_of_file(CANDIDATE_ARTIFACTS[8]),
            "candidate7_artifact_sha256": _sha_of_file(CANDIDATE_ARTIFACTS[7]),
            "forensics5_artifact_sha256": _sha_of_file(FORENSICS_RUN),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ARTIFACT_PATH))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "optional per-seed cache directory (resumable chunked runs; the "
            "assembled artifact is identical to an uncached run)"
        ),
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="assemble the artifact from an existing complete cache dir",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="score the given seeds into the cache dir without assembling",
    )
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    artifact = run(
        verbose=True,
        seeds=seeds,
        n_draws=args.draws,
        artifact_path=Path(args.out),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        score_only=args.score_only,
    )
    if args.score_only:
        return
    artifacts.write_new(Path(args.out), _json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
