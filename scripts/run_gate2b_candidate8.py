"""Gate-2b candidate 8: proven levers + band-signed adult-child retention.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b; 46 gated household/relationship cells, K=20 mean-over-draws,
per-seed train/holdout protocol). Registered on issue #42 comment 4948604739
BEFORE this run; published REGARDLESS of verdict.

Candidate 8 is candidate 7 (registration 4948186843; PR #145) with EXACTLY THREE
frozen deltas, each designed against a graded gate-2b forensics-5 finding
(``runs/gate2b_forensics5_v1.json``, grading 4948603337). Everything candidate 7
cleared or carried -- the certified tranche-2a marital core, the carried
``coresident_parent`` / ``multigen`` (stock + transitions) / ``parental_home_exit``
/ ``coresident_grandchild``, and every ``coresident_spouse`` band EXCEPT the
25-34|female overlay lifted by delta 2 -- is carried BYTE-FAITHFULLY (candidate 8
REUSES the candidate-7 generator and re-runs its exact 0xB2B / 0xC2 / 0xC3 / 0xC4
/ 0xC5 / 0xC6 / 0xC7 streams; the three candidate-8 deltas modify the composed
frame on the isolated 0xC8 stream).

Delta 1 -- fertility-core lift (``hh_size.5+``, ``coresident_child.55-64|male``):
swap the sim completed-family-size distribution to the train 3+-child
distribution per parent cohort x sex, holding the sim's own kernels (Q15
analytic application; reproduces hh_size.5+ 0.128 -> 0.144, 55-64|male
0.213 -> 0.255).
Delta 2 -- cohabitation-overlay lift at 25-34|female: lift the currently-
non-spouse mass by the forensics-3 Q9 measured -0.045 overlay shortfall
(Bernoulli superposition; reproduces 0.588 -> 0.606), age-band-specific.
Delta 3 -- band-signed adult-child retention refit at parent 45+ + link-coverage
inclusion at older parent ages: close the Q14 EXIT-ORIGIN channel band-signed
(lift 65-74 under-retention, reduce 45-54|female over-retention) and the
LINK-COVERAGE channel (-0.020 55-64|male, -0.016 65-74|male); the v7 interaction
is the named residual.

Estimator, artifact schema, undefined-draw rule, and protocol are the locked
contract. Artifact ``runs/gate2b_hazard_v8.json``; ``artifacts.write_new``
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

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v8.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v8"
RUN_NAME = "gate2b_hazard_v8"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS_RUN = ROOT / "runs" / "gate2b_forensics5_v1.json"
CANDIDATE_ARTIFACTS = {
    1: ROOT / "runs" / "gate2b_hazard_v1.json",
    2: ROOT / "runs" / "gate2b_hazard_v2.json",
    3: ROOT / "runs" / "gate2b_hazard_v3.json",
    4: ROOT / "runs" / "gate2b_hazard_v4.json",
    5: ROOT / "runs" / "gate2b_hazard_v5.json",
    6: ROOT / "runs" / "gate2b_hazard_v6.json",
    7: ROOT / "runs" / "gate2b_hazard_v7.json",
}
REGISTRATION_POINTER = "4948604739"
CANDIDATE7_REGISTRATION_POINTER = "4948186843"
CANDIDATE7_GRADING_POINTER = "4948429354"
GRADING_POINTER = "4948603337"  # forensics-5 grading (c8 designs against it)
FORENSICS_REGISTRATION_POINTER = "4948430423"
SPEC_REGISTRATION = (
    "issue #42 comment 4948604739: gate-2b candidate 8, the two proven levers "
    "(fertility-core lift, cohabitation-overlay lift) and the band-signed "
    "adult-child retention refit + link-coverage inclusion at parent ages 45+"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12
FIT_VS_RAW_SEED = 0

#: Families carried BYTE-IDENTICAL to candidate 7: no candidate-8 delta touches
#: their streams. This INCLUDES all of coresident_spouse EXCEPT the delta-2
#: overlay-lifted 25-34|female cell (which is a delta target, not a carry).
CANDIDATE7_STRICT_CARRIED_FAMILIES = (
    "coresident_parent",
    "coresident_spouse",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_grandchild",
)
#: Candidate-7 families whose pass rate was 1.0 and must STAY cleared.
CANDIDATE7_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_grandchild",
)
MULTIGEN_MARGINAL_CELLS_PREFIX = ("multigen.",)
MULTIGEN_MARGINAL_CELLS_EXACT = ("multigen_entry", "multigen_exit")
#: The delta-2 overlay-lifted cell (was the candidate-7 fragile carry).
LIFTED_SPOUSE_CELL = "coresident_spouse.25-34|female"

#: Per-delta target family.
PER_DELTA_TARGET_FAMILY = {
    "delta_1_fertility_core_lift": ["coresident_child", "hh_size"],
    "delta_2_cohab_overlay_lift": ["coresident_spouse"],
    "delta_3_retention_link_refit": ["coresident_child"],
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.60-0.75",
    "majority_side": True,
    "named_expectations": [
        "hh_size.5+ clears on its failing seeds (Q15-proven fertility lift)",
        "coresident_child.55-64|male clears (Q15-proven)",
        "coresident_spouse.25-34|female clears incl. the fragility pair "
        "(Q16-proven cohab-overlay lift)",
        "coresident_child.65-74|male clears or lands within 1.2x (the "
        "multi-channel cell -- modal residual if anything survives)",
        "coresident_child.45-54|female returns to tolerance from the over side "
        "(the sign test of delta 3)",
        "all cleared child cells and the byte-identical carries hold",
    ],
    "modal_outcome_if_fail": (
        "coresident_child.65-74|male alone -- the multi-channel cell whose v7 "
        "persistence/enumeration interaction (-0.010) is the named residual "
        "left for a targeted candidate-9 forensics"
    ),
    "named_residual_risk": (
        "the v7 persistence/enumeration interaction at the older male bands "
        "(+0.008 / -0.010), left untouched; and the hh_size re-emission bleed "
        "into hh_size.1/.2 (both OVER in candidate 7, so the fertility swap is "
        "corrective, not regressive)"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT grade "
        "it."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "rng_byte_identical_carried_families": (
        "Candidate 8 re-runs candidate 7's exact 0xB2B / 0xC2 / 0xC3 / 0xC4 / "
        "0xC5 / 0xC6 / 0xC7 composition (household_composition_sim_v8._compose_"
        "v7 is a faithful copy of hcs7.simulate_draw_v7's body up to the panel "
        "build). The three candidate-8 deltas modify only the composed "
        "coresident_child / hh_size / coresident_spouse.25-34|female arrays on a "
        "SEPARATE SeedSequence([5200+k, 0xC8]); coresident_parent, multigen and "
        "coresident_grandchild are taken from the candidate-7 composition "
        "UNCHANGED. Every carried coresident_parent, coresident_spouse (all "
        "bands except 25-34|female), multigen (stock + transitions), "
        "parental_home_exit and coresident_grandchild cell is bit-identical to "
        "candidate 7 on every draw and seed (regression is impossible by "
        "construction)."
    ),
    "delta_1_fertility_core_lift": (
        "Forensics-5 Q15 proved the single fertility lever moves hh_size.5+ "
        "into tolerance on its failing seeds (0.128 -> 0.144 vs 0.139) and "
        "coresident_child.55-64|male (0.213 -> 0.255) while holding hh_size.3/.4 "
        "and the cleared child cells. RESOLUTION: candidate 8 swaps the sim's "
        "completed-family-size distribution D_sim[S] to the train D_train[S] per "
        "(band, sex), holding the sim's own coresidence-given-size kernel "
        "K_sim(coresident|S) and hh_size|size resample kernel -- the Q15 "
        "analytic application realized per-person on the isolated 0xC8 stream. "
        "hh_size.1/.2 (both OVER in candidate 7) move toward rate_a, so the "
        "population swap is corrective."
    ),
    "delta_2_cohab_overlay_lift": (
        "Forensics-5 Q16 proved a cohabitation-overlay lift at 25-34|female "
        "sized to the forensics-3 Q9 measured -0.045 overlay shortfall clears "
        "the fragile cell without collateral (0.588 -> 0.606 vs 0.621; other "
        "female spouse bands unmoved). RESOLUTION: candidate 8 lifts the "
        "currently-non-spouse mass at 25-34|female by 0.045 (Bernoulli "
        "superposition new = old + 0.045 * (1 - old)), an age-band-specific "
        "override. The candidate-7 fragile 2/5 split-seed cell becomes a delta "
        "target; every other female band is byte-identical to candidate 7."
    ),
    "delta_3_retention_link_refit": (
        "Forensics-5 Q14 decomposed the older-parent adult-child deficit into "
        "FERTILITY-ORIGIN (delta 1), EXIT-ORIGIN (band-signed: 65-74|male -0.022 "
        "under, 45-54|female +0.079 over, 65-74|female -0.016 under), "
        "LINK-COVERAGE (-0.020 55-64|male, -0.016 65-74|male) and the v7 "
        "persistence/enumeration interaction. RESOLUTION: candidate 8 refits the "
        "coresident-adult-child exit/retention hazards by parent age band x sex "
        "on train, closing the EXIT-ORIGIN channel band-signed (a positive "
        "additive shift lifts under-retained cells; a negative shift reduces the "
        "45-54|female over-retention) plus the LINK-COVERAGE channel at the "
        "older-male bands (the enumerated joinable children whose links exist in "
        "train but sit outside the current draw basis at those ages enter it). "
        "Fitted on side B over train draws; applied to the delta-1-lifted "
        "coresident_child cell via a conditional flip. The v7 interaction "
        "(+0.008 / -0.010) is the NAMED residual, left for candidate 9."
    ),
    "v7_interaction_named_residual": (
        "The v7 persistence/enumeration interaction at 55-64|male (+0.008) and "
        "65-74|male (-0.010) is deliberately UNTOUCHED -- the named residual "
        "(analytic independent-per-wave occupancy vs observed episode among "
        "joinable children). If 65-74|male is the modal survivor, candidate 9 is "
        "a single-cell run with this channel measured in a targeted forensics."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "VERBATIM from candidate 7: each simulated side-A holdout person is "
        "seeded from their OWN observed state at their first 2b wave, then "
        "evolved with train-fitted hazards; no parameter is estimated from side "
        "A. The candidate-8 fertility-core target D_train, the delta-3 channels "
        "and the delta-2 lift are fitted on train side B only (no holdout "
        "tuning surface)."
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
        "delta_1_fertility_core_lift": {
            "completed_size_dist_train_all": (
                fert["completed_size_dist_train_all"]
            ),
            "mean_realized_hh_size_5plus_lift": None,
        },
        "delta_2_cohab_overlay_lift": {
            "band": hcs8.COHAB_OVERLAY_LIFT_BAND,
            "overlay_shortfall": model.cohab_overlay_lift,
            "mean_realized_lift": dmean_dict("cohab_overlay_lift").get(
                "realized_lift"
            ),
        },
        "delta_3_retention_link_refit": {
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

    model = hcs8.fit_household_model_v8(
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
        sim_panel, diag = hcs8.simulate_draw_v8(
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
    if compute_fit_checks:
        fit_checks = hcs8.c8_delta_checks(model, forensics)

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
        "c8_delta_checks": fit_checks,
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
            "delta-2 cohabitation-overlay lift and the delta-3 band-signed "
            "retention + link-coverage additive closures. A material gap "
            "between target and realized is a spec violation, not a finding."
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
        "CARRIED candidate 7 legal registry UNION cohabitation overlay UNION "
        "legal-residual overlay -- byte-identical to candidate 7 on every band "
        "EXCEPT 25-34|female, where delta 2 lifts the cohabitation overlay by "
        "the measured -0.045 shortfall (Bernoulli superposition)."
    ),
    "coresident_parent": (
        "CARRIED candidate 1 logistic exit hazard (byte-faithful, RNG-isolated)."
    ),
    "coresident_child": (
        "candidate-7 observed-link + maternal + shadow composition, then the "
        "delta-1 fertility-core lift (swap D_sim[S] -> D_train[S] holding "
        "K_sim) and the delta-3 band-signed retention + link-coverage refit at "
        "parent 45+ on the isolated 0xC8 stream."
    ),
    "coresident_grandchild": (
        "CARRIED candidate 7 (the 55+ coupling and the 45-54 composed cell taken "
        "from the candidate-7 composition UNCHANGED; byte-identical)."
    ),
    "multigen_stock": (
        "CARRIED candidate 1 initial state + train band x sex entry/exit."
    ),
    "multigen_transition": (
        "CARRIED candidate 1 pooled entry/exit rates (byte-faithful)."
    ),
    "parental_home_exit": (
        "CARRIED candidate 1 (byte-faithful, RNG-isolated)."
    ),
    "hh_size": (
        "composed ego-centric family unit + the CARRIED count-conditional "
        "non-family bridge, then the delta-1 fertility-core hh_size resample "
        "(swap D_sim[S] -> D_train[S] holding the hh_size|size kernel)."
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
# Candidate 1 -> ... -> 8 progression + regression + byte carry + checks
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
    for fam, d8 in decomposition.items():
        r7 = decomp[7].get(fam, {}).get("cell_seed_pass_rate")
        per_family[fam] = {
            f"candidate{n}_pass_rate": decomp[n]
            .get(fam, {})
            .get("cell_seed_pass_rate")
            for n in cand
        }
        per_family[fam]["candidate8_pass_rate"] = d8["cell_seed_pass_rate"]
        per_family[fam]["delta_c7_to_c8"] = (
            round(d8["cell_seed_pass_rate"] - r7, 4)
            if r7 is not None
            else None
        )
        per_family[fam]["candidate7_worst_cell"] = (
            decomp[7].get(fam, {}).get("worst_cell")
        )
        per_family[fam]["candidate8_worst_cell"] = d8["worst_cell"]
        per_family[fam]["candidate8_worst_mean_abs_ln"] = d8[
            "worst_cell_mean_abs_ln"
        ]

    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE7_CLEARED_FAMILIES:
        r7 = decomp[7].get(fam, {}).get("cell_seed_pass_rate")
        r8 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r8 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate7_pass_rate": r7,
            "candidate8_pass_rate": r8,
            "still_clears": bool(holds),
        }

    c7_by_seed = {s["seed"]: s for s in cand[7]["per_seed"]}
    c8_by_seed = {s["seed"]: s for s in per_seed}

    def _max_dev(cells: list[str]) -> float:
        d = 0.0
        for seed in (s["seed"] for s in per_seed):
            for cell in cells:
                s8 = c8_by_seed[seed]["gated_cells"][cell]["score"]
                s7 = c7_by_seed[seed]["gated_cells"][cell]["score"]
                if math.isfinite(s8) and math.isfinite(s7):
                    d = max(d, abs(s8 - s7))
        return d

    # Byte-identical carried families vs candidate 7 (spouse EXCLUDES the
    # delta-2 overlay-lifted 25-34|female cell).
    strict_carried = [
        c
        for c in tol
        if (
            c.startswith(
                (
                    "coresident_parent.",
                    "multigen.",
                    "parental_home_",
                    "coresident_grandchild.",
                )
            )
            or c in ("multigen_entry", "multigen_exit")
            or (c.startswith("coresident_spouse.") and c != LIFTED_SPOUSE_CELL)
        )
    ]
    max_carry_dev = _max_dev(strict_carried)

    # The delta-2 lifted spouse cell (was the candidate-7 fragile carry).
    lifted_per_seed = {
        seed: {
            "candidate8_score": c8_by_seed[seed]["gated_cells"][
                LIFTED_SPOUSE_CELL
            ]["score"],
            "candidate7_score": c7_by_seed[seed]["gated_cells"][
                LIFTED_SPOUSE_CELL
            ]["score"],
            "candidate8_rbar": c8_by_seed[seed]["gated_cells"][
                LIFTED_SPOUSE_CELL
            ]["rbar"],
            "candidate7_rbar": c7_by_seed[seed]["gated_cells"][
                LIFTED_SPOUSE_CELL
            ]["rbar"],
            "candidate8_pass": c8_by_seed[seed]["gated_cells"][
                LIFTED_SPOUSE_CELL
            ]["pass"],
            "candidate7_pass": c7_by_seed[seed]["gated_cells"][
                LIFTED_SPOUSE_CELL
            ]["pass"],
        }
        for seed in (s["seed"] for s in per_seed)
    }

    # Multigen-marginal-unchanged check.
    multigen_cells = [
        c
        for c in tol
        if c.startswith(MULTIGEN_MARGINAL_CELLS_PREFIX)
        or c in MULTIGEN_MARGINAL_CELLS_EXACT
    ]
    mg_detail = {cell: _max_dev([cell]) for cell in sorted(multigen_cells)}
    max_mg_dev = max(mg_detail.values()) if mg_detail else 0.0

    # coresident_child target movement vs candidate 7 (the delta effect).
    child_cells = sorted(c for c in tol if c.startswith("coresident_child."))
    child_movement = {}
    for cell in child_cells:
        child_movement[cell] = {
            "candidate7_pass_all_seeds": all(
                c7_by_seed[seed]["gated_cells"][cell]["pass"]
                for seed in (s["seed"] for s in per_seed)
            ),
            "candidate8_pass_all_seeds": all(
                c8_by_seed[seed]["gated_cells"][cell]["pass"]
                for seed in (s["seed"] for s in per_seed)
            ),
            "candidate8_per_seed_pass": {
                seed: c8_by_seed[seed]["gated_cells"][cell]["pass"]
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
            "families": list(CANDIDATE7_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-7 cleared families stay cleared: the strictly-"
                "carried families (parent / spouse minus 25-34|female / "
                "multigen / parental-home / grandchild) are byte-identical, "
                "taken from the candidate-7 composition unchanged."
            ),
        },
        "byte_identical_carried_family_score_check": {
            "strict_carried_cells": sorted(strict_carried),
            "max_abs_score_deviation_vs_candidate7": max_carry_dev,
            "byte_identical": bool(max_carry_dev <= EXACT_ATOL),
            "note": (
                "Every strictly-carried cell's per-seed gated score equals "
                "candidate 7's to bit precision (parent / spouse all bands "
                "EXCEPT 25-34|female / multigen / parental-home / grandchild), "
                "because candidate 8 changes ONLY coresident_child / hh_size / "
                "coresident_spouse.25-34|female on the isolated 0xC8 stream."
            ),
        },
        "lifted_spouse_cell_delta2": {
            "cell": LIFTED_SPOUSE_CELL,
            "per_seed": lifted_per_seed,
            "candidate7_n_seeds_pass": sum(
                v["candidate7_pass"] for v in lifted_per_seed.values()
            ),
            "candidate8_n_seeds_pass": sum(
                v["candidate8_pass"] for v in lifted_per_seed.values()
            ),
            "note": (
                "The candidate-7 fragile coresident_spouse.25-34|female cell "
                "(2/5 split-seed exceedance) is LIFTED by delta 2 (the measured "
                "-0.045 cohabitation-overlay shortfall); its candidate-7 vs "
                "candidate-8 pass counts are recorded (the Q16 sign test)."
            ),
        },
        "multigen_marginal_unchanged_check": {
            "multigen_cells": sorted(multigen_cells),
            "max_abs_score_deviation_vs_candidate7": max_mg_dev,
            "per_cell_max_abs_score_deviation": mg_detail,
            "marginal_unchanged": bool(max_mg_dev <= EXACT_ATOL),
            "note": (
                "multigen is carried from candidate 1's simulate_draw and taken "
                "from the candidate-7 composition unchanged, so every multigen "
                "stock and transition cell is byte-identical to candidate 7."
            ),
        },
        "coresident_child_target_movement": {
            "cells": child_cells,
            "detail": child_movement,
            "note": (
                "coresident_child MOVES under the delta-1 fertility-core lift "
                "and the delta-3 band-signed retention + link-coverage refit; "
                "the candidate-7 vs candidate-8 pass status is recorded per cell."
            ),
        },
    }


def carried_blocker_analysis(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Classify each seed's failing cells as delta-target or carried-blocker.

    A CARRIED cell (byte-identical to candidate 7) that fails caps the seed
    regardless of the deltas. For candidate 8 the coresident_spouse.25-34|female
    cell is a delta-2 TARGET (lifted), so it is NOT a carried blocker.
    """
    strict_carried_families = {
        "coresident_parent",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
        "coresident_grandchild",
    }
    per_seed_out: dict[int, Any] = {}
    for s in per_seed:
        fails = [c for c in sorted(tol) if not s["gated_cells"][c]["pass"]]
        carried_blockers = []
        delta_targets = []
        for c in fails:
            fam = _family_of(c)
            is_carried = fam in strict_carried_families or (
                fam == "coresident_spouse" and c != LIFTED_SPOUSE_CELL
            )
            (carried_blockers if is_carried else delta_targets).append(c)
        per_seed_out[s["seed"]] = {
            "n_fail": len(fails),
            "carried_blockers": carried_blockers,
            "delta_target_or_downstream_fails": delta_targets,
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
            "A carried (byte-identical to candidate 7) cell that fails caps the "
            "seed regardless of the three candidate-8 deltas. The "
            "coresident_spouse.25-34|female cell is a delta-2 target, not a "
            "carried blocker."
        ),
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
        if rec.get("c8_delta_checks") is not None:
            fit_checks = rec["c8_delta_checks"]
        rec.pop("c8_delta_checks", None)
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
                f"(c7 {c['candidate7_pass_rate']}, d_c7c8 "
                f"{c['delta_c7_to_c8']}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} tol={d['worst_cell_tolerance']}"
            )
        byt = comparison["byte_identical_carried_family_score_check"]
        mg = comparison["multigen_marginal_unchanged_check"]
        print(
            "  strict-carried byte-identical vs c7="
            f"{byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate7']:.2e}); "
            f"multigen-marginal-unchanged={mg['marginal_unchanged']}"
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
        "candidate": "candidate 8",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate7_registration_pointer": CANDIDATE7_REGISTRATION_POINTER,
        "candidate7_grading_pointer": CANDIDATE7_GRADING_POINTER,
        "grading_pointer": GRADING_POINTER,
        "forensics_registration_pointer": FORENSICS_REGISTRATION_POINTER,
        "forensics_artifact": "runs/gate2b_forensics5_v1.json",
        "one_shot": (
            "Registered on issue #42 comment 4948604739 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_7": [
            "delta 1: fertility-core lift -- swap the sim completed-family-size "
            "distribution D_sim[S] to the train D_train[S] per (band, sex), "
            "holding the sim's own coresidence-given-size and hh_size|size "
            "kernels (the Q15 analytic application; hh_size.5+ 0.128 -> 0.144, "
            "coresident_child.55-64|male 0.213 -> 0.255)",
            "delta 2: cohabitation-overlay lift at 25-34|female -- lift the "
            "currently-non-spouse mass by the forensics-3 Q9 measured -0.045 "
            "overlay shortfall (Bernoulli superposition new = old + 0.045 * "
            "(1 - old)); age-band-specific, other female bands untouched "
            "(0.588 -> 0.606)",
            "delta 3: band-signed adult-child retention refit at parent 45+ + "
            "link-coverage inclusion -- close the Q14 EXIT-ORIGIN channel "
            "band-signed (lift 65-74 under-retention, reduce 45-54|female "
            "over-retention) and the LINK-COVERAGE channel at the older-male "
            "bands (-0.020 55-64|male, -0.016 65-74|male); the v7 interaction is "
            "the named residual",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 7's generator REUSED byte-faithfully (which reuses "
                "candidates 6-1, the enumeration-conditioned episode-persistent "
                "linked draw, the multigen--adult-child coupling, the female "
                "cohabitation refit and the count-conditional bridge), plus "
                "three train-fitted deltas on the composed frame: the "
                "fertility-core lift (delta 1), the cohabitation-overlay lift "
                "(delta 2) and the band-signed retention + link-coverage refit "
                "(delta 3), on the isolated 0xC8 stream."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate7_module": (
                "populace_dynamics.models.household_composition_sim_v7"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v8"
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
            "n_fit_draws": hcs8.N_FIT_DRAWS,
            "delta_stream_tag_v7": hcs8.DELTA_STREAM_TAG_V7,
            "delta_stream_tag_v8": hcs8.DELTA_STREAM_TAG_V8,
            "components": [
                "coresident_spouse<-CARRIED candidate 7 on every band EXCEPT "
                "25-34|female, where delta 2 lifts the cohab overlay by -0.045",
                "coresident_parent<-CARRIED candidate 1 logistic exit hazard",
                "multigen<-CARRIED candidate 7 (byte-identical)",
                "coresident_child<-candidate-7 composition + delta-1 "
                "fertility-core lift + delta-3 band-signed retention + "
                "link-coverage refit (0xC8)",
                "coresident_grandchild<-CARRIED candidate 7 (byte-identical)",
                "hh_size<-candidate-7 composition + delta-1 fertility-core "
                "hh_size resample (0xC8)",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from the candidate-7 streams UNCHANGED "
                "(0xB2B occupancy; 0xC2 cohabitation/child; 0xC3 non-family/"
                "skipgen; 0xC4 legal-residual; 0xC5 coupling/parent-count; 0xC6 "
                "maternal leave; 0xC7 linked episode); the three candidate-8 "
                "deltas draw the fertility-core lift, the retention/link refit "
                "and the cohab-overlay lift from a separate "
                "SeedSequence([5200+k, 0xC8])"
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
        "c8_delta_checks": {
            "computed_on_seed": FIT_VS_RAW_SEED,
            "note": (
                "The three deltas' fit-vs-raw checks vs the forensics-5 measured "
                "quantities (fit seed's train side B): the Q15 fertility-lever "
                "headline (hh_size.5+ 0.128 -> 0.144; 55-64|male 0.213 -> "
                "0.255), the Q16 cohab-overlay lift (0.588 -> 0.606), and the "
                "Q14 band-signed EXIT-ORIGIN + LINK-COVERAGE channels (fitted vs "
                "measured), with the v7 interaction as the named residual."
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
