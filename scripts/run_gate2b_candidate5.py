"""Gate-2b candidate 5: coupling + custodial correction + bridge reach.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b; 46 gated household/relationship cells, K=20 mean-over-draws,
per-seed train/holdout protocol). Registered on issue #42 comment 4945159933
BEFORE this run; published REGARDLESS of verdict.

Candidate 5 is candidate 4 (registration 4941160621; PR #138) with EXACTLY
THREE frozen deltas, each designed against a corrected mechanism the gate-2b
forensics-2 decomposition (``runs/gate2b_forensics2_v1.json``, grading
4945156926) measured. Everything candidate 4 cleared or carried -- the
certified tranche-2a marital core and maternal births, the carried
``coresident_parent`` / ``multigen`` (stock + transitions) /
``parental_home_exit``, AND the candidate-4 ``coresident_spouse`` family (legal
registry + cohabitation overlay + legal-spouse residual overlay) -- is carried
BYTE-FAITHFULLY (candidate 5 REUSES the candidate-4 module and re-runs its exact
0xB2B / 0xC2 / 0xC3 / 0xC4 streams; the carried families' per-seed scores are
IDENTICAL to candidate 4 to bit precision).

Delta 1 -- multigen--adult-child coupling (``coresident_grandchild``
55+|female): replace, for 55+ egos, the independent coresident-own-adult-child
input to the composed grandchild with a train-fitted JOINT
``P(child | multigen state, band, sex)`` on the simulated multigen occupancy
(0xC5). The multigen MARGINAL is unchanged (load-bearing spec constraint).
Delta 2 -- not-married custodial correction (``coresident_child`` male): for
NOT-married linked fathers, swap the observable-basis custodial probability for
the child-record-basis rate; the young-married gate is untouched.
Delta 3 -- bridge reach + parent_count composition (``hh_size``): re-fit the
non-family bridge incidence conditional on core size (lift size-3 cores) and
draw the coresident-parent count (1 vs 2) from the train composition (0xC5)
instead of the fixed parent_count=2.

Estimator, artifact schema, undefined-draw rule, and protocol are candidate 4's
(locked contract). Artifact ``runs/gate2b_hazard_v5.json``;
``artifacts.write_new`` refuses to overwrite. Reproduce with the ``.venv-gate``
and the PSID products staged (``POPULACE_DYNAMICS_PSID_DIR``).
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
from populace_dynamics.models import household_composition_sim_v5 as hcs5

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v5.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v5"
RUN_NAME = "gate2b_hazard_v5"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS_RUN = ROOT / "runs" / "gate2b_forensics2_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
CANDIDATE3_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v3.json"
CANDIDATE4_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v4.json"
REGISTRATION_POINTER = "4945159933"
CANDIDATE4_REGISTRATION_POINTER = "4941160621"
CANDIDATE3_REGISTRATION_POINTER = "4939960467"
CANDIDATE2_REGISTRATION_POINTER = "4939456379"
CANDIDATE1_REGISTRATION_POINTER = "4938726107"
GRADING_POINTER = "4945156926"
FORENSICS_POINTER = "4942005972"
SPEC_REGISTRATION = (
    "issue #42 comment 4945159933: gate-2b candidate 5, coupling + the "
    "not-married custodial cell + bridge reach (multigen--adult-child "
    "coupling; not-married custodial correction; bridge reach conditional on "
    "core size + parent_count composition)"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12
FIT_VS_RAW_SEED = 0

#: Candidate 4's carried families -- byte-identical to candidate 4 in c5 (no c5
#: delta touches their streams). coresident_spouse is NOW carried (c4's deltas 1
#: and 2 that targeted it are themselves carried byte-faithfully here).
CANDIDATE4_CARRIED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_spouse",
)
#: The candidate-4 families that CLEARED (pass rate 1.0): the regression check
#: confirms candidate 5 carries them byte-faithfully (still 1.0).
CANDIDATE4_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)
#: The multigen cells the delta-1 coupling constraint pins byte-identical (the
#: coupling reads the multigen state but never changes the marginal).
MULTIGEN_MARGINAL_CELLS_PREFIX = ("multigen.",)
MULTIGEN_MARGINAL_CELLS_EXACT = ("multigen_entry", "multigen_exit")

#: Per-delta target family (each delta feeds a disjoint failure surface).
PER_DELTA_TARGET_FAMILY = {
    "delta_1_multigen_child_coupling": "coresident_grandchild",
    "delta_2_not_married_custodial": "coresident_child",
    "delta_3_bridge_reach_and_parent_count": "hh_size",
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.40-0.55",
    "deliberately_not_majority_side": True,
    "named_expectations": [
        "coresident_child male cells clear (arithmetic correction of a "
        "measured overstatement)",
        "hh_size.3 clears or lands within 1.2x (mechanical mass reallocation)",
        "grandchild 55+|female improves past half its remaining gap with "
        "clearing possible but uncertain (new coupling machinery under a "
        "fitted joint -- the modal residual)",
        "multigen stock/transition cells stay clearing (the coupling "
        "constraint is load-bearing -- any movement there is a spec "
        "violation, not a finding)",
        "all other carried families byte-identical",
    ],
    "modal_outcome_if_fail": (
        "grandchild 55+|female alone, or grandchild plus one hh_size middle "
        "cell -- either way candidate 6 would be a single-mechanism run"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT grade "
        "it."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "rng_byte_identical_carried_families": (
        "Candidate 5 adds two stochastic components (the coupled adult-child "
        "draw; the per-ego parent-count draw) on a SEPARATE "
        "SeedSequence([5200+k, 0xC5]).spawn(2), and re-fits two existing "
        "probability tables (the not-married custodial rate; the non-family "
        "class thresholds now keyed on the simulated core size) whose per-wave "
        "draws consume RNG BY SHAPE ONLY -- so the candidate-1 (0xB2B), "
        "candidate-2 (0xC2), candidate-3 (0xC3) and candidate-4 (0xC4) streams "
        "stay byte-identical in consumption and only the targeted states move. "
        "The carried coresident_parent, multigen (stock + transitions), "
        "parental_home_exit AND coresident_spouse families come off the "
        "candidate-4 streams UNCHANGED, so they are bit-identical to candidate "
        "4 on every draw and seed (regression is impossible by construction)."
    ),
    "delta_1_multigen_child_coupling": (
        "Forensics-2 Q6 measured the reference 55+|female coresident_grandchild "
        "residual as a DECOUPLING: the reference couples the multigen state and "
        "a coresident own child into one household fact (train joint "
        "multigen AND child AND NOT-parent ~0.0384, ~5x the independence "
        "product ~0.0077), while candidate 4 draws multigen (the carried "
        "candidate-1 hazard) and a coresident child (aged-out custodial / "
        "maternal) from separate components so their joint collapses to the "
        "product ~0.0063. RESOLUTION: candidate 5 replaces, FOR 55+ EGOS ONLY, "
        "the independent coresident-own-adult-child input to the COMPOSED "
        "grandchild with a train-fitted P(coresident own child | multigen "
        "state, 55+ band, sex) drawn on the simulated multigen occupancy "
        "(isolated 0xC5); a (band, sex, multigen) stratum thinner than 20 "
        "waves falls back to the pooled 55+ (sex, multigen) rate. LOAD-BEARING "
        "CONSTRAINT: the multigen occupancy MARGINAL is UNCHANGED -- multigen "
        "comes off candidate 1's simulate_draw unchanged and the coupling only "
        "READS it, so every multigen stock / transition cell is byte-identical "
        "to candidate 4 (the coupling reallocates WHICH multigen egos carry "
        "the adult child, not how many egos are multigen). The coupled "
        "indicator feeds the grandchild composition ONLY (never coresident_"
        "child, hh_size, or multigen), exactly as candidate 4's skip-gen "
        "occupancy is unioned into the grandchild alone; ages below 55 keep "
        "the candidate-4 composed grandchild byte-identical."
    ),
    "delta_2_not_married_custodial": (
        "Forensics-2 Q5 measured the observable (father, child, wave) custodial "
        "basis OVER-stating coresidence for NOT-married fathers by ~0.096 at "
        "school ages against the less-selected child-record basis (denominator "
        "= the child's OWN enumerated waves), while the young-MARRIED gate is "
        "FAITHFUL (the gap REVERSES). RESOLUTION: candidate 5 replaces, for "
        "NOT-married linked fathers only, the observable-basis custodial "
        "probability with the child-record-basis rate by child age band (fit "
        "on train side B via the same denominator/event as the forensics); "
        "MARRIED fathers keep candidate 4's observable-basis lookup UNCHANGED. "
        "The per-child custodial draw is unchanged in shape (byte-identical "
        "0xC3 custodial stream); women's coresident_child (maternal only, no "
        "paternal link) and married-father coresident_child are byte-identical "
        "to candidate 4."
    ),
    "delta_3_bridge_reach_and_parent_count": (
        "Forensics-2 Q7 measured the 0.088 size-3 core-vs-actual gap splitting "
        "EXACTLY into a non-core-member part ~0.051 (reference size-3 CORES "
        "that are truly 4+ households -- a sibling/roomer present, the bridge "
        "must lift them) and a composition part ~0.037 (the over-produced "
        "three-adults route = ego + parent_count=2). RESOLUTION: (a) the "
        "non-family bridge's 0/1/2+ class incidence is re-fit CONDITIONAL ON "
        "CORE SIZE from train (dense (core, band, sex) cells; sparse fall back "
        "to candidate 4's (band, sex) shares), applied on the SIMULATED core "
        "size so size-3 family cores are lifted at the train rate -- the class "
        "draw is byte-identical in shape on 0xC3, only the per-core thresholds "
        "move, and the 2+ count spread is candidate 4's 0xC4 draw. (b) the "
        "parent_count=2 assumption is corrected to a per-ego draw of 1 vs 2 "
        "coresident parents from the train coresident-parent-count composition "
        "(P(two parents | coresident parent, band, sex)) on the isolated 0xC5 "
        "stream, relaxing the three-adults concentration and enabling the "
        "couple+parent route. Both feed hh_size ONLY (coresident_parent stays "
        "the boolean, byte-identical)."
    ),
    "carried_families_byte_faithful": (
        "The four candidate-4 cleared families -- coresident_parent, multigen "
        "stock, the multigen entry/exit transitions, and parental_home_exit -- "
        "plus the candidate-4 coresident_spouse family come off the candidate-4 "
        "streams UNCHANGED before any candidate-5 delta, and the deltas draw "
        "from streams that do not touch those, so their per-seed scores are "
        "IDENTICAL to candidate 4 to bit precision."
    ),
    "multigen_marginal_unchanged": (
        "The delta-1 coupling constraint: the multigen occupancy marginal must "
        "be unchanged. Verified on the scored run -- every multigen stock cell "
        "and the multigen_entry / multigen_exit transition cells have per-seed "
        "gated scores byte-identical to candidate 4 (max deviation 0.0), "
        "because multigen is carried from candidate 1's simulate_draw and the "
        "coupling only reads it."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "VERBATIM from candidate 4: each simulated side-A holdout person is "
        "seeded from their OWN observed state at their first 2b wave, then "
        "evolved with train-fitted hazards; no parameter is estimated from "
        "side A. The candidate-5 coupled adult-child and parent-count draws are "
        "conditional Bernoulli devices on the isolated 0xC5 stream, not "
        "observed states."
    ),
    "gates_yaml_path": (
        "The locked block is gates.yaml gates.gate_2.gate_2b.thresholds; its "
        "46 gated tolerances and the floor gate_partition are read at runtime, "
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
def _delta_stats(model: hcs5.HouseholdCompositionModelV5, dmean) -> dict:
    return {
        "delta_1_multigen_child_coupling": {
            "n_coupled_grandchild_waves_simulated": dmean(
                "n_coupled_grandchild_waves_simulated"
            ),
            "fitted_p_child_given_multigen_true_female": model.meta[
                "coupling_multigen_child"
            ]["pooled_p_child_given_multigen_true_female"],
            "fitted_p_child_given_multigen_false_female": model.meta[
                "coupling_multigen_child"
            ]["pooled_p_child_given_multigen_false_female"],
            "p_child_given_multigen_true_female_by_band": model.meta[
                "coupling_multigen_child"
            ]["p_child_given_multigen_true_female_by_band"],
            "sim_gc55f_multigen": dmean("coupling_gc55f_multigen"),
            "sim_gc55f_joint_mg_child_notparent": dmean(
                "coupling_gc55f_joint_mg_child_notparent"
            ),
            "sim_gc55f_independence_product": dmean(
                "coupling_gc55f_independence_product"
            ),
            "sim_gc55f_composed_grandchild": dmean(
                "coupling_gc55f_composed_grandchild"
            ),
            "sim_gc55f_skiponly": dmean("coupling_gc55f_skiponly"),
            "sim_gc55f_union": dmean("coupling_gc55f_union"),
        },
        "delta_2_not_married_custodial": {
            "n_linked_child_coresident_wave_units": dmean(
                "n_linked_child_coresident_wave_units"
            ),
            "not_married_child_record_by_band": model.meta[
                "custodial_child_record"
            ]["not_married_child_record_by_band"],
            "n_child_record_waves_train": model.meta["custodial_child_record"][
                "n_child_record_waves_train"
            ],
        },
        "delta_3_bridge_reach_and_parent_count": {
            "mean_nonfamily_count_simulated": dmean(
                "mean_nonfamily_count_simulated"
            ),
            "p_noncore_member_present_by_core": model.meta[
                "nonfamily_by_core"
            ]["p_noncore_member_present_by_core"],
            "n_dense_core_band_sex_cells": model.meta["nonfamily_by_core"][
                "n_dense_core_band_sex_cells"
            ],
            "pooled_p_two_parents": model.meta["parent_count_composition"][
                "pooled_p_two_parents_given_coresident_parent"
            ],
            "sim_mean_n_parents_among_coresident_parent": dmean(
                "mean_n_parents_among_coresident_parent"
            ),
            "sim_size3_core_total": dmean("size3_core_total"),
            "sim_size3_full_total": dmean("size3_full_total"),
            "sim_size3_route_three_adults": dmean("size3_route_three_adults"),
            "sim_size3_route_couple_plus_parent": dmean(
                "size3_route_couple_plus_parent"
            ),
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

    model = hcs5.fit_household_model_v5(
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
        sim_panel, diag = hcs5.simulate_draw_v5(
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

    delta_stats = _delta_stats(model, _dmean)
    coverage = {
        **coverage,
        "mean_paternal_linked_births": _dmean("n_paternal_linked_births"),
        "mean_paternal_shadow_births": _dmean("n_paternal_shadow_births"),
        "mean_maternal_births": _dmean("n_maternal_births"),
    }
    fit_checks = None
    if compute_fit_checks:
        fit_checks = hcs5.coupling_and_gap_checks(model, forensics)

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
        "component_meta": {
            k: v
            for k, v in model.meta.items()
            if k
            in (
                "family_transitions_spec_sha256",
                "n_train_persons",
                "parent_count",
                "grandchild_coupling_age_lo",
                "core_size_cap",
            )
        },
        "father_link_coverage": coverage,
        "delta_stats": delta_stats,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "undefined_gated_draws": undefined,
        "coupling_and_gap_checks": fit_checks,
        "elapsed_seconds": elapsed,
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
        "certified legal-marriage registry UNION the age-refined cohabitation "
        "overlay UNION the legal-spouse residual overlay (candidate 4, carried "
        "byte-faithfully; no candidate-5 delta touches it)."
    ),
    "coresident_parent": (
        "directly fitted logistic exit hazard (candidate 1, byte-faithful, "
        "RNG-isolated); expected to clear."
    ),
    "coresident_child": (
        "observed father->child links gated per wave by the custodial "
        "probability, with the delta-2 not-married child-record correction; "
        "maternal + unlinked shadow byte-faithful (women unchanged)."
    ),
    "coresident_grandchild": (
        "composed implication (multigen AND child AND NOT parent) with the "
        "delta-1 multigen--adult-child coupling at 55+ UNION the carried "
        "5-year skip-gen occupancy."
    ),
    "multigen_stock": (
        "carried initial state + train band x sex entry/exit (candidate 1, "
        "byte-faithful, RNG-isolated); the delta-1 coupling READS it but never "
        "changes the marginal."
    ),
    "multigen_transition": (
        "directly fitted pooled entry/exit rates (candidate 1, byte-faithful, "
        "RNG-isolated); unchanged by the coupling."
    ),
    "parental_home_exit": (
        "directly fitted (candidate 1, byte-faithful, RNG-isolated); expected "
        "to clear."
    ),
    "hh_size": (
        "composed ego-centric family unit (parent count drawn 1 vs 2, delta "
        "3b) PLUS the non-family bridge re-fit conditional on core size (delta "
        "3a) so size-3 cores are lifted upward at the train rate."
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
        finite = [x for x in scores if math.isfinite(x)]
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
            "The 47 report-only cells (below the 20-event floor, above the "
            "T_max power cap, or superseded by a gating aggregate). Same "
            "|ln(rbar / rate_a)| statistic; never gated."
        ),
        "cells": out,
    }


# --------------------------------------------------------------------------
# Candidate 1 -> ... -> 5 progression + regression + byte carry + checks
# --------------------------------------------------------------------------
def comparison_across_candidates(
    decomposition: dict[str, Any],
    per_seed: list[dict[str, Any]],
    tol: dict[str, float],
) -> dict[str, Any]:
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())
    c3 = json.loads(CANDIDATE3_ARTIFACT.read_text())
    c4 = json.loads(CANDIDATE4_ARTIFACT.read_text())
    c1d = c1["per_family_decomposition"]
    c2d = c2["per_family_decomposition"]
    c3d = c3["per_family_decomposition"]
    c4d = c4["per_family_decomposition"]
    per_family: dict[str, Any] = {}
    for fam, d5 in decomposition.items():
        r4 = c4d.get(fam, {}).get("cell_seed_pass_rate")
        per_family[fam] = {
            "candidate1_pass_rate": c1d.get(fam, {}).get(
                "cell_seed_pass_rate"
            ),
            "candidate2_pass_rate": c2d.get(fam, {}).get(
                "cell_seed_pass_rate"
            ),
            "candidate3_pass_rate": c3d.get(fam, {}).get(
                "cell_seed_pass_rate"
            ),
            "candidate4_pass_rate": r4,
            "candidate5_pass_rate": d5["cell_seed_pass_rate"],
            "delta_c4_to_c5": (
                round(d5["cell_seed_pass_rate"] - r4, 4)
                if r4 is not None
                else None
            ),
            "candidate4_worst_cell": c4d.get(fam, {}).get("worst_cell"),
            "candidate4_worst_mean_abs_ln": c4d.get(fam, {}).get(
                "worst_cell_mean_abs_ln"
            ),
            "candidate5_worst_cell": d5["worst_cell"],
            "candidate5_worst_mean_abs_ln": d5["worst_cell_mean_abs_ln"],
        }

    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE4_CLEARED_FAMILIES:
        r4 = c4d.get(fam, {}).get("cell_seed_pass_rate")
        r5 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r5 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate4_pass_rate": r4,
            "candidate5_pass_rate": r5,
            "still_clears": bool(holds),
        }

    # Byte-identical carried-family score check vs candidate 4 (now INCLUDING
    # coresident_spouse -- no candidate-5 delta touches it).
    c4_by_seed = {s["seed"]: s for s in c4["per_seed"]}
    c5_by_seed = {s["seed"]: s for s in per_seed}
    carried_cells = [
        c
        for c in tol
        if c.startswith(
            (
                "coresident_parent.",
                "coresident_spouse.",
                "multigen.",
                "parental_home_",
            )
        )
        or c in ("multigen_entry", "multigen_exit")
    ]
    max_carry_dev = 0.0
    for seed in (s["seed"] for s in per_seed):
        for cell in carried_cells:
            s5 = c5_by_seed[seed]["gated_cells"][cell]["score"]
            s4 = c4_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s5) and math.isfinite(s4):
                max_carry_dev = max(max_carry_dev, abs(s5 - s4))

    # Multigen-marginal-unchanged check (the delta-1 load-bearing constraint).
    multigen_cells = [
        c
        for c in tol
        if c.startswith(MULTIGEN_MARGINAL_CELLS_PREFIX)
        or c in MULTIGEN_MARGINAL_CELLS_EXACT
    ]
    max_mg_dev = 0.0
    mg_detail = {}
    for cell in sorted(multigen_cells):
        cell_max = 0.0
        for seed in (s["seed"] for s in per_seed):
            s5 = c5_by_seed[seed]["gated_cells"][cell]["score"]
            s4 = c4_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s5) and math.isfinite(s4):
                cell_max = max(cell_max, abs(s5 - s4))
        mg_detail[cell] = cell_max
        max_mg_dev = max(max_mg_dev, cell_max)

    return {
        "candidate1_artifact": "runs/gate2b_hazard_v1.json",
        "candidate2_artifact": "runs/gate2b_hazard_v2.json",
        "candidate3_artifact": "runs/gate2b_hazard_v3.json",
        "candidate4_artifact": "runs/gate2b_hazard_v4.json",
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "candidate2_registration_pointer": CANDIDATE2_REGISTRATION_POINTER,
        "candidate3_registration_pointer": CANDIDATE3_REGISTRATION_POINTER,
        "candidate4_registration_pointer": CANDIDATE4_REGISTRATION_POINTER,
        "candidate1_verdict": {
            "gate_2b_pass": c1["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c1["verdict"]["n_seeds_pass"],
        },
        "candidate2_verdict": {
            "gate_2b_pass": c2["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c2["verdict"]["n_seeds_pass"],
        },
        "candidate3_verdict": {
            "gate_2b_pass": c3["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c3["verdict"]["n_seeds_pass"],
        },
        "candidate4_verdict": {
            "gate_2b_pass": c4["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c4["verdict"]["n_seeds_pass"],
        },
        "per_family_progression": per_family,
        "cleared_family_regression_check": {
            "families": list(CANDIDATE4_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-4 cleared families are carried byte-faithfully "
                "and RNG-isolated from the three deltas, so they stay cleared; "
                "this check confirms it on the scored run."
            ),
        },
        "byte_identical_carried_family_score_check": {
            "carried_cells": sorted(carried_cells),
            "max_abs_score_deviation_vs_candidate4": max_carry_dev,
            "byte_identical": bool(max_carry_dev <= EXACT_ATOL),
            "note": (
                "Every carried cell's per-seed gated score equals candidate "
                "4's to bit precision (parent / spouse / multigen / "
                "parental-home / multigen transitions). coresident_spouse IS "
                "carried in candidate 5 (no c5 delta targets it). The three "
                "deltas either re-fit shape-preserving probability tables or "
                "draw from the isolated 0xC5 stream, and cannot perturb the "
                "carried families."
            ),
        },
        "multigen_marginal_unchanged_check": {
            "multigen_cells": sorted(multigen_cells),
            "max_abs_score_deviation_vs_candidate4": max_mg_dev,
            "per_cell_max_abs_score_deviation": mg_detail,
            "marginal_unchanged": bool(max_mg_dev <= EXACT_ATOL),
            "note": (
                "The delta-1 load-bearing constraint: the coupling reallocates "
                "WHICH multigen egos carry the adult child but never changes "
                "how many egos are multigen. multigen comes off candidate 1's "
                "simulate_draw unchanged and the coupling only reads it, so "
                "every multigen stock and transition cell is byte-identical to "
                "candidate 4."
            ),
        },
    }


def carried_blocker_analysis(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Classify each seed's failing cells as delta-target or carried-blocker.

    A CARRIED cell (byte-identical to candidate 4) that fails caps the seed
    regardless of the three deltas -- the explicit, falsifiable record of the
    'measured deltas can work while the mechanism LIST stays incomplete'
    pattern the candidate-4 grading named.
    """
    carried_families = {
        "coresident_spouse",
        "coresident_parent",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
    }
    # coresident_child female cells are carried byte-identical too (delta 2
    # touches paternal-linked MALE fathers only; women have no paternal link).
    per_seed_out: dict[int, Any] = {}
    for s in per_seed:
        fails = [c for c in sorted(tol) if not s["gated_cells"][c]["pass"]]
        carried_blockers = []
        delta_targets = []
        for c in fails:
            fam = _family_of(c)
            is_carried = fam in carried_families or (
                fam == "coresident_child" and c.endswith("|female")
            )
            (carried_blockers if is_carried else delta_targets).append(c)
        per_seed_out[s["seed"]] = {
            "n_fail": len(fails),
            "carried_blockers": carried_blockers,
            "delta_target_fails": delta_targets,
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
            "Carried blockers are cells byte-identical to candidate 4 that "
            "still fail (coresident_spouse.25-34|female; coresident_child."
            "45-54|female -- women's coresident_child is maternal-only and "
            "untouched by the male-father delta 2). They cap the affected "
            "seeds regardless of the three deltas: the honest record of an "
            "incomplete mechanism list, not a delta regression."
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

    per_seed: list[dict[str, Any]] = []
    fit_checks = None
    for seed in seeds:
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
        if rec["coupling_and_gap_checks"] is not None:
            fit_checks = rec["coupling_and_gap_checks"]
        rec.pop("coupling_and_gap_checks", None)
        per_seed.append(rec)

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
                f"(c4 {c['candidate4_pass_rate']}, d_c4c5 "
                f"{c['delta_c4_to_c5']}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} tol={d['worst_cell_tolerance']}"
            )
        byt = comparison["byte_identical_carried_family_score_check"]
        mg = comparison["multigen_marginal_unchanged_check"]
        print(
            "  carried byte-identical vs c4="
            f"{byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate4']:.2e}); "
            f"multigen-marginal-unchanged={mg['marginal_unchanged']}"
        )
        print(
            "  carried blockers cap "
            f"{blocker['n_seeds_capped_by_carried_cell']} seed(s); "
            f"max attainable = "
            f"{blocker['max_attainable_seeds_given_carried_blockers']}/5"
        )

    from populace_dynamics.models import household_composition_sim_v4 as hcs4

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "candidate": "candidate 5",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate4_registration_pointer": CANDIDATE4_REGISTRATION_POINTER,
        "candidate3_registration_pointer": CANDIDATE3_REGISTRATION_POINTER,
        "candidate2_registration_pointer": CANDIDATE2_REGISTRATION_POINTER,
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "grading_pointer": GRADING_POINTER,
        "forensics_pointer": FORENSICS_POINTER,
        "forensics_artifact": "runs/gate2b_forensics2_v1.json",
        "one_shot": (
            "Registered on issue #42 comment 4945159933 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_4": [
            "delta 1: multigen--adult-child coupling -- for 55+ egos, replace "
            "the independent coresident-own-adult-child input to the composed "
            "grandchild with a train-fitted P(child | multigen, band, sex) on "
            "the simulated multigen occupancy (0xC5); multigen MARGINAL "
            "unchanged",
            "delta 2: not-married custodial correction -- for NOT-married "
            "linked fathers, swap the observable-basis custodial probability "
            "for the child-record-basis rate; the young-married gate untouched",
            "delta 3: bridge reach conditional on core size (lift size-3 "
            "cores) + parent_count composition (draw 1 vs 2 coresident parents "
            "from the train composition on 0xC5) -- both feed hh_size only",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 4's generator REUSED byte-faithfully (which reuses "
                "candidates 3, 2 and 1), plus three train-fitted deltas: a "
                "multigen--adult-child coupling at 55+ (delta 1), a "
                "not-married child-record custodial correction (delta 2), and "
                "a core-size-conditional non-family bridge plus a per-ego "
                "parent-count composition (delta 3)."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate4_module": (
                "populace_dynamics.models.household_composition_sim_v4"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v5"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "grandchild_coupling_age_lo": hcs5.GRANDCHILD_LO,
            "core_size_cap": hcs5.CORE_SIZE_CAP,
            "coupling_age_bands_55plus": [
                list(b) for b in hcs5.COUPLING_AGE_BANDS_55PLUS
            ],
            "custodial_child_age_bands": [
                list(b) for b in hcs3.CUSTODIAL_CHILD_AGE_BANDS
            ],
            "delta_stream_tag_v5": hcs5.DELTA_STREAM_TAG_V5,
            "components": [
                "coresident_spouse<-CARRIED candidate 4 (legal|cohab|residual)",
                "coresident_parent<-CARRIED candidate 1 logistic exit hazard",
                "multigen<-CARRIED candidate 1 (coupling reads, never changes)",
                "coresident_child<-custodial_gated with not-married child-record"
                " correction (delta 2) + maternal + shadow (women byte-faithful)",
                "coresident_grandchild<-composed(multigen&child&~parent) with "
                "55+ multigen--adult-child coupling (delta 1) | skipgen",
                "hh_size<-composed(1+spouse+children+parent_count_drawn) + "
                "nonfamily bridge conditional on core size (delta 3)",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from the candidate-4 streams UNCHANGED "
                "(0xB2B occupancy; 0xC2 cohabitation/child; 0xC3 skipgen; 0xC4 "
                "legal-residual/nonfamily-2+); the candidate-3 custodial and "
                "non-family class draws are re-run byte-identically in "
                "consumption (delta 2 and delta 3a change probability VALUES, "
                "not draw shape); the two new candidate-5 components (the "
                "coupled adult-child draw; the per-ego parent-count draw) draw "
                "from a separate SeedSequence([5200+k, 0xC5]).spawn(2)"
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
        "coupling_and_gap_checks": {
            "computed_on_seed": FIT_VS_RAW_SEED,
            "note": (
                "Delta 1 joint-vs-product and delta 3 gap-closure fits vs the "
                "forensics-2 measured quantities (fit seed's train side B). "
                "The realized sim joint / size-3 routes (mean over the scored "
                "holdout draws) are in per_seed[].delta_stats."
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
            "candidate4_artifact_sha256": _sha_of_file(CANDIDATE4_ARTIFACT),
            "forensics_artifact_sha256": _sha_of_file(FORENSICS_RUN),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _ = hcs4  # imported for provenance parity; carried families come off it
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ARTIFACT_PATH))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    artifact = run(
        verbose=True,
        seeds=seeds,
        n_draws=args.draws,
        artifact_path=Path(args.out),
    )
    artifacts.write_new(Path(args.out), _json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
