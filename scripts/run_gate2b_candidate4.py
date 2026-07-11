"""Gate-2b candidate 4: the forensics-1 quartet.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b; 46 gated household/relationship cells, K=20 mean-over-draws,
per-seed train/holdout protocol). Registered on issue #42 comment 4941160621
BEFORE this run; published REGARDLESS of verdict.

Candidate 4 is candidate 3 (registration 4939960467; PR #136) with EXACTLY
FOUR frozen deltas, one per residual mechanism the gate-2b forensics-1
decomposition (``runs/gate2b_forensics1_v1.json``, grading 4941157359)
quantified, each feeding a DISJOINT failure surface. Everything candidate 3
cleared or carried -- the certified tranche-2a marital core and maternal
births, the parental-home exit hazard, the multigen entry/exit machinery,
``coresident_parent`` -- is carried BYTE-FAITHFULLY (the candidate-4 model
REUSES the candidate-3 module and re-runs candidate 3's exact draw streams; the
carried families come off candidate 1's ``simulate_draw`` UNCHANGED, so their
per-seed scores are IDENTICAL to candidate 3 to bit precision).

Delta 1 -- age-refined cohabitation overlay (``coresident_spouse`` young):
re-fit the code-22 entry/exit hazards by single year of age within 15-34.
Delta 2 -- legal-spouse residual top-up (``coresident_spouse`` older male): an
additive train-fitted occupancy overlay for the code-20 legal mass the core
under-produces, unioned into ``coresident_spouse`` on an isolated 0xC4 stream.
Delta 3 -- custodial-gate refinement (single-year child age x era x father
marital) + non-family 2+ tail spread (``coresident_child`` male / ``hh_size``).
Delta 4 -- skipped-generation level rebuild (5-year entry/exit within 55+ so
the stationary stock tracks the raw age-graded train stock).

Estimator, artifact schema, undefined-draw rule, and protocol are candidate
3's (locked contract). Artifact ``runs/gate2b_hazard_v4.json``;
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
from populace_dynamics.models import household_composition_sim_v4 as hcs4

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v4.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v4"
RUN_NAME = "gate2b_hazard_v4"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS_RUN = ROOT / "runs" / "gate2b_forensics1_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
CANDIDATE3_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v3.json"
REGISTRATION_POINTER = "4941160621"
CANDIDATE3_REGISTRATION_POINTER = "4939960467"
CANDIDATE2_REGISTRATION_POINTER = "4939456379"
CANDIDATE1_REGISTRATION_POINTER = "4938726107"
GRADING_POINTER = "4941157359"
FORENSICS_POINTER = "4940442065"
SPEC_REGISTRATION = (
    "issue #42 comment 4941160621: gate-2b candidate 4, the forensics-1 "
    "quartet (age-refined cohabitation overlay; legal-spouse residual top-up; "
    "custodial-gate refit + non-family 2+ tail spread; skip-gen level rebuild)"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12
FIT_VS_RAW_SEED = 0

#: The candidate-3 families that CLEARED (grading 4939958136 / forensics 1):
#: the regression check confirms candidate 4 carries them byte-faithfully.
#: (coresident_spouse is NO LONGER carried -- deltas 1 and 2 target it.)
CANDIDATE3_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)

#: Per-delta target family (each delta feeds a disjoint failure surface;
#: recorded so attribution stays clean and any leak is visible).
PER_DELTA_TARGET_FAMILY = {
    "delta_1_age_refined_cohabitation": "coresident_spouse",
    "delta_2_legal_spouse_residual": "coresident_spouse",
    "delta_3_custodial_refit_and_nonfamily_tail": [
        "coresident_child",
        "hh_size",
    ],
    "delta_4_skipgen_level_rebuild": "coresident_grandchild",
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.50-0.65",
    "named_expectations": [
        "the spouse family clears in both directions (young via delta 1, "
        "older via delta 2)",
        "coresident_child male cells clear or sit within 1.2x tolerance",
        "hh_size.3 and .4 clear with 5+ improving to within ~1.5x tolerance "
        "(deep-tail composition keeps residual risk)",
        "grandchild 55+|female reaches within ~1.5x tolerance (stock-building "
        "under a high fitted exit is the least certain mechanism -- named as "
        "the modal residual together with hh_size.5+)",
        "carried families stay byte-identical",
    ],
    "modal_outcome_if_fail": (
        "1-2 chronic cells among {hh_size.5+, grandchild 55+|female}, "
        "everything else clearing -- candidate 5 would then be a "
        "single-mechanism run"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42 (grading 4941157359 "
        "context); this run does NOT grade it."
    ),
}

#: Candidate 3's resolutions carry (the carried components are byte-faithful),
#: plus the four candidate-4 delta resolutions. Adopted per the registration
#: ("ambiguities resolved per the locked contract text and documented").
SPEC_RESOLUTION_NOTES = {
    "rng_byte_identical_carried_families": (
        "Candidate 4 adds two stochastic components (the legal-spouse residual "
        "overlay; the non-family 2+ count spread) on a SEPARATE "
        "SeedSequence([5200+k, 0xC4]).spawn(2), and re-fits three existing "
        "hazard/probability tables (single-year cohabitation; custodial "
        "coresidence; 5-year skip-gen) whose _evolve_two_state / per-child "
        "draws consume RNG BY SHAPE ONLY -- so the candidate-2 (0xC2) and "
        "candidate-3 (0xC3) streams stay byte-identical in consumption and "
        "only the targeted states move. The carried coresident_parent, "
        "multigen (stock + transitions) and parental_home_exit families come "
        "off candidate 1's simulate_draw at 0xB2B UNCHANGED, so they are "
        "bit-identical to candidate 3 on every draw and seed (regression is "
        "impossible by construction). coresident_spouse is NO LONGER "
        "byte-identical (deltas 1 and 2 target it)."
    ),
    "delta_1_age_refined_cohabitation": (
        "The candidate-2 cohabitation (MX8 code-22) overlay is fit "
        "band-constant; the forensics measured a ~351x within-15-24 "
        "single-year stock gradient (male) a flat band hazard mis-places, "
        "over-producing the young overshoot and under-accumulating the 25-34 "
        "stock. Candidate 4 re-fits the code-22 entry/exit hazards by SINGLE "
        "YEAR of age within 15-34 (the forensics spouse single-year window), "
        "train-side, replacing the band hazards on those ages; ages 35+ keep "
        "the carried band hazards. A single-year stratum thinner than 20 "
        "weighted at-risk waves falls back to the carried band hazard. The "
        "cohabitation _evolve_two_state draw is unchanged in shape so its "
        "stream is byte-identical; only the young cohabitation stock moves "
        "(down at 15-18 where the raw stock is near-zero, up at 20-24 where "
        "it accumulates and carries into 25-34)."
    ),
    "delta_2_legal_spouse_residual": (
        "The certified 2a registry under-produces the code-20 LEGAL spouse "
        "stock the forensics found at the 65+ male cells. Candidate 4 adds an "
        "ADDITIVE two-state occupancy overlay unioned into coresident_spouse "
        "on the isolated 0xC4 stream (the certified 2a machinery is "
        "UNTOUCHED). RESOLUTIONS: (a) the residual is sized per band x sex to "
        "target = max(0, ref_code20_stock - core_legal_stock), the train "
        "observed code-20 stock minus the certified core's simulated legal "
        "stock (one core simulate on the TRAIN persons at seed 5200, side B "
        "only -- no holdout contact); where target<=0 the overlay is off. "
        "(b) to make the union's ADDED mass equal target (the overlay "
        "overlaps the core, both being legal), the overlay's marginal "
        "stationary stock is target/(1-core_legal). (c) the overlay is "
        "initialized from a Bernoulli(marginal) draw and its exit hazard is "
        "floored at 0.5 so it reaches the marginal stock across a band tenure "
        "regardless of the age a person enters the panel (a low legal exit "
        "would build far too slowly for people aging into 65+); the residual "
        "transition is report-only (spousal_loss is not gated), so the fast "
        "turnover has no gated effect -- only the stock cells receive the "
        "top-up."
    ),
    "delta_3a_custodial_refit": (
        "The forensics located the male coresident_child overshoot in the "
        "custodial gate's young-child probabilities. Candidate 4 re-fits "
        "P(linked child coresident) with SINGLE-YEAR child age and ERA (the "
        "floor era slices: pre-1997 / 1997-2009 / 2010-2023), train-side. "
        "RESOLUTION: the registration names 'single-year child age and era'; "
        "candidate 4 refines candidate 3's (child age band x father marital) "
        "table to single-year age plus era and RETAINS the father-marital "
        "gate (the draining lever the custodial-gate diagnosis rests on; "
        "dropping it RAISES rather than drains the overshoot -- verified). An "
        "(age, era, marital) stratum thinner than 20 exposures falls back to "
        "(age, marital), then to the carried candidate-3 (band, marital), "
        "then to the overall rate. The per-child draw is unchanged in shape "
        "(byte-identical custodial stream); the gate is applied on the "
        "SIMULATED father marital, exactly as candidate 3. NOTE (recorded, "
        "not tuned): the single-year refit does NOT drain the male overshoot "
        "-- un-averaging the band exposes that the common young-child ages "
        "carry ~0.95-0.99 coresidence (higher than the band average, which "
        "the low age-0 and the 25-60 tail dilute), so the fitted probability "
        "faithfully reproduces the high young-child coresidence rather than "
        "lowering it; the overshoot lives in the observable-subset selection "
        "the custodial fit conditions on, which no age/era refinement "
        "changes. The delta is implemented per the registration and its "
        "effect published as measured."
    ),
    "delta_3b_nonfamily_tail_spread": (
        "Candidate 3's non-family bridge reads a sampled 2+ class as exactly "
        "2 (the minimal cap); the forensics found the 2+ households truly "
        "average ~2.84 members, so the cap under-fills hh_size.4/5+. Candidate "
        "4 draws the 2+ count from the train 2+ count distribution by ego age "
        "band x sex (support 2, 3, 4, ...). RESOLUTION: the 0/1/2+ CLASS draw "
        "is byte-identical to candidate 3 (same shares, same 0xC3 non-family "
        "stream shape), and the ACTUAL 2+ count is drawn on the isolated 0xC4 "
        "stream, so which households are 2+ is unchanged and only their depth "
        "spreads. The count feeds hh_size ONLY."
    ),
    "delta_4_skipgen_level_rebuild": (
        "The band-constant skip-gen entry/exit pair cannot build the rising "
        "older-band stock the forensics measured. Candidate 4 re-fits the "
        "skipped-generation entry AND exit hazards by 5-year age band within "
        "55+ (single-year would be too sparse), train-side, so the stationary "
        "stock tracks the raw age-graded train stock (~0.020-0.028 rising "
        "with age); a 5-year stratum thinner than 20 at-risk waves falls back "
        "to the carried candidate-3 composition-band hazard. The skip-gen "
        "_evolve_two_state draw is unchanged in shape (byte-identical "
        "stream); the state is unioned into the composed grandchild ONLY "
        "(never multigen). NOTE (recorded, not tuned): the grandchild "
        "55+|female cell is structurally bounded below the reference -- the "
        "reference (~0.063) is dominated by three-generation grandchildren "
        "the composed multigen path CANNOT reach (55+ women in three-gen "
        "homes are coded coresident_parent, excluded by the NOT-parent "
        "construction), and the skip-gen path tops out near the raw skip-gen "
        "stock (~0.024 pooled); the registration names this cell the modal "
        "residual."
    ),
    "carried_families_byte_faithful": (
        "The four candidate-3 cleared families -- coresident_parent, multigen "
        "stock, the multigen entry/exit transitions, and parental_home_exit "
        "-- come off candidate 1's simulate_draw UNCHANGED (0xB2B) before any "
        "candidate-4 delta, and the deltas draw from streams that do not "
        "touch that occupancy stream, so their per-seed scores are IDENTICAL "
        "to candidate 3 to bit precision."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "VERBATIM from candidate 3: each simulated side-A holdout person is "
        "seeded from their OWN observed coresident-parent / multigen / "
        "cohabitation / skipped-generation state at their first 2b wave, then "
        "evolved with train-fitted hazards. Using a holdout person's own "
        "window-entry state is not fitting (no parameter is estimated from "
        "side A). The delta-2 legal-residual overlay is initialized from a "
        "Bernoulli(marginal) draw (a device, not an observed state) because "
        "the certified core already carries the observed-initial legal "
        "spouse."
    ),
    "gates_yaml_path": (
        "The locked block is gates.yaml gates.gate_2.gate_2b.thresholds; its "
        "46 gated tolerances and the floor gate_partition are read at "
        "runtime, never hardcoded (candidate convention)."
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
    legal_flag = hcs4.legal_spouse_flag(rel_map)
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
def _delta_stats(model: hcs4.HouseholdCompositionModelV4, dmean) -> dict:
    cohab_young = {
        f"{age}|{sex}": round(model.cohab_entry_age[(age, sex)], 5)
        for age in range(15, 25)
        for sex in hc.SEXES
    }
    return {
        "delta_1_age_refined_cohabitation": {
            "n_cohab_person_waves_simulated": dmean(
                "n_cohab_person_waves_simulated"
            ),
            "cohab_single_year_entry_15_24": cohab_young,
            "cohab_single_year_window": [
                hcs4.COHAB_SINGLE_YEAR_LO,
                hcs4.COHAB_SINGLE_YEAR_HI,
            ],
        },
        "delta_2_legal_spouse_residual": {
            "n_legal_residual_person_waves_simulated": dmean(
                "n_legal_residual_person_waves_simulated"
            ),
            "residual_target_stock_by_band_sex": (
                model.meta["legal_residual"]["residual_target_stock"]
            ),
            "ref_code20_stock_by_band_sex": (
                model.meta["legal_residual"]["ref_code20_stock"]
            ),
            "core_legal_stock_by_band_sex": (
                model.meta["legal_residual"]["core_legal_stock"]
            ),
            "residual_marginal_by_band_sex": {
                f"{b}|{s}": round(v, 5)
                for (b, s), v in model.legal_residual_marginal.items()
            },
            "n_bands_active": model.meta["legal_residual"]["n_bands_active"],
        },
        "delta_3a_custodial_refit": {
            "n_linked_child_coresident_wave_units": dmean(
                "n_linked_child_coresident_wave_units"
            ),
            "custodial_n_train_exposure": model.meta["custodial_era"][
                "n_exposure"
            ],
            "custodial_n_train_coresident": model.meta["custodial_era"][
                "n_coresident"
            ],
            "custodial_overall_rate": model.custodial_overall,
            "n_age_era_marital_cells": model.meta["custodial_era"][
                "n_age_era_marital_cells"
            ],
        },
        "delta_3b_nonfamily_tail_spread": {
            "mean_nonfamily_count_simulated": dmean(
                "mean_nonfamily_count_simulated"
            ),
            "mean_nonfamily_count_within_2plus_simulated": dmean(
                "mean_nonfamily_count_within_2plus_simulated"
            ),
            "nonfamily_2plus_train_mean_within": model.meta["nonfamily_2plus"][
                "mean_count_within_2plus_train"
            ],
            "nonfamily_2plus_train_true_mean": model.meta["nonfamily_2plus"][
                "true_weighted_mean_count_train"
            ],
        },
        "delta_4_skipgen_level_rebuild": {
            "n_skipgen_person_waves_simulated": dmean(
                "n_skipgen_person_waves_simulated"
            ),
            "skipgen_entry_5yr_female": model.meta["skipgen_entry_5yr_female"],
            "skipgen_exit_5yr_female": model.meta["skipgen_exit_5yr_female"],
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
    compute_fit_vs_raw: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    hh = data["hh"]
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs4.fit_household_model_v4(
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
        sim_panel, diag = hcs4.simulate_draw_v4(
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
    fit_vs_raw = None
    if compute_fit_vs_raw:
        fit_vs_raw = hcs4.fit_vs_raw_checks(model, hh, ids_b, forensics)

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
                "custodial_era_overall",
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
        "fit_vs_raw_checks": fit_vs_raw,
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
        "certified legal-marriage registry UNION the age-refined "
        "cohabitation (code-22) overlay (delta 1) UNION the additive "
        "legal-spouse residual overlay (delta 2, isolated 0xC4)."
    ),
    "coresident_parent": (
        "directly fitted logistic exit hazard (candidate 1, byte-faithful, "
        "RNG-isolated); expected to clear."
    ),
    "coresident_child": (
        "observed father->child links gated per wave by the custodial "
        "coresidence probability, re-fit by single-year child age x era x "
        "father marital (delta 3a); maternal + unlinked shadow byte-faithful."
    ),
    "coresident_grandchild": (
        "composed implication (multigen AND child AND NOT parent) UNION the "
        "5-year-refit skipped-generation occupancy (delta 4); structurally "
        "bounded below the reference by the NOT-parent exclusion."
    ),
    "multigen_stock": (
        "carried initial state + train band x sex entry/exit (candidate 1, "
        "byte-faithful, RNG-isolated); no candidate-4 delta feeds it."
    ),
    "multigen_transition": (
        "directly fitted pooled entry/exit rates (candidate 1, byte-faithful, "
        "RNG-isolated); expected to clear."
    ),
    "parental_home_exit": (
        "directly fitted (candidate 1, byte-faithful, RNG-isolated); the "
        "transition expected to clear."
    ),
    "hh_size": (
        "composed ego-centric family unit PLUS the non-family bridge with the "
        "delta-3b 2+ tail spread; the size-3 family-core overshoot is the "
        "named residual (not addressed by the 2+ tail)."
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
# Candidate 1 -> 2 -> 3 -> 4 progression + regression + byte carry
# --------------------------------------------------------------------------
def comparison_across_candidates(
    decomposition: dict[str, Any],
    per_seed: list[dict[str, Any]],
    tol: dict[str, float],
) -> dict[str, Any]:
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())
    c3 = json.loads(CANDIDATE3_ARTIFACT.read_text())
    c1d = c1["per_family_decomposition"]
    c2d = c2["per_family_decomposition"]
    c3d = c3["per_family_decomposition"]
    per_family: dict[str, Any] = {}
    for fam, d4 in decomposition.items():
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
            "candidate4_pass_rate": d4["cell_seed_pass_rate"],
            "delta_c3_to_c4": (
                round(d4["cell_seed_pass_rate"] - r3, 4)
                if (r3 := c3d.get(fam, {}).get("cell_seed_pass_rate"))
                is not None
                else None
            ),
            "candidate3_worst_cell": c3d.get(fam, {}).get("worst_cell"),
            "candidate3_worst_mean_abs_ln": c3d.get(fam, {}).get(
                "worst_cell_mean_abs_ln"
            ),
            "candidate4_worst_cell": d4["worst_cell"],
            "candidate4_worst_mean_abs_ln": d4["worst_cell_mean_abs_ln"],
        }

    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE3_CLEARED_FAMILIES:
        r3 = c3d.get(fam, {}).get("cell_seed_pass_rate")
        r4 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r4 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate3_pass_rate": r3,
            "candidate4_pass_rate": r4,
            "still_clears": bool(holds),
        }

    # Byte-identical carried-family score check vs candidate 3 (NOT spouse:
    # deltas 1 and 2 target coresident_spouse).
    c3_by_seed = {s["seed"]: s for s in c3["per_seed"]}
    c4_by_seed = {s["seed"]: s for s in per_seed}
    carried_cells = [
        c
        for c in tol
        if c.startswith(("coresident_parent.", "multigen.", "parental_home_"))
        or c in ("multigen_entry", "multigen_exit")
    ]
    max_carry_dev = 0.0
    for seed in (s["seed"] for s in per_seed):
        for cell in carried_cells:
            s4 = c4_by_seed[seed]["gated_cells"][cell]["score"]
            s3 = c3_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s4) and math.isfinite(s3):
                max_carry_dev = max(max_carry_dev, abs(s4 - s3))
    return {
        "candidate1_artifact": "runs/gate2b_hazard_v1.json",
        "candidate2_artifact": "runs/gate2b_hazard_v2.json",
        "candidate3_artifact": "runs/gate2b_hazard_v3.json",
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "candidate2_registration_pointer": CANDIDATE2_REGISTRATION_POINTER,
        "candidate3_registration_pointer": CANDIDATE3_REGISTRATION_POINTER,
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
        "per_family_progression": per_family,
        "cleared_family_regression_check": {
            "families": list(CANDIDATE3_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-3 cleared families are carried byte-faithfully "
                "(read off candidate 1's simulate_draw unchanged) and their "
                "occupancy draws are RNG-isolated from the four deltas, so "
                "they are expected to stay cleared; this check confirms it on "
                "the scored run."
            ),
        },
        "byte_identical_carried_family_score_check": {
            "carried_cells": sorted(carried_cells),
            "max_abs_score_deviation_vs_candidate3": max_carry_dev,
            "byte_identical": bool(max_carry_dev <= EXACT_ATOL),
            "note": (
                "Every carried cell's per-seed gated score equals candidate "
                "3's to bit precision (parent / multigen / parental-home / "
                "multigen transitions). coresident_spouse is NOT carried "
                "(deltas 1 and 2 target it). This is the strong regression "
                "proof: the four deltas either re-fit shape-preserving hazard "
                "tables or draw from an isolated 0xC4 stream, and cannot "
                "perturb the carried families."
            ),
        },
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
    fit_vs_raw = None
    for seed in seeds:
        rec = score_seed(
            seed,
            data,
            floor,
            tol,
            report_only,
            forensics,
            verbose,
            compute_fit_vs_raw=(seed == FIT_VS_RAW_SEED),
        )
        if rec["fit_vs_raw_checks"] is not None:
            fit_vs_raw = rec["fit_vs_raw_checks"]
        rec.pop("fit_vs_raw_checks", None)
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
                f"(c3 {c['candidate3_pass_rate']}, d_c3c4 "
                f"{c['delta_c3_to_c4']}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} tol={d['worst_cell_tolerance']}"
            )
        byt = comparison["byte_identical_carried_family_score_check"]
        print(
            "  carried byte-identical vs c3="
            f"{byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate3']:.2e})"
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "candidate": "candidate 4",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate3_registration_pointer": CANDIDATE3_REGISTRATION_POINTER,
        "candidate2_registration_pointer": CANDIDATE2_REGISTRATION_POINTER,
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "grading_pointer": GRADING_POINTER,
        "forensics_pointer": FORENSICS_POINTER,
        "forensics_artifact": "runs/gate2b_forensics1_v1.json",
        "one_shot": (
            "Registered on issue #42 comment 4941160621 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_3": [
            "delta 1: age-refined cohabitation overlay -- single-year (15-34) "
            "code-22 entry/exit hazards replacing the band-constant pair",
            "delta 2: legal-spouse residual top-up -- an additive occupancy "
            "overlay sized to (ref_code20 - core_legal) per band x sex, "
            "unioned into coresident_spouse on the isolated 0xC4 stream",
            "delta 3: custodial-gate refit (single-year child age x era x "
            "father marital) + non-family 2+ count tail spread from the train "
            "2+ distribution",
            "delta 4: skip-gen level rebuild -- 5-year (55+) entry AND exit "
            "hazards so the stationary stock tracks the raw age-graded stock",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 3's generator REUSED byte-faithfully (which "
                "reuses candidates 2 and 1), plus four train-fitted deltas: "
                "an age-refined single-year cohabitation overlay (delta 1), "
                "an additive legal-spouse residual overlay (delta 2), a "
                "custodial refit by single-year child age x era x father "
                "marital plus a non-family 2+ tail spread (delta 3), and a "
                "5-year skipped-generation level rebuild (delta 4)."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate3_module": (
                "populace_dynamics.models.household_composition_sim_v3"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v4"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "cohab_single_year_window": [
                hcs4.COHAB_SINGLE_YEAR_LO,
                hcs4.COHAB_SINGLE_YEAR_HI,
            ],
            "skipgen_5yr_bands": [
                list(b) for b in hcs4.SKIPGEN_AGE_BANDS_55PLUS
            ],
            "custodial_era_slices": list(hcs4.CUSTODIAL_ERA_SLICES),
            "legal_spouse_code": hcs4.LEGAL_SPOUSE_CODE,
            "components": [
                "coresident_spouse<-certified_married|UNION|cohab_code22"
                "_single_year|UNION|legal_residual_overlay",
                "coresident_parent<-logistic_exit_hazard_age_spline_sex",
                "multigen<-train_band_sex_entry_exit_carried_initial",
                "coresident_child<-custodial_gated(single_year_age_x_era_x"
                "_marital)_father_links+maternal_kernel+shadow_unlinked",
                "hh_size<-composed(1+spouse+children+parents)+nonfamily"
                "_bridge_2plus_tail_spread",
                "coresident_grandchild<-composed(multigen&child&~parent)"
                "|UNION|skipgen_occupancy_5yr",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from candidate 1's simulate_draw "
                "UNCHANGED (occupancy tag 0xB2B); the candidate-2 (0xC2) and "
                "candidate-3 (0xC3) streams are re-run byte-identically in "
                "consumption (the delta re-fits change hazard VALUES, not draw "
                "shape); the two new candidate-4 components (legal-spouse "
                "residual; non-family 2+ count spread) draw from a separate "
                "SeedSequence([5200+k, 0xC4]).spawn(2)"
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
        "fit_vs_raw_checks": {
            "computed_on_seed": FIT_VS_RAW_SEED,
            "note": (
                "Each delta's fitted objects verified against the forensics-1 "
                "raw gradients on the fit seed's train side B (the fitted "
                "objects are a property of the fit, not the holdout)."
            ),
            "checks": fit_vs_raw,
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
        "report_only": report_block,
        "revision_pins": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "head_sha": _git_sha(),
            "base_sha": _merge_base(),
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "candidate3_artifact_sha256": _sha_of_file(CANDIDATE3_ARTIFACT),
            "forensics_artifact_sha256": _sha_of_file(FORENSICS_RUN),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
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
