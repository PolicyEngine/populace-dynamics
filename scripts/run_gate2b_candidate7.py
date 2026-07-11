"""Gate-2b candidate 7: enumeration conditioning + episode persistence.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b; 46 gated household/relationship cells, K=20 mean-over-draws,
per-seed train/holdout protocol). Registered on issue #42 comment 4948186843
BEFORE this run; published REGARDLESS of verdict.

Candidate 7 is candidate 6 (registration 4946285556; PR #143) with EXACTLY TWO
frozen deltas, each designed against a graded gate-2b forensics-4 finding
(``runs/gate2b_forensics4_v1.json``, grading 4948183531). Everything candidate 6
cleared or carried -- the certified tranche-2a marital core, the carried
``coresident_parent`` / ``coresident_spouse`` (INCLUDING the fragile
``coresident_spouse.25-34|female`` cell, carried UNTOUCHED with its 2/5
split-seed fragility on the record) / ``multigen`` (stock + transitions) /
``parental_home_exit``, the 55+ coupled grandchild, and the count-conditional
bridge -- is carried BYTE-FAITHFULLY (candidate 7 REUSES the candidate-6
generator and re-runs its exact 0xB2B / 0xC2 / 0xC3 / 0xC4 / 0xC5 / 0xC6
streams; the two candidate-7 deltas REPLACE the linked-father child coresidence
draw on the isolated 0xC7 stream).

Delta 1 -- enumeration conditioning (``coresident_child`` 25-34|male, 35-44|
male): restrict the paternal-linked coresidence draw to ENUMERATED children (the
joinable (parent, birth_year) keys); the non-joinable biological children the
committed candidate-6 draw coresides but the reference roster can never observe
(25.8% of linked exposure, 9,500 of 36,887) are excluded, removing the dominant
unenumerated_nonjoinable_supply channel (+0.035/+0.036).
Delta 2 -- episode persistence (``coresident_child`` 25-34|male, 35-44|male):
replace the independent per-wave occupancy (sim mean episode 3.57 waves) with a
correlated entry/persist/exit process fitted to the train episode-length mean
(~5.93 waves), CONSTRAINED to preserve the per-wave custodial marginal by band
exactly -- removing the spell_length fragmentation channel (+0.018/+0.022).

The shadow (unlinked imputed-paternal) channel is a NAMED residual, untouched.
Estimator, artifact schema, undefined-draw rule, and protocol are the locked
contract. Artifact ``runs/gate2b_hazard_v7.json``; ``artifacts.write_new``
refuses to overwrite. Reproduce with ``.venv-gate`` and the PSID products
staged (``POPULACE_DYNAMICS_PSID_DIR``).
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
from populace_dynamics.models import household_composition_sim_v7 as hcs7

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v7.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v7"
RUN_NAME = "gate2b_hazard_v7"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS_RUN = ROOT / "runs" / "gate2b_forensics4_v1.json"
CANDIDATE_ARTIFACTS = {
    1: ROOT / "runs" / "gate2b_hazard_v1.json",
    2: ROOT / "runs" / "gate2b_hazard_v2.json",
    3: ROOT / "runs" / "gate2b_hazard_v3.json",
    4: ROOT / "runs" / "gate2b_hazard_v4.json",
    5: ROOT / "runs" / "gate2b_hazard_v5.json",
    6: ROOT / "runs" / "gate2b_hazard_v6.json",
}
REGISTRATION_POINTER = "4948186843"
CANDIDATE6_REGISTRATION_POINTER = "4946285556"
CANDIDATE6_GRADING_POINTER = "4947225286"
GRADING_POINTER = "4948183531"  # forensics-4 grading (c7 designs against it)
FORENSICS_REGISTRATION_POINTER = "4947226688"
SPEC_REGISTRATION = (
    "issue #42 comment 4948186843: gate-2b candidate 7, enumeration "
    "conditioning (restrict the paternal-linked draw to enumerated children) "
    "and episode persistence (correlated entry/persist/exit for linked-father "
    "coresidence, fitted to the 5.93-wave episode mean, preserving the per-wave "
    "marginal by band)"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12
FIT_VS_RAW_SEED = 0

#: Families carried BYTE-IDENTICAL to candidate 6: no candidate-7 delta touches
#: their streams (they come off 0xB2B / 0xC2 / 0xC4 / 0xC5 before the linked
#: draw, and the two deltas draw only from the isolated 0xC7 stream). This
#: INCLUDES all of coresident_spouse -- candidate 7 makes NO spouse change, so
#: the fragile coresident_spouse.25-34|female cell is carried UNTOUCHED.
CANDIDATE6_STRICT_CARRIED_FAMILIES = (
    "coresident_parent",
    "coresident_spouse",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)
#: Candidate-6 families whose pass rate was 1.0 and must STAY cleared. The
#: coresident_grandchild.55+|female coupled cell is byte-identical (independent
#: of the linked child count); the 45-54 composed cell has wide tolerance.
CANDIDATE6_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_grandchild",
)
MULTIGEN_MARGINAL_CELLS_PREFIX = ("multigen.",)
MULTIGEN_MARGINAL_CELLS_EXACT = ("multigen_entry", "multigen_exit")
#: The fragile cell carried UNTOUCHED (byte-identical to candidate 6).
FRAGILE_SPOUSE_CELL = "coresident_spouse.25-34|female"
#: hh_size cells the registration names as carries to watch (they move via the
#: delta-1 linked-child-count reduction; .3/.4 are expected to still clear).
HH_SIZE_CARRY_CELLS = ("hh_size.3", "hh_size.4")

#: Per-delta target family (both deltas feed the coresident_child failure).
PER_DELTA_TARGET_FAMILY = {
    "delta_1_enumeration_conditioning": "coresident_child",
    "delta_2_episode_persistence": "coresident_child",
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.55-0.70",
    "majority_side": True,
    "named_expectations": [
        "both child male cells (25-34|male, 35-44|male) clear on >= 4 seeds "
        "(the two exactly-measured levers clear both cells with margin in the "
        "channel arithmetic)",
        "the marginal-preservation check holds (any violation is a spec "
        "violation, not a finding)",
        "carries byte-identical incl. grandchild 1.00, the multigen marginal, "
        "and hh_size.3/.4",
        "the fragile coresident_spouse.25-34|female cell is carried UNTOUCHED "
        "(2/5 split-seed exceedance priced into the forecast, not assumed)",
    ],
    "modal_outcome_if_fail": (
        "seed 3 (hh_size.5+ structural fertility deficit) plus one spouse-"
        "capped seed -- the budget spent twice"
    ),
    "named_residual_risk": (
        "channel interactions under the new composition (the forensics-4 "
        "channels were measured under candidate-6 composition; the negative "
        "marital-joint / non-linked offsets may shift)"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT grade "
        "it."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "rng_byte_identical_carried_families": (
        "Candidate 7 re-runs candidate 6's exact 0xB2B / 0xC2 / 0xC3 / 0xC4 / "
        "0xC5 / 0xC6 streams. The two candidate-7 deltas REPLACE the linked-"
        "father child coresidence draw with an enumeration-conditioned, "
        "episode-persistent draw on a SEPARATE SeedSequence([5200+k, 0xC7]); "
        "the candidate-6 0xC3 custodial spawn is retired (its non-family and "
        "skip-gen sibling spawns are preserved so those stay byte-identical). "
        "The carried coresident_parent, coresident_spouse (ALL bands), multigen "
        "(stock + transitions) and parental_home_exit families come off the "
        "carried streams UNCHANGED, so they are bit-identical to candidate 6 on "
        "every draw and seed (regression is impossible by construction)."
    ),
    "delta_1_enumeration_conditioning": (
        "Forensics-4 Q11 found the dominant linked driver at both male child "
        "cells is unenumerated_nonjoinable_supply (+0.035 at 25-34|male, +0.036 "
        "at 35-44|male): the committed candidate-6 draw coresides over "
        "model.father_links (father_link_births), which does NOT require an "
        "enumerated child -- 25.8% of its exposure rows (9,500 of 36,887) are "
        "non-joinable biological children with no enumerated household record, "
        "coresidence the reference roster can never observe (a concept artifact, "
        "not a behavioral claim). RESOLUTION: candidate 7 restricts the paternal-"
        "linked draw to ENUMERATED children (the (parent_person_id, birth_year) "
        "keys present in father_links_child); a non-joinable exposure row cannot "
        "be drawn coresident. Deterministic filter (no RNG); the non-joinable "
        "share is recorded per child band."
    ),
    "delta_2_episode_persistence": (
        "Forensics-4 Q11 found the SPELL channel (+0.018 at 25-34|male, +0.022 "
        "at 35-44|male) is the gap between the analytic INDEPENDENT per-wave "
        "occupancy (faithful custody prob) and the observed correlated "
        "coresidence: the committed candidate-6 draw applies the per-wave "
        "custody probability as INDEPENDENT per-wave occupancy, fragmenting the "
        "spells (sim mean episode 3.57 waves vs reference 5.93; sim single-wave "
        "share 0.331 vs 0.165). RESOLUTION: candidate 7 replaces the independent "
        "per-wave draw with a correlated entry/persist/exit process -- a per-"
        "father latent frailty Z_f (shared across the father's children) that a "
        "fitted fraction rho of linked children follow across all their waves (a "
        "persistent, sibling-synchronized episode: coresident while the faithful "
        "custody prob exceeds the shared latent), the rest keeping the "
        "candidate-6 independent per-wave draw. The mixture PRESERVES the per-"
        "wave custodial marginal by band EXACTLY (rho*p_c + (1-rho)*p_c = p_c), "
        "so the faithful per-wave probabilities do NOT move; rho is fitted on "
        "train side B to the episode-length mean (~5.93 waves). The persistent, "
        "sibling-synchronized fraction concentrates coresidence into fewer "
        "father-waves, reshaping the father-wave stock (the spell channel) "
        "without touching the per-wave marginal. The per-draw marginal-"
        "preservation check (simulated per-wave marginal vs train, per band) is "
        "recorded; any material violation is a spec violation, not a finding."
    ),
    "shadow_channel_named_residual": (
        "The shadow (unlinked imputed-paternal) channel (+0.032 at 25-34|male, "
        "+0.060 at 35-44|male) is a NAMED residual, deliberately UNTOUCHED -- it "
        "is offset by the marital-joint and non-linked negatives, and touching "
        "it without a measurement would be tuning. The marital-state joint "
        "(simulated vs observed father marital) is the other named residual."
    ),
    "fragile_spouse_cell_carried_untouched": (
        "coresident_spouse.25-34|female is carried UNTOUCHED (candidate 7 makes "
        "no spouse change; the cell is byte-identical to candidate 6). "
        "Forensics-4 Q12 found the cell is inherited candidate-4/5 machinery "
        "riding the tolerance line (fragile-marginal: +0.0118 seed-mean margin, "
        "2/5 SPLIT seeds exceed tolerance), NOT a candidate-6 interaction and "
        "not luck the next candidate could lose by re-drawing. Its fragility is "
        "priced into the forecast, not assumed away."
    ),
    "hh_size_5plus_structural_seed": (
        "Forensics-4 Q13 found hh_size.5+ fails split seed 3 as STRUCTURE (all "
        "20 holdout draws below reference, ~4.3 SE past the line), not noise: "
        "the upstream core-5+ / 3+-own-child fertility deficit the count-"
        "conditional bridge cannot lift (it only adds non-core members). "
        "Candidate 7 does not chase it; hh_size.5+ is priced as the structural "
        "seed-3 risk. hh_size moves under the delta-1 linked-child-count "
        "reduction; hh_size.3/.4 are expected to still clear."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "VERBATIM from candidate 6: each simulated side-A holdout person is "
        "seeded from their OWN observed state at their first 2b wave, then "
        "evolved with train-fitted hazards; no parameter is estimated from side "
        "A. The candidate-7 episode-persistence rho is fitted on train side B "
        "only (no holdout tuning surface)."
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
    model: hcs7.HouseholdCompositionModelV7, dmean, dmean_dict
) -> dict:
    enum = model.meta["enumeration_conditioning"]
    epi = model.meta["linked_episode_persistence"]
    band_share = {
        band: v.get("share", 0.0)
        for band, v in dmean_dict("linked_nonjoinable_share_by_band").items()
    }
    return {
        "delta_1_enumeration_conditioning": {
            "n_linked_exposure_rows": enum["n_linked_exposure_rows"],
            "n_joinable_exposure_rows": enum["n_joinable_exposure_rows"],
            "n_nonjoinable_exposure_rows": enum["n_nonjoinable_exposure_rows"],
            "nonjoinable_share": enum["nonjoinable_share"],
            "n_joinable_keys": enum["n_joinable_keys"],
            "sim_nonjoinable_share_by_band": band_share,
            "n_linked_child_coresident_wave_units": dmean(
                "n_linked_child_coresident_wave_units"
            ),
        },
        "delta_2_episode_persistence": {
            "fitted_persistence_rho": epi["fitted_persistence_rho"],
            "target_reference_episode_mean_train": epi[
                "target_reference_episode_mean_train"
            ],
            "candidate6_independent_episode_mean_train": epi[
                "candidate6_independent_episode_mean_train"
            ],
            "achieved_episode_mean_at_rho_train": epi[
                "achieved_episode_mean_at_rho_train"
            ],
            "sim_holdout_episode_mean_length": dmean(
                "linked_sim_episode_mean_length"
            ),
            "marginal_preservation_max_abs_dev_mean": dmean(
                "linked_marginal_preservation_max_abs_dev"
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

    model = hcs7.fit_household_model_v7(
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
        sim_panel, diag = hcs7.simulate_draw_v7(
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
        keys = draw_diagnostics[0][key].keys()
        out: dict[str, Any] = {}
        for k in keys:
            vals = [d[key][k] for d in draw_diagnostics]
            if isinstance(vals[0], dict):
                out[k] = {
                    kk: float(np.mean([v[kk] for v in vals]))
                    for kk in vals[0]
                    if isinstance(vals[0][kk], int | float)
                }
            else:
                out[k] = float(np.mean(vals))
        return out

    delta_stats = _delta_stats(model, _dmean, _dmean_dict)
    marginal_preservation = _marginal_preservation_block(draw_diagnostics)
    episode_fit_vs_raw = _episode_fit_vs_raw_block(
        model, draw_diagnostics, forensics
    )
    coverage = {
        **coverage,
        "mean_paternal_linked_births": _dmean("n_paternal_linked_births"),
        "mean_paternal_shadow_births": _dmean("n_paternal_shadow_births"),
        "mean_maternal_births": _dmean("n_maternal_births"),
    }
    fit_checks = None
    if compute_fit_checks:
        fit_checks = hcs7.c7_delta_checks(model, forensics)

    if verbose:
        fails = [k for k, v in gated_cells.items() if not v["pass"]]
        print(
            f"seed {seed}: {n_gated_pass}/{len(tol)} gated pass "
            f"(seed_pass={seed_pass}); K={N_DRAWS}; rho="
            f"{model.linked_episode_persistence:.4f}; "
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
            "grandchild_coupling_age_lo": hcs7.GRANDCHILD_LO,
            "core_size_cap": hcs7.CORE_SIZE_CAP,
            "custodial_revert_band": list(hcs7.CUSTODIAL_REVERT_BAND),
            "delta_stream_tag_v6": hcs7.DELTA_STREAM_TAG_V6,
            "delta_stream_tag_v7": hcs7.DELTA_STREAM_TAG_V7,
        },
        "father_link_coverage": coverage,
        "delta_stats": delta_stats,
        "marginal_preservation_check": marginal_preservation,
        "episode_length_fit_vs_raw": episode_fit_vs_raw,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "undefined_gated_draws": undefined,
        "c7_delta_checks": fit_checks,
        "elapsed_seconds": elapsed,
    }


def _marginal_preservation_block(
    draw_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """The delta-2 marginal-preservation check, aggregated over the K draws."""
    per_draw_max = [
        float(d["linked_marginal_preservation_max_abs_dev"])
        for d in draw_diagnostics
    ]
    bands = list(draw_diagnostics[0]["linked_marginal_preservation_by_band"])
    per_band: dict[str, Any] = {}
    for band in bands:
        targets = [
            d["linked_marginal_preservation_by_band"][band][
                "target_mean_custody_prob"
            ]
            for d in draw_diagnostics
        ]
        sims = [
            d["linked_marginal_preservation_by_band"][band][
                "sim_coresident_share"
            ]
            for d in draw_diagnostics
        ]
        per_band[band] = {
            "mean_target_custody_prob": float(np.mean(targets)),
            "mean_sim_coresident_share": float(np.mean(sims)),
            "mean_abs_deviation": float(
                np.mean(np.abs(np.asarray(sims) - np.asarray(targets)))
            ),
            "max_abs_deviation": float(
                np.max(np.abs(np.asarray(sims) - np.asarray(targets)))
            ),
        }
    return {
        "note": (
            "Delta-2 constraint: the simulated per-wave linked-coresidence "
            "marginal (over joinable exposure) vs the faithful target (mean "
            "custody prob) by child band, per draw. The mixture preserves the "
            "marginal EXACTLY in expectation, so the per-draw deviation is "
            "Monte-Carlo only; a material systematic deviation is a spec "
            "violation, not a finding."
        ),
        "mean_max_abs_dev_over_draws": float(np.mean(per_draw_max)),
        "worst_draw_max_abs_dev": float(np.max(per_draw_max)),
        "per_band": per_band,
    }


def _episode_fit_vs_raw_block(
    model: hcs7.HouseholdCompositionModelV7,
    draw_diagnostics: list[dict[str, Any]],
    forensics: dict[str, Any],
) -> dict[str, Any]:
    """The delta-2 episode-length fit-vs-raw: v6 raw 3.57 -> v7 fit -> ref 5.93."""
    epi = forensics["question_11_linked_father_child_supply"][
        "episode_length_distributions"
    ]
    sim_means = [
        float(d["linked_sim_episode_mean_length"]) for d in draw_diagnostics
    ]
    return {
        "raw_candidate6_sim_episode_mean": epi["sim"]["mean_episode_length"],
        "raw_candidate6_sim_episode_distribution": epi["sim"]["distribution"],
        "reference_episode_mean": epi["reference"]["mean_episode_length"],
        "reference_episode_distribution": epi["reference"]["distribution"],
        "fitted_persistence_rho": float(model.linked_episode_persistence),
        "train_fit_target_episode_mean": model.episode_fit[
            "target_reference_episode_mean_train"
        ],
        "train_fit_achieved_episode_mean": model.episode_fit[
            "achieved_episode_mean_at_rho_train"
        ],
        "sim_holdout_episode_mean_over_draws": float(np.mean(sim_means)),
        "sim_holdout_episode_distribution_draw0": draw_diagnostics[0][
            "linked_sim_episode_length_distribution"
        ],
        "note": (
            "Delta 2 lifts the linked coresidence episode mean from the "
            "candidate-6 fragmented 3.57 waves toward the reference 5.93 by "
            "fitting rho on train; the holdout mean is the achieved lift under "
            "the fitted rho."
        ),
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
        "CARRIED candidate 6: certified legal-marriage registry UNION the "
        "age-refined cohabitation overlay (with the carried delta-3 female "
        "25-44 refit) UNION the legal-spouse residual overlay -- byte-identical "
        "to candidate 6 (no candidate-7 spouse change)."
    ),
    "coresident_parent": (
        "CARRIED candidate 1 logistic exit hazard (byte-faithful, RNG-isolated)."
    ),
    "coresident_child": (
        "observed father->child links, ENUMERATION-CONDITIONED (delta 1: "
        "joinable children only) and drawn as correlated episodes (delta 2: the "
        "persistent, sibling-synchronized frailty on 0xC7); maternal own-birth "
        "and shadow children carried from candidate 6."
    ),
    "coresident_grandchild": (
        "composed implication (multigen AND child AND NOT parent) with the "
        "CARRIED 55+ coupling (byte-identical) UNION the carried skip-gen "
        "occupancy; the 45-54 composed cell moves with the linked child count."
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
        "composed ego-centric family unit + the CARRIED delta-4 count-"
        "conditional non-family bridge; moves with the delta-1 linked-child-"
        "count reduction."
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
# Candidate 1 -> ... -> 7 progression + regression + byte carry + checks
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
    for fam, d7 in decomposition.items():
        r6 = decomp[6].get(fam, {}).get("cell_seed_pass_rate")
        per_family[fam] = {
            f"candidate{n}_pass_rate": decomp[n]
            .get(fam, {})
            .get("cell_seed_pass_rate")
            for n in cand
        }
        per_family[fam]["candidate7_pass_rate"] = d7["cell_seed_pass_rate"]
        per_family[fam]["delta_c6_to_c7"] = (
            round(d7["cell_seed_pass_rate"] - r6, 4)
            if r6 is not None
            else None
        )
        per_family[fam]["candidate6_worst_cell"] = (
            decomp[6].get(fam, {}).get("worst_cell")
        )
        per_family[fam]["candidate7_worst_cell"] = d7["worst_cell"]
        per_family[fam]["candidate7_worst_mean_abs_ln"] = d7[
            "worst_cell_mean_abs_ln"
        ]

    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE6_CLEARED_FAMILIES:
        r6 = decomp[6].get(fam, {}).get("cell_seed_pass_rate")
        r7 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r7 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate6_pass_rate": r6,
            "candidate7_pass_rate": r7,
            "still_clears": bool(holds),
        }

    c6_by_seed = {s["seed"]: s for s in cand[6]["per_seed"]}
    c7_by_seed = {s["seed"]: s for s in per_seed}

    def _max_dev(cells: list[str]) -> float:
        d = 0.0
        for seed in (s["seed"] for s in per_seed):
            for cell in cells:
                s7 = c7_by_seed[seed]["gated_cells"][cell]["score"]
                s6 = c6_by_seed[seed]["gated_cells"][cell]["score"]
                if math.isfinite(s7) and math.isfinite(s6):
                    d = max(d, abs(s7 - s6))
        return d

    # Byte-identical carried families vs candidate 6 (incl. ALL spouse bands).
    strict_carried = [
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
    max_carry_dev = _max_dev(strict_carried)

    # The fragile spouse cell, carried UNTOUCHED (byte-identical to candidate 6).
    fragile_dev = _max_dev([FRAGILE_SPOUSE_CELL])
    fragile_per_seed = {
        seed: {
            "candidate7_score": c7_by_seed[seed]["gated_cells"][
                FRAGILE_SPOUSE_CELL
            ]["score"],
            "candidate6_score": c6_by_seed[seed]["gated_cells"][
                FRAGILE_SPOUSE_CELL
            ]["score"],
            "candidate7_pass": c7_by_seed[seed]["gated_cells"][
                FRAGILE_SPOUSE_CELL
            ]["pass"],
        }
        for seed in (s["seed"] for s in per_seed)
    }

    # Grandchild 55+ byte-identity (coupling independent of the linked count).
    gc_dev = _max_dev(["coresident_grandchild.55+|female"])

    # Multigen-marginal-unchanged check.
    multigen_cells = [
        c
        for c in tol
        if c.startswith(MULTIGEN_MARGINAL_CELLS_PREFIX)
        or c in MULTIGEN_MARGINAL_CELLS_EXACT
    ]
    mg_detail = {cell: _max_dev([cell]) for cell in sorted(multigen_cells)}
    max_mg_dev = max(mg_detail.values()) if mg_detail else 0.0

    # hh_size.3/.4 carry (they MOVE via the delta-1 linked-child-count
    # reduction; the registration expects them to still clear).
    hh_carry = {}
    for cell in HH_SIZE_CARRY_CELLS:
        hh_carry[cell] = {
            "max_abs_score_deviation_vs_candidate6": _max_dev([cell]),
            "candidate7_pass_all_seeds": all(
                c7_by_seed[seed]["gated_cells"][cell]["pass"]
                for seed in (s["seed"] for s in per_seed)
            ),
            "per_seed_pass": {
                seed: c7_by_seed[seed]["gated_cells"][cell]["pass"]
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
            "families": list(CANDIDATE6_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-6 cleared families (incl. coresident_grandchild "
                "at 1.00) stay cleared: the strictly-carried families are byte-"
                "identical, the grandchild 55+ coupled cell is independent of "
                "the linked count, and the 45-54 composed cell has wide "
                "tolerance."
            ),
        },
        "byte_identical_carried_family_score_check": {
            "strict_carried_cells": sorted(strict_carried),
            "max_abs_score_deviation_vs_candidate6": max_carry_dev,
            "byte_identical": bool(max_carry_dev <= EXACT_ATOL),
            "coresident_grandchild_55plus_female_max_abs_score_deviation": (
                gc_dev
            ),
            "coresident_grandchild_55plus_byte_identical": bool(
                gc_dev <= EXACT_ATOL
            ),
            "note": (
                "Every strictly-carried cell's per-seed gated score equals "
                "candidate 6's to bit precision (parent / spouse ALL bands / "
                "multigen / parental-home / multigen transitions), because "
                "candidate 7 changes ONLY the linked child coresidence draw on "
                "the isolated 0xC7 stream. coresident_grandchild.55+|female is "
                "byte-identical (the coupling is independent of the linked "
                "count)."
            ),
        },
        "fragile_spouse_cell_carried_untouched": {
            "cell": FRAGILE_SPOUSE_CELL,
            "max_abs_score_deviation_vs_candidate6": fragile_dev,
            "byte_identical": bool(fragile_dev <= EXACT_ATOL),
            "per_seed": fragile_per_seed,
            "forensics4_stability_verdict": (
                "fragile_marginal_inherited_not_c6_interaction"
            ),
            "forensics4_n_split_seeds_over_tolerance": 2,
            "note": (
                "The fragile coresident_spouse.25-34|female cell is carried "
                "UNTOUCHED (byte-identical to candidate 6); its 2/5 split-seed "
                "fragility is on the record and priced into the forecast, not "
                "banked as solved."
            ),
        },
        "multigen_marginal_unchanged_check": {
            "multigen_cells": sorted(multigen_cells),
            "max_abs_score_deviation_vs_candidate6": max_mg_dev,
            "per_cell_max_abs_score_deviation": mg_detail,
            "marginal_unchanged": bool(max_mg_dev <= EXACT_ATOL),
            "note": (
                "multigen is carried from candidate 1's simulate_draw and the "
                "coupling only reads it (applied ONLY at 55+), so every multigen "
                "stock and transition cell is byte-identical to candidate 6."
            ),
        },
        "hh_size_carry_check": {
            "cells": list(HH_SIZE_CARRY_CELLS),
            "detail": hh_carry,
            "note": (
                "hh_size.3/.4 MOVE under the delta-1 linked-child-count "
                "reduction (they are downstream of the child count, not byte-"
                "identical); the registration expects them to still clear. The "
                "deviation vs candidate 6 and the pass status are recorded."
            ),
        },
    }


def carried_blocker_analysis(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Classify each seed's failing cells as delta-target or carried-blocker.

    A CARRIED cell (byte-identical to candidate 6) that fails caps the seed
    regardless of the deltas. The fragile coresident_spouse.25-34|female is
    carried UNTOUCHED, so it is a carried blocker on the seeds where it exceeds
    tolerance (the honest, falsifiable record of the priced fragility).
    """
    strict_carried_families = {
        "coresident_parent",
        "coresident_spouse",
        "multigen_stock",
        "multigen_transition",
        "parental_home_exit",
    }
    per_seed_out: dict[int, Any] = {}
    for s in per_seed:
        fails = [c for c in sorted(tol) if not s["gated_cells"][c]["pass"]]
        carried_blockers = []
        delta_targets = []
        for c in fails:
            fam = _family_of(c)
            is_carried = fam in strict_carried_families
            (carried_blockers if is_carried else delta_targets).append(c)
        per_seed_out[s["seed"]] = {
            "n_fail": len(fails),
            "carried_blockers": carried_blockers,
            "delta_target_or_downstream_fails": delta_targets,
            "seed_capped_by_carried_cell": bool(carried_blockers),
            "fragile_spouse_cell_caps_seed": (
                FRAGILE_SPOUSE_CELL in carried_blockers
            ),
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
            "A carried (byte-identical to candidate 6) cell that fails caps the "
            "seed regardless of the two candidate-7 deltas. The fragile "
            "coresident_spouse.25-34|female is carried UNTOUCHED; where it "
            "exceeds tolerance it is recorded as a carried blocker."
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
        if rec.get("c7_delta_checks") is not None:
            fit_checks = rec["c7_delta_checks"]
        rec.pop("c7_delta_checks", None)
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
                f"(c6 {c['candidate6_pass_rate']}, d_c6c7 "
                f"{c['delta_c6_to_c7']}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} tol={d['worst_cell_tolerance']}"
            )
        byt = comparison["byte_identical_carried_family_score_check"]
        mg = comparison["multigen_marginal_unchanged_check"]
        fr = comparison["fragile_spouse_cell_carried_untouched"]
        print(
            "  strict-carried byte-identical vs c6="
            f"{byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate6']:.2e}); "
            f"grandchild55+ byte-identical="
            f"{byt['coresident_grandchild_55plus_byte_identical']}; "
            f"multigen-marginal-unchanged={mg['marginal_unchanged']}; "
            f"fragile-spouse byte-identical={fr['byte_identical']}"
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
        "candidate": "candidate 7",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate6_registration_pointer": CANDIDATE6_REGISTRATION_POINTER,
        "candidate6_grading_pointer": CANDIDATE6_GRADING_POINTER,
        "grading_pointer": GRADING_POINTER,
        "forensics_registration_pointer": FORENSICS_REGISTRATION_POINTER,
        "forensics_artifact": "runs/gate2b_forensics4_v1.json",
        "one_shot": (
            "Registered on issue #42 comment 4948186843 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_6": [
            "delta 1: enumeration conditioning -- restrict the paternal-linked "
            "coresidence draw to ENUMERATED children (the joinable "
            "(parent, birth_year) keys in father_links_child); the non-joinable "
            "biological children (25.8% of linked exposure, 9,500 of 36,887) "
            "cannot be drawn coresident; deterministic filter, non-joinable "
            "share recorded per band",
            "delta 2: episode persistence -- replace the independent per-wave "
            "occupancy with a correlated entry/persist/exit process (a per-"
            "father latent frailty a fitted fraction rho of children follow, "
            "sibling-synchronized) on the isolated 0xC7 stream, fitted to the "
            "train episode-length mean (~5.93 waves), CONSTRAINED to preserve "
            "the per-wave custodial marginal by band exactly",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 6's generator REUSED byte-faithfully (which reuses "
                "candidates 5, 4, 3, 2 and 1, the multigen--adult-child "
                "coupling at 55+, the female cohabitation refit and the count-"
                "conditional bridge), plus two train-fitted deltas on the "
                "linked child coresidence draw: enumeration conditioning "
                "(delta 1) and episode persistence (delta 2), on the isolated "
                "0xC7 stream."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate6_module": (
                "populace_dynamics.models.household_composition_sim_v6"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v7"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "grandchild_coupling_age_lo": hcs7.GRANDCHILD_LO,
            "core_size_cap": hcs7.CORE_SIZE_CAP,
            "custodial_revert_band": list(hcs7.CUSTODIAL_REVERT_BAND),
            "custodial_child_age_bands": [
                list(b) for b in hcs3.CUSTODIAL_CHILD_AGE_BANDS
            ],
            "spell_child_max_age": hcs7.SPELL_CHILD_MAX_AGE,
            "delta_stream_tag_v6": hcs7.DELTA_STREAM_TAG_V6,
            "delta_stream_tag_v7": hcs7.DELTA_STREAM_TAG_V7,
            "components": [
                "coresident_spouse<-CARRIED candidate 6 (byte-identical, incl. "
                "the fragile 25-34|female cell)",
                "coresident_parent<-CARRIED candidate 1 logistic exit hazard",
                "multigen<-CARRIED candidate 1 (coupling reads, never changes)",
                "coresident_child<-ENUMERATION-CONDITIONED linked draw (delta 1) "
                "+ episode-persistent correlated coresidence (delta 2, 0xC7) + "
                "maternal own-birth and shadow children CARRIED from candidate 6",
                "coresident_grandchild<-composed(multigen&child&~parent) with "
                "the CARRIED 55+ coupling (byte-identical) | skipgen",
                "hh_size<-composed(1+spouse+children+parent_count) + the CARRIED "
                "delta-4 count-conditional bridge; moves with the linked count",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from the candidate-6 streams UNCHANGED "
                "(0xB2B occupancy; 0xC2 cohabitation/child; 0xC3 non-family/"
                "skipgen; 0xC4 legal-residual; 0xC5 coupling/parent-count; 0xC6 "
                "maternal leave); the candidate-6 0xC3 custodial spawn is "
                "retired (its non-family and skip-gen sibling spawns preserved "
                "so those stay byte-identical); the two candidate-7 deltas draw "
                "the linked episode coresidence from a separate "
                "SeedSequence([5200+k, 0xC7])"
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
        "c7_delta_checks": {
            "computed_on_seed": FIT_VS_RAW_SEED,
            "note": (
                "The two deltas' fit-vs-raw checks vs the forensics-4 measured "
                "channels (fit seed's train side B): the enumeration split "
                "(delta 1, the unenumerated_nonjoinable channel), the episode-"
                "length fit target 5.93 (delta 2, the spell channel), the "
                "channel-removal arithmetic (raw candidate-6 cell miss minus "
                "the two removed channels), and the shadow named residual."
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
            "candidate6_artifact_sha256": _sha_of_file(CANDIDATE_ARTIFACTS[6]),
            "forensics4_artifact_sha256": _sha_of_file(FORENSICS_RUN),
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
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    artifact = run(
        verbose=True,
        seeds=seeds,
        n_draws=args.draws,
        artifact_path=Path(args.out),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    artifacts.write_new(Path(args.out), _json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
