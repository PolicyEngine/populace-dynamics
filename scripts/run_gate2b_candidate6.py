"""Gate-2b candidate 6: the four measured levers.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b; 46 gated household/relationship cells, K=20 mean-over-draws,
per-seed train/holdout protocol). Registered on issue #42 comment 4946285556
BEFORE this run; published REGARDLESS of verdict.

Candidate 6 is candidate 5 (registration 4945159933; PR #141) with EXACTLY FOUR
frozen deltas, each designed against a graded gate-2b forensics-3 finding
(``runs/gate2b_forensics3_v1.json``, grading 4946281888). Everything candidate 5
cleared or carried -- the certified tranche-2a marital core, the carried
``coresident_parent`` / ``multigen`` (stock + transitions) /
``parental_home_exit``, the delta-1 multigen--adult-child coupling at 55+ (which
took ``coresident_grandchild`` to 1.00), and the delta-3b per-ego parent-count
composition -- is carried BYTE-FAITHFULLY (candidate 6 REUSES the candidate-5
generator and re-runs its exact 0xB2B / 0xC2 / 0xC3 / 0xC4 / 0xC5 streams).

Delta 1 -- 0-4 basis revert (``coresident_child`` 15-24|male): revert the
not-married custodial swap to the OBSERVABLE basis at child ages 0-4 (keep the
child-record swap at 5-17); the child-record 0-4 gap is a join-denominator
artifact.
Delta 2 -- adult-child exit timing (``coresident_child`` 35-44|male, 45-54|
female): re-fit the single-year child-age home-exit hazard over 18-30 and apply
it to the maternal own-birth leave (0xC6) AND to a new linked-married adult-child
leave (0xC6); the multigen coupling stays at 55+ EXACTLY (no downward extension).
Delta 3 -- female cohabitation lift at 25-34 (``coresident_spouse`` 25-34|
female): re-fit the FEMALE single-year cohabitation entry/exit over 25-44 (the
overlay shortfall); the legal top-up is NOT applied.
Delta 4 -- count-conditional bridge (``hh_size`` 3/4/5+): draw the FULL non-core
count from the train joint P(count | capped core) -- the forensics-3-proven
parameterization (0.1887/0.1709/0.1303).

Estimator, artifact schema, undefined-draw rule, and protocol are candidate 5's
(locked contract). Artifact ``runs/gate2b_hazard_v6.json``; ``artifacts.
write_new`` refuses to overwrite. Reproduce with ``.venv-gate`` and the PSID
products staged (``POPULACE_DYNAMICS_PSID_DIR``).
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
from populace_dynamics.models import household_composition_sim_v6 as hcs6

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v6.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v6"
RUN_NAME = "gate2b_hazard_v6"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS_RUN = ROOT / "runs" / "gate2b_forensics3_v1.json"
CANDIDATE_ARTIFACTS = {
    1: ROOT / "runs" / "gate2b_hazard_v1.json",
    2: ROOT / "runs" / "gate2b_hazard_v2.json",
    3: ROOT / "runs" / "gate2b_hazard_v3.json",
    4: ROOT / "runs" / "gate2b_hazard_v4.json",
    5: ROOT / "runs" / "gate2b_hazard_v5.json",
}
REGISTRATION_POINTER = "4946285556"
CANDIDATE5_REGISTRATION_POINTER = "4945159933"
CANDIDATE4_REGISTRATION_POINTER = "4941160621"
CANDIDATE3_REGISTRATION_POINTER = "4939960467"
CANDIDATE2_REGISTRATION_POINTER = "4939456379"
CANDIDATE1_REGISTRATION_POINTER = "4938726107"
CANDIDATE5_GRADING_POINTER = "4945697846"
GRADING_POINTER = "4946281888"  # forensics-3 grading (c6 designs against it)
FORENSICS_REGISTRATION_POINTER = "4945702151"
SPEC_REGISTRATION = (
    "issue #42 comment 4946285556: gate-2b candidate 6, the four measured "
    "levers (0-4 basis revert; adult-child exit timing both parent sides; "
    "female cohabitation lift at 25-34; count-conditional bridge)"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12
FIT_VS_RAW_SEED = 0

#: Families carried BYTE-IDENTICAL to candidate 5 (no candidate-6 delta touches
#: their streams: they come off 0xB2B / 0xC5 before any delta and the deltas
#: draw from streams that do not perturb them).
CANDIDATE5_STRICT_CARRIED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)
#: Candidate-5 families whose pass rate was 1.0 and must STAY cleared under the
#: byte-faithful carry (the regression guard the registration names, incl.
#: coresident_grandchild at 1.00 -- the 55+ coupled cell is byte-identical and
#: the 45-54 composed cell has wide tolerance).
CANDIDATE5_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
    "coresident_grandchild",
)
MULTIGEN_MARGINAL_CELLS_PREFIX = ("multigen.",)
MULTIGEN_MARGINAL_CELLS_EXACT = ("multigen_entry", "multigen_exit")

#: Per-delta target family (each delta feeds a distinct failure surface).
PER_DELTA_TARGET_FAMILY = {
    "delta_1_zero_four_revert": "coresident_child",
    "delta_2_adult_child_exit_timing": "coresident_child",
    "delta_3_female_cohab_lift": "coresident_spouse",
    "delta_4_count_conditional_bridge": "hh_size",
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.55-0.70",
    "majority_side": True,
    "named_expectations": [
        "hh_size.3/.4/.5+ all clear (the forensics-3-proven count-conditional "
        "parameterization)",
        "coresident_child.15-24|male clears (the 0-4 artifact removed at its "
        "cause)",
        "coresident_spouse.25-34|female clears or lands within 1.1x (overlay "
        "refit on the measured shortfall)",
        "coresident_child.35-44|male is the modal residual (the supply/timing "
        "mechanism is the least precisely measured of the four)",
        "coresident_child.45-54|female clears or near-clears via the same exit "
        "timing",
        "carries byte-identical incl. grandchild 1.00 and the multigen "
        "marginal",
    ],
    "modal_outcome_if_fail": (
        "coresident_child.35-44|male alone -- one cell from the first 2b pass"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT grade "
        "it."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "rng_byte_identical_carried_families": (
        "Candidate 6 re-runs candidate 5's exact 0xB2B / 0xC2 / 0xC3 / 0xC4 / "
        "0xC5 streams. Delta 1 (0-4 revert) and delta 4 (count-conditional "
        "bridge) re-key probability tables whose per-wave draw consumes RNG BY "
        "SHAPE ONLY, so the 0xC3 custodial / non-family / skip-gen streams stay "
        "byte-identical. Delta 3 (female cohab override) consumes the 0xC2 "
        "cohabitation draw by shape only. The two genuinely new components -- "
        "the maternal single-year leave-year refit and the linked-married "
        "adult-child leave (delta 2) -- draw from a SEPARATE "
        "SeedSequence([5200+k, 0xC6]).spawn(2). The carried coresident_parent, "
        "multigen (stock + transitions) and parental_home_exit families come "
        "off candidate 1's simulate_draw UNCHANGED, so they are bit-identical "
        "to candidate 5 on every draw and seed (regression is impossible by "
        "construction)."
    ),
    "delta_1_zero_four_revert": (
        "Forensics-3 Q8 adjudicated the candidate-5 not-married 0-4 child-record "
        "swap a mistake: the child-record basis runs HIGHER than observable at "
        "0-4 (+0.048) but LOWER at school ages -- a sign inversion marking the "
        "0-4 gap as a join-denominator ARTIFACT (selective under-enumeration of "
        "young children living away from the father). The reference gate cell "
        "is ego-anchored, so the observable (father-wave) basis is the matched "
        "concept at 0-4. RESOLUTION: candidate 6 reverts the not-married 0-4 "
        "custodial cell to candidate 4's observable lookup (the exposure-"
        "weighted observable there ~0.710 is BELOW the child-record ~0.742, so "
        "the revert drains the young-father over-production) and KEEPS the "
        "child-record swap at 5-17|not-married. Byte-identical 0xC3 custodial "
        "draw shape; only the not-married 0-4 probability moves. 18-24 and "
        "25-60|not-married keep the candidate-5 child-record basis (the "
        "registration scopes the revert to 0-4 only)."
    ),
    "delta_2_adult_child_exit_timing": (
        "Forensics-3 Q8 localized 35-44|male to MARRIED linked-father coresidence "
        "supply / leave timing (the married custodial basis is faithful, so the "
        "lever is NOT the custodial probability) and 45-54|female to maternal "
        "adult-child aging-out timing (entirely maternal, over-produced, and the "
        "reference coupling there is weak -- lift x1.75 vs ~5x at 55+ -- so the "
        "multigen coupling must NOT extend downward). RESOLUTION: candidate 6 "
        "re-fits the single-year child-age home-exit hazard over 18-30 on train "
        "(weighted exit among coresident-parent at-risk waves by ego own age x "
        "sex) and applies the exit timing to both parent sides: (a) MATERNAL -- "
        "the maternal own-birth leave-year OVERRIDES candidate 1's spline at "
        "18-30 (the empirical exceeds the spline for sons), on the isolated 0xC6 "
        "stream; the SHADOW leave-year is kept byte-identical to candidate 5. "
        "(b) LINKED-MARRIED -- candidate 4's single-year OBSERVABLE married "
        "custodial probability ALREADY declines from ~0.85 at child age 18 to "
        "~0.14 at age 30: it IS the coresident-adult-child home-exit timing, "
        "applied UNCHANGED. DOCUMENTED RESOLUTION OF THE 'both parent sides' "
        "AMBIGUITY: a hard linked-married leave ON TOP of that declining prob "
        "would DOUBLE-COUNT the aging-out and over-drain the older married-linked "
        "male cells (55-64|male, 65-74|male), contradicting the registration's "
        "own 'modal failure: 35-44|male alone'; the 35-44|male over-production "
        "is therefore the linked-father child SUPPLY, deferred to candidate 7 as "
        "the registration anticipated ('if the single-year exit refit does not "
        "drain it, c7 needs a forensics-4 look at linked-father child supply'). "
        "So the linked-married custodial coresidence is byte-identical to "
        "candidate 5 except the delta-1 0-4 not-married revert. The multigen "
        "coupling stays at 55+ EXACTLY (extending it to 45-54 is a spec "
        "violation)."
    ),
    "delta_3_female_cohab_lift": (
        "Forensics-3 Q9 reclassified the 25-34|female miss as a cohabitation-"
        "OVERLAY shortfall (overlay gap -0.045), NOT the under-produced-legal "
        "mechanism of the male bands (the legal core is +0.011 OVER). "
        "RESOLUTION: candidate 6 re-fits the FEMALE single-year cohabitation "
        "entry/exit over ages 25-44 (the same single-year estimator candidate "
        "4's delta 1 used for 15-34) and applies it as a female override; the "
        "candidate-4 legal top-up is explicitly NOT applied (it would push the "
        "already-adequate legal core further over). DOCUMENTED AMBIGUITY: the "
        "faithful single-year female refit COINCIDES with candidate 4's train "
        "estimate at 25-34 (same estimator, same train at-risk waves), so "
        "coresident_spouse.25-34|female is ESSENTIALLY UNCHANGED from candidate "
        "5 (a negligible ~2.6e-7 cohab-persistence propagation from the 35-44 "
        "refit on 1 of 20 draws never flips its pass/fail); the new structure "
        "is the 35-44 single-year refit (candidate 4 fit single-year only to "
        "34). Re-estimating the same hazards offers no free lift at 25-34 -- "
        "the registration's 'lift' is the named expectation, not a calibration; "
        "the mechanism is the refit, implemented literally, and the 25-34|"
        "female outcome is whatever the unchanged overlay produces (within 1.1x "
        "on every candidate-5 seed)."
    ),
    "delta_4_count_conditional_bridge": (
        "Forensics-3 Q10 PROVED the binding candidate-5 defect is the BRIDGE "
        "COUNT parameterization: candidate 5 conditioned the non-core INCIDENCE "
        "on core size (delta 3a) but drew the 2+ non-core COUNT from the (band, "
        "sex) table INDEPENDENTLY of core size, over-producing hh_size.3 and "
        "starving hh_size.5+. RESOLUTION: candidate 6 draws the FULL non-core "
        "member count from the train joint P(non-core count = j | capped core "
        "size = k); convolving that conditional with the sim's own core "
        "distribution clears hh3/hh4/hh5+ simultaneously (~0.1887/0.1709/"
        "0.1303, reproduced in the fit-vs-raw check). The count draw consumes "
        "ONE uniform per ego on the 0xC3 non-family stream (byte-identical in "
        "shape to candidate 5's 0/1/2+ class draw); the candidate-5 0xC4 2+ "
        "count stream is retired (its legal-residual sibling spawn is preserved "
        "so the legal-residual overlay stays byte-identical). The delta-3b per-"
        "ego parent-count composition is CARRIED (it feeds the core the bridge "
        "conditions on). The secondary core-5+ fertility deficit is named, "
        "carried, and NOT chased (the honest joint absorbs it)."
    ),
    "carried_families_byte_faithful": (
        "The candidate-5 strictly-carried families -- coresident_parent, "
        "multigen stock, the multigen entry/exit transitions, and "
        "parental_home_exit -- come off the carried streams UNCHANGED before "
        "any candidate-6 delta, so their per-seed scores are IDENTICAL to "
        "candidate 5 to bit precision. coresident_spouse is carried EXCEPT the "
        "delta-3 female 25-44 override (males and females outside 25-44 are "
        "byte-identical); coresident_child female cells are NO LONGER byte-"
        "identical (delta 2's maternal leave touches them); coresident_"
        "grandchild.55+|female is byte-identical (the coupling is independent "
        "of the maternal child leaves)."
    ),
    "multigen_marginal_unchanged": (
        "The delta-1-of-candidate-5 coupling constraint is carried: the "
        "multigen occupancy marginal is unchanged. Every multigen stock cell "
        "and the multigen_entry / multigen_exit transition cells have per-seed "
        "gated scores byte-identical to candidate 5 (max deviation 0.0), "
        "because multigen is carried from candidate 1's simulate_draw and the "
        "coupling only reads it and is applied ONLY at 55+ (never extended to "
        "45-54)."
    ),
    "no_coupling_extension_below_55": (
        "Forensics-3 Q8 verified the reference multigen--adult-child coupling "
        "at 45-54 is weak (joint 0.0481 vs product 0.0275, lift x1.75, far "
        "below the ~5x at 55+) and the 45-54|female cell OVER-produces, so "
        "extending the coupling downward would worsen the overshoot. Candidate "
        "6 pins GRANDCHILD_LO = 55 and applies the coupling ONLY at is_55_row; "
        "the recorded invariant max_p_coupled_below_55 = 0.0 confirms it."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "VERBATIM from candidate 5: each simulated side-A holdout person is "
        "seeded from their OWN observed state at their first 2b wave, then "
        "evolved with train-fitted hazards; no parameter is estimated from side "
        "A. The candidate-6 leave draws are conditional survival devices on the "
        "isolated 0xC6 stream, not observed states."
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
def _delta_stats(
    model: hcs6.HouseholdCompositionModelV6, dmean, dmean_dict
) -> dict:
    exit_grid = model.meta["child_exit_single_year"]["grid_ages_biennial"]
    exit_tab = model.meta["child_exit_single_year"][
        "single_year_hazard_vs_spline"
    ]
    return {
        "delta_1_zero_four_revert": {
            "revert_band": model.meta["custodial_revert_band"],
            "fitted_child_record_0_4_not_married": model.custodial_child_record[
                (hc.band_label(0, 4), "not_married")
            ],
            "not_married_child_record_by_band": model.meta[
                "custodial_child_record"
            ]["not_married_child_record_by_band"],
            "n_linked_child_coresident_wave_units": dmean(
                "n_linked_child_coresident_wave_units"
            ),
        },
        "delta_2_adult_child_exit_timing": {
            "refit_range": model.meta["child_exit_refit_range"],
            "grid_ages": exit_grid,
            "single_year_hazard_vs_spline_male": {
                str(a): exit_tab[f"{a}|male"] for a in exit_grid
            },
            "single_year_hazard_vs_spline_female": {
                str(a): exit_tab[f"{a}|female"] for a in exit_grid
            },
            "n_maternal_child_coresident_wave_units": dmean(
                "n_maternal_child_coresident_wave_units"
            ),
            "n_linked_child_coresident_wave_units": dmean(
                "n_linked_child_coresident_wave_units"
            ),
            "coupling_stays_at_55_plus": hcs6.GRANDCHILD_LO == 55,
            "max_p_coupled_below_55": dmean("no_coupling_below_55_max_p"),
        },
        "delta_3_female_cohab_lift": {
            "refit_range": model.meta["cohab_female_refit_range"],
            "fitted_female_entry_25_34": {
                str(a): round(model.cohab_entry_age_female[a], 5)
                for a in range(25, 35)
            },
            "fitted_female_entry_35_44": {
                str(a): round(model.cohab_entry_age_female[a], 5)
                for a in range(35, 45)
            },
            "n_cohab_person_waves_simulated": dmean(
                "n_cohab_person_waves_simulated"
            ),
        },
        "delta_4_count_conditional_bridge": {
            "fitted_noncore_incidence_by_core": model.meta[
                "nonfamily_count_by_core"
            ]["noncore_incidence_by_capped_core"],
            "fitted_mean_noncore_count_by_core": model.meta[
                "nonfamily_count_by_core"
            ]["mean_noncore_count_by_capped_core"],
            "mean_nonfamily_count_simulated": dmean(
                "mean_nonfamily_count_simulated"
            ),
            "sim_noncore_incidence_by_core": dmean_dict(
                "noncore_incidence_by_core"
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

    model = hcs6.fit_household_model_v6(
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
        sim_panel, diag = hcs6.simulate_draw_v6(
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

    def _dmean_dict(key: str) -> dict[str, float]:
        keys = draw_diagnostics[0][key].keys()
        return {
            k: float(np.mean([d[key][k] for d in draw_diagnostics]))
            for k in keys
        }

    delta_stats = _delta_stats(model, _dmean, _dmean_dict)
    coverage = {
        **coverage,
        "mean_paternal_linked_births": _dmean("n_paternal_linked_births"),
        "mean_paternal_shadow_births": _dmean("n_paternal_shadow_births"),
        "mean_maternal_births": _dmean("n_maternal_births"),
    }
    fit_checks = None
    if compute_fit_checks:
        fit_checks = hcs6.c6_delta_checks(model, forensics)

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
                "custodial_revert_band",
                "child_exit_refit_range",
                "cohab_female_refit_range",
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
        "c6_delta_checks": fit_checks,
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
        "overlay (with the delta-3 female 25-44 single-year refit) UNION the "
        "legal-spouse residual overlay."
    ),
    "coresident_parent": (
        "directly fitted logistic exit hazard (candidate 1, byte-faithful, "
        "RNG-isolated); expected to clear."
    ),
    "coresident_child": (
        "observed father->child links gated per wave by the custodial "
        "probability (delta-1 0-4 not-married revert), linked-married children "
        "aged out by the delta-2 leave; maternal own-birth children aged out by "
        "the delta-2 single-year 18-30 exit refit; shadow byte-faithful."
    ),
    "coresident_grandchild": (
        "composed implication (multigen AND child AND NOT parent) with the "
        "delta-1-of-candidate-5 coupling at 55+ (carried) UNION the carried "
        "5-year skip-gen occupancy; the coupling is NOT extended to 45-54."
    ),
    "multigen_stock": (
        "carried initial state + train band x sex entry/exit (candidate 1, "
        "byte-faithful, RNG-isolated)."
    ),
    "multigen_transition": (
        "directly fitted pooled entry/exit rates (candidate 1, byte-faithful, "
        "RNG-isolated)."
    ),
    "parental_home_exit": (
        "directly fitted (candidate 1, byte-faithful, RNG-isolated); expected "
        "to clear."
    ),
    "hh_size": (
        "composed ego-centric family unit (parent count drawn 1 vs 2, carried "
        "delta-3b) PLUS the delta-4 count-conditional non-family bridge (the "
        "full non-core count drawn from the train joint P(count | core size))."
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
        # a report-only cell's score may be non-finite (rate_a or rbar == 0);
        # the artifact / per-seed cache stores that as None (json can't hold
        # inf), so guard both representations.
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
# Candidate 1 -> ... -> 6 progression + regression + byte carry + checks
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
    for fam, d6 in decomposition.items():
        r5 = decomp[5].get(fam, {}).get("cell_seed_pass_rate")
        per_family[fam] = {
            f"candidate{n}_pass_rate": decomp[n]
            .get(fam, {})
            .get("cell_seed_pass_rate")
            for n in cand
        }
        per_family[fam]["candidate6_pass_rate"] = d6["cell_seed_pass_rate"]
        per_family[fam]["delta_c5_to_c6"] = (
            round(d6["cell_seed_pass_rate"] - r5, 4)
            if r5 is not None
            else None
        )
        per_family[fam]["candidate5_worst_cell"] = (
            decomp[5].get(fam, {}).get("worst_cell")
        )
        per_family[fam]["candidate6_worst_cell"] = d6["worst_cell"]
        per_family[fam]["candidate6_worst_mean_abs_ln"] = d6[
            "worst_cell_mean_abs_ln"
        ]

    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE5_CLEARED_FAMILIES:
        r5 = decomp[5].get(fam, {}).get("cell_seed_pass_rate")
        r6 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r6 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate5_pass_rate": r5,
            "candidate6_pass_rate": r6,
            "still_clears": bool(holds),
        }

    # Byte-identical carried-family score check vs candidate 5.
    c5_by_seed = {s["seed"]: s for s in cand[5]["per_seed"]}
    c6_by_seed = {s["seed"]: s for s in per_seed}
    strict_carried = [
        c
        for c in tol
        if c.startswith(("coresident_parent.", "multigen.", "parental_home_"))
        or c in ("multigen_entry", "multigen_exit")
    ]
    max_carry_dev = 0.0
    for seed in (s["seed"] for s in per_seed):
        for cell in strict_carried:
            s6 = c6_by_seed[seed]["gated_cells"][cell]["score"]
            s5 = c5_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s6) and math.isfinite(s5):
                max_carry_dev = max(max_carry_dev, abs(s6 - s5))

    # coresident_spouse byte-carry EXCEPT the delta-3 female 25-44 cells.
    delta3_cells = {
        "coresident_spouse.25-34|female",
        "coresident_spouse.35-44|female",
    }
    spouse_carry_dev = 0.0
    spouse_moved: dict[str, float] = {}
    for cell in [c for c in tol if c.startswith("coresident_spouse.")]:
        cell_dev = 0.0
        for seed in (s["seed"] for s in per_seed):
            s6 = c6_by_seed[seed]["gated_cells"][cell]["score"]
            s5 = c5_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s6) and math.isfinite(s5):
                cell_dev = max(cell_dev, abs(s6 - s5))
        if cell in delta3_cells or "|female" in cell:
            spouse_moved[cell] = round(cell_dev, 6)
        else:
            spouse_carry_dev = max(spouse_carry_dev, cell_dev)

    # grandchild 55+ byte-identity (coupling independent of the deltas).
    gc_cell = "coresident_grandchild.55+|female"
    gc_dev = 0.0
    for seed in (s["seed"] for s in per_seed):
        s6 = c6_by_seed[seed]["gated_cells"][gc_cell]["score"]
        s5 = c5_by_seed[seed]["gated_cells"][gc_cell]["score"]
        if math.isfinite(s6) and math.isfinite(s5):
            gc_dev = max(gc_dev, abs(s6 - s5))

    # Multigen-marginal-unchanged check.
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
            s6 = c6_by_seed[seed]["gated_cells"][cell]["score"]
            s5 = c5_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s6) and math.isfinite(s5):
                cell_max = max(cell_max, abs(s6 - s5))
        mg_detail[cell] = cell_max
        max_mg_dev = max(max_mg_dev, cell_max)

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
            "families": list(CANDIDATE5_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-5 cleared families (incl. coresident_grandchild "
                "at 1.00) stay cleared: the strictly-carried families are byte-"
                "identical, and the grandchild 55+ coupled cell is independent "
                "of the deltas while the 45-54 composed cell has wide "
                "tolerance."
            ),
        },
        "byte_identical_carried_family_score_check": {
            "strict_carried_cells": sorted(strict_carried),
            "max_abs_score_deviation_vs_candidate5": max_carry_dev,
            "byte_identical": bool(max_carry_dev <= EXACT_ATOL),
            "coresident_spouse_carry_excluding_female_25_44": {
                "max_abs_score_deviation": spouse_carry_dev,
                "byte_identical": bool(spouse_carry_dev <= EXACT_ATOL),
                "delta3_and_female_cells_moved": spouse_moved,
            },
            "coresident_grandchild_55plus_female_max_abs_score_deviation": (
                gc_dev
            ),
            "coresident_grandchild_55plus_byte_identical": bool(
                gc_dev <= EXACT_ATOL
            ),
            "note": (
                "Every strictly-carried cell's per-seed gated score equals "
                "candidate 5's to bit precision (parent / multigen / parental-"
                "home / multigen transitions). coresident_spouse is byte-"
                "identical except the delta-3 female 25-44 override (and its "
                "female-band cohab persistence); coresident_grandchild.55+|"
                "female is byte-identical (the coupling is independent of the "
                "maternal child leaves)."
            ),
        },
        "multigen_marginal_unchanged_check": {
            "multigen_cells": sorted(multigen_cells),
            "max_abs_score_deviation_vs_candidate5": max_mg_dev,
            "per_cell_max_abs_score_deviation": mg_detail,
            "marginal_unchanged": bool(max_mg_dev <= EXACT_ATOL),
            "note": (
                "The delta-1-of-candidate-5 coupling constraint is carried: the "
                "coupling reads the multigen state (applied ONLY at 55+) but "
                "never changes how many egos are multigen, so every multigen "
                "stock and transition cell is byte-identical to candidate 5."
            ),
        },
    }


def carried_blocker_analysis(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Classify each seed's failing cells as delta-target or carried-blocker.

    A CARRIED cell (byte-identical to candidate 5) that fails caps the seed
    regardless of the deltas. In candidate 6 the delta-3 female cohab refit
    coincides with candidate 5 at coresident_spouse.25-34|female, so that cell
    is byte-identical -- a carried blocker on the seeds where candidate 5 failed
    it (the honest, falsifiable record of the delta's inert-at-target behavior).
    """
    strict_carried_families = {
        "coresident_parent",
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
            is_carried = fam in strict_carried_families or (
                c == "coresident_spouse.25-34|female"
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
            "coresident_spouse.25-34|female is essentially unchanged from "
            "candidate 5 (the delta-3 female single-year refit coincides with "
            "candidate 4's train estimate there; a negligible ~2.6e-7 cohab-"
            "persistence propagation from the 35-44 refit never flips its "
            "pass/fail); it caps the seeds where candidate 5 failed it. The "
            "honest record of a delta that is faithful to its registered "
            "mechanism but inert at its named target -- the 'measured deltas "
            "can be faithful while the mechanism list stays incomplete' "
            "pattern, not a regression."
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
        if rec.get("c6_delta_checks") is not None:
            fit_checks = rec["c6_delta_checks"]
        rec.pop("c6_delta_checks", None)
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
                f"(c5 {c['candidate5_pass_rate']}, d_c5c6 "
                f"{c['delta_c5_to_c6']}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} tol={d['worst_cell_tolerance']}"
            )
        byt = comparison["byte_identical_carried_family_score_check"]
        mg = comparison["multigen_marginal_unchanged_check"]
        print(
            "  strict-carried byte-identical vs c5="
            f"{byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate5']:.2e}); "
            f"grandchild55+ byte-identical="
            f"{byt['coresident_grandchild_55plus_byte_identical']}; "
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
        "candidate": "candidate 6",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate5_registration_pointer": CANDIDATE5_REGISTRATION_POINTER,
        "candidate5_grading_pointer": CANDIDATE5_GRADING_POINTER,
        "grading_pointer": GRADING_POINTER,
        "forensics_registration_pointer": FORENSICS_REGISTRATION_POINTER,
        "forensics_artifact": "runs/gate2b_forensics3_v1.json",
        "one_shot": (
            "Registered on issue #42 comment 4946285556 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_5": [
            "delta 1: 0-4 basis revert -- revert the not-married custodial swap "
            "to the observable basis at child ages 0-4 (keep the child-record "
            "swap at 5-17); byte-identical 0xC3 custodial draw shape",
            "delta 2: adult-child exit timing -- re-fit the single-year 18-30 "
            "child-age home-exit hazard; OVERRIDE the maternal own-birth leave "
            "(isolated 0xC6); the linked-married side is candidate 4's single-"
            "year observable declining married custodial prob UNCHANGED (a hard "
            "leave would double-count and over-drain; 35-44|male is supply -> "
            "c7); the multigen coupling stays at 55+ EXACTLY",
            "delta 3: female cohabitation lift at 25-34 -- re-fit the FEMALE "
            "single-year cohabitation entry/exit over 25-44 (byte-identical "
            "0xC2 draw shape); the legal top-up is NOT applied",
            "delta 4: count-conditional bridge -- draw the FULL non-core count "
            "from the train joint P(count | capped core); byte-identical 0xC3 "
            "non-family draw shape (the 0xC4 2+ count is retired)",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 5's generator REUSED byte-faithfully (which reuses "
                "candidates 4, 3, 2 and 1, the multigen--adult-child coupling "
                "at 55+ and the per-ego parent-count composition), plus four "
                "train-fitted deltas: the 0-4 not-married custodial revert "
                "(delta 1), the single-year 18-30 adult-child exit timing on "
                "both parent sides (delta 2), the female 25-44 cohabitation "
                "refit (delta 3), and the count-conditional non-family bridge "
                "(delta 4)."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate5_module": (
                "populace_dynamics.models.household_composition_sim_v5"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v6"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "grandchild_coupling_age_lo": hcs6.GRANDCHILD_LO,
            "core_size_cap": hcs6.CORE_SIZE_CAP,
            "custodial_revert_band": list(hcs6.CUSTODIAL_REVERT_BAND),
            "child_exit_refit_range": [
                hcs6.CHILD_EXIT_REFIT_LO,
                hcs6.CHILD_EXIT_REFIT_HI,
            ],
            "cohab_female_refit_range": [
                hcs6.COHAB_FEMALE_REFIT_LO,
                hcs6.COHAB_FEMALE_REFIT_HI,
            ],
            "custodial_child_age_bands": [
                list(b) for b in hcs3.CUSTODIAL_CHILD_AGE_BANDS
            ],
            "delta_stream_tag_v5": hcs6.DELTA_STREAM_TAG_V5,
            "delta_stream_tag_v6": hcs6.DELTA_STREAM_TAG_V6,
            "components": [
                "coresident_spouse<-legal registry | cohab overlay (delta-3 "
                "female 25-44 refit) | legal residual",
                "coresident_parent<-CARRIED candidate 1 logistic exit hazard",
                "multigen<-CARRIED candidate 1 (coupling reads, never changes)",
                "coresident_child<-custodial_gated (delta-1 0-4 revert) + "
                "linked-married leave (delta 2) + maternal own-birth leave "
                "refit (delta 2) + shadow (byte-faithful)",
                "coresident_grandchild<-composed(multigen&child&~parent) with "
                "the CARRIED 55+ coupling (never extended to 45-54) | skipgen",
                "hh_size<-composed(1+spouse+children+parent_count_drawn) + "
                "count-conditional non-family bridge (delta 4)",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from the candidate-5 streams UNCHANGED "
                "(0xB2B occupancy; 0xC2 cohabitation/child; 0xC3 skipgen; 0xC4 "
                "legal-residual; 0xC5 coupling/parent-count); the candidate-3 "
                "custodial and non-family draws are re-run byte-identically in "
                "consumption (delta 1 and delta 4 change probability VALUES / "
                "the count support, not draw shape; the 0xC4 2+ count is "
                "retired); the two new candidate-6 components (the maternal "
                "single-year leave refit and the linked-married leave) draw "
                "from a separate SeedSequence([5200+k, 0xC6]).spawn(2)"
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
        "c6_delta_checks": {
            "computed_on_seed": FIT_VS_RAW_SEED,
            "note": (
                "The four deltas' fit-vs-raw checks vs the forensics-3 measured "
                "quantities (fit seed's train side B): the 0-4 basis (Q8), the "
                "single-year exit hazard vs spline both parent sides + the "
                "no-coupling-extension invariant (Q8), the female overlay gap "
                "(Q9), and the count-conditional feasibility reproduction of "
                "0.1887/0.1709/0.1303 (Q10)."
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
            "candidate5_artifact_sha256": _sha_of_file(CANDIDATE_ARTIFACTS[5]),
            "forensics3_artifact_sha256": _sha_of_file(FORENSICS_RUN),
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
