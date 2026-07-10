"""Gate-2b candidate 3: custodial conditioning + household bridge + skip-gen.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b, ratified 2026-07-10; 46 gated household/relationship cells,
K=20 mean-over-draws, per-seed train/holdout protocol). Registered on issue
#42 comment 4939960467 BEFORE this run; published REGARDLESS of verdict.

Candidate 3 is candidate 2 (registration 4939456379; PR #134) with EXACTLY
THREE frozen deltas, each feeding a DISJOINT family the candidate-2 grading
(4939958136) isolated. Everything candidate 2 cleared or carried -- the
certified tranche-2a marital core and maternal births, the parental-home exit
hazard, the multigen entry/exit machinery, ``coresident_parent``, the
cohabitation (MX8 code-22) overlay unioned into ``coresident_spouse``, the
observed cah85_23 father->child links, the retained unlinked shadow kernel,
and the household-size composition rule -- is carried BYTE-FAITHFULLY (the
candidate-3 model REUSES the candidate-2 module and re-runs candidate 2's exact
draw streams; it does not copy them), and the scoring path here is candidate
2's, unchanged.

Delta 1 -- custodial paternal conditioning. A father-linked child counts as
the man's coresident child in a wave only with the train-fitted probability
``P(linked child coresident | child age band x father marital state)``, drawn
per wave; the maternal side and the unlinked shadow kernel are untouched.
Delta 2 -- household bridge. A train-fitted non-family household-member count
(0/1/2+ by ego age band x sex) is added to the ``hh_size`` composition ONLY.
Delta 3 -- skipped-generation coresidence. Train-fitted entry/exit hazards for
the ``coresident_grandchild AND NOT multigen`` state, evolved from observed
initial states, are unioned into the composed grandchild ONLY -- never into
``multigen`` (which stays byte-identical; the regression risk is named).

Estimator, artifact schema, undefined-draw rule, and protocol are candidate
2's (locked contract). Artifact ``runs/gate2b_hazard_v3.json``;
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
from populace_dynamics.models import household_composition_sim as hcs
from populace_dynamics.models import household_composition_sim_v2 as hcs2
from populace_dynamics.models import household_composition_sim_v3 as hcs3

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v3.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v3"
RUN_NAME = "gate2b_hazard_v3"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
REGISTRATION_POINTER = "4939960467"
CANDIDATE2_REGISTRATION_POINTER = "4939456379"
CANDIDATE1_REGISTRATION_POINTER = "4938726107"
GRADING_POINTER = "4939958136"
SPEC_REGISTRATION = (
    "issue #42 comment 4939960467: gate-2b candidate 3, custodial "
    "conditioning + household bridge + skipped-generation state"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol; candidate 2).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12

#: The candidate-2 families that CLEARED (grading comment 4939958136): the
#: regression check confirms candidate 3 carries them non-interacting. These
#: are the same four candidate 1 cleared; candidate 3 reads them off candidate
#: 1's simulate_draw unchanged, so they stay byte-identical.
CANDIDATE2_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)

#: Per-delta target family (each delta feeds a disjoint family; recorded so
#: attribution stays clean and any leak is visible).
PER_DELTA_TARGET_FAMILY = {
    "delta_1_custodial_paternal_conditioning": "coresident_child",
    "delta_2_household_bridge": "hh_size",
    "delta_3_skipped_generation_coresidence": "coresident_grandchild",
}

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.40-0.55",
    "expected_effects": [
        "coresident_child male cells clear or near-clear (the overshoot is "
        "removed at its cause -- non-custodial biological fathers no longer "
        "attribute a coresident child)",
        "hh_size.3/.4 clear; hh_size.5+ improves and is the single most "
        "likely residual failure (deep-tail composition)",
        "coresident_grandchild improves materially, with 55+|female the "
        "second-likeliest residual",
        "all carried families remain byte-identical (the bridge feeds only "
        "hh_size -- a leak into any coresidence cell is a spec violation, not "
        "a finding)",
    ],
    "modal_outcome_if_fail": (
        "1-2 chronic cells (hh_size.5+ and/or grandchild 55+|female) failing "
        "across seeds, everything else clearing -- candidate 4 one targeted "
        "delta from the first 2b pass"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT "
        "grade it."
    ),
}

#: Candidate 2's resolutions carried VERBATIM (the carried components are
#: byte-faithful, so their resolutions are unchanged) plus the three-delta
#: resolutions. Adopted per the registration ("spec ambiguities resolved per
#: the locked contract text and documented").
SPEC_RESOLUTION_NOTES = {
    "rng_two_stream": (
        "VERBATIM from candidate 2 (locked contract wins): each of the K=20 "
        "draws re-simulates the whole composition keyed to draw seed 5200+k; "
        "default_rng(4200+seed) is the single-draw provenance seed and does "
        "not drive the mean-over-K estimator."
    ),
    "rng_byte_identical_carried_families": (
        "Candidate 3 adds THREE stochastic deltas (custodial gate; non-family "
        "count; skipped-generation occupancy). To carry candidate 2 (and "
        "through it candidate 1) BYTE-FAITHFULLY, simulate_draw_v3 REPRODUCES "
        "candidate 2's draw exactly -- candidate 1's simulate_draw at the same "
        "draw seed 5200+k and occupancy tag 0xB2B (byte-identical "
        "coresident_parent / multigen / legal-marriage spouse), the "
        "candidate-2 cohabitation overlay and father-link / maternal / shadow "
        "child roster on the same SeedSequence([5200+k, 0xC2]).spawn(2) "
        "streams (byte-identical cohabitation and maternal + shadow child "
        "leave-years) -- then draws the three deltas from a SEPARATE "
        "SeedSequence([5200+k, 0xC3]).spawn(3) -> [custodial, non-family, "
        "skipped-generation], isolated from the 0xB2B and 0xC2 streams. The "
        "carried coresident_parent, multigen, parental_home_exit and multigen "
        "transition families are therefore bit-identical to candidate 2 on "
        "every draw and seed (regression is impossible by construction). "
        "coresident_spouse is also byte-identical (no delta touches it, though "
        "it did not clear in candidate 2). The maternal side is byte-identical "
        "(the pooled leave-year call is unchanged; only the LINKED children's "
        "coresidence is overridden by the delta-1 custodial gate)."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "Each simulated side-A holdout person is seeded from their OWN "
        "observed coresident-parent / multigen / cohabitation / "
        "SKIPPED-GENERATION state at their first 2b wave, then evolved with "
        "the train-fitted hazards. Using a holdout person's own window-entry "
        "state is not fitting (no parameter is estimated from side A); every "
        "fitted parameter is estimated on side B only. VERBATIM from candidate "
        "2, extended to the skipped-generation state (same convention)."
    ),
    "custodial_paternal_conditioning": (
        "Delta 1. Candidate 2 attributed EVERY cah85_23 father->child "
        "biological birth as coresident (aged out under the parental-home "
        "hazard), which over-represents coresidence (non-custodial fathers -- "
        "the OVERSHOOT the grading isolated). Candidate 3 instead lets a "
        "father-linked child count as the man's coresident child in a wave "
        "with the train-fitted probability P(linked child coresident with "
        "father | child age band x father marital state), drawn per wave per "
        "draw (independent per child-wave -- a point-in-time STOCK, matching "
        "the reference; there is no coresident_child transition cell). "
        "RESOLUTIONS: (a) coresidence is measured at the SPECIFIC (father, "
        "child, wave) grain -- the father is coded a parent (MX8 {50,53,55,56} "
        "= hc.CORESIDENCE_LINKS['coresident_child'], verified against the raw "
        "MX23REL formats) of that child that wave -- using the joinable "
        "cah85_23 child_person_id (child_person_id present, per the births "
        "module: ~67k joinable events); the probability is fit on that "
        "observable subset and applied to candidate 2's FULL linked-child set "
        "(the standard fit-on-observable / apply-to-population step, keeping "
        "the candidate-2 linked/unlinked split unchanged). (b) child age bands "
        "(0-4, 5-12, 13-17, 18-24, 25-60) are fixed a priori to the "
        "coresidence life-cycle, NOT tuned to the holdout; a linked child "
        "older than 60 is never counted (candidate-1 open top). (c) 'father "
        "marital state' is the certified marital core's married vs not-married "
        "binary (the salient custodial axis; dense support in both cells) -- "
        "fit on the OBSERVED marital state, applied on the SIMULATED marital "
        "state (self-consistent: the simulated household's marital trajectory "
        "gates its own children), uncovered waves falling back to the observed "
        "spouse-union state then not-married."
    ),
    "household_bridge_nonfamily_members": (
        "Delta 2. The reference hh_size counts every enumerated household "
        "member; the composed ego-centric family unit (1 + spouse + children "
        "+ parents) omits non-family coresidents (siblings, other relatives, "
        "roomers). Candidate 3 fits the train distribution of the NON-FAMILY "
        "member count -- enumerated hh_size minus the ego's 1 + spouse-links + "
        "child-links + parent-links (the same MX8 link families the "
        "composition rule uses; verified codes), clipped at 0 and binned "
        "0/1/2+ by ego age band x sex -- samples it per draw, and adds it ONLY "
        "to the hh_size composition. RESOLUTION: the registration names the "
        "support 'count 0/1/2+'; a sampled '2+' contributes the minimal "
        "guaranteed count (2) to hh_size (the label 2+ guarantees exactly 2; a "
        "larger value would invent mass the 3-category distribution does not "
        "specify), so the deep composition tail (hh_size.5+) stays the named "
        "residual rather than being over-shot. The count feeds NO coresidence "
        "family cell (a leak would be a spec violation)."
    ),
    "skipped_generation_coresidence": (
        "Delta 3. The locked reference EXCLUDES skipped-generation households "
        "(grandparent + grandchild, no middle generation) from multigen (two "
        "generations, not three -- Census B11017; verified in the household "
        "composition module docstring and _MULTIGEN_MIN_GENERATIONS=3), so "
        "the composed grandchild (multigen AND coresident_child AND NOT "
        "coresident_parent) misses grandparents raising grandchildren alone. "
        "Candidate 3 fits train entry/exit hazards by age band x sex for the "
        "OBSERVED skipped-generation state -- coresident_grandchild AND NOT "
        "multigen (the coresident_grandchild flag is MX8 {66,68,82,87,88}, "
        "verified against the raw formats; both flags are already on the "
        "reference roster) -- evolves it from each holdout person's observed "
        "initial state (candidate-2 convention), and UNIONS it into the "
        "composed grandchild concept ONLY. It is explicitly NOT counted toward "
        "multigen (which stays byte-identical). The union cannot double-count "
        "in the boolean grandchild stock; the disjoint construction (composed "
        "is on multigen=True waves, the observed skip-gen state on "
        "multigen=False waves) keeps them separable."
    ),
    "coresident_children_maternal_side_untouched": (
        "The maternal side is UNCHANGED from candidate 2: women's maternal "
        "births come from the certified tranche-2a registry simulate and age "
        "out under the fitted parental-home hazard on the same 0xC2 child "
        "stream (byte-identical). Only the paternal LINKED stream changes "
        "(delta 1); the unlinked shadow kernel is byte-identical."
    ),
    "household_size_composition": (
        "hh_size = 1 + coresident_spouse + n_coresident_children + "
        "(parent_count if coresident_parent else 0) + non_family_count -- the "
        "candidate-1 composition rule (byte-faithful, never separately tuned) "
        "plus the delta-2 non-family term. It improves as the spouse and "
        "children inputs carry and the non-family bridge supplies the "
        "enumerated-household mass the family unit omits."
    ),
    "coresident_grandchild_composed_plus_skipgen": (
        "coresident_grandchild = (multigen AND coresident_child AND NOT "
        "coresident_parent) UNION the delta-3 skipped-generation state -- the "
        "candidate-1 composed implication plus the skip-gen occupancy the "
        "locked reference includes in the grandchild concept but excludes from "
        "multigen."
    ),
    "coresident_spouse_fallback": (
        "For side-A 2b waves the certified simulation does not cover, the "
        "legal-marriage component carries the person's OWN observed first-wave "
        "coresident-spouse state (candidate-1 fallback, byte-faithful) BEFORE "
        "the cohabitation union is applied (candidate-2 convention)."
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
    """Recursively convert numpy scalars and non-finite floats for the file."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
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
# Data loading (candidate-2 loader + the seed-independent delta-1/2 inputs)
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
    """Load the MX23REL household panel, the marital panel, loaders, and the
    seed-independent candidate-3 delta inputs (resolved once, reused per
    seed)."""
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

    # Seed-independent delta inputs (the train filter differs per seed).
    father_links_child = hcs3.father_link_births_with_child(bh)
    parent_pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    fu_sizes = hcs3.family_unit_sizes(rel_map)
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
# Per-seed scoring (mean over K=20 draws) -- candidate-2 scoring, v3 model
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    data: dict[str, Any],
    floor: dict[str, Any],
    tol: dict[str, float],
    report_only: list[str],
    verbose: bool,
) -> dict[str, Any]:
    t0 = time.time()
    hh = data["hh"]
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs3.fit_household_model_v3(
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
        sim_panel, diag = hcs3.simulate_draw_v3(
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
        if rbar > 0 and rate_a > 0:
            s = float(abs(math.log(rbar / rate_a)))
        else:
            s = float("inf")
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

    delta_stats = {
        "delta_1_custodial": {
            "mean_linked_child_coresident_wave_units": _dmean(
                "n_linked_child_coresident_wave_units"
            ),
            "mean_paternal_linked_births": _dmean("n_paternal_linked_births"),
            "custodial_probability_by_band_marital": {
                f"{b}|{m}": round(v, 5)
                for (b, m), v in model.custodial.items()
            },
            "custodial_n_train_exposure": model.meta["custodial_n_exposure"],
            "custodial_n_train_coresident": model.meta[
                "custodial_n_coresident"
            ],
            "custodial_train_overall_rate": model.meta[
                "custodial_overall_rate"
            ],
        },
        "delta_2_household_bridge": {
            "mean_nonfamily_count_simulated": _dmean(
                "mean_nonfamily_count_simulated"
            ),
            "nonfamily_train_overall_p0_p1_p2plus": model.meta[
                "nonfamily_overall_p0_p1_p2plus"
            ],
            "nonfamily_train_weighted_mean_count": model.meta[
                "nonfamily_weighted_mean_count"
            ],
            "nonfamily_distribution_by_band_sex": {
                f"{b}|{s}": [round(x, 5) for x in v]
                for (b, s), v in model.nonfamily.items()
            },
        },
        "delta_3_skipgen": {
            "mean_skipgen_person_waves_simulated": _dmean(
                "n_skipgen_person_waves_simulated"
            ),
            "skipgen_entry_overall": model.meta["skipgen_entry_overall"],
            "skipgen_exit_overall": model.meta["skipgen_exit_overall"],
            "skipgen_train_person_waves": model.meta[
                "skipgen_train_person_waves"
            ],
        },
    }
    coverage = {
        **coverage,
        "mean_paternal_linked_births": _dmean("n_paternal_linked_births"),
        "mean_paternal_shadow_births": _dmean("n_paternal_shadow_births"),
        "mean_maternal_births": _dmean("n_maternal_births"),
    }
    if verbose:
        fails = [k for k, v in gated_cells.items() if not v["pass"]]
        print(
            f"seed {seed}: {n_gated_pass}/{len(tol)} gated pass "
            f"(seed_pass={seed_pass}); K={N_DRAWS}; "
            f"undefined={len(undefined)}; father_cov="
            f"{coverage['coverage_fraction_men']}; fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "estimator": "mean_over_K20_draws",
        "draw_seeds": draw_seeds,
        "sim_seed_single_draw_provenance": SIM_SEED_PROVENANCE_BASE + seed,
        "component_meta": model.meta,
        "father_link_coverage": coverage,
        "delta_stats": delta_stats,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "undefined_gated_draws": undefined,
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
# Verdict + per-family decomposition (candidate taxonomy)
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


def per_family_decomposition(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Per family: seed x cell pass rate, worst cell, |ln| vs tol, mechanism."""
    mechanisms = {
        "coresident_spouse": (
            "certified legal-marriage registry UNION the cohabitation "
            "(code-22) overlay (candidate 2, byte-faithful, RNG-isolated); no "
            "candidate-3 delta touches it."
        ),
        "coresident_parent": (
            "directly fitted logistic exit hazard from observed initial "
            "states (candidate 1, byte-faithful, RNG-isolated); expected to "
            "clear."
        ),
        "coresident_child": (
            "observed father->child links for linked men, now GATED per wave "
            "by the train-fitted custodial coresidence probability (delta 1); "
            "unlinked shadow kernel and the maternal certified kernel are "
            "byte-faithful."
        ),
        "coresident_grandchild": (
            "composed implication (multigen AND coresident_child AND NOT "
            "coresident_parent) UNION the delta-3 skipped-generation "
            "occupancy; inherits the child fix and the carried multigen."
        ),
        "multigen_stock": (
            "carried initial state + train band x sex entry/exit (candidate "
            "1, byte-faithful, RNG-isolated); the skip-gen delta does NOT "
            "feed it."
        ),
        "multigen_transition": (
            "directly fitted pooled entry/exit rates (candidate 1, "
            "byte-faithful, RNG-isolated); expected to clear."
        ),
        "parental_home_exit": (
            "directly fitted (candidate 1, byte-faithful, RNG-isolated); the "
            "transition expected to clear."
        ),
        "hh_size": (
            "composed ego-centric family unit (1 + spouse + children + "
            "parents) PLUS the delta-2 non-family member bridge (0/1/2+); the "
            "5+ deep tail stays the named residual."
        ),
    }
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
            "mechanism": mechanisms.get(fam, ""),
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
# Candidate 1 -> 2 -> 3 progression + cleared-family regression + byte carry
# --------------------------------------------------------------------------
def comparison_across_candidates(
    decomposition: dict[str, Any],
    per_seed: list[dict[str, Any]],
    tol: dict[str, float],
) -> dict[str, Any]:
    """Per-family c1 -> c2 -> c3 pass rates, cleared-family regression, and
    the byte-identical carried-family score check against candidate 2."""
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())
    c1_decomp = c1["per_family_decomposition"]
    c2_decomp = c2["per_family_decomposition"]
    per_family: dict[str, Any] = {}
    for fam, d3 in decomposition.items():
        r1 = c1_decomp.get(fam, {}).get("cell_seed_pass_rate")
        r2 = c2_decomp.get(fam, {}).get("cell_seed_pass_rate")
        r3 = d3["cell_seed_pass_rate"]
        per_family[fam] = {
            "candidate1_pass_rate": r1,
            "candidate2_pass_rate": r2,
            "candidate3_pass_rate": r3,
            "delta_c2_to_c3": (round(r3 - r2, 4) if r2 is not None else None),
            "candidate2_worst_cell": c2_decomp.get(fam, {}).get("worst_cell"),
            "candidate2_worst_mean_abs_ln": c2_decomp.get(fam, {}).get(
                "worst_cell_mean_abs_ln"
            ),
            "candidate3_worst_cell": d3["worst_cell"],
            "candidate3_worst_mean_abs_ln": d3["worst_cell_mean_abs_ln"],
        }

    # Cleared-family regression check vs candidate 2's cleared families.
    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE2_CLEARED_FAMILIES:
        r2 = c2_decomp.get(fam, {}).get("cell_seed_pass_rate")
        r3 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r3 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate2_pass_rate": r2,
            "candidate3_pass_rate": r3,
            "still_clears": bool(holds),
        }

    # Byte-identical carried-family score check: every carried cell's per-seed
    # score equals candidate 2's (parent / multigen / parental-home /
    # transitions, plus the carried-but-uncleared coresident_spouse).
    c2_by_seed = {s["seed"]: s for s in c2["per_seed"]}
    c3_by_seed = {s["seed"]: s for s in per_seed}
    carried_cells = [
        c
        for c in tol
        if c.startswith(
            (
                "coresident_parent.",
                "multigen.",
                "parental_home_",
                "coresident_spouse.",
            )
        )
        or c in ("multigen_entry", "multigen_exit")
    ]
    max_carry_dev = 0.0
    for seed in (s["seed"] for s in per_seed):
        for cell in carried_cells:
            s3 = c3_by_seed[seed]["gated_cells"][cell]["score"]
            s2 = c2_by_seed[seed]["gated_cells"][cell]["score"]
            if math.isfinite(s3) and math.isfinite(s2):
                max_carry_dev = max(max_carry_dev, abs(s3 - s2))
    return {
        "candidate1_artifact": "runs/gate2b_hazard_v1.json",
        "candidate2_artifact": "runs/gate2b_hazard_v2.json",
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "candidate2_registration_pointer": CANDIDATE2_REGISTRATION_POINTER,
        "candidate1_verdict": {
            "gate_2b_pass": c1["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c1["verdict"]["n_seeds_pass"],
        },
        "candidate2_verdict": {
            "gate_2b_pass": c2["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c2["verdict"]["n_seeds_pass"],
        },
        "per_family_progression": per_family,
        "cleared_family_regression_check": {
            "families": list(CANDIDATE2_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-2 families that cleared are carried "
                "byte-faithfully (read off candidate 1's simulate_draw "
                "unchanged) and their occupancy draws are RNG-isolated from "
                "the three deltas, so they are expected to stay cleared; this "
                "check confirms it on the scored run."
            ),
        },
        "byte_identical_carried_family_score_check": {
            "carried_cells": sorted(carried_cells),
            "max_abs_score_deviation_vs_candidate2": max_carry_dev,
            "byte_identical": bool(max_carry_dev <= EXACT_ATOL),
            "note": (
                "Every carried cell's per-seed gated score equals candidate "
                "2's to bit precision (parent / multigen / parental-home / "
                "multigen transitions, plus the carried-but-uncleared "
                "coresident_spouse). This is the strong regression proof: the "
                "three deltas draw from an isolated 0xC3 stream and cannot "
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
            f"{precheck['all_reproduced_exactly']} "
            f"(ref dev={precheck['reference_moments_max_abs_deviation']:.2e}, "
            f"rate_a dev={precheck['rate_a_max_abs_deviation']:.2e}, "
            f"sha_all={precheck['holdout_sha256_all_match']})"
        )
    if not precheck["all_reproduced_exactly"]:
        raise RuntimeError(
            "Scoring path does not reproduce the committed gate-2b floor "
            "(reference moments / per-seed rate_a / holdout sha256) to bit "
            "precision; refusing to proceed."
        )

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(
            score_seed(seed, data, floor, tol, report_only, verbose)
        )

    per_draw_cube = _per_draw_cube(per_seed, tol)
    undefined_block = _undefined_block(per_seed)
    dispersion_block = _dispersion_block(per_seed, tol)
    if undefined_block["run_invalidated"]:
        raise RuntimeError(
            "RUN INVALIDATED (fresh_run_artifact_schema.undefined_draw_rule): "
            f"{undefined_block['n_undefined_gated_draws']} undefined gated "
            "cell draw(s) (empty simulated denominator); the run must be "
            "re-registered and re-run."
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
                f"(c1 {c['candidate1_pass_rate']}, c2 "
                f"{c['candidate2_pass_rate']}, d_c2c3 {c['delta_c2_to_c3']}); "
                f"worst {d['worst_cell']} |ln|={d['worst_cell_mean_abs_ln']} "
                f"tol={d['worst_cell_tolerance']}"
            )
        chk = comparison["cleared_family_regression_check"]
        byt = comparison["byte_identical_carried_family_score_check"]
        print(
            "  cleared-family regression: all_still_clear="
            f"{chk['all_cleared_families_still_clear']}; carried byte-identical"
            f"={byt['byte_identical']} (max dev "
            f"{byt['max_abs_score_deviation_vs_candidate2']:.2e})"
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "candidate": "candidate 3",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate2_registration_pointer": CANDIDATE2_REGISTRATION_POINTER,
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "grading_pointer": GRADING_POINTER,
        "one_shot": (
            "Registered on issue #42 comment 4939960467 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_2": [
            "delta 1: custodial paternal conditioning -- father-linked "
            "children counted per wave with the train-fitted P(coresident | "
            "child age band x father marital state)",
            "delta 2: household bridge -- train-fitted non-family member count "
            "(0/1/2+ by ego band x sex) added to hh_size only",
            "delta 3: skipped-generation coresidence -- train-fitted "
            "entry/exit hazards for coresident_grandchild AND NOT multigen, "
            "unioned into the composed grandchild only (never multigen)",
        ],
        "per_delta_target_family": PER_DELTA_TARGET_FAMILY,
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 2's generator REUSED byte-faithfully (which "
                "reuses candidate 1), plus three train-fitted deltas: a "
                "custodial coresidence gate on the linked paternal children "
                "(delta 1), a non-family household-member count added to "
                "hh_size (delta 2), and a skipped-generation grandchild "
                "occupancy unioned into coresident_grandchild (delta 3)."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "candidate2_module": (
                "populace_dynamics.models.household_composition_sim_v2"
            ),
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v3"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "parental_exit_knots": list(hcs.PARENTAL_EXIT_KNOTS),
            "cohabitation_partner_code": hcs2.PARTNER_CODE,
            "custodial_child_age_bands": [
                hc.band_label(lo, hi)
                for lo, hi in hcs3.CUSTODIAL_CHILD_AGE_BANDS
            ],
            "nonfamily_classes": list(hcs3.NONFAMILY_CLASSES),
            "components": [
                "coresident_spouse<-certified_married|UNION|cohab_code22",
                "coresident_parent<-logistic_exit_hazard_age_spline_sex",
                "multigen<-train_band_sex_entry_exit_carried_initial",
                "coresident_child<-custodial_gated_father_links+maternal_kernel"
                "+shadow_unlinked_residual",
                "hh_size<-composed(1+spouse+children+parents)+nonfamily_bridge",
                "coresident_grandchild<-composed(multigen&child&~parent)"
                "|UNION|skipgen_occupancy",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from candidate 1's simulate_draw "
                "UNCHANGED (occupancy tag 0xB2B); the candidate-2 cohabitation "
                "and child streams are SeedSequence([5200+k, 0xC2]).spawn(2); "
                "the three candidate-3 deltas draw from a separate "
                "SeedSequence([5200+k, 0xC3]).spawn(3) -> [custodial, "
                "non-family, skipped-generation], isolated from 0xB2B / 0xC2"
            ),
            "sim_seed_single_draw_provenance": (
                "numpy.random.default_rng(4200 + seed) (recorded, not used "
                "by the mean-over-K estimator)"
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
        "data": {
            "holdout_basis": ["MX23REL"],
            "paternal_link_basis": ["cah85_23"],
            "n_person_waves": int(len(hh.person_waves)),
            "n_persons": int(hh.person_waves.person_id.nunique()),
            "floor_run": "runs/gate2b_floors_v1.json",
            "floor_run_sha256": _sha_of_file(FLOOR_RUN),
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
