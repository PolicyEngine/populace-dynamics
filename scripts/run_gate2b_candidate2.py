"""Gate-2b candidate 2: cohabitation overlay + paternal child attribution.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b, ratified 2026-07-10; 46 gated household/relationship cells,
K=20 mean-over-draws, per-seed train/holdout protocol). Registered on issue
#42 comment 4939456379 BEFORE this run; published REGARDLESS of verdict.

Candidate 2 is candidate 1 (registration 4938726107; PR #133) with EXACTLY
TWO frozen deltas; everything that cleared in candidate 1 -- the certified
tranche-2a marital core and maternal births, the parental-home exit hazard,
the multigen entry/exit machinery, ``coresident_parent``, the household-size
composition rule, and the composed-only grandchild -- is carried byte-
faithfully (the delta model REUSES the candidate-1 module, it does not copy
it), and the scoring path here is candidate 1's, unchanged.

Delta 1 -- cohabiting-partner overlay. The 2b reference spouse concept is the
MX8 spouse-OR-partner flag (codes 20/22); the staged formats decode 20='Legal
Spouse', 22='Partner - cohabiting, not legally married'. The certified 2a
registry generates the legal (code-20) mass, so candidate 2 fits a
cohabitation occupancy on the code-22 partner spells ONLY -- train entry/exit
hazards by age band x sex, like the multigen machinery, from each holdout
person's observed initial state -- and ``coresident_spouse`` becomes the UNION
of the legal-marriage state and the cohabitation state.

Delta 2 -- paternal child attribution. Coresident children of men come from
the observed cah85_23 father->child birth links (each man's own recorded
biological births, a data initial condition), aged out under the SAME fitted
home-exit hazard; the candidate-1 spousal-gap-shifted shadow kernel is
retained ONLY for the residual men with no recorded father link (the split is
recorded in the artifact).

Estimator, artifact schema, undefined-draw rule, and protocol are candidate
1's (locked contract). Artifact ``runs/gate2b_hazard_v2.json``;
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

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v2"
RUN_NAME = "gate2b_hazard_v2"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
REGISTRATION_POINTER = "4939456379"
CANDIDATE1_REGISTRATION_POINTER = "4938726107"
SPEC_REGISTRATION = (
    "issue #42 comment 4939456379: gate-2b candidate 2, cohabitation "
    "overlay + paternal child attribution"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol; candidate 1).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
EXACT_ATOL = 1e-12

#: The candidate-1 families that CLEARED (grading comment 4939453260): the
#: regression check confirms candidate 2 carries them non-interacting.
CANDIDATE1_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.35-0.50",
    "expected_effects": [
        "coresident_spouse young cells clear or near-clear (the overlay "
        "supplies the missing code-22 partner concept mass)",
        "coresident_child male cells improve materially but may not fully "
        "clear (father-link coverage is partial; the unlinked shadow is a "
        "named residual)",
        "hh_size improves as its spouse/children inputs improve, but "
        "hh_size.5+ stays the modal residual failure (family-unit vs "
        "enumerated-household concept)",
        "coresident_grandchild improves via the child fix but its 55+|female "
        "cell may stay out",
        "everything that cleared in candidate 1 stays cleared (the union is "
        "additive only; the deltas are RNG-isolated from the cleared "
        "families)",
    ],
    "regression_risk_named": (
        "the overlay must not disturb the marital state -- the union is "
        "additive only, and the parental-home / multigen occupancy draws are "
        "isolated on dedicated substreams"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT "
        "grade it."
    ),
}

#: Candidate 1's resolutions carried VERBATIM (the carried components are
#: byte-faithful, so their resolutions are unchanged) plus the two-delta
#: resolutions. Adopted per the registration ("spec ambiguities resolved per
#: the locked contract text and documented in the artifact").
SPEC_RESOLUTION_NOTES = {
    "rng_two_stream": (
        "The registration RNG clause reads 'default_rng(4200+seed) outer, "
        "5200+k draws, K=20'. The LOCKED gate_2b protocol (gates.yaml "
        "gate_2.gate_2b.thresholds.protocol) defines the estimator as the "
        "mean over K=20 household-simulation draws at "
        "numpy.random.default_rng(5200 + k), k=0..19, and names no per-seed "
        "outer stream. Resolution (locked contract wins, VERBATIM from "
        "candidate 1): each of the K=20 draws re-simulates the whole "
        "composition keyed to draw seed 5200+k (the certified tranche-2a "
        "simulate uses default_rng(5200+k); the occupancy overlays use a "
        "distinct substream SeedSequence([5200+k, 0xB2B])). "
        "default_rng(4200+seed) is recorded as the single-draw provenance "
        "seed; it does not drive the mean-over-K estimator."
    ),
    "rng_byte_identical_carried_families": (
        "Candidate 2 adds TWO stochastic deltas (cohabitation occupancy; a "
        "changed paternal-child stream). To carry the candidate-1 families "
        "that cleared BYTE-FAITHFULLY -- the registration's explicit intent "
        "('carried byte-faithfully; the union is additive only; everything "
        "that cleared stays cleared') -- candidate 2 does NOT re-simulate the "
        "parental-home / multigen occupancy: it calls candidate 1's "
        "simulate_draw UNCHANGED (same draw seed 5200+k, same occupancy tag "
        "0xB2B) and reads coresident_parent and multigen straight off its "
        "panel. Those states -- and therefore the coresident_parent, "
        "multigen, parental_home_exit and multigen entry/exit families -- are "
        "bit-identical to candidate 1 on every draw and every seed, so the "
        "deltas cannot perturb them (regression is impossible by "
        "construction, not merely improbable; a spawn-based re-simulation "
        "would re-randomize a knife-edge multigen cell). The two deltas draw "
        "from a distinct SeedSequence([5200+k, 0xC2]) (spawned into a child "
        "stream and a cohabitation stream), isolated from the candidate-1 "
        "occupancy stream; the household is then recomposed from the "
        "byte-identical carried states plus the union spouse (delta 1) and "
        "the father-link children (delta 2). The comparison_to_candidate_1 "
        "block confirms the cleared families stay cleared."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "Each simulated side-A holdout person is seeded from their OWN "
        "observed coresident-parent / multigen / COHABITATION state at their "
        "first 2b wave, then evolved with the train-fitted hazards -- exactly "
        "as the certified tranche-2a simulate seeds each holdout person from "
        "their own observed initial marital state. Using a holdout person's "
        "own window-entry state as the initial condition is not fitting (no "
        "parameter is estimated from side A); every fitted parameter -- the "
        "registry components, the parental-exit logistic, the multigen and "
        "cohabitation hazards -- is estimated on side B only. VERBATIM from "
        "candidate 1, extended to the cohabitation state (same convention)."
    ),
    "cohabitation_overlay_is_code_22_partner_only": (
        "Delta 1. The 2b reference coresident_spouse flag is MX8 codes "
        "{20, 22} (hc.CORESIDENCE_LINKS). The staged MX23REL formats file "
        "(MX23REL_formats.sps, verified 2026-07-10, the standing raw-format "
        "lesson) decodes 20='Legal Spouse - EGO is the legal spouse to "
        "ALTER' and 22='Partner - EGO is the cohabiting partner to ALTER, "
        "not legally married'. The certified 2a registry ALREADY generates "
        "legal marriage (the code-20 mass), so the cohabitation occupancy "
        "overlay is fit on the code-22 partner spells ONLY -- the "
        "partner-inclusion concept mass the legal registry omits -- and "
        "coresident_spouse = legal_marriage_sim (~20) UNION cohabitation_sim "
        "(22). Fitting on {20,22} would double-model the code-20 mass the "
        "registry already carries; the registration's 'MX8 codes 20/22' names "
        "the reference concept, and the raw-format decode dictates the "
        "code-22-only overlay that supplies the MISSING mass. The code-20 and "
        "code-22 masses are disjoint by construction, so the union cannot "
        "double-count."
    ),
    "paternal_attribution_from_observed_father_links": (
        "Delta 2. Coresident children of men come from the observed cah85_23 "
        "father->child biological birth links (records with parent_sex=male, "
        "record_type=birth, a non-NA child birth year; CAH5 decodes 1=Male, "
        "2=Female, verified against the raw formats). A side-A man is LINKED "
        "if he is the male parent of >= 1 such record; his OWN recorded "
        "births (birth years) are attributed and aged out under the same "
        "fitted home-exit hazard -- a data initial condition of the same "
        "species as the observed initial states (no side-A parameter is "
        "estimated; the birth years are the man's own recorded data, and "
        "coresidence is DECIDED by the train-fitted aging hazard, not copied "
        "from the scored roster). 'where the train data records them' is read "
        "as 'where cah85_23 records a father link', i.e. the man's own "
        "birth-history record, not a train-persons restriction (parity with "
        "the observed-initial resolution). The candidate-1 spousal-gap "
        "shadow kernel is retained ONLY for the residual UNLINKED men (men "
        "with no recorded father link -- denial reports and men absent from "
        "cah85_23); linked men are excluded from the shadow, so no man is "
        "double-counted. Biological births only (adoptions excluded) to "
        "mirror the fertility kernel the shadow replaces. Consequence "
        "(reported, not tuned): observed father links attribute biological "
        "paternity, which for men over-represents coresidence relative to "
        "the maternal-custodial norm (non-coresident fathers), so the male "
        "coresident_child cells can over-shoot where candidate 1 under-shot "
        "-- a concept residual, not a fit choice."
    ),
    "coresident_children_from_certified_kernel_maternal_side": (
        "The maternal side is UNCHANGED from candidate 1: women's own "
        "maternal births come from the certified tranche-2a registry "
        "simulate; only the paternal (male) stream changes (delta 2). "
        "Children age out under the fitted parental-home-exit hazard at the "
        "child's age (sex drawn 0.5), open-topped at age 60 -- the candidate-"
        "1 mechanism, byte-faithful."
    ),
    "household_size_composition": (
        "hh_size = 1 + coresident_spouse + n_coresident_children + "
        "(parent_count if coresident_parent else 0) -- the candidate-1 "
        "composition rule, byte-faithful and never separately tuned. It "
        "improves mechanically as the spouse (delta 1) and children (delta 2) "
        "inputs improve; it stays the ego-centric FAMILY-UNIT size, narrower "
        "than the enumerated PSID HOUSEHOLD (it omits siblings, other "
        "relatives, roomers), so the 5+ tail stays under-weighted."
    ),
    "coresident_grandchild_composed_only": (
        "coresident_grandchild = multigen AND coresident_child AND NOT "
        "coresident_parent -- a deterministic composed implication of the "
        "simulated states with no fitted parameter (candidate-1 mechanism, "
        "byte-faithful). It inherits the child fix but its top-generation "
        "elderly-female cell is largely unaffected by a male-child delta."
    ),
    "coresident_spouse_fallback": (
        "For side-A 2b waves the certified simulation does not cover "
        "(attrition-year gaps; persons with no marriage-file record), the "
        "legal-marriage component carries the person's OWN observed "
        "first-wave coresident-spouse state (candidate-1 fallback, "
        "byte-faithful) BEFORE the cohabitation union is applied."
    ),
    "gates_yaml_path": (
        "The locked block is gates.yaml gates.gate_2.gate_2b.thresholds; its "
        "46 gated tolerances and the floor gate_partition are read at "
        "runtime, never hardcoded (candidate-1 convention)."
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
# Data loading (candidate-1 loader; the delta inputs are derived in the model)
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
    """Load the MX23REL household panel, the marital panel, and loaders."""
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
    return {
        "hh": hh,
        "mpanel": mpanel,
        "demo": demo,
        "mh": mh,
        "bh": bh,
        "rel_map": rel_map,
        "order_map": order_map,
    }


# --------------------------------------------------------------------------
# Precheck: reproduce the committed floor bit-for-bit (candidate-1 mirror)
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
            "Hard-stop precheck (candidate-1 mirror): the scoring path must "
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
# Per-seed scoring (mean over K=20 draws) -- candidate-1 scoring, v2 model
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

    model = hcs2.fit_household_model_v2(
        hh,
        data["mpanel"],
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b,
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
        sim_panel, diag = hcs2.simulate_draw_v2(
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
    coverage = {
        **coverage,
        "mean_paternal_linked_births": float(
            np.mean([d["n_paternal_linked_births"] for d in draw_diagnostics])
        ),
        "mean_paternal_shadow_births": float(
            np.mean([d["n_paternal_shadow_births"] for d in draw_diagnostics])
        ),
        "mean_maternal_births": float(
            np.mean([d["n_maternal_births"] for d in draw_diagnostics])
        ),
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
# Verdict + per-family decomposition (candidate-1 taxonomy)
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
            "certified legal-marriage registry UNION the train-fitted "
            "cohabitation (MX8 code-22) overlay (delta 1); the young-adult "
            "shortfall the overlay supplies is the partner-inclusion concept."
        ),
        "coresident_parent": (
            "directly fitted logistic exit hazard from observed initial "
            "states (candidate 1, byte-faithful, RNG-isolated); expected to "
            "clear."
        ),
        "coresident_child": (
            "observed cah85_23 father->child links for linked men (delta 2) "
            "with the candidate-1 shadow kernel for the unlinked residual; "
            "the maternal side is the certified kernel, unchanged."
        ),
        "coresident_grandchild": (
            "composed implication only (multigen AND coresident_child AND NOT "
            "coresident_parent); inherits the child fix and the multigen "
            "residual."
        ),
        "multigen_stock": (
            "carried initial state + train band x sex entry/exit (candidate "
            "1, byte-faithful, RNG-isolated)."
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
            "parents); improves as its spouse/children inputs improve, but "
            "the 5+ tail stays under-weighted."
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
# Candidate 1 -> candidate 2 comparison + cleared-family regression check
# --------------------------------------------------------------------------
def comparison_to_candidate_1(
    decomposition: dict[str, Any],
) -> dict[str, Any]:
    """Per-family candidate-1 vs candidate-2 pass rates + regression check.

    Reads the committed candidate-1 artifact (``runs/gate2b_hazard_v1.json``)
    and reports, per family, the candidate-1 and candidate-2 cell-seed pass
    rates and their delta; the regression check asserts every candidate-1
    CLEARED family still clears (cell-seed pass rate == 1.0) in candidate 2.
    """
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())
    c1_decomp = c1["per_family_decomposition"]
    per_family: dict[str, Any] = {}
    for fam, d2 in decomposition.items():
        d1 = c1_decomp.get(fam, {})
        r1 = d1.get("cell_seed_pass_rate")
        r2 = d2["cell_seed_pass_rate"]
        per_family[fam] = {
            "candidate1_pass_rate": r1,
            "candidate2_pass_rate": r2,
            "delta": (round(r2 - r1, 4) if r1 is not None else None),
            "candidate1_worst_cell": d1.get("worst_cell"),
            "candidate1_worst_mean_abs_ln": d1.get("worst_cell_mean_abs_ln"),
            "candidate2_worst_cell": d2["worst_cell"],
            "candidate2_worst_mean_abs_ln": d2["worst_cell_mean_abs_ln"],
        }
    cleared_check = {}
    all_cleared_hold = True
    for fam in CANDIDATE1_CLEARED_FAMILIES:
        r1 = c1_decomp.get(fam, {}).get("cell_seed_pass_rate")
        r2 = decomposition.get(fam, {}).get("cell_seed_pass_rate")
        holds = r2 == 1.0
        all_cleared_hold = all_cleared_hold and holds
        cleared_check[fam] = {
            "candidate1_pass_rate": r1,
            "candidate2_pass_rate": r2,
            "still_clears": bool(holds),
        }
    return {
        "candidate1_artifact": "runs/gate2b_hazard_v1.json",
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "candidate1_verdict": {
            "gate_2b_pass": c1["verdict"]["gate_2b_pass"],
            "n_seeds_pass": c1["verdict"]["n_seeds_pass"],
        },
        "per_family": per_family,
        "cleared_family_regression_check": {
            "families": list(CANDIDATE1_CLEARED_FAMILIES),
            "detail": cleared_check,
            "all_cleared_families_still_clear": bool(all_cleared_hold),
            "note": (
                "The candidate-1 families that cleared are carried "
                "byte-faithfully and their occupancy draws are RNG-isolated "
                "from the two deltas, so they are expected to stay cleared; "
                "this check confirms it on the scored run."
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
    comparison = comparison_to_candidate_1(decomposition)
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
            c = comparison["per_family"][fam]
            print(
                f"  {fam:22s} pass {d['cell_seed_pass_rate']:.2f} "
                f"(c1 {c['candidate1_pass_rate']:.2f}, "
                f"d {c['delta']:+.2f}); worst {d['worst_cell']} "
                f"|ln|={d['worst_cell_mean_abs_ln']} "
                f"tol={d['worst_cell_tolerance']}"
            )
        chk = comparison["cleared_family_regression_check"]
        print(
            "  cleared-family regression check: all_still_clear="
            f"{chk['all_cleared_families_still_clear']}"
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "candidate": "candidate 2",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate1_registration_pointer": CANDIDATE1_REGISTRATION_POINTER,
        "grading_pointer": "4939453260",
        "one_shot": (
            "Registered on issue #42 comment 4939456379 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "deltas_vs_candidate_1": [
            "delta 1: cohabiting-partner (MX8 code-22) occupancy overlay "
            "unioned into coresident_spouse",
            "delta 2: paternal child attribution from observed cah85_23 "
            "father->child links, shadow kernel retained for the unlinked "
            "residual",
        ],
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Candidate 1's six-component structural generator REUSED "
                "byte-faithfully, plus a train-fitted cohabitation (MX8 "
                "code-22) occupancy overlay unioned into coresident_spouse "
                "(delta 1) and observed cah85_23 father->child paternal "
                "attribution with the candidate-1 shadow kernel retained for "
                "the unlinked residual (delta 2)."
            ),
            "base_module": "populace_dynamics.models.household_composition_sim",
            "delta_module": (
                "populace_dynamics.models.household_composition_sim_v2"
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "parental_exit_knots": list(hcs.PARENTAL_EXIT_KNOTS),
            "cohabitation_partner_code": hcs2.PARTNER_CODE,
            "components": [
                "coresident_spouse<-certified_registry_married"
                "|UNION|cohabitation_code22_overlay",
                "coresident_parent<-logistic_exit_hazard_age_spline_sex",
                "multigen<-train_band_sex_entry_exit_carried_initial",
                "coresident_child<-father_links_paternal+maternal_kernel"
                "+shadow_unlinked_residual",
                "hh_size<-composed(1+spouse+children+parents)",
                "coresident_grandchild<-composed(multigen&child&~parent)",
            ],
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "occupancy_substream": (
                "carried families come from candidate 1's simulate_draw "
                "UNCHANGED (occupancy tag 0xB2B) -> byte-identical "
                "coresident_parent / multigen; the two deltas draw from a "
                "separate SeedSequence([5200 + k, 0xC2]).spawn(2) -> "
                "[child, cohabitation], isolated from the candidate-1 stream"
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
        "comparison_to_candidate_1": comparison,
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
