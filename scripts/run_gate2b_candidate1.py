"""Gate-2b candidate 1: structural composition from the certified marital core.

ONE-SHOT scored run against the LOCKED gate_2b contract (``gates.yaml``
gate_2.gate_2b, ratified 2026-07-10; 46 gated household/relationship cells,
K=20 mean-over-draws, per-seed train/holdout protocol). Registered on issue
#42 comment 4938726107 BEFORE this run; published REGARDLESS of verdict. The
forecast is failure-likely (P(gate>=4/5) = 0.15-0.30); the per-family
decomposition is the primary value (it is what candidate 2 targets, the
tranche-2a ladder method).

Six-component structural generator (registration 4938726107):

1. coresident spouse -- the certified tranche-2a family-transition registry
   (``family_transitions.CANDIDATE_16``), per-seed refit on the train
   complement; simulated ``married`` -> coresident spouse.
2. coresident parent -- a logistic parental-home exit hazard (age spline x
   sex) fit on train MX23REL person-waves, evolved from the OBSERVED initial
   state at window entry.
3. multigenerational membership -- train-fitted entry/exit hazards by age
   band x sex, carried from the observed initial state.
4. coresident children -- the certified fertility kernel (maternal births
   from the registry simulate + a paternal shadow for married men), aging out
   under the fitted home-exit hazard.
5. household size -- COMPOSED (1 + spouse + coresident children + coresident
   parents), never separately tuned.
6. coresident grandchild -- NOT separately modelled; a composed implication.

Estimator: the ratified tranche-2a mean-over-K=20-draws statistic (gate_2b
protocol). Each draw re-simulates the whole composition at
``numpy.random.default_rng(5200 + k)`` and scores ``|ln(rbar / rate_a)|`` on
the 20-draw mean rate; a seed passes iff all 46 gated cells hold; the gate
passes iff >=4 of 5 seeds pass. Fitting is train-only (side B); the holdout
persons (side A) are simulated and scored. Run ONCE; ``artifacts.write_new``
refuses to overwrite. Reproduce with the repo ``.venv`` and the PSID products
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
from populace_dynamics.models import household_composition_sim as hcs

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_hazard_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2b_hazard.v1"
RUN_NAME = "gate2b_hazard_v1"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
REGISTRATION_POINTER = "4938726107"
SPEC_REGISTRATION = (
    "issue #42 comment 4938726107: gate-2b candidate 1, structural "
    "composition from the certified marital core"
)

#: Locked gate seeds and the K=20 draw stream (gate_2b protocol).
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
#: The single-draw provenance stream (candidates 1-9 lineage; recorded, not
#: used by the mean-over-K estimator). The registration names it "outer".
SIM_SEED_PROVENANCE_BASE = 4200
#: Bit-exact reproduction tolerance for the committed floor (gate-1/2a mirror).
EXACT_ATOL = 1e-12

PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.15-0.30",
    "modal_failure_classes_in_order": [
        "multigen stock cells (carried-state accumulation drift)",
        "coresident_grandchild family (unmodelled, composed-only)",
        "hh_size.5+ (tail composition)",
    ],
    "transitions_expected": "parental-home exit clears (directly fitted)",
    "primary_value_if_fail": (
        "the per-family decomposition that candidate 2 targets (the "
        "tranche-2a ladder method)"
    ),
    "grading_note": (
        "the orchestrator grades the forecast on #42; this run does NOT "
        "grade it."
    ),
}

SPEC_RESOLUTION_NOTES = {
    "rng_two_stream": (
        "The registration RNG clause reads 'default_rng(4200+seed) outer, "
        "5200+k draws, K=20'. The LOCKED gate_2b protocol (gates.yaml "
        "gate_2.gate_2b.thresholds.protocol.estimator / .candidate) defines "
        "the estimator as the mean over K=20 household-simulation draws at "
        "numpy.random.default_rng(5200 + k), k=0..19, and names no per-seed "
        "outer stream. Resolution (locked contract wins): each of the K=20 "
        "draws re-simulates the WHOLE composition keyed to draw seed 5200+k "
        "(the certified tranche-2a simulate uses default_rng(5200+k); the "
        "parental-home / multigen / paternal-shadow occupancy overlays use a "
        "distinct substream SeedSequence([5200+k, 0xB2B]) so the two "
        "subsystems never share draws yet the draw is a deterministic "
        "function of 5200+k). default_rng(4200+seed) is recorded as the "
        "single-draw provenance seed (the candidates 1-9 lineage the "
        "registration calls 'outer'); it does not drive the mean-over-K "
        "estimator, mirroring the certified tranche-2a candidate-10..16 runs "
        "(sim_seed_single_draw_provenance)."
    ),
    "observed_initial_states_are_the_holdout_persons_own": (
        "Registration component 2 uses 'OBSERVED initial states at window "
        "entry (the 2a initial-state lesson applied from the start)' and "
        "component 3 'carried initial states'. Resolution: each simulated "
        "side-A holdout person is seeded from their OWN observed "
        "coresident-parent / multigen state at their FIRST 2b wave, then "
        "evolved forward with the train-fitted hazards -- exactly as the "
        "certified tranche-2a simulate seeds each holdout person from their "
        "own observed initial marital state (family_transitions.simulate, "
        "entry_state from the holdout person-years). Using a holdout person's "
        "own window-entry state as the simulation's initial condition is not "
        "fitting (no parameter is estimated from side A); every fitted "
        "parameter -- the registry components, the parental-exit logistic, "
        "the multigen and child-attribution parameters -- is estimated on "
        "side B only. This is the reading the locked contract's estimand and "
        "the registration's explicit 'OBSERVED initial states' dictate; it "
        "supersedes any narrower paraphrase."
    ),
    "coresident_spouse_is_legal_marriage": (
        "Component 1 maps the certified registry's simulated 'married' state "
        "to a coresident spouse. The MX8 coresident-spouse flag also counts "
        "unmarried partners (codes 20/22), which the legal-marriage registry "
        "(mh85_23) does not carry; the resulting young-adult gap is the "
        "spouse-OR-partner concept delta the gate_2b external anchor names, "
        "reported not fixed (no partner model is in the registered spec)."
    ),
    "coresident_spouse_fallback": (
        "For side-A 2b waves the certified simulation does not cover "
        "(attrition-year gaps, ~7%; and persons with no marriage-file record "
        "at all), coresident_spouse carries the person's OWN observed "
        "first-wave state (the observed-initial lesson; it also recovers "
        "cohabitors the legal file misses). ~93% of side-A waves are covered "
        "by the certified simulation."
    ),
    "coresident_children_from_certified_kernel": (
        "Component 4 draws children from the CERTIFIED tranche-2a fertility "
        "kernel: women's own maternal births come from the registry "
        "simulate; married men are attributed a paternal shadow using the "
        "SAME kernel evaluated at the wife's imputed age (man age + the "
        "certified male spousal age gap) and the household parity. The "
        "certified kernel is female- and marriage-UNCONDITIONAL, so the "
        "paternal shadow structurally under-counts (married women "
        "out-reproduce the pooled kernel); this is a reported consequence of "
        "composing from the certified kernel, not a tuned choice. Children "
        "age out under the fitted parental-home-exit hazard at the child's "
        "age (sex drawn 0.5), open-topped at age 60."
    ),
    "household_size_composition": (
        "Component 5: hh_size = 1 + coresident_spouse + n_coresident_children "
        "+ (parent_count if coresident_parent else 0), where parent_count is "
        "the train-fitted rounded mean number of coresident parents (a "
        "two-parent home). It is the ego-centric FAMILY-UNIT size, narrower "
        "than the enumerated PSID HOUSEHOLD (it omits siblings, other "
        "relatives, roomers), so the upper tail is under-weighted -- the "
        "hh_size.5+ tail failure the forecast names. Never separately tuned."
    ),
    "coresident_grandchild_composed_only": (
        "Component 6: coresident_grandchild = multigen AND coresident_child "
        "AND NOT coresident_parent (the top generation of a three-generation "
        "household), a deterministic composed implication of the simulated "
        "states with no fitted parameter and no separate model, as the "
        "registration requires ('composed implication only')."
    ),
    "gates_yaml_path": (
        "The locked block is gates.yaml gates.gate_2.gate_2b.thresholds (the "
        "2b tranche nests under gate_2); its 46 gated tolerances and the "
        "floor gate_partition are read at runtime, never hardcoded."
    ),
}


# --------------------------------------------------------------------------
# Locked-threshold loading (read at runtime; no threshold hardcoded)
# --------------------------------------------------------------------------
def load_gate2b_thresholds() -> dict[str, Any]:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_2"]["gate_2b"]["thresholds"]


def gated_tolerances(thresholds: dict[str, Any]) -> dict[str, float]:
    tol: dict[str, float] = {}
    for view in thresholds["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars and non-finite floats for the file.

    A composed report-only cell can have a zero rate, whose ``|ln|`` score is
    ``inf``; strict JSON (and the committed-artifact convention) forbids the
    ``Infinity`` token, so a non-finite float serializes as ``null``. In-memory
    scores keep ``inf`` for the seed-conjunction logic; only the on-disk copy
    is sanitized.
    """
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
# Precheck: reproduce the committed floor bit-for-bit (gate-1/2a mirror)
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
            "Hard-stop precheck (gate-1/2a mirror): the scoring path must "
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

    model = hcs.fit_household_model(
        hh,
        data["mpanel"],
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b,
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]
    all_cells = sorted(set(tol) | set(report_only))
    draw_seeds = [DRAW_SEED_BASE + k for k in range(N_DRAWS)]
    per_draw_rate: dict[str, list[float]] = {c: [] for c in all_cells}
    per_draw_den: dict[str, list[float]] = {c: [] for c in all_cells}
    per_draw_nev: dict[str, list[int]] = {c: [] for c in all_cells}
    for draw_seed in draw_seeds:
        sim_panel = hcs.simulate_draw(
            hh, data["mpanel"], model, ids_a, draw_seed
        )
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
        "component_meta": model.meta,
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


def per_family_decomposition(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Per family: seed x cell pass rate, worst cell, |ln| vs tol, mechanism."""
    mechanisms = {
        "coresident_spouse": (
            "certified legal-marriage registry; the young-adult shortfall is "
            "the spouse-OR-partner (MX8 20/22) concept the registry omits."
        ),
        "coresident_parent": (
            "directly fitted logistic exit hazard from observed initial "
            "states; expected to clear."
        ),
        "coresident_child": (
            "certified fertility kernel; the male shortfall is the kernel's "
            "female/marriage-unconditional calibration under paternal shadow "
            "attribution, and elderly coresidence is the adult-child return "
            "the age-out model omits."
        ),
        "coresident_grandchild": (
            "composed implication only (multigen AND coresident_child AND NOT "
            "coresident_parent); inherits the multigen and child residuals."
        ),
        "multigen_stock": (
            "carried initial state + train band x sex entry/exit; the "
            "forecast's chief accumulation-drift class."
        ),
        "multigen_transition": (
            "directly fitted pooled entry/exit rates; expected to clear."
        ),
        "parental_home_exit": (
            "directly fitted; the transition the forecast expects to clear."
        ),
        "hh_size": (
            "composed ego-centric family unit (1 + spouse + children + "
            "parents); narrower than the enumerated household, so the 5+ tail "
            "is under-weighted."
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
        # worst cell by mean |ln|/tol across seeds.
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
            print(
                f"  {fam:22s} pass {d['cell_seed_pass_rate']:.2f}; worst "
                f"{d['worst_cell']} |ln|={d['worst_cell_mean_abs_ln']} "
                f"tol={d['worst_cell_tolerance']}"
            )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "candidate": "candidate 1",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "one_shot": (
            "Registered on issue #42 comment 4938726107 before this run; "
            "published REGARDLESS of verdict; artifacts.write_new refuses to "
            "overwrite an existing artifact."
        ),
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "description": (
                "Six-component structural household-composition generator "
                "composed from the certified tranche-2a marital core: "
                "coresident spouse (registry), coresident parent (fitted "
                "exit hazard), multigen (fitted entry/exit), coresident "
                "children (fertility kernel + paternal shadow, aging out), "
                "composed household size, composed grandchild."
            ),
            "family_transitions_spec": ft.CANDIDATE_16.candidate_id,
            "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
            "parental_exit_knots": list(hcs.PARENTAL_EXIT_KNOTS),
            "components": [
                "coresident_spouse<-certified_registry_married",
                "coresident_parent<-logistic_exit_hazard_age_spline_sex",
                "multigen<-train_band_sex_entry_exit_carried_initial",
                "coresident_child<-fertility_kernel_maternal+paternal_shadow",
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
                "numpy.random.default_rng(SeedSequence([5200 + k, 0xB2B]))"
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
