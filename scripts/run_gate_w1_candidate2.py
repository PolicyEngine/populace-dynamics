"""W1 candidate 2 -- the three proven levers (one-shot, scored).

Registered at issue #42 comment 4952253568 (FROZEN spec; the registration
wins) against the AMENDED ``gate_w1`` contract (``gates.yaml`` post-#165: 55
gated = 53 family-A joints + 2 family-C fingerprint reversals; family B fully
report-only). Composes the certified generators onto the certified populace
CPS frame with the three forensics-1 levers (Q1 entry-state seeding, Q2
boundary earnings, Q3 child attachment + coresidence) and scores the target
families. Publishes regardless of the verdict (no holdout tuning).

Phases (memory guard -- family A is chunked by gate seed as separate processes
per the candidate-1 pattern):

  fit        -- fit the v2 generators once, cache to scratch
  seed N     -- family A for gate seed N -> scratch/fa2_seed{N}.json
  family-b   -- family B (report-only vs retained_anchors) -> scratch/fb2.json
  family-c   -- family C (compression fingerprints) -> scratch/fc2.json
  assemble   -- read all phase outputs -> runs/gate_w1_candidate2_v1.json

The frame is resolved by ``POPULACE_DYNAMICS_FRAME_PICKLE`` (sha-verified
export). PSID at ``POPULACE_DYNAMICS_PSID_DIR``; the pe-us oracle at
``POPULACE_DYNAMICS_PE_US_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402 -- imports follow the sys.path bootstrap above (script).
from populace_dynamics import artifacts
from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.models import transport_deployment_v2 as td

REGISTRATION_POINTER = "4952253568"
CANDIDATE1_POINTER = "4950931131"
FORENSICS1_POINTER = "4951218279"
ARTIFACT_PATH = ROOT / "runs" / "gate_w1_candidate2_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate_w1_candidate1_v1.json"
SCRATCH = ROOT / "scratch"
GENS_CACHE = SCRATCH / "gens_cache_v2.pkl"
M4_ARTIFACT = ROOT / "runs" / "m4_disability_v1.json"


def _frame_path() -> str:
    p = os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE")
    if not p or not Path(p).exists():
        raise SystemExit(
            "set POPULACE_DYNAMICS_FRAME_PICKLE to the sha-verified frame "
            "export (scripts/export_frame_persons.py, policyengine.py .venv)"
        )
    return p


def load_frame() -> pd.DataFrame:
    return pd.read_pickle(_frame_path())


def load_contracts():
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gw1 = gates["gates"]["gate_w1"]
    fa = gw1["thresholds"]["family_a"]
    tolerances = {}
    for block in fa["views"].values():
        for cell, tol in block["tolerances"].items():
            tolerances[cell] = float(tol)
    floor = json.load(open(ROOT / "runs" / "gate_w1_floors_v1.json"))
    gated_a = list(floor["gate_partition"]["gate_eligible"])
    fb = gw1["thresholds"]["family_b"]
    return {
        "tolerances_a": tolerances,
        "gated_a": gated_a,
        "floor": floor,
        "family_b_gated_cells": fb.get("gated_cells", {}),
        "family_b_retained_anchors": fb["retained_anchors"],
        "family_b_report_reasons": fb["report_reasons"],
        "family_c": gw1["thresholds"]["family_c"],
    }


def load_generators() -> td.DeployedGeneratorsV2:
    if GENS_CACHE.exists():
        return pickle.load(open(GENS_CACHE, "rb"))
    gens = td.fit_generators(str(M4_ARTIFACT))
    SCRATCH.mkdir(exist_ok=True)
    pickle.dump(gens, open(GENS_CACHE, "wb"))
    return gens


def phase_fit():
    SCRATCH.mkdir(exist_ok=True)
    t = time.time()
    gens = td.fit_generators(str(M4_ARTIFACT))
    pickle.dump(gens, open(GENS_CACHE, "wb"))
    print(f"[fit] {time.time()-t:.1f}s -> {GENS_CACHE}", flush=True)
    print(json.dumps(gens.fit_provenance, indent=1), flush=True)


def phase_seed(seed: int):
    t = time.time()
    persons = load_frame()
    gens = load_generators()
    c = load_contracts()
    res = td.family_a_score(
        persons,
        gens,
        c["floor"],
        c["tolerances_a"],
        c["gated_a"],
        seeds=(seed,),
        k_draws=td.K_DRAWS,
        progress=True,
    )
    out = SCRATCH / f"fa2_seed{seed}.json"
    json.dump(res, open(out, "w"))
    s = res["per_seed"][0]
    print(
        f"[seed {seed}] {'PASS' if s['seed_pass'] else 'FAIL'} "
        f"({s['n_cells_pass']}/{len(c['gated_a'])}) {time.time()-t:.0f}s -> {out}",
        flush=True,
    )


def phase_family_b():
    t = time.time()
    persons = load_frame()
    gens = load_generators()
    c = load_contracts()
    res = td.family_b_report(
        persons,
        gens,
        c["family_b_retained_anchors"],
        c["family_b_report_reasons"],
        pe_us_dir=os.environ.get("POPULACE_DYNAMICS_PE_US_DIR"),
    )
    json.dump(res, open(SCRATCH / "fb2.json", "w"))
    print(
        f"[family-b] REPORT-ONLY {res['n_report_cells_within_tolerance']}"
        f"/{res['n_report_cells']} within tol (gates nothing) {time.time()-t:.0f}s",
        flush=True,
    )


def phase_family_c():
    t = time.time()
    persons = load_frame()
    gens = load_generators()
    c = load_contracts()
    res = td.family_c(persons, gens, c["family_c"])
    json.dump(res, open(SCRATCH / "fc2.json", "w"))
    print(
        f"[family-c] both_reverse={res['both_reverse']} "
        f"c1_rev={res['fingerprints']['c1']['reversed_to_anchor']} "
        f"c2_swap={res['fingerprints']['c2']['required_swap_realised']} "
        f"{time.time()-t:.0f}s",
        flush=True,
    )


def _frame_provenance() -> dict:
    prov = {"frame_pickle": os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE")}
    provfile = SCRATCH / "frame_provenance.json"
    if provfile.exists():
        prov.update(json.load(open(provfile)))
    prov["certified_pin"] = dfm.CERTIFIED_PIN
    return prov


def _byte_carry_regression(per_seed, gated_a) -> dict:
    """Regression vs candidate 1: which gated family-A cells are byte-carried
    (untouched by the three deltas) and whether v2 reproduces c1 on them."""
    c1 = json.load(open(CANDIDATE1_ARTIFACT))
    c1_ps = {s["seed"]: s["per_cell"] for s in c1["family_a"]["per_seed"]}
    # Cells the deltas do NOT touch: earnings participation/profile/dispersion
    # for the in-support 25-54 bands (base marginals + preserved u draw). The
    # boundary (18-24/55-61/62-69) earnings, all marital/coresident, and
    # hh_size are delta-touched.
    delta_prefixes = ("marital_share.", "coresident_spouse.", "hh_size_share.")
    boundary_tokens = ("18-24", "55-61", "62-69")
    carried = []
    for cell in gated_a:
        if cell.startswith(delta_prefixes):
            continue
        if cell.startswith("earnings_") and any(
            tok in cell for tok in boundary_tokens
        ):
            continue
        carried.append(cell)
    max_dev = 0.0
    worst = None
    checked = 0
    for s in per_seed:
        seed = s["seed"]
        for cell in carried:
            if cell in s["per_cell"] and cell in c1_ps[seed]:
                dev = abs(
                    s["per_cell"][cell]["rbar"] - c1_ps[seed][cell]["rbar"]
                )
                checked += 1
                if dev > max_dev:
                    max_dev = dev
                    worst = {"cell": cell, "seed": seed, "abs_dev": dev}
    return {
        "carried_cells": carried,
        "n_carried_cells": len(carried),
        "n_comparisons": checked,
        "max_abs_rbar_deviation_vs_candidate1": max_dev,
        "bit_identical": bool(max_dev <= 1e-9),
        "worst": worst,
        "note": (
            "non-boundary in-support earnings cells (25-54) are byte-carried "
            "from candidate 1 -- same base marginals, same preserved u draw. "
            "A near-zero max deviation confirms the deltas are surgical."
        ),
    }


def phase_assemble():
    c = load_contracts()
    gens = load_generators()
    seeds = list(td.GATE_SEEDS)

    per_seed = []
    n_cell = len(c["gated_a"])
    cube = np.full((td.K_DRAWS, n_cell, len(seeds)), np.nan)
    for si, s in enumerate(seeds):
        fa = json.load(open(SCRATCH / f"fa2_seed{s}.json"))
        sub = np.array(fa["cube"])
        cube[:, :, si] = sub[:, :, 0]
        per_seed.append(fa["per_seed"][0])
    n_seed_pass = sum(1 for s in per_seed if s["seed_pass"])

    max_sd = 0.0
    max_absln = 0.0
    for s in per_seed:
        for pcv in s["per_cell"].values():
            max_sd = max(max_sd, pcv["across_draw_sd"])
            max_absln = max(max_absln, pcv["max_per_draw_abs_ln"])
    conformance = {
        "regenerated_surface": True,
        "identity_candidate": False,
        "max_across_draw_sd": max_sd,
        "max_per_draw_abs_ln": max_absln,
        "note": (
            "non-zero across-draw dispersion => the candidate REGENERATES "
            "every scored column (the seeded marital/earnings/household draws "
            "vary per k), not the prohibited identity map."
        ),
    }
    family_a = {
        "pricing": "100-seed household-disjoint half-split floor",
        "gated_cells": c["gated_a"],
        "cube_shape": [td.K_DRAWS, n_cell, len(seeds)],
        "cube_dims": "K_draws x gated_cells x gate_seeds",
        "cube": cube.tolist(),
        "conformance": conformance,
        "per_seed": per_seed,
        "n_seed_pass": n_seed_pass,
        "family_a_pass": n_seed_pass >= 4,
        "byte_carry_regression_vs_candidate1": _byte_carry_regression(
            per_seed, c["gated_a"]
        ),
    }
    fb = json.load(open(SCRATCH / "fb2.json"))
    fc = json.load(open(SCRATCH / "fc2.json"))

    # amended gate: family A AND family C (family B report-only, gates nothing)
    gate_pass = bool(family_a["family_a_pass"] and fc["family_c_pass"])
    verdict = {
        "gate": "gate_w1",
        "candidate": "w1_candidate2",
        "registration_pointer": REGISTRATION_POINTER,
        "amended_contract": "gates.yaml post-#165 (55 gated = 53 family-A + 2 family-C; family B report-only)",
        "gate_rule": "family A (>= 4/5 seeds) AND family C (both fingerprints reverse); family B contributes nothing",
        "family_a_pass": family_a["family_a_pass"],
        "family_b_pass": None,
        "family_b_gates": False,
        "family_c_pass": fc["family_c_pass"],
        "n_seed_pass_family_a": n_seed_pass,
        "gate_pass": gate_pass,
        "per_seed_family_a": [
            {
                "seed": s["seed"],
                "pass": s["seed_pass"],
                "cells_pass": s["n_cells_pass"],
                "cells_fail": s["n_cells_fail"],
            }
            for s in per_seed
        ],
        "fingerprints": {
            "c1_ppi_nra": {
                "reversed_to_anchor": fc["fingerprints"]["c1"][
                    "reversed_to_anchor"
                ],
                "kendall_tau_vs_required": fc["fingerprints"]["c1"][
                    "kendall_tau_vs_required"
                ],
                "required_swap_realised": fc["fingerprints"]["c1"][
                    "required_swap_realised"
                ],
            },
            "c2_elimination_plus2pp": {
                "reversed_to_anchor": fc["fingerprints"]["c2"][
                    "reversed_to_anchor"
                ],
                "kendall_tau_vs_required": fc["fingerprints"]["c2"][
                    "kendall_tau_vs_required"
                ],
                "required_swap_realised": fc["fingerprints"]["c2"][
                    "required_swap_realised"
                ],
            },
        },
    }

    artifact = {
        "schema_version": "gate_w1_candidate2_v1",
        "run": "gate_w1_candidate2_v1",
        "candidate": "W1 candidate 2: the three proven levers",
        "registration": {
            "issue": 42,
            "comment_id": REGISTRATION_POINTER,
            "url": (
                "https://github.com/PolicyEngine/populace-dynamics/issues/42"
                f"#issuecomment-{REGISTRATION_POINTER}"
            ),
        },
        "lineage": {
            "candidate1": CANDIDATE1_POINTER,
            "candidate1_artifact": "runs/gate_w1_candidate1_v1.json",
            "forensics1": FORENSICS1_POINTER,
            "forensics1_artifact": "runs/gate_w1_forensics1_v1.json",
        },
        "gate": "gate_w1",
        "protocol": {
            "one_shot": True,
            "publishes_regardless": True,
            "no_holdout_tuning": True,
            "k_draws": td.K_DRAWS,
            "gate_seeds": seeds,
            "family_a_draw_stream_base": td.FAMILY_A_STREAM_BASE,
            "family_b_draw_stream_base": td.FAMILY_B_STREAM_BASE,
            "family_c_transitory_stream": td.FAMILY_C_TRANSITORY_STREAM,
            "deltas_vs_candidate1": [
                "Q1 entry-state seeding (marital + coresident)",
                "Q2 boundary earnings extension (18-24/60-69 + sex covariate)",
                "Q3 child attachment + coresidence repair through the seeding",
            ],
        },
        "deployment_frame": dfm.CERTIFIED_PIN,
        "frame_provenance": _frame_provenance(),
        "generator_fit_provenance": gens.fit_provenance,
        "fit_vs_raw": gens.fit_vs_raw,
        "verdict": verdict,
        "family_a": family_a,
        "family_b": fb,
        "family_c": fc,
        "decomposition": {
            "family_a": {
                "pass": family_a["family_a_pass"],
                "n_seed_pass": n_seed_pass,
                "n_gated_cells": n_cell,
                "modal_failure": _modal_failure(per_seed, c["gated_a"]),
                "c1_to_c2_progression": _progression(per_seed),
            },
            "family_b": {
                "reported_not_gated": True,
                "n_report_cells": fb["n_report_cells"],
                "n_report_cells_within_tolerance": fb[
                    "n_report_cells_within_tolerance"
                ],
                "note": "family B gates nothing after amendment 1; disclosed for completeness",
            },
            "family_c": {
                "pass": fc["family_c_pass"],
                "both_reverse": fc["both_reverse"],
                "c1_reversed": fc["fingerprints"]["c1"]["reversed_to_anchor"],
                "c2_reversed": fc["fingerprints"]["c2"]["reversed_to_anchor"],
                "c2_swap_realised": fc["fingerprints"]["c2"][
                    "required_swap_realised"
                ],
            },
        },
        "spec_resolutions": td.SPEC_RESOLUTIONS,
        "elapsed_wall_note": "families run as parallel phase processes",
    }

    artifacts.write_new(ARTIFACT_PATH, artifact, sidecar=True)
    print(f"[assemble] wrote {ARTIFACT_PATH} (+ sidecar)", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)


def _modal_failure(per_seed, gated) -> dict:
    import collections

    fail_counts = collections.Counter()
    for s in per_seed:
        for cell, pc in s["per_cell"].items():
            if not pc["pass"]:
                fail_counts[cell.split(".")[0]] += 1
    return dict(fail_counts.most_common())


def _progression(per_seed) -> dict:
    """Per-family c1 -> c2 progression on the gated family-A cells."""
    c1 = json.load(open(CANDIDATE1_ARTIFACT))
    c1_ps = {s["seed"]: s["per_cell"] for s in c1["family_a"]["per_seed"]}
    fams = (
        "earnings_participation",
        "earnings_profile",
        "earnings_p90p50",
        "earnings_p50p10",
        "marital_share",
        "hh_size_share",
        "coresident_spouse",
    )
    out = {}
    for fam in fams:
        c1_pass = c2_pass = total = 0
        for s in per_seed:
            for cell, pc in s["per_cell"].items():
                if cell.split(".")[0] != fam:
                    continue
                total += 1
                c2_pass += int(pc["pass"])
                c1_pass += int(c1_ps[s["seed"]][cell]["pass"])
        out[fam] = {
            "cell_seed_slots": total,
            "candidate1_pass": c1_pass,
            "candidate2_pass": c2_pass,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "phase", choices=["fit", "seed", "family-b", "family-c", "assemble"]
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "fit":
        phase_fit()
    elif args.phase == "seed":
        phase_seed(args.seed)
    elif args.phase == "family-b":
        phase_family_b()
    elif args.phase == "family-c":
        phase_family_c()
    elif args.phase == "assemble":
        phase_assemble()


if __name__ == "__main__":
    main()
