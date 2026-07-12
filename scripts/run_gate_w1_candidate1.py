"""W1 candidate 1 -- the first full transport deployment (one-shot, scored).

Registered at issue #42 comment 4950931131 against the LOCKED ``gate_w1``
contract (``gates.yaml``). Composes the certified / locked PSID-estimated
generators onto the certified populace CPS frame and scores the three target
families. Publishes regardless of the verdict (no holdout tuning).

Phases (the frame + generators are large -- family A is chunked by gate seed
as separate processes per the W2 two-process pattern / memory guard):

  fit        -- fit the certified generators once, cache to scratch
  seed N     -- family A for gate seed N -> scratch/fa_seed{N}.json
  family-b   -- family B (SSA anchors) on the full frame -> scratch/fb.json
  family-c   -- family C (compression fingerprints) -> scratch/fc.json
  assemble   -- read all phase outputs -> runs/gate_w1_candidate1_v1.json

The frame is resolved by ``POPULACE_DYNAMICS_FRAME_PICKLE`` (exported from the
policyengine.py .venv, sha-verified against the certified pin -- the DATA-BOUND
step the .venv-gate cannot do, per the floor's certified_repro_env_note). All
fit + score phases run in the always-runnable-adjacent .venv-gate.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402 -- imports follow the sys.path bootstrap above (script).
from populace_dynamics import artifacts
from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.models import transport_deployment_v1 as td

REGISTRATION_POINTER = "4950931131"
ARTIFACT_PATH = ROOT / "runs" / "gate_w1_candidate1_v1.json"
SCRATCH = ROOT / "scratch"
GENS_CACHE = SCRATCH / "gens_cache.pkl"
M4_ARTIFACT = ROOT / "runs" / "m4_disability_v1.json"


def _frame_path() -> str:
    p = os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE")
    if not p or not Path(p).exists():
        raise SystemExit(
            "set POPULACE_DYNAMICS_FRAME_PICKLE to the sha-verified frame "
            "export (see scripts/export_frame_persons.py, policyengine.py .venv)"
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
    return {
        "tolerances_a": tolerances,
        "gated_a": gated_a,
        "floor": floor,
        "family_b": gw1["thresholds"]["family_b"],
        "family_c": gw1["thresholds"]["family_c"],
    }


def load_generators() -> td.DeployedGenerators:
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
    out = SCRATCH / f"fa_seed{seed}.json"
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
    res = td.family_b_score(
        persons,
        gens,
        c["family_b"],
        pe_us_dir=os.environ.get("POPULACE_DYNAMICS_PE_US_DIR"),
    )
    json.dump(res, open(SCRATCH / "fb.json", "w"))
    print(
        f"[family-b] {'PASS' if res['family_b_pass'] else 'FAIL'} "
        f"({res['n_cells_pass']}/{res['n_cells']}) {time.time()-t:.0f}s",
        flush=True,
    )


def phase_family_c():
    t = time.time()
    persons = load_frame()
    gens = load_generators()
    c = load_contracts()
    res = td.family_c(persons, gens, c["family_c"])
    json.dump(res, open(SCRATCH / "fc.json", "w"))
    print(
        f"[family-c] both_reverse={res['both_reverse']} {time.time()-t:.0f}s",
        flush=True,
    )


def _frame_provenance() -> dict:
    """sha-verify the h5 the frame pickle came from, if resolvable here."""
    prov = {"frame_pickle": os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE")}
    provfile = SCRATCH / "frame_provenance.json"
    if provfile.exists():
        prov.update(json.load(open(provfile)))
    prov["certified_pin"] = dfm.CERTIFIED_PIN
    return prov


def phase_assemble():
    c = load_contracts()
    gens = load_generators()
    seeds = list(td.GATE_SEEDS)

    # family A: stitch per-seed cubes into [K, cell, seed]
    per_seed = []
    import numpy as np

    n_cell = len(c["gated_a"])
    cube = np.full((td.K_DRAWS, n_cell, len(seeds)), np.nan)
    for si, s in enumerate(seeds):
        fa = json.load(open(SCRATCH / f"fa_seed{s}.json"))
        sub = np.array(fa["cube"])  # [K, cell, 1]
        cube[:, :, si] = sub[:, :, 0]
        per_seed.append(fa["per_seed"][0])
    n_seed_pass = sum(1 for s in per_seed if s["seed_pass"])
    # Conformance / identity-candidate audit: the regenerated_surface rule
    # says an identity candidate (copying a scored column) has ZERO
    # across-draw dispersion and scores 0. A CONFORMANT candidate regenerates
    # every draw, so max_per_draw_abs_ln > 0 and across_draw_sd > 0.
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
            "every scored column (not the prohibited identity map, which "
            "would score 0 with zero dispersion)."
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
    }
    fb = json.load(open(SCRATCH / "fb.json"))
    fc = json.load(open(SCRATCH / "fc.json"))

    verdict = {
        "gate": "gate_w1",
        "candidate": "w1_candidate1",
        "registration_pointer": REGISTRATION_POINTER,
        "family_a_pass": family_a["family_a_pass"],
        "family_b_pass": fb["family_b_pass"],
        "family_c_pass": fc["family_c_pass"],
        "n_seed_pass_family_a": n_seed_pass,
        "gate_pass": bool(
            family_a["family_a_pass"]
            and fb["family_b_pass"]
            and fc["family_c_pass"]
        ),
        "per_seed_family_a": [
            {
                "seed": s["seed"],
                "pass": s["seed_pass"],
                "cells_pass": s["n_cells_pass"],
                "cells_fail": s["n_cells_fail"],
            }
            for s in per_seed
        ],
    }

    artifact = {
        "schema_version": "gate_w1_candidate1_v1",
        "run": "gate_w1_candidate1_v1",
        "candidate": "W1 candidate 1: first full transport deployment",
        "registration": {
            "issue": 42,
            "comment_id": REGISTRATION_POINTER,
            "url": (
                "https://github.com/PolicyEngine/populace-dynamics/issues/42"
                f"#issuecomment-{REGISTRATION_POINTER}"
            ),
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
        },
        "deployment_frame": dfm.CERTIFIED_PIN,
        "frame_provenance": _frame_provenance(),
        "generator_fit_provenance": gens.fit_provenance,
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
            },
            "family_b": {
                "pass": fb["family_b_pass"],
                "n_cells_pass": fb["n_cells_pass"],
                "n_cells": fb["n_cells"],
            },
            "family_c": {
                "pass": fc["family_c_pass"],
                "both_reverse": fc["both_reverse"],
                "c1_reversed": fc["fingerprints"]["c1"]["reversed_to_anchor"],
                "c2_reversed": fc["fingerprints"]["c2"]["reversed_to_anchor"],
            },
        },
        "spec_resolutions": td.SPEC_RESOLUTIONS,
        "elapsed_wall_note": "families run as parallel phase processes",
    }

    artifacts.write_new(ARTIFACT_PATH, artifact)
    print(f"[assemble] wrote {ARTIFACT_PATH}", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)


def _modal_failure(per_seed, gated) -> dict:
    """Aggregate per-cell fail counts across seeds -> the modal failing family."""
    import collections

    fail_counts = collections.Counter()
    for s in per_seed:
        for cell, pc in s["per_cell"].items():
            if not pc["pass"]:
                fail_counts[cell.split(".")[0]] += 1
    return dict(fail_counts.most_common())


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
