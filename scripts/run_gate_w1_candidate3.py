"""W1 candidate 3 -- the three forensics-2 levers (one-shot, BUILD ONLY here).

Composes the certified generators onto the certified populace CPS frame with
the three forensics-2 levers (lever 1 CPS-anchored entry, lever 2 interior sex
covariate, lever 3 co-designed roster/fertility) and scores the target
families. The gated PARTITION is bound from ``gates.yaml`` at RUN time -- the
family-A gated cells are ``union(view tolerances) - report_only`` and the gated
family-C fingerprints are ``family_c.gate_partition.gate_eligible`` -- so the
candidate automatically scores whatever surface is locked when its registered
one-shot runs. Before the amendment-2 flip that is 53 family-A joints + 2
family-C fingerprints; after the flip (which demotes the 18-24 pair, the 65+
quad, and C1) it is 47 family-A joints + the C2 fingerprint, with NO code change.

Registration: the scored run is registered on issue #42 AFTER amendment 2
ratifies. This script is committed BUILD ONLY -- it has never touched the pinned
frame. ``main()`` refuses to run once the output artifact exists (write_new
one-shot semantics).

Phases (memory guard -- family A is chunked by gate seed as separate processes
per the candidate-1/2 pattern):

  fit        -- fit the v3 generators once, cache to scratch
  seed N     -- family A for gate seed N -> scratch/fa3_seed{N}.json
  family-b   -- family B (report-only vs retained_anchors) -> scratch/fb3.json
  family-c   -- family C (compression fingerprints)        -> scratch/fc3.json
  assemble   -- read all phase outputs -> runs/gate_w1_candidate3_v1.json

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
from populace_dynamics.models import transport_deployment_v3 as td

# Candidate 3 is registered on issue #42 only AFTER amendment 2 ratifies; the
# scored-run comment id does not exist yet, so it is left None here (BUILD ONLY).
REGISTRATION_POINTER = None
CANDIDATE2_POINTER = "4952253568"
FORENSICS2_POINTER = "4953088871"
AMENDMENT2_DOC = (
    "docs/amendments/gate_w1_amendment_2_family_a_concept_cells.md"
)
ARTIFACT_PATH = ROOT / "runs" / "gate_w1_candidate3_v1.json"
SCRATCH = ROOT / "scratch"
GENS_CACHE = SCRATCH / "gens_cache_v3.pkl"
M4_ARTIFACT = ROOT / "runs" / "m4_disability_v1.json"
FLOOR_ARTIFACT = ROOT / "runs" / "gate_w1_floors_v1.json"


def _refuse_if_artifact_exists() -> None:
    """Refuse the one-shot once the committed artifact (or sidecar) exists.

    Mirrors ``artifacts.write_new``: a completed one-shot must never be
    re-run. This guard fires for EVERY phase, before any frame is touched, so
    an accidental re-run cannot peek at the pinned frame.
    """
    for target in (ARTIFACT_PATH, Path(f"{ARTIFACT_PATH}.env.json")):
        if os.path.lexists(target):
            raise FileExistsError(
                f"{target} already exists; the candidate-3 one-shot is "
                "complete -- refusing to re-run (write_new semantics)"
            )


# --------------------------------------------------------------------------
# gates.yaml-bound partition (pure helpers -- unit-tested on synthetic blocks).
# --------------------------------------------------------------------------
def family_a_partition(
    family_a_block: dict,
) -> tuple[dict[str, float], list[str]]:
    """Bind the family-A tolerances and GATED cell list from a gates.yaml block.

    The gated surface is ``union(view tolerances) - report_only`` -- so the
    amendment-2 flip (which adds the demoted cells to ``report_only``) is picked
    up automatically. Every gated cell is guaranteed a tolerance (it is drawn
    from the tolerance union), so scoring never KeyErrors on the locked surface.
    """
    tolerances: dict[str, float] = {}
    for view in family_a_block["views"].values():
        for cell, tol in view["tolerances"].items():
            tolerances[cell] = float(tol)
    report_only = set(family_a_block.get("report_only", []))
    gated = [cell for cell in tolerances if cell not in report_only]
    return tolerances, gated


def family_c_gated_fingerprints(family_c_block: dict) -> list[str]:
    """Bind the GATED family-C fingerprint ids (c1/c2) from a gates.yaml block.

    Maps ``family_c.gate_partition.gate_eligible`` (``fingerprint.<id>`` keys)
    back to the fingerprint spec keys (``c1``/``c2``) via each spec's ``id``.
    After the amendment-2 flip only ``fingerprint.elimination_plus2pp`` (C2)
    remains, so this returns ``["c2"]`` and C1's non-reversal no longer gates.
    """
    gate_eligible = set(
        family_c_block.get("gate_partition", {}).get("gate_eligible", [])
    )
    gated: list[str] = []
    for fid, spec in family_c_block["fingerprints"].items():
        if f"fingerprint.{spec['id']}" in gate_eligible:
            gated.append(fid)
    return gated


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


def load_contracts() -> dict:
    """Bind the whole contract from gates.yaml at RUN time (+ the committed floor
    for rate_a pricing)."""
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gw1 = gates["gates"]["gate_w1"]["thresholds"]
    tolerances_a, gated_a = family_a_partition(gw1["family_a"])
    floor = json.load(open(FLOOR_ARTIFACT))
    fb = gw1["family_b"]
    fc = gw1["family_c"]
    return {
        "tolerances_a": tolerances_a,
        "gated_a": gated_a,
        "floor": floor,
        "family_b_retained_anchors": fb["retained_anchors"],
        "family_b_report_reasons": fb["report_reasons"],
        "family_c": fc,
        "family_c_gated_fids": family_c_gated_fingerprints(fc),
    }


def load_generators() -> td.DeployedGeneratorsV3:
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
    out = SCRATCH / f"fa3_seed{seed}.json"
    json.dump(res, open(out, "w"))
    s = res["per_seed"][0]
    print(
        f"[seed {seed}] {'PASS' if s['seed_pass'] else 'FAIL'} "
        f"({s['n_cells_pass']}/{len(c['gated_a'])}) {time.time()-t:.0f}s "
        f"-> {out}",
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
    json.dump(res, open(SCRATCH / "fb3.json", "w"))
    print(
        f"[family-b] REPORT-ONLY {res['n_report_cells_within_tolerance']}"
        f"/{res['n_report_cells']} within tol (gates nothing) "
        f"{time.time()-t:.0f}s",
        flush=True,
    )


def phase_family_c():
    t = time.time()
    persons = load_frame()
    gens = load_generators()
    c = load_contracts()
    res = td.family_c(persons, gens, c["family_c"])
    res["gated_fingerprints"] = c["family_c_gated_fids"]
    json.dump(res, open(SCRATCH / "fc3.json", "w"))
    print(
        f"[family-c] gated={c['family_c_gated_fids']} "
        f"c1_rev={res['fingerprints']['c1']['reversed_to_anchor']} "
        f"c2_rev={res['fingerprints']['c2']['reversed_to_anchor']} "
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


def _family_c_pass(fc: dict, gated_fids: list[str]) -> bool:
    """Family C passes iff every GATED fingerprint reverses to its anchor."""
    return bool(
        gated_fids
        and all(
            fc["fingerprints"][fid]["reversed_to_anchor"] for fid in gated_fids
        )
    )


def phase_assemble():
    c = load_contracts()
    gens = load_generators()
    seeds = list(td.GATE_SEEDS)
    gated_a = c["gated_a"]
    gated_fids = c["family_c_gated_fids"]

    per_seed = []
    n_cell = len(gated_a)
    cube = np.full((td.K_DRAWS, n_cell, len(seeds)), np.nan)
    for si, s in enumerate(seeds):
        fa = json.load(open(SCRATCH / f"fa3_seed{s}.json"))
        sub = np.array(fa["cube"])
        cube[:, :, si] = sub[:, :, 0]
        per_seed.append(fa["per_seed"][0])
    n_seed_pass = sum(1 for s in per_seed if s["seed_pass"])
    entry_anchor = json.load(open(SCRATCH / f"fa3_seed{seeds[0]}.json")).get(
        "cps_entry_anchor"
    )

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
            "non-zero across-draw dispersion => the candidate REGENERATES every "
            "scored column (the CPS-anchored marital seed, the interior "
            "sex-covariate earnings draw, and the co-designed roster/fertility "
            "household draw all vary per k), not the prohibited identity map."
        ),
    }
    family_a = {
        "pricing": "100-seed household-disjoint half-split floor",
        "gated_cells": gated_a,
        "n_gated_cells": n_cell,
        "cps_entry_anchor": entry_anchor,
        "cube_shape": [td.K_DRAWS, n_cell, len(seeds)],
        "cube_dims": "K_draws x gated_cells x gate_seeds",
        "cube": cube.tolist(),
        "conformance": conformance,
        "per_seed": per_seed,
        "n_seed_pass": n_seed_pass,
        "family_a_pass": n_seed_pass >= 4,
    }
    fb = json.load(open(SCRATCH / "fb3.json"))
    fc = json.load(open(SCRATCH / "fc3.json"))
    family_c_pass = _family_c_pass(fc, gated_fids)

    gate_pass = bool(family_a["family_a_pass"] and family_c_pass)
    verdict = {
        "gate": "gate_w1",
        "candidate": "w1_candidate3",
        "registration_pointer": REGISTRATION_POINTER,
        "registration_note": (
            "registered on issue #42 AFTER amendment 2 ratifies; this artifact "
            "is BUILD ONLY until then"
        ),
        "partition_binding": {
            "source": "gates.yaml (bound at run time)",
            "family_a_gated_cells": n_cell,
            "family_a_rule": "union(view tolerances) - report_only",
            "family_c_gated_fingerprints": gated_fids,
            "family_c_rule": "family_c.gate_partition.gate_eligible",
            "note": (
                "the amendment-2 flip (demote 18-24 pair + 65+ quad + C1) is "
                "picked up automatically -- no code change; pre-flip this scores "
                "53 family-A + {c1,c2}, post-flip 47 family-A + {c2}"
            ),
        },
        "gate_rule": (
            "family A (>= 4/5 seeds) AND family C (every GATED fingerprint "
            "reverses); family B contributes nothing"
        ),
        "family_a_pass": family_a["family_a_pass"],
        "family_b_pass": None,
        "family_b_gates": False,
        "family_c_pass": family_c_pass,
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
            fid: {
                "gated": fid in gated_fids,
                "reversed_to_anchor": fc["fingerprints"][fid][
                    "reversed_to_anchor"
                ],
                "kendall_tau_vs_required": fc["fingerprints"][fid][
                    "kendall_tau_vs_required"
                ],
                "required_swap_realised": fc["fingerprints"][fid][
                    "required_swap_realised"
                ],
            }
            for fid in ("c1", "c2")
        },
    }

    artifact = {
        "schema_version": "gate_w1_candidate3_v1",
        "run": "gate_w1_candidate3_v1",
        "candidate": "W1 candidate 3: CPS-anchored entry + interior sex "
        "covariate + co-designed roster/fertility",
        "registration": {
            "issue": 42,
            "comment_id": REGISTRATION_POINTER,
            "status": "pending amendment-2 ratification (build only)",
        },
        "lineage": {
            "candidate2": CANDIDATE2_POINTER,
            "candidate2_artifact": "runs/gate_w1_candidate2_v1.json",
            "forensics2": FORENSICS2_POINTER,
            "forensics2_artifact": "runs/gate_w1_forensics2_v1.json",
            "amendment2_doc": AMENDMENT2_DOC,
        },
        "gate": "gate_w1",
        "protocol": {
            "one_shot": True,
            "publishes_regardless": True,
            "no_holdout_tuning": True,
            "k_draws": td.K_DRAWS,
            "estimator": "mean over K=20 draws",
            "gate_seeds": seeds,
            "family_a_draw_stream_base": td.FAMILY_A_STREAM_BASE,
            "family_b_draw_stream_base": td.FAMILY_B_STREAM_BASE,
            "family_c_transitory_stream": td.FAMILY_C_TRANSITORY_STREAM,
            "gated_partition_bound_from": "gates.yaml at run time",
            "levers_vs_candidate2": [
                "lever 1 CPS-anchored entry-level marital model (Q6)",
                "lever 2 interior sex covariate on the earnings marginals (Q8)",
                "lever 3 co-designed coresident_parent roster + fertility "
                "window (Q7)",
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
                "modal_failure": _modal_failure(per_seed),
            },
            "family_b": {
                "reported_not_gated": True,
                "n_report_cells": fb["n_report_cells"],
                "n_report_cells_within_tolerance": fb[
                    "n_report_cells_within_tolerance"
                ],
                "note": (
                    "family B gates nothing after amendment 1; disclosed for "
                    "completeness"
                ),
            },
            "family_c": {
                "pass": family_c_pass,
                "gated_fingerprints": gated_fids,
                "c1_reversed": fc["fingerprints"]["c1"]["reversed_to_anchor"],
                "c2_reversed": fc["fingerprints"]["c2"]["reversed_to_anchor"],
            },
        },
        "spec_resolutions": td.SPEC_RESOLUTIONS,
        "elapsed_wall_note": "families run as parallel phase processes",
    }

    artifacts.write_new(ARTIFACT_PATH, artifact, sidecar=True)
    print(f"[assemble] wrote {ARTIFACT_PATH} (+ sidecar)", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)


def _modal_failure(per_seed) -> dict:
    import collections

    fail_counts = collections.Counter()
    for s in per_seed:
        for cell, pc in s["per_cell"].items():
            if not pc["pass"]:
                fail_counts[cell.split(".")[0]] += 1
    return dict(fail_counts.most_common())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "phase", choices=["fit", "seed", "family-b", "family-c", "assemble"]
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    # One-shot guard: refuse EVERY phase once the artifact exists, before any
    # frame is touched (an accidental re-run must never peek at the frame).
    _refuse_if_artifact_exists()
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
