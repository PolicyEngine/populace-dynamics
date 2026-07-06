"""Build the PIA-proxy anchor floor: real-vs-real at deployment scale.

This artifact is the empirical anchor the gate-1 amendment proposal
(:mod:`gates.yaml`, ``gate_1.thresholds.amendment_proposed``) cites for
the KS threshold of the proposed ``benefit_space`` block. It answers one
question: how far apart do two independent real PSID samples look in
PIA-proxy space, at the deployment scale a Social Security analysis
consumes? Every proposed benefit-space band is a claim about candidate
gaps against THIS real-vs-real floor.

REPORTED ANCHOR, NOT A GATE RUN. This reads no gate and, on its own,
decides nothing; it is committed evidence pinned by a reproduction test,
exactly like the other floor artifacts in ``runs/``. The amendment that
would put it to work is a separate object under the pre-registration
contract and changes nothing until a fresh referee round runs and the
maintainer ratifies by merging.

Construction (per seed s in 0-4), on the LOCKED filtered panel (the full
prime-age 25-59, reference-year-1998-2022, positive-weight family panel --
NOT a train split; this is a deployment-scale floor):

* ctx20 real-vs-real, exactly the construction in
  :mod:`scripts.build_gate1_floor_artifacts`: draw 40% of persons
  (``split_panel_by_person``, fraction=0.4, seed=1000+s), halve it
  person-disjointly (fraction=0.5, seed=s), giving two DISJOINT real
  samples A and B at ~20%-of-persons scale.
* Push both A's and B's persons through the pinned PIA-proxy functional
  from the merged :mod:`scripts.build_downstream_relevance` -- the SAME
  ``person_pia_proxy`` (wage-base cap, NAWI-to-2022 indexing, top
  ``min(10, n)`` mean over observed positives, 2022 415(a)/415(g) PIA),
  zero-positive persons included at 0, each person weighted by their
  anchor-period weight.
* Measure the real-vs-real distribution gap of the two PIA-proxy samples:
  ``|mean % gap|``, ``|median % gap|``, per-decile % gaps, and the
  weighted KS distance; and, on the Q0 (zero-anchor-earnings) subgroup of
  each side, the real-vs-real ``mean % gap`` -- the floor for the group
  candidate 7 misses by +9.3%. Q0 membership uses the seed-stable
  full-panel anchor-earnings quintile edges (Q0 = zero anchor earnings).

Per seed and the across-seed mean/sd of each statistic are recorded. The
KS ``noise_floor_seeds_0_4`` block follows the committed-floor convention
(``mean``/``sd``/``values``) so the machine-checkable KS threshold in
``gates.yaml`` derives from it exactly as
``tests/test_gates_derivations.py`` asserts.

The proxy functional is imported from
:mod:`scripts.build_downstream_relevance` so it is byte-for-byte the
merged, pinned functional -- single source of truth. NO candidate is
generated here (this floor is real-vs-real), so the oracle import path
does NOT pull ``populace.fit``; the reproduction test needs only PSID.

Oracle parameters (NAWI, wage base, PIA bend points and factors) load
ONCE from the pinned policyengine-us checkout; the recorded revision is
written into the artifact. Point the loader at the pin and run from the
repository root with the PSID family files staged:

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/build_pia_proxy_floor.py

(The dedicated gate venv is used only for parity with the other builders;
this script itself needs no populace-fit and runs equally in the repo
``.venv``.)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Imports below reuse the merged, pinned pieces so this floor shares one
# identical functional with the candidate measurements:
# - build_downstream_relevance: the pinned PIA-proxy functional and its
#   weighted-statistic helpers, verbatim (single source of truth).
# - run_gate1_baseline / run_gate1_candidate5b: the locked filter-first
#   load, SEEDS, and the anchor definition.
# The floor is REAL-vs-real, so NONE of the candidate-generation machinery
# is imported and the participation gate (the sole populace-fit dependency
# in the gate runners) is never touched; every populace-fit import in the
# gate chain is lazy, inside candidate-fitting functions this floor never
# calls, so importing these modules does NOT load populace.fit.
from build_downstream_relevance import (  # noqa: E402
    DECILES,
    SUCCESS_CRITERION_PCT,
    SUCCESS_CRITERION_TEXT,
    _assign_quintiles,
    anchor_quintile_cutpoints,
    distribution_gaps,
    panel_pia_proxy,
)
from run_gate1_baseline import (  # noqa: E402
    SEEDS,
    load_filtered_panel,
)
from run_gate1_candidate5b import anchor_rows  # noqa: E402

from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "pia_proxy_floor_9822.json"
ARTIFACT_SCHEMA_VERSION = "pia_proxy_floor.v1"

#: The zero-anchor-earnings subgroup index (Q0) in the seed-stable
#: full-panel anchor-earnings quintiles.
Q0_INDEX = 0


# --------------------------------------------------------------------------
# Per-seed real-vs-real measurement
# --------------------------------------------------------------------------
def measure_seed(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    cutpoints: np.ndarray,
    params: Any,
    verbose: bool,
) -> dict[str, Any]:
    """One seed of the real-vs-real PIA-proxy floor on the full panel.

    ctx20 on the full filtered panel: draw 40% of persons
    (``split_panel_by_person``, fraction=0.4, seed=1000+s), halve it
    person-disjointly (fraction=0.5, seed=s), push both disjoint halves
    through the pinned proxy, and report the distribution gap of the two
    PIA-proxy samples plus the Q0 (zero-anchor-earnings) subgroup mean %
    gap. The two halves are disjoint persons, so only distribution-gap
    measures apply (no person-level alignment).
    """
    t0 = time.time()
    forty, _ = hpanel.split_panel_by_person(
        panel, "person_id", fraction=0.4, seed=1000 + seed
    )
    side_a, side_b = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=seed
    )

    a_px = panel_pia_proxy(side_a, all_anchor, params)
    b_px = panel_pia_proxy(side_b, all_anchor, params)

    a = a_px["pia_proxy"].to_numpy(dtype=np.float64)
    aw = a_px["weight"].to_numpy(dtype=np.float64)
    b = b_px["pia_proxy"].to_numpy(dtype=np.float64)
    bw = b_px["weight"].to_numpy(dtype=np.float64)

    overall = distribution_gaps(a, aw, b, bw)

    # Q0 subgroup: zero-anchor-earnings persons on each disjoint side,
    # membership fixed by the seed-stable full-panel quintile edges. Real
    # side A's Q0 persons vs real side B's Q0 persons -- the real-vs-real
    # floor for the group candidate 7 misses (its +9.3% Q0 mean % gap).
    q_of_a = _assign_quintiles(
        all_anchor, a_px["person_id"].to_numpy(), cutpoints
    )
    q_of_b = _assign_quintiles(
        all_anchor, b_px["person_id"].to_numpy(), cutpoints
    )
    a_q0 = a_px[[q_of_a[int(p)] == Q0_INDEX for p in a_px["person_id"]]]
    b_q0 = b_px[[q_of_b[int(p)] == Q0_INDEX for p in b_px["person_id"]]]
    q0_gap = distribution_gaps(
        a_q0["pia_proxy"].to_numpy(dtype=np.float64),
        a_q0["weight"].to_numpy(dtype=np.float64),
        b_q0["pia_proxy"].to_numpy(dtype=np.float64),
        b_q0["weight"].to_numpy(dtype=np.float64),
    )

    result = {
        "seed": seed,
        "construction": (
            "ctx20 on the full filtered panel: "
            "split_panel_by_person(panel, 'person_id', fraction=0.4, "
            "seed=1000+s) then split_panel_by_person(forty, 'person_id', "
            "fraction=0.5, seed=s); side A scored against side B"
        ),
        "n_persons_side_a": int(len(a_px)),
        "n_persons_side_b": int(len(b_px)),
        "n_persons_q0_side_a": int(len(a_q0)),
        "n_persons_q0_side_b": int(len(b_q0)),
        "distribution": overall,
        "q0_zero_anchor": {
            "definition": (
                "zero anchor earnings (no earnings in the person's last "
                "observed period); real side-A Q0 persons vs real side-B "
                "Q0 persons -- the real-vs-real floor for the subgroup "
                "candidate 7 misses by +9.3%"
            ),
            "mean_pct_diff": q0_gap["mean"]["pct_diff"],
            "median_pct_diff": q0_gap["median"]["pct_diff"],
            "ks_distance": q0_gap["ks_distance"],
        },
    }
    if verbose:
        d = result["distribution"]
        q0 = result["q0_zero_anchor"]
        print(
            f"  seed {seed}: mean%={d['mean']['pct_diff']:+.3f} "
            f"median%={d['median']['pct_diff']:+.3f} "
            f"KS={d['ks_distance']:.4f} "
            f"Q0_mean%={q0['mean_pct_diff']:+.3f} "
            f"(A={len(a_px)},B={len(b_px)}; "
            f"Q0 A={len(a_q0)},B={len(b_q0)}) "
            f"({time.time() - t0:.0f}s)"
        )
    return result


# --------------------------------------------------------------------------
# Across-seed pooling (mean/sd of the anchor statistics)
# --------------------------------------------------------------------------
def _summary(values: list[float]) -> dict[str, Any]:
    """Mean/sd/min/max and the raw per-seed values (float64, ddof=1 sd)."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
        "values": [float(v) for v in arr],
    }


def pool_floor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Across-seed floor: mean/sd of each real-vs-real anchor statistic.

    The spec records ``|mean % gap|``, ``|median % gap|``, per-decile %
    gaps, weighted KS, and the Q0 mean % gap. Percent-gap magnitudes are
    reported as ABSOLUTE values (``abs_...``) -- the amendment's bands are
    on absolute magnitude -- alongside the signed pool for transparency.
    KS is a non-negative distance, so its pool is already a magnitude.
    """

    def signed(path: list[str]) -> list[float]:
        vals = []
        for r in rows:
            node: Any = r["distribution"]
            for key in path:
                node = node[key]
            if node is not None:
                vals.append(float(node))
        return vals

    mean_signed = signed(["mean", "pct_diff"])
    median_signed = signed(["median", "pct_diff"])
    ks_vals = signed(["ks_distance"])
    q0_signed = [
        float(r["q0_zero_anchor"]["mean_pct_diff"])
        for r in rows
        if r["q0_zero_anchor"]["mean_pct_diff"] is not None
    ]

    decile_abs = {}
    decile_signed = {}
    for level in DECILES:
        key = f"d{int(round(level * 10))}"
        raw = signed(["deciles", key, "pct_diff"])
        decile_signed[key] = _summary(raw)
        decile_abs[key] = _summary([abs(v) for v in raw])

    # Pooled (across-seed mean) magnitude of the Q0 gap. The per-seed Q0
    # real-vs-real floor is noisy (the subgroup is ~900 persons/side, with
    # near-zero PIA-proxy denominators), so a per-seed 5% Q0 gate would
    # reject reality on some seeds; the ACROSS-SEED MEAN of the signed Q0
    # gap cancels that noise (real is unbiased) and cleanly separates real
    # (|pooled| ~2.7%) from the candidate-7 residual (+9.3%). The proposed
    # Q0 gate is therefore on this pooled magnitude, at the a-priori 5%
    # band -- see gates.yaml gate_1.thresholds.amendment_proposed.
    pooled_abs_q0 = abs(float(np.mean(q0_signed))) if q0_signed else 0.0

    return {
        # KS block in the committed-floor convention (mean/sd/values) so
        # the gates.yaml KS threshold derives from it directly.
        "ks_distance": _summary(ks_vals),
        "abs_mean_pct_diff": _summary([abs(v) for v in mean_signed]),
        "abs_median_pct_diff": _summary([abs(v) for v in median_signed]),
        "abs_q0_mean_pct_diff": _summary([abs(v) for v in q0_signed]),
        "abs_decile_pct_diff": decile_abs,
        # Pooled Q0 magnitude (|across-seed mean of signed Q0 gap|), the
        # level the proposed Q0 gate is scored on.
        "pooled_abs_q0_mean_pct_diff": pooled_abs_q0,
        # Signed pools kept for transparency (not gated on).
        "signed_mean_pct_diff": _summary(mean_signed),
        "signed_median_pct_diff": _summary(median_signed),
        "signed_q0_mean_pct_diff": _summary(q0_signed),
        "signed_decile_pct_diff": decile_signed,
        # Per-decile count of seeds whose real-vs-real gap exceeds the 5%
        # a-priori band. d1 AND d2 are denominator-fragile (near-zero
        # PIA-proxy at the bottom), so their real-vs-real floor clips 5% on
        # a majority of seeds; this is the machine-visible evidence for
        # reporting-not-gating d1 and d2 in the amendment.
        "decile_seeds_clipping_5pct": {
            f"d{int(round(level * 10))}": int(
                sum(
                    abs(v) > SUCCESS_CRITERION_PCT
                    for v in decile_signed[f"d{int(round(level * 10))}"][
                        "values"
                    ]
                )
            )
            for level in DECILES
        },
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full real-vs-real PIA-proxy floor (reported anchor)."""
    started = time.time()

    params = load_ssa_parameters()
    if verbose:
        print(f"SSA oracle parameters: pe_us_revision={params.pe_us_revision}")

    panel = load_filtered_panel()
    all_anchor = anchor_rows(panel)
    cutpoints = anchor_quintile_cutpoints(all_anchor)
    n_persons = int(panel.person_id.nunique())
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, {n_persons} "
            f"persons; {len(all_anchor)} anchors; positive-anchor quartile "
            f"cuts {[round(float(c)) for c in cutpoints]}"
        )

    seed_rows = [
        measure_seed(s, panel, all_anchor, cutpoints, params, verbose)
        for s in SEEDS
    ]
    floor = pool_floor(seed_rows)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "pia_proxy_floor_9822",
        "reported_anchor_not_gated": True,
        "purpose": (
            "Real-vs-real PIA-proxy distribution gaps at deployment scale: "
            "the empirical anchor the gate-1 amendment proposal cites for "
            "the proposed benefit_space KS threshold and the scale against "
            "which every proposed benefit-space band is judged. This reads "
            "no gate and changes no gate on its own; the amendment that "
            "would use it requires a fresh referee round and maintainer "
            "ratification."
        ),
        "not_full_415b": (
            "The functional is a STATUTE-SHAPED PROXY, not the full 42 USC "
            "415(b) AIME: the filtered panel's careers are partial "
            "(biennial PSID, prime age 25-59, reference years 1998-2022), "
            "so a faithful highest-35 AIME cannot be formed. The proxy is "
            "monotone in lifetime earnings, routes through the exact "
            "415(a)/415(g) PIA bend-point formula, and uses a constant "
            "scale (top min(10,n) indexed years / (count*12*2)) that "
            "cancels in every real-vs-real and real-vs-candidate "
            "comparison. It is byte-for-byte the functional in "
            "scripts/build_downstream_relevance.py."
        ),
        "functional": {
            "source": (
                "scripts/build_downstream_relevance.py:person_pia_proxy "
                "(imported, not re-implemented; single source of truth)"
            ),
            "steps": [
                "positive-earnings periods only, within the locked filter "
                "(age 25-59, periods 1998-2022)",
                "cap each year at that year's wage base (wage_base_for)",
                "index each to 2022 by nawi[2022]/nawi[year]",
                "AIME-proxy = sum(top min(10, n_pos) indexed) / "
                "(min(10, n_pos) * 12 * 2)",
                "PIA-proxy = pia(AIME-proxy, 2022, params) "
                "[2022-eligibility 415(a)/415(g)]",
                "zero positive observations -> PIA-proxy 0 (person included)",
            ],
            "weight": (
                "each person weighted by their anchor-period weight (the "
                "weight on their chronologically last observed period)"
            ),
            "q0_definition": (
                "Q0 = zero anchor earnings (no earnings in the person's "
                "last observed period); membership fixed once on the full "
                "filtered panel (seed-stable), so each seed's disjoint "
                "halves partition the same population map"
            ),
            "positive_anchor_quartile_cutpoints": [
                float(c) for c in cutpoints
            ],
        },
        "oracle": {
            "source": "policyengine-us (loaded once via load_ssa_parameters)",
            "pe_us_revision": params.pe_us_revision,
            "pe_us_dir_env": "POPULACE_DYNAMICS_PE_US_DIR",
            "eligibility_year": 2022,
            "bend_points_2022": list(params.bend_points(2022)),
            "pia_factors": list(params.pia_factors),
            "nawi_2022": params.nawi[2022],
        },
        "data": (
            "PSID family-file head/spouse labor income attached to persons; "
            "age 25-59, reference years 1998-2022, positive weights; the "
            "FULL locked filtered panel (deployment scale, not a train "
            "split)"
        ),
        "method": (
            "candidate-context (ctx20) real-vs-real floor at deployment "
            "scale: per seed s, draw 40% of persons "
            "(split_panel_by_person, fraction=0.4, seed=1000+s), halve it "
            "person-disjointly (fraction=0.5, seed=s), push both disjoint "
            "halves through the pinned PIA-proxy functional, and measure "
            "the distribution gap of the two PIA-proxy samples plus the Q0 "
            "subgroup mean % gap -- real vs real at the ~20%-of-persons "
            "scale a candidate is scored at"
        ),
        "n_person_periods": int(len(panel)),
        "n_persons": n_persons,
        "protocol": {
            "seeds": list(SEEDS),
            "measurements": [
                "distribution gaps of the two disjoint PIA-proxy samples: "
                "|mean % gap|, |median % gap|, decile % gaps, weighted KS",
                "Q0 (zero-anchor-earnings) subgroup real-vs-real mean % gap",
            ],
            "note": (
                "disjoint persons across the two halves, so only "
                "distribution-gap measures apply; there is no person-level "
                "alignment (that is candidate-vs-real only)"
            ),
        },
        "success_criterion": {
            "pct": SUCCESS_CRITERION_PCT,
            "text": SUCCESS_CRITERION_TEXT,
        },
        # Headline: the KS block in the committed-floor convention, so the
        # gates.yaml KS threshold derives as round(mean + k*sd) and
        # tests/test_gates_derivations.py binds it.
        "noise_floor_seeds_0_4": {
            "ks_distance": floor["ks_distance"],
        },
        "floor_seeds_0_4": floor,
        "per_seed": seed_rows,
        "revision_pins": {
            "populace_dynamics_sha": _sha(ROOT),
            "pe_us_revision": params.pe_us_revision,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        f = floor
        print(
            "\nFLOOR (mean +/- sd across seeds 0-4): "
            f"|mean %|={f['abs_mean_pct_diff']['mean']:.3f}"
            f"+/-{f['abs_mean_pct_diff']['sd']:.3f} "
            f"|median %|={f['abs_median_pct_diff']['mean']:.3f}"
            f"+/-{f['abs_median_pct_diff']['sd']:.3f} "
            f"KS={f['ks_distance']['mean']:.4f}"
            f"+/-{f['ks_distance']['sd']:.4f} "
            f"|Q0 mean %|={f['abs_q0_mean_pct_diff']['mean']:.3f}"
            f"+/-{f['abs_q0_mean_pct_diff']['sd']:.3f}"
        )
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
