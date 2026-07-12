"""M4 candidate 1: the disability hazards scored under the locked gate_m4.

ONE-SHOT scored run, pre-registered on issue #42 comment 4950931158 (the
FROZEN registration; the registration wins). Publishes regardless; no
holdout tuning. Do NOT grade here -- verification-before-record follows
per the standing addendum.

What is scored
--------------
The candidate is :mod:`populace_dynamics.models.disability_hazard_sim` --
"the M4 module's hazard machinery" of
:mod:`populace_dynamics.data.disability`, refit per split seed on the
TRAIN half (side B) and simulated on the HOLDOUT support (side A), with
the ratified K=20 mean-over-draws estimator (numpy
``default_rng(5200 + k)``, k=0..19) and the pinned START-WAVE weights.
The gate is the locked ``gate_m4`` surface: 12 gated cells --

* 8 INTERNAL mixed-k reproduction cells (flow hazards @k=3, the
  prevalence.50-59 occupancy STOCK @k=4), scored
  ``|ln(rbar_candidate,s / rate_a,s)| <= tolerance`` where the tolerance
  is ``round(floor mean + k*sd, 3)`` capped at ``ln(1.5)`` on the frozen
  floor ``runs/m4_gate_floors_v1.json`` (== the locked gate_m4 values);
* 4 ANCHOR cells (concept-bridged prevalence age-shape + conversion-exit
  dominance, per sex), scored on the candidate's own per-seed simulated
  invariant (min adjacent prevalence gap; retirement-exit dominance
  margin) with margin ``>= MARGIN_K=3 x`` the committed real half-split
  sd -- the MARGIN reading, not the bare ordinal.

A seed passes iff every gated cell holds; the gate passes iff ``>= 4 of
5`` gate seeds pass. Per the ``fresh_run_artifact_schema`` the run
commits the ``[20, 8, 5]`` per-draw per-cell rate cube, invalidates on
any undefined gated draw (empty simulated denominator), and discloses
the per-draw dispersion (report-only). The artifact
``runs/gate_m4_hazard_v1.json`` is written via ``write_new`` (the
one-shot rule).

Run (the scored run needs no populace-fit and no policyengine-us -- it is
real-holdout vs simulated-holdout only)::

    .venv-gate/bin/python scripts/run_gate_m4_candidate1.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics import artifacts
from populace_dynamics.data import deaths, disability
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import disability_hazard_sim as dhs

# The floor builder is a sibling script; add scripts/ so the shared rate
# helpers (conversion_exit_shares, prevalence_shares) import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_m4_gate_floors as bf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOOR_RUN = ROOT / "runs" / "m4_gate_floors_v1.json"
DEFAULT_OUT = ROOT / "runs" / "gate_m4_hazard_v1.json"
SCHEMA_VERSION = "gate_m4_hazard.v1"

#: The frozen registration (issue #42), embedded as a first-class pointer.
REGISTRATION_POINTER = "4950931158"
SPEC_REGISTRATION = (
    "M4 candidate 1, pre-registered on issue #42 comment "
    f"{REGISTRATION_POINTER}: the M4 module's hazard machinery refit per "
    "seed on the train half; K=20 mean-over-draws (streams 5200+k); "
    "start-wave weights; fresh_run_artifact_schema [20,8,5] with "
    "undefined-draw invalidation and dispersion disclosure; internal "
    "mixed-k reproduction cells + >=3 sigma anchor margins."
)
PRE_REGISTERED_FORECAST = (
    "P(gate >= 4/5 seeds) = 0.55-0.75. The internal cells are reproduction "
    "cells for the very hazard family the floor priced (faithful OC 0.9689); "
    "the module analogs held the anchor statistics at 4.8-12.8 sigma on "
    "100/100 halves. Named residual risks: the two thin recovery.60-66 "
    "cells and stock-vs-flow composition at prevalence.50-59 under k=4. "
    "Modal failure shape: one recovery.60-66 cell on 2 seeds."
)

GATE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SPLIT_FRACTION = 0.5
SPLIT_COLUMN = "person_id"

FLOW_K = 3
STOCK_K = 4
ROUNDING = 3
MARGIN_K = 3
T_MAX = math.log(1.5)

#: The 8 locked internal tolerances (gate_m4.thresholds.internal_surface);
#: the run DERIVES them from the frozen floor and asserts equality, so a
#: floor / gates drift fails loudly rather than scoring against a stale
#: tolerance.
LOCKED_TOLERANCES: dict[str, float] = {
    "incidence.40-49|female": 0.358,
    "incidence.50-59|female": 0.334,
    "incidence.50-59|male": 0.354,
    "prevalence.50-59|female": 0.371,
    "prevalence.50-59|male": 0.387,
    "recovery.50-59|female": 0.4,
    "recovery.60-66|female": 0.332,
    "recovery.60-66|male": 0.4,
}
#: The 4 locked anchor half-split sds (gate_m4.anchor_surface.*
#: real_half_split_sd); read from the floor + asserted equal.
LOCKED_ANCHOR_SD: dict[str, float] = {
    "prevalence_ageshape.comonotone|female": 0.01090291052862014,
    "prevalence_ageshape.comonotone|male": 0.010965466629052454,
    "conversion_exit.retirement_dominant|female": 0.043471378797961174,
    "conversion_exit.retirement_dominant|male": 0.028006631583008267,
}
BRIDGED_BANDS = bf.BRIDGED_BANDS
ALL_BANDS = bf.ALL_BANDS


def _cell_k(cell: str) -> int:
    return STOCK_K if cell.split(".", 1)[0] == "prevalence" else FLOW_K


def _derive_tolerance(mean: float, sd: float, k: int) -> float:
    return min(round(mean + k * sd, ROUNDING), T_MAX)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def load_panel() -> disability.DisabilityPanel:
    status = disability.read_disability_status()
    death_records = deaths.read_death_records()
    return disability.build_disability_panel(status, death_records)


def resolve_contract(floor: dict[str, Any]) -> dict[str, Any]:
    """Derive the internal tolerances from the frozen floor and read the
    anchor sds, asserting both equal the locked gate_m4 values."""
    noise = floor["internal_noise_floor"]["noise_floor_seeds_0_99"]
    internal_cells = sorted(floor["gate_partition"]["internal_gate_eligible"])
    anchor_cells = sorted(floor["gate_partition"]["anchor_gate_eligible"])

    tolerances: dict[str, float] = {}
    tol_derivation: dict[str, Any] = {}
    for cell in internal_cells:
        k = _cell_k(cell)
        mean, sd = noise[cell]["mean"], noise[cell]["sd"]
        tol = _derive_tolerance(mean, sd, k)
        tolerances[cell] = tol
        tol_derivation[cell] = {
            "floor_mean": mean,
            "floor_sd": sd,
            "k": k,
            "raw": round(mean + k * sd, ROUNDING),
            "t_max": T_MAX,
            "tolerance": tol,
            "locked_gate_value": LOCKED_TOLERANCES[cell],
        }
        if abs(tol - LOCKED_TOLERANCES[cell]) > 1e-12:
            raise RuntimeError(
                f"tolerance drift for {cell}: derived {tol} != locked "
                f"{LOCKED_TOLERANCES[cell]} (floor/gates out of sync)."
            )

    anchor_sd: dict[str, float] = {}
    for cell in anchor_cells:
        ac = floor["anchor_checks"][cell]
        hsf = ac["half_split_floor"]
        sd = (hsf["min_gap"] if "min_gap" in hsf else hsf["share"])["sd"]
        anchor_sd[cell] = sd
        if abs(sd - LOCKED_ANCHOR_SD[cell]) > 1e-15:
            raise RuntimeError(
                f"anchor sd drift for {cell}: floor {sd} != locked "
                f"{LOCKED_ANCHOR_SD[cell]}."
            )

    return {
        "internal_cells": internal_cells,
        "anchor_cells": anchor_cells,
        "tolerances": tolerances,
        "tolerance_derivation": tol_derivation,
        "anchor_sd": anchor_sd,
    }


def _prev_min_gap(shares: dict[str, np.ndarray], sex: str) -> float:
    sh = shares[sex]
    gaps = [float(sh[i + 1] - sh[i]) for i in range(len(BRIDGED_BANDS) - 1)]
    return min(gaps)


def score_seed(
    seed: int,
    panel: disability.DisabilityPanel,
    contract: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Fit on side B, simulate K=20 draws on side A, score the 12 cells."""
    started = time.time()
    internal_cells: list[str] = contract["internal_cells"]
    anchor_cells: list[str] = contract["anchor_cells"]
    tolerances: dict[str, float] = contract["tolerances"]
    anchor_sd: dict[str, float] = contract["anchor_sd"]

    ids_frame = panel.person_years[[SPLIT_COLUMN]].drop_duplicates()
    left, right = hpanel.split_panel_by_person(
        ids_frame, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
    )
    ids_a = set(left[SPLIT_COLUMN])  # holdout (side A)
    ids_b = set(right[SPLIT_COLUMN])  # train complement (side B)

    mom_a = disability.reference_moments(panel, ids_a)
    model = dhs.fit(panel, ids_b)

    draw_seeds = [DRAW_SEED_BASE + k for k in range(N_DRAWS)]
    # Per-draw collections.
    per_draw_rate = {c: [] for c in internal_cells}
    per_draw_nrisk = {c: [] for c in internal_cells}
    prev_shares_draws: list[dict[str, np.ndarray]] = []
    conv_draws: list[dict[str, dict[str, float]]] = []

    for ds in draw_seeds:
        sim = dhs.simulate_draw(panel, model, ids_a, ds)
        mom = disability.reference_moments(sim)
        for c in internal_cells:
            per_draw_rate[c].append(float(mom[c]["rate"]))
            per_draw_nrisk[c].append(int(mom[c]["n_at_risk"]))
        prev_shares_draws.append(bf.prevalence_shares(mom))
        conv_draws.append(bf.conversion_exit_shares(sim))

    # --- undefined-draw rule (fresh_run_artifact_schema) -----------------
    undefined = []
    for c in internal_cells:
        for k, nrisk in enumerate(per_draw_nrisk[c]):
            if nrisk == 0:
                undefined.append(
                    {"cell": c, "seed": seed, "draw_seed": draw_seeds[k]}
                )
    # the conversion anchor is also undefined on an empty exit denominator
    conv_undefined = []
    for sex in disability.SEXES:
        for k in range(N_DRAWS):
            if conv_draws[k][sex]["n_exits"] == 0:
                conv_undefined.append(
                    {
                        "cell": f"conversion_exit.retirement_dominant|{sex}",
                        "seed": seed,
                        "draw_seed": draw_seeds[k],
                    }
                )

    # --- internal cells: rbar, score, pass -------------------------------
    internal_result: dict[str, Any] = {}
    for c in internal_cells:
        rates = np.array(per_draw_rate[c], dtype=np.float64)
        rbar = float(rates.mean())
        rate_a = float(mom_a[c]["rate"])
        if rbar > 0 and rate_a > 0:
            score = abs(math.log(rbar / rate_a))
        else:
            score = math.inf
        per_draw_ln = [
            abs(math.log(r / rate_a)) for r in rates if r > 0 and rate_a > 0
        ]
        internal_result[c] = {
            "rbar": rbar,
            "rate_a": rate_a,
            "score": score,
            "tolerance": tolerances[c],
            "pass": bool(score <= tolerances[c]),
            "k": _cell_k(c),
            "per_draw_rate": [float(r) for r in rates],
            "per_draw_rate_sd": (
                float(rates.std(ddof=1)) if rates.size > 1 else 0.0
            ),
            "max_per_draw_abs_ln": (max(per_draw_ln) if per_draw_ln else None),
            "unweighted_note": "gated statistic is start-wave weighted",
            "n_at_risk_min": int(min(per_draw_nrisk[c])),
            "n_draws_defined": int(
                sum(1 for nr in per_draw_nrisk[c] if nr > 0)
            ),
        }

    # --- anchor cells: rbar-basis invariant + >= MARGIN_K sigma ----------
    anchor_result: dict[str, Any] = {}
    # prevalence age-shape (rbar over the K=20 per-draw shares)
    mean_shares = {
        sex: np.mean(
            [prev_shares_draws[k][sex] for k in range(N_DRAWS)], axis=0
        )
        for sex in disability.SEXES
    }
    for sex in disability.SEXES:
        cell = f"prevalence_ageshape.comonotone|{sex}"
        sd = anchor_sd[cell]
        min_gap = _prev_min_gap(mean_shares, sex)
        per_draw_gap = [
            _prev_min_gap(prev_shares_draws[k], sex) for k in range(N_DRAWS)
        ]
        margin_sigma = min_gap / sd if sd > 0 else None
        anchor_result[cell] = {
            "statistic": "prevalence min adjacent gap over bridged bands",
            "invariant_value": float(min_gap),
            "real_half_split_sd": sd,
            "margin_k": MARGIN_K,
            "threshold": MARGIN_K * sd,
            "margin_sigma_units": (
                float(margin_sigma) if margin_sigma is not None else None
            ),
            "pass": bool(
                margin_sigma is not None and margin_sigma >= MARGIN_K
            ),
            "rbar_shares_by_band": {
                b: float(mean_shares[sex][i]) for i, b in enumerate(ALL_BANDS)
            },
            "per_draw_min_gap": [float(g) for g in per_draw_gap],
        }
    # conversion-exit dominance (rbar over the K=20 per-draw shares)
    for sex in disability.SEXES:
        cell = f"conversion_exit.retirement_dominant|{sex}"
        sd = anchor_sd[cell]
        shares = np.array(
            [conv_draws[k][sex]["share"] for k in range(N_DRAWS)],
            dtype=np.float64,
        )
        rbar_share = float(shares.mean())
        margin = rbar_share - 0.5
        margin_sigma = margin / sd if sd > 0 else None
        anchor_result[cell] = {
            "statistic": "retirement-exit dominance margin (share - 0.5)",
            "rbar_retirement_exit_share": rbar_share,
            "invariant_value": float(margin),
            "real_half_split_sd": sd,
            "margin_k": MARGIN_K,
            "threshold": MARGIN_K * sd,
            "margin_sigma_units": (
                float(margin_sigma) if margin_sigma is not None else None
            ),
            "pass": bool(
                margin_sigma is not None and margin_sigma >= MARGIN_K
            ),
            "per_draw_share": [float(s) for s in shares],
            "mean_n_exits": float(
                np.mean(
                    [conv_draws[k][sex]["n_exits"] for k in range(N_DRAWS)]
                )
            ),
        }

    n_internal_pass = sum(v["pass"] for v in internal_result.values())
    n_anchor_pass = sum(v["pass"] for v in anchor_result.values())
    n_gated = len(internal_cells) + len(anchor_cells)
    n_gated_pass = n_internal_pass + n_anchor_pass
    seed_pass = bool(n_gated_pass == n_gated)

    if verbose:
        print(
            f"seed {seed}: internal {n_internal_pass}/{len(internal_cells)} "
            f"anchor {n_anchor_pass}/{len(anchor_cells)} -> "
            f"seed_pass={seed_pass} ({round(time.time() - started, 1)}s)"
        )

    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "estimator": "mean_over_K20_draws",
        "draw_seeds": draw_seeds,
        "internal_cells": internal_result,
        "anchor_cells": anchor_result,
        "n_internal_pass": int(n_internal_pass),
        "n_anchor_pass": int(n_anchor_pass),
        "n_gated_pass": int(n_gated_pass),
        "seed_pass": seed_pass,
        "undefined_internal_draws": undefined,
        "undefined_conversion_anchor_draws": conv_undefined,
        "fit_meta": model.meta,
        "elapsed_seconds": round(time.time() - started, 1),
    }


def build_cube(
    per_seed: list[dict[str, Any]], internal_cells: list[str]
) -> list[list[list[float]]]:
    """The [20, 8, 5] per-draw per-cell rate cube (K x cells x seeds)."""
    by_seed = {s["seed"]: s for s in per_seed}
    cube: list[list[list[float]]] = []
    for k in range(N_DRAWS):
        plane: list[list[float]] = []
        for cell in internal_cells:
            plane.append(
                [
                    by_seed[s]["internal_cells"][cell]["per_draw_rate"][k]
                    for s in GATE_SEEDS
                ]
            )
        cube.append(plane)
    return cube


def dispersion_disclosure(
    per_seed: list[dict[str, Any]], internal_cells: list[str]
) -> dict[str, Any]:
    """Report-only per-cell per-draw dispersion (no cap gates the run)."""
    by_seed = {s["seed"]: s for s in per_seed}
    return {
        "note": (
            "REPORT-ONLY: no dispersion cap gates the run; it lets a "
            "referee see whether a passing 20-draw mean conceals a wild "
            "individual draw."
        ),
        "per_cell_per_draw_sd": {
            cell: {
                str(s): by_seed[s]["internal_cells"][cell]["per_draw_rate_sd"]
                for s in GATE_SEEDS
            }
            for cell in internal_cells
        },
        "max_per_draw_abs_ln_per_cell": {
            cell: {
                str(s): by_seed[s]["internal_cells"][cell][
                    "max_per_draw_abs_ln"
                ]
                for s in GATE_SEEDS
            }
            for cell in internal_cells
        },
    }


def build_verdict(
    per_seed: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    internal_cells = contract["internal_cells"]
    anchor_cells = contract["anchor_cells"]
    seed_pass = {str(s["seed"]): bool(s["seed_pass"]) for s in per_seed}
    n_pass = sum(seed_pass.values())

    failing_internal = []
    failing_anchor = []
    for s in per_seed:
        for cell, rec in s["internal_cells"].items():
            if not rec["pass"]:
                failing_internal.append(
                    {
                        "cell": cell,
                        "seed": s["seed"],
                        "family": cell.split(".", 1)[0],
                        "score": rec["score"],
                        "tolerance": rec["tolerance"],
                        "score_over_tolerance": (
                            rec["score"] / rec["tolerance"]
                            if math.isfinite(rec["score"])
                            else None
                        ),
                    }
                )
        for cell, rec in s["anchor_cells"].items():
            if not rec["pass"]:
                failing_anchor.append(
                    {
                        "cell": cell,
                        "seed": s["seed"],
                        "margin_sigma_units": rec["margin_sigma_units"],
                        "margin_k_required": MARGIN_K,
                    }
                )

    internal_seed_pass = {
        str(s["seed"]): int(s["n_internal_pass"]) for s in per_seed
    }
    anchor_seed_pass = {
        str(s["seed"]): int(s["n_anchor_pass"]) for s in per_seed
    }

    return {
        "gate": "gate_m4",
        "n_gate_seeds": len(per_seed),
        "n_gated_cells": len(internal_cells) + len(anchor_cells),
        "n_gated_internal": len(internal_cells),
        "n_gated_anchor": len(anchor_cells),
        "seed_pass": seed_pass,
        "n_seeds_pass": int(n_pass),
        "gate_m4_pass": bool(n_pass >= 4),
        "rule": (
            "seed passes iff every gated cell holds (internal "
            "|ln(rbar/rate_a)| <= tolerance AND anchor margin >= "
            "MARGIN_K=3 sigma); gate passes iff >= 4 of 5 gate seeds pass"
        ),
        "internal_vs_anchor_decomposition": {
            "internal": {
                "n_gated": len(internal_cells),
                "per_seed_n_pass": internal_seed_pass,
                "all_failing_internal_cells": sorted(
                    failing_internal, key=lambda r: (r["seed"], r["cell"])
                ),
                "note": (
                    "reproduction cells: |ln(rbar_candidate / rate_a)| vs "
                    "round(floor mean + k*sd, 3), flow k=3 / stock k=4"
                ),
            },
            "anchor": {
                "n_gated": len(anchor_cells),
                "per_seed_n_pass": anchor_seed_pass,
                "all_failing_anchor_cells": sorted(
                    failing_anchor, key=lambda r: (r["seed"], r["cell"])
                ),
                "note": (
                    "concept-bridged invariants on the candidate's simulated "
                    "holdout: prevalence min adjacent gap; retirement-exit "
                    "dominance margin; each > MARGIN_K=3 x real half-split sd"
                ),
            },
        },
        "all_failing_gated_cells": sorted(
            [
                {"cell": r["cell"], "seed": r["seed"], "kind": "internal"}
                for r in failing_internal
            ]
            + [
                {"cell": r["cell"], "seed": r["seed"], "kind": "anchor"}
                for r in failing_anchor
            ],
            key=lambda r: (r["seed"], r["cell"]),
        ),
    }


def run(out_path: Path, verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    assert "populace.fit" not in sys.modules

    floor = artifacts_read_json(FLOOR_RUN)
    contract = resolve_contract(floor)
    panel = load_panel()

    if verbose:
        print(
            f"panel: {len(panel.person_years)} py, "
            f"{panel.person_years['person_id'].nunique()} persons, "
            f"{len(panel.pairs)} pairs"
        )

    per_seed = [score_seed(s, panel, contract, verbose) for s in GATE_SEEDS]

    # Undefined-draw hard stop (fresh_run_artifact_schema): the run is
    # invalidated -- not silently repaired -- if any gated draw is
    # undefined. No draw is dropped or re-rolled.
    undefined = [u for s in per_seed for u in s["undefined_internal_draws"]]
    undefined += [
        u for s in per_seed for u in s["undefined_conversion_anchor_draws"]
    ]
    if undefined:
        raise RuntimeError(
            "RUN INVALIDATED (undefined_draw_rule): empty simulated "
            f"denominator on gated draws {undefined}. Re-register and re-run; "
            "no draw may be dropped, substituted, or re-rolled."
        )

    cube = build_cube(per_seed, contract["internal_cells"])
    verdict = build_verdict(per_seed, contract)
    dispersion = dispersion_disclosure(per_seed, contract["internal_cells"])

    if verbose:
        print(
            f"VERDICT: {verdict['n_seeds_pass']}/5 seeds pass -> "
            f"gate_m4_pass = {verdict['gate_m4_pass']}"
        )

    # Strip the bulky per-draw / fit_meta detail into report-only sidecars;
    # keep the per-seed verdicts + decomposition leading and auditable.
    per_seed_out = per_seed

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": "gate_m4_hazard_v1",
        "gate": "gate_m4",
        "candidate": "candidate 1",
        "registration_pointer": REGISTRATION_POINTER,
        "spec_registration": SPEC_REGISTRATION,
        "one_shot": (
            "one-shot scored run registered on issue #42 comment "
            f"{REGISTRATION_POINTER}; independent verification by rerun; "
            "publishes regardless; no holdout tuning"
        ),
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "verdict": verdict,
        "per_seed": per_seed_out,
        "internal_cell_order": contract["internal_cells"],
        "anchor_cell_order": contract["anchor_cells"],
        "per_draw_per_cell_rates": {
            "shape": [
                N_DRAWS,
                len(contract["internal_cells"]),
                len(GATE_SEEDS),
            ],
            "shape_dims": "K_draws x gated_internal_cells x gate_seeds",
            "axis_order": {
                "draws": [DRAW_SEED_BASE + k for k in range(N_DRAWS)],
                "cells": contract["internal_cells"],
                "seeds": list(GATE_SEEDS),
            },
            "cube": cube,
            "rule": (
                "r[k, cell, s] for all K=20 draws (default_rng(5200+k)), all "
                "8 gated internal cells, all 5 gate seeds -- so rbar and "
                "|ln(rbar/rate_a)| recompute cell-by-cell, not on trust"
            ),
        },
        "per_draw_dispersion_disclosure": dispersion,
        "protocol": {
            "candidate_estimator": (
                "mean-over-K=20-draws (5200+k), scored ONCE per cell as "
                "|ln(rbar_candidate,s / rate_a,s)| -- not the mean of "
                "per-draw scores"
            ),
            "candidate_draws": N_DRAWS,
            "candidate_draw_stream": "numpy.random.default_rng(5200 + k), k=0..19",
            "draw_stream_base": DRAW_SEED_BASE,
            "gate_seeds": list(GATE_SEEDS),
            "split": (
                "person-disjoint 50/50 (split_panel_by_person, fraction=0.5, "
                "seed=s); side A holdout, side B train complement"
            ),
            "weight_definition": (
                "START-WAVE cross-sectional PSID weight of each transition "
                "pair (the pinned M4 convention); no unweighted gated "
                "statistic"
            ),
            "internal_statistic": (
                "|ln(rbar_candidate,s / rate_a,s)| <= round(floor mean + "
                "k*sd, 3) capped at ln(1.5); flow k=3, stock k=4"
            ),
            "anchor_statistic": (
                "candidate's own per-seed simulated invariant on the rbar "
                "basis (prevalence min adjacent gap; retirement-exit "
                "dominance margin) > MARGIN_K=3 x the committed real "
                "half-split sd -- the margin reading"
            ),
            "conjunction": (
                "seed passes iff every gated cell holds; gate passes iff "
                ">= 4 of 5 gate seeds"
            ),
            "fresh_run_artifact_schema": {
                "per_draw_per_cell_rates_shape": [
                    N_DRAWS,
                    len(contract["internal_cells"]),
                    len(GATE_SEEDS),
                ],
                "undefined_draw_rule": (
                    "any undefined gated draw (empty simulated denominator) "
                    "INVALIDATES the run; enforced as a hard stop before "
                    "write"
                ),
                "per_draw_dispersion_disclosure": "report-only (committed)",
            },
        },
        "tolerance_derivation": contract["tolerance_derivation"],
        "anchor_margin_basis": {
            cell: {
                "real_half_split_sd": contract["anchor_sd"][cell],
                "margin_k": MARGIN_K,
                "threshold": MARGIN_K * contract["anchor_sd"][cell],
                "locked_gate_value": LOCKED_ANCHOR_SD[cell],
            }
            for cell in contract["anchor_cells"]
        },
        "model": {
            "module": "populace_dynamics.models.disability_hazard_sim",
            "description": (
                "the M4 module's hazard machinery: fit the unmodified "
                "populace_dynamics.data.disability weighted band x sex "
                "incidence / recovery / prevalence rates + the FRA-window "
                "retirement-exit share on the train half (side B); simulate "
                "the holdout support (side A) forward and score the same "
                "reference_moments / prevalence_shares / conversion_exit_"
                "shares code the floor used. No free hyperparameter."
            ),
            "n_gated_internal_cells": len(contract["internal_cells"]),
            "n_gated_anchor_cells": len(contract["anchor_cells"]),
        },
        "data": {
            "holdout_basis": [
                "ind2023er_employment_status (PSID self-report)",
                "m4_gate_floors_v1 (frozen 100-seed person-disjoint floor)",
            ],
            "floor_run": str(FLOOR_RUN.relative_to(ROOT)),
            "floor_run_sha256": bf._sha256_bytes(FLOOR_RUN),
            "n_persons": int(panel.person_years["person_id"].nunique()),
            "n_person_years": int(len(panel.person_years)),
            "n_transition_pairs": int(len(panel.pairs)),
        },
        "revision_pins": {
            "artifact_schema_version": SCHEMA_VERSION,
            "base_sha": floor.get("revision_pins", {}).get(
                "origin_master_sha"
            ),
            "floor_base_sha": floor.get("revision_pins", {}).get("base_sha"),
            "floor_run_sha256": bf._sha256_bytes(FLOOR_RUN),
            "candidate_head_sha": _rev("HEAD"),
            "candidate_merge_base_sha": _merge_base(),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return _json_safe(artifact)


def artifacts_read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text())


def _rev(ref: str) -> str | None:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", ref],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _merge_base() -> str | None:
    import subprocess

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    artifact = run(args.out, verbose=not args.quiet)
    artifacts.write_new(args.out, artifact)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
