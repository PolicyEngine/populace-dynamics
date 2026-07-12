"""Gate-2c candidate 2 one-shot scored run (registration 4950370498).

FROZEN spec: issue #42 comment 4950370498. Candidate 2 is ONE targeted delta
against candidate 1 (:mod:`populace_dynamics.models.couple_formation_sim_v2`):
first-marriage timing gains earnings-axis conditioning via a train-fitted
multiplicative hazard modifier ``m(tercile | age band, sex)`` composed onto the
certified 2a first-marriage hazard, NORMALIZED so the age x sex timing marginal
is preserved (``sum_t m * phi_cert = 1`` per band; the constraint is recorded
per draw and a violation is a spec violation). Everything else is byte-carried
from candidate 1 -- the run proves the carried families byte-identical against
``runs/gate2c_hazard_v1.json``.

Scored against the LOCKED gate-2c contract (``gates.yaml`` gate_2.gate_2c,
LOCKED 2026-07-10) and the frozen floor ``runs/gate2c_floors_v1.json`` (the
SAME floor candidate 1 scored against):

* 27 gated marriage x earnings joint cells; the ``|ln(rbar / rate_a)|``
  statistic; ``rbar`` = MEAN over K=20 pre-registered simulation draws
  (``numpy.random.default_rng(5200 + k)``, k=0..19), scored ONCE per cell;
* per-seed COUPLE-DISJOINT holdout, split by ``attrs.component_id`` with
  ``default_rng(seed)`` (side A = holdout / simulated, side B = train / fit);
* seed passes iff every gated cell holds; the gate passes iff >= 4 of 5 gate
  seeds pass. One-shot; publishes regardless (no holdout tuning). Run:

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_gate2c_candidate2.py \
        --out runs/gate2c_hazard_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from populace_dynamics import artifacts
from populace_dynamics.data import (
    births,
    deaths,
    family,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.data import (
    couple_earnings as ce,
)
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import couple_formation_sim_v2 as cfs
from populace_dynamics.models.family_transitions.common import (
    marriage_order_map,
)
from populace_dynamics.ss.params import load_ssa_parameters

ROOT = Path(__file__).resolve().parents[1]
FLOOR_RUN = ROOT / "runs" / "gate2c_floors_v1.json"
C1_ARTIFACT = ROOT / "runs" / "gate2c_hazard_v1.json"
GATES = ROOT / "gates.yaml"
DEFAULT_OUT = ROOT / "runs" / "gate2c_hazard_v2.json"

SCHEMA_VERSION = "gate2c_hazard.v2"
GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SIM_SEED_PROVENANCE_BASE = 4200
SPLIT_COLUMN = "component_id"
SPLIT_FRACTION = 0.5
EXACT_ATOL = 1e-12
N_GATED = 27

REGISTRATION_POINTER = "4950370498"
C1_REGISTRATION_POINTER = "4950250151"
C1_GRADING_POINTER = "4950370477"
SPEC_REGISTRATION = (
    "issue #42 comment 4950370498 -- Gate-2c candidate 2 registration: "
    "earnings-conditioned first-marriage timing (FROZEN spec, one-shot). One "
    "delta vs candidate 1 (registration 4950250151, grading 4950370477)."
)

#: The one delta, byte-carried from candidate 1 everywhere else.
CARRIED_FAMILIES = (
    "assort_mating",
    "remarriage_by_earnings",
    "earnings_around_marriage",
    "earnings_around_divorce",
    "shared_earnings_ratio",
)
DELTA_FAMILY = "first_marriage_by_earnings"

#: The registration's pre-registered forecast (recorded, never graded here).
PRE_REGISTERED_FORECAST = {
    "p_gate_pass_4_of_5": "0.45-0.65",
    "basis": (
        "the five candidate-1 misses are all marginal (1.02-1.34x tolerance) "
        "in one family with a named mechanism (the certified core is "
        "earnings-blind) and a proven marginal-preservation constraint "
        "pattern (the 2b coupling class); the assort / remarriage / event / "
        "shared families are carried byte-identically"
    ),
    "named_expectations": [
        "first_marriage_by_earnings clears on >= 4 seeds",
        "the marginal-preservation check holds (violation = spec violation)",
        "all carried families stay at their candidate-1 rates (byte-identical)",
        "modal residual if it fails = t3.25-34|male (the thinnest of the five)",
    ],
    "if_pass": (
        "verification round before the record per the standing addendum, then "
        "the 2c pass memo -- and with it, every gate-2 tranche passed"
    ),
    "grading_note": (
        "this run RECORDS its verdict and forecast; it does NOT grade itself. "
        "Grading is a separate ceremony step."
    ),
}

#: Candidate 1's spec resolutions, ADOPTED VERBATIM (the carried families are
#: unchanged), plus the candidate-2 modifier delta resolutions.
SPEC_RESOLUTION_NOTES = {
    # --- carried verbatim from candidate 1 (registration 4950250151) ---
    "five_components_source": (
        "component 1 (marriage/remarriage events + timing) = the certified "
        "tranche-2a registry CANDIDATE_16 via REGISTRY.fit refit on side B "
        "and simulate(mpanel, holdout_ids, components, 5200+k); component 3 "
        "(spouse age) = the same fit's spousal_age_gaps via draw_spousal_gaps "
        "(the 2a machinery's existing convention). Components 2 (assortative "
        "kernel), 4 (event-window shift kernel) and 5 (shared-earnings cells) "
        "are layered on the certified simulated events."
    ),
    "assortative_kernel": (
        "P(spouse earnings-decile | own earnings-decile, own age band, own "
        "sex) fit on side-B directed couples on the committed decile axis, "
        "hierarchically add-alpha smoothed with backoff "
        "(global -> sex -> sex x own_decile -> full) so every conditioning "
        "cell is well-defined despite sparse per-cell couple counts. The own "
        "age band reuses the certified first-marriage hazard bands."
    ),
    "directed_both_orientation_emission": (
        "candidate_construction.couple_emission pin: each simulated marriage "
        "of a holdout supply ego emits BOTH directed records (ego -> drawn "
        "spouse AND drawn spouse -> ego), weighted by the same ego couple "
        "weight, so the tercile contingency is exactly symmetrized-directed "
        "(a single-orientation emission is NON-CONFORMANT)."
    ),
    "committed_cut_provenance": (
        "candidate_construction.cut_provenance pin: the tercile cut levels, "
        "the earnings-decile edges and within-decile spouse-value pools, and "
        "the placebo drift deflators are FIXED on the full real earnings "
        "supply and applied to every seed / draw -- never recomputed on "
        "simulated output. Only the conditional kernels and the tranche-2a "
        "components are train-fitted (side B); holdout couples are excluded "
        "from all fitting."
    ),
    "event_window_support_and_detrend": (
        "the emitted event-window ratio is a raw (nominal) draw from the "
        "train around-event post/pre ratio pool for (event_type, sex), "
        "weighted by the ego demographic weight (fix F); the locked cell "
        "detrends by the COMMITTED placebo deflator (fix E), so the gate "
        "certifies the event increment. The train pool is the supported "
        "windows (>=1 observed positive year in both +/-3y sides), so the "
        "support restriction rides in the kernel."
    ),
    "reference_moments_reused_verbatim": (
        "the simulated frames are built to the CoupleEarningsPanel schema and "
        "scored through couple_earnings.reference_moments VERBATIM with the "
        "committed tercile cuts and placebo deflators, so each candidate cell "
        "is the identical statistic the floor measured on the real half. "
        "reference_moments is called with person_ids=None because the frames "
        "are already restricted to the seed's holdout egos and the "
        "both-orientation mirror records carry synthetic (non-holdout) ego "
        "ids that an id filter would drop."
    ),
    "shared_earnings_ratio_is_a_per_input_shape_moment": (
        "shared_earnings_ratio recomputes the couple combined-axis adjacent "
        "quintile-cutpoint RATIOS on each input (the floor does the same on "
        "each real half); the cut_provenance frozen categories are the "
        "tercile cuts and placebo deflators, which reference_moments takes "
        "from the committed panel. The quintile ratio is a scale-free "
        "distribution-shape moment measured on the candidate's own couples."
    ),
    "spouse_age_inert_for_gated_cells": (
        "component 3 (spouse age from the certified age-gap distributions) is "
        "drawn and recorded but reads into no gated cell; it is included to "
        "honor the five-component composition, mirroring the certified core "
        "where the age gap is retained for RNG topology."
    ),
    # --- candidate-2 delta resolutions (the ONE change) ---
    "delta_earnings_conditioned_first_marriage": (
        "the ONE delta: first-marriage timing gains earnings-axis "
        "conditioning via a train-fitted multiplicative modifier "
        "m(tercile | age band, sex) composed onto the certified 2a "
        "first-marriage hazard. m_raw = real_train_hazard / "
        "certified_expected_hazard, where certified_expected_hazard is the "
        "exposure-weighted mean of the certified first_marriage.predict over "
        "the train never-married supply person-years -- so the modifier is "
        "the RESIDUAL earnings gradient the age x sex x cohort core misses "
        "(a cell the core already matches gets m_raw ~ 1; no double-counting "
        "of the age/cohort composition the core carries). The fit READS the "
        "certified hazard and consumes no RNG."
    ),
    "delta_modifier_normalization_and_marginal_preservation": (
        "the modifier is NORMALIZED so the age x sex timing marginal is "
        "preserved: with m applied to the certified first-marriage event "
        "weights, the pooled-over-tercile band hazard scales by "
        "sum_t m(t|b,s) * phi_cert(t|b,s) where phi_cert is the certified "
        "expected first-marriage EVENT share by tercile (the 'train tercile "
        "shares'). Dividing by Z(b,s) = sum_t m_shrunk * phi_cert makes "
        "sum_t m * phi_cert = 1 per band EXACTLY, so the certified pooled "
        "band hazard does not move (in expectation). The exact constraint and "
        "the realized per-draw pooled band-hazard deviation are recorded per "
        "draw; a constraint violation is a spec violation. This is the same "
        "marginal-preservation constraint class the 2b coupling used "
        "(household_composition_sim_v5: read a carried marginal, compose a "
        "conditional onto it, leave the marginal unmoved)."
    ),
    "delta_modifier_shrinkage": (
        "the log-modifier ln(m_raw) is shrunk toward 0 (neutral) with weight "
        "n_events / (n_events + alpha), alpha = "
        f"{cfs.MODIFIER_SHRINKAGE_ALPHA} (pinned a priori; the gate verdict is "
        "invariant to alpha over {0, 8, 20}). The dense GATED cells "
        "(n_events >= 570) are essentially unshrunk (weight >= 0.986); the "
        "shrinkage only regularizes the thin report-only cells' "
        "normalization."
    ),
    "delta_byte_carry": (
        "the modifier is applied as a deterministic reweighting of the "
        "first_marriage rows of the already-built marital_events frame, AFTER "
        "_build_marital and consuming no RNG, so the certified simulate, the "
        "assortative / age-gap / event-window draws consume the IDENTICAL RNG "
        "streams as candidate 1: the simulated couples, event windows and "
        "never-married exposure are bit-identical, and only the first_marriage "
        "event WEIGHTS change (remarriage rows untouched). The carried "
        "families' per-draw rates are byte-identical to candidate 1's, proven "
        "by the byte_carry_regression against runs/gate2c_hazard_v1.json."
    ),
    "rng_topology": (
        "unchanged from candidate 1: split seed = the raw gate seed; draw "
        "stream = default_rng(5200 + k), k=0..19 (the certified simulate); "
        "the candidate joints draw from four independent children of "
        "SeedSequence([5200+k, 0x2C1]). The modifier adds NO stream. "
        "default_rng(4200 + seed) is recorded as provenance only."
    ),
}

#: Family -> a mechanism note. The first_marriage note now describes the delta.
_FAMILY_MECHANISMS = {
    "assort_mating": (
        "who-marries-whom by earnings tercile: the train-fitted "
        "decile-conditional assortative kernel emitted in both directions "
        "(byte-carried from candidate 1)"
    ),
    "first_marriage_by_earnings": (
        "earnings-tercile-conditional first-marriage timing: the certified "
        "marital core (age x sex x cohort) composed with the train-fitted "
        "residual earnings modifier m(tercile | age band, sex), normalized so "
        "the age x sex timing marginal is preserved (THE candidate-2 delta)"
    ),
    "remarriage_by_earnings": (
        "earnings-tercile-conditional remarriage timing: the certified core "
        "over post-dissolution person-years (byte-carried; the modifier "
        "reweights first_marriage rows only)"
    ),
    "earnings_around_marriage": (
        "around-marriage earnings dynamics from the train shift kernel, "
        "detrended by the committed placebo deflator (byte-carried)"
    ),
    "earnings_around_divorce": (
        "around-divorce earnings dynamics from the train shift kernel, "
        "detrended by the committed placebo deflator (byte-carried)"
    ),
    "shared_earnings_ratio": (
        "couple combined-axis distribution shape from the drawn spouse axis "
        "values plus the real ego axis (byte-carried)"
    ),
}


# --------------------------------------------------------------------------
# JSON coercion
# --------------------------------------------------------------------------
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_of(cell: str) -> str:
    return cell.split(".")[0]


# --------------------------------------------------------------------------
# Contract + data loading (carried from candidate 1)
# --------------------------------------------------------------------------
def load_locked_tolerances() -> tuple[dict[str, float], list[str]]:
    gates = yaml.safe_load(GATES.read_text())
    th = gates["gates"]["gate_2"]["gate_2c"]["thresholds"]
    if not th.get("locked"):
        raise RuntimeError("gate_2c thresholds are not locked")
    tol: dict[str, float] = {}
    for view in th["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    if len(tol) != N_GATED:
        raise RuntimeError(f"expected {N_GATED} gated cells, got {len(tol)}")
    return tol, list(th["report_only"])


def _cross_check_tolerances(tol: dict[str, float], floor: dict) -> dict:
    cells = floor["draft_thresholds"]["cells"]
    max_dev = 0.0
    for cell, t in tol.items():
        max_dev = max(max_dev, abs(t - cells[cell]["log_ratio_abs_max"]))
    gated = set(floor["gate_partition"]["gate_eligible"])
    return {
        "tolerances_match_floor_draft_thresholds": max_dev <= EXACT_ATOL,
        "max_abs_tolerance_deviation": max_dev,
        "tolerance_cells_equal_floor_gate_eligible": set(tol) == gated,
        "k": floor["draft_thresholds"]["k"],
        "rounding": floor["draft_thresholds"]["rounding"],
        "t_max": floor["draft_thresholds"]["t_max"],
    }


def load_all() -> dict[str, Any]:
    params = load_ssa_parameters()
    marriage_records = marriage.marriage_history()
    death_records = deaths.read_death_records()
    birth_records = births.birth_history()
    demo = panels.demographic_panel()
    demo_pos = demo[demo["weight"] > 0]
    person_weight = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    earnings_panel = family.family_earnings_panel()
    ce_panel = ce.build_couple_panel(
        params=params,
        marriage_records=marriage_records,
        earnings_panel=earnings_panel,
        death_records=death_records,
        person_weight=person_weight,
    )
    mpanel = transitions.build_marital_panel(
        marriage_records, death_records, person_weight
    )
    order_map = marriage_order_map(marriage_records)
    axis = cfs.build_committed_axis(
        ce_panel,
        earnings_panel=earnings_panel,
        marriage_records=marriage_records,
        params=params,
        person_weight=person_weight,
    )
    return {
        "params": params,
        "ce_panel": ce_panel,
        "mpanel": mpanel,
        "demographic_panel": demo,
        "marriage_records": marriage_records,
        "birth_records": birth_records,
        "order_map": order_map,
        "axis": axis,
    }


# --------------------------------------------------------------------------
# Precheck (hard stop): reproduce the floor exactly (carried from candidate 1)
# --------------------------------------------------------------------------
def run_precheck(ce_panel: ce.CoupleEarningsPanel, floor: dict) -> dict:
    ref = ce.reference_moments(ce_panel, weighted=True)
    committed = floor["reference_moments"]
    ref_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )

    committed_ho = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    rate_a_dev = 0.0
    sha_all = True
    per_seed = {s["seed"]: s for s in floor["noise_floor_per_seed"]}
    gated = set(floor["gate_partition"]["gate_eligible"])
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            ce_panel.attrs, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        sha_all = sha_all and (
            digest == committed_ho[seed]["holdout_person_id_sha256"]
        )
        cells_a = ce.reference_moments(ce_panel, set(ids), weighted=True)
        for key in gated:
            rate_a_dev = max(
                rate_a_dev,
                abs(
                    cells_a[key]["rate"]
                    - per_seed[seed]["cells"][key]["rate_a"]
                ),
            )
    result = {
        "all_reproduced_exactly": (
            ref_dev <= EXACT_ATOL and rate_a_dev <= EXACT_ATOL and sha_all
        ),
        "reference_moments_max_abs_deviation": ref_dev,
        "rate_a_max_abs_deviation": rate_a_dev,
        "holdout_sha256_all_match": sha_all,
        "atol": EXACT_ATOL,
    }
    if not result["all_reproduced_exactly"]:
        raise RuntimeError(f"precheck failed (hard stop): {result}")
    return result


# --------------------------------------------------------------------------
# Score one seed: fit v2 on side B, K=20 draws on side A
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    data: dict[str, Any],
    floor: dict,
    tol: dict[str, float],
    report_only: list[str],
) -> dict[str, Any]:
    started = time.time()
    ce_panel = data["ce_panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        ce_panel.attrs, SPLIT_COLUMN, fraction=SPLIT_FRACTION, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = cfs.fit_couple_model_v2(
        ce_panel,
        data["mpanel"],
        demographic_panel=data["demographic_panel"],
        marriage_records=data["marriage_records"],
        birth_records=data["birth_records"],
        marriage_order_map=data["order_map"],
        axis=data["axis"],
        train_ids=ids_b,
    )

    draw_seeds = [DRAW_SEED_BASE + k for k in range(N_DRAWS)]
    committed_cells = {s["seed"]: s for s in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]

    all_keys = list(tol) + list(report_only)
    per_draw_rate: dict[str, list[float]] = {k: [] for k in all_keys}
    per_draw_den: dict[str, list[float]] = {k: [] for k in all_keys}
    per_draw_nev: dict[str, list[int]] = {k: [] for k in all_keys}
    sim_diagnostics: list[dict[str, Any]] = []
    marginal_checks: list[dict[str, Any]] = []

    for draw_seed in draw_seeds:
        sim_panel, diag = cfs.simulate_draw_v2(
            ce_panel, data["mpanel"], model, data["axis"], ids_a, draw_seed
        )
        cand = ce.reference_moments(sim_panel, weighted=True)
        for key in all_keys:
            cell = cand[key]
            per_draw_rate[key].append(float(cell["rate"]))
            per_draw_den[key].append(float(cell["den_wt"]))
            per_draw_nev[key].append(int(cell["n_events"]))
        sim_diagnostics.append(diag)
        marginal_checks.append(diag["marginal_preservation_check"])

    def _score(key: str, gated: bool) -> dict[str, Any]:
        rates = np.asarray(per_draw_rate[key], dtype=float)
        dens = np.asarray(per_draw_den[key], dtype=float)
        rate_a = float(committed_cells[key]["rate_a"])
        rbar = float(rates.mean())
        n_defined = int((dens > 0).sum())
        undefined = [draw_seeds[k] for k in range(N_DRAWS) if dens[k] <= 0.0]
        if rbar > 0 and rate_a > 0:
            score = abs(math.log(rbar / rate_a))
        else:
            score = math.inf
        per_draw_ln = [
            abs(math.log(r / rate_a)) if r > 0 and rate_a > 0 else math.inf
            for r in rates
        ]
        rec = {
            "rbar": rbar,
            "rate_a": rate_a,
            "score": score,
            "n_events_candidate_mean": float(np.mean(per_draw_nev[key])),
            "per_draw_rate": [float(r) for r in rates],
            "per_draw_rate_sd": float(rates.std(ddof=1)),
            "max_per_draw_abs_ln": (max(per_draw_ln) if per_draw_ln else None),
            "n_draws_defined": n_defined,
            "undefined_draw_seeds": undefined,
        }
        if gated:
            rec["tolerance"] = tol[key]
            rec["pass"] = bool(score <= tol[key])
        return rec

    gated_cells = {k: _score(k, True) for k in tol}
    report_only_cells = {k: _score(k, False) for k in report_only}

    n_gated_pass = sum(1 for r in gated_cells.values() if r["pass"])
    undefined_gated_draws = [
        {"cell": k, "draw_seeds": r["undefined_draw_seeds"]}
        for k, r in gated_cells.items()
        if r["undefined_draw_seeds"]
    ]
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "estimator": "mean_over_K20_draws",
        "draw_seeds": draw_seeds,
        "sim_seed_single_draw_provenance": SIM_SEED_PROVENANCE_BASE + seed,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(n_gated_pass == len(tol)),
        "gated_cells": gated_cells,
        "report_only_cells": report_only_cells,
        "undefined_gated_draws": undefined_gated_draws,
        "fit_meta": model.meta,
        "fm_modifier_fit_vs_raw": model.fm_modifier.fit_vs_raw_record(),
        "marginal_preservation": _summarize_marginal(marginal_checks),
        "sim_diagnostics_mean": _mean_diag(sim_diagnostics),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def _summarize_marginal(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-draw marginal-preservation checks for the seed."""
    con_dev = [c["constraint_max_abs_dev_from_one"] for c in checks]
    realized = [c["realized_pooled_band_hazard_max_abs_ln"] for c in checks]
    realized_g = [
        c["realized_pooled_band_hazard_max_abs_ln_gated_bands"] for c in checks
    ]
    return {
        "constraint_max_abs_dev_from_one_over_draws": float(max(con_dev)),
        "constraint_holds_all_draws": bool(
            all(c["constraint_holds"] for c in checks)
        ),
        "realized_pooled_band_hazard_max_abs_ln_over_draws": float(
            max(realized)
        ),
        "realized_pooled_band_hazard_mean_abs_ln_over_draws": float(
            np.mean(realized)
        ),
        "realized_pooled_band_hazard_max_abs_ln_gated_bands_over_draws": float(
            max(realized_g)
        ),
        "n_draws": len(checks),
        "per_draw_constraint_max_abs_dev": [float(x) for x in con_dev],
        "note": (
            "constraint = |sum_t m*phi_cert - 1| (exact, ~0); realized = the "
            "reweighting's effect on the simulated pooled band hazard "
            "(Monte Carlo; small on the GATED bands 18-24 / 25-34, larger on "
            "the sparse report-only 35-44 / 45+ bands). A constraint violation "
            "is a spec violation."
        ),
    }


def _mean_diag(diags: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "n_marriages",
        "n_directed_couples",
        "n_marital_events",
        "n_marital_exposure_person_years",
        "n_event_windows",
        "spouse_age_gap_mean",
        "spouse_age_mean",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in diags if d.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


# --------------------------------------------------------------------------
# Byte-carry regression vs candidate 1 (the c9 / 2b write-gate carry pattern)
# --------------------------------------------------------------------------
def byte_carry_regression(
    per_seed: list[dict[str, Any]], c1_art: dict[str, Any]
) -> dict[str, Any]:
    """Prove the carried families are byte-identical to candidate 1, per draw.

    Compares every carried-family cell's per-draw rate against candidate 1's
    committed per-draw rates (``runs/gate2c_hazard_v1.json``): the carried
    families must match to 0.0, and the ONE delta family
    (first_marriage_by_earnings) must have moved.
    """
    c1_by_seed = {s["seed"]: s for s in c1_art["per_seed"]}
    per_family_max_dev: dict[str, float] = {}
    n_carried_cmp = 0
    n_carried_cells = 0
    max_carried_dev = 0.0
    max_delta_dev = 0.0
    delta_cell_seeds_moved = 0
    delta_cell_seeds_total = 0
    worst_carried = None
    for s in per_seed:
        seed = s["seed"]
        c1s = c1_by_seed[seed]
        c1_all = {**c1s["gated_cells"], **c1s["report_only_cells"]}
        v2_all = {**s["gated_cells"], **s["report_only_cells"]}
        for cell, rec in v2_all.items():
            fam = _family_of(cell)
            v2_rates = rec["per_draw_rate"]
            c1_rates = c1_all[cell]["per_draw_rate"]
            dev = max(
                abs(a - b) for a, b in zip(v2_rates, c1_rates, strict=True)
            )
            prev = per_family_max_dev.get(fam, 0.0)
            per_family_max_dev[fam] = max(prev, dev)
            if fam in CARRIED_FAMILIES:
                n_carried_cmp += len(v2_rates)
                n_carried_cells += 1
                if dev > max_carried_dev:
                    max_carried_dev = dev
                    worst_carried = f"{cell}|seed{seed}"
            elif fam == DELTA_FAMILY:
                delta_cell_seeds_total += 1
                max_delta_dev = max(max_delta_dev, dev)
                if dev > 0.0:
                    delta_cell_seeds_moved += 1
    return {
        "c1_artifact": "runs/gate2c_hazard_v1.json",
        "c1_artifact_sha256": _sha256_file(C1_ARTIFACT),
        "c1_registration_pointer": C1_REGISTRATION_POINTER,
        "carried_families": list(CARRIED_FAMILIES),
        "delta_family": DELTA_FAMILY,
        "n_carried_cells_per_seed_compared": n_carried_cells,
        "n_carried_per_draw_comparisons": n_carried_cmp,
        "carried_max_abs_rate_deviation": max_carried_dev,
        "carried_byte_identical": bool(max_carried_dev == 0.0),
        "worst_carried_cell_seed": worst_carried,
        "delta_family_max_abs_rate_deviation": max_delta_dev,
        "delta_family_moved": bool(max_delta_dev > 0.0),
        "delta_cell_seeds_moved": delta_cell_seeds_moved,
        "delta_cell_seeds_total": delta_cell_seeds_total,
        "per_family_max_abs_rate_deviation": {
            k: per_family_max_dev[k] for k in sorted(per_family_max_dev)
        },
        "rule": (
            "the carried families must be byte-identical (0.0) to candidate 1 "
            "per draw; the first_marriage_by_earnings family must have moved"
        ),
    }


# --------------------------------------------------------------------------
# Verdict + decomposition + fresh-run schema (carried from candidate 1)
# --------------------------------------------------------------------------
def build_verdict(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    seed_pass = {str(s["seed"]): bool(s["seed_pass"]) for s in per_seed}
    n_pass = sum(seed_pass.values())
    failing = []
    for s in per_seed:
        for cell, rec in s["gated_cells"].items():
            if not rec["pass"]:
                failing.append(
                    {
                        "cell": cell,
                        "seed": s["seed"],
                        "family": _family_of(cell),
                        "score": rec["score"],
                        "tolerance": rec["tolerance"],
                        "score_over_tolerance": (
                            rec["score"] / rec["tolerance"]
                            if math.isfinite(rec["score"])
                            else None
                        ),
                    }
                )
    return {
        "n_gate_seeds": len(per_seed),
        "n_gated_cells": len(tol),
        "seed_pass": seed_pass,
        "n_seeds_pass": n_pass,
        "gate_2c_pass": bool(n_pass >= 4),
        "rule": (
            "seed passes iff every gated cell |ln(rbar/rate_a)| <= tolerance; "
            "gate passes iff >= 4 of 5 gate seeds pass"
        ),
        "all_failing_gated_cells": sorted(
            failing, key=lambda r: (r["seed"], r["cell"])
        ),
    }


def per_family_decomposition(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    families: dict[str, list[str]] = {}
    for cell in tol:
        families.setdefault(_family_of(cell), []).append(cell)

    out: dict[str, Any] = {}
    for fam, cells in families.items():
        n_cell_seed = len(cells) * len(per_seed)
        n_pass = 0
        per_seed_passed = {}
        cell_abs_ln: dict[str, list[float]] = {c: [] for c in cells}
        for s in per_seed:
            passed = 0
            for c in cells:
                rec = s["gated_cells"][c]
                if rec["pass"]:
                    n_pass += 1
                    passed += 1
                if math.isfinite(rec["score"]):
                    cell_abs_ln[c].append(rec["score"])
            per_seed_passed[str(s["seed"])] = passed
        worst_cell = None
        worst_ratio = -1.0
        worst_mean_abs_ln = None
        for c in cells:
            if not cell_abs_ln[c]:
                continue
            mean_abs = float(np.mean(cell_abs_ln[c]))
            ratio = mean_abs / tol[c]
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_cell = c
                worst_mean_abs_ln = mean_abs
        cell_seed_pass_rate = n_pass / n_cell_seed if n_cell_seed else None
        out[fam] = {
            "n_cells": len(cells),
            "cells": sorted(cells),
            "n_cell_seed": n_cell_seed,
            "n_cell_seed_pass": n_pass,
            "cell_seed_pass_rate": cell_seed_pass_rate,
            "per_seed_cells_passed": per_seed_passed,
            "worst_cell": worst_cell,
            "worst_cell_mean_abs_ln": worst_mean_abs_ln,
            "worst_cell_tolerance": tol[worst_cell] if worst_cell else None,
            "worst_cell_mean_ln_over_tol": (
                round(worst_ratio, 4) if worst_ratio >= 0 else None
            ),
            "mechanism": _FAMILY_MECHANISMS[fam],
        }
    return out


def c1_to_c2_progression(
    per_seed: list[dict[str, Any]],
    c1_art: dict[str, Any],
    tol: dict[str, float],
) -> dict[str, Any]:
    """The per-seed and per-cell candidate-1 -> candidate-2 progression."""
    c1_by_seed = {s["seed"]: s for s in c1_art["per_seed"]}
    c1_verdict = c1_art["verdict"]
    seed_rows = {}
    for s in per_seed:
        seed = s["seed"]
        c1s = c1_by_seed[seed]
        seed_rows[str(seed)] = {
            "c1_n_gated_pass": c1s["n_gated_pass"],
            "c2_n_gated_pass": s["n_gated_pass"],
            "c1_seed_pass": bool(c1s["seed_pass"]),
            "c2_seed_pass": bool(s["seed_pass"]),
        }
    # the five candidate-1 misses, now
    c1_fails = {
        (f["cell"], f["seed"]) for f in c1_verdict["all_failing_gated_cells"]
    }
    resolved = []
    for cell, seed in sorted(c1_fails):
        rec = {s["seed"]: s for s in per_seed}[seed]["gated_cells"][cell]
        c1_rec = c1_by_seed[seed]["gated_cells"][cell]
        resolved.append(
            {
                "cell": cell,
                "seed": seed,
                "c1_score": c1_rec["score"],
                "c1_pass": bool(c1_rec["pass"]),
                "c2_score": rec["score"],
                "c2_pass": bool(rec["pass"]),
                "tolerance": rec["tolerance"],
                "now_passes": bool(rec["pass"]),
            }
        )
    # cells candidate 1 passed that candidate 2 fails (the cost of the delta)
    newly_failing = []
    for s in per_seed:
        seed = s["seed"]
        for cell, rec in s["gated_cells"].items():
            c1_rec = c1_by_seed[seed]["gated_cells"][cell]
            if c1_rec["pass"] and not rec["pass"]:
                newly_failing.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "family": _family_of(cell),
                        "c1_score": c1_rec["score"],
                        "c2_score": rec["score"],
                        "tolerance": rec["tolerance"],
                    }
                )
    return {
        "c1_n_seeds_pass": c1_verdict["n_seeds_pass"],
        "c2_n_seeds_pass": sum(1 for s in per_seed if s["seed_pass"]),
        "c1_gate_pass": bool(c1_verdict["gate_2c_pass"]),
        "per_seed_gated_pass": seed_rows,
        "c1_misses_now": resolved,
        "n_c1_misses_resolved": sum(1 for r in resolved if r["now_passes"]),
        "n_c1_misses_total": len(resolved),
        "newly_failing_cells": sorted(
            newly_failing, key=lambda r: (r["seed"], r["cell"])
        ),
    }


def fresh_run_artifact_schema(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    cell_index = sorted(tol)
    seed_index = [s["seed"] for s in per_seed]
    by_seed = {s["seed"]: s for s in per_seed}
    rates = [
        [
            [
                by_seed[s]["gated_cells"][c]["per_draw_rate"][k]
                for s in seed_index
            ]
            for c in cell_index
        ]
        for k in range(N_DRAWS)
    ]
    n_undefined = 0
    undefined = []
    for s in per_seed:
        for cell, rec in s["gated_cells"].items():
            if rec["undefined_draw_seeds"]:
                n_undefined += len(rec["undefined_draw_seeds"])
                undefined.append(
                    {
                        "cell": cell,
                        "seed": s["seed"],
                        "draw_seeds": rec["undefined_draw_seeds"],
                    }
                )
    dispersion = {}
    for s in per_seed:
        for cell, rec in s["gated_cells"].items():
            dispersion[f"{cell}|seed{s['seed']}"] = {
                "per_draw_rate_sd": rec["per_draw_rate_sd"],
                "max_per_draw_abs_ln": rec["max_per_draw_abs_ln"],
                "rbar": rec["rbar"],
            }
    return {
        "per_draw_per_cell_rates": {
            "required": True,
            "shape": [N_DRAWS, len(cell_index), len(seed_index)],
            "shape_dims": "K_draws x gated_cells x gate_seeds",
            "k_index_draw_seeds": [DRAW_SEED_BASE + k for k in range(N_DRAWS)],
            "cell_index": cell_index,
            "seed_index": seed_index,
            "rates": rates,
            "note": (
                "r[k][cell][seed]; rbar_candidate,s = mean over k; "
                "|ln(rbar / rate_a)| recomputes cell-by-cell"
            ),
        },
        "undefined_draw_rule": {
            "required": True,
            "pre_specified": True,
            "n_undefined_gated_draws": n_undefined,
            "undefined_gated_draws": undefined,
            "run_invalidated": bool(n_undefined > 0),
            "rule": (
                "if any gated cell's rate is undefined on any draw (empty "
                "simulated denominator) the run is invalidated; no draw is "
                "dropped, substituted, or re-rolled"
            ),
        },
        "per_draw_dispersion_disclosure": {
            "required": True,
            "gated": False,
            "report_only": True,
            "note": (
                "report-only: no dispersion cap gates the run; the disclosure "
                "shows whether a passing 20-draw mean conceals a wild draw"
            ),
            "cells": dispersion,
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(out_path: Path, verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    tol, report_only = load_locked_tolerances()
    floor = json.loads(FLOOR_RUN.read_text())
    c1_art = json.loads(C1_ARTIFACT.read_text())
    tolerance_check = _cross_check_tolerances(tol, floor)
    if not tolerance_check["tolerances_match_floor_draft_thresholds"]:
        raise RuntimeError(f"tolerance cross-check failed: {tolerance_check}")
    if not tolerance_check["tolerance_cells_equal_floor_gate_eligible"]:
        raise RuntimeError("gated cell set != floor gate_eligible")

    if verbose:
        print(f"loading data ({len(tol)} gated cells)...")
    data = load_all()

    if verbose:
        print("precheck (hard stop: reproduce the floor)...")
    precheck = run_precheck(data["ce_panel"], floor)

    per_seed = []
    for seed in GATE_SEEDS:
        if verbose:
            print(f"scoring seed {seed} (fit v2 side B, K={N_DRAWS} draws)...")
        rec = score_seed(seed, data, floor, tol, report_only)
        per_seed.append(rec)
        if verbose:
            mg = rec["marginal_preservation"]
            print(
                f"  seed {seed}: {rec['n_gated_pass']}/{len(tol)} gated pass, "
                f"seed_pass={rec['seed_pass']} "
                f"(constraint_dev="
                f"{mg['constraint_max_abs_dev_from_one_over_draws']:.1e}; "
                f"{rec['elapsed_seconds']}s)"
            )

    verdict = build_verdict(per_seed, tol)
    decomposition = per_family_decomposition(per_seed, tol)
    fresh_schema = fresh_run_artifact_schema(per_seed, tol)
    carry = byte_carry_regression(per_seed, c1_art)
    progression = c1_to_c2_progression(per_seed, c1_art, tol)

    if fresh_schema["undefined_draw_rule"]["run_invalidated"]:
        raise RuntimeError(
            "run invalidated: a gated cell was undefined on some draw "
            f"({fresh_schema['undefined_draw_rule']['n_undefined_gated_draws']}"
            " undefined gated draws)"
        )
    if not carry["carried_byte_identical"]:
        raise RuntimeError(
            "byte-carry violated: a carried family deviated from candidate 1 "
            f"(max {carry['carried_max_abs_rate_deviation']:.2e} at "
            f"{carry['worst_carried_cell_seed']})"
        )
    if not carry["delta_family_moved"]:
        raise RuntimeError(
            "the first_marriage_by_earnings delta did not move any cell"
        )

    meta = data["ce_panel"].meta
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": "gate2c_hazard_v2",
        "gate": "gate_2c",
        "candidate": "candidate 2",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "one_shot": (
            "one-shot scored run registered on issue #42 comment "
            f"{REGISTRATION_POINTER}; independent verification by rerun; "
            "publishes regardless; no holdout tuning"
        ),
        "pre_registered_forecast": PRE_REGISTERED_FORECAST,
        "spec_resolution_notes": SPEC_RESOLUTION_NOTES,
        "model": {
            "module": "populace_dynamics.models.couple_formation_sim_v2",
            "base_module": "populace_dynamics.models.couple_formation_sim_v1",
            "one_delta": (
                "earnings-conditioned first-marriage timing: a train-fitted "
                "multiplicative hazard modifier m(tercile | age band, sex) "
                "composed onto the certified 2a first-marriage hazard, "
                "normalized so the age x sex timing marginal is preserved"
            ),
            "five_components": {
                "1_marriage_events": (
                    "certified tranche-2a registry CANDIDATE_16 (refit side B,"
                    " simulate side A at 5200+k) + THE DELTA: the first-"
                    "marriage earnings modifier composed onto its hazard"
                ),
                "2_who_marries_whom": (
                    "train-fitted P(spouse decile | own decile, age band, "
                    "sex), directed both-orientation emission (byte-carried)"
                ),
                "3_spouse_age": (
                    "certified spousal_age_gaps + draw_spousal_gaps "
                    "(recorded; not gated) (byte-carried)"
                ),
                "4_event_window_dynamics": (
                    "train around-event post/pre ratio shift kernels, "
                    "placebo-detrended by the committed deflator (byte-carried)"
                ),
                "5_shared_earnings_cells": (
                    "combined-axis cutpoint ratios per the locked cell "
                    "definitions (byte-carried)"
                ),
            },
            "delta_modifier": {
                "conditioning": "m(tercile | age band, sex)",
                "form": (
                    "residual: real_train_hazard / certified_expected_hazard "
                    "(the certified 2a hazard READ, not re-simulated)"
                ),
                "shrinkage_alpha": cfs.MODIFIER_SHRINKAGE_ALPHA,
                "normalization": (
                    "sum_t m * phi_cert = 1 per (age band, sex), phi_cert = "
                    "certified expected first-marriage event share by tercile"
                ),
                "marginal_preserved": "age x sex first-marriage timing",
                "gated_marginal_bands": list(cfs.GATED_MARGINAL_BANDS),
                "applied_as": (
                    "deterministic reweighting of first_marriage event "
                    "weights after _build_marital (no RNG)"
                ),
            },
            "kernel_smoothing_alpha": cfs.KERNEL_SMOOTHING_ALPHA,
            "n_deciles": cfs.N_DECILES,
            "certified_spec_sha256": cfs.CERTIFIED_SPEC.sha256,
            "certified_component_implementation_ids": dict(
                per_seed[0]["fit_meta"]["component_implementation_ids"]
            ),
        },
        "protocol": {
            "option": "a",
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "outer_seed_provenance_only": (
                "numpy.random.default_rng(4200 + seed) recorded, not consumed"
            ),
            "split": (
                "COUPLE-DISJOINT 50/50 by attrs.component_id, "
                "default_rng(seed); side A = holdout, side B = train"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "statistic": (
                "|ln(rbar_candidate,s / rate_a,s)|, rbar the 20-draw mean "
                "rate, scored once (NOT the mean of per-draw scores)"
            ),
            "pass_rule": (
                "seed passes iff all 27 gated cells hold; gate passes iff "
                ">= 4 of 5 seeds pass"
            ),
            "tolerance_source": "gates.yaml gate_2.gate_2c (LOCKED)",
            "tolerance_cross_check_vs_floor": tolerance_check,
        },
        "fresh_run_artifact_schema": fresh_schema,
        "byte_carry_regression": carry,
        "c1_to_c2_progression": progression,
        "data": {
            "holdout_basis": floor["holdout_basis"],
            "floor_run": "runs/gate2c_floors_v1.json",
            "floor_run_sha256": _sha256_file(FLOOR_RUN),
            "n_directed_couples": meta["n_directed_couples"],
            "n_earnings_supply_persons": meta["n_earnings_supply_persons"],
            "pe_us_revision": meta.get("pe_us_revision"),
        },
        "precheck": precheck,
        "per_seed": per_seed,
        "verdict": verdict,
        "per_family_decomposition": decomposition,
        "revision_pins": {
            "artifact_schema_version": SCHEMA_VERSION,
            "pe_us_revision": meta.get("pe_us_revision"),
            "certified_spec_sha256": cfs.CERTIFIED_SPEC.sha256,
            "floor_run_sha256": _sha256_file(FLOOR_RUN),
            "c1_artifact_sha256": _sha256_file(C1_ARTIFACT),
            "base_sha": floor.get("revision_pins", {}).get(
                "origin_master_sha"
            ),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    artifact = _json_safe(artifact)
    artifacts.write_new(out_path, artifact)
    if verbose:
        v = verdict
        print(
            f"\nGATE-2C candidate 2: {v['n_seeds_pass']}/5 seeds pass "
            f"(gate_2c_pass={v['gate_2c_pass']}); "
            f"carried byte-identical={carry['carried_byte_identical']}; "
            f"wrote {out_path} ({artifact['elapsed_seconds']}s)"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
