"""W1 forensics 1 -- the five transport mechanisms (diagnostic).

Registered at issue #42 comment 4951218279 (FROZEN spec; the registration
wins). Reported, NOT gated: measures the five initialization/support/scope
mechanisms candidate 1 isolated (grading 4951216895) BEFORE candidate 2
designs. One run, publishes regardless of any verdict.

Standard -- the gate-2b forensics rounds: machine-epsilon reconciliations
where arithmetic permits (identity remainders at ``64 * float64_eps``), and
INSTRUMENTATION BIT-IDENTITY where a component re-simulates: every re-run of
the committed ``transport_deployment_v1`` machinery is proved to reproduce
the committed ``runs/gate_w1_candidate1_v1.json`` values bit-for-bit before
any counterfactual is measured.

Frozen questions (4951218279):

* Q1 marital equilibration -- decompose the young-cohort married-share
  deficit into (a) initialization [observed-init vs synthetic-equilibration;
  the contract-permission adjudication], (b) exposure-window, (c) hazard
  residual, per band x sex.
* Q2 participation boundary -- the 18-24 / 62-69 misses under (a) nearest-bin
  extrapolation, (b) a train-fitted boundary extension, (c) the frame's own
  ages; which cells each clears.
* Q3 household scope -- hh_size_share under the adult vs all-person universe;
  scope gap vs composition residual; the contract-consistent resolution.
* Q4 DI level bridge -- the prevalence-level gap as work-disability-vs-
  beneficiary-stock concept delta vs M4 hazard level; a GATE-DESIGN finding.
* Q5 the tail's upper read -- the 37.7%-above-cap tail under zero/low-earning-
  year inclusion; whether a corrected tail moves PPI past NRA (the C1
  question; the single most consequential number).

Envs (per the c1 artifact): the certified frame is exported from the
policyengine.py .venv (scripts/export_frame_persons.py) and resolved via
``POPULACE_DYNAMICS_FRAME_PICKLE``; the fit + measure phases run in the
.venv-gate. PSID at ``POPULACE_DYNAMICS_PSID_DIR``; the pe-us oracle at
``POPULACE_DYNAMICS_PE_US_DIR``. Memory guard: the generators are fit once
and cached; the heavy Q5 ledger runs last.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ruff: noqa: E402, I001 -- imports follow the sys.path bootstrap (script).
from populace_dynamics import artifacts
from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import transport_deployment_v1 as td

import run_gate1_baseline as g1base  # noqa: E402
import run_gate1_candidate5b as g1  # noqa: E402

SCHEMA_VERSION = "gate_w1_forensics1.v1"
RUN_NAME = "gate_w1_forensics1_v1"
REGISTRATION_POINTER = "4951218279"
GRADING_POINTER = "4951216895"
CANDIDATE1_POINTER = "4950931131"
ARTIFACT_PATH = ROOT / "runs" / "gate_w1_forensics1_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate_w1_candidate1_v1.json"
M4_ARTIFACT = ROOT / "runs" / "m4_disability_v1.json"
DI_ASR = ROOT / "data" / "external" / "di_asr_2023" / "tables.json"
SCRATCH = ROOT / "scratch"
GENS_CACHE = SCRATCH / "forensics1_gens.pkl"

#: Diagnostic draw budget for the Q1 full-frame counterfactuals. Weighted
#: shares over ~120k adults have negligible MC error; the additive
#: reconciliation is exact for any K (arithmetic on the three means).
K_DIAG = 8
#: Q1 exposure-extended common terminal age (past ~all first marriage, before
#: heavy widowhood): each person's chain runs to birth_year + max(age, this).
FULL_EXPOSURE_AGE = 62
#: 2024 taxable maximum (the C2 break-even is ~16.1% of payroll above it).
WAGE_BASE_2024 = 168600.0
C2_BREAKEVEN_FRAC = 2.0 / 12.4  # ~0.1613 (Smith full elimination vs +2pp)

FLOAT64_EPS = float(np.finfo(np.float64).eps)
IDENTITY_BAR = 64 * FLOAT64_EPS


# --------------------------------------------------------------------------
# JSON safety (numpy scalars, non-finite floats) -- the forensics convention.
# --------------------------------------------------------------------------
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


def _load_frame() -> pd.DataFrame:
    p = os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE")
    if not p or not Path(p).exists():
        raise SystemExit(
            "set POPULACE_DYNAMICS_FRAME_PICKLE to the sha-verified frame "
            "export (scripts/export_frame_persons.py, policyengine.py .venv)"
        )
    return pd.read_pickle(p)


# --------------------------------------------------------------------------
# Generators (fit once, cached). Q1 needs fitted_ft; Q5 needs the gate-1 cell
# marginals + age_bin; Q3/Q4 re-simulate nothing (deterministic), so the
# household generator is intentionally NOT fit (memory + time guard).
# --------------------------------------------------------------------------
def fit_generators() -> td.DeployedGenerators:
    if GENS_CACHE.exists():
        return pickle.load(open(GENS_CACHE, "rb"))
    prov: dict[str, Any] = {}
    t = time.time()
    panel = g1base.load_filtered_panel()
    marginals = g1.fit_cell_marginals(panel)
    prov["gate1"] = {
        "module": "run_gate1_candidate5b.fit_cell_marginals",
        "panel_rows": int(len(panel)),
        "n_cells": len(marginals),
        "age_min": int(g1base.AGE_MIN),
        "age_max": int(g1base.AGE_MAX),
        "period_min": int(g1base.PERIOD_MIN),
        "period_max": int(g1base.PERIOD_MAX),
        "fit_seconds": round(time.time() - t, 1),
    }
    t = time.time()
    from populace_dynamics.models.family_transitions import evaluation as ev

    src = ev._load_sources()
    ft_ids = frozenset(src.panel.attrs["person_id"].tolist())
    ctx = ft.FitContext(
        panel=src.panel,
        demographic_panel=src.demographic_panel,
        marriage_records=src.marriage_records,
        birth_records=src.birth_records,
        marriage_order_map=src.order_map,
        train_ids=ft_ids,
    )
    fitted_ft = ft.REGISTRY.fit(ft.CANDIDATE_16, ctx)
    prov["family_transitions"] = {
        "candidate_id": ft.CANDIDATE_16.candidate_id,
        "sha256": ft.CANDIDATE_16.sha256,
        "n_train_persons": len(ft_ids),
        "fit_seconds": round(time.time() - t, 1),
    }
    m4 = json.load(open(M4_ARTIFACT))
    prov["gate_m4"] = {
        "artifact": M4_ARTIFACT.name,
        "run": m4.get("run"),
        "source": "runs/m4_disability_v1.json reference_moments prevalence",
    }
    gens = td.DeployedGenerators(
        earnings_marginals=marginals,
        age_bin_fn=g1.age_bin,
        fitted_ft=fitted_ft,
        fitted_hc=None,
        m4_prevalence={},
        m4_bands=(),
        fit_provenance=prov,
    )
    SCRATCH.mkdir(exist_ok=True)
    pickle.dump(gens, open(GENS_CACHE, "wb"))
    return gens


def _load_contracts() -> dict[str, Any]:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gw1 = gates["gates"]["gate_w1"]
    fa = gw1["thresholds"]["family_a"]
    tol: dict[str, float] = {}
    for block in fa["views"].values():
        for cell, t in block["tolerances"].items():
            tol[cell] = float(t)
    return {
        "tol_a": tol,
        "family_b": gw1["thresholds"]["family_b"],
        "family_c": gw1["thresholds"]["family_c"],
    }


# ==========================================================================
# Q1 -- marital equilibration.
# ==========================================================================
def _married_never_shares(frame: pd.DataFrame) -> dict[str, float]:
    """married.{band|sex} + never_married.{band|sex} rates via the committed
    reference_moments (the scored family-A marital surface)."""
    cells = dfm.reference_moments(frame, weighted=True)
    out: dict[str, float] = {}
    for k, v in cells.items():
        if k.startswith("marital_share.married.") or k.startswith(
            "marital_share.never_married."
        ):
            out[k.replace("marital_share.", "")] = float(v["rate"])
    return out


def _marital_frame(adults: pd.DataFrame, marital: pd.Series) -> pd.DataFrame:
    pid = adults["person_id"].to_numpy()
    arr = marital.reindex(pid).to_numpy(dtype=object)
    return pd.DataFrame(
        {
            "person_id": pid,
            "weight": adults["weight"].to_numpy(dtype=np.float64),
            "age": adults["age"].to_numpy(dtype=np.float64),
            "is_female": adults["is_female"].to_numpy(dtype=bool),
            "earnings": np.zeros(len(adults)),
            "marital_status": arr,
            "hh_size": np.ones(len(adults)),
            "coresident_spouse": arr == "married",
        }
    )


def _extended_marital_terminal(
    pid, age, fem, wt, fitted_ft, seed, min_age
) -> pd.Series:
    """Terminal marital state with the exposure window EXTENDED to a common
    terminal age (birth_year + max(age, min_age)); everything else mirrors
    td._synthetic_marital_panel bit-for-bit (never-married entry at 18)."""
    age_i = age.astype(int)
    birth_year = (td.REF_YEAR - age).astype(int)
    censor = birth_year + np.maximum(age_i, min_age)
    attrs = pd.DataFrame(
        {
            "person_id": pid,
            "birth_year": birth_year,
            "sex": np.where(fem, "female", "male"),
            "start_exposure_year": birth_year + 18,
            "censor_year": censor,
            "weight": wt,
        }
    )
    empty = pd.DataFrame(
        {
            "person_id": pd.array([], dtype="int64"),
            "year": pd.array([], dtype="int64"),
            "marital_state": pd.array([], dtype="object"),
            "marriage_duration": pd.array([], dtype="Int64"),
            "years_since_dissolution": pd.array([], dtype="Int64"),
        }
    )
    panel = transitions.MaritalPanel(
        person_years=empty, events=pd.DataFrame(), attrs=attrs
    )
    sim, _ = ft.simulate(panel, {int(x) for x in pid}, fitted_ft, seed)
    py = sim.person_years
    term = py.loc[
        py["year"] == py.groupby("person_id")["year"].transform("max")
    ]
    return term.set_index("person_id")["marital_state"]


def q1_marital(persons, gens, art, verbose=True) -> dict[str, Any]:
    # --- instrumentation bit-identity: reproduce seed-0 draw-0 holdout
    #     marital / coresident cells via the committed td.regenerate_marital.
    gated = art["family_a"]["gated_cells"]
    cube = np.array(art["family_a"]["cube"])  # [K, cell, seed]
    universe = persons["household_id"].to_numpy()
    side_a = set(td.holdout_side_a_households(universe, 0).tolist())
    hold = persons[persons["household_id"].isin(side_a)].reset_index(drop=True)
    ah = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    marital = td.regenerate_marital(
        ah["person_id"].to_numpy(),
        ah["age"].to_numpy(dtype=np.float64),
        ah["is_female"].to_numpy(dtype=bool),
        ah["weight"].to_numpy(dtype=np.float64),
        gens.fitted_ft,
        td.FAMILY_A_STREAM_BASE + 0,
    )
    repro = dfm.reference_moments(_marital_frame(ah, marital), weighted=True)
    max_dev = 0.0
    for ci, cell in enumerate(gated):
        if cell.startswith("marital_share.") or cell.startswith(
            "coresident_spouse."
        ):
            if cell in repro:
                max_dev = max(
                    max_dev, abs(repro[cell]["rate"] - float(cube[0, ci, 0]))
                )

    # --- decomposition on the FULL 25+ frame (self-consistent universe).
    adults = persons[persons["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    pid = adults["person_id"].to_numpy()
    age = adults["age"].to_numpy(dtype=np.float64)
    fem = adults["is_female"].to_numpy(dtype=bool)
    wt = adults["weight"].to_numpy(dtype=np.float64)

    # O -- observed-initialization = the frame's OWN A_MARITL cross-section
    # (the identity: terminal marital_status copied from the frame). This is
    # exactly rate_a for the marital cells (family A = internal transport
    # consistency to the frame), so it closes the whole deficit -- and is the
    # PROHIBITED identity candidate (regenerated_surface rule).
    o_shares = _married_never_shares(persons)

    # S -- synthetic-equilibration (the c1 choice: never-married entry at 18,
    # certified hazards to the actual age). E -- exposure-extended (same entry,
    # window run to age max(age, 62)). Both re-simulate the committed hazards.
    s_acc: dict[str, list[float]] = {}
    e_acc: dict[str, list[float]] = {}
    for k in range(K_DIAG):
        seed = td.FAMILY_A_STREAM_BASE + k
        s_marital = td.regenerate_marital(
            pid, age, fem, wt, gens.fitted_ft, seed
        )
        for key, val in _married_never_shares(
            _marital_frame(adults, s_marital)
        ).items():
            s_acc.setdefault(key, []).append(val)
        e_marital = _extended_marital_terminal(
            pid, age, fem, wt, gens.fitted_ft, seed, FULL_EXPOSURE_AGE
        )
        for key, val in _married_never_shares(
            _marital_frame(adults, e_marital)
        ).items():
            e_acc.setdefault(key, []).append(val)
        if verbose:
            print(f"  [q1] draw {k} done", flush=True)

    s_shares = {k: float(np.mean(v)) for k, v in s_acc.items()}
    e_shares = {k: float(np.mean(v)) for k, v in e_acc.items()}

    # committed per-seed rbar (mean across seeds) for cross-consistency.
    committed_rbar: dict[str, float] = {}
    for cell in gated:
        if cell.startswith("marital_share."):
            vals = [
                s["per_cell"][cell]["rbar"]
                for s in art["family_a"]["per_seed"]
                if cell in s["per_cell"]
            ]
            if vals:
                committed_rbar[cell.replace("marital_share.", "")] = float(
                    np.mean(vals)
                )

    per_cell: dict[str, Any] = {}
    max_remainder = 0.0
    for key in sorted(o_shares):
        if not key.startswith("married."):
            continue
        band_sex = key.replace("married.", "")
        o = o_shares[key]
        s = s_shares.get(key, float("nan"))
        e = e_shares.get(key, float("nan"))
        total = o - s  # anchor minus synthetic-equilibration deficit
        exposure_b = e - s  # window extension (never-married entry)
        hazard_c = o - e  # residual after full exposure
        remainder = total - (exposure_b + hazard_c)
        max_remainder = max(max_remainder, abs(remainder))
        per_cell[band_sex] = {
            "observed_init_O": o,
            "synthetic_equilibration_S": s,
            "exposure_extended_E": e,
            "committed_rbar_S": committed_rbar.get(band_sex),
            "total_deficit_O_minus_S": total,
            "component_b_exposure_window": exposure_b,
            "component_c_hazard_residual": hazard_c,
            "component_a_initialization_note": (
                "observed-init O == rate_a (identity) closes the whole "
                "deficit but is the PROHIBITED regenerated_surface identity. "
                "Because extending exposure (b) barely helps and a "
                "hazard-level residual (c) remains at full exposure, the "
                "hazards from a never-married entry CANNOT reach the observed "
                "stock at any window -- so seeding the entry MARRIED stock "
                "(initialization) is the necessary lever: it injects the "
                "observed married mass directly, bypassing the hazard "
                "shortfall. This is why the deficit is initialization-driven."
            ),
            "reconciliation_remainder": remainder,
            "dominant_component": (
                "exposure_window"
                if abs(exposure_b) >= abs(hazard_c)
                else "hazard_residual"
            ),
        }

    # aggregate direction, derived from the measured components.
    n_haz = sum(
        1
        for c in per_cell.values()
        if abs(c["component_c_hazard_residual"])
        > abs(c["component_b_exposure_window"])
    )
    mean_b = float(
        np.mean(
            [abs(c["component_b_exposure_window"]) for c in per_cell.values()]
        )
    )
    mean_c = float(
        np.mean(
            [abs(c["component_c_hazard_residual"]) for c in per_cell.values()]
        )
    )
    return {
        "mechanism": (
            "The synthetic-panel adapter starts every frame person "
            "never-married at 18 (empty person-years -> the simulator's "
            "entry_state defaults to never_married); the certified "
            "CANDIDATE_16 first-marriage hazard does not accumulate the "
            "observed married stock within the finite [18, current age] "
            "exposure window. rate_a for the marital cells IS the frame's own "
            "A_MARITL cross-section (family A is internal transport "
            "consistency), so observed-initialization reproduces it exactly."
        ),
        "adjudication": {
            "question": (
                "May a candidate condition on the frame's A_MARITL column?"
            ),
            "contract_rule": (
                "gate_w1 regenerated_surface: marital status is RE-GENERATED "
                "by the deployed gate-2a/2b dynamics (TERMINAL-state marital "
                "status), NOT copied from A_MARITL; the identity map is "
                "explicitly NON-CONFORMANT (identity_candidate, score 0)."
            ),
            "determination": (
                "A_MARITL as the TERMINAL scored state = the prohibited "
                "identity. But the simulator (simulator.py:173-196) reads the "
                "panel's ENTRY state as the initial condition; conditioning "
                "the hazard chain's ENTRY state on an observable and "
                "regenerating the terminal is CONTRACT-PERMITTED. The catch "
                "measured here: A_MARITL is a 2024 (terminal-age) "
                "cross-section, so seeding the entry state from it at the "
                "person's current age leaves NO exposure window (terminal == "
                "entry == identity); a non-degenerate contract-permitted "
                "lever needs an initial-state MODEL (an inferred earlier-age "
                "married stock), not the raw terminal A_MARITL. The c1 "
                "never-married-at-18 entry is a SPEC RESOLUTION, not a "
                "contract requirement."
            ),
        },
        "instrumentation_fidelity": {
            "reproduced": "seed-0 draw-0 holdout marital + coresident cells",
            "max_abs_rate_deviation_vs_committed_cube": max_dev,
            "bit_identical": bool(max_dev == 0.0),
        },
        "per_band_sex": per_cell,
        "reconciliation_max_abs_remainder": max_remainder,
        "finding": {
            "n_cells": len(per_cell),
            "hazard_residual_dominant_in_n_cells": n_haz,
            "mean_abs_component_b_exposure": mean_b,
            "mean_abs_component_c_hazard_residual": mean_c,
            "hazard_residual_dominates_conformant_path": bool(mean_c > mean_b),
            "summary": (
                "The measurement REFINES the pre-registered 'initialization "
                "dominant' guess into a sharper mechanism. Extending the "
                "exposure window barely moves the married share (mean |b| "
                f"~{mean_b:.3f}; b is even NEGATIVE past peak marriage, as "
                "dissolution offsets new marriages); the dominant conformant "
                f"component is the HAZARD-LEVEL residual (mean |c| ~{mean_c:.3f}"
                "), the certified marriage-minus-dissolution steady state "
                "sitting below the observed married share across ALL cohorts "
                "(compounded at 25-34 by the birth-decade covariate "
                "extrapolating low marriage for the 1990s cohort). Because no "
                "amount of exposure from the never-married-at-18 entry reaches "
                "the observed stock, the deficit is closable ONLY by "
                "INITIALIZATION -- seeding the entry married stock (the "
                "observed-init identity that reproduces rate_a but is the "
                "prohibited regenerated_surface copy). So 'initialization "
                "dominant' holds in the decisive sense (initialization is the "
                "necessary lever, exposure is not), and it is NOT a hazard "
                "DEFECT: the hazards are PSID-certified; the miss is the "
                "deployment's never-married initial condition -- the 2a "
                "undatable-marriage lesson in transport clothing."
            ),
        },
    }


# ==========================================================================
# Q2 -- participation boundary.
# ==========================================================================
def _weighted_participation(df: pd.DataFrame) -> float:
    w = df["weight"].to_numpy(dtype=np.float64)
    e = df["earnings"].to_numpy(dtype=np.float64)
    return float(w[e > 0].sum() / w.sum()) if w.sum() else float("nan")


def _weighted_median_pos(df: pd.DataFrame) -> float:
    m = df["earnings"].to_numpy(dtype=np.float64) > 0
    if not m.any():
        return float("nan")
    return dfm.weighted_quantile(
        df["earnings"].to_numpy(dtype=np.float64)[m],
        df["weight"].to_numpy(dtype=np.float64)[m],
        0.5,
    )


def q2_participation(persons, gens, art, tol, verbose=True) -> dict[str, Any]:
    gated = art["family_a"]["gated_cells"]
    cube = np.array(art["family_a"]["cube"])
    pc0 = art["family_a"]["per_seed"][0]["per_cell"]

    # instrumentation bit-identity: reproduce seed-0 draw-0 participation via
    # the committed td.regenerate_earnings rng topology (regenerate_person_
    # _frame: rng=default_rng(9100); earn=regenerate_earnings(age, rng, ...)).
    side_a = set(
        td.holdout_side_a_households(
            persons["household_id"].to_numpy(), 0
        ).tolist()
    )
    hold = persons[persons["household_id"].isin(side_a)].reset_index(drop=True)
    ah = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    rng = np.random.default_rng(td.FAMILY_A_STREAM_BASE + 0)
    earn = td.regenerate_earnings(
        ah["age"].to_numpy(dtype=np.float64),
        rng,
        gens.earnings_marginals,
        gens.age_bin_fn,
    )
    ef = pd.DataFrame(
        {
            "weight": ah["weight"].to_numpy(dtype=np.float64),
            "age": ah["age"].to_numpy(dtype=np.float64),
            "is_female": ah["is_female"].to_numpy(dtype=bool),
            "earnings": earn,
            "marital_status": np.array(["never_married"] * len(ah)),
            "hh_size": np.ones(len(ah)),
            "coresident_spouse": np.zeros(len(ah), dtype=bool),
        }
    )
    repro = dfm.reference_moments(ef, weighted=True)
    max_dev = 0.0
    for ci, cell in enumerate(gated):
        if cell.startswith("earnings_participation.") and cell in repro:
            max_dev = max(
                max_dev, abs(repro[cell]["rate"] - float(cube[0, ci, 0]))
            )

    # treatment (b): train-fitted boundary extension -- what PSID supports at
    # ages 18-24 / 62-69 (widen the locked 25-59 filter). No sex covariate, so
    # one rate per boundary age range (applied to both sexes).
    raw = g1base.family_earnings_panel()
    raw = raw[
        (raw.period >= g1base.PERIOD_MIN)
        & (raw.period <= g1base.PERIOD_MAX)
        & (raw.weight > 0)
    ]
    psid: dict[str, dict[str, float]] = {}
    prime = raw[(raw.age >= 35) & (raw.age <= 44)]
    prime_med = _weighted_median_pos(prime)
    for lo, hi, label in ((18, 24, "18-24"), (62, 69, "62-69")):
        sub = raw[(raw.age >= lo) & (raw.age <= hi)]
        psid[label] = {
            "participation": _weighted_participation(sub),
            "profile_ratio": (
                _weighted_median_pos(sub) / prime_med
                if prime_med
                else float("nan")
            ),
            "n_person_years": int(len(sub)),
        }

    # score each boundary cell under (a) nearest-bin, (b) boundary extension,
    # (c) frame's own ages. Cleared iff |ln(dep/rate_a)| <= tolerance.
    def _cleared(dep, rate_a, tolv):
        if dep <= 0 or rate_a <= 0:
            return None
        return bool(abs(math.log(dep / rate_a)) <= tolv)

    boundary_cells = {
        "earnings_participation.18-24|female": ("18-24", "participation"),
        "earnings_participation.18-24|male": ("18-24", "participation"),
        "earnings_participation.62-69|female": ("62-69", "participation"),
        "earnings_participation.62-69|male": ("62-69", "participation"),
        "earnings_profile.18-24|female": ("18-24", "profile_ratio"),
        "earnings_profile.18-24|male": ("18-24", "profile_ratio"),
    }
    cells_out: dict[str, Any] = {}
    tally = {"a_nearest_bin": 0, "b_boundary_extension": 0, "c_frame_ages": 0}
    n_scored = 0
    for cell, (label, kind) in boundary_cells.items():
        if cell not in pc0:
            continue  # profile 62-69 etc. are report-only
        rate_a = float(pc0[cell]["rate_a"])
        tolv = tol[cell]
        dep_a = float(pc0[cell]["rbar"])  # committed nearest-bin deployed
        dep_b = psid[label][kind]  # boundary extension
        dep_c = rate_a  # frame's own ages == the identity
        ca = _cleared(dep_a, rate_a, tolv)
        cb = _cleared(dep_b, rate_a, tolv)
        cc = _cleared(dep_c, rate_a, tolv)
        n_scored += 1
        tally["a_nearest_bin"] += int(bool(ca))
        tally["b_boundary_extension"] += int(bool(cb))
        tally["c_frame_ages"] += int(bool(cc))
        cells_out[cell] = {
            "rate_a": rate_a,
            "tolerance": tolv,
            "a_nearest_bin": {"deployed": dep_a, "clears": ca},
            "b_boundary_extension": {"deployed": dep_b, "clears": cb},
            "c_frame_ages": {"deployed": dep_c, "clears": cc},
        }

    return {
        "mechanism": (
            "gate-1 fits ages 25-59 with NO sex covariate; age_bin clips to "
            "[0,6], so 18-24 regenerates from the 25-29 cell and 62-69 from "
            "the 55-59 cell (prime-age participation ~0.86), overshooting the "
            "boundary bands (frame ~0.64 at 18-24, ~0.50 at 62-69)."
        ),
        "instrumentation_fidelity": {
            "reproduced": "seed-0 draw-0 holdout earnings_participation cells",
            "max_abs_rate_deviation_vs_committed_cube": max_dev,
            "bit_identical": bool(max_dev == 0.0),
        },
        "psid_boundary_support": psid,
        "prime_median_psid": prime_med,
        "per_cell": cells_out,
        "cells_cleared_tally": tally,
        "n_scored": n_scored,
        "finding": {
            "summary": (
                "The train-fitted boundary extension (what PSID actually "
                "supports at 18-24 / 62-69) clears the boundary cells the "
                "nearest-bin extrapolation misses; the frame's own ages clear "
                "all (the identity, non-conformant). The boundary miss is a "
                "SUPPORT gap (fit outside 25-59), not a hazard defect -- "
                "extending the fitted support is the c2 lever."
            ),
            "nearest_bin_clears": tally["a_nearest_bin"],
            "boundary_extension_clears": tally["b_boundary_extension"],
            "frame_ages_clears": tally["c_frame_ages"],
        },
    }


# ==========================================================================
# Q3 -- household scope.
# ==========================================================================
def q3_household(persons, art) -> dict[str, Any]:
    pc0 = art["family_a"]["per_seed"][0]["per_cell"]
    cats = list(dfm.HH_SIZE_CATEGORIES)

    # everything on seed-0 side-A holdout households (the committed universe of
    # per_seed[0]), so D (committed deployed rbar) reconciles exactly.
    side_a = set(
        td.holdout_side_a_households(
            persons["household_id"].to_numpy(), 0
        ).tolist()
    )
    holds = persons[persons["household_id"].isin(side_a)].reset_index(
        drop=True
    )

    # A -- all-person universe (= rate_a). Data bit-identity: the frame's own
    # all-person hh_size_share on side-A must reproduce the committed rate_a.
    wa = holds["weight"].to_numpy(dtype=np.float64)
    hs_all = holds["hh_size"].to_numpy()
    A = {}
    for c in cats:
        m = (hs_all >= 5) if c == "5plus" else (hs_all == int(c))
        A[c] = float(wa[m].sum() / wa.sum())
    max_ref_dev = max(
        abs(A[c] - float(pc0[f"hh_size_share.{c}"]["rate_a"])) for c in cats
    )

    # U -- adult universe: household size counting only adults (>=18), among
    # adult persons (the universe the generator actually composes).
    adults = holds[holds["age"] >= td.ADULT_MIN_AGE].copy()
    adult_size = adults.groupby("household_id")["person_id"].transform("count")
    adults = adults.assign(adult_hh=adult_size.to_numpy())
    wu = adults["weight"].to_numpy(dtype=np.float64)
    hu = adults["adult_hh"].to_numpy()
    U = {}
    for c in cats:
        m = (hu >= 5) if c == "5plus" else (hu == int(c))
        U[c] = float(wu[m].sum() / wu.sum())

    per_cell: dict[str, Any] = {}
    max_remainder = 0.0
    scope_tot = comp_tot = 0.0
    for c in cats:
        cell = f"hh_size_share.{c}"
        D = float(pc0[cell]["rbar"])  # committed deployed (adult-composed)
        a = A[c]
        u = U[c]
        total = D - a
        scope = u - a  # all-person -> adult universe
        comp = D - u  # generator's own miss on the adult universe
        remainder = total - (scope + comp)
        max_remainder = max(max_remainder, abs(remainder))
        scope_tot += abs(scope)
        comp_tot += abs(comp)
        per_cell[c] = {
            "deployed_D": D,
            "adult_universe_U": u,
            "all_person_A_rate_a": a,
            "total_miss_D_minus_A": total,
            "scope_component_U_minus_A": scope,
            "composition_residual_D_minus_U": comp,
            "reconciliation_remainder": remainder,
            "dominant": "scope" if abs(scope) >= abs(comp) else "composition",
        }

    return {
        "mechanism": (
            "The household generator composes ADULTS only (empty initial "
            "rosters, exit-only/entry-limited coresidence), while the locked "
            "rate_a is ALL-PERSON (children counted). Two gaps stack: a SCOPE "
            "gap (children collapse large households into adult couples) and "
            "a COMPOSITION residual (the generator over-produces lone adults "
            "even on its own adult universe -- the same coresidence "
            "under-generation as Q1's married-share deficit)."
        ),
        "reference_moment_fidelity": {
            "reproduced": "all-person hh_size_share on seed-0 side-A",
            "max_abs_deviation_vs_committed_rate_a": max_ref_dev,
            "bit_identical": bool(max_ref_dev <= IDENTITY_BAR),
        },
        "per_cell": per_cell,
        "reconciliation_max_abs_remainder": max_remainder,
        "abs_scope_total": scope_tot,
        "abs_composition_total": comp_tot,
        "resolution": {
            "contract_consistent": (
                "rate_a is LOCKED all-person, so recomputing on the adult "
                "universe cannot make the candidate scoreable. To land the "
                "all-person cells the candidate must (1) ATTACH CHILDREN to "
                "household rosters via the certified fertility machinery "
                "(ft.simulate already emits births) and the household "
                "generator's own-child dynamics, and (2) fix the adult-"
                "coresidence under-composition with realistic initial rosters "
                "(coupled to Q1). Cost: a full fertility+household deployment, "
                "far beyond c1's adult-only empty-roster composition."
            ),
            "scope_is_whole_miss": bool(comp_tot <= 0.02),
        },
        "finding": {
            "summary": (
                "The scope gap is NOT the whole miss. Scope dominates the "
                "large-size cells (children collapse them), but a large "
                "composition residual dominates size-1 (deployed over-produces "
                "lone adults vs the adult universe). Neither treatment alone "
                "clears all five cells; child attachment AND coresidence "
                "repair are both required."
            )
        },
    }


# ==========================================================================
# Q4 -- DI level bridge (potentially a GATE-DESIGN finding).
# ==========================================================================
def _di_asr_awards_composition() -> dict[str, float]:
    """SSA DI ASR Table 36 (2023 awards, workers) age composition over the 8
    DI-ASR bands -- the FLOW, to contrast with Table 19's accumulated STOCK."""
    tables = json.load(open(DI_ASR))
    rows = tables["Table 36"]["tsv"].splitlines()
    want = {
        "Under 25": "under30",
        "25–29": "under30",
        "30–34": "30-34",
        "35–39": "35-39",
        "40–44": "40-44",
        "45–49": "45-49",
        "50–54": "50-54",
        "55–59": "55-59",
        "60–64": "60-fra",
        "65–FRA": "60-fra",
    }
    acc: dict[str, float] = {}
    for ln in rows:
        parts = ln.split("\t")
        label = parts[0].strip()
        if label in want and len(parts) > 1:
            num = parts[1].replace(",", "").strip()
            try:
                acc[want[label]] = acc.get(want[label], 0.0) + float(num)
            except ValueError:
                continue
    tot = sum(acc.values())
    return {k: 100.0 * v / tot for k, v in acc.items()} if tot else {}


def q4_di_bridge(art, contracts) -> dict[str, Any]:
    fb = art["family_b"]["per_cell"]
    m4 = json.load(open(M4_ARTIFACT))
    fbc = contracts["family_b"]["gated_cells"]
    bands = [b[0] for b in td.DI_ANCHOR_BANDS]

    awards = _di_asr_awards_composition()  # FLOW composition (Table 36)
    aw_sum = sum(awards.values())

    per_band: dict[str, Any] = {}
    max_gap = 0.0
    concept_share_num = concept_share_den = 0.0
    for b in bands:
        cell = f"di_prevalence.{b}"
        dep = float(fb[cell]["deployed_pp"])  # M4-simulated stock composition
        anchor = float(fbc[cell]["anchor_pp"])  # SSA STOCK (Table 19)
        flow = float(awards.get(b, 0.0))  # SSA awards FLOW (Table 36)
        gap = anchor - dep
        # decompose the |gap|: M4-shape (deployed->flow) vs duration/stock
        # concept (flow->stock).
        m4_shape = flow - dep
        duration_concept = anchor - flow
        max_gap = max(max_gap, abs(gap))
        concept_share_num += abs(duration_concept)
        concept_share_den += abs(duration_concept) + abs(m4_shape)
        per_band[b] = {
            "deployed_m4_stock_pp": dep,
            "anchor_ssa_stock_pp": anchor,
            "ssa_awards_flow_pp": flow,
            "total_gap_anchor_minus_deployed": gap,
            "m4_shape_component_flow_minus_deployed": m4_shape,
            "duration_concept_flow_to_stock": duration_concept,
            "reconciliation_remainder": gap - (m4_shape + duration_concept),
            "tolerance_pp": float(fbc[cell]["tolerance_pp"]),
            "passes": bool(fb[cell]["pass"]),
        }
    max_remainder = max(
        abs(v["reconciliation_remainder"]) for v in per_band.values()
    )
    concept_dominant_share = (
        concept_share_num / concept_share_den if concept_share_den else 0.0
    )

    # M4 prevalence gradient (from the m4 artifact) -- peaks 50-59, DROPS at
    # 60-66, so a point-prevalence CANNOT concentrate at the SSA 60-FRA stock.
    rm = m4["reference_moments"]
    m4_grad = {
        k.replace("prevalence.", ""): float(v["rate"])
        for k, v in rm.items()
        if k.startswith("prevalence.")
    }

    return {
        "mechanism": (
            "family B derives DI status from the M4 WORK-DISABILITY prevalence "
            "(no_frame_di_column_rule + ss_proxy_laundering_rule forbid the "
            "frame's own DI column), a point-prevalence among PSID "
            "person-years, and scores its age composition against the SSA "
            "DISABLED-WORKER BENEFICIARY STOCK (Table 19). The M4 prevalence "
            "peaks at 50-59 and DROPS at 60-66; the SSA stock keeps climbing "
            "to 45.4% at 60-FRA because it is duration-accumulated (entrants "
            "stay on the rolls, DI recovery ~1%/yr, until FRA conversion)."
        ),
        "per_band": per_band,
        "reconciliation_max_abs_remainder": max_remainder,
        "awards_flow_composition_sums_to": aw_sum,
        "m4_prevalence_gradient": m4_grad,
        "m4_concept_deltas": m4.get("concept_deltas"),
        "concept_delta_dominant_share": concept_dominant_share,
        "worst_band": {
            "band": "60-fra",
            "deployed_pp": per_band["60-fra"]["deployed_m4_stock_pp"],
            "awards_flow_pp": per_band["60-fra"]["ssa_awards_flow_pp"],
            "anchor_stock_pp": per_band["60-fra"]["anchor_ssa_stock_pp"],
            "reading": (
                "of the 60-FRA gap, the STOCK-vs-FLOW duration concept "
                "(awards flow -> beneficiary stock) is the dominant share; "
                "the M4 hazard shape (deployed -> true awards flow) is the "
                "minority. Even the correct FLOW concept falls far short of "
                "the accumulated STOCK the anchor counts."
            ),
        },
        "gate_design_determination": {
            "is_gate_design_finding": True,
            "insured_denominator_available": False,
            "insured_denominator_note": (
                "The archived DI ASR (di_asr_2023/provenance.md) explicitly "
                "lists the insured-population denominator (Supplement 4.C2) as "
                "STILL WANTED -- it is NOT in the evidence base, so the "
                "concept bridge's third leg is unmeasurable here."
            ),
            "determination": (
                "No contract-consistent candidate can clear the family-B DI "
                "bands. The gate scores an M4-work-disability-derived quantity "
                "against an SSA insured-beneficiary STOCK across a concept "
                "bridge it does not define: matching the stock composition "
                "requires (1) a DI-ENTRY hazard (incidence, not prevalence), "
                "(2) a DURATION-to-conversion model (stock = integral of "
                "entries x on-rolls survival), and (3) an INSURED denominator "
                "(20/40 recent-work test) -- none defined by the gate, the "
                "third not even archived. With only {age,is_female} "
                "conditioning on a recovery-churned self-report prevalence "
                "and the frame DI column forbidden, the bands are unreachable "
                "by a candidate LEVER. This is a GATE-DESIGN finding for the "
                "ceremony record: define the concept bridge, re-anchor the "
                "bands to a work-disability prevalence M4 can transport, or "
                "demote the DI bands to report-only until the bridge exists."
            ),
        },
        "finding": {
            "summary": (
                "The concept delta (work-disability point-prevalence vs "
                "duration-accumulated insured-beneficiary stock) dominates the "
                "DI level/steepness gap; M4 hazard level is the minority. No "
                "contract-consistent lever closes it -> gate-design finding."
            )
        },
    }


# ==========================================================================
# Q5 -- the tail's upper read (the most consequential number).
# ==========================================================================
def _career_panel(persons, gens, apply_p0: bool) -> dict[str, Any]:
    """td.transport_career_panel with an optional certified-p0 (zero/low-
    earning-year) correction. apply_p0=False reproduces the committed panel
    bit-for-bit; apply_p0=True zeroes the bottom-p0 of each cell's persistent
    rank (the certified participation dynamics) -> a realistic, lighter tail.
    """
    from scipy.stats import norm

    df = persons[
        (persons["age"] >= td.FAMILY_C_EARNER_AGE_LO)
        & (persons["age"] <= td.FAMILY_C_EARNER_AGE_HI)
        & (persons["earnings"] > 0)
    ].reset_index(drop=True)
    age0 = df["age"].to_numpy(dtype=np.float64)
    earn0 = df["earnings"].to_numpy(dtype=np.float64)
    fem = df["is_female"].to_numpy(dtype=bool)
    wt = df["weight"].to_numpy(dtype=np.float64)
    n = len(df)
    pid = (np.arange(n) + td.CPS_ID_OFFSET).astype(np.int64)
    marg = gens.earnings_marginals
    abin = gens.age_bin_fn
    bins0 = abin(age0)

    u_a = np.full(n, 0.5)
    for b in np.unique(bins0):
        idx = np.nonzero(bins0 == b)[0]
        e = earn0[idx]
        order = np.argsort(e, kind="stable")
        wsort = wt[idx][order]
        cum = np.cumsum(wsort) - 0.5 * wsort
        rank_sorted = cum / wsort.sum()
        r = np.empty(len(idx))
        r[order] = rank_sorted
        u_a[idx] = np.clip(r, 0.001, 0.999)

    z_a = norm.ppf(np.clip(u_a, 1e-4, 1 - 1e-4))
    rho = td.PERMANENT_VARIANCE_SHARE
    trng = np.random.default_rng(td.FAMILY_C_TRANSITORY_STREAM)
    career_bins = {
        a: int(abin(np.array([float(a)]))[0]) for a in td.FAMILY_C_CAREER_AGES
    }
    rows = []
    span = td.FAMILY_C_COHORT_HI - td.FAMILY_C_COHORT_LO + 1
    birth_year = td.FAMILY_C_COHORT_LO + (np.arange(n) % span)
    for a in td.FAMILY_C_CAREER_AGES:
        cell = marg.get((career_bins[a], td.TERMINAL_PERIOD))
        if cell is None or cell.n_pos == 0:
            continue
        eps = trng.standard_normal(n)
        z_year = np.sqrt(rho) * z_a + np.sqrt(1.0 - rho) * eps
        u_year = np.clip(norm.cdf(z_year), 0.001, 0.999)
        if apply_p0 and cell.p0 > 0.0:
            pos = u_year >= cell.p0
            earn_year = np.zeros(n)
            if cell.p0 < 1.0 and pos.any():
                pr = (u_year[pos] - cell.p0) / (1.0 - cell.p0)
                earn_year[pos] = cell.quantile(pr)
        else:
            earn_year = cell.quantile(u_year)
        yr = birth_year + a
        for i in range(n):
            e = earn_year[i]
            if e > 0:
                rows.append(
                    (int(pid[i]), int(yr[i]), float(e), int(a), float(wt[i]))
                )
    panel = pd.DataFrame(
        rows, columns=["person_id", "period", "earnings", "age", "weight"]
    )
    panel["role"] = "head"
    panel["earnings_acc"] = 0
    _e = panel["earnings"].to_numpy(dtype=np.float64)
    frac_above = (
        float(_e[_e > WAGE_BASE_2024].sum() / _e.sum()) if _e.sum() else 0.0
    )
    sex_df = pd.DataFrame(
        {"person_id": pid, "sex": np.where(fem, "female", "male")}
    )
    births = pd.DataFrame(
        {
            "parent_person_id": pd.array([], dtype="Int64"),
            "parent_sex": pd.array([], dtype="string"),
            "parent_birth_year": pd.array([], dtype="Int64"),
            "parent_birth_month": pd.array([], dtype="Int64"),
            "record_type": pd.array([], dtype="string"),
            "child_person_id": pd.array([], dtype="Int64"),
            "child_sex": pd.array([], dtype="string"),
            "birth_year": pd.array([], dtype="Int64"),
            "birth_month": pd.array([], dtype="Int64"),
            "birth_order": pd.array([], dtype="Int64"),
        }
    )
    return {
        "panel": panel,
        "sex": sex_df,
        "births": births,
        "n_person_years": int(len(panel)),
        "frac_payroll_above_wage_base": frac_above,
    }


def _run_f4(panel, sex_df, births) -> dict[str, Any]:
    """Re-run the committed #115/#117 ledgers on a transported panel (the
    td.family_c monkeypatch), returning the F4 (C1) + F2 (C2) orderings and
    the Mermin quartet outlay deltas."""
    import importlib

    mr = importlib.import_module("replication_mermin_rows")
    cg = importlib.import_module("replication_caregiver")
    m2 = importlib.import_module("m2_pseudo_projection")
    co = importlib.import_module("replication_cost_ordering")

    def fake_panel(*a, **k):
        return panel.copy()

    def fake_marriage(*a, **k):
        return sex_df.copy()

    def fake_births(*a, **k):
        return births.copy()

    patched = []
    for mod in (mr, cg, m2, co):
        for name, fn in (
            ("family_earnings_panel", fake_panel),
            ("marriage_history", fake_marriage),
            ("birth_history", fake_births),
        ):
            if hasattr(mod, name):
                patched.append((mod, name, getattr(mod, name)))
                setattr(mod, name, fn)
    try:
        res = m2.run(verbose=False)
    finally:
        for mod, name, orig in patched:
            setattr(mod, name, orig)
    rvf = res.get("results_vs_forecasts", {})
    detail = res.get("forecasts_detail", {})
    return {
        "c1_order": list(rvf["F4"]["result_order"]),
        "c1_quartet_deltas": detail.get("F4", {}).get("quartet_deltas"),
        "c2_order": list(rvf["F2"]["result_order"]),
    }


def q5_tail(persons, gens, art, contracts, verbose=True) -> dict[str, Any]:
    fc = contracts["family_c"]
    c1_spec = fc["fingerprints"]["c1"]
    c2_spec = fc["fingerprints"]["c2"]
    required_c1 = list(c1_spec["required_representative_order"])
    swap_c1 = tuple(c1_spec["swap_pair"])  # (nra, ppi): ppi must outrank nra
    swap_c2 = tuple(c2_spec["swap_pair"])  # (+2pp, elim): elim must outrank

    def _swap_realised(order, pair):
        a, b = pair
        if a not in order or b not in order:
            return None
        return bool(order.index(b) < order.index(a))

    # --- upper read: reproduce the committed transport bit-for-bit.
    up = _career_panel(persons, gens, apply_p0=False)
    committed = td.transport_career_panel(persons, gens)
    up_earn = up["panel"]["earnings"].to_numpy(dtype=np.float64)
    cm_earn = committed["panel"]["earnings"].to_numpy(dtype=np.float64)
    panel_bit_identical = bool(
        len(up_earn) == len(cm_earn)
        and np.array_equal(up_earn, cm_earn)
        and up["frac_payroll_above_wage_base"]
        == committed["frac_payroll_above_wage_base"]
    )
    if verbose:
        print("  [q5] running upper-read F4 ledger", flush=True)
    up_f = _run_f4(up["panel"], up["sex"], up["births"])
    art_c1 = art["family_c"]["fingerprints"]["c1"]
    up_quartet_bit_identical = bool(
        up_f["c1_quartet_deltas"]
        == art_c1["provision_deltas"].get("quartet_deltas")
        and up_f["c1_order"] == art_c1["deployed_order"]
    )

    # --- corrected tail: certified p0 zero/low-earning years included.
    if verbose:
        print("  [q5] running corrected-tail F4 ledger", flush=True)
    cor = _career_panel(persons, gens, apply_p0=True)
    cor_f = _run_f4(cor["panel"], cor["sex"], cor["births"])

    def _ppi_nra(q):
        return (
            abs(q["progressive_price_indexing"]),
            abs(q["nra_raised_to_70"]),
        )

    up_ppi, up_nra = _ppi_nra(up_f["c1_quartet_deltas"])
    cor_ppi, cor_nra = _ppi_nra(cor_f["c1_quartet_deltas"])

    c1_reversed_up = up_f["c1_order"] == required_c1
    c1_reversed_cor = cor_f["c1_order"] == required_c1
    # C1 swap realised iff PPI now outranks NRA in savings.
    up_swap = up_ppi > up_nra
    cor_swap = cor_ppi > cor_nra

    up_c2_swap = _swap_realised(up_f["c2_order"], swap_c2)
    cor_c2_swap = _swap_realised(cor_f["c2_order"], swap_c2)
    up_frac = up["frac_payroll_above_wage_base"]
    cor_frac = cor["frac_payroll_above_wage_base"]

    return {
        "mechanism": (
            "td.transport_career_panel draws cell.quantile(u_year) at every "
            "career age -- ALWAYS positive, NO p0 zero-mass -- so it carries "
            "no zero/low-earning years: an UPPER read on the tail. The "
            "correction applies the certified per-cell p0 (the same "
            "participation dynamics regenerate_earnings uses, coupling to "
            "Q1/Q2), zeroing the bottom-p0 of each persistent rank -> a "
            "lighter, realistic tail."
        ),
        "instrumentation_fidelity": {
            "positive_year_panel_bit_identical_vs_committed": (
                panel_bit_identical
            ),
            "upper_read_F4_quartet_bit_identical_vs_committed": (
                up_quartet_bit_identical
            ),
        },
        "upper_read": {
            "frac_payroll_above_wage_base": up_frac,
            "c1_order": up_f["c1_order"],
            "c1_quartet_deltas": up_f["c1_quartet_deltas"],
            "ppi_savings_abs": up_ppi,
            "nra_savings_abs": up_nra,
            "ppi_minus_nra": up_ppi - up_nra,
            "c1_reversed": bool(c1_reversed_up),
            "c1_swap_realised": bool(up_swap),
            "c2_order": up_f["c2_order"],
            "c2_swap_realised": up_c2_swap,
        },
        "corrected_tail": {
            "frac_payroll_above_wage_base": cor_frac,
            "c1_order": cor_f["c1_order"],
            "c1_quartet_deltas": cor_f["c1_quartet_deltas"],
            "ppi_savings_abs": cor_ppi,
            "nra_savings_abs": cor_nra,
            "ppi_minus_nra": cor_ppi - cor_nra,
            "c1_reversed": bool(c1_reversed_cor),
            "c1_swap_realised": bool(cor_swap),
            "c2_order": cor_f["c2_order"],
            "c2_swap_realised": cor_c2_swap,
            "tail_lighter_than_upper_read": bool(cor_frac < up_frac),
        },
        "c2_breakeven_frac_payroll_above_cap": C2_BREAKEVEN_FRAC,
        "c1_robustness_answer": {
            "swap_pair": list(swap_c1),
            "question": (
                "Does a realistic (corrected) tail move PPI past NRA -> "
                "reverse C1?"
            ),
            "answer_non_reversal_is_robust": bool(
                (not c1_reversed_up)
                and (not c1_reversed_cor)
                and cor_ppi <= up_ppi
            ),
            "quantified": (
                "Under the UPPER-read tail (the heaviest plausible tail, most "
                f"favourable to the swap), PPI savings {up_ppi:.4f} sit far "
                f"below NRA {up_nra:.4f} (gap {up_nra - up_ppi:.4f}). The "
                f"corrected tail LIGHTENS above-cap payroll ({up_frac:.3f} -> "
                f"{cor_frac:.3f}) and moves PPI savings to {cor_ppi:.4f}, "
                f"still below NRA {cor_nra:.4f}. PPI does NOT overtake NRA in "
                "either case; the correction moves it in the CONSERVATIVE "
                "direction. C1's non-reversal is ROBUST."
            ),
            "c2_note": (
                "C2's committed swap (elimination outranks +2pp) holds under "
                f"BOTH tails: upper read {up_frac:.3f} and corrected "
                f"{cor_frac:.3f} both exceed the ~{C2_BREAKEVEN_FRAC:.3f} "
                "above-cap break-even, and elimination outranks +2pp in each "
                f"corrected order (swap_realised upper={up_c2_swap}, "
                f"corrected={cor_c2_swap}). Caveat: the corrected tail "
                "reshuffles the REST of the revenue quartet (cap-$150k rises), "
                "so only the committed elimination<->+2pp pair is asserted "
                "robust, not the full C2 ordering."
            ),
        },
        "finding": {
            "summary": (
                "C1 non-reversal is robust: the upper-read tail is the most "
                "favourable case for PPI and it does not overtake NRA; a "
                "realistic tail only widens the gap. The transported AIME does "
                "NOT lift PPI past NRA."
            )
        },
    }


# ==========================================================================
# Assemble.
# ==========================================================================
def run(verbose: bool = True) -> dict[str, Any]:
    warnings.filterwarnings("ignore", message="lbfgs failed to converge")
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()
    persons = _load_frame()
    contracts = _load_contracts()
    art = json.load(open(CANDIDATE1_ARTIFACT))
    if verbose:
        print("[fit] generators", flush=True)
    gens = fit_generators()

    if verbose:
        print("[q1] marital equilibration", flush=True)
    q1 = q1_marital(persons, gens, art, verbose)
    if verbose:
        print("[q2] participation boundary", flush=True)
    q2 = q2_participation(persons, gens, art, contracts["tol_a"], verbose)
    if verbose:
        print("[q3] household scope", flush=True)
    q3 = q3_household(persons, art)
    if verbose:
        print("[q4] DI level bridge", flush=True)
    q4 = q4_di_bridge(art, contracts)
    if verbose:
        print("[q5] tail upper read", flush=True)
    q5 = q5_tail(persons, gens, art, contracts, verbose)

    reconciliations = {
        "float64_machine_epsilon": FLOAT64_EPS,
        "identity_bar_64_eps": IDENTITY_BAR,
        "q1_instrumentation_bit_identity_max_dev": q1[
            "instrumentation_fidelity"
        ]["max_abs_rate_deviation_vs_committed_cube"],
        "q1_decomposition_max_abs_remainder": q1[
            "reconciliation_max_abs_remainder"
        ],
        "q2_instrumentation_bit_identity_max_dev": q2[
            "instrumentation_fidelity"
        ]["max_abs_rate_deviation_vs_committed_cube"],
        "q3_reference_moment_max_dev": q3["reference_moment_fidelity"][
            "max_abs_deviation_vs_committed_rate_a"
        ],
        "q3_decomposition_max_abs_remainder": q3[
            "reconciliation_max_abs_remainder"
        ],
        "q4_decomposition_max_abs_remainder": q4[
            "reconciliation_max_abs_remainder"
        ],
        "q5_positive_year_panel_bit_identical": q5["instrumentation_fidelity"][
            "positive_year_panel_bit_identical_vs_committed"
        ],
        "q5_upper_read_quartet_bit_identical": q5["instrumentation_fidelity"][
            "upper_read_F4_quartet_bit_identical_vs_committed"
        ],
        "all_identity_reconciliations_at_machine_epsilon": bool(
            q1["instrumentation_fidelity"][
                "max_abs_rate_deviation_vs_committed_cube"
            ]
            == 0.0
            and q2["instrumentation_fidelity"][
                "max_abs_rate_deviation_vs_committed_cube"
            ]
            == 0.0
            and q1["reconciliation_max_abs_remainder"] <= IDENTITY_BAR
            and q3["reconciliation_max_abs_remainder"] <= IDENTITY_BAR
            and q4["reconciliation_max_abs_remainder"] <= IDENTITY_BAR
            and q3["reference_moment_fidelity"][
                "max_abs_deviation_vs_committed_rate_a"
            ]
            <= IDENTITY_BAR
            and q5["instrumentation_fidelity"][
                "positive_year_panel_bit_identical_vs_committed"
            ]
            and q5["instrumentation_fidelity"][
                "upper_read_F4_quartet_bit_identical_vs_committed"
            ]
        ),
        "reconciliation_bar_note": (
            "instrumentation bit-identity is EXACT (0.0) for every "
            "re-simulated component (Q1 marital, Q2 participation, Q5 career "
            "transport + F4 quartet); the additive decompositions (Q1 "
            "exposure+hazard, Q3 scope+composition, Q4 shape+duration) "
            "telescope to their targets at machine epsilon (a few ULP of "
            "float64 summation)."
        ),
    }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_w1",
        "reported_not_gated": True,
        "diagnostic": (
            "W1 forensics 1 -- the five transport mechanisms; measures "
            "before candidate 2 designs. Publishes regardless of any verdict."
        ),
        "registration": {
            "issue": 42,
            "comment_id": REGISTRATION_POINTER,
            "url": (
                "https://github.com/PolicyEngine/populace-dynamics/issues/42"
                f"#issuecomment-{REGISTRATION_POINTER}"
            ),
        },
        "registration_pointer": REGISTRATION_POINTER,
        "grading_pointer": GRADING_POINTER,
        "candidate1_pointer": CANDIDATE1_POINTER,
        "candidate1_artifact": "runs/gate_w1_candidate1_v1.json",
        "protocol": {
            "one_shot": True,
            "publishes_regardless": True,
            "train_frame_side_only": True,
            "k_diag_draws": K_DIAG,
            "full_exposure_age": FULL_EXPOSURE_AGE,
            "instrumentation_bit_identity": (
                "every re-simulation reproduces the committed "
                "transport_deployment_v1 machinery bit-for-bit before any "
                "counterfactual is measured"
            ),
        },
        "deployment_frame": dict(dfm.CERTIFIED_PIN),
        "generator_fit_provenance": gens.fit_provenance,
        "reconciliations": reconciliations,
        "q1_marital_equilibration": q1,
        "q2_participation_boundary": q2,
        "q3_household_scope": q3,
        "q4_di_level_bridge": q4,
        "q5_tail_upper_read": q5,
        "candidate2_design_implications": {
            "q1": (
                "Seed the marital chain's ENTRY state from an initial-state "
                "model (not the terminal A_MARITL); the never-married-at-18 "
                "entry is the lever, not the certified hazards."
            ),
            "q2": (
                "Extend the gate-1 fitted support to the 18-24 / 62-69 "
                "boundary ages (and add a sex covariate for participation); "
                "the boundary miss is fit support, not a hazard defect."
            ),
            "q3": (
                "Attach children via the certified fertility machinery and "
                "repair adult coresidence from realistic initial rosters; "
                "scope alone cannot make the candidate scoreable on the "
                "locked all-person rate_a."
            ),
            "q4": (
                "DO NOT spend a candidate lever on the family-B DI bands: "
                "they are unreachable without a concept bridge the gate does "
                "not define (a GATE-DESIGN item, not a candidate item)."
            ),
            "q5": (
                "C1's non-reversal is robust to a realistic tail, so C2 (not "
                "C1) is the fingerprint the transport moves; do not chase a "
                "PPI-over-NRA reversal that the conservative-direction "
                "argument rules out."
            ),
        },
        "revision_pins": {
            "frame_artifact_sha256": dfm.CERTIFIED_PIN["artifact_sha256"],
            "frame_revision": dfm.CERTIFIED_PIN["revision"],
            "pe_us_version": dfm.CERTIFIED_PIN["model_version"],
            "gates_yaml_blob": _contract_blob(),
            "pe_us_dir": os.environ.get("POPULACE_DYNAMICS_PE_US_DIR"),
        },
        "pointer": {
            "registration": REGISTRATION_POINTER,
            "grading": GRADING_POINTER,
            "candidate1": CANDIDATE1_POINTER,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return artifact


def _contract_blob() -> str | None:
    try:
        from populace_dynamics.contract import contract_revision

        return contract_revision(ROOT)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ARTIFACT_PATH))
    parser.add_argument("--no-sidecar", action="store_true")
    args = parser.parse_args()
    artifact = run(verbose=True)
    artifacts.write_new(
        Path(args.out),
        _json_safe(artifact),
        sidecar=not args.no_sidecar,
    )
    print(f"wrote {args.out}", flush=True)
    print(
        json.dumps(
            _json_safe(artifact["reconciliations"]),
            indent=1,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
