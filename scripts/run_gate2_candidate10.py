"""Gate-2 candidate 10 (run 1): candidate 8 + two named deltas, scored under
the amended mean-over-K=20-draws estimator.

The TENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics, and the FIRST fresh registration under the
amended (mean-over-20-draws) estimator ratified 2026-07-08 (gates.yaml
amendment 1; ceremony PR #96 -> flip #97). Its frozen specification is issue
#42 comment 4917059482 (``SPEC_REGISTRATION``): candidate 8's frozen spec
(comment 4912995860) verbatim EXCEPT TWO deltas registered from the committed
gate-2 forensics (#94, ``runs/gate2_forensics_v1.json``). One-shot; no
constant moves after the registration comment; published REGARDLESS of
verdict.

The amended estimator (gates.yaml gate_2, amendment 1)
------------------------------------------------------
The per-cell candidate score is ``|ln(rbar_candidate,s / rate_a,s)|`` where
``rbar_candidate,s`` is the MEAN over K=20 pre-registered draws
(``numpy.random.default_rng(5200 + k)``, k=0..19) of the cell RATE for the
seed-s holdout, scored once (NOT the mean of the per-draw scores). Tolerances
and the 46-cell 4-of-5-seed conjunction are byte-identical to the lock. The
run artifact conforms to ``protocol.fresh_run_artifact_schema``: it commits
every draw's per-cell rate (shape [20, 46, 5] = K x gated cells x gate
seeds), invalidates on any undefined gated-cell draw, and reports (never
gates) per-draw dispersion. The single-draw stream ``4200 + seed`` used by
candidates 1-9 is retained only for provenance; the amended stream
``5200 + k`` is distinct so the committed single-draw artifacts stay
auditable.

The two deltas vs candidate 8 (everything else byte-identical)
--------------------------------------------------------------
**Delta 1 -- observed undatable-marriage lifetime-count initial state (Q1's
exact residual, candidate 9's delta 1 UNCHANGED).** Each holdout person's
simulated lifetime-marriage count initialises at their OBSERVED count of
marriages not representable as datable in-exposure episodes -- the forensics
Q1 residual ``R[i] = n_marriages[i] - (# datable first_marriage/remarriage
transition events for person i)`` -- then accumulates the simulated datable
transitions exactly as candidate 8 does. It aggregates to the forensics
reference residual (0.116 M / 0.090 F per person) and reconciles to remainder
0.0 (a train-side hard gate BEFORE the one-shot). An OBSERVED initial state
per protocol; RNG-neutral (a post-assembly count add); hazards untouched.
:func:`run_gate2_candidate9.observed_residual_counts` and
:func:`run_gate2_candidate9._delta1_reconciliation` are IMPORTED and reused
byte-for-byte.

**Delta 2 -- age-band-conditioned remarriage (candidate 8's rescale
REMOVED).** Candidate 8's remarriage is the order-split (2nd vs 3rd+) weighted
empirical hazard by years-since-dissolution band x origin x sex, then
rescaled by one train-side scalar to preserve the aggregate. Candidate 10
REPLACES that with the remarriage hazard conditioned on the ego's AGE BAND
(18-34 / 35-49 / 50+) x years-since-dissolution band x origin
(divorced/widowed) x sex, with the SAME add-one smoothing (``wbar_diss``,
byte-identical to candidate 1) and candidate 8's aggregate-preservation
rescale REMOVED. The forensics (#94) located candidate 8's in-exposure
over-production (-0.050 M / -0.036 F per person) in OLD dissolved exposure
inheriting the pooled (younger, higher) remarriage rate; age conditioning
addresses that misallocation directly (the rescale only papered over it), so
the scalar is superseded. The order-bit stratum is dropped in favour of the
age-band stratum. Fertility = candidate 8's single-year triangular-kernel
construction, UNCHANGED (candidate 7/9's fertility deltas stand falsified;
the c1970s tilt is sub-tolerance under the mean estimator).

Designed count-cell cancellation (registration)
-----------------------------------------------
Combined with delta 1, the marriage-count tilt is designed to land near zero
from both sides: delta 1's +residual (~+0.046 ln-scale) minus delta 2's
removed over-production (~-0.046 ln-scale). The registration's honest
uncertainty is whether age conditioning under- or over-removes the
over-production (the cancellation is designed, not measured). The
``count_cell_tilt`` block reports the realised net tilt against that design.

Structure
---------
This runner IMPORTS candidate 8's machinery (which chains candidates 6/5/4/1
and candidate 7's estimator) and candidate 9's delta 1:
:func:`fit_components` starts from ``candidate8.fit_components`` and swaps ONLY
``remarriage`` for the age-banded table (delta 2); :func:`simulate_holdout`
mirrors ``candidate8.simulate_holdout`` byte-for-byte EXCEPT the remarriage
probability is looked up by the ego's age band (delta 2), then applies the
delta-1 count add. Because delta 2 moves only the remarriage THRESHOLD (the
per-year uniform blocks ``rng.random(n_active)`` then ``rng.random(n_fertile)``
are drawn in the same order and size as candidate 8) and delta 1 is
RNG-neutral, first marriage, the ever-married shares, and fertility are
byte-identical to candidate 8 at a shared draw seed; only remarriage, the
marriage counts, and everything downstream of the dissolved-state
trajectory move. The precheck and verdict assembly are candidate 1's,
imported unchanged.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed holdout-id
sha256, BEFORE any candidate is simulated. A SECOND hard gate (candidate 9's,
train-side, licensed) then verifies the delta-1 reconciliation to remainder
0.0. Any mismatch is a hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate10.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 8 supplies the machinery this build deltas twice: its fit chain
# (candidates 6/5/4/1 and candidate 7's estimator), its vectorised annual
# simulation (mirrored here), the precheck, the verdict assembly and the
# report-only summary. Candidate 9 supplies delta 1 (the observed residual
# and its reconciliation), reused byte-for-byte. Only remarriage is re-fit
# (delta 2) and only the marriage-count initial state is added (delta 1).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate8 as c8  # noqa: E402
import run_gate2_candidate9 as c9  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v10.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE9_ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS_ARTIFACT = ROOT / "runs" / "gate2_forensics_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v10"
RUN_NAME = "gate2_hazard_v10"

#: This run's frozen-spec registration (issue #42, comment 4917059482): the
#: first fresh registration under the amended mean-over-20-draws estimator.
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4917059482"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4917059482"
#: The candidate-8 spec this build deltas twice (comment 4912995860).
CANDIDATE8_REGISTRATION = c8.SPEC_REGISTRATION
#: Candidate 9 (comment 4914111252): the source of delta 1, reused verbatim.
CANDIDATE9_REGISTRATION = c9.SPEC_REGISTRATION
CANDIDATE6_REGISTRATION = c8.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c8.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c8.CANDIDATE1_REGISTRATION
#: The forensics diagnostic the deltas cite (#94, gate2_forensics_v1.json).
FORENSICS_REGISTRATION = c9.FORENSICS_REGISTRATION

# --- Amended-estimator draw stream (gates.yaml gate_2, amendment 1). -------
#: The amended 20-draw stream base: draw k uses default_rng(5200 + k). The
#: committed forensics convention (#94), distinct from the single-draw
#: 4200 + seed stream candidates 1-9 registered.
DRAW_SEED_BASE = 5200
N_DRAWS = 20

# --- Delta 2 dials (registration comment 4917059482). ----------------------
#: The remarriage age bands: 18-34 / 35-49 / 50+ (the ego's calendar-year age
#: at the dissolved person-year / remarriage event). Fixed by the
#: registration; the searchsorted band index clips ages below 18 into the
#: youngest band (there are essentially no dissolved person-years there).
REM_AGE_BANDS: tuple[tuple[int, int], ...] = ((18, 34), (35, 49), (50, 120))
REM_AGE_LOWERS = np.array([lo for lo, _ in REM_AGE_BANDS], dtype=np.int64)
_REM_AGE_LABEL = {0: "18-34", 1: "35-49", 2: "50+"}

#: The two named deltas (registration comment 4917059482).
DELTA_VS_CANDIDATE8 = (
    "TWO deltas vs candidate 8, scored under the amended mean-over-K=20-draws "
    "estimator (5200 + k). (1) OBSERVED UNDATABLE-MARRIAGE LIFETIME-COUNT "
    "INITIAL STATE (candidate 9's delta 1, UNCHANGED): each holdout person's "
    "simulated marriage count initialises at their OBSERVED residual "
    "R = n_marriages - datable in-exposure marriage events (the forensics Q1 "
    "residual), then accumulates the simulated datable transitions; an "
    "OBSERVED initial state per protocol, RNG-neutral, hazards untouched; "
    "reconciles to remainder 0.0 train-side before the one-shot. (2) "
    "AGE-BAND-CONDITIONED REMARRIAGE: remarriage hazard conditioned on the "
    "ego's age band (18-34 / 35-49 / 50+) x years-since-dissolution band x "
    "origin (divorced/widowed) x sex, same add-one smoothing (wbar_diss) as "
    "candidate 1, with candidate 8's order-bit stratum and its "
    "aggregate-preservation rescale REMOVED (age conditioning addresses the "
    "old-exposure-inherits-young-rates over-production the rescale papered "
    "over). Fertility = candidate 8's kernel construction, UNCHANGED. Delta 2 "
    "moves only the remarriage threshold (same draw order/size), so first "
    "marriage, the ever-married shares and fertility are byte-identical to "
    "candidate 8 at a shared draw seed; only remarriage, the marriage counts, "
    "and the dissolved-state trajectory move"
)

# --- Frozen dials + pure helpers, reused (byte-identical; imported). -------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE  # 4200 (single-draw stream; provenance only)
EXACT_ATOL = c1.EXACT_ATOL
Components = c1.Components

YSD_BANDS = c1.YSD_BANDS
YSD_LOWERS = c1.YSD_LOWERS
_bands_vec = c1._bands_vec

# Simulation helpers reused byte-for-byte from candidate 8's chain (aliased
# for provenance; simulate_holdout below mirrors candidate 8 exactly except
# the remarriage lookup).
_STATE = c8._STATE
_STATE_ABSORB = c8._STATE_ABSORB
_ASFR_LO = c8._ASFR_LO
_ASFR_HI = c8._ASFR_HI
_assemble_sim_panel = c8._assemble_sim_panel
_divorce_probs = c8._divorce_probs
_widow_probs = c8._widow_probs
_fertility_probs_single = c8._fertility_probs_single
WIDOW_BANDS = c8.WIDOW_BANDS
WIDOW_LOWERS = c8.WIDOW_LOWERS

# Delta 1 imported and reused byte-for-byte from candidate 9.
observed_residual_counts = c9.observed_residual_counts
_delta1_reconciliation = c9._delta1_reconciliation

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate10_run1_cache.json"
)

#: The marriage-count cells the two deltas are designed to cancel on.
COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
#: The gated remarriage cells delta 2 directly reshapes.
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
#: The registered modal failure (comment 4917059482): the count cells, if age
#: conditioning under- or over-removes the in-exposure over-production so the
#: designed cancellation misses.
REGISTERED_MODAL_CELLS = COUNT_CELLS
#: The cells the two deltas target (registration): the count cells (delta 1 +
#: delta 2 net) and the remarriage flows (delta 2).
TARGETED_CELLS = COUNT_CELLS + REMARRIAGE_GATED_CELLS


# --------------------------------------------------------------------------
# DELTA 2: age-band-conditioned remarriage (candidate 8's rescale removed)
# --------------------------------------------------------------------------
def fit_remarriage_age_banded(
    panel: transitions.MaritalPanel,
    train_ids: set[int],
) -> tuple[dict[tuple[int, int, str, str], float], float]:
    """Remarriage hazard by (age_band, ysd_band, origin, sex), add-one smoothed.

    Candidate 1's origin-split remarriage construction -- weighted empirical
    hazard among divorced/widowed person-years by years-since-dissolution band
    x origin (divorced/widowed) x sex, add-one smoothed at the train mean
    dissolved person-year weight ``wbar_diss`` -- ADDITIONALLY stratified by
    the ego's AGE BAND (18-34 / 35-49 / 50+) at the person-year, and with the
    order-bit stratum DROPPED. The age band is the ego's calendar-year age
    (``year - birth_year``), computed identically in the denominator
    (dissolved person-years) and the numerator (remarriage events). The
    per-cell add-one smoothing (``(wnum + wbar_diss) / (wden + 2*wbar_diss)``)
    and ``wbar_diss`` (computed over ALL dissolved rows, byte-identical to
    candidate 1) are UNCHANGED -- only the stratifying key changes, and NO
    aggregate-preservation rescale is applied (candidate 8's scalar is removed;
    the forensics locate the misallocation age conditioning fixes at the cell
    level).

    Returns the ``(age_band, ysd_band, origin, sex) -> rate`` table and
    ``wbar_diss`` (byte-identical to candidate 1's).
    """
    py = panel.person_years
    ev = panel.events
    train_py = py[py["person_id"].isin(train_ids)]
    train_ev = ev[ev["person_id"].isin(train_ids)]

    diss = train_py[
        train_py["marital_state"].isin(("divorced", "widowed"))
        & train_py["years_since_dissolution"].notna()
    ].copy()
    rem_ev = train_ev[
        (train_ev["transition"] == "remarriage")
        & train_ev["years_since_dissolution"].notna()
    ].copy()
    wbar_diss = float(diss["weight"].mean()) if len(diss) else 1.0

    for df in (diss, rem_ev):
        df["ysd_band"] = _bands_vec(
            df["years_since_dissolution"].astype("int64").to_numpy(),
            YSD_LOWERS,
            len(YSD_BANDS),
        )
        df["age_band"] = _bands_vec(
            np.rint(df["age"].to_numpy()).astype(np.int64),
            REM_AGE_LOWERS,
            len(REM_AGE_BANDS),
        )

    rem_den = diss.groupby(["age_band", "ysd_band", "marital_state", "sex"])[
        "weight"
    ].sum()
    rem_num = rem_ev.groupby(["age_band", "ysd_band", "origin", "sex"])[
        "weight"
    ].sum()

    remarriage: dict[tuple[int, int, str, str], float] = {}
    for ab in range(len(REM_AGE_BANDS)):
        for b in range(len(YSD_BANDS)):
            for origin in ("divorced", "widowed"):
                for sex in ("female", "male"):
                    wnum = float(rem_num.get((ab, b, origin, sex), 0.0))
                    wden = float(rem_den.get((ab, b, origin, sex), 0.0))
                    remarriage[(ab, b, origin, sex)] = (wnum + wbar_diss) / (
                        wden + 2.0 * wbar_diss
                    )
    return remarriage, wbar_diss


def _remarriage_age_band_diag(
    panel: transitions.MaritalPanel,
    train_ids: set[int],
    remarriage: dict[tuple[int, int, str, str], float],
) -> dict[str, Any]:
    """Provenance of the age-conditioned remarriage table + per-cell exposure.

    Records each cell's hazard, its train weighted exposure and its raw
    person-year count, and flags the thin cells (< 20 train person-years) the
    add-one smoothing pulls toward 0.5 -- the registration's flagged secondary
    risk (thin 50+ cells misfiring under the added stratum).
    """
    py = panel.person_years
    train_py = py[py["person_id"].isin(train_ids)]
    diss = train_py[
        train_py["marital_state"].isin(("divorced", "widowed"))
        & train_py["years_since_dissolution"].notna()
    ].copy()
    diss["ysd_band"] = _bands_vec(
        diss["years_since_dissolution"].astype("int64").to_numpy(),
        YSD_LOWERS,
        len(YSD_BANDS),
    )
    diss["age_band"] = _bands_vec(
        np.rint(diss["age"].to_numpy()).astype(np.int64),
        REM_AGE_LOWERS,
        len(REM_AGE_BANDS),
    )
    wexp = diss.groupby(["age_band", "ysd_band", "marital_state", "sex"])[
        "weight"
    ].sum()
    npy = diss.groupby(["age_band", "ysd_band", "marital_state", "sex"]).size()

    cells: dict[str, Any] = {}
    thin: list[str] = []
    for (ab, b, origin, sex), h in sorted(remarriage.items()):
        label = f"age{_REM_AGE_LABEL[ab]}|ysd{b}|{origin}|{sex}"
        n = int(npy.get((ab, b, origin, sex), 0))
        cells[label] = {
            "hazard": float(h),
            "train_person_years": n,
            "train_weighted_exposure": float(
                wexp.get((ab, b, origin, sex), 0.0)
            ),
        }
        if n < 20:
            thin.append(label)
    return {
        "representation": (
            "weighted empirical remarriage hazard by ego age band "
            "(18-34/35-49/50+) x years-since-dissolution band x origin "
            "(divorced/widowed) x sex, add-one smoothed at wbar_diss; "
            "candidate 8's order-bit stratum and aggregate-preservation "
            "rescale REMOVED (DELTA 2 vs candidate 8)"
        ),
        "age_bands": [list(b) for b in REM_AGE_BANDS],
        "ysd_bands": [list(b) for b in YSD_BANDS],
        "n_cells": len(remarriage),
        "n_thin_cells_lt_20_py": len(thin),
        "thin_cells_lt_20_py": thin,
        "cells": cells,
    }


def _remarriage_probs_age_banded(
    age: np.ndarray,
    ysd: np.ndarray,
    origin_state: np.ndarray,
    is_male: np.ndarray,
    rem_arr: np.ndarray,
) -> np.ndarray:
    """DELTA 2: remarriage probability by (age band, ysd, origin, sex).

    ``age`` is the ego's calendar-year age entering the year; the age band is
    the same 18-34 / 35-49 / 50+ split the train fit estimates. ``rem_arr`` is
    indexed ``[age_band, ysd_band, origin(0=div,1=wid), sex(0=f,1=m)]``.
    Otherwise identical to candidate 1's ``_remarriage_probs`` (no order axis).
    """
    ab = _bands_vec(
        np.rint(age).astype(np.int64), REM_AGE_LOWERS, len(REM_AGE_BANDS)
    )
    yb = _bands_vec(ysd, YSD_LOWERS, len(YSD_BANDS))
    origin_idx = (origin_state == 3).astype(np.int64)  # 2 div->0, 3 wid->1
    return rem_arr[ab, yb, origin_idx, is_male.astype(np.int64)]


# --------------------------------------------------------------------------
# Fitted components (candidate 8's, with the delta-2 remarriage swapped)
# --------------------------------------------------------------------------
def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Candidate 8's components with ONLY the remarriage table delta'd.

    Starts from :func:`candidate8.fit_components` -- so first marriage,
    divorce, the surviving-spouse widowhood level, the committed NCHS betas,
    the spousal-gap draw AND the single-year triangular-kernel fertility are
    byte-identical to candidate 8 by construction. Then DELTA 2 replaces
    ``remarriage`` with the age-band-conditioned table
    (:func:`fit_remarriage_age_banded`); candidate 8's order-split table, its
    rescale scalar and its rescale aggregates are dropped from the meta and
    the age-band table's provenance and per-cell exposure are recorded. DELTA
    1 is NOT a fitted component: it is an observed initial state applied at
    simulation-assembly time (:func:`simulate_holdout`).
    """
    base = c8.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # DELTA 2: age-band-conditioned remarriage, candidate 8's rescale removed.
    remarriage, wbar_diss = fit_remarriage_age_banded(panel, train_ids)
    base.remarriage = remarriage

    # Scrub candidate 8's order-split / rescale provenance; record delta 2's.
    for key in (
        "remarriage_order_diagnostics",
        "remarriage_order_prerescale_cells",
        "remarriage_candidate6",
        "remarriage_rescale",
        "remarriage_mean_dissolved_weight_check",
        "delta_vs_candidate6",
    ):
        base.meta.pop(key, None)
    base.meta["remarriage_representation"] = (
        "weighted empirical hazard by ego age band (18-34/35-49/50+) x "
        "years-since-dissolution band x origin (divorced/widowed) x sex, "
        "add-one smoothed at wbar_diss -- candidate 1's origin-split table "
        "re-stratified on age band, candidate 8's order-bit stratum and "
        "aggregate-preservation rescale REMOVED (DELTA 2)"
    )
    base.meta["remarriage_age_banded"] = _remarriage_age_band_diag(
        panel, train_ids, remarriage
    )
    base.meta["remarriage_mean_dissolved_weight"] = float(wbar_diss)
    base.meta["delta_vs_candidate8"] = DELTA_VS_CANDIDATE8
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 8's, with the age-banded remarriage
# lookup swapped in and the delta-1 count add appended)
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC10:
    mort_arr: np.ndarray  # [widow_band, sex] surviving-spouse level
    beta_arr: np.ndarray  # [sex] per-sex log-linear period slope
    rem_arr: np.ndarray  # [age_band, ysd_band, origin, sex] (DELTA 2)
    fert_arr: np.ndarray  # [age-FERT_AGE_LO, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC10:
    """Candidate 8's widowhood level + NCHS slope + single-year fertility,
    with the DELTA-2 remarriage lookup gaining an age-band axis (no order).
    """
    mort_arr = np.zeros((len(WIDOW_BANDS), 2), dtype=np.float64)
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            mort_arr[b, si] = components.mortality.get(f"{band}|{sex}", 0.0)

    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)

    # DELTA 2: remarriage indexed [age_band, ysd_band, origin, sex].
    rem_arr = np.zeros(
        (len(REM_AGE_BANDS), len(YSD_BANDS), 2, 2), dtype=np.float64
    )
    for (ab, b, origin, sex), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[ab, b, oi, si] = v

    decades = sorted({d for (_a, _p, d) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = c5.FERT_AGE_HI - c5.FERT_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1)), dtype=np.float64)
    for (age, pb, d), v in components.fertility.items():
        fert_arr[age - c5.FERT_AGE_LO, pb, decade_map[d]] = v
    return _SimLookupsC10(mort_arr, beta_arr, rem_arr, fert_arr, decade_map)


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 8's simulation with the DELTA-2 age-banded remarriage and the
    DELTA-1 count add.

    Byte-identical to :func:`candidate8.simulate_holdout` EXCEPT the remarriage
    probability is looked up by the ego's AGE BAND
    (:func:`_remarriage_probs_age_banded`) instead of candidate 8's order-split
    lookup, and after the panel is assembled each holdout person's simulated
    lifetime-marriage count is initialised at their observed residual (delta 1,
    a pure post-assembly count add on ``sim_panel.attrs['n_marriages']``). The
    per-year uniform blocks (``rng.random(n_active)`` then
    ``rng.random(n_fertile)``) and the spawned gap-draw stream are drawn in the
    same order and size as candidate 8 -- only the remarriage THRESHOLD moves,
    and delta 1 perturbs NO draw.
    """
    attrs = panel.attrs[panel.attrs["person_id"].isin(holdout_ids)].copy()
    attrs = attrs.sort_values("person_id").reset_index(drop=True)
    n = len(attrs)
    pid = attrs["person_id"].to_numpy(dtype=np.int64)
    by = attrs["birth_year"].to_numpy(dtype=np.float64)
    sex = attrs["sex"].to_numpy()
    is_male = (sex == "male").astype(np.float64)
    sy = attrs["start_exposure_year"].to_numpy(dtype=np.int64)
    ey = attrs["censor_year"].to_numpy(dtype=np.int64)
    decade = (by // 10 * 10).astype(np.int64)

    py = panel.person_years
    entry = (
        py[py["person_id"].isin(holdout_ids)]
        .sort_values("year")
        .groupby("person_id", as_index=False)
        .first()
    )
    entry_state = (
        entry.set_index("person_id")["marital_state"].reindex(pid).to_numpy()
    )
    entry_dur = entry.set_index("person_id")["marriage_duration"].reindex(pid)
    entry_ysd = entry.set_index("person_id")[
        "years_since_dissolution"
    ].reindex(pid)

    state = np.zeros(n, dtype=np.int64)
    cur_start = np.full(n, -1, dtype=np.int64)
    order = np.zeros(n, dtype=np.int64)
    diss_year = np.full(n, -1, dtype=np.int64)
    parity = np.zeros(n, dtype=np.int64)
    open_start = np.full(n, -1, dtype=np.int64)
    open_order = np.zeros(n, dtype=np.int64)

    for i in range(n):
        st = entry_state[i]
        if pd.isna(st) or st == "never_married":
            state[i] = 0
        elif st == "married":
            state[i] = 1
            d = entry_dur.iloc[i]
            d0 = int(d) if not pd.isna(d) else 0
            cur_start[i] = int(sy[i]) - d0
            order[i] = 1
            open_start[i] = cur_start[i]
            open_order[i] = 1
        elif st in ("divorced", "widowed"):
            state[i] = _STATE[st]
            j = entry_ysd.iloc[i]
            j0 = int(j) if not pd.isna(j) else 0
            diss_year[i] = int(sy[i]) - j0
            order[i] = 1
        else:
            state[i] = _STATE_ABSORB

    # Registered simulation RNG + the SPAWNED gap-draw stream (candidate 4's
    # delta 2, retained): the spawn does not advance rng's bit stream. Under
    # the amended estimator sim_seed = 5200 + k for draw k.
    rng = np.random.default_rng(sim_seed)
    gap_seed_seq = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_seq)

    gap_dist = components.gap_dist_by_sex
    gap_arr = np.empty(n, dtype=np.float64)
    fem_mask = is_male == 0.0
    male_mask = is_male == 1.0
    n_fem = int(fem_mask.sum())
    n_male = int(male_mask.sum())
    if n_fem:
        gap_arr[fem_mask] = gap_rng.choice(gap_dist["female"], size=n_fem)
    if n_male:
        gap_arr[male_mask] = gap_rng.choice(gap_dist["male"], size=n_male)
    opp_is_male = 1.0 - is_male

    lookups = _build_sim_lookups(components)
    fert_didx = np.array(
        [lookups.decade_map.get(int(d), -1) for d in decade], dtype=np.int64
    )

    ep_person: list[int] = []
    ep_order: list[int] = []
    ep_start: list[int] = []
    ep_end: list[Any] = []
    ep_how: list[str] = []
    bi_person: list[int] = []
    bi_year: list[int] = []
    bi_order: list[int] = []

    def close_ep(idx_arr: np.ndarray, how: str, end_year: int) -> None:
        for i in idx_arr:
            ep_person.append(int(pid[i]))
            ep_order.append(int(open_order[i]))
            ep_start.append(int(open_start[i]))
            ep_end.append(int(end_year))
            ep_how.append(how)

    y0, y1 = int(sy.min()), int(ey.max())
    for y in range(y0, y1 + 1):
        active = (sy <= y) & (y <= ey)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            continue
        age = y - by[idx]
        u = rng.random(idx.size)
        st = state[idx]

        nm = st == 0
        if nm.any():
            sub = idx[nm]
            p_fm = components.first_marriage.predict(
                age[nm], is_male[sub], decade[sub]
            )
            marry = u[nm] < p_fm
            gi = sub[marry]
            order[gi] += 1
            cur_start[gi] = y
            state[gi] = 1
            open_start[gi] = y
            open_order[gi] = order[gi]

        mar = st == 1
        if mar.any():
            sub = idx[mar]
            dur = (y - cur_start[sub]).astype(np.int64)
            p_div = _divorce_probs(dur, order[sub], components.divorce)
            sp_age = age[mar] + gap_arr[sub]
            p_wid = _widow_probs(
                age[mar],
                is_male[sub],
                sp_age,
                opp_is_male[sub],
                y,
                lookups.mort_arr,
                lookups.beta_arr,
            )
            um = u[mar]
            div = um < p_div
            wid = (~div) & (um < p_div + p_wid)
            gdi = sub[div]
            close_ep(gdi, "divorce", y)
            state[gdi] = 2
            diss_year[gdi] = y
            gwi = sub[wid]
            close_ep(gwi, "widowhood", y)
            state[gwi] = 3
            diss_year[gwi] = y

        diss = (st == 2) | (st == 3)
        if diss.any():
            sub = idx[diss]
            ysd = (y - diss_year[sub]).astype(np.int64)
            origin = st[diss]
            # DELTA 2: remarriage looked up by the ego's age band (18-34 /
            # 35-49 / 50+) x ysd x origin x sex; NO order axis, NO rescale.
            # Reuses candidate 8's u[diss] draw byte-for-byte -- only the
            # threshold moves.
            p_rm = _remarriage_probs_age_banded(
                age[diss], ysd, origin, is_male[sub], lookups.rem_arr
            )
            rm = u[diss] < p_rm
            gri = sub[rm]
            order[gri] += 1
            cur_start[gri] = y
            state[gri] = 1
            diss_year[gri] = -1
            open_start[gri] = y
            open_order[gri] = order[gri]

        # Fertility: women aged 15-49, any marital state (candidate 8's
        # single-year kernel lookup; NO delta -- byte-identical to candidate
        # 8, same uf draw block, same threshold).
        age_all = (y - by).astype(np.int64)
        fert = (
            active
            & (sex == "female")
            & (age_all >= _ASFR_LO)
            & (age_all <= _ASFR_HI)
        )
        fidx = np.nonzero(fert)[0]
        if fidx.size:
            uf = rng.random(fidx.size)
            fage = (y - by[fidx]).astype(np.int64)
            p_birth = _fertility_probs_single(
                fage, parity[fidx], fert_didx[fidx], lookups.fert_arr
            )
            born = uf < p_birth
            gbi = fidx[born]
            for i in gbi:
                bi_person.append(int(pid[i]))
                bi_year.append(int(y))
                bi_order.append(int(parity[i]) + 1)
            parity[gbi] += 1

    still = np.nonzero(state == 1)[0]
    for i in still:
        ep_person.append(int(pid[i]))
        ep_order.append(int(open_order[i]))
        ep_start.append(int(open_start[i]))
        ep_end.append(pd.NA)
        ep_how.append("intact")

    sim_panel = _assemble_sim_panel(
        attrs, ep_person, ep_order, ep_start, ep_end, ep_how
    )
    sim_births = pd.DataFrame(
        {
            "parent_person_id": np.array(bi_person, dtype=np.int64),
            "birth_year": pd.array(bi_year, dtype="Int64"),
            "birth_order": pd.array(bi_order, dtype="Int64"),
            "record_type": pd.array(
                ["birth"] * len(bi_person), dtype="string"
            ),
            "is_event": np.ones(len(bi_person), dtype=bool),
        }
    )

    # DELTA 1: observed undatable-marriage lifetime-count initial state (a pure
    # post-assembly count add on n_marriages; RNG-neutral). Candidate 9's
    # observed_residual_counts, reused byte-for-byte.
    residual = observed_residual_counts(panel)
    add = (
        sim_panel.attrs["person_id"]
        .map(residual)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    sim_panel.attrs["n_marriages"] = (
        sim_panel.attrs["n_marriages"].to_numpy(dtype="float64") + add
    )
    return sim_panel, sim_births


def _draw_moments(
    panel: transitions.MaritalPanel,
    ids_a: set[int],
    components: Components,
    draw_seed: int,
) -> dict[str, dict[str, float]]:
    """One draw's per-cell reference moments for the seed-A holdout.

    Simulates side A once at ``draw_seed`` (= 5200 + k under the amended
    estimator), builds the fertility panel, and returns
    :func:`transitions.reference_moments` (weighted). The single unit the
    mean-over-K estimator averages; the live seed-0 single-draw test pins one
    cell's value here to the committed per-draw artifact.
    """
    sim_panel, sim_births = simulate_holdout(
        panel, ids_a, components, draw_seed
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    return transitions.reference_moments(
        sim_panel, sim_fert, ids_a, weighted=True
    )


# --------------------------------------------------------------------------
# Per-seed scoring under the amended mean-over-K=20-draws estimator
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    floor: dict[str, Any],
    tol: dict[str, float],
    report_only: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """Fit side B once, simulate side A at K=20 draws, score the 20-draw mean.

    The amended estimator (gates.yaml gate_2, amendment 1): the candidate fits
    on side B, then simulates side A's persons at K=20 pre-registered draws
    (``default_rng(5200 + k)``, k=0..19). ``rbar`` is the mean over the 20
    draws of each cell's RATE; the score is ``|ln(rbar / rate_a)|`` (NOT the
    mean of the per-draw scores). The full [K, cell] per-draw rate matrix, the
    per-cell per-draw sd and the max per-draw ``|ln|`` are committed for the
    fresh-run artifact schema; any undefined gated-cell draw (empty simulated
    denominator) is flagged for run invalidation.
    """
    t0 = time.time()
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    components = fit_components(
        panel, demo, death_records, mh_records, birth_records, order_map, ids_b
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]
    all_cells = sorted(set(tol) | set(report_only))

    draw_seeds = [DRAW_SEED_BASE + k for k in range(N_DRAWS)]
    # Per-cell arrays over the K draws: rate, den_wt, n_events.
    per_draw_rate: dict[str, list[float]] = {c: [] for c in all_cells}
    per_draw_den: dict[str, list[float]] = {c: [] for c in all_cells}
    per_draw_nev: dict[str, list[int]] = {c: [] for c in all_cells}
    for draw_seed in draw_seeds:
        cand = _draw_moments(panel, ids_a, components, draw_seed)
        for c in all_cells:
            cell = cand[c]
            per_draw_rate[c].append(float(cell["rate"]))
            per_draw_den[c].append(float(cell.get("den_wt", 0.0)))
            per_draw_nev[c].append(int(cell.get("n_events", 0)))

    # Undefined-draw detection (fresh_run_artifact_schema.undefined_draw_rule):
    # a gated cell with an empty simulated denominator on any draw invalidates.
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
        # Report-only per-draw dispersion (schema): per-cell sd across the 20
        # draws and the worst single-draw |ln| excursion.
        sd = float(rates.std(ddof=1)) if rates.size > 1 else 0.0
        if rate_a > 0:
            with np.errstate(divide="ignore"):
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
            f"(seed_pass={seed_pass}); K={N_DRAWS} draws; "
            f"undefined={len(undefined)}; fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "estimator": "mean_over_K20_draws",
        "draw_seeds": draw_seeds,
        "sim_seed_single_draw_provenance": SIM_SEED_BASE + seed,
        "component_meta": components.meta,
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
# Fresh-run artifact-schema assembly (amendment 1)
# --------------------------------------------------------------------------
def _per_draw_per_cell_rates_block(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """The [K=20, 46 gated cells, 5 gate seeds] per-draw rate cube.

    Commits every draw's per-cell rate ``r[k, cell, s]`` so a referee can
    recompute ``rbar_candidate,s = mean_k r[k, cell, s]`` cell-by-cell and
    re-derive the certified ``|ln(rbar / rate_a)|`` independently of the
    committed mean.
    """
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


def _undefined_draw_block(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The undefined-draw check (schema.undefined_draw_rule).

    Aggregates every seed's undefined gated-cell draws. Per the pre-specified
    rule, ANY undefined gated draw (empty simulated denominator) invalidates
    the run -- no draw may be dropped, substituted or re-rolled.
    """
    all_undefined: list[dict[str, Any]] = []
    for s in per_seed:
        for u in s["undefined_gated_draws"]:
            all_undefined.append({"seed": s["seed"], **u})
    return {
        "required": True,
        "pre_specified": True,
        "rule": (
            "any gated cell UNDEFINED (empty simulated denominator) on any of "
            "the K=20 draws invalidates the run; no draw dropped, skipped, "
            "substituted or re-rolled; rbar is the mean over ALL K draws"
        ),
        "n_undefined_gated_draws": len(all_undefined),
        "undefined_gated_draws": all_undefined,
        "run_invalidated": bool(all_undefined),
    }


def _per_draw_dispersion_block(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    """Report-only per-draw dispersion (schema.per_draw_dispersion_disclosure).

    Per gated cell per gate seed: the sd of the cell rate across the K=20
    draws, and the worst single-draw ``|ln(r[k]/rate_a)|`` excursion. Never
    gated; exposes over-dispersion the 20-draw mean hides.
    """
    cell_index = sorted(tol)
    by_seed = {s["seed"]: s for s in per_seed}
    seeds = [s["seed"] for s in per_seed]
    per_cell_sd: dict[str, dict[str, float]] = {}
    max_abs_ln: dict[str, dict[str, Any]] = {}
    for c in cell_index:
        per_cell_sd[c] = {
            str(s): float(by_seed[s]["gated_cells"][c]["per_draw_rate_sd"])
            for s in seeds
        }
        max_abs_ln[c] = {
            str(s): by_seed[s]["gated_cells"][c]["max_per_draw_abs_ln"]
            for s in seeds
        }
    # Notables: the cells whose worst single-draw excursion most exceeds their
    # certified 20-draw-mean score (draw noise the mean absorbs), and the
    # highest per-draw-sd cells.
    notables_excursion = []
    for c in cell_index:
        for s in seeds:
            rec = by_seed[s]["gated_cells"][c]
            mx = rec["max_per_draw_abs_ln"]
            if mx is None:
                continue
            notables_excursion.append(
                {
                    "cell": c,
                    "seed": s,
                    "certified_score": rec["score"],
                    "max_per_draw_abs_ln": mx,
                    "tolerance": rec["tolerance"],
                    "worst_draw_exceeds_tol": bool(mx > rec["tolerance"]),
                    "mean_absorbs": bool(
                        mx > rec["tolerance"] >= rec["score"]
                    ),
                }
            )
    notables_excursion.sort(
        key=lambda d: d["max_per_draw_abs_ln"] - d["certified_score"],
        reverse=True,
    )
    return {
        "required": True,
        "gated": False,
        "report_only": True,
        "per_cell_per_draw_sd": per_cell_sd,
        "max_per_draw_abs_ln_per_cell": max_abs_ln,
        "n_cells_worst_draw_exceeds_tol_but_mean_passes": sum(
            1 for d in notables_excursion if d["mean_absorbs"]
        ),
        "top_excursions": notables_excursion[:12],
        "note": (
            "REPORT-ONLY: no dispersion cap gates the run. 'mean_absorbs' "
            "flags cells whose worst single draw would clip the tolerance but "
            "whose 20-draw mean passes -- exactly the draw noise the amended "
            "estimator averages out."
        ),
    }


def _count_cell_tilt(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The designed count-cell cancellation, measured.

    The registration designs the marriage-count tilt to land near zero:
    delta 1's +residual (~+0.046 ln) minus delta 2's removed over-production
    (~-0.046 ln). Reports candidate 10's realised signed ln-tilt
    ``ln(rbar / rate_a)`` per seed for each count cell against the committed
    candidate 8 (no delta 1) and candidate 9 (delta-1-heavy, failed)
    single-draw scores, so the cancellation is auditable.
    """
    c8art = (
        json.loads(CANDIDATE8_ARTIFACT.read_text())
        if CANDIDATE8_ARTIFACT.exists()
        else None
    )
    c9art = (
        json.loads(CANDIDATE9_ARTIFACT.read_text())
        if CANDIDATE9_ARTIFACT.exists()
        else None
    )
    by8 = {s["seed"]: s for s in c8art["per_seed"]} if c8art else {}
    by9 = {s["seed"]: s for s in c9art["per_seed"]} if c9art else {}

    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4917059482: the count tilt is designed to "
            "land near zero from both sides -- delta 1's +residual "
            "(~+0.046 ln-scale) minus delta 2's removed over-production "
            "(~-0.046 ln-scale). The cancellation is DESIGNED, not measured; "
            "this block reports the realised net tilt."
        ),
        "design_residual_ln": 0.046,
        "design_over_production_ln": -0.046,
        "cells": {},
    }
    for cell in COUNT_CELLS:
        rows = []
        signed = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            rbar = rec["rbar"]
            rate_a = rec["rate_a"]
            tilt = (
                float(math.log(rbar / rate_a))
                if rbar > 0 and rate_a > 0
                else None
            )
            if tilt is not None:
                signed.append(tilt)
            rows.append(
                {
                    "seed": s["seed"],
                    "rbar": rbar,
                    "rate_a": rate_a,
                    "signed_ln_tilt": tilt,
                    "score_abs_ln": rec["score"],
                    "tolerance": rec["tolerance"],
                    "pass": rec["pass"],
                    "candidate8_single_draw_score": (
                        by8[s["seed"]]["gated_cells"][cell]["score"]
                        if by8
                        else None
                    ),
                    "candidate9_single_draw_score": (
                        by9[s["seed"]]["gated_cells"][cell]["score"]
                        if by9
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "mean_signed_ln_tilt": (
                float(np.mean(signed)) if signed else None
            ),
            "mean_abs_ln_score": float(
                np.mean([r["score_abs_ln"] for r in rows])
            ),
            "n_seeds_pass": n_pass,
            "cancellation_landed": bool(n_pass >= 4),
        }
    both_pass = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    out["designed_cancellation_succeeded"] = bool(both_pass)
    out["summary"] = (
        "the designed +residual/-over-production cancellation "
        + ("HELD" if both_pass else "MISSED")
        + " on the marriage-count cells (>=4/5 seeds pass on both)"
        if both_pass
        else (
            "the designed +residual/-over-production cancellation MISSED on "
            "at least one marriage-count cell (< 4/5 seeds pass)"
        )
    )
    return out


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (count cells), targeted cells, and the gate decider."""
    fails_by_cell: dict[str, list[int]] = {}
    for f in verdict["all_failing_gated_cells"]:
        fails_by_cell.setdefault(f["cell"], []).append(f["seed"])

    def track(cell: str) -> dict[str, Any]:
        return {
            "tolerance": per_seed[0]["gated_cells"][cell]["tolerance"],
            "per_seed_score": {
                s["seed"]: s["gated_cells"][cell]["score"] for s in per_seed
            },
            "per_seed_pass": {
                s["seed"]: s["gated_cells"][cell]["pass"] for s in per_seed
            },
            "failed_seeds": sorted(fails_by_cell.get(cell, [])),
        }

    def seeds_pass_if_forgiven(forgiven: set[str]) -> int:
        n = 0
        for s in per_seed:
            ok = all(
                rec["pass"]
                for cell, rec in s["gated_cells"].items()
                if cell not in forgiven
            )
            n += ok
        return n

    gate_pass = verdict["gate_2_pass"]
    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven(set(REGISTERED_MODAL_CELLS))
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    modal_failed = any(c in fails_by_cell for c in REGISTERED_MODAL_CELLS)
    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }

    if gate_pass:
        decider = "none (gate passed)"
    elif seeds_pass_if_forgiven(set(TARGETED_CELLS)) >= 4:
        decider = (
            "the delta-targeted cells (forgiving the count/remarriage targets "
            "flips >= 4 seeds to pass)"
        )
    elif n_pass_no_modal >= 4:
        decider = (
            "the registered modal count cells (forgiving them flips >= 4 "
            "seeds to pass)"
        )
    else:
        decider = (
            "broader than the registered modal + delta-targeted cells "
            "(other gated cells also hold the gate below 4 passing seeds)"
        )

    return {
        "registered_modal": (
            "the marriage-count cells (mean_lifetime_marriages|male/female) -- "
            "if age conditioning under- or over-removes the in-exposure "
            "over-production so delta 1's +residual and delta 2's removed "
            "over-production fail to cancel"
        ),
        "modal_cells": list(REGISTERED_MODAL_CELLS),
        "modal_failed": modal_failed,
        "modal_failed_seeds": {
            c: sorted(fails_by_cell.get(c, [])) for c in REGISTERED_MODAL_CELLS
        },
        "modal_track": {c: track(c) for c in REGISTERED_MODAL_CELLS},
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "distinct_failing_cells": sorted(distinct_fail_cells),
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "decider": decider,
        },
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins + the candidate-10 schema, c1-c9 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 7, 8, 9):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate9_artifact"] = "runs/gate2_hazard_v9.json"
    pins["candidate9_artifact_sha256"] = c1._sha_of_file(CANDIDATE9_ARTIFACT)
    pins["candidate8_artifact"] = "runs/gate2_hazard_v8.json"
    pins["candidate8_artifact_sha256"] = c1._sha_of_file(CANDIDATE8_ARTIFACT)
    pins["forensics_runner"] = "scripts/gate2_forensics.py"
    pins["forensics_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics.py"
    )
    pins["forensics_artifact"] = "runs/gate2_forensics_v1.json"
    pins["forensics_artifact_sha256"] = c1._sha_of_file(FORENSICS_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["delta"] = (
        "delta 1: observed undatable-marriage lifetime-count initial state "
        "(candidate 9, RNG-neutral); delta 2: age-band-conditioned remarriage "
        "(18-34/35-49/50+ x ysd x origin x sex), candidate 8's order-bit "
        "stratum and aggregate-preservation rescale removed"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the two candidate-10 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 8 + two named "
            "deltas: an observed undatable-marriage lifetime-count initial "
            "state, and age-band-conditioned remarriage with candidate 8's "
            "aggregate-preservation rescale removed), scored under the amended "
            "mean-over-K=20-draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate8": DELTA_VS_CANDIDATE8,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 8 at a shared draw seed (no delta "
                "touches first marriage)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 8 "
                "(events downstream of the remarriage trajectory move under "
                "delta 2)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; surviving-spouse "
                "marriage-history widowhood level x candidate 5's committed "
                "NCHS betas -- estimator BYTE-IDENTICAL to candidate 8"
            ),
            "remarriage": (
                "DELTA 2: weighted empirical hazard by ego age band "
                "(18-34/35-49/50+) x years-since-dissolution band x origin "
                "(divorced/widowed) x sex, add-one smoothed at wbar_diss; "
                "candidate 8's order-bit stratum and aggregate-preservation "
                "rescale REMOVED. Age conditioning fixes the "
                "old-exposure-inherits-young-rates over-production the "
                "forensics (#94) located (-0.050 M / -0.036 F per person)"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 8 (no fertility delta; candidate 7/9's fertility "
                "deltas stand falsified)"
            ),
            "lifetime_marriage_count_initial_state": (
                "DELTA 1 (candidate 9, unchanged): each holdout person's "
                "simulated lifetime-marriage count initialises at their "
                "OBSERVED residual (n_marriages minus datable in-exposure "
                "marriage events -- the forensics Q1 residual) and accumulates "
                "the simulated datable transitions. An observed initial state "
                "per protocol; RNG-neutral; hazards untouched"
            ),
        },
        "estimator": (
            "AMENDED (gates.yaml gate_2 amendment 1, ratified 2026-07-08): per "
            "cell rbar_candidate,s = mean over K=20 draws (default_rng(5200 + "
            "k), k=0..19) of the cell rate; score |ln(rbar / rate_a,s)| scored "
            "once (NOT the mean of per-draw scores); tolerances and the "
            "46-cell 4-of-5-seed conjunction byte-identical to the lock"
        ),
        "registered_ambiguity_resolutions": {
            "remarriage_age_bands": (
                "18-34 / 35-49 / 50+ on the ego's calendar-year age "
                "(year - birth_year) at the dissolved person-year and the "
                "remarriage event; the searchsorted band index clips ages "
                "below 18 into the youngest band"
            ),
            "remarriage_smoothing": (
                "the SAME add-one smoothing (wnum + wbar_diss)/(wden + "
                "2*wbar_diss) and the SAME wbar_diss (mean weight over ALL "
                "dissolved rows) as candidate 1; only the stratifying key "
                "changes; NO aggregate-preservation rescale (candidate 8's "
                "scalar removed)"
            ),
            "observed_residual_definition": (
                "candidate 9's delta 1 verbatim: R[i] = n_marriages[i] - (# "
                "datable in-exposure first_marriage/remarriage transition "
                "events for person i); reconciles to the forensics reference "
                "residual and its five buckets (remainder 0.0); applied per "
                "holdout person as an observed initial state"
            ),
            "rng_neutrality": (
                "delta 2 moves only the remarriage threshold (per-year uniform "
                "blocks drawn in the same order and size as candidate 8) and "
                "delta 1 adds no draw; at a shared draw seed first marriage, "
                "the ever-married shares and fertility are byte-identical to "
                "candidate 8"
            ),
            "everything_else": (
                "the surviving-spouse widowhood level, the committed NCHS "
                "betas, the knot-at-22 first-marriage spline, divorce, the "
                "spousal-gap draw, the single-year triangular-kernel "
                "fertility, the competing-risk step, the spawned gap-draw "
                "stream, one sequence per person per draw, and the locked "
                "protocol are byte-identical to candidate 8"
            ),
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = c1._load_cache(cache_path)

    thresholds = c1.load_gate2_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    tol = c1.gated_tolerances(thresholds)
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

    # Preflight: candidate 9 (delta 1), candidate 8 (the fit base) and its fit
    # chain (6/5/7), and the forensics artifact must be present, plus the
    # candidate-5 NCHS references.
    for name, path in (
        ("candidate-9", CANDIDATE9_ARTIFACT),
        ("candidate-8", CANDIDATE8_ARTIFACT),
        ("candidate-7", CANDIDATE7_ARTIFACT),
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
        ("forensics", FORENSICS_ARTIFACT),
    ):
        if not path.exists():
            raise RuntimeError(
                f"{name} artifact missing at {path}; required for the run."
            )
    for year, path in c5.NCHS_LIFE_TABLE_PATHS.items():
        if not path.exists():
            raise RuntimeError(
                f"NCHS life-table reference for {year} missing at {path}; "
                "run scripts/fetch_nchs_life_tables_historical.py first."
            )
    c6._committed_beta_v5()  # fail fast if the committed betas drifted

    mh_records = marriage.marriage_history()
    birth_records = g2f.births.birth_history()
    death_records = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, fert, data_meta = g2f.load_panels()
    order_map = c1._order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons; "
            f"estimator: mean over K={N_DRAWS} draws (5200 + k)"
        )

    # Hard gate 1: bit-exact reproduction of the committed floor.
    precheck = c1.run_precheck(panel, fert, floor)
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
            "Scoring path does not reproduce the committed gate-2 floor "
            "(reference moments / per-seed rate_a / holdout sha256) to bit "
            "precision; refusing to proceed."
        )

    # Hard gate 2: the delta-1 reconciliation (train-side, licensed; candidate
    # 9's, reused). Must reconcile to remainder 0.0 BEFORE the one-shot.
    reconciliation = _delta1_reconciliation(panel, mh_records, GATE_SEEDS)
    if verbose:
        pid_max = reconciliation["per_person_identity_max_abs_residual"]
        agg_max = reconciliation["aggregate_reconciliation_max_abs_remainder"]
        print(
            "delta-1 reconciliation reconciled="
            f"{reconciliation['reconciled']} "
            f"(per-person identity max={pid_max:.2e}, "
            f"aggregate max remainder={agg_max:.2e}, "
            f"residual>=0: {reconciliation['residual_nonnegative']})"
        )
    if not reconciliation["reconciled"]:
        raise RuntimeError(
            "delta-1 reconciliation failed: the observed residual does not "
            "reproduce the forensics Q1 reconciliation to remainder 0.0 on "
            "train; refusing to proceed."
        )

    per_seed: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"seed_{seed}"
        if key in cache:
            if verbose:
                print(f"seed {seed}: cached")
            per_seed.append(cache[key])
            continue
        result = score_seed(
            seed,
            panel,
            fert,
            demo,
            death_records,
            mh_records,
            birth_records,
            order_map,
            floor,
            tol,
            report_only,
            verbose,
        )
        cache[key] = json.loads(json.dumps(result, default=c1._json_default))
        c1._save_cache(cache_path, cache)
        per_seed.append(cache[key])

    # Fresh-run artifact-schema blocks (amendment 1).
    per_draw_cube = _per_draw_per_cell_rates_block(per_seed, tol)
    undefined_block = _undefined_draw_block(per_seed)
    dispersion_block = _per_draw_dispersion_block(per_seed, tol)

    # The undefined-draw rule is pre-specified: any undefined gated draw
    # invalidates the run (no verdict may be certified from an invalid run).
    if undefined_block["run_invalidated"]:
        raise RuntimeError(
            "RUN INVALIDATED (fresh_run_artifact_schema.undefined_draw_rule): "
            f"{undefined_block['n_undefined_gated_draws']} undefined gated "
            "cell draw(s) (empty simulated denominator); the run must be "
            "re-registered and re-run. No draw may be dropped or substituted."
        )

    verdict = c1.build_verdict(per_seed, tol)
    report_block = c1.report_only_summary(per_seed, report_only)
    seed_conjunction = [
        {
            "seed": s["seed"],
            "n_gated_pass": s["n_gated_pass"],
            "n_gated_fail": s["n_gated_fail"],
            "seed_pass": s["seed_pass"],
        }
        for s in per_seed
    ]

    modal = _modal_failure_check(verdict, per_seed)
    count_tilt = _count_cell_tilt(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 10",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "forensics_diagnostic": FORENSICS_REGISTRATION,
        "forensics_artifact": "runs/gate2_forensics_v1.json (#94)",
        "delta_vs_candidate8": DELTA_VS_CANDIDATE8,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate; the "
            "first fresh registration under the amended estimator"
        ),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79); amendment 1 flipped "
            "live (#97). Protocol/views/tolerances/schema read at runtime; no "
            "threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.65-0.75",
            "conjunction_estimate": 0.70,
            "modal_failure": (
                "the count cells, if age conditioning under- or over-removes "
                "the in-exposure over-production so delta 1's +residual and "
                "delta 2's removed over-production fail to cancel (the "
                "cancellation is designed, not measured); secondary: thin 50+ "
                "remarriage cells misfiring under the added stratum"
            ),
            "registration": SPEC_REGISTRATION,
        },
        "model": _model_block(),
        "protocol": {
            "option": (
                "a (gate-1 mirror; LOCKED gates.yaml gate_2, amendment 1 live)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "the holdout, side B = the train complement"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "single_draw_provenance_rng_rule": (
                "numpy.random.default_rng(4200 + seed) (candidates 1-9; "
                "retained for provenance only, not scored)"
            ),
            "one_sequence_per_person_per_draw": True,
            "scored_against": (
                "side A's own empirical rate (rate_a in "
                "runs/gate2_floors_v2.json noise_floor_per_seed)"
            ),
            "statistic": (
                "|ln(rbar_candidate,s / rate_a,s)| per cell, rbar the 20-draw "
                "mean rate, scored once (NOT the mean of per-draw scores)"
            ),
            "conjunction": (
                "all 46 gated cells per seed AND >= 4 of 5 gate seeds"
            ),
            "weight_definition": (
                "person-constant most-recent positive PSID cross-sectional "
                "weight; every gated statistic weighted, none unweighted"
            ),
        },
        "fresh_run_artifact_schema": {
            "applies_to": (
                "the fresh candidate-10 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4917059482)"
            ),
            "per_draw_per_cell_rates": per_draw_cube,
            "undefined_draw_rule": undefined_block,
            "per_draw_dispersion_disclosure": dispersion_block,
        },
        "data": data_meta,
        "precheck": precheck,
        "delta1_reconciliation": reconciliation,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "count_cell_tilt": count_tilt,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate9_registration": CANDIDATE9_REGISTRATION,
            "candidate8_registration": CANDIDATE8_REGISTRATION,
            "forensics_diagnostic": "runs/gate2_forensics_v1.json",
            "floor_run": "runs/gate2_floors_v2.json",
            "faithful_candidate_oc": floor["faithful_candidate_oc"][
                "p_gate_pass_4_of_5"
            ],
        },
        "revision_pins": _revision_pins(thresholds),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_2_pass={v['gate_2_pass']} "
            f"({v['n_seeds_pass']}/5 seeds pass)"
        )
        print(f"seed_pass: {v['seed_pass']}")
        ct = count_tilt
        print(
            "count-cell cancellation "
            f"succeeded={ct['designed_cancellation_succeeded']}: "
            + "; ".join(
                f"{c} mean_signed_ln_tilt="
                f"{ct['cells'][c]['mean_signed_ln_tilt']:.4f} "
                f"({ct['cells'][c]['n_seeds_pass']}/5 pass)"
                for c in COUNT_CELLS
            )
        )
        print(
            f"decider={modal['decider_analysis']['decider']}; "
            f"distinct failing cells={modal['distinct_failing_cells']}"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    artifact = run(verbose=True, cache_path=Path(args.cache))
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
