"""Gate-2 candidate 4 (run 1): candidate 3 + two named fixes.

The FOURTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4911532899 (``SPEC_REGISTRATION``): candidate 3's frozen spec
(comment 4911357564) verbatim EXCEPT two named deltas. One-shot; no
constant moves after the registration comment.

The two deltas vs candidate 3 (everything else byte-identical)
--------------------------------------------------------------
1. **Spouse-death hazard (replaces the decade-period table + shrinkage
   entirely).** Candidates 2 and 3 stratified the spouse-death table by
   decade-period x age band x sex (candidate 2 add-one smoothed, candidate
   3 shrunk toward the pooled rate with ``K = 500``); under both, the female
   widowed-stock cluster (``share_widowed.65-74|female``,
   ``share_widowed.75+|female``, ``widowhood.45-64|female``) failed every
   seed, because the drift lives in the thin early-period cells a decade
   table cannot resolve. Candidate 4 drops the decade table and the
   shrinkage constant altogether and models the period effect
   **parametrically**:

       rate(band, sex, year) = pooled(band, sex) * exp(beta_sex * (year - 1995))

   where ``pooled(band, sex)`` is candidate 1's pooled (time-invariant)
   weighted central rate for that age band x sex
   (``build_mortality_floors.weighted_hazards``'s ``psid_m`` on the train
   slices -- the exact table candidate 1 integrates), the 1995 anchor, and
   ``beta_sex`` is a **single per-sex log-linear period slope** fit by a
   **weighted Poisson (log-linear) GLM** on the train ``(age band x
   start_wave)`` death cells with **age-band fixed effects** (one slope per
   sex, all ages pooled). The GLM soaks up the age-band baseline in its
   fixed effects so ``beta_sex`` estimates the pure period trend, free of
   the age-composition shift across the panel; the slope is applied to the
   pooled table at the simulated calendar year. Thin-cell-proof by
   construction: the slope pools every age within a sex, and no cell-wise
   period rate is formed anywhere.
2. **Spousal age where the spouse's record lacks it: drawn from the train
   sex-specific empirical age-gap distribution (1-year bins) instead of the
   mean.** Candidates 1-3 imputed the simulated spouse's age at the person's
   age + the train *mean* spousal age gap by the person's sex, so every
   imputed spouse of a given sex carried the same age. Candidate 4 draws
   each person's imputed spousal age gap **from the empirical 1-year-binned
   distribution** of that sex's train gaps (the same record selection whose
   mean candidates 1-3 used), so widowhood timing inherits the gap variance
   -- targeting the 45-64 female incidence miss. The per-person draw comes
   from an RNG stream **spawned from the registered
   ``numpy.random.default_rng(4200 + seed)``** (``SeedSequence.spawn``); the
   spawn does not advance the registered generator's bit stream, so the
   per-year uniform blocks (``rng.random(n_active)`` then
   ``rng.random(n_fertile)``) are byte-identical to candidates 1-3 and the
   RNG-isolated fertility subprocess is bit-for-bit unchanged.

Everything else -- the first-marriage hazard (candidate 3's knot-at-22
spline 20/22/25/30/40 with the age-spline x sex + age-spline x cohort
design), divorce, remarriage, fertility, the competing-risk step, the RNG
rule ``numpy.random.default_rng(4200 + seed)``, one simulated sequence per
person, and the LOCKED gate-2 protocol (gates.yaml ``gate_2``, ratified PR
#79 + flip #81) -- is byte-identical to candidate 3. This runner IMPORTS
candidate 3's machinery (``run_gate2_candidate3``, which chains candidates 2
and 1) and reuses every unchanged function: the unchanged components come
straight from ``candidate3.fit_components`` (so first marriage, divorce,
remarriage, fertility, and the spousal-gap MEAN are provably identical to
candidate 3), and only the two delta'd fields are recomputed -- the
mortality representation (pooled anchor + per-sex Poisson trend, replacing
candidate 3's mortality table) and the spousal-age-gap DISTRIBUTION (added
alongside the retained mean). The scoring path, precheck, verdict assembly,
and report-only handling are candidate 1's, imported unchanged.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels -- the Poisson GLM is a self-contained IRLS).
Run from the repository root with the PSID history files staged::

    .venv/bin/python scripts/run_gate2_candidate4.py
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

# Candidate 3 supplies the machinery this build minimally deltas again: its
# knot-at-22 first-marriage fitter and design class (reused unchanged --
# neither delta touches first marriage), its pooled band x sex rate helper
# (``pooled_band_sex_rates`` = candidate 1's exact pooled table, the delta-1
# anchor), and -- transitively, via candidate 1 -- the divorce / remarriage /
# fertility / spousal-gap-MEAN fitters, the vectorised simulation helpers,
# the precheck, the verdict assembly, and the report-only summary. Only the
# two delta'd fitters and the two delta'd simulation steps are
# re-implemented here. ``build_mortality_floors`` supplies the
# mortality-foundation construction (exposure slices, pooled weighted
# hazards, bands).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import build_mortality_floors as mort  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate2 as c2  # noqa: E402
import run_gate2_candidate3 as c3  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v4.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE3_ARTIFACT = ROOT / "runs" / "gate2_hazard_v3.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v4"
RUN_NAME = "gate2_hazard_v4"

#: This run's frozen-spec registration (issue #42, comment 4911532899).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911532899"
)
#: The candidate-3 spec this build minimally deltas (comment 4911357564).
CANDIDATE3_REGISTRATION = c3.SPEC_REGISTRATION
#: The candidate-2 spec candidate 3 minimally deltas (comment 4911167286).
CANDIDATE2_REGISTRATION = c2.SPEC_REGISTRATION
#: The candidate-1 spec candidate 2 minimally deltas (comment 4910914098).
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The two named deltas (registration comment 4911532899).
DELTAS_VS_CANDIDATE3 = (
    "spouse-death hazard replaces candidate 3's decade-period table + "
    "pooled-rate shrinkage entirely with a parametric per-sex log-linear "
    "period trend on candidate 1's pooled band x sex rate: rate(band, sex, "
    "year) = pooled(band, sex) * exp(beta_sex * (year - 1995)), beta_sex "
    "from a weighted Poisson/log-linear GLM on the train (band x start_wave) "
    "death cells with age-band fixed effects (one slope per sex, all ages "
    "pooled; thin-cell-proof, no cell-wise period rates)",
    "missing spousal ages drawn from the train sex-specific empirical "
    "age-gap distribution (1-year bins) instead of imputed at the train "
    "mean; per-person draw from an RNG stream spawned from the registered "
    "default_rng(4200 + seed) (isolated so the fertility subprocess stays "
    "byte-identical)",
)

# --- Frozen dials + band constants + pure helpers, reused from candidate 1
# (byte-identical; imported, never redefined). ---------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
EXACT_ATOL = c1.EXACT_ATOL

#: Candidate 3's first-marriage spline knots (kept for provenance/tests).
SPLINE_KNOTS_C3 = c3.SPLINE_KNOTS_C3  # (20, 22, 25, 30, 40)

DIV_BANDS = c1.DIV_BANDS
YSD_BANDS = c1.YSD_BANDS
ASFR_BANDS = c1.ASFR_BANDS
MORT_BANDS = c1.MORT_BANDS
DIV_LOWERS = c1.DIV_LOWERS
YSD_LOWERS = c1.YSD_LOWERS
ASFR_LOWERS = c1.ASFR_LOWERS
MORT_LOWERS = c1.MORT_LOWERS
_ASFR_LO = c1._ASFR_LO
_ASFR_HI = c1._ASFR_HI
_STATE = c1._STATE
_STATE_ABSORB = c1._STATE_ABSORB

_bands_vec = c1._bands_vec
_divorce_probs = c1._divorce_probs
_remarriage_probs = c1._remarriage_probs
_fertility_probs = c1._fertility_probs
_assemble_sim_panel = c1._assemble_sim_panel
Components = c1.Components

# DELTA 1 constants: the log-linear trend anchor year, and the a-priori
# Poisson-GLM IRLS convergence rule (disclosed, not tuned -- round,
# conservative constants chosen before any run; the fit is a full weighted
# MLE at these settings).
TREND_ANCHOR_YEAR = 1995.0
POISSON_MAX_ITER = 100
POISSON_TOL = 1e-12

# The candidate-4 first-marriage model IS candidate 3's (knot-at-22 spline);
# neither delta touches it. Aliased (not redefined) so the identity is
# provable: ``run_gate2_candidate4.fit_first_marriage is
# run_gate2_candidate3.fit_first_marriage``.
FirstMarriageModelC4 = c3.FirstMarriageModelC3
fit_first_marriage = c3.fit_first_marriage

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate4_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA 1: parametric per-sex log-linear mortality trend
# --------------------------------------------------------------------------
def _poisson_loglinear_irls(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    max_iter: int = POISSON_MAX_ITER,
    tol: float = POISSON_TOL,
) -> tuple[np.ndarray, int, bool]:
    """Unpenalised weighted Poisson (log-link) GLM by IRLS.

    Fits ``y ~ Poisson(mu)``, ``log(mu) = offset + x @ b`` (canonical link),
    the standard iteratively-reweighted-least-squares Newton step: working
    response ``z = (eta - offset) + (y - mu) / mu`` regressed on ``x`` with
    weights ``mu``. No penalty, so the fixed point is the exact weighted MLE
    (statsmodels is absent from the gate venv; this hand-rolled IRLS matches
    ``sklearn.PoissonRegressor`` at ``alpha -> 0`` to machine precision on
    validation). Returns ``(coefficients, n_iter, converged)``.
    """
    b = np.zeros(x.shape[1], dtype=np.float64)
    n_iter = 0
    converged = False
    while n_iter < max_iter:
        n_iter += 1
        eta = offset + x @ b
        mu = np.exp(eta)
        mu_safe = np.maximum(mu, 1e-300)
        z = (eta - offset) + (y - mu) / mu_safe
        xtw = x.T * mu  # weights = mu (Poisson canonical IRLS weight)
        b_new = np.linalg.solve(xtw @ x, xtw @ z)
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            converged = True
            break
        b = b_new
    return b, n_iter, converged


def fit_mortality_trend(
    slices: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-sex log-linear period slope ``beta_sex`` (DELTA 1).

    ``slices`` are the train exposure slices
    (``build_mortality_floors.build_exposure_slices`` restricted to the
    train persons). For each sex the train slices are aggregated to
    ``(age band x start_wave)`` cells -- weighted deaths ``Wd = sum(w *
    death)`` and weighted exposure ``We = sum(w * exposure)`` -- and a
    weighted Poisson (log-linear) GLM is fit with **age-band fixed effects**
    (one dummy per age band that carries any train death, no intercept) and
    a single period slope on ``start_wave - 1995``; the offset is
    ``log(We)``. Because the covariates are constant within a cell, the
    aggregated (Wd, We) Poisson fit has the identical score equations, hence
    the identical MLE, as the person-level weighted Poisson -- so this is the
    weighted Poisson regression on train deaths, exactly. The single
    coefficient on the period term is ``beta_sex``. Age bands with zero train
    weighted deaths carry no period signal and are dropped from the fit (their
    fixed effect is degenerate); the slope pools every surviving age band.
    Returns ``({sex: beta}, diagnostics)``.
    """
    df = slices.copy()
    df["we"] = df["weight"] * df["exposure"]
    df["wd"] = df["weight"] * df["death"]
    beta_by_sex: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for sex in ("female", "male"):
        sub = df[df["sex"] == sex]
        grouped = (
            sub.groupby(["band", "start_wave"], observed=True)
            .agg(we=("we", "sum"), wd=("wd", "sum"))
            .reset_index()
        )
        grouped = grouped[grouped["we"] > 0.0].copy()
        # Age-band fixed effects only for bands that carry train deaths.
        band_deaths = grouped.groupby("band", observed=True)["wd"].sum()
        live_bands = sorted(b for b, w in band_deaths.items() if w > 0.0)
        cells = grouped[grouped["band"].isin(live_bands)].copy()
        band_index = {b: i for i, b in enumerate(live_bands)}
        bi = cells["band"].map(band_index).to_numpy()
        t = cells["start_wave"].to_numpy(dtype=np.float64) - TREND_ANCHOR_YEAR
        dummies = np.zeros((len(cells), len(live_bands)), dtype=np.float64)
        dummies[np.arange(len(cells)), bi] = 1.0
        design = np.column_stack([dummies, t])  # band FE + period slope
        y = cells["wd"].to_numpy(dtype=np.float64)
        offset = np.log(cells["we"].to_numpy(dtype=np.float64))
        coef, n_iter, converged = _poisson_loglinear_irls(design, y, offset)
        beta_by_sex[sex] = float(coef[-1])
        diagnostics[sex] = {
            "beta": float(coef[-1]),
            "n_cells": int(len(cells)),
            "n_bands": len(live_bands),
            "bands": [str(b) for b in live_bands],
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "weighted_deaths": float(y.sum()),
            "weighted_exposure": float(cells["we"].sum()),
            "start_wave_min": int(cells["start_wave"].min()),
            "start_wave_max": int(cells["start_wave"].max()),
        }
    return beta_by_sex, diagnostics


# --------------------------------------------------------------------------
# DELTA 2: train sex-specific empirical spousal-age-gap distribution
# --------------------------------------------------------------------------
def spousal_gap_distribution(
    mh_records: pd.DataFrame,
    train_ids: set[int],
) -> dict[str, np.ndarray]:
    """Train sex-specific empirical spousal age-gap arrays (1-year bins).

    The SAME record selection as ``candidate1._spousal_gap`` (train
    self-persons whose marriage record joins a spouse with a known birth
    year; ``gap = self_birth - spouse_birth = spouse_age - self_age``), but
    returns the per-sex integer-rounded (1-year-binned) gap ARRAY instead of
    its mean. Sampling uniformly from this array reproduces the empirical
    1-year-binned distribution; its mean equals candidate 1's ``gap_by_sex``
    (the gaps are integer year differences, so rounding is a no-op) -- the
    delta is exactly mean -> draw. Keyed by the person's sex.
    """
    person_birth = (
        mh_records.dropna(subset=["birth_year"])
        .groupby("person_id")["birth_year"]
        .first()
    )
    rec = mh_records[
        mh_records["is_marriage"]
        & mh_records["spouse_person_id"].notna()
        & mh_records["person_id"].isin(train_ids)
    ].copy()
    rec["self_birth"] = rec["person_id"].map(person_birth).astype("float64")
    rec["spouse_birth"] = (
        rec["spouse_person_id"].map(person_birth).astype("float64")
    )
    rec = rec[rec["self_birth"].notna() & rec["spouse_birth"].notna()]
    rec["gap"] = rec["self_birth"] - rec["spouse_birth"]
    dist: dict[str, np.ndarray] = {}
    for sex in ("female", "male"):
        sub = rec[rec["sex"] == sex]["gap"]
        if len(sub):
            dist[sex] = np.rint(sub.to_numpy(dtype=np.float64)).astype(
                np.int64
            )
        else:
            dist[sex] = np.zeros(1, dtype=np.int64)
    return dist


def _gap_dist_summary(dist: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compact per-sex summary of the gap distribution (for the artifact)."""
    out: dict[str, Any] = {}
    for sex, arr in dist.items():
        out[sex] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "sd": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "n_bins": int(np.unique(arr).size),
        }
    return out


# --------------------------------------------------------------------------
# Fitted components (candidate 3's, with the two delta'd fields swapped)
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
    """Fit all five components on side B, deltas 1 and 2 applied.

    Starts from :func:`candidate3.fit_components` -- so the first-marriage
    model (candidate 3's knot-at-22 spline), divorce, remarriage, fertility,
    and the spousal-gap MEAN are byte-identical to candidate 3 by
    construction, not by re-implementation. Then the two delta'd fields are
    swapped in:

    * DELTA 1 -- the spouse-death mortality REPRESENTATION becomes candidate
      1's pooled band x sex table (the 1995 anchor) plus the per-sex
      log-linear period slope :func:`fit_mortality_trend`; candidate 3's
      decade-period shrinkage table and its ``K`` are dropped entirely (their
      meta keys are removed so the artifact carries no stale period/shrinkage
      provenance).
    * DELTA 2 -- the empirical spousal-age-gap DISTRIBUTION
      (:func:`spousal_gap_distribution`) is attached alongside the retained
      mean; the simulation draws from it per person.
    """
    base = c3.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # DELTA 1: pooled band x sex anchor (candidate 1's exact pooled table)
    # plus the per-sex log-linear period trend; replaces candidate 3's
    # decade-period shrinkage table entirely.
    slices = mort.build_exposure_slices(demo, death_records)
    slices = slices[slices["person_id"].isin(train_ids)].copy()
    pooled = c3.pooled_band_sex_rates(slices)
    beta_by_sex, trend_diag = fit_mortality_trend(slices)
    base.mortality = pooled  # the 1995 anchor; keyed "band|sex"

    # DELTA 2: empirical spousal-age-gap distribution (kept off ``meta`` so
    # the artifact stays lean; a summary is recorded instead).
    gap_dist = spousal_gap_distribution(mh_records, train_ids)
    base.gap_dist_by_sex = gap_dist

    # --- Rewrite the mortality meta for the parametric trend (candidate 3's
    # shrinkage/period keys are removed; they do not describe candidate 4). ---
    for stale in (
        "mortality_periods",
        "mortality_prior_strength_K",
    ):
        base.meta.pop(stale, None)
    base.meta["mortality_cells"] = len(pooled)
    base.meta["mortality_stratification"] = (
        "pooled band x sex (candidate 1 anchor) x per-sex log-linear "
        "period trend"
    )
    base.meta["mortality_smoothing"] = (
        "none (parametric per-sex log-linear period trend replaces the "
        "decade-period table and the shrinkage constant)"
    )
    base.meta["mortality_trend"] = (
        "rate(band, sex, year) = pooled(band, sex) * "
        "exp(beta_sex * (year - 1995))"
    )
    base.meta["mortality_trend_anchor_year"] = TREND_ANCHOR_YEAR
    base.meta["mortality_trend_estimator"] = (
        "weighted Poisson (log-linear) GLM on train (band x start_wave) "
        "death cells, age-band fixed effects, one slope per sex, IRLS"
    )
    base.meta["mortality_beta_by_sex"] = beta_by_sex
    base.meta["mortality_trend_diagnostics"] = trend_diag

    # --- Spousal-gap distribution meta (the mean stays in gap_by_sex). ---
    base.meta["spousal_gap_imputation"] = (
        "per-person draw from the train sex-specific empirical age-gap "
        "distribution (1-year bins) via an RNG stream spawned from the "
        "registered default_rng(4200 + seed); mean retained in gap_by_sex "
        "for provenance"
    )
    base.meta["spousal_gap_dist_summary"] = _gap_dist_summary(gap_dist)
    base.meta["deltas_vs_candidate3"] = list(DELTAS_VS_CANDIDATE3)
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 3's, with the two deltas)
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC4:
    mort_arr: np.ndarray  # [mort_band, sex(0=f,1=m)]  pooled 1995 anchor
    beta_arr: np.ndarray  # [sex(0=f,1=m)]  per-sex log-linear period slope
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex(0=f,1=m)]
    fert_arr: np.ndarray  # [asfr_band, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC4:
    """Candidate 1's pooled/remarriage/fertility lookups + the period slope.

    The mortality lookup is candidate 1's pooled band x sex table (built by
    the reused :func:`candidate1._build_sim_lookups` from the ``"band|sex"``
    keys of the delta-1 anchor), and the per-sex log-linear slope
    ``beta_arr`` is read from the fitted meta. The remarriage and fertility
    lookups are candidate 1's, unchanged.
    """
    base = c1._build_sim_lookups(components)
    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)
    return _SimLookupsC4(
        mort_arr=base.mort_arr,
        beta_arr=beta_arr,
        rem_arr=base.rem_arr,
        fert_arr=base.fert_arr,
        decade_map=base.decade_map,
    )


def _widow_probs(
    spouse_age: np.ndarray,
    spouse_is_male: np.ndarray,
    year: int,
    mort_arr: np.ndarray,
    beta_arr: np.ndarray,
) -> np.ndarray:
    """Pooled band x sex rate times the per-sex log-linear period trend.

    ``rate = pooled(band, sex) * exp(beta_sex * (year - 1995))`` (DELTA 1),
    the spouse-death hazard at the simulated calendar ``year``. Banding is
    candidate 1's (``rint(spouse_age)`` into the mortality bands).
    """
    bands = _bands_vec(
        np.rint(spouse_age).astype(np.int64), MORT_LOWERS, len(MORT_BANDS)
    )
    sidx = spouse_is_male.astype(np.int64)
    pooled = mort_arr[bands, sidx]
    trend = np.exp(beta_arr[sidx] * (float(year) - TREND_ANCHOR_YEAR))
    return pooled * trend


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 3's simulation with the delta-1 trend and delta-2 gap draw.

    Byte-identical to :func:`candidate1.simulate_holdout` /
    :func:`candidate2.simulate_holdout` EXCEPT:

    * the spouse-death hazard is the pooled band x sex rate times the
      per-sex log-linear period trend at the simulated year (DELTA 1),
      replacing candidate 3's decade-period table lookup; and
    * each person's imputed spousal age gap is DRAWN from the train
      sex-specific empirical gap distribution (DELTA 2) rather than set to
      the mean.

    The gap draw uses an isolated stream spawned from the registered
    generator's seed sequence -- ``rng.bit_generator.seed_seq.spawn(1)``
    does not advance ``rng``'s bit stream -- so the per-year uniform blocks
    (``rng.random(n_active)`` then ``rng.random(n_fertile)``) are drawn in
    the same order and size as candidates 1-3, and the RNG-isolated
    fertility subprocess is byte-identical (test-pinned against candidate 1).
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

    # Observed initial state at entry (min-year person-year per person).
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
        else:  # separated / other
            state[i] = _STATE_ABSORB

    # The registered simulation RNG. The gap-draw stream is SPAWNED from it
    # (SeedSequence.spawn) before any per-year draw; the spawn does not
    # advance ``rng``'s bit stream, so ``rng.random`` below is byte-identical
    # to candidates 1-3 and the fertility subprocess is bit-for-bit unchanged.
    rng = np.random.default_rng(sim_seed)
    gap_seed_seq = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_seq)

    # DELTA 2: per-person imputed spousal age gap, drawn from the train
    # sex-specific empirical distribution (females then males, each in
    # person-id order -- a fixed, disclosed draw order). Candidates 1-3 used
    # the mean gap by sex here.
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
    opp_is_male = 1.0 - is_male  # spouse opposite sex

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
            # DELTA 1: pooled band x sex rate x per-sex log-linear trend at y.
            p_wid = _widow_probs(
                sp_age, opp_is_male[sub], y, lookups.mort_arr, lookups.beta_arr
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
            origin = st[diss]  # 2 divorced, 3 widowed
            p_rm = _remarriage_probs(
                ysd, origin, is_male[sub], lookups.rem_arr
            )
            rm = u[diss] < p_rm
            gri = sub[rm]
            order[gri] += 1
            cur_start[gri] = y
            state[gri] = 1
            diss_year[gri] = -1
            open_start[gri] = y
            open_order[gri] = order[gri]

        # Fertility: women aged 15-49, any marital state (incl. absorbed).
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
            p_birth = _fertility_probs(
                fage, parity[fidx], fert_didx[fidx], lookups.fert_arr
            )
            born = uf < p_birth
            gbi = fidx[born]
            for i in gbi:
                bi_person.append(int(pid[i]))
                bi_year.append(int(y))
                bi_order.append(int(parity[i]) + 1)
            parity[gbi] += 1

    # Close still-open marriages at censor (intact).
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
    return sim_panel, sim_births


# --------------------------------------------------------------------------
# Per-seed scoring (candidate 1's, calling the candidate-4 fit + simulate)
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
    """Fit side B, simulate side A, score every cell against rate_a.

    Identical to :func:`candidate1.score_seed` except it calls the
    candidate-4 :func:`fit_components` and :func:`simulate_holdout` (the two
    deltas). The split, scoring statistic, gated/report partition, and
    per-seed record are candidate 1's.
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
    sim_panel, sim_births = simulate_holdout(
        panel, ids_a, components, SIM_SEED_BASE + seed
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    cand = transitions.reference_moments(
        sim_panel, sim_fert, ids_a, weighted=True
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]

    def score_cell(key: str) -> dict[str, Any]:
        rate_a = float(committed_cells[key]["rate_a"])
        r_cand = float(cand[key]["rate"])
        n_cand = int(cand[key]["n_events"])
        if r_cand > 0 and rate_a > 0:
            s = float(abs(math.log(r_cand / rate_a)))
        else:
            s = float("inf")
        return {
            "r_candidate": r_cand,
            "rate_a": rate_a,
            "n_events_candidate": n_cand,
            "log_ratio_abs": s if math.isfinite(s) else None,
            "score": s,
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
            f"(seed_pass={seed_pass}); fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "sim_seed": SIM_SEED_BASE + seed,
        "component_meta": components.meta,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Modal-failure check (registered c4 modal + widowed-stock-cluster decider)
# --------------------------------------------------------------------------
#: The registered modal failure (comment 4911532899): the sequence statistic
#: at its very tight 0.047 tolerance that persisted through candidate 3.
REGISTERED_MODAL_CELL = "mean_lifetime_marriages|male"
#: The female widowed-stock cluster that was candidate 3's ISOLATED failure
#: (each failed all five seeds under both time-invariant c1 and shrunk-period
#: c3 mortality) -- the cluster delta 1's parametric trend directly attacks.
WIDOWED_STOCK_CLUSTER = (
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)
#: Candidate 3's residual gated failures, tracked for movement vs candidate 3.
CANDIDATE3_RESIDUAL_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal, the widowed-stock cluster, and which decided it.

    Registered modal failure (comment 4911532899):
    ``mean_lifetime_marriages|male`` at its very tight 0.047 tolerance.
    Registered secondary: the 65-74 female stock partial fix. The check
    tracks both the modal and the female widowed-stock cluster the
    parametric trend targets, and computes -- by counterfactual -- whether
    the modal cell or the stock cluster is the decider of the verdict.
    """
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

    # Counterfactual decider analysis: for each candidate "cause" (the modal
    # cell alone, or the widowed-stock cluster), recompute how many seeds
    # would pass if ONLY that cause's cells were forgiven. If forgiving the
    # cluster flips the seed count to >= 4, the cluster is a decider;
    # likewise the modal. This directly answers "modal or cluster decided
    # it".
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

    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven({REGISTERED_MODAL_CELL})
    n_pass_no_cluster = seeds_pass_if_forgiven(set(WIDOWED_STOCK_CLUSTER))
    n_pass_no_both = seeds_pass_if_forgiven(
        {REGISTERED_MODAL_CELL, *WIDOWED_STOCK_CLUSTER}
    )
    modal_failed = REGISTERED_MODAL_CELL in fails_by_cell
    cluster_failed = any(c in fails_by_cell for c in WIDOWED_STOCK_CLUSTER)
    gate_pass = verdict["gate_2_pass"]

    if gate_pass:
        decider = "none (gate passed)"
    else:
        modal_flips = n_pass_no_modal >= 4
        cluster_flips = n_pass_no_cluster >= 4
        if modal_flips and cluster_flips:
            decider = (
                "both independently decisive (forgiving either the modal or "
                "the cluster alone flips the gate to pass)"
            )
        elif cluster_flips:
            decider = "widowed_stock_cluster"
        elif modal_flips:
            decider = "mean_lifetime_marriages|male (the registered modal)"
        elif n_pass_no_both >= 4:
            decider = (
                "modal AND cluster jointly (forgiving both flips the gate; "
                "neither alone suffices)"
            )
        else:
            decider = (
                "broader than the modal + cluster (other gated cells also "
                "hold the gate below 4 passing seeds)"
            )

    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (the lifetime-marriage sequence "
            "statistic at its very tight 0.047 tolerance; persisted through "
            "candidate 3)"
        ),
        "modal_cell": REGISTERED_MODAL_CELL,
        "modal_failed": modal_failed,
        "modal_failed_seeds": sorted(
            fails_by_cell.get(REGISTERED_MODAL_CELL, [])
        ),
        "modal_track": track(REGISTERED_MODAL_CELL),
        "registered_secondary": (
            "share_widowed.65-74|female partial fix (the 65-74 female stock)"
        ),
        "widowed_stock_cluster": list(WIDOWED_STOCK_CLUSTER),
        "widowed_stock_cluster_failed": cluster_failed,
        "widowed_stock_cluster_track": {
            c: track(c) for c in WIDOWED_STOCK_CLUSTER
        },
        "any_materialized": modal_failed,
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_cluster_forgiven": n_pass_no_cluster,
            "n_seeds_pass_if_both_forgiven": n_pass_no_both,
            "decider": decider,
        },
        "candidate3_residual_track": {
            c: track(c) for c in CANDIDATE3_RESIDUAL_CELLS
        },
    }


def _candidate3_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-3 -> candidate-4 movement for the tracked cluster.

    Reads the committed candidate-3 artifact (``runs/gate2_hazard_v3.json``)
    and records, for the registered modal and the widowed-stock cluster,
    each cell's per-seed candidate-3 and candidate-4 scores and pass flags --
    so the artifact carries the vs-candidate-3 movement directly (the mean of
    the |ln| score across seeds, and the pass-count change).
    """
    if not CANDIDATE3_ARTIFACT.exists():
        return {"available": False}
    a3 = json.loads(CANDIDATE3_ARTIFACT.read_text())
    by3 = {s["seed"]: s for s in a3["per_seed"]}
    by4 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by4)
    cells = (REGISTERED_MODAL_CELL, *WIDOWED_STOCK_CLUSTER)
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in cells:
        c3_scores = {s: by3[s]["gated_cells"][cell]["score"] for s in seeds}
        c4_scores = {s: by4[s]["gated_cells"][cell]["score"] for s in seeds}
        c3_pass = sum(by3[s]["gated_cells"][cell]["pass"] for s in seeds)
        c4_pass = sum(by4[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "tolerance": by4[seeds[0]]["gated_cells"][cell]["tolerance"],
            "candidate3_per_seed_score": c3_scores,
            "candidate4_per_seed_score": c4_scores,
            "candidate3_mean_score": float(np.mean(list(c3_scores.values()))),
            "candidate4_mean_score": float(np.mean(list(c4_scores.values()))),
            "candidate3_n_seeds_pass": c3_pass,
            "candidate4_n_seeds_pass": c4_pass,
        }
    return out


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-4 schema + c1/c2/c3 shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for n in (1, 2, 3):
        pins[f"candidate{n}_runner"] = f"scripts/run_gate2_candidate{n}.py"
        pins[f"candidate{n}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{n}.py"
        )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 3's model block, edited for the two candidate-4 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 3 + two named "
            "fixes: a parametric per-sex log-linear mortality period trend; "
            "an empirical spousal-age-gap distribution draw)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "deltas_vs_candidate3": list(DELTAS_VS_CANDIDATE3),
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic (restricted) "
                "spline on age, knots 20/22/25/30/40, sex, birth-decade "
                "cohort; main effects + age-spline x sex + age-spline x "
                "cohort interactions; sklearn LogisticRegression(penalty="
                "'l2', C=1.0, lbfgs), sample_weight = person-year PSID "
                "weight -- BYTE-IDENTICAL to candidate 3 (neither delta "
                "touches first marriage)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x "
                "marriage order (1st vs 2+), add-one (Laplace) smoothed at "
                "the train mean married person-year weight (byte-identical "
                "to candidates 1-3)"
            ),
            "widowhood": (
                "COMPOSED and now PARAMETRIC in period (DELTA 1): the "
                "spouse-death hazard is candidate 1's pooled band x sex "
                "weighted central rate (the 1995 anchor) times a per-sex "
                "log-linear period trend, rate(band, sex, year) = "
                "pooled(band, sex) * exp(beta_sex * (year - 1995)); beta_sex "
                "is fit by a weighted Poisson/log-linear GLM on the train "
                "(band x start_wave) death cells with age-band fixed effects "
                "(one slope per sex, all ages pooled). Candidate 3's "
                "decade-period shrinkage table and its K = 500 constant are "
                "dropped entirely. The spouse is opposite sex; the spousal "
                "age gap is DELTA 2's empirical draw; widowhood = induced "
                "transition"
            ),
            "remarriage": (
                "weighted empirical hazard by years-since-dissolution band x "
                "origin (divorced/widowed) x sex, add-one smoothed at the "
                "train mean dissolved person-year weight (byte-identical to "
                "candidates 1-3)"
            ),
            "fertility": (
                "weighted empirical age-band x parity (0/1/2/3+) rates by "
                "birth-decade cohort, train-estimated (no smoothing; "
                "byte-identical to candidate 1, and RNG-isolated from the "
                "marital process so its per-seed outcomes reproduce "
                "candidate 1 bit-for-bit -- the delta-2 gap draw is spawned "
                "off the registered stream and cannot perturb it)"
            ),
        },
        "registered_ambiguity_resolutions": {
            "mortality_trend": (
                "rate(band, sex, year) = pooled(band, sex) * exp(beta_sex * "
                "(year - 1995)); pooled(band, sex) = candidate 1's pooled "
                "weighted central rate (build_mortality_floors."
                "weighted_hazards.psid_m on the train slices), the 1995 "
                "anchor; beta_sex = the single period-slope coefficient of a "
                "weighted Poisson (log-linear) GLM on the train (band x "
                "start_wave) weighted death cells (response = weighted "
                "deaths, offset = log weighted exposure) with age-band fixed "
                "effects (one dummy per train-death-bearing band, no "
                "intercept) and one slope per sex, fit by unpenalised IRLS "
                "(max_iter 100, tol 1e-12; the aggregated (Wd, We) Poisson "
                "has the identical MLE as the person-level weighted Poisson) "
                "-- thin-cell-proof: the slope pools every age within a sex, "
                "and no cell-wise period rate is formed"
            ),
            "spousal_gap_distribution": (
                "each simulated person's imputed spousal age gap is drawn "
                "from the train sex-specific empirical 1-year-binned gap "
                "distribution (the SAME record selection whose mean "
                "candidates 1-3 used; gap = self_birth - spouse_birth, "
                "integer year differences, so the 1-year bins are exact and "
                "the distribution mean equals candidate 1's gap_by_sex). The "
                "per-person draw (numpy Generator.choice, with replacement) "
                "comes from a stream SPAWNED from the registered "
                "default_rng(4200 + seed) via SeedSequence.spawn(1); the "
                "spawn does not advance the registered generator, so the "
                "per-year u/uf uniform blocks are byte-identical to "
                "candidates 1-3 and fertility is unperturbed. Draw order: "
                "females then males, each in person-id order"
            ),
            "everything_else": (
                "the first-marriage hazard (candidate 3's knot-at-22 "
                "20/22/25/30/40 spline with the age-spline x sex + "
                "age-spline x cohort design), divorce, remarriage, "
                "fertility, the competing-risk step, the RNG rule "
                "default_rng(4200 + seed), one sequence per person, and the "
                "locked protocol are byte-identical to candidate 3"
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

    mh_records = marriage.marriage_history()
    birth_records = g2f.births.birth_history()
    death_records = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, fert, data_meta = g2f.load_panels()
    order_map = c1._order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons"
        )

    # Hard-stop precheck BEFORE any candidate is simulated (candidate 1's).
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
    candidate3_comparison = _candidate3_comparison(per_seed)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 4",
        "spec_registration": SPEC_REGISTRATION,
        "candidate3_registration": CANDIDATE3_REGISTRATION,
        "candidate2_registration": CANDIDATE2_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "deltas_vs_candidate3": list(DELTAS_VS_CANDIDATE3),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 merge 82006877 + flip "
            "#81); protocol/views/tolerances read at runtime, no threshold "
            "moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.45-0.55",
            "conjunction_estimate": 0.50,
            "component_probabilities": {
                "trend_attacks_stock_drift_mechanism": 0.7,
                "widowed_stock_cluster_holds": 0.55,
                "share_widowed_65-74_female_fixes": 0.65,
                "mean_lifetime_marriages_male_passes": 0.6,
            },
            "modal_failure": (
                "mean_lifetime_marriages|male at its 0.047 boundary; "
                "secondary: share_widowed.65-74|female partial fix"
            ),
            "secondary_failure": [
                "share_widowed.65-74|female (65-74 female stock partial fix)"
            ],
            "diagnostic_flag": (
                "if candidate 4 fails ONLY on mean_lifetime_marriages|male at "
                "<= 0.07, the next step is a DIAGNOSTIC (not a candidate): "
                "whether that cell's tolerance is floor-coherent for sequence "
                "statistics, taken through the amendment process if warranted "
                "-- flagged pre-run so it cannot be a post-hoc rescue"
            ),
            "deltas_vs_candidate3": list(DELTAS_VS_CANDIDATE3),
            "registration": SPEC_REGISTRATION,
        },
        "model": _model_block(),
        "protocol": {
            "option": "a (gate-1 mirror; LOCKED gates.yaml gate_2)",
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "the holdout, side B = the train complement"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "sim_rng_rule": "numpy.random.default_rng(4200 + seed)",
            "one_sequence_per_person": True,
            "scored_against": (
                "side A's own empirical rate (rate_a in "
                "runs/gate2_floors_v2.json noise_floor_per_seed)"
            ),
            "statistic": "|ln(r_candidate / rate_a)| per cell",
            "conjunction": (
                "all 46 gated cells per seed AND >= 4 of 5 gate seeds"
            ),
            "weight_definition": (
                "person-constant most-recent positive PSID cross-sectional "
                "weight (populace_dynamics.data.panels.demographic_panel); "
                "every gated statistic weighted, none unweighted"
            ),
        },
        "data": data_meta,
        "precheck": precheck,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "modal_failure_materialized": modal,
        "candidate3_comparison": candidate3_comparison,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "candidate3_registration": CANDIDATE3_REGISTRATION,
            "candidate2_registration": CANDIDATE2_REGISTRATION,
            "candidate1_registration": CANDIDATE1_REGISTRATION,
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
        print(
            "registered modal (mean_lifetime_marriages|male) materialized: "
            f"{modal['modal_failed']} (seeds {modal['modal_failed_seeds']}); "
            f"decider={modal['decider_analysis']['decider']}"
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
