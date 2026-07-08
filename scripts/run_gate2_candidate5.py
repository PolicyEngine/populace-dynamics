"""Gate-2 candidate 5 (run 1): candidate 4 + three named deltas.

The FIFTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4911788302 (``SPEC_REGISTRATION``): candidate 4's frozen spec
(comment 4911532899) verbatim EXCEPT three named deltas. One-shot; no
constant moves after the registration comment.

The three deltas vs candidate 4 (everything else byte-identical)
----------------------------------------------------------------
1. **Mortality trend source: EXTERNAL NCHS rather than train-fit.** The
   spouse-death hazard keeps candidate 4's functional form exactly --
   ``rate(age, sex, year) = pooled_PSID(age, sex) * exp(beta_sex *
   (year - 1995))`` with ``pooled_PSID`` candidate 1's pooled band x sex
   weighted central rate (the 1995 anchor, unchanged) -- but ``beta_sex`` is
   no longer the train-fit Poisson slope. It is fixed from EXTERNAL NCHS US
   life tables: the log-linear slope of the age-45-84-band-average ``qx``
   across THREE vintages per sex. Candidate 4's committed reference carries
   the most recent vintage (``nchs_life_tables_2023.json``); this run adds
   two historical vintages fetched with full provenance
   (``nchs_life_tables_2000.json`` -- NVSR 51-3, and
   ``nchs_life_tables_2010.json`` -- NVSR 63-7,
   ``scripts/fetch_nchs_life_tables_historical.py``). PSID sets the LEVEL;
   NCHS sets the TREND. This closes candidate 4's finding that the male PSID
   mortality trend is not robustly estimable within splits (``beta_male``
   flips positive on seed 1).
2. **Fertility: single-year-of-age within parity x cohort, triangular-kernel
   smoothed.** Candidate 1-4's fertility is a 5-year ASFR age-band x parity
   (0/1/2/3+) x birth-decade table. Candidate 5 estimates single-year-of-age
   rates within parity x cohort and kernel-smooths them over age (triangular
   kernel, bandwidth 3 years, pre-registered) -- targeting the ``asfr.20-24``
   boundary clip and the c1970s completed-fertility clip. Because the delta
   changes the fertility THRESHOLD only (not the number or order of the
   per-year uniform draws), the marital process is unperturbed; fertility
   byte-identity vs candidate 1 NO LONGER holds by design (delta 2 changes
   fertility), so the NEW construction is pinned by its own seed-0
   reproduction instead.
3. **Remarriage: duration-band hazards estimated separately by origin.** The
   registration fixes the remarriage duration-band hazards estimated
   separately by origin (after-divorce vs after-widowhood), same bands and
   smoothing as the current table. Candidate 1's remarriage table -- inherited
   byte-identically through candidates 2-4 -- is ALREADY this construction:
   an empirical weighted hazard keyed ``(ysd_band, origin, sex)`` with the
   numerator split by the dissolved marriage's ``origin`` and the denominator
   by the person-year ``marital_state`` (the same bands, the same add-one
   smoothing at the train mean dissolved person-year weight). So this delta
   is the current origin-split construction pinned explicitly; the remarriage
   HAZARD table is byte-identical to candidate 4, and the
   ``remarriage.after_divorce`` scoring cell moves only through the indirect
   effect of delta 1 (mortality) on the simulated marital histories.

Everything else -- the first-marriage hazard (candidate 3's knot-at-22
20/22/25/30/40 spline), divorce, the spousal-age-gap DISTRIBUTION draw
(candidate 4's delta 2), the pooled band x sex mortality anchor, the
competing-risk step, the RNG rule ``numpy.random.default_rng(4200 + seed)``,
the spawned gap-draw stream, one simulated sequence per person, and the
LOCKED gate-2 protocol -- is byte-identical to candidate 4. This runner
IMPORTS candidate 4's machinery (which chains candidates 3, 2, 1) and reuses
every unchanged function: the unchanged components come straight from
``candidate4.fit_components`` (so first marriage, divorce, remarriage, the
mortality pooled anchor, and the spousal-gap distribution are provably
identical to candidate 4), and only the two truly delta'd fields are
recomputed -- the mortality trend SOURCE (NCHS-external ``beta_sex``,
replacing the train-fit slope) and the fertility REPRESENTATION (single-year
triangular-kernel, replacing the 5-year band table). The scoring path,
precheck, and verdict assembly are candidate 1's, imported unchanged.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). The two historical NCHS references are
committed, so the gate run needs no network. Run from the repository root
with the PSID history files staged::

    .venv/bin/python scripts/run_gate2_candidate5.py
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

# Candidate 4 supplies the machinery this build minimally deltas again: its
# pooled band x sex mortality anchor + parametric-trend widowhood, its
# spousal-gap DISTRIBUTION draw, and -- transitively, via candidates 3/2/1 --
# the knot-at-22 first-marriage fitter, divorce / remarriage / fertility
# fitters, the vectorised simulation helpers, the precheck, the verdict
# assembly, and the report-only summary. Only the two delta'd fields (the
# mortality trend SOURCE and the fertility REPRESENTATION) are re-implemented.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import build_mortality_floors as mort  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate2 as c2  # noqa: E402
import run_gate2_candidate3 as c3  # noqa: E402
import run_gate2_candidate4 as c4  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v5.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE4_ARTIFACT = ROOT / "runs" / "gate2_hazard_v4.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2_hazard_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v5"
RUN_NAME = "gate2_hazard_v5"

#: This run's frozen-spec registration (issue #42, comment 4911788302).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911788302"
)
#: The candidate-4 spec this build minimally deltas (comment 4911532899).
CANDIDATE4_REGISTRATION = c4.SPEC_REGISTRATION
CANDIDATE3_REGISTRATION = c3.SPEC_REGISTRATION
CANDIDATE2_REGISTRATION = c2.SPEC_REGISTRATION
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The three named deltas (registration comment 4911788302).
DELTAS_VS_CANDIDATE4 = (
    "mortality trend beta_sex is fixed from EXTERNAL NCHS life tables rather "
    "than the train-fit Poisson slope: beta_sex = the log-linear slope of the "
    "age-45-84-band-average qx across three NCHS US life-table vintages "
    "(2000, 2010, 2023) per sex, applied with candidate 4's unchanged form "
    "rate(age, sex, year) = pooled_PSID(age, sex) * exp(beta_sex * "
    "(year - 1995)); PSID sets the level, NCHS sets the trend",
    "fertility is estimated at single-year-of-age within parity x cohort and "
    "kernel-smoothed over age (triangular kernel, bandwidth 3 years, "
    "pre-registered), replacing the 5-year ASFR age-band table; fertility "
    "byte-identity vs candidate 1 no longer holds by design and the new "
    "construction is pinned by its own seed-0 reproduction",
    "remarriage duration-band hazards estimated separately by origin "
    "(after-divorce vs after-widowhood), same bands and smoothing as the "
    "current table -- candidate 1's remarriage table (inherited byte-"
    "identically through candidates 2-4) is already this (ysd_band, origin, "
    "sex) origin-split construction, so this delta pins it explicitly and the "
    "remarriage hazard is byte-identical to candidate 4",
)

# --- Frozen dials + band constants + pure helpers, reused (byte-identical;
# imported, never redefined). ---------------------------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
EXACT_ATOL = c1.EXACT_ATOL
SPLINE_KNOTS_C3 = c3.SPLINE_KNOTS_C3  # (20, 22, 25, 30, 40)
TREND_ANCHOR_YEAR = c4.TREND_ANCHOR_YEAR  # 1995.0 (unchanged)

DIV_BANDS = c1.DIV_BANDS
YSD_BANDS = c1.YSD_BANDS
ASFR_BANDS = c1.ASFR_BANDS
MORT_BANDS = c1.MORT_BANDS
DIV_LOWERS = c1.DIV_LOWERS
YSD_LOWERS = c1.YSD_LOWERS
MORT_LOWERS = c1.MORT_LOWERS
_ASFR_LO = c1._ASFR_LO
_ASFR_HI = c1._ASFR_HI
_STATE = c1._STATE
_STATE_ABSORB = c1._STATE_ABSORB

_bands_vec = c1._bands_vec
_divorce_probs = c1._divorce_probs
_remarriage_probs = c1._remarriage_probs
_assemble_sim_panel = c1._assemble_sim_panel
_widow_probs = c4._widow_probs  # pooled(band,sex) * exp(beta_sex*(year-1995))
Components = c1.Components

# The candidate-5 first-marriage model IS candidate 3's (knot-at-22 spline);
# no delta touches it. Aliased so the identity is provable.
FirstMarriageModelC5 = c3.FirstMarriageModelC3
fit_first_marriage = c3.fit_first_marriage

# DELTA 1 constants: the external NCHS vintages and the age band whose
# average qx defines the per-sex log-linear trend slope. Anchor year 1995 is
# candidate 4's, unchanged.
NCHS_VINTAGE_YEARS = (2000, 2010, 2023)
NCHS_BAND_AGE_LO = 45
NCHS_BAND_AGE_HI = 84
NCHS_LIFE_TABLE_PATHS = {
    year: ROOT / "data" / "external" / f"nchs_life_tables_{year}.json"
    for year in NCHS_VINTAGE_YEARS
}

# DELTA 2 constants: single-year fertility age range and the triangular
# kernel bandwidth (pre-registered). The kernel weight at integer age offset
# d is max(0, 1 - |d| / BANDWIDTH); with bandwidth 3 the support is |d| <= 2,
# weights (1/3, 2/3, 1, 2/3, 1/3). The rate is smoothed exposure-weighted
# (numerator and denominator each convolved with the kernel, then divided),
# so the kernel's normalisation cancels in the ratio.
FERT_AGE_LO = _ASFR_LO  # 15
FERT_AGE_HI = _ASFR_HI  # 49
FERT_KERNEL_BANDWIDTH = 3

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate5_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA 1: external NCHS per-sex log-linear mortality trend
# --------------------------------------------------------------------------
def _band_average_qx(vintage: dict[str, Any], sex: str) -> float:
    """Arithmetic mean of qx over ages NCHS_BAND_AGE_LO..HI for one sex."""
    rows = {r["age"]: r for r in vintage["tables"][sex]}
    qs = [
        float(rows[age]["qx"])
        for age in range(NCHS_BAND_AGE_LO, NCHS_BAND_AGE_HI + 1)
    ]
    return float(np.mean(qs))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Closed-form OLS slope of ``y`` on ``x`` (deterministic)."""
    xbar = float(x.mean())
    ybar = float(y.mean())
    dx = x - xbar
    return float((dx @ (y - ybar)) / (dx @ dx))


def fit_mortality_trend_nchs() -> tuple[dict[str, float], dict[str, Any]]:
    """Per-sex log-linear period slope ``beta_sex`` from external NCHS (DELTA 1).

    Loads the three committed NCHS US life-table vintages (2000, 2010, 2023),
    computes the age-45-84-band-average ``qx`` per sex per vintage, and sets
    ``beta_sex`` to the OLS slope of ``ln(band_average_qx)`` on the vintage
    year -- the log-linear slope of the band-average qx across the three
    vintages. Returns ``({sex: beta}, diagnostics)`` where the diagnostics
    carry, per sex, the three per-vintage band-average qx, their logs, the
    fitted slope/intercept, and the fit residual; and, once, the vintage
    citations. No PSID data enters -- this is external truth.
    """
    vintages = {
        year: json.loads(NCHS_LIFE_TABLE_PATHS[year].read_text())
        for year in NCHS_VINTAGE_YEARS
    }
    years = np.array(NCHS_VINTAGE_YEARS, dtype=np.float64)
    beta_by_sex: dict[str, float] = {}
    per_sex: dict[str, Any] = {}
    for sex in ("female", "male"):
        qbar = np.array(
            [_band_average_qx(vintages[y], sex) for y in NCHS_VINTAGE_YEARS],
            dtype=np.float64,
        )
        ln_qbar = np.log(qbar)
        beta = _ols_slope(years, ln_qbar)
        intercept = float(ln_qbar.mean() - beta * years.mean())
        fitted = intercept + beta * years
        resid = ln_qbar - fitted
        ss_res = float(resid @ resid)
        ss_tot = float((ln_qbar - ln_qbar.mean()) @ (ln_qbar - ln_qbar.mean()))
        beta_by_sex[sex] = beta
        per_sex[sex] = {
            "beta": beta,
            "intercept": intercept,
            "band_average_qx_by_vintage": {
                str(int(y)): float(q)
                for y, q in zip(NCHS_VINTAGE_YEARS, qbar, strict=True)
            },
            "ln_band_average_qx_by_vintage": {
                str(int(y)): float(v)
                for y, v in zip(NCHS_VINTAGE_YEARS, ln_qbar, strict=True)
            },
            "fit_ss_resid": ss_res,
            "fit_r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        }
    citations = {
        str(int(year)): {
            "vintage_year": vintages[year]["vintage_year"],
            "nvsr_citation": vintages[year]["report"]["nvsr_citation"],
            "report_pdf_url": vintages[year]["report"]["report_pdf_url"],
            "source_format": vintages[year]["fetch"]["source_format"],
            "reference_json": (
                f"data/external/nchs_life_tables_{int(year)}.json"
            ),
        }
        for year in NCHS_VINTAGE_YEARS
    }
    diagnostics = {
        "vintages": list(NCHS_VINTAGE_YEARS),
        "band_age_lo": NCHS_BAND_AGE_LO,
        "band_age_hi": NCHS_BAND_AGE_HI,
        "anchor_year": TREND_ANCHOR_YEAR,
        "estimator": (
            "OLS slope of ln(age-45-84 band-average qx) on vintage year, "
            "one slope per sex, from three external NCHS US life-table "
            "vintages (2000, 2010, 2023)"
        ),
        "per_sex": per_sex,
        "vintage_citations": citations,
    }
    return beta_by_sex, diagnostics


# --------------------------------------------------------------------------
# DELTA 2: single-year-of-age fertility, triangular-kernel smoothed over age
# --------------------------------------------------------------------------
def _triangular_kernel_weights() -> dict[int, float]:
    """Triangular kernel weights at integer age offsets (bandwidth 3).

    ``w(d) = max(0, 1 - |d| / bandwidth)``; with bandwidth 3 the support is
    ``|d| <= 2`` with weights ``(1/3, 2/3, 1, 2/3, 1/3)``. The absolute
    normalisation is immaterial: the rate is the ratio of the kernel-smoothed
    numerator to the kernel-smoothed denominator, so a common scale cancels.
    """
    h = FERT_KERNEL_BANDWIDTH
    weights: dict[int, float] = {}
    for d in range(-(h - 1), h):
        w = 1.0 - abs(d) / h
        if w > 0.0:
            weights[d] = w
    return weights


def fit_fertility_single_year(
    panel: transitions.MaritalPanel,
    birth_records: pd.DataFrame,
    train_ids: set[int],
    birth_decade: pd.Series,
) -> dict[tuple[int, int, int], float]:
    """Single-year-of-age fertility within parity x cohort, kernel-smoothed.

    The SAME weighted numerator (mother-weighted births) and denominator
    (weighted woman-years) selection as :func:`candidate1._fit_fertility` --
    train women aged 15-49, running parity capped at 3, births censored at the
    mother's censor year -- but at single-year-of-age resolution instead of
    the 5-year ASFR band. Within each ``(parity_band, birth_decade)`` stratum
    the single-year numerator and denominator are each convolved over age with
    the pre-registered triangular kernel (bandwidth 3), and the smoothed rate
    is their ratio (0 where the smoothed denominator is 0). Keyed
    ``(age, parity_band, decade)`` for ages 15-49.
    """
    py = panel.person_years
    attrs = panel.attrs
    women_ids = set(attrs[attrs["sex"] == "female"]["person_id"]) & train_ids
    lo, hi = FERT_AGE_LO, FERT_AGE_HI
    wy = py[
        py["person_id"].isin(women_ids) & (py["age"] >= lo) & (py["age"] <= hi)
    ][["person_id", "year", "age", "weight"]].copy()

    be = g2f.births.birth_events(birth_records)
    be = be[
        (be["record_type"] == "birth")
        & be["parent_person_id"].isin(women_ids)
        & be["birth_year"].notna()
    ].copy()
    be = be.rename(columns={"parent_person_id": "person_id"})
    be["birth_year"] = be["birth_year"].astype("int64")
    births_by = {
        int(p): np.sort(g["birth_year"].to_numpy())
        for p, g in be.groupby("person_id")
    }

    wy = wy.reset_index(drop=True)
    wy["parity"] = c1._parity_vec(
        wy["person_id"].to_numpy(), wy["year"].to_numpy(), births_by
    )
    wy["decade"] = wy["person_id"].map(birth_decade).to_numpy()
    wy["parity_band"] = np.minimum(wy["parity"].to_numpy(), 3)

    attr_by = attrs.set_index("person_id")
    be["mother_birth"] = (
        be["person_id"].map(attr_by["birth_year"]).astype("float64")
    )
    be["mother_censor"] = (
        be["person_id"].map(attr_by["censor_year"]).astype("float64")
    )
    be["mother_age"] = be["birth_year"] - be["mother_birth"]
    be = be[
        (be["mother_age"] >= lo)
        & (be["mother_age"] <= hi)
        & (be["birth_year"] <= be["mother_censor"])
    ].reset_index(drop=True)
    be["decade"] = be["person_id"].map(birth_decade).to_numpy()
    be["weight"] = be["person_id"].map(attr_by["weight"]).to_numpy()
    be["parity"] = c1._parity_vec(
        be["person_id"].to_numpy(), be["birth_year"].to_numpy(), births_by
    )
    be["parity_band"] = np.minimum(be["parity"].to_numpy(), 3)

    den = (
        wy.groupby(["age", "parity_band", "decade"])["weight"].sum().to_dict()
    )
    num = (
        be.groupby(["mother_age", "parity_band", "decade"])["weight"]
        .sum()
        .to_dict()
    )

    # Strata present in the denominator (every woman-year cohort/parity cell).
    strata = {(int(pb), int(dec)) for (_a, pb, dec) in den}
    kernel = _triangular_kernel_weights()
    table: dict[tuple[int, int, int], float] = {}
    for pb, dec in strata:
        for age in range(lo, hi + 1):
            num_s = 0.0
            den_s = 0.0
            for d, w in kernel.items():
                a = age + d
                num_s += w * float(num.get((a, pb, dec), 0.0))
                den_s += w * float(den.get((a, pb, dec), 0.0))
            if den_s > 0.0:
                table[(age, pb, dec)] = num_s / den_s
    return table


def _fertility_probs_single(
    age: np.ndarray,
    parity: np.ndarray,
    didx: np.ndarray,
    fert_arr: np.ndarray,
) -> np.ndarray:
    """Single-year-of-age fertility probability lookup (DELTA 2).

    ``fert_arr`` is indexed ``[age - FERT_AGE_LO, parity_band, decade_idx]``;
    ages are clipped into 15-49 (the simulation only calls fertility in that
    range) and parity capped at 3. A person whose birth decade is absent from
    the train fertility table (``didx < 0``) gets probability 0, exactly as
    candidate 1's band lookup.
    """
    ai = np.clip(age - FERT_AGE_LO, 0, fert_arr.shape[0] - 1)
    pb = np.minimum(parity, 3)
    safe = np.where(didx >= 0, didx, 0)
    vals = fert_arr[ai, pb, safe]
    return np.where(didx >= 0, vals, 0.0)


def _fertility_meta(
    table: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    """Compact provenance summary of the single-year fertility table."""
    decades = sorted({d for (_a, _p, d) in table})
    ages = sorted({a for (a, _p, _d) in table})
    return {
        "n_cells": len(table),
        "age_resolution": "single_year_of_age",
        "age_range": [FERT_AGE_LO, FERT_AGE_HI],
        "n_ages": len(ages),
        "parity_bands": [0, 1, 2, 3],
        "decades": decades,
        "n_decades": len(decades),
    }


# --------------------------------------------------------------------------
# Fitted components (candidate 4's, with the two delta'd fields swapped and
# delta 3's already-origin-split remarriage pinned)
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
    """Fit all five components on side B, deltas 1-3 applied.

    Starts from :func:`candidate4.fit_components` -- so first marriage, divorce,
    remarriage, the pooled band x sex mortality anchor, and the spousal-gap
    DISTRIBUTION draw are byte-identical to candidate 4 by construction. Then:

    * DELTA 1 -- ``beta_sex`` is replaced by the external NCHS slope
      (:func:`fit_mortality_trend_nchs`); the pooled anchor and the applied
      functional form are unchanged. Candidate 4's train-fit betas are
      retained under a provenance key for the vs-candidate-4 comparison.
    * DELTA 2 -- the fertility table is replaced by the single-year-of-age
      triangular-kernel construction (:func:`fit_fertility_single_year`).
    * DELTA 3 -- remarriage is already the origin-split ``(ysd_band, origin,
      sex)`` table (inherited from candidate 1); it is left byte-identical and
      the origin split is recorded explicitly.
    """
    base = c4.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )
    attr_by = panel.attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")

    # DELTA 1: external NCHS per-sex trend replaces the train-fit slope. The
    # pooled band x sex anchor (base.mortality) is candidate 4's / candidate
    # 1's, unchanged. Retain candidate 4's train-fit betas for the comparison.
    c4_trainfit_beta = dict(base.meta["mortality_beta_by_sex"])
    c4_trainfit_diag = base.meta.get("mortality_trend_diagnostics")
    beta_nchs, nchs_diag = fit_mortality_trend_nchs()
    base.meta["mortality_beta_by_sex"] = beta_nchs
    base.meta["mortality_beta_by_sex_candidate4_trainfit"] = c4_trainfit_beta
    base.meta["mortality_trend_source"] = (
        "external NCHS US life tables (2000, 2010, 2023)"
    )
    base.meta["mortality_trend_estimator"] = (
        "beta_sex = OLS slope of ln(age-45-84 band-average qx) on vintage "
        "year across three external NCHS US life-table vintages (2000, 2010, "
        "2023); one slope per sex; PSID sets the level (pooled band x sex "
        "anchor), NCHS sets the trend"
    )
    base.meta["mortality_trend_diagnostics_nchs"] = nchs_diag
    base.meta["mortality_trend_diagnostics_candidate4_trainfit"] = (
        c4_trainfit_diag
    )

    # DELTA 2: single-year-of-age fertility, triangular-kernel smoothed.
    fertility = fit_fertility_single_year(
        panel, birth_records, train_ids, birth_decade
    )
    base.fertility = fertility
    base.meta["fertility_representation"] = (
        "single-year-of-age rates within parity (0/1/2/3+) x birth-decade "
        "cohort, kernel-smoothed over age (triangular kernel, bandwidth 3 "
        "years); numerator and denominator each convolved over age with the "
        "kernel, rate = ratio (replaces the 5-year ASFR age-band table)"
    )
    base.meta["fertility_kernel"] = "triangular"
    base.meta["fertility_kernel_bandwidth"] = FERT_KERNEL_BANDWIDTH
    base.meta["fertility_single_year_summary"] = _fertility_meta(fertility)

    # DELTA 3: remarriage is already origin-split (candidate 1's (ysd_band,
    # origin, sex) table, inherited byte-identically through candidates 2-4);
    # pinned explicitly, byte-identical hazard.
    base.meta["remarriage_origin_split"] = (
        "remarriage duration-band hazards estimated separately by origin "
        "(after-divorce vs after-widowhood), keyed (ysd_band, origin, sex); "
        "numerator split by the dissolved marriage's origin (prev_how_ended), "
        "denominator by the person-year marital_state; same bands and add-one "
        "smoothing as candidate 1 -- byte-identical to candidate 4"
    )

    base.meta["deltas_vs_candidate4"] = list(DELTAS_VS_CANDIDATE4)
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 4's, with delta-2 single-year
# fertility; delta 1 rides in through the fitted beta on candidate 4's trend)
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC5:
    mort_arr: np.ndarray  # [mort_band, sex(0=f,1=m)]  pooled 1995 anchor
    beta_arr: np.ndarray  # [sex(0=f,1=m)]  per-sex log-linear period slope
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex(0=f,1=m)]
    fert_arr: np.ndarray  # [age-FERT_AGE_LO, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC5:
    """Candidate 4's mort/remarriage lookups + the NCHS slope + single-year fert.

    The mortality anchor and per-sex slope and the remarriage table are
    candidate 4's / candidate 1's exactly; only the fertility lookup is the
    delta-2 single-year-of-age array (age index = ``age - FERT_AGE_LO``).
    """
    mort_arr = np.zeros((len(MORT_BANDS), 2), dtype=np.float64)
    for b, (lo, hi) in enumerate(MORT_BANDS):
        band = mort.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            mort_arr[b, si] = components.mortality.get(f"{band}|{sex}", 0.0)

    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)

    rem_arr = np.zeros((len(YSD_BANDS), 2, 2), dtype=np.float64)
    for (b, origin, sex), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[b, oi, si] = v

    decades = sorted({d for (_a, _p, d) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = FERT_AGE_HI - FERT_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1)), dtype=np.float64)
    for (age, pb, d), v in components.fertility.items():
        fert_arr[age - FERT_AGE_LO, pb, decade_map[d]] = v
    return _SimLookupsC5(mort_arr, beta_arr, rem_arr, fert_arr, decade_map)


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 4's simulation with delta-2 single-year fertility.

    Byte-identical to :func:`candidate4.simulate_holdout` EXCEPT the fertility
    probability is looked up at single-year-of-age from the delta-2 kernel-
    smoothed table (:func:`_fertility_probs_single`) instead of the 5-year
    ASFR band. Delta 1 rides in transparently through the fitted ``beta_arr``
    on candidate 4's unchanged widowhood trend. The per-year uniform blocks
    (``rng.random(n_active)`` then ``rng.random(n_fertile)``) are drawn in the
    same order and size as candidate 4 -- only the fertility THRESHOLD changes
    -- so the marital process is byte-identical and only fertility outcomes
    move; the spousal-gap draw remains the candidate-4 spawned-stream draw.
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
    # delta 2, retained): the spawn does not advance rng's bit stream.
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
            # DELTA 1 rides in via beta_arr (external NCHS slope) on candidate
            # 4's unchanged pooled-anchor x per-sex log-linear trend at year y.
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
            origin = st[diss]
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

        # Fertility: women aged 15-49, any marital state. DELTA 2: single-year
        # kernel-smoothed lookup (same uf draw block, only the threshold moves).
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
    return sim_panel, sim_births


# --------------------------------------------------------------------------
# Per-seed scoring (candidate 1's, calling the candidate-5 fit + simulate)
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
    candidate-5 :func:`fit_components` and :func:`simulate_holdout`.
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
# Modal-failure check + targeted-cell / candidate-4 movement
# --------------------------------------------------------------------------
#: The registered modal failure (comment 4911788302): the untargeted sequence
#: statistic at its very tight 0.047 tolerance -- the modal failure if
#: candidate 5 fails.
REGISTERED_MODAL_CELL = "mean_lifetime_marriages|male"
#: The four cells the three deltas target (registration component reads):
#: delta 1 -> 75+ female widowed stock; delta 2 -> asfr.20-24 and the c1970s
#: completed-fertility clip; delta 3 -> remarriage.after_divorce.
TARGETED_CELLS = (
    "share_widowed.75+|female",
    "asfr.20-24",
    "completed_fertility.c1970s",
    "remarriage.after_divorce",
)
#: The female widowed-stock cluster tracked since candidate 3.
WIDOWED_STOCK_CLUSTER = (
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal, the targeted cells, and which decided the verdict."""
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

    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven({REGISTERED_MODAL_CELL})
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    n_pass_no_both = seeds_pass_if_forgiven(
        {REGISTERED_MODAL_CELL, *TARGETED_CELLS}
    )
    modal_failed = REGISTERED_MODAL_CELL in fails_by_cell
    gate_pass = verdict["gate_2_pass"]

    if gate_pass:
        decider = "none (gate passed)"
    else:
        modal_flips = n_pass_no_modal >= 4
        targeted_flips = n_pass_no_targeted >= 4
        modal_is_sole = modal_failed and all(
            (c not in fails_by_cell) or c == REGISTERED_MODAL_CELL
            for c in {f["cell"] for f in verdict["all_failing_gated_cells"]}
        )
        if modal_flips and targeted_flips:
            decider = (
                "both independently decisive (forgiving either the modal or "
                "the targeted cells alone flips the gate to pass)"
            )
        elif modal_flips:
            decider = (
                "mean_lifetime_marriages|male (the registered modal alone "
                "holds the gate; forgiving it flips >=4 seeds to pass)"
            )
        elif targeted_flips:
            decider = "targeted_cells"
        elif n_pass_no_both >= 4:
            decider = (
                "modal AND targeted cells jointly (forgiving both flips the "
                "gate; neither alone suffices)"
            )
        else:
            decider = (
                "broader than the modal + targeted cells (other gated cells "
                "also hold the gate below 4 passing seeds)"
            )
        if modal_is_sole:
            decider += (
                " [mean_lifetime_marriages|male is the SOLE distinct failing "
                "gated cell]"
            )

    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (the untargeted lifetime-marriage "
            "sequence statistic at its very tight 0.047 tolerance; the modal "
            "failure if candidate 5 fails)"
        ),
        "modal_cell": REGISTERED_MODAL_CELL,
        "modal_failed": modal_failed,
        "modal_failed_seeds": sorted(
            fails_by_cell.get(REGISTERED_MODAL_CELL, [])
        ),
        "modal_track": track(REGISTERED_MODAL_CELL),
        "modal_is_sole_failing_cell": (
            len({f["cell"] for f in verdict["all_failing_gated_cells"]}) == 1
            and modal_failed
        ),
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "widowed_stock_cluster": list(WIDOWED_STOCK_CLUSTER),
        "widowed_stock_cluster_track": {
            c: track(c) for c in WIDOWED_STOCK_CLUSTER
        },
        "any_materialized": modal_failed,
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "n_seeds_pass_if_both_forgiven": n_pass_no_both,
            "decider": decider,
            "modal_decided": (not gate_pass) and modal_flips,
        },
    }


def _candidate4_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-4 -> candidate-5 movement for the tracked cells.

    Reads the committed candidate-4 artifact and records, for the registered
    modal and the four targeted cells, each cell's per-seed candidate-4 and
    candidate-5 scores and pass counts -- the vs-candidate-4 movement.
    """
    if not CANDIDATE4_ARTIFACT.exists():
        return {"available": False}
    a4 = json.loads(CANDIDATE4_ARTIFACT.read_text())
    by4 = {s["seed"]: s for s in a4["per_seed"]}
    by5 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by5)
    cells = (REGISTERED_MODAL_CELL, *TARGETED_CELLS)
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in cells:
        c4_scores = {s: by4[s]["gated_cells"][cell]["score"] for s in seeds}
        c5_scores = {s: by5[s]["gated_cells"][cell]["score"] for s in seeds}
        c4_pass = sum(by4[s]["gated_cells"][cell]["pass"] for s in seeds)
        c5_pass = sum(by5[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "tolerance": by5[seeds[0]]["gated_cells"][cell]["tolerance"],
            "candidate4_per_seed_score": c4_scores,
            "candidate5_per_seed_score": c5_scores,
            "candidate4_mean_score": float(np.mean(list(c4_scores.values()))),
            "candidate5_mean_score": float(np.mean(list(c5_scores.values()))),
            "candidate4_n_seeds_pass": c4_pass,
            "candidate5_n_seeds_pass": c5_pass,
        }
    return out


def _beta_comparison(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """NCHS betas (constant across seeds) vs candidate 4's per-seed train-fit."""
    nchs = per_seed[0]["component_meta"]["mortality_beta_by_sex"]
    trainfit = {
        s["seed"]: s["component_meta"][
            "mortality_beta_by_sex_candidate4_trainfit"
        ]
        for s in per_seed
    }
    return {
        "beta_sex_nchs": nchs,
        "beta_sex_candidate4_trainfit_per_seed": trainfit,
        "note": (
            "beta_sex_nchs is external and constant across seeds; the "
            "candidate-4 train-fit betas are per-seed (fit on each seed's "
            "train complement) and were unstable for males (beta_male could "
            "flip positive on seed 1)"
        ),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-5 schema + c1-c4 + NCHS shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for n in (1, 2, 3, 4):
        pins[f"candidate{n}_runner"] = f"scripts/run_gate2_candidate{n}.py"
        pins[f"candidate{n}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{n}.py"
        )
    pins["nchs_fetch_script"] = "scripts/fetch_nchs_life_tables_historical.py"
    pins["nchs_fetch_script_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "fetch_nchs_life_tables_historical.py"
    )
    pins["nchs_life_table_references"] = {
        str(year): {
            "path": f"data/external/nchs_life_tables_{year}.json",
            "sha256": c1._sha_of_file(NCHS_LIFE_TABLE_PATHS[year]),
        }
        for year in NCHS_VINTAGE_YEARS
    }
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 4's model block, edited for the three candidate-5 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 4 + three named "
            "fixes: an EXTERNAL NCHS per-sex mortality period trend; "
            "single-year-of-age triangular-kernel fertility; and an explicit "
            "origin-split remarriage table)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "deltas_vs_candidate4": list(DELTAS_VS_CANDIDATE4),
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort; main effects "
                "+ age-spline x sex + age-spline x cohort -- BYTE-IDENTICAL to "
                "candidate 4 (no delta touches first marriage)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order "
                "(1st vs 2+), add-one smoothed (byte-identical to candidates "
                "1-4)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; DELTA 1: the spouse-death "
                "hazard keeps candidate 4's form rate(band, sex, year) = "
                "pooled(band, sex) * exp(beta_sex * (year - 1995)) with the "
                "pooled band x sex anchor unchanged, but beta_sex is fixed "
                "from EXTERNAL NCHS life tables -- the OLS slope of the "
                "age-45-84 band-average qx across the 2000/2010/2023 US life-"
                "table vintages, one slope per sex (PSID sets the level, NCHS "
                "the trend). The spouse is opposite sex; the spousal age gap "
                "is candidate 4's empirical distribution draw"
            ),
            "remarriage": (
                "DELTA 3 (explicit): weighted empirical hazard by years-since-"
                "dissolution band x origin (divorced/widowed) x sex, "
                "estimated separately by origin, add-one smoothed at the train "
                "mean dissolved person-year weight -- candidate 1's origin-"
                "split table, inherited byte-identically through candidates "
                "2-4 and pinned here"
            ),
            "fertility": (
                "DELTA 2: single-year-of-age rates within parity (0/1/2/3+) x "
                "birth-decade cohort, kernel-smoothed over age (triangular "
                "kernel, bandwidth 3 years; numerator and denominator each "
                "convolved over age, rate = ratio), replacing the 5-year ASFR "
                "age-band table. RNG-isolated from the marital process (only "
                "the fertility threshold changes), so the marital cells are "
                "unperturbed by the delta; fertility byte-identity vs "
                "candidate 1 no longer holds by design"
            ),
        },
        "registered_ambiguity_resolutions": {
            "mortality_trend_source": (
                "beta_sex = OLS slope of ln(age-45-84 band-average qx) on the "
                "vintage year, using three NCHS US life-table vintages: 2000 "
                "(NVSR 51-3), 2010 (NVSR 63-7), 2023 (NVSR 74-6). "
                "'band-average qx' is the arithmetic mean of the single-year "
                "qx over ages 45-84 inclusive; the slope is the ordinary "
                "least-squares slope of the three (year, ln qbar) points. The "
                "applied rate is candidate 4's pooled_PSID(band, sex) * "
                "exp(beta_sex * (year - 1995)); only the SOURCE of beta_sex "
                "changes (train-fit Poisson -> external NCHS), not the anchor "
                "(1995) or the functional form"
            ),
            "fertility_kernel": (
                "single-year-of-age fertility rate cells (age x parity_band x "
                "birth-decade), with the numerator (mother-weighted births) "
                "and denominator (weighted woman-years) each convolved over "
                "age within each parity_band x decade stratum by the "
                "triangular kernel w(d) = max(0, 1 - |d| / 3) (support "
                "|d| <= 2, weights 1/3, 2/3, 1, 2/3, 1/3), and the smoothed "
                "rate is their ratio -- the exposure-weighted kernel-smoothed "
                "rate (kernel normalisation cancels in the ratio). The same "
                "numerator/denominator selection as candidate 1 (train women "
                "15-49, running parity capped at 3, births censored at the "
                "mother's censor year), only at single-year resolution"
            ),
            "remarriage_origin_split": (
                "candidate 1's remarriage table is already estimated "
                "separately by origin -- keyed (ysd_band, origin, sex), the "
                "numerator split by the dissolved marriage's origin "
                "(prev_how_ended -> divorced/widowed) and the denominator by "
                "the person-year marital_state, with the same 0-4/5-9/10+ "
                "bands and the same add-one smoothing at the train mean "
                "dissolved person-year weight. This delta pins that "
                "construction explicitly; the remarriage HAZARD is byte-"
                "identical to candidate 4, and remarriage.after_divorce moves "
                "only through delta 1's effect on the simulated marital "
                "histories"
            ),
            "everything_else": (
                "the first-marriage hazard (candidate 3's knot-at-22 "
                "20/22/25/30/40 spline), divorce, the spousal-gap distribution "
                "draw (candidate 4's delta 2), the pooled band x sex mortality "
                "anchor, the competing-risk step, the RNG rule "
                "default_rng(4200 + seed), the spawned gap-draw stream, one "
                "sequence per person, and the locked protocol are byte-"
                "identical to candidate 4"
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

    # DELTA 1 preflight: the three committed NCHS references must be present
    # so the external trend is reproducible offline.
    for year, path in NCHS_LIFE_TABLE_PATHS.items():
        if not path.exists():
            raise RuntimeError(
                f"NCHS life-table reference for {year} missing at {path}; "
                "run scripts/fetch_nchs_life_tables_historical.py first."
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
    candidate4_comparison = _candidate4_comparison(per_seed)
    beta_comparison = _beta_comparison(per_seed)
    nchs_citations = per_seed[0]["component_meta"][
        "mortality_trend_diagnostics_nchs"
    ]["vintage_citations"]

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 5",
        "spec_registration": SPEC_REGISTRATION,
        "candidate4_registration": CANDIDATE4_REGISTRATION,
        "candidate3_registration": CANDIDATE3_REGISTRATION,
        "candidate2_registration": CANDIDATE2_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "deltas_vs_candidate4": list(DELTAS_VS_CANDIDATE4),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 + flip #81); "
            "protocol/views/tolerances read at runtime, no threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.35-0.45",
            "conjunction_estimate": 0.40,
            "component_probabilities": {
                "share_widowed_75plus_female_fixes": 0.65,
                "fertility_clips_fix": 0.75,
                "remarriage_after_divorce_passes": 0.8,
                "mean_lifetime_marriages_male_passes": 0.45,
            },
            "modal_failure": (
                "mean_lifetime_marriages|male at its 0.047 tolerance (the "
                "untargeted sequence statistic; origin-split remarriage moves "
                "it, direction uncertain -- the modal failure if candidate 5 "
                "fails)"
            ),
            "component_reads": (
                "75+ female widowed stock ~0.65 (the NCHS male trend is "
                "strongly negative where PSID's was unstable -- the seed-1 "
                "failure mode should close); fertility clips ~0.75; "
                "remarriage.after_divorce ~0.8; the untargeted "
                "mean_lifetime_marriages|male at its 0.047 tolerance ~0.45"
            ),
            "next_step_if_fail": (
                "if candidate 5 fails with mean_lifetime_marriages as the sole "
                ">=3-seed cell, the next registration targets it directly "
                "(remarriage level calibration is the plausible mechanism), "
                "still under the one-shot rule"
            ),
            "deltas_vs_candidate4": list(DELTAS_VS_CANDIDATE4),
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
                "weight; every gated statistic weighted, none unweighted"
            ),
        },
        "data": data_meta,
        "precheck": precheck,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "modal_failure_materialized": modal,
        "candidate4_comparison": candidate4_comparison,
        "mortality_trend_beta_comparison": beta_comparison,
        "nchs_vintage_citations": nchs_citations,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "candidate4_registration": CANDIDATE4_REGISTRATION,
            "candidate3_registration": CANDIDATE3_REGISTRATION,
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
