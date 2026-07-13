"""Ceremony step 1 for ``gate_m6``: the temporal-holdout TRUTH-SIDE floors.

This builds ``runs/m6_holdout_floors_v1.json`` -- the deployment-scale
sampling null the proposed ``gate_m6`` (``docs/design/m6_projection_engine.md``
sec. 4, sec. 9) prices its tolerances against. It runs NO projection engine,
scores NO candidate, and writes NO gate: it computes the held-out PSID truth
(2015-2022) and the real-vs-real half-split floor of every proposed cell, then
partitions the surface into the gated family-A flows and the report-only
margins per the design's shock (sec. 4.1), attrition (sec. 4.4), mixed-k /
power (sec. 4.6) and measurability dispositions.

PSID wave facts (verified against the readers): interviews are biennial ODD
years from 1999 (``panels.demographic_panel`` periods ..., 2013, 2015, 2017,
2019, 2021, 2023); income REFERENCE years are even (``family.family_earnings``
..., 2014, 2016, 2018, 2020, 2022). So the boundary ``T* = 2014`` realizes its
seed state at the **2015 interview** (reference year 2014), and reference year
2021 is unobserved (odd) -- exactly the dual-axis pin of sec. 4.1.

Frozen once written (``artifacts.write_new`` refuses overwrite). Run from the
repository root with PSID staged at ``~/PolicyEngine/psid-data``::

    .venv-wt/bin/python scripts/build_m6_holdout_floors.py
"""

from __future__ import annotations

import hashlib
import math
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics import artifacts
from populace_dynamics.data import (
    deaths,
    disability,
    family,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.evaluation import derive_tolerance
from populace_dynamics.harness import moments
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m6_holdout_floors_v1.json"
SCHEMA_VERSION = "m6_holdout_floors.v1"
RUN = "m6_holdout_floors_v1"

# --------------------------------------------------------------------------
# Design-pinned constants (docs/design/m6_projection_engine.md sec. 4, sec. 10)
# --------------------------------------------------------------------------
T_STAR = 2014
SEED_WAVE = 2015  # interview realizing the reference-year-2014 (T*) state
#: Gated flow event years (sec. 4.1 flows axis): 2015-2019 annual.
GATED_FLOW_YEARS: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019)
#: Report-only shock flow event years (excess deaths + marriage collapse).
SHOCK_FLOW_YEARS: tuple[int, ...] = (2020, 2021)
#: Gated earnings reference years (sec. 4.1 earnings axis): 2 pre-shock waves.
GATED_EARN_YEARS: tuple[int, ...] = (2016, 2018)
#: The T* earnings anchor: the change 2014->2016 lands its endpoint in-holdout.
EARN_ANCHOR_YEAR = 2014
#: Report-only shock earnings reference years (2021 unobserved, odd).
SHOCK_EARN_YEARS: tuple[int, ...] = (2020, 2022)
#: Interview waves opening a gated flow interval (odd biennial).
GATED_START_WAVES: tuple[int, ...] = (2015, 2017, 2019)

FLOOR_SEEDS: tuple[int, ...] = tuple(range(100))
GATE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SPLIT_FRACTION = 0.5

#: Mixed-k discipline (sec. 4.6): FLOW k=3, STOCK k=4; margin check k=3.
K_FLOW = 3
K_STOCK = 4
MARGIN_K = 3
ROUNDING = 3
T_MAX = math.log(1.5)
T_MAX_SOURCE = "ln(1.5)"
#: Reliability floor: >= 20 events on the weaker half of the worst seed
#: (NCHS's own <20-death unreliable flag; the gate_m4 / mortality standard).
MIN_EVENTS_FOR_GATE = 20

#: OC weak-power threshold (sec. 4.9, sec. 9). Both directions:
#: the gate PAUSES for surface redesign if p_gate falls below the precedent
#: band floor (near-unpassable) OR the gated surface is empty / single-cell
#: (near-vacuous). The precedent p_gate band is gate_2a 0.9685 / 2b 0.9678 /
#: 2c 0.988 / m4 0.9689 / w1 0.9481-0.9623; the named floor is set at the
#: bottom of that lived band, rounded down.
WEAK_POWER_P_GATE_FLOOR = 0.90
#: Vacuity guard: at least this many gated family-A cells, and not every
#: gated tolerance pinned at the ln(1.5) cap.
MIN_GATED_CELLS_FOR_POWER = 4

SEXES = ("female", "male")

MORTALITY_BANDS = (
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 84),
    (85, 120),
)
#: Mortality bands demoted report-only PRE-LOCK for attrition (sec. 4.4, F7:
#: PSID panel attrition -- death-vs-dropout ambiguity + nursing-home non-
#: interview -- confounds the oldest-old death hazard beyond repair; "mortality
#: worst"). Younger retained cells' residual bias is disclosed, not demoted.
MORTALITY_ATTRITION_BANDS = ("85+",)
MARITAL_BANDS = ((18, 29), (30, 44), (45, 64), (65, 120))
DISABILITY_BANDS = (
    disability.DI_AGE_BANDS
)  # (20-29),(30-39),(40-49),(50-59),(60-66)
#: Earnings cohorts: age at reference year (a person-period cohort proxy).
EARN_COHORTS = (("prime", 25, 44), ("older", 45, 64))
MOBILITY_HORIZONS = (1, 2)
AUTOCORR_LAGS = (1, 2, 5)


def band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 120 else f"{lo}+"


def band_of(age: object, bands: tuple[tuple[int, int], ...]) -> str | None:
    if pd.isna(age):
        return None
    a = int(age)
    for lo, hi in bands:
        if lo <= a <= hi:
            return band_label(lo, hi)
    return None


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# The T*-anchored frame: household identity + F6 start-wave weight (sec. 9,
# sec. 4.7). Rule (written into the artifact): a holdout person's household_id
# and F6 weight are the family interview number (ER30001-scoped) and cross-
# sectional weight at the person's REALIZED START-OF-HOLDOUT WAVE -- the
# earliest gated interview wave (>= the 2015 T* seed) at which they are present
# -- held FIXED across the window (F6: not per-year re-weighted). Persons never
# present in a gated interview wave fall back to a singleton household_id
# (their own person_id); the count is disclosed.
# --------------------------------------------------------------------------
def build_anchor_frame(demo: pd.DataFrame) -> pd.DataFrame:
    """One row per holdout person: person_id, household_id, weight, anchor_wave.

    The holdout (closed-panel, sec. 4.8) universe is every person present in a
    gated start wave; the anchor is their earliest such wave.
    """
    present = demo[
        demo.period.isin(GATED_START_WAVES) & (demo.weight > 0)
    ].copy()
    present = present.sort_values(["person_id", "period"])
    first = present.groupby("person_id", as_index=False).first()
    return pd.DataFrame(
        {
            "person_id": first.person_id.to_numpy(),
            "household_id": first.interview.astype("int64").to_numpy(),
            "weight": first.weight.astype("float64").to_numpy(),
            "anchor_wave": first.period.astype("int64").to_numpy(),
        }
    )


def presence_by_wave(demo: pd.DataFrame) -> dict[int, set[int]]:
    """person_ids present (sequence 1-20, positive weight) per interview wave
    that opens a gated or shock flow interval."""
    out: dict[int, set[int]] = {}
    for wave in (2013, 2015, 2017, 2019, 2021):
        sub = demo[(demo.period == wave) & (demo.weight > 0)]
        out[wave] = set(sub.person_id.to_numpy().tolist())
    return out


# --------------------------------------------------------------------------
# Family-A cell tables (presence-conditioned, windowed, F6-weighted). Each
# builder returns a tidy long frame carrying person_id (for the split), the
# fixed F6 weight, and the cell attributes. Cells are computed later on a
# subset of person_ids (a floor half).
# --------------------------------------------------------------------------
def mortality_slices(
    demo: pd.DataFrame, death_records: pd.DataFrame, anchor: pd.DataFrame
) -> pd.DataFrame:
    """Single-year exposure/death slices (mortality precedent), F6-weighted,
    carrying calendar year for shock windowing. Presence-conditioned by
    construction (demographic_panel is sequence 1-20 present only)."""
    grid = sorted(int(w) for w in demo.period.unique())
    nxt = {w: grid[i + 1] for i, w in enumerate(grid[:-1])}
    obs = demo[(demo.age <= 120) & (demo.weight > 0)].copy()
    obs["sex"] = obs.person_id.map(death_records.set_index("person_id")["sex"])
    obs["dyear"] = obs.person_id.map(
        death_records.set_index("person_id")["death_year"]
    )
    obs = obs[obs.sex.isin(SEXES)]
    obs["nw"] = obs.period.map(nxt)
    obs = obs[obs.nw.notna()]
    obs["nw"] = obs.nw.astype(int)
    w = obs.period.to_numpy(np.int64)
    nw = obs.nw.to_numpy(np.int64)
    d = obs.dyear.astype("float64").to_numpy()
    die = obs.dyear.notna().to_numpy() & (d >= w) & (d < nw)
    frames = []
    s0d = die & (d == w)
    frames.append(
        pd.DataFrame(
            {
                "person_id": obs.person_id.to_numpy(),
                "sex": obs.sex.to_numpy(),
                "age": obs.age.to_numpy(np.int64),
                "cal_year": w,
                "exposure": np.where(s0d, 0.5, 1.0),
                "death": s0d.astype(float),
            }
        )
    )
    two = (nw - w == 2) & ~s0d
    s1d = die & (d == w + 1)
    frames.append(
        pd.DataFrame(
            {
                "person_id": obs.person_id.to_numpy()[two],
                "sex": obs.sex.to_numpy()[two],
                "age": (obs.age.to_numpy(np.int64) + 1)[two],
                "cal_year": (w + 1)[two],
                "exposure": np.where(s1d, 0.5, 1.0)[two],
                "death": s1d.astype(float)[two],
            }
        )
    )
    sl = pd.concat(frames, ignore_index=True)
    sl["band"] = sl.age.map(lambda a: band_of(a, MORTALITY_BANDS))
    sl = sl[sl.band.notna()]
    sl = sl.merge(anchor[["person_id", "weight"]], on="person_id", how="inner")
    window = np.where(
        sl.cal_year.isin(GATED_FLOW_YEARS),
        "gated",
        np.where(sl.cal_year.isin(SHOCK_FLOW_YEARS), "shock", "out"),
    )
    sl["window"] = window
    return sl[sl.window != "out"].reset_index(drop=True)


def marital_tables(
    death_records: pd.DataFrame,
    anchor: pd.DataFrame,
    present: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marital events + at-risk person-years, presence-conditioned on the
    opening biennial interview, F6-weighted, age x sex per transition."""
    mh = marriage.marriage_history()
    person_weight = anchor.set_index("person_id")["weight"]
    panel = transitions.build_marital_panel(mh, death_records, person_weight)

    def opening_wave(year: np.ndarray) -> np.ndarray:
        return np.where(year % 2 == 1, year, year - 1)

    def keep_present(frame: pd.DataFrame) -> pd.DataFrame:
        ow = opening_wave(frame.year.to_numpy())
        mask = np.array(
            [
                pid in present.get(int(w), set())
                for pid, w in zip(frame.person_id, ow, strict=True)
            ]
        )
        return frame.loc[mask]

    ev = panel.events.merge(
        anchor[["person_id", "weight"]],
        on="person_id",
        how="inner",
        suffixes=("_raw", ""),
    )
    ev = ev[ev.year.isin(GATED_FLOW_YEARS + SHOCK_FLOW_YEARS)].copy()
    ev["band"] = ev.age.map(lambda a: band_of(a, MARITAL_BANDS))
    ev = ev[ev.band.notna() & ev.sex.isin(SEXES)]
    ev = keep_present(ev)
    ev["window"] = np.where(ev.year.isin(GATED_FLOW_YEARS), "gated", "shock")

    py = panel.person_years.merge(
        anchor[["person_id", "weight"]],
        on="person_id",
        how="inner",
        suffixes=("_raw", ""),
    )
    py = py[py.year.isin(GATED_FLOW_YEARS + SHOCK_FLOW_YEARS)].copy()
    py["band"] = py.age.map(lambda a: band_of(a, MARITAL_BANDS))
    py = py[py.band.notna() & py.sex.isin(SEXES)]
    py = keep_present(py)
    py["window"] = np.where(py.year.isin(GATED_FLOW_YEARS), "gated", "shock")
    return ev.reset_index(drop=True), py.reset_index(drop=True)


#: at-risk marital state per transition (denominator).
MARITAL_AT_RISK = {
    "first_marriage": ("never_married",),
    "divorce": ("married",),
    "widowhood": ("married",),
    "remarriage": ("divorced", "widowed"),
}


def disability_pairs(
    status: pd.DataFrame,
    death_records: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    """Grid-adjacent disability transition pairs carrying the START wave, so
    the gated holdout intervals (start 2015, 2017; reproduction mode) window
    cleanly. Presence-conditioned (read_disability_status is present-only)."""
    with_sex = disability.attach_sex(status, death_records)
    s = with_sex.sort_values(["person_id", "period"])
    g = s.groupby("person_id", sort=False)
    s = s.assign(
        np_=g.period.shift(-1),
        nd_=g.disabled.shift(-1),
    )
    s["intv"] = s["np_"] - s["period"]
    keep = s.np_.notna() & (s.intv >= 1) & (s.intv <= 2)
    pairs = pd.DataFrame(
        {
            "person_id": s.person_id[keep].to_numpy(),
            "sex": s.sex[keep].to_numpy(),
            "age": s.age[keep].astype(np.int64).to_numpy(),
            "start_wave": s.period[keep].astype(np.int64).to_numpy(),
            "from_disabled": s.disabled[keep].to_numpy(bool),
            "to_disabled": s.nd_[keep].to_numpy(bool),
        }
    )
    pairs = pairs.merge(
        anchor[["person_id", "weight"]], on="person_id", how="inner"
    )
    pairs["band"] = pairs.age.map(lambda a: band_of(a, DISABILITY_BANDS))
    pairs = pairs[pairs.band.notna() & pairs.sex.isin(SEXES)]
    # gated: both endpoints in holdout (start 2015 or 2017 -> end 2017 / 2019);
    # start 2019 -> end 2021 shock (report-only).
    window = np.where(
        pairs.start_wave.isin((2015, 2017)),
        "gated",
        np.where(pairs.start_wave == 2019, "shock", "out"),
    )
    pairs["window"] = window
    return pairs[pairs.window != "out"].reset_index(drop=True)


def earnings_frame(ep: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Person-period earnings on the anchor + gated + shock reference years,
    F6-weighted, with an age-at-reference-year cohort. Presence-conditioned
    (family_earnings_panel is present, positive-weight, valid-earnings)."""
    years = (EARN_ANCHOR_YEAR,) + GATED_EARN_YEARS + SHOCK_EARN_YEARS
    e = ep[ep.period.isin(years)].copy()
    e = e.merge(
        anchor[["person_id", "weight"]],
        on="person_id",
        how="inner",
        suffixes=("_raw", ""),
    )

    def cohort_of(age: object) -> str | None:
        if pd.isna(age):
            return None
        a = int(age)
        for name, lo, hi in EARN_COHORTS:
            if lo <= a <= hi:
                return name
        return None

    e["cohort"] = e.age.map(cohort_of)
    e = e[e.cohort.notna()]
    return e.reset_index(drop=True)


# --------------------------------------------------------------------------
# Cell computation on a subset of persons (one floor half).
# --------------------------------------------------------------------------
def _rate(num_w: float, den_w: float) -> float:
    return float(num_w / den_w) if den_w > 0 else 0.0


def mortality_cells(sl: pd.DataFrame) -> dict[str, dict[str, float]]:
    g = sl.assign(we=sl.weight * sl.exposure, wd=sl.weight * sl.death)
    agg = g.groupby(["band", "sex"], observed=True).agg(
        we=("we", "sum"),
        wd=("wd", "sum"),
        nd=("death", "sum"),
        na=("death", "size"),
    )
    out: dict[str, dict[str, float]] = {}
    for (band, sex), r in agg.iterrows():
        out[f"death.{band}|{sex}"] = {
            "rate": _rate(r["wd"], r["we"]),
            "n_events": int(round(r["nd"])),
            "n_at_risk": int(r["na"]),
        }
    return out


def marital_cells(
    ev: pd.DataFrame, py: pd.DataFrame
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for trans, states in MARITAL_AT_RISK.items():
        te = ev[ev.transition == trans]
        num = (
            te.assign(w=te.weight)
            .groupby(["band", "sex"], observed=True)
            .agg(numw=("w", "sum"), ne=("w", "size"))
        )
        atr = py[py.marital_state.isin(states)]
        den = (
            atr.assign(w=atr.weight)
            .groupby(["band", "sex"], observed=True)
            .agg(denw=("w", "sum"), na=("w", "size"))
        )
        joined = num.join(den, how="outer")
        for (band, sex), r in joined.iterrows():
            numw = 0.0 if pd.isna(r["numw"]) else float(r["numw"])
            denw = 0.0 if pd.isna(r["denw"]) else float(r["denw"])
            n_ev = 0 if pd.isna(r["ne"]) else int(r["ne"])
            n_at = 0 if pd.isna(r["na"]) else int(r["na"])
            out[f"{trans}.{band}|{sex}"] = {
                "rate": _rate(numw, denw),
                "n_events": n_ev,
                "n_at_risk": n_at,
            }
    return out


def disability_cells(pairs: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    inc = pairs[~pairs.from_disabled]
    g = (
        inc.assign(w=inc.weight, wto=inc.weight * inc.to_disabled)
        .groupby(["band", "sex"], observed=True)
        .agg(
            denw=("w", "sum"),
            numw=("wto", "sum"),
            na=("w", "size"),
            ne=("to_disabled", "sum"),
        )
    )
    for (band, sex), r in g.iterrows():
        out[f"incidence.{band}|{sex}"] = {
            "rate": _rate(r["numw"], r["denw"]),
            "n_events": int(round(r["ne"])),
            "n_at_risk": int(r["na"]),
        }
    rec = pairs[pairs.from_disabled]
    recovered = ~rec.to_disabled.to_numpy()
    g2 = (
        rec.assign(w=rec.weight, wr=rec.weight * recovered)
        .groupby(["band", "sex"], observed=True)
        .agg(denw=("w", "sum"), numw=("wr", "sum"), na=("w", "size"))
    )
    ev2 = (
        rec.assign(rec=recovered)
        .groupby(["band", "sex"], observed=True)["rec"]
        .sum()
    )
    for (band, sex), r in g2.iterrows():
        out[f"recovery.{band}|{sex}"] = {
            "rate": _rate(r["numw"], r["denw"]),
            "n_events": int(round(ev2.get((band, sex), 0))),
            "n_at_risk": int(r["na"]),
        }
    return out


def _wquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")
    cutoff = q * cw[-1]
    return float(v[np.searchsorted(cw, cutoff)])


def earnings_cells(
    e: pd.DataFrame,
    *,
    level_years: tuple[int, ...] = GATED_EARN_YEARS,
    change_years: tuple[int, ...] = (EARN_ANCHOR_YEAR,) + GATED_EARN_YEARS,
) -> dict[str, dict[str, Any]]:
    """All earnings moment cells (gated + report-only) for one half.

    ``level_years`` = the reference years for the level moments (quantiles,
    zero rate); ``change_years`` = the anchor + endpoint years for the change
    / mobility / autocorrelation moments. The shock diagnostic reuses this
    with the shock windows.
    """
    out: dict[str, dict[str, Any]] = {}
    gated_lvls = e[e.period.isin(level_years)]
    change = e[e.period.isin(change_years)]
    # age_profile log-quantiles + zero rate, per cohort (level quantiles;
    # |ln(q_a/q_b)| == |Δ log-quantile|).
    for name, _lo, _hi in EARN_COHORTS:
        sub = gated_lvls[gated_lvls.cohort == name]
        pos = sub[sub.earnings > 0]
        v = pos.earnings.to_numpy(float)
        w = pos.weight.to_numpy(float)
        for q, tag in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
            out[f"earn_{tag}.{name}"] = {
                "value": _wquantile(v, w, q) if len(v) else 0.0,
                "n_obs": int(len(v)),
                "metric": "log_ratio",
            }
        allw = sub.weight.to_numpy(float)
        zerow = sub.loc[sub.earnings == 0, "weight"].to_numpy(float).sum()
        out[f"earn_zero_rate.{name}"] = {
            "value": _rate(zerow, allw.sum()),
            "n_obs": int(len(sub)),
            "metric": "log_ratio",
        }
        # change_moments of Delta log earnings over change_years.
        base = change[change.cohort == name]
        cm = moments.change_moments(
            base,
            id_col="person_id",
            period_col="period",
            value_col="earnings",
            weight_col="weight",
            horizon=1,
            period_step=2,
            log=True,
        ).set_index("moment")["value"]
        out[f"earn_dlog_sd.{name}"] = {
            "value": float(cm.get("sd", 0.0)),
            "n_obs": int(len(base)),
            "metric": "log_ratio",
        }
        # change-mean is a mean of log ratios -> gated on abs_gap in log units
        # (natural ln(1.5) cap). skew / kurtosis are dimensionless shape
        # moments with no natural power cap at this N -> report-only.
        out[f"earn_dlog_mean.{name}"] = {
            "value": float(cm.get("mean", 0.0)),
            "n_obs": int(len(base)),
            "metric": "abs_gap_log",
        }
        for mo, tag in (("skew", "skew"), ("kurtosis", "kurt")):
            out[f"earn_dlog_{tag}.{name}"] = {
                "value": float(cm.get(mo, 0.0)),
                "n_obs": int(len(base)),
                "metric": "report_only_shape",
            }
    # mobility diagonal retention, horizons 1 & 2 (pooled), positive bins.
    for h in MOBILITY_HORIZONS:
        mm = moments.mobility_matrix(
            change,
            id_col="person_id",
            period_col="period",
            value_col="earnings",
            weight_col="weight",
            horizon=h,
            period_step=2,
            n_bins=5,
            zero_bin=True,
        )
        diag = mm[(mm.origin == mm.destination) & (mm.origin > 0)]
        p = float(diag.probability.mean()) if len(diag) else 0.0
        out[f"earn_mob_h{h}_diag"] = {
            "value": p,
            "n_obs": int(len(mm)),
            "metric": "log_ratio",
        }
    # autocorrelation of log earnings, lags 1,2 gated on abs_gap (the gate_1
    # autocorr_log absolute-tolerance precedent); lag 5 = report-only by
    # measurability (a 10-year statistic exceeds the holdout span, so it is
    # undefined inside change_years and returns nan -> report_only_horizon).
    ac = moments.autocorrelation(
        change,
        id_col="person_id",
        period_col="period",
        value_col="earnings",
        weight_col="weight",
        lags=AUTOCORR_LAGS,
        period_step=2,
        log=True,
    ).set_index("lag")["value"]
    for lag in AUTOCORR_LAGS:
        raw = ac.get(lag, float("nan"))
        val = float(raw) if pd.notna(raw) else float("nan")
        metric = "abs_gap_corr" if lag < 5 else "report_only_horizon"
        out[f"earn_autocorr_lag{lag}"] = {
            "value": val,
            "n_obs": int(change.person_id.nunique()),
            "metric": metric,
        }
    return out


# --------------------------------------------------------------------------
# Floor: 100-seed real-vs-real half split of a cell table.
# --------------------------------------------------------------------------
def _score(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    """|ln(rate_a/rate_b)| for rate/log_ratio cells; |a-b| otherwise. None if
    a side is undefined (zero rate / nan) -- the undefined-draw guard."""
    va = a.get(key)
    vb = b.get(key)
    if va is None or vb is None:
        return None
    x = va.get("rate", va.get("value"))
    y = vb.get("rate", vb.get("value"))
    metric = va.get("metric", "log_ratio")
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if metric == "log_ratio" or "rate" in va:
        if x <= 0 or y <= 0:
            return None
        return abs(math.log(x / y))
    if isinstance(y, float) and math.isnan(y):
        return None
    return abs(x - y)


def run_floor(
    anchor: pd.DataFrame,
    compute,
    split_col: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Score |ln(a/b)| (or |a-b|) between two `split_col`-disjoint halves over
    100 seeds; return per-cell (mean, sd, realized_sigma, min_events) + the
    gate-seed detail. `compute(person_ids)` returns the cell dict for a half.
    """
    persons = anchor[["person_id", "household_id"]].drop_duplicates(
        "person_id"
    )
    per_seed: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = {}
    min_events: dict[str, int] = {}
    for seed in FLOOR_SEEDS:
        left, right = hpanel.split_panel_by_person(
            persons, split_col, fraction=SPLIT_FRACTION, seed=seed
        )
        ids_a = set(left.person_id.to_numpy().tolist())
        ids_b = set(right.person_id.to_numpy().tolist())
        cells_a = compute(ids_a)
        cells_b = compute(ids_b)
        keys = set(cells_a) | set(cells_b)
        seed_cells: dict[str, Any] = {}
        for key in keys:
            s = _score(cells_a, cells_b, key)
            scores.setdefault(key, [])
            if s is not None:
                scores[key].append(s)
            ea = cells_a.get(key, {})
            eb = cells_b.get(key, {})
            na = ea.get("n_events", ea.get("n_obs", 0))
            nb = eb.get("n_events", eb.get("n_obs", 0))
            weaker = min(na, nb)
            if key not in min_events or weaker < min_events[key]:
                min_events[key] = weaker
            seed_cells[key] = {
                "score": None if s is None else round(s, 6),
                "n_events_a": int(na),
                "n_events_b": int(nb),
            }
        if seed in GATE_SEEDS:
            per_seed.append({"seed": seed, "cells": seed_cells})
    floor: dict[str, dict[str, float]] = {}
    for key, vals in scores.items():
        arr = np.array(vals, dtype=float)
        n_def = len(arr)
        mean = float(arr.mean()) if n_def else 0.0
        sd = float(arr.std(ddof=1)) if n_def > 1 else 0.0
        # realized_sigma is the underlying-normal sigma of the folded |ln(a/b)|
        # score: for a half-normal, E[|X|]^2 + Var[|X|] = sigma^2, so the RMS
        # of the scores IS sigma (the gate_m4 / mortality-floor convention the
        # faithful-candidate OC prices against). NOT the folded sd.
        floor[key] = {
            "mean": mean,
            "sd": sd,
            "realized_sigma": math.sqrt(mean**2 + sd**2),
            "n_defined_seeds": n_def,
            "min_events_weaker_half": int(min_events.get(key, 0)),
        }
    return floor, per_seed


# --------------------------------------------------------------------------
# Tolerance, partition, OC.
# --------------------------------------------------------------------------
def _tol(mean: float, sd: float, k: int) -> float:
    return float(
        derive_tolerance(Decimal(str(mean)), Decimal(str(sd)), k, ROUNDING)
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def oc_4of5(
    floor: dict[str, dict[str, float]],
    tolerances: dict[str, float],
    gated: list[str],
) -> dict[str, Any]:
    """Faithful-candidate OC on the draw-noise-free half-normal basis (sec.
    4.9; the gate_2a / gate_m4 machinery, implemented independently)."""
    per_cell: dict[str, Any] = {}
    p_seed = 1.0
    for key in sorted(gated):
        sigma = floor[key]["realized_sigma"]
        tol = tolerances[key]
        p = (2.0 * _normal_cdf(tol / sigma) - 1.0) if sigma > 0 else 1.0
        per_cell[key] = {
            "tolerance": tol,
            "realized_sigma": sigma,
            "cell_pass_prob": round(p, 6),
        }
        p_seed *= p
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return {
        "method": (
            "Independence-approx normal OC on the DRAW-NOISE-FREE half-normal "
            "basis: a faithful candidate's per-cell score ~ "
            "half-normal(realized_sigma); cell pass = 2*Phi(tolerance/sigma)-1; "
            "seed pass = product over the gated family-A cells; gate = "
            "P(>=4 of 5) = p^5 + 5 p^4 (1-p). The K=20-draw mean estimator "
            "shares this basis (sec. 4.7)."
        ),
        "n_gated_cells": len(gated),
        "p_seed_pass": round(p_seed, 4),
        "p_gate_pass_4_of_5": round(p_gate, 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# Metric -> (power cap, k). Every gated cell shares round(mean + k*sd, 3);
# the cap differs by metric type (the log-ratio surface caps at ln(1.5), the
# log-unit change-mean at the same, the correlation cells at the gate-1
# autocorr-tolerance-scaled cap). Report-only metrics do not gate.
# --------------------------------------------------------------------------
#: gate-1 scores autocorr_log on an ABSOLUTE tolerance (0.05-0.07, comment
#: block gate_1.thresholds); a floor-derived corr tolerance must clear a cap
#: at ~2x the gate-1 10yr tolerance to discriminate the persistence regime.
AUTOCORR_ABS_CAP = 0.15

METRIC_CAP = {
    "log_ratio": T_MAX,
    "abs_gap_log": T_MAX,
    "abs_gap_corr": AUTOCORR_ABS_CAP,
}
METRIC_CAP_SOURCE = {
    "log_ratio": "ln(1.5)",
    "abs_gap_log": "ln(1.5) (change-mean is in log units)",
    "abs_gap_corr": "0.15 = ~2x gate_1 autocorr_log_10yr_tolerance 0.07",
}
GATEABLE_METRICS = frozenset(METRIC_CAP)


def cell_family(key: str) -> str:
    head = key.split(".")[0]
    if head == "death":
        return "mortality"
    if head in ("first_marriage", "divorce", "widowhood", "remarriage"):
        return "marital"
    if head in ("incidence", "recovery"):
        return "disability"
    if head == "prevalence":
        return "disability_stock"
    if head == "married_share":
        return "marital_stock"
    return "earnings"


#: Design split unit per family (sec. 4.5 / 4.7): demographic FLOWS with
#: within-household correlation (mortality, marital -- widowhood/divorce couple
#: a household; mortality shares household frailty) split HOUSEHOLD-disjoint
#: (couple-disjoint subsumed); disability + earnings split PERSON-disjoint.
HOUSEHOLD_SPLIT_FAMILIES = frozenset({"mortality", "marital", "marital_stock"})


def split_col_for_family(fam: str) -> str:
    return "household_id" if fam in HOUSEHOLD_SPLIT_FAMILIES else "person_id"


def partition(
    floor: dict[str, dict[str, float]],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Gated vs report-only, with a machine reason per report-only cell."""
    tolerances: dict[str, float] = {}
    gated: list[str] = []
    report_only: dict[str, str] = {}
    for key, fl in floor.items():
        m = meta[key]
        metric = m["metric"]
        fam = m["family"]
        k = K_STOCK if m.get("quantity_type") == "stock" else K_FLOW
        raw_tol = _tol(fl["mean"], fl["sd"], k)
        tolerances[key] = raw_tol
        # report-only dispositions (in disposition precedence order):
        if fam.endswith("_stock"):
            report_only[key] = "stock_report_only_margin"
            continue
        if metric == "report_only_horizon":
            report_only[key] = "horizon_exceeds_holdout_span"
            continue
        if metric == "report_only_shape":
            report_only[key] = "dimensionless_shape_no_power_cap"
            continue
        if key.split(".")[0] == "death" and key.split("|")[0].split(".")[
            1
        ] in (MORTALITY_ATTRITION_BANDS):
            report_only[key] = "attrition_confounded_truth"
            continue
        if metric not in GATEABLE_METRICS:
            report_only[key] = "non_gateable_metric"
            continue
        if fl["n_defined_seeds"] < len(FLOOR_SEEDS):
            report_only[key] = "undefined_on_some_seed"
            continue
        if fl["min_events_weaker_half"] < MIN_EVENTS_FOR_GATE:
            report_only[key] = "below_20_events_weaker_half"
            continue
        if raw_tol > METRIC_CAP[metric] + 1e-12:
            report_only[key] = "tolerance_above_power_cap"
            continue
        gated.append(key)
    return {
        "gated": sorted(gated),
        "report_only": dict(sorted(report_only.items())),
        "tolerances": tolerances,
        "n_gated": len(gated),
        "n_report_only": len(report_only),
    }


def split_width_diagnostic(
    household_floor: dict[str, dict[str, float]],
    person_floor: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Person-disjoint vs household-disjoint floor WIDTHS on the household-split
    cells (the ceremony watch-item, PR #170 comment 4953958949): a household-ID
    rule that biased the split would inflate/deflate the household floor sigma
    relative to person. Ratio ~1 means the T*-anchored household rule does not
    bias the family-A floor."""
    per_cell: dict[str, Any] = {}
    ratios: list[float] = []
    for key in sorted(set(household_floor) & set(person_floor)):
        hh = household_floor[key]["realized_sigma"]
        pp = person_floor[key]["realized_sigma"]
        ratio = float(hh / pp) if pp > 0 else None
        per_cell[key] = {
            "household_sigma": round(hh, 6),
            "person_sigma": round(pp, 6),
            "household_over_person": (
                None if ratio is None else round(ratio, 4)
            ),
        }
        if ratio is not None:
            ratios.append(ratio)
    arr = np.array(ratios) if ratios else np.array([1.0])
    return {
        "note": (
            "Per household-split family-A cell, realized_sigma under the "
            "HOUSEHOLD-disjoint split (the design unit) over the PERSON-disjoint "
            "split. Ratio > 1 is the expected mild inflation from positive "
            "within-household correlation (fewer effective independent units); "
            "a ratio far from 1 would flag a biasing household-ID rule."
        ),
        "n_cells": len(ratios),
        "mean_household_over_person": round(float(arr.mean()), 4),
        "median_household_over_person": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# End-of-window STOCK cells (report-only margin, sec. 4.6 gate_m4 >=3 sigma
# style). Occupancy levels at the last gated wave (2019); their half-split
# floor sigma is published as the margin basis, never |ln|-level-gated.
# --------------------------------------------------------------------------
def disability_occupancy(
    status: pd.DataFrame,
    death_records: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    """Disabled occupancy person-rows at the 2019 gated wave, F6-weighted."""
    with_sex = disability.attach_sex(status, death_records)
    occ = with_sex[with_sex.period == 2019].copy()
    occ = occ.merge(
        anchor[["person_id", "weight"]],
        on="person_id",
        how="inner",
        suffixes=("_raw", ""),
    )
    occ["band"] = occ.age.map(lambda a: band_of(a, DISABILITY_BANDS))
    occ = occ[occ.band.notna() & occ.sex.isin(SEXES)]
    return occ.reset_index(drop=True)


def occupancy_cells(occ: pd.DataFrame) -> dict[str, dict[str, Any]]:
    g = (
        occ.assign(w=occ.weight, wd=occ.weight * occ.disabled.astype(float))
        .groupby(["band", "sex"], observed=True)
        .agg(w=("w", "sum"), wd=("wd", "sum"), n=("w", "size"))
    )
    out: dict[str, dict[str, Any]] = {}
    for (band, sex), r in g.iterrows():
        out[f"prevalence.{band}|{sex}"] = {
            "rate": _rate(r.wd, r.w),
            "n_events": int(r.n),
            "n_at_risk": int(r.n),
        }
    return out


def marital_share_cells(py: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Married-share stock at the 2019 gated wave, by band x sex."""
    at19 = py[(py.year == 2019) & (py.window == "gated")]
    g = (
        at19.assign(
            w=at19.weight, wm=at19.weight * (at19.marital_state == "married")
        )
        .groupby(["band", "sex"], observed=True)
        .agg(w=("w", "sum"), wm=("wm", "sum"), n=("w", "size"))
    )
    out: dict[str, dict[str, Any]] = {}
    for (band, sex), r in g.iterrows():
        out[f"married_share.{band}|{sex}"] = {
            "rate": _rate(r.wm, r.w),
            "n_events": int(r.n),
            "n_at_risk": int(r.n),
        }
    return out


def _meta_from_reference(
    ref: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-cell metric / family / quantity_type from a full-sample cell dict."""
    meta: dict[str, dict[str, Any]] = {}
    for key, val in ref.items():
        fam = cell_family(key)
        if "rate" in val:
            metric = "log_ratio"
        else:
            metric = val.get("metric", "log_ratio")
        quantity = "stock" if fam.endswith("_stock") else "flow"
        meta[key] = {
            "family": fam,
            "metric": metric,
            "quantity_type": quantity,
        }
    return meta


def build_artifact(verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    demo = panels.demographic_panel()
    death_records = deaths.read_death_records()
    status = disability.read_disability_status()
    ep = family.family_earnings_panel()

    anchor = build_anchor_frame(demo)
    present = presence_by_wave(demo)
    n_fallback = 0  # every holdout person has a gated-wave interview by build

    # Family cell tables (presence-conditioned, windowed, F6-weighted).
    sl = mortality_slices(demo, death_records, anchor)
    ev, py = marital_tables(death_records, anchor, present)
    dpairs = disability_pairs(status, death_records, anchor)
    earn = earnings_frame(ep, anchor)
    occ = disability_occupancy(status, death_records, anchor)

    sl_g, ev_g, py_g = (
        sl[sl.window == "gated"],
        ev[ev.window == "gated"],
        py[py.window == "gated"],
    )
    dp_g = dpairs[dpairs.window == "gated"]

    def c_mort(ids: set[int]) -> dict[str, Any]:
        return mortality_cells(sl_g[sl_g.person_id.isin(ids)])

    def c_marital(ids: set[int]) -> dict[str, Any]:
        return marital_cells(
            ev_g[ev_g.person_id.isin(ids)], py_g[py_g.person_id.isin(ids)]
        )

    def c_disab(ids: set[int]) -> dict[str, Any]:
        return disability_cells(dp_g[dp_g.person_id.isin(ids)])

    def c_earn(ids: set[int]) -> dict[str, Any]:
        return earnings_cells(earn[earn.person_id.isin(ids)])

    def c_prev(ids: set[int]) -> dict[str, Any]:
        return occupancy_cells(occ[occ.person_id.isin(ids)])

    def c_mshare(ids: set[int]) -> dict[str, Any]:
        return marital_share_cells(py[py.person_id.isin(ids)])

    all_ids = set(anchor.person_id.to_numpy().tolist())
    reference = {
        **c_mort(all_ids),
        **c_marital(all_ids),
        **c_disab(all_ids),
        **c_earn(all_ids),
        **c_prev(all_ids),
        **c_mshare(all_ids),
    }
    meta = _meta_from_reference(reference)

    # Primary floors on the design split unit per family.
    if verbose:
        print("floor: mortality (household + person diag)...")
    mort_hh, mort_hh_seed = run_floor(anchor, c_mort, "household_id")
    mort_pp, _ = run_floor(anchor, c_mort, "person_id")
    if verbose:
        print("floor: marital (household + person diag)...")
    mar_hh, mar_hh_seed = run_floor(anchor, c_marital, "household_id")
    mar_pp, _ = run_floor(anchor, c_marital, "person_id")
    if verbose:
        print("floor: disability (person)...")
    dis_floor, dis_seed = run_floor(anchor, c_disab, "person_id")
    if verbose:
        print("floor: earnings (person)...")
    earn_floor, earn_seed = run_floor(anchor, c_earn, "person_id")
    if verbose:
        print("floor: stocks (prevalence person / married_share household)...")
    prev_floor, _ = run_floor(anchor, c_prev, "person_id")
    mshare_floor, _ = run_floor(anchor, c_mshare, "household_id")

    # Shock-window diagnostics (sec. 4.1): reference + half-split floor of the
    # 2020-2022 exogenous-shock cells, COMPUTED and published but PARTITIONED
    # OUT of the gated set (machine reason exogenous_shock_outside_model_class).
    sl_s, ev_s, py_s = (
        sl[sl.window == "shock"],
        ev[ev.window == "shock"],
        py[py.window == "shock"],
    )
    dp_s = dpairs[dpairs.window == "shock"]

    def cs_mort(ids: set[int]) -> dict[str, Any]:
        return mortality_cells(sl_s[sl_s.person_id.isin(ids)])

    def cs_marital(ids: set[int]) -> dict[str, Any]:
        return marital_cells(
            ev_s[ev_s.person_id.isin(ids)], py_s[py_s.person_id.isin(ids)]
        )

    def cs_disab(ids: set[int]) -> dict[str, Any]:
        return disability_cells(dp_s[dp_s.person_id.isin(ids)])

    def cs_earn(ids: set[int]) -> dict[str, Any]:
        return earnings_cells(
            earn[earn.person_id.isin(ids)],
            level_years=SHOCK_EARN_YEARS,
            change_years=(2018,) + SHOCK_EARN_YEARS,
        )

    if verbose:
        print("floor: shock-window diagnostics (report-only)...")
    shock_ref = {
        **cs_mort(all_ids),
        **cs_marital(all_ids),
        **cs_disab(all_ids),
        **cs_earn(all_ids),
    }
    shock_floor: dict[str, dict[str, float]] = {}
    for compute, col in (
        (cs_mort, "household_id"),
        (cs_marital, "household_id"),
        (cs_disab, "person_id"),
        (cs_earn, "person_id"),
    ):
        fl, _ = run_floor(anchor, compute, col)
        shock_floor.update(fl)

    # Authoritative floor = the design-unit split per family.
    floor: dict[str, dict[str, float]] = {}
    floor.update(mort_hh)
    floor.update(mar_hh)
    floor.update(dis_floor)
    floor.update(earn_floor)
    floor.update(prev_floor)
    floor.update(mshare_floor)

    part = partition(floor, meta)
    tolerances = part["tolerances"]

    # OC on the draw-noise-free half-normal basis (sec. 4.9). Reported over
    # THREE surfaces: the certifiable FLOW family-A primary (mortality +
    # marital + disability -- the "dynamics-drift flows" a PASS certifies,
    # sec. 4.6), the gated EARNINGS moments (a distinct deployment-scale
    # sub-family), and the COMBINED gated surface (the precedent's one-OC
    # convention).
    flow_gated = [
        k
        for k in part["gated"]
        if meta[k]["family"] in ("mortality", "marital", "disability")
    ]
    earn_gated = [k for k in part["gated"] if meta[k]["family"] == "earnings"]
    oc_flows = oc_4of5(floor, tolerances, flow_gated)
    oc_earnings = oc_4of5(floor, tolerances, earn_gated)
    oc = oc_4of5(floor, tolerances, part["gated"])

    # The split-width diagnostic on the household-split (mortality + marital)
    # family-A cells.
    hh_floor = {**mort_hh, **mar_hh}
    pp_floor = {**mort_pp, **mar_pp}
    diag = split_width_diagnostic(hh_floor, pp_floor)

    # Weak-power OC-before-lock check (both directions, sec. 4.9 / sec. 9).
    n_gated = part["n_gated"]
    n_flow_gated = len(flow_gated)
    at_cap = [
        k
        for k in part["gated"]
        if abs(tolerances[k] - METRIC_CAP[meta[k]["metric"]]) <= 1e-9
    ]
    # near-unpassable: the certifiable surface's faithful OC drops below the
    # named floor. near-vacuous: fewer than MIN_GATED_CELLS gated FLOW cells
    # (the primary certifiable surface -- earnings padding cannot certify the
    # dynamics-drift flows), OR every gated tolerance pinned at its cap.
    clears_lower = oc["p_gate_pass_4_of_5"] >= WEAK_POWER_P_GATE_FLOOR
    clears_flow_power = n_flow_gated >= MIN_GATED_CELLS_FOR_POWER
    clears_not_all_capped = len(at_cap) < n_gated
    clears = bool(clears_lower and clears_flow_power and clears_not_all_capped)

    if verbose:
        print(
            f"gated={n_gated} (flow={n_flow_gated} earn={len(earn_gated)}) "
            f"report_only={part['n_report_only']} "
            f"p_gate_combined={oc['p_gate_pass_4_of_5']} "
            f"p_gate_flows={oc_flows['p_gate_pass_4_of_5']} "
            f"clears_weak_power={clears}"
        )

    per_seed = {
        "mortality": mort_hh_seed,
        "marital": mar_hh_seed,
        "disability": dis_seed,
        "earnings": earn_seed,
    }

    families = {}
    for key in floor:
        families.setdefault(meta[key]["family"], []).append(key)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "reported_truth_side_only": True,
        "no_projection_no_candidate": True,
        "referee_round": (
            "gate_m6 floors ceremony step 1; design PR #170 (1d83a22) "
            "VERIFIED comment 4953958949; adjudication issue #42 4953722912"
        ),
        "component": (
            "M6 temporal-holdout gate (gate_m6): the PSID held-out truth "
            "(2015-2022) + real-vs-real half-split floor for the proposed "
            "family-A dynamics-drift flows and the report-only margins."
        ),
        "purpose": (
            "Price gate_m6's tolerances against the deployment-scale sampling "
            "null of the held-out observables, on the post-demotion surface "
            "(shock sec. 4.1, attrition sec. 4.4, mixed-k / power sec. 4.6). "
            "Builds NO engine and scores NO candidate; gates.yaml is untouched."
        ),
        "gates_yaml_untouched": True,
        "design_pins": {
            "boundary_T_star": T_STAR,
            "seed_wave": SEED_WAVE,
            "seed_wave_note": (
                "PSID interviews are biennial ODD years; T*=2014 realizes its "
                "seed state at the 2015 interview (reference year 2014)."
            ),
            "gated_flow_event_years": list(GATED_FLOW_YEARS),
            "shock_flow_event_years": list(SHOCK_FLOW_YEARS),
            "gated_earnings_reference_years": list(GATED_EARN_YEARS),
            "shock_earnings_reference_years": list(SHOCK_EARN_YEARS),
            "reference_year_2021_unobserved": True,
            "mixed_k": {"flow": K_FLOW, "stock": K_STOCK, "margin": MARGIN_K},
            "rounding": ROUNDING,
            "t_max": T_MAX,
            "t_max_source": T_MAX_SOURCE,
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
            "floor_seeds": [FLOOR_SEEDS[0], FLOOR_SEEDS[-1]],
            "gate_seeds": list(GATE_SEEDS),
            "conjunction": "4 of 5 seeds",
            "k20_estimator_note": (
                "the candidate side (not built here) is the K=20-draw mean on "
                "numpy.random.default_rng(5200 + k), scored once |ln(rbar/"
                "rate_a)| against side-A truth (sec. 4.7)."
            ),
        },
        "presence_conditioning": {
            "rule": (
                "Every truth cell is computed on the realized-wave-presence "
                "set (PSID sequence 1-20, positive weight), SYMMETRICALLY on "
                "both halves and (for the candidate, not built here) both "
                "sides (sec. 4.4). demographic_panel / read_disability_status "
                "/ family_earnings_panel are present-only by construction; "
                "marital events + at-risk person-years are conditioned on "
                "presence at the biennial interview OPENING the interval that "
                "contains the event year."
            ),
            "per_family": {
                "mortality": "present at the interval-start interview (slices "
                "credited only from an observed wave); conditioning-not-"
                "leakage (PSID retention is unmodelled).",
                "marital": "present at the opening biennial interview of the "
                "event's interval; conditioning-not-leakage.",
                "disability": "reproduction mode -- realized support preserved "
                "(read_disability_status present-only); a PASS certifies a "
                "TEMPORAL gate_m4 extension, NOT M6 forward disability "
                "(sec. 4.4, finding 7).",
                "earnings": "present with a valid positive-weight earnings "
                "observation at the reference-year wave.",
            },
        },
        "f6_weight_rule": {
            "definition": (
                "Every gated rate, every truth cell, and the floor use the "
                "person's realized start-of-holdout START-WAVE cross-sectional "
                "PSID weight, held FIXED across the window (F6, sec. 4.7). The "
                "per-year calibrated panel weight is NOT used (it would put a "
                "different weight on the two sides and bias every ratio); it "
                "is a report-only alternative."
            ),
            "start_wave": (
                "the earliest gated interview wave (2015 / 2017 / 2019, >= the "
                "2015 T* seed) at which the person is present."
            ),
        },
        "household_id_weight_carriage_rule": {
            "finding": "13 (sec. 9); the ceremony watch-item of comment 4953958949",
            "household_id": (
                "the PSID family interview number (ER30001-scoped, "
                "panels.demographic_panel `interview`) at the person's realized "
                "start-of-holdout wave, held FIXED across the window -- "
                "consistent with the F6 start-wave anchoring."
            ),
            "divorce_splitoff_rule": (
                "a within-window divorce / splitoff does NOT reassign "
                "household_id; both ex-partners keep their fixed start-wave "
                "household (conservative: keeps a correlated ex-couple on ONE "
                "side of the split, so the split cannot understate correlation)."
            ),
            "entrant_child_rule": (
                "synthetic births + immigrant cohorts are the family-B closed-"
                "panel entrants (sec. 4.8) and are OUT of the family-A floor; "
                "children present at the start wave carry that wave's household."
            ),
            "fallback": (
                "a person with no gated-wave interview number falls back to a "
                "singleton household (their own person_id)."
            ),
            "n_singleton_fallback": n_fallback,
            "split_bias_diagnostic": (
                "runs comparison: person-disjoint vs household-disjoint floor "
                "widths on the household-split family-A cells (see "
                "split_width_diagnostic); a rule that biased the split would "
                "move the ratio far from 1."
            ),
        },
        "split_units": {
            "note": (
                "correlation-respecting half split (sec. 4.7): the demographic "
                "flows with within-household correlation split HOUSEHOLD-"
                "disjoint (couple-disjoint subsumed); disability + earnings "
                "split PERSON-disjoint (sec. 4.5)."
            ),
            "household_disjoint_families": sorted(HOUSEHOLD_SPLIT_FAMILIES),
            "person_disjoint_families": [
                "disability",
                "earnings",
                "disability_stock",
            ],
            "machine": "populace_dynamics.harness.panel.split_panel_by_person",
            "split_fraction": SPLIT_FRACTION,
        },
        "data": {
            "n_holdout_persons": int(anchor.person_id.nunique()),
            "n_households": int(anchor.household_id.nunique()),
            "psid_population": (
                "populace_dynamics.data.panels.demographic_panel + deaths "
                "(ER32050) + read_disability_status + family_earnings_panel + "
                "marriage.marriage_history"
            ),
            "psid_wave_calendar": "biennial odd-year interviews 1999-2023; "
            "even-year income reference years",
        },
        "reference_moments": reference,
        "floor": {
            "method": (
                "100-seed real-vs-real half split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction 0.5, seeds 0-99) of the presence-conditioned, F6-"
                "weighted holdout truth; floor statistic |ln(rate_a/rate_b)| "
                "for rate / log_ratio cells, |value_a - value_b| for the "
                "typed abs_gap cells. (mean, sd, realized_sigma) per cell "
                "derive round(mean + k*sd, 3) capped at the metric's power cap."
            ),
            "cells": floor,
        },
        "metric_caps": {
            m: {"cap": METRIC_CAP[m], "source": METRIC_CAP_SOURCE[m]}
            for m in METRIC_CAP
        },
        "cell_meta": meta,
        "partition": {
            "gated": part["gated"],
            "report_only": part["report_only"],
            "n_gated": part["n_gated"],
            "n_report_only": part["n_report_only"],
            "by_family_gated": {
                fam: sorted(
                    k for k in part["gated"] if meta[k]["family"] == fam
                )
                for fam in sorted({meta[k]["family"] for k in part["gated"]})
            },
        },
        "tolerances": {k: tolerances[k] for k in part["gated"]},
        "tolerances_all_cells": tolerances,
        "faithful_candidate_oc": {
            "combined": oc,
            "family_a_flows": oc_flows,
            "earnings_subfamily": oc_earnings,
            "p_seed_pass": oc["p_seed_pass"],
            "p_gate_pass_4_of_5": oc["p_gate_pass_4_of_5"],
        },
        "oc_before_lock": {
            "weak_power_p_gate_floor": WEAK_POWER_P_GATE_FLOOR,
            "weak_power_floor_source": (
                "bottom of the lived precedent p_gate band (gate_2a 0.9685 / "
                "2b 0.9678 / 2c 0.988 / m4 0.9689 / w1 0.9481-0.9623), rounded "
                "down to 0.90."
            ),
            "min_gated_flow_cells": MIN_GATED_CELLS_FOR_POWER,
            "n_gated_cells": n_gated,
            "n_gated_flow_cells": n_flow_gated,
            "n_gated_earnings_cells": len(earn_gated),
            "n_gated_tolerances_at_cap": len(at_cap),
            "p_gate_combined": oc["p_gate_pass_4_of_5"],
            "p_gate_flows": oc_flows["p_gate_pass_4_of_5"],
            "p_gate_earnings": oc_earnings["p_gate_pass_4_of_5"],
            "clears_lower_bound_not_unpassable": bool(clears_lower),
            "clears_flow_surface_power": bool(clears_flow_power),
            "clears_not_all_tolerances_capped": bool(clears_not_all_capped),
            "clears_weak_power_threshold": clears,
            "ceremony_may_proceed": clears,
            "interpretation": (
                "Both directions (sec. 4.9, sec. 9). Near-unpassable: the "
                "combined faithful p_gate < the named floor. Near-vacuous: "
                "fewer than the minimum gated FLOW cells (the mortality / "
                "marital / disability primary that certifies the dynamics-"
                "drift claim per sec. 4.6 -- earnings-moment cells cannot "
                "certify the flow claim), OR every gated tolerance pinned at "
                "its power cap. If ANY fails the ceremony PAUSES for surface "
                "redesign rather than locking a near-vacuous gate."
            ),
            "pause_finding": (
                (
                    "The 8-year biennial holdout at age-band x sex granularity "
                    "yields ~0.1 half-split log-ratio noise per flow cell, so at "
                    "the design-pinned k=3 most flow tolerances land just above "
                    "the ln(1.5) cap and demote on power; the post-demotion FLOW "
                    "surface supports too few gated cells to certify the dynamics-"
                    "drift claim. The earnings moments gate cleanly (large N), but "
                    "cannot stand in for the flows. RECOMMENDATION for the surface "
                    "redesign the pause triggers: coarsen the sparse marital / "
                    "mortality strata (e.g. sex-pool the low-frequency transitions "
                    "or widen the oldest bands), and/or extend the flow event "
                    "window, then rebuild the floor -- not a threshold widening."
                )
                if not clears
                else "surface clears; ceremony may proceed to lock"
            ),
        },
        "split_width_diagnostic": diag,
        "shock_window_diagnostics": {
            "machine_reason": "exogenous_shock_outside_model_class",
            "note": (
                "The 2020-2022 COVID / inflation shock is outside the engine's "
                "model class (no macro / epidemic channel) and hit mortality "
                "(2020-21 excess deaths) and marital rates (2020 marriage "
                "collapse) at least as hard as earnings (sec. 4.1). These "
                "cells are a first-class PUBLISHED diagnostic -- how far a "
                "mechanical engine misses a pandemic, quantified against held-"
                "out truth -- but are PARTITIONED OUT of every gated / report-"
                "only-margin set above. Dual axis: earnings reference years "
                f"{list(SHOCK_EARN_YEARS)} (2021 unobserved), flow event years "
                f"{list(SHOCK_FLOW_YEARS)}."
            ),
            "reference": shock_ref,
            "floor": shock_floor,
            "n_cells": len(shock_ref),
        },
        "report_reasons": sorted(set(part["report_only"].values())),
        "families": {fam: sorted(keys) for fam, keys in families.items()},
        "per_seed": per_seed,
        "revision_pins": {
            "artifact_schema_version": SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = build_artifact(verbose=True)
    artifacts.write_new(ARTIFACT_PATH, artifact, sidecar=True)
    print(f"wrote {ARTIFACT_PATH}")
    print(f"sha256 {_sha256_bytes(ARTIFACT_PATH)}")


if __name__ == "__main__":
    main()
