"""Shared truth-side cell machinery for the M6 temporal-holdout gate.

This module is the single source imported by both the frozen floor builders and
the scored-run harness.  The reductions remain the ceremony implementations;
support restrictions (including the earnings domain) are applied by callers.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import disability, marriage, transitions
from populace_dynamics.evaluation import derive_tolerance
from populace_dynamics.harness import moments
from populace_dynamics.harness import panel as hpanel

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
    return marital_tables_from_panel(panel, anchor, present)


def marital_tables_from_panel(
    panel: transitions.MaritalPanel,
    anchor: pd.DataFrame,
    present: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the shared M6 truth support law to a certified marital panel.

    The reader-owning floor entry point above and the runner input assembler
    both call this transform, keeping production truth preparation on the same
    extracted byte-identity path.
    """

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
    pairs = disability_scoring_universe(pairs)
    # gated: both endpoints in holdout (start 2015 or 2017 -> end 2017 / 2019);
    # start 2019 -> end 2021 shock (report-only).
    window = np.where(
        pairs.start_wave.isin((2015, 2017)),
        "gated",
        np.where(pairs.start_wave == 2019, "shock", "out"),
    )
    pairs["window"] = window
    return pairs[pairs.window != "out"].reset_index(drop=True)


def disability_scoring_universe(pairs: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared coded-sex, hazard-band disability universe."""
    out = pairs.copy()
    out["band"] = out.age.map(lambda age: band_of(age, DISABILITY_BANDS))
    return out[out.band.notna() & out.sex.isin(SEXES)]


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
    *,
    retained_seeds: Iterable[int] = GATE_SEEDS,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Score |ln(a/b)| (or |a-b|) between two `split_col`-disjoint halves over
    100 seeds; return per-cell (mean, sd, realized_sigma, min_events) + the
    requested seed detail. ``compute(person_ids)`` returns the cell dict for a
    half. Detail remains limited to the five gate seeds by default; a floor
    ceremony may retain all 100 without changing the split or reduction math.
    """
    retained = frozenset(retained_seeds)
    unknown = retained - frozenset(FLOOR_SEEDS)
    if unknown:
        raise ValueError(
            f"retained floor seeds are outside 0..99: {sorted(unknown)}"
        )
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
        if seed in retained:
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


# --------------------------------------------------------------------------
# The v3 ceremony's adopted coarsened flow reductions.  The v2/v3 scripts and
# the scored-run harness import these functions from the same source.
# --------------------------------------------------------------------------
def _stratum(
    frame: pd.DataFrame,
    bands: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    sex_pool: bool,
) -> pd.DataFrame:
    f = frame.copy()
    f["band2"] = f["age"].map(lambda a: band_of(a, tuple(bands)))
    f = f[f["band2"].notna()]
    if sex_pool:
        f["stratum"] = f["band2"]
    else:
        f["stratum"] = f["band2"].astype(str) + "|" + f["sex"].astype(str)
    return f


def coarsened_mortality_cells(
    sl: pd.DataFrame,
    bands: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    sex_pool: bool,
) -> dict[str, dict[str, float]]:
    f = _stratum(sl, bands, sex_pool)
    g = (
        f.assign(we=f.weight * f.exposure, wd=f.weight * f.death)
        .groupby("stratum", observed=True)
        .agg(
            we=("we", "sum"),
            wd=("wd", "sum"),
            nd=("death", "sum"),
            na=("death", "size"),
        )
    )
    return {
        f"death.{s}": {
            "rate": _rate(r["wd"], r["we"]),
            "n_events": int(round(r["nd"])),
            "n_at_risk": int(r["na"]),
        }
        for s, r in g.iterrows()
    }


def coarsened_marital_cells(
    ev: pd.DataFrame,
    py: pd.DataFrame,
    trans: str,
    bands: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    sex_pool: bool,
) -> dict[str, dict[str, float]]:
    states = MARITAL_AT_RISK[trans]
    te = _stratum(ev[ev.transition == trans], bands, sex_pool)
    atr = _stratum(py[py.marital_state.isin(states)], bands, sex_pool)
    num = te.groupby("stratum", observed=True).agg(
        numw=("weight", "sum"), ne=("weight", "size")
    )
    den = atr.groupby("stratum", observed=True).agg(
        denw=("weight", "sum"), na=("weight", "size")
    )
    joined = num.join(den, how="outer")
    out: dict[str, dict[str, float]] = {}
    for s, r in joined.iterrows():
        numw = 0.0 if pd.isna(r["numw"]) else float(r["numw"])
        denw = 0.0 if pd.isna(r["denw"]) else float(r["denw"])
        n_ev = 0 if pd.isna(r["ne"]) else int(r["ne"])
        n_at = 0 if pd.isna(r["na"]) else int(r["na"])
        out[f"{trans}.{s}"] = {
            "rate": _rate(numw, denw),
            "n_events": n_ev,
            "n_at_risk": n_at,
        }
    return out


def coarsened_disability_cells(
    pairs: pd.DataFrame,
    kind: str,
    bands: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    sex_pool: bool,
) -> dict[str, dict[str, float]]:
    if kind == "incidence":
        sub = pairs[~pairs.from_disabled].copy()
        sub["_ev"] = sub.to_disabled.astype(float)
    else:
        sub = pairs[pairs.from_disabled].copy()
        sub["_ev"] = (~sub.to_disabled).astype(float)
    f = _stratum(sub, bands, sex_pool)
    g = (
        f.assign(w=f.weight, wev=f.weight * f._ev)
        .groupby("stratum", observed=True)
        .agg(
            denw=("w", "sum"),
            numw=("wev", "sum"),
            na=("w", "size"),
            ne=("_ev", "sum"),
        )
    )
    return {
        f"{kind}.{s}": {
            "rate": _rate(r["numw"], r["denw"]),
            "n_events": int(round(r["ne"])),
            "n_at_risk": int(r["na"]),
        }
        for s, r in g.iterrows()
    }
