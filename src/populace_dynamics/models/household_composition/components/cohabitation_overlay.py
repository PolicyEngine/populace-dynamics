"""Cohabitation occupancy and candidate-8 overlay lift.

Copied from ``household_composition_sim_v2.py:83-88,116-206``,
``household_composition_sim_v4.py:100-104,216-267``,
``household_composition_sim_v6.py:145-148,307-363``, and
``household_composition_sim_v8.py:115-118,971-1011``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import relmap
from populace_dynamics.models.household_composition.common import (
    weighted_rate,
    weighted_share,
)

PARTNER_CODE = relmap.PARTNER
COHAB_SINGLE_YEAR_LO = hc.START_AGE
COHAB_SINGLE_YEAR_HI = 34
COHAB_FEMALE_REFIT_LO = 25
COHAB_FEMALE_REFIT_HI = 44
COHAB_OVERLAY_LIFT_BAND = "25-34"
COHAB_OVERLAY_LIFT = 0.045
MIN_STRATUM_N = 20


def cohabitation_flag(rel_map: pd.DataFrame) -> pd.DataFrame:
    """Return observed code-22 partner state by person-wave.

    Copied from ``household_composition_sim_v2.py:116-137``.
    """
    nonself = rel_map[rel_map["ego_rel_to_alter"] != relmap.SELF]
    partner = nonself[nonself["ego_rel_to_alter"] == PARTNER_CODE]
    hits = (
        partner.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("_n")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    hits["cohabiting"] = hits["_n"] > 0
    return hits[["person_id", "year", "cohabiting"]]


def attach_cohabitation(
    person_waves: pd.DataFrame, cohab_flag: pd.DataFrame
) -> pd.DataFrame:
    """Attach current and next cohabitation states.

    Copied from ``household_composition_sim_v2.py:140-156``.
    """
    pw = person_waves.merge(cohab_flag, on=["person_id", "year"], how="left")
    pw["cohabiting"] = pw["cohabiting"].fillna(False).astype(bool)
    pw = pw.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw["next_cohabiting"] = pw.groupby("person_id", sort=False)[
        "cohabiting"
    ].shift(-1)
    return pw


def fit_cohabitation_rates(
    train_pw: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Fit band-by-sex cohabitation entry and exit.

    Copied from ``household_composition_sim_v2.py:159-206``.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry_pool = has_next[~has_next["cohabiting"]]
    exit_pool = has_next[has_next["cohabiting"]]
    entry_overall = weighted_rate(
        entry_pool, entry_pool["next_cohabiting"].to_numpy(dtype=np.float64)
    )
    exit_overall = weighted_rate(
        exit_pool,
        exit_pool["next_cohabiting"].eq(False).to_numpy(dtype=np.float64),
    )
    entry: dict[tuple[str, str], float] = {}
    exit_: dict[tuple[str, str], float] = {}
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            e = entry_pool[
                (entry_pool["band"] == band) & (entry_pool["sex"] == sex)
            ]
            x = exit_pool[
                (exit_pool["band"] == band) & (exit_pool["sex"] == sex)
            ]
            entry[(band, sex)] = (
                weighted_rate(e, e["next_cohabiting"].to_numpy(np.float64))
                if len(e)
                else entry_overall
            )
            exit_[(band, sex)] = (
                weighted_rate(
                    x, x["next_cohabiting"].eq(False).to_numpy(np.float64)
                )
                if len(x)
                else exit_overall
            )
    return entry, exit_


def band_bounds(age: int) -> tuple[int, int]:
    """Return the composition band containing ``age``.

    Copied from ``household_composition_sim_v4.py:263-267``.
    """
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        if lo <= age <= hi:
            return lo, hi
    return hc.COMPOSITION_AGE_BANDS[-1]


def fit_cohab_single_year(
    person_waves: pd.DataFrame,
    cohab_flag: pd.DataFrame,
    train_ids: set[int],
    band_entry: dict[tuple[str, str], float],
    band_exit: dict[tuple[str, str], float],
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float]]:
    """Fit the ages 15--34 single-year refinement.

    Copied from ``household_composition_sim_v4.py:216-260``.
    """
    pw = attach_cohabitation(person_waves, cohab_flag)
    pw = pw[pw["person_id"].isin(train_ids)]
    hasn = pw[pw["has_next"] & pw["band"].notna()]
    entry: dict[tuple[int, str], float] = {}
    exit_: dict[tuple[int, str], float] = {}
    for age in range(COHAB_SINGLE_YEAR_LO, COHAB_SINGLE_YEAR_HI + 1):
        for sex in hc.SEXES:
            sub = hasn[(hasn["age"] == age) & (hasn["sex"] == sex)]
            band = hc.band_label(*band_bounds(age))
            fallback_entry = band_entry.get((band, sex), 0.0)
            fallback_exit = band_exit.get((band, sex), 0.0)
            ep = sub[~sub["cohabiting"]]
            xp = sub[sub["cohabiting"]]
            entry[(age, sex)] = (
                weighted_rate(
                    ep, ep["next_cohabiting"].fillna(False).to_numpy(float)
                )
                if len(ep) >= MIN_STRATUM_N
                else fallback_entry
            )
            exit_[(age, sex)] = (
                weighted_rate(
                    xp, xp["next_cohabiting"].eq(False).to_numpy(float)
                )
                if len(xp) >= MIN_STRATUM_N
                else fallback_exit
            )
    return entry, exit_


def fit_female_cohab_single_year(
    person_waves: pd.DataFrame,
    cohab_flag: pd.DataFrame,
    train_ids: set[int],
    band_entry: dict[tuple[str, str], float],
    band_exit: dict[tuple[str, str], float],
) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    """Fit the female ages 25--44 refinement.

    Copied from ``household_composition_sim_v6.py:307-363``.
    """
    pw = attach_cohabitation(person_waves, cohab_flag)
    pw = pw[pw["person_id"].isin(train_ids)]
    hasn = pw[pw["has_next"] & pw["band"].notna()]
    entry: dict[int, float] = {}
    exit_: dict[int, float] = {}
    diag_rows: dict[str, dict[str, float | int]] = {}
    for age in range(COHAB_FEMALE_REFIT_LO, COHAB_FEMALE_REFIT_HI + 1):
        sub = hasn[(hasn["age"] == age) & (hasn["sex"] == "female")]
        band = hc.band_label(*band_bounds(age))
        fallback_entry = band_entry.get((band, "female"), 0.0)
        fallback_exit = band_exit.get((band, "female"), 0.0)
        ep = sub[~sub["cohabiting"]]
        xp = sub[sub["cohabiting"]]
        entry_rate = (
            weighted_rate(
                ep, ep["next_cohabiting"].fillna(False).to_numpy(np.float64)
            )
            if len(ep) >= MIN_STRATUM_N
            else fallback_entry
        )
        exit_rate = (
            weighted_rate(
                xp, xp["next_cohabiting"].eq(False).to_numpy(np.float64)
            )
            if len(xp) >= MIN_STRATUM_N
            else fallback_exit
        )
        entry[age] = entry_rate
        exit_[age] = exit_rate
        diag_rows[str(age)] = {
            "entry": round(entry_rate, 5),
            "exit": round(exit_rate, 5),
            "equilibrium": (
                round(entry_rate / (entry_rate + exit_rate), 5)
                if (entry_rate + exit_rate) > 0
                else 0.0
            ),
            "n_entry_atrisk": int(len(ep)),
            "n_exit_atrisk": int(len(xp)),
        }
    return (
        entry,
        exit_,
        {
            "refit_age_range": [COHAB_FEMALE_REFIT_LO, COHAB_FEMALE_REFIT_HI],
            "female_single_year": diag_rows,
        },
    )


def apply_overlay_lift(
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    spouse: np.ndarray,
    lift: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the 25--34 female Bernoulli superposition.

    Copied from ``household_composition_sim_v8.py:971-1011``.
    """
    spouse_new = np.asarray(spouse, dtype=bool).copy()
    subset = (
        (band == COHAB_OVERLAY_LIFT_BAND) & (sex == "female") & (~spouse_new)
    )
    before = weighted_share(
        weight,
        spouse_new,
        (sex == "female") & (band == COHAB_OVERLAY_LIFT_BAND),
    )
    if subset.any():
        u = rng.random(len(spouse_new))
        flip = subset & (u < lift)
        spouse_new[flip] = True
    after = weighted_share(
        weight,
        spouse_new,
        (sex == "female") & (band == COHAB_OVERLAY_LIFT_BAND),
    )
    return spouse_new, {
        "band": COHAB_OVERLAY_LIFT_BAND,
        "lift": float(lift),
        "rate_before": before,
        "rate_after": after,
        "realized_lift": after - before,
    }
