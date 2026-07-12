"""Skipped-generation grandchild occupancy component.

Copied from ``household_composition_sim_v3.py:395-458`` and
``household_composition_sim_v4.py:106-117,614-651``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.models.household_composition.common import weighted_rate
from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
    band_bounds,
)

MIN_STRATUM_N = 20
SKIPGEN_AGE_BANDS_55PLUS: tuple[tuple[int, int], ...] = (
    (55, 59),
    (60, 64),
    (65, 69),
    (70, 74),
    (75, 79),
    (80, 120),
)


def attach_skipgen(person_waves: pd.DataFrame) -> pd.DataFrame:
    """Attach observed current and next skipped-generation state.

    Copied from ``household_composition_sim_v3.py:395-411``.
    """
    pw = person_waves.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw["skipgen"] = pw["coresident_grandchild"].to_numpy(dtype=bool) & ~pw[
        "multigen"
    ].to_numpy(dtype=bool)
    pw["next_skipgen"] = pw.groupby("person_id", sort=False)["skipgen"].shift(
        -1
    )
    return pw


def fit_skipgen_rates(
    train_pw: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Fit band-by-sex skipped-generation hazards.

    Copied from ``household_composition_sim_v3.py:414-458``.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry_pool = has_next[~has_next["skipgen"]]
    exit_pool = has_next[has_next["skipgen"]]
    entry_overall = weighted_rate(
        entry_pool, entry_pool["next_skipgen"].to_numpy(dtype=np.float64)
    )
    exit_overall = weighted_rate(
        exit_pool,
        exit_pool["next_skipgen"].eq(False).to_numpy(dtype=np.float64),
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
                weighted_rate(e, e["next_skipgen"].to_numpy(np.float64))
                if len(e)
                else entry_overall
            )
            exit_[(band, sex)] = (
                weighted_rate(
                    x, x["next_skipgen"].eq(False).to_numpy(np.float64)
                )
                if len(x)
                else exit_overall
            )
    return entry, exit_


def fit_skipgen_5yr(
    train_pw: pd.DataFrame,
    band_entry: dict[tuple[str, str], float],
    band_exit: dict[tuple[str, str], float],
) -> tuple[
    dict[tuple[tuple[int, int], str], float],
    dict[tuple[tuple[int, int], str], float],
]:
    """Fit 55+ five-year skipped-generation hazards.

    Copied from ``household_composition_sim_v4.py:614-651``.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry: dict[tuple[tuple[int, int], str], float] = {}
    exit_: dict[tuple[tuple[int, int], str], float] = {}
    for lo, hi in SKIPGEN_AGE_BANDS_55PLUS:
        gate_band = hc.band_label(*band_bounds(lo))
        ages = has_next[(has_next["age"] >= lo) & (has_next["age"] <= hi)]
        for sex in hc.SEXES:
            subset = ages[ages["sex"] == sex]
            ep = subset[~subset["skipgen"]]
            xp = subset[subset["skipgen"]]
            entry[((lo, hi), sex)] = (
                weighted_rate(ep, ep["next_skipgen"].to_numpy(np.float64))
                if len(ep) >= MIN_STRATUM_N
                else band_entry.get((gate_band, sex), 0.0)
            )
            exit_[((lo, hi), sex)] = (
                weighted_rate(
                    xp, xp["next_skipgen"].eq(False).to_numpy(np.float64)
                )
                if len(xp) >= MIN_STRATUM_N
                else band_exit.get((gate_band, sex), 0.0)
            )
    return entry, exit_
