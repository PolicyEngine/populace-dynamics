"""Multigenerational occupancy and adult-child coupling component.

The occupancy fit is copied from ``household_composition_sim.py:227-272``.
The 55+ adult-child coupling is copied from
``household_composition_sim_v5.py:104-125,135-140,196-258``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.models.household_composition.common import weighted_rate

GRANDCHILD_LO = 55
DELTA_STREAM_TAG_V5 = 0xC5
MIN_STRATUM_N = 20
COUPLING_AGE_BANDS_55PLUS: tuple[tuple[int, int], ...] = tuple(
    (lo, hi) for lo, hi in hc.COMPOSITION_AGE_BANDS if lo >= GRANDCHILD_LO
)


def fit_multigen_rates(
    train_pw: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Fit multigenerational entry and exit by band and sex.

    Copied from ``household_composition_sim.py:227-272``.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry_pool = has_next[~has_next["multigen"]]
    exit_pool = has_next[has_next["multigen"]]
    entry_overall = weighted_rate(
        entry_pool, entry_pool["next_multigen"].to_numpy(dtype=np.float64)
    )
    exit_overall = weighted_rate(
        exit_pool,
        exit_pool["next_multigen"].eq(False).to_numpy(dtype=np.float64),
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
                weighted_rate(e, e["next_multigen"].to_numpy(np.float64))
                if len(e)
                else entry_overall
            )
            exit_[(band, sex)] = (
                weighted_rate(
                    x, x["next_multigen"].eq(False).to_numpy(np.float64)
                )
                if len(x)
                else exit_overall
            )
    return entry, exit_


def fit_multigen_child_coupling(
    person_waves: pd.DataFrame, train_ids: set[int]
) -> tuple[
    dict[tuple[str, str, bool], float],
    dict[tuple[str, bool], float],
    dict[str, Any],
]:
    """Fit ``P(child | multigen, 55+ band, sex)``.

    Copied from ``household_composition_sim_v5.py:196-258``.
    """
    pw = person_waves[
        person_waves["person_id"].isin(train_ids)
        & (person_waves["age"] >= GRANDCHILD_LO)
    ]
    pooled: dict[tuple[str, bool], float] = {}
    for sex in hc.SEXES:
        for multigen in (False, True):
            sub = pw[(pw["sex"] == sex) & (pw["multigen"] == multigen)]
            pooled[(sex, multigen)] = weighted_rate(
                sub, sub["coresident_child"].to_numpy(np.float64)
            )
    table: dict[tuple[str, str, bool], float] = {}
    for lo, hi in COUPLING_AGE_BANDS_55PLUS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            for multigen in (False, True):
                sub = pw[
                    (pw["band"] == band)
                    & (pw["sex"] == sex)
                    & (pw["multigen"] == multigen)
                ]
                key = (band, sex, multigen)
                table[key] = (
                    weighted_rate(
                        sub, sub["coresident_child"].to_numpy(np.float64)
                    )
                    if len(sub) >= MIN_STRATUM_N
                    else pooled[(sex, multigen)]
                )
    diag = {
        "p_child_given_multigen_true_female_by_band": {
            hc.band_label(lo, hi): round(
                table[(hc.band_label(lo, hi), "female", True)], 5
            )
            for lo, hi in COUPLING_AGE_BANDS_55PLUS
        },
        "pooled_p_child_given_multigen_true_female": round(
            pooled[("female", True)], 5
        ),
        "pooled_p_child_given_multigen_false_female": round(
            pooled[("female", False)], 5
        ),
        "n_train_55plus_female_waves": int(len(pw[pw["sex"] == "female"])),
    }
    return table, pooled, diag
