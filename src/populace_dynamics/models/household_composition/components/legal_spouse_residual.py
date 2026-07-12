"""Legal-spouse residual top-up for resolved candidate 9.

Copied from ``household_composition_sim_v4.py:125-143,273-410``.  The
simulate-time state evolution remains in the flattened simulator so its C4
stream ordering is explicit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import relmap, transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models.household_composition.common import weighted_rate
from populace_dynamics.models.household_composition.components.marital_core_adapter import (
    spouse_from_marital,
)

LEGAL_SPOUSE_CODE = relmap.SPOUSE
MIN_STRATUM_N = 20
LEGAL_RESIDUAL_MIX_EXIT = 0.5


def legal_spouse_flag(relationship_map: pd.DataFrame) -> pd.DataFrame:
    """Return observed code-20 legal-spouse state by person-wave.

    Copied from ``household_composition_sim_v4.py:273-292``.
    """
    nonself = relationship_map[
        relationship_map["ego_rel_to_alter"] != relmap.SELF
    ]
    legal = nonself[nonself["ego_rel_to_alter"] == LEGAL_SPOUSE_CODE]
    hits = (
        legal.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("_n")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    hits["legal_spouse_obs"] = hits["_n"] > 0
    return hits[["person_id", "year", "legal_spouse_obs"]]


def fit_legal_residual(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    family_transitions: ft.FittedFamilyTransitions,
    train_ids: set[int],
    legal_flag: pd.DataFrame,
    *,
    core_seed: int = 5200,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, Any],
]:
    """Fit the additive legal-spouse residual overlay.

    Copied from ``household_composition_sim_v4.py:295-410``.
    """
    pw = hh.person_waves.merge(
        legal_flag, on=["person_id", "year"], how="left"
    )
    pw["legal_spouse_obs"] = pw["legal_spouse_obs"].fillna(False).astype(bool)
    pw = pw.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw["next_legal"] = pw.groupby("person_id", sort=False)[
        "legal_spouse_obs"
    ].shift(-1)
    train = pw[pw["person_id"].isin(train_ids)]

    sim_panel, _ = ft.simulate(
        mpanel, train_ids, family_transitions, core_seed
    )
    train_ordered = (
        hh.person_waves[hh.person_waves["person_id"].isin(train_ids)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )
    core_spouse = spouse_from_marital(train_ordered, sim_panel.person_years)
    train_ordered = train_ordered.assign(_core_spouse=core_spouse)

    entry: dict[tuple[str, str], float] = {}
    exit_: dict[tuple[str, str], float] = {}
    marginal: dict[tuple[str, str], float] = {}
    target: dict[tuple[str, str], float] = {}
    ref_code20: dict[tuple[str, str], float] = {}
    core_legal: dict[tuple[str, str], float] = {}
    hasn = train[train["has_next"] & train["band"].notna()]
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            key = (band, sex)
            sub = train[(train["band"] == band) & (train["sex"] == sex)]
            ref = weighted_rate(sub, sub["legal_spouse_obs"].to_numpy(float))
            csub = train_ordered[
                (train_ordered["band"] == band) & (train_ordered["sex"] == sex)
            ]
            core = weighted_rate(csub, csub["_core_spouse"].to_numpy(float))
            ref_code20[key] = ref
            core_legal[key] = core
            residual_target = max(0.0, ref - core)
            target[key] = residual_target
            xp = hasn[
                (hasn["band"] == band)
                & (hasn["sex"] == sex)
                & hasn["legal_spouse_obs"]
            ]
            fitted_exit = (
                weighted_rate(xp, xp["next_legal"].eq(False).to_numpy(float))
                if len(xp) >= MIN_STRATUM_N
                else 0.1
            )
            exit_rate = max(fitted_exit, LEGAL_RESIDUAL_MIX_EXIT)
            exit_[key] = exit_rate
            marg = residual_target / (1.0 - core) if core < 1.0 else 0.0
            marg = float(min(max(marg, 0.0), 0.95))
            marginal[key] = marg
            entry[key] = marg * exit_rate / (1.0 - marg) if marg < 1.0 else 1.0
    diag = {
        "ref_code20_stock": {
            f"{band}|{sex}": round(value, 5)
            for (band, sex), value in ref_code20.items()
        },
        "core_legal_stock": {
            f"{band}|{sex}": round(value, 5)
            for (band, sex), value in core_legal.items()
        },
        "residual_target_stock": {
            f"{band}|{sex}": round(value, 5)
            for (band, sex), value in target.items()
        },
        "core_seed": core_seed,
        "n_bands_active": int(sum(value > 0 for value in target.values())),
    }
    return entry, exit_, marginal, target, diag
