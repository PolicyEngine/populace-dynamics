"""Candidate-16 support-composed surviving-spouse widowhood component.

The seven bands and aggregate fit port
``scripts/run_gate2_candidate14.py:262-280,379-430``. Candidate 15's untrended
application is resolved directly. Candidate 16's two support-stratum fits,
exposure-weighted recombination, and probability lookup port
``scripts/run_gate2_candidate16.py:497-634,757-823``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import transitions
from populace_dynamics.models.family_transitions.common import band_indices

__all__ = [
    "AGE_BANDS",
    "SUPPORT_STRATA",
    "WidowhoodModel",
    "fit_widowhood",
    "widowhood_probabilities",
]

AGE_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 84),
    (85, 120),
)
AGE_LOWERS = np.array([lo for lo, _ in AGE_BANDS], dtype=np.int64)
SUPPORT_STRATA = (0, 1)


@dataclass(frozen=True)
class WidowhoodModel:
    """Stratified widowhood levels and their aggregate reconciliation."""

    level_by_stratum: dict[int, dict[str, float]]
    lookup: np.ndarray
    recombination: dict[str, Any]


def _hazard_cells(
    person_years: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    cells = transitions._hazard_by_band(
        events[events["transition"] == "widowhood"],
        person_years[person_years["marital_state"] == "married"],
        "age",
        AGE_BANDS,
        prefix="widowhood",
        by_sex=True,
        weighted=True,
    )
    prefix = "widowhood."
    return {
        key[len(prefix) :]: dict(cell)
        for key, cell in cells.items()
        if key.startswith(prefix)
    }


def _recombine(
    stratified: dict[int, dict[str, dict[str, float]]],
    aggregate: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Recombine stratum rates in candidate-16 operation order.

    The multiplication by each stratum exposure is deliberate: replacing it
    with summed numerators can change binary64 results. This is the exact
    arithmetic from ``scripts/run_gate2_candidate16.py:543-634``.
    """
    cells: dict[str, Any] = {}
    max_rate_residual = 0.0
    max_numerator_residual = 0.0
    max_denominator_residual = 0.0
    for key in sorted(aggregate):
        aggregate_cell = aggregate[key]
        numerator_0 = float(stratified[0][key]["num_wt"])
        denominator_0 = float(stratified[0][key]["den_wt"])
        rate_0 = float(stratified[0][key]["rate"])
        numerator_1 = float(stratified[1][key]["num_wt"])
        denominator_1 = float(stratified[1][key]["den_wt"])
        rate_1 = float(stratified[1][key]["rate"])
        aggregate_numerator = float(aggregate_cell["num_wt"])
        aggregate_denominator = float(aggregate_cell["den_wt"])
        aggregate_rate = float(aggregate_cell["rate"])
        denominator_sum = denominator_0 + denominator_1
        recombined = (
            (denominator_0 * rate_0 + denominator_1 * rate_1) / denominator_sum
            if denominator_sum > 0
            else 0.0
        )
        rate_residual = abs(recombined - aggregate_rate)
        numerator_residual = abs(
            (numerator_0 + numerator_1) - aggregate_numerator
        )
        denominator_residual = abs(denominator_sum - aggregate_denominator)
        max_rate_residual = max(max_rate_residual, rate_residual)
        max_numerator_residual = max(
            max_numerator_residual, numerator_residual
        )
        max_denominator_residual = max(
            max_denominator_residual, denominator_residual
        )
        cells[key] = {
            "stratum_0": {
                "rate": rate_0,
                "num_wt": numerator_0,
                "den_wt": denominator_0,
            },
            "stratum_1": {
                "rate": rate_1,
                "num_wt": numerator_1,
                "den_wt": denominator_1,
            },
            "aggregate_rate": aggregate_rate,
            "recombined_rate": recombined,
            "abs_residual_recombined_vs_aggregate": rate_residual,
        }
    return {
        "max_abs_residual_recombined_vs_aggregate": max_rate_residual,
        "max_abs_residual_num_wt": max_numerator_residual,
        "max_abs_residual_den_wt": max_denominator_residual,
        "reconciled": bool(
            max_rate_residual <= 1e-9
            and max_numerator_residual <= 1e-6
            and max_denominator_residual <= 1e-6
        ),
        "cells": cells,
    }


def fit_widowhood(
    panel: transitions.MaritalPanel,
    train_ids: set[int],
    support_by_person: pd.Series,
) -> WidowhoodModel:
    """Fit seven-band widowhood rates by sex and observed support stratum.

    Each stratum calls the gate reference's own weighted hazard implementation
    independently, with no add-one smoothing, exactly as in
    ``scripts/run_gate2_candidate16.py:497-540``.
    """
    train_person_years = panel.person_years[
        panel.person_years["person_id"].isin(train_ids)
    ]
    train_events = panel.events[panel.events["person_id"].isin(train_ids)]
    married = train_person_years[
        train_person_years["marital_state"] == "married"
    ].copy()
    widowhood = train_events[train_events["transition"] == "widowhood"].copy()
    married["stratum"] = (
        married["person_id"].map(support_by_person).fillna(0).astype(np.int64)
    )
    widowhood["stratum"] = (
        widowhood["person_id"]
        .map(support_by_person)
        .fillna(0)
        .astype(np.int64)
    )
    stratified: dict[int, dict[str, dict[str, float]]] = {}
    for stratum in SUPPORT_STRATA:
        stratified[stratum] = _hazard_cells(
            married[married["stratum"] == stratum],
            widowhood[widowhood["stratum"] == stratum],
        )
    aggregate = _hazard_cells(train_person_years, train_events)
    level_by_stratum = {
        stratum: {
            key: float(cell["rate"])
            for key, cell in stratified[stratum].items()
        }
        for stratum in SUPPORT_STRATA
    }
    lookup = np.zeros((len(AGE_BANDS), 2, len(SUPPORT_STRATA)), np.float64)
    for band_index, (lower, upper) in enumerate(AGE_BANDS):
        label = transitions.band_label(lower, upper)
        for sex_index, sex in enumerate(("female", "male")):
            for stratum in SUPPORT_STRATA:
                lookup[band_index, sex_index, stratum] = level_by_stratum[
                    stratum
                ].get(f"{label}|{sex}", 0.0)
    recombination = _recombine(stratified, aggregate)
    if not recombination["reconciled"]:
        raise RuntimeError("support-stratum widowhood rates did not recombine")
    return WidowhoodModel(
        level_by_stratum=level_by_stratum,
        lookup=lookup,
        recombination=recombination,
    )


def widowhood_probabilities(
    ego_age: np.ndarray,
    ego_is_male: np.ndarray,
    support_stratum: np.ndarray,
    lookup: np.ndarray,
) -> np.ndarray:
    """Return the untrended candidate-16 surviving-spouse hazard.

    The married ego's own age, sex, and support stratum index the level. No
    NCHS period multiplier is applied, preserving
    ``scripts/run_gate2_candidate16.py:795-823``.
    """
    bands = band_indices(
        np.rint(ego_age).astype(np.int64), AGE_LOWERS, len(AGE_BANDS)
    )
    sex_index = ego_is_male.astype(np.int64)
    return lookup[bands, sex_index, support_stratum]
