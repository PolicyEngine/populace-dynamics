"""Candidate-16 empirical remarriage component.

The five current-age bands come from
``scripts/run_gate2_candidate11.py:159-171``. The weighted
age-band-by-years-since-dissolution-by-origin-by-sex fit and dense lookup port
``scripts/run_gate2_candidate10.py:260-328`` and
``scripts/run_gate2_candidate10.py:395-414`` without importing either frozen
runner.
"""

from __future__ import annotations

import numpy as np

from populace_dynamics.data import transitions
from populace_dynamics.models.family_transitions.common import band_indices

__all__ = [
    "AGE_BANDS",
    "YSD_BANDS",
    "build_remarriage_lookup",
    "fit_remarriage",
    "remarriage_probabilities",
]

AGE_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 49),
    (50, 64),
    (65, 74),
    (75, 120),
)
YSD_BANDS = transitions.REMARRIAGE_YSD_BANDS
AGE_LOWERS = np.array([lo for lo, _ in AGE_BANDS], dtype=np.int64)
YSD_LOWERS = np.array([lo for lo, _ in YSD_BANDS], dtype=np.int64)


def fit_remarriage(
    panel: transitions.MaritalPanel,
    train_ids: set[int],
) -> dict[tuple[int, int, str, str], float]:
    """Fit the candidate-16 banded remarriage hazard.

    Each cell receives the frozen mean-weight add-one smoothing
    ``(wnum + wbar) / (wden + 2*wbar)``. Selection, grouping, loop order, and
    arithmetic preserve ``scripts/run_gate2_candidate10.py:260-328`` while
    :data:`AGE_BANDS` supplies candidate 11's resolved five-band table.
    """
    train_person_years = panel.person_years[
        panel.person_years["person_id"].isin(train_ids)
    ]
    train_events = panel.events[panel.events["person_id"].isin(train_ids)]
    dissolved = train_person_years[
        train_person_years["marital_state"].isin(("divorced", "widowed"))
        & train_person_years["years_since_dissolution"].notna()
    ].copy()
    remarriages = train_events[
        (train_events["transition"] == "remarriage")
        & train_events["years_since_dissolution"].notna()
    ].copy()
    mean_weight = float(dissolved["weight"].mean()) if len(dissolved) else 1.0
    for frame in (dissolved, remarriages):
        frame["ysd_band"] = band_indices(
            frame["years_since_dissolution"].astype("int64").to_numpy(),
            YSD_LOWERS,
            len(YSD_BANDS),
        )
        frame["age_band"] = band_indices(
            np.rint(frame["age"].to_numpy()).astype(np.int64),
            AGE_LOWERS,
            len(AGE_BANDS),
        )
    denominator = dissolved.groupby(
        ["age_band", "ysd_band", "marital_state", "sex"]
    )["weight"].sum()
    numerator = remarriages.groupby(["age_band", "ysd_band", "origin", "sex"])[
        "weight"
    ].sum()
    table: dict[tuple[int, int, str, str], float] = {}
    for age_band in range(len(AGE_BANDS)):
        for ysd_band in range(len(YSD_BANDS)):
            for origin in ("divorced", "widowed"):
                for sex in ("female", "male"):
                    weighted_numerator = float(
                        numerator.get((age_band, ysd_band, origin, sex), 0.0)
                    )
                    weighted_denominator = float(
                        denominator.get((age_band, ysd_band, origin, sex), 0.0)
                    )
                    table[(age_band, ysd_band, origin, sex)] = (
                        weighted_numerator + mean_weight
                    ) / (weighted_denominator + 2.0 * mean_weight)
    return table


def build_remarriage_lookup(
    table: dict[tuple[int, int, str, str], float],
) -> np.ndarray:
    """Build ``[age_band, ysd_band, origin, sex]`` dense rates.

    The axis ordering is the candidate-16 lookup ordering from
    ``scripts/run_gate2_candidate16.py:778-784``.
    """
    lookup = np.zeros((len(AGE_BANDS), len(YSD_BANDS), 2, 2), dtype=np.float64)
    for (age_band, ysd_band, origin, sex), value in table.items():
        origin_index = 0 if origin == "divorced" else 1
        sex_index = 0 if sex == "female" else 1
        lookup[age_band, ysd_band, origin_index, sex_index] = value
    return lookup


def remarriage_probabilities(
    age: np.ndarray,
    years_since_dissolution: np.ndarray,
    origin_state: np.ndarray,
    is_male: np.ndarray,
    lookup: np.ndarray,
) -> np.ndarray:
    """Look up candidate-16 remarriage probabilities for dissolved egos."""
    age_band = band_indices(
        np.rint(age).astype(np.int64), AGE_LOWERS, len(AGE_BANDS)
    )
    ysd_band = band_indices(
        years_since_dissolution, YSD_LOWERS, len(YSD_BANDS)
    )
    origin_index = (origin_state == 3).astype(np.int64)
    return lookup[
        age_band,
        ysd_band,
        origin_index,
        is_male.astype(np.int64),
    ]
