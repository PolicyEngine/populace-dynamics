"""Vestigial candidate-16 spousal-age-gap component.

Candidate 16 retains the candidate-12 age-banded empirical draw even though
its untrended surviving-spouse hazard no longer reads spouse age. The fit and
fallback port ``scripts/run_gate2_candidate12.py:317-460`` and the draw ports
``scripts/run_gate2_candidate12.py:793-831``. Keeping the component preserves
the registered RNG topology without importing a frozen runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.models.family_transitions.common import band_indices

__all__ = ["AGE_BANDS", "draw_spousal_gaps", "fit_spousal_age_gaps"]

AGE_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 49),
    (50, 64),
    (65, 120),
)
AGE_LOWERS = np.array([lo for lo, _ in AGE_BANDS], dtype=np.int64)
MIN_WEIGHTED_COUPLES = 200.0


def _fallback_group(band: int, weighted_n: np.ndarray) -> list[int]:
    """Extend toward younger bands until the effective count reaches 200."""
    total = float(weighted_n[band])
    first = band
    while total < MIN_WEIGHTED_COUPLES and first > 0:
        first -= 1
        total += float(weighted_n[first])
    return list(range(first, band + 1))


def fit_spousal_age_gaps(
    marriage_records: pd.DataFrame,
    attrs: pd.DataFrame,
    train_ids: set[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Fit empirical gap arrays by ego sex and age at marriage.

    Record selection, normalized effective weights, adjacent fallback, and
    concatenation order preserve ``scripts/run_gate2_candidate12.py:317-430``.
    Undatable marriages remain available only to the sex-pooled final fallback,
    as in the frozen implementation.
    """
    person_birth = (
        marriage_records.dropna(subset=["birth_year"])
        .groupby("person_id")["birth_year"]
        .first()
    )
    records = marriage_records[
        marriage_records["is_marriage"]
        & marriage_records["spouse_person_id"].notna()
        & marriage_records["person_id"].isin(train_ids)
    ].copy()
    records["self_birth"] = (
        records["person_id"].map(person_birth).astype("float64")
    )
    records["spouse_birth"] = (
        records["spouse_person_id"].map(person_birth).astype("float64")
    )
    records = records[
        records["self_birth"].notna() & records["spouse_birth"].notna()
    ].copy()
    records["gap"] = np.rint(
        records["self_birth"] - records["spouse_birth"]
    ).astype(np.int64)
    records["ego_age_at_marriage"] = (
        records["start_year"].astype("float64") - records["self_birth"]
    )
    weight_by_person = attrs.set_index("person_id")["weight"]
    records["couple_weight"] = (
        records["person_id"]
        .map(weight_by_person)
        .astype("float64")
        .fillna(0.0)
    )

    distributions: dict[str, dict[int, np.ndarray]] = {}
    for sex in ("female", "male"):
        sex_records = records[records["sex"] == sex]
        datable = sex_records[
            sex_records["ego_age_at_marriage"].notna()
        ].copy()
        age_band = band_indices(
            np.rint(datable["ego_age_at_marriage"].to_numpy()).astype(
                np.int64
            ),
            AGE_LOWERS,
            len(AGE_BANDS),
        )
        datable = datable.assign(gap_band=age_band)
        weights = datable["couple_weight"].to_numpy(dtype=np.float64)
        mean_weight = (
            float(weights.mean())
            if weights.size and weights.mean() > 0
            else 1.0
        )
        effective_weight = weights / mean_weight
        gaps_by_band: dict[int, np.ndarray] = {}
        weighted_n = np.zeros(len(AGE_BANDS), dtype=np.float64)
        for band in range(len(AGE_BANDS)):
            mask = age_band == band
            gaps_by_band[band] = datable.loc[mask, "gap"].to_numpy(
                dtype=np.int64
            )
            weighted_n[band] = float(effective_weight[mask].sum())
        pooled = sex_records["gap"].to_numpy(dtype=np.int64)
        if pooled.size == 0:
            pooled = np.zeros(1, dtype=np.int64)
        sex_distribution: dict[int, np.ndarray] = {}
        for band in range(len(AGE_BANDS)):
            group = _fallback_group(band, weighted_n)
            values = np.concatenate(
                [gaps_by_band[index] for index in group]
                + [np.empty(0, dtype=np.int64)]
            )
            if values.size == 0:
                values = pooled
            sex_distribution[band] = values
        distributions[sex] = sex_distribution
    return distributions


def draw_spousal_gaps(
    rng: np.random.Generator,
    indices: np.ndarray,
    marriage_age: np.ndarray,
    is_male: np.ndarray,
    distributions: dict[str, dict[int, np.ndarray]],
) -> np.ndarray:
    """Draw gaps in the frozen female-bands-then-male-bands order."""
    output = np.empty(indices.size, dtype=np.float64)
    bands = band_indices(
        np.rint(marriage_age).astype(np.int64),
        AGE_LOWERS,
        len(AGE_BANDS),
    )
    male = is_male[indices] == 1.0
    for sex_index, sex_mask in ((0, ~male), (1, male)):
        sex = ("female", "male")[sex_index]
        for band in range(len(AGE_BANDS)):
            mask = sex_mask & (bands == band)
            count = int(mask.sum())
            if count:
                output[mask] = rng.choice(distributions[sex][band], size=count)
    return output
