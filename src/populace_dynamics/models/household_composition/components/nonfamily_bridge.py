"""Count-conditional non-family household-member bridge.

The roster core-size input is copied from
``household_composition_sim_v3.py:224-251``.  Candidate 9's effective full
count fit and draw are copied from
``household_composition_sim_v6.py:369-424,655-681``.  Superseded 0/1/2+
fitters are intentionally omitted because this full conditional replaces them
and consumes the same one-uniform-per-row C3 substream shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import relmap

CORE_SIZE_CAP = 5


def family_unit_sizes(relationship_map: pd.DataFrame) -> pd.DataFrame:
    """Return ego-centric nuclear family-unit size by person-wave.

    Copied from ``household_composition_sim_v3.py:224-251``.
    """
    nonself = relationship_map[
        relationship_map["ego_rel_to_alter"] != relmap.SELF
    ]
    family_codes = (
        hc.CORESIDENCE_LINKS["coresident_spouse"]
        | hc.CORESIDENCE_LINKS["coresident_child"]
        | hc.CORESIDENCE_LINKS["coresident_parent"]
    )
    family = nonself[nonself["ego_rel_to_alter"].isin(family_codes)]
    counts = (
        family.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("_n_fam")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    counts["family_unit_size"] = counts["_n_fam"].astype("int64") + 1
    return counts[["person_id", "year", "family_unit_size"]]


def fit_nonfamily_count_by_core(
    person_waves: pd.DataFrame,
    family_sizes: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Fit full non-core count conditional on capped core size.

    Copied from ``household_composition_sim_v6.py:369-424``.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "weight", "hh_size"]
    ].merge(family_sizes, on=["person_id", "year"], how="left")
    pw["family_unit_size"] = pw["family_unit_size"].fillna(1).astype("int64")
    residual = np.clip(
        pw["hh_size"].to_numpy() - pw["family_unit_size"].to_numpy(), 0, None
    ).astype(np.int64)
    capped = np.clip(
        pw["family_unit_size"].to_numpy(), 1, CORE_SIZE_CAP
    ).astype(np.int64)
    weight = pw["weight"].to_numpy(np.float64)
    table: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    incidence: dict[str, float] = {}
    mean_count: dict[str, float] = {}
    for core_size in range(1, CORE_SIZE_CAP + 1):
        mask = capped == core_size
        cell_weight = weight[mask]
        cell_residual = residual[mask]
        total = float(cell_weight.sum())
        if total <= 0:
            table[core_size] = (
                np.array([0], dtype=np.int64),
                np.array([1.0]),
            )
            incidence[str(core_size)] = 0.0
            mean_count[str(core_size)] = 0.0
            continue
        values = np.arange(0, int(cell_residual.max()) + 1, dtype=np.int64)
        probabilities = np.array(
            [
                float(cell_weight[cell_residual == value].sum() / total)
                for value in values
            ],
            dtype=np.float64,
        )
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        table[core_size] = (values, cumulative)
        incidence[str(core_size)] = round(float(1.0 - probabilities[0]), 5)
        mean_count[str(core_size)] = round(
            float((values * probabilities).sum()), 5
        )
    return table, {
        "core_size_cap": CORE_SIZE_CAP,
        "noncore_incidence_by_capped_core": incidence,
        "mean_noncore_count_by_capped_core": mean_count,
        "max_noncore_count_by_core": {
            str(key): int(table[key][0][-1]) for key in table
        },
    }


def sample_nonfamily_count(
    pw: pd.DataFrame,
    table: dict[int, tuple[np.ndarray, np.ndarray]],
    count_rng: np.random.Generator,
    core_size: np.ndarray,
) -> np.ndarray:
    """Draw full non-core counts from the C3 non-family stream.

    Copied from ``household_composition_sim_v6.py:655-681``.
    """
    n = len(pw)
    core = np.clip(np.asarray(core_size, dtype=np.int64), 1, CORE_SIZE_CAP)
    uniform = count_rng.random(n)
    counts = np.zeros(n, dtype=np.int64)
    for key in range(1, CORE_SIZE_CAP + 1):
        mask = core == key
        if not mask.any():
            continue
        values, cumulative = table[key]
        index = np.searchsorted(cumulative, uniform[mask], side="left")
        index = np.clip(index, 0, len(values) - 1)
        counts[mask] = values[index]
    return counts
