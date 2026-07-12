"""Household-size composition and parent-count mixture.

The base composition identity is copied from
``household_composition_sim.py:82-109``.  Candidate 9's effective per-ego
parent-count inputs and fit are copied from
``household_composition_sim_v5.py:456-521``.  The final simulator composes the
core, adds the count-conditional bridge, and then applies the scoped
fertility-core resample in the frozen order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.models.household_composition.common import weighted_rate

MIN_STRATUM_N = 20


def compose_states(
    spouse: np.ndarray,
    parent: np.ndarray,
    multigen: np.ndarray,
    child_counts: np.ndarray,
    parent_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose child flag, grandchild flag, and base household size.

    Copied from ``household_composition_sim.py:82-109``.
    """
    spouse = np.asarray(spouse, dtype=bool)
    parent = np.asarray(parent, dtype=bool)
    multigen = np.asarray(multigen, dtype=bool)
    child_counts = np.asarray(child_counts, dtype=np.int64)
    coresident_child = child_counts > 0
    n_parents = np.where(parent, int(parent_count), 0)
    hh_size = (1 + spouse.astype(np.int64) + child_counts + n_parents).astype(
        np.int64
    )
    coresident_grandchild = multigen & coresident_child & (~parent)
    return coresident_child, coresident_grandchild, hh_size


def parent_link_counts(relationship_map: pd.DataFrame) -> pd.DataFrame:
    """Return the observed count of coresident-parent links.

    Copied from ``household_composition_sim_v5.py:456-475``.
    """
    codes = hc.CORESIDENCE_LINKS["coresident_parent"]
    links = relationship_map[relationship_map["ego_rel_to_alter"].isin(codes)]
    counts = (
        links.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("n_parent_links")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    return counts[["person_id", "year", "n_parent_links"]]


def fit_parent_count_composition(
    person_waves: pd.DataFrame,
    parent_counts: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[tuple[str, str], float], float, dict[str, Any]]:
    """Fit ``P(two parents | coresident parent, band, sex)``.

    Copied from ``household_composition_sim_v5.py:478-521``.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)][
        [
            "person_id",
            "year",
            "band",
            "sex",
            "weight",
            "coresident_parent",
        ]
    ].merge(parent_counts, on=["person_id", "year"], how="left")
    pw["n_parent_links"] = pw["n_parent_links"].fillna(0).astype("int64")
    cp = pw[pw["coresident_parent"] & (pw["n_parent_links"] >= 1)]
    two = (cp["n_parent_links"] >= 2).to_numpy(np.float64)
    pooled = weighted_rate(cp, two)
    table: dict[tuple[str, str], float] = {}
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            sub = cp[(cp["band"] == band) & (cp["sex"] == sex)]
            table[(band, sex)] = (
                weighted_rate(
                    sub, (sub["n_parent_links"] >= 2).to_numpy(np.float64)
                )
                if len(sub) >= MIN_STRATUM_N
                else pooled
            )
    return (
        table,
        pooled,
        {
            "pooled_p_two_parents_given_coresident_parent": round(pooled, 5),
            "p_two_parents_by_band_female": {
                hc.band_label(lo, hi): round(
                    table[(hc.band_label(lo, hi), "female")], 5
                )
                for lo, hi in hc.COMPOSITION_AGE_BANDS
            },
            "n_coresident_parent_waves_train": int(len(cp)),
        },
    )
