"""Shared, numerically stable helpers for family-transition components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.data import marriage

__all__ = ["band_indices", "marriage_order_map", "prior_event_counts"]


def band_indices(
    values: np.ndarray, lowers: np.ndarray, n_bands: int
) -> np.ndarray:
    """Return clipped band indices for ``values``.

    The operation order is ported from
    ``scripts/run_gate2_candidate1.py:345-347`` and is shared by every
    candidate-16 empirical hazard lookup.
    """
    return np.clip(
        np.searchsorted(lowers, values, side="right") - 1,
        0,
        n_bands - 1,
    )


def prior_event_counts(
    person_ids: np.ndarray,
    years: np.ndarray,
    events_by_person: dict[int, np.ndarray],
) -> np.ndarray:
    """Count strictly prior events for each person-year.

    This is the candidate-1 parity helper from
    ``scripts/run_gate2_candidate1.py:617-634``. Candidate 16 uses it when
    fitting the single-year fertility component.
    """
    out = np.zeros(len(person_ids), dtype=np.int64)
    order = np.argsort(person_ids, kind="stable")
    sorted_person = person_ids[order]
    sorted_year = years[order]
    starts = np.searchsorted(
        sorted_person, np.unique(sorted_person), side="left"
    )
    bounds = np.append(starts, len(sorted_person))
    for i in range(len(starts)):
        segment = slice(bounds[i], bounds[i + 1])
        person_id = int(sorted_person[bounds[i]])
        event_years = events_by_person.get(person_id)
        if event_years is None:
            continue
        out[order[segment]] = np.searchsorted(
            event_years, sorted_year[segment], side="left"
        )
    return out


def marriage_order_map(marriage_records: pd.DataFrame) -> pd.DataFrame:
    """Build the datable per-person marriage order by start year.

    Ported without changing sort or duplicate semantics from
    ``scripts/run_gate2_candidate1.py:1296-1305``.
    """
    episodes = marriage.marriage_episodes(marriage_records)
    episodes = episodes[episodes["start_year"].notna()].copy()
    episodes["start_year"] = episodes["start_year"].astype("int64")
    episodes = episodes.sort_values(["person_id", "start_year"])
    episodes["order"] = episodes.groupby("person_id").cumcount() + 1
    return episodes[["person_id", "start_year", "order"]].drop_duplicates(
        ["person_id", "start_year"]
    )
