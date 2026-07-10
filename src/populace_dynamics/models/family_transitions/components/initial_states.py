"""Observed initial-state components selected by candidate 16.

The marriage-count residual is ported from
``scripts/run_gate2_candidate9.py:189-232``. The entry-widowed cells and
injection are ported from ``scripts/run_gate2_candidate12.py:469-526`` and
``scripts/run_gate2_candidate12.py:1103-1133``. The support-composition
stratum is ported from ``scripts/run_gate2_candidate16.py:410-486``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from populace_dynamics.data import transitions

__all__ = [
    "EntryWidowedCells",
    "ObservedInitialStates",
    "SupportComposition",
    "apply_entry_widowed",
    "entry_widowed_cells",
    "fit_observed_initial_states",
    "observed_marriage_residual",
    "observed_support",
    "support_composition",
]

SUPPORT_THRESHOLD_AGE = 75


@dataclass(frozen=True)
class EntryWidowedCells:
    """Sorted reference person-year keys that enter observation widowed."""

    key_sorted: np.ndarray
    years_since_dissolution_sorted: np.ndarray
    person_id_sorted: np.ndarray
    weight_sorted: np.ndarray


@dataclass(frozen=True)
class SupportComposition:
    """Observed binary support stratum keyed by person."""

    by_person: pd.Series

    def align(self, person_ids: np.ndarray) -> np.ndarray:
        """Align the 0/1 stratum to ``person_ids``; missing people map to 0.

        The Series mapping sequence matches
        ``scripts/run_gate2_candidate16.py:477-486``.
        """
        return (
            pd.Series(person_ids)
            .map(self.by_person)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )


@dataclass(frozen=True)
class ObservedInitialStates:
    """Candidate-16 split-independent observed state bundle."""

    marriage_residual_by_person: dict[int, float]
    entry_widowed: EntryWidowedCells
    support: SupportComposition


def observed_support(demographic_panel: pd.DataFrame) -> pd.DataFrame:
    """Return each person's first and last observed demographic wave.

    This is the helper used by candidate 12 and candidate 16, copied from
    ``scripts/gate2_forensics3.py:183-185`` rather than importing that frozen
    diagnostic module.
    """
    support = demographic_panel.groupby("person_id")["period"].agg(
        ["min", "max"]
    )
    support.columns = ["first_wave", "last_wave"]
    return support


def observed_marriage_residual(
    panel: transitions.MaritalPanel,
) -> dict[int, float]:
    """Return undatable observed marriages not represented by panel events.

    Candidate 16 applies this observed count only after assembling the
    simulated panel. The arithmetic and undefined-count rule preserve
    ``scripts/run_gate2_candidate9.py:189-232``.
    """
    attrs = panel.attrs
    starts = panel.events[
        panel.events["transition"].isin(("first_marriage", "remarriage"))
    ]
    event_count = starts.groupby("person_id").size()
    datable = (
        attrs["person_id"]
        .map(event_count)
        .fillna(0)
        .astype("int64")
        .to_numpy()
    )
    observed = attrs["n_marriages"].to_numpy(dtype="float64")
    residual = observed - datable
    residual = np.where(np.isnan(residual), 0.0, residual)
    return {
        int(person_id): float(value)
        for person_id, value in zip(
            attrs["person_id"].to_numpy(), residual, strict=True
        )
    }


def entry_widowed_cells(
    panel: transitions.MaritalPanel, demographic_panel: pd.DataFrame
) -> EntryWidowedCells:
    """Identify widowed person-years whose onset predates observed support.

    The onset classification, stable sort, and integer key preserve
    ``scripts/run_gate2_candidate12.py:474-526`` exactly.
    """
    support = observed_support(demographic_panel)
    widowed = panel.person_years[
        panel.person_years["marital_state"] == "widowed"
    ].copy()
    years_since = widowed["years_since_dissolution"].astype("float64")
    onset_year = widowed["year"].to_numpy(
        dtype=np.float64
    ) - years_since.to_numpy(dtype=np.float64)
    first_wave = (
        widowed["person_id"]
        .map(support["first_wave"])
        .to_numpy(dtype=np.float64)
    )
    carried = (
        ~np.isnan(onset_year)
        & ~np.isnan(first_wave)
        & (onset_year < first_wave)
    )
    cells = widowed.loc[carried, ["person_id", "year", "weight"]].copy()
    cells["years_since_dissolution"] = years_since.to_numpy(dtype=np.float64)[
        carried
    ].astype(np.int64)
    person_id = cells["person_id"].to_numpy(dtype=np.int64)
    year = cells["year"].to_numpy(dtype=np.int64)
    key = person_id * 10000 + year
    order = np.argsort(key, kind="stable")
    return EntryWidowedCells(
        key_sorted=key[order],
        years_since_dissolution_sorted=cells[
            "years_since_dissolution"
        ].to_numpy(dtype=np.int64)[order],
        person_id_sorted=person_id[order],
        weight_sorted=cells["weight"].to_numpy(dtype=np.float64)[order],
    )


def support_composition(
    panel: transitions.MaritalPanel, demographic_panel: pd.DataFrame
) -> SupportComposition:
    """Build the observed window-reaches-age-75 support stratum.

    ``min(last_wave, censor_year) >= birth_year + 75`` and the missing-support
    fallback preserve ``scripts/run_gate2_candidate16.py:410-474``.
    """
    support = observed_support(demographic_panel)
    attrs = panel.attrs
    person_id = attrs["person_id"].to_numpy(dtype=np.int64)
    birth_year = attrs["birth_year"].to_numpy(dtype=np.float64)
    censor_year = attrs["censor_year"].to_numpy(dtype=np.float64)
    last_wave = (
        attrs["person_id"].map(support["last_wave"]).to_numpy(dtype=np.float64)
    )
    observed_end = np.where(
        np.isnan(last_wave),
        -np.inf,
        np.minimum(last_wave, censor_year),
    )
    stratum = (
        observed_end >= (birth_year + float(SUPPORT_THRESHOLD_AGE))
    ).astype(np.int64)
    return SupportComposition(
        by_person=pd.Series(
            stratum,
            index=pd.Index(person_id, name="person_id"),
        )
    )


def fit_observed_initial_states(
    panel: transitions.MaritalPanel, demographic_panel: pd.DataFrame
) -> ObservedInitialStates:
    """Fit candidate 16's three observed, split-independent state components."""
    return ObservedInitialStates(
        marriage_residual_by_person=observed_marriage_residual(panel),
        entry_widowed=entry_widowed_cells(panel, demographic_panel),
        support=support_composition(panel, demographic_panel),
    )


def apply_entry_widowed(
    simulated_panel: transitions.MaritalPanel,
    carried: EntryWidowedCells,
) -> None:
    """Overwrite reference-carried widowed states on the simulated grid.

    This intentionally changes person-year states only; it does not create
    simulated episodes or events. The search and assignments port
    ``scripts/run_gate2_candidate12.py:1103-1133``.
    """
    key_sorted = carried.key_sorted
    if key_sorted.size == 0:
        return
    person_years = simulated_panel.person_years
    simulated_key = person_years["person_id"].to_numpy(
        dtype=np.int64
    ) * 10000 + person_years["year"].to_numpy(dtype=np.int64)
    position = np.searchsorted(key_sorted, simulated_key)
    position = np.clip(position, 0, key_sorted.size - 1)
    match = key_sorted[position] == simulated_key
    if not match.any():
        return
    state = person_years["marital_state"].to_numpy(dtype=object)
    state[match] = "widowed"
    person_years["marital_state"] = state
    years_since = person_years["years_since_dissolution"].to_numpy(
        dtype=object
    )
    years_since[match] = carried.years_since_dissolution_sorted[position][
        match
    ]
    person_years["years_since_dissolution"] = pd.array(
        years_since, dtype="Int64"
    )
    duration = person_years["marriage_duration"].to_numpy(dtype=object)
    duration[match] = pd.NA
    person_years["marriage_duration"] = pd.array(duration, dtype="Int64")
