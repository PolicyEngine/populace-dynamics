"""Realized-anchor population materialization for the M6 holdout.

The builder mirrors the truth-side anchor and support readers.  It keeps the
2015 bulk in the period-zero slice and schedules 2017/2019 presence openers at
their own anchor interview, so no module evolves a later opener before its
realized initial condition exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from populace_dynamics.data import disability
from populace_dynamics.engine.loop import SCHEDULED_ENTRIES_KEY
from populace_dynamics.engine.panel_builders import PanelBuilderInputs
from populace_dynamics.engine.support import StartWaveWeightSnapshot
from populace_dynamics.harness.m6_cells import (
    EARN_ANCHOR_YEAR,
    SEED_WAVE,
    SEXES,
    build_anchor_frame,
    earnings_frame,
    presence_by_wave,
)

_MARITAL_SEED_COLUMNS = (
    "marital_state",
    "marriage_duration",
    "years_since_dissolution",
)
_HOUSEHOLD_SEED_COLUMNS = (
    "coresident_spouse",
    "coresident_parent",
    "coresident_child",
    "coresident_grandchild",
    "multigen",
    "hh_size",
)
_DISABILITY_SEED_COLUMNS = (
    "status_code",
    "disabled",
    "retired",
)


@dataclass(frozen=True)
class M6RealizedPopulation:
    """All realized support objects needed after the cutoff refit."""

    anchor: pd.DataFrame
    initial_slice: pd.DataFrame
    scheduled_entries_by_year: dict[int, pd.DataFrame]
    holdout_ids: frozenset[int]
    earnings_domain_ids: frozenset[int]
    earnings_support: pd.DataFrame
    presence: dict[int, set[int]]
    start_weights: StartWaveWeightSnapshot
    disability_panel: disability.DisabilityPanel
    disability_ids: frozenset[int]
    panel_builder_inputs: PanelBuilderInputs

    def projection_metadata(self) -> dict[str, object]:
        """Return the population-owned metadata consumed by the engine."""
        return {
            SCHEDULED_ENTRIES_KEY: {
                year: frame.copy()
                for year, frame in self.scheduled_entries_by_year.items()
            },
            "m6_panel_builder_inputs": self.panel_builder_inputs,
        }


def subset_realized_population(
    population: M6RealizedPopulation,
    person_ids: set[object] | frozenset[object],
) -> M6RealizedPopulation:
    """Return one gate-seed side without changing any realized support law."""
    selected = frozenset(int(value) for value in person_ids)
    unknown = selected - population.holdout_ids
    if unknown:
        raise ValueError(
            f"side-A contains unknown persons {sorted(unknown)[:10]}"
        )
    anchor = population.anchor[
        population.anchor["person_id"].isin(selected)
    ].copy()
    builders = PanelBuilderInputs(
        anchor=anchor,
        marital=population.panel_builder_inputs.marital,
        household=population.panel_builder_inputs.household,
        cohabitation=population.panel_builder_inputs.cohabitation,
        projection_end_year=population.panel_builder_inputs.projection_end_year,
    )
    scheduled = {
        year: frame[frame["person_id"].isin(selected)].copy()
        for year, frame in population.scheduled_entries_by_year.items()
    }
    scheduled = {
        year: frame for year, frame in scheduled.items() if len(frame)
    }
    return M6RealizedPopulation(
        anchor=anchor,
        initial_slice=population.initial_slice[
            population.initial_slice["person_id"].isin(selected)
        ].copy(),
        scheduled_entries_by_year=scheduled,
        holdout_ids=selected,
        earnings_domain_ids=population.earnings_domain_ids & selected,
        earnings_support=population.earnings_support[
            population.earnings_support["person_id"].isin(selected)
        ].copy(),
        presence={
            wave: set(ids) & selected
            for wave, ids in population.presence.items()
        },
        start_weights=StartWaveWeightSnapshot.from_frame(
            anchor[["person_id", "weight"]],
            boundary_period=EARN_ANCHOR_YEAR,
        ),
        disability_panel=population.disability_panel,
        disability_ids=population.disability_ids & selected,
        panel_builder_inputs=builders,
    )


def _anchor_rows(
    frame: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    period_column: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Select exactly the row at each person's realized anchor interview."""
    required = {"person_id", period_column, *columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"anchor source is missing columns {sorted(missing)}")
    selected = frame[frame["person_id"].isin(anchor["person_id"])].copy()
    selected = selected.merge(
        anchor[["person_id", "anchor_wave"]],
        on="person_id",
        how="inner",
        validate="many_to_one",
    )
    selected = selected[selected[period_column] == selected["anchor_wave"]]
    if selected["person_id"].duplicated().any():
        raise ValueError("anchor source has duplicate person-wave rows")
    return selected[["person_id", *columns]]


def _person_sex_map(sex_records: pd.DataFrame) -> pd.Series:
    """Return a ``person_id`` -> coded-sex Series from a death-records frame.

    Person sex is ``ER32000`` from the PSID cross-year individual file, read by
    :func:`populace_dynamics.data.deaths.read_death_records`.  The demographic
    panel never carries ``sex`` (§2.8.3f; grading #42 comment 4972045579), so it
    is joined by ``person_id`` from the death records -- the same canonical
    attach the certified builders use
    (:func:`populace_dynamics.data.household_composition.join_demographics`,
    :func:`populace_dynamics.data.disability.attach_sex`).  Only the two coded
    sexes are retained and exactly one coded value per person is required; two
    conflicting coded-sex rows for one person raise.
    """
    missing = {"person_id", "sex"} - set(sex_records.columns)
    if missing:
        raise ValueError(f"sex source is missing columns {sorted(missing)}")
    coded = sex_records.loc[
        sex_records["sex"].isin(SEXES), ["person_id", "sex"]
    ].drop_duplicates()
    if coded["person_id"].duplicated().any():
        conflicting = sorted(
            int(value)
            for value in coded.loc[
                coded["person_id"].duplicated(keep=False), "person_id"
            ].unique()
        )
        raise ValueError(
            "sex source has conflicting coded sex for persons "
            f"{conflicting[:10]}"
        )
    return coded.set_index("person_id")["sex"].astype("string")


def build_realized_population(
    *,
    demographic_panel: pd.DataFrame,
    death_records: pd.DataFrame,
    earnings_panel: pd.DataFrame,
    disability_panel: disability.DisabilityPanel,
    panel_builder_inputs: PanelBuilderInputs,
    earnings_domain_ids: set[int] | frozenset[int],
) -> M6RealizedPopulation:
    """Build the closed-panel seed and support bases fixed by section 2.8.3.

    ``death_records`` is the :func:`populace_dynamics.data.deaths.read_death_records`
    frame; it is the canonical person-constant sex source for the demographic
    seed (§2.8.3f), since the demographic panel carries no ``sex`` column.
    """
    anchor = build_anchor_frame(demographic_panel)
    expected_anchor = panel_builder_inputs.anchor[
        ["person_id", "household_id", "weight", "anchor_wave"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        anchor.reset_index(drop=True),
        expected_anchor,
        check_dtype=True,
        check_like=False,
        obj="panel-builder anchor",
    )
    holdout_ids = frozenset(int(value) for value in anchor["person_id"])
    domain = frozenset(int(value) for value in earnings_domain_ids)
    unknown_domain = domain - holdout_ids
    if unknown_domain:
        raise ValueError(
            "earnings domain contains persons outside the closed panel: "
            f"{sorted(unknown_domain)[:10]}"
        )

    # Person sex is not on the demographic panel; source it canonically from
    # the death-records individual file (§2.8.3f) and join it by person_id.
    # Sex is person-constant, so attaching it to every demo row and selecting
    # the anchor-wave row yields the realized anchor sex.
    sex_by_person = _person_sex_map(death_records)
    demographic_with_sex = demographic_panel.assign(
        sex=demographic_panel["person_id"].map(sex_by_person).astype("string")
    )
    demographic_seed = _anchor_rows(
        demographic_with_sex,
        anchor,
        period_column="period",
        columns=("age", "sex", "interview"),
    )
    if set(demographic_seed["person_id"]) != holdout_ids:
        raise ValueError(
            "demographic panel does not cover every anchor person"
        )
    uncoded = demographic_seed.loc[
        ~demographic_seed["sex"].isin(SEXES), "person_id"
    ]
    if len(uncoded):
        raise ValueError(
            "anchor persons have no coded sex in the death-records "
            f"individual file: {sorted(int(v) for v in uncoded)[:10]}"
        )
    seed = anchor.merge(
        demographic_seed,
        on="person_id",
        how="inner",
        validate="one_to_one",
    )
    if not np.array_equal(
        seed["household_id"].to_numpy(dtype=np.int64),
        seed["interview"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("anchor household_id disagrees with interview number")
    seed = seed.drop(columns="interview")

    marital_seed = _anchor_rows(
        panel_builder_inputs.marital.person_years,
        anchor,
        period_column="year",
        columns=_MARITAL_SEED_COLUMNS,
    )
    seed = seed.merge(
        marital_seed,
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    household_seed = _anchor_rows(
        panel_builder_inputs.household.person_waves,
        anchor,
        period_column="year",
        columns=_HOUSEHOLD_SEED_COLUMNS,
    )
    seed = seed.merge(
        household_seed,
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    cohabitation_seed = _anchor_rows(
        panel_builder_inputs.cohabitation,
        anchor,
        period_column="year",
        columns=("cohabiting",),
    )
    seed = seed.merge(
        cohabitation_seed,
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    # The certified cohabitation overlay is sparse; no exact anchor record is
    # the observed false state, not an unknown carried from a later wave.
    seed["cohabiting"] = seed["cohabiting"].fillna(False).astype(bool)
    disability_seed = _anchor_rows(
        disability_panel.person_years,
        anchor,
        period_column="period",
        columns=_DISABILITY_SEED_COLUMNS,
    )
    seed = seed.merge(
        disability_seed,
        on="person_id",
        how="left",
        validate="one_to_one",
    )

    # The frame coordinate is the reference year immediately before the
    # anchor interview, while age is the realized collection-wave age.  The
    # loop's first mortality step therefore sees the observed anchor age, and
    # its later aging step advances toward the next collection-wave age used
    # by the forward earnings fit.
    seed["year"] = seed["anchor_wave"].astype(np.int64) - 1
    seed["age"] = seed["age"].astype(np.int64)
    seed["earnings_domain"] = seed["person_id"].isin(domain)
    seed["earnings"] = 0.0

    initial = seed[seed["anchor_wave"] == SEED_WAVE].copy()
    initial["year"] = EARN_ANCHOR_YEAR
    initial = initial.sort_values("person_id").reset_index(drop=True)
    scheduled = {
        int(anchor_wave): rows.sort_values("person_id").reset_index(drop=True)
        for anchor_wave, rows in seed[
            seed["anchor_wave"] != SEED_WAVE
        ].groupby("anchor_wave", sort=True)
    }
    if set(scheduled) - {2017, 2019}:
        raise ValueError(
            f"unexpected gated anchor waves {sorted(set(scheduled) - {2017, 2019})}"
        )

    disability_ids = frozenset(
        int(value)
        for value in disability_panel.person_years.loc[
            disability_panel.person_years["person_id"].isin(holdout_ids),
            "person_id",
        ].unique()
    )
    return M6RealizedPopulation(
        anchor=anchor,
        initial_slice=initial,
        scheduled_entries_by_year=scheduled,
        holdout_ids=holdout_ids,
        earnings_domain_ids=domain,
        earnings_support=earnings_frame(earnings_panel, anchor),
        presence=presence_by_wave(demographic_panel),
        start_weights=StartWaveWeightSnapshot.from_frame(
            anchor[["person_id", "weight"]],
            boundary_period=EARN_ANCHOR_YEAR,
        ),
        disability_panel=disability_panel,
        disability_ids=disability_ids,
        panel_builder_inputs=panel_builder_inputs,
    )
