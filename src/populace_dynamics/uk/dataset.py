"""Three-table UK panel dataset contract.

The archived policyengine-uk-data#346 machinery operated on
``policyengine_uk.data.UKSingleYearDataset``. This repo does not
depend on policyengine-uk, so the port runs on a minimal structural
equivalent: three DataFrames (person, benunit, household) linked by
ID columns, plus the fiscal year. ``from_policyengine`` adapts a real
``UKSingleYearDataset`` when that package is installed.

ID contract (the #345 panel-ID contract):

- ``person``: ``person_id``, ``person_benunit_id``,
  ``person_household_id``, ``age``, ``gender`` (``"MALE"``/``"FEMALE"``).
- ``benunit``: ``benunit_id``.
- ``household``: ``household_id``.

Person IDs of survivors are preserved across a transition; deaths and
emigration remove IDs, births and immigration add fresh IDs.

:func:`prune_orphaned_entities` is the fix for the first blocker on
#346's review: any operation that removes person rows must not leave
benunit/household rows (with live weights) that no surviving person
references.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "PANEL_ID_COLUMNS",
    "UKPanelDataset",
    "prune_orphaned_entities",
    "classify_panel_ids",
]

#: The panel keys, per table.
PANEL_ID_COLUMNS: dict[str, str] = {
    "person": "person_id",
    "benunit": "benunit_id",
    "household": "household_id",
}

_PERSON_REQUIRED = (
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "age",
    "gender",
)


@dataclass
class UKPanelDataset:
    """A single-year UK population as three linked tables."""

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    fiscal_year: int = 2023

    def __post_init__(self) -> None:
        for col in _PERSON_REQUIRED:
            if col not in self.person.columns:
                raise ValueError(f"person table missing column {col!r}")
        for table in ("benunit", "household"):
            key = PANEL_ID_COLUMNS[table]
            if key not in getattr(self, table).columns:
                raise ValueError(f"{table} table missing column {key!r}")

    def copy(self) -> "UKPanelDataset":
        return UKPanelDataset(
            person=self.person.copy(),
            benunit=self.benunit.copy(),
            household=self.household.copy(),
            fiscal_year=self.fiscal_year,
        )

    def validate_ids(self) -> None:
        """Assert referential integrity of the three tables."""
        for key in ("person_id",):
            if self.person[key].duplicated().any():
                raise ValueError("duplicate person_id values")
        missing_bu = ~self.person["person_benunit_id"].isin(
            self.benunit["benunit_id"]
        )
        if missing_bu.any():
            raise ValueError(
                f"{int(missing_bu.sum())} persons reference a "
                "benunit_id absent from the benunit table"
            )
        missing_hh = ~self.person["person_household_id"].isin(
            self.household["household_id"]
        )
        if missing_hh.any():
            raise ValueError(
                f"{int(missing_hh.sum())} persons reference a "
                "household_id absent from the household table"
            )

    @classmethod
    def from_policyengine(cls, ds) -> "UKPanelDataset":
        """Adapt a ``policyengine_uk.data.UKSingleYearDataset``."""
        return cls(
            person=ds.person.copy(),
            benunit=ds.benunit.copy(),
            household=ds.household.copy(),
            fiscal_year=int(getattr(ds, "fiscal_year", 2023)),
        )


def prune_orphaned_entities(ds: UKPanelDataset) -> UKPanelDataset:
    """Drop benunit/household rows no surviving person references.

    Review fix for #346: mortality (and any other person-removal path)
    previously filtered only the person table, leaving fully-deceased
    benunits and households behind with their weights and inputs.
    Every removal path in this package routes through this helper.
    """
    out = ds.copy()
    live_bu = set(out.person["person_benunit_id"])
    live_hh = set(out.person["person_household_id"])
    out.benunit = out.benunit[
        out.benunit["benunit_id"].isin(live_bu)
    ].reset_index(drop=True)
    out.household = out.household[
        out.household["household_id"].isin(live_hh)
    ].reset_index(drop=True)
    return out


def classify_panel_ids(
    before: UKPanelDataset, after: UKPanelDataset
) -> dict[str, set]:
    """Classify person IDs across a transition.

    Returns ``{"survivors": ..., "removed": ..., "added": ...}`` —
    deaths/emigration land in ``removed``, births/immigration in
    ``added``.
    """
    ids_before = set(before.person["person_id"])
    ids_after = set(after.person["person_id"])
    return {
        "survivors": ids_before & ids_after,
        "removed": ids_before - ids_after,
        "added": ids_after - ids_before,
    }
