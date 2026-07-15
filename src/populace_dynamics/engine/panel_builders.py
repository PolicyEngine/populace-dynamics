"""Realized-anchor native-panel builders for the M6 scored harness.

The gated M6 reproduction test starts each PSID person from their realized
start-of-holdout interview.  These adapters expose that initial condition in
the two certified native schemas without consulting the mortality-attrited
projection slice.  Candidate 16 and candidate 9 then evolve the state over the
whole registered window.

Callers place one :class:`PanelBuilderInputs` object in ``PeriodContext``
metadata under :data:`PANEL_BUILDER_INPUTS_KEY`.  The two public functions
therefore have exactly the ``(frame, context) -> (panel, holdout_ids)`` shape
required by :mod:`populace_dynamics.engine.assembly`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.engine.loop import PeriodContext

PANEL_BUILDER_INPUTS_KEY = "m6_panel_builder_inputs"
PROJECTION_END_YEAR = 2022

_ANCHOR_COLUMNS = ("person_id", "anchor_wave", "weight")
_HOUSEHOLD_SUPPORT_COLUMNS = (
    "person_id",
    "year",
    "age",
    "band",
    "sex",
    "weight",
)
_HOUSEHOLD_STATE_COLUMNS = (
    "hh_size",
    *hc.CORESIDENCE_LINKS,
    "multigen",
)
_HOUSEHOLD_TRANSITION_COLUMNS = (
    "has_next",
    "next_coresident_parent",
    "next_coresident_spouse",
    "next_multigen",
)

__all__ = [
    "PANEL_BUILDER_INPUTS_KEY",
    "PROJECTION_END_YEAR",
    "PanelBuilderInputs",
    "household_panel_builder",
    "marital_panel_builder",
]


@dataclass(frozen=True)
class PanelBuilderInputs:
    """Certified realized panels and per-person M6 anchor coordinates.

    ``marital`` is the output of :func:`transitions.build_marital_panel`
    built with the anchor-weight series.  ``household`` is the output of
    :func:`hc.build_household_panel`.  ``cohabitation`` is the sparse
    ``person_id, year, cohabiting`` frame produced from the same realized
    roster by candidate 9's certified ``cohabitation_flag`` helper.

    The classmethod is the production construction path for the marital
    input.  Direct construction remains useful for fully synthetic tests that
    already hold schema-compatible certified panels.
    """

    anchor: pd.DataFrame
    marital: transitions.MaritalPanel
    household: hc.HouseholdCompositionPanel
    cohabitation: pd.DataFrame
    projection_end_year: int = PROJECTION_END_YEAR

    @classmethod
    def from_realized_histories(
        cls,
        *,
        anchor: pd.DataFrame,
        marriage_records: pd.DataFrame,
        death_records: pd.DataFrame,
        household: hc.HouseholdCompositionPanel,
        cohabitation: pd.DataFrame,
        projection_end_year: int = PROJECTION_END_YEAR,
    ) -> PanelBuilderInputs:
        """Build the marital source through the certified truth-side reader."""
        _validate_anchor(anchor)
        weights = anchor.set_index("person_id")["weight"]
        marital = transitions.build_marital_panel(
            marriage_records,
            death_records,
            weights,
        )
        return cls(
            anchor=anchor.copy(),
            marital=marital,
            household=household,
            cohabitation=cohabitation.copy(),
            projection_end_year=int(projection_end_year),
        )


def _validate_anchor(anchor: pd.DataFrame) -> None:
    missing = set(_ANCHOR_COLUMNS) - set(anchor.columns)
    if missing:
        raise ValueError(f"anchor frame is missing columns {sorted(missing)}")
    if anchor["person_id"].duplicated().any():
        raise ValueError("anchor frame has duplicate person_id rows")
    if anchor["anchor_wave"].isna().any():
        raise ValueError("anchor frame has missing anchor_wave values")
    if (anchor["weight"] <= 0).any() or anchor["weight"].isna().any():
        raise ValueError("anchor frame weights must be positive")


def _builder_inputs(context: PeriodContext) -> PanelBuilderInputs:
    value = context.metadata.get(PANEL_BUILDER_INPUTS_KEY)
    if not isinstance(value, PanelBuilderInputs):
        raise TypeError(
            f"context metadata {PANEL_BUILDER_INPUTS_KEY!r} must be a "
            "PanelBuilderInputs"
        )
    _validate_anchor(value.anchor)
    if int(value.projection_end_year) != PROJECTION_END_YEAR:
        raise ValueError(
            "M6 scored panel builders require the pinned 2022 horizon"
        )
    return value


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an empty frame preserving every certified column dtype."""
    return frame.iloc[0:0].copy()


def _assign_compatible(
    frame: pd.DataFrame, column: str, values: pd.Series
) -> None:
    """Assign values using the certified source column's exact dtype."""
    dtype = frame[column].dtype
    frame[column] = pd.Series(values.to_numpy(), index=frame.index).astype(
        dtype
    )


def marital_panel_builder(
    frame: pd.DataFrame,
    context: PeriodContext,
) -> tuple[transitions.MaritalPanel, set[int]]:
    """Build candidate 16's whole-window realized-anchor seed panel.

    The current projection frame is deliberately not a population filter:
    gated marital flows sit on realized presence support.  A person is kept
    only when they are in both the floor's anchor universe and the certified
    marriage-history universe, and their anchor interview is within their
    realized report/death censor.  Candidate 16 receives exactly one entry
    row per person, so its existing state machine carries the realized
    marriage start or dissolution year into simulated episodes.
    """
    del frame
    inputs = _builder_inputs(context)
    source = inputs.marital
    anchor = inputs.anchor[[*_ANCHOR_COLUMNS]].copy()

    attrs = source.attrs.merge(
        anchor,
        on="person_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_anchor"),
        sort=False,
    )
    if "weight_anchor" not in attrs:
        raise ValueError("certified marital attrs have no weight column")

    clipped_censor = np.minimum(
        attrs["censor_year"].to_numpy(dtype=np.float64),
        float(inputs.projection_end_year),
    )
    attrs["censor_year"] = clipped_censor.astype(
        source.attrs["censor_year"].dtype
    )
    # Amendment 3g (§2.8.2g): clamp the projected seed wave to the certified
    # marital risk-set entry.  For the bulk (anchor_wave >= birth+START_AGE)
    # this is anchor_wave, byte-unchanged; for the sub-START_AGE-at-anchor
    # class (anchor_wave < birth+START_AGE) it is birth_year + START_AGE, the
    # certified person_years entry the read below (:200-205) finds by
    # construction.  Reads only existing attrs columns; the truth side is
    # untouched, so the frozen floors stay byte-identical.
    clamped_start = np.maximum(
        attrs["anchor_wave"].to_numpy(dtype=np.float64),
        attrs["birth_year"].to_numpy(dtype=np.float64) + transitions.START_AGE,
    )
    attrs["start_exposure_year"] = clamped_start.astype(
        source.attrs["start_exposure_year"].dtype
    )
    attrs["weight"] = attrs["weight_anchor"].astype(
        source.attrs["weight"].dtype
    )
    attrs = attrs[attrs["start_exposure_year"] <= attrs["censor_year"]].copy()

    # Restore the reader's exact attrs schema and column order.  In
    # particular, lifetime MH18 n_marriages is carried unchanged and inert.
    attrs = attrs[list(source.attrs.columns)].reset_index(drop=True)
    valid_ids = set(attrs["person_id"].tolist())

    anchor_wave = attrs.set_index("person_id")["start_exposure_year"]
    entry = source.person_years[
        source.person_years["person_id"].isin(valid_ids)
    ].copy()
    expected_wave = entry["person_id"].map(anchor_wave)
    entry = entry[entry["year"] == expected_wave].copy()
    if entry["person_id"].duplicated().any():
        raise ValueError("certified marital panel has duplicate anchor rows")
    found_ids = set(entry["person_id"].tolist())
    if found_ids != valid_ids:
        missing = sorted(valid_ids - found_ids)
        raise ValueError(
            "certified marital panel has no entry row at anchor for "
            f"person_ids {missing[:10]}"
        )
    entry = entry.sort_values(["person_id", "year"]).reset_index(drop=True)

    # The certified reader was built with the same anchor-weight series.  Set
    # the entry weight explicitly so F6 remains fixed even for injected
    # synthetic panels, while retaining its byte-compatible dtype.
    entry_weights = entry["person_id"].map(
        attrs.set_index("person_id")["weight"]
    )
    _assign_compatible(entry, "weight", entry_weights)

    panel = transitions.MaritalPanel(
        person_years=entry,
        events=_empty_like(source.events),
        attrs=attrs,
    )
    return panel, valid_ids


def _household_seed_rows(
    source: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    """Return each person's last observed row no later than their anchor."""
    anchor_wave = anchor.set_index("person_id")["anchor_wave"]
    candidates = source[source["person_id"].isin(anchor_wave.index)].copy()
    candidates["_anchor_wave"] = candidates["person_id"].map(anchor_wave)
    candidates = candidates[candidates["year"] <= candidates["_anchor_wave"]]
    candidates = candidates.sort_values(["person_id", "year"])
    return candidates.groupby("person_id", as_index=False, sort=False).tail(1)


def household_panel_builder(
    frame: pd.DataFrame,
    context: PeriodContext,
) -> tuple[hc.HouseholdCompositionPanel, set[int]]:
    """Build candidate 9's whole-window realized-support seed panel.

    Support coordinates come from the certified realized household panel.
    Household state is read only through each person's anchor interview and
    copied over later support rows as inert input: candidate 9 reads the first
    row as ``obs_*`` and replaces every projected row.  The copy makes the
    leakage fence explicit and prevents realized post-anchor states from
    entering the candidate.  Transition helper columns are always rebuilt
    after the per-person slice.
    """
    del frame
    inputs = _builder_inputs(context)
    source = inputs.household.person_waves
    anchor = inputs.anchor[[*_ANCHOR_COLUMNS]].copy()

    required = set(_HOUSEHOLD_SUPPORT_COLUMNS) | set(_HOUSEHOLD_STATE_COLUMNS)
    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            "certified household panel is missing columns "
            f"{sorted(missing)}"
        )
    cohab_required = {"person_id", "year", "cohabiting"}
    cohab_missing = cohab_required - set(inputs.cohabitation.columns)
    if cohab_missing:
        raise ValueError(
            "cohabitation seed frame is missing columns "
            f"{sorted(cohab_missing)}"
        )
    if inputs.cohabitation.duplicated(["person_id", "year"]).any():
        raise ValueError("cohabitation seed frame has duplicate person-years")

    seed = _household_seed_rows(source, anchor)
    if seed["person_id"].duplicated().any():
        raise ValueError("certified household panel has duplicate seed rows")
    seed = seed.merge(
        anchor[["person_id", "anchor_wave"]],
        on="person_id",
        how="inner",
        validate="one_to_one",
    )
    # A household initial condition must be observed at the actual anchor
    # interview, not carried from an earlier wave.
    seed = seed[seed["year"] == seed["anchor_wave"]].copy()
    seed_ids = set(seed["person_id"].tolist())

    support = source[list(_HOUSEHOLD_SUPPORT_COLUMNS)].copy()
    support = support[support["person_id"].isin(seed_ids)]
    anchor_wave = anchor.set_index("person_id")["anchor_wave"]
    support_anchor = support["person_id"].map(anchor_wave)
    support = support[
        (support["year"] >= support_anchor)
        & (support["year"] <= inputs.projection_end_year)
    ].copy()

    state = seed[["person_id", *_HOUSEHOLD_STATE_COLUMNS]].copy()
    person_waves = support.merge(
        state,
        on="person_id",
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    fixed_weight = anchor.set_index("person_id")["weight"]
    person_waves["weight"] = (
        person_waves["person_id"]
        .map(fixed_weight)
        .astype(source["weight"].dtype)
    )

    # The cohabitation flag is sparse: absence at the exact anchor means
    # false.  As with the other obs_* states, carry only the anchor value.
    cohab_seed = seed[["person_id", "year"]].merge(
        inputs.cohabitation[["person_id", "year", "cohabiting"]],
        on=["person_id", "year"],
        how="left",
        validate="one_to_one",
    )
    cohab_seed["cohabiting"] = (
        cohab_seed["cohabiting"].fillna(False).astype(bool)
    )
    person_waves = person_waves.merge(
        cohab_seed[["person_id", "cohabiting"]],
        on="person_id",
        how="left",
        validate="many_to_one",
    )

    base_columns = [
        column
        for column in source.columns
        if column not in _HOUSEHOLD_TRANSITION_COLUMNS
    ]
    person_waves = person_waves[[*base_columns, "cohabiting"]]
    person_waves = hc._add_transitions(person_waves)
    # Keep every certified reader column in byte-compatible order, then the
    # injected seed consumed by the M6 candidate-9 adapter.
    person_waves = person_waves[[*source.columns, "cohabiting"]].reset_index(
        drop=True
    )

    holdout_ids = set(person_waves["person_id"].tolist())
    attrs = inputs.household.attrs[
        inputs.household.attrs["person_id"].isin(holdout_ids)
    ].copy()
    attrs = attrs.reset_index(drop=True)
    return hc.HouseholdCompositionPanel(person_waves, attrs), holdout_ids
