#!/usr/bin/env python3
"""Train-only M6 remarriage transport-delta investigation.

This is an analysis helper, not a gate runner. It deliberately does not import
the M6 scoring layer, read ``gates.yaml``, or write an artifact. Its only output
is a JSON document on stdout; progress goes to stderr.

The candidate set, pseudo-boundaries, seeds, and selector are frozen in
``docs/analysis/m6_remarriage_train_only_delta.md``. All candidate inputs are
sanitized to an information boundary no later than 2014 before fitting. A
calendar-2014 marital row is excluded because its establishing interview is in
2015.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from populace_dynamics.data import (
    births,
    deaths,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.engine import refit
from populace_dynamics.engine.marital import (
    _simulate_candidate16_with_generators,
)
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.models.family_transitions import registry as ft_registry
from populace_dynamics.models.family_transitions.common import (
    band_indices,
    marriage_order_map,
)
from populace_dynamics.models.family_transitions.components import (
    remarriage as rem,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)

MAX_INFORMATION_YEAR = 2014
PSEUDO_BOUNDARIES = (2006, 2008, 2010)
DRAW_SEEDS = tuple(range(7200, 7240))
DRAW_BLOCKS = (tuple(range(7200, 7220)), tuple(range(7220, 7240)))
LAWS = ("L0", "L1", "L2", "L3")
NONZERO_COMPLEXITY = ("L1", "L2", "L3")
TARGET_AGE_LO = 18
TARGET_AGE_HI = 64
MIN_PARENT_EVENTS = 20
FLOAT_TOL = 1e-12


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if pd.isna(value):
        return None
    return value


def _frame_checksum(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(list(columns), kind="stable").reset_index(
        drop=True
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _max_year(frame: pd.DataFrame, column: str) -> int | None:
    if frame.empty or column not in frame:
        return None
    years = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(years.max()) if len(years) else None


def _assert_at_most(
    frame: pd.DataFrame, column: str, boundary: int, label: str
) -> None:
    maximum = _max_year(frame, column)
    if maximum is not None and maximum > boundary:
        raise AssertionError(
            f"{label}.{column} reaches {maximum}, beyond {boundary}"
        )


def _sanitize_deaths(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("death_year", "death_year_lo", "death_year_hi"):
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        out.loc[values > MAX_INFORMATION_YEAR, column] = pd.NA
        if str(frame[column].dtype) == "Int64":
            out[column] = out[column].astype("Int64")
    _assert_at_most(out, "death_year", MAX_INFORMATION_YEAR, "deaths")
    return out


def _load_sanitized_context() -> tuple[ft_registry.FitContext, dict[str, Any]]:
    """Load staged sources, then sever every post-2014 candidate input."""
    _progress("loading staged PSID sources (read-only)")
    raw_demo = panels.demographic_panel()
    raw_marriages = marriage.marriage_history()
    raw_births = births.birth_history()
    raw_deaths = deaths.read_death_records()

    # PSID is biennial in this era. The last pre-boundary interview is 2013;
    # no 2015 column is allowed to establish a calendar-2014 flow.
    demo = raw_demo.loc[
        pd.to_numeric(raw_demo["period"], errors="coerce")
        < MAX_INFORMATION_YEAR
    ].copy()
    marriage_records = refit._truncate_marriage_records(
        raw_marriages, MAX_INFORMATION_YEAR
    )
    birth_records = refit._truncate_birth_records(
        raw_births, MAX_INFORMATION_YEAR
    )
    death_records = _sanitize_deaths(raw_deaths)

    _assert_at_most(demo, "period", 2013, "demographic")
    actual_marriages = marriage_records[
        marriage_records["is_marriage"].fillna(False)
    ]
    _assert_at_most(
        actual_marriages,
        "start_year",
        MAX_INFORMATION_YEAR,
        "marriage_records",
    )
    _assert_at_most(
        actual_marriages,
        "most_recent_report_year",
        MAX_INFORMATION_YEAR,
        "marriage_records",
    )
    actual_births = birth_records[birth_records["is_event"].fillna(False)]
    _assert_at_most(
        actual_births, "birth_year", MAX_INFORMATION_YEAR, "birth_records"
    )
    _assert_at_most(
        actual_births,
        "most_recent_child_report_year",
        MAX_INFORMATION_YEAR,
        "birth_records",
    )

    positive = demo[demo["weight"] > 0]
    person_weight = (
        positive.sort_values(["person_id", "period"])
        .groupby("person_id", sort=False)
        .tail(1)
        .set_index("person_id")["weight"]
        .astype("float64")
    )
    marital_panel = transitions.build_marital_panel(
        marriage_records, death_records, person_weight
    )
    order_map = marriage_order_map(marriage_records)
    _assert_at_most(
        marital_panel.person_years,
        "year",
        MAX_INFORMATION_YEAR,
        "marital_person_years",
    )
    _assert_at_most(
        marital_panel.events,
        "year",
        MAX_INFORMATION_YEAR,
        "marital_events",
    )

    valid_ids = frozenset(
        set(int(value) for value in positive["person_id"].unique())
        & set(
            int(value) for value in marital_panel.attrs["person_id"].unique()
        )
    )
    context = ft_registry.FitContext(
        panel=marital_panel,
        demographic_panel=demo,
        marriage_records=marriage_records,
        birth_records=birth_records,
        marriage_order_map=order_map,
        train_ids=valid_ids,
    )
    audit = {
        "raw_source_is_retrospective_product": True,
        "candidate_input_max_information_year": MAX_INFORMATION_YEAR,
        "calendar_2014_flow_excluded": True,
        "raw_rows": {
            "demographic": len(raw_demo),
            "marriage_records": len(raw_marriages),
            "birth_records": len(raw_births),
            "deaths": len(raw_deaths),
        },
        "sanitized_rows": {
            "demographic": len(demo),
            "marriage_records": len(marriage_records),
            "actual_marriages": len(actual_marriages),
            "birth_records": len(birth_records),
            "actual_births": len(actual_births),
            "deaths": len(death_records),
            "marital_person_years": len(marital_panel.person_years),
            "marital_events": len(marital_panel.events),
        },
        "sanitized_max_year": {
            "demographic_period": _max_year(demo, "period"),
            "actual_marriage_start": _max_year(actual_marriages, "start_year"),
            "actual_marriage_report": _max_year(
                actual_marriages, "most_recent_report_year"
            ),
            "actual_birth": _max_year(actual_births, "birth_year"),
            "actual_birth_report": _max_year(
                actual_births, "most_recent_child_report_year"
            ),
            "death": _max_year(death_records, "death_year"),
            "marital_person_year": _max_year(
                marital_panel.person_years, "year"
            ),
            "marital_event": _max_year(marital_panel.events, "year"),
        },
        "checksums": {
            "demographic": _frame_checksum(
                demo, ("person_id", "period", "weight", "interview")
            ),
            "marriage_records": _frame_checksum(
                marriage_records,
                (
                    "person_id",
                    "start_year",
                    "end_year",
                    "how_ended",
                    "most_recent_report_year",
                    "is_marriage",
                ),
            ),
            "marital_person_years": _frame_checksum(
                marital_panel.person_years,
                ("person_id", "year", "marital_state", "weight"),
            ),
            "marital_events": _frame_checksum(
                marital_panel.events,
                ("person_id", "year", "transition", "weight"),
            ),
        },
    }
    return context, audit


def _remarriage_frames(
    context: ft_registry.FitContext,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    train_ids = set(context.train_ids)
    person_years = context.panel.person_years[
        context.panel.person_years["person_id"].isin(train_ids)
    ]
    events = context.panel.events[
        context.panel.events["person_id"].isin(train_ids)
    ]
    dissolved = person_years[
        person_years["marital_state"].isin(("divorced", "widowed"))
        & person_years["years_since_dissolution"].notna()
    ].copy()
    remarriages = events[
        (events["transition"] == "remarriage")
        & events["years_since_dissolution"].notna()
    ].copy()
    mean_weight = float(dissolved["weight"].mean()) if len(dissolved) else 1.0
    for frame in (dissolved, remarriages):
        frame["ysd_band"] = band_indices(
            frame["years_since_dissolution"].astype("int64").to_numpy(),
            rem.YSD_LOWERS,
            len(rem.YSD_BANDS),
        )
        frame["age_band"] = band_indices(
            np.rint(frame["age"].to_numpy()).astype(np.int64),
            rem.AGE_LOWERS,
            len(rem.AGE_BANDS),
        )
    return dissolved, remarriages, mean_weight


def _solve_recent_delta(
    recent: pd.DataFrame, base_table: dict[tuple[int, int, str, str], float]
) -> float:
    keys = list(
        zip(
            recent["age_band"],
            recent["ysd_band"],
            recent["marital_state"],
            recent["sex"],
            strict=True,
        )
    )
    base = np.array([base_table[key] for key in keys], dtype=np.float64)
    base = np.clip(base, np.finfo(float).eps, 1 - np.finfo(float).eps)
    y = recent["event"].to_numpy(dtype=np.float64)
    weight = recent["weight"].to_numpy(dtype=np.float64)
    offset = logit(base)

    def equation(delta: float) -> float:
        return float(np.sum(weight * (y - expit(offset + delta))))

    lower = equation(-40.0)
    upper = equation(40.0)
    if not (lower >= 0 and upper <= 0):
        raise ValueError("recent log-odds intercept has no finite bracket")
    return float(brentq(equation, -40.0, 40.0, xtol=1e-13, rtol=1e-14))


def _candidate_tables(
    context: ft_registry.FitContext,
    boundary: int,
    incumbent: dict[tuple[int, int, str, str], float] | None = None,
) -> tuple[dict[str, dict[Any, float] | None], dict[str, Any]]:
    dissolved, remarriages, mean_weight = _remarriage_frames(context)
    if dissolved.empty:
        raise ValueError(f"boundary {boundary} has no dissolved fit rows")

    denominator = dissolved.groupby(
        ["age_band", "ysd_band", "marital_state", "sex"]
    )["weight"].sum()
    numerator = remarriages.groupby(["age_band", "ysd_band", "origin", "sex"])[
        "weight"
    ].sum()
    row_count = dissolved.groupby(
        ["age_band", "ysd_band", "marital_state", "sex"]
    ).size()
    event_count = remarriages.groupby(
        ["age_band", "ysd_band", "origin", "sex"]
    ).size()

    total_exposure = float(dissolved["weight"].sum())
    total_numerator = float(remarriages["weight"].sum())
    global_mu = total_numerator / total_exposure

    parent_exposure = dissolved.groupby(["marital_state", "sex"])[
        "weight"
    ].sum()
    parent_numerator = remarriages.groupby(["origin", "sex"])["weight"].sum()
    parent_event_count = remarriages.groupby(["origin", "sex"]).size()
    parent_mu: dict[tuple[str, str], float] = {}
    parent_events: dict[tuple[str, str], int] = {}
    l2_eligible = True
    for origin in ("divorced", "widowed"):
        for sex in ("female", "male"):
            key = (origin, sex)
            exposure = float(parent_exposure.get(key, 0.0))
            weighted_events = float(parent_numerator.get(key, 0.0))
            events = int(parent_event_count.get(key, 0))
            parent_events[key] = events
            if exposure <= 0 or events < MIN_PARENT_EVENTS:
                l2_eligible = False
            parent_mu[key] = (
                weighted_events / exposure if exposure > 0 else float("nan")
            )

    tables: dict[str, dict[Any, float] | None] = {law: {} for law in LAWS}
    cell_rows: list[dict[str, Any]] = []
    for age_band in range(len(rem.AGE_BANDS)):
        for ysd_band in range(len(rem.YSD_BANDS)):
            for origin in ("divorced", "widowed"):
                for sex in ("female", "male"):
                    key = (age_band, ysd_band, origin, sex)
                    weighted_exposure = float(denominator.get(key, 0.0))
                    weighted_events = float(numerator.get(key, 0.0))
                    count_rows = int(row_count.get(key, 0))
                    count_events = int(event_count.get(key, 0))
                    h0 = (weighted_events + mean_weight) / (
                        weighted_exposure + 2.0 * mean_weight
                    )
                    h1 = (weighted_events + 2.0 * mean_weight * global_mu) / (
                        weighted_exposure + 2.0 * mean_weight
                    )
                    h2 = (
                        weighted_events
                        + 2.0 * mean_weight * parent_mu[(origin, sex)]
                    ) / (weighted_exposure + 2.0 * mean_weight)
                    assert isinstance(tables["L0"], dict)
                    assert isinstance(tables["L1"], dict)
                    assert isinstance(tables["L2"], dict)
                    tables["L0"][key] = h0
                    tables["L1"][key] = h1
                    tables["L2"][key] = h2
                    cell_rows.append(
                        {
                            "age_band": rem.AGE_BANDS[age_band],
                            "ysd_band": rem.YSD_BANDS[ysd_band],
                            "origin": origin,
                            "sex": sex,
                            "row_count": count_rows,
                            "event_count": count_events,
                            "weighted_exposure": weighted_exposure,
                            "weighted_events": weighted_events,
                            "effective_exposure_mean_weights": (
                                weighted_exposure / mean_weight
                            ),
                            "empty": count_rows == 0,
                            "thin_lt20_mean_weights": (
                                weighted_exposure / mean_weight < 20
                            ),
                            "L0": h0,
                            "L1": h1,
                            "L2": h2,
                        }
                    )

    if incumbent is not None:
        difference = max(
            abs(float(tables["L0"][key]) - float(incumbent[key]))
            for key in incumbent
        )
        if difference != 0.0:
            raise AssertionError(
                f"L0 failed incumbent bit-equivalence: max diff {difference}"
            )

    event_keys = {
        (int(person_id), int(year))
        for person_id, year in zip(
            remarriages["person_id"], remarriages["year"], strict=True
        )
    }
    if len(event_keys) != len(remarriages):
        raise ValueError("duplicate fit remarriage person-year events")
    recent = dissolved[
        dissolved["year"].between(boundary - 3, boundary)
    ].copy()
    recent["event"] = [
        int((int(person_id), int(year)) in event_keys)
        for person_id, year in zip(
            recent["person_id"], recent["year"], strict=True
        )
    ]
    recent_events = int(recent["event"].sum())
    l3_eligible = (
        recent_events >= MIN_PARENT_EVENTS
        and float(recent["weight"].sum()) > 0
    )
    delta: float | None = None
    if l3_eligible:
        assert isinstance(tables["L1"], dict)
        delta = _solve_recent_delta(recent, tables["L1"])
        tables["L3"] = {
            key: float(expit(logit(value) + delta))
            for key, value in tables["L1"].items()
        }
        for row in cell_rows:
            key = (
                rem.AGE_BANDS.index(tuple(row["age_band"])),
                rem.YSD_BANDS.index(tuple(row["ysd_band"])),
                row["origin"],
                row["sex"],
            )
            row["L3"] = tables["L3"][key]
    else:
        tables["L3"] = None
        for row in cell_rows:
            row["L3"] = None

    diagnostics = {
        "boundary": boundary,
        "fit_person_year_rows": len(context.panel.person_years),
        "fit_event_rows": len(context.panel.events),
        "dissolved_rows": len(dissolved),
        "remarriage_events": len(remarriages),
        "mean_weight": mean_weight,
        "weighted_dissolved_exposure": total_exposure,
        "weighted_remarriage_events": total_numerator,
        "global_training_hazard": global_mu,
        "parent_hazards": {
            f"{origin}|{sex}": parent_mu[(origin, sex)]
            for origin, sex in parent_mu
        },
        "parent_event_counts": {
            f"{origin}|{sex}": parent_events[(origin, sex)]
            for origin, sex in parent_events
        },
        "L2_parameter_eligible": l2_eligible,
        "recent_effective_years": sorted(
            int(value) for value in recent["year"].unique()
        ),
        "recent_rows": len(recent),
        "recent_events": recent_events,
        "L3_parameter_eligible": l3_eligible,
        "L3_delta": delta,
        "empty_cells": int(sum(row["empty"] for row in cell_rows)),
        "thin_cells_lt20_mean_weights": int(
            sum(row["thin_lt20_mean_weights"] for row in cell_rows)
        ),
        "fit_max_year": {
            "person_year": _max_year(context.panel.person_years, "year"),
            "event": _max_year(context.panel.events, "year"),
            "demographic_interview": _max_year(
                context.demographic_panel, "period"
            ),
        },
        "checksums": {
            "dissolved": _frame_checksum(
                dissolved,
                (
                    "person_id",
                    "year",
                    "marital_state",
                    "years_since_dissolution",
                    "weight",
                ),
            ),
            "remarriages": _frame_checksum(
                remarriages,
                ("person_id", "year", "origin", "weight"),
            ),
        },
        "cells": cell_rows,
    }
    return tables, diagnostics


def _pseudo_calendar(boundary: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    anchor_waves = (boundary + 1, boundary + 3)
    candidate_years = tuple(range(boundary + 1, boundary + 5))
    # Annual year y is established by the first odd interview >=y.
    evaluation_years = tuple(
        year
        for year in candidate_years
        if (year if year % 2 else year + 1) <= MAX_INFORMATION_YEAR
    )
    if not evaluation_years:
        raise ValueError(
            f"boundary {boundary} has no admissible pseudo future"
        )
    return anchor_waves, evaluation_years


def _pseudo_anchor(
    demo: pd.DataFrame, anchor_waves: tuple[int, ...]
) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    present: dict[int, set[int]] = {}
    for wave in anchor_waves:
        rows = demo[(demo["period"] == wave) & (demo["weight"] > 0)]
        present[wave] = set(int(value) for value in rows["person_id"])
    rows = demo[
        demo["period"].isin(anchor_waves) & (demo["weight"] > 0)
    ].copy()
    rows = rows.sort_values(["person_id", "period"])
    first = rows.groupby("person_id", as_index=False, sort=False).first()
    anchor = pd.DataFrame(
        {
            "person_id": first["person_id"].astype("int64").to_numpy(),
            "household_id": first["interview"].astype("int64").to_numpy(),
            "weight": first["weight"].astype("float64").to_numpy(),
            "anchor_wave": first["period"].astype("int64").to_numpy(),
        }
    )
    return anchor, present


def _seed_panel(
    source: transitions.MaritalPanel,
    anchor: pd.DataFrame,
    horizon: int,
) -> tuple[transitions.MaritalPanel, set[int], int]:
    attrs = source.attrs.merge(
        anchor,
        on="person_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_anchor"),
        sort=False,
    )
    attrs["censor_year"] = np.minimum(
        attrs["censor_year"].to_numpy(dtype=np.float64), float(horizon)
    ).astype(source.attrs["censor_year"].dtype)
    attrs["start_exposure_year"] = np.maximum(
        attrs["anchor_wave"].to_numpy(dtype=np.float64),
        attrs["birth_year"].to_numpy(dtype=np.float64) + transitions.START_AGE,
    ).astype(source.attrs["start_exposure_year"].dtype)
    attrs["weight"] = attrs["weight_anchor"].astype(
        source.attrs["weight"].dtype
    )
    attrs = attrs[attrs["start_exposure_year"] <= attrs["censor_year"]].copy()
    attrs = attrs[list(source.attrs.columns)].reset_index(drop=True)
    valid_ids = set(int(value) for value in attrs["person_id"])

    start_by_person = attrs.set_index("person_id")["start_exposure_year"]
    entry = source.person_years[
        source.person_years["person_id"].isin(valid_ids)
    ].copy()
    entry = entry[
        entry["year"] == entry["person_id"].map(start_by_person)
    ].copy()
    if entry["person_id"].duplicated().any():
        raise ValueError("pseudo seed has duplicate entry rows")
    found = set(int(value) for value in entry["person_id"])
    if found != valid_ids:
        missing = sorted(valid_ids - found)
        raise ValueError(f"pseudo seed missing entry rows {missing[:10]}")
    entry = entry.sort_values(["person_id", "year"]).reset_index(drop=True)
    entry["weight"] = entry["person_id"].map(
        attrs.set_index("person_id")["weight"]
    )
    entry_dissolved = int(
        entry["marital_state"].isin(("divorced", "widowed")).sum()
    )
    panel = transitions.MaritalPanel(
        person_years=entry,
        events=source.events.iloc[0:0].copy(),
        attrs=attrs,
    )
    return panel, valid_ids, entry_dissolved


def _opening_wave(year: int) -> int:
    return year if year % 2 else year - 1


def _weighted_support(
    panel: transitions.MaritalPanel,
    anchor: pd.DataFrame,
    present: dict[int, set[int]],
    years: tuple[int, ...],
    valid_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_year_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    anchor_weight = anchor.set_index("person_id")["weight"]
    for year in years:
        wave = _opening_wave(year)
        year_ids = present.get(wave, set()) & valid_ids
        py = panel.person_years[
            (panel.person_years["year"] == year)
            & panel.person_years["person_id"].isin(year_ids)
            & panel.person_years["age"].between(TARGET_AGE_LO, TARGET_AGE_HI)
            & panel.person_years["sex"].isin(("female", "male"))
        ].copy()
        ev = panel.events[
            (panel.events["year"] == year)
            & panel.events["person_id"].isin(year_ids)
            & panel.events["age"].between(TARGET_AGE_LO, TARGET_AGE_HI)
            & panel.events["sex"].isin(("female", "male"))
        ].copy()
        py["weight"] = py["person_id"].map(anchor_weight).astype("float64")
        ev["weight"] = ev["person_id"].map(anchor_weight).astype("float64")
        person_year_parts.append(py)
        event_parts.append(ev)
    person_years = pd.concat(person_year_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    if person_years.duplicated(["person_id", "year"]).any():
        raise ValueError("duplicate pseudo-support person-year")
    return events, person_years


def _rate_summary(
    events: pd.DataFrame, person_years: pd.DataFrame
) -> dict[str, Any]:
    risk = person_years[
        person_years["marital_state"].isin(("divorced", "widowed"))
    ]
    remarriages = events[events["transition"] == "remarriage"]
    exposure = float(risk["weight"].sum())
    numerator = float(remarriages["weight"].sum())
    if exposure <= 0 or numerator <= 0:
        raise ValueError("pseudo-holdout remarriage rate is undefined")
    return {
        "risk_rows": len(risk),
        "event_rows": len(remarriages),
        "exposure": exposure,
        "numerator": numerator,
        "rate": numerator / exposure,
    }


def _origin_summary(
    events: pd.DataFrame, person_years: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for origin in ("divorced", "widowed"):
        risk = person_years[person_years["marital_state"] == origin]
        remarriages = events[
            (events["transition"] == "remarriage")
            & (events["origin"] == origin)
        ]
        exposure = float(risk["weight"].sum())
        numerator = float(remarriages["weight"].sum())
        out[origin] = {
            "risk_rows": len(risk),
            "event_rows": len(remarriages),
            "exposure": exposure,
            "numerator": numerator,
            "rate": numerator / exposure if exposure > 0 else None,
        }
    return out


def _direct_standardization(
    table: dict[tuple[int, int, str, str], float],
    truth_events: pd.DataFrame,
    truth_person_years: pd.DataFrame,
) -> dict[str, Any]:
    risk = truth_person_years[
        truth_person_years["marital_state"].isin(("divorced", "widowed"))
        & truth_person_years["years_since_dissolution"].notna()
    ].copy()
    risk["ysd_band"] = band_indices(
        risk["years_since_dissolution"].astype("int64").to_numpy(),
        rem.YSD_LOWERS,
        len(rem.YSD_BANDS),
    )
    risk["age_band"] = band_indices(
        np.rint(risk["age"].to_numpy()).astype(np.int64),
        rem.AGE_LOWERS,
        len(rem.AGE_BANDS),
    )
    keys = list(
        zip(
            risk["age_band"],
            risk["ysd_band"],
            risk["marital_state"],
            risk["sex"],
            strict=True,
        )
    )
    probability = np.array([table[key] for key in keys], dtype=np.float64)
    if (
        not np.isfinite(probability).all()
        or not ((probability > 0) & (probability < 1)).all()
    ):
        raise ValueError("direct-standardized probability outside (0,1)")
    event_frame = truth_events[truth_events["transition"] == "remarriage"]
    event_keys = {
        (int(person_id), int(year))
        for person_id, year in zip(
            event_frame["person_id"], event_frame["year"], strict=True
        )
    }
    if len(event_keys) != len(event_frame):
        raise ValueError("duplicate pseudo-truth remarriage person-year")
    risk_keys = {
        (int(person_id), int(year))
        for person_id, year in zip(
            risk["person_id"], risk["year"], strict=True
        )
    }
    if not event_keys.issubset(risk_keys):
        raise ValueError(
            "pseudo-truth remarriage event lacks dissolved risk row"
        )
    event = np.array(
        [
            int((int(person_id), int(year)) in event_keys)
            for person_id, year in zip(
                risk["person_id"], risk["year"], strict=True
            )
        ],
        dtype=np.float64,
    )
    weight = risk["weight"].to_numpy(dtype=np.float64)
    exposure = float(weight.sum())
    actual_numerator = float(np.sum(weight * event))
    expected_numerator = float(np.sum(weight * probability))
    source_numerator = float(event_frame["weight"].sum())
    if not math.isclose(
        actual_numerator, source_numerator, rel_tol=0.0, abs_tol=1e-8
    ):
        raise AssertionError(
            "truth event weights disagree with risk indicators"
        )
    deviance = float(
        -2.0
        * np.sum(
            weight
            * (
                event * np.log(probability)
                + (1.0 - event) * np.log1p(-probability)
            )
        )
        / exposure
    )
    return {
        "risk_rows": len(risk),
        "event_rows": int(event.sum()),
        "exposure": exposure,
        "actual_numerator": actual_numerator,
        "actual_rate": actual_numerator / exposure,
        "expected_numerator": expected_numerator,
        "expected_rate": expected_numerator / exposure,
        "expected_over_actual": expected_numerator / actual_numerator,
        "log_expected_over_actual": math.log(
            expected_numerator / actual_numerator
        ),
        "weighted_bernoulli_deviance": deviance,
    }


def _transport_ratios(
    projected: dict[str, Any], truth: dict[str, Any]
) -> dict[str, float]:
    exposure_ratio = projected["exposure"] / truth["exposure"]
    numerator_ratio = projected["numerator"] / truth["numerator"]
    rate_ratio = projected["rate"] / truth["rate"]
    return {
        "exposure_ratio": exposure_ratio,
        "numerator_ratio": numerator_ratio,
        "rate_ratio": rate_ratio,
        "log_exposure_ratio": math.log(exposure_ratio),
        "log_numerator_ratio": math.log(numerator_ratio),
        "log_rate_ratio": math.log(rate_ratio),
    }


def _mean_projection(
    per_seed: list[dict[str, Any]], truth: dict[str, Any]
) -> dict[str, Any]:
    exposure = float(
        np.mean([row["projected"]["exposure"] for row in per_seed])
    )
    numerator = float(
        np.mean([row["projected"]["numerator"] for row in per_seed])
    )
    rate = float(np.mean([row["projected"]["rate"] for row in per_seed]))
    projected = {
        "exposure": exposure,
        "numerator": numerator,
        "rate": rate,
    }
    return {**projected, **_transport_ratios(projected, truth)}


def _block_projection(
    per_seed: list[dict[str, Any]], truth: dict[str, Any]
) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in per_seed}
    out: dict[str, Any] = {}
    for index, block in enumerate(DRAW_BLOCKS, start=1):
        rows = [by_seed[seed] for seed in block]
        out[f"block_{index}"] = _mean_projection(rows, truth)
    return out


def _evaluate_boundary(
    full_context: ft_registry.FitContext, boundary: int
) -> dict[str, Any]:
    _progress(f"boundary {boundary}: truncating and fitting family components")
    truncated = refit._truncate_family_context(full_context, boundary)
    for label, frame in (
        ("person_years", truncated.panel.person_years),
        ("events", truncated.panel.events),
    ):
        _assert_at_most(frame, "year", boundary, f"fit_{label}")
        _assert_at_most(
            frame,
            refit.REQUIRED_INTERVIEW_COLUMN,
            boundary,
            f"fit_{label}",
        )
    components = ft_registry.REGISTRY.fit(ft_registry.CANDIDATE_16, truncated)
    tables, fit_diagnostics = _candidate_tables(
        truncated, boundary, incumbent=components.remarriage
    )

    anchor_waves, evaluation_years = _pseudo_calendar(boundary)
    if max(evaluation_years) > 2013:
        raise AssertionError(
            "a forbidden calendar-2014 flow entered evaluation"
        )
    anchor, present = _pseudo_anchor(
        full_context.demographic_panel, anchor_waves
    )
    seed_panel, valid_ids, carrier_count = _seed_panel(
        full_context.panel, anchor, max(evaluation_years)
    )
    truth_events, truth_person_years = _weighted_support(
        full_context.panel,
        anchor,
        present,
        evaluation_years,
        valid_ids,
    )
    truth = _rate_summary(truth_events, truth_person_years)
    truth_origin = _origin_summary(truth_events, truth_person_years)
    truth_support_keys = set(
        zip(
            truth_person_years["person_id"],
            truth_person_years["year"],
            strict=True,
        )
    )
    laws: dict[str, Any] = {}
    for law in LAWS:
        table = tables[law]
        parameter_eligible = table is not None
        if law == "L2":
            parameter_eligible = parameter_eligible and bool(
                fit_diagnostics["L2_parameter_eligible"]
            )
        if law == "L3":
            parameter_eligible = parameter_eligible and bool(
                fit_diagnostics["L3_parameter_eligible"]
            )
        if not parameter_eligible:
            laws[law] = {
                "parameter_eligible": False,
                "direct": None,
                "per_seed": [],
                "mean": None,
                "blocks": None,
            }
            continue
        assert isinstance(table, dict)
        direct = _direct_standardization(
            table, truth_events, truth_person_years
        )
        candidate_components: FittedFamilyTransitions = replace(
            components, remarriage=table
        )
        per_seed: list[dict[str, Any]] = []
        _progress(f"boundary {boundary}: {law}, 40 common-random-number draws")
        for offset, seed in enumerate(DRAW_SEEDS, start=1):
            registry = ProjectionRNGRegistry(
                draw_index=seed - 5200,
                n_periods=max(evaluation_years) - boundary,
            )
            projected_panel, _ = _simulate_candidate16_with_generators(
                seed_panel,
                valid_ids,
                candidate_components,
                registry.generator(0, ProjectionModule.MARITAL_CORE),
                registry.child_generator(0, ProjectionModule.MARITAL_CORE, 1),
            )
            projected_events, projected_person_years = _weighted_support(
                projected_panel,
                anchor,
                present,
                evaluation_years,
                valid_ids,
            )
            projected_support_keys = set(
                zip(
                    projected_person_years["person_id"],
                    projected_person_years["year"],
                    strict=True,
                )
            )
            if projected_support_keys != truth_support_keys:
                missing = len(truth_support_keys - projected_support_keys)
                extra = len(projected_support_keys - truth_support_keys)
                raise AssertionError(
                    f"support mismatch at {boundary}/{law}/{seed}: "
                    f"missing={missing}, extra={extra}"
                )
            projected = _rate_summary(projected_events, projected_person_years)
            per_seed.append(
                {
                    "seed": seed,
                    "projected": projected,
                    "ratios": _transport_ratios(projected, truth),
                    "origin": _origin_summary(
                        projected_events, projected_person_years
                    ),
                }
            )
            if offset % 10 == 0:
                _progress(f"boundary {boundary}: {law} draw {offset}/40")
        laws[law] = {
            "parameter_eligible": True,
            "direct": direct,
            "per_seed": per_seed,
            "mean": _mean_projection(per_seed, truth),
            "blocks": _block_projection(per_seed, truth),
        }

    return {
        "boundary": boundary,
        "anchor_waves": anchor_waves,
        "evaluation_years": evaluation_years,
        "anchor_persons_before_marital_intersection": int(
            anchor["person_id"].nunique()
        ),
        "anchor_households_before_marital_intersection": int(
            anchor["household_id"].nunique()
        ),
        "projected_persons": len(valid_ids),
        "entry_dissolved_carriers": carrier_count,
        "truth": truth,
        "truth_origin": truth_origin,
        "truth_support_rows": len(truth_person_years),
        "truth_same_year_ysd0_events": int(
            (
                (truth_events["transition"] == "remarriage")
                & (truth_events["years_since_dissolution"] == 0)
            ).sum()
        ),
        "support_checksum": _frame_checksum(
            truth_person_years,
            ("person_id", "year", "age", "sex", "weight"),
        ),
        "fit": fit_diagnostics,
        "laws": laws,
    }


def _selector(
    boundaries: dict[str, Any], final_parameters: dict[str, Any]
) -> dict[str, Any]:
    ordered = [boundaries[str(boundary)] for boundary in PSEUDO_BOUNDARIES]
    loss: dict[str, float | None] = {}
    block_loss: dict[str, dict[str, float] | None] = {}
    per_seed_loss: dict[str, dict[int, float] | None] = {}
    direct_deviance: dict[str, float | None] = {}
    parameter_eligible: dict[str, bool] = {}
    for law in LAWS:
        parameter_eligible[law] = bool(
            final_parameters[law]["eligible"]
        ) and all(
            boundary["laws"][law]["parameter_eligible"] for boundary in ordered
        )
        if not parameter_eligible[law]:
            loss[law] = None
            block_loss[law] = None
            per_seed_loss[law] = None
            direct_deviance[law] = None
            continue
        logs = [
            boundary["laws"][law]["mean"]["log_rate_ratio"]
            for boundary in ordered
        ]
        loss[law] = float(np.mean(np.square(logs)))
        block_loss[law] = {
            f"block_{index}": float(
                np.mean(
                    [
                        boundary["laws"][law]["blocks"][f"block_{index}"][
                            "log_rate_ratio"
                        ]
                        ** 2
                        for boundary in ordered
                    ]
                )
            )
            for index in (1, 2)
        }
        per_seed_loss[law] = {
            seed: float(
                np.mean(
                    [
                        next(
                            row
                            for row in boundary["laws"][law]["per_seed"]
                            if row["seed"] == seed
                        )["ratios"]["log_rate_ratio"]
                        ** 2
                        for boundary in ordered
                    ]
                )
            )
            for seed in DRAW_SEEDS
        }
        direct_deviance[law] = float(
            np.mean(
                [
                    boundary["laws"][law]["direct"][
                        "weighted_bernoulli_deviance"
                    ]
                    for boundary in ordered
                ]
            )
        )

    if loss["L0"] is None or block_loss["L0"] is None:
        raise AssertionError("the no-op law must be eligible")
    eligibility: dict[str, dict[str, Any]] = {}
    eligible_nonzero: list[str] = []
    for law in NONZERO_COMPLEXITY:
        checks: dict[str, Any] = {"parameters": parameter_eligible[law]}
        if not parameter_eligible[law]:
            checks.update(
                {
                    "full_J": False,
                    "both_block_J": False,
                    "rate_boundaries": False,
                    "exposure_boundaries": False,
                    "direct_deviance": False,
                }
            )
            eligibility[law] = {"eligible": False, "checks": checks}
            continue
        assert loss[law] is not None
        assert block_loss[law] is not None
        assert direct_deviance[law] is not None
        checks["full_J"] = loss[law] < float(loss["L0"]) - FLOAT_TOL
        checks["both_block_J"] = all(
            block_loss[law][f"block_{index}"]
            < block_loss["L0"][f"block_{index}"] - FLOAT_TOL
            for index in (1, 2)
        )
        rate_improvement = []
        rate_no_worse = []
        exposure_no_worse = []
        deviance_improvement = []
        deviance_no_worse = []
        for boundary in ordered:
            candidate = boundary["laws"][law]
            baseline = boundary["laws"]["L0"]
            candidate_rate = abs(candidate["mean"]["log_rate_ratio"])
            baseline_rate = abs(baseline["mean"]["log_rate_ratio"])
            rate_improvement.append(candidate_rate < baseline_rate - FLOAT_TOL)
            rate_no_worse.append(candidate_rate <= baseline_rate + FLOAT_TOL)
            candidate_exposure = abs(candidate["mean"]["log_exposure_ratio"])
            baseline_exposure = abs(baseline["mean"]["log_exposure_ratio"])
            exposure_no_worse.append(
                candidate_exposure <= baseline_exposure + FLOAT_TOL
            )
            candidate_dev = candidate["direct"]["weighted_bernoulli_deviance"]
            baseline_dev = baseline["direct"]["weighted_bernoulli_deviance"]
            deviance_improvement.append(
                candidate_dev < baseline_dev - FLOAT_TOL
            )
            deviance_no_worse.append(candidate_dev <= baseline_dev + FLOAT_TOL)
        checks["rate_boundaries"] = sum(rate_improvement) >= 2 and all(
            rate_no_worse
        )
        checks["exposure_boundaries"] = all(exposure_no_worse)
        checks["direct_deviance"] = (
            direct_deviance[law] < float(direct_deviance["L0"]) - FLOAT_TOL
            and sum(deviance_improvement) >= 2
            and all(deviance_no_worse)
        )
        is_eligible = all(bool(value) for value in checks.values())
        eligibility[law] = {"eligible": is_eligible, "checks": checks}
        if is_eligible:
            eligible_nonzero.append(law)

    selected = "L0"
    one_se: dict[str, Any] | None = None
    if eligible_nonzero:
        minimum = min(eligible_nonzero, key=lambda law: float(loss[law]))
        seed_values = np.array(
            list(per_seed_loss[minimum].values()), dtype=np.float64
        )
        standard_error = float(
            seed_values.std(ddof=1) / math.sqrt(len(seed_values))
        )
        cutoff = float(loss[minimum]) + standard_error
        within = [
            law
            for law in NONZERO_COMPLEXITY
            if law in eligible_nonzero
            and float(loss[law]) <= cutoff + FLOAT_TOL
        ]
        if float(loss["L0"]) <= cutoff + FLOAT_TOL:
            selected = "L0"
        else:
            selected = within[0]
        one_se = {
            "minimum_law": minimum,
            "minimum_J": loss[minimum],
            "standard_error": standard_error,
            "cutoff": cutoff,
            "eligible_within_cutoff": within,
            "L0_within_cutoff": float(loss["L0"]) <= cutoff + FLOAT_TOL,
        }

    return {
        "parameter_eligible": parameter_eligible,
        "J": loss,
        "block_J": block_loss,
        "direct_deviance_equal_boundary_mean": direct_deviance,
        "nonzero_eligibility": eligibility,
        "one_standard_error": one_se,
        "selected_law": selected,
        "disposition": (
            "RATIFIABLE_LAW_FOR_LATER_AMENDMENT"
            if selected != "L0"
            else "NO_OP_DESIGNED_PAUSE"
        ),
    }


def _final_parameters(full_context: ft_registry.FitContext) -> dict[str, Any]:
    boundary = MAX_INFORMATION_YEAR
    truncated = refit._truncate_family_context(full_context, boundary)
    tables, diagnostics = _candidate_tables(truncated, boundary)
    out: dict[str, Any] = {}
    for law in LAWS:
        table = tables[law]
        eligible = table is not None
        if law == "L2":
            eligible = eligible and bool(diagnostics["L2_parameter_eligible"])
        if law == "L3":
            eligible = eligible and bool(diagnostics["L3_parameter_eligible"])
        out[law] = {
            "eligible": eligible,
            "probability_min": (
                min(table.values()) if isinstance(table, dict) else None
            ),
            "probability_max": (
                max(table.values()) if isinstance(table, dict) else None
            ),
        }
    out["fit"] = diagnostics
    return out


def main() -> int:
    context, source_audit = _load_sanitized_context()
    boundaries: dict[str, Any] = {}
    for boundary in PSEUDO_BOUNDARIES:
        boundaries[str(boundary)] = _evaluate_boundary(context, boundary)
    _progress("fitting final <=2014-information parameters")
    final_parameters = _final_parameters(context)
    selection = _selector(boundaries, final_parameters)
    payload = {
        "schema": "m6_remarriage_train_only_delta.v1",
        "authority": {
            "program_merge": "051b4494ecce9345da14d68488bb2833ed476d22",
            "verification_comment": 5001901052,
        },
        "protocol": {
            "max_information_year": MAX_INFORMATION_YEAR,
            "pseudo_boundaries": PSEUDO_BOUNDARIES,
            "draw_seeds": DRAW_SEEDS,
            "draw_blocks": DRAW_BLOCKS,
            "laws": LAWS,
            "target": "sex-pooled remarriage ages 18-64",
            "gate_scorer_called": False,
            "gate_tolerance_read": False,
            "runs_artifact_written": False,
        },
        "source_audit": source_audit,
        "boundaries": boundaries,
        "final_2014_information_fit": final_parameters,
        "selection": selection,
    }
    print(
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    _progress(f"selected disposition: {selection['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
