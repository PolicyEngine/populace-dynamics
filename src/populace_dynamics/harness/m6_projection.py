"""Projection-side support preparation for the M6 reproduction scorer."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from populace_dynamics.data import disability, transitions
from populace_dynamics.engine.earnings_domain import wrap_earnings_domain
from populace_dynamics.engine.loop import PeriodContext
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.engine.steps import apply_earnings
from populace_dynamics.engine.support import (
    EvaluationMode,
    PreparedSupport,
    PresenceBasis,
    StartWaveWeightSnapshot,
    prepare_evaluation_support,
)
from populace_dynamics.harness.m6_cells import (
    EARN_COHORTS,
    GATED_EARN_YEARS,
    GATED_FLOW_YEARS,
    MARITAL_BANDS,
    SEXES,
    band_of,
    disability_scoring_universe,
)
from populace_dynamics.harness.m6_scoring import (
    restrict_earnings_domain_support,
)

PROJECTION_END_YEAR = 2022


def prepare_gated_realized_support(
    projection: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    realized_presence: pd.DataFrame,
    start_weights: StartWaveWeightSnapshot,
    presence_basis: PresenceBasis,
    period_column: str,
) -> PreparedSupport:
    """Bind an M6 scored surface to the engine's gated support contract.

    The frozen floor is aggregate-only, so the truth frame is also its
    row-support witness here.  Floor cell-function identity is enforced by the
    separate truth-side byte test; this call enforces symmetric keys and F6
    weights on the projected and realized scored frames themselves.
    """
    prepared = prepare_evaluation_support(
        projection,
        mode=EvaluationMode.GATED_REALIZED,
        truth=truth,
        floor=truth.copy(),
        realized_presence=realized_presence,
        presence_basis=presence_basis,
        start_weights=start_weights,
        period_column=period_column,
    )
    if prepared.mode is not EvaluationMode.GATED_REALIZED:
        raise RuntimeError("M6 scorer escaped gated-realized support mode")
    return prepared


def prepare_projected_marital(
    panel: transitions.MaritalPanel,
    anchor: pd.DataFrame,
    presence: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the truth-side window, presence, and F6 rules to a simulated panel."""

    def opening_wave(year: np.ndarray) -> np.ndarray:
        return np.where(year % 2 == 1, year, year - 1)

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[frame["year"].isin(GATED_FLOW_YEARS)].copy()
        out = out.merge(
            anchor[["person_id", "weight"]],
            on="person_id",
            how="inner",
            suffixes=("_raw", ""),
        )
        wave = opening_wave(out["year"].to_numpy(dtype=np.int64))
        keep = np.asarray(
            [
                person_id in presence.get(int(start_wave), set())
                for person_id, start_wave in zip(
                    out["person_id"], wave, strict=True
                )
            ],
            dtype=bool,
        )
        out = out.loc[keep].copy()
        out["band"] = out["age"].map(
            lambda value: band_of(value, MARITAL_BANDS)
        )
        return out[out["band"].notna() & out["sex"].isin(SEXES)].reset_index(
            drop=True
        )

    return prepare(panel.events), prepare(panel.person_years)


def prepare_projected_disability(
    panel: disability.DisabilityPanel,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    """Build the gated 2015/2017 realized-support transition pairs."""
    person_years = panel.person_years.sort_values(
        ["person_id", "period"], kind="stable"
    )
    grouped = person_years.groupby("person_id", sort=False)
    next_period = grouped["period"].shift(-1)
    next_disabled = grouped["disabled"].shift(-1)
    interval = next_period - person_years["period"]
    keep = (
        next_period.notna()
        & interval.between(1, disability.MAX_INTERVAL)
        & person_years["period"].isin((2015, 2017))
    )
    pairs = pd.DataFrame(
        {
            "person_id": person_years.loc[keep, "person_id"].to_numpy(),
            "sex": person_years.loc[keep, "sex"].to_numpy(),
            "age": person_years.loc[keep, "age"].to_numpy(dtype=np.int64),
            "start_wave": person_years.loc[keep, "period"].to_numpy(
                dtype=np.int64
            ),
            "from_disabled": person_years.loc[keep, "disabled"].to_numpy(
                dtype=bool
            ),
            "to_disabled": next_disabled.loc[keep].to_numpy(dtype=bool),
        }
    )
    pairs = pairs.merge(
        anchor[["person_id", "weight"]],
        on="person_id",
        how="inner",
        validate="many_to_one",
    )
    return disability_scoring_universe(pairs).reset_index(drop=True)


def _earnings_state_frame(initial_slice: pd.DataFrame) -> pd.DataFrame:
    required = {"person_id", "year", "age", "sex", "earnings_domain"}
    missing = required - set(initial_slice)
    if missing:
        raise ValueError(
            f"earnings initial slice is missing columns {sorted(missing)}"
        )
    # This allow-list is the scored-path leakage fence.  In particular, a
    # poisoned realized post-boundary ``nawi`` column cannot reach generate().
    return initial_slice[
        ["person_id", "year", "age", "sex", "earnings_domain"]
    ].copy()


def project_earnings_on_realized_support(
    *,
    initial_slice: pd.DataFrame,
    truth_support: pd.DataFrame,
    generator: object,
    domain_person_ids: Iterable[object],
    all_person_ids: Iterable[object],
    draw_index: int,
) -> pd.DataFrame:
    """Evaluate the chain for every realized-present in-domain person-period.

    Mortality is intentionally absent.  Stable ordinals are assigned over the
    complete side-A population, so the earnings stream for a supported person
    is identical to the main engine stream and independent of simulated death.
    """
    domain = frozenset(domain_person_ids)
    state = _earnings_state_frame(initial_slice)
    state = state[state["person_id"].isin(domain)].copy()
    if set(state["person_id"]) != set(domain):
        missing = sorted(set(domain) - set(state["person_id"]))[:10]
        raise ValueError(
            f"earnings domain is not present at the 2014 anchor: {missing}"
        )
    model = wrap_earnings_domain(generator)
    materialize = getattr(model, "materialize_initial_frame", None)
    if materialize is None:
        raise TypeError("scored earnings generator has no initializer")
    state = materialize(state)

    all_ids = sorted(set(all_person_ids))
    ordinals = {person_id: index for index, person_id in enumerate(all_ids)}
    if not set(domain).issubset(ordinals):
        raise ValueError("earnings domain lies outside the RNG population")
    registry = ProjectionRNGRegistry(
        draw_index=int(draw_index), n_periods=PROJECTION_END_YEAR - 2014
    )
    rows = [
        state[["person_id", "earnings"]]
        .assign(period=2014)
        .loc[:, ["person_id", "period", "earnings"]]
    ]
    for period_index, year in enumerate(range(2015, 2019), start=1):
        state = state.copy()
        state["age"] = state["age"].to_numpy(dtype=np.int64) + 1
        state["year"] = year
        context = PeriodContext(
            period_index=period_index,
            year=year,
            draw_index=int(draw_index),
            metadata={},
            rng_registry=registry,
            person_ordinals=ordinals,
        )
        state = apply_earnings(
            state,
            context,
            registry.generator(period_index, ProjectionModule.EARNINGS),
            model=model,
        )
        if year in GATED_EARN_YEARS:
            rows.append(
                state[["person_id", "earnings"]]
                .assign(period=year)
                .loc[:, ["person_id", "period", "earnings"]]
            )
    projected = pd.concat(rows, ignore_index=True)
    support_columns = [
        column
        for column in truth_support.columns
        if column not in {"earnings"}
    ]
    projected = projected.merge(
        truth_support[support_columns],
        on=["person_id", "period"],
        how="inner",
        validate="one_to_one",
    )
    projected, _truth = restrict_earnings_domain_support(
        projected,
        truth_support,
        domain,
    )
    return projected


def earnings_cohort(age: object) -> str | None:
    """Expose the shared truth-side cohort rule for synthetic projections."""
    if pd.isna(age):
        return None
    value = int(age)
    for name, lower, upper in EARN_COHORTS:
        if lower <= value <= upper:
            return name
    return None
