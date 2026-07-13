"""The ordered M6 annual projection loop.

The loop is deliberately small: fitted component adapters own state-specific
math, while this module owns the binding operation order and the single-source
marital dataflow.  Fertility and household composition receive the exact same
step-3 result object; no other step can replace it or simulate another core.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)

FrameStep = Callable[
    [pd.DataFrame, "PeriodContext", np.random.Generator], pd.DataFrame
]
MaritalStep = Callable[
    [pd.DataFrame, "PeriodContext", np.random.Generator], "MaritalStepResult"
]
MaritalReaderStep = Callable[
    [
        pd.DataFrame,
        "PeriodContext",
        "MaritalStepResult",
        np.random.Generator,
    ],
    pd.DataFrame,
]


@dataclass
class SyntheticPersonIdAllocator:
    """Projection-wide monotone allocator for synthetic roster entries."""

    next_id: int

    def allocate(self, count: int) -> np.ndarray:
        """Return ``count`` never-before-used integer person identifiers."""
        if count < 0:
            raise ValueError("allocation count cannot be negative")
        allocated = np.arange(
            self.next_id, self.next_id + count, dtype=np.int64
        )
        self.next_id += count
        return allocated


@dataclass(frozen=True)
class MaritalStepResult:
    """Authoritative step-3 state and its deterministic gate-2c view."""

    sim_years: pd.DataFrame
    births: pd.DataFrame
    panel: Any = None
    weighted_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposure: pd.DataFrame = field(default_factory=pd.DataFrame)
    modifier_check: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeriodContext:
    """Immutable coordinates and carried metadata for one projected year."""

    period_index: int
    year: int
    draw_index: int
    metadata: Mapping[str, Any]
    rng_registry: ProjectionRNGRegistry | None = None
    person_ordinals: Mapping[object, int] = field(default_factory=dict)

    @property
    def synthetic_id_allocator(self) -> SyntheticPersonIdAllocator:
        """Return the projection-wide synthetic-person ID allocator."""
        allocator = self.metadata.get("synthetic_id_allocator")
        if not isinstance(allocator, SyntheticPersonIdAllocator):
            raise RuntimeError(
                "period context has no synthetic person ID allocator"
            )
        return allocator

    def person_generator(
        self, module: ProjectionModule | str, person_id: object
    ) -> np.random.Generator:
        """Return the stable initial-roster ordinal stream for one person."""
        if self.rng_registry is None:
            raise RuntimeError("period context has no RNG registry")
        try:
            ordinal = self.person_ordinals[person_id]
        except KeyError as error:
            raise KeyError(
                f"no stable RNG ordinal for person {person_id!r}"
            ) from error
        return self.rng_registry.person_generator(
            self.period_index, module, ordinal
        )


@dataclass(frozen=True)
class PeriodModules:
    """All eight required adapters in the binding section 2.2 order."""

    mortality: FrameStep
    aging: FrameStep
    marital_core: MaritalStep
    fertility: MaritalReaderStep
    disability: FrameStep
    earnings: FrameStep
    claiming: FrameStep
    household_composition: MaritalReaderStep


@dataclass(frozen=True)
class PeriodTrace:
    """Auditable operation trace for one wave."""

    year: int
    steps: tuple[str, ...]
    authoritative_marital_state: MaritalStepResult


@dataclass(frozen=True)
class ProjectionResult:
    """Projected slices, including the realized starting slice."""

    slices: tuple[pd.DataFrame, ...]
    traces: tuple[PeriodTrace, ...]
    draw_index: int

    @property
    def panel(self) -> pd.DataFrame:
        """Concatenate all period slices in chronological order."""
        return pd.concat(self.slices, ignore_index=True)


class ProjectionEngine:
    """Compose the eight certified-module adapters year by year."""

    def __init__(self, modules: PeriodModules) -> None:
        self.modules = modules

    @staticmethod
    def _validate_slice(frame: pd.DataFrame, label: str) -> None:
        required = {"person_id", "year"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns {sorted(missing)}")
        if frame["person_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate person_id rows")

    def project(
        self,
        initial_slice: pd.DataFrame,
        *,
        end_year: int,
        draw_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProjectionResult:
        """Project ``initial_slice`` through ``end_year`` inclusively."""
        self._validate_slice(initial_slice, "initial_slice")
        start_years = initial_slice["year"].unique()
        if len(start_years) != 1:
            raise ValueError("initial_slice must contain exactly one year")
        start_year = int(start_years[0])
        if end_year < start_year:
            raise ValueError("end_year precedes the initial slice")

        n_periods = end_year - start_year
        streams = ProjectionRNGRegistry(draw_index, n_periods)
        current = initial_slice.copy()
        slices = [current.copy()]
        traces: list[PeriodTrace] = []
        shared_metadata = dict(metadata or {})
        next_person_id = (
            int(initial_slice["person_id"].max()) + 1
            if len(initial_slice)
            else 1
        )
        supplied_allocator = shared_metadata.setdefault(
            "synthetic_id_allocator",
            SyntheticPersonIdAllocator(next_person_id),
        )
        if not isinstance(supplied_allocator, SyntheticPersonIdAllocator):
            raise TypeError(
                "metadata synthetic_id_allocator must be a "
                "SyntheticPersonIdAllocator"
            )
        person_ordinals = {
            person_id: ordinal
            for ordinal, person_id in enumerate(
                sorted(initial_slice["person_id"].unique())
            )
        }

        for period_index, year in enumerate(
            range(start_year + 1, end_year + 1), start=1
        ):
            context = PeriodContext(
                period_index=period_index,
                year=year,
                draw_index=draw_index,
                metadata=shared_metadata,
                rng_registry=streams,
                person_ordinals=person_ordinals,
            )
            steps: list[str] = []

            current = self.modules.mortality(
                current,
                context,
                streams.generator(period_index, ProjectionModule.MORTALITY),
            )
            steps.append(ProjectionModule.MORTALITY.value)
            current = self.modules.aging(
                current,
                context,
                streams.generator(period_index, ProjectionModule.AGING),
            )
            steps.append(ProjectionModule.AGING.value)
            marital = self.modules.marital_core(
                current,
                context,
                streams.generator(period_index, ProjectionModule.MARITAL_CORE),
            )
            if not isinstance(marital, MaritalStepResult):
                raise TypeError("marital_core must return MaritalStepResult")
            steps.append(ProjectionModule.MARITAL_CORE.value)
            current = self.modules.fertility(
                current,
                context,
                marital,
                streams.generator(period_index, ProjectionModule.FERTILITY),
            )
            steps.append(ProjectionModule.FERTILITY.value)
            current = self.modules.disability(
                current,
                context,
                streams.generator(period_index, ProjectionModule.DISABILITY),
            )
            steps.append(ProjectionModule.DISABILITY.value)
            current = self.modules.earnings(
                current,
                context,
                streams.generator(period_index, ProjectionModule.EARNINGS),
            )
            steps.append(ProjectionModule.EARNINGS.value)
            current = self.modules.claiming(
                current,
                context,
                streams.generator(period_index, ProjectionModule.CLAIMING),
            )
            steps.append(ProjectionModule.CLAIMING.value)
            current = self.modules.household_composition(
                current,
                context,
                marital,
                streams.generator(
                    period_index, ProjectionModule.HOUSEHOLD_COMPOSITION
                ),
            )
            steps.append(ProjectionModule.HOUSEHOLD_COMPOSITION.value)

            self._validate_slice(current, f"projected slice {year}")
            if set(current["year"].unique()) != {year}:
                raise ValueError(
                    f"projected slice {year} does not carry its target year"
                )
            slices.append(current.copy())
            for person_id in sorted(current["person_id"].unique()):
                if person_id not in person_ordinals:
                    person_ordinals[person_id] = len(person_ordinals)
            traces.append(
                PeriodTrace(
                    year=year,
                    steps=tuple(steps),
                    authoritative_marital_state=marital,
                )
            )

        return ProjectionResult(
            slices=tuple(slices),
            traces=tuple(traces),
            draw_index=draw_index,
        )
