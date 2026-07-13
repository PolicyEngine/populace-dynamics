"""Small projection-time adapters for the non-registry M6 steps.

These adapters contain no fitted constants and perform no loading.  Mortality,
earnings, and claiming distributions are supplied by the cutoff refit bundle;
the functions here only apply them with the engine's injected generator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from populace_dynamics.engine.loop import MaritalStepResult, PeriodContext
from populace_dynamics.engine.marital import simulate_maternal_births
from populace_dynamics.engine.rng import ProjectionModule
from populace_dynamics.models.family_transitions.components.fertility import (
    build_fertility_lookup,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)
from populace_dynamics.models.household_composition.components.marital_core_adapter import (
    paternal_births,
)

__all__ = [
    "AgeSexMortalityModel",
    "ClaimingSchedule",
    "EarningsGenerator",
    "FertilityDraws",
    "advance_age",
    "apply_claiming",
    "apply_earnings",
    "apply_fertility",
    "apply_mortality",
    "materialize_maternal_births",
    "simulate_fertility",
]


@dataclass(frozen=True)
class AgeSexMortalityModel:
    """Annual death probabilities by inclusive age band and sex."""

    bands: tuple[tuple[int, int], ...]
    probability: Mapping[tuple[str, str], float]

    def __post_init__(self) -> None:
        if not self.bands or self.bands[0][0] != 0:
            raise ValueError("mortality bands must start at age zero")
        previous_upper = -1
        for lower, upper in self.bands:
            if lower != previous_upper + 1 or upper < lower:
                raise ValueError(
                    "mortality bands must be contiguous and non-overlapping"
                )
            previous_upper = upper
        if previous_upper < 120:
            raise ValueError("mortality bands must cover through age 120")
        expected = {
            (self.band_label(lower, upper), sex)
            for lower, upper in self.bands
            for sex in ("female", "male")
        }
        if set(self.probability) != expected:
            raise ValueError(
                "mortality probabilities must cover every age-band x sex cell"
            )
        values = np.asarray(list(self.probability.values()), dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError("mortality probabilities must lie in [0, 1]")

    @staticmethod
    def band_label(lower: int, upper: int) -> str:
        return f"{lower}-{upper}" if upper < 120 else f"{lower}+"

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Align fitted probabilities to a canonical person-sorted frame."""
        missing = {"age", "sex"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"mortality frame is missing columns {sorted(missing)}"
            )
        age = frame["age"].to_numpy(dtype=np.int64)
        sex = frame["sex"].astype(str).to_numpy()
        out = np.zeros(len(frame), dtype=np.float64)
        for lower, upper in self.bands:
            label = self.band_label(lower, upper)
            in_band = (age >= lower) & (age <= upper)
            for sex_label in np.unique(sex[in_band]):
                try:
                    value = float(self.probability[(label, sex_label)])
                except KeyError as error:
                    raise ValueError(
                        f"unknown mortality sex {sex_label!r}"
                    ) from error
                out[in_band & (sex == sex_label)] = value
        return out


def apply_mortality(
    frame: pd.DataFrame,
    context: PeriodContext,
    rng: np.random.Generator,
    *,
    model: AgeSexMortalityModel,
) -> pd.DataFrame:
    """Draw deaths first and return only the period's survivors."""
    ordered = frame.sort_values("person_id", kind="stable")
    if context.rng_registry is None:
        uniform = rng.random(len(ordered))
    else:
        uniform = np.asarray(
            [
                context.person_generator(
                    ProjectionModule.MORTALITY, person_id
                ).random()
                for person_id in ordered["person_id"]
            ]
        )
    death = uniform < model.probabilities(ordered)
    return ordered.loc[~death].reset_index(drop=True)


def advance_age(
    frame: pd.DataFrame,
    context: PeriodContext,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Advance survivor age and calendar year without consuming RNG."""
    del rng
    if "age" not in frame:
        raise ValueError("aging frame is missing column 'age'")
    out = frame.copy()
    out["age"] = out["age"].to_numpy(dtype=np.int64) + 1
    out["year"] = context.year
    for metadata_key, column in (
        ("nawi_by_year", "nawi"),
        ("wage_base_by_year", "taxable_max"),
    ):
        series = context.metadata.get(metadata_key)
        if series is None:
            if column in out:
                raise ValueError(
                    f"aging cannot roll {column!r} without {metadata_key!r}"
                )
            continue
        try:
            out[column] = float(series[context.year])
        except KeyError as error:
            raise ValueError(
                f"{metadata_key!r} has no value for {context.year}"
            ) from error
    return out


class EarningsGenerator(Protocol):
    """Projection interface implemented by the cutoff-refit earnings fit."""

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FertilityDraws:
    """Step-4 births split by open-population and paternal-shadow use."""

    maternal: pd.DataFrame
    paternal: pd.DataFrame


def simulate_fertility(
    marital: MaritalStepResult,
    components: FittedFamilyTransitions,
    holdout_ids: set[int],
    male_gap: float,
    rng: np.random.Generator,
) -> FertilityDraws:
    """Draw maternal births and the married-state paternal shadow at step 4."""
    maternal = simulate_maternal_births(
        marital.panel, holdout_ids, components, rng
    )
    lookup, decade_map = build_fertility_lookup(components.fertility)
    paternal = paternal_births(
        marital.sim_years,
        marital.panel.attrs[
            marital.panel.attrs["person_id"].isin(holdout_ids)
        ],
        male_gap,
        lookup,
        decade_map,
        rng,
    )
    return FertilityDraws(maternal=maternal, paternal=paternal)


def apply_earnings(
    frame: pd.DataFrame,
    context: PeriodContext,
    rng: np.random.Generator,
    *,
    model: EarningsGenerator,
) -> pd.DataFrame:
    """Draw annual earnings from the refitted chained generator."""
    if context.rng_registry is None:
        generated = np.asarray(model.generate(frame, context.year, rng))
    else:
        generated = np.zeros(len(frame), dtype=np.float64)
        eligible = (
            frame.get("age", pd.Series(18, index=frame.index)).to_numpy() >= 15
        )
        for index in np.flatnonzero(eligible):
            person_id = frame.iloc[index]["person_id"]
            person_rng = context.person_generator(
                ProjectionModule.EARNINGS, person_id
            )
            draw = np.asarray(
                model.generate(frame.iloc[[index]], context.year, person_rng)
            )
            if draw.shape != (1,):
                raise ValueError(
                    "earnings generator returned the wrong row shape"
                )
            generated[index] = float(draw[0])
    if generated.shape != (len(frame),):
        raise ValueError("earnings generator returned the wrong row shape")
    if not np.isfinite(generated).all() or (generated < 0).any():
        raise ValueError("earnings generator returned invalid earnings")
    out = frame.copy()
    if (
        context.year % 2 == 0
        and "gen_earn_w2" in frame
        and "gen_earn_w4" in frame
    ):
        out["gen_earn_w4"] = frame["gen_earn_w2"].to_numpy(
            dtype=np.float64, copy=True
        )
        out["gen_earn_w2"] = generated.astype(np.float64)
    out["earnings"] = generated.astype(np.float64)
    return out


@dataclass(frozen=True)
class ClaimingSchedule:
    """A cutoff-frozen claim-age PMF by sex and entitlement year."""

    pmf: Mapping[tuple[str, int], Mapping[int, float]]

    def distribution(
        self, sex: str, year: int
    ) -> tuple[np.ndarray, np.ndarray]:
        available = sorted(y for s, y in self.pmf if s == sex)
        if not available:
            raise KeyError(f"no claiming distribution for sex {sex!r}")
        selected_year = min(
            available, key=lambda candidate: abs(candidate - year)
        )
        values = self.pmf[(sex, selected_year)]
        ages = np.asarray(sorted(values), dtype=np.int64)
        probability = np.asarray(
            [values[int(age)] for age in ages], dtype=np.float64
        )
        if (
            not len(ages)
            or not np.isfinite(probability).all()
            or (probability < 0).any()
            or probability.sum() <= 0
        ):
            raise ValueError(f"invalid claiming PMF for {sex}|{selected_year}")
        return ages, probability / probability.sum()


def apply_claiming(
    frame: pd.DataFrame,
    context: PeriodContext,
    rng: np.random.Generator,
    *,
    schedule: ClaimingSchedule,
) -> pd.DataFrame:
    """Draw a planned claim age once and advance claiming state.

    M4 disability conversions are supplied by step 5 through an optional
    ``di_converted`` flag and do not enter the behavioral claim-age draw.
    """
    required = {"age", "sex"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"claiming frame is missing columns {sorted(missing)}"
        )
    out = frame.copy()
    if "claim_age" not in out:
        out["claim_age"] = pd.array([pd.NA] * len(out), dtype="Int64")
    unassigned = out["claim_age"].isna().to_numpy() & (
        out["age"].to_numpy(dtype=np.int64) >= 50
    )
    for sex in sorted(out.loc[unassigned, "sex"].astype(str).unique()):
        rows = unassigned & (out["sex"].astype(str).to_numpy() == sex)
        ages, probability = schedule.distribution(sex, context.year)
        row_indices = np.flatnonzero(rows)
        if context.rng_registry is None:
            chosen = rng.choice(ages, size=len(row_indices), p=probability)
        else:
            chosen = np.asarray(
                [
                    context.person_generator(
                        ProjectionModule.CLAIMING,
                        out.iloc[index]["person_id"],
                    ).choice(ages, p=probability)
                    for index in row_indices
                ]
            )
        out.loc[rows, "claim_age"] = chosen
    age = out["age"].to_numpy(dtype=np.int64)
    claim_age = out["claim_age"].fillna(10_000).to_numpy(dtype=np.int64)
    converted = (
        out.get("di_converted", pd.Series(False, index=out.index))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    previously_claimed = (
        out.get("claimed", pd.Series(False, index=out.index))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    out["claimed"] = previously_claimed | converted | (age >= claim_age)
    new_claim = out["claimed"].to_numpy(dtype=bool) & ~previously_claimed
    if "claim_year" not in out:
        out["claim_year"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[new_claim, "claim_year"] = context.year
    return out


def materialize_maternal_births(
    frame: pd.DataFrame,
    births: pd.DataFrame,
    context: PeriodContext,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Attach step-3 maternal births as new child person rows.

    The candidate-16 birth records identify mothers.  Paternal shadow births
    are household-composition conditioning draws and are deliberately not
    duplicated into open-population child rows.
    """
    roster_required = {"person_id", "year", "age", "sex", "weight"}
    roster_missing = roster_required - set(frame.columns)
    if roster_missing:
        raise ValueError(f"roster is missing columns {sorted(roster_missing)}")
    required = {"parent_person_id", "birth_year"}
    missing = required - set(births.columns)
    if missing:
        if births.empty:
            return frame.copy()
        raise ValueError(f"birth frame is missing columns {sorted(missing)}")
    current = births[
        births["birth_year"].astype("Int64") == context.year
    ].reset_index(drop=True)
    if current.empty:
        return frame.copy()

    parent = frame.set_index("person_id", drop=False)
    unknown = set(current["parent_person_id"]) - set(parent.index)
    if unknown:
        raise ValueError(
            f"birth parents are absent from the roster: {unknown}"
        )
    children = pd.DataFrame(
        index=np.arange(len(current)), columns=frame.columns
    )
    children["person_id"] = context.synthetic_id_allocator.allocate(
        len(current)
    )
    children["year"] = context.year
    children["age"] = 0
    children["birth_year"] = context.year
    children["sex"] = np.where(
        rng.random(len(current)) < 0.5, "female", "male"
    )
    children["parent_person_id"] = current["parent_person_id"].to_numpy()
    children["synthetic_entry"] = True
    for carried in ("household_id", "weight", "start_weight"):
        if carried in children:
            children[carried] = current["parent_person_id"].map(
                parent[carried]
            )
    existing = frame.copy()
    if "synthetic_entry" not in existing:
        existing["synthetic_entry"] = False
    return pd.concat([existing, children], ignore_index=True)


def apply_fertility(
    frame: pd.DataFrame,
    context: PeriodContext,
    marital: MaritalStepResult,
    rng: np.random.Generator,
    *,
    components: FittedFamilyTransitions | None = None,
    holdout_ids: set[int] | None = None,
    male_gap: float = -2.0,
    birth_store: dict[int, FertilityDraws] | None = None,
) -> pd.DataFrame:
    """Run step-4 fertility, materializing only maternal child rows.

    ``marital.births`` is accepted only as an explicit precomputed test seam.
    Production assembly supplies ``components`` and records both maternal and
    paternal draws for candidate-9's step-8 roster reconciliation.
    """
    if components is None:
        draws = FertilityDraws(
            maternal=marital.births,
            paternal=pd.DataFrame(
                {
                    "parent_person_id": pd.Series(dtype="int64"),
                    "birth_year": pd.Series(dtype="int64"),
                }
            ),
        )
    else:
        ids = holdout_ids or set(int(value) for value in frame["person_id"])
        draws = simulate_fertility(marital, components, ids, male_gap, rng)
    if birth_store is not None:
        birth_store[context.year] = draws
    return materialize_maternal_births(frame, draws.maternal, context, rng)
