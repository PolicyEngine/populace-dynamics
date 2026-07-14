"""Year-zero domain wrapper for the certified forward earnings generator.

The forward law is fitted only for people with both pieces of its 2014 state.
This adapter keeps that support restriction outside the certified stochastic
generator: it materializes state only for supported people and returns the
pinned non-scored zero for every other person.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

EARNINGS_DOMAIN_COLUMN = "earnings_domain"
EARNINGS_CHAIN_STATE_COLUMNS = (
    "u_w",
    "realized_earn_2014",
    "realized_earn_2012",
    "gen_earn_w2",
    "gen_earn_w4",
)

_MATERIALIZED_COLUMN_ORDER = (
    "u_w",
    "realized_earn_2014",
    "realized_earn_2012",
    "earnings",
    "gen_earn_w2",
    "gen_earn_w4",
)


class _CertifiedEarningsGenerator(Protocol):
    """The fitted state surface wrapped by :class:`EarningsDomainAdapter`."""

    u_w_by_person: Mapping[int, float]
    realized_earn_2014_by_person: Mapping[int, float]

    def materialize_initial_frame(
        self, frame: pd.DataFrame
    ) -> pd.DataFrame: ...

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray: ...


def _integer_person_ids(frame: pd.DataFrame) -> np.ndarray:
    if "person_id" not in frame:
        raise ValueError("earnings-domain frame is missing column 'person_id'")
    numeric = pd.to_numeric(frame["person_id"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError(
            "earnings-domain person_id values must be finite integers"
        )
    return numeric.astype(np.int64)


def earnings_domain_person_ids(generator: object) -> frozenset[int]:
    """Return the exact realized-2014 state intersection for ``generator``."""
    existing = getattr(generator, "domain_person_ids", None)
    if existing is not None:
        return frozenset(int(key) for key in existing)
    try:
        realized = generator.realized_earn_2014_by_person
        permanent = generator.u_w_by_person
    except AttributeError as error:
        raise TypeError(
            "earnings-domain generator must expose realized 2014 and u_w maps"
        ) from error
    if not isinstance(realized, Mapping) or not isinstance(permanent, Mapping):
        raise TypeError("earnings-domain state surfaces must be mappings")
    return frozenset(int(key) for key in realized) & frozenset(
        int(key) for key in permanent
    )


def earnings_domain_mask(frame: pd.DataFrame) -> np.ndarray:
    """Read the pinned bool marker, treating a new row's missing value as false."""
    if EARNINGS_DOMAIN_COLUMN not in frame:
        raise ValueError(
            f"earnings frame is missing column {EARNINGS_DOMAIN_COLUMN!r}"
        )
    mask = np.zeros(len(frame), dtype=bool)
    for index, value in enumerate(frame[EARNINGS_DOMAIN_COLUMN].to_numpy()):
        if pd.isna(value):
            continue
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("earnings_domain values must be bool or missing")
        mask[index] = bool(value)
    return mask


def _validate_initial_frame(frame: pd.DataFrame, generator: object) -> None:
    required = {"person_id", "year", "age", "sex"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"earnings initial frame is missing columns {sorted(missing)}"
        )
    numeric_year = pd.to_numeric(frame["year"], errors="coerce")
    if (
        numeric_year.isna().any()
        or not np.equal(
            numeric_year.to_numpy(dtype=np.float64),
            np.floor(numeric_year.to_numpy(dtype=np.float64)),
        ).all()
    ):
        raise ValueError("earnings initial frame has an invalid year")
    years = numeric_year.unique()
    if len(years) != 1:
        raise ValueError(
            "earnings initial frame must contain exactly one year"
        )
    boundary_year = getattr(generator, "boundary_year", None)
    if boundary_year is not None and int(years[0]) != int(boundary_year):
        raise ValueError(
            f"earnings state must be materialized at {int(boundary_year)}"
        )


@dataclass(frozen=True)
class EarningsDomainAdapter:
    """Apply the section 2.8.3a support law around a certified generator."""

    generator: _CertifiedEarningsGenerator
    domain_person_ids: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "domain_person_ids",
            earnings_domain_person_ids(self.generator),
        )

    def membership(self, frame: pd.DataFrame) -> np.ndarray:
        """Compute membership from the fitted state maps, independent of age."""
        person_ids = _integer_person_ids(frame)
        return np.asarray(
            [
                int(person_id) in self.domain_person_ids
                for person_id in person_ids
            ],
            dtype=bool,
        )

    def materialize_initial_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Materialize supported rows, then restore the full marked frame."""
        _validate_initial_frame(frame, self.generator)
        membership = self.membership(frame)
        in_domain = frame.loc[membership].copy()
        materialized: pd.DataFrame | None = None
        if len(in_domain):
            materialized = self.generator.materialize_initial_frame(in_domain)
            if len(materialized) != len(in_domain) or not np.array_equal(
                _integer_person_ids(materialized),
                _integer_person_ids(in_domain),
            ):
                raise ValueError(
                    "earnings initializer changed the in-domain person rows"
                )

        out = frame.copy()
        for column in _MATERIALIZED_COLUMN_ORDER:
            values = np.full(len(frame), np.nan, dtype=np.float64)
            if materialized is not None:
                if column not in materialized:
                    raise ValueError(
                        f"earnings initializer omitted column {column!r}"
                    )
                values[membership] = pd.to_numeric(
                    materialized[column], errors="raise"
                ).to_numpy(dtype=np.float64)
            if column == "earnings":
                values[~membership] = 0.0
            out[column] = values
        out[EARNINGS_DOMAIN_COLUMN] = membership
        return out

    def validate_domain(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the marker after checking its fitted-state equivalence."""
        marked = earnings_domain_mask(frame)
        fitted_membership = self.membership(frame)
        if not np.array_equal(marked, fitted_membership):
            raise ValueError(
                "earnings_domain marker disagrees with fitted 2014 state"
            )
        return marked

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Guard the certified per-person call with the pinned domain predicate."""
        marked = self.validate_domain(frame)
        if not marked.any():
            return np.zeros(len(frame), dtype=np.float64)
        if len(frame) != 1:
            raise ValueError(
                "earnings-domain generation must be called once per person"
            )
        return np.asarray(self.generator.generate(frame, year, rng))


def wrap_earnings_domain(generator: object) -> object:
    """Wrap a fitted forward generator while leaving generic test seams intact."""
    if isinstance(generator, EarningsDomainAdapter):
        return generator
    required = (
        "materialize_initial_frame",
        "realized_earn_2014_by_person",
        "u_w_by_person",
    )
    if all(hasattr(generator, name) for name in required):
        return EarningsDomainAdapter(generator)
    return generator
