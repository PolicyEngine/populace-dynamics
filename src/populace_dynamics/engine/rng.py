"""Deterministic random streams for the M6 projection wave loop.

The registry adds the period axis that the frozen single-period generators do
not have.  A call to :meth:`ProjectionRNGRegistry.generator` always returns a
fresh generator, so a module's stream is a pure function of draw, period, and
module rather than of call order elsewhere in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

DRAW_SEED_BASE = 5200
GATE_M6_TAG = 0x4D36


class ProjectionModule(str, Enum):
    """The fixed section 2.2 module order."""

    MORTALITY = "mortality"
    AGING = "aging"
    MARITAL_CORE = "marital_core"
    FERTILITY = "fertility"
    DISABILITY = "disability"
    EARNINGS = "earnings"
    CLAIMING = "claiming"
    HOUSEHOLD_COMPOSITION = "household_composition"


MODULE_ORDER = tuple(ProjectionModule)
_MODULE_INDEX = {module: index for index, module in enumerate(MODULE_ORDER)}


@dataclass(frozen=True)
class ProjectionRNGRegistry:
    """The ``draw x period x module x person`` M6 spawn tree.

    ``draw_index`` is the ensemble index ``k`` rather than a raw seed.  The
    root entropy therefore remains the inherited ``5200 + k`` convention.
    Period zero is addressable for deployment and compatibility checks; the
    projection loop normally requests periods one through ``n_periods``.
    """

    draw_index: int
    n_periods: int

    def __post_init__(self) -> None:
        if self.draw_index < 0:
            raise ValueError("draw_index must be non-negative")
        if self.n_periods < 0:
            raise ValueError("n_periods must be non-negative")

    @property
    def draw_seed(self) -> int:
        """Inherited ensemble root seed, ``5200 + k``."""
        return DRAW_SEED_BASE + self.draw_index

    def _period_sequence(self, period: int) -> np.random.SeedSequence:
        if not 0 <= period <= self.n_periods:
            raise ValueError(
                f"period must be in [0, {self.n_periods}], got {period}"
            )
        root = np.random.SeedSequence([self.draw_seed, GATE_M6_TAG])
        return root.spawn(self.n_periods + 1)[period]

    def seed_sequence(
        self, period: int, module: ProjectionModule | str
    ) -> np.random.SeedSequence:
        """Return the fixed module child sequence for one period."""
        resolved = ProjectionModule(module)
        period_sequence = self._period_sequence(period)
        return period_sequence.spawn(len(MODULE_ORDER))[
            _MODULE_INDEX[resolved]
        ]

    def generator(
        self, period: int, module: ProjectionModule | str
    ) -> np.random.Generator:
        """Return a fresh generator for one ``(module, period)`` stream."""
        return np.random.default_rng(self.seed_sequence(period, module))

    def child_generator(
        self,
        period: int,
        module: ProjectionModule | str,
        child_index: int,
    ) -> np.random.Generator:
        """Return a deterministic component child of a module stream."""
        if child_index < 0:
            raise ValueError("child_index must be non-negative")
        sequence = self.seed_sequence(period, module)
        child = np.random.SeedSequence(
            entropy=sequence.entropy,
            spawn_key=(*sequence.spawn_key, child_index),
            pool_size=sequence.pool_size,
        )
        return np.random.default_rng(child)

    def tagged_generator(
        self,
        period: int,
        module: ProjectionModule | str,
        component_tag: int,
    ) -> np.random.Generator:
        """Return a component-tagged stream with period-zero compatibility.

        Candidate-9's established single-period components use
        ``SeedSequence([5200 + k, tag])``.  Period zero preserves that exact
        stream.  Later periods place the same tag below the M6 period/module
        child, preventing the frozen seed from repeating across years.
        """
        if component_tag < 0:
            raise ValueError("component_tag must be non-negative")
        if period == 0:
            return np.random.default_rng(
                np.random.SeedSequence([self.draw_seed, component_tag])
            )
        sequence = self.seed_sequence(period, module)
        child = np.random.SeedSequence(
            entropy=sequence.entropy,
            spawn_key=(*sequence.spawn_key, component_tag),
            pool_size=sequence.pool_size,
        )
        return np.random.default_rng(child)

    def tagged_child_generator(
        self,
        period: int,
        module: ProjectionModule | str,
        component_tag: int,
        child_index: int,
    ) -> np.random.Generator:
        """Return one spawned child below a component's tagged stream."""
        if component_tag < 0 or child_index < 0:
            raise ValueError(
                "component_tag and child_index must be non-negative"
            )
        if period == 0:
            tagged = np.random.SeedSequence([self.draw_seed, component_tag])
            return np.random.default_rng(
                tagged.spawn(child_index + 1)[child_index]
            )
        sequence = self.seed_sequence(period, module)
        child = np.random.SeedSequence(
            entropy=sequence.entropy,
            spawn_key=(
                *sequence.spawn_key,
                component_tag,
                child_index,
            ),
            pool_size=sequence.pool_size,
        )
        return np.random.default_rng(child)

    def person_generator(
        self,
        period: int,
        module: ProjectionModule | str,
        person_ordinal: int,
    ) -> np.random.Generator:
        """Return the stream for a person in canonical ID-sorted order.

        The ordinal, rather than the identifier's representation, is part of
        the spawn key.  Callers must sort unique ``person_id`` values first.
        """
        return self.child_generator(period, module, person_ordinal)


def seed_from_generator(generator: np.random.Generator) -> int:
    """Bridge an injected generator to a frozen integer-seed-only API.

    New engine adapters consume generators directly.  This helper is limited
    to certified APIs for which a copied generator path is not necessary; it
    still removes their fixed-seed/period collision without changing fitted
    parameters.
    """
    return int(generator.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
