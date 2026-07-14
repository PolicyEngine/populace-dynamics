"""Synthetic tests for the section 2.8.3a earnings-domain wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.engine.assembly import (
    CertifiedEngineInputs,
    assemble_period_modules,
)
from populace_dynamics.engine.earnings_domain import (
    EARNINGS_CHAIN_STATE_COLUMNS,
    EarningsDomainAdapter,
    earnings_domain_person_ids,
)
from populace_dynamics.engine.loop import PeriodContext
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.engine.steps import apply_earnings


class _RecordingCertifiedGenerator:
    def __init__(self):
        # Only person 10 has both pieces of the certified year-zero state.
        self.realized_earn_2014_by_person = {10: 1_010.0, 20: 2_020.0}
        self.realized_earn_2012_by_person = {10: 909.0}
        self.u_w_by_person = {10: 0.1, 30: 0.3}
        self.materialized_person_ids: list[list[int]] = []
        self.generated_person_ids: list[int] = []

    def materialize_initial_frame(self, frame):
        person_ids = frame["person_id"].astype(int).tolist()
        self.materialized_person_ids.append(person_ids)
        out = frame.copy()
        out["u_w"] = [
            self.u_w_by_person[person_id] for person_id in person_ids
        ]
        out["realized_earn_2014"] = [
            self.realized_earn_2014_by_person[person_id]
            for person_id in person_ids
        ]
        out["realized_earn_2012"] = [
            self.realized_earn_2012_by_person.get(person_id, np.nan)
            for person_id in person_ids
        ]
        out["earnings"] = out["realized_earn_2014"]
        out["gen_earn_w2"] = out["realized_earn_2014"]
        out["gen_earn_w4"] = out["realized_earn_2012"]
        return out

    def generate(self, frame, year, rng):
        del year
        person_ids = frame["person_id"].astype(int).tolist()
        assert len(person_ids) == 1
        self.generated_person_ids.extend(person_ids)
        return np.asarray([rng.uniform(100.0, 200.0)])


def _initial_frame():
    return pd.DataFrame(
        {
            "person_id": [30, 10, 20],
            "year": [2014, 2014, 2014],
            "age": [35, 35, 35],
            "sex": ["female", "female", "male"],
        },
        index=[7, 3, 9],
    )


def _context(person_ids, *, year, period_index):
    registry = ProjectionRNGRegistry(draw_index=4, n_periods=4)
    return PeriodContext(
        period_index=period_index,
        year=year,
        draw_index=4,
        metadata={},
        rng_registry=registry,
        person_ordinals={
            person_id: ordinal
            for ordinal, person_id in enumerate(sorted(person_ids))
        },
    )


def test_initialize_materializes_only_the_realized_2014_state_intersection():
    certified = _RecordingCertifiedGenerator()
    adapter = EarningsDomainAdapter(certified)

    result = adapter.materialize_initial_frame(_initial_frame())

    assert earnings_domain_person_ids(certified) == frozenset({10})
    assert earnings_domain_person_ids(adapter) == frozenset({10})
    assert certified.materialized_person_ids == [[10]]
    assert result.index.tolist() == [7, 3, 9]
    assert result["person_id"].tolist() == [30, 10, 20]
    assert result["earnings_domain"].dtype == bool
    assert result["earnings_domain"].tolist() == [False, True, False]
    assert result["earnings"].tolist() == [0.0, 1_010.0, 0.0]
    outside = result.loc[~result["earnings_domain"]]
    assert outside[list(EARNINGS_CHAIN_STATE_COLUMNS)].isna().all().all()


def test_assembly_wraps_the_certified_initializer_without_changing_generator():
    certified = _RecordingCertifiedGenerator()
    family = SimpleNamespace(name="same-cutoff-fit")
    inputs = CertifiedEngineInputs(
        family=family,
        modifier=object(),
        permanent_axis=object(),
        household=SimpleNamespace(family_transitions=family),
        mortality=object(),
        disability=object(),
        claiming=object(),
        earnings=certified,
        marital_panel_builder=lambda frame, context: (object(), set()),
        household_panel_builder=lambda frame, context: (object(), set()),
        disability_panel=object(),
        disability_ids=set(),
        start_weights=object(),
        male_gap=-2.0,
    )

    modules = assemble_period_modules(inputs)
    assert modules.initialize is not None
    initialized = modules.initialize(_initial_frame())

    assert certified.materialized_person_ids == [[10]]
    assert initialized["earnings_domain"].tolist() == [False, True, False]


def test_apply_skips_outside_domain_and_preserves_in_domain_person_stream():
    certified = _RecordingCertifiedGenerator()
    adapter = EarningsDomainAdapter(certified)
    initialized = adapter.materialize_initial_frame(_initial_frame())
    context = _context((10, 20, 30), year=2016, period_index=2)
    registry = context.rng_registry
    assert registry is not None

    with_outside = apply_earnings(
        initialized,
        context,
        registry.generator(2, ProjectionModule.EARNINGS),
        model=adapter,
    )
    supported_only = initialized.loc[
        initialized["earnings_domain"]
    ].reset_index(drop=True)
    alone = apply_earnings(
        supported_only,
        context,
        registry.generator(2, ProjectionModule.EARNINGS),
        model=adapter,
    )

    supported_value = with_outside.loc[
        with_outside["person_id"] == 10, "earnings"
    ].item()
    assert supported_value == alone["earnings"].item()
    assert certified.generated_person_ids == [10, 10]
    outside = with_outside.loc[~with_outside["earnings_domain"]]
    assert outside["earnings"].eq(0.0).all()
    assert outside[list(EARNINGS_CHAIN_STATE_COLUMNS)].isna().all().all()


def test_later_opener_keeps_the_zero_marker_across_odd_and_even_years():
    certified = _RecordingCertifiedGenerator()
    adapter = EarningsDomainAdapter(certified)
    opener = adapter.materialize_initial_frame(_initial_frame().loc[[9]])
    assert opener["person_id"].tolist() == [20]

    odd_context = _context((20,), year=2015, period_index=1)
    odd_registry = odd_context.rng_registry
    assert odd_registry is not None
    odd = apply_earnings(
        opener,
        odd_context,
        odd_registry.generator(1, ProjectionModule.EARNINGS),
        model=adapter,
    )
    even_context = _context((20,), year=2016, period_index=2)
    even_registry = even_context.rng_registry
    assert even_registry is not None
    even = apply_earnings(
        odd,
        even_context,
        even_registry.generator(2, ProjectionModule.EARNINGS),
        model=adapter,
    )

    assert certified.generated_person_ids == []
    assert not bool(even.loc[9, "earnings_domain"])
    assert even.loc[9, "earnings"] == 0.0
    assert even.loc[[9], list(EARNINGS_CHAIN_STATE_COLUMNS)].isna().all().all()


def test_wrapper_rejects_a_true_marker_outside_the_fitted_domain():
    certified = _RecordingCertifiedGenerator()
    adapter = EarningsDomainAdapter(certified)
    row = adapter.materialize_initial_frame(_initial_frame().loc[[9]])
    row["earnings_domain"] = True

    with pytest.raises(ValueError, match="marker disagrees"):
        adapter.generate(row, 2016, np.random.default_rng(1))


def test_apply_rejects_a_false_marker_inside_the_fitted_domain():
    certified = _RecordingCertifiedGenerator()
    adapter = EarningsDomainAdapter(certified)
    row = adapter.materialize_initial_frame(_initial_frame().loc[[3]])
    row["earnings_domain"] = False
    context = _context((10,), year=2016, period_index=2)
    registry = context.rng_registry
    assert registry is not None

    with pytest.raises(ValueError, match="marker disagrees"):
        apply_earnings(
            row,
            context,
            registry.generator(2, ProjectionModule.EARNINGS),
            model=adapter,
        )
