"""Synthetic unit tests for the M6 operation order and RNG registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodModules,
    ProjectionEngine,
)
from populace_dynamics.engine.rng import (
    MODULE_ORDER,
    ProjectionModule,
    ProjectionRNGRegistry,
)


def test_rng_streams_replay_and_are_disjoint_by_period_module_and_person():
    registry = ProjectionRNGRegistry(draw_index=3, n_periods=2)
    values = {
        (period, module): registry.generator(period, module).random(4)
        for period in (0, 1, 2)
        for module in MODULE_ORDER
    }
    replay = ProjectionRNGRegistry(draw_index=3, n_periods=2)
    for key, expected in values.items():
        assert np.array_equal(replay.generator(*key).random(4), expected)
    assert not np.array_equal(
        values[(1, ProjectionModule.MORTALITY)],
        values[(2, ProjectionModule.MORTALITY)],
    )
    assert not np.array_equal(
        registry.person_generator(1, ProjectionModule.EARNINGS, 0).random(4),
        registry.person_generator(1, ProjectionModule.EARNINGS, 1).random(4),
    )


@pytest.mark.parametrize("tag", [0xC2, 0xC5, 0xC7, 0xB2B])
def test_period_zero_tagged_streams_are_single_period_compatible(tag):
    registry = ProjectionRNGRegistry(draw_index=4, n_periods=1)
    expected = np.random.default_rng(
        np.random.SeedSequence([5204, tag])
    ).random(8)
    actual = registry.tagged_generator(
        0, ProjectionModule.HOUSEHOLD_COMPOSITION, tag
    ).random(8)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(
        registry.tagged_generator(
            1, ProjectionModule.HOUSEHOLD_COMPOSITION, tag
        ).random(8),
        expected,
    )


def test_period_loop_order_and_single_marital_object_injection():
    calls: list[tuple[str, int, int | None]] = []

    def frame_step(name):
        def apply(frame, context, rng):
            calls.append((name, context.year, None))
            out = frame.copy()
            out["year"] = context.year
            out[name] = rng.random(len(out))
            return out

        return apply

    marital_results = []

    def marital(frame, context, rng):
        calls.append(("marital_core", context.year, None))
        result = MaritalStepResult(
            sim_years=pd.DataFrame(
                {
                    "person_id": frame.person_id,
                    "year": context.year,
                    "marital_state": "married",
                }
            ),
            births=pd.DataFrame(),
            modifier_check={"constraint_holds": True},
        )
        marital_results.append(result)
        return result

    def marital_reader(name):
        def apply(frame, context, marital_state, rng):
            assert marital_state is marital_results[-1]
            calls.append((name, context.year, id(marital_state)))
            out = frame.copy()
            out["year"] = context.year
            out[name] = rng.random(len(out))
            return out

        return apply

    modules = PeriodModules(
        mortality=frame_step("mortality"),
        aging=frame_step("aging"),
        marital_core=marital,
        fertility=marital_reader("fertility"),
        disability=frame_step("disability"),
        earnings=frame_step("earnings"),
        claiming=frame_step("claiming"),
        household_composition=marital_reader("household_composition"),
    )
    initial = pd.DataFrame({"person_id": [2, 1], "year": [2014, 2014]})
    result = ProjectionEngine(modules).project(
        initial, end_year=2016, draw_index=0
    )

    expected = [module.value for module in MODULE_ORDER]
    assert [name for name, year, _ in calls if year == 2015] == expected
    assert [name for name, year, _ in calls if year == 2016] == expected
    assert [trace.steps for trace in result.traces] == [
        tuple(expected),
        tuple(expected),
    ]
    for year in (2015, 2016):
        readers = [
            row for row in calls if row[1] == year and row[2] is not None
        ]
        assert len(readers) == 2
        assert readers[0][2] == readers[1][2]


def test_loop_rejects_a_second_year_or_duplicate_person_in_a_slice():
    def no_op(frame, context, rng):
        return frame

    modules = PeriodModules(
        mortality=no_op,
        aging=no_op,
        marital_core=lambda frame, context, rng: MaritalStepResult(
            pd.DataFrame(), pd.DataFrame()
        ),
        fertility=lambda frame, context, marital, rng: frame,
        disability=no_op,
        earnings=no_op,
        claiming=no_op,
        household_composition=lambda frame, context, marital, rng: frame,
    )
    with pytest.raises(ValueError, match="exactly one year"):
        ProjectionEngine(modules).project(
            pd.DataFrame({"person_id": [1, 2], "year": [2013, 2014]}),
            end_year=2014,
            draw_index=0,
        )
    with pytest.raises(ValueError, match="duplicate person_id"):
        ProjectionEngine(modules).project(
            pd.DataFrame({"person_id": [1, 1], "year": [2014, 2014]}),
            end_year=2014,
            draw_index=0,
        )
