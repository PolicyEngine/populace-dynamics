"""Synthetic unit tests for the M6 operation order and RNG registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.engine.earnings_domain import wrap_earnings_domain
from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodModules,
    ProjectionEngine,
    SyntheticPersonIdAllocator,
)
from populace_dynamics.engine.rng import (
    MODULE_ORDER,
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.engine.steps import (
    advance_age,
    apply_earnings,
    materialize_maternal_births,
)


class _NamespaceEarnings:
    boundary_year = 2014

    def __init__(self, realized_ids, permanent_ids):
        self.realized_earn_2014_by_person = {
            int(person_id): 100.0 for person_id in realized_ids
        }
        self.realized_earn_2012_by_person = {
            int(person_id): 90.0 for person_id in realized_ids
        }
        self.u_w_by_person = {
            int(person_id): 0.5 for person_id in permanent_ids
        }

    def materialize_initial_frame(self, frame):
        out = frame.copy()
        out["u_w"] = 0.5
        out["realized_earn_2014"] = 100.0
        out["realized_earn_2012"] = 90.0
        out["earnings"] = 100.0
        out["gen_earn_w2"] = 100.0
        out["gen_earn_w4"] = 90.0
        return out

    def generate(self, frame, year, rng):
        del year, rng
        return np.full(len(frame), 100.0)


def _project_one_birth(
    earnings,
    *,
    end_year=2015,
    allocator=None,
    rng_bytes=None,
):
    model = wrap_earnings_domain(earnings)
    births = pd.DataFrame(
        {
            "parent_person_id": [5],
            "birth_year": pd.array([2015], dtype="Int64"),
        }
    )

    def mortality(frame, context, rng):
        del rng
        if rng_bytes is not None:
            rng_bytes.append(
                b"".join(
                    context.person_generator(
                        ProjectionModule.MORTALITY, person_id
                    ).bytes(32)
                    for person_id in sorted(frame["person_id"])
                )
            )
        return frame.copy()

    def marital(frame, context, rng):
        del frame, context, rng
        return MaritalStepResult(pd.DataFrame(), births)

    def fertility(frame, context, marital_result, rng):
        return materialize_maternal_births(
            frame, marital_result.births, context, rng
        )

    def passthrough(frame, context, rng):
        del context, rng
        return frame.copy()

    modules = PeriodModules(
        mortality=mortality,
        aging=advance_age,
        marital_core=marital,
        fertility=fertility,
        disability=passthrough,
        earnings=lambda frame, context, rng: apply_earnings(
            frame, context, rng, model=model
        ),
        claiming=passthrough,
        household_composition=lambda frame, context, marital_result, rng: frame.copy(),
        initialize=model.materialize_initial_frame,
    )
    metadata = (
        {} if allocator is None else {"synthetic_id_allocator": allocator}
    )
    initial = pd.DataFrame(
        {
            "person_id": [5],
            "year": [2014],
            "age": [30],
            "sex": ["female"],
            "weight": [1.0],
        }
    )
    return ProjectionEngine(modules).project(
        initial,
        end_year=end_year,
        draw_index=0,
        metadata=metadata,
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


def test_global_real_namespace_prevents_newborn_alias_without_shifting_rng():
    with pytest.raises(RuntimeError, match="reserved real-person namespace"):
        SyntheticPersonIdAllocator(6, frozenset({6})).allocate(1)

    control = _NamespaceEarnings({5}, {5})
    local_control = _project_one_birth(control)
    reserved_control = _project_one_birth(
        control,
        allocator=SyntheticPersonIdAllocator(6, frozenset({5})),
    )
    assert local_control.panel.to_csv(index=False).encode() == (
        reserved_control.panel.to_csv(index=False).encode()
    )

    aliased = _NamespaceEarnings({5, 6}, {5, 6})
    with pytest.raises(
        ValueError,
        match="earnings_domain marker disagrees with fitted 2014 state",
    ):
        _project_one_birth(aliased)

    reserved = frozenset({5, 6})
    protected = _project_one_birth(
        aliased,
        allocator=SyntheticPersonIdAllocator(7, reserved),
    )
    newborn = protected.slices[-1].loc[protected.slices[-1]["person_id"] != 5]
    assert newborn["person_id"].tolist() == [7]
    assert newborn["earnings_domain"].tolist() == [False]
    assert newborn["earnings"].tolist() == [0.0]

    union_generator = _NamespaceEarnings({5, 6, 9}, {5, 6})
    global_reserved = frozenset({5, 6, 9})
    first_allocator = SyntheticPersonIdAllocator(10, global_reserved)
    second_allocator = SyntheticPersonIdAllocator(10, global_reserved)
    first = _project_one_birth(
        union_generator,
        allocator=first_allocator,
    )
    replay = _project_one_birth(
        union_generator,
        allocator=second_allocator,
    )
    assert first_allocator is not second_allocator
    for result in (first, replay):
        child_ids = set(result.slices[-1]["person_id"]) - {5}
        assert child_ids == {10}
        assert child_ids.isdisjoint(global_reserved)
        assert all(
            not frame["person_id"].duplicated().any()
            for frame in result.slices
        )

    local_rng_bytes = []
    global_rng_bytes = []
    _project_one_birth(
        control,
        end_year=2016,
        rng_bytes=local_rng_bytes,
    )
    _project_one_birth(
        control,
        end_year=2016,
        allocator=SyntheticPersonIdAllocator(10, frozenset({5, 9})),
        rng_bytes=global_rng_bytes,
    )
    assert b"".join(global_rng_bytes) == b"".join(local_rng_bytes)
