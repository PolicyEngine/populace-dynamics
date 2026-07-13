"""Synthetic tests for the projection-time non-registry step adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodContext,
    SyntheticPersonIdAllocator,
)
from populace_dynamics.engine.rng import ProjectionRNGRegistry
from populace_dynamics.engine.steps import (
    AgeSexMortalityModel,
    ClaimingSchedule,
    advance_age,
    apply_claiming,
    apply_earnings,
    apply_fertility,
    apply_mortality,
    materialize_maternal_births,
)


def _context(year=2015, metadata=None):
    values = dict(metadata or {})
    values.setdefault("synthetic_id_allocator", SyntheticPersonIdAllocator(11))
    return PeriodContext(1, year, 0, values)


def test_mortality_runs_before_aging_on_canonical_person_order():
    frame = pd.DataFrame(
        {
            "person_id": [2, 1],
            "year": [2014, 2014],
            "age": [70, 30],
            "sex": ["male", "female"],
        }
    )
    model = AgeSexMortalityModel(
        bands=((0, 64), (65, 120)),
        probability={
            ("0-64", "female"): 0.0,
            ("0-64", "male"): 0.0,
            ("65+", "female"): 0.0,
            ("65+", "male"): 1.0,
        },
    )
    survivors = apply_mortality(
        frame, _context(), np.random.default_rng(0), model=model
    )
    aged = advance_age(
        survivors,
        _context(
            metadata={
                "nawi_by_year": {2015: 50_000.0},
                "wage_base_by_year": {2015: 120_000.0},
            }
        ),
        np.random.default_rng(1),
    )
    assert aged[["person_id", "age", "year"]].to_dict("records") == [
        {"person_id": 1, "age": 31, "year": 2015}
    ]
    assert aged["nawi"].tolist() == [50_000.0]
    assert aged["taxable_max"].tolist() == [120_000.0]


def test_mortality_model_rejects_an_uncovered_cell():
    with pytest.raises(ValueError, match="every age-band x sex"):
        AgeSexMortalityModel(
            bands=((0, 120),), probability={("0+", "female"): 0.01}
        )


def test_earnings_adapter_consumes_the_injected_generator():
    class Model:
        def generate(self, frame, year, rng):
            assert year == 2015
            return rng.uniform(10, 20, len(frame))

    frame = pd.DataFrame({"person_id": [1, 2]})
    first = apply_earnings(
        frame, _context(), np.random.default_rng(7), model=Model()
    )
    replay = apply_earnings(
        frame, _context(), np.random.default_rng(7), model=Model()
    )
    pd.testing.assert_frame_equal(first, replay)


def test_even_year_earnings_adapter_shifts_generated_lags():
    class Model:
        def generate(self, frame, year, rng):
            del frame, rng
            assert year == 2016
            return np.array([300.0, 400.0])

    frame = pd.DataFrame(
        {
            "person_id": [1, 2],
            "earnings": [100.0, 200.0],
            "gen_earn_w2": [90.0, 180.0],
            "gen_earn_w4": [80.0, 160.0],
        }
    )

    result = apply_earnings(
        frame, _context(2016), np.random.default_rng(7), model=Model()
    )

    assert result["earnings"].tolist() == [300.0, 400.0]
    assert result["gen_earn_w2"].tolist() == [300.0, 400.0]
    assert result["gen_earn_w4"].tolist() == [90.0, 180.0]
    assert frame["gen_earn_w2"].tolist() == [90.0, 180.0]
    assert frame["gen_earn_w4"].tolist() == [80.0, 160.0]


def test_odd_year_earnings_adapter_preserves_generated_lags():
    class Model:
        def generate(self, frame, year, rng):
            del rng
            assert year == 2017
            return frame["earnings"].to_numpy(dtype=np.float64)

    frame = pd.DataFrame(
        {
            "person_id": [1, 2],
            "earnings": [300.0, 400.0],
            "gen_earn_w2": [300.0, 400.0],
            "gen_earn_w4": [90.0, 180.0],
        }
    )

    result = apply_earnings(
        frame, _context(2017), np.random.default_rng(8), model=Model()
    )

    assert result["earnings"].tolist() == [300.0, 400.0]
    assert result["gen_earn_w2"].tolist() == [300.0, 400.0]
    assert result["gen_earn_w4"].tolist() == [90.0, 180.0]


def test_person_earnings_stream_does_not_shift_when_an_earlier_id_is_absent():
    class Model:
        def generate(self, frame, year, rng):
            del year
            return rng.random(len(frame))

    registry = ProjectionRNGRegistry(draw_index=0, n_periods=1)
    context = PeriodContext(
        1,
        2015,
        0,
        {},
        rng_registry=registry,
        person_ordinals={1: 0, 2: 1},
    )
    both = pd.DataFrame(
        {"person_id": [1, 2], "age": [30, 30], "earnings": [0.0, 0.0]}
    )
    second_only = both[both["person_id"] == 2].reset_index(drop=True)
    with_both = apply_earnings(
        both, context, registry.generator(1, "earnings"), model=Model()
    )
    alone = apply_earnings(
        second_only,
        context,
        registry.generator(1, "earnings"),
        model=Model(),
    )
    assert with_both.loc[1, "earnings"] == alone.loc[0, "earnings"]


def test_claiming_excludes_behavioral_redraw_for_di_conversion():
    schedule = ClaimingSchedule(
        {
            ("female", 2014): {62: 1.0},
            ("male", 2014): {70: 1.0},
        }
    )
    frame = pd.DataFrame(
        {
            "person_id": [1, 2],
            "age": [62, 63],
            "sex": ["female", "male"],
            "di_converted": [False, True],
        }
    )
    result = apply_claiming(
        frame,
        _context(2020),
        np.random.default_rng(2),
        schedule=schedule,
    )
    assert result["claim_age"].tolist() == [62, 70]
    assert result["claimed"].tolist() == [True, True]
    assert result["claim_year"].tolist() == [2020, 2020]


def test_birth_rows_inherit_maternal_household_and_fixed_weight():
    frame = pd.DataFrame(
        {
            "person_id": [10],
            "year": [2015],
            "age": [30],
            "sex": ["female"],
            "birth_year": [1985],
            "household_id": [55],
            "weight": [123.0],
            "start_weight": [123.0],
            "synthetic_entry": [False],
        }
    )
    births = pd.DataFrame(
        {"parent_person_id": [10, 10], "birth_year": [2015, 2016]}
    )
    result = materialize_maternal_births(
        frame, births, _context(), np.random.default_rng(0)
    )
    child = result.iloc[1]
    assert child["person_id"] == 11
    assert child["age"] == 0
    assert child["parent_person_id"] == 10
    assert child["household_id"] == 55
    assert child["weight"] == 123.0
    assert child["start_weight"] == 123.0
    assert bool(child["synthetic_entry"])

    via_step = apply_fertility(
        frame,
        _context(),
        MaritalStepResult(pd.DataFrame(), births),
        np.random.default_rng(0),
    )
    pd.testing.assert_frame_equal(result, via_step)


def test_birth_ids_are_never_reused_after_a_synthetic_child_exits():
    frame = pd.DataFrame(
        {
            "person_id": [10],
            "year": [2015],
            "age": [30],
            "sex": ["female"],
            "weight": [123.0],
        }
    )
    births = pd.DataFrame(
        {"parent_person_id": [10, 10], "birth_year": [2015, 2016]}
    )
    metadata = {
        "synthetic_id_allocator": SyntheticPersonIdAllocator(next_id=11)
    }
    first = materialize_maternal_births(
        frame,
        births,
        _context(2015, metadata),
        np.random.default_rng(0),
    )
    assert first.loc[first["age"] == 0, "person_id"].tolist() == [11]

    parent_only = first[first["person_id"] == 10].copy()
    parent_only["year"] = 2016
    second = materialize_maternal_births(
        parent_only,
        births,
        _context(2016, metadata),
        np.random.default_rng(1),
    )
    assert second.loc[second["age"] == 0, "person_id"].tolist() == [12]
