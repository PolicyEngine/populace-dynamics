"""Synthetic unit tests for the leakage-safe M6 forward earnings law."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.engine.forward_earnings import (
    FRAME_COLUMNS,
    RANK_CLAMP_HI,
    RANK_CLAMP_LO,
    SUBSTREAM_CODES,
    CellMarginal,
    ForwardEarningsGenerator,
    ProjectedWageIndex,
    age_bin,
    fit_age_marginals,
    fit_forward_earnings,
    fit_projected_wage_index,
)
from populace_dynamics.engine.rng import seed_from_generator


class _RecordingGate:
    def __init__(self, sign: int = 1) -> None:
        self.sign = int(sign)
        self.uniforms: list[np.ndarray] = []

    def draw_sign(
        self,
        current_level: np.ndarray,
        target_age: np.ndarray,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        assert current_level.shape == target_age.shape == uniforms.shape
        self.uniforms.append(uniforms.copy())
        return np.full(len(uniforms), self.sign, dtype=np.int64)


class _RecordingQRFFactory:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.fit_calls: list[tuple[pd.DataFrame, dict[str, object]]] = []

    def __call__(self, *, seed: int):
        self.seeds.append(int(seed))
        owner = self

        class Model:
            def fit(self, frame: pd.DataFrame, **kwargs):
                owner.fit_calls.append((frame.copy(), dict(kwargs)))
                return _RecordingGate(1)

        return Model()


class _NawiPoisonFrame(pd.DataFrame):
    """A frame that raises if code asks pandas for its ``nawi`` column."""

    @property
    def _constructor(self):
        return _NawiPoisonFrame

    def __getitem__(self, key):
        selected = [key] if isinstance(key, str) else list(key)
        if "nawi" in selected:
            raise AssertionError("scored earnings path read realized NAWI")
        return super().__getitem__(key)


def _cell(scale: float = 1.0) -> CellMarginal:
    return CellMarginal(
        p0=0.2,
        wtil=np.asarray([0.1, 0.2, 0.5, 0.8, 0.9]),
        yval=scale * np.asarray([10.0, 20.0, 50.0, 80.0, 90.0]),
        n_pos=5,
        w_total=5.0,
    )


def _marginals(*, distinct: bool = False) -> dict[int, CellMarginal]:
    return {
        index: _cell(float(index + 1) if distinct else 1.0)
        for index in range(8)
    }


def _transition_pool(
    ranks: tuple[float, ...], *, triple: bool
) -> dict[str, np.ndarray]:
    count = len(ranks)
    pool = {
        "u_t": np.full(count, 0.5, dtype=np.float64),
        "u_tp2": np.asarray(ranks, dtype=np.float64),
        "u_A": np.full(count, 0.5, dtype=np.float64),
        "u_w": np.full(count, 0.5, dtype=np.float64),
        "weight": np.ones(count, dtype=np.float64),
        "person_id": np.arange(count, dtype=np.int64),
        "period_tp2": np.full(count, 2014, dtype=np.int64),
    }
    if triple:
        pool["u_tm2"] = np.full(count, 0.5, dtype=np.float64)
    return pool


def _reentry_pool(ranks: tuple[float, ...]) -> dict[str, np.ndarray]:
    count = len(ranks)
    return {
        "u_tp2": np.asarray(ranks, dtype=np.float64),
        "u_A": np.full(count, 0.5, dtype=np.float64),
        "u_w": np.full(count, 0.5, dtype=np.float64),
        "weight": np.ones(count, dtype=np.float64),
        "person_id": np.arange(count, dtype=np.int64),
        "period_tp2": np.full(count, 2014, dtype=np.int64),
    }


def _pools(
    *,
    pair: tuple[float, ...] = (0.2,),
    triple: tuple[float, ...] = (0.8,),
    reentry: tuple[float, ...] = (0.5,),
) -> dict[str, Mapping[str, np.ndarray]]:
    return {
        "pairs": _transition_pool(pair, triple=False),
        "triples": _transition_pool(triple, triple=True),
        "reentry": _reentry_pool(reentry),
    }


def _generator(
    *,
    gate: _RecordingGate | None = None,
    pools: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    marginals: Mapping[int, CellMarginal] | None = None,
) -> ForwardEarningsGenerator:
    selected_gate = gate or _RecordingGate(1)
    fitted_state = {1: 0.5, 10: 0.5, 20: 0.5}
    realized_2014 = {1: 50.0, 10: 50.0, 20: 50.0}
    realized_2012 = {1: 40.0, 10: 40.0, 20: 40.0}
    return ForwardEarningsGenerator(
        shared_gate=selected_gate,
        zero_anchor_gate=selected_gate,
        marginals=marginals or _marginals(),
        pools=pools or _pools(),
        wage_index=ProjectedWageIndex(
            actual={2012: 1.0, 2014: 1.0},
            intercept=0.0,
            slope=0.0,
        ),
        u_w_by_person=fitted_state,
        realized_earn_2014_by_person=realized_2014,
        realized_earn_2012_by_person=realized_2012,
    )


def _frame(
    person_ids: tuple[int, ...] = (1,),
    *,
    current: tuple[float, ...] | None = None,
    lag: tuple[float, ...] | None = None,
    prior: tuple[float, ...] | None = None,
    anchor: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    count = len(person_ids)

    def values(
        supplied: tuple[float, ...] | None, default: float
    ) -> np.ndarray:
        chosen = supplied or tuple(default for _ in range(count))
        return np.asarray(chosen, dtype=np.float64)

    return pd.DataFrame(
        {
            "person_id": person_ids,
            "age": np.full(count, 32.0),
            "sex": ["female"] * count,
            "u_w": np.full(count, 0.5),
            "realized_earn_2014": values(anchor, 50.0),
            "realized_earn_2012": np.full(count, 40.0),
            "earnings": values(current, 50.0),
            "gen_earn_w2": values(lag, 50.0),
            "gen_earn_w4": values(prior, 40.0),
        }
    )


def test_age_grid_has_eight_bins_and_clips_both_outer_ranges():
    ages = np.asarray(
        [0, 24.999, 25, 29.999, 30, 34.999, 35, 40, 45, 50, 55, 60, 64, 99]
    )
    assert age_bin(ages).tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
    ]


def test_age_marginals_pool_waves_only_after_nawi_normalization():
    rows = []
    for index, age in enumerate(range(27, 67, 5), start=1):
        rows.extend(
            [
                {
                    "period": 2012,
                    "age": age,
                    "earnings": 100.0 * index,
                    "weight": 1.0,
                },
                {
                    "period": 2014,
                    "age": age,
                    "earnings": 200.0 * index,
                    "weight": 1.0,
                },
            ]
        )
    fitted = fit_age_marginals(pd.DataFrame(rows), {2012: 100.0, 2014: 200.0})
    assert set(fitted) == set(range(8))
    for index, marginal in fitted.items():
        assert marginal.yval.tolist() == [float(index + 1)] * 2


def test_cell_marginal_round_trip_corners_and_ties_are_deterministic():
    marginal = CellMarginal(
        p0=0.25,
        wtil=np.asarray([0.0, 0.25, 0.75, 1.0]),
        yval=np.asarray([1.0, 2.0, 4.0, 8.0]),
        n_pos=4,
        w_total=4.0,
    )
    interior = np.asarray([0.25, 0.75])
    assert np.array_equal(
        np.asarray(
            [marginal.rank(value) for value in marginal.quantile(interior)]
        ),
        interior,
    )
    assert marginal.quantile(np.asarray([-1.0, 2.0])).tolist() == [1.0, 8.0]
    assert marginal.rank(-100.0) == RANK_CLAMP_LO
    assert marginal.rank(100.0) == RANK_CLAMP_HI

    tied = CellMarginal(
        p0=0.0,
        wtil=np.asarray([0.1, 0.3, 0.5, 0.9]),
        yval=np.asarray([1.0, 2.0, 2.0, 3.0]),
        n_pos=4,
        w_total=4.0,
    )
    # This is literal candidate-5b/np.interp behavior: an exact duplicate
    # y-value maps to the rightmost plotting position on the stable grid.
    assert tied.rank(2.0) == 0.5
    assert tied.quantile(np.asarray([0.3, 0.5])).tolist() == [2.0, 2.0]


def test_generator_rank_level_round_trip_is_exact_in_every_bin():
    generator = _generator(marginals=_marginals(distinct=True))
    ranks = np.asarray([0.2, 0.5] * 8)
    ages = np.repeat(np.arange(27, 67, 5), 2)
    levels = generator.rank_to_level(ranks, ages, 2018)
    assert np.array_equal(generator.level_to_rank(levels, ages, 2018), ranks)


def test_odd_year_carries_without_rng_and_even_year_consumes_one_seed():
    generator = _generator(gate=_RecordingGate(0))
    frame = _frame(current=(123.0,))

    odd_rng = np.random.default_rng(81)
    carried = generator.generate(frame, 2015, odd_rng)
    odd_reference = np.random.default_rng(81)
    assert carried.tolist() == [123.0]
    assert np.array_equal(odd_rng.random(8), odd_reference.random(8))

    even_rng = np.random.default_rng(82)
    assert generator.generate(frame, 2016, even_rng).tolist() == [0.0]
    even_reference = np.random.default_rng(82)
    seed_from_generator(even_reference)
    assert np.array_equal(even_rng.random(8), even_reference.random(8))


def test_scored_even_path_never_reads_frames_realized_nawi_column():
    frame = _NawiPoisonFrame(
        _frame(current=(50.0,), lag=(50.0,), prior=(40.0,))
    )
    frame["nawi"] = [object()]
    generator = _generator(gate=_RecordingGate(1))
    assert generator.generate(
        frame, 2018, np.random.default_rng(3)
    ).tolist() == [80.0]


def test_2016_uses_pair_pool_and_2018_uses_full_triple_memory():
    generator = _generator(
        pools=_pools(pair=(0.2,), triple=(0.8,), reentry=(0.5,))
    )
    initial = generator.materialize_initial_frame(
        pd.DataFrame(
            {
                "person_id": [1],
                "year": [2014],
                "age": [30],
                "sex": ["female"],
            }
        )
    )
    assert initial.loc[0, "gen_earn_w2"] == 50.0
    assert initial.loc[0, "gen_earn_w4"] == 40.0

    wave_2016 = initial.copy()
    wave_2016["age"] = 32
    drawn_2016 = generator.generate(wave_2016, 2016, np.random.default_rng(10))
    assert drawn_2016.tolist() == [20.0]

    wave_2018 = wave_2016.copy()
    wave_2018["age"] = 34
    wave_2018["earnings"] = drawn_2016
    wave_2018["gen_earn_w4"] = wave_2016["gen_earn_w2"]
    wave_2018["gen_earn_w2"] = drawn_2016
    drawn_2018 = generator.generate(wave_2018, 2018, np.random.default_rng(11))
    assert drawn_2018.tolist() == [80.0]


def test_missing_2012_observation_is_an_unused_nan_in_the_2016_ramp():
    generator = replace(
        _generator(gate=_RecordingGate(0)),
        realized_earn_2012_by_person={},
    )
    initial = generator.materialize_initial_frame(
        pd.DataFrame(
            {
                "person_id": [1],
                "year": [2014],
                "age": [30],
                "sex": ["female"],
            }
        )
    )
    assert np.isnan(initial.loc[0, "realized_earn_2012"])
    assert np.isnan(initial.loc[0, "gen_earn_w4"])

    wave_2016 = initial.copy()
    wave_2016["age"] = 32
    assert generator.generate(
        wave_2016, 2016, np.random.default_rng(12)
    ).tolist() == [0.0]


def test_substream_bridge_matches_codes_and_replays_byte_for_byte():
    gate = _RecordingGate(1)
    generator = _generator(
        gate=gate,
        pools=_pools(
            pair=(0.2, 0.8),
            triple=(0.2, 0.8),
            reentry=(0.2, 0.8),
        ),
    )
    # Deliberately reverse row order. Substreams operate in canonical ID order,
    # while the returned ndarray must restore the caller's order.
    frame = _frame(
        (20, 10),
        current=(0.0, 50.0),
        lag=(0.0, 50.0),
        prior=(40.0, 40.0),
    )
    actual_parent = np.random.default_rng(2026)
    actual = generator.generate(frame, 2016, actual_parent)

    expected_parent = np.random.default_rng(2026)
    seed = seed_from_generator(expected_parent)
    expected_gate = np.random.default_rng(
        np.random.SeedSequence([seed, SUBSTREAM_CODES["gate"]])
    ).random(2)
    donor_uniform = np.random.default_rng(
        np.random.SeedSequence([seed, SUBSTREAM_CODES["donor-draw"]])
    ).random()
    reentry_uniform = np.random.default_rng(
        np.random.SeedSequence([seed, SUBSTREAM_CODES["re-entry-draw"]])
    ).random()

    assert np.array_equal(gate.uniforms[0], expected_gate)
    donor_rank = 0.2 if donor_uniform <= 0.5 else 0.8
    reentry_rank = 0.2 if reentry_uniform <= 0.5 else 0.8
    expected = np.asarray([100.0 * reentry_rank, 100.0 * donor_rank])
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_parent.random(8), expected_parent.random(8))

    replay = generator.generate(
        frame.copy(), 2016, np.random.default_rng(2026)
    )
    assert replay.tobytes() == actual.tobytes()


def test_wage_projection_ignores_realized_postcutoff_values():
    baseline = {
        year: 100.0 * 1.02 ** (year - 2005) for year in range(2005, 2019)
    }
    poisoned = dict(baseline)
    poisoned[2016] = 1.0
    poisoned[2018] = 1_000_000_000.0
    first = fit_projected_wage_index(baseline)
    second = fit_projected_wage_index(poisoned)
    fit_years = np.arange(2005, 2015, dtype=np.float64)
    expected_slope, expected_intercept = np.polyfit(
        fit_years,
        np.log([baseline[int(year)] for year in fit_years]),
        1,
    )
    assert set(first.actual) == set(range(2005, 2015))
    assert first.slope == pytest.approx(expected_slope)
    assert first.intercept == pytest.approx(expected_intercept)
    assert first.projected(2018) == pytest.approx(
        np.exp(expected_intercept + expected_slope * 2018)
    )
    assert first.projected(2016) == pytest.approx(second.projected(2016))
    assert first.projected(2018) == pytest.approx(second.projected(2018))


def _fit_panel() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    periods = tuple(range(2002, 2016, 2))
    for bin_index, age in enumerate(range(27, 67, 5)):
        person_id = bin_index + 1
        for wave_index, period in enumerate(periods):
            earnings = float(10_000 + 1_000 * bin_index + 125 * wave_index)
            if person_id == 1 and period == 2014:
                earnings = 0.0
            rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "earnings": earnings,
                    "age": age,
                    "weight": float(person_id),
                }
            )

    # These rows must not reach any fit surface: two are outside the pinned age
    # support and one is a post-cutoff future observation for a fitted person.
    for person_id, age in ((900, 24), (901, 65)):
        for period in periods:
            rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "earnings": 999_999.0,
                    "age": age,
                    "weight": 1.0,
                }
            )
    rows.append(
        {
            "person_id": 1,
            "period": 2016,
            "earnings": 888_888.0,
            "age": 27,
            "weight": 1.0,
        }
    )
    return pd.DataFrame(rows)


def test_refit_cutoff_and_25_64_support_reach_every_fitted_surface():
    factory = _RecordingQRFFactory()
    nawi = {year: 100.0 for year in range(2002, 2019)}
    fitted = fit_forward_earnings(
        _fit_panel(), nawi, seed=17, qrf_factory=factory
    )

    estimation = fitted.estimation_panel
    assert estimation["period"].max() == 2014
    assert estimation["age"].between(25, 64).all()
    assert set(estimation["person_id"]) == set(range(1, 9))
    assert not estimation["earnings"].isin([888_888.0, 999_999.0]).any()
    assert set(fitted.generator.marginals) == set(range(8))
    roster_ids = set(range(1, 9)) | {900, 901}
    assert set(fitted.generator.realized_earn_2014_by_person) == set(
        roster_ids
    )
    assert set(fitted.generator.realized_earn_2012_by_person) == set(
        roster_ids
    )
    assert set(fitted.generator.u_w_by_person) == roster_ids
    assert fitted.generator.u_w_by_person[900] == 0.5
    assert fitted.generator.u_w_by_person[901] == 0.5

    assert factory.seeds == [17, 17]
    assert len(factory.fit_calls) == 2
    for gate_frame, kwargs in factory.fit_calls:
        assert gate_frame["period_tp2"].max() <= 2014
        assert gate_frame["age"].between(25, 64).all()
        assert gate_frame["age_tp2"].between(25, 64).all()
        assert kwargs == {
            "predictors": ["earnings", "age_tp2"],
            "targets": ["earnings_tp2"],
            "weights": "weight_tp2",
        }

    for pool in fitted.generator.pools.values():
        assert set(pool["person_id"]).issubset(set(range(1, 9)))
        assert np.all(pool["period_tp2"] <= 2014)


def test_concrete_frame_schema_is_exactly_the_pinned_conditioning_set():
    assert FRAME_COLUMNS == (
        "person_id",
        "age",
        "sex",
        "u_w",
        "realized_earn_2014",
        "realized_earn_2012",
        "earnings",
        "gen_earn_w2",
        "gen_earn_w4",
    )
