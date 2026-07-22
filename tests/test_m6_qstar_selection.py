"""Synthetic proofs for the frozen amendment-4 q-star selector."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reduce_m6_qstar_selection as reducer  # noqa: E402
import select_m6_qstar_train_only as selector  # noqa: E402

from populace_dynamics.engine.forward_earnings import (  # noqa: E402
    CellMarginal,
    ForwardEarningsGenerator,
    ProjectedWageIndex,
)


class _ThresholdGate:
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def draw_sign(
        self,
        current_level: np.ndarray,
        target_age: np.ndarray,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        assert current_level.shape == target_age.shape == uniforms.shape
        return (current_level >= self.threshold).astype(np.int64)


def _cell() -> CellMarginal:
    return CellMarginal(
        p0=0.2,
        wtil=np.asarray([0.1, 0.2, 0.5, 0.8, 0.9]),
        yval=np.asarray([10.0, 20.0, 50.0, 80.0, 90.0]),
        n_pos=5,
        w_total=5.0,
    )


def _pool(rank: float, *, triple: bool = False) -> dict[str, np.ndarray]:
    pool = {
        "u_t": np.asarray([0.5]),
        "u_tp2": np.asarray([rank]),
        "u_A": np.asarray([0.5]),
        "u_w": np.asarray([0.5]),
        "weight": np.asarray([1.0]),
        "person_id": np.asarray([1], dtype=np.int64),
        "period_tp2": np.asarray([2014], dtype=np.int64),
    }
    if triple:
        pool["u_tm2"] = np.asarray([0.5])
    return pool


def _generator(*, threshold: float = 0.0) -> ForwardEarningsGenerator:
    gate = _ThresholdGate(threshold)
    return ForwardEarningsGenerator(
        shared_gate=gate,
        zero_anchor_gate=gate,
        marginals={index: _cell() for index in range(8)},
        pools={
            "pairs": _pool(0.8),
            "triples": _pool(0.8, triple=True),
            "reentry": {
                key: value for key, value in _pool(0.5).items() if key != "u_t"
            },
        },
        wage_index=ProjectedWageIndex(
            actual={2012: 1.0, 2014: 1.0},
            intercept=0.0,
            slope=0.0,
        ),
        u_w_by_person={1: 0.5},
        realized_earn_2014_by_person={1: 50.0},
        realized_earn_2012_by_person={1: 40.0},
    )


def _stable_pools(rank: float = 0.2) -> dict[int, dict[str, np.ndarray]]:
    return {index: _pool(rank) for index in range(8)}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1],
            "age": [32.0],
            "sex": [selector.SCHEMA_SEX_SENTINEL],
            "u_w": [0.5],
            "realized_earn_2014": [50.0],
            "realized_earn_2012": [40.0],
            "earnings": [50.0],
            "gen_earn_w2": [50.0],
            "gen_earn_w4": [40.0],
        }
    )


def test_registered_grid_seeds_cells_and_substreams_are_exact():
    assert selector.Q_GRID == tuple(
        round(index * 0.05, 2) for index in range(21)
    )
    assert selector.PSEUDO_BOUNDARIES == (2006, 2008, 2010)
    assert selector.FIT_SEED == 5200
    assert selector.SELECTION_DRAW_SEEDS == tuple(range(6200, 6220))
    assert selector.FIRST_HALF_DRAW_SEEDS == tuple(range(6200, 6210))
    assert selector.SECOND_HALF_DRAW_SEEDS == tuple(range(6210, 6220))
    assert selector.SUBSTREAM_CODES == {
        "gate": 1,
        "donor-draw": 2,
        "re-entry-draw": 3,
        "memory-refresh-gate": 4,
        "memory-refresh-rank": 5,
    }
    assert tuple(selector.FLOOR_SEEDS) == tuple(range(100))
    assert selector.SELECTED_CELLS == (
        "earn_p10.prime",
        "earn_dlog_mean.prime",
        "earn_dlog_sd.older",
        "earn_mob_h1_diag",
        "earn_autocorr_lag2",
        "earn_zero_rate.older",
    )
    assert selector.OBJECTIVE_CELLS == (
        "earn_p10.prime",
        "earn_dlog_mean.prime",
        "earn_mob_h1_diag",
        "earn_autocorr_lag2",
    )
    assert selector.FEASIBILITY_CELLS == (
        "earn_dlog_sd.older",
        "earn_zero_rate.older",
    )
    assert selector.fe.K_NEIGHBORS == 25


def test_metric_aware_draw_values_reject_nonpositive_log_ratios_only():
    assert (
        selector._selection_cell_value({"value": 0.0, "metric": "log_ratio"})
        is None
    )
    assert selector._selection_cell_value({"rate": 0.0}) is None
    assert (
        selector._selection_cell_value(
            {"value": 0.0, "metric": "abs_gap_corr"}
        )
        == 0.0
    )
    assert (
        selector._selection_cell_value({"value": 1.5, "metric": "log_ratio"})
        == 1.5
    )


def test_field_capped_nawi_stops_at_2014_newline(tmp_path):
    admitted = (
        b"description: fixture\n"
        b"values:\n"
        b"  2013-01-01: 44_888.16\n"
        b"  2014-01-01: 46_481.52\n"
    )
    forbidden = b"  2015-01-01: TRIPWIRE_MUST_NOT_BE_READ\n"
    path = tmp_path / "nawi.yaml"
    path.write_bytes(admitted + forbidden)
    values, audit = selector._read_historical_nawi(path, maximum_year=2014)
    assert values == {2013: 44_888.16, 2014: 46_481.52}
    assert audit["stopped_after_maximum_key"]
    assert audit["post_maximum_key_bytes_read"] is False
    assert audit["bytes_consumed_through_maximum_key"] == len(admitted)
    assert (
        audit["admitted_prefix_sha256"] == hashlib.sha256(admitted).hexdigest()
    )

    missing = tmp_path / "missing.yaml"
    missing.write_bytes(
        b"description: fixture\nvalues:\n  2013-01-01: 44_888.16\n"
    )
    with pytest.raises(ValueError, match="expected 2014"):
        selector._read_historical_nawi(missing, maximum_year=2014)


def test_psid_reader_requests_only_explicit_field_dated_waves(monkeypatch):
    waves = (2007, 2009, 2011, 2013, 2015)
    calls = []

    def ind_person_period(concepts, *, data_dir, waves):
        calls.append((concepts, data_dir, tuple(waves)))
        rows = []
        for wave in waves:
            rows.append(
                {
                    "person_id": 1,
                    "period": wave,
                    "age": 30 + wave - 2007,
                    "sequence": 1,
                    "relationship": 10,
                    "weight": 2.0,
                    "interview": wave * 10,
                }
            )
        return pd.DataFrame(rows)

    def labor(wave, *, data_dir):
        assert data_dir == Path("/staged")
        return pd.DataFrame(
            {
                "interview": [wave * 10],
                "head_labor": [float(wave)],
                "spouse_labor": [0.0],
            }
        )

    monkeypatch.setattr(
        selector.panels, "ind_person_period", ind_person_period
    )
    monkeypatch.setattr(selector, "_read_family_labor_levels", labor)
    monkeypatch.setattr(
        selector.panels,
        "demographic_panel",
        lambda **_kwargs: pytest.fail("broad demographic read"),
    )
    monkeypatch.setattr(
        selector.family,
        "family_earnings_panel",
        lambda **_kwargs: pytest.fail("broad earnings read"),
    )
    earnings, anchor, audit = selector._load_field_capped_psid(
        Path("/staged"), collection_waves=waves
    )
    assert calls[0][2] == waves
    assert tuple(calls[0][0]) == (
        "age",
        "sequence",
        "relationship",
        "weight",
        "interview",
    )
    assert earnings["period"].tolist() == [2006, 2008, 2010, 2012, 2014]
    assert anchor["period"].tolist() == [2007, 2009, 2011]
    assert audit["collection_waves_after_2014"] == [2015]
    assert not audit["post_2014_reference_field_requested"]
    with pytest.raises(AssertionError, match="capped|post-2014"):
        selector._load_field_capped_psid(
            Path("/staged"), collection_waves=(*waves, 2017)
        )


def test_q0_delegates_bit_exactly_and_preserves_old_stream_states():
    base = _generator()
    frame = _frame()
    incumbent = selector._TracedIncumbentGenerator(base)
    candidate = selector.RankRefreshPrototype(
        base=base,
        q=0.0,
        stable_pools=_stable_pools(),
        trace_incumbent=True,
    )
    incumbent_rng = np.random.default_rng(314159)
    candidate_rng = np.random.default_rng(314159)
    incumbent_value = incumbent.generate(frame, 2016, incumbent_rng)
    candidate_value = candidate.generate(frame, 2016, candidate_rng)

    assert incumbent_value.tobytes() == candidate_value.tobytes()
    assert incumbent.records == candidate.incumbent_records
    assert selector._rng_state_checksum(
        incumbent_rng
    ) == selector._rng_state_checksum(candidate_rng)
    assert candidate.refresh_records[0]["eligible_count"] == 1
    assert candidate.refresh_records[0]["refreshed_count"] == 0
    assert set(candidate.refresh_records[0]["stream_final_state_sha256"]) == {
        "memory-refresh-gate",
        "memory-refresh-rank",
    }


def test_odd_year_consumes_no_rng_and_delegates_maps_and_frame_state():
    base = _generator()
    candidate = selector.RankRefreshPrototype(
        base=base,
        q=0.0,
        stable_pools=_stable_pools(),
        trace_incumbent=True,
    )
    assert candidate.shared_gate is base.shared_gate
    assert candidate.zero_anchor_gate is base.zero_anchor_gate
    assert candidate.marginals is base.marginals
    assert candidate.wage_index is base.wage_index
    assert candidate.pools is base.pools

    initial = pd.DataFrame(
        {
            "person_id": [1],
            "year": [2014],
            "age": [30],
            "sex": ["selection_schema_only"],
        }
    )
    pd.testing.assert_frame_equal(
        candidate.materialize_initial_frame(initial),
        base.materialize_initial_frame(initial),
    )
    ranks = np.asarray([0.2, 0.8])
    ages = np.asarray([30.0, 45.0])
    np.testing.assert_array_equal(
        candidate.rank_to_level(ranks, ages, 2016),
        base.rank_to_level(ranks, ages, 2016),
    )
    levels = base.rank_to_level(ranks, ages, 2016)
    np.testing.assert_array_equal(
        candidate.level_to_rank(levels, ages, 2016),
        base.level_to_rank(levels, ages, 2016),
    )

    frame = _frame()
    rng = np.random.default_rng(2025)
    before = selector._rng_state_checksum(rng)
    odd = candidate.generate(frame, 2015, rng)
    assert selector._rng_state_checksum(rng) == before
    np.testing.assert_array_equal(odd, frame["earnings"].to_numpy())
    assert candidate.refresh_records == []
    assert candidate.incumbent_records[0]["parent_bridge_seed"] is None
    assert (
        candidate.incumbent_records[0]["incumbent_stream_final_state_sha256"]
        == {}
    )

    context = selector.PeriodContext(
        period_index=2,
        year=2016,
        draw_index=0,
        metadata={},
    )
    incumbent_rng = np.random.default_rng(88)
    wrapper_rng = np.random.default_rng(88)
    incumbent_state = selector.apply_earnings(
        frame, context, incumbent_rng, model=base
    )
    wrapper_state = selector.apply_earnings(
        frame,
        context,
        wrapper_rng,
        model=selector.RankRefreshPrototype(
            base=base, q=0.0, stable_pools=_stable_pools()
        ),
    )
    pd.testing.assert_frame_equal(incumbent_state, wrapper_state)
    assert selector._rng_state_checksum(
        incumbent_rng
    ) == selector._rng_state_checksum(wrapper_rng)


class _PrototypeBase:
    boundary_year = 2014

    def generate(self, frame, year, rng):
        rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        return np.full(len(frame), 0.9)

    @staticmethod
    def _anchor_ranks(level, current_age, target_year):
        return np.full(len(level), 0.5)

    @staticmethod
    def _third_coordinate(pool, q0):
        return ForwardEarningsGenerator._third_coordinate(pool, q0)

    @staticmethod
    def rank_to_level(rank, age, year):
        return np.asarray(rank)


def _coordinate_pools():
    pools = {}
    for bin_index in range(8):
        pools[bin_index] = {
            "u_A": np.asarray([0.4, 0.7]),
            "u_w": np.asarray([0.2, 0.9]),
            "u_t": np.asarray([10.0 + bin_index, 20.0 + bin_index]),
            "u_tm2": np.asarray([30.0 + bin_index, 40.0 + bin_index]),
            "u_tp2": np.asarray([0.1 + bin_index / 10.0, 0.8]),
            "weight": np.asarray([2.0, 3.0]),
        }
    return pools


def test_stable_draw_uses_exact_target_bin_and_asymmetric_coordinates(
    monkeypatch,
):
    captured = []

    def knn(distance, weight, donor_rank, uniforms):
        captured.append(
            {
                "distance": np.asarray(distance).copy(),
                "weight": np.asarray(weight).copy(),
                "donor_rank": np.asarray(donor_rank).copy(),
                "uniforms": np.asarray(uniforms).copy(),
            }
        )
        return np.full(len(distance), float(donor_rank[0]))

    monkeypatch.setattr(selector.fe, "_knn_draw", knn)
    frame = pd.DataFrame(
        {
            "person_id": [20, 10],
            "age": [37.0, 37.0],
            "u_w": [0.99, 0.01],
            "realized_earn_2014": [0.0, 10.0],
            "realized_earn_2012": [5.0, 6.0],
            "earnings": [100.0, 200.0],
            "gen_earn_w2": [50.0, 60.0],
            "gen_earn_w4": [1.0, 2.0],
        }
    )
    candidate = selector.RankRefreshPrototype(
        base=_PrototypeBase(), q=1.0, stable_pools=_coordinate_pools()
    )
    result = candidate.generate(frame, 2016, np.random.default_rng(41))
    np.testing.assert_allclose(result, [0.3, 0.3])
    assert len(captured) == 2
    positive, q0 = captured
    np.testing.assert_allclose(positive["distance"], [[0.12, 0.22]])
    np.testing.assert_allclose(q0["distance"], [[0.1, 0.2]])
    for call in captured:
        np.testing.assert_array_equal(call["weight"], [2.0, 3.0])
        np.testing.assert_allclose(call["donor_rank"], [0.3, 0.8])

    first_distances = [record["distance"].copy() for record in captured]
    captured.clear()
    mutated = frame.copy()
    mutated["u_w"] = [0.01, 0.99]
    mutated["earnings"] = [9_999.0, 8_888.0]
    mutated["gen_earn_w4"] = [7_777.0, 6_666.0]
    candidate.generate(mutated, 2016, np.random.default_rng(41))
    for before, after in zip(first_distances, captured, strict=True):
        np.testing.assert_array_equal(before, after["distance"])


def test_codes_four_and_five_are_common_across_q_thresholds():
    base = _generator()
    q0 = selector.RankRefreshPrototype(
        base=base,
        q=0.0,
        stable_pools=_stable_pools(0.2),
        trace_incumbent=True,
    )
    q1 = selector.RankRefreshPrototype(
        base=base,
        q=1.0,
        stable_pools=_stable_pools(0.2),
        trace_incumbent=True,
    )
    frame = _frame()
    out0 = q0.generate(frame, 2016, np.random.default_rng(1234))
    out1 = q1.generate(frame, 2016, np.random.default_rng(1234))
    assert q0.incumbent_records == q1.incumbent_records
    assert (
        q0.refresh_records[0]["parent_bridge_seed"]
        == q1.refresh_records[0]["parent_bridge_seed"]
    )
    assert (
        q0.refresh_records[0]["stream_final_state_sha256"]
        == q1.refresh_records[0]["stream_final_state_sha256"]
    )
    assert q0.refresh_records[0]["eligible_count"] == 1
    assert q1.refresh_records[0]["eligible_count"] == 1
    assert out0.tolist() != out1.tolist()


def test_q1_overwrites_only_positive_continuers_with_exact_bin_stable_draw():
    base = _generator()
    frame = _frame()
    baseline = base.generate(frame, 2016, np.random.default_rng(9))
    candidate = selector.RankRefreshPrototype(
        base=base,
        q=1.0,
        stable_pools=_stable_pools(0.2),
    )
    refreshed = candidate.generate(frame, 2016, np.random.default_rng(9))
    assert baseline.tolist() == [80.0]
    assert refreshed.tolist() == [20.0]

    reentry = frame.copy()
    reentry["earnings"] = 0.0
    reentry["gen_earn_w2"] = 0.0
    baseline_reentry = base.generate(reentry, 2016, np.random.default_rng(10))
    candidate_reentry = candidate.generate(
        reentry, 2016, np.random.default_rng(10)
    )
    assert baseline_reentry.tobytes() == candidate_reentry.tobytes()


def test_refreshed_carried_level_can_change_later_unchanged_gate():
    base = _generator(threshold=40.0)
    initial = _frame()
    q0 = selector.RankRefreshPrototype(
        base=base,
        q=0.0,
        stable_pools=_stable_pools(0.2),
    )
    q1 = selector.RankRefreshPrototype(
        base=base,
        q=1.0,
        stable_pools=_stable_pools(0.2),
    )
    q0_2016 = q0.generate(initial, 2016, np.random.default_rng(21))
    q1_2016 = q1.generate(initial, 2016, np.random.default_rng(21))
    assert q0_2016.tolist() == [80.0]
    assert q1_2016.tolist() == [20.0]

    def advance(level: np.ndarray) -> pd.DataFrame:
        frame = initial.copy()
        frame["age"] = 34.0
        frame["earnings"] = level
        frame["gen_earn_w4"] = frame["gen_earn_w2"]
        frame["gen_earn_w2"] = level
        return frame

    q0_2018 = q0.generate(advance(q0_2016), 2018, np.random.default_rng(22))
    q1_2018 = q1.generate(advance(q1_2016), 2018, np.random.default_rng(22))
    assert q0_2018[0] > 0
    assert q1_2018.tolist() == [0.0]


def test_stable_pool_partition_reuses_incumbent_pair_order_and_all_bins():
    rows = []
    for index, age in enumerate(range(27, 67, 5), start=1):
        rows.append(
            {
                "person_id": index,
                "period": 2012,
                "period_tp2": 2014,
                "earnings": 10.0,
                "earnings_tp2": 20.0,
                "age": age - 2,
                "age_tp2": age,
                "weight": 1.0,
                "weight_tp2": float(index),
            }
        )
    forward_pairs = pd.DataFrame(rows[::-1])
    ordered = forward_pairs.sort_values(
        ["person_id", "period_tp2"], kind="stable"
    )
    pair_pool = {
        "u_t": np.full(8, 0.4),
        "u_tp2": np.linspace(0.1, 0.8, 8),
        "u_A": np.full(8, 0.5),
        "u_w": np.full(8, 0.6),
        "weight": ordered["weight_tp2"].to_numpy(float),
        "person_id": ordered["person_id"].to_numpy(np.int64),
        "period_tp2": ordered["period_tp2"].to_numpy(np.int64),
    }
    fitted = SimpleNamespace(
        forward_pairs=forward_pairs,
        generator=SimpleNamespace(pools={"pairs": pair_pool}),
    )
    pools, audit = selector._stable_pools(fitted)
    assert set(pools) == set(range(8))
    assert audit["counts_by_bin"] == {str(index): 1 for index in range(8)}
    assert sum(len(pool["person_id"]) for pool in pools.values()) == 8
    assert audit["empty_pool_check_passed"]


def test_boundary_floor_splits_full_anchor_before_domain(monkeypatch):
    selected = {
        cell: {"value": 1.0, "metric": "abs_gap_corr", "n_obs": 10}
        for cell in selector.SELECTED_CELLS
    }
    seen_compute_ids = []

    def cells(frame, **_kwargs):
        seen_compute_ids.append(set(frame["person_id"].astype(int)))
        return selected

    def floor(full_anchor, compute, id_column):
        assert id_column == "person_id"
        assert set(full_anchor["person_id"].astype(int)) == {1, 2, 3}
        computed = compute({1, 3})
        assert computed == selected
        records = {
            cell: {"realized_sigma": 1.0, "n_defined_seeds": 100}
            for cell in selector.SELECTED_CELLS
        }
        return records, []

    monkeypatch.setattr(selector, "earnings_cells", cells)
    monkeypatch.setattr(selector, "_selected_cells", lambda *_args: selected)
    monkeypatch.setattr(selector, "run_floor", floor)
    generator = SimpleNamespace(
        realized_earn_2014_by_person={1: 10.0, 2: 20.0},
        u_w_by_person={1: 0.3, 2: 0.7},
    )
    fitted = SimpleNamespace(
        generator=generator,
        anchors=pd.DataFrame(
            {
                "person_id": [1, 2],
                "period": [2006, 2006],
                "earnings": [10.0, 20.0],
                "age": [30, 40],
                "weight": [1.0, 1.0],
            }
        ),
    )
    anchor_demo = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "period": [2007, 2007, 2007],
            "weight": [1.0, 2.0, 3.0],
            "interview": [11, 22, 33],
        }
    )
    rows = []
    for person_id, age in ((1, 30), (2, 40)):
        for period in (2006, 2008, 2010):
            rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "earnings": 10.0 + person_id + period - 2006,
                    "age": age + period - 2006,
                    "weight": float(person_id),
                }
            )
    context = selector._boundary_context(
        fitted, pd.DataFrame(rows), anchor_demo, 2006
    )
    assert seen_compute_ids == [{1}]
    assert context.support_audit["n_full_anchor"] == 3
    assert context.support_audit["n_domain"] == 2
    assert context.support_audit["split_before_domain_intersection"]
    assert context.rng_manifest["n_full_anchor_ordinals"] == 3


class _FakeProbabilityGate:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, scale):
        self.scale = float(scale)

    def predict_proba(self, features):
        positive = np.clip(
            self.scale * np.asarray(features)[:, 0] / 100.0, 0.05, 0.95
        )
        return np.column_stack((1.0 - positive, positive))


def _fake_fitted_gate(scale):
    target = SimpleNamespace(
        columns=("earnings", "age_tp2"),
        gate=_FakeProbabilityGate(scale),
        regime="zero_positive",
    )
    return SimpleNamespace(_target_models={"earnings_tp2": target})


def test_gate_audit_changes_with_exact_state_and_probability_surface():
    pairs = pd.DataFrame(
        {
            "person_id": [2, 1],
            "period_tp2": [2014, 2012],
            "earnings": [20.0, 80.0],
            "age_tp2": [35.0, 45.0],
        }
    )
    first = selector._participation_gate_audit(
        _fake_fitted_gate(0.5), pairs, name="shared_gate"
    )
    second = selector._participation_gate_audit(
        _fake_fitted_gate(1.0), pairs, name="shared_gate"
    )
    assert first["gate_state_sha256"] != second["gate_state_sha256"]
    assert (
        first["canonical_surface_sha256"] != second["canonical_surface_sha256"]
    )


def _synthetic_boundary_record(q: float) -> dict[str, object]:
    truth = {
        cell: {"value": 10.0, "metric": "abs_gap_corr", "n_obs": 100}
        for cell in selector.SELECTED_CELLS
    }
    draws = []
    for offset, seed in enumerate(selector.SELECTION_DRAW_SEEDS):
        alternating = -0.01 if offset % 2 == 0 else 0.01
        if q == 0.10:
            objective_error = (
                2.2 + alternating if offset < 10 else 1.2 + alternating
            )
            guard_error = 0.1 + alternating
        elif q == 0.15:
            objective_error = 1.5 + alternating
            guard_error = 1.21 + alternating / 10
        elif q == 0.25:
            objective_error = 1.2 + alternating
            guard_error = 0.1 + alternating
        elif q == 0.50:
            objective_error = 0.0 if offset % 2 == 0 else 2.0
            guard_error = 0.1 + alternating
        else:
            objective_error = 2.0 + alternating
            guard_error = 0.2 + alternating
        values = {
            cell: 10.0
            + (
                objective_error
                if cell in selector.OBJECTIVE_CELLS
                else guard_error
            )
            for cell in selector.SELECTED_CELLS
        }
        draws.append(
            {
                "draw_seed": seed,
                "moment_values": values,
                "annual_level_sha256": f"level-{q:.2f}-{seed}",
            }
        )
    return {
        "truth_moments": truth,
        "standardizers": {cell: 1.0 for cell in selector.SELECTED_CELLS},
        "per_draw": draws,
        "q0_equivalence": {"passed": True if q == 0 else None},
    }


def test_selector_applies_all_halves_jackknife_and_smallest_q_one_se():
    payload = {"rungs": {}}
    for q in selector.Q_GRID:
        payload["rungs"][f"{q:.2f}"] = {
            "q": q,
            "boundaries": {
                str(boundary): _synthetic_boundary_record(q)
                for boundary in selector.PSEUDO_BOUNDARIES
            },
        }
    selector._finalize_selector(payload)
    result = payload["selector"]
    assert result["q_min"] == 0.5
    assert result["selected_q"] == 0.25
    assert payload["rungs"]["0.10"]["strict_improvement_vs_q0"] == {
        "all_20": True,
        "first_10": False,
        "second_10": True,
    }
    assert not payload["rungs"]["0.10"]["retained_for_one_se"]
    assert not payload["rungs"]["0.15"]["feasible"]
    assert payload["rungs"]["0.25"]["retained_for_one_se"]
    assert payload["rungs"]["0.50"]["retained_for_one_se"]
    assert len(payload["rungs"]["0.50"]["objectives"]["delete_one"]) == 20
    deletes = np.asarray(
        [
            record["total"]
            for record in payload["rungs"]["0.50"]["objectives"]["delete_one"]
        ]
    )
    expected_se = np.sqrt(
        (19.0 / 20.0) * np.sum((deletes - deletes.mean()) ** 2)
    )
    assert result["q_min_jackknife_standard_error"] == pytest.approx(
        expected_se
    )
    assert payload["rungs"]["0.50"]["strict_improvement_vs_q0"] == {
        "all_20": True,
        "first_10": True,
        "second_10": True,
    }


def _weak_note_boundary_record(q: float) -> dict[str, object]:
    truth = {
        cell: {"value": 10.0, "metric": "abs_gap_corr", "n_obs": 100}
        for cell in selector.SELECTED_CELLS
    }
    draws = []
    for offset, seed in enumerate(selector.SELECTION_DRAW_SEEDS):
        variation = -0.125 if offset % 2 == 0 else 0.125
        if q == 0.05:
            objective_value = 10.5 + variation
        else:
            objective_value = (
                13.0 + variation if offset < 10 else 8.0 + variation
            )
        guard_value = 10.25 + variation
        draws.append(
            {
                "draw_seed": seed,
                "moment_values": {
                    cell: (
                        objective_value
                        if cell in selector.OBJECTIVE_CELLS
                        else guard_value
                    )
                    for cell in selector.SELECTED_CELLS
                },
                "annual_level_sha256": f"weak-{q:.2f}-{seed}",
            }
        )
    return {
        "truth_moments": truth,
        "standardizers": {cell: 1.0 for cell in selector.SELECTED_CELLS},
        "per_draw": draws,
        "q0_equivalence": {"passed": True if q == 0 else None},
    }


def test_weak_all_draw_equality_cannot_change_smallest_q_outcome():
    payload = {"rungs": {}}
    for q in selector.Q_GRID:
        payload["rungs"][f"{q:.2f}"] = {
            "q": q,
            "boundaries": {
                str(boundary): _weak_note_boundary_record(q)
                for boundary in selector.PSEUDO_BOUNDARIES
            },
        }
    selector._finalize_selector(payload)
    result = payload["selector"]
    assert not payload["rungs"]["0.05"]["retained_for_one_se"]
    assert 0.05 in result["weak_improvement_counterfactual"]["retained_q"]
    assert result["strict_vs_weak_improvement_outcome_invariant"]
    assert result["selected_q"] == 0
    assert result["weak_improvement_counterfactual"]["selected_q"] == 0


def _raw_reducer_fixture() -> bytes:
    rungs = {}
    for q in selector.Q_GRID:
        boundaries = {}
        for boundary in selector.PSEUDO_BOUNDARIES:
            support_hash = f"support-{boundary}"
            draws = []
            for offset, seed in enumerate(selector.SELECTION_DRAW_SEEDS):
                values = {
                    cell: 1.0 + offset / 100.0
                    for cell in selector.SELECTED_CELLS
                }
                draws.append(
                    {
                        "draw_seed": seed,
                        "moment_values": values,
                        "annual_level_sha256": f"l-{q}-{boundary}-{seed}",
                        "annual_participation_sha256": (
                            f"p-{q}-{boundary}-{seed}"
                        ),
                        "support_ids_sha256": support_hash,
                        "fresh_initial_state": True,
                    }
                )
            equivalent = []
            if q == 0:
                equivalent = [
                    {
                        "draw_seed": seed,
                        "person_period_keys_equal": True,
                        "level_bytes_equal": True,
                        "participation_states_equal": True,
                        "all_six_moments_equal": True,
                        "streams_1_3_final_states_equal": True,
                        "passed": True,
                        "n_incumbent_person_period_calls": 10,
                        "n_refresh_period_records": 5,
                        "old_stream_trace_sha256": f"old-{seed}",
                        "new_stream_trace_sha256": f"new-{seed}",
                    }
                    for seed in selector.SELECTION_DRAW_SEEDS
                ]
            boundaries[str(boundary)] = {
                "fit": {"sentinel": f"fit-{q}-{boundary}"},
                "floor": {"sentinel": f"floor-{q}-{boundary}"},
                "standardizers": {"sentinel": 1.0},
                "aggregates": {"sentinel": f"aggregate-{q}-{boundary}"},
                "support": {"truth_support_ids_sha256": support_hash},
                "per_draw": draws,
                "q0_equivalence": {
                    "required": q == 0,
                    "passed": True if q == 0 else None,
                    "per_draw": equivalent,
                },
            }
        rungs[f"{q:.2f}"] = {
            "q": q,
            "valid": True,
            "boundaries": boundaries,
            "objectives": {"sentinel": f"objective-{q}"},
            "feasibility_guards": {"sentinel": f"guard-{q}"},
        }
    payload = {
        "schema": reducer.RAW_SCHEMA,
        "protocol": {"q_grid": selector.Q_GRID},
        "rungs": rungs,
        "selector": {"selected_q": 0.25, "sentinel": "selector"},
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def test_reducer_hashes_exact_stdout_and_drops_only_repetitive_arrays():
    raw = _raw_reducer_fixture()
    reduced = reducer.reduce(raw)
    assert reduced["schema"] == reducer.FINDINGS_SCHEMA
    assert reduced["full_stdout_sha256"] == hashlib.sha256(raw).hexdigest()
    q0 = reduced["rungs"]["0.00"]["boundaries"]["2006"]
    assert "per_draw" not in q0
    assert q0["per_draw_summary"]["n_draws"] == 20
    assert q0["per_draw_summary"]["projected_support_ids_sha256"] == (
        q0["support"]["truth_support_ids_sha256"]
    )
    assert q0["per_draw_summary"]["truth_projection_support_equal_all_draws"]
    assert q0["fit"]["sentinel"] == "fit-0.0-2006"
    assert q0["floor"]["sentinel"] == "floor-0.0-2006"
    assert q0["aggregates"]["sentinel"] == "aggregate-0.0-2006"
    assert reduced["rungs"]["0.00"]["objectives"] == {
        "sentinel": "objective-0.0"
    }
    assert reduced["rungs"]["0.00"]["feasibility_guards"] == {
        "sentinel": "guard-0.0"
    }
    assert reduced["selector"] == {
        "selected_q": 0.25,
        "sentinel": "selector",
    }
    assert q0["q0_equivalence"]["passed"]
    nonzero = reduced["rungs"]["0.05"]["boundaries"]["2006"]
    assert nonzero["q0_equivalence"] == {
        "required": False,
        "passed": None,
        "n_draws": 0,
    }


def test_reducer_rejects_support_drift_and_q0_seed_reordering():
    support_drift = json.loads(_raw_reducer_fixture())
    support_drift["rungs"]["0.05"]["boundaries"]["2008"]["per_draw"][3][
        "support_ids_sha256"
    ] = "wrong-support"
    with pytest.raises(ValueError, match="projected support hashes"):
        reducer.reduce(json.dumps(support_drift).encode())

    seed_drift = json.loads(_raw_reducer_fixture())
    records = seed_drift["rungs"]["0.00"]["boundaries"]["2006"][
        "q0_equivalence"
    ]["per_draw"]
    records[0], records[1] = records[1], records[0]
    with pytest.raises(ValueError, match="equivalence seed order"):
        reducer.reduce(json.dumps(seed_drift).encode())


def test_reducer_retains_undefined_draws_for_an_ineligible_nonzero_rung():
    ledger = json.loads(_raw_reducer_fixture())
    rung = ledger["rungs"]["0.05"]
    rung["valid"] = False
    rung["boundaries"]["2006"]["per_draw"][2]["moment_values"][
        "earn_p10.prime"
    ] = None
    reduced = reducer.reduce(json.dumps(ledger).encode())
    summary = reduced["rungs"]["0.05"]["boundaries"]["2006"][
        "per_draw_summary"
    ]
    assert not summary["all_cells_defined"]
    assert summary["undefined_draw_seeds_by_cell"]["earn_p10.prime"] == [6202]

    invalid_claim = json.loads(_raw_reducer_fixture())
    invalid_claim["rungs"]["0.05"]["boundaries"]["2006"]["per_draw"][2][
        "moment_values"
    ]["earn_p10.prime"] = None
    with pytest.raises(ValueError, match="raw rung is marked valid"):
        reducer.reduce(json.dumps(invalid_claim).encode())
