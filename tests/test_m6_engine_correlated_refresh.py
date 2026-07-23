"""Engine proofs for the M6 candidate-3 correlated-refresh prototype."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import populace_dynamics.engine.forward_earnings as fe
from populace_dynamics.engine.candidates import CANDIDATE_2
from populace_dynamics.engine.earnings_domain import (
    EARNINGS_CHAIN_STATE_COLUMNS,
)
from tests.test_m6_engine_forward_earnings import (
    FRAME_COLUMNS,
    _fit_panel,
    _frame,
    _generator,
    _pools,
    _RecordingGate,
    _RecordingQRFFactory,
)
from tests.test_m6_engine_rank_refresh import (
    _candidate,
    _project,
    _with_refresh,
)


class _FixedStream:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self, size=None):
        if size is None:
            return self.value
        return np.full(int(size), self.value, dtype=np.float64)


def _correlated(
    *,
    rho: float,
    gate: _RecordingGate | None = None,
    q: float = 0.55,
):
    base = _generator(
        gate=gate,
        pools=_pools(pair=(0.8,), triple=(0.8,), reentry=(0.5,)),
    )
    return replace(
        _with_refresh(base, q=q, rank=0.2),
        rank_refresh_rho=rho,
    )


def _state_frame(
    *,
    state: float = np.nan,
    age: float = 32.0,
    current: float = 50.0,
    lag: float = 50.0,
    prior: float = 40.0,
) -> pd.DataFrame:
    frame = _frame(
        current=(current,),
        lag=(lag,),
        prior=(prior,),
    )
    frame["age"] = age
    frame[fe.REFRESH_STATE_COLUMN] = state
    return frame


def _advance(
    frame: pd.DataFrame,
    result: fe.EarningsGenerationResult,
) -> pd.DataFrame:
    out = frame.copy()
    out["age"] = out["age"].to_numpy(dtype=np.float64) + 2.0
    out["gen_earn_w4"] = frame["gen_earn_w2"].to_numpy(
        dtype=np.float64, copy=True
    )
    out["gen_earn_w2"] = result.earnings
    out["earnings"] = result.earnings
    out[fe.REFRESH_STATE_COLUMN] = result.frame_updates[
        fe.REFRESH_STATE_COLUMN
    ]
    return out


def test_correlated_state_is_opt_in_and_rho_is_admissibility_checked():
    candidate2 = _with_refresh(_generator(), q=0.55)
    initial = pd.DataFrame(
        {"person_id": [1], "year": [2014], "age": [30], "sex": ["female"]}
    )
    candidate2_frame = candidate2.materialize_initial_frame(initial)
    assert candidate2.earnings_frame_update_columns == ()
    assert fe.REFRESH_STATE_COLUMN not in candidate2_frame

    candidate3 = replace(candidate2, rank_refresh_rho=-0.80)
    candidate3_frame = candidate3.materialize_initial_frame(initial)
    assert candidate3.earnings_frame_update_columns == (
        fe.REFRESH_STATE_COLUMN,
    )
    assert candidate3_frame[fe.REFRESH_STATE_COLUMN].isna().all()
    assert tuple(FRAME_COLUMNS) == (
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
    assert EARNINGS_CHAIN_STATE_COLUMNS == (
        "u_w",
        "realized_earn_2014",
        "realized_earn_2012",
        "gen_earn_w2",
        "gen_earn_w4",
    )
    with pytest.raises(ValueError, match="stationary Markov"):
        replace(candidate2, rank_refresh_rho=-0.82)
    with pytest.raises(ValueError, match="requires the complete"):
        replace(_generator(), rank_refresh_rho=0.0)
    with pytest.raises(
        ValueError, match="requires generate_with_frame_updates"
    ):
        candidate3.generate(_state_frame(), 2016, np.random.default_rng(1))


def test_threshold_law_uses_null_one_and_zero_states_with_strict_inequality():
    state = np.asarray([np.nan, 1.0, 0.0])
    threshold = fe._correlated_refresh_threshold(0.55, -0.80, state)
    assert threshold.tolist() == pytest.approx([0.55, 0.19, 0.99])

    generator = _correlated(rho=-0.80)
    output = np.asarray([0.81, 0.82, 0.83], dtype=np.float64)
    refreshed = generator._draw_correlated_rank_refresh(
        np.arange(3, dtype=np.int64),
        np.zeros(3, dtype=bool),
        np.full(3, 0.5),
        np.full(3, 32.0),
        state,
        _FixedStream(0.55),
        _FixedStream(0.25),
        output,
    )
    assert refreshed.tolist() == [False, False, True]
    assert output.tolist() == [0.81, 0.82, 0.2]


def test_rho_zero_matches_candidate2_levels_parent_and_streams_one_to_five(
    monkeypatch: pytest.MonkeyPatch,
):
    original_substream = fe._substream

    def run(generator, *, correlated: bool):
        streams = {}

        def traced(seed, label):
            child = original_substream(seed, label)
            streams[label] = child
            return child

        monkeypatch.setattr(fe, "_substream", traced)
        parent = np.random.default_rng(20260722)
        frame = _state_frame(state=0.0)
        if correlated:
            result = generator.generate_with_frame_updates(frame, 2016, parent)
            earnings = result.earnings
        else:
            earnings = generator.generate(frame, 2016, parent)
        states = {
            label: pickle.dumps(child.bit_generator.state, protocol=5)
            for label, child in streams.items()
        }
        return earnings, pickle.dumps(parent.bit_generator.state), states

    candidate2 = _with_refresh(
        _generator(pools=_pools(pair=(0.2, 0.8))), q=0.55
    )
    candidate3 = replace(candidate2, rank_refresh_rho=0.0)
    incumbent = run(candidate2, correlated=False)
    prototype = run(candidate3, correlated=True)
    assert prototype[0].tobytes() == incumbent[0].tobytes()
    assert prototype[1] == incumbent[1]
    assert prototype[2] == incumbent[2]
    assert set(prototype[2]) == set(fe.SUBSTREAM_CODES)


@pytest.mark.parametrize(
    ("label", "gate", "frame", "year"),
    (
        (
            "nonparticipation",
            _RecordingGate(0),
            _state_frame(state=0.0),
            2016,
        ),
        (
            "zero_earnings",
            _RecordingGate(0),
            _state_frame(state=1.0, current=0.0),
            2016,
        ),
        (
            "stream3_reentry",
            _RecordingGate(1),
            _state_frame(state=1.0, current=0.0, lag=0.0),
            2016,
        ),
        (
            "support_exit",
            _RecordingGate(1),
            _state_frame(state=0.0, age=65.0),
            2016,
        ),
    ),
)
def test_every_gap_class_resets_refresh_state(label, gate, frame, year):
    del label
    generator = _correlated(rho=-0.80, gate=gate)
    result = generator.generate_with_frame_updates(
        frame, year, np.random.default_rng(31)
    )
    assert np.isnan(result.frame_updates[fe.REFRESH_STATE_COLUMN][0])


def test_reset_fixture_post_gap_uses_q_not_p_zero_or_p_one(
    monkeypatch: pytest.MonkeyPatch,
):
    values = {
        "gate": 0.0,
        "donor-draw": 0.0,
        "re-entry-draw": 0.0,
        "memory-refresh-rank": 0.0,
    }

    def set_refresh_uniform(value: float) -> None:
        def fixed_substream(seed, label):
            del seed
            return _FixedStream(
                value if label == "memory-refresh-gate" else values[label]
            )

        monkeypatch.setattr(fe, "_substream", fixed_substream)

    generator = _correlated(rho=-0.80)

    set_refresh_uniform(0.80)
    post_zero_gap = _state_frame(state=np.nan)
    result = generator.generate_with_frame_updates(
        post_zero_gap, 2018, np.random.default_rng(1)
    )
    assert result.earnings.tolist() == [80.0]
    assert result.frame_updates[fe.REFRESH_STATE_COLUMN].tolist() == [0.0]
    counterfactual_zero = generator.generate_with_frame_updates(
        _state_frame(state=0.0), 2018, np.random.default_rng(1)
    )
    assert counterfactual_zero.earnings.tolist() == [20.0]

    set_refresh_uniform(0.30)
    post_one_gap = generator.generate_with_frame_updates(
        _state_frame(state=np.nan), 2018, np.random.default_rng(2)
    )
    assert post_one_gap.earnings.tolist() == [20.0]
    counterfactual_one = generator.generate_with_frame_updates(
        _state_frame(state=1.0), 2018, np.random.default_rng(2)
    )
    assert counterfactual_one.earnings.tolist() == [80.0]


def test_odd_year_consumes_no_rng_carries_state_and_resets_support_exit():
    generator = _correlated(rho=-0.80)
    frame = _state_frame(state=1.0)
    parent = np.random.default_rng(44)
    before = pickle.dumps(parent.bit_generator.state, protocol=5)
    result = generator.generate_with_frame_updates(frame, 2017, parent)
    assert pickle.dumps(parent.bit_generator.state, protocol=5) == before
    assert result.earnings.tolist() == [50.0]
    assert result.frame_updates[fe.REFRESH_STATE_COLUMN].tolist() == [1.0]

    exited = _state_frame(state=1.0, age=65.0)
    reset = generator.generate_with_frame_updates(exited, 2017, parent)
    assert np.isnan(reset.frame_updates[fe.REFRESH_STATE_COLUMN][0])
    assert pickle.dumps(parent.bit_generator.state, protocol=5) == before

    zero = _state_frame(state=1.0, current=0.0)
    reset = generator.generate_with_frame_updates(zero, 2017, parent)
    assert np.isnan(reset.frame_updates[fe.REFRESH_STATE_COLUMN][0])
    assert pickle.dumps(parent.bit_generator.state, protocol=5) == before


def test_negative_rho_changes_later_refresh_then_unchanged_participation(
    monkeypatch: pytest.MonkeyPatch,
):
    def fixed_substream(seed, label):
        del seed
        values = {
            "gate": 0.0,
            "donor-draw": 0.0,
            "re-entry-draw": 0.0,
            "memory-refresh-gate": 0.80,
            "memory-refresh-rank": 0.0,
        }
        return _FixedStream(values[label])

    monkeypatch.setattr(fe, "_substream", fixed_substream)

    class ThresholdGate(_RecordingGate):
        def __init__(self) -> None:
            super().__init__(1)
            self.levels: list[np.ndarray] = []

        def draw_sign(self, current_level, target_age, uniforms):
            del target_age
            self.uniforms.append(uniforms.copy())
            self.levels.append(current_level.copy())
            return (current_level >= 40.0).astype(np.int64)

    iid_gate = ThresholdGate()
    negative_gate = ThresholdGate()
    iid = _correlated(rho=0.0, gate=iid_gate)
    negative = _correlated(rho=-0.80, gate=negative_gate)
    iid_frame = _state_frame()
    negative_frame = _state_frame()

    iid_2016 = iid.generate_with_frame_updates(
        iid_frame, 2016, np.random.default_rng(1)
    )
    negative_2016 = negative.generate_with_frame_updates(
        negative_frame, 2016, np.random.default_rng(1)
    )
    assert (
        iid_2016.earnings.tolist() == negative_2016.earnings.tolist() == [80.0]
    )
    assert iid_2016.frame_updates[fe.REFRESH_STATE_COLUMN].tolist() == [0.0]
    assert negative_2016.frame_updates[fe.REFRESH_STATE_COLUMN].tolist() == [
        0.0
    ]

    iid_frame = _advance(iid_frame, iid_2016)
    negative_frame = _advance(negative_frame, negative_2016)
    iid_2018 = iid.generate_with_frame_updates(
        iid_frame, 2018, np.random.default_rng(2)
    )
    negative_2018 = negative.generate_with_frame_updates(
        negative_frame, 2018, np.random.default_rng(2)
    )
    assert (
        iid_gate.uniforms[1].tobytes() == negative_gate.uniforms[1].tobytes()
    )
    assert iid_2018.earnings.tolist() == [80.0]
    assert negative_2018.earnings.tolist() == [20.0]

    iid_frame = _advance(iid_frame, iid_2018)
    negative_frame = _advance(negative_frame, negative_2018)
    iid_2020 = iid.generate_with_frame_updates(
        iid_frame, 2020, np.random.default_rng(3)
    )
    negative_2020 = negative.generate_with_frame_updates(
        negative_frame, 2020, np.random.default_rng(3)
    )
    assert (
        iid_gate.levels[1].tolist()
        == negative_gate.levels[1].tolist()
        == [80.0]
    )
    assert iid_gate.levels[2].tolist() == [80.0]
    assert negative_gate.levels[2].tolist() == [20.0]
    assert iid_2020.earnings.tolist() == [80.0]
    assert negative_2020.earnings.tolist() == [0.0]


def test_rho_zero_full_projection_matches_candidate2_common_frame_bytes():
    nawi = {year: 100.0 for year in range(2002, 2019)}
    candidate2 = fe.fit_forward_earnings(
        _fit_panel(),
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=CANDIDATE_2,
    ).generator
    candidate3 = replace(candidate2, rank_refresh_rho=0.0)
    incumbent = _project(candidate2, draw_index=9).panel
    prototype = _project(candidate3, draw_index=9).panel
    common = list(incumbent.columns)
    pd.testing.assert_frame_equal(prototype[common], incumbent)
    assert fe.REFRESH_STATE_COLUMN not in incumbent
    assert fe.REFRESH_STATE_COLUMN in prototype


def test_q_zero_correlated_path_reproduces_candidate1_and_state_one_unreachable():
    nawi = {year: 100.0 for year in range(2002, 2019)}
    candidate1 = fe.fit_forward_earnings(
        _fit_panel(),
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
    ).generator
    q_zero = replace(
        fe.fit_forward_earnings(
            _fit_panel(),
            nawi,
            seed=17,
            qrf_factory=_RecordingQRFFactory(),
            candidate_spec=_candidate(0.0),
        ).generator,
        rank_refresh_rho=-0.80,
    )
    incumbent = _project(candidate1, draw_index=12).panel
    prototype = _project(q_zero, draw_index=12).panel
    pd.testing.assert_frame_equal(
        prototype[list(incumbent.columns)], incumbent
    )
    realized = prototype[fe.REFRESH_STATE_COLUMN].dropna().unique().tolist()
    assert set(realized).issubset({0.0})


def test_correlated_clone_reuses_every_pinned_fitted_object():
    candidate2 = _with_refresh(_generator(), q=0.55)
    candidate3 = replace(candidate2, rank_refresh_rho=-0.50)
    assert candidate3.shared_gate is candidate2.shared_gate
    assert candidate3.zero_anchor_gate is candidate2.zero_anchor_gate
    assert candidate3.marginals is candidate2.marginals
    assert candidate3.pools is candidate2.pools
    assert candidate3.stable_pools is candidate2.stable_pools
    assert candidate3.wage_index is candidate2.wage_index
    assert candidate3.u_w_by_person is candidate2.u_w_by_person
    assert (
        candidate3.rank_refresh_fit_audit is candidate2.rank_refresh_fit_audit
    )
