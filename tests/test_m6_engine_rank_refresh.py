"""Engine-level proofs for the ratified M6 earnings rank refresh."""

from __future__ import annotations

import inspect
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import populace_dynamics.engine as engine
import populace_dynamics.engine.forward_earnings as fe
import populace_dynamics.harness.m6_runner as m6_runner
from populace_dynamics.engine.candidates import (
    CANDIDATE_2,
    CANDIDATE_3,
    CORRELATED_REFRESH_OPERATION_ID,
    CORRELATED_REFRESH_OPERATION_KIND,
    RANK_REFRESH_OPERATION_ID,
    RANK_REFRESH_OPERATION_KIND,
    REGISTRY,
    CandidateSpec,
    OperationSpec,
)
from populace_dynamics.engine.forward_earnings import (
    RankRefreshFitAudit,
    RankRefreshPreflightAbort,
    fit_forward_earnings,
)
from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodModules,
    ProjectionEngine,
)
from populace_dynamics.engine.refit import (
    M6RefitBundle,
    refit_earnings_chained_generator,
)
from populace_dynamics.engine.steps import apply_earnings
from populace_dynamics.models import family_transitions as ft
from tests.test_m6_engine_forward_earnings import (
    _fit_panel,
    _frame,
    _generator,
    _pools,
    _RecordingGate,
    _RecordingQRFFactory,
)

ROOT = Path(__file__).resolve().parents[1]


def _candidate(q: float) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=f"synthetic_rank_refresh_q_{q}",
        contract_revision="engine_q_equivalence_fixture",
        operations=(
            OperationSpec(
                kind=RANK_REFRESH_OPERATION_KIND,
                implementation_id=RANK_REFRESH_OPERATION_ID,
                params={
                    "q": q,
                    "substream_codes": {
                        "memory-refresh-gate": 4,
                        "memory-refresh-rank": 5,
                    },
                },
            ),
        ),
    )


def _stable_pools(rank: float = 0.2) -> dict[int, dict[str, np.ndarray]]:
    return {
        bin_index: {
            "u_t": np.asarray([0.5]),
            "u_tp2": np.asarray([rank]),
            "u_A": np.asarray([0.5]),
            "u_w": np.asarray([0.5]),
            "weight": np.asarray([1.0]),
            "person_id": np.asarray([bin_index], dtype=np.int64),
            "period_tp2": np.asarray([2014], dtype=np.int64),
        }
        for bin_index in range(fe.N_AGE_BINS)
    }


def _eligible_audit() -> RankRefreshFitAudit:
    return RankRefreshFitAudit(
        source="synthetic incumbent positive-to-positive pair pool",
        sort=("person_id", "period_tp2"),
        target_age_bin_width=fe.AGE_BIN_WIDTH,
        k=fe.K_NEIGHBORS,
        counts_by_bin={str(index): 1 for index in range(fe.N_AGE_BINS)},
        checksums_by_bin={
            str(index): f"synthetic-{index}" for index in range(fe.N_AGE_BINS)
        },
        partition_sha256="synthetic",
        empty_bins=(),
    )


def _with_refresh(generator, q: float, rank: float = 0.2):
    return replace(
        generator,
        rank_refresh_q=q,
        stable_pools=_stable_pools(rank),
        rank_refresh_fit_audit=_eligible_audit(),
    )


def _initial_slice(generator) -> pd.DataFrame:
    anchor = _fit_panel()
    anchor = anchor[
        (anchor["period"] == 2014) & anchor["person_id"].between(2, 8)
    ].copy()
    return pd.DataFrame(
        {
            "person_id": anchor["person_id"].to_numpy(dtype=np.int64),
            "year": np.full(len(anchor), 2014, dtype=np.int64),
            "age": anchor["age"].to_numpy(dtype=np.int64),
            "sex": ["female"] * len(anchor),
            "weight": anchor["weight"].to_numpy(dtype=np.float64),
        }
    )


def _project(generator, *, draw_index: int = 3):
    def identity(frame, context, rng):
        del context, rng
        return frame.copy()

    def aging(frame, context, rng):
        del rng
        out = frame.copy()
        out["age"] = out["age"].to_numpy(dtype=np.int64) + 1
        out["year"] = context.year
        return out

    def marital(frame, context, rng):
        del frame, context, rng
        return MaritalStepResult(pd.DataFrame(), pd.DataFrame())

    def reader(frame, context, marital_result, rng):
        del context, marital_result, rng
        return frame.copy()

    def earnings(frame, context, rng):
        return apply_earnings(frame, context, rng, model=generator)

    modules = PeriodModules(
        mortality=identity,
        aging=aging,
        marital_core=marital,
        fertility=reader,
        disability=identity,
        earnings=earnings,
        claiming=identity,
        household_composition=reader,
        initialize=generator.materialize_initial_frame,
    )
    return ProjectionEngine(modules).project(
        _initial_slice(generator), end_year=2018, draw_index=draw_index
    )


def _projection_bytes(result) -> bytes:
    panel = result.panel.sort_values(
        ["person_id", "year"], kind="stable"
    ).reset_index(drop=True)
    payload = bytearray()
    for column, dtype in (
        ("person_id", np.int64),
        ("year", np.int64),
        ("earnings", np.float64),
        ("gen_earn_w2", np.float64),
        ("gen_earn_w4", np.float64),
    ):
        payload.extend(panel[column].to_numpy(dtype=dtype).tobytes())
    payload.extend((panel["earnings"] > 0).to_numpy(dtype=np.uint8).tobytes())
    return bytes(payload)


def test_engine_registry_binds_additive_candidate3_without_changing_prior_laws():
    assert dict(REGISTRY) == {2: CANDIDATE_2, 3: CANDIDATE_3}
    assert engine.CANDIDATE_3 is CANDIDATE_3
    operation = CANDIDATE_2.operation(RANK_REFRESH_OPERATION_KIND)
    assert operation is not None
    assert operation.implementation_id == RANK_REFRESH_OPERATION_ID
    assert operation.params["q"] == 0.55
    assert dict(operation.params["substream_codes"]) == {
        "memory-refresh-gate": 4,
        "memory-refresh-rank": 5,
    }
    assert CANDIDATE_2.sha256 == (
        "8fbfcf4130fd9051aa063061bf7b2d8514773fc6a900c900caab18717ad8e14c"
    )

    # Candidate 3 byte-carries candidate 2's complete operation as its exact
    # prefix, then adds only the ratified correlated-refresh state law.
    assert len(CANDIDATE_3.operations) == len(CANDIDATE_2.operations) + 1
    assert CANDIDATE_3.operations[:-1] == CANDIDATE_2.operations
    assert CANDIDATE_3.operations[0] is CANDIDATE_2.operations[0]
    correlated = CANDIDATE_3.operations[-1]
    assert correlated.kind == CORRELATED_REFRESH_OPERATION_KIND
    assert correlated.implementation_id == CORRELATED_REFRESH_OPERATION_ID
    assert dict(correlated.params) == {"rho": -0.60}
    assert CANDIDATE_3.canonical_dict()["operations"][:-1] == (
        CANDIDATE_2.canonical_dict()["operations"]
    )
    assert CANDIDATE_3.sha256 == (
        "c9be28a28d6fcc3911723872386906af559f6d0e0d5c89a87f741a5b2c3eacd6"
    )

    assert ft.CANDIDATE_16.sha256 == (
        "6d4d2b2beadc87d17404a3deb64a272c2456d7471b3ad6f1cef779d807765aa1"
    )
    assert "0.55" not in inspect.getsource(fe)
    assert "-0.60" not in inspect.getsource(fe)


def test_candidate_and_audit_inputs_are_immutable_snapshots():
    operation = OperationSpec(
        kind=RANK_REFRESH_OPERATION_KIND,
        implementation_id=RANK_REFRESH_OPERATION_ID,
        params={"q": 0.55},
    )
    operations = [operation]
    candidate = CandidateSpec(
        candidate_id="immutability_fixture",
        contract_revision="immutability_fixture",
        operations=operations,  # type: ignore[arg-type]
    )
    candidate_sha = candidate.sha256
    operations.append(
        OperationSpec(
            kind="synthetic.other_operation",
            implementation_id="synthetic.v1",
            params={},
        )
    )
    assert candidate.operations == (operation,)
    assert candidate.sha256 == candidate_sha
    restored_candidate = pickle.loads(pickle.dumps(candidate, protocol=5))
    assert restored_candidate.canonical_dict() == candidate.canonical_dict()
    assert restored_candidate.sha256 == candidate_sha

    counts = {str(index): 1 for index in range(fe.N_AGE_BINS)}
    checksums = {
        str(index): f"synthetic-{index}" for index in range(fe.N_AGE_BINS)
    }
    audit = replace(
        _eligible_audit(),
        counts_by_bin=counts,
        checksums_by_bin=checksums,
    )
    counts["0"] = 999
    checksums["0"] = "changed"
    assert audit.counts_by_bin["0"] == 1
    assert audit.checksums_by_bin["0"] == "synthetic-0"
    with pytest.raises(TypeError):
        audit.counts_by_bin["0"] = 2  # type: ignore[index]

    assert "__reduce__" in type(audit).__dict__
    restored_audit = pickle.loads(pickle.dumps(audit, protocol=5))
    assert restored_audit.as_dict() == audit.as_dict()
    with pytest.raises(TypeError):
        restored_audit.counts_by_bin["0"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        restored_audit.checksums_by_bin["0"] = "changed"  # type: ignore[index]


def test_substream_registry_appends_four_and_five_without_renumbering():
    assert fe.SUBSTREAM_CODES == {
        "gate": 1,
        "donor-draw": 2,
        "re-entry-draw": 3,
        "memory-refresh-gate": 4,
        "memory-refresh-rank": 5,
    }


def test_generator_rejects_every_partial_rank_refresh_binding():
    base = _generator()
    complete = {
        "rank_refresh_q": 0.55,
        "stable_pools": _stable_pools(),
        "rank_refresh_fit_audit": _eligible_audit(),
    }
    keys = tuple(complete)
    for supplied in (
        (keys[0],),
        (keys[1],),
        (keys[2],),
        (keys[0], keys[1]),
        (keys[0], keys[2]),
        (keys[1], keys[2]),
    ):
        kwargs = {
            key: value for key, value in complete.items() if key in supplied
        }
        with pytest.raises(ValueError, match="must be bound together"):
            replace(base, **kwargs)


def test_registered_q55_threshold_refreshes_exact_strict_subset():
    operation = CANDIDATE_2.operation(RANK_REFRESH_OPERATION_KIND)
    assert operation is not None
    generator = _with_refresh(
        _generator(), float(operation.params["q"]), rank=0.2
    )

    class FixedRng:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def random(self, size=None):
            assert size == len(self.values)
            return self.values.copy()

    output = np.asarray([0.81, 0.82, 0.83, 0.84], dtype=np.float64)
    generator._draw_rank_refresh(
        np.arange(4, dtype=np.int64),
        np.zeros(4, dtype=bool),
        np.full(4, 0.5, dtype=np.float64),
        np.full(4, 30, dtype=np.int64),
        FixedRng([0.0, 0.549999, 0.55, 0.9]),
        FixedRng([0.1, 0.2, 0.3, 0.4]),
        output,
    )
    assert output.tolist() == [0.2, 0.2, 0.83, 0.84]


def test_candidate1_absent_operation_and_q0_are_bit_equivalent_with_old_states(
    monkeypatch: pytest.MonkeyPatch,
):
    original_substream = fe._substream

    class RecordingRng:
        def __init__(self, rng):
            self.rng = rng
            self.calls: list[int | None] = []

        def random(self, size=None):
            self.calls.append(size)
            return self.rng.random(size)

    def run(generator):
        streams = {}

        def recording_substream(seed, label):
            stream = RecordingRng(original_substream(seed, label))
            streams[label] = stream
            return stream

        monkeypatch.setattr(fe, "_substream", recording_substream)
        parent = np.random.default_rng(20260718)
        frame = _frame(
            (20, 10, 1),
            current=(0.0, 50.0, 50.0),
            lag=(0.0, 50.0, 50.0),
            prior=(40.0, 40.0, 40.0),
        )
        output = generator.generate(frame, 2016, parent)
        states = {
            label: pickle.dumps(stream.rng.bit_generator.state, protocol=5)
            for label, stream in streams.items()
        }
        return output, parent.random(8), streams, states

    incumbent = _generator(
        pools=_pools(
            pair=(0.2, 0.8),
            triple=(0.2, 0.8),
            reentry=(0.2, 0.8),
        )
    )
    incumbent_output, incumbent_parent, incumbent_streams, incumbent_states = (
        run(incumbent)
    )
    assert set(incumbent_streams) == {
        "gate",
        "donor-draw",
        "re-entry-draw",
    }

    refresh_records = {}
    for q in (0.0, 0.55, 1.0):
        refreshed = _with_refresh(
            _generator(
                pools=_pools(
                    pair=(0.2, 0.8),
                    triple=(0.2, 0.8),
                    reentry=(0.2, 0.8),
                )
            ),
            q,
        )
        output, parent_tail, streams, states = run(refreshed)
        refresh_records[q] = (streams, states)
        assert parent_tail.tobytes() == incumbent_parent.tobytes()
        for label in ("gate", "donor-draw", "re-entry-draw"):
            assert states[label] == incumbent_states[label]
        if q == 0:
            assert output.tobytes() == incumbent_output.tobytes()
        assert streams["memory-refresh-gate"].calls == [2]
        assert streams["memory-refresh-rank"].calls == [2]

    for label in ("memory-refresh-gate", "memory-refresh-rank"):
        assert (
            refresh_records[0.0][1][label]
            == refresh_records[0.55][1][label]
            == refresh_records[1.0][1][label]
        )


def test_q0_reproduces_incumbent_full_engine_path_bit_for_bit():
    nawi = {year: 100.0 for year in range(2002, 2019)}
    incumbent = fit_forward_earnings(
        _fit_panel(), nawi, seed=17, qrf_factory=_RecordingQRFFactory()
    )
    q0 = fit_forward_earnings(
        _fit_panel(),
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=_candidate(0.0),
    )

    incumbent_result = _project(incumbent.generator)
    q0_result = _project(q0.generator)
    pd.testing.assert_frame_equal(incumbent_result.panel, q0_result.panel)
    assert _projection_bytes(incumbent_result) == _projection_bytes(q0_result)


def test_candidate_spec_without_refresh_reproduces_candidate1_bytes():
    nawi = {year: 100.0 for year in range(2002, 2019)}
    incumbent = fit_forward_earnings(
        _fit_panel(), nawi, seed=17, qrf_factory=_RecordingQRFFactory()
    )
    no_refresh = fit_forward_earnings(
        _fit_panel(),
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=CandidateSpec(
            candidate_id="synthetic_candidate1_no_engine_delta",
            contract_revision="candidate1_byte_equivalence_fixture",
            operations=(),
        ),
    )
    assert no_refresh.generator.rank_refresh_q is None
    assert no_refresh.generator.stable_pools is None
    assert no_refresh.q_invariant_fit_signature_sha256 is None
    assert _projection_bytes(_project(no_refresh.generator)) == (
        _projection_bytes(_project(incumbent.generator))
    )


def test_fit_signature_and_all_fitted_surfaces_are_q_invariant():
    nawi = {year: 100.0 for year in range(2002, 2019)}
    fitted = [
        fit_forward_earnings(
            _fit_panel(),
            nawi,
            seed=17,
            qrf_factory=_RecordingQRFFactory(),
            candidate_spec=_candidate(q),
        )
        for q in (0.0, 0.55, 1.0)
    ]
    assert len({item.q_invariant_fit_signature_sha256 for item in fitted}) == 1
    for left, right in zip(fitted, fitted[1:], strict=False):
        assert left.generator.pools.keys() == right.generator.pools.keys()
        for name in left.generator.pools:
            for field in left.generator.pools[name]:
                assert np.array_equal(
                    left.generator.pools[name][field],
                    right.generator.pools[name][field],
                )
        assert left.generator.wage_index == right.generator.wage_index
        for bin_index in range(fe.N_AGE_BINS):
            left_cell = left.generator.marginals[bin_index]
            right_cell = right.generator.marginals[bin_index]
            assert left_cell.p0 == right_cell.p0
            assert np.array_equal(left_cell.wtil, right_cell.wtil)
            assert np.array_equal(left_cell.yval, right_cell.yval)

    changed = _fit_panel()
    changed.loc[
        (changed["person_id"] == 2) & (changed["period"] == 2010),
        "earnings",
    ] += 100.0
    perturbed = fit_forward_earnings(
        changed,
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=_candidate(0.55),
    )
    assert (
        perturbed.q_invariant_fit_signature_sha256
        != fitted[1].q_invariant_fit_signature_sha256
    )


def test_fit_partitions_incumbent_pair_order_by_exact_target_age_bin():
    fitted = fit_forward_earnings(
        _fit_panel(),
        {year: 100.0 for year in range(2002, 2019)},
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=CANDIDATE_2,
    )
    positive = fitted.forward_pairs.loc[
        (fitted.forward_pairs["earnings"] > 0)
        & (fitted.forward_pairs["earnings_tp2"] > 0)
    ].copy()
    positive = positive.sort_values(
        ["person_id", "period_tp2"], kind="stable"
    ).reset_index(drop=True)
    bins = fe.age_bin(positive["age_tp2"])
    stable = fitted.generator.stable_pools
    assert stable is not None

    for bin_index in range(fe.N_AGE_BINS):
        expected = positive.loc[bins == bin_index]
        pool = stable[bin_index]
        assert np.array_equal(
            pool["person_id"], expected["person_id"].to_numpy(np.int64)
        )
        assert np.array_equal(
            pool["period_tp2"], expected["period_tp2"].to_numpy(np.int64)
        )
        assert np.array_equal(
            pool["weight"], expected["weight_tp2"].to_numpy(np.float64)
        )


def test_stable_draw_uses_exact_bin_and_asymmetric_donor_coordinate(
    monkeypatch: pytest.MonkeyPatch,
):
    pools = _stable_pools()
    pools[2] = {
        "u_t": np.asarray([0.01, 0.99]),
        "u_tp2": np.asarray([0.2, 0.8]),
        "u_A": np.asarray([0.2, 0.8]),
        "u_w": np.asarray([0.9, 0.1]),
        "weight": np.asarray([2.0, 3.0]),
        "person_id": np.asarray([1, 2], dtype=np.int64),
        "period_tp2": np.asarray([2012, 2014], dtype=np.int64),
    }
    generator = replace(
        _generator(),
        rank_refresh_q=1.0,
        stable_pools=pools,
        rank_refresh_fit_audit=_eligible_audit(),
    )
    captured = []

    def knn(distance, weight, donor_rank, uniforms):
        captured.append(
            (
                np.asarray(distance).copy(),
                np.asarray(weight).copy(),
                np.asarray(donor_rank).copy(),
                np.asarray(uniforms).copy(),
            )
        )
        return np.full(len(distance), donor_rank[0])

    monkeypatch.setattr(fe, "_knn_draw", knn)
    output = np.full(2, np.nan)
    generator._draw_rank_refresh(
        np.asarray([0, 1]),
        np.asarray([False, True]),
        np.asarray([0.4, 0.6]),
        np.asarray([37.0, 37.0]),
        np.random.default_rng(1),
        np.random.default_rng(2),
        output,
    )

    assert len(captured) == 2
    non_q0_coordinate = 0.1 * pools[2]["u_w"] + 0.9 * pools[2]["u_A"]
    assert np.array_equal(
        captured[0][0], np.abs(non_q0_coordinate[None, :] - 0.4)
    )
    assert np.array_equal(
        captured[1][0], np.abs(pools[2]["u_A"][None, :] - 0.6)
    )
    assert captured[0][1].tolist() == [2.0, 3.0]
    assert captured[0][2].tolist() == [0.2, 0.8]
    assert output.tolist() == [0.2, 0.2]


class _ThresholdGate(_RecordingGate):
    def __init__(self, threshold: float) -> None:
        super().__init__(1)
        self.threshold = float(threshold)
        self.levels: list[np.ndarray] = []

    def draw_sign(self, current_level, target_age, uniforms):
        del target_age
        self.uniforms.append(uniforms.copy())
        self.levels.append(current_level.copy())
        return (current_level >= self.threshold).astype(np.int64)


def test_refresh_cannot_change_same_step_but_can_change_later_participation():
    q0_gate = _ThresholdGate(40.0)
    q1_gate = _ThresholdGate(40.0)
    q0 = _with_refresh(
        _generator(
            gate=q0_gate,
            pools=_pools(pair=(0.8,), triple=(0.8,), reentry=(0.5,)),
        ),
        0.0,
        rank=0.2,
    )
    q1 = _with_refresh(
        _generator(
            gate=q1_gate,
            pools=_pools(pair=(0.8,), triple=(0.8,), reentry=(0.5,)),
        ),
        1.0,
        rank=0.2,
    )
    frame = _frame()
    q0_2016 = q0.generate(frame, 2016, np.random.default_rng(9))
    q1_2016 = q1.generate(frame, 2016, np.random.default_rng(9))
    assert q0_gate.levels[0].tolist() == q1_gate.levels[0].tolist() == [50.0]
    assert q0_gate.uniforms[0].tobytes() == q1_gate.uniforms[0].tobytes()
    assert q0_2016.tolist() == [80.0]
    assert q1_2016.tolist() == [20.0]

    def advance(level: np.ndarray) -> pd.DataFrame:
        out = frame.copy()
        out["age"] = 34.0
        out["earnings"] = level
        out["gen_earn_w4"] = frame["gen_earn_w2"]
        out["gen_earn_w2"] = level
        return out

    q0_2018 = q0.generate(advance(q0_2016), 2018, np.random.default_rng(10))
    q1_2018 = q1.generate(advance(q1_2016), 2018, np.random.default_rng(10))
    assert q0_gate.levels[1].tolist() == [80.0]
    assert q1_gate.levels[1].tolist() == [20.0]
    assert q0_2018.tolist() == [80.0]
    assert q1_2018.tolist() == [0.0]


def _empty_last_stable_bin_panel() -> pd.DataFrame:
    panel = _fit_panel()
    rows = panel[panel["person_id"] == 8].sort_values("period").index
    for offset, index in enumerate(rows):
        if offset % 2:
            panel.loc[index, "earnings"] = 0.0
    return panel


def test_empty_exact_bin_is_publishable_nonregisterable_preflight():
    panel = _empty_last_stable_bin_panel()
    nawi = {year: 100.0 for year in range(2002, 2019)}
    fitted = fit_forward_earnings(
        panel,
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=CANDIDATE_2,
    )
    audit = fitted.rank_refresh_fit_audit
    assert audit is not None
    assert audit.empty_bins == (7,)
    assert audit.as_dict()["disposition"] == (
        "NO_REGISTERABLE_EARNINGS_REFRESH_FIT"
    )
    assert not audit.as_dict()["empty_pool_check_passed"]

    with pytest.raises(RankRefreshPreflightAbort) as caught:
        refit_earnings_chained_generator(
            panel,
            nawi,
            seed=17,
            qrf_factory=_RecordingQRFFactory(),
            candidate_spec=CANDIDATE_2,
        )
    assert caught.value.audit.as_dict() == audit.as_dict()
    assert "permits no donor fallback" in str(caught.value)
    restored = pickle.loads(pickle.dumps(caught.value, protocol=5))
    assert isinstance(restored, RankRefreshPreflightAbort)
    assert restored.audit.as_dict() == audit.as_dict()
    assert str(restored) == str(caught.value)


def test_runner_lineage_is_additive_only_for_engine_candidate():
    incumbent_earnings = SimpleNamespace(
        provenance={"fixture": "incumbent"},
    )
    incumbent = m6_runner._refit_lineage(
        M6RefitBundle(boundary_year=2014, earnings=incumbent_earnings)
    )
    assert set(incumbent) == {
        "boundary_year",
        "earnings_spec_registration",
        "earnings_spec_sha256",
        "resolved_spec_sha256s",
        "refit_provenance",
        "certified_full_window_artifacts_read",
        "certified_full_window_artifacts_written",
    }
    assert "engine_candidate" not in incumbent["resolved_spec_sha256s"]

    audit = _eligible_audit().as_dict()
    active_earnings = SimpleNamespace(
        provenance={"fixture": "active"},
        engine_candidate_id=CANDIDATE_2.candidate_id,
        engine_candidate_spec_sha256=CANDIDATE_2.sha256,
        q_invariant_fit_signature_sha256="synthetic-q-invariant-signature",
        rank_refresh_fit_audit=audit,
    )
    active = m6_runner._refit_lineage(
        M6RefitBundle(boundary_year=2014, earnings=active_earnings)
    )
    assert active["resolved_spec_sha256s"]["engine_candidate"] == (
        CANDIDATE_2.sha256
    )
    assert active["engine_candidate_id"] == CANDIDATE_2.candidate_id
    assert active["q_invariant_fit_signature_sha256"] == (
        "synthetic-q-invariant-signature"
    )
    assert active["rank_refresh_fit_audit"] == audit


def test_active_full_projection_is_deterministic_and_changes_incumbent():
    nawi = {year: 100.0 for year in range(2002, 2019)}
    incumbent = fit_forward_earnings(
        _fit_panel(), nawi, seed=17, qrf_factory=_RecordingQRFFactory()
    )
    active = fit_forward_earnings(
        _fit_panel(),
        nawi,
        seed=17,
        qrf_factory=_RecordingQRFFactory(),
        candidate_spec=CANDIDATE_2,
    )
    first = _project(active.generator, draw_index=0)
    second = _project(active.generator, draw_index=0)
    baseline = _project(incumbent.generator, draw_index=0)
    assert _projection_bytes(first) == _projection_bytes(second)
    assert _projection_bytes(first) != _projection_bytes(baseline)


def test_committed_rank_refresh_synthetic_smoke_is_reproducible():
    script = ROOT / "scripts/smoke_m6_rank_refresh_engine.py"
    artifact = ROOT / "docs/analysis/m6_rank_refresh_engine_synthetic.json"
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout == artifact.read_text(encoding="utf-8")
