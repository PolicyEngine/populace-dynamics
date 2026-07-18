#!/usr/bin/env python3
"""Publish deterministic synthetic evidence for the M6 rank-refresh engine.

No staged or realized source is opened.  The happy path traverses the annual
``ProjectionEngine`` with the registered candidate-2 q binding.  The
degenerate path proves that an empty exact-age donor bin becomes a publishable
non-registerable preflight disposition, never a widened-pool fallback.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.engine.candidates import (
    CANDIDATE_2,
    RANK_REFRESH_OPERATION_ID,
    RANK_REFRESH_OPERATION_KIND,
    CandidateSpec,
    OperationSpec,
)
from populace_dynamics.engine.forward_earnings import (
    RankRefreshPreflightAbort,
)
from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodModules,
    ProjectionEngine,
)
from populace_dynamics.engine.refit import (
    refit_earnings_chained_generator,
)
from populace_dynamics.engine.steps import apply_earnings


class _AlwaysParticipateGate:
    def draw_sign(
        self,
        current_level: np.ndarray,
        target_age: np.ndarray,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        if current_level.shape != target_age.shape or uniforms.shape != (
            len(current_level),
        ):
            raise AssertionError("synthetic gate received misaligned inputs")
        return np.ones(len(current_level), dtype=np.int64)


class _SyntheticQRFFactory:
    def __call__(self, *, seed: int):
        del seed

        class Model:
            def fit(self, frame: pd.DataFrame, **kwargs: Any):
                del frame, kwargs
                return _AlwaysParticipateGate()

        return Model()


def _candidate(q: float) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=f"synthetic_rank_refresh_q_{q}",
        contract_revision="engine_q_equivalence_smoke",
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


def _panel(*, empty_last_bin: bool = False) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    periods = tuple(range(2002, 2016, 2))
    for bin_index, age in enumerate(range(27, 67, 5)):
        person_id = bin_index + 1
        for wave_index, period in enumerate(periods):
            earnings = float(10_000 + 1_000 * bin_index + 175 * wave_index**2)
            if empty_last_bin and bin_index == 7 and wave_index % 2:
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
    return pd.DataFrame(rows)


def _nawi() -> dict[int, float]:
    return {year: 100.0 for year in range(2002, 2019)}


def _initial_slice(panel: pd.DataFrame) -> pd.DataFrame:
    anchor = panel[panel["period"] == 2014].copy()
    return pd.DataFrame(
        {
            "person_id": anchor["person_id"].to_numpy(dtype=np.int64),
            "year": np.full(len(anchor), 2014, dtype=np.int64),
            "age": anchor["age"].to_numpy(dtype=np.int64),
            "sex": ["female"] * len(anchor),
            "weight": anchor["weight"].to_numpy(dtype=np.float64),
        }
    )


def _project(generator, initial: pd.DataFrame, *, draw_index: int):
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
        initial, end_year=2018, draw_index=draw_index
    )


def _ordered_panel(result) -> pd.DataFrame:
    return result.panel.sort_values(
        ["person_id", "year"], kind="stable"
    ).reset_index(drop=True)


def _projection_bytes(result) -> bytes:
    panel = _ordered_panel(result)
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


def _fit(panel: pd.DataFrame, candidate_spec: CandidateSpec | None):
    return refit_earnings_chained_generator(
        panel,
        _nawi(),
        seed=17,
        qrf_factory=_SyntheticQRFFactory(),
        candidate_spec=candidate_spec,
    )


def main() -> int:
    panel = _panel()
    initial = _initial_slice(panel)
    incumbent = _fit(panel, None)
    q0 = _fit(panel, _candidate(0.0))
    active = _fit(panel, CANDIDATE_2)

    incumbent_result = _project(incumbent.generator, initial, draw_index=0)
    q0_result = _project(q0.generator, initial, draw_index=0)
    first = _project(active.generator, initial, draw_index=0)
    second = _project(active.generator, initial, draw_index=0)
    incumbent_bytes = _projection_bytes(incumbent_result)
    q0_bytes = _projection_bytes(q0_result)
    first_bytes = _projection_bytes(first)
    second_bytes = _projection_bytes(second)
    active_panel = _ordered_panel(first)
    incumbent_panel = _ordered_panel(incumbent_result)
    changed = active_panel["earnings"].to_numpy(dtype=np.float64) != (
        incumbent_panel["earnings"].to_numpy(dtype=np.float64)
    )
    if incumbent_bytes != q0_bytes:
        raise AssertionError("q=0 failed incumbent bit equivalence")
    if first_bytes != second_bytes:
        raise AssertionError(
            "active rank-refresh projection is not deterministic"
        )
    if not changed.any():
        raise AssertionError(
            "registered q did not activate on synthetic smoke"
        )

    degenerate_projection_calls = 0
    try:
        _fit(_panel(empty_last_bin=True), CANDIDATE_2)
    except RankRefreshPreflightAbort as error:
        degenerate_audit = error.audit.as_dict()
        degenerate_message = str(error)
    else:
        raise AssertionError("empty stable pool did not stop preflight")

    operation = CANDIDATE_2.operation(RANK_REFRESH_OPERATION_KIND)
    if operation is None:
        raise AssertionError("candidate 2 lost its refresh operation")
    payload = {
        "schema": "m6.rank_refresh_engine.synthetic_smoke.v1",
        "status": "PASS",
        "synthetic_only": True,
        "real_data_projection_run": False,
        "candidate_2_binding": {
            "candidate_id": CANDIDATE_2.candidate_id,
            "candidate_spec_sha256": CANDIDATE_2.sha256,
            "operation": operation.canonical_dict(),
        },
        "happy_path": {
            "full_annual_projection_engine": True,
            "draw_index": 0,
            "q": operation.params["q"],
            "fit_signature_sha256": (active.q_invariant_fit_signature_sha256),
            "exact_age_bin_counts": active.rank_refresh_fit_audit[
                "counts_by_bin"
            ],
            "changed_person_period_rows_vs_incumbent": int(changed.sum()),
            "refresh_active": True,
            "deterministic_replay": first_bytes == second_bytes,
            "projection_sha256": hashlib.sha256(first_bytes).hexdigest(),
        },
        "equivalence": {
            "q0_incumbent_bit_equivalent": incumbent_bytes == q0_bytes,
            "candidate1_operation_absent": (
                incumbent.engine_candidate_id is None
            ),
            "candidate1_projection_sha256": hashlib.sha256(
                incumbent_bytes
            ).hexdigest(),
            "q0_projection_sha256": hashlib.sha256(q0_bytes).hexdigest(),
        },
        "degenerate_empty_pool": {
            "status": "NO_REGISTERABLE_EARNINGS_REFRESH_FIT_PUBLISHED",
            "published": True,
            "preflight_exception": RankRefreshPreflightAbort.__name__,
            "message": degenerate_message,
            "audit": degenerate_audit,
            "fallback": "NONE_BY_RATIFIED_LAW",
            "adjacent_or_pooled_donor_used": False,
            "projection_calls": degenerate_projection_calls,
            "candidate_artifact_write_calls": 0,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
