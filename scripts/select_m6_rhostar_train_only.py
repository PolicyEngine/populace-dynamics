#!/usr/bin/env python3
"""Run the amendment-6 train-only correlated-refresh rho selector.

The runner extends the frozen q-star selector machinery at fixed ``q*=0.55``.
It reads no gate or candidate scored artifact and writes nothing below
``runs/``.  Full mode executes every section 2.7.8.6 preflight before it
creates or reduces a rho rung; ``--preflight-only`` stops at that same pass
gate without computing a ladder value.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import select_m6_qstar_train_only as parent

from populace_dynamics.engine import forward_earnings as fe
from populace_dynamics.engine.candidates import CANDIDATE_2
from populace_dynamics.engine.earnings_domain import (
    EARNINGS_CHAIN_STATE_COLUMNS,
)
from populace_dynamics.engine.refit import (
    EarningsChainedRefit,
    refit_earnings_chained_generator,
    truncate_estimation_frame,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
QSTAR_LEDGER_PATH = (
    ROOT / "docs/analysis/m6_qstar_train_only_selection_results.json"
)
QSTAR_LEDGER_SHA256 = (
    "d25b8e159384f8a84ed7f2218d863ca63d96fc9cb244536853b0a1f05c4025bb"
)
PROGRESS_PATH = (
    ROOT / "docs/analysis/m6_rhostar_train_only_selection_progress.txt"
)
FINDINGS_PATH = (
    ROOT / "docs/analysis/m6_rhostar_train_only_selection_results.json"
)
FINDINGS_TMP_PATH = Path(f"{FINDINGS_PATH}.tmp")

RAW_SCHEMA = "m6_rhostar_train_only_selection.v1"
PREFLIGHT_SCHEMA = "m6_rhostar_train_only_selection.preflight.v1"
FIXED_Q = 0.55
RHO_GRID = tuple(round(-0.80 + 0.05 * index, 2) for index in range(17))
PSEUDO_BOUNDARIES = parent.PSEUDO_BOUNDARIES
FIT_SEED = parent.FIT_SEED
SELECTION_DRAW_SEEDS = parent.SELECTION_DRAW_SEEDS
FIRST_HALF_DRAW_SEEDS = parent.FIRST_HALF_DRAW_SEEDS
SECOND_HALF_DRAW_SEEDS = parent.SECOND_HALF_DRAW_SEEDS
SELECTED_CELLS = parent.SELECTED_CELLS
OBJECTIVE_CELLS = parent.OBJECTIVE_CELLS
FEASIBILITY_CELLS = parent.FEASIBILITY_CELLS
SUBSTREAM_CODES = dict(fe.SUBSTREAM_CODES)

THREAD_ENVIRONMENT_KEYS = (
    "LOKY_MAX_CPU_COUNT",
    "POPULACE_FIT_N_JOBS",
    "POPULACE_FIT_PREDICT_WORKERS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
EXPECTED_THREAD_ENVIRONMENT = {key: "8" for key in THREAD_ENVIRONMENT_KEYS}
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.14.4",
    "numpy": "2.5.1",
    "pandas": "3.0.3",
    "scikit_learn": "1.8.0",
    "scipy": "1.18.0",
    "quantile_forest": "1.4.2",
    "populace_fit": "0.1.0",
    "populace_frame": "0.1.0",
    "policyengine_us": "1.752.2",
    "policyengine_core": "3.30.3",
}
EXPECTED_POPULACE_HEAD = "ee8f7fc139271de5d4e448549c35e8c5eb992534"
EXPECTED_POPULACE_FIT_TREE = "5c866378fdf5906b7a61da9977b8d028d1d36e9f"
EXPECTED_POPULACE_FRAME_TREE = "7cfb9ee78beb74911963913f202a4471aae2f52b"
EXPECTED_OLD_FRAME_COLUMNS = (
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
EXPECTED_OLD_CHAIN_COLUMNS = (
    "u_w",
    "realized_earn_2014",
    "realized_earn_2012",
    "gen_earn_w2",
    "gen_earn_w4",
)
GENERATED_OUTPUT_PATHS = frozenset(
    path.relative_to(ROOT).as_posix()
    for path in (PROGRESS_PATH, FINDINGS_PATH, FINDINGS_TMP_PATH)
)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_qstar_ledger() -> tuple[dict[str, Any], bytes]:
    raw = QSTAR_LEDGER_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != QSTAR_LEDGER_SHA256:
        raise RuntimeError(f"q-star ledger hash drifted: {digest}")
    ledger = json.loads(raw)
    if ledger.get("schema") != "m6_qstar_train_only_selection.findings.v1":
        raise RuntimeError("q-star ledger schema drifted")
    if float(ledger["selector"]["selected_q"]) != FIXED_Q:
        raise RuntimeError("q-star ledger no longer selects the frozen q")
    return ledger, raw


def _editable_source(package: str, tree_path: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(package)
    direct_text = distribution.read_text("direct_url.json")
    if direct_text is None:
        raise RuntimeError(f"{package} is not installed editable")
    direct = json.loads(direct_text)
    if not direct.get("dir_info", {}).get("editable"):
        raise RuntimeError(f"{package} is not installed editable")
    source = Path(unquote(urlparse(direct["url"]).path)).resolve()
    repository = Path(_git(source, "rev-parse", "--show-toplevel"))
    status = _git(repository, "status", "--porcelain", "--", tree_path)
    if status:
        raise RuntimeError(f"{package} editable source is dirty:\n{status}")
    return {
        "version": distribution.version,
        "editable_source": str(source),
        "direct_url": direct,
        "repository_root": str(repository),
        "repository_head": _git(repository, "rev-parse", "HEAD"),
        "repository_branch": _git(repository, "branch", "--show-current"),
        "source_tree_sha1": _git(repository, "rev-parse", f"HEAD:{tree_path}"),
        "tracked_source_clean": True,
    }


def _repository_freeze() -> dict[str, Any]:
    status_lines = _git(
        ROOT, "status", "--porcelain", "--untracked-files=all"
    ).splitlines()
    allowed = []
    rejected = []
    for line in status_lines:
        path = line[3:]
        if line.startswith("?? ") and path in GENERATED_OUTPUT_PATHS:
            allowed.append(path)
        else:
            rejected.append(line)
    if rejected:
        raise RuntimeError(
            "rho selector requires committed source bytes; worktree has:\n"
            + "\n".join(rejected)
        )

    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "quantile_forest": importlib.metadata.version("quantile-forest"),
        "populace_fit": importlib.metadata.version("populace-fit"),
        "populace_frame": importlib.metadata.version("populace-frame"),
        "policyengine_us": importlib.metadata.version("policyengine-us"),
        "policyengine_core": importlib.metadata.version("policyengine-core"),
    }
    if runtime != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(
            f"gate runtime {runtime} differs from {EXPECTED_RUNTIME_VERSIONS}"
        )
    thread_environment = {
        key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS
    }
    if thread_environment != EXPECTED_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "thread environment must be the exact eight-variable 8-thread "
            f"block: {thread_environment}"
        )

    fit_source = _editable_source(
        "populace-fit", "packages/populace-fit/src/populace/fit"
    )
    frame_source = _editable_source(
        "populace-frame", "packages/populace-frame/src/populace/frame"
    )
    for record in (fit_source, frame_source):
        if record["repository_head"] != EXPECTED_POPULACE_HEAD:
            raise RuntimeError("editable populace repository HEAD drifted")
    if fit_source["source_tree_sha1"] != EXPECTED_POPULACE_FIT_TREE:
        raise RuntimeError("editable populace-fit source tree drifted")
    if frame_source["source_tree_sha1"] != EXPECTED_POPULACE_FRAME_TREE:
        raise RuntimeError("editable populace-frame source tree drifted")

    frozen_paths = (
        "scripts/select_m6_rhostar_train_only.py",
        "scripts/reduce_m6_rhostar_selection.py",
        "tests/test_m6_rhostar_selection.py",
        "tests/test_m6_engine_correlated_refresh.py",
        "docs/analysis/m6_rhostar_train_only_selection.md",
    )
    head = _git(ROOT, "rev-parse", "HEAD")
    return {
        "freeze_commit": head,
        "branch": _git(ROOT, "branch", "--show-current"),
        "worktree_clean_except_generated_outputs": True,
        "allowed_generated_outputs_present": allowed,
        "frozen_blob_sha1": {
            path: _git(ROOT, "rev-parse", f"HEAD:{path}")
            for path in frozen_paths
        },
        "source_tree_sha1": _git(
            ROOT, "rev-parse", "HEAD:src/populace_dynamics"
        ),
        "runtime": runtime,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "thread_environment": thread_environment,
        "editable_fitting_stack": {
            "populace_fit": fit_source,
            "populace_frame": frame_source,
        },
    }


CHAIN_KEYS = (
    "empty_to_0",
    "empty_to_1",
    "0_to_0",
    "0_to_1",
    "1_to_0",
    "1_to_1",
    "nonparticipation",
    "zero_earnings",
    "stream3_reentry",
    "support_exit",
)


@dataclass
class _TraceAccumulator:
    digest: Any = field(default_factory=hashlib.sha256)
    n_calls: int = 0
    n_even_calls: int = 0
    n_odd_calls: int = 0
    all_even_stream_sets_exact: bool = True
    all_odd_calls_stream_free: bool = True

    def add(self, record: Mapping[str, Any]) -> None:
        raw = parent._canonical_bytes(record)
        self.digest.update(len(raw).to_bytes(8, "little"))
        self.digest.update(raw)
        self.n_calls += 1
        if int(record["year"]) % fe.PERIOD_STEP:
            self.n_odd_calls += 1
            self.all_odd_calls_stream_free &= not bool(
                record["stream_final_state_sha256"]
            )
        else:
            self.n_even_calls += 1
            self.all_even_stream_sets_exact &= set(
                record["stream_final_state_sha256"]
            ) == set(SUBSTREAM_CODES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.digest.hexdigest(),
            "n_calls": self.n_calls,
            "n_even_calls": self.n_even_calls,
            "n_odd_calls": self.n_odd_calls,
            "all_even_stream_sets_exact": self.all_even_stream_sets_exact,
            "all_odd_calls_stream_free": self.all_odd_calls_stream_free,
        }


@dataclass
class _ChainCounter:
    counts: Counter = field(default_factory=Counter)

    def add(
        self,
        frame: pd.DataFrame,
        year: int,
        earnings: np.ndarray,
        updates: Mapping[str, np.ndarray],
    ) -> None:
        if int(year) % fe.PERIOD_STEP:
            return
        if len(frame) != 1:
            raise AssertionError(
                "rho projection must call one person at a time"
            )
        age = float(frame.iloc[0]["age"])
        lag = float(frame.iloc[0]["gen_earn_w2"])
        output = float(earnings[0])
        previous = float(frame.iloc[0][fe.REFRESH_STATE_COLUMN])
        next_state = float(updates[fe.REFRESH_STATE_COLUMN][0])
        self.counts["even_calls"] += 1

        if age < fe.AGE_MIN or age > fe.AGE_MAX:
            category = "support_exit"
        elif output <= 0.0 and lag > 0.0:
            category = "nonparticipation"
        elif output <= 0.0:
            category = "zero_earnings"
        elif lag <= 0.0:
            category = "stream3_reentry"
        else:
            self.counts["eligible_transitions"] += 1
            if np.isnan(next_state):
                raise AssertionError("eligible transition did not carry state")
            realized = int(next_state)
            if np.isnan(previous):
                category = f"empty_to_{realized}"
            else:
                category = f"{int(previous)}_to_{realized}"
            self.counts[category] += 1
            return

        self.counts[category] += 1
        if not np.isnan(next_state):
            raise AssertionError(f"gap {category} did not reset refresh state")

    def as_dict(self) -> dict[str, Any]:
        stationary = {
            key: int(self.counts[key]) for key in ("empty_to_0", "empty_to_1")
        }
        pairs = {
            key: int(self.counts[key])
            for key in ("0_to_0", "0_to_1", "1_to_0", "1_to_1")
        }
        resets = {
            key: int(self.counts[key])
            for key in (
                "nonparticipation",
                "zero_earnings",
                "stream3_reentry",
                "support_exit",
            )
        }
        eligible = int(self.counts["eligible_transitions"])
        entries_and_pairs = sum(stationary.values()) + sum(pairs.values())
        even_calls = int(self.counts["even_calls"])
        conservation = eligible + sum(resets.values()) == even_calls
        return {
            "even_calls": even_calls,
            "eligible_transitions": eligible,
            "stationary_entries": stationary,
            "consecutive_pairs": pairs,
            "resets": resets,
            "eligible_decomposition_conserves": entries_and_pairs == eligible,
            "even_call_conservation_passed": conservation,
        }


def _call_with_trace(
    call: Callable[[], Any],
    frame: pd.DataFrame,
    year: int,
    rng: np.random.Generator,
    accumulator: _TraceAccumulator,
) -> Any:
    captured: dict[str, np.random.Generator] = {}
    seeds: dict[str, int] = {}
    original = fe._substream

    def traced(seed: int, label: str) -> np.random.Generator:
        child = original(seed, label)
        captured[label] = child
        seeds[label] = int(seed)
        return child

    before = parent._rng_state_checksum(rng)
    fe._substream = traced
    try:
        result = call()
    finally:
        fe._substream = original
    after = parent._rng_state_checksum(rng)
    if int(year) % fe.PERIOD_STEP:
        if captured or before != after:
            raise AssertionError("odd-year correlated path consumed RNG")
        bridge_seed = None
    else:
        if set(captured) != set(SUBSTREAM_CODES):
            raise AssertionError("even-year stream registry use changed")
        if len(set(seeds.values())) != 1:
            raise AssertionError(
                "earnings streams did not share one bridge seed"
            )
        bridge_seed = next(iter(seeds.values()))
    accumulator.add(
        {
            "person_id": int(frame.iloc[0]["person_id"]),
            "year": int(year),
            "parent_state_before_sha256": before,
            "parent_state_after_sha256": after,
            "parent_bridge_seed": bridge_seed,
            "stream_final_state_sha256": {
                label: parent._rng_state_checksum(child)
                for label, child in sorted(captured.items())
            },
        }
    )
    return result


@dataclass
class _AuditedGenerator:
    base: Any
    trace_streams: bool = False
    trace: _TraceAccumulator = field(default_factory=_TraceAccumulator)
    chain: _ChainCounter = field(default_factory=_ChainCounter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def earnings_frame_update_columns(self) -> tuple[str, ...]:
        return tuple(self.base.earnings_frame_update_columns)

    def materialize_initial_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.base.materialize_initial_frame(frame)

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray:
        def call():
            return self.base.generate(frame, year, rng)

        if self.trace_streams:
            return np.asarray(
                _call_with_trace(call, frame, year, rng, self.trace)
            )
        return np.asarray(call())

    def generate_with_frame_updates(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> fe.EarningsGenerationResult:
        def call():
            return self.base.generate_with_frame_updates(frame, year, rng)

        result = (
            _call_with_trace(call, frame, year, rng, self.trace)
            if self.trace_streams
            else call()
        )
        self.chain.add(
            frame,
            year,
            np.asarray(result.earnings),
            result.frame_updates,
        )
        return result


def _production_stable_pool_proof(
    fitted: EarningsChainedRefit,
    script_pools: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    production = fitted.generator.stable_pools
    if production is None:
        raise AssertionError("candidate-2 fit omitted stable donor pools")
    if set(production) != set(script_pools):
        raise AssertionError("production and selector stable bins differ")
    for bin_index in production:
        if set(production[bin_index]) != set(script_pools[bin_index]):
            raise AssertionError("stable donor fields differ")
        for name in production[bin_index]:
            if not np.array_equal(
                np.asarray(production[bin_index][name]),
                np.asarray(script_pools[bin_index][name]),
            ):
                raise AssertionError(
                    "production and selector stable pools differ"
                )


def _projection_record(
    scored: pd.DataFrame,
    annual: pd.DataFrame,
    boundary: int,
    draw_seed: int,
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    cells = parent._selected_cells(scored, boundary)
    return {
        "draw_seed": int(draw_seed),
        "moments": cells,
        "moment_values": {
            name: parent._selection_cell_value(record)
            for name, record in cells.items()
        },
        "support_ids_sha256": parent._key_checksum(scored),
        "annual_level_sha256": parent._level_checksum(annual),
        "annual_participation_sha256": parent._participation_checksum(annual),
        "transition_chain": dict(chain),
        "fresh_initial_state": True,
    }


class _SyntheticGate:
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)
        self.levels: list[float] = []
        self.uniforms: list[float] = []

    def draw_sign(self, current_level, target_age, uniforms):
        del target_age
        self.levels.extend(float(value) for value in current_level)
        self.uniforms.extend(float(value) for value in uniforms)
        return (current_level >= self.threshold).astype(np.int64)


class _FixedStream:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self, size=None):
        if size is None:
            return self.value
        return np.full(int(size), self.value, dtype=np.float64)


def _synthetic_generator(rho: float, gate: Any) -> fe.ForwardEarningsGenerator:
    cell = fe.CellMarginal(
        p0=0.2,
        wtil=np.asarray([0.1, 0.2, 0.5, 0.8, 0.9]),
        yval=np.asarray([10.0, 20.0, 50.0, 80.0, 90.0]),
        n_pos=5,
        w_total=5.0,
    )

    def transition(rank: float, *, triple: bool) -> dict[str, np.ndarray]:
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

    reentry = transition(0.5, triple=False)
    reentry.pop("u_t")
    stable = {
        index: transition(0.2, triple=False) for index in range(fe.N_AGE_BINS)
    }
    audit = fe.RankRefreshFitAudit(
        source="synthetic positive-pair pool",
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
    return fe.ForwardEarningsGenerator(
        shared_gate=gate,
        zero_anchor_gate=gate,
        marginals={index: cell for index in range(fe.N_AGE_BINS)},
        pools={
            "pairs": transition(0.8, triple=False),
            "triples": transition(0.8, triple=True),
            "reentry": reentry,
        },
        wage_index=fe.ProjectedWageIndex(
            actual={2012: 1.0, 2014: 1.0},
            intercept=0.0,
            slope=0.0,
        ),
        u_w_by_person={1: 0.5},
        realized_earn_2014_by_person={1: 50.0},
        realized_earn_2012_by_person={1: 40.0},
        rank_refresh_q=FIXED_Q,
        stable_pools=stable,
        rank_refresh_fit_audit=audit,
        rank_refresh_rho=float(rho),
    )


def _synthetic_frame(
    *,
    state: float = np.nan,
    age: float = 32.0,
    current: float = 50.0,
    lag: float = 50.0,
    prior: float = 40.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1],
            "age": [age],
            "sex": ["selection_schema_only"],
            "u_w": [0.5],
            "realized_earn_2014": [50.0],
            "realized_earn_2012": [40.0],
            "earnings": [current],
            "gen_earn_w2": [lag],
            "gen_earn_w4": [prior],
            fe.REFRESH_STATE_COLUMN: [state],
        }
    )


def _advance_synthetic(
    frame: pd.DataFrame, result: fe.EarningsGenerationResult
) -> pd.DataFrame:
    out = frame.copy()
    out["age"] = out["age"].to_numpy(dtype=np.float64) + 2.0
    out["gen_earn_w4"] = frame["gen_earn_w2"].to_numpy(dtype=np.float64)
    out["gen_earn_w2"] = result.earnings
    out["earnings"] = result.earnings
    out[fe.REFRESH_STATE_COLUMN] = result.frame_updates[
        fe.REFRESH_STATE_COLUMN
    ]
    return out


def _reset_law_preflight() -> dict[str, Any]:
    rho = RHO_GRID[0]
    state = np.asarray([np.nan, 1.0, 0.0])
    thresholds = fe._correlated_refresh_threshold(FIXED_Q, rho, state)
    expected = np.asarray(
        [
            FIXED_Q,
            FIXED_Q + rho * (1.0 - FIXED_Q),
            FIXED_Q * (1.0 - rho),
        ]
    )
    if thresholds.tobytes() != expected.tobytes():
        raise AssertionError("correlated threshold formula drifted")

    original_substream = fe._substream
    histories = []
    for prior, uniform in ((0, 0.80), (1, 0.30)):
        gate = _SyntheticGate(100.0)
        generator = _synthetic_generator(rho, gate)
        frame = _synthetic_frame(state=float(prior))

        def fixed_substream(
            seed: int, label: str, refresh_uniform: float = uniform
        ):
            del seed
            values = {
                "gate": 0.0,
                "donor-draw": 0.0,
                "re-entry-draw": 0.0,
                "memory-refresh-gate": refresh_uniform,
                "memory-refresh-rank": 0.0,
            }
            return _FixedStream(values[label])

        fe._substream = fixed_substream
        try:
            gap = generator.generate_with_frame_updates(
                frame, 2016, np.random.default_rng(1)
            )
            frame = _advance_synthetic(frame, gap)
            gate.threshold = 0.0
            reentry = generator.generate_with_frame_updates(
                frame, 2018, np.random.default_rng(2)
            )
            frame = _advance_synthetic(frame, reentry)
            post_gap = generator.generate_with_frame_updates(
                frame, 2020, np.random.default_rng(3)
            )
            remembered_frame = frame.copy()
            remembered_frame[fe.REFRESH_STATE_COLUMN] = float(prior)
            remembered = generator.generate_with_frame_updates(
                remembered_frame, 2020, np.random.default_rng(3)
            )
        finally:
            fe._substream = original_substream

        gap_state = float(gap.frame_updates[fe.REFRESH_STATE_COLUMN][0])
        reentry_state = float(
            reentry.frame_updates[fe.REFRESH_STATE_COLUMN][0]
        )
        if gap.earnings[0] != 0.0 or not np.isnan(gap_state):
            raise AssertionError(
                "non-participation did not reset refresh state"
            )
        if reentry.earnings[0] <= 0.0 or not np.isnan(reentry_state):
            raise AssertionError("stream-3 re-entry did not retain null state")

        post_gap_threshold = float(
            fe._correlated_refresh_threshold(
                FIXED_Q,
                rho,
                frame[fe.REFRESH_STATE_COLUMN].to_numpy(dtype=np.float64),
            )[0]
        )
        prior_threshold = float(
            fe._correlated_refresh_threshold(
                FIXED_Q, rho, np.asarray([float(prior)])
            )[0]
        )
        post_gap_refresh = bool(
            post_gap.frame_updates[fe.REFRESH_STATE_COLUMN][0]
        )
        remembered_refresh = bool(
            remembered.frame_updates[fe.REFRESH_STATE_COLUMN][0]
        )
        if post_gap_refresh == remembered_refresh:
            raise AssertionError("reset fixture does not discriminate its gap")
        if post_gap_refresh != (uniform < post_gap_threshold):
            raise AssertionError(
                "post-gap draw did not use the null threshold"
            )
        if remembered_refresh != (uniform < prior_threshold):
            raise AssertionError("counterfactual did not use remembered state")
        histories.append(
            {
                "synthetic_person": 1,
                "prior_realized_state": prior,
                "gap_type": "nonparticipation_then_stream3_reentry",
                "gap_level": float(gap.earnings[0]),
                "gap_resets_to": None,
                "reentry_level": float(reentry.earnings[0]),
                "reentry_state": None,
                "post_gap_uniform": uniform,
                "post_gap_threshold": post_gap_threshold,
                "remembered_state_threshold": prior_threshold,
                "post_gap_refresh": post_gap_refresh,
                "remembered_state_refresh": remembered_refresh,
                "discriminating": True,
            }
        )
    return {
        "passed": True,
        "rho": rho,
        "thresholds": {
            "empty": float(thresholds[0]),
            "one": float(thresholds[1]),
            "zero": float(thresholds[2]),
        },
        "histories": histories,
        "gap_classes_covered_by_guard_tests": [
            "nonparticipation",
            "zero_earnings",
            "stream3_reentry",
            "support_exit",
        ],
    }


def _participation_feedback_preflight() -> dict[str, Any]:
    original_substream = fe._substream

    def fixed_substream(seed: int, label: str):
        del seed
        values = {
            "gate": 0.0,
            "donor-draw": 0.0,
            "re-entry-draw": 0.0,
            "memory-refresh-gate": 0.80,
            "memory-refresh-rank": 0.0,
        }
        return _FixedStream(values[label])

    iid_gate = _SyntheticGate(40.0)
    negative_gate = _SyntheticGate(40.0)
    iid = _synthetic_generator(0.0, iid_gate)
    negative = _synthetic_generator(RHO_GRID[0], negative_gate)
    iid_frame = _synthetic_frame()
    negative_frame = _synthetic_frame()
    fe._substream = fixed_substream
    try:
        iid_2016 = iid.generate_with_frame_updates(
            iid_frame, 2016, np.random.default_rng(1)
        )
        negative_2016 = negative.generate_with_frame_updates(
            negative_frame, 2016, np.random.default_rng(1)
        )
        iid_frame = _advance_synthetic(iid_frame, iid_2016)
        negative_frame = _advance_synthetic(negative_frame, negative_2016)
        iid_2018 = iid.generate_with_frame_updates(
            iid_frame, 2018, np.random.default_rng(2)
        )
        negative_2018 = negative.generate_with_frame_updates(
            negative_frame, 2018, np.random.default_rng(2)
        )
        iid_frame = _advance_synthetic(iid_frame, iid_2018)
        negative_frame = _advance_synthetic(negative_frame, negative_2018)
        iid_2020 = iid.generate_with_frame_updates(
            iid_frame, 2020, np.random.default_rng(3)
        )
        negative_2020 = negative.generate_with_frame_updates(
            negative_frame, 2020, np.random.default_rng(3)
        )
    finally:
        fe._substream = original_substream

    conditions = {
        "first_transition_levels_equal": (
            iid_2016.earnings.tobytes() == negative_2016.earnings.tobytes()
        ),
        "same_step_2018_gate_inputs_equal": (
            iid_gate.levels[1] == negative_gate.levels[1]
        ),
        "later_refresh_changes_carried_level": (
            iid_2018.earnings.tobytes() != negative_2018.earnings.tobytes()
        ),
        "unchanged_2020_gate_reads_changed_level": (
            iid_gate.levels[2] != negative_gate.levels[2]
        ),
        "later_participation_differs": (
            (iid_2020.earnings > 0).tobytes()
            != (negative_2020.earnings > 0).tobytes()
        ),
        "gate_uniforms_equal_at_every_shared_call": (
            iid_gate.uniforms == negative_gate.uniforms
        ),
    }
    if not all(conditions.values()):
        raise AssertionError(
            f"participation feedback fixture failed: {conditions}"
        )
    return {
        "passed": True,
        "conditions": conditions,
        "levels": {
            "rho0": [
                float(iid_2016.earnings[0]),
                float(iid_2018.earnings[0]),
                float(iid_2020.earnings[0]),
            ],
            "negative_rho": [
                float(negative_2016.earnings[0]),
                float(negative_2018.earnings[0]),
                float(negative_2020.earnings[0]),
            ],
        },
    }


def _source_sha256(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _object_level_preflight(
    fitted: EarningsChainedRefit,
    context: parent.BoundaryContext,
) -> dict[str, Any]:
    candidate2 = fitted.generator
    proof_rho = RHO_GRID[0]
    prototype = replace(candidate2, rank_refresh_rho=proof_rho)
    identities = {
        "shared_participation_gate": (
            prototype.shared_gate is candidate2.shared_gate
        ),
        "zero_anchor_participation_gate": (
            prototype.zero_anchor_gate is candidate2.zero_anchor_gate
        ),
        "cell_marginals": prototype.marginals is candidate2.marginals,
        "incumbent_donor_pools": prototype.pools is candidate2.pools,
        "stable_donor_pools": (
            prototype.stable_pools is candidate2.stable_pools
        ),
        "projected_wage_index": (
            prototype.wage_index is candidate2.wage_index
        ),
        "permanent_rank_map": (
            prototype.u_w_by_person is candidate2.u_w_by_person
        ),
        "fit_audit": (
            prototype.rank_refresh_fit_audit
            is candidate2.rank_refresh_fit_audit
        ),
    }
    if not all(identities.values()):
        raise AssertionError(
            f"rho clone changed a fitted object: {identities}"
        )

    realized_anchor = context.initial_slice["person_id"].map(
        candidate2.realized_earn_2014_by_person
    )
    initial = (
        context.initial_slice.loc[
            context.initial_slice["age"].between(fe.AGE_MIN, fe.AGE_MAX - 2)
            & (realized_anchor > 0.0)
        ]
        .iloc[:16]
        .copy()
    )
    if initial.empty:
        raise AssertionError("object proof has no in-support anchor row")
    incumbent_frame = candidate2.materialize_initial_frame(initial)
    prototype_frame = prototype.materialize_initial_frame(initial)
    pd.testing.assert_frame_equal(
        prototype_frame[incumbent_frame.columns], incumbent_frame
    )
    if not prototype_frame[fe.REFRESH_STATE_COLUMN].isna().all():
        raise AssertionError("refresh state did not initialize to null")
    if fe.REFRESH_STATE_COLUMN in incumbent_frame:
        raise AssertionError("candidate-2 frame acquired candidate-3 state")
    if tuple(fe.FRAME_COLUMNS) != EXPECTED_OLD_FRAME_COLUMNS:
        raise AssertionError("incumbent forward frame columns changed")
    if tuple(EARNINGS_CHAIN_STATE_COLUMNS) != EXPECTED_OLD_CHAIN_COLUMNS:
        raise AssertionError("incumbent lag-shift columns changed")

    rho_zero = replace(candidate2, rank_refresh_rho=0.0)
    incumbent_model = parent.wrap_earnings_domain(candidate2)
    rho_zero_model = parent.wrap_earnings_domain(rho_zero)
    incumbent_state = incumbent_model.materialize_initial_frame(
        initial.iloc[[0]].copy()
    )
    rho_zero_state = rho_zero_model.materialize_initial_frame(
        initial.iloc[[0]].copy()
    )
    incumbent_state_parent = np.random.default_rng(780)
    rho_zero_state_parent = np.random.default_rng(780)
    for period_index, year in enumerate(
        range(context.boundary + 1, context.boundary + 5), start=1
    ):
        period_context = parent.PeriodContext(
            period_index=period_index,
            year=year,
            draw_index=0,
            metadata={},
        )
        incumbent_state = incumbent_state.copy()
        rho_zero_state = rho_zero_state.copy()
        for frame in (incumbent_state, rho_zero_state):
            frame["age"] = frame["age"].to_numpy(dtype=np.float64) + 1.0
            frame["year"] = year
        incumbent_state = parent.apply_earnings(
            incumbent_state,
            period_context,
            incumbent_state_parent,
            model=incumbent_model,
        )
        rho_zero_state = parent.apply_earnings(
            rho_zero_state,
            period_context,
            rho_zero_state_parent,
            model=rho_zero_model,
        )
        pd.testing.assert_frame_equal(
            rho_zero_state[incumbent_state.columns], incumbent_state
        )
    frame_state_conditions = {
        "all_four_period_common_columns_equal": True,
        "parent_state_equal": (
            parent._rng_state_checksum(incumbent_state_parent)
            == parent._rng_state_checksum(rho_zero_state_parent)
        ),
        "incumbent_has_no_refresh_state_column": (
            fe.REFRESH_STATE_COLUMN not in incumbent_state
        ),
        "rho_zero_has_refresh_state_column": (
            fe.REFRESH_STATE_COLUMN in rho_zero_state
        ),
        "old_frame_state_sha256": parent._frame_checksum(
            incumbent_state,
            (
                "person_id",
                "year",
                "age",
                "earnings",
                *EARNINGS_CHAIN_STATE_COLUMNS,
            ),
        ),
    }
    if not all(
        value
        for name, value in frame_state_conditions.items()
        if name != "old_frame_state_sha256"
    ):
        raise AssertionError(
            f"rho-zero frame-state proof failed: {frame_state_conditions}"
        )

    prototype_frame[fe.REFRESH_STATE_COLUMN] = 1.0

    incumbent_parent = np.random.default_rng(781)
    prototype_parent = np.random.default_rng(781)
    incumbent_before = parent._rng_state_checksum(incumbent_parent)
    prototype_before = parent._rng_state_checksum(prototype_parent)
    odd_year = context.boundary + 1
    incumbent_odd = candidate2.generate(
        incumbent_frame.iloc[[0]], odd_year, incumbent_parent
    )
    prototype_odd = prototype.generate_with_frame_updates(
        prototype_frame.iloc[[0]], odd_year, prototype_parent
    )
    odd_conditions = {
        "level_bytes_equal": (
            incumbent_odd.tobytes() == prototype_odd.earnings.tobytes()
        ),
        "candidate2_parent_unchanged": (
            parent._rng_state_checksum(incumbent_parent) == incumbent_before
        ),
        "prototype_parent_unchanged": (
            parent._rng_state_checksum(prototype_parent) == prototype_before
        ),
        "nonnull_state_carried": (
            prototype_odd.frame_updates[fe.REFRESH_STATE_COLUMN][0] == 1.0
        ),
    }
    if not all(odd_conditions.values()):
        raise AssertionError(f"odd-year object proof failed: {odd_conditions}")

    incumbent_even_frame = None
    prototype_even_frame = None
    even_parent_seed = None
    for position in range(len(incumbent_frame)):
        incumbent_trial = incumbent_frame.iloc[[position]].copy()
        prototype_trial = prototype_frame.iloc[[position]].copy()
        for frame in (incumbent_trial, prototype_trial):
            frame["age"] = frame["age"].to_numpy(dtype=np.float64) + 2.0
        for parent_seed in range(782, 882):
            trial = candidate2.generate(
                incumbent_trial,
                context.boundary + 2,
                np.random.default_rng(parent_seed),
            )
            if trial[0] > 0.0:
                incumbent_even_frame = incumbent_trial
                prototype_even_frame = prototype_trial
                even_parent_seed = parent_seed
                break
        if even_parent_seed is not None:
            break
    if (
        incumbent_even_frame is None
        or prototype_even_frame is None
        or even_parent_seed is None
    ):
        raise AssertionError(
            "nonzero-rho object proof found no positive continuer"
        )

    incumbent_even_parent = np.random.default_rng(even_parent_seed)
    prototype_even_parent = np.random.default_rng(even_parent_seed)
    incumbent_trace = _TraceAccumulator()
    prototype_trace = _TraceAccumulator()

    def incumbent_even_call():
        return candidate2.generate(
            incumbent_even_frame,
            context.boundary + 2,
            incumbent_even_parent,
        )

    def prototype_even_call():
        return prototype.generate_with_frame_updates(
            prototype_even_frame,
            context.boundary + 2,
            prototype_even_parent,
        )

    incumbent_even = _call_with_trace(
        incumbent_even_call,
        incumbent_even_frame,
        context.boundary + 2,
        incumbent_even_parent,
        incumbent_trace,
    )
    prototype_even = _call_with_trace(
        prototype_even_call,
        prototype_even_frame,
        context.boundary + 2,
        prototype_even_parent,
        prototype_trace,
    )
    even_conditions = {
        "streams_1_5_final_states_equal": (
            incumbent_trace.as_dict() == prototype_trace.as_dict()
        ),
        "parent_state_equal": (
            parent._rng_state_checksum(incumbent_even_parent)
            == parent._rng_state_checksum(prototype_even_parent)
        ),
        "participation_state_equal": (
            (np.asarray(incumbent_even) > 0).tobytes()
            == (np.asarray(prototype_even.earnings) > 0).tobytes()
        ),
        "eligible_positive_continuer_exercised": (
            incumbent_even[0] > 0.0
            and prototype_even.earnings[0] > 0.0
            and not np.isnan(
                prototype_even.frame_updates[fe.REFRESH_STATE_COLUMN][0]
            )
        ),
        "parent_seed": even_parent_seed,
    }
    if not all(
        value
        for name, value in even_conditions.items()
        if name != "parent_seed"
    ):
        raise AssertionError(
            f"nonzero-rho object proof failed: {even_conditions}"
        )

    source_hashes = {
        "participation_formula": _source_sha256(fe._gate_sign_draw),
        "cell_marginal": _source_sha256(fe.CellMarginal),
        "rank_to_level": _source_sha256(
            fe.ForwardEarningsGenerator.rank_to_level
        ),
        "inverse_rank": _source_sha256(
            fe.ForwardEarningsGenerator.level_to_rank
        ),
        "frame_initializer": _source_sha256(
            fe.ForwardEarningsGenerator.materialize_initial_frame
        ),
        "candidate2_generate": _source_sha256(
            fe.ForwardEarningsGenerator.generate
        ),
        "old_substream_constructor": _source_sha256(fe._substream),
    }
    return {
        "passed": True,
        "nonzero_rho_path_checked": True,
        "proof_rho": proof_rho,
        "same_fitted_object_identity": identities,
        "candidate2_common_frame_bytes_equal": True,
        "candidate2_has_no_refresh_state_column": True,
        "prototype_state_initializes_null": True,
        "old_frame_columns_exact": list(fe.FRAME_COLUMNS),
        "old_chain_shift_columns_exact": list(EARNINGS_CHAIN_STATE_COLUMNS),
        "rho_zero_frame_state": frame_state_conditions,
        "odd_year": odd_conditions,
        "nonzero_rho_even_year": even_conditions,
        "pinned_source_sha256": source_hashes,
        "fit_signature_sha256": (fitted.q_invariant_fit_signature_sha256),
    }


def _fit_boundary(
    earnings: pd.DataFrame,
    nawi_path: Path,
    boundary: int,
    *,
    label: str,
) -> tuple[
    EarningsChainedRefit,
    dict[int, dict[str, np.ndarray]],
    dict[str, Any],
    pd.DataFrame,
    dict[int, float],
    dict[str, Any],
]:
    fit_input = truncate_estimation_frame(
        earnings,
        boundary_year=boundary,
        year_column="period",
        flow=False,
        label=label,
    )
    parent._assert_at_most(fit_input, "period", boundary, label)
    boundary_nawi, nawi_audit = parent._read_historical_nawi(
        nawi_path, maximum_year=boundary
    )
    expected_nawi = parent.EXPECTED_BOUNDARY_NAWI[boundary]
    if (
        nawi_audit["bytes_consumed_through_maximum_key"]
        != expected_nawi["prefix_bytes"]
        or nawi_audit["admitted_prefix_sha256"]
        != expected_nawi["prefix_sha256"]
        or parent._canonical_sha256(boundary_nawi)
        != expected_nawi["mapping_sha256"]
    ):
        raise RuntimeError(f"certified NAWI prefix through {boundary} drifted")
    with redirect_stdout(sys.stderr):
        fitted = refit_earnings_chained_generator(
            fit_input,
            boundary_nawi,
            seed=FIT_SEED,
            boundary_year=boundary,
            candidate_spec=CANDIDATE_2,
        )
    if fitted.generator.rank_refresh_q != FIXED_Q:
        raise AssertionError("candidate-2 fit did not bind the frozen q")
    stable_pools, stable_audit = parent._stable_pools(fitted)
    _production_stable_pool_proof(fitted, stable_pools)
    return (
        fitted,
        stable_pools,
        stable_audit,
        fit_input,
        boundary_nawi,
        nawi_audit,
    )


def _rho_zero_equivalence(
    earnings: pd.DataFrame,
    anchor_demo: pd.DataFrame,
    nawi_path: Path,
) -> tuple[dict[str, Any], dict[int, parent.BoundaryContext]]:
    by_boundary: dict[str, Any] = {}
    contexts: dict[int, parent.BoundaryContext] = {}
    object_proof: dict[str, Any] | None = None
    for boundary in PSEUDO_BOUNDARIES:
        _progress(f"preflight rho=0 boundary={boundary}: complete QRF refit")
        (
            fitted,
            _stable_pools,
            stable_audit,
            fit_input,
            boundary_nawi,
            nawi_audit,
        ) = _fit_boundary(
            earnings,
            nawi_path,
            boundary,
            label=f"rho-zero preflight boundary={boundary}",
        )
        fit_audit = parent._fit_audit(fitted, stable_audit, boundary)
        context = parent._boundary_context(
            fitted, earnings, anchor_demo, boundary
        )
        contexts[boundary] = context
        if object_proof is None:
            object_proof = _object_level_preflight(fitted, context)

        draw_records = []
        for draw_number, draw_seed in enumerate(SELECTION_DRAW_SEEDS, start=1):
            incumbent = _AuditedGenerator(fitted.generator, trace_streams=True)
            correlated = _AuditedGenerator(
                replace(fitted.generator, rank_refresh_rho=0.0),
                trace_streams=True,
            )
            incumbent_scored, incumbent_annual = parent._project(
                incumbent, context, draw_seed
            )
            correlated_scored, correlated_annual = parent._project(
                correlated, context, draw_seed
            )
            incumbent_cells = parent._selected_cells(
                incumbent_scored, boundary
            )
            correlated_cells = parent._selected_cells(
                correlated_scored, boundary
            )
            trace_incumbent = incumbent.trace.as_dict()
            trace_correlated = correlated.trace.as_dict()
            checks = {
                "person_period_keys_equal": (
                    parent._key_checksum(incumbent_annual)
                    == parent._key_checksum(correlated_annual)
                ),
                "level_bytes_equal": (
                    incumbent_annual["earnings"]
                    .to_numpy(dtype=np.float64)
                    .tobytes()
                    == correlated_annual["earnings"]
                    .to_numpy(dtype=np.float64)
                    .tobytes()
                ),
                "participation_states_equal": (
                    (incumbent_annual["earnings"] > 0)
                    .to_numpy(dtype=np.uint8)
                    .tobytes()
                    == (correlated_annual["earnings"] > 0)
                    .to_numpy(dtype=np.uint8)
                    .tobytes()
                ),
                "all_six_moments_equal": (
                    parent._canonical_bytes(incumbent_cells)
                    == parent._canonical_bytes(correlated_cells)
                ),
                "streams_1_5_final_states_equal": (
                    trace_incumbent == trace_correlated
                ),
                "truth_projection_support_equal": (
                    parent._key_checksum(correlated_scored)
                    == context.support_audit["truth_support_ids_sha256"]
                ),
                "chain_count_conservation": all(
                    correlated.chain.as_dict()[key]
                    for key in (
                        "eligible_decomposition_conserves",
                        "even_call_conservation_passed",
                    )
                ),
            }
            if not all(checks.values()):
                raise AssertionError(
                    "rho=0 equivalence failed at "
                    f"boundary={boundary} draw={draw_seed}: {checks}"
                )
            draw_records.append(
                {
                    "draw_seed": draw_seed,
                    **checks,
                    "annual_rows": int(len(correlated_annual)),
                    "level_sha256": parent._level_checksum(correlated_annual),
                    "participation_sha256": (
                        parent._participation_checksum(correlated_annual)
                    ),
                    "six_moments_sha256": parent._canonical_sha256(
                        correlated_cells
                    ),
                    "stream_trace_sha256": trace_correlated["sha256"],
                    "n_person_period_calls": trace_correlated["n_calls"],
                    "transition_chain": correlated.chain.as_dict(),
                    "passed": True,
                }
            )
            if draw_number % 5 == 0:
                _progress(
                    f"preflight rho=0 boundary={boundary}: "
                    f"completed {draw_number}/20 paired draws"
                )
        by_boundary[str(boundary)] = {
            "passed": all(record["passed"] for record in draw_records),
            "fit_input_rows": int(len(fit_input)),
            "fit_input_max_period": parent._max_year(fit_input, "period"),
            "fit_input_checksum": parent._frame_checksum(
                fit_input,
                ("person_id", "period", "earnings", "age", "weight"),
            ),
            "nawi_key_max": max(boundary_nawi),
            "nawi_checksum": parent._canonical_sha256(boundary_nawi),
            "nawi_field_read": nawi_audit,
            "fit": fit_audit,
            "support": context.support_audit,
            "rng_registry": context.rng_manifest,
            "per_draw": draw_records,
        }
    assert object_proof is not None
    result = {
        "passed": all(record["passed"] for record in by_boundary.values()),
        "required_boundaries": list(PSEUDO_BOUNDARIES),
        "required_draw_seeds": list(SELECTION_DRAW_SEEDS),
        "n_boundary_draw_equivalence_cells": sum(
            len(record["per_draw"]) for record in by_boundary.values()
        ),
        "boundaries": by_boundary,
        "object_level_unchanged": object_proof,
    }
    return result, contexts


def _run_preflights(
    earnings: pd.DataFrame,
    anchor_demo: pd.DataFrame,
    nawi_path: Path,
) -> dict[str, Any]:
    _progress("preflight: reset-law discriminating fixture")
    reset = _reset_law_preflight()
    _progress("preflight: endogenous participation feedback fixture")
    feedback = _participation_feedback_preflight()
    _progress("preflight: 3 boundaries x 20 draws rho-zero equivalence")
    equivalence, _contexts = _rho_zero_equivalence(
        earnings, anchor_demo, nawi_path
    )
    records = {
        "rho_zero_candidate2_equivalence": equivalence,
        "reset_law_discriminating_fixture": reset,
        "endogenous_participation_feedback": feedback,
        "object_level_unchanged": equivalence["object_level_unchanged"],
    }
    all_passed = all(bool(record["passed"]) for record in records.values())
    if not all_passed:
        raise RuntimeError("one or more amendment-6 preflights failed")
    return {
        "all_passed": True,
        "ladder_values_computed_before_pass": False,
        "failure_disposition": "STOP_AND_INVALIDATE_MECHANISM",
        "records": records,
    }


def _require_preflights(preflights: Mapping[str, Any]) -> None:
    if not bool(preflights.get("all_passed")):
        raise RuntimeError(
            "rho ladder is forbidden because the preflight pass gate failed"
        )
    if bool(preflights.get("ladder_values_computed_before_pass")):
        raise RuntimeError("preflight ordering attestation is false")


def _sum_chain_counts(per_draw: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stationary = Counter()
    pairs = Counter()
    resets = Counter()
    even_calls = 0
    eligible = 0
    all_conserve = True
    records = []
    for draw in per_draw:
        chain = draw["transition_chain"]
        even_calls += int(chain["even_calls"])
        eligible += int(chain["eligible_transitions"])
        stationary.update(chain["stationary_entries"])
        pairs.update(chain["consecutive_pairs"])
        resets.update(chain["resets"])
        all_conserve &= bool(chain["eligible_decomposition_conserves"])
        all_conserve &= bool(chain["even_call_conservation_passed"])
        records.append({"draw_seed": int(draw["draw_seed"]), **dict(chain)})
    aggregate = {
        "even_calls": even_calls,
        "eligible_transitions": eligible,
        "stationary_entries": {
            key: int(stationary[key]) for key in ("empty_to_0", "empty_to_1")
        },
        "consecutive_pairs": {
            key: int(pairs[key])
            for key in ("0_to_0", "0_to_1", "1_to_0", "1_to_1")
        },
        "resets": {
            key: int(resets[key])
            for key in (
                "nonparticipation",
                "zero_earnings",
                "stream3_reentry",
                "support_exit",
            )
        },
    }
    aggregate["eligible_decomposition_conserves"] = (
        sum(aggregate["stationary_entries"].values())
        + sum(aggregate["consecutive_pairs"].values())
        == eligible
    )
    aggregate["even_call_conservation_passed"] = (
        eligible + sum(aggregate["resets"].values()) == even_calls
    )
    return {
        "per_draw": records,
        "all_20": aggregate,
        "all_draws_conserve": all_conserve
        and aggregate["eligible_decomposition_conserves"]
        and aggregate["even_call_conservation_passed"],
    }


def _objective_across_boundaries(
    rung: Mapping[str, Any], draw_seeds: Sequence[int]
) -> dict[str, Any]:
    by_boundary: dict[str, float | None] = {}
    for boundary in PSEUDO_BOUNDARIES:
        aggregate = parent._aggregate_boundary(
            rung["boundaries"][str(boundary)], draw_seeds
        )
        by_boundary[str(boundary)] = aggregate["objective"]
    total = (
        None
        if any(value is None for value in by_boundary.values())
        else float(sum(float(value) for value in by_boundary.values()))
    )
    return {
        "draw_seeds": list(draw_seeds),
        "by_boundary": by_boundary,
        "total": total,
    }


def _train_f1_disclosure(rung: Mapping[str, Any]) -> dict[str, Any]:
    cell = "earn_dlog_mean.prime"
    by_boundary = {}
    for boundary in PSEUDO_BOUNDARIES:
        record = rung["boundaries"][str(boundary)]
        aggregate = record["aggregates"]["all_20"]
        truth = parent._selection_cell_value(record["truth_moments"][cell])
        projected = aggregate["projected_moments"][cell]
        by_boundary[str(boundary)] = {
            "truth_moment": truth,
            "projected_moment": projected,
            "gap_truth_minus_projected": (
                None
                if truth is None or projected is None
                else float(truth) - float(projected)
            ),
            "registered_score": aggregate["scores"][cell],
            "standardized_score": aggregate["standardized_scores"][cell],
            "objective_contribution": aggregate["objective_contributions"][
                cell
            ],
        }
    return {
        "cell": cell,
        "status": "DISCLOSURE_WITH_EXISTING_J_ROLE_ONLY",
        "adds_no_selection_criterion": True,
        "by_boundary": by_boundary,
    }


def _finalize_selector(payload: dict[str, Any]) -> None:
    rungs = payload["rungs"]
    for rho in RHO_GRID:
        label = f"{rho:.2f}"
        rung = rungs[label]
        invalid_reasons: list[str] = []
        for boundary in PSEUDO_BOUNDARIES:
            record = rung["boundaries"][str(boundary)]
            full = parent._aggregate_boundary(record, SELECTION_DRAW_SEEDS)
            first = parent._aggregate_boundary(record, FIRST_HALF_DRAW_SEEDS)
            second = parent._aggregate_boundary(record, SECOND_HALF_DRAW_SEEDS)
            record["aggregates"] = {
                "all_20": full,
                "first_10": first,
                "second_10": second,
            }
            regeneration = {}
            for name in SELECTED_CELLS:
                values = [
                    draw["moment_values"].get(name)
                    for draw in record["per_draw"]
                ]
                regenerated = all(
                    value is not None for value in values
                ) and any(value != values[0] for value in values[1:])
                regeneration[name] = regenerated
                if not regenerated:
                    invalid_reasons.append(
                        f"boundary {boundary} cell {name} not regenerated"
                    )
            record["regeneration"] = {
                "by_cell": regeneration,
                "all_six_cells_regenerated": all(regeneration.values()),
                "distinct_annual_level_surfaces": len(
                    {
                        draw["annual_level_sha256"]
                        for draw in record["per_draw"]
                    }
                ),
            }
            record["transition_pair_counts"] = _sum_chain_counts(
                record["per_draw"]
            )
            if not record["transition_pair_counts"]["all_draws_conserve"]:
                invalid_reasons.append(
                    f"boundary {boundary} transition counts do not conserve"
                )
            if full["objective"] is None:
                invalid_reasons.append(
                    f"boundary {boundary} has undefined all-draw objective"
                )
        rung["objectives"] = {
            "all_20": _objective_across_boundaries(rung, SELECTION_DRAW_SEEDS),
            "first_10": _objective_across_boundaries(
                rung, FIRST_HALF_DRAW_SEEDS
            ),
            "second_10": _objective_across_boundaries(
                rung, SECOND_HALF_DRAW_SEEDS
            ),
            "delete_one": [
                {
                    "omitted_draw_seed": omitted,
                    **_objective_across_boundaries(
                        rung,
                        tuple(
                            seed
                            for seed in SELECTION_DRAW_SEEDS
                            if seed != omitted
                        ),
                    ),
                }
                for omitted in SELECTION_DRAW_SEEDS
            ],
        }
        if any(
            rung["objectives"][name]["total"] is None
            for name in ("all_20", "first_10", "second_10")
        ):
            invalid_reasons.append(
                "one or more fixed-block objectives undefined"
            )
        rung["valid"] = not invalid_reasons
        rung["invalid_reasons"] = invalid_reasons
        rung["train_f1_analog_disclosure"] = _train_f1_disclosure(rung)

    baseline = rungs["0.00"]
    if not baseline["valid"]:
        raise RuntimeError(
            "global rho=0 validity checks failed: "
            + "; ".join(baseline["invalid_reasons"])
        )
    for rho in RHO_GRID:
        label = f"{rho:.2f}"
        rung = rungs[label]
        guards = {}
        feasible = bool(rung["valid"])
        for boundary in PSEUDO_BOUNDARIES:
            guard_cells = {}
            candidate_scores = rung["boundaries"][str(boundary)]["aggregates"][
                "all_20"
            ]["standardized_scores"]
            baseline_scores = baseline["boundaries"][str(boundary)][
                "aggregates"
            ]["all_20"]["standardized_scores"]
            for name in FEASIBILITY_CELLS:
                candidate = candidate_scores[name]
                incumbent = baseline_scores[name]
                passed = (
                    candidate is not None
                    and incumbent is not None
                    and float(candidate) <= float(incumbent) + 1.0
                )
                guard_cells[name] = {
                    "candidate_standardized_score": candidate,
                    "rho0_standardized_score": incumbent,
                    "limit": None if incumbent is None else incumbent + 1.0,
                    "passed": passed,
                }
                feasible &= passed
            guards[str(boundary)] = guard_cells
        rung["feasibility_guards"] = guards
        rung["feasible"] = feasible
        if rho == 0.0:
            improves = {
                "all_20": True,
                "first_10": True,
                "second_10": True,
            }
            retained = True
        else:
            improves = {}
            for name in ("all_20", "first_10", "second_10"):
                candidate_total = rung["objectives"][name]["total"]
                baseline_total = baseline["objectives"][name]["total"]
                improves[name] = (
                    candidate_total is not None
                    and baseline_total is not None
                    and float(candidate_total) < float(baseline_total)
                )
            retained = feasible and all(improves.values())
        rung["strict_improvement_vs_rho0"] = improves
        rung["retained_for_one_se"] = retained

    retained_labels = [
        f"{rho:.2f}"
        for rho in RHO_GRID
        if rungs[f"{rho:.2f}"]["retained_for_one_se"]
    ]
    rho_min_label = min(
        retained_labels,
        key=lambda candidate: (
            float(rungs[candidate]["objectives"]["all_20"]["total"]),
            abs(float(candidate)),
        ),
    )
    deletes = np.asarray(
        [
            record["total"]
            for record in rungs[rho_min_label]["objectives"]["delete_one"]
        ],
        dtype=np.float64,
    )
    delete_mean = float(deletes.mean())
    standard_error = float(
        np.sqrt((19.0 / 20.0) * np.sum((deletes - delete_mean) ** 2))
    )
    rho_min_objective = float(
        rungs[rho_min_label]["objectives"]["all_20"]["total"]
    )
    cutoff = rho_min_objective + standard_error
    within_one_se = [
        label
        for label in retained_labels
        if float(rungs[label]["objectives"]["all_20"]["total"]) <= cutoff
    ]
    selected_label = min(within_one_se, key=lambda value: abs(float(value)))
    for label, rung in rungs.items():
        rung["within_one_se_cutoff"] = label in within_one_se
        rung["selected"] = label == selected_label

    weak_retained_labels = ["0.00"]
    for rho in RHO_GRID:
        if rho == 0.0:
            continue
        label = f"{rho:.2f}"
        rung = rungs[label]
        if not rung["feasible"]:
            continue
        weak_improves = (
            float(rung["objectives"]["all_20"]["total"])
            <= float(baseline["objectives"]["all_20"]["total"])
            and float(rung["objectives"]["first_10"]["total"])
            < float(baseline["objectives"]["first_10"]["total"])
            and float(rung["objectives"]["second_10"]["total"])
            < float(baseline["objectives"]["second_10"]["total"])
        )
        if weak_improves:
            weak_retained_labels.append(label)
    weak_min_label = min(
        weak_retained_labels,
        key=lambda candidate: (
            float(rungs[candidate]["objectives"]["all_20"]["total"]),
            abs(float(candidate)),
        ),
    )
    weak_deletes = np.asarray(
        [
            record["total"]
            for record in rungs[weak_min_label]["objectives"]["delete_one"]
        ],
        dtype=np.float64,
    )
    weak_se = float(
        np.sqrt(
            (19.0 / 20.0) * np.sum((weak_deletes - weak_deletes.mean()) ** 2)
        )
    )
    weak_cutoff = (
        float(rungs[weak_min_label]["objectives"]["all_20"]["total"]) + weak_se
    )
    weak_selected_label = min(
        (
            label
            for label in weak_retained_labels
            if float(rungs[label]["objectives"]["all_20"]["total"])
            <= weak_cutoff
        ),
        key=lambda value: abs(float(value)),
    )
    if weak_selected_label != selected_label:
        raise AssertionError(
            "strict-versus-weak improvement changed selected rho"
        )

    selected_rho = float(selected_label)
    payload["selector"] = {
        "baseline_rho_retained": True,
        "effective_search_size": {
            "grid_rungs": len(RHO_GRID),
            "valid_rungs": sum(rung["valid"] for rung in rungs.values()),
            "feasible_rungs_including_rho0": sum(
                rung["feasible"] for rung in rungs.values()
            ),
            "retained_rungs_including_rho0": len(retained_labels),
            "retained_nonzero_rungs": sum(
                float(label) != 0.0 for label in retained_labels
            ),
        },
        "retained_rho": [float(label) for label in retained_labels],
        "rho_min": float(rho_min_label),
        "rho_min_objective": rho_min_objective,
        "rho_min_delete_one_mean": delete_mean,
        "rho_min_jackknife_standard_error": standard_error,
        "one_se_cutoff": cutoff,
        "rho_within_one_se": [float(label) for label in within_one_se],
        "selected_rho": selected_rho,
        "selected_rho_label": selected_label,
        "disposition": (
            "DESIGNED_PAUSE"
            if selected_rho == 0.0
            else "LOCK_ADDENDUM_ELIGIBLE"
        ),
        "closest_to_zero_tie_break_applied": True,
        "strict_vs_weak_improvement_outcome_invariant": True,
        "weak_improvement_counterfactual": {
            "weakened_comparison": "all_20 only; fixed halves remain strict",
            "retained_rho": [float(label) for label in weak_retained_labels],
            "rho_min": float(weak_min_label),
            "jackknife_standard_error": weak_se,
            "one_se_cutoff": weak_cutoff,
            "selected_rho": float(weak_selected_label),
        },
    }


def _run_ladder(
    earnings: pd.DataFrame,
    anchor_demo: pd.DataFrame,
    nawi_path: Path,
    preflights: Mapping[str, Any],
) -> dict[str, Any]:
    _require_preflights(preflights)
    result: dict[str, Any] = {"boundaries": {}, "rungs": {}}
    contexts: dict[int, parent.BoundaryContext] = {}
    fit_signatures: dict[int, str] = {}
    for rho_index, rho in enumerate(RHO_GRID, start=1):
        label = f"{rho:.2f}"
        _progress(f"starting rho={label} ({rho_index}/{len(RHO_GRID)})")
        rung = {"rho": rho, "fixed_q": FIXED_Q, "boundaries": {}}
        for boundary_index, boundary in enumerate(PSEUDO_BOUNDARIES, start=1):
            _progress(
                f"rho={label} boundary={boundary} "
                f"({boundary_index}/3): fresh complete QRF refit"
            )
            (
                fitted,
                _stable_pools,
                stable_audit,
                fit_input,
                boundary_nawi,
                nawi_audit,
            ) = _fit_boundary(
                earnings,
                nawi_path,
                boundary,
                label=f"rho={label} boundary={boundary} earnings",
            )
            fit_audit = parent._fit_audit(fitted, stable_audit, boundary)
            signature = fit_audit["q_invariant_fit_signature_sha256"]
            if boundary not in contexts:
                context = parent._boundary_context(
                    fitted, earnings, anchor_demo, boundary
                )
                contexts[boundary] = context
                fit_signatures[boundary] = signature
                result["boundaries"][str(boundary)] = {
                    "cutoff": boundary,
                    "fit_input_rows": int(len(fit_input)),
                    "fit_input_max_period": parent._max_year(
                        fit_input, "period"
                    ),
                    "fit_input_checksum": parent._frame_checksum(
                        fit_input,
                        (
                            "person_id",
                            "period",
                            "earnings",
                            "age",
                            "weight",
                        ),
                    ),
                    "nawi_key_max": max(boundary_nawi),
                    "nawi_checksum": parent._canonical_sha256(boundary_nawi),
                    "nawi_field_read": nawi_audit,
                    "truth_moments": context.truth_cells,
                    "floor": context.floor,
                    "floor_gate_seed_detail": context.floor_gate_seed_detail,
                    "standardizers": context.standardizers,
                    "support": context.support_audit,
                    "rng_registry": context.rng_manifest,
                }
            else:
                context = contexts[boundary]
                if signature != fit_signatures[boundary]:
                    raise AssertionError(
                        f"rho={label} boundary={boundary} fit/pool "
                        "signature differs despite fixed inputs and seed"
                    )

            draws = []
            for draw_number, draw_seed in enumerate(
                SELECTION_DRAW_SEEDS, start=1
            ):
                prototype = replace(
                    fitted.generator, rank_refresh_rho=float(rho)
                )
                audited = _AuditedGenerator(prototype)
                scored, annual = parent._project(audited, context, draw_seed)
                draws.append(
                    _projection_record(
                        scored,
                        annual,
                        boundary,
                        draw_seed,
                        audited.chain.as_dict(),
                    )
                )
                if draw_number % 5 == 0:
                    _progress(
                        f"rho={label} boundary={boundary}: "
                        f"completed {draw_number}/20 draws"
                    )
            rung["boundaries"][str(boundary)] = {
                "cutoff": boundary,
                "fit": fit_audit,
                "truth_moments": context.truth_cells,
                "floor": context.floor,
                "standardizers": context.standardizers,
                "support": context.support_audit,
                "rng_registry_sha256": context.rng_manifest["sha256"],
                "nawi_key_max": max(boundary_nawi),
                "nawi_checksum": parent._canonical_sha256(boundary_nawi),
                "nawi_field_read": nawi_audit,
                "per_draw": draws,
            }
            _progress(
                f"rho={label} boundary={boundary}: complete "
                f"(fit signature {signature[:12]})"
            )
        result["rungs"][label] = rung
        _progress(f"completed rho={label} ({rho_index}/{len(RHO_GRID)})")

    _progress(
        "all 51 refits and 1,020 ladder draws complete; reducing selector"
    )
    _finalize_selector(result)
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Run the amendment-6 train-only rho selector; strict JSON stdout "
            "and progress stderr"
        )
    )
    command.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the mandatory pass gate and stop before all ladder values",
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if RHO_GRID != tuple(
        round(-0.80 + 0.05 * index, 2) for index in range(17)
    ):
        raise RuntimeError("registered rho grid drifted")
    if RHO_GRID[-1] != 0.0 or len(RHO_GRID) != 17:
        raise RuntimeError("registered rho grid endpoints drifted")
    if PSEUDO_BOUNDARIES != (2006, 2008, 2010):
        raise RuntimeError("pseudo-boundary set drifted")
    if FIT_SEED != 5200 or SELECTION_DRAW_SEEDS != tuple(range(6200, 6220)):
        raise RuntimeError("fit or draw seed protocol drifted")
    if SUBSTREAM_CODES != {
        "gate": 1,
        "donor-draw": 2,
        "re-entry-draw": 3,
        "memory-refresh-gate": 4,
        "memory-refresh-rank": 5,
    }:
        raise RuntimeError("earnings substream registry drifted")
    if set(parent.FLOOR_SEEDS) != set(range(100)):
        raise RuntimeError("floor seed protocol drifted")

    freeze = _repository_freeze()
    qstar_ledger, qstar_raw = _load_qstar_ledger()
    _progress("loading field-capped train-only PSID sources")
    with redirect_stdout(sys.stderr):
        earnings, anchor_demo, nawi_path, source_audit = (
            parent._load_train_only_sources()
        )

    psid_dir = os.environ.get("POPULACE_DYNAMICS_PSID_DIR")
    pe_us_dir = os.environ.get("POPULACE_DYNAMICS_PE_US_DIR")
    payload: dict[str, Any] = {
        "schema": (
            PREFLIGHT_SCHEMA if arguments.preflight_only else RAW_SCHEMA
        ),
        "status": "PREFLIGHTS_RUNNING",
        "freeze": freeze,
        "environment": {
            "python_executable": str(Path(sys.executable).resolve()),
            "runtime": freeze["runtime"],
            "POPULACE_DYNAMICS_PSID_DIR": psid_dir,
            "POPULACE_DYNAMICS_PE_US_DIR": pe_us_dir,
            "thread_environment": freeze["thread_environment"],
        },
        "protocol": {
            "authority": (
                "docs/design/m6_projection_engine.md section 2.7.8, "
                "ratified merge 5b5c9c641d0cf38926d42afb839b45434c4b1b60"
            ),
            "fixed_q": FIXED_Q,
            "rho_grid": list(RHO_GRID),
            "pseudo_boundaries": list(PSEUDO_BOUNDARIES),
            "fit_seed": FIT_SEED,
            "selection_draw_seeds": list(SELECTION_DRAW_SEEDS),
            "fixed_halves": [
                list(FIRST_HALF_DRAW_SEEDS),
                list(SECOND_HALF_DRAW_SEEDS),
            ],
            "floor_seeds": list(parent.FLOOR_SEEDS),
            "selected_cells": list(SELECTED_CELLS),
            "objective_cells": list(OBJECTIVE_CELLS),
            "feasibility_cells": list(FEASIBILITY_CELLS),
            "fresh_complete_qrf_refit_per_rho_boundary": True,
            "common_random_numbers_across_rungs_at_fixed_seed": True,
            "draw_seed_to_registry_index": ("draw_index = draw_seed - 5200"),
            "substream_codes": SUBSTREAM_CODES,
            "retention_rule": (
                "nonzero rho retained only if feasible and strictly improves "
                "on rho=0 in all_20, first_10, and second_10"
            ),
            "one_se_rule": (
                "argmin J tie toward abs(rho)=0; "
                "sqrt((19/20)*sum((J_-r-mean(J_-r))^2)); select smallest "
                "abs(rho) at or below J(rho_min)+SE"
            ),
            "rho_zero_disposition": "DESIGNED_PAUSE",
            "no_candidate_1_or_candidate_2_artifact_read": True,
            "no_gate_score": True,
            "no_runs_write": True,
        },
        "fences": {
            "no_candidate_1_or_candidate_2_artifact_read": True,
            "no_gate_score": True,
            "no_runs_write": True,
        },
        "cross_pins": {
            "qstar_ledger_path": str(QSTAR_LEDGER_PATH.relative_to(ROOT)),
            "qstar_ledger_sha256": QSTAR_LEDGER_SHA256,
            "qstar_ledger_bytes": len(qstar_raw),
            "qstar_ledger_schema": qstar_ledger["schema"],
            "qstar_selected_q": qstar_ledger["selector"]["selected_q"],
            "candidate2_engine_spec_sha256": CANDIDATE_2.sha256,
        },
        "implementation": {
            "runner": str(SCRIPT_PATH.relative_to(ROOT)),
            "parent_selector_machinery": (
                "scripts/select_m6_qstar_train_only.py pure source, support, "
                "floor, scoring, aggregation, and checksum helpers"
            ),
            "candidate3_registry_entry_created": False,
            "rho_binding": (
                "dataclasses.replace on a fresh production candidate-2 fit; "
                "no refit object or incumbent path mutation"
            ),
            "refresh_state_column": fe.REFRESH_STATE_COLUMN,
            "refresh_state_values": [None, 0, 1],
            "reset_causes": [
                "nonparticipation",
                "zero_earnings",
                "stream3_reentry",
                "support_exit",
            ],
            "progress_stream": "stderr",
            "result_stream": "strict JSON stdout only",
            "generated_progress_path": str(PROGRESS_PATH.relative_to(ROOT)),
            "generated_findings_path": str(FINDINGS_PATH.relative_to(ROOT)),
        },
        "source_audit": source_audit,
    }

    preflights = _run_preflights(earnings, anchor_demo, nawi_path)
    payload["preflights"] = preflights
    payload["status"] = "PREFLIGHTS_PASSED"
    if arguments.preflight_only:
        _progress("all preflights passed; stopping before every ladder value")
        print(
            json.dumps(
                parent._plain(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    ladder = _run_ladder(earnings, anchor_demo, nawi_path, preflights)
    payload.update(ladder)
    payload["status"] = (
        "DESIGNED_PAUSE"
        if payload["selector"]["selected_rho"] == 0.0
        else "SELECTION_COMPLETE"
    )
    _progress(
        "selector complete: rho*="
        f"{payload['selector']['selected_rho_label']} "
        f"({payload['selector']['disposition']})"
    )
    print(
        json.dumps(
            parent._plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
