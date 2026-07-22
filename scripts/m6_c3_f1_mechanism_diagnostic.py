#!/usr/bin/env python3
"""Run the candidate-3 section 5.1 train-only F1 mechanism split.

The diagnostic replays the q-star selector's deepest pseudo-boundary with
the ratified ``q=0.55`` production engine.  For each of the same twenty draw
addresses it runs two arms from fresh state:

* the candidate with the section 2.7.6 projected wage-index path; and
* a counterfactual whose rank-to-level and inverse re-ranking index is the
  realized path through 2014.

Only the field-capped selector source loader, its boundary-2010 fit/support
factory, its projection helper, and the shared earnings-cell reducer are
used.  The script emits one sorted, strict-JSON findings payload on stdout;
progress goes to stderr.  It has no design-changing command-line options.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import select_m6_qstar_train_only as selector

from populace_dynamics.engine.candidates import CANDIDATE_2
from populace_dynamics.engine.refit import (
    refit_earnings_chained_generator,
    truncate_estimation_frame,
)
from populace_dynamics.harness.m6_cells import earnings_cells

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
LEDGER_PATH = ROOT / "docs/analysis/m6_qstar_train_only_selection_results.json"
LEDGER_SHA256 = (
    "d25b8e159384f8a84ed7f2218d863ca63d96fc9cb244536853b0a1f05c4025bb"
)
SCHEMA = "m6_c3_f1_mechanism_diagnostic.findings.v1"
BOUNDARY = 2010
REFERENCE_YEARS = (2012, 2014)
INTERVIEW_YEARS = (2013, 2015)
REFERENCE_WINDOW = (BOUNDARY, *REFERENCE_YEARS)
ANCHOR_INTERVIEW_YEAR = 2011
PROJECTED_INDEX_FIT_YEARS = tuple(range(2001, 2011))
SELECTED_Q = 0.55
FIT_SEED = 5200
DRAW_SEEDS = tuple(range(6200, 6220))
F1_CELL = "earn_dlog_mean.prime"
F2_CELL = "earn_autocorr_lag2"
EXPECTED_2014_NAWI_PREFIX = {
    "prefix_bytes": 1684,
    "prefix_sha256": (
        "9660f7a89b04a9f735607d66699a8919f01812cddb40ddff26a3e907fe914d89"
    ),
    "mapping_sha256": (
        "72bc5f32b9d2417b3573527525b7c9b6d29c901e5af5242059a9f9daada8150d"
    ),
}
VALIDITY_CAVEATS = (
    "(i) the evaluated waves sat inside q*'s selection evidence, so it is "
    "not out-of-sample with respect to the engine's selection; (ii) an "
    "error specific to the 2016/2018 scored regime cannot manifest in a "
    "≤2014 window, so any routing carries regime-mismatch risk."
)
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


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class RealizedWageIndexPath:
    """Replace the target-year projected index with realized train values."""

    base: Any
    realized: Mapping[int, float]
    swap_years: tuple[int, ...] = REFERENCE_YEARS

    def __post_init__(self) -> None:
        realized = {
            int(year): float(value) for year, value in self.realized.items()
        }
        swap_years = tuple(int(year) for year in self.swap_years)
        if set(realized) != set(swap_years):
            raise ValueError(
                "realized index mapping must contain exactly the swap years"
            )
        if any(year > 2014 for year in swap_years):
            raise ValueError("realized index swap cannot extend beyond 2014")
        if any(
            not math.isfinite(realized[year]) or realized[year] <= 0
            for year in swap_years
        ):
            raise ValueError(
                "realized index values must be finite and positive"
            )
        object.__setattr__(self, "realized", MappingProxyType(realized))
        object.__setattr__(self, "swap_years", swap_years)

    def projected(self, year: int) -> float:
        """Use realized NAWI at swapped targets, the fitted path otherwise."""
        year = int(year)
        if year in self.realized:
            return float(self.realized[year])
        return float(self.base.projected(year))

    def normalization_index(self, year: int) -> float:
        """Use the same counterfactual path for inverse re-ranking."""
        year = int(year)
        if year in self.realized:
            return float(self.realized[year])
        return float(self.base.normalization_index(year))


def with_realized_index_path(
    generator: Any,
    *,
    realized_nawi: Mapping[int, float],
) -> Any:
    """Clone a fitted generator, replacing only its wage-index dependency."""
    path = RealizedWageIndexPath(
        base=generator.wage_index,
        realized={year: realized_nawi[year] for year in REFERENCE_YEARS},
    )
    return replace(generator, wage_index=path)


def _cell_value(record: Mapping[str, Any]) -> float:
    value = selector._selection_cell_value(record)
    if value is None or not math.isfinite(value):
        raise ValueError("diagnostic moment is undefined")
    return float(value)


def reduce_diagnostic_moments(frame: pd.DataFrame) -> dict[str, Any]:
    """Reduce one truth/projection frame through the exact F1/F2 machinery."""
    pooled = earnings_cells(
        frame,
        level_years=REFERENCE_YEARS,
        change_years=REFERENCE_WINDOW,
    )
    by_reference_year: dict[str, float] = {}
    for target_year in REFERENCE_YEARS:
        transition = earnings_cells(
            frame,
            level_years=(target_year,),
            change_years=(target_year - 2, target_year),
        )
        by_reference_year[str(target_year)] = _cell_value(transition[F1_CELL])
    return {
        "primary_f1": {
            "window_aggregate": _cell_value(pooled[F1_CELL]),
            "by_reference_year": by_reference_year,
        },
        "secondary_lag2_autocorrelation": _cell_value(pooled[F2_CELL]),
    }


def decompose_gap(
    *,
    truth_value: float,
    candidate_value: float,
    realized_index_value: float,
) -> dict[str, float | None]:
    """Split ``truth - candidate`` into index-explained plus residual."""
    truth_value = float(truth_value)
    candidate_value = float(candidate_value)
    realized_index_value = float(realized_index_value)
    values = (truth_value, candidate_value, realized_index_value)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("decomposition inputs must be finite")
    gap = truth_value - candidate_value
    index_explained = realized_index_value - candidate_value
    residual = truth_value - realized_index_value
    closure_error = gap - index_explained - residual
    return {
        "truth_moment": truth_value,
        "candidate_projected_index_moment": candidate_value,
        "candidate_realized_index_moment": realized_index_value,
        "gap_truth_minus_candidate": gap,
        "absolute_gap": abs(gap),
        "index_explained_component": index_explained,
        "residual_conditional_on_index": residual,
        "index_explained_percent": (
            None if gap == 0.0 else 100.0 * index_explained / gap
        ),
        "residual_percent": None if gap == 0.0 else 100.0 * residual / gap,
        "closure_error": closure_error,
    }


def _candidate_q() -> float:
    operations = CANDIDATE_2.canonical_dict()["operations"]
    if len(operations) != 1:
        raise RuntimeError("candidate-2 engine operation registry drifted")
    return float(operations[0]["params"]["q"])


def _load_ledger() -> tuple[dict[str, Any], bytes]:
    raw = LEDGER_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LEDGER_SHA256:
        raise RuntimeError(f"q-star findings ledger hash drifted: {digest}")
    ledger = json.loads(raw)
    if ledger.get("schema") != "m6_qstar_train_only_selection.findings.v1":
        raise RuntimeError("q-star findings ledger schema drifted")
    protocol = ledger["protocol"]
    selector_record = ledger["selector"]
    if (
        float(selector_record["selected_q"]) != SELECTED_Q
        or int(protocol["fit_seed"]) != FIT_SEED
        or tuple(protocol["selection_draw_seeds"]) != DRAW_SEEDS
        or BOUNDARY not in protocol["pseudo_boundaries"]
    ):
        raise RuntimeError("q-star findings ledger protocol pins drifted")
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
    return {
        "version": distribution.version,
        "editable_source": str(source),
        "repository_root": str(repository),
        "repository_head": _git(repository, "rev-parse", "HEAD"),
        "repository_branch": _git(repository, "branch", "--show-current"),
        "source_tree_sha1": _git(repository, "rev-parse", f"HEAD:{tree_path}"),
        "tracked_source_clean": not bool(
            _git(repository, "status", "--porcelain", "--", tree_path)
        ),
    }


def _environment() -> dict[str, Any]:
    status = _git(ROOT, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "diagnostic execution requires a committed clean source freeze; "
            f"worktree status is:\n{status}"
        )
    runtime_packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "quantile-forest",
        "populace-fit",
        "populace-frame",
        "policyengine-us",
        "policyengine-core",
    )
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in runtime_packages
        },
        "thread_environment": {
            key: os.environ[key]
            for key in THREAD_ENVIRONMENT_KEYS
            if key in os.environ
        },
        "repository": {
            "root": str(ROOT),
            "head": _git(ROOT, "rev-parse", "HEAD"),
            "branch": _git(ROOT, "branch", "--show-current"),
            "worktree_clean_at_start": True,
            "script_blob_sha1": _git(
                ROOT,
                "rev-parse",
                "HEAD:scripts/m6_c3_f1_mechanism_diagnostic.py",
            ),
            "script_sha256": _file_sha256(SCRIPT_PATH),
            "source_tree_sha1": _git(
                ROOT, "rev-parse", "HEAD:src/populace_dynamics"
            ),
        },
        "editable_fitting_stack": {
            "populace_fit": _editable_source(
                "populace-fit",
                "packages/populace-fit/src/populace/fit",
            ),
            "populace_frame": _editable_source(
                "populace-frame",
                "packages/populace-frame/src/populace/frame",
            ),
        },
    }


def _validate_nawi_prefix(
    mapping: Mapping[int, float],
    audit: Mapping[str, Any],
) -> None:
    if (
        max(mapping) != 2014
        or int(audit["maximum_admitted_key_year"]) != 2014
        or not audit["stopped_after_maximum_key"]
        or audit["post_maximum_key_bytes_read"]
        or int(audit["bytes_consumed_through_maximum_key"])
        != EXPECTED_2014_NAWI_PREFIX["prefix_bytes"]
        or audit["admitted_prefix_sha256"]
        != EXPECTED_2014_NAWI_PREFIX["prefix_sha256"]
        or selector._canonical_sha256(mapping)
        != EXPECTED_2014_NAWI_PREFIX["mapping_sha256"]
    ):
        raise RuntimeError("certified NAWI prefix through 2014 drifted")


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label} differs: {actual!r} != {expected!r}")


def _projection_record(
    scored: pd.DataFrame,
    annual: pd.DataFrame,
    draw_seed: int,
) -> dict[str, Any]:
    cells = selector._selected_cells(scored, BOUNDARY)
    return {
        "draw_seed": draw_seed,
        "moments": cells,
        "moment_values": {
            name: selector._selection_cell_value(record)
            for name, record in cells.items()
        },
        "support_ids_sha256": selector._key_checksum(scored),
        "annual_level_sha256": selector._level_checksum(annual),
        "annual_participation_sha256": selector._participation_checksum(
            annual
        ),
        "fresh_initial_state": True,
    }


def _aggregate_values(
    records: Sequence[Mapping[str, Any]],
    *path: str,
) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _decomposition_findings(
    truth: Mapping[str, Any],
    draw_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary_per_draw = []
    secondary_per_draw = []
    for record in draw_records:
        primary_per_draw.append(
            {
                "draw_seed": int(record["draw_seed"]),
                "window_aggregate": decompose_gap(
                    truth_value=truth["primary_f1"]["window_aggregate"],
                    candidate_value=record["candidate_moments"]["primary_f1"][
                        "window_aggregate"
                    ],
                    realized_index_value=record["realized_index_moments"][
                        "primary_f1"
                    ]["window_aggregate"],
                ),
                "by_reference_year": {
                    str(year): decompose_gap(
                        truth_value=truth["primary_f1"]["by_reference_year"][
                            str(year)
                        ],
                        candidate_value=record["candidate_moments"][
                            "primary_f1"
                        ]["by_reference_year"][str(year)],
                        realized_index_value=record["realized_index_moments"][
                            "primary_f1"
                        ]["by_reference_year"][str(year)],
                    )
                    for year in REFERENCE_YEARS
                },
            }
        )
        secondary_per_draw.append(
            {
                "draw_seed": int(record["draw_seed"]),
                **decompose_gap(
                    truth_value=truth["secondary_lag2_autocorrelation"],
                    candidate_value=record["candidate_moments"][
                        "secondary_lag2_autocorrelation"
                    ],
                    realized_index_value=record["realized_index_moments"][
                        "secondary_lag2_autocorrelation"
                    ],
                ),
            }
        )

    primary_aggregate = {
        "draw_seeds": list(DRAW_SEEDS),
        "window_aggregate": decompose_gap(
            truth_value=truth["primary_f1"]["window_aggregate"],
            candidate_value=_aggregate_values(
                draw_records,
                "candidate_moments",
                "primary_f1",
                "window_aggregate",
            ),
            realized_index_value=_aggregate_values(
                draw_records,
                "realized_index_moments",
                "primary_f1",
                "window_aggregate",
            ),
        ),
        "by_reference_year": {
            str(year): decompose_gap(
                truth_value=truth["primary_f1"]["by_reference_year"][
                    str(year)
                ],
                candidate_value=_aggregate_values(
                    draw_records,
                    "candidate_moments",
                    "primary_f1",
                    "by_reference_year",
                    str(year),
                ),
                realized_index_value=_aggregate_values(
                    draw_records,
                    "realized_index_moments",
                    "primary_f1",
                    "by_reference_year",
                    str(year),
                ),
            )
            for year in REFERENCE_YEARS
        },
    }
    secondary_aggregate = {
        "draw_seeds": list(DRAW_SEEDS),
        **decompose_gap(
            truth_value=truth["secondary_lag2_autocorrelation"],
            candidate_value=_aggregate_values(
                draw_records,
                "candidate_moments",
                "secondary_lag2_autocorrelation",
            ),
            realized_index_value=_aggregate_values(
                draw_records,
                "realized_index_moments",
                "secondary_lag2_autocorrelation",
            ),
        ),
    }
    return {
        "primary_f1_mean_dlog": {
            "label": "primary F1 mechanism split",
            "cell": F1_CELL,
            "metric": "abs_gap_log",
            "cohort": "prime (ages 25-44)",
            "exact_gate_analog": (
                "weighted pooled mean of positive-to-positive adjacent "
                "biennial log changes over reference years 2010/2012/2014"
            ),
            "aggregate": primary_aggregate,
            "per_draw": primary_per_draw,
        },
        "secondary_lag2_autocorrelation": {
            "label": "secondary",
            "cell": F2_CELL,
            "metric": "abs_gap_corr",
            "reference_year": 2014,
            "exact_gate_analog": (
                "weighted Pearson correlation of positive log earnings at "
                "lag 2 on the biennial 2010/2012/2014 grid"
            ),
            "aggregate": secondary_aggregate,
            "per_draw": secondary_per_draw,
        },
    }


def _run() -> dict[str, Any]:
    if (
        BOUNDARY != selector.PSEUDO_BOUNDARIES[-1]
        or FIT_SEED != selector.FIT_SEED
        or DRAW_SEEDS != selector.SELECTION_DRAW_SEEDS
        or _candidate_q() != SELECTED_Q
    ):
        raise RuntimeError("diagnostic protocol constants drifted")
    environment = _environment()
    ledger, ledger_raw = _load_ledger()
    ledger_boundary = ledger["boundaries"][str(BOUNDARY)]
    ledger_rung = ledger["rungs"][f"{SELECTED_Q:.2f}"]["boundaries"][
        str(BOUNDARY)
    ]

    with redirect_stdout(sys.stderr):
        earnings, anchor_demo, nawi_path, source_audit = (
            selector._load_train_only_sources()
        )
    fit_input = truncate_estimation_frame(
        earnings,
        boundary_year=BOUNDARY,
        year_column="period",
        flow=False,
        label="candidate-3 F1 diagnostic boundary-2010 earnings",
    )
    selector._assert_at_most(
        fit_input, "period", BOUNDARY, "diagnostic_fit_input"
    )
    boundary_nawi, boundary_nawi_audit = selector._read_historical_nawi(
        nawi_path, maximum_year=BOUNDARY
    )
    realized_nawi, realized_nawi_audit = selector._read_historical_nawi(
        nawi_path, maximum_year=2014
    )
    _validate_nawi_prefix(realized_nawi, realized_nawi_audit)
    _assert_equal(
        selector._canonical_sha256(boundary_nawi),
        ledger_boundary["nawi_checksum"],
        "boundary NAWI checksum",
    )

    _progress("fitting certified candidate-2 earnings engine at boundary 2010")
    with redirect_stdout(sys.stderr):
        fitted = refit_earnings_chained_generator(
            fit_input,
            boundary_nawi,
            seed=FIT_SEED,
            boundary_year=BOUNDARY,
            candidate_spec=CANDIDATE_2,
        )
    if fitted.generator.rank_refresh_q != SELECTED_Q:
        raise AssertionError("fitted engine did not bind q=0.55")
    stable_pools, stable_audit = selector._stable_pools(fitted)
    if fitted.generator.stable_pools is None:
        raise AssertionError("fitted engine omitted stable donor pools")
    for bin_index in stable_pools:
        for field in stable_pools[bin_index]:
            if not np.array_equal(
                stable_pools[bin_index][field],
                fitted.generator.stable_pools[bin_index][field],
            ):
                raise AssertionError(
                    "production and selector stable pools differ"
                )
    fit_audit = selector._fit_audit(fitted, stable_audit, BOUNDARY)
    _assert_equal(
        fit_audit["q_invariant_fit_signature_sha256"],
        ledger_rung["fit"]["q_invariant_fit_signature_sha256"],
        "boundary fit signature",
    )
    context = selector._boundary_context(
        fitted, earnings, anchor_demo, BOUNDARY
    )
    for key in (
        "domain_ids_sha256",
        "truth_support_ids_sha256",
        "truth_frame_checksum",
    ):
        _assert_equal(
            context.support_audit[key],
            ledger_boundary["support"][key],
            f"boundary support {key}",
        )
    _assert_equal(
        context.rng_manifest["sha256"],
        ledger_boundary["rng_registry"]["sha256"],
        "boundary RNG registry",
    )

    truth_moments = reduce_diagnostic_moments(context.truth_support)
    realized_generator = with_realized_index_path(
        fitted.generator,
        realized_nawi=realized_nawi,
    )
    draw_records: list[dict[str, Any]] = []
    candidate_projection_records: list[dict[str, Any]] = []
    projection_audit: list[dict[str, Any]] = []
    for draw_number, draw_seed in enumerate(DRAW_SEEDS, start=1):
        candidate_scored, candidate_annual = selector._project(
            fitted.generator, context, draw_seed
        )
        realized_scored, realized_annual = selector._project(
            realized_generator, context, draw_seed
        )
        candidate_support = selector._key_checksum(candidate_scored)
        realized_support = selector._key_checksum(realized_scored)
        if candidate_support != realized_support:
            raise AssertionError("counterfactual changed scoring support")
        candidate_projection_records.append(
            _projection_record(
                candidate_scored,
                candidate_annual,
                draw_seed,
            )
        )
        draw_records.append(
            {
                "draw_seed": draw_seed,
                "candidate_moments": reduce_diagnostic_moments(
                    candidate_scored
                ),
                "realized_index_moments": reduce_diagnostic_moments(
                    realized_scored
                ),
            }
        )
        projection_audit.append(
            {
                "draw_seed": draw_seed,
                "support_ids_sha256": candidate_support,
                "candidate_annual_level_sha256": selector._level_checksum(
                    candidate_annual
                ),
                "realized_index_annual_level_sha256": (
                    selector._level_checksum(realized_annual)
                ),
                "candidate_annual_participation_sha256": (
                    selector._participation_checksum(candidate_annual)
                ),
                "realized_index_annual_participation_sha256": (
                    selector._participation_checksum(realized_annual)
                ),
                "fresh_initial_state_both_arms": True,
                "common_rng_address": True,
            }
        )
        if draw_number % 5 == 0:
            _progress(f"completed {draw_number}/20 paired projections")

    candidate_records_sha256 = selector._canonical_sha256(
        candidate_projection_records
    )
    _assert_equal(
        candidate_records_sha256,
        ledger_rung["per_draw_summary"]["records_sha256"],
        "candidate per-draw replay checksum",
    )
    findings = _decomposition_findings(truth_moments, draw_records)
    primary = findings["primary_f1_mean_dlog"]["aggregate"]
    secondary = findings["secondary_lag2_autocorrelation"]["aggregate"]
    ledger_truth = ledger_rung["truth_moments"]
    ledger_aggregate = ledger_rung["aggregates"]["all_20"]["projected_moments"]
    _assert_equal(
        primary["window_aggregate"]["truth_moment"],
        ledger_truth[F1_CELL]["value"],
        "F1 truth moment replay",
    )
    _assert_equal(
        primary["window_aggregate"]["candidate_projected_index_moment"],
        ledger_aggregate[F1_CELL],
        "F1 candidate moment replay",
    )
    _assert_equal(
        secondary["truth_moment"],
        ledger_truth[F2_CELL]["value"],
        "lag-2 truth moment replay",
    )
    _assert_equal(
        secondary["candidate_projected_index_moment"],
        ledger_aggregate[F2_CELL],
        "lag-2 candidate moment replay",
    )

    projected_index = fitted.generator.wage_index
    artifact = {
        "schema": SCHEMA,
        "status": "COMPUTED_TRAIN_ONLY_FINDING",
        "authority": {
            "program": "docs/design/m6_candidate3_program.md section 5.1",
            "projected_index_law": (
                "docs/design/m6_projection_engine.md section 2.7.6"
            ),
            "pattern": "train-only findings artifact; no contract surface",
        },
        "protocol": {
            "pseudo_boundary": BOUNDARY,
            "projected_index_fit_years": list(PROJECTED_INDEX_FIT_YEARS),
            "earnings_fit_rule": "all earnings reference rows through 2010",
            "reference_years": list(REFERENCE_YEARS),
            "interview_years": list(INTERVIEW_YEARS),
            "reference_to_interview_year": {
                str(reference): interview
                for reference, interview in zip(
                    REFERENCE_YEARS, INTERVIEW_YEARS, strict=True
                )
            },
            "anchor_interview_year": ANCHOR_INTERVIEW_YEAR,
            "q": SELECTED_Q,
            "fit_seed": FIT_SEED,
            "draw_seeds": list(DRAW_SEEDS),
            "draw_indices": [
                seed - selector.DRAW_SEED_BASE for seed in DRAW_SEEDS
            ],
            "n_draws": len(DRAW_SEEDS),
            "maximum_information_year": 2014,
            "sign_convention": (
                "gap = truth - candidate; index-explained = candidate under "
                "realized index - candidate under projected index; residual "
                "= truth - candidate under realized index"
            ),
            "share_convention": (
                "signed component / signed gap * 100; shares are not clamped "
                "and can lie outside [0,100] when the swap over-closes or "
                "moves against the gap"
            ),
            "routing_threshold": None,
            "routing_note": (
                "section 5.1 defines no numerical predominantly/mixed cutoff; "
                "this findings artifact reports components without inventing one"
            ),
        },
        "cross_pins": {
            "qstar_ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
            "qstar_ledger_sha256": LEDGER_SHA256,
            "qstar_ledger_bytes": len(ledger_raw),
            "qstar_schema": ledger["schema"],
            "qstar_raw_stdout_sha256": ledger["full_stdout_sha256"],
            "qstar_freeze_commit": ledger["freeze"]["freeze_commit"],
            "qstar_selected_q": ledger["selector"]["selected_q"],
            "boundary_2010_fit_signature_sha256": fit_audit[
                "q_invariant_fit_signature_sha256"
            ],
            "boundary_2010_rng_registry_sha256": context.rng_manifest[
                "sha256"
            ],
            "candidate_per_draw_replay_records_sha256": (
                candidate_records_sha256
            ),
            "candidate_per_draw_replay_matches_ledger": True,
        },
        "intervention": {
            "name": "realized-index path through 2014",
            "projected_index_fit": "OLS ln(NAWI) on year over 2001-2010",
            "swap_years": list(REFERENCE_YEARS),
            "projected_index": {
                str(year): projected_index.projected(year)
                for year in REFERENCE_YEARS
            },
            "realized_index": {
                str(year): realized_nawi[year] for year in REFERENCE_YEARS
            },
            "rank_to_level_index_swapped": True,
            "inverse_reranking_index_swapped": True,
            "full_projection_replayed": True,
            "post_hoc_moment_rescaling": False,
            "all_non_index_fitted_surfaces_preserved": True,
            "same_draw_addresses_across_arms": True,
            "fresh_initial_state_per_arm_and_draw": True,
        },
        "findings": findings,
        "projection_audit": {
            "per_draw": projection_audit,
            "fit_input": {
                "rows": int(len(fit_input)),
                "maximum_reference_year": selector._max_year(
                    fit_input, "period"
                ),
                "checksum": selector._frame_checksum(
                    fit_input,
                    ("person_id", "period", "earnings", "age", "weight"),
                ),
            },
            "fit": fit_audit,
            "support": context.support_audit,
            "rng_registry": context.rng_manifest,
            "truth_support_ids_sha256": context.support_audit[
                "truth_support_ids_sha256"
            ],
            "candidate_records_sha256": candidate_records_sha256,
            "all_support_equal": True,
            "all_draws_defined": True,
        },
        "information_boundary": {
            "maximum_earnings_reference_year_requested": source_audit[
                "requested_earnings_reference_max"
            ],
            "maximum_collection_wave_requested": source_audit[
                "psid_field_read"
            ]["maximum_collection_wave"],
            "collection_waves_after_2014": source_audit["psid_field_read"][
                "collection_waves_after_2014"
            ],
            "collection_wave_2015_use": (
                "income-reference year 2014 labor fields and incumbent "
                "collection-wave covariates only"
            ),
            "maximum_realized_macro_year_read": max(realized_nawi),
            "realized_macro_post_2014_read": False,
            "post_2014_earnings_reference_row_requested": source_audit[
                "post_2014_earnings_reference_row_requested"
            ],
            "candidate_or_gate_artifact_read": source_audit[
                "candidate_or_gate_artifact_read"
            ],
            "gate_configuration_read": source_audit["gate_configuration_read"],
            "run_artifact_read": False,
            "run_artifact_written": False,
            "source_audit": source_audit,
            "boundary_nawi_read_audit": boundary_nawi_audit,
            "realized_nawi_read_audit": realized_nawi_audit,
        },
        "environment": environment,
        "validity_caveats_verbatim": VALIDITY_CAVEATS,
    }
    return artifact


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run the fixed candidate-3 section 5.1 train-only diagnostic; "
            "strict JSON stdout and progress stderr."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser().parse_args(argv)
    artifact = _run()
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
