#!/usr/bin/env python3
"""Strictly validate and reduce the frozen round-3 selector stdout.

Usage::

    python scripts/reduce_m6_remarriage_round5.py \
      < docs/analysis/m6_remarriage_round3_selection_full.json \
      > docs/analysis/m6_remarriage_round3_selection_results.json

The reducer independently reconstructs the projected means, fixed-block and
delete-one objectives, all seven eligibility rules, and the one-SE decision.
Only the 600 repetitive per-seed publication-group arrays are removed.  The
full stdout and removed arrays receive separate SHA-256 commitments.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit as scipy_expit
from scipy.special import logit as scipy_logit

CONFIG_FILENAME = "m6_remarriage_round3_selector_config.json"
CONFIG_SCHEMA = "m6.remarriage.round3.selector_config.v1"
RAW_SCHEMA = "m6.remarriage.learning_plan.round3.selection.full.v1"
FINDINGS_SCHEMA = "m6.remarriage.learning_plan.round3.selection.findings.v1"
BOUNDARIES = ("2006", "2008", "2010")
BOUNDARY_INTS = (2006, 2008, 2010)
LAWS = (
    "R0",
    "R_D50_W00",
    "R_D75_W00",
    "R_D50_W05",
    "R_D75_W05",
)
SEEDS = tuple(range(7240, 7280))
BLOCKS = (tuple(range(7240, 7260)), tuple(range(7260, 7280)))
ORIGINS = ("divorced", "widowed")
AGE_BANDS = ((18, 34), (35, 49), (50, 64))
YSD_BANDS = ((0, 4), (5, 9), (10, 120))
TOLERANCE = 1e-12
AREA_TOLERANCE = 1e-10
WIDOWED_BUDGET = 0.08956860182931886
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

EXPECTED_GROUPS = tuple(
    (
        f"{origin}|age_{age[0]}_{age[1]}|ysd_{ysd[0]}_{ysd[1]}",
        origin,
        age,
        ysd,
    )
    for origin in ORIGINS
    for age in AGE_BANDS
    for ysd in YSD_BANDS
)

RATE_FIELDS = (
    "risk_rows",
    "event_rows",
    "exposure",
    "numerator",
    "rate",
)
RATIO_FIELDS = tuple(
    name
    for quantity in ("exposure", "numerator", "rate")
    for name in (f"{quantity}_ratio", f"log_{quantity}_ratio")
)
PROJECTED_FIELDS = RATE_FIELDS + RATIO_FIELDS
MEAN_FIELDS = (
    "seed_count",
    "mean_risk_rows",
    "mean_event_rows",
    "exposure",
    "numerator",
    "rate",
) + RATIO_FIELDS
DIRECT_FIELDS = (
    "risk_rows",
    "event_rows",
    "matchable_positive_weight_event_rows",
    "matchable_event_rows",
    "unmatched_same_year_event_rows",
    "unmatched_same_year_event_weight",
    "exposure",
    "deviance_exposure",
    "actual_numerator",
    "matchable_numerator",
    "actual_rate",
    "row_expected_numerator",
    "expected_numerator",
    "expected_rate",
    "qdir",
    "weighted_deviance_numerator",
    "weighted_bernoulli_deviance",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_strict_json(raw: bytes, *, source: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one top-level object")
    return value


def _mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _array(value: Any, *, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _strict_keys(
    value: dict[str, Any], expected: tuple[str, ...] | set[str], *, where: str
) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        missing = sorted(expected_set.difference(value))
        extra = sorted(set(value).difference(expected_set))
        raise ValueError(
            f"{where} fields drifted; missing={missing}, extra={extra}"
        )


def _finite_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} must be a JSON number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{where} must be finite")
    return numeric


def _nonnegative_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a nonnegative integer")
    return value


def _true(value: Any, *, where: str) -> None:
    if value is not True:
        raise ValueError(f"{where} must be true")


def _false(value: Any, *, where: str) -> None:
    if value is not False:
        raise ValueError(f"{where} must be false")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _assert_same(actual: Any, expected: Any, *, where: str) -> None:
    if _canonical_bytes(actual) != _canonical_bytes(expected):
        raise ValueError(
            f"{where} does not match its independent recomputation"
        )


def _selection_diff(actual: Any, expected: Any) -> dict[str, Any]:
    key_presence_diffs: list[dict[str, Any]] = []
    value_diffs: list[dict[str, Any]] = []

    def walk(actual_value: Any, expected_value: Any, path: str) -> None:
        if isinstance(actual_value, dict) and isinstance(expected_value, dict):
            actual_keys = set(actual_value)
            expected_keys = set(expected_value)
            for key in sorted(actual_keys - expected_keys):
                key_presence_diffs.append(
                    {
                        "path": f"{path}.{key}",
                        "raw": {
                            "present": True,
                            "value": actual_value[key],
                        },
                        "recomputed": {"present": False},
                    }
                )
            for key in sorted(expected_keys - actual_keys):
                key_presence_diffs.append(
                    {
                        "path": f"{path}.{key}",
                        "raw": {"present": False},
                        "recomputed": {
                            "present": True,
                            "value": expected_value[key],
                        },
                    }
                )
            for key in sorted(actual_keys & expected_keys):
                walk(
                    actual_value[key],
                    expected_value[key],
                    f"{path}.{key}",
                )
            return
        if isinstance(actual_value, list) and isinstance(expected_value, list):
            if len(actual_value) != len(expected_value):
                value_diffs.append(
                    {
                        "path": path,
                        "raw_value": actual_value,
                        "recomputed_value": expected_value,
                    }
                )
                return
            for index, (actual_item, expected_item) in enumerate(
                zip(actual_value, expected_value, strict=True)
            ):
                walk(
                    actual_item,
                    expected_item,
                    f"{path}[{index}]",
                )
            return
        if _canonical_bytes(actual_value) != _canonical_bytes(expected_value):
            value_diffs.append(
                {
                    "path": path,
                    "raw_value": actual_value,
                    "recomputed_value": expected_value,
                }
            )

    walk(actual, expected, "selection")
    return {
        "schema": "m6.remarriage.round6.selection_diff.v1",
        "where": "selection and one-SE outcome",
        "key_presence_diffs": key_presence_diffs,
        "value_diffs": value_diffs,
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unexpected config schema {config.get('schema')!r}")
    if config.get("status") != "frozen_train_only_selector_configuration":
        raise ValueError("config status drifted")

    runtime = _mapping(
        config.get("runtime_numeric_identity"),
        where="config.runtime_numeric_identity",
    )
    expected_runtime = {
        "python_implementation": "CPython",
        "python_version": "3.13.12",
        "numpy_version": "2.4.2",
    }
    _assert_same(runtime, expected_runtime, where="config runtime identity")

    frozen_paths = _array(
        _mapping(config.get("freeze"), where="config.freeze").get("paths"),
        where="config.freeze.paths",
    )
    expected_paths = [
        "scripts/select_m6_remarriage_round3.py",
        f"scripts/{CONFIG_FILENAME}",
        "scripts/reduce_m6_remarriage_round3.py",
        "scripts/analyze_m6_remarriage_train_delta.py",
    ]
    if frozen_paths != expected_paths:
        raise ValueError("config frozen path list drifted")
    if config["freeze"].get("require_clean_committed_head") is not True:
        raise ValueError("config does not require a clean committed head")
    if config["freeze"].get("mismatch_status") != "ROOT_VALIDATION_MISMATCH":
        raise ValueError("config root mismatch status drifted")

    protocol = _mapping(config.get("protocol"), where="config.protocol")
    if protocol.get("maximum_information_year") != 2014:
        raise ValueError("config maximum information year drifted")
    if protocol.get("boundaries") != list(BOUNDARY_INTS):
        raise ValueError("config boundary order drifted")
    expected_years = {
        "2006": [2007, 2008, 2009, 2010],
        "2008": [2009, 2010, 2011, 2012],
        "2010": [2011, 2012, 2013],
    }
    if protocol.get("evaluation_years") != expected_years:
        raise ValueError("config evaluation years drifted")
    if protocol.get("n_periods") != {"2006": 4, "2008": 4, "2010": 3}:
        raise ValueError("config RNG period counts drifted")
    if protocol.get("seeds") != list(SEEDS):
        raise ValueError("config seed bank drifted")
    if protocol.get("blocks") != [list(block) for block in BLOCKS]:
        raise ValueError("config fixed blocks drifted")
    if protocol.get("law_order") != list(LAWS):
        raise ValueError("config law order drifted")

    information = _mapping(
        config.get("information_boundary"), where="config.information_boundary"
    )
    expected_information = {
        "evidence_cutoff": 2014,
        "maximum_demographic_interview_year": 2013,
        "maximum_event_or_establishing_report_year": 2014,
        "maximum_evaluation_year": 2013,
        "calendar_2014_flow_excluded": True,
        "prohibited_selection_years": [2015, 2016, 2017, 2018, 2019],
    }
    _assert_same(
        information, expected_information, where="config information boundary"
    )

    holdouts = _mapping(
        config.get("pseudo_holdouts"), where="config.pseudo_holdouts"
    )
    if holdouts.get("boundary_order") != list(BOUNDARY_INTS):
        raise ValueError("config pseudo-boundary order drifted")
    for boundary in BOUNDARIES:
        record = _mapping(
            holdouts["boundaries"].get(boundary),
            where=f"config.pseudo_holdouts.boundaries.{boundary}",
        )
        if record.get("evaluation_years") != expected_years[boundary]:
            raise ValueError(f"config holdout years drifted at {boundary}")
        if record.get("n_periods") != protocol["n_periods"][boundary]:
            raise ValueError(f"config holdout periods drifted at {boundary}")
    if holdouts.get("final_information_fit") != {
        "boundary": 2014,
        "fit_max_year": 2013,
    }:
        raise ValueError("config final information fit drifted")

    rng = _mapping(config.get("rng"), where="config.rng")
    if rng.get("seed_root") != 5200:
        raise ValueError("config RNG seed root drifted")
    if rng.get("selection_seeds") != list(SEEDS):
        raise ValueError("config RNG seed bank drifted")
    if rng.get("selection_seed_blocks") != [list(block) for block in BLOCKS]:
        raise ValueError("config RNG blocks drifted")
    if rng.get("transition_uniform_address") != {
        "child_index": 0,
        "module": "MARITAL_CORE",
        "period": 0,
    }:
        raise ValueError("config transition RNG address drifted")
    if rng.get("spouse_gap_address") != {
        "child_index": 1,
        "module": "MARITAL_CORE",
        "period": 0,
    }:
        raise ValueError("config spouse-gap RNG address drifted")

    family = _mapping(config.get("family"), where="config.family")
    if family.get("origins") != list(ORIGINS):
        raise ValueError("config origins drifted")
    if family.get("law_order") != list(LAWS):
        raise ValueError("config family law order drifted")
    if family.get("nonzero_law_order") != list(LAWS[1:]):
        raise ValueError("config nonzero law order drifted")
    if family.get("cumulative_nonzero_laws_two_rounds") != 7:
        raise ValueError("config accumulated-law count drifted")
    components = _mapping(
        family.get("law_components"), where="config.family.law_components"
    )
    expected_components = {
        "R0": (0.0, 0.0, {"divorced": "NONE", "widowed": "NONE"}),
        "R_D50_W00": (
            0.5,
            0.0,
            {"divorced": "R_D50", "widowed": "NONE"},
        ),
        "R_D75_W00": (
            0.75,
            0.0,
            {"divorced": "R_D75", "widowed": "NONE"},
        ),
        "R_D50_W05": (
            0.5,
            0.05,
            {"divorced": "R_D50", "widowed": "R_W05"},
        ),
        "R_D75_W05": (
            0.75,
            0.05,
            {"divorced": "R_D75", "widowed": "R_W05"},
        ),
    }
    if set(components) != set(expected_components):
        raise ValueError("config law components drifted")
    for law, (k, omega, outcome) in expected_components.items():
        component = _mapping(
            components[law], where=f"config.family.law_components.{law}"
        )
        if (
            component.get("k") != k
            or component.get("divorced_k") != k
            or component.get("omega") != omega
            or component.get("widowed_omega") != omega
            or component.get("per_origin_outcome") != outcome
        ):
            raise ValueError(f"config law component drifted for {law}")

    construction = _mapping(
        config.get("construction"), where="config.construction"
    )
    critical_construction = {
        "raw_age_delta_domain": [18, 64],
        "outside_raw_age_domain": "exact_R0_probability",
        "ysd_band_order": ["0-4", "5-9", "10-120"],
        "ysd_contrast": [1.0, 0.0, -1.0],
        "divorced_strengths": [0.5, 0.75],
        "widowed_logit_options": [0.0, 0.05],
        "widowed_log_rate_budget": WIDOWED_BUDGET,
        "area_tolerance": AREA_TOLERANCE,
        "sum_algorithm": "fixed_adjacent_pairwise_float64",
        "root_algorithm": "first_accepted_midpoint_bisection",
        "root_bracket": [0.0, 16.0],
        "root_maximum_iterations": 200,
        "root_midpoint_expression": "lo + (hi - lo) / 2",
    }
    for key, expected in critical_construction.items():
        if construction.get(key) != expected:
            raise ValueError(f"config construction field {key} drifted")

    numeric = _mapping(
        config.get("numeric_protocol"), where="config.numeric_protocol"
    )
    if numeric.get("area_relative_tolerance") != AREA_TOLERANCE:
        raise ValueError("config numeric area tolerance drifted")
    if numeric.get("selector_comparison_tolerance") != TOLERANCE:
        raise ValueError("config numeric selector tolerance drifted")
    if numeric.get("sum_algorithm") != construction["sum_algorithm"]:
        raise ValueError("config summation declarations disagree")

    selector = _mapping(config.get("selector"), where="config.selector")
    if selector.get("comparison_tolerance") != TOLERANCE:
        raise ValueError("config selector tolerance drifted")
    if selector.get("rules") != 7:
        raise ValueError("config selector rule count drifted")
    if selector.get("jackknife_replicates") != 40:
        raise ValueError("config jackknife replicate count drifted")
    if selector.get("delete_one_seed_count") != 39:
        raise ValueError("config jackknife retained count drifted")
    if (
        selector.get("jackknife_delete_same_seed_across_all_boundaries")
        is not True
    ):
        raise ValueError("config jackknife deletion rule drifted")
    if selector.get("jackknife_reselect_inside_replicate") is not False:
        raise ValueError("config jackknife reselection rule drifted")
    if selector.get("selected_outcome_if_no_op") != "NO_OP_DESIGNED_PAUSE":
        raise ValueError("config no-op disposition drifted")
    if (
        selector.get("selected_outcome_if_law")
        != "SELECTED_LAW_FOR_AMENDMENT_PROPOSAL"
    ):
        raise ValueError("config selected-law disposition drifted")

    output = _mapping(config.get("output"), where="config.output")
    expected_output = {
        "selector_script": "scripts/select_m6_remarriage_round3.py",
        "config_file": f"scripts/{CONFIG_FILENAME}",
        "reducer_script": "scripts/reduce_m6_remarriage_round3.py",
        "raw_schema": RAW_SCHEMA,
        "raw_stdout_external_filename": (
            "m6_remarriage_round3_selection_full.json"
        ),
        "full_stdout_json": (
            "docs/analysis/m6_remarriage_round3_selection_full.json"
        ),
        "findings_schema": FINDINGS_SCHEMA,
        "findings_json": (
            "docs/analysis/m6_remarriage_round3_selection_results.json"
        ),
        "findings_report": ("docs/analysis/m6_remarriage_round3_selection.md"),
    }
    _assert_same(output, expected_output, where="config output lock")


def _validate_config_against_round1(
    config: dict[str, Any], round1: dict[str, Any]
) -> None:
    _assert_same(
        config["expected_source_audit"],
        round1.get("source_audit"),
        where="config source audit versus pinned round-1 ledger",
    )
    round1_boundaries = _mapping(
        round1.get("boundaries"), where="round1.boundaries"
    )
    for boundary in BOUNDARIES:
        source = _mapping(
            round1_boundaries.get(boundary),
            where=f"round1.boundaries.{boundary}",
        )
        locked = _mapping(
            config["expected_pseudo_holdouts"].get(boundary),
            where=f"config.expected_pseudo_holdouts.{boundary}",
        )
        for key, expected in locked.items():
            if key not in source:
                raise ValueError(
                    f"round-1 ledger lacks frozen pseudo field {boundary}.{key}"
                )
            _assert_same(
                source[key],
                expected,
                where=f"config pseudo lock {boundary}.{key}",
            )


def _validate_freeze(
    ledger: dict[str, Any], config: dict[str, Any], root: Path
) -> None:
    if ledger.get("authority") != config.get("authority"):
        raise ValueError(
            "raw authority does not equal frozen config authority"
        )
    if ledger.get("protocol") != config.get("protocol"):
        raise ValueError("raw protocol does not equal frozen config protocol")

    freeze = _mapping(ledger.get("freeze"), where="freeze")
    _strict_keys(
        freeze,
        {
            "freeze_commit",
            "branch",
            "worktree_clean",
            "frozen_blob_sha1",
            "frozen_file_sha256",
            "authority_sha256",
            "source_tree_sha1",
            "import_paths",
            "all_imports_from_frozen_tree",
        },
        where="freeze",
    )
    commit = freeze["freeze_commit"]
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise ValueError("freeze.freeze_commit must be a full lowercase SHA-1")
    if _git(root, "rev-parse", commit) != commit:
        raise ValueError("freeze commit is not locally resolvable")
    if not isinstance(freeze["branch"], str) or not freeze["branch"]:
        raise ValueError("freeze branch is missing")
    _true(freeze["worktree_clean"], where="freeze.worktree_clean")
    _true(
        freeze["all_imports_from_frozen_tree"],
        where="freeze.all_imports_from_frozen_tree",
    )

    frozen_paths = tuple(config["freeze"]["paths"])
    blobs = _mapping(
        freeze["frozen_blob_sha1"], where="freeze.frozen_blob_sha1"
    )
    hashes = _mapping(
        freeze["frozen_file_sha256"], where="freeze.frozen_file_sha256"
    )
    if set(blobs) != set(frozen_paths) or set(hashes) != set(frozen_paths):
        raise ValueError("freeze file maps do not match configured paths")
    for relative in frozen_paths:
        blob = blobs[relative]
        if not isinstance(blob, str) or HEX40.fullmatch(blob) is None:
            raise ValueError(f"freeze blob hash is invalid for {relative}")
        if _git(root, "rev-parse", f"{commit}:{relative}") != blob:
            raise ValueError(
                f"freeze blob does not match commit for {relative}"
            )
        observed = _sha256_file(root / relative)
        if (
            _sha256(hashes[relative], where=f"freeze hash {relative}")
            != observed
        ):
            raise ValueError(
                f"current bytes differ from frozen run: {relative}"
            )

    source_tree = freeze["source_tree_sha1"]
    if (
        _git(root, "rev-parse", f"{commit}:src/populace_dynamics")
        != source_tree
    ):
        raise ValueError("frozen source-tree object does not match commit")

    authority_paths = {
        "design_sha256": (
            root / "docs/design/m6_remarriage_learning_plan_round2.md"
        ),
        "validation_sha256": (
            root
            / "docs/design/m6_remarriage_learning_plan_round2_validation.json"
        ),
        "round1_ledger_sha256": (
            root / "docs/analysis/m6_remarriage_train_only_delta_results.json"
        ),
    }
    expected_authority = {
        key: _sha256_file(path) for key, path in authority_paths.items()
    }
    _assert_same(
        freeze["authority_sha256"],
        expected_authority,
        where="freeze authority hashes",
    )
    for key, observed in expected_authority.items():
        if config["authority"].get(key) != observed:
            raise ValueError(f"config authority hash drifted for {key}")

    imports = _mapping(freeze["import_paths"], where="freeze.import_paths")
    if set(imports) != {
        "round1_chassis",
        "marital_engine",
        "refit",
        "remarriage_component",
    }:
        raise ValueError("freeze import audit fields drifted")
    if not all(isinstance(path, str) and path for path in imports.values()):
        raise ValueError("freeze import audit contains an invalid path")


def _validate_runtime(ledger: dict[str, Any], config: dict[str, Any]) -> None:
    runtime = _mapping(ledger.get("runtime"), where="runtime")
    _strict_keys(
        runtime,
        {
            "python_implementation",
            "python_version",
            "numpy_version",
            "python_executable",
            "dont_write_bytecode",
            "dependencies",
            "forbidden_m6_modules_imported",
        },
        where="runtime",
    )
    expected = config["runtime_numeric_identity"]
    observed = {key: runtime[key] for key in expected}
    _assert_same(observed, expected, where="raw runtime identity")
    if (
        not isinstance(runtime["python_executable"], str)
        or not runtime["python_executable"]
    ):
        raise ValueError("raw runtime executable is invalid")
    _true(runtime["dont_write_bytecode"], where="runtime.dont_write_bytecode")
    if runtime["forbidden_m6_modules_imported"] != []:
        raise ValueError("raw runtime imported a forbidden M6 module")
    dependencies = _mapping(
        runtime["dependencies"], where="runtime.dependencies"
    )
    if set(dependencies) != {
        "pandas",
        "scipy",
        "scikit-learn",
        "quantile-forest",
        "policyengine-us",
        "populace-fit",
        "populace-frame",
    }:
        raise ValueError("runtime dependency audit fields drifted")
    if any(
        not isinstance(value, str) or not value
        for value in dependencies.values()
    ):
        raise ValueError("runtime fitting dependency is missing")
    reducer_runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    _assert_same(
        reducer_runtime,
        expected,
        where="reducer numeric runtime identity",
    )


def _ratio_fields(
    projected: dict[str, Any], truth: dict[str, Any]
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for quantity in ("exposure", "numerator", "rate"):
        numerator = projected[quantity]
        denominator = truth[quantity]
        ratio = (
            float(numerator / denominator)
            if numerator is not None
            and denominator is not None
            and float(denominator) > 0.0
            else None
        )
        output[f"{quantity}_ratio"] = ratio
        output[f"log_{quantity}_ratio"] = (
            math.log(ratio) if ratio is not None and ratio > 0.0 else None
        )
    return output


def _validate_rate_record(
    value: Any,
    *,
    where: str,
    truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = _mapping(value, where=where)
    expected_fields = PROJECTED_FIELDS if truth is not None else RATE_FIELDS
    _strict_keys(record, expected_fields, where=where)
    risk_rows = _nonnegative_int(
        record["risk_rows"], where=f"{where}.risk_rows"
    )
    event_rows = _nonnegative_int(
        record["event_rows"], where=f"{where}.event_rows"
    )
    exposure = _finite_number(record["exposure"], where=f"{where}.exposure")
    numerator = _finite_number(record["numerator"], where=f"{where}.numerator")
    if exposure < 0.0 or numerator < 0.0:
        raise ValueError(f"{where} exposure/numerator must be nonnegative")
    expected_rate = numerator / exposure if exposure > 0.0 else None
    _assert_same(record["rate"], expected_rate, where=f"{where}.rate")
    if record["rate"] is not None:
        _finite_number(record["rate"], where=f"{where}.rate")
    if truth is not None:
        expected_ratios = _ratio_fields(record, truth)
        actual_ratios = {key: record[key] for key in RATIO_FIELDS}
        _assert_same(actual_ratios, expected_ratios, where=f"{where} ratios")
    return {
        **record,
        "risk_rows": risk_rows,
        "event_rows": event_rows,
    }


def _validate_mean_record(
    value: Any,
    *,
    where: str,
    truth: dict[str, Any],
    contributors: list[dict[str, Any]],
) -> dict[str, Any]:
    record = _mapping(value, where=where)
    _strict_keys(record, MEAN_FIELDS, where=where)
    _nonnegative_int(record["seed_count"], where=f"{where}.seed_count")
    for key in (
        "mean_risk_rows",
        "mean_event_rows",
        "exposure",
        "numerator",
    ):
        number = _finite_number(record[key], where=f"{where}.{key}")
        if number < 0.0:
            raise ValueError(f"{where}.{key} must be nonnegative")
    any_null_rate = any(
        contributor["rate"] is None for contributor in contributors
    )
    if (record["rate"] is None) != any_null_rate:
        raise ValueError(
            f"{where}.rate must be null iff at least one contributing "
            "per-seed rate is null"
        )
    if record["rate"] is not None:
        rate = _finite_number(record["rate"], where=f"{where}.rate")
        if rate < 0.0:
            raise ValueError(f"{where}.rate must be nonnegative")
    expected_ratios = _ratio_fields(record, truth)
    _assert_same(
        {key: record[key] for key in RATIO_FIELDS},
        expected_ratios,
        where=f"{where} ratios",
    )
    return record


def _validate_group_array(
    value: Any,
    *,
    where: str,
    validator: Callable[..., dict[str, Any]],
    truth_groups: dict[str, dict[str, Any]] | None = None,
    contributor_groups: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    groups = _array(value, where=where)
    if len(groups) != len(EXPECTED_GROUPS):
        raise ValueError(f"{where} must contain exactly 18 publication groups")
    output: list[dict[str, Any]] = []
    for index, (group, origin, age, ysd) in enumerate(EXPECTED_GROUPS):
        item_where = f"{where}[{index}]"
        item = _mapping(groups[index], where=item_where)
        if item.get("group") != group:
            raise ValueError(f"{item_where}.group drifted")
        if item.get("origin") != origin:
            raise ValueError(f"{item_where}.origin drifted")
        if item.get("age_band") != list(age):
            raise ValueError(f"{item_where}.age_band drifted")
        if item.get("ysd_band") != list(ysd):
            raise ValueError(f"{item_where}.ysd_band drifted")
        payload = {
            key: value
            for key, value in item.items()
            if key not in {"group", "origin", "age_band", "ysd_band"}
        }
        truth = None if truth_groups is None else truth_groups[group]
        if contributor_groups is None:
            validated = validator(payload, where=item_where, truth=truth)
        else:
            validated = validator(
                payload,
                where=item_where,
                truth=truth,
                contributors=[groups[index] for groups in contributor_groups],
            )
        expected_keys = {
            "group",
            "origin",
            "age_band",
            "ysd_band",
            *validated.keys(),
        }
        if set(item) != expected_keys:
            raise ValueError(f"{item_where} fields drifted")
        output.append(item)
    return output


def _validate_direct_record(
    value: Any,
    *,
    where: str,
    truth: Any = None,
    allow_zero_exposure: bool = False,
) -> dict[str, Any]:
    del truth
    record = _mapping(value, where=where)
    _strict_keys(record, DIRECT_FIELDS, where=where)
    for key in (
        "risk_rows",
        "event_rows",
        "matchable_positive_weight_event_rows",
        "matchable_event_rows",
        "unmatched_same_year_event_rows",
    ):
        _nonnegative_int(record[key], where=f"{where}.{key}")
    for key in (
        "unmatched_same_year_event_weight",
        "exposure",
        "deviance_exposure",
        "actual_numerator",
        "matchable_numerator",
        "row_expected_numerator",
        "expected_numerator",
        "weighted_deviance_numerator",
    ):
        number = _finite_number(record[key], where=f"{where}.{key}")
        if number < 0.0:
            raise ValueError(f"{where}.{key} must be nonnegative")
    exposure = float(record["exposure"])
    if exposure < 0.0 or (exposure == 0.0 and not allow_zero_exposure):
        raise ValueError(f"{where}.exposure must be positive")
    _assert_same(
        record["deviance_exposure"],
        exposure,
        where=f"{where}.deviance_exposure",
    )
    actual_rate = record["actual_numerator"] / exposure if exposure else None
    _assert_same(
        record["actual_rate"], actual_rate, where=f"{where}.actual_rate"
    )
    _assert_same(
        record["expected_numerator"],
        record["row_expected_numerator"],
        where=f"{where}.expected_numerator",
    )
    expected_rate = (
        record["expected_numerator"] / exposure if exposure else None
    )
    _assert_same(
        record["expected_rate"], expected_rate, where=f"{where}.expected_rate"
    )
    _assert_same(record["qdir"], expected_rate, where=f"{where}.qdir")
    expected_deviance = (
        record["weighted_deviance_numerator"] / exposure if exposure else None
    )
    _assert_same(
        record["weighted_bernoulli_deviance"],
        expected_deviance,
        where=f"{where}.weighted_bernoulli_deviance",
    )
    return record


def _validate_direct_group_record(
    value: Any, *, where: str, truth: Any = None
) -> dict[str, Any]:
    return _validate_direct_record(
        value,
        where=where,
        truth=truth,
        allow_zero_exposure=True,
    )


def _plain_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean_record(
    records: list[dict[str, Any]], truth: dict[str, Any]
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty seed list")
    projected = {
        quantity: float(
            np.mean(
                np.asarray(
                    [record[quantity] for record in records],
                    dtype=np.float64,
                )
            )
        )
        for quantity in ("exposure", "numerator", "rate")
    }
    projected = {
        quantity: _plain_float(value) for quantity, value in projected.items()
    }
    return {
        "seed_count": len(records),
        "mean_risk_rows": float(
            np.mean([record["risk_rows"] for record in records])
        ),
        "mean_event_rows": float(
            np.mean([record["event_rows"] for record in records])
        ),
        **projected,
        **_ratio_fields(projected, truth),
    }


def _aggregate_projection(
    rows: list[dict[str, Any]],
    truth: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    wanted = set(seeds)
    selected = [row for row in rows if row["seed"] in wanted]
    if [row["seed"] for row in selected] != list(seeds):
        raise ValueError("aggregate seed order drifted")
    truth_groups = {row["group"]: row for row in truth["groups"]}
    groups: list[dict[str, Any]] = []
    for index, (group, origin, age, ysd) in enumerate(EXPECTED_GROUPS):
        group_rows = [row["publication_groups"][index] for row in selected]
        groups.append(
            {
                "group": group,
                "origin": origin,
                "age_band": list(age),
                "ysd_band": list(ysd),
                **_mean_record(group_rows, truth_groups[group]),
            }
        )
    return {
        "pooled": _mean_record(
            [row["pooled"] for row in selected], truth["pooled"]
        ),
        "origin": {
            origin: _mean_record(
                [row["origin"][origin] for row in selected],
                truth["origin"][origin],
            )
            for origin in ORIGINS
        },
        "groups": groups,
    }


LAW_PUBLIC_FIELDS = {
    "law",
    "divorced_k",
    "widowed_omega",
    "beta_frontload",
    "table_sha256",
    "cells",
    "Delta_divorced",
    "Delta_widowed",
    "applied_divorced_direction",
    "divorced_area_target",
    "divorced_area",
    "divorced_area_relative_residual",
    "widowed_area_target",
    "widowed_area",
    "widowed_area_relative_residual",
    "positive_exposure_divorced_cells_rising",
    "positive_exposure_divorced_cells_falling",
    "widowed_working_age_cell_logit_shifts",
    "widowed_table_construction_exact",
    "probability_min",
    "probability_max",
    "raw_age_outside_18_64_exact_R0",
}
FIT_FIELDS = {
    "fit_max_year",
    "fit_person_year_rows",
    "fit_event_rows",
    "dissolved_rows",
    "remarriage_events",
    "wbar",
    "incumbent_table_sha256",
    "fit_support_by_origin_ysd",
    "fit_exposure_center",
    "centered_contrast",
    "reference_spells",
    "reference_exclusion_category_hashes",
    "divorced_calibration",
    "widowed_targets",
    "support_struck_named_laws",
    "validation_match",
    "field_boundary_audit",
    "laws",
}
REFERENCE_AUDIT_FIELDS = {
    "R0_area",
    "eligible_spells_before_exclusions",
    "duplicate_key_groups",
    "duplicate_spells_excluded",
    "duplicate_spells_excluded_weight",
    "duplicate_spells_checksum_sha256",
    "same_year_remarriage_spells_excluded",
    "same_year_remarriage_spells_excluded_weight",
    "same_year_remarriage_spells_checksum_sha256",
    "missing_required_or_nonpositive_weight_spells_excluded",
    "missing_required_or_nonpositive_weight_spells_excluded_weight",
    "missing_required_or_nonpositive_weight_checksum_sha256",
    "no_potential_path_spells_excluded",
    "no_potential_path_spells_excluded_weight",
    "no_potential_path_spells_checksum_sha256",
    "included_spells",
    "included_spell_weight",
    "included_spells_checksum_sha256",
    "potential_path_years",
    "working_age_path_year_terms",
    "path_checksum_sha256",
}
REFERENCE_AUDIT_ADDITIVE_FIELDS = {
    "duplicate_spells_checksum_sha256",
    "same_year_remarriage_spells_checksum_sha256",
    "missing_required_or_nonpositive_weight_spells_excluded_weight",
    "missing_required_or_nonpositive_weight_checksum_sha256",
    "no_potential_path_spells_excluded_weight",
    "no_potential_path_spells_checksum_sha256",
    "included_spells_checksum_sha256",
}
REFERENCE_EXCLUSION_CATEGORY_HASH_FIELDS = (
    "duplicate_spells_checksum_sha256",
    "same_year_remarriage_spells_checksum_sha256",
    "missing_required_or_nonpositive_weight_checksum_sha256",
    "no_potential_path_spells_checksum_sha256",
    "included_spells_checksum_sha256",
    "path_checksum_sha256",
)
DIVORCED_CALIBRATION_FIELDS = {
    "alpha_divorced",
    "beta_frontload",
    "fit_exposure_center",
    "centered_contrast",
    "bracket_residual_low",
    "bracket_residual_high",
    "root_iterations",
    "area_R0",
    "candidate_area",
    "area_relative_residual",
    "pairwise_term_count",
    "effective_divorced_shift",
    "minimum_cell_logit_shift",
    "maximum_cell_logit_shift",
    "candidate_probability_min",
    "candidate_probability_max",
}
DIVORCED_CALIBRATION_ADDITIVE_FIELDS = {
    "area_R0",
    "candidate_area",
    "pairwise_term_count",
}
CELL_AGE_BANDS = ((18, 34), (35, 49), (50, 64), (65, 74), (75, 120))
CELL_FIELDS = {
    "age_band_index",
    "age_band",
    "ysd_band_index",
    "ysd_band",
    "origin",
    "sex",
    "risk_rows",
    "event_rows",
    "weighted_exposure",
    "weighted_events",
    "R0_probability",
    "probability",
}


def _table_sha256(cells: list[dict[str, Any]]) -> str:
    table = {
        (
            int(cell["age_band_index"]),
            int(cell["ysd_band_index"]),
            str(cell["origin"]),
            str(cell["sex"]),
        ): float(cell["probability"])
        for cell in cells
    }
    ordered = sorted(table, key=lambda key: (key[2], key[0], key[1], key[3]))
    rows = [
        "|".join(map(str, key)) + "|" + float(table[key]).hex()
        for key in ordered
    ]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _validate_law_public(
    value: Any, *, where: str, law: str, config: dict[str, Any]
) -> dict[str, Any]:
    public = _mapping(value, where=where)
    _strict_keys(public, LAW_PUBLIC_FIELDS, where=where)
    component = config["family"]["law_components"][law]
    if public["law"] != law:
        raise ValueError(f"{where}.law drifted")
    if public["divorced_k"] != component["divorced_k"]:
        raise ValueError(f"{where}.divorced_k drifted")
    if public["widowed_omega"] != component["widowed_omega"]:
        raise ValueError(f"{where}.widowed_omega drifted")
    for key in (
        "beta_frontload",
        "Delta_divorced",
        "Delta_widowed",
        "applied_divorced_direction",
        "divorced_area_target",
        "divorced_area",
        "divorced_area_relative_residual",
        "widowed_area_target",
        "widowed_area",
        "widowed_area_relative_residual",
        "probability_min",
        "probability_max",
    ):
        _finite_number(public[key], where=f"{where}.{key}")
    for key in (
        "positive_exposure_divorced_cells_rising",
        "positive_exposure_divorced_cells_falling",
    ):
        _nonnegative_int(public[key], where=f"{where}.{key}")
    _true(
        public["raw_age_outside_18_64_exact_R0"],
        where=f"{where}.raw_age_outside_18_64_exact_R0",
    )
    if not isinstance(public["widowed_table_construction_exact"], bool):
        raise ValueError(f"{where}.widowed_table_construction_exact invalid")
    shifts = _array(
        public["widowed_working_age_cell_logit_shifts"],
        where=f"{where}.widowed_working_age_cell_logit_shifts",
    )
    if len(shifts) != 18:
        raise ValueError(f"{where} must publish 18 widowed cell shifts")
    for index, shift in enumerate(shifts):
        _finite_number(shift, where=f"{where}.widowed_shift[{index}]")

    cells = _array(public["cells"], where=f"{where}.cells")
    if len(cells) != 60:
        raise ValueError(f"{where}.cells must contain all 60 table cells")
    expected_cells = [
        (age_index, age, ysd_index, ysd, origin, sex)
        for age_index, age in enumerate(CELL_AGE_BANDS)
        for ysd_index, ysd in enumerate(YSD_BANDS)
        for origin in ORIGINS
        for sex in ("female", "male")
    ]
    probabilities: list[float] = []
    for index, expected in enumerate(expected_cells):
        cell_where = f"{where}.cells[{index}]"
        cell = _mapping(cells[index], where=cell_where)
        _strict_keys(cell, CELL_FIELDS, where=cell_where)
        age_index, age, ysd_index, ysd, origin, sex = expected
        identity = (
            cell["age_band_index"],
            tuple(cell["age_band"]),
            cell["ysd_band_index"],
            tuple(cell["ysd_band"]),
            cell["origin"],
            cell["sex"],
        )
        if identity != (age_index, age, ysd_index, ysd, origin, sex):
            raise ValueError(f"{cell_where} identity/order drifted")
        _nonnegative_int(cell["risk_rows"], where=f"{cell_where}.risk_rows")
        _nonnegative_int(cell["event_rows"], where=f"{cell_where}.event_rows")
        for key in ("weighted_exposure", "weighted_events"):
            if _finite_number(cell[key], where=f"{cell_where}.{key}") < 0.0:
                raise ValueError(f"{cell_where}.{key} must be nonnegative")
        baseline = _finite_number(
            cell["R0_probability"], where=f"{cell_where}.R0_probability"
        )
        probability = _finite_number(
            cell["probability"], where=f"{cell_where}.probability"
        )
        if not (0.0 < baseline < 1.0 and 0.0 < probability < 1.0):
            raise ValueError(f"{cell_where} probability outside (0,1)")
        if law == "R0" and probability != baseline:
            raise ValueError(f"{cell_where} R0 table is not exact")
        if age_index >= 3 and probability != baseline:
            raise ValueError(
                f"{cell_where} outside-domain probability changed"
            )
        if age_index < 3 and origin == "widowed":
            omega = float(component["widowed_omega"])
            expected_widowed = (
                baseline
                if omega == 0.0
                else float(
                    scipy_expit(
                        np.asarray(
                            float(
                                scipy_logit(
                                    np.asarray(baseline, dtype=np.float64)
                                )
                            )
                            + omega,
                            dtype=np.float64,
                        )
                    )
                )
            )
            if probability != expected_widowed:
                raise ValueError(
                    f"{cell_where} widowed table construction drifted"
                )
        probabilities.append(probability)
    if public["probability_min"] != min(probabilities):
        raise ValueError(f"{where}.probability_min is inconsistent")
    if public["probability_max"] != max(probabilities):
        raise ValueError(f"{where}.probability_max is inconsistent")
    if _sha256(
        public["table_sha256"], where=f"{where}.table_sha256"
    ) != _table_sha256(cells):
        raise ValueError(f"{where}.table_sha256 is inconsistent")
    return public


def _validate_reference_spells_audit(
    value: Any, expected: Any, *, where: str
) -> dict[str, Any]:
    references = _mapping(value, where=where)
    expected_references = _mapping(expected, where=f"{where} validation")
    _strict_keys(references, set(ORIGINS), where=where)
    _strict_keys(
        expected_references,
        set(ORIGINS),
        where=f"{where} validation",
    )
    shared_references: dict[str, Any] = {}
    for origin in ORIGINS:
        audit_where = f"{where}.{origin}"
        audit = _mapping(references[origin], where=audit_where)
        expected_audit = _mapping(
            expected_references[origin],
            where=f"{audit_where} validation",
        )
        _strict_keys(audit, REFERENCE_AUDIT_FIELDS, where=audit_where)
        _strict_keys(
            expected_audit,
            REFERENCE_AUDIT_FIELDS.difference(REFERENCE_AUDIT_ADDITIVE_FIELDS),
            where=f"{audit_where} validation",
        )
        for key in REFERENCE_EXCLUSION_CATEGORY_HASH_FIELDS:
            _sha256(audit[key], where=f"{audit_where}.{key}")
        for key in (
            "missing_required_or_nonpositive_weight_spells_excluded_weight",
            "no_potential_path_spells_excluded_weight",
        ):
            if _finite_number(audit[key], where=f"{audit_where}.{key}") < 0.0:
                raise ValueError(f"{audit_where}.{key} must be nonnegative")
        shared_references[origin] = {key: audit[key] for key in expected_audit}
    _assert_same(shared_references, expected_references, where=where)
    return references


def _validate_divorced_calibration(
    value: Any, expected: Any, *, where: str
) -> dict[str, Any]:
    calibration = _mapping(value, where=where)
    expected_calibration = _mapping(
        expected,
        where=f"{where} validation",
    )
    expected_subset = {
        alpha: expected_calibration[alpha] for alpha in calibration
    }
    shared_calibration: dict[str, Any] = {}
    for alpha in calibration:
        alpha_where = f"{where}.{alpha}"
        alpha_calibration = _mapping(
            calibration[alpha],
            where=alpha_where,
        )
        expected_alpha = _mapping(
            expected_subset[alpha],
            where=f"{alpha_where} validation",
        )
        _strict_keys(
            alpha_calibration,
            DIVORCED_CALIBRATION_FIELDS,
            where=alpha_where,
        )
        _strict_keys(
            expected_alpha,
            DIVORCED_CALIBRATION_FIELDS.difference(
                DIVORCED_CALIBRATION_ADDITIVE_FIELDS
            ),
            where=f"{alpha_where} validation",
        )
        for key in ("area_R0", "candidate_area"):
            _finite_number(
                alpha_calibration[key],
                where=f"{alpha_where}.{key}",
            )
        _nonnegative_int(
            alpha_calibration["pairwise_term_count"],
            where=f"{alpha_where}.pairwise_term_count",
        )
        shared_calibration[alpha] = {
            key: alpha_calibration[key] for key in expected_alpha
        }
    _assert_same(shared_calibration, expected_subset, where=where)
    return calibration


def _validate_fit_public(
    value: Any,
    *,
    where: str,
    boundary: str,
    laws: tuple[str, ...],
    config: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    fit = _mapping(value, where=where)
    _strict_keys(fit, FIT_FIELDS, where=where)
    expected = _mapping(
        validation["boundaries"][boundary],
        where=f"validation.boundaries.{boundary}",
    )
    if fit["fit_max_year"] != expected["fit_max_year"]:
        raise ValueError(f"{where}.fit_max_year differs from validation")
    for key in (
        "fit_person_year_rows",
        "fit_event_rows",
        "dissolved_rows",
        "remarriage_events",
    ):
        _nonnegative_int(fit[key], where=f"{where}.{key}")
    if _finite_number(fit["wbar"], where=f"{where}.wbar") <= 0.0:
        raise ValueError(f"{where}.wbar must be positive")
    if fit["incumbent_table_sha256"] != expected["incumbent_table_sha256"]:
        raise ValueError(f"{where} incumbent table differs from validation")
    references: dict[str, Any]
    for key in (
        "fit_support_by_origin_ysd",
        "reference_spells",
        "divorced_calibration",
        "widowed_targets",
        "support_struck_named_laws",
    ):
        expected_value = expected[key]
        if key == "reference_spells":
            references = _validate_reference_spells_audit(
                fit[key],
                expected_value,
                where=f"{where}.{key}",
            )
            continue
        if key == "divorced_calibration":
            _validate_divorced_calibration(
                fit[key],
                expected_value,
                where=f"{where}.{key}",
            )
            continue
        if key == "widowed_targets":
            expected_value = {name: expected_value[name] for name in fit[key]}
        _assert_same(fit[key], expected_value, where=f"{where}.{key}")
    if fit["support_struck_named_laws"] != []:
        raise ValueError(f"{where} unexpectedly struck a named law")
    _true(fit["validation_match"], where=f"{where}.validation_match")
    field_audit = _mapping(
        fit["field_boundary_audit"], where=f"{where}.field_boundary_audit"
    )
    _strict_keys(
        field_audit,
        {
            "boundary",
            "field_maxima",
            "post_boundary_end_and_separation_fields_nulled",
            "marriage_counts_recomputed",
            "every_fit_frame_field_asserted",
        },
        where=f"{where}.field_boundary_audit",
    )
    if field_audit["boundary"] != int(boundary):
        raise ValueError(f"{where} field-audit boundary drifted")
    maxima = _mapping(
        field_audit["field_maxima"],
        where=f"{where}.field_boundary_audit.field_maxima",
    )
    for name, maximum in maxima.items():
        if not isinstance(name, str) or (
            maximum is not None
            and (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum > int(boundary)
            )
        ):
            raise ValueError(f"{where} invalid field maximum {name!r}")
    for key in (
        "post_boundary_end_and_separation_fields_nulled",
        "marriage_counts_recomputed",
        "every_fit_frame_field_asserted",
    ):
        _true(field_audit[key], where=f"{where}.field_boundary_audit.{key}")
    center = _finite_number(
        fit["fit_exposure_center"], where=f"{where}.fit_exposure_center"
    )
    centered = _array(
        fit["centered_contrast"], where=f"{where}.centered_contrast"
    )
    if len(centered) != 3 or not all(
        math.isfinite(float(value)) for value in centered
    ):
        raise ValueError(f"{where}.centered_contrast is invalid")
    if centered != [1.0 - center, -center, -1.0 - center]:
        raise ValueError(f"{where}.centered_contrast is inconsistent")
    hashes = _mapping(
        fit["reference_exclusion_category_hashes"],
        where=f"{where}.reference_exclusion_category_hashes",
    )
    expected_hashes = {
        origin: {
            key: references[origin][key]
            for key in REFERENCE_EXCLUSION_CATEGORY_HASH_FIELDS
        }
        for origin in ORIGINS
    }
    _assert_same(hashes, expected_hashes, where=f"{where} reference hashes")
    public_laws = _mapping(fit["laws"], where=f"{where}.laws")
    if set(public_laws) != set(laws):
        raise ValueError(f"{where}.laws content drifted")
    for law in laws:
        _validate_law_public(
            public_laws[law],
            where=f"{where}.laws.{law}",
            law=law,
            config=config,
        )
    return fit


def _validate_input_audit(
    ledger: dict[str, Any], config: dict[str, Any]
) -> None:
    audit = _mapping(
        ledger.get("input_and_file_open_audit"),
        where="input_and_file_open_audit",
    )
    _strict_keys(
        audit,
        {
            "selection_relevant_reads",
            "maximum_information_year",
            "source_audit",
            "dynamic_open_audit_installed_before_selection_reads",
            "observed_open_path_count",
            "observed_open_paths",
            "gates_yaml_read",
            "runs_artifact_read",
            "M6_scorer_imported",
            "post_2014_selection_data_read",
            "helper_wrote_files",
            "stdout_machine_documents",
        },
        where="input_and_file_open_audit",
    )
    if audit["maximum_information_year"] != 2014:
        raise ValueError("input audit maximum information year drifted")
    if audit["stdout_machine_documents"] != 1:
        raise ValueError(
            "selector stdout did not contain one machine document"
        )
    _true(
        audit["dynamic_open_audit_installed_before_selection_reads"],
        where="input_and_file_open_audit.dynamic_open_audit_installed",
    )
    observed_paths = _array(
        audit["observed_open_paths"],
        where="input_and_file_open_audit.observed_open_paths",
    )
    if (
        audit["observed_open_path_count"] != len(observed_paths)
        or observed_paths != sorted(set(observed_paths))
        or not all(isinstance(path, str) and path for path in observed_paths)
    ):
        raise ValueError("dynamic file-open path audit is invalid")
    root = Path(__file__).resolve().parents[1]
    required_observed = {
        str((root / relative).resolve())
        for relative in (
            f"scripts/{CONFIG_FILENAME}",
            "docs/design/m6_remarriage_learning_plan_round2.md",
            "docs/design/m6_remarriage_learning_plan_round2_validation.json",
            "docs/analysis/m6_remarriage_train_only_delta_results.json",
        )
    }
    if not required_observed.issubset(set(observed_paths)):
        raise ValueError("dynamic file-open audit omitted an authority read")
    for key in (
        "gates_yaml_read",
        "runs_artifact_read",
        "M6_scorer_imported",
        "post_2014_selection_data_read",
        "helper_wrote_files",
    ):
        _false(audit[key], where=f"input_and_file_open_audit.{key}")
    expected_reads = [
        f"scripts/{CONFIG_FILENAME}",
        "docs/design/m6_remarriage_learning_plan_round2.md",
        "docs/design/m6_remarriage_learning_plan_round2_validation.json",
        "docs/analysis/m6_remarriage_train_only_delta_results.json",
        "sanitized staged PSID sources through the frozen round-1 chassis",
    ]
    if audit["selection_relevant_reads"] != expected_reads:
        raise ValueError("selection-relevant read audit drifted")
    _assert_same(
        audit["source_audit"],
        config["expected_source_audit"],
        where="source audit",
    )


def _validate_publication(
    ledger: dict[str, Any], config: dict[str, Any]
) -> None:
    publication = _mapping(ledger.get("publication"), where="publication")
    expected = {
        "full_stdout_path": config["output"]["full_stdout_json"],
        "reduced_findings_path": config["output"]["findings_json"],
        "report_path": config["output"]["findings_report"],
        "publish_regardless_of_outcome": True,
        "cumulative_nonzero_laws_two_rounds": 7,
    }
    _assert_same(publication, expected, where="publication record")


def _validate_truth(
    value: Any,
    *,
    where: str,
    locked: dict[str, Any],
) -> dict[str, Any]:
    truth = _mapping(value, where=where)
    _strict_keys(truth, {"pooled", "origin", "groups"}, where=where)
    pooled = _validate_rate_record(truth["pooled"], where=f"{where}.pooled")
    _assert_same(pooled, locked["truth"], where=f"{where}.pooled lock")
    origin = _mapping(truth["origin"], where=f"{where}.origin")
    if tuple(origin) != ORIGINS or set(origin) != set(ORIGINS):
        raise ValueError(f"{where}.origin order/content drifted")
    for name in ORIGINS:
        record = _validate_rate_record(
            origin[name], where=f"{where}.origin.{name}"
        )
        _assert_same(
            record,
            locked["truth_origin"][name],
            where=f"{where}.origin.{name} lock",
        )
    _validate_group_array(
        truth["groups"],
        where=f"{where}.groups",
        validator=_validate_rate_record,
    )
    return truth


def _validate_direct(value: Any, *, where: str) -> dict[str, Any]:
    direct = _mapping(value, where=where)
    _strict_keys(direct, {"pooled", "origin", "groups"}, where=where)
    _validate_direct_record(direct["pooled"], where=f"{where}.pooled")
    origin = _mapping(direct["origin"], where=f"{where}.origin")
    if tuple(origin) != ORIGINS or set(origin) != set(ORIGINS):
        raise ValueError(f"{where}.origin order/content drifted")
    for name in ORIGINS:
        _validate_direct_record(origin[name], where=f"{where}.origin.{name}")
    _validate_group_array(
        direct["groups"],
        where=f"{where}.groups",
        validator=_validate_direct_group_record,
    )
    return direct


def _validate_direct_truth_and_invariance(
    direct: dict[str, Any],
    *,
    truth: dict[str, Any],
    r0_direct: dict[str, Any] | None,
    where: str,
) -> None:
    invariant_fields = (
        "risk_rows",
        "event_rows",
        "matchable_positive_weight_event_rows",
        "matchable_event_rows",
        "unmatched_same_year_event_rows",
        "unmatched_same_year_event_weight",
        "exposure",
        "deviance_exposure",
        "actual_numerator",
        "matchable_numerator",
        "actual_rate",
    )
    scopes = (("pooled", None), *(("origin", origin) for origin in ORIGINS))
    for scope, origin in scopes:
        record = direct[scope] if origin is None else direct[scope][origin]
        truth_record = truth[scope] if origin is None else truth[scope][origin]
        for key in ("risk_rows", "event_rows", "exposure"):
            _assert_same(
                record[key],
                truth_record[key],
                where=f"{where}.{scope}.{origin or 'pooled'}.{key} truth",
            )
        _assert_same(
            record["actual_numerator"],
            truth_record["numerator"],
            where=f"{where}.{scope}.{origin or 'pooled'}.actual_numerator truth",
        )
        _assert_same(
            record["actual_rate"],
            truth_record["rate"],
            where=f"{where}.{scope}.{origin or 'pooled'}.actual_rate truth",
        )
        if r0_direct is not None:
            baseline = (
                r0_direct[scope]
                if origin is None
                else r0_direct[scope][origin]
            )
            _assert_same(
                {key: record[key] for key in invariant_fields},
                {key: baseline[key] for key in invariant_fields},
                where=f"{where}.{scope}.{origin or 'pooled'} invariants",
            )


def _validate_aggregate_record(
    value: Any,
    *,
    where: str,
    truth: dict[str, Any],
    contributors: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = _mapping(value, where=where)
    _strict_keys(aggregate, {"pooled", "origin", "groups"}, where=where)
    _validate_mean_record(
        aggregate["pooled"],
        where=f"{where}.pooled",
        truth=truth["pooled"],
        contributors=[row["pooled"] for row in contributors],
    )
    origins = _mapping(aggregate["origin"], where=f"{where}.origin")
    if tuple(origins) != ORIGINS or set(origins) != set(ORIGINS):
        raise ValueError(f"{where}.origin order/content drifted")
    for origin in ORIGINS:
        _validate_mean_record(
            origins[origin],
            where=f"{where}.origin.{origin}",
            truth=truth["origin"][origin],
            contributors=[row["origin"][origin] for row in contributors],
        )
    truth_groups = {record["group"]: record for record in truth["groups"]}
    _validate_group_array(
        aggregate["groups"],
        where=f"{where}.groups",
        validator=_validate_mean_record,
        truth_groups=truth_groups,
        contributor_groups=[row["publication_groups"] for row in contributors],
    )
    return aggregate


def _validate_seed_row(
    value: Any,
    *,
    where: str,
    seed: int,
    law: str,
    boundary: str,
    truth: dict[str, Any],
    boundary_record: dict[str, Any],
    config: dict[str, Any],
    uniform_reference: dict[int, str],
    gap_reference: dict[int, dict[str, int]],
) -> dict[str, Any]:
    row = _mapping(value, where=where)
    _strict_keys(
        row,
        {
            "seed",
            "pooled",
            "origin",
            "publication_groups",
            "carrier_checks",
            "support_checks",
            "uniform_checks",
            "downstream",
        },
        where=where,
    )
    if row["seed"] != seed:
        raise ValueError(f"{where}.seed drifted")
    _validate_rate_record(
        row["pooled"], where=f"{where}.pooled", truth=truth["pooled"]
    )
    origins = _mapping(row["origin"], where=f"{where}.origin")
    if tuple(origins) != ORIGINS or set(origins) != set(ORIGINS):
        raise ValueError(f"{where}.origin order/content drifted")
    for origin in ORIGINS:
        _validate_rate_record(
            origins[origin],
            where=f"{where}.origin.{origin}",
            truth=truth["origin"][origin],
        )
    truth_groups = {record["group"]: record for record in truth["groups"]}
    _validate_group_array(
        row["publication_groups"],
        where=f"{where}.publication_groups",
        validator=_validate_rate_record,
        truth_groups=truth_groups,
    )

    carriers = _mapping(row["carrier_checks"], where=f"{where}.carrier_checks")
    _strict_keys(
        carriers,
        {
            "expected_entry_dissolved_carriers",
            "verified_entry_dissolved_carriers",
            "exact",
        },
        where=f"{where}.carrier_checks",
    )
    expected_carriers = boundary_record["entry_dissolved_carriers"]
    if (
        carriers["expected_entry_dissolved_carriers"] != expected_carriers
        or carriers["verified_entry_dissolved_carriers"] != expected_carriers
    ):
        raise ValueError(f"{where} carrier count drifted")
    _true(carriers["exact"], where=f"{where}.carrier_checks.exact")

    support = _mapping(row["support_checks"], where=f"{where}.support_checks")
    _strict_keys(
        support,
        {
            "exact_truth_keys",
            "truth_key_count",
            "projected_key_count",
            "truth_key_sha256",
            "projected_key_sha256",
            "weighted_support_sha256",
            "weighted_support_exact_truth",
            "event_semantics",
        },
        where=f"{where}.support_checks",
    )
    _true(support["exact_truth_keys"], where=f"{where}.support exact keys")
    if (
        support["truth_key_count"] != boundary_record["truth_support_rows"]
        or support["projected_key_count"]
        != boundary_record["truth_support_rows"]
    ):
        raise ValueError(f"{where} support key count drifted")
    if (
        support["truth_key_sha256"]
        != boundary_record["truth_support_key_sha256"]
        or support["projected_key_sha256"]
        != boundary_record["truth_support_key_sha256"]
    ):
        raise ValueError(f"{where} support key checksum drifted")
    if (
        support["weighted_support_sha256"]
        != boundary_record["support_checksum"]
    ):
        raise ValueError(f"{where} weighted support checksum drifted")
    _true(
        support["weighted_support_exact_truth"],
        where=f"{where}.weighted_support_exact_truth",
    )
    semantics = _mapping(
        support["event_semantics"], where=f"{where}.support event semantics"
    )
    _strict_keys(
        semantics,
        {
            "remarriage_event_rows",
            "origins_valid",
            "years_since_dissolution_defined",
            "unique_person_year",
            "matchable_origin_and_ysd_exact_risk_row",
            "unmatched_same_year_event_rows",
        },
        where=f"{where}.support event semantics",
    )
    _nonnegative_int(
        semantics["remarriage_event_rows"],
        where=f"{where}.remarriage_event_rows",
    )
    _nonnegative_int(
        semantics["unmatched_same_year_event_rows"],
        where=f"{where}.unmatched_same_year_event_rows",
    )
    for key in (
        "origins_valid",
        "years_since_dissolution_defined",
        "unique_person_year",
        "matchable_origin_and_ysd_exact_risk_row",
    ):
        _true(semantics[key], where=f"{where}.event_semantics.{key}")

    uniform = _mapping(row["uniform_checks"], where=f"{where}.uniform_checks")
    _strict_keys(
        uniform,
        {
            "draw_index",
            "n_periods",
            "transition_address",
            "spouse_gap_address",
            "transition_uniform_sha256",
            "exact_R0_stream",
        },
        where=f"{where}.uniform_checks",
    )
    if uniform["draw_index"] != seed - config["rng"]["seed_root"]:
        raise ValueError(f"{where} RNG draw index drifted")
    if uniform["n_periods"] != config["protocol"]["n_periods"][boundary]:
        raise ValueError(f"{where} RNG period count drifted")
    if (
        uniform["transition_address"]
        != config["rng"]["transition_uniform_address"]
    ):
        raise ValueError(f"{where} transition RNG address drifted")
    if uniform["spouse_gap_address"] != config["rng"]["spouse_gap_address"]:
        raise ValueError(f"{where} spouse-gap RNG address drifted")
    checksum = _sha256(
        uniform["transition_uniform_sha256"],
        where=f"{where}.transition_uniform_sha256",
    )
    _true(uniform["exact_R0_stream"], where=f"{where}.exact_R0_stream")
    if law == "R0":
        uniform_reference[seed] = checksum
    elif uniform_reference.get(seed) != checksum:
        raise ValueError(f"{where} transition uniform differs from R0")

    downstream = _mapping(row["downstream"], where=f"{where}.downstream")
    _strict_keys(
        downstream,
        {
            "person_year_rows",
            "event_rows",
            "birth_rows",
            "person_years_sha256",
            "events_sha256",
            "births_sha256",
            "spouse_gap_consumption",
            "spouse_gap_consumption_difference_from_R0",
            "R0_spouse_gap_consumption_exact_incumbent",
            "R0_projection_exact_incumbent",
        },
        where=f"{where}.downstream",
    )
    for key in ("person_year_rows", "event_rows", "birth_rows"):
        _nonnegative_int(downstream[key], where=f"{where}.downstream.{key}")
    for key in ("person_years_sha256", "events_sha256", "births_sha256"):
        _sha256(downstream[key], where=f"{where}.downstream.{key}")
    gap_consumption = _mapping(
        downstream["spouse_gap_consumption"],
        where=f"{where}.downstream.spouse_gap_consumption",
    )
    gap_difference = _mapping(
        downstream["spouse_gap_consumption_difference_from_R0"],
        where=f"{where}.downstream.spouse_gap_consumption_difference_from_R0",
    )
    for name, record in (
        ("spouse_gap_consumption", gap_consumption),
        ("spouse_gap_consumption_difference_from_R0", gap_difference),
    ):
        _strict_keys(record, {"calls", "draws"}, where=f"{where}.{name}")
        for key in ("calls", "draws"):
            value = record[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (name == "spouse_gap_consumption" and value < 0)
            ):
                raise ValueError(f"{where}.{name}.{key} is invalid")
    if law == "R0" and gap_difference != {"calls": 0, "draws": 0}:
        raise ValueError(f"{where} R0 spouse-gap difference is nonzero")
    if law == "R0":
        gap_reference[seed] = {
            "calls": gap_consumption["calls"],
            "draws": gap_consumption["draws"],
        }
    expected_gap_difference = {
        key: gap_consumption[key] - gap_reference[seed][key]
        for key in ("calls", "draws")
    }
    _assert_same(
        gap_difference,
        expected_gap_difference,
        where=f"{where}.spouse_gap_consumption_difference_from_R0",
    )
    expected_incumbent = True if law == "R0" else None
    _assert_same(
        downstream["R0_projection_exact_incumbent"],
        expected_incumbent,
        where=f"{where}.R0 projection equivalence",
    )
    _assert_same(
        downstream["R0_spouse_gap_consumption_exact_incumbent"],
        expected_incumbent,
        where=f"{where}.R0 spouse-gap consumption equivalence",
    )
    return row


def _validate_cube(
    ledger: dict[str, Any],
    *,
    config: dict[str, Any],
    validation: dict[str, Any],
) -> list[tuple[dict[str, Any], list[Any]]]:
    fit_validation = _mapping(
        ledger.get("fit_validation"), where="fit_validation"
    )
    if tuple(fit_validation) != BOUNDARIES or set(fit_validation) != set(
        BOUNDARIES
    ):
        raise ValueError("fit_validation boundary order/content drifted")
    for boundary in BOUNDARIES:
        _validate_fit_public(
            fit_validation[boundary],
            where=f"fit_validation.{boundary}",
            boundary=boundary,
            laws=LAWS,
            config=config,
            validation=validation,
        )

    boundaries = _mapping(ledger.get("boundaries"), where="boundaries")
    if tuple(boundaries) != BOUNDARIES or set(boundaries) != set(BOUNDARIES):
        raise ValueError("raw boundary order/content drifted")
    removable: list[tuple[dict[str, Any], list[Any]]] = []
    for boundary in BOUNDARIES:
        where = f"boundaries.{boundary}"
        record = _mapping(boundaries[boundary], where=where)
        _strict_keys(
            record,
            {
                "boundary",
                "anchor_waves",
                "evaluation_years",
                "anchor_households_before_marital_intersection",
                "anchor_persons_before_marital_intersection",
                "projected_persons",
                "entry_dissolved_carriers",
                "truth_support_rows",
                "support_checksum",
                "truth_required_interview_year_max",
                "truth_same_year_ysd0_events",
                "truth_same_year_ysd0_event_weight",
                "truth_origin",
                "truth",
                "truth_support_key_sha256",
                "pseudo_input_hashes",
                "fit",
                "laws",
            },
            where=where,
        )
        if record["boundary"] != int(boundary):
            raise ValueError(f"{where}.boundary drifted")
        holdout = config["pseudo_holdouts"]["boundaries"][boundary]
        if record["anchor_waves"] != holdout["anchor_waves"]:
            raise ValueError(f"{where}.anchor_waves drifted")
        if record["evaluation_years"] != holdout["evaluation_years"]:
            raise ValueError(f"{where}.evaluation_years drifted")
        locked = config["expected_pseudo_holdouts"][boundary]
        for key in (
            "anchor_households_before_marital_intersection",
            "anchor_persons_before_marital_intersection",
            "projected_persons",
            "entry_dissolved_carriers",
            "truth_support_rows",
            "support_checksum",
            "truth_required_interview_year_max",
            "truth_same_year_ysd0_events",
            "truth_same_year_ysd0_event_weight",
        ):
            if record[key] != locked[key]:
                raise ValueError(
                    f"{where}.{key} differs from frozen truth lock"
                )
        _sha256(record["support_checksum"], where=f"{where}.support_checksum")
        _sha256(
            record["truth_support_key_sha256"],
            where=f"{where}.truth_support_key_sha256",
        )
        input_hashes = _mapping(
            record["pseudo_input_hashes"],
            where=f"{where}.pseudo_input_hashes",
        )
        _strict_keys(
            input_hashes,
            {
                "anchor_sha256",
                "seed_attrs_sha256",
                "seed_entry_person_years_sha256",
                "truth_support_key_sha256",
                "truth_support_weighted_sha256",
            },
            where=f"{where}.pseudo_input_hashes",
        )
        for key, digest in input_hashes.items():
            _sha256(digest, where=f"{where}.pseudo_input_hashes.{key}")
        if (
            input_hashes["truth_support_key_sha256"]
            != record["truth_support_key_sha256"]
            or input_hashes["truth_support_weighted_sha256"]
            != record["support_checksum"]
        ):
            raise ValueError(f"{where} pseudo-input truth hashes drifted")
        truth = _validate_truth(
            record["truth"], where=f"{where}.truth", locked=locked
        )
        _assert_same(
            record["truth_origin"],
            truth["origin"],
            where=f"{where}.truth_origin",
        )
        _assert_same(
            record["fit"], fit_validation[boundary], where=f"{where}.fit"
        )

        laws = _mapping(record["laws"], where=f"{where}.laws")
        if set(laws) != set(LAWS):
            raise ValueError(f"{where}.laws content drifted")
        uniform_reference: dict[int, str] = {}
        gap_reference: dict[int, dict[str, int]] = {}
        r0_direct: dict[str, Any] | None = None
        for law in LAWS:
            law_where = f"{where}.laws.{law}"
            law_record = _mapping(laws[law], where=law_where)
            _strict_keys(
                law_record,
                {
                    "construction",
                    "direct",
                    "g_widowed_log_qdir_ratio",
                    "per_seed",
                    "mean",
                    "blocks",
                    "carrier_conformance_all_draws",
                    "support_exact_all_draws",
                    "uniform_exact_all_draws",
                    "non_remarriage_components_exact_R0",
                    "R0_projection_exact_incumbent_all_draws",
                    "R0_spouse_gap_consumption_exact_incumbent_all_draws",
                },
                where=law_where,
            )
            _assert_same(
                law_record["construction"],
                fit_validation[boundary]["laws"][law],
                where=f"{law_where}.construction",
            )
            direct = _validate_direct(
                law_record["direct"], where=f"{law_where}.direct"
            )
            _validate_direct_truth_and_invariance(
                direct,
                truth=truth,
                r0_direct=r0_direct,
                where=f"{law_where}.direct",
            )
            if law == "R0":
                r0_direct = direct
            assert r0_direct is not None
            expected_g = math.log(
                direct["origin"]["widowed"]["qdir"]
                / r0_direct["origin"]["widowed"]["qdir"]
            )
            _assert_same(
                law_record["g_widowed_log_qdir_ratio"],
                expected_g,
                where=f"{law_where}.g_widowed_log_qdir_ratio",
            )

            rows = _array(
                law_record["per_seed"], where=f"{law_where}.per_seed"
            )
            if len(rows) != 40:
                raise ValueError(f"{law_where}.per_seed must contain 40 draws")
            validated_rows: list[dict[str, Any]] = []
            for index, seed in enumerate(SEEDS):
                seed_where = f"{law_where}.per_seed[{index}]"
                row = _validate_seed_row(
                    rows[index],
                    where=seed_where,
                    seed=seed,
                    law=law,
                    boundary=boundary,
                    truth=truth,
                    boundary_record=record,
                    config=config,
                    uniform_reference=uniform_reference,
                    gap_reference=gap_reference,
                )
                removable.append((row, row["publication_groups"]))
                validated_rows.append(row)

            expected_mean = _aggregate_projection(validated_rows, truth, SEEDS)
            _validate_aggregate_record(
                law_record["mean"],
                where=f"{law_where}.mean",
                truth=truth,
                contributors=validated_rows,
            )
            _assert_same(
                law_record["mean"], expected_mean, where=f"{law_where}.mean"
            )
            blocks = _mapping(
                law_record["blocks"], where=f"{law_where}.blocks"
            )
            _strict_keys(
                blocks, {"block_1", "block_2"}, where=f"{law_where}.blocks"
            )
            for block_index, seeds in enumerate(BLOCKS, start=1):
                name = f"block_{block_index}"
                _validate_aggregate_record(
                    blocks[name],
                    where=f"{law_where}.blocks.{name}",
                    truth=truth,
                    contributors=[
                        row for row in validated_rows if row["seed"] in seeds
                    ],
                )
                expected_block = _aggregate_projection(
                    validated_rows, truth, seeds
                )
                _assert_same(
                    blocks[name],
                    expected_block,
                    where=f"{law_where}.blocks.{name}",
                )
            for key in (
                "carrier_conformance_all_draws",
                "support_exact_all_draws",
                "uniform_exact_all_draws",
                "non_remarriage_components_exact_R0",
            ):
                _true(law_record[key], where=f"{law_where}.{key}")
            expected_incumbent = True if law == "R0" else None
            _assert_same(
                law_record["R0_projection_exact_incumbent_all_draws"],
                expected_incumbent,
                where=f"{law_where}.R0 projection aggregate",
            )
            _assert_same(
                law_record[
                    "R0_spouse_gap_consumption_exact_incumbent_all_draws"
                ],
                expected_incumbent,
                where=f"{law_where}.R0 spouse-gap aggregate",
            )
    if len(removable) != 3 * 5 * 40:
        raise ValueError("validated per-seed cube did not contain 600 rows")
    return removable


def _objective_for_seeds(
    boundaries: dict[str, Any], law: str, seeds: tuple[int, ...]
) -> float:
    wanted = set(seeds)
    terms: list[float] = []
    for boundary in BOUNDARIES:
        rows = [
            row
            for row in boundaries[boundary]["laws"][law]["per_seed"]
            if row["seed"] in wanted
        ]
        if [row["seed"] for row in rows] != list(seeds):
            raise ValueError("objective seed order drifted")
        mean_rate = float(
            np.mean(
                np.asarray(
                    [row["pooled"]["rate"] for row in rows],
                    dtype=np.float64,
                )
            )
        )
        truth_rate = float(boundaries[boundary]["truth"]["pooled"]["rate"])
        if mean_rate <= 0.0 or truth_rate <= 0.0:
            raise ValueError("objective log argument is non-positive")
        terms.append(math.log(mean_rate / truth_rate) ** 2)
    return float(np.mean(np.asarray(terms, dtype=np.float64)))


def _pooled_direct(
    boundaries: dict[str, Any], law: str, origin: str | None = None
) -> dict[str, float]:
    records = []
    for boundary in BOUNDARIES:
        direct = boundaries[boundary]["laws"][law]["direct"]
        records.append(
            direct["pooled"] if origin is None else direct["origin"][origin]
        )
    numerator = float(
        sum(record["weighted_deviance_numerator"] for record in records)
    )
    exposure = float(sum(record["deviance_exposure"] for record in records))
    if exposure <= 0.0:
        raise ValueError("pooled direct exposure is non-positive")
    return {
        "weighted_deviance_numerator": numerator,
        "deviance_exposure": exposure,
        "weighted_bernoulli_deviance": numerator / exposure,
    }


def _selector_metrics(boundaries: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for law in LAWS:
        full_j = _objective_for_seeds(boundaries, law, SEEDS)
        block_j = {
            f"block_{index}": _objective_for_seeds(boundaries, law, block)
            for index, block in enumerate(BLOCKS, start=1)
        }
        delete_one = []
        for deleted in SEEDS:
            retained = tuple(seed for seed in SEEDS if seed != deleted)
            delete_one.append(
                {
                    "deleted_seed": deleted,
                    "retained_seed_count": len(retained),
                    "J_delete_one": _objective_for_seeds(
                        boundaries, law, retained
                    ),
                }
            )
        jackknife_mean = float(
            np.mean(
                np.asarray(
                    [record["J_delete_one"] for record in delete_one],
                    dtype=np.float64,
                )
            )
        )
        standard_error = math.sqrt(
            (39.0 / 40.0)
            * sum(
                (record["J_delete_one"] - jackknife_mean) ** 2
                for record in delete_one
            )
        )
        boundary_metrics = {}
        for boundary in BOUNDARIES:
            record = boundaries[boundary]
            law_record = record["laws"][law]
            mean = law_record["mean"]
            truth = record["truth"]
            boundary_metrics[boundary] = {
                "rate_error": abs(
                    math.log(mean["pooled"]["rate"] / truth["pooled"]["rate"])
                ),
                "exposure_error": abs(
                    math.log(
                        mean["pooled"]["exposure"]
                        / truth["pooled"]["exposure"]
                    )
                ),
                "working_age_widow_exposure_error": abs(
                    math.log(
                        mean["origin"]["widowed"]["exposure"]
                        / truth["origin"]["widowed"]["exposure"]
                    )
                ),
                "pooled_direct_deviance": law_record["direct"]["pooled"][
                    "weighted_bernoulli_deviance"
                ],
                "divorced_direct_deviance": law_record["direct"]["origin"][
                    "divorced"
                ]["weighted_bernoulli_deviance"],
                "widowed_direct_deviance": law_record["direct"]["origin"][
                    "widowed"
                ]["weighted_bernoulli_deviance"],
                "widowed_matchable_positive_weight_event_rows": law_record[
                    "direct"
                ]["origin"]["widowed"]["matchable_positive_weight_event_rows"],
                "widowed_direct_risk_exposure": law_record["direct"]["origin"][
                    "widowed"
                ]["deviance_exposure"],
                "g_widowed_log_qdir_ratio": law_record[
                    "g_widowed_log_qdir_ratio"
                ],
            }
        metrics[law] = {
            "J": full_j,
            "block_J": block_j,
            "jackknife": {
                "replicates": delete_one,
                "Jjack_bar": jackknife_mean,
                "SE_J": standard_error,
                "same_seed_deleted_across_all_boundaries": True,
                "reselection_inside_replicates": False,
            },
            "pooled_direct": _pooled_direct(boundaries, law),
            "origin_pooled_direct": {
                origin: _pooled_direct(boundaries, law, origin)
                for origin in ORIGINS
            },
            "boundary": boundary_metrics,
        }
    return metrics


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _rule_one(
    boundaries: dict[str, Any], law: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    required_values: list[Any] = [
        metrics["J"],
        metrics["block_J"]["block_1"],
        metrics["block_J"]["block_2"],
        metrics["jackknife"]["SE_J"],
        metrics["pooled_direct"]["weighted_bernoulli_deviance"],
    ]
    conformance: list[bool] = []
    positive_logs: list[bool] = []
    for boundary in BOUNDARIES:
        law_record = boundaries[boundary]["laws"][law]
        construction = law_record["construction"]
        required_values.extend(
            [
                law_record["mean"]["pooled"]["rate"],
                law_record["mean"]["pooled"]["exposure"],
                law_record["mean"]["origin"]["widowed"]["exposure"],
                law_record["direct"]["pooled"]["weighted_bernoulli_deviance"],
                law_record["direct"]["origin"]["divorced"][
                    "weighted_bernoulli_deviance"
                ],
                law_record["direct"]["origin"]["widowed"][
                    "weighted_bernoulli_deviance"
                ],
                law_record["g_widowed_log_qdir_ratio"],
                construction["probability_min"],
                construction["probability_max"],
            ]
        )
        truth = boundaries[boundary]["truth"]
        positive_logs.extend(
            [
                law_record["mean"]["pooled"]["rate"] > 0.0,
                law_record["mean"]["pooled"]["exposure"] > 0.0,
                truth["pooled"]["rate"] > 0.0,
                truth["pooled"]["exposure"] > 0.0,
                law_record["mean"]["origin"]["widowed"]["exposure"] > 0.0,
                truth["origin"]["widowed"]["exposure"] > 0.0,
            ]
        )
        conformance.extend(
            [
                law_record["carrier_conformance_all_draws"],
                law_record["support_exact_all_draws"],
                law_record["uniform_exact_all_draws"],
                construction["raw_age_outside_18_64_exact_R0"],
                boundaries[boundary]["fit"]["validation_match"],
            ]
        )
        if law == "R0":
            conformance.append(
                law_record["R0_projection_exact_incumbent_all_draws"]
            )
        for row in law_record["per_seed"]:
            for scope in [row["pooled"], *row["origin"].values()]:
                required_values.extend(
                    [scope["exposure"], scope["numerator"], scope["rate"]]
                )
                positive_logs.append(scope["exposure"] > 0.0)
            conformance.extend(
                [
                    row["carrier_checks"]["exact"],
                    row["support_checks"]["exact_truth_keys"],
                    row["uniform_checks"]["exact_R0_stream"],
                    row["support_checks"]["event_semantics"]["origins_valid"],
                    row["support_checks"]["event_semantics"][
                        "years_since_dissolution_defined"
                    ],
                    row["support_checks"]["event_semantics"][
                        "unique_person_year"
                    ],
                ]
            )
    checks = {
        "all_required_selector_values_finite": all(
            _finite(value) for value in required_values
        ),
        "all_required_log_arguments_positive": all(positive_logs),
        "support_carrier_event_rng_conformance": all(conformance),
        "fit_side_validation_match": all(
            boundaries[boundary]["fit"]["validation_match"]
            for boundary in BOUNDARIES
        ),
    }
    return {"pass": all(checks.values()), "checks": checks}


def _strictly_better(value: float, baseline: float) -> bool:
    return value < baseline - TOLERANCE


def _no_worse(value: float, baseline: float) -> bool:
    return value <= baseline + TOLERANCE


def _comparison_rule(
    candidate: list[float], baseline: list[float], *, strict_minimum: int
) -> dict[str, Any]:
    strict = [
        _strictly_better(value, reference)
        for value, reference in zip(candidate, baseline, strict=True)
    ]
    no_worse = [
        _no_worse(value, reference)
        for value, reference in zip(candidate, baseline, strict=True)
    ]
    return {
        "pass": sum(strict) >= strict_minimum and all(no_worse),
        "strict_improvements": strict,
        "no_worse": no_worse,
        "strict_improvement_count": sum(strict),
        "required_strict_improvements": strict_minimum,
    }


def _law_components(config: dict[str, Any], law: str) -> tuple[float, float]:
    record = config["family"]["law_components"][law]
    return float(record["divorced_k"]), float(record["widowed_omega"])


def _construction_rule(
    boundaries: dict[str, Any], config: dict[str, Any], law: str
) -> dict[str, Any]:
    k, omega = _law_components(config, law)
    by_boundary = {}
    for boundary in BOUNDARIES:
        record = boundaries[boundary]["laws"][law]
        construction = record["construction"]
        g_value = record["g_widowed_log_qdir_ratio"]
        checks = {
            "divorced_area_within_tolerance": abs(
                construction["divorced_area_relative_residual"]
            )
            <= AREA_TOLERANCE,
            "widowed_area_within_tolerance": abs(
                construction["widowed_area_relative_residual"]
            )
            <= AREA_TOLERANCE,
            "divorced_delta_exact_target": abs(
                construction["Delta_divorced"] + k
            )
            <= TOLERANCE,
            "widowed_delta_exact_target": abs(
                construction["Delta_widowed"] - omega
            )
            <= TOLERANCE,
            "positive_divorced_beta": construction["beta_frontload"] > 0.0,
            "positive_exposure_divorced_cell_rises": construction[
                "positive_exposure_divorced_cells_rising"
            ]
            > 0,
            "positive_exposure_divorced_cell_falls": construction[
                "positive_exposure_divorced_cells_falling"
            ]
            > 0,
            "widowed_cells_exact_uniform_shift": construction[
                "widowed_table_construction_exact"
            ]
            and all(
                value == omega
                for value in construction[
                    "widowed_working_age_cell_logit_shifts"
                ]
            ),
            "g_nonnegative": g_value >= -TOLERANCE,
            "g_no_greater_than_omega": g_value <= omega + TOLERANCE,
            "omega_nonnegative": omega >= -TOLERANCE,
            "omega_within_budget": omega <= WIDOWED_BUDGET + TOLERANCE,
        }
        by_boundary[boundary] = {
            "pass": all(checks.values()),
            "checks": checks,
            "g": g_value,
            "omega": omega,
            "B_W": WIDOWED_BUDGET,
        }
    return {
        "pass": all(record["pass"] for record in by_boundary.values()),
        "boundaries": by_boundary,
    }


def _independent_selection(
    boundaries: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    metrics = _selector_metrics(boundaries)
    eligibility: dict[str, Any] = {}
    r0_rule_one = _rule_one(boundaries, "R0", metrics["R0"])
    r0_construction = all(
        boundaries[boundary]["laws"]["R0"]["construction"]["table_sha256"]
        == boundaries[boundary]["fit"]["incumbent_table_sha256"]
        and boundaries[boundary]["laws"]["R0"][
            "R0_projection_exact_incumbent_all_draws"
        ]
        for boundary in BOUNDARIES
    )
    if not r0_rule_one["pass"] or not r0_construction:
        raise ValueError("R0 failed independently reconstructed conformance")
    eligibility["R0"] = {
        "eligible_as_baseline": True,
        "rule_1": r0_rule_one,
        "R0_bit_equivalence": r0_construction,
    }

    eligible_nonzero: list[str] = []
    for law in LAWS[1:]:
        rule_1 = _rule_one(boundaries, law, metrics[law])
        rule_2 = _construction_rule(boundaries, config, law)
        rule_3_checks = {
            "full_J_strictly_better_R0": _strictly_better(
                metrics[law]["J"], metrics["R0"]["J"]
            ),
            "block_1_J_strictly_better_R0": _strictly_better(
                metrics[law]["block_J"]["block_1"],
                metrics["R0"]["block_J"]["block_1"],
            ),
            "block_2_J_strictly_better_R0": _strictly_better(
                metrics[law]["block_J"]["block_2"],
                metrics["R0"]["block_J"]["block_2"],
            ),
        }
        rule_3 = {"pass": all(rule_3_checks.values()), "checks": rule_3_checks}
        rate_candidate = [
            metrics[law]["boundary"][boundary]["rate_error"]
            for boundary in BOUNDARIES
        ]
        rate_baseline = [
            metrics["R0"]["boundary"][boundary]["rate_error"]
            for boundary in BOUNDARIES
        ]
        rule_4 = _comparison_rule(
            rate_candidate, rate_baseline, strict_minimum=2
        )
        exposure_candidate = [
            metrics[law]["boundary"][boundary]["exposure_error"]
            for boundary in BOUNDARIES
        ]
        exposure_baseline = [
            metrics["R0"]["boundary"][boundary]["exposure_error"]
            for boundary in BOUNDARIES
        ]
        exposure_comparison = _comparison_rule(
            exposure_candidate, exposure_baseline, strict_minimum=0
        )
        rule_5 = {
            **exposure_comparison,
            "pass": all(exposure_comparison["no_worse"]),
        }
        direct_candidate = [
            metrics[law]["boundary"][boundary]["pooled_direct_deviance"]
            for boundary in BOUNDARIES
        ]
        direct_baseline = [
            metrics["R0"]["boundary"][boundary]["pooled_direct_deviance"]
            for boundary in BOUNDARIES
        ]
        direct_boundaries = _comparison_rule(
            direct_candidate, direct_baseline, strict_minimum=2
        )
        pooled_direct_strict = _strictly_better(
            metrics[law]["pooled_direct"]["weighted_bernoulli_deviance"],
            metrics["R0"]["pooled_direct"]["weighted_bernoulli_deviance"],
        )
        rule_6 = {
            "pass": pooled_direct_strict and direct_boundaries["pass"],
            "pooled_direct_strictly_better_R0": pooled_direct_strict,
            "boundary_comparison": direct_boundaries,
        }
        divorced_candidate = [
            metrics[law]["boundary"][boundary]["divorced_direct_deviance"]
            for boundary in BOUNDARIES
        ]
        divorced_baseline = [
            metrics["R0"]["boundary"][boundary]["divorced_direct_deviance"]
            for boundary in BOUNDARIES
        ]
        divorced_check = _comparison_rule(
            divorced_candidate, divorced_baseline, strict_minimum=2
        )
        _k, omega = _law_components(config, law)
        widowed_branches = {}
        for boundary in BOUNDARIES:
            candidate = metrics[law]["boundary"][boundary]
            baseline = metrics["R0"]["boundary"][boundary]
            event_rows = candidate[
                "widowed_matchable_positive_weight_event_rows"
            ]
            branch_law_independent = (
                event_rows
                == baseline["widowed_matchable_positive_weight_event_rows"]
            )
            g_value = candidate["g_widowed_log_qdir_ratio"]
            if event_rows > 0:
                branch_pass = bool(
                    branch_law_independent
                    and _no_worse(
                        candidate["widowed_direct_deviance"],
                        baseline["widowed_direct_deviance"],
                    )
                )
                branch = "positive_event_deviance_compared"
                details = {"widowed_deviance_no_worse_R0": branch_pass}
            else:
                branch_pass = bool(
                    branch_law_independent
                    and candidate["widowed_direct_risk_exposure"] > 0.0
                    and g_value >= -TOLERANCE
                    and g_value <= omega + TOLERANCE
                    and omega >= -TOLERANCE
                    and omega <= WIDOWED_BUDGET + TOLERANCE
                )
                branch = "zero_event_publish_not_compare_g_guard"
                details = {
                    "positive_risk_exposure": candidate[
                        "widowed_direct_risk_exposure"
                    ]
                    > 0.0,
                    "g_guard": branch_pass,
                    "widowed_deviance_compared": False,
                }
            widowed_branches[boundary] = {
                "pass": branch_pass,
                "branch": branch,
                "matchable_positive_weight_event_rows": event_rows,
                "truth_branch_law_independent": branch_law_independent,
                **details,
            }
        widow_exposure_candidate = [
            metrics[law]["boundary"][boundary][
                "working_age_widow_exposure_error"
            ]
            for boundary in BOUNDARIES
        ]
        widow_exposure_baseline = [
            metrics["R0"]["boundary"][boundary][
                "working_age_widow_exposure_error"
            ]
            for boundary in BOUNDARIES
        ]
        widow_exposure = _comparison_rule(
            widow_exposure_candidate,
            widow_exposure_baseline,
            strict_minimum=0,
        )
        rule_7 = {
            "pass": (
                divorced_check["pass"]
                and all(record["pass"] for record in widowed_branches.values())
                and all(widow_exposure["no_worse"])
            ),
            "divorced_direct": divorced_check,
            "widowed_truth_defined_branches": widowed_branches,
            "working_age_widow_exposure": {
                **widow_exposure,
                "pass": all(widow_exposure["no_worse"]),
            },
        }
        rules = {
            "rule_1_defined_and_conformant": rule_1,
            "rule_2_per_origin_construction_and_budget": rule_2,
            "rule_3_full_and_block_loss": rule_3,
            "rule_4_boundary_rate_transport": rule_4,
            "rule_5_endogenous_exposure_protection": rule_5,
            "rule_6_direct_fit": rule_6,
            "rule_7_origin_protection": rule_7,
        }
        eligible = all(record["pass"] for record in rules.values())
        eligibility[law] = {"eligible": eligible, "rules": rules}
        if eligible:
            eligible_nonzero.append(law)

    best: str | None = None
    cutoff: float | None = None
    if not eligible_nonzero:
        selected = "R0"
        reason = "no_eligible_nonzero_law"
    else:
        best = min(
            eligible_nonzero,
            key=lambda law: (metrics[law]["J"], LAWS.index(law)),
        )
        cutoff = metrics[best]["J"] + metrics[best]["jackknife"]["SE_J"]
        if metrics["R0"]["J"] <= cutoff + TOLERANCE:
            selected = "R0"
            reason = "R0_within_one_SE_of_best_eligible_law"
        else:
            selected = next(
                law
                for law in LAWS[1:]
                if law in eligible_nonzero
                and metrics[law]["J"] <= cutoff + TOLERANCE
            )
            reason = "first_eligible_law_within_one_SE_in_simplicity_order"
    return {
        "selected_law": selected,
        "selected_joint_law": selected,
        "per_origin_outcome": config["family"]["law_components"][selected][
            "per_origin_outcome"
        ],
        "disposition": (
            "NO_OP_DESIGNED_PAUSE"
            if selected == "R0"
            else "SELECTED_LAW_FOR_AMENDMENT_PROPOSAL"
        ),
        "selection_reason": reason,
        "eligible_nonzero_laws": eligible_nonzero,
        "Lbest": best,
        "one_SE_cutoff": cutoff,
        "comparison_tolerance": TOLERANCE,
        "simplicity_order": list(LAWS),
        "metrics": metrics,
        "eligibility": eligibility,
    }


def _validate_final_fit(
    ledger: dict[str, Any],
    *,
    selection: dict[str, Any],
    config: dict[str, Any],
    validation: dict[str, Any],
) -> bool:
    final = _mapping(
        ledger.get("final_information_fit"), where="final_information_fit"
    )
    selected = selection["selected_law"]
    if selected == "R0":
        expected = {
            "boundary": 2014,
            "selected_law": "R0",
            "status": "NOT_RUN_R0_SELECTED",
            "construction_pass": True,
            "designed_pause_continues": True,
        }
        _assert_same(final, expected, where="final R0 information fit")
        if ledger.get("status") != "SELECTION_COMPLETE":
            raise ValueError("raw status is inconsistent with R0 selection")
        return True

    _strict_keys(
        final,
        {
            "boundary",
            "selected_law",
            "status",
            "construction_pass",
            "checks",
            "fit",
            "selected_law_table",
            "designed_pause_continues",
        },
        where="final_information_fit",
    )
    if final["boundary"] != 2014 or final["selected_law"] != selected:
        raise ValueError("final fit boundary/selected law drifted")
    fit = _validate_fit_public(
        final["fit"],
        where="final_information_fit.fit",
        boundary="2014",
        laws=(selected,),
        config=config,
        validation=validation,
    )
    construction = fit["laws"][selected]
    _assert_same(
        final["selected_law_table"],
        construction,
        where="final selected-law table",
    )
    k, omega = _law_components(config, selected)
    checks = {
        "fit_max_year_2013": fit["fit_max_year"] == 2013,
        "validation_match": fit["validation_match"],
        "divorced_area_within_tolerance": abs(
            construction["divorced_area_relative_residual"]
        )
        <= AREA_TOLERANCE,
        "widowed_area_within_tolerance": abs(
            construction["widowed_area_relative_residual"]
        )
        <= AREA_TOLERANCE,
        "divorced_delta_exact_target": abs(construction["Delta_divorced"] + k)
        <= TOLERANCE,
        "widowed_delta_exact_target": abs(
            construction["Delta_widowed"] - omega
        )
        <= TOLERANCE,
        "positive_beta": construction["beta_frontload"] > 0.0,
        "divorced_rise_and_fall": (
            construction["positive_exposure_divorced_cells_rising"] > 0
            and construction["positive_exposure_divorced_cells_falling"] > 0
        ),
        "widowed_table_construction_exact": construction[
            "widowed_table_construction_exact"
        ],
        "omega_within_budget": (
            omega >= -TOLERANCE and omega <= WIDOWED_BUDGET + TOLERANCE
        ),
    }
    _assert_same(final["checks"], checks, where="final construction checks")
    passed = all(checks.values())
    if final["construction_pass"] is not passed:
        raise ValueError("final construction_pass is inconsistent")
    expected_status = "PASS" if passed else "FINAL_FIT_FAILURE_DESIGNED_PAUSE"
    if final["status"] != expected_status:
        raise ValueError("final fit status is inconsistent")
    if final["designed_pause_continues"] is not (not passed):
        raise ValueError("final designed-pause flag is inconsistent")
    expected_raw_status = (
        "SELECTION_COMPLETE"
        if passed
        else "SELECTION_COMPLETE_FINAL_FIT_FAILURE"
    )
    if ledger.get("status") != expected_raw_status:
        raise ValueError("raw status is inconsistent with final fit")
    return passed


def _validate_raw_top(ledger: dict[str, Any]) -> None:
    reserved = {"full_stdout_sha256", "reducer"}.intersection(ledger)
    if reserved:
        raise ValueError(
            f"raw stdout contains reserved reducer keys {sorted(reserved)}"
        )
    _strict_keys(
        ledger,
        {
            "schema",
            "status",
            "candidate_outcome_contact",
            "authority",
            "freeze",
            "runtime",
            "protocol",
            "input_and_file_open_audit",
            "fit_validation",
            "boundaries",
            "selection",
            "final_information_fit",
            "publication",
        },
        where="selector stdout",
    )
    if ledger["schema"] != RAW_SCHEMA:
        raise ValueError(f"unexpected raw schema {ledger['schema']!r}")
    _true(
        ledger["candidate_outcome_contact"], where="candidate_outcome_contact"
    )


def reduce(raw: bytes) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]
    config_path = script_path.with_name(CONFIG_FILENAME)
    config_raw = config_path.read_bytes()
    config = _load_strict_json(config_raw, source=str(config_path))
    _validate_config(config)

    validation_path = (
        root / "docs/design/m6_remarriage_learning_plan_round2_validation.json"
    )
    validation = _load_strict_json(
        validation_path.read_bytes(), source=str(validation_path)
    )
    if (
        validation.get("schema")
        != "m6.remarriage.learning_plan.round2.fit_side_validation.v1"
    ):
        raise ValueError("fit-side validation schema drifted")
    round1_path = (
        root / "docs/analysis/m6_remarriage_train_only_delta_results.json"
    )
    round1 = _load_strict_json(
        round1_path.read_bytes(), source=str(round1_path)
    )
    _validate_config_against_round1(config, round1)

    ledger = _load_strict_json(raw, source="selector stdout")
    _validate_raw_top(ledger)
    _validate_freeze(ledger, config, root)
    _validate_runtime(ledger, config)
    _validate_input_audit(ledger, config)
    _validate_publication(ledger, config)
    removable = _validate_cube(ledger, config=config, validation=validation)

    actual_selection = _mapping(ledger.get("selection"), where="selection")
    _strict_keys(
        actual_selection,
        {
            "selected_law",
            "selected_joint_law",
            "per_origin_outcome",
            "disposition",
            "selection_reason",
            "eligible_nonzero_laws",
            "Lbest",
            "one_SE_cutoff",
            "comparison_tolerance",
            "simplicity_order",
            "metrics",
            "eligibility",
        },
        where="selection",
    )
    independent_selection = _independent_selection(
        ledger["boundaries"], config
    )
    final_pass = _validate_final_fit(
        ledger,
        selection=independent_selection,
        config=config,
        validation=validation,
    )
    if independent_selection["selected_law"] != "R0" and not final_pass:
        independent_selection["disposition"] = (
            "FINAL_FIT_FAILURE_DESIGNED_PAUSE"
        )
    if _canonical_bytes(actual_selection) != _canonical_bytes(
        independent_selection
    ):
        print(
            _canonical_bytes(
                _selection_diff(actual_selection, independent_selection)
            ).decode("utf-8"),
            file=sys.stderr,
        )
    _assert_same(
        actual_selection,
        independent_selection,
        where="selection and one-SE outcome",
    )

    removed_groups = [groups for _row, groups in removable]
    expected_arrays = len(BOUNDARIES) * len(LAWS) * len(SEEDS)
    if len(removed_groups) != expected_arrays:
        raise ValueError("publication-group removal count drifted")
    if any(len(groups) != 18 for groups in removed_groups):
        raise ValueError("publication-group array width drifted")
    for row, _groups in removable:
        del row["publication_groups"]

    ledger["schema"] = FINDINGS_SCHEMA
    ledger["full_stdout_sha256"] = hashlib.sha256(raw).hexdigest()
    ledger["reducer"] = {
        "script": f"scripts/{script_path.name}",
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "config_file": f"scripts/{CONFIG_FILENAME}",
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "validation": {
            "strict_raw_schema": True,
            "freeze_bytes_and_commit_bound": True,
            "exact_boundary_law_seed_cube": True,
            "publication_group_width": 18,
            "projected_means_recomputed": True,
            "block_objectives_recomputed": True,
            "delete_one_jackknife_recomputed": True,
            "seven_rules_recomputed": True,
            "one_SE_selection_recomputed": True,
            "final_fit_and_disposition_validated": True,
        },
        "removed": (
            "only boundaries[b].laws[law].per_seed[*].publication_groups; "
            "600 arrays in boundary/law/seed order"
        ),
        "removed_publication_groups_arrays": expected_arrays,
        "removed_publication_group_records": sum(
            len(groups) for groups in removed_groups
        ),
        "removed_publication_groups_canonical_sha256": hashlib.sha256(
            _canonical_bytes(removed_groups)
        ).hexdigest(),
        "retained": (
            "every non-publication-group raw field, including each seed, "
            "pooled and per-origin exposure/numerator/rate record, carrier/"
            "support/uniform/downstream check, aggregate, delete-one and "
            "fixed-block objective, eligibility guard, and selection field"
        ),
    }
    return ledger


def main() -> int:
    raw = sys.stdin.buffer.read()
    ledger = reduce(raw)
    print(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
