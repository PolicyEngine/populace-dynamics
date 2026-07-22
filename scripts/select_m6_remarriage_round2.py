#!/usr/bin/env python3.13
"""Run the ratified round-2 train-only remarriage selector.

This is an analysis helper, not a gate runner.  It implements the frozen
protocol in ``docs/design/m6_remarriage_learning_plan_round2.md``.  It does
not import the M6 scorer, read ``gates.yaml`` or ``runs/``, or write files.
Its only machine output is one strict-JSON document on stdout; progress is
flushed to stderr.

The selection runtime is frozen to CPython 3.13.12 and NumPy 2.4.2.  The
remaining fitting stack must match the candidate-16 runner environment.  From
the repository root, the full run is invoked only from a clean freeze commit::

    PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS='ignore::FutureWarning' \
      PYTHONPATH=src \
      POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data \
      POPULACE_DYNAMICS_PE_US_DIR=/path/to/site-packages \
      /path/to/pinned-venv/bin/python \
      scripts/select_m6_remarriage_round2.py \
      > /safe/path/m6-remarriage-r2-full.json \
      2> /safe/path/m6-remarriage-r2-progress.log

There are no selector-changing command-line options.  ``--smoke`` is a
synthetic-only arithmetic/import check and never opens staged data.
``--preflight`` loads only information-admissible fit/reference inputs and
validates the frozen tables, paths, roots, and runtime before any pseudo-
holdout truth or candidate outcome is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit as scipy_expit
from scipy.special import logit as scipy_logit

import analyze_m6_remarriage_train_delta as round1
from populace_dynamics.data import marriage, transitions
from populace_dynamics.engine import marital as marital_engine
from populace_dynamics.engine import refit
from populace_dynamics.engine.rng import (
    ProjectionModule,
    ProjectionRNGRegistry,
)
from populace_dynamics.models.family_transitions import registry as ft_registry
from populace_dynamics.models.family_transitions.common import band_indices
from populace_dynamics.models.family_transitions.components import (
    remarriage as rem,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts/m6_remarriage_round2_selector_config.json"
VALIDATION_PATH = (
    ROOT / "docs/design/m6_remarriage_learning_plan_round2_validation.json"
)
RAW_SCHEMA = "m6.remarriage.learning_plan.round2.selection.full.v1"
SMOKE_SCHEMA = "m6.remarriage.learning_plan.round2.synthetic_smoke.v1"
PREFLIGHT_SCHEMA = "m6.remarriage.learning_plan.round2.preflight.v1"
ORIGINS = ("divorced", "widowed")
SEXES = ("female", "male")
WORKING_AGE_BANDS = (0, 1, 2)
OBSERVED_OPEN_PATHS: set[str] = set()
FILE_OPEN_AUDIT_INSTALLED = False


class ProtocolAbort(RuntimeError):
    """A protocol/conformance failure that must not feed selection."""


class RootFailure(ProtocolAbort):
    """A frozen root failure carrying its prescribed status code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _install_file_open_audit() -> None:
    global FILE_OPEN_AUDIT_INSTALLED
    if FILE_OPEN_AUDIT_INSTALLED:
        return

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        path = arguments[0]
        if not isinstance(path, str | bytes | Path):
            return
        try:
            text_path = (
                path.decode(errors="replace")
                if isinstance(path, bytes)
                else str(path)
            )
            resolved = Path(text_path)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            OBSERVED_OPEN_PATHS.add(str(resolved.resolve(strict=False)))
        except (OSError, RuntimeError, ValueError):
            OBSERVED_OPEN_PATHS.add(str(path))

    sys.addaudithook(audit)
    FILE_OPEN_AUDIT_INSTALLED = True


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProtocolAbort(f"{path} must contain a JSON object")
    return value


def _load_config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    if config.get("schema") != "m6.remarriage.round2.selector_config.v1":
        raise ProtocolAbort("unexpected round-2 selector config schema")
    return config


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_freeze(config: dict[str, Any]) -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProtocolAbort(
            "round-2 selector requires a committed clean freeze; "
            f"worktree status is:\n{status}"
        )
    head = _git("rev-parse", "HEAD")
    frozen_paths = tuple(config["freeze"]["paths"])
    blobs: dict[str, str] = {}
    sha256: dict[str, str] = {}
    for relative in frozen_paths:
        path = ROOT / relative
        blob = _git("rev-parse", f"HEAD:{relative}")
        if _git("hash-object", relative) != blob:
            raise ProtocolAbort(f"working bytes differ from HEAD: {relative}")
        blobs[relative] = blob
        sha256[relative] = _sha256_file(path)

    expected_authority = config["authority"]
    authority_paths = {
        "design_sha256": ROOT
        / "docs/design/m6_remarriage_learning_plan_round2.md",
        "validation_sha256": VALIDATION_PATH,
        "round1_ledger_sha256": ROOT
        / "docs/analysis/m6_remarriage_train_only_delta_results.json",
    }
    observed_authority = {
        key: _sha256_file(path) for key, path in authority_paths.items()
    }
    for key, observed in observed_authority.items():
        if observed != expected_authority[key]:
            raise ProtocolAbort(
                f"frozen authority mismatch for {key}: {observed}"
            )
    authority_objects = {
        expected_authority["design_file"]["path"]: expected_authority[
            "design_file"
        ]["git_object"],
        expected_authority["validation_file"]["path"]: expected_authority[
            "validation_file"
        ]["git_object"],
        expected_authority["round1_ledger"]["path"]: expected_authority[
            "round1_ledger"
        ]["git_object"],
    }
    for relative, expected_object in authority_objects.items():
        if _git("rev-parse", f"HEAD:{relative}") != expected_object:
            raise ProtocolAbort(
                f"authority git object mismatch for {relative}"
            )
    design_commit = str(expected_authority["design_commit"])
    if _git("rev-parse", design_commit) != design_commit:
        raise ProtocolAbort("ratified design commit is not locally resolvable")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", design_commit, head],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        != 0
    ):
        raise ProtocolAbort(
            "freeze commit does not descend from design commit"
        )

    source_root = (ROOT / "src/populace_dynamics").resolve()
    import_paths = {
        "round1_chassis": Path(inspect.getfile(round1)).resolve(),
        "marital_engine": Path(inspect.getfile(marital_engine)).resolve(),
        "refit": Path(inspect.getfile(refit)).resolve(),
        "remarriage_component": Path(inspect.getfile(rem)).resolve(),
    }
    scripts_root = (ROOT / "scripts").resolve()
    for name, path in import_paths.items():
        allowed = scripts_root if name == "round1_chassis" else source_root
        try:
            path.relative_to(allowed)
        except ValueError as error:
            raise ProtocolAbort(
                f"{name} imported from {path}, outside {allowed}"
            ) from error

    return {
        "freeze_commit": head,
        "branch": _git("branch", "--show-current"),
        "worktree_clean": True,
        "frozen_blob_sha1": blobs,
        "frozen_file_sha256": sha256,
        "authority_sha256": observed_authority,
        "source_tree_sha1": _git("rev-parse", "HEAD:src/populace_dynamics"),
        "import_paths": {
            name: str(path) for name, path in import_paths.items()
        },
        "all_imports_from_frozen_tree": True,
    }


def _runtime_audit(config: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    expected = config["runtime_numeric_identity"]
    if observed != expected:
        raise RootFailure(
            "ROOT_VALIDATION_MISMATCH",
            f"numeric runtime {observed} differs from {expected}",
        )
    if not sys.dont_write_bytecode:
        raise ProtocolAbort(
            "PYTHONDONTWRITEBYTECODE=1 is required so the helper writes "
            "no import artifacts"
        )
    dependencies: dict[str, str | None] = {}
    for distribution in (
        "pandas",
        "scipy",
        "scikit-learn",
        "quantile-forest",
        "policyengine-us",
        "populace-fit",
        "populace-frame",
    ):
        try:
            dependencies[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name.startswith("populace_dynamics.harness.m6_scoring")
        or name.startswith("populace_dynamics.harness.m6_runner")
    )
    if forbidden_modules:
        raise ProtocolAbort(
            f"forbidden M6 scoring modules imported: {forbidden_modules}"
        )
    return {
        **observed,
        "python_executable": sys.executable,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "dependencies": dependencies,
        "forbidden_m6_modules_imported": forbidden_modules,
    }


def _assert_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if tuple(protocol["boundaries"]) != (2006, 2008, 2010):
        raise ProtocolAbort("boundary list drifted")
    expected_years = {
        "2006": [2007, 2008, 2009, 2010],
        "2008": [2009, 2010, 2011, 2012],
        "2010": [2011, 2012, 2013],
    }
    if protocol["evaluation_years"] != expected_years:
        raise ProtocolAbort("evaluation-year lists drifted")
    if protocol["n_periods"] != {"2006": 4, "2008": 4, "2010": 3}:
        raise ProtocolAbort("RNG period counts drifted")
    if protocol["seeds"] != list(range(7240, 7280)):
        raise ProtocolAbort("selection seed bank drifted")
    if protocol["blocks"] != [
        list(range(7240, 7260)),
        list(range(7260, 7280)),
    ]:
        raise ProtocolAbort("selection seed blocks drifted")
    expected_laws = [
        "R0",
        "R_D50_W00",
        "R_D75_W00",
        "R_D50_W05",
        "R_D75_W05",
    ]
    if protocol["law_order"] != expected_laws:
        raise ProtocolAbort("law order drifted")
    if config["construction"]["ysd_contrast"] != [1.0, 0.0, -1.0]:
        raise ProtocolAbort("YSD contrast drifted")
    if float(config["construction"]["area_tolerance"]) != 1e-10:
        raise ProtocolAbort("area tolerance drifted")
    if float(config["selector"]["comparison_tolerance"]) != 1e-12:
        raise ProtocolAbort("selector tolerance drifted")
    if int(config["protocol"]["maximum_information_year"]) != 2014:
        raise ProtocolAbort("information boundary drifted")
    if config["runtime_numeric_identity"] != {
        "python_implementation": "CPython",
        "python_version": "3.13.12",
        "numpy_version": "2.4.2",
    }:
        raise ProtocolAbort("numeric runtime identity lock drifted")
    if config["freeze"]["paths"] != [
        "scripts/select_m6_remarriage_round2.py",
        "scripts/m6_remarriage_round2_selector_config.json",
        "scripts/reduce_m6_remarriage_round2.py",
        "scripts/analyze_m6_remarriage_train_delta.py",
    ]:
        raise ProtocolAbort("frozen path list drifted")
    if config["rng"]["seed_root"] != 5200:
        raise ProtocolAbort("RNG seed root drifted")
    if config["rng"]["selection_seeds"] != protocol["seeds"]:
        raise ProtocolAbort("duplicated RNG seed lock drifted")
    if config["rng"]["selection_seed_blocks"] != protocol["blocks"]:
        raise ProtocolAbort("duplicated RNG block lock drifted")
    if config["construction"]["widowed_log_rate_budget"] != (
        0.08956860182931886
    ):
        raise ProtocolAbort("widowed budget drifted")
    construction = config["construction"]
    if (
        construction["raw_age_delta_domain"] != [18, 64]
        or construction["outside_raw_age_domain"] != "exact_R0_probability"
        or construction["divorced_strengths"] != [0.5, 0.75]
        or construction["widowed_logit_options"] != [0.0, 0.05]
        or construction["sum_algorithm"] != "fixed_adjacent_pairwise_float64"
        or construction["root_algorithm"]
        != "first_accepted_midpoint_bisection"
        or construction["root_bracket"] != [0.0, 16.0]
        or construction["root_maximum_iterations"] != 200
        or construction["root_midpoint_expression"] != "lo + (hi - lo) / 2"
    ):
        raise ProtocolAbort("construction algorithm lock drifted")
    numeric = config["numeric_protocol"]
    if (
        numeric["sum_algorithm"] != config["construction"]["sum_algorithm"]
        or numeric["area_relative_tolerance"]
        != config["construction"]["area_tolerance"]
        or numeric["selector_comparison_tolerance"]
        != config["selector"]["comparison_tolerance"]
        or numeric["root"]["bracket"] != config["construction"]["root_bracket"]
        or numeric["root"]["maximum_iterations"]
        != config["construction"]["root_maximum_iterations"]
    ):
        raise ProtocolAbort("duplicated numeric protocol lock drifted")
    for boundary in protocol["boundaries"]:
        key = str(boundary)
        pseudo = config["pseudo_holdouts"]["boundaries"][key]
        if (
            pseudo["evaluation_years"] != protocol["evaluation_years"][key]
            or pseudo["n_periods"] != protocol["n_periods"][key]
            or pseudo["anchor_waves"] != [boundary + 1, boundary + 3]
            or max(pseudo["evaluation_years"]) > 2013
            or pseudo["truth_required_interview_year_max"] > 2014
        ):
            raise ProtocolAbort(f"pseudo-holdout lock drifted at {boundary}")
    if set(config["expected_pseudo_holdouts"]) != {"2006", "2008", "2010"}:
        raise ProtocolAbort("expected pseudo-holdout keys drifted")
    family = config["family"]
    if family["law_order"] != expected_laws or family["origins"] != list(
        ORIGINS
    ):
        raise ProtocolAbort("family order drifted")
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
    for law, (k, omega, outcome) in expected_components.items():
        component = family["law_components"][law]
        if (
            component["divorced_k"] != k
            or component["widowed_omega"] != omega
            or component["per_origin_outcome"] != outcome
        ):
            raise ProtocolAbort(f"law component lock drifted for {law}")
    selector = config["selector"]
    if (
        selector["jackknife_replicates"] != 40
        or selector["delete_one_seed_count"] != 39
        or not selector["jackknife_delete_same_seed_across_all_boundaries"]
        or selector["jackknife_reselect_inside_replicate"]
        or selector["rules"] != 7
        or set(selector["rule_definitions"])
        != {str(index) for index in range(1, 8)}
    ):
        raise ProtocolAbort("selector algorithm lock drifted")
    if config["output"]["full_stdout_json"] != (
        "docs/analysis/m6_remarriage_round2_selection_full.json"
    ):
        raise ProtocolAbort("full stdout publication path drifted")


def _assert_validation_index(
    config: dict[str, Any], validation: dict[str, Any]
) -> None:
    index = config["validation_expectations"]
    for boundary, expected in index["boundaries"].items():
        observed = validation["boundaries"][boundary]
        if (
            observed["support_struck_named_laws"]
            != index["support_struck_named_laws"]
        ):
            raise RootFailure(
                "ROOT_VALIDATION_MISMATCH",
                f"support-strike index drifted at {boundary}",
            )
        if (
            observed["incumbent_table_sha256"]
            != expected["incumbent_table_sha256"]
        ):
            raise RootFailure(
                "ROOT_VALIDATION_MISMATCH",
                f"validation table index drifted at {boundary}",
            )
        for strength, root_expected in expected["roots"].items():
            _assert_expected_subset(
                observed["divorced_calibration"][strength],
                root_expected,
                f"validation index {boundary}/{strength}",
            )


def _pairwise_sum(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("pairwise sum requires one dimension")
    work = [float(value) for value in array.tolist()]
    if not work:
        return 0.0
    while len(work) > 1:
        reduced = [
            work[index] + work[index + 1]
            for index in range(0, len(work) - 1, 2)
        ]
        if len(work) % 2:
            reduced.append(work[-1])
        work = reduced
    return work[0]


def _expit(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    result = scipy_expit(array)
    return float(result) if result.ndim == 0 else result


def _logit(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    result = scipy_logit(array)
    return float(result) if result.ndim == 0 else result


def _strictly_better(value: float, baseline: float, tolerance: float) -> bool:
    return value < baseline - tolerance


def _no_worse(value: float, baseline: float, tolerance: float) -> bool:
    return value <= baseline + tolerance


def _max_year(frame: pd.DataFrame, column: str) -> int | None:
    return round1._max_year(frame, column)


def _assert_at_most(
    frame: pd.DataFrame, column: str, boundary: int, label: str
) -> None:
    round1._assert_at_most(frame, column, boundary, label)


def _frame_checksum(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    return round1._frame_checksum(frame, columns)


def _table_checksum(table: dict[tuple[int, int, str, str], float]) -> str:
    rows = []
    ordered = sorted(table, key=lambda key: (key[2], key[0], key[1], key[3]))
    for key in ordered:
        rows.append("|".join(map(str, key)) + "|" + float(table[key]).hex())
    return _sha256_bytes("\n".join(rows).encode())


def _age_band(age: np.ndarray) -> np.ndarray:
    return band_indices(
        np.rint(age).astype(np.int64),
        rem.AGE_LOWERS,
        len(rem.AGE_BANDS),
    )


def _ysd_band(ysd: np.ndarray) -> np.ndarray:
    return band_indices(
        ysd.astype(np.int64), rem.YSD_LOWERS, len(rem.YSD_BANDS)
    )


def _cell_key_records(
    frame: pd.DataFrame, *, origin_column: str
) -> list[tuple[int, int, str, str]]:
    age_band = _age_band(frame["age"].to_numpy(dtype=np.float64))
    ysd_band = _ysd_band(
        frame["years_since_dissolution"].to_numpy(dtype=np.int64)
    )
    return list(
        zip(
            age_band,
            ysd_band,
            frame[origin_column].astype(str),
            frame["sex"].astype(str),
            strict=True,
        )
    )


def _lookup_probabilities(
    frame: pd.DataFrame,
    candidate: dict[tuple[int, int, str, str], float],
    baseline: dict[tuple[int, int, str, str], float],
    *,
    origin_column: str,
) -> np.ndarray:
    keys = _cell_key_records(frame, origin_column=origin_column)
    candidate_value = np.asarray(
        [candidate[key] for key in keys], dtype=np.float64
    )
    baseline_value = np.asarray(
        [baseline[key] for key in keys], dtype=np.float64
    )
    age = frame["age"].to_numpy(dtype=np.float64)
    in_domain = (age >= 18.0) & (age <= 64.0)
    result = np.where(in_domain, candidate_value, baseline_value)
    if not np.array_equal(result[~in_domain], baseline_value[~in_domain]):
        raise AssertionError("raw-age outside-domain probability changed")
    return result


def _remarriage_fit(
    context: ft_registry.FitContext,
    incumbent: dict[tuple[int, int, str, str], float],
) -> tuple[
    dict[tuple[int, int, str, str], float],
    list[dict[str, Any]],
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    dissolved, remarriages, mean_weight = round1._remarriage_frames(context)
    if dissolved.empty or mean_weight <= 0:
        raise ProtocolAbort("fit has no positive dissolved exposure")
    if not (dissolved["weight"] > 0).all():
        raise ProtocolAbort("fit includes non-positive dissolved weight")
    denominator = dissolved.groupby(
        ["age_band", "ysd_band", "marital_state", "sex"]
    )["weight"].sum()
    numerator = remarriages.groupby(["age_band", "ysd_band", "origin", "sex"])[
        "weight"
    ].sum()
    row_count = dissolved.groupby(
        ["age_band", "ysd_band", "marital_state", "sex"]
    ).size()
    event_count = remarriages.groupby(
        ["age_band", "ysd_band", "origin", "sex"]
    ).size()
    table: dict[tuple[int, int, str, str], float] = {}
    cells: list[dict[str, Any]] = []
    for age_band in range(len(rem.AGE_BANDS)):
        for ysd_band in range(len(rem.YSD_BANDS)):
            for origin in ORIGINS:
                for sex in SEXES:
                    key = (age_band, ysd_band, origin, sex)
                    exposure = float(denominator.get(key, 0.0))
                    events = float(numerator.get(key, 0.0))
                    probability = (events + mean_weight) / (
                        exposure + 2.0 * mean_weight
                    )
                    table[key] = probability
                    cells.append(
                        {
                            "age_band_index": age_band,
                            "age_band": list(rem.AGE_BANDS[age_band]),
                            "ysd_band_index": ysd_band,
                            "ysd_band": list(rem.YSD_BANDS[ysd_band]),
                            "origin": origin,
                            "sex": sex,
                            "risk_rows": int(row_count.get(key, 0)),
                            "event_rows": int(event_count.get(key, 0)),
                            "weighted_exposure": exposure,
                            "weighted_events": events,
                            "R0_probability": probability,
                        }
                    )
    if set(table) != set(incumbent):
        raise ProtocolAbort("R0 table keys differ from incumbent")
    if any(table[key] != incumbent[key] for key in table):
        maximum = max(abs(table[key] - incumbent[key]) for key in table)
        raise ProtocolAbort(
            f"R0 table is not bit-identical to incumbent: {maximum}"
        )
    incumbent_lookup = rem.build_remarriage_lookup(incumbent)
    rebuilt_lookup = rem.build_remarriage_lookup(table)
    if not np.array_equal(incumbent_lookup, rebuilt_lookup):
        raise ProtocolAbort("R0 lookup is not bit-identical to incumbent")
    return table, cells, mean_weight, dissolved, remarriages


def _fit_support_by_origin_ysd(
    dissolved: pd.DataFrame,
    remarriages: pd.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    risk = dissolved[dissolved["age"].between(18, 64)].copy()
    events = remarriages[remarriages["age"].between(18, 64)].copy()
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for origin in ORIGINS:
        output[origin] = {}
        for ysd_band, label in enumerate(("0-4", "5-9", "10-120")):
            selected_risk = risk[
                risk["marital_state"].eq(origin)
                & risk["ysd_band"].eq(ysd_band)
            ]
            selected_events = events[
                events["origin"].eq(origin) & events["ysd_band"].eq(ysd_band)
            ]
            output[origin][label] = {
                "risk_rows": len(selected_risk),
                "unweighted_events": len(selected_events),
                "weighted_exposure": _pairwise_sum(
                    selected_risk["weight"].astype(float).tolist()
                ),
            }
    return output


def _divorced_center(
    dissolved: pd.DataFrame, contrast: tuple[float, ...]
) -> tuple[float, list[float], pd.DataFrame]:
    selected = dissolved[
        dissolved["marital_state"].eq("divorced")
        & dissolved["years_since_dissolution"].notna()
        & dissolved["age"].between(18, 64)
    ].copy()
    ysd_index = selected["ysd_band"].to_numpy(dtype=np.int64)
    weights = selected["weight"].to_numpy(dtype=np.float64)
    contrast_array = np.asarray(contrast, dtype=np.float64)
    numerator = _pairwise_sum((weights * contrast_array[ysd_index]).tolist())
    denominator = _pairwise_sum(weights.tolist())
    if denominator <= 0:
        raise ProtocolAbort(
            "divorced working-age fit exposure is non-positive"
        )
    center = numerator / denominator
    centered = [float(value - center) for value in contrast]
    centered_array = np.asarray(centered, dtype=np.float64)
    centered_residual = (
        _pairwise_sum((weights * centered_array[ysd_index]).tolist())
        / denominator
    )
    if abs(centered_residual) > 1e-12:
        raise ProtocolAbort("divorced exposure centering failed")
    return center, centered, selected


def _category_checksum(records: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(records))


def _legacy_reference_spells(
    context: ft_registry.FitContext,
    boundary: int,
    origin: str,
) -> dict[str, Any]:
    episodes = marriage.marriage_episodes(context.marriage_records).copy()
    episodes = episodes.sort_values(
        ["person_id", "marriage_order", "start_year"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    episodes["next_start_year"] = episodes.groupby("person_id", sort=False)[
        "start_year"
    ].shift(-1)
    dissolution_label = "divorce" if origin == "divorced" else "widowhood"
    spells = episodes[
        (episodes["how_ended"] == dissolution_label)
        & episodes["episode_end_year"].notna()
    ][
        [
            "person_id",
            "episode_end_year",
            "next_start_year",
        ]
    ].copy()
    spells = spells.rename(columns={"episode_end_year": "dissolution_year"})
    spells["origin"] = origin
    attrs = context.panel.attrs[
        ["person_id", "sex", "birth_year", "censor_year", "weight"]
    ].copy()
    spells = spells.merge(
        attrs,
        on="person_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    required = (
        "person_id",
        "dissolution_year",
        "origin",
        "sex",
        "birth_year",
        "censor_year",
        "weight",
    )
    missing_mask = spells[list(required)].isna().any(axis=1) | ~(
        pd.to_numeric(spells["weight"], errors="coerce") > 0
    )
    missing = spells[missing_mask].copy()
    retained = spells[~missing_mask].copy()
    for column in (
        "person_id",
        "dissolution_year",
        "birth_year",
        "censor_year",
    ):
        retained[column] = retained[column].astype("int64")
    retained["weight"] = retained["weight"].astype("float64")
    retained = retained.sort_values(
        ["person_id", "dissolution_year", "origin"], kind="stable"
    ).reset_index(drop=True)
    eligible_count = len(retained)

    duplicate_mask = retained.duplicated(
        ["person_id", "dissolution_year", "origin"], keep=False
    )
    duplicate = retained[duplicate_mask].copy()
    duplicate_groups = int(
        duplicate[["person_id", "dissolution_year", "origin"]]
        .drop_duplicates()
        .shape[0]
    )
    retained = retained[~duplicate_mask].copy()

    same_year_mask = (
        pd.to_numeric(retained["next_start_year"], errors="coerce")
        == retained["dissolution_year"]
    )
    same_year = retained[same_year_mask].copy()
    retained = retained[~same_year_mask].copy()

    retained["path_end"] = np.minimum.reduce(
        [
            np.full(len(retained), boundary, dtype=np.int64),
            retained["censor_year"].to_numpy(dtype=np.int64),
            retained["birth_year"].to_numpy(dtype=np.int64) + 64,
        ]
    )
    retained["path_start"] = (
        retained["dissolution_year"].to_numpy(dtype=np.int64) + 1
    )
    no_path_mask = retained["path_end"] < retained["path_start"]
    no_path = retained[no_path_mask].copy()
    included = retained[~no_path_mask].copy()

    path_records: list[dict[str, Any]] = []
    spell_records: list[dict[str, Any]] = []
    potential_path_years = 0
    working_age_terms = 0
    for row in included.itertuples(index=False):
        years = tuple(range(int(row.path_start), int(row.path_end) + 1))
        working = tuple(
            year for year in years if year >= int(row.birth_year) + 18
        )
        potential_path_years += len(years)
        working_age_terms += len(working)
        record = {
            "person_id": int(row.person_id),
            "dissolution_year": int(row.dissolution_year),
            "origin": origin,
            "sex": str(row.sex),
            "birth_year": int(row.birth_year),
            "censor_year": int(row.censor_year),
            "weight": float(row.weight),
            "path_start": int(row.path_start),
            "path_end": int(row.path_end),
            "years": years,
            "working_years": working,
        }
        spell_records.append(record)
        for year in years:
            path_records.append(
                {
                    "origin": origin,
                    "person_id": int(row.person_id),
                    "dissolution_year": int(row.dissolution_year),
                    "year": year,
                    "sex": str(row.sex),
                    "birth_year": int(row.birth_year),
                    "censor_year": int(row.censor_year),
                    "weight": float(row.weight),
                }
            )
    path_frame = pd.DataFrame(path_records)
    path_columns = (
        "origin",
        "person_id",
        "dissolution_year",
        "year",
        "sex",
        "birth_year",
        "censor_year",
        "weight",
    )
    path_checksum = (
        _frame_checksum(path_frame, path_columns)
        if len(path_frame)
        else _sha256_bytes(b"")
    )

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        columns = [
            "person_id",
            "dissolution_year",
            "origin",
            "sex",
            "birth_year",
            "censor_year",
            "weight",
        ]
        present = [column for column in columns if column in frame]
        output = (
            frame[present]
            .copy()
            .sort_values(
                [
                    column
                    for column in ("person_id", "dissolution_year", "origin")
                    if column in present
                ],
                kind="stable",
            )
        )
        return _plain(output.to_dict(orient="records"))

    audit = {
        "eligible_spells_before_exclusions": eligible_count,
        "missing_required_or_nonpositive_weight_spells_excluded": len(missing),
        "missing_required_or_nonpositive_weight_spells_excluded_weight": float(
            pd.to_numeric(missing.get("weight"), errors="coerce")
            .fillna(0)
            .sum()
            if "weight" in missing
            else 0.0
        ),
        "missing_required_or_nonpositive_weight_checksum_sha256": (
            _category_checksum(records(missing))
        ),
        "duplicate_key_groups": duplicate_groups,
        "duplicate_spells_excluded": len(duplicate),
        "duplicate_spells_excluded_weight": float(duplicate["weight"].sum()),
        "duplicate_spells_checksum_sha256": _category_checksum(
            records(duplicate)
        ),
        "same_year_remarriage_spells_excluded": len(same_year),
        "same_year_remarriage_spells_excluded_weight": float(
            same_year["weight"].sum()
        ),
        "same_year_remarriage_spells_checksum_sha256": _category_checksum(
            records(same_year)
        ),
        "no_potential_path_spells_excluded": len(no_path),
        "no_potential_path_spells_excluded_weight": float(
            no_path["weight"].sum()
        ),
        "no_potential_path_spells_checksum_sha256": _category_checksum(
            records(no_path)
        ),
        "included_spells": len(included),
        "included_spell_weight": float(included["weight"].sum()),
        "included_spells_checksum_sha256": _category_checksum(
            records(included)
        ),
        "potential_path_years": potential_path_years,
        "working_age_path_year_terms": working_age_terms,
        "path_checksum_sha256": path_checksum,
    }
    return {"audit": audit, "spells": spell_records}


def _legacy_path_hazard(
    spell: dict[str, Any],
    year: int,
    table: dict[tuple[int, int, str, str], float],
    baseline: dict[tuple[int, int, str, str], float],
) -> float:
    age = year - int(spell["birth_year"])
    ysd = year - int(spell["dissolution_year"])
    age_index = int(_age_band(np.asarray([age], dtype=np.float64))[0])
    ysd_index = int(_ysd_band(np.asarray([ysd], dtype=np.int64))[0])
    key = (age_index, ysd_index, spell["origin"], spell["sex"])
    return table[key] if 18 <= age <= 64 else baseline[key]


def _legacy_reference_area(
    reference: dict[str, Any],
    table: dict[tuple[int, int, str, str], float],
    baseline: dict[tuple[int, int, str, str], float],
) -> tuple[float, int]:
    contributions: list[float] = []
    spells = sorted(
        reference["spells"],
        key=lambda row: (
            row["origin"],
            row["person_id"],
            row["dissolution_year"],
        ),
    )
    for spell in spells:
        survival = np.float64(1.0)
        working = set(spell["working_years"])
        for year in spell["years"]:
            if year in working:
                contributions.append(
                    float(np.float64(spell["weight"]) * survival)
                )
            hazard = np.float64(
                _legacy_path_hazard(spell, year, table, baseline)
            )
            survival = survival * (np.float64(1.0) - hazard)
    return _pairwise_sum(contributions), len(contributions)


def _spell_category_records(
    frame: pd.DataFrame, boundary: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        dissolution = (
            None
            if pd.isna(row.dissolution_year)
            else int(row.dissolution_year)
        )
        birth = None if pd.isna(row.birth_year) else int(row.birth_year)
        censor = None if pd.isna(row.censor_year) else int(row.censor_year)
        records.append(
            {
                "person_id": int(row.person_id),
                "dissolution_year": dissolution,
                "origin": str(row.origin),
                "sex": None if pd.isna(row.sex) else str(row.sex),
                "birth_year": birth,
                "censor_year": censor,
                "weight": None if pd.isna(row.weight) else float(row.weight),
                "path_start": (
                    None if dissolution is None else dissolution + 1
                ),
                "path_end": (
                    None
                    if birth is None or censor is None
                    else min(boundary, censor, birth + 64)
                ),
            }
        )
    return records


def _reference_spells(
    context: ft_registry.FitContext,
    baseline: dict[tuple[int, int, str, str], float],
    boundary: int,
    origin: str,
) -> dict[str, Any]:
    events = context.panel.events
    dissolutions = events[
        events["transition"].isin(("divorce", "widowhood"))
    ].copy()
    dissolutions["origin_key"] = dissolutions["transition"].map(
        {"divorce": "divorced", "widowhood": "widowed"}
    )
    dissolutions = dissolutions[dissolutions["origin_key"].eq(origin)]
    spells = dissolutions[["person_id", "year", "origin_key"]].rename(
        columns={"year": "dissolution_year", "origin_key": "origin"}
    )
    spells = spells.merge(
        context.panel.attrs[
            ["person_id", "sex", "birth_year", "censor_year", "weight"]
        ],
        on="person_id",
        how="left",
        validate="many_to_one",
    )
    eligible = (
        spells["sex"].isin(SEXES)
        & spells["birth_year"].notna()
        & spells["censor_year"].notna()
        & spells["weight"].gt(0)
    )
    missing = spells[~eligible].copy()
    spells = spells[eligible].copy()
    duplicate_mask = spells.duplicated(
        ["person_id", "dissolution_year", "origin"], keep=False
    )
    duplicates = spells[duplicate_mask].copy()
    spells = spells[~duplicate_mask].copy()
    remarriages = events[events["transition"].eq("remarriage")]
    same_year_keys = set(
        zip(
            remarriages["person_id"].astype(int),
            remarriages["year"].astype(int),
            remarriages["origin"],
            strict=True,
        )
    )
    same_year_mask = [
        (int(person), int(year), spell_origin) in same_year_keys
        for person, year, spell_origin in zip(
            spells["person_id"],
            spells["dissolution_year"],
            spells["origin"],
            strict=True,
        )
    ]
    same_year = spells[same_year_mask].copy()
    spells = spells[np.logical_not(same_year_mask)].sort_values(
        ["person_id", "dissolution_year", "origin"], kind="stable"
    )

    paths: list[dict[str, Any]] = []
    included_records: list[dict[str, Any]] = []
    no_path_records: list[dict[str, Any]] = []
    included_weights: list[float] = []
    checksum_rows: list[str] = []
    potential_years = 0
    working_years = 0
    for row in spells.itertuples(index=False):
        birth = int(row.birth_year)
        start = int(row.dissolution_year) + 1
        end = min(boundary, int(row.censor_year), birth + 64)
        serialized = {
            "person_id": int(row.person_id),
            "dissolution_year": int(row.dissolution_year),
            "origin": origin,
            "sex": str(row.sex),
            "birth_year": birth,
            "censor_year": int(row.censor_year),
            "weight": float(row.weight),
            "path_start": start,
            "path_end": end,
        }
        if start > end:
            no_path_records.append(serialized)
            continue
        years = np.arange(start, end + 1, dtype=np.int64)
        ages = years - birth
        ysd = years - int(row.dissolution_year)
        age_index = np.clip(
            np.searchsorted(rem.AGE_LOWERS, ages, side="right") - 1,
            0,
            len(rem.AGE_BANDS) - 1,
        )
        ysd_index = np.clip(
            np.searchsorted(rem.YSD_LOWERS, ysd, side="right") - 1,
            0,
            len(rem.YSD_BANDS) - 1,
        )
        logits = np.asarray(
            [
                scipy_logit(
                    baseline[
                        (int(age_band), int(ysd_band), origin, str(row.sex))
                    ]
                )
                for age_band, ysd_band in zip(
                    age_index, ysd_index, strict=True
                )
            ],
            dtype=np.float64,
        )
        working = (ages >= 18) & (ages <= 64)
        paths.append(
            {
                "weight": float(row.weight),
                "logits": logits,
                "ysd_index": ysd_index,
                "working": working,
            }
        )
        included_records.append(serialized)
        included_weights.append(float(row.weight))
        potential_years += len(years)
        working_years += int(working.sum())
        checksum_rows.append(
            f"{int(row.person_id)}|{int(row.dissolution_year)}|{origin}|"
            f"{row.sex}|{birth}|{int(row.censor_year)}|"
            f"{float(row.weight).hex()}|{start}|{end}"
        )

    duplicate_records = _spell_category_records(duplicates, boundary)
    same_year_records = _spell_category_records(same_year, boundary)
    missing_records = _spell_category_records(missing, boundary)
    missing_weights = [
        float(value)
        for value in missing.get("weight", pd.Series(dtype=float)).dropna()
        if float(value) > 0
    ]
    audit = {
        "eligible_spells_before_exclusions": (
            len(spells) + len(duplicates) + len(same_year)
        ),
        "duplicate_key_groups": (
            int(
                duplicates.groupby(
                    ["person_id", "dissolution_year", "origin"]
                ).ngroups
            )
            if len(duplicates)
            else 0
        ),
        "duplicate_spells_excluded": len(duplicates),
        "duplicate_spells_excluded_weight": _pairwise_sum(
            duplicates["weight"].astype(float).tolist()
        ),
        "duplicate_spells_checksum_sha256": _category_checksum(
            duplicate_records
        ),
        "same_year_remarriage_spells_excluded": len(same_year),
        "same_year_remarriage_spells_excluded_weight": _pairwise_sum(
            same_year["weight"].astype(float).tolist()
        ),
        "same_year_remarriage_spells_checksum_sha256": _category_checksum(
            same_year_records
        ),
        "missing_required_or_nonpositive_weight_spells_excluded": len(missing),
        "missing_required_or_nonpositive_weight_spells_excluded_weight": (
            _pairwise_sum(missing_weights)
        ),
        "missing_required_or_nonpositive_weight_checksum_sha256": (
            _category_checksum(missing_records)
        ),
        "no_potential_path_spells_excluded": len(no_path_records),
        "no_potential_path_spells_excluded_weight": _pairwise_sum(
            [row["weight"] for row in no_path_records]
        ),
        "no_potential_path_spells_checksum_sha256": _category_checksum(
            no_path_records
        ),
        "included_spells": len(paths),
        "included_spell_weight": _pairwise_sum(included_weights),
        "included_spells_checksum_sha256": _category_checksum(
            included_records
        ),
        "potential_path_years": potential_years,
        "working_age_path_year_terms": working_years,
        "path_checksum_sha256": _sha256_bytes(
            "\n".join(checksum_rows).encode()
        ),
    }
    return {"paths": paths, "audit": audit}


def _reference_area(
    reference: dict[str, Any],
    origin: str,
    centered: list[float],
    *,
    k: float,
    beta: float,
    omega: float,
) -> tuple[float, int]:
    contributions: list[float] = []
    for path in reference["paths"]:
        survival = 1.0
        for offset in range(len(path["logits"])):
            if bool(path["working"][offset]):
                contributions.append(float(path["weight"]) * survival)
            delta = 0.0
            if bool(path["working"][offset]):
                delta = (
                    -k + beta * centered[int(path["ysd_index"][offset])]
                    if origin == "divorced"
                    else omega
                )
            survival *= 1.0 - float(
                scipy_expit(float(path["logits"][offset]) + delta)
            )
    return _pairwise_sum(contributions), len(contributions)


def _apply_law(
    baseline: dict[tuple[int, int, str, str], float],
    centered: list[float],
    *,
    k: float,
    beta: float,
    omega: float,
) -> dict[tuple[int, int, str, str], float]:
    table = dict(baseline)
    for age_band in WORKING_AGE_BANDS:
        for ysd_band in range(len(rem.YSD_BANDS)):
            for sex in SEXES:
                divorced_key = (age_band, ysd_band, "divorced", sex)
                divorced_shift = -k + beta * centered[ysd_band]
                table[divorced_key] = float(
                    _expit(_logit(baseline[divorced_key]) + divorced_shift)
                )
                widowed_key = (age_band, ysd_band, "widowed", sex)
                table[widowed_key] = (
                    baseline[widowed_key]
                    if omega == 0.0
                    else float(_expit(_logit(baseline[widowed_key]) + omega))
                )
    return table


def _applied_direction(risk: pd.DataFrame, shifts: np.ndarray) -> float:
    ysd_index = risk["ysd_band"].to_numpy(dtype=np.int64)
    weights = risk["weight"].to_numpy(dtype=np.float64)
    return _pairwise_sum((weights * shifts[ysd_index]).tolist()) / (
        _pairwise_sum(weights.tolist())
    )


def _solve_divorced_root(
    baseline: dict[tuple[int, int, str, str], float],
    divorced_risk: pd.DataFrame,
    center: float,
    centered: list[float],
    reference: dict[str, Any],
    *,
    k: float,
    omega: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[dict[tuple[int, int, str, str], float], dict[str, Any]]:
    baseline_area, term_count = _reference_area(
        reference,
        "divorced",
        [0.0, 0.0, 0.0],
        k=0.0,
        beta=0.0,
        omega=0.0,
    )
    if not math.isfinite(baseline_area) or baseline_area <= 0:
        raise RootFailure("ROOT_NONFINITE", "invalid R0 divorced area")

    def evaluate(beta: float) -> tuple[float, dict[Any, float], float]:
        table = _apply_law(
            baseline,
            centered,
            k=k,
            beta=beta,
            omega=omega,
        )
        area, observed_terms = _reference_area(
            reference,
            "divorced",
            centered,
            k=k,
            beta=beta,
            omega=0.0,
        )
        if observed_terms != term_count:
            raise AssertionError("reference-area term count changed")
        residual = area / baseline_area - 1.0
        if not math.isfinite(residual):
            raise RootFailure("ROOT_NONFINITE", f"beta={beta}")
        return residual, table, area

    low_residual, _, _ = evaluate(0.0)
    high_residual, _, _ = evaluate(16.0)
    if not (low_residual > 0.0 > high_residual):
        raise RootFailure(
            "ROOT_NO_BRACKET",
            f"G(0)={low_residual}, G(16)={high_residual}",
        )
    low = 0.0
    high = 16.0
    accepted: tuple[float, dict[Any, float], float, float, int] | None = None
    for iteration in range(1, maximum_iterations + 1):
        midpoint = low + (high - low) / 2.0
        if midpoint == low or midpoint == high:
            raise RootFailure("ROOT_STAGNATION", f"iteration={iteration}")
        residual, table, area = evaluate(midpoint)
        if abs(residual) <= tolerance:
            accepted = (midpoint, table, area, residual, iteration)
            break
        if residual > 0.0:
            low = midpoint
        else:
            high = midpoint
    if accepted is None:
        raise RootFailure(
            "ROOT_ITERATION_LIMIT", f"no root after {maximum_iterations}"
        )
    beta, table, area, residual, iterations = accepted
    shifts = -k + beta * np.asarray(centered, dtype=np.float64)
    probabilities = [
        float(_expit(_logit(value) + shifts[cell_ysd]))
        for (cell_age, cell_ysd, origin, _sex), value in baseline.items()
        if cell_age < 3 and origin == "divorced"
    ]
    direction = _applied_direction(divorced_risk, shifts)
    ledger = {
        "alpha_divorced": -k,
        "beta_frontload": beta,
        "fit_exposure_center": center,
        "centered_contrast": centered,
        "bracket_residual_low": low_residual,
        "bracket_residual_high": high_residual,
        "root_iterations": iterations,
        "area_R0": baseline_area,
        "candidate_area": area,
        "area_relative_residual": residual,
        "pairwise_term_count": term_count,
        "effective_divorced_shift": direction,
        "minimum_cell_logit_shift": float(np.min(shifts)),
        "maximum_cell_logit_shift": float(np.max(shifts)),
        "candidate_probability_min": min(probabilities),
        "candidate_probability_max": max(probabilities),
    }
    return table, ledger


def _assert_expected_subset(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise RootFailure(
                "ROOT_VALIDATION_MISMATCH", f"{label} is not an object"
            )
        for key, expected_value in expected.items():
            if key not in observed:
                raise RootFailure(
                    "ROOT_VALIDATION_MISMATCH",
                    f"{label} lacks {key!r}",
                )
            _assert_expected_subset(
                observed[key], expected_value, f"{label}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RootFailure(
                "ROOT_VALIDATION_MISMATCH", f"{label} list differs"
            )
        for index, (observed_value, expected_value) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _assert_expected_subset(
                observed_value, expected_value, f"{label}[{index}]"
            )
        return
    if observed != expected:
        raise RootFailure(
            "ROOT_VALIDATION_MISMATCH",
            f"{label}: observed={observed!r}, expected={expected!r}",
        )


def _table_cells(
    base_cells: list[dict[str, Any]],
    table: dict[tuple[int, int, str, str], float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in base_cells:
        key = (
            row["age_band_index"],
            row["ysd_band_index"],
            row["origin"],
            row["sex"],
        )
        output.append({**row, "probability": table[key]})
    return output


def _table_delta(
    risk: pd.DataFrame,
    table: dict[tuple[int, int, str, str], float],
    baseline: dict[tuple[int, int, str, str], float],
    origin: str,
) -> float:
    selected = risk[
        risk["marital_state"].eq(origin) & risk["age"].between(18, 64)
    ]
    keys = _cell_key_records(selected, origin_column="marital_state")
    weights = selected["weight"].to_numpy(dtype=np.float64)
    shifts = np.asarray(
        [
            float(scipy_logit(table[key]) - scipy_logit(baseline[key]))
            for key in keys
        ],
        dtype=np.float64,
    )
    return _pairwise_sum((weights * shifts).tolist()) / _pairwise_sum(
        weights.tolist()
    )


def _law_components(config: dict[str, Any], law: str) -> tuple[float, float]:
    record = config["family"]["law_components"][law]
    return float(record["divorced_k"]), float(record["widowed_omega"])


def _assert_fit_context_boundary(
    context: ft_registry.FitContext, boundary: int
) -> dict[str, Any]:
    actual_marriages = context.marriage_records[
        context.marriage_records["is_marriage"].fillna(False)
    ]
    actual_births = context.birth_records[
        context.birth_records["is_event"].fillna(False)
    ]
    frames_and_fields = (
        ("panel_person_years", context.panel.person_years, "year"),
        (
            "panel_person_years",
            context.panel.person_years,
            refit.REQUIRED_INTERVIEW_COLUMN,
        ),
        ("panel_events", context.panel.events, "year"),
        (
            "panel_events",
            context.panel.events,
            refit.REQUIRED_INTERVIEW_COLUMN,
        ),
        ("panel_attrs", context.panel.attrs, "start_exposure_year"),
        ("panel_attrs", context.panel.attrs, "censor_year"),
        ("demographic_panel", context.demographic_panel, "period"),
        ("actual_marriages", actual_marriages, "start_year"),
        ("actual_marriages", actual_marriages, "end_year"),
        ("actual_marriages", actual_marriages, "separation_year"),
        (
            "actual_marriages",
            actual_marriages,
            "most_recent_report_year",
        ),
        ("actual_births", actual_births, "birth_year"),
        (
            "actual_births",
            actual_births,
            "most_recent_child_report_year",
        ),
        ("marriage_order_map", context.marriage_order_map, "start_year"),
    )
    maxima: dict[str, int | None] = {}
    for label, frame, column in frames_and_fields:
        if column not in frame:
            continue
        _assert_at_most(frame, column, boundary, f"fit_{label}")
        maxima[f"{label}.{column}"] = _max_year(frame, column)
    _assert_at_most(
        actual_marriages,
        "start_year",
        boundary,
        "fit_actual_marriages",
    )
    _assert_at_most(actual_births, "birth_year", boundary, "fit_actual_births")
    if "n_marriages" in context.marriage_records:
        counts = actual_marriages.groupby("person_id").size()
        expected = (
            context.marriage_records["person_id"]
            .map(counts)
            .fillna(0)
            .astype("Int64")
        )
        observed = context.marriage_records["n_marriages"].astype("Int64")
        if not observed.equals(expected):
            raise ProtocolAbort("truncated marriage counts were not refreshed")
    if "n_marriages" in context.panel.attrs:
        counts = actual_marriages.groupby("person_id").size()
        expected = (
            context.panel.attrs["person_id"]
            .map(counts)
            .fillna(0)
            .astype(float)
        )
        observed = context.panel.attrs["n_marriages"].astype(float)
        if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
            raise ProtocolAbort("truncated panel marriage counts drifted")
    return {
        "boundary": boundary,
        "field_maxima": maxima,
        "post_boundary_end_and_separation_fields_nulled": True,
        "marriage_counts_recomputed": True,
        "every_fit_frame_field_asserted": True,
    }


def _fit_boundary(
    context: ft_registry.FitContext,
    config: dict[str, Any],
    validation: dict[str, Any],
    boundary: int,
    law_order: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    _progress(f"fit-side preflight: boundary {boundary}")
    truncated = refit._truncate_family_context(context, boundary)
    field_boundary_audit = _assert_fit_context_boundary(truncated, boundary)
    components = ft_registry.REGISTRY.fit(ft_registry.CANDIDATE_16, truncated)
    (
        baseline,
        base_cells,
        mean_weight,
        dissolved,
        remarriages,
    ) = _remarriage_fit(truncated, components.remarriage)
    support = _fit_support_by_origin_ysd(dissolved, remarriages)
    contrast = tuple(
        float(value) for value in config["construction"]["ysd_contrast"]
    )
    center, centered, divorced_risk = _divorced_center(dissolved, contrast)
    references = {
        origin: _reference_spells(truncated, baseline, boundary, origin)
        for origin in ORIGINS
    }
    r0_areas: dict[str, float] = {}
    for origin in ORIGINS:
        r0_areas[origin], term_count = _reference_area(
            references[origin],
            origin,
            [0.0, 0.0, 0.0],
            k=0.0,
            beta=0.0,
            omega=0.0,
        )
        if (
            term_count
            != references[origin]["audit"]["working_age_path_year_terms"]
        ):
            raise RootFailure(
                "ROOT_VALIDATION_MISMATCH",
                f"{boundary}/{origin} pairwise term count differs",
            )
        references[origin]["audit"]["R0_area"] = r0_areas[origin]

    tolerance = float(config["construction"]["area_tolerance"])
    maximum_iterations = int(config["construction"]["root_maximum_iterations"])
    requested_laws = (
        tuple(config["protocol"]["law_order"])
        if law_order is None
        else law_order
    )
    unknown_laws = set(requested_laws).difference(
        config["protocol"]["law_order"]
    )
    if unknown_laws or not requested_laws:
        raise ProtocolAbort(
            f"invalid requested law subset at {boundary}: {requested_laws}"
        )
    requested_components = {
        law: _law_components(config, law) for law in requested_laws
    }

    calibration: dict[str, dict[str, Any]] = {}
    beta_by_k: dict[float, float] = {}
    requested_strengths = sorted(
        {k for k, _omega in requested_components.values() if k > 0.0}
    )
    for k in requested_strengths:
        _, ledger = _solve_divorced_root(
            baseline,
            divorced_risk,
            center,
            centered,
            references["divorced"],
            k=k,
            omega=0.0,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        calibration[f"{k:.2f}"] = ledger
        beta_by_k[k] = float(ledger["beta_frontload"])

    widowed_targets: dict[str, dict[str, Any]] = {}
    budget = float(config["construction"]["widowed_log_rate_budget"])
    requested_omegas = sorted(
        {omega for _k, omega in requested_components.values()}
    )
    for omega in requested_omegas:
        target, term_count = _reference_area(
            references["widowed"],
            "widowed",
            [0.0, 0.0, 0.0],
            k=0.0,
            beta=0.0,
            omega=omega,
        )
        if (
            term_count
            != references["widowed"]["audit"]["working_age_path_year_terms"]
        ):
            raise RootFailure(
                "ROOT_VALIDATION_MISMATCH",
                f"{boundary}/widowed target term count differs",
            )
        probabilities = [
            float(scipy_expit(scipy_logit(value) + omega))
            for (cell_age, _cell_ysd, origin, _sex), value in baseline.items()
            if cell_age < 3 and origin == "widowed"
        ]
        widowed_targets[f"{omega:.2f}"] = {
            "alpha_widowed": omega,
            "effective_widowed_shift": omega,
            "minimum_cell_logit_shift": omega,
            "maximum_cell_logit_shift": omega,
            "target_area": target,
            "target_over_R0_area": target / r0_areas["widowed"],
            "target_area_relative_residual": 0.0,
            "candidate_probability_min": min(probabilities),
            "candidate_probability_max": max(probabilities),
            "within_log_rate_budget": omega <= budget,
        }

    expected = validation["boundaries"][str(boundary)]
    observed_validation = {
        "fit_max_year": int(truncated.panel.person_years["year"].max()),
        "incumbent_table_sha256": _table_checksum(baseline),
        "fit_support_by_origin_ysd": support,
        "reference_spells": {
            origin: references[origin]["audit"] for origin in ORIGINS
        },
        "divorced_calibration": calibration,
        "widowed_targets": widowed_targets,
        "support_struck_named_laws": [],
    }
    expected_relevant = {
        "fit_max_year": expected["fit_max_year"],
        "incumbent_table_sha256": expected["incumbent_table_sha256"],
        "fit_support_by_origin_ysd": expected["fit_support_by_origin_ysd"],
        "reference_spells": expected["reference_spells"],
        "divorced_calibration": {
            key: expected["divorced_calibration"][key] for key in calibration
        },
        "widowed_targets": {
            key: expected["widowed_targets"][key] for key in widowed_targets
        },
        "support_struck_named_laws": expected["support_struck_named_laws"],
    }
    _assert_expected_subset(
        observed_validation,
        expected_relevant,
        f"validation.boundaries.{boundary}",
    )

    laws: dict[str, Any] = {}
    for law in requested_laws:
        k, omega = _law_components(config, law)
        beta = 0.0 if k == 0.0 else beta_by_k[k]
        table = (
            dict(baseline)
            if law == "R0"
            else _apply_law(baseline, centered, k=k, beta=beta, omega=omega)
        )
        divorced_area, _ = _reference_area(
            references["divorced"],
            "divorced",
            centered,
            k=k,
            beta=beta,
            omega=0.0,
        )
        widowed_area, _ = _reference_area(
            references["widowed"],
            "widowed",
            [0.0, 0.0, 0.0],
            k=0.0,
            beta=0.0,
            omega=omega,
        )
        divorced_shifts = (
            np.zeros(3, dtype=np.float64)
            if law == "R0"
            else -k + beta * np.asarray(centered, dtype=np.float64)
        )
        positive_divorced_cells = [
            row
            for row in base_cells
            if row["origin"] == "divorced"
            and row["age_band_index"] in WORKING_AGE_BANDS
            and row["weighted_exposure"] > 0.0
        ]
        widowed_table_construction_exact = all(
            table[(age_band, ysd_band, "widowed", sex)]
            == (
                baseline[(age_band, ysd_band, "widowed", sex)]
                if omega == 0.0
                else float(
                    _expit(
                        _logit(baseline[(age_band, ysd_band, "widowed", sex)])
                        + omega
                    )
                )
            )
            for age_band in WORKING_AGE_BANDS
            for ysd_band in range(len(rem.YSD_BANDS))
            for sex in SEXES
        )
        laws[law] = {
            "table": table,
            "public": {
                "law": law,
                "divorced_k": k,
                "widowed_omega": omega,
                "beta_frontload": beta,
                "table_sha256": _table_checksum(table),
                "cells": _table_cells(base_cells, table),
                "Delta_divorced": _table_delta(
                    dissolved, table, baseline, "divorced"
                ),
                "Delta_widowed": _table_delta(
                    dissolved, table, baseline, "widowed"
                ),
                "applied_divorced_direction": _applied_direction(
                    divorced_risk, divorced_shifts
                ),
                "divorced_area_target": r0_areas["divorced"],
                "divorced_area": divorced_area,
                "divorced_area_relative_residual": (
                    divorced_area / r0_areas["divorced"] - 1.0
                ),
                "widowed_area_target": widowed_targets[f"{omega:.2f}"][
                    "target_area"
                ],
                "widowed_area": widowed_area,
                "widowed_area_relative_residual": (
                    widowed_area
                    / widowed_targets[f"{omega:.2f}"]["target_area"]
                    - 1.0
                ),
                "positive_exposure_divorced_cells_rising": sum(
                    divorced_shifts[row["ysd_band_index"]] > 0.0
                    for row in positive_divorced_cells
                ),
                "positive_exposure_divorced_cells_falling": sum(
                    divorced_shifts[row["ysd_band_index"]] < 0.0
                    for row in positive_divorced_cells
                ),
                "widowed_working_age_cell_logit_shifts": [
                    omega
                    for _age_band in WORKING_AGE_BANDS
                    for _ysd_band in range(len(rem.YSD_BANDS))
                    for _sex in SEXES
                ],
                "widowed_table_construction_exact": (
                    widowed_table_construction_exact
                ),
                "probability_min": min(table.values()),
                "probability_max": max(table.values()),
                "raw_age_outside_18_64_exact_R0": True,
            },
        }
    return {
        "boundary": boundary,
        "truncated": truncated,
        "components": components,
        "baseline": baseline,
        "laws": laws,
        "public": {
            "fit_max_year": observed_validation["fit_max_year"],
            "fit_person_year_rows": len(truncated.panel.person_years),
            "fit_event_rows": len(truncated.panel.events),
            "dissolved_rows": len(dissolved),
            "remarriage_events": len(remarriages),
            "wbar": mean_weight,
            "incumbent_table_sha256": observed_validation[
                "incumbent_table_sha256"
            ],
            "fit_support_by_origin_ysd": support,
            "fit_exposure_center": center,
            "centered_contrast": centered,
            "reference_spells": observed_validation["reference_spells"],
            "reference_exclusion_category_hashes": {
                origin: {
                    key: value
                    for key, value in references[origin]["audit"].items()
                    if key.endswith("checksum_sha256")
                }
                for origin in ORIGINS
            },
            "divorced_calibration": calibration,
            "widowed_targets": widowed_targets,
            "support_struck_named_laws": [],
            "validation_match": True,
            "field_boundary_audit": field_boundary_audit,
            "laws": {law: record["public"] for law, record in laws.items()},
        },
    }


def _prepare_fits(
    context: ft_registry.FitContext,
    config: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    boundaries = config["protocol"]["boundaries"]
    prepared = {
        str(boundary): _fit_boundary(
            context, config, validation, int(boundary)
        )
        for boundary in boundaries
    }
    return prepared


def _ratio_record(
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


def _rate_record(risk: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    exposure = float(risk["weight"].sum())
    numerator = float(events["weight"].sum())
    return {
        "risk_rows": len(risk),
        "event_rows": len(events),
        "exposure": exposure,
        "numerator": numerator,
        "rate": numerator / exposure if exposure > 0.0 else None,
    }


def _publication_group_key(origin: str, age_band: int, ysd_band: int) -> str:
    age = rem.AGE_BANDS[age_band]
    ysd = rem.YSD_BANDS[ysd_band]
    return f"{origin}|age_{age[0]}_{age[1]}|ysd_{ysd[0]}_{ysd[1]}"


def _rate_summary(
    events: pd.DataFrame,
    person_years: pd.DataFrame,
) -> dict[str, Any]:
    risk = person_years[
        person_years["marital_state"].isin(ORIGINS)
        & person_years["years_since_dissolution"].notna()
    ].copy()
    remarriages = events[events["transition"].eq("remarriage")].copy()
    pooled = _rate_record(risk, remarriages)
    if pooled["exposure"] <= 0.0:
        raise ProtocolAbort("pooled dissolved exposure is non-positive")
    origin_records = {
        origin: _rate_record(
            risk[risk["marital_state"].eq(origin)],
            remarriages[remarriages["origin"].eq(origin)],
        )
        for origin in ORIGINS
    }
    for frame in (risk, remarriages):
        if len(frame):
            frame["age_band_index"] = _age_band(
                frame["age"].to_numpy(dtype=np.float64)
            )
            frame["ysd_band_index"] = _ysd_band(
                frame["years_since_dissolution"].to_numpy(dtype=np.int64)
            )
    groups: list[dict[str, Any]] = []
    for origin in ORIGINS:
        for age_band in WORKING_AGE_BANDS:
            for ysd_band in range(len(rem.YSD_BANDS)):
                group_risk = risk[
                    risk["marital_state"].eq(origin)
                    & risk["age_band_index"].eq(age_band)
                    & risk["ysd_band_index"].eq(ysd_band)
                ]
                group_events = remarriages[
                    remarriages["origin"].eq(origin)
                    & remarriages["age_band_index"].eq(age_band)
                    & remarriages["ysd_band_index"].eq(ysd_band)
                ]
                groups.append(
                    {
                        "group": _publication_group_key(
                            origin, age_band, ysd_band
                        ),
                        "origin": origin,
                        "age_band": list(rem.AGE_BANDS[age_band]),
                        "ysd_band": list(rem.YSD_BANDS[ysd_band]),
                        **_rate_record(group_risk, group_events),
                    }
                )
    return {"pooled": pooled, "origin": origin_records, "groups": groups}


def _summary_with_truth(
    projected: dict[str, Any], truth: dict[str, Any]
) -> dict[str, Any]:
    truth_groups = {record["group"]: record for record in truth["groups"]}
    projected_groups = []
    for record in projected["groups"]:
        truth_record = truth_groups[record["group"]]
        projected_groups.append(
            {**record, **_ratio_record(record, truth_record)}
        )
    return {
        "pooled": {
            **projected["pooled"],
            **_ratio_record(projected["pooled"], truth["pooled"]),
        },
        "origin": {
            origin: {
                **projected["origin"][origin],
                **_ratio_record(
                    projected["origin"][origin], truth["origin"][origin]
                ),
            }
            for origin in ORIGINS
        },
        "groups": projected_groups,
    }


def _direct_standardization(
    table: dict[tuple[int, int, str, str], float],
    baseline: dict[tuple[int, int, str, str], float],
    truth_events: pd.DataFrame,
    truth_person_years: pd.DataFrame,
) -> dict[str, Any]:
    risk = truth_person_years[
        truth_person_years["marital_state"].isin(ORIGINS)
        & truth_person_years["years_since_dissolution"].notna()
    ].copy()
    if not (risk["weight"] > 0).all():
        raise ProtocolAbort("direct risk contains non-positive weight")
    probability = _lookup_probabilities(
        risk, table, baseline, origin_column="marital_state"
    )
    if (
        not np.isfinite(probability).all()
        or not ((probability > 0.0) & (probability < 1.0)).all()
    ):
        raise ProtocolAbort("direct probability lies outside (0,1)")
    event_frame = truth_events[
        truth_events["transition"].eq("remarriage")
    ].copy()
    if (
        not event_frame["origin"].isin(ORIGINS).all()
        or not event_frame["years_since_dissolution"].notna().all()
    ):
        raise ProtocolAbort("truth remarriage event labels are incomplete")
    event_keys = {
        (int(person), int(year))
        for person, year in zip(
            event_frame["person_id"], event_frame["year"], strict=True
        )
    }
    if len(event_keys) != len(event_frame):
        raise ProtocolAbort("duplicate truth remarriage person-year")
    risk_keys = {
        (int(person), int(year))
        for person, year in zip(risk["person_id"], risk["year"], strict=True)
    }
    unmatched_keys = event_keys - risk_keys
    if unmatched_keys:
        event_index = event_frame.set_index(["person_id", "year"])
        unmatched = event_index.loc[sorted(unmatched_keys)].reset_index()
        invalid = unmatched[
            unmatched["years_since_dissolution"].isna()
            | unmatched["years_since_dissolution"].ne(0)
        ]
        if len(invalid):
            raise ProtocolAbort("unmatched nonzero-YSD remarriage event")
    else:
        unmatched = event_frame.iloc[0:0].copy()
    matchable_events = event_frame[
        [
            (int(person), int(year)) in risk_keys
            for person, year in zip(
                event_frame["person_id"],
                event_frame["year"],
                strict=True,
            )
        ]
    ]
    if len(matchable_events):
        matched = matchable_events[
            [
                "person_id",
                "year",
                "origin",
                "years_since_dissolution",
            ]
        ].merge(
            risk[
                [
                    "person_id",
                    "year",
                    "marital_state",
                    "years_since_dissolution",
                ]
            ],
            on=["person_id", "year"],
            how="left",
            validate="one_to_one",
            suffixes=("_event", "_risk"),
        )
        origin_matches = matched["origin"].eq(matched["marital_state"])
        event_ysd = pd.to_numeric(
            matched["years_since_dissolution_event"], errors="coerce"
        )
        risk_ysd = pd.to_numeric(
            matched["years_since_dissolution_risk"], errors="coerce"
        )
        if not origin_matches.all() or not event_ysd.eq(risk_ysd).all():
            raise ProtocolAbort(
                "truth remarriage event origin/YSD differs from risk row"
            )
    event = np.asarray(
        [
            int((int(person), int(year)) in event_keys)
            for person, year in zip(
                risk["person_id"], risk["year"], strict=True
            )
        ],
        dtype=np.float64,
    )
    risk["_probability"] = probability
    risk["_event"] = event
    risk["age_band_index"] = _age_band(risk["age"].to_numpy(dtype=np.float64))
    risk["ysd_band_index"] = _ysd_band(
        risk["years_since_dissolution"].to_numpy(dtype=np.int64)
    )
    if len(unmatched):
        unmatched["age_band_index"] = _age_band(
            unmatched["age"].to_numpy(dtype=np.float64)
        )
        unmatched["ysd_band_index"] = _ysd_band(
            unmatched["years_since_dissolution"].to_numpy(dtype=np.int64)
        )

    def record(
        selected_risk: pd.DataFrame, selected_events: pd.DataFrame
    ) -> dict[str, Any]:
        selected_weight = selected_risk["weight"].to_numpy(dtype=np.float64)
        selected_probability = selected_risk["_probability"].to_numpy(
            dtype=np.float64
        )
        selected_event = selected_risk["_event"].to_numpy(dtype=np.float64)
        exposure = float(selected_weight.sum())
        row_expected = float(np.sum(selected_weight * selected_probability))
        actual_numerator = float(selected_events["weight"].sum())
        selected_event_keys = set(
            zip(
                selected_events["person_id"].astype(int),
                selected_events["year"].astype(int),
                strict=True,
            )
        )
        selected_risk_keys = set(
            zip(
                selected_risk["person_id"].astype(int),
                selected_risk["year"].astype(int),
                strict=True,
            )
        )
        selected_unmatched = selected_events[
            [
                (int(person), int(year)) not in selected_risk_keys
                for person, year in zip(
                    selected_events["person_id"],
                    selected_events["year"],
                    strict=True,
                )
            ]
        ]
        unmatched_weight = float(selected_unmatched["weight"].sum())
        matchable_numerator = float(np.sum(selected_weight * selected_event))
        if not math.isclose(
            matchable_numerator + unmatched_weight,
            actual_numerator,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ProtocolAbort(
                "direct actual numerator differs from matchable plus "
                "same-year numerator"
            )
        matchable_rows = sum(
            key in selected_risk_keys for key in selected_event_keys
        )
        deviance_numerator = float(
            -2.0
            * np.sum(
                selected_weight
                * (
                    selected_event * np.log(selected_probability)
                    + (1.0 - selected_event) * np.log1p(-selected_probability)
                )
            )
        )
        return {
            "risk_rows": len(selected_risk),
            "event_rows": len(selected_events),
            "matchable_positive_weight_event_rows": int(
                selected_risk.loc[
                    selected_risk["_event"].eq(1)
                    & selected_risk["weight"].gt(0)
                ].shape[0]
            ),
            "matchable_event_rows": int(matchable_rows),
            "unmatched_same_year_event_rows": len(selected_unmatched),
            "unmatched_same_year_event_weight": unmatched_weight,
            "exposure": exposure,
            "deviance_exposure": exposure,
            "actual_numerator": actual_numerator,
            "matchable_numerator": matchable_numerator,
            "actual_rate": (
                actual_numerator / exposure if exposure > 0.0 else None
            ),
            "row_expected_numerator": row_expected,
            "expected_numerator": row_expected,
            "expected_rate": (
                row_expected / exposure if exposure > 0.0 else None
            ),
            "qdir": row_expected / exposure if exposure > 0.0 else None,
            "weighted_deviance_numerator": deviance_numerator,
            "weighted_bernoulli_deviance": (
                deviance_numerator / exposure if exposure > 0.0 else None
            ),
        }

    pooled = record(risk, event_frame)
    origin_records = {
        origin: record(
            risk[risk["marital_state"].eq(origin)],
            event_frame[event_frame["origin"].eq(origin)],
        )
        for origin in ORIGINS
    }
    groups = []
    for origin in ORIGINS:
        for age_band in WORKING_AGE_BANDS:
            for ysd_band in range(len(rem.YSD_BANDS)):
                groups.append(
                    {
                        "group": _publication_group_key(
                            origin, age_band, ysd_band
                        ),
                        "origin": origin,
                        "age_band": list(rem.AGE_BANDS[age_band]),
                        "ysd_band": list(rem.YSD_BANDS[ysd_band]),
                        **record(
                            risk[
                                risk["marital_state"].eq(origin)
                                & risk["age_band_index"].eq(age_band)
                                & risk["ysd_band_index"].eq(ysd_band)
                            ],
                            event_frame[
                                event_frame["origin"].eq(origin)
                                & event_frame["age"].between(
                                    *rem.AGE_BANDS[age_band]
                                )
                                & event_frame[
                                    "years_since_dissolution"
                                ].between(*rem.YSD_BANDS[ysd_band])
                            ],
                        ),
                    }
                )
    return {"pooled": pooled, "origin": origin_records, "groups": groups}


def _transition_uniform_checksum(
    seed_panel: transitions.MaritalPanel,
    seed: int,
    n_periods: int,
) -> str:
    registry = ProjectionRNGRegistry(
        draw_index=seed - 5200, n_periods=n_periods
    )
    generator = registry.generator(0, ProjectionModule.MARITAL_CORE)
    attrs = seed_panel.attrs.sort_values("person_id").reset_index(drop=True)
    start = attrs["start_exposure_year"].to_numpy(dtype=np.int64)
    end = attrs["censor_year"].to_numpy(dtype=np.int64)
    digest = hashlib.sha256()
    for year in range(int(start.min()), int(end.max()) + 1):
        count = int(((start <= year) & (year <= end)).sum())
        digest.update(generator.random(count).tobytes())
    return digest.hexdigest()


def _project_with_raw_age_guard(
    seed_panel: transitions.MaritalPanel,
    valid_ids: set[int],
    components: FittedFamilyTransitions,
    baseline: dict[tuple[int, int, str, str], float],
    registry: ProjectionRNGRegistry,
) -> tuple[transitions.MaritalPanel, pd.DataFrame, dict[str, int]]:
    original = marital_engine.remarriage_probabilities
    original_gap_draw = marital_engine.draw_spousal_gaps
    baseline_lookup = rem.build_remarriage_lookup(baseline)
    gap_consumption = {"calls": 0, "draws": 0}

    def guarded(
        age: np.ndarray,
        years_since_dissolution: np.ndarray,
        origin_state: np.ndarray,
        is_male: np.ndarray,
        lookup: np.ndarray,
    ) -> np.ndarray:
        candidate_probability = original(
            age, years_since_dissolution, origin_state, is_male, lookup
        )
        baseline_probability = original(
            age,
            years_since_dissolution,
            origin_state,
            is_male,
            baseline_lookup,
        )
        in_domain = (age >= 18.0) & (age <= 64.0)
        output = np.where(
            in_domain, candidate_probability, baseline_probability
        )
        if not np.array_equal(
            output[~in_domain], baseline_probability[~in_domain]
        ):
            raise AssertionError("projection raw-age guard failed")
        return output

    def counted_gap_draw(
        rng: np.random.Generator,
        indices: np.ndarray,
        marriage_age: np.ndarray,
        is_male: np.ndarray,
        distributions: dict[str, dict[int, np.ndarray]],
    ) -> np.ndarray:
        gap_consumption["calls"] += 1
        gap_consumption["draws"] += int(indices.size)
        return original_gap_draw(
            rng, indices, marriage_age, is_male, distributions
        )

    marital_engine.remarriage_probabilities = guarded
    marital_engine.draw_spousal_gaps = counted_gap_draw
    try:
        panel, births = marital_engine._simulate_candidate16_with_generators(
            seed_panel,
            valid_ids,
            components,
            registry.generator(0, ProjectionModule.MARITAL_CORE),
            registry.child_generator(0, ProjectionModule.MARITAL_CORE, 1),
        )
        return panel, births, gap_consumption
    finally:
        marital_engine.remarriage_probabilities = original
        marital_engine.draw_spousal_gaps = original_gap_draw


def _support_key_checksum(frame: pd.DataFrame) -> str:
    keys = sorted(
        (int(person), int(year))
        for person, year in zip(frame["person_id"], frame["year"], strict=True)
    )
    return _sha256_bytes(
        "\n".join(f"{person}|{year}" for person, year in keys).encode()
    )


def _projected_panel_checksums(
    panel: transitions.MaritalPanel, births: pd.DataFrame
) -> dict[str, Any]:
    person_year_columns = tuple(
        column
        for column in (
            "person_id",
            "year",
            "marital_state",
            "years_since_dissolution",
            "marriage_duration",
            "age",
            "sex",
            "weight",
        )
        if column in panel.person_years
    )
    event_columns = tuple(
        column
        for column in (
            "person_id",
            "year",
            "transition",
            "origin",
            "years_since_dissolution",
            "age",
            "sex",
            "weight",
        )
        if column in panel.events
    )
    birth_columns = tuple(births.columns)
    return {
        "person_year_rows": len(panel.person_years),
        "event_rows": len(panel.events),
        "birth_rows": len(births),
        "person_years_sha256": _frame_checksum(
            panel.person_years, person_year_columns
        ),
        "events_sha256": _frame_checksum(panel.events, event_columns),
        "births_sha256": _frame_checksum(births, birth_columns),
    }


def _panels_exactly_equal(
    left: transitions.MaritalPanel,
    right: transitions.MaritalPanel,
    left_births: pd.DataFrame,
    right_births: pd.DataFrame,
) -> bool:
    return bool(
        left.person_years.equals(right.person_years)
        and left.events.equals(right.events)
        and left.attrs.equals(right.attrs)
        and left_births.equals(right_births)
    )


def _mean_projection_record(
    records: list[dict[str, Any]], truth: dict[str, Any]
) -> dict[str, Any]:
    if not records:
        raise ProtocolAbort("projection aggregate received no seed records")
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
    return {
        "seed_count": len(records),
        "mean_risk_rows": float(
            np.mean([record["risk_rows"] for record in records])
        ),
        "mean_event_rows": float(
            np.mean([record["event_rows"] for record in records])
        ),
        **projected,
        **_ratio_record(projected, truth),
    }


def _aggregate_projection(
    per_seed: list[dict[str, Any]],
    truth: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    wanted = set(seeds)
    rows = [row for row in per_seed if int(row["seed"]) in wanted]
    if [int(row["seed"]) for row in rows] != list(seeds):
        raise ProtocolAbort("projection aggregate seed order drifted")
    pooled = _mean_projection_record(
        [row["pooled"] for row in rows], truth["pooled"]
    )
    origin = {
        value: _mean_projection_record(
            [row["origin"][value] for row in rows],
            truth["origin"][value],
        )
        for value in ORIGINS
    }
    truth_groups = {row["group"]: row for row in truth["groups"]}
    group_names = [row["group"] for row in rows[0]["publication_groups"]]
    groups: list[dict[str, Any]] = []
    for index, group_name in enumerate(group_names):
        group_records = [row["publication_groups"][index] for row in rows]
        if any(record["group"] != group_name for record in group_records):
            raise ProtocolAbort("publication group order drifted across seeds")
        first = group_records[0]
        groups.append(
            {
                "group": group_name,
                "origin": first["origin"],
                "age_band": first["age_band"],
                "ysd_band": first["ysd_band"],
                **_mean_projection_record(
                    group_records, truth_groups[group_name]
                ),
            }
        )
    return {"pooled": pooled, "origin": origin, "groups": groups}


def _validate_projected_event_semantics(
    events: pd.DataFrame, person_years: pd.DataFrame
) -> dict[str, Any]:
    remarriages = events[events["transition"].eq("remarriage")]
    origin_valid = remarriages["origin"].isin(ORIGINS).all()
    ysd_defined = remarriages["years_since_dissolution"].notna().all()
    unique_person_year = not remarriages.duplicated(
        ["person_id", "year"]
    ).any()
    if not (origin_valid and ysd_defined and unique_person_year):
        raise ProtocolAbort("projected remarriage event semantics drifted")
    risk = person_years[
        person_years["marital_state"].isin(ORIGINS)
        & person_years["years_since_dissolution"].notna()
    ]
    matched = remarriages.merge(
        risk[
            [
                "person_id",
                "year",
                "marital_state",
                "years_since_dissolution",
            ]
        ],
        on=["person_id", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("_event", "_risk"),
        indicator=True,
    )
    matchable = matched[matched["_merge"].eq("both")]
    unmatched = matched[matched["_merge"].eq("left_only")]
    origin_matches = matchable["origin"].eq(matchable["marital_state"])
    event_ysd = pd.to_numeric(
        matchable["years_since_dissolution_event"], errors="coerce"
    )
    risk_ysd = pd.to_numeric(
        matchable["years_since_dissolution_risk"], errors="coerce"
    )
    ysd_matches = event_ysd.eq(risk_ysd)
    unmatched_same_year = pd.to_numeric(
        unmatched["years_since_dissolution_event"], errors="coerce"
    ).eq(0)
    label_conformance = bool(
        origin_matches.all()
        and ysd_matches.all()
        and unmatched_same_year.all()
    )
    if not (
        origin_valid
        and ysd_defined
        and unique_person_year
        and label_conformance
    ):
        raise ProtocolAbort("projected remarriage event semantics drifted")
    return {
        "remarriage_event_rows": len(remarriages),
        "origins_valid": bool(origin_valid),
        "years_since_dissolution_defined": bool(ysd_defined),
        "unique_person_year": bool(unique_person_year),
        "matchable_origin_and_ysd_exact_risk_row": label_conformance,
        "unmatched_same_year_event_rows": len(unmatched),
    }


def _expected_pseudo_record(
    *,
    boundary: int,
    anchor: pd.DataFrame,
    valid_ids: set[int],
    carrier_count: int,
    truth_events: pd.DataFrame,
    truth_person_years: pd.DataFrame,
    truth: dict[str, Any],
) -> dict[str, Any]:
    same_year = truth_events[
        truth_events["transition"].eq("remarriage")
        & truth_events["years_since_dissolution"].eq(0).fillna(False)
    ]
    return {
        "anchor_households_before_marital_intersection": int(
            anchor["household_id"].nunique()
        ),
        "anchor_persons_before_marital_intersection": int(
            anchor["person_id"].nunique()
        ),
        "projected_persons": len(valid_ids),
        "entry_dissolved_carriers": carrier_count,
        "truth_support_rows": len(truth_person_years),
        "support_checksum": _frame_checksum(
            truth_person_years,
            (
                "person_id",
                "year",
                "required_interview_year",
                "age",
                "sex",
                "weight",
            ),
        ),
        "truth_required_interview_year_max": _max_year(
            truth_person_years, "required_interview_year"
        ),
        "truth_same_year_ysd0_events": len(same_year),
        "truth_same_year_ysd0_event_weight": float(same_year["weight"].sum()),
        "truth": truth["pooled"],
        "truth_origin": truth["origin"],
        "boundary": boundary,
    }


def _evaluate_boundary(
    full_context: ft_registry.FitContext,
    prepared: dict[str, Any],
    config: dict[str, Any],
    boundary: int,
) -> dict[str, Any]:
    years = tuple(config["protocol"]["evaluation_years"][str(boundary)])
    anchor_waves = (boundary + 1, boundary + 3)
    if (
        list(anchor_waves)
        != config["pseudo_holdouts"]["boundaries"][str(boundary)][
            "anchor_waves"
        ]
    ):
        raise ProtocolAbort(f"anchor-wave lock drifted at {boundary}")
    if max(years) > 2013:
        raise ProtocolAbort("forbidden calendar-2014 flow entered evaluation")
    anchor, present = round1._pseudo_anchor(
        full_context.demographic_panel, anchor_waves
    )
    _assert_at_most(anchor, "anchor_wave", 2013, "pseudo_anchor")
    seed_panel, valid_ids, carrier_count = round1._seed_panel(
        full_context.panel, anchor, max(years)
    )
    _assert_at_most(seed_panel.person_years, "year", max(years), "seed_panel")
    truth_events, truth_person_years = round1._weighted_support(
        full_context.panel, anchor, present, years, valid_ids
    )
    for label, frame in (
        ("truth_events", truth_events),
        ("truth_person_years", truth_person_years),
    ):
        _assert_at_most(frame, "year", 2013, label)
        _assert_at_most(frame, "required_interview_year", 2014, label)
    truth = _rate_summary(truth_events, truth_person_years)
    expected_record = _expected_pseudo_record(
        boundary=boundary,
        anchor=anchor,
        valid_ids=valid_ids,
        carrier_count=carrier_count,
        truth_events=truth_events,
        truth_person_years=truth_person_years,
        truth=truth,
    )
    expected_locked = config["expected_pseudo_holdouts"][str(boundary)]
    _assert_expected_subset(
        expected_record,
        expected_locked,
        f"expected_pseudo_holdouts.{boundary}",
    )

    truth_support_keys = set(
        zip(
            truth_person_years["person_id"].astype(int),
            truth_person_years["year"].astype(int),
            strict=True,
        )
    )
    truth_key_checksum = _support_key_checksum(truth_person_years)
    pseudo_input_hashes = {
        "anchor_sha256": _frame_checksum(
            anchor,
            ("person_id", "household_id", "weight", "anchor_wave"),
        ),
        "seed_attrs_sha256": _frame_checksum(
            seed_panel.attrs,
            tuple(
                column
                for column in (
                    "person_id",
                    "birth_year",
                    "sex",
                    "start_exposure_year",
                    "censor_year",
                    "weight",
                    "n_marriages",
                )
                if column in seed_panel.attrs
            ),
        ),
        "seed_entry_person_years_sha256": _frame_checksum(
            seed_panel.person_years,
            tuple(
                column
                for column in (
                    "person_id",
                    "year",
                    "marital_state",
                    "years_since_dissolution",
                    "marriage_duration",
                    "age",
                    "sex",
                    "weight",
                )
                if column in seed_panel.person_years
            ),
        ),
        "truth_support_key_sha256": truth_key_checksum,
        "truth_support_weighted_sha256": expected_record["support_checksum"],
    }
    n_periods = int(config["protocol"]["n_periods"][str(boundary)])
    seeds = tuple(int(seed) for seed in config["protocol"]["seeds"])
    uniform_by_seed = {
        seed: _transition_uniform_checksum(seed_panel, seed, n_periods)
        for seed in seeds
    }
    laws: dict[str, Any] = {}
    r0_uniforms: dict[int, str] = {}
    r0_gap_consumption: dict[int, dict[str, int]] = {}
    r0_direct: dict[str, Any] | None = None
    for law in config["protocol"]["law_order"]:
        _progress(
            f"candidate outcome: boundary {boundary}, {law}, "
            f"{len(seeds)} CRN draws"
        )
        law_fit = prepared["laws"][law]
        table = law_fit["table"]
        direct = _direct_standardization(
            table,
            prepared["baseline"],
            truth_events,
            truth_person_years,
        )
        if law == "R0":
            r0_direct = direct
        assert r0_direct is not None
        r0_qdir = float(r0_direct["origin"]["widowed"]["qdir"])
        candidate_qdir = float(direct["origin"]["widowed"]["qdir"])
        if r0_qdir <= 0.0 or candidate_qdir <= 0.0:
            raise ProtocolAbort("widowed direct qdir is non-positive")
        g_widowed = math.log(candidate_qdir / r0_qdir)
        candidate_components: FittedFamilyTransitions = replace(
            prepared["components"], remarriage=table
        )
        unchanged_components = all(
            getattr(candidate_components, field.name)
            is getattr(prepared["components"], field.name)
            for field in fields(FittedFamilyTransitions)
            if field.name != "remarriage"
        )
        if not unchanged_components:
            raise ProtocolAbort("a non-remarriage component changed")
        per_seed: list[dict[str, Any]] = []
        for offset, seed in enumerate(seeds, start=1):
            registry = ProjectionRNGRegistry(
                draw_index=seed - int(config["rng"]["seed_root"]),
                n_periods=n_periods,
            )
            projected_panel, births, gap_consumption = (
                _project_with_raw_age_guard(
                    seed_panel,
                    valid_ids,
                    candidate_components,
                    prepared["baseline"],
                    registry,
                )
            )
            if law == "R0":
                r0_gap_consumption[seed] = gap_consumption
            gap_difference = {
                key: gap_consumption[key] - r0_gap_consumption[seed][key]
                for key in ("calls", "draws")
            }
            incumbent_exact: bool | None = None
            incumbent_gap_exact: bool | None = None
            if law == "R0":
                incumbent_registry = ProjectionRNGRegistry(
                    draw_index=seed - int(config["rng"]["seed_root"]),
                    n_periods=n_periods,
                )
                (
                    incumbent_panel,
                    incumbent_births,
                    incumbent_gap_consumption,
                ) = _project_with_raw_age_guard(
                    seed_panel,
                    valid_ids,
                    prepared["components"],
                    prepared["baseline"],
                    incumbent_registry,
                )
                incumbent_gap_exact = (
                    gap_consumption == incumbent_gap_consumption
                )
                incumbent_exact = _panels_exactly_equal(
                    projected_panel,
                    incumbent_panel,
                    births,
                    incumbent_births,
                )
                if not incumbent_exact or not incumbent_gap_exact:
                    raise ProtocolAbort(
                        f"R0 projection differs from incumbent at "
                        f"{boundary}/{seed}"
                    )
            carrier_rows = round1._assert_carrier_conformance(
                seed_panel, projected_panel
            )
            if carrier_rows != carrier_count:
                raise ProtocolAbort("entry carrier count changed")
            projected_events, projected_person_years = (
                round1._weighted_support(
                    projected_panel,
                    anchor,
                    present,
                    years,
                    valid_ids,
                )
            )
            for label, frame in (
                ("projected_events", projected_events),
                ("projected_person_years", projected_person_years),
            ):
                _assert_at_most(frame, "year", 2013, label)
                _assert_at_most(frame, "required_interview_year", 2014, label)
            projected_keys = set(
                zip(
                    projected_person_years["person_id"].astype(int),
                    projected_person_years["year"].astype(int),
                    strict=True,
                )
            )
            if projected_keys != truth_support_keys:
                raise ProtocolAbort(
                    f"support mismatch at {boundary}/{law}/{seed}"
                )
            projected = _summary_with_truth(
                _rate_summary(projected_events, projected_person_years),
                truth,
            )
            weighted_support_checksum = _frame_checksum(
                projected_person_years,
                (
                    "person_id",
                    "year",
                    "required_interview_year",
                    "age",
                    "sex",
                    "weight",
                ),
            )
            weighted_support_exact = (
                weighted_support_checksum
                == expected_record["support_checksum"]
            )
            if not weighted_support_exact:
                raise ProtocolAbort(
                    f"weighted support mismatch at {boundary}/{law}/{seed}"
                )
            uniform_checksum = uniform_by_seed[seed]
            if law == "R0":
                r0_uniforms[seed] = uniform_checksum
            uniform_unchanged = uniform_checksum == r0_uniforms[seed]
            if not uniform_unchanged:
                raise ProtocolAbort("transition uniform stream changed by law")
            per_seed.append(
                {
                    "seed": seed,
                    "pooled": projected["pooled"],
                    "origin": projected["origin"],
                    "publication_groups": projected["groups"],
                    "carrier_checks": {
                        "expected_entry_dissolved_carriers": carrier_count,
                        "verified_entry_dissolved_carriers": carrier_rows,
                        "exact": carrier_rows == carrier_count,
                    },
                    "support_checks": {
                        "exact_truth_keys": True,
                        "truth_key_count": len(truth_support_keys),
                        "projected_key_count": len(projected_keys),
                        "truth_key_sha256": truth_key_checksum,
                        "projected_key_sha256": _support_key_checksum(
                            projected_person_years
                        ),
                        "weighted_support_sha256": weighted_support_checksum,
                        "weighted_support_exact_truth": weighted_support_exact,
                        "event_semantics": (
                            _validate_projected_event_semantics(
                                projected_events, projected_person_years
                            )
                        ),
                    },
                    "uniform_checks": {
                        "draw_index": seed - int(config["rng"]["seed_root"]),
                        "n_periods": n_periods,
                        "transition_address": config["rng"][
                            "transition_uniform_address"
                        ],
                        "spouse_gap_address": config["rng"][
                            "spouse_gap_address"
                        ],
                        "transition_uniform_sha256": uniform_checksum,
                        "exact_R0_stream": uniform_unchanged,
                    },
                    "downstream": {
                        **_projected_panel_checksums(projected_panel, births),
                        "spouse_gap_consumption": gap_consumption,
                        "spouse_gap_consumption_difference_from_R0": (
                            gap_difference
                        ),
                        "R0_spouse_gap_consumption_exact_incumbent": (
                            incumbent_gap_exact
                        ),
                        "R0_projection_exact_incumbent": incumbent_exact,
                    },
                }
            )
            if offset % 10 == 0:
                _progress(
                    f"boundary {boundary}: {law} draw "
                    f"{offset}/{len(seeds)}"
                )
        mean = _aggregate_projection(per_seed, truth, seeds)
        blocks = {
            f"block_{index}": _aggregate_projection(
                per_seed,
                truth,
                tuple(int(seed) for seed in block),
            )
            for index, block in enumerate(
                config["protocol"]["blocks"], start=1
            )
        }
        laws[law] = {
            "construction": law_fit["public"],
            "direct": direct,
            "g_widowed_log_qdir_ratio": g_widowed,
            "per_seed": per_seed,
            "mean": mean,
            "blocks": blocks,
            "carrier_conformance_all_draws": all(
                row["carrier_checks"]["exact"] for row in per_seed
            ),
            "support_exact_all_draws": all(
                row["support_checks"]["exact_truth_keys"]
                and row["support_checks"]["weighted_support_exact_truth"]
                for row in per_seed
            ),
            "uniform_exact_all_draws": all(
                row["uniform_checks"]["exact_R0_stream"] for row in per_seed
            ),
            "non_remarriage_components_exact_R0": unchanged_components,
            "R0_projection_exact_incumbent_all_draws": (
                all(
                    row["downstream"]["R0_projection_exact_incumbent"]
                    for row in per_seed
                )
                if law == "R0"
                else None
            ),
            "R0_spouse_gap_consumption_exact_incumbent_all_draws": (
                all(
                    row["downstream"][
                        "R0_spouse_gap_consumption_exact_incumbent"
                    ]
                    for row in per_seed
                )
                if law == "R0"
                else None
            ),
        }
    return {
        "boundary": boundary,
        "anchor_waves": list(anchor_waves),
        "evaluation_years": list(years),
        **expected_record,
        "truth": truth,
        "truth_support_key_sha256": truth_key_checksum,
        "pseudo_input_hashes": pseudo_input_hashes,
        "fit": prepared["public"],
        "laws": laws,
    }


def _objective_for_seeds(
    boundaries: dict[str, Any],
    law: str,
    seeds: tuple[int, ...],
) -> float:
    terms: list[float] = []
    wanted = set(seeds)
    for boundary in ("2006", "2008", "2010"):
        record = boundaries[boundary]
        rows = [
            row
            for row in record["laws"][law]["per_seed"]
            if int(row["seed"]) in wanted
        ]
        if [int(row["seed"]) for row in rows] != list(seeds):
            raise ProtocolAbort("objective seed order drifted")
        mean_rate = float(
            np.mean(
                np.asarray(
                    [row["pooled"]["rate"] for row in rows],
                    dtype=np.float64,
                )
            )
        )
        truth_rate = float(record["truth"]["pooled"]["rate"])
        if mean_rate <= 0.0 or truth_rate <= 0.0:
            raise ProtocolAbort("objective log argument is non-positive")
        terms.append(math.log(mean_rate / truth_rate) ** 2)
    return float(np.mean(np.asarray(terms, dtype=np.float64)))


def _pooled_direct_deviance(
    boundaries: dict[str, Any], law: str, origin: str | None = None
) -> dict[str, float]:
    records = []
    for boundary in ("2006", "2008", "2010"):
        direct = boundaries[boundary]["laws"][law]["direct"]
        records.append(
            direct["pooled"] if origin is None else direct["origin"][origin]
        )
    numerator = float(
        sum(record["weighted_deviance_numerator"] for record in records)
    )
    exposure = float(sum(record["deviance_exposure"] for record in records))
    if exposure <= 0.0:
        raise ProtocolAbort("pooled direct deviance exposure is non-positive")
    return {
        "weighted_deviance_numerator": numerator,
        "deviance_exposure": exposure,
        "weighted_bernoulli_deviance": numerator / exposure,
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _rule_one_checks(
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
    for boundary in ("2006", "2008", "2010"):
        law_record = boundaries[boundary]["laws"][law]
        construction = law_record["construction"]
        boundary_metrics = metrics["boundary"][boundary]
        required_values.extend(
            [
                boundary_metrics["rate_error"],
                boundary_metrics["exposure_error"],
                boundary_metrics["working_age_widow_exposure_error"],
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
                law_record["non_remarriage_components_exact_R0"],
                construction["raw_age_outside_18_64_exact_R0"],
                boundaries[boundary]["fit"]["validation_match"],
            ]
        )
        if law == "R0":
            conformance.extend(
                [
                    law_record["R0_projection_exact_incumbent_all_draws"],
                    law_record[
                        "R0_spouse_gap_consumption_exact_incumbent_all_draws"
                    ],
                ]
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
                    row["support_checks"]["weighted_support_exact_truth"],
                    row["uniform_checks"]["exact_R0_stream"],
                    row["support_checks"]["event_semantics"]["origins_valid"],
                    row["support_checks"]["event_semantics"][
                        "years_since_dissolution_defined"
                    ],
                    row["support_checks"]["event_semantics"][
                        "unique_person_year"
                    ],
                    row["support_checks"]["event_semantics"][
                        "matchable_origin_and_ysd_exact_risk_row"
                    ],
                ]
            )
    finite_required = all(_finite(value) for value in required_values)
    checks = {
        "all_required_selector_values_finite": finite_required,
        "all_required_log_arguments_positive": all(positive_logs),
        "support_carrier_event_rng_conformance": all(conformance),
        "fit_side_validation_match": all(
            boundaries[boundary]["fit"]["validation_match"]
            for boundary in ("2006", "2008", "2010")
        ),
    }
    return {"pass": all(checks.values()), "checks": checks}


def _selector_metrics(
    boundaries: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in config["protocol"]["seeds"])
    metrics: dict[str, Any] = {}
    for law in config["protocol"]["law_order"]:
        full_j = _objective_for_seeds(boundaries, law, seeds)
        block_j = {
            f"block_{index}": _objective_for_seeds(
                boundaries,
                law,
                tuple(int(seed) for seed in block),
            )
            for index, block in enumerate(
                config["protocol"]["blocks"], start=1
            )
        }
        delete_one: list[dict[str, Any]] = []
        for deleted in seeds:
            retained = tuple(seed for seed in seeds if seed != deleted)
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
                    [row["J_delete_one"] for row in delete_one],
                    dtype=np.float64,
                )
            )
        )
        se = math.sqrt(
            (39.0 / 40.0)
            * sum(
                (row["J_delete_one"] - jackknife_mean) ** 2
                for row in delete_one
            )
        )
        boundary_metrics: dict[str, Any] = {}
        for boundary in ("2006", "2008", "2010"):
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
                "SE_J": se,
                "same_seed_deleted_across_all_boundaries": True,
                "reselection_inside_replicates": False,
            },
            "pooled_direct": _pooled_direct_deviance(boundaries, law),
            "origin_pooled_direct": {
                origin: _pooled_direct_deviance(boundaries, law, origin)
                for origin in ORIGINS
            },
            "boundary": boundary_metrics,
        }
    return metrics


def _construction_rule(
    boundaries: dict[str, Any],
    config: dict[str, Any],
    law: str,
) -> dict[str, Any]:
    area_tolerance = float(config["construction"]["area_tolerance"])
    comparison_tolerance = float(config["selector"]["comparison_tolerance"])
    budget = float(config["construction"]["widowed_log_rate_budget"])
    k, omega = _law_components(config, law)
    by_boundary: dict[str, Any] = {}
    for boundary in ("2006", "2008", "2010"):
        record = boundaries[boundary]["laws"][law]
        construction = record["construction"]
        g_value = record["g_widowed_log_qdir_ratio"]
        checks = {
            "divorced_area_within_tolerance": abs(
                construction["divorced_area_relative_residual"]
            )
            <= area_tolerance,
            "widowed_area_within_tolerance": abs(
                construction["widowed_area_relative_residual"]
            )
            <= area_tolerance,
            "divorced_delta_exact_target": abs(
                construction["Delta_divorced"] + k
            )
            <= comparison_tolerance,
            "widowed_delta_exact_target": abs(
                construction["Delta_widowed"] - omega
            )
            <= comparison_tolerance,
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
            "g_nonnegative": g_value >= -comparison_tolerance,
            "g_no_greater_than_omega": (
                g_value <= omega + comparison_tolerance
            ),
            "omega_nonnegative": omega >= -comparison_tolerance,
            "omega_within_budget": omega <= budget + comparison_tolerance,
        }
        by_boundary[boundary] = {
            "pass": all(checks.values()),
            "checks": checks,
            "g": g_value,
            "omega": omega,
            "B_W": budget,
        }
    return {
        "pass": all(record["pass"] for record in by_boundary.values()),
        "boundaries": by_boundary,
    }


def _comparison_rule(
    candidate: list[float],
    baseline: list[float],
    *,
    strict_minimum: int,
    tolerance: float,
) -> dict[str, Any]:
    strict = [
        _strictly_better(value, reference, tolerance)
        for value, reference in zip(candidate, baseline, strict=True)
    ]
    no_worse = [
        _no_worse(value, reference, tolerance)
        for value, reference in zip(candidate, baseline, strict=True)
    ]
    return {
        "pass": sum(strict) >= strict_minimum and all(no_worse),
        "strict_improvements": strict,
        "no_worse": no_worse,
        "strict_improvement_count": sum(strict),
        "required_strict_improvements": strict_minimum,
    }


def _select_law(
    boundaries: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    tolerance = float(config["selector"]["comparison_tolerance"])
    laws = config["protocol"]["law_order"]
    metrics = _selector_metrics(boundaries, config)
    eligibility: dict[str, Any] = {}
    r0_rule_one = _rule_one_checks(boundaries, "R0", metrics["R0"])
    r0_construction = all(
        boundaries[boundary]["laws"]["R0"]["construction"]["table_sha256"]
        == boundaries[boundary]["fit"]["incumbent_table_sha256"]
        and boundaries[boundary]["laws"]["R0"][
            "R0_projection_exact_incumbent_all_draws"
        ]
        for boundary in ("2006", "2008", "2010")
    )
    if not r0_rule_one["pass"] or not r0_construction:
        raise ProtocolAbort("R0 failed baseline conformance")
    eligibility["R0"] = {
        "eligible_as_baseline": True,
        "rule_1": r0_rule_one,
        "R0_bit_equivalence": r0_construction,
    }

    eligible_nonzero: list[str] = []
    for law in laws[1:]:
        rule_1 = _rule_one_checks(boundaries, law, metrics[law])
        rule_2 = _construction_rule(boundaries, config, law)
        rule_3_checks = {
            "full_J_strictly_better_R0": _strictly_better(
                metrics[law]["J"], metrics["R0"]["J"], tolerance
            ),
            "block_1_J_strictly_better_R0": _strictly_better(
                metrics[law]["block_J"]["block_1"],
                metrics["R0"]["block_J"]["block_1"],
                tolerance,
            ),
            "block_2_J_strictly_better_R0": _strictly_better(
                metrics[law]["block_J"]["block_2"],
                metrics["R0"]["block_J"]["block_2"],
                tolerance,
            ),
        }
        rule_3 = {
            "pass": all(rule_3_checks.values()),
            "checks": rule_3_checks,
        }
        rate_candidate = [
            metrics[law]["boundary"][boundary]["rate_error"]
            for boundary in ("2006", "2008", "2010")
        ]
        rate_baseline = [
            metrics["R0"]["boundary"][boundary]["rate_error"]
            for boundary in ("2006", "2008", "2010")
        ]
        rule_4 = _comparison_rule(
            rate_candidate,
            rate_baseline,
            strict_minimum=2,
            tolerance=tolerance,
        )
        exposure_candidate = [
            metrics[law]["boundary"][boundary]["exposure_error"]
            for boundary in ("2006", "2008", "2010")
        ]
        exposure_baseline = [
            metrics["R0"]["boundary"][boundary]["exposure_error"]
            for boundary in ("2006", "2008", "2010")
        ]
        exposure_comparison = _comparison_rule(
            exposure_candidate,
            exposure_baseline,
            strict_minimum=0,
            tolerance=tolerance,
        )
        rule_5 = {
            **exposure_comparison,
            "pass": all(exposure_comparison["no_worse"]),
        }
        direct_candidate = [
            metrics[law]["boundary"][boundary]["pooled_direct_deviance"]
            for boundary in ("2006", "2008", "2010")
        ]
        direct_baseline = [
            metrics["R0"]["boundary"][boundary]["pooled_direct_deviance"]
            for boundary in ("2006", "2008", "2010")
        ]
        direct_boundaries = _comparison_rule(
            direct_candidate,
            direct_baseline,
            strict_minimum=2,
            tolerance=tolerance,
        )
        pooled_direct_strict = _strictly_better(
            metrics[law]["pooled_direct"]["weighted_bernoulli_deviance"],
            metrics["R0"]["pooled_direct"]["weighted_bernoulli_deviance"],
            tolerance,
        )
        rule_6 = {
            "pass": pooled_direct_strict and direct_boundaries["pass"],
            "pooled_direct_strictly_better_R0": pooled_direct_strict,
            "boundary_comparison": direct_boundaries,
        }
        divorced_candidate = [
            metrics[law]["boundary"][boundary]["divorced_direct_deviance"]
            for boundary in ("2006", "2008", "2010")
        ]
        divorced_baseline = [
            metrics["R0"]["boundary"][boundary]["divorced_direct_deviance"]
            for boundary in ("2006", "2008", "2010")
        ]
        divorced_check = _comparison_rule(
            divorced_candidate,
            divorced_baseline,
            strict_minimum=2,
            tolerance=tolerance,
        )
        _k, omega = _law_components(config, law)
        budget = float(config["construction"]["widowed_log_rate_budget"])
        widowed_branches: dict[str, Any] = {}
        for boundary in ("2006", "2008", "2010"):
            candidate_boundary = metrics[law]["boundary"][boundary]
            baseline_boundary = metrics["R0"]["boundary"][boundary]
            event_rows = candidate_boundary[
                "widowed_matchable_positive_weight_event_rows"
            ]
            branch_law_independent = (
                event_rows
                == baseline_boundary[
                    "widowed_matchable_positive_weight_event_rows"
                ]
            )
            g_value = candidate_boundary["g_widowed_log_qdir_ratio"]
            if event_rows > 0:
                branch_pass = bool(
                    branch_law_independent
                    and _no_worse(
                        candidate_boundary["widowed_direct_deviance"],
                        baseline_boundary["widowed_direct_deviance"],
                        tolerance,
                    )
                )
                branch = "positive_event_deviance_compared"
                details = {"widowed_deviance_no_worse_R0": branch_pass}
            else:
                branch_pass = bool(
                    branch_law_independent
                    and candidate_boundary["widowed_direct_risk_exposure"]
                    > 0.0
                    and g_value >= -tolerance
                    and g_value <= omega + tolerance
                    and omega >= -tolerance
                    and omega <= budget + tolerance
                )
                branch = "zero_event_publish_not_compare_g_guard"
                details = {
                    "positive_risk_exposure": candidate_boundary[
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
            for boundary in ("2006", "2008", "2010")
        ]
        widow_exposure_baseline = [
            metrics["R0"]["boundary"][boundary][
                "working_age_widow_exposure_error"
            ]
            for boundary in ("2006", "2008", "2010")
        ]
        widow_exposure = _comparison_rule(
            widow_exposure_candidate,
            widow_exposure_baseline,
            strict_minimum=0,
            tolerance=tolerance,
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
    reason: str
    if not eligible_nonzero:
        selected = "R0"
        reason = "no_eligible_nonzero_law"
    else:
        best = min(
            eligible_nonzero,
            key=lambda law: (metrics[law]["J"], laws.index(law)),
        )
        cutoff = metrics[best]["J"] + metrics[best]["jackknife"]["SE_J"]
        if metrics["R0"]["J"] <= cutoff + tolerance:
            selected = "R0"
            reason = "R0_within_one_SE_of_best_eligible_law"
        else:
            selected = next(
                law
                for law in laws[1:]
                if law in eligible_nonzero
                and metrics[law]["J"] <= cutoff + tolerance
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
        "comparison_tolerance": tolerance,
        "simplicity_order": laws,
        "metrics": metrics,
        "eligibility": eligibility,
    }


def _final_information_fit(
    context: ft_registry.FitContext,
    config: dict[str, Any],
    validation: dict[str, Any],
    selected_law: str,
) -> dict[str, Any]:
    if selected_law == "R0":
        return {
            "boundary": 2014,
            "selected_law": selected_law,
            "status": "NOT_RUN_R0_SELECTED",
            "construction_pass": True,
            "designed_pause_continues": True,
        }
    _progress(f"final 2014 information fit: {selected_law}")
    try:
        fitted = _fit_boundary(
            context,
            config,
            validation,
            2014,
            law_order=(selected_law,),
        )
    except RootFailure as error:
        return {
            "boundary": 2014,
            "selected_law": selected_law,
            "status": "FINAL_FIT_FAILURE_DESIGNED_PAUSE",
            "construction_pass": False,
            "failure_code": error.code,
            "failure": str(error),
            "selected_pseudo_holdout_law_retained": True,
            "substitution_or_reselection": False,
            "designed_pause_continues": True,
        }
    except ProtocolAbort as error:
        return {
            "boundary": 2014,
            "selected_law": selected_law,
            "status": "FINAL_FIT_FAILURE_DESIGNED_PAUSE",
            "construction_pass": False,
            "failure_code": "FINAL_PROTOCOL_ABORT",
            "failure": str(error),
            "selected_pseudo_holdout_law_retained": True,
            "substitution_or_reselection": False,
            "designed_pause_continues": True,
        }
    construction = fitted["laws"][selected_law]["public"]
    tolerance = float(config["construction"]["area_tolerance"])
    comparison_tolerance = float(config["selector"]["comparison_tolerance"])
    budget = float(config["construction"]["widowed_log_rate_budget"])
    k, omega = _law_components(config, selected_law)
    checks = {
        "fit_max_year_2013": fitted["public"]["fit_max_year"] == 2013,
        "validation_match": fitted["public"]["validation_match"],
        "divorced_area_within_tolerance": abs(
            construction["divorced_area_relative_residual"]
        )
        <= tolerance,
        "widowed_area_within_tolerance": abs(
            construction["widowed_area_relative_residual"]
        )
        <= tolerance,
        "divorced_delta_exact_target": abs(construction["Delta_divorced"] + k)
        <= comparison_tolerance,
        "widowed_delta_exact_target": abs(
            construction["Delta_widowed"] - omega
        )
        <= comparison_tolerance,
        "positive_beta": construction["beta_frontload"] > 0.0,
        "divorced_rise_and_fall": (
            construction["positive_exposure_divorced_cells_rising"] > 0
            and construction["positive_exposure_divorced_cells_falling"] > 0
        ),
        "widowed_table_construction_exact": construction[
            "widowed_table_construction_exact"
        ],
        "omega_within_budget": (
            omega >= -comparison_tolerance
            and omega <= budget + comparison_tolerance
        ),
    }
    passed = all(checks.values())
    return {
        "boundary": 2014,
        "selected_law": selected_law,
        "status": "PASS" if passed else "FINAL_FIT_FAILURE_DESIGNED_PAUSE",
        "construction_pass": passed,
        "checks": checks,
        "fit": fitted["public"],
        "selected_law_table": construction,
        "designed_pause_continues": not passed,
    }


def _input_and_file_open_audit(
    config: dict[str, Any], source_audit: dict[str, Any]
) -> dict[str, Any]:
    observed = sorted(OBSERVED_OPEN_PATHS)
    gates_path = (ROOT / "gates.yaml").resolve()
    runs_path = (ROOT / "runs").resolve()
    gates_read = False
    runs_read = False
    for value in observed:
        path = Path(value)
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        gates_read = gates_read or resolved == gates_path
        runs_read = (
            runs_read or resolved == runs_path or runs_path in resolved.parents
        )
    if gates_read or runs_read:
        raise ProtocolAbort("forbidden gate or run artifact was opened")
    return {
        "selection_relevant_reads": [
            "scripts/m6_remarriage_round2_selector_config.json",
            "docs/design/m6_remarriage_learning_plan_round2.md",
            "docs/design/m6_remarriage_learning_plan_round2_validation.json",
            "docs/analysis/m6_remarriage_train_only_delta_results.json",
            "sanitized staged PSID sources through the frozen round-1 chassis",
        ],
        "maximum_information_year": config["protocol"][
            "maximum_information_year"
        ],
        "source_audit": source_audit,
        "dynamic_open_audit_installed_before_selection_reads": (
            FILE_OPEN_AUDIT_INSTALLED
        ),
        "observed_open_path_count": len(observed),
        "observed_open_paths": observed,
        "gates_yaml_read": gates_read,
        "runs_artifact_read": runs_read,
        "M6_scorer_imported": False,
        "post_2014_selection_data_read": False,
        "helper_wrote_files": False,
        "stdout_machine_documents": 1,
    }


def _synthetic_smoke() -> dict[str, Any]:
    pairwise = _pairwise_sum([1.0, 2.0, 3.0, 4.0, 5.0])
    if pairwise != 15.0:
        raise ProtocolAbort("synthetic pairwise sum failed")
    baseline = {
        (age, ysd, origin, sex): 0.05
        for age in range(len(rem.AGE_BANDS))
        for ysd in range(len(rem.YSD_BANDS))
        for origin in ORIGINS
        for sex in SEXES
    }
    centered = [1.0, 0.0, -1.0]
    years = np.arange(1, 21, dtype=np.int64)
    synthetic_reference = {
        "paths": [
            {
                "weight": 1.0,
                "logits": np.full(
                    len(years), float(scipy_logit(0.05)), dtype=np.float64
                ),
                "ysd_index": np.clip(
                    np.searchsorted(rem.YSD_LOWERS, years, side="right") - 1,
                    0,
                    len(rem.YSD_BANDS) - 1,
                ),
                "working": np.ones(len(years), dtype=bool),
            }
        ]
    }
    synthetic_risk = pd.DataFrame(
        {
            "ysd_band": [0, 1, 2],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    table, root = _solve_divorced_root(
        baseline,
        synthetic_risk,
        0.0,
        centered,
        synthetic_reference,
        k=0.5,
        omega=0.0,
        tolerance=1e-10,
        maximum_iterations=200,
    )
    omega_zero_widowed_exact = all(
        table[key] == baseline[key] for key in baseline if key[2] == "widowed"
    )
    if not omega_zero_widowed_exact:
        raise ProtocolAbort("synthetic omega-zero widowed identity failed")
    synthetic = pd.DataFrame(
        {
            "age": [17.0, 18.0, 64.0, 65.0],
            "years_since_dissolution": [1, 1, 1, 1],
            "origin": ["divorced"] * 4,
            "sex": ["female"] * 4,
        }
    )
    probability = _lookup_probabilities(
        synthetic,
        table,
        baseline,
        origin_column="origin",
    )
    raw_age_guard = bool(
        probability[0] == 0.05
        and probability[3] == 0.05
        and probability[1] == table[(0, 0, "divorced", "female")]
        and probability[2] == table[(2, 0, "divorced", "female")]
    )
    if not raw_age_guard:
        raise ProtocolAbort("synthetic raw-age guard failed")
    return {
        "schema": SMOKE_SCHEMA,
        "status": "SYNTHETIC_SMOKE_PASS",
        "synthetic_only": True,
        "staged_data_opened": False,
        "pseudo_holdout_truth_constructed": False,
        "candidate_outcome_computed": False,
        "selector_run": False,
        "checks": {
            "pairwise_sum": pairwise,
            "root_first_accepted_midpoint": root,
            "raw_age_guard": raw_age_guard,
            "omega_zero_widowed_exact_R0": omega_zero_widowed_exact,
            "table_sha256": _table_checksum(table),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run synthetic-only checks without opening staged data",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="validate admissible inputs and fits without outcome contact",
    )
    mode.add_argument(
        "--select",
        action="store_true",
        help="run the one frozen full selector",
    )
    return parser.parse_args()


def main() -> int:
    _install_file_open_audit()
    args = _parse_args()
    config = _load_config()
    _assert_config(config)
    freeze = _repository_freeze(config)
    runtime = _runtime_audit(config)
    validation = _load_json(VALIDATION_PATH)
    if (
        validation.get("schema")
        != "m6.remarriage.learning_plan.round2.fit_side_validation.v1"
    ):
        raise RootFailure(
            "ROOT_VALIDATION_MISMATCH", "validation schema drifted"
        )
    _assert_validation_index(config, validation)
    if args.smoke:
        payload = {
            **_synthetic_smoke(),
            "freeze": freeze,
            "runtime": runtime,
            "config_sha256": _sha256_file(CONFIG_PATH),
        }
        print(
            json.dumps(
                _plain(payload), indent=2, sort_keys=True, allow_nan=False
            )
        )
        return 0

    full_context, source_audit = round1._load_sanitized_context()
    _assert_expected_subset(
        source_audit,
        config["expected_source_audit"],
        "expected_source_audit",
    )
    prepared = _prepare_fits(full_context, config, validation)
    public_fits = {
        boundary: record["public"] for boundary, record in prepared.items()
    }
    if args.preflight:
        payload = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "PRE_OUTCOME_PREFLIGHT_PASS",
            "candidate_outcome_contact": False,
            "pseudo_holdout_truth_constructed": False,
            "freeze": freeze,
            "runtime": runtime,
            "source_audit": source_audit,
            "fit_validation": public_fits,
        }
        print(
            json.dumps(
                _plain(payload), indent=2, sort_keys=True, allow_nan=False
            )
        )
        return 0

    boundaries = {
        str(boundary): _evaluate_boundary(
            full_context,
            prepared[str(boundary)],
            config,
            int(boundary),
        )
        for boundary in config["protocol"]["boundaries"]
    }
    selection = _select_law(boundaries, config)
    final_fit = _final_information_fit(
        full_context,
        config,
        validation,
        selection["selected_law"],
    )
    if not final_fit["construction_pass"]:
        selection["disposition"] = "FINAL_FIT_FAILURE_DESIGNED_PAUSE"
    payload = {
        "schema": RAW_SCHEMA,
        "status": (
            "SELECTION_COMPLETE"
            if final_fit["construction_pass"]
            else "SELECTION_COMPLETE_FINAL_FIT_FAILURE"
        ),
        "candidate_outcome_contact": True,
        "authority": config["authority"],
        "freeze": freeze,
        "runtime": runtime,
        "protocol": config["protocol"],
        "input_and_file_open_audit": _input_and_file_open_audit(
            config, source_audit
        ),
        "fit_validation": public_fits,
        "boundaries": boundaries,
        "selection": selection,
        "final_information_fit": final_fit,
        "publication": {
            "full_stdout_path": config["output"]["full_stdout_json"],
            "reduced_findings_path": config["output"]["findings_json"],
            "report_path": config["output"]["findings_report"],
            "publish_regardless_of_outcome": True,
            "cumulative_nonzero_laws_two_rounds": config["family"][
                "cumulative_nonzero_laws_two_rounds"
            ],
        },
    }
    print(
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
