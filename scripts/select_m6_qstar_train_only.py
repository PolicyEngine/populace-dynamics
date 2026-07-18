#!/usr/bin/env python3
"""Run the registered amendment-4 train-only q-star selector.

This is a non-scoring analysis runner.  It implements section 2.7.7 of
``docs/design/m6_projection_engine.md`` without importing the M6 gate runner,
reading a gate artifact, or writing below ``runs/``.  Its only machine output
is one strict-JSON payload on stdout; progress is flushed to stderr.

The selector needs the same two-stack environment as the registered M6 run.
The fitting stack (``populace-fit`` / ``populace-frame`` with scikit-learn
``<1.9``) must be installed in a dedicated environment.  The certified
``policyengine-us==1.752.2`` install must be importable there, and
``POPULACE_DYNAMICS_PE_US_DIR`` must point at the directory containing its
``policyengine_us/`` package.  The staged PSID directory is selected only by
``POPULACE_DYNAMICS_PSID_DIR``.  For example, from the repository root::

    PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS='ignore::FutureWarning' \
      PYTHONPATH=src \
      POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data \
      POPULACE_DYNAMICS_PE_US_DIR=/path/to/site-packages \
      /path/to/two-stack-venv/bin/python \
      scripts/select_m6_qstar_train_only.py \
      > /safe/path/m6-qstar-full.json \
      2> /safe/path/m6-qstar-progress.log

There are deliberately no selector-changing command-line options.  The Q
grid, pseudo-boundaries, fit seed, draw seeds, halves, floor seeds, objective,
feasibility guards, jackknife, and smallest-q one-SE rule are constants below.

Implementation choices frozen with this runner:

* each q-by-boundary cell performs a fresh complete QRF refit, even though q
  does not enter the fit; q-invariant fit and donor checksums must agree;
* the full ``b`` anchor is the positive-weight demographic roster at the exact
  collection wave ``b+1``.  It is split/addressed before intersection with the
  exact-b earnings domain.  No later pseudo-window opener enters the anchor;
* the projection frame carries a fixed nonmissing sex sentinel.  Sex is a
  schema check but is absent from every fit, gate feature, donor distance, and
  reduced cell, so this avoids an unrelated death-data read without changing
  a numerical input;
* the refresh prototype delegates the complete incumbent ``generate`` call.
  It advances the parent generator through the incumbent seed bridge once,
  replays that exact integer into the delegated call, then uses isolated
  ``SeedSequence([seed, 4])`` and ``SeedSequence([seed, 5])`` children.  At
  q=0 the exact ndarray returned by the incumbent is returned untouched after
  the mandatory code-4/code-5 draws;
* raw earnings are requested only for income-reference periods no later than
  2014.  Every boundary fit receives a separately truncated ``period <= b``
  frame and a NAWI mapping containing no key later than ``b``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from registered_m6_inputs import (
    assert_pe_us_param_dir,
    assert_pe_us_version,
)

from populace_dynamics.data import family, panels, psid
from populace_dynamics.engine import forward_earnings as fe
from populace_dynamics.engine.earnings_domain import (
    EARNINGS_DOMAIN_COLUMN,
    earnings_domain_person_ids,
    wrap_earnings_domain,
)
from populace_dynamics.engine.loop import PeriodContext
from populace_dynamics.engine.refit import (
    EarningsChainedRefit,
    refit_earnings_chained_generator,
    truncate_estimation_frame,
)
from populace_dynamics.engine.rng import (
    DRAW_SEED_BASE,
    GATE_M6_TAG,
    MODULE_ORDER,
    ProjectionModule,
    ProjectionRNGRegistry,
    seed_from_generator,
)
from populace_dynamics.engine.steps import apply_earnings
from populace_dynamics.harness.m6_cells import (
    EARN_COHORTS,
    FLOOR_SEEDS,
    earnings_cells,
    run_floor,
)
from populace_dynamics.harness.m6_preflight import verify_external_sign_path
from populace_dynamics.harness.m6_scoring import (
    restrict_earnings_domain_support,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_INFORMATION_YEAR = 2014
PSEUDO_BOUNDARIES = (2006, 2008, 2010)
Q_GRID = tuple(round(index * 0.05, 2) for index in range(21))
FIT_SEED = 5200
SELECTION_DRAW_SEEDS = tuple(range(6200, 6220))
FIRST_HALF_DRAW_SEEDS = tuple(range(6200, 6210))
SECOND_HALF_DRAW_SEEDS = tuple(range(6210, 6220))
SUBSTREAM_CODES = {
    **fe.SUBSTREAM_CODES,
    "memory-refresh-gate": 4,
    "memory-refresh-rank": 5,
}
SELECTED_CELLS = (
    "earn_p10.prime",
    "earn_dlog_mean.prime",
    "earn_dlog_sd.older",
    "earn_mob_h1_diag",
    "earn_autocorr_lag2",
    "earn_zero_rate.older",
)
OBJECTIVE_CELLS = (
    "earn_p10.prime",
    "earn_dlog_mean.prime",
    "earn_mob_h1_diag",
    "earn_autocorr_lag2",
)
FEASIBILITY_CELLS = (
    "earn_dlog_sd.older",
    "earn_zero_rate.older",
)
SCHEMA_SEX_SENTINEL = "selection_schema_only"
MAX_ANCHOR_INTERVIEW_YEAR = max(boundary + 1 for boundary in PSEUDO_BOUNDARIES)
EARNINGS_COLLECTION_WAVES = tuple(
    wave
    for wave in family.FAMILY_WAVES
    if 1969 <= wave and wave - 1 <= MAX_INFORMATION_YEAR
)
ANCHOR_COLLECTION_WAVES = tuple(boundary + 1 for boundary in PSEUDO_BOUNDARIES)
RAW_SCHEMA = "m6_qstar_train_only_selection.v1"
EXPECTED_POPULACE_HEAD = "af02c917fcb3c50816bf3af9c2b64509e928889a"
EXPECTED_POPULACE_FIT_TREE = "5c866378fdf5906b7a61da9977b8d028d1d36e9f"
EXPECTED_POPULACE_FRAME_TREE = "7cfb9ee78beb74911963913f202a4471aae2f52b"
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.14.4",
    "numpy": "2.5.1",
    "pandas": "3.0.3",
    "scikit_learn": "1.8.0",
    "scipy": "1.18.0",
    "quantile_forest": "1.4.2",
    "populace_fit": "0.1.0",
    "populace_frame": "0.1.0",
}
EXPECTED_BOUNDARY_NAWI = {
    2006: {
        "prefix_bytes": 1492,
        "prefix_sha256": (
            "0f9f6ea4cc0c95dd10142c4673dbcb8c9447327f1294860ee4117ac55ce0fcf2"
        ),
        "mapping_sha256": (
            "749b034a756468e1b1b3a164e30465dc313ca474e8e3e5663c2a38b23e40b67c"
        ),
    },
    2008: {
        "prefix_bytes": 1540,
        "prefix_sha256": (
            "d3b20c353081a4094a4044c189354ffad6e426a5ce3b5bd6535d589ee46786c4"
        ),
        "mapping_sha256": (
            "37ca4ec34d2342f624c3ba9a47d9980e08864d07c4a5cd6d579bf614bfb2434e"
        ),
    },
    2010: {
        "prefix_bytes": 1588,
        "prefix_sha256": (
            "e272a7bc63a1e23fa4ca4c6abdd7bc35ef0e5fa56943e49b215a262c2e8440d7"
        ),
        "mapping_sha256": (
            "26b6a0498ccff6f5a23bf226ec6fe47347dca853b9844bb2da1c8a831579582c"
        ),
    },
}


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
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
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _frame_checksum(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(list(columns), kind="stable").reset_index(
        drop=True
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _array_mapping_checksum(mapping: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        array = np.ascontiguousarray(np.asarray(mapping[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _id_checksum(values: Iterable[object]) -> str:
    ordered = sorted(int(value) for value in set(values))
    raw = "".join(f"{value}\n" for value in ordered).encode()
    return hashlib.sha256(raw).hexdigest()


def _key_checksum(frame: pd.DataFrame) -> str:
    keys = frame[["person_id", "period"]].drop_duplicates()
    keys = keys.sort_values(["person_id", "period"], kind="stable")
    raw = "".join(
        f"{int(row.person_id)}|{int(row.period)}\n"
        for row in keys.itertuples(index=False)
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _level_checksum(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["person_id", "period"], kind="stable")
    digest = hashlib.sha256()
    digest.update(ordered["person_id"].to_numpy(dtype=np.int64).tobytes())
    digest.update(ordered["period"].to_numpy(dtype=np.int64).tobytes())
    digest.update(ordered["earnings"].to_numpy(dtype=np.float64).tobytes())
    return digest.hexdigest()


def _participation_checksum(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["person_id", "period"], kind="stable")
    participation = (
        ordered["earnings"].to_numpy(dtype=np.float64) > 0
    ).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(ordered["person_id"].to_numpy(dtype=np.int64).tobytes())
    digest.update(ordered["period"].to_numpy(dtype=np.int64).tobytes())
    digest.update(participation.tobytes())
    return digest.hexdigest()


def _max_year(frame: pd.DataFrame, column: str) -> int | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.max()) if len(values) else None


def _assert_at_most(
    frame: pd.DataFrame, column: str, boundary: int, label: str
) -> None:
    maximum = _max_year(frame, column)
    if maximum is not None and maximum > boundary:
        raise AssertionError(
            f"{label}.{column} reaches {maximum}, beyond {boundary}"
        )


def _cell_value(record: Mapping[str, Any]) -> float | None:
    raw = record.get("rate", record.get("value"))
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _selection_cell_value(record: Mapping[str, Any]) -> float | None:
    """Return a draw value only when its registered metric defines it."""
    value = _cell_value(record)
    if value is None:
        return None
    metric = str(record.get("metric", "log_ratio"))
    if (metric == "log_ratio" or "rate" in record) and value <= 0:
        return None
    return value


def _selected_cells(frame: pd.DataFrame, boundary: int) -> dict[str, Any]:
    reduced = earnings_cells(
        frame,
        level_years=(boundary + 2, boundary + 4),
        change_years=(boundary, boundary + 2, boundary + 4),
    )
    missing = set(SELECTED_CELLS) - set(reduced)
    if missing:
        raise ValueError(
            f"earnings reducer omitted selected cells {sorted(missing)}"
        )
    return {name: reduced[name] for name in SELECTED_CELLS}


def _score(
    projected_value: float,
    truth_record: Mapping[str, Any],
) -> float | None:
    truth_value = _selection_cell_value(truth_record)
    if truth_value is None or not math.isfinite(projected_value):
        return None
    metric = str(truth_record.get("metric", "log_ratio"))
    if metric == "log_ratio" or "rate" in truth_record:
        if projected_value <= 0:
            return None
        return abs(math.log(projected_value / truth_value))
    return abs(projected_value - truth_value)


def _cohort(age: object) -> str | None:
    if pd.isna(age):
        return None
    value = int(age)
    for name, lower, upper in EARN_COHORTS:
        if lower <= value <= upper:
            return name
    return None


def _read_family_labor_levels(wave: int, *, data_dir: Path) -> pd.DataFrame:
    """Read only the three family-file fields needed by this selector.

    The general family reader also materializes assignment-code fields.  Those
    fields are irrelevant here, so this audit-critical reader deliberately
    resolves and reads only interview ID plus the two role-specific labor
    income levels.  Private helpers are reused to keep the incumbent label and
    reference-year validation byte-for-byte aligned.
    """
    sps_path, txt_path = family._family_paths(wave, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    if wave in family._PRE94:
        entry = family._PRE94[wave]
        interview = family._verified(labels, *entry["interview"], wave)
        head = family._verified(labels, *entry["head"], wave)
        spouse = family._verified(labels, *entry["wife"], wave)
    else:
        interview = family._single(
            labels,
            rf"^{wave} (INTERVIEW #|FAMILY INTERVIEW \(ID\) NUMBER)$",
            wave,
            "interview number",
        )
        head = family._single(
            labels, family._HEAD_LABOR, wave, "head labor income"
        )
        spouse = family._single(
            labels, family._SPOUSE_LABOR, wave, "spouse labor income"
        )
        family._check_reference_year(labels[head], wave)
        family._check_reference_year(labels[spouse], wave)

    layout = psid.parse_sps_layout(sps_path).set_index("name")
    names = [interview, head, spouse]
    colspecs = [
        (
            int(layout.loc[name, "start"]) - 1,
            int(layout.loc[name, "end"]),
        )
        for name in names
    ]
    raw = pd.read_fwf(
        txt_path,
        colspecs=colspecs,
        names=names,
        header=None,
    )
    return raw.rename(
        columns={
            interview: "interview",
            head: "head_labor",
            spouse: "spouse_labor",
        }
    )


def _load_field_capped_psid(
    data_dir: Path,
    *,
    collection_waves: tuple[int, ...] = EARNINGS_COLLECTION_WAVES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build incumbent-shaped earnings and anchors without a broad read."""
    if not collection_waves:
        raise ValueError("field-capped collection-wave set is empty")
    if max(collection_waves) != MAX_INFORMATION_YEAR + 1:
        raise AssertionError(
            "2014 earnings collection wave is not uniquely capped"
        )
    if any(wave - 1 > MAX_INFORMATION_YEAR for wave in collection_waves):
        raise AssertionError("post-2014 earnings reference field requested")

    concepts = {
        name: panels.DEMOGRAPHIC_CONCEPTS[name]
        for name in ("age", "sequence", "relationship", "weight", "interview")
    }
    person_wave = panels.ind_person_period(
        concepts,
        data_dir=data_dir,
        waves=list(collection_waves),
    )
    present = person_wave["sequence"].between(1, 20)
    person_wave = person_wave.loc[present].reset_index(drop=True)
    if _max_year(person_wave, "period") != max(collection_waves):
        raise AssertionError("field-capped individual wave read is incomplete")

    anchor_demo = person_wave.loc[
        person_wave["period"].isin(ANCHOR_COLLECTION_WAVES)
        & (pd.to_numeric(person_wave["weight"], errors="coerce") > 0),
        ["person_id", "period", "weight", "interview"],
    ].copy()

    frames: list[pd.DataFrame] = []
    for wave in collection_waves:
        labor = _read_family_labor_levels(wave, data_dir=data_dir)
        wave_people = person_wave.loc[person_wave["period"] == wave]
        merged = wave_people.merge(labor, on="interview", how="inner")
        head_codes, spouse_codes = family._relationship_codes(wave)
        is_head = merged["relationship"].isin(head_codes)
        is_spouse = merged["relationship"].isin(spouse_codes)
        merged = merged.loc[is_head | is_spouse].copy()
        head_mask = merged["relationship"].isin(head_codes)
        merged["earnings"] = (
            merged["head_labor"]
            .where(head_mask, merged["spouse_labor"])
            .astype("float64")
        )
        merged["period"] = wave - 1
        frames.append(
            merged[["person_id", "period", "earnings", "age", "weight"]]
        )
    earnings = pd.concat(frames, ignore_index=True)
    keep = (earnings["earnings"] < family._MISSING) & (earnings["weight"] > 0)
    earnings = (
        earnings.loc[keep]
        .sort_values(["person_id", "period"], kind="stable")
        .reset_index(drop=True)
    )
    audit = {
        "individual_reader": (
            "panels.ind_person_period with an explicit collection-wave list"
        ),
        "individual_concepts_requested": list(concepts),
        "family_fields_requested_per_wave": [
            "interview",
            "head_labor",
            "spouse_labor",
        ],
        "collection_waves_requested": list(collection_waves),
        "maximum_collection_wave": max(collection_waves),
        "collection_waves_after_2014": [
            wave for wave in collection_waves if wave > MAX_INFORMATION_YEAR
        ],
        "post_2014_reference_field_requested": False,
        "broad_demographic_panel_called": False,
        "broad_family_earnings_panel_called": False,
    }
    return earnings, anchor_demo, audit


_NAWI_LINE = re.compile(
    rb"^  (?P<year>\d{4})-\d{2}-\d{2}:\s*(?P<value>[^#\r\n]+)"
)


def _read_historical_nawi(
    path: Path, *, maximum_year: int
) -> tuple[dict[int, float], dict[str, Any]]:
    """Read NAWI through ``maximum_year`` without reading the next key.

    The pinned policyengine-us YAML is year-sorted with scalar values on the
    key line.  An ordinary YAML load necessarily materializes realized future
    values.  This unbuffered byte reader stops at the newline terminating the
    maximum admitted entry, before any byte of the next key enters memory.
    """
    values: dict[int, float] = {}
    line = bytearray()
    consumed = bytearray()
    stopped_after_maximum = False
    with Path(path).open("rb", buffering=0) as stream:
        while True:
            byte = stream.read(1)
            if not byte:
                if line:
                    match = _NAWI_LINE.match(bytes(line))
                    if match:
                        year = int(match.group("year"))
                        values[year] = float(
                            match.group("value").replace(b"_", b"")
                        )
                break
            line.extend(byte)
            consumed.extend(byte)
            if (
                len(line) == 6
                and line[:2] == b"  "
                and bytes(line[2:6]).isdigit()
                and int(bytes(line[2:6])) > maximum_year
            ):
                raise AssertionError(
                    "NAWI reader reached a post-maximum key before the "
                    "required maximum entry"
                )
            if byte == b"\n":
                match = _NAWI_LINE.match(bytes(line))
                if match:
                    year = int(match.group("year"))
                    if year > maximum_year:
                        raise AssertionError(
                            "NAWI reader crossed the maximum admitted key"
                        )
                    value = float(match.group("value").replace(b"_", b""))
                    if values and year <= max(values):
                        raise ValueError(
                            "NAWI keys are not strictly increasing"
                        )
                    values[year] = value
                    if year == maximum_year:
                        stopped_after_maximum = True
                        break
                line.clear()
    if not values or max(values) != maximum_year:
        raise ValueError(
            f"NAWI prefix ends at {max(values) if values else None}, "
            f"expected {maximum_year}"
        )
    if not stopped_after_maximum:
        raise ValueError(
            "NAWI reader did not stop at the maximum admitted key"
        )
    return values, {
        "path": str(Path(path).resolve()),
        "maximum_admitted_key_year": max(values),
        "stopped_after_maximum_key": True,
        "post_maximum_key_bytes_read": False,
        "bytes_consumed_through_maximum_key": len(consumed),
        "admitted_prefix_sha256": hashlib.sha256(consumed).hexdigest(),
    }


def _load_train_only_sources() -> (
    tuple[pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]
):
    psid_dir = os.environ.get("POPULACE_DYNAMICS_PSID_DIR")
    pe_us_dir = os.environ.get("POPULACE_DYNAMICS_PE_US_DIR")
    if not psid_dir:
        raise RuntimeError("POPULACE_DYNAMICS_PSID_DIR must be set explicitly")
    if not pe_us_dir:
        raise RuntimeError(
            "POPULACE_DYNAMICS_PE_US_DIR must be set explicitly"
        )
    if not Path(psid_dir).is_dir():
        raise RuntimeError(f"PSID directory does not exist: {psid_dir}")
    if not Path(pe_us_dir).is_dir():
        raise RuntimeError(
            f"policyengine-us package directory does not exist: {pe_us_dir}"
        )

    _progress("loading field-capped staged PSID sources (read-only)")
    earnings, anchor_demo, psid_read_audit = _load_field_capped_psid(
        Path(psid_dir)
    )
    _assert_at_most(
        earnings,
        "period",
        MAX_INFORMATION_YEAR,
        "sanitized_earnings",
    )
    if earnings.duplicated(["person_id", "period"]).any():
        raise ValueError("sanitized earnings has duplicate person-period rows")
    numeric = earnings[["earnings", "age", "weight"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("sanitized earnings contains non-finite values")
    if (earnings["earnings"] < 0).any() or (earnings["weight"] <= 0).any():
        raise ValueError("sanitized earnings has negative levels or weights")

    _assert_at_most(
        anchor_demo,
        "period",
        MAX_ANCHOR_INTERVIEW_YEAR,
        "sanitized_anchor_demographic",
    )
    if anchor_demo.duplicated(["person_id", "period"]).any():
        raise ValueError("anchor demographic has duplicate person-period rows")

    _progress("binding certified boundary-capped NAWI source")
    version = assert_pe_us_version()
    parameter_dir = assert_pe_us_param_dir()
    nawi_path = parameter_dir / "nawi.yaml"
    if not nawi_path.is_file():
        raise FileNotFoundError(
            f"certified NAWI source is absent: {nawi_path}"
        )

    audit = {
        "raw_source_is_retrospective_product": True,
        "maximum_selection_information_year": MAX_INFORMATION_YEAR,
        "candidate_or_gate_artifact_read": False,
        "gate_configuration_read": False,
        "runs_path_written": False,
        "post_2014_earnings_reference_row_requested": False,
        "psid_field_read": psid_read_audit,
        "requested_earnings_reference_max": _max_year(earnings, "period"),
        "anchor_demographic_max_interview": _max_year(anchor_demo, "period"),
        "policyengine_us_version": version,
        "policyengine_us_parameter_dir": str(parameter_dir),
        "nawi_path": str(nawi_path.resolve()),
        "nawi_read_policy": (
            "no NAWI value is loaded during source assembly; each q-by-"
            "boundary cell performs an unbuffered prefix read that stops at "
            "that exact b and asserts its frozen prefix and mapping hashes"
        ),
        "expected_boundary_nawi": EXPECTED_BOUNDARY_NAWI,
        "psid_dir": str(Path(psid_dir).resolve()),
        "sanitized_rows": {
            "earnings": int(len(earnings)),
            "anchor_demographic": int(len(anchor_demo)),
        },
        "checksums": {
            "earnings": _frame_checksum(
                earnings,
                ("person_id", "period", "earnings", "age", "weight"),
            ),
            "anchor_demographic": _frame_checksum(
                anchor_demo,
                ("person_id", "period", "weight", "interview"),
            ),
        },
        "dating_note": (
            "family labor-income levels are field-dated by their verified "
            "income-reference year. For the amendment-required 2014 earnings "
            "row, the incumbent family-panel convention reads the 2015 "
            "collection-wave age, sequence, relationship, weight, and "
            "interview fields: relationship/interview attach the two family "
            "labor fields, sequence defines presence, and age/weight enter "
            "support and reduction. This is the sole collection wave above "
            "2014; no 2015 earnings-reference field is requested"
        ),
    }
    return earnings, anchor_demo, nawi_path, audit


def _repository_freeze() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "q-star selector requires the committed clean pre-outcome freeze; "
            f"worktree status is:\n{status}"
        )
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    frozen_paths = (
        "scripts/select_m6_qstar_train_only.py",
        "scripts/reduce_m6_qstar_selection.py",
        "tests/test_m6_qstar_selection.py",
        "docs/analysis/m6_qstar_train_only_selection.md",
    )
    blobs = {path: git("rev-parse", f"HEAD:{path}") for path in frozen_paths}
    local_package_root = (ROOT / "src" / "populace_dynamics").resolve()
    local_imports = {
        "forward_earnings": Path(inspect.getfile(fe)).resolve(),
        "family": Path(inspect.getfile(family)).resolve(),
        "panels": Path(inspect.getfile(panels)).resolve(),
        "refit_earnings_chained_generator": Path(
            inspect.getfile(refit_earnings_chained_generator)
        ).resolve(),
        "earnings_cells": Path(inspect.getfile(earnings_cells)).resolve(),
    }
    for name, path in local_imports.items():
        try:
            path.relative_to(local_package_root)
        except ValueError as error:
            raise RuntimeError(
                f"{name} imported from {path}, outside frozen branch source "
                f"{local_package_root}"
            ) from error
    local_source_tree = git("rev-parse", "HEAD:src/populace_dynamics")
    qrf_class = importlib.import_module("populace.fit.qrf").RegimeGatedQRF
    qrf_path = Path(inspect.getfile(qrf_class)).resolve()
    frame_class = importlib.import_module("populace.frame").Frame
    frame_path = Path(inspect.getfile(frame_class)).resolve()
    populace_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=qrf_path.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    qrf_relative = qrf_path.relative_to(populace_root).as_posix()
    expected_fit_root = (populace_root / "packages" / "populace-fit").resolve()
    expected_frame_root = (
        populace_root / "packages" / "populace-frame"
    ).resolve()
    try:
        qrf_path.relative_to(expected_fit_root / "src")
        frame_path.relative_to(expected_frame_root / "src")
    except ValueError as error:
        raise RuntimeError(
            "editable fitting-stack imports do not resolve inside the "
            "frozen populace repository"
        ) from error

    def populace_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=populace_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    tracked_status = populace_git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "packages/populace-fit/src",
        "packages/populace-frame/src",
    )
    if tracked_status:
        raise RuntimeError(
            "the editable populace fitting-stack source differs from its "
            f"repository HEAD:\n{tracked_status}"
        )
    populace_head = populace_git("rev-parse", "HEAD")
    if populace_head != EXPECTED_POPULACE_HEAD:
        raise RuntimeError(
            f"editable populace HEAD {populace_head} differs from frozen "
            f"{EXPECTED_POPULACE_HEAD}"
        )
    fit_tree = populace_git(
        "rev-parse", "HEAD:packages/populace-fit/src/populace/fit"
    )
    frame_tree = populace_git(
        "rev-parse", "HEAD:packages/populace-frame/src/populace/frame"
    )
    if fit_tree != EXPECTED_POPULACE_FIT_TREE:
        raise RuntimeError("editable populace-fit source tree drifted")
    if frame_tree != EXPECTED_POPULACE_FRAME_TREE:
        raise RuntimeError("editable populace-frame source tree drifted")
    qrf_blob = populace_git("rev-parse", f"HEAD:{qrf_relative}")
    if populace_git("hash-object", qrf_relative) != qrf_blob:
        raise RuntimeError("editable populace-fit QRF bytes do not match HEAD")

    def direct_url(distribution: str) -> Any:
        raw = importlib.metadata.distribution(distribution).read_text(
            "direct_url.json"
        )
        return None if raw is None else json.loads(raw)

    fit_direct_url = direct_url("populace-fit")
    frame_direct_url = direct_url("populace-frame")
    expected_direct_urls = {
        "populace-fit": {
            "url": f"file://{expected_fit_root}",
            "dir_info": {"editable": True},
        },
        "populace-frame": {
            "url": f"file://{expected_frame_root}",
            "dir_info": {"editable": True},
        },
    }
    if (
        fit_direct_url != expected_direct_urls["populace-fit"]
        or frame_direct_url != expected_direct_urls["populace-frame"]
    ):
        raise RuntimeError(
            "editable fitting-stack direct URLs do not match the frozen "
            f"package roots: fit={fit_direct_url}, frame={frame_direct_url}"
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
    }
    if runtime != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(
            f"two-stack runtime {runtime} differs from frozen "
            f"{EXPECTED_RUNTIME_VERSIONS}"
        )
    return {
        "freeze_commit": head,
        "branch": branch,
        "worktree_clean": True,
        "frozen_blob_sha1": blobs,
        "local_populace_dynamics": {
            "source_root": str(local_package_root),
            "source_tree_sha1": local_source_tree,
            "import_paths": {
                name: str(path) for name, path in local_imports.items()
            },
            "all_imports_from_frozen_branch": True,
        },
        "editable_fitting_stack": {
            "populace_repository_root": str(populace_root),
            "populace_repository_head": populace_head,
            "populace_repository_branch": populace_git(
                "branch", "--show-current"
            ),
            "qrf_path": str(qrf_path),
            "frame_path": str(frame_path),
            "qrf_git_blob_sha1": qrf_blob,
            "qrf_sha256": hashlib.sha256(qrf_path.read_bytes()).hexdigest(),
            "qrf_tracked_bytes_match_head": True,
            "populace_fit_source_tree_sha1": fit_tree,
            "populace_frame_source_tree_sha1": frame_tree,
            "populace_fit_direct_url": fit_direct_url,
            "populace_frame_direct_url": frame_direct_url,
        },
        "runtime": runtime,
    }


def _stable_pools(
    fitted: EarningsChainedRefit,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    pairs = fitted.forward_pairs
    positive = pairs.loc[
        (pairs["earnings"] > 0) & (pairs["earnings_tp2"] > 0)
    ].copy()
    person = positive["person_id"].to_numpy(dtype=np.int64)
    period = positive["period_tp2"].to_numpy(dtype=np.int64)
    order = np.lexsort((period, person))
    positive = positive.iloc[order].reset_index(drop=True)
    pair_pool = fitted.generator.pools["pairs"]
    if not np.array_equal(
        positive["person_id"].to_numpy(dtype=np.int64),
        np.asarray(pair_pool["person_id"], dtype=np.int64),
    ) or not np.array_equal(
        positive["period_tp2"].to_numpy(dtype=np.int64),
        np.asarray(pair_pool["period_tp2"], dtype=np.int64),
    ):
        raise AssertionError(
            "stable-pool source does not align with incumbent pair-pool order"
        )
    if not positive["age_tp2"].between(fe.AGE_MIN, fe.AGE_MAX).all():
        raise AssertionError("stable-pool target ages escaped 25-64")
    bins = fe.age_bin(positive["age_tp2"].to_numpy(dtype=np.float64))
    by_bin: dict[int, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    checksums: dict[str, str] = {}
    for bin_index in range(fe.N_AGE_BINS):
        selected = bins == bin_index
        pool = {
            name: np.asarray(values)[selected]
            for name, values in pair_pool.items()
        }
        count = int(selected.sum())
        if count == 0:
            raise ValueError(
                f"boundary fit has empty exact target-age bin {bin_index}"
            )
        weights = np.asarray(pool["weight"], dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(
                f"stable target-age bin {bin_index} has invalid weights"
            )
        by_bin[bin_index] = pool
        counts[str(bin_index)] = count
        checksums[str(bin_index)] = _array_mapping_checksum(pool)
    if sum(counts.values()) != len(pair_pool["person_id"]):
        raise AssertionError("stable age-bin partition lost donor rows")
    audit = {
        "source": "incumbent positive-to-positive pair pool",
        "sort": ["person_id", "period_tp2"],
        "target_age_bin_width": fe.AGE_BIN_WIDTH,
        "k": fe.K_NEIGHBORS,
        "empty_pool_check_passed": True,
        "counts_by_bin": counts,
        "checksums_by_bin": checksums,
        "partition_checksum": _canonical_sha256(
            {
                "age_bin": bins,
                **{
                    name: np.asarray(value)
                    for name, value in pair_pool.items()
                },
            }
        ),
    }
    return by_bin, audit


class _ReplaySeedGenerator:
    """Replay one already-consumed parent seed into the incumbent bridge."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.calls = 0

    def integers(
        self,
        low: int,
        high: int | None = None,
        *,
        dtype: Any = np.int64,
        **_kwargs: Any,
    ) -> np.uint64:
        expected_high = int(np.iinfo(np.uint64).max)
        if int(low) != 0 or int(high or -1) != expected_high:
            raise AssertionError("incumbent parent-seed bridge changed bounds")
        if np.dtype(dtype) != np.dtype(np.uint64) or self.calls:
            raise AssertionError("incumbent parent-seed bridge changed shape")
        self.calls += 1
        return np.uint64(self.seed)


def _rng_state_checksum(rng: np.random.Generator) -> str:
    return _canonical_sha256(rng.bit_generator.state)


def _call_with_stream_trace(
    generator: Any,
    frame: pd.DataFrame,
    year: int,
    rng: Any,
    *,
    parent_before: str,
    parent_after: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    captured: dict[str, np.random.Generator] = {}
    captured_seeds: dict[str, int] = {}
    original = fe._substream

    def traced(seed: int, label: str) -> np.random.Generator:
        child = original(seed, label)
        captured[label] = child
        captured_seeds[label] = int(seed)
        return child

    fe._substream = traced
    try:
        result = np.asarray(generator.generate(frame, year, rng))
    finally:
        fe._substream = original
    if year % fe.PERIOD_STEP == 0:
        if set(captured) != set(fe.SUBSTREAM_CODES):
            raise AssertionError("incumbent substream registry use changed")
        if len(set(captured_seeds.values())) != 1:
            raise AssertionError("incumbent substreams did not share a seed")
        seed = next(iter(captured_seeds.values()))
    else:
        if captured:
            raise AssertionError(
                "odd-year incumbent call consumed a substream"
            )
        seed = None
    record = {
        "person_id": int(frame.iloc[0]["person_id"]),
        "year": int(year),
        "parent_state_before_sha256": parent_before,
        "parent_state_after_sha256": parent_after,
        "parent_bridge_seed": seed,
        "incumbent_stream_final_state_sha256": {
            label: _rng_state_checksum(child)
            for label, child in sorted(captured.items())
        },
        "level_sha256": hashlib.sha256(
            np.asarray(result, dtype=np.float64).tobytes()
        ).hexdigest(),
        "participating": [bool(value > 0) for value in result],
    }
    return result, record


@dataclass
class _TracedIncumbentGenerator:
    base: Any
    records: list[dict[str, Any]] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def materialize_initial_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.base.materialize_initial_frame(frame)

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray:
        before = _rng_state_checksum(rng)
        result, record = _call_with_stream_trace(
            self.base,
            frame,
            year,
            rng,
            parent_before=before,
        )
        record["parent_state_after_sha256"] = _rng_state_checksum(rng)
        self.records.append(record)
        return result


@dataclass
class RankRefreshPrototype:
    """Script-local candidate-2 generator delegating the incumbent law."""

    base: Any
    q: float
    stable_pools: Mapping[int, Mapping[str, np.ndarray]]
    trace_incumbent: bool = False
    incumbent_records: list[dict[str, Any]] = field(default_factory=list)
    refresh_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.q not in Q_GRID:
            raise ValueError(f"q={self.q} is outside the registered Q grid")
        if set(self.stable_pools) != set(range(fe.N_AGE_BINS)):
            raise ValueError("stable pools do not cover all exact age bins")
        if fe.SUBSTREAM_CODES != {
            "gate": 1,
            "donor-draw": 2,
            "re-entry-draw": 3,
        }:
            raise RuntimeError("incumbent earnings substream registry changed")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def materialize_initial_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.base.materialize_initial_frame(frame)

    def _delegate_even(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, int]:
        before = _rng_state_checksum(rng) if self.trace_incumbent else ""
        seed = seed_from_generator(rng)
        after = _rng_state_checksum(rng) if self.trace_incumbent else ""
        replay = _ReplaySeedGenerator(seed)
        if self.trace_incumbent:
            result, record = _call_with_stream_trace(
                self.base,
                frame,
                year,
                replay,
                parent_before=before,
                parent_after=after,
            )
            self.incumbent_records.append(record)
        else:
            result = np.asarray(self.base.generate(frame, year, replay))
        if replay.calls != 1:
            raise AssertionError(
                "incumbent generate changed parent consumption"
            )
        return result, seed

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray:
        year = int(year)
        if year % fe.PERIOD_STEP:
            if self.trace_incumbent:
                before = _rng_state_checksum(rng)
                result, record = _call_with_stream_trace(
                    self.base,
                    frame,
                    year,
                    rng,
                    parent_before=before,
                )
                record["parent_state_after_sha256"] = _rng_state_checksum(rng)
                self.incumbent_records.append(record)
                return result
            return np.asarray(self.base.generate(frame, year, rng))

        incumbent, seed = self._delegate_even(frame, year, rng)
        refresh_rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, SUBSTREAM_CODES["memory-refresh-gate"]]
            )
        )
        stable_rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, SUBSTREAM_CODES["memory-refresh-rank"]]
            )
        )

        order = np.argsort(
            frame["person_id"].to_numpy(dtype=np.int64), kind="stable"
        )
        sorted_frame = frame.iloc[order]
        sorted_incumbent = incumbent[order]
        lag_positive = (
            sorted_frame["gen_earn_w2"].to_numpy(dtype=np.float64) > 0
        )
        eligible = np.flatnonzero((sorted_incumbent > 0) & lag_positive)
        refresh_uniforms = refresh_rng.random(len(eligible))
        stable_uniforms = stable_rng.random(len(eligible))

        stable_ranks = np.full(len(eligible), np.nan, dtype=np.float64)
        if len(eligible):
            ages = sorted_frame["age"].to_numpy(dtype=np.float64)
            anchor_levels = sorted_frame["realized_earn_2014"].to_numpy(
                dtype=np.float64
            )
            anchor_ranks = self.base._anchor_ranks(anchor_levels, ages, year)
            anchor_is_zero = anchor_levels == 0
            target_bins = fe.age_bin(ages)
            for bin_index in range(fe.N_AGE_BINS):
                pool = self.stable_pools[bin_index]
                for is_q0 in (False, True):
                    selected_positions = np.flatnonzero(
                        (target_bins[eligible] == bin_index)
                        & (anchor_is_zero[eligible] == is_q0)
                    )
                    if not len(selected_positions):
                        continue
                    query_indices = eligible[selected_positions]
                    coordinate = self.base._third_coordinate(pool, is_q0)
                    distance = np.abs(
                        coordinate[None, :]
                        - anchor_ranks[query_indices][:, None]
                    )
                    stable_ranks[selected_positions] = fe._knn_draw(
                        distance,
                        np.asarray(pool["weight"]),
                        np.asarray(pool["u_tp2"]),
                        stable_uniforms[selected_positions],
                    )
        if len(eligible) and not np.isfinite(stable_ranks).all():
            raise AssertionError(
                "stable donor draw left an eligible rank unset"
            )

        refreshed = refresh_uniforms < self.q
        if self.trace_incumbent:
            self.refresh_records.append(
                {
                    "person_id": int(frame.iloc[0]["person_id"]),
                    "year": year,
                    "parent_bridge_seed": seed,
                    "eligible_count": int(len(eligible)),
                    "refreshed_count": int(refreshed.sum()),
                    "stream_final_state_sha256": {
                        "memory-refresh-gate": _rng_state_checksum(
                            refresh_rng
                        ),
                        "memory-refresh-rank": _rng_state_checksum(stable_rng),
                    },
                }
            )

        if self.q == 0:
            return incumbent
        output = np.asarray(incumbent, dtype=np.float64).copy()
        if refreshed.any():
            selected_sorted = eligible[refreshed]
            selected_original = order[selected_sorted]
            ages = sorted_frame["age"].to_numpy(dtype=np.float64)
            output[selected_original] = self.base.rank_to_level(
                stable_ranks[refreshed], ages[selected_sorted], year
            )
        return output


@dataclass(frozen=True)
class BoundaryContext:
    boundary: int
    full_anchor: pd.DataFrame
    domain_ids: frozenset[int]
    initial_slice: pd.DataFrame
    truth_support: pd.DataFrame
    truth_cells: Mapping[str, Mapping[str, Any]]
    floor: Mapping[str, Mapping[str, Any]]
    floor_gate_seed_detail: Sequence[Mapping[str, Any]]
    standardizers: Mapping[str, float]
    support_audit: Mapping[str, Any]
    rng_manifest: Mapping[str, Any]


def _participation_gate_audit(
    fitted_gate: Any | None,
    all_pairs: pd.DataFrame,
    *,
    name: str,
) -> dict[str, Any]:
    """Hash exact fitted HGB state and its canonical operative surface."""
    if fitted_gate is None:
        return {"name": name, "present": False}
    target_models = getattr(fitted_gate, "_target_models", None)
    if target_models is None or "earnings_tp2" not in target_models:
        raise RuntimeError(f"{name} lacks the certified earnings target model")
    target = target_models["earnings_tp2"]
    if tuple(target.columns) != ("earnings", "age_tp2"):
        raise RuntimeError(
            f"{name} predictor columns drifted: {tuple(target.columns)}"
        )
    gate = target.gate
    if gate is None:
        raise RuntimeError(f"{name} has no fitted participation gate")
    model_name = f"{type(gate).__module__}.{type(gate).__qualname__}"
    metadata = {
        "name": name,
        "present": True,
        "regime": str(target.regime),
        "columns": list(target.columns),
        "gate_class": model_name,
    }
    ordered = all_pairs.sort_values(["person_id", "period_tp2"], kind="stable")
    features = np.ascontiguousarray(
        ordered.loc[:, list(target.columns)].to_numpy(dtype="<f8")
    )
    classes = np.ascontiguousarray(np.asarray(gate.classes_, dtype="<i8"))
    probability = np.ascontiguousarray(
        np.asarray(gate.predict_proba(features), dtype="<f8")
    )
    digest = hashlib.sha256(_canonical_bytes(metadata))
    for array in (features, classes, probability):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return {
        **metadata,
        "gate_state_pickle_protocol": 5,
        "gate_state_sha256": hashlib.sha256(
            pickle.dumps(gate, protocol=5)
        ).hexdigest(),
        "canonical_surface_rows": int(len(features)),
        "canonical_surface_sha256": digest.hexdigest(),
        "classes": classes.tolist(),
    }


def _fit_audit(
    fitted: EarningsChainedRefit,
    stable_audit: Mapping[str, Any],
    boundary: int,
) -> dict[str, Any]:
    _assert_at_most(
        fitted.estimation_panel,
        "period",
        boundary,
        "fitted_estimation_panel",
    )
    _assert_at_most(
        fitted.forward_pairs,
        "period_tp2",
        boundary,
        "fitted_forward_pairs",
    )
    if set(fitted.anchors["period"].astype(int)) != {boundary}:
        raise AssertionError("fitted anchors do not equal the pseudo-boundary")
    pool_audit: dict[str, Any] = {}
    for name, pool in fitted.generator.pools.items():
        if np.any(np.asarray(pool["period_tp2"], dtype=np.int64) > boundary):
            raise AssertionError(
                f"{name} donor pool escaped boundary {boundary}"
            )
        pool_audit[name] = {
            "n_rows": int(len(pool["person_id"])),
            "checksum": _array_mapping_checksum(pool),
            "max_period_tp2": int(
                np.max(np.asarray(pool["period_tp2"], dtype=np.int64))
            ),
        }
    marginal_payload = {
        str(index): {
            "p0": cell.p0,
            "wtil": cell.wtil,
            "yval": cell.yval,
            "n_pos": cell.n_pos,
            "w_total": cell.w_total,
        }
        for index, cell in fitted.generator.marginals.items()
    }
    permanent = pd.DataFrame(
        sorted(fitted.generator.u_w_by_person.items()),
        columns=["person_id", "u_w"],
    )
    sign_path = asdict(verify_external_sign_path(fitted.generator))
    participation_gates = {
        "shared_gate": _participation_gate_audit(
            fitted.generator.shared_gate,
            fitted.forward_pairs,
            name="shared_gate",
        ),
        "zero_anchor_gate": _participation_gate_audit(
            fitted.generator.zero_anchor_gate,
            fitted.forward_pairs,
            name="zero_anchor_gate",
        ),
    }
    record = {
        "fit_seed": fitted.seed,
        "boundary": boundary,
        "spec_registration": fitted.spec_registration,
        "adapter_spec_sha256": fitted.adapter_spec_sha256,
        "provenance": asdict(fitted.provenance),
        "n_zero_anchor_pairs": fitted.n_zero_anchor_pairs,
        "checksums": {
            "estimation_rows": _frame_checksum(
                fitted.estimation_panel,
                ("person_id", "period", "earnings", "age", "weight"),
            ),
            "forward_pairs": _frame_checksum(
                fitted.forward_pairs,
                (
                    "person_id",
                    "period",
                    "period_tp2",
                    "earnings",
                    "earnings_tp2",
                    "age",
                    "age_tp2",
                    "weight",
                    "weight_tp2",
                ),
            ),
            "anchors": _frame_checksum(
                fitted.anchors,
                ("person_id", "period", "earnings", "age", "weight"),
            ),
            "u_w": _frame_checksum(permanent, ("person_id", "u_w")),
            "marginals": _canonical_sha256(marginal_payload),
            "wage_index": _canonical_sha256(
                {
                    "actual": fitted.generator.wage_index.actual,
                    "intercept": fitted.generator.wage_index.intercept,
                    "slope": fitted.generator.wage_index.slope,
                    "boundary": fitted.generator.wage_index.boundary_year,
                    "projected": {
                        str(
                            boundary + 2
                        ): fitted.generator.wage_index.projected(boundary + 2),
                        str(
                            boundary + 4
                        ): fitted.generator.wage_index.projected(boundary + 4),
                    },
                }
            ),
        },
        "donor_pools": pool_audit,
        "stable_pools": stable_audit,
        "u_w_diagnostics": fitted.u_w_diagnostics,
        "participation_gates": participation_gates,
        "external_sign_path": sign_path,
        "empty_pool_check_passed": True,
    }
    record["q_invariant_fit_signature_sha256"] = _canonical_sha256(record)
    return record


def _boundary_context(
    fitted: EarningsChainedRefit,
    earnings: pd.DataFrame,
    anchor_demo: pd.DataFrame,
    boundary: int,
) -> BoundaryContext:
    anchor_wave = boundary + 1
    full_anchor = anchor_demo.loc[
        anchor_demo["period"] == anchor_wave,
        ["person_id", "interview", "weight"],
    ].copy()
    full_anchor = full_anchor.rename(columns={"interview": "household_id"})
    full_anchor = full_anchor.sort_values(
        "person_id", kind="stable"
    ).reset_index(drop=True)
    if full_anchor.empty or full_anchor["person_id"].duplicated().any():
        raise ValueError(f"full {boundary} anchor is empty or duplicated")
    if (full_anchor["weight"] <= 0).any():
        raise ValueError(f"full {boundary} anchor has non-positive weights")

    fitted_domain = earnings_domain_person_ids(fitted.generator)
    domain_ids = frozenset(
        int(value)
        for value in set(full_anchor["person_id"].astype(int)) & fitted_domain
    )
    fitted_anchor_ids = frozenset(fitted.anchors["person_id"].astype(int))
    if domain_ids != fitted_anchor_ids:
        missing = sorted(fitted_anchor_ids - domain_ids)[:10]
        extra = sorted(domain_ids - fitted_anchor_ids)[:10]
        raise AssertionError(
            f"full-anchor/domain mismatch missing={missing} extra={extra}"
        )

    fixed_weight = full_anchor.set_index("person_id")["weight"]
    periods = (boundary, boundary + 2, boundary + 4)
    truth = earnings.loc[
        earnings["period"].isin(periods)
        & earnings["person_id"].isin(domain_ids)
    ].copy()
    truth["raw_row_weight"] = truth["weight"].astype("float64")
    truth["weight"] = truth["person_id"].map(fixed_weight).astype("float64")
    truth["cohort"] = truth["age"].map(_cohort)
    truth = truth[truth["cohort"].notna()].copy()
    truth = truth.sort_values(
        ["person_id", "period"], kind="stable"
    ).reset_index(drop=True)
    if truth.duplicated(["person_id", "period"]).any():
        raise ValueError("truth support has duplicate person-period rows")
    endpoint = truth[truth["period"].isin((boundary + 2, boundary + 4))]
    if (
        endpoint.empty
        or not endpoint["age"].between(fe.AGE_MIN, fe.AGE_MAX).all()
    ):
        raise ValueError("truth endpoint support is empty or outside 25-64")

    truth_cells = _selected_cells(truth, boundary)
    for name, record in truth_cells.items():
        value = _selection_cell_value(record)
        if value is None:
            raise ValueError(f"truth cell {name!r} is undefined")

    def compute(person_ids: set[object]) -> dict[str, Any]:
        selected_ids = set(int(value) for value in person_ids) & domain_ids
        return earnings_cells(
            truth[truth["person_id"].isin(selected_ids)],
            level_years=(boundary + 2, boundary + 4),
            change_years=(boundary, boundary + 2, boundary + 4),
        )

    floor, floor_detail = run_floor(full_anchor, compute, "person_id")
    selected_floor = {name: floor[name] for name in SELECTED_CELLS}
    standardizers: dict[str, float] = {}
    for name, record in selected_floor.items():
        sigma = float(record["realized_sigma"])
        if (
            int(record["n_defined_seeds"]) != len(FLOOR_SEEDS)
            or not math.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError(
                f"floor standardizer {name!r} is not positive over all seeds"
            )
        standardizers[name] = sigma

    anchor_state = fitted.anchors[
        fitted.anchors["person_id"].isin(domain_ids)
    ][["person_id", "age"]].copy()
    anchor_state = anchor_state.sort_values("person_id", kind="stable")
    initial_slice = anchor_state.assign(
        year=boundary,
        sex=SCHEMA_SEX_SENTINEL,
        **{EARNINGS_DOMAIN_COLUMN: True},
    )

    ordinals = sorted(int(value) for value in full_anchor["person_id"])
    rng_manifest = {
        "draw_seed_base": DRAW_SEED_BASE,
        "selection_draw_seeds": SELECTION_DRAW_SEEDS,
        "selection_draw_indices": [
            seed - DRAW_SEED_BASE for seed in SELECTION_DRAW_SEEDS
        ],
        "gate_m6_tag": GATE_M6_TAG,
        "module_order": [module.value for module in MODULE_ORDER],
        "earnings_module_index": list(MODULE_ORDER).index(
            ProjectionModule.EARNINGS
        ),
        "period_indices": {
            str(boundary + offset): offset for offset in range(1, 5)
        },
        "n_periods": 4,
        "substream_codes": SUBSTREAM_CODES,
        "full_anchor_ordinal_ids_sha256": _id_checksum(ordinals),
        "n_full_anchor_ordinals": len(ordinals),
    }
    rng_manifest["sha256"] = _canonical_sha256(rng_manifest)
    support_audit = {
        "full_anchor_definition": (
            "positive-weight demographic roster at exact collection wave b+1"
        ),
        "anchor_wave": anchor_wave,
        "n_full_anchor": int(len(full_anchor)),
        "full_anchor_ids_sha256": _id_checksum(full_anchor["person_id"]),
        "n_domain": int(len(domain_ids)),
        "domain_ids_sha256": _id_checksum(domain_ids),
        "split_before_domain_intersection": True,
        "fixed_boundary_anchor_weight": True,
        "truth_support_ids_sha256": _key_checksum(truth),
        "truth_support_rows": int(len(truth)),
        "truth_support_rows_by_period": {
            str(year): int((truth["period"] == year).sum()) for year in periods
        },
        "endpoint_support_ids_sha256": _key_checksum(endpoint),
        "endpoint_support_rows": int(len(endpoint)),
        "support_age_min": int(endpoint["age"].min()),
        "support_age_max": int(endpoint["age"].max()),
        "truth_frame_checksum": _frame_checksum(
            truth,
            ("person_id", "period", "earnings", "age", "weight", "cohort"),
        ),
        "sex_frame_value": SCHEMA_SEX_SENTINEL,
        "sex_is_schema_only": True,
    }
    return BoundaryContext(
        boundary=boundary,
        full_anchor=full_anchor,
        domain_ids=domain_ids,
        initial_slice=initial_slice,
        truth_support=truth,
        truth_cells=truth_cells,
        floor=selected_floor,
        floor_gate_seed_detail=floor_detail,
        standardizers=standardizers,
        support_audit=support_audit,
        rng_manifest=rng_manifest,
    )


def _project(
    generator: Any,
    context: BoundaryContext,
    draw_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if draw_seed not in SELECTION_DRAW_SEEDS:
        raise ValueError("projection draw seed is outside the registered set")
    model = wrap_earnings_domain(generator)
    state = model.materialize_initial_frame(context.initial_slice.copy())
    state = state.sort_values("person_id", kind="stable").reset_index(
        drop=True
    )
    full_ids = sorted(int(value) for value in context.full_anchor["person_id"])
    ordinals = {person_id: index for index, person_id in enumerate(full_ids)}
    if not context.domain_ids.issubset(ordinals):
        raise AssertionError("earnings domain lies outside the full anchor")
    registry = ProjectionRNGRegistry(
        draw_index=draw_seed - DRAW_SEED_BASE,
        n_periods=4,
    )
    annual = [
        state[["person_id", "earnings"]]
        .assign(period=context.boundary)
        .loc[:, ["person_id", "period", "earnings"]]
    ]
    for period_index, year in enumerate(
        range(context.boundary + 1, context.boundary + 5), start=1
    ):
        state = state.copy()
        state["age"] = state["age"].to_numpy(dtype=np.int64) + 1
        state["year"] = year
        period_context = PeriodContext(
            period_index=period_index,
            year=year,
            draw_index=draw_seed - DRAW_SEED_BASE,
            metadata={},
            rng_registry=registry,
            person_ordinals=ordinals,
        )
        state = apply_earnings(
            state,
            period_context,
            registry.generator(period_index, ProjectionModule.EARNINGS),
            model=model,
        )
        annual.append(
            state[["person_id", "earnings"]]
            .assign(period=year)
            .loc[:, ["person_id", "period", "earnings"]]
        )
    annual_frame = pd.concat(annual, ignore_index=True)
    scored_periods = (
        context.boundary,
        context.boundary + 2,
        context.boundary + 4,
    )
    scored = annual_frame[annual_frame["period"].isin(scored_periods)].copy()
    support_columns = [
        column
        for column in context.truth_support.columns
        if column != "earnings"
    ]
    scored = scored.merge(
        context.truth_support[support_columns],
        on=["person_id", "period"],
        how="inner",
        validate="one_to_one",
    )
    scored, truth = restrict_earnings_domain_support(
        scored,
        context.truth_support,
        context.domain_ids,
        periods=scored_periods,
    )
    if _key_checksum(scored) != _key_checksum(truth):
        raise AssertionError("truth/projection support checksum differs")
    return scored, annual_frame


def _q0_equivalence(
    incumbent_scored: pd.DataFrame,
    incumbent_annual: pd.DataFrame,
    candidate_scored: pd.DataFrame,
    candidate_annual: pd.DataFrame,
    incumbent_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    refresh_records: Sequence[Mapping[str, Any]],
    boundary: int,
    draw_seed: int,
) -> dict[str, Any]:
    incumbent_annual = incumbent_annual.sort_values(
        ["person_id", "period"], kind="stable"
    ).reset_index(drop=True)
    candidate_annual = candidate_annual.sort_values(
        ["person_id", "period"], kind="stable"
    ).reset_index(drop=True)
    keys_equal = incumbent_annual[["person_id", "period"]].equals(
        candidate_annual[["person_id", "period"]]
    )
    level_bytes_equal = (
        incumbent_annual["earnings"].to_numpy(dtype=np.float64).tobytes()
        == candidate_annual["earnings"].to_numpy(dtype=np.float64).tobytes()
    )
    participation_equal = np.array_equal(
        incumbent_annual["earnings"].to_numpy(dtype=np.float64) > 0,
        candidate_annual["earnings"].to_numpy(dtype=np.float64) > 0,
    )
    old_stream_states_equal = list(incumbent_records) == list(
        candidate_records
    )
    incumbent_cells = _selected_cells(incumbent_scored, boundary)
    candidate_cells = _selected_cells(candidate_scored, boundary)
    moments_equal = _canonical_bytes(incumbent_cells) == _canonical_bytes(
        candidate_cells
    )
    passed = all(
        (
            keys_equal,
            level_bytes_equal,
            participation_equal,
            old_stream_states_equal,
            moments_equal,
        )
    )
    if not passed:
        raise AssertionError(
            f"q=0 equivalence failed at boundary={boundary} seed={draw_seed}"
        )
    return {
        "draw_seed": draw_seed,
        "annual_rows": int(len(candidate_annual)),
        "person_period_keys_equal": keys_equal,
        "level_bytes_equal": level_bytes_equal,
        "participation_states_equal": participation_equal,
        "all_six_moments_equal": moments_equal,
        "streams_1_3_final_states_equal": old_stream_states_equal,
        "incumbent_level_sha256": _level_checksum(incumbent_annual),
        "candidate_level_sha256": _level_checksum(candidate_annual),
        "incumbent_participation_sha256": _participation_checksum(
            incumbent_annual
        ),
        "candidate_participation_sha256": _participation_checksum(
            candidate_annual
        ),
        "old_stream_trace_sha256": _canonical_sha256(incumbent_records),
        "new_stream_trace_sha256": _canonical_sha256(refresh_records),
        "n_incumbent_person_period_calls": len(incumbent_records),
        "n_refresh_period_records": len(refresh_records),
        "passed": True,
    }


def _run_boundary_draws(
    fitted: EarningsChainedRefit,
    stable_pools: Mapping[int, Mapping[str, np.ndarray]],
    context: BoundaryContext,
    q: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    draws: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for draw_number, draw_seed in enumerate(SELECTION_DRAW_SEEDS, start=1):
        candidate = RankRefreshPrototype(
            base=fitted.generator,
            q=q,
            stable_pools=stable_pools,
            trace_incumbent=(q == 0),
        )
        candidate_scored, candidate_annual = _project(
            candidate, context, draw_seed
        )
        if q == 0:
            incumbent = _TracedIncumbentGenerator(fitted.generator)
            incumbent_scored, incumbent_annual = _project(
                incumbent, context, draw_seed
            )
            equivalence.append(
                _q0_equivalence(
                    incumbent_scored,
                    incumbent_annual,
                    candidate_scored,
                    candidate_annual,
                    incumbent.records,
                    candidate.incumbent_records,
                    candidate.refresh_records,
                    context.boundary,
                    draw_seed,
                )
            )
        cells = _selected_cells(candidate_scored, context.boundary)
        values = {
            name: _selection_cell_value(record)
            for name, record in cells.items()
        }
        draws.append(
            {
                "draw_seed": draw_seed,
                "moments": cells,
                "moment_values": values,
                "support_ids_sha256": _key_checksum(candidate_scored),
                "annual_level_sha256": _level_checksum(candidate_annual),
                "annual_participation_sha256": _participation_checksum(
                    candidate_annual
                ),
                "fresh_initial_state": True,
            }
        )
        if draw_number % 5 == 0:
            _progress(
                f"q={q:.2f} boundary={context.boundary}: "
                f"completed {draw_number}/20 draws"
            )
    return draws, equivalence


def _aggregate_boundary(
    boundary_record: Mapping[str, Any],
    draw_seeds: Sequence[int],
) -> dict[str, Any]:
    by_seed = {
        int(record["draw_seed"]): record
        for record in boundary_record["per_draw"]
    }
    selected = [by_seed[int(seed)] for seed in draw_seeds]
    truth = boundary_record["truth_moments"]
    standardizers = boundary_record["standardizers"]
    projected: dict[str, float | None] = {}
    scores: dict[str, float | None] = {}
    standardized: dict[str, float | None] = {}
    for name in SELECTED_CELLS:
        values = [record["moment_values"].get(name) for record in selected]
        if any(value is None for value in values):
            projected[name] = None
            scores[name] = None
            standardized[name] = None
            continue
        mean = float(np.mean(np.asarray(values, dtype=np.float64)))
        score = _score(mean, truth[name])
        projected[name] = mean
        scores[name] = score
        standardized[name] = (
            None if score is None else score / float(standardizers[name])
        )
    contributions = {
        name: (
            None
            if standardized[name] is None
            else float(standardized[name]) ** 2
        )
        for name in OBJECTIVE_CELLS
    }
    objective = (
        None
        if any(value is None for value in contributions.values())
        else float(sum(float(value) for value in contributions.values()))
    )
    return {
        "draw_seeds": list(draw_seeds),
        "projected_moments": projected,
        "scores": scores,
        "standardized_scores": standardized,
        "objective_contributions": contributions,
        "objective": objective,
    }


def _objective_across_boundaries(
    rung: Mapping[str, Any], draw_seeds: Sequence[int]
) -> dict[str, Any]:
    by_boundary: dict[str, float | None] = {}
    for boundary in PSEUDO_BOUNDARIES:
        aggregate = _aggregate_boundary(
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


def _finalize_selector(payload: dict[str, Any]) -> None:
    rungs = payload["rungs"]
    for q in Q_GRID:
        rung = rungs[f"{q:.2f}"]
        invalid_reasons: list[str] = []
        for boundary in PSEUDO_BOUNDARIES:
            record = rung["boundaries"][str(boundary)]
            full = _aggregate_boundary(record, SELECTION_DRAW_SEEDS)
            first = _aggregate_boundary(record, FIRST_HALF_DRAW_SEEDS)
            second = _aggregate_boundary(record, SECOND_HALF_DRAW_SEEDS)
            record["aggregates"] = {
                "all_20": full,
                "first_10": first,
                "second_10": second,
            }
            regeneration: dict[str, bool] = {}
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
            if full["objective"] is None:
                invalid_reasons.append(
                    f"boundary {boundary} has undefined all-draw objective"
                )
            if q == 0 and not record["q0_equivalence"]["passed"]:
                invalid_reasons.append(
                    f"boundary {boundary} q0 equivalence failed"
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

    baseline = rungs["0.00"]
    if not baseline["valid"]:
        raise RuntimeError(
            "global q=0 validity checks failed: "
            + "; ".join(baseline["invalid_reasons"])
        )
    for q in Q_GRID:
        label = f"{q:.2f}"
        rung = rungs[label]
        guards: dict[str, Any] = {}
        feasible = bool(rung["valid"])
        for boundary in PSEUDO_BOUNDARIES:
            guard_cells: dict[str, Any] = {}
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
                    "q0_standardized_score": incumbent,
                    "limit": None if incumbent is None else incumbent + 1.0,
                    "passed": passed,
                }
                feasible &= passed
            guards[str(boundary)] = guard_cells
        rung["feasibility_guards"] = guards
        rung["feasible"] = feasible
        if q == 0:
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
                    and candidate_total < baseline_total
                )
            retained = feasible and all(improves.values())
        rung["strict_improvement_vs_q0"] = improves
        rung["retained_for_one_se"] = retained

    retained_labels = [
        f"{q:.2f}" for q in Q_GRID if rungs[f"{q:.2f}"]["retained_for_one_se"]
    ]
    q_min_label = min(
        retained_labels,
        key=lambda label: (
            float(rungs[label]["objectives"]["all_20"]["total"]),
            float(label),
        ),
    )
    q_min_deletes = np.asarray(
        [
            record["total"]
            for record in rungs[q_min_label]["objectives"]["delete_one"]
        ],
        dtype=np.float64,
    )
    q_min_delete_mean = float(q_min_deletes.mean())
    standard_error = float(
        np.sqrt(
            (19.0 / 20.0) * np.sum((q_min_deletes - q_min_delete_mean) ** 2)
        )
    )
    q_min_objective = float(
        rungs[q_min_label]["objectives"]["all_20"]["total"]
    )
    cutoff = q_min_objective + standard_error
    within_one_se = [
        label
        for label in retained_labels
        if float(rungs[label]["objectives"]["all_20"]["total"]) <= cutoff
    ]
    selected_label = min(within_one_se, key=float)
    for label, rung in rungs.items():
        rung["within_one_se_cutoff"] = label in within_one_se
        rung["selected"] = label == selected_label

    weak_retained_labels = ["0.00"]
    for q in Q_GRID[1:]:
        label = f"{q:.2f}"
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
    weak_q_min_label = min(
        weak_retained_labels,
        key=lambda label: (
            float(rungs[label]["objectives"]["all_20"]["total"]),
            float(label),
        ),
    )
    weak_deletes = np.asarray(
        [
            record["total"]
            for record in rungs[weak_q_min_label]["objectives"]["delete_one"]
        ],
        dtype=np.float64,
    )
    weak_se = float(
        np.sqrt(
            (19.0 / 20.0) * np.sum((weak_deletes - weak_deletes.mean()) ** 2)
        )
    )
    weak_cutoff = (
        float(rungs[weak_q_min_label]["objectives"]["all_20"]["total"])
        + weak_se
    )
    weak_selected_label = min(
        (
            label
            for label in weak_retained_labels
            if float(rungs[label]["objectives"]["all_20"]["total"])
            <= weak_cutoff
        ),
        key=float,
    )
    if weak_selected_label != selected_label:
        raise AssertionError(
            "strict-versus-weak all-draw improvement changed the selected q"
        )

    payload["selector"] = {
        "baseline_q_retained": True,
        "effective_search_size": {
            "grid_rungs": len(Q_GRID),
            "valid_rungs": sum(rung["valid"] for rung in rungs.values()),
            "feasible_rungs_including_q0": sum(
                rung["feasible"] for rung in rungs.values()
            ),
            "retained_rungs_including_q0": len(retained_labels),
            "retained_nonzero_rungs": sum(
                float(label) > 0 for label in retained_labels
            ),
        },
        "retained_q": [float(label) for label in retained_labels],
        "q_min": float(q_min_label),
        "q_min_objective": q_min_objective,
        "q_min_delete_one_mean": q_min_delete_mean,
        "q_min_jackknife_standard_error": standard_error,
        "one_se_cutoff": cutoff,
        "q_within_one_se": [float(label) for label in within_one_se],
        "selected_q": float(selected_label),
        "selected_q_label": selected_label,
        "strict_vs_weak_improvement_outcome_invariant": (
            weak_selected_label == selected_label
        ),
        "weak_improvement_counterfactual": {
            "weakened_comparison": "all_20 only; fixed halves remain strict",
            "retained_q": [float(label) for label in weak_retained_labels],
            "q_min": float(weak_q_min_label),
            "jackknife_standard_error": weak_se,
            "one_se_cutoff": weak_cutoff,
            "selected_q": float(weak_selected_label),
        },
        "q0_dispatch_disposition": (
            "q=0 is a valid selector outcome; lock-addendum posture remains "
            "DRAFT_NOT_OPERATIVE pending referee/orchestrator ratification"
        ),
    }


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run the frozen amendment-4 q-star selector; JSON stdout, "
            "progress stderr. There are no selector-changing options."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser().parse_args(argv)
    if fe.SUBSTREAM_CODES != {
        "gate": 1,
        "donor-draw": 2,
        "re-entry-draw": 3,
    }:
        raise RuntimeError("incumbent earnings substream registry drifted")
    if len(Q_GRID) != 21 or Q_GRID[0] != 0 or Q_GRID[-1] != 1:
        raise RuntimeError("registered Q grid drifted")
    if set(FLOOR_SEEDS) != set(range(100)):
        raise RuntimeError("registered floor seeds drifted")

    freeze = _repository_freeze()
    with redirect_stdout(sys.stderr):
        earnings, anchor_demo, nawi_path, source_audit = (
            _load_train_only_sources()
        )
    payload: dict[str, Any] = {
        "schema": RAW_SCHEMA,
        "freeze": freeze,
        "protocol": {
            "authority": (
                "docs/design/m6_projection_engine.md section 2.7.7, "
                "amendment merge 64ec6c04bf8f3e6a6f4fcaf71c086a128056a86f"
            ),
            "design_commit_flip": ("43fd65eedc225555ac368e24b9510446ca85e1b3"),
            "q_grid": Q_GRID,
            "pseudo_boundaries": PSEUDO_BOUNDARIES,
            "fit_seed": FIT_SEED,
            "selection_draw_seeds": SELECTION_DRAW_SEEDS,
            "fixed_halves": [
                FIRST_HALF_DRAW_SEEDS,
                SECOND_HALF_DRAW_SEEDS,
            ],
            "floor_seeds": FLOOR_SEEDS,
            "selected_cells": SELECTED_CELLS,
            "objective_cells": OBJECTIVE_CELLS,
            "feasibility_cells": FEASIBILITY_CELLS,
            "qrf_refit_per_q_boundary": True,
            "common_random_numbers_across_rungs_at_fixed_seed": True,
            "draw_seed_to_registry_index": "draw_index = draw_seed - 5200",
            "substream_codes": SUBSTREAM_CODES,
            "one_se_rule": (
                "sqrt((19/20) * sum((J_-r - mean(J_-r))^2)); "
                "smallest retained q at or below J(q_min)+SE"
            ),
            "no_gate_score": True,
            "no_candidate_1_or_candidate_2_artifact_read": True,
            "no_runs_write": True,
        },
        "implementation": {
            "runner": str(Path(__file__).resolve().relative_to(ROOT)),
            "full_boundary_anchor": (
                "all positive-weight demographic persons at exact wave b+1; "
                "split and ordinal assignment precede earnings-domain intersection"
            ),
            "sex_frame": (
                "fixed nonmissing schema-only sentinel; sex is absent from "
                "every numerical earnings input"
            ),
            "refresh_wrapper": (
                "advance incumbent parent bridge once, replay exact seed into "
                "unchanged incumbent generate, then isolated codes 4 and 5"
            ),
            "stable_pool": (
                "positive-to-positive pair donors partitioned by exact target "
                "five-year age bin; incumbent order, ranks, weights, kNN helper"
            ),
            "fixed_weight": "exact b+1 full-anchor weight held across the window",
            "progress_stream": "stderr",
            "result_stream": "strict JSON stdout only",
        },
        "source_audit": source_audit,
        "boundaries": {},
        "rungs": {},
    }
    contexts: dict[int, BoundaryContext] = {}
    baseline_signatures: dict[int, str] = {}

    for q_index, q in enumerate(Q_GRID, start=1):
        label = f"{q:.2f}"
        _progress(f"starting q={label} ({q_index}/21)")
        rung: dict[str, Any] = {"q": q, "boundaries": {}}
        for boundary_index, boundary in enumerate(PSEUDO_BOUNDARIES, start=1):
            _progress(
                f"q={label} boundary={boundary} ({boundary_index}/3): "
                "fresh complete QRF refit"
            )
            fit_input = truncate_estimation_frame(
                earnings,
                boundary_year=boundary,
                year_column="period",
                flow=False,
                label=f"q={label} boundary={boundary} earnings",
            )
            _assert_at_most(
                fit_input, "period", boundary, "boundary_fit_input"
            )
            boundary_nawi, nawi_read_audit = _read_historical_nawi(
                nawi_path, maximum_year=boundary
            )
            expected_nawi = EXPECTED_BOUNDARY_NAWI[boundary]
            if (
                nawi_read_audit["bytes_consumed_through_maximum_key"]
                != expected_nawi["prefix_bytes"]
                or nawi_read_audit["admitted_prefix_sha256"]
                != expected_nawi["prefix_sha256"]
                or _canonical_sha256(boundary_nawi)
                != expected_nawi["mapping_sha256"]
            ):
                raise RuntimeError(
                    f"certified NAWI prefix through {boundary} drifted"
                )
            if not boundary_nawi or max(boundary_nawi) != boundary:
                raise AssertionError(
                    "boundary NAWI mapping escaped its cutoff"
                )
            if any(
                not math.isfinite(value) or value <= 0
                for value in boundary_nawi.values()
            ):
                raise ValueError("boundary NAWI contains non-positive values")
            required_nawi = set(range(boundary - 9, boundary + 1))
            if missing := sorted(required_nawi - set(boundary_nawi)):
                raise ValueError(
                    f"boundary {boundary} NAWI misses trailing years {missing}"
                )
            with redirect_stdout(sys.stderr):
                fitted = refit_earnings_chained_generator(
                    fit_input,
                    boundary_nawi,
                    seed=FIT_SEED,
                    boundary_year=boundary,
                )
            stable_pools, stable_audit = _stable_pools(fitted)
            fit_audit = _fit_audit(fitted, stable_audit, boundary)
            signature = fit_audit["q_invariant_fit_signature_sha256"]
            if q == 0:
                context = _boundary_context(
                    fitted, earnings, anchor_demo, boundary
                )
                contexts[boundary] = context
                baseline_signatures[boundary] = signature
                payload["boundaries"][str(boundary)] = {
                    "cutoff": boundary,
                    "fit_input_rows": int(len(fit_input)),
                    "fit_input_max_period": _max_year(fit_input, "period"),
                    "fit_input_checksum": _frame_checksum(
                        fit_input,
                        ("person_id", "period", "earnings", "age", "weight"),
                    ),
                    "nawi_key_max": max(boundary_nawi),
                    "nawi_checksum": _canonical_sha256(boundary_nawi),
                    "nawi_field_read": nawi_read_audit,
                    "truth_moments": context.truth_cells,
                    "floor": context.floor,
                    "floor_gate_seed_detail": context.floor_gate_seed_detail,
                    "standardizers": context.standardizers,
                    "support": context.support_audit,
                    "rng_registry": context.rng_manifest,
                }
            else:
                context = contexts[boundary]
                if signature != baseline_signatures[boundary]:
                    raise AssertionError(
                        f"q={label} boundary={boundary} fit/pool checksum "
                        "differs from q=0 despite fixed inputs and seed"
                    )

            draws, equivalence = _run_boundary_draws(
                fitted, stable_pools, context, q
            )
            q0_equivalence = {
                "required": q == 0,
                "passed": (
                    all(record["passed"] for record in equivalence)
                    if q == 0
                    else None
                ),
                "per_draw": equivalence,
            }
            rung["boundaries"][str(boundary)] = {
                "cutoff": boundary,
                "fit": fit_audit,
                "truth_moments": context.truth_cells,
                "floor": context.floor,
                "standardizers": context.standardizers,
                "support": context.support_audit,
                "rng_registry_sha256": context.rng_manifest["sha256"],
                "nawi_key_max": max(boundary_nawi),
                "nawi_checksum": _canonical_sha256(boundary_nawi),
                "nawi_field_read": nawi_read_audit,
                "per_draw": draws,
                "q0_equivalence": q0_equivalence,
            }
            _progress(
                f"q={label} boundary={boundary}: complete "
                f"(fit signature {signature[:12]})"
            )
        payload["rungs"][label] = rung
        _progress(f"completed q={label} ({q_index}/21)")

    _progress(
        "all 63 refits and 1,260 q-boundary draws complete; reducing selector"
    )
    _finalize_selector(payload)
    selected = payload["selector"]["selected_q_label"]
    _progress(f"selector complete: q*={selected}")
    print(
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
