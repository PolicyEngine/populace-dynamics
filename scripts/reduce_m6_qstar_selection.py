#!/usr/bin/env python3
"""Reduce exact q-star selector stdout to the public findings ledger.

Usage::

    python scripts/reduce_m6_qstar_selection.py \
      < m6-qstar-full.json \
      > docs/analysis/m6_qstar_train_only_selection_results.json

The reducer follows the PR-231 precedent: hash the exact input bytes, assert
the raw schema, remove only repetitive per-draw arrays, derive retained ranges,
undefined-draw indices, and checksums from those arrays, rename the schema, and
emit strict sorted JSON.
Every q-by-boundary aggregate, full/fixed-half/delete-one objective, fit/pool/
support/RNG checksum, standardizer, feasibility guard, and selection field is
retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

RAW_SCHEMA = "m6_qstar_train_only_selection.v1"
FINDINGS_SCHEMA = "m6_qstar_train_only_selection.findings.v1"
SELECTED_CELLS = (
    "earn_p10.prime",
    "earn_dlog_mean.prime",
    "earn_dlog_sd.older",
    "earn_mob_h1_diag",
    "earn_autocorr_lag2",
    "earn_zero_rate.older",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _draw_summary(
    draws: list[dict[str, Any]], *, truth_support_sha256: str
) -> dict[str, Any]:
    if len(draws) != 20:
        raise ValueError(f"expected 20 per-draw records, found {len(draws)}")
    seeds = [int(record["draw_seed"]) for record in draws]
    if seeds != list(range(6200, 6220)):
        raise ValueError(f"unexpected draw seed order: {seeds}")
    ranges: dict[str, dict[str, float | None]] = {}
    undefined: dict[str, list[int]] = {}
    for cell in SELECTED_CELLS:
        values = [record["moment_values"][cell] for record in draws]
        bad = [
            seeds[index]
            for index, value in enumerate(values)
            if value is None or not math.isfinite(float(value))
        ]
        numeric = [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        undefined[cell] = bad
        ranges[cell] = {
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
        }
    support_hashes = sorted(
        {str(record["support_ids_sha256"]) for record in draws}
    )
    if support_hashes != [truth_support_sha256]:
        raise ValueError(
            "projected support hashes do not equal the boundary truth "
            f"support hash: {support_hashes} != {[truth_support_sha256]}"
        )
    return {
        "n_draws": len(draws),
        "draw_seeds": seeds,
        "moment_range": ranges,
        "undefined_draw_seeds_by_cell": undefined,
        "all_cells_defined": not any(undefined.values()),
        "records_sha256": hashlib.sha256(_canonical_bytes(draws)).hexdigest(),
        "distinct_annual_level_surfaces": len(
            {record["annual_level_sha256"] for record in draws}
        ),
        "distinct_annual_participation_surfaces": len(
            {record["annual_participation_sha256"] for record in draws}
        ),
        "all_fresh_initial_state": all(
            bool(record["fresh_initial_state"]) for record in draws
        ),
        "projected_support_ids_sha256": support_hashes[0],
        "truth_projection_support_equal_all_draws": True,
    }


def _equivalence_summary(
    records: list[dict[str, Any]], *, required: bool
) -> dict[str, Any]:
    if not required:
        if records:
            raise ValueError(
                "nonzero q unexpectedly carries q=0 preflight rows"
            )
        return {"required": False, "passed": None, "n_draws": 0}
    if len(records) != 20:
        raise ValueError(
            f"q=0 preflight expected 20 draw records, found {len(records)}"
        )
    seeds = [int(record["draw_seed"]) for record in records]
    if seeds != list(range(6200, 6220)):
        raise ValueError(f"unexpected q=0 equivalence seed order: {seeds}")
    booleans = (
        "person_period_keys_equal",
        "level_bytes_equal",
        "participation_states_equal",
        "all_six_moments_equal",
        "streams_1_3_final_states_equal",
        "passed",
    )
    return {
        "required": True,
        "passed": all(
            all(bool(record[field]) for field in booleans)
            for record in records
        ),
        "n_draws": len(records),
        "draw_seeds": seeds,
        "records_sha256": hashlib.sha256(
            _canonical_bytes(records)
        ).hexdigest(),
        "n_person_period_calls": sum(
            int(record["n_incumbent_person_period_calls"])
            for record in records
        ),
        "n_refresh_period_records": sum(
            int(record["n_refresh_period_records"]) for record in records
        ),
        "old_stream_trace_sha256": hashlib.sha256(
            _canonical_bytes(
                [record["old_stream_trace_sha256"] for record in records]
            )
        ).hexdigest(),
        "new_stream_trace_sha256": hashlib.sha256(
            _canonical_bytes(
                [record["new_stream_trace_sha256"] for record in records]
            )
        ).hexdigest(),
    }


def reduce(raw: bytes) -> dict[str, Any]:
    ledger = json.loads(raw)
    if ledger.get("schema") != RAW_SCHEMA:
        raise ValueError(f"unexpected raw schema {ledger.get('schema')!r}")
    q_grid = [float(value) for value in ledger["protocol"]["q_grid"]]
    if q_grid != [round(index * 0.05, 2) for index in range(21)]:
        raise ValueError("raw ledger Q grid drifted")
    if sorted(ledger["rungs"]) != [f"{value:.2f}" for value in q_grid]:
        raise ValueError("raw ledger rung keys do not match the Q grid")

    for q_label, rung in ledger["rungs"].items():
        for boundary in ("2006", "2008", "2010"):
            record = rung["boundaries"][boundary]
            draws = record.pop("per_draw")
            record["per_draw_summary"] = _draw_summary(
                draws,
                truth_support_sha256=record["support"][
                    "truth_support_ids_sha256"
                ],
            )
            if not record["per_draw_summary"]["all_cells_defined"] and bool(
                rung.get("valid")
            ):
                raise ValueError(
                    f"q={q_label} boundary={boundary} has undefined draws "
                    "but the raw rung is marked valid"
                )
            equivalence = record["q0_equivalence"]
            details = equivalence.pop("per_draw")
            summary = _equivalence_summary(
                details, required=float(q_label) == 0
            )
            if equivalence.get("passed") != summary["passed"]:
                raise ValueError(
                    f"q={q_label} boundary={boundary} preflight summary differs"
                )
            record["q0_equivalence"] = summary

    ledger["schema"] = FINDINGS_SCHEMA
    ledger["full_stdout_sha256"] = hashlib.sha256(raw).hexdigest()
    ledger["reducer"] = {
        "script": "scripts/reduce_m6_qstar_selection.py",
        "removed": (
            "21 x 3 x 20 repetitive projected-draw records and the "
            "3 x 20 q=0 per-draw equivalence records"
        ),
        "retained": (
            "all registered fit/pool/support/RNG checksums, truth moments, "
            "standardizers, aggregate projected moments and scores, full/half/"
            "delete-one objectives, guards, one-SE cutoff, and selection"
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
