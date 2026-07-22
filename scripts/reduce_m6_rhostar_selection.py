#!/usr/bin/env python3
"""Reduce exact rho-selector stdout to the public findings ledger."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

RAW_SCHEMA = "m6_rhostar_train_only_selection.v1"
FINDINGS_SCHEMA = "m6_rhostar_train_only_selection.findings.v1"
RHO_GRID = tuple(round(-0.80 + 0.05 * index, 2) for index in range(17))
DRAW_SEEDS = tuple(range(6200, 6220))
BOUNDARIES = (2006, 2008, 2010)
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
SUBSTREAM_CODES = {
    "gate": 1,
    "donor-draw": 2,
    "re-entry-draw": 3,
    "memory-refresh-gate": 4,
    "memory-refresh-rank": 5,
}
REQUIRED_FENCES = {
    "no_candidate_1_or_candidate_2_artifact_read": True,
    "no_gate_score": True,
    "no_runs_write": True,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _validate_preflights(ledger: dict[str, Any]) -> None:
    preflights = ledger.get("preflights", {})
    if not preflights.get("all_passed"):
        raise ValueError("raw ledger does not carry a passed preflight gate")
    if preflights.get("ladder_values_computed_before_pass"):
        raise ValueError("raw ledger violates preflight ordering")
    records = preflights.get("records", {})
    required = {
        "rho_zero_candidate2_equivalence",
        "reset_law_discriminating_fixture",
        "endogenous_participation_feedback",
        "object_level_unchanged",
    }
    if set(records) != required:
        raise ValueError("preflight record set drifted")
    if not all(bool(record.get("passed")) for record in records.values()):
        raise ValueError("one or more raw preflight records failed")
    equivalence = records["rho_zero_candidate2_equivalence"]
    if equivalence.get("n_boundary_draw_equivalence_cells") != 60:
        raise ValueError("rho-zero preflight does not contain 60 cells")
    for boundary in BOUNDARIES:
        rows = equivalence["boundaries"][str(boundary)]["per_draw"]
        if [int(row["draw_seed"]) for row in rows] != list(DRAW_SEEDS):
            raise ValueError("rho-zero equivalence draw order drifted")
        required_checks = (
            "person_period_keys_equal",
            "level_bytes_equal",
            "participation_states_equal",
            "all_six_moments_equal",
            "streams_1_5_final_states_equal",
            "truth_projection_support_equal",
            "chain_count_conservation",
            "passed",
        )
        if not all(
            all(bool(row[name]) for name in required_checks) for row in rows
        ):
            raise ValueError("rho-zero equivalence contains a failed draw")


def _validate_protocol_and_fences(ledger: dict[str, Any]) -> None:
    protocol = ledger.get("protocol", {})
    required = {
        "fixed_q": 0.55,
        "rho_grid": list(RHO_GRID),
        "pseudo_boundaries": list(BOUNDARIES),
        "fit_seed": 5200,
        "selection_draw_seeds": list(DRAW_SEEDS),
        "fixed_halves": [list(DRAW_SEEDS[:10]), list(DRAW_SEEDS[10:])],
        "selected_cells": list(SELECTED_CELLS),
        "objective_cells": list(OBJECTIVE_CELLS),
        "feasibility_cells": list(FEASIBILITY_CELLS),
        "substream_codes": SUBSTREAM_CODES,
        "fresh_complete_qrf_refit_per_rho_boundary": True,
        "common_random_numbers_across_rungs_at_fixed_seed": True,
        "rho_zero_disposition": "DESIGNED_PAUSE",
        **REQUIRED_FENCES,
    }
    drifted = {
        name: protocol.get(name)
        for name, expected in required.items()
        if protocol.get(name) != expected
    }
    if drifted:
        raise ValueError(f"raw selector protocol drifted: {drifted}")
    if ledger.get("fences") != REQUIRED_FENCES:
        raise ValueError(
            "raw ledger fence fields are missing, false, or extra"
        )


def _draw_summary(
    draws: list[dict[str, Any]], *, truth_support_sha256: str
) -> dict[str, Any]:
    if len(draws) != len(DRAW_SEEDS):
        raise ValueError(f"expected 20 per-draw records, found {len(draws)}")
    seeds = [int(record["draw_seed"]) for record in draws]
    if seeds != list(DRAW_SEEDS):
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
            "projected support hashes do not equal boundary truth support"
        )
    if not all(
        record["transition_chain"]["eligible_decomposition_conserves"]
        and record["transition_chain"]["even_call_conservation_passed"]
        for record in draws
    ):
        raise ValueError("per-draw transition-chain counts do not conserve")
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
        "transition_chain_records_sha256": hashlib.sha256(
            _canonical_bytes([record["transition_chain"] for record in draws])
        ).hexdigest(),
    }


def reduce(raw: bytes) -> dict[str, Any]:
    ledger = json.loads(raw)
    if ledger.get("schema") != RAW_SCHEMA:
        raise ValueError(f"unexpected raw schema {ledger.get('schema')!r}")
    _validate_preflights(ledger)
    _validate_protocol_and_fences(ledger)
    grid = [float(value) for value in ledger["protocol"]["rho_grid"]]
    if grid != list(RHO_GRID):
        raise ValueError("raw ledger rho grid drifted")
    expected_labels = {f"{value:.2f}" for value in RHO_GRID}
    if set(ledger["rungs"]) != expected_labels:
        raise ValueError("raw ledger rung keys do not match the rho grid")
    for rho in RHO_GRID:
        label = f"{rho:.2f}"
        rung = ledger["rungs"][label]
        if float(rung["rho"]) != rho:
            raise ValueError(f"rung {label} carries a different rho")
        if set(rung["boundaries"]) != {str(value) for value in BOUNDARIES}:
            raise ValueError(f"rung {label} boundary set drifted")
        for boundary in BOUNDARIES:
            record = rung["boundaries"][str(boundary)]
            draws = record.pop("per_draw")
            summary = _draw_summary(
                draws,
                truth_support_sha256=record["support"][
                    "truth_support_ids_sha256"
                ],
            )
            record["per_draw_summary"] = summary
            if not summary["all_cells_defined"] and bool(rung.get("valid")):
                raise ValueError(
                    f"rho={label} boundary={boundary} has undefined draws "
                    "but is marked valid"
                )
            transition = record["transition_pair_counts"]
            if [
                int(item["draw_seed"]) for item in transition["per_draw"]
            ] != list(DRAW_SEEDS):
                raise ValueError("transition-pair draw order drifted")
            if not transition["all_draws_conserve"]:
                raise ValueError("transition-pair aggregate does not conserve")

    selector = ledger.get("selector", {})
    selected_label = str(selector.get("selected_rho_label"))
    if selected_label not in expected_labels:
        raise ValueError("selector chose a value outside the rho grid")
    if float(selected_label) == 0.0 and selector.get("disposition") != (
        "DESIGNED_PAUSE"
    ):
        raise ValueError("rho=0 result was not published as a designed pause")

    ledger["schema"] = FINDINGS_SCHEMA
    ledger["full_stdout_sha256"] = hashlib.sha256(raw).hexdigest()
    ledger["reducer"] = {
        "script": "scripts/reduce_m6_rhostar_selection.py",
        "removed": (
            "17 x 3 x 20 repetitive projected-draw moment records only"
        ),
        "retained": (
            "all preflights; every fit, pool, support, RNG checksum; truth "
            "moments; standardizers; aggregate simulated moments; full, half, "
            "and delete-one objectives; guards; retention; F1 disclosure; "
            "one-SE result; and all 1,020 seed-specific transition counts"
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
