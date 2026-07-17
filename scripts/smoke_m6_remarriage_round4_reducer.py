#!/usr/bin/env python3
"""Exercise both remarriage reducers on one selector-shaped synthetic cube.

The fixture inherits only frozen locks and candidate-blind fit tables from the
committed preflight.  Every outcome-bearing cube value is synthetic.  This
script never opens the round-3 full selector stdout or a staged data source.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import reduce_m6_remarriage_round4 as reducer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts/m6_remarriage_round3_selector_config.json"
PREFLIGHT_PATH = (
    ROOT / "docs/analysis/m6_remarriage_round3_selection_preflight.json"
)
VALIDATION_PATH = (
    ROOT / "docs/design/m6_remarriage_learning_plan_round2_validation.json"
)
ROUND3_REDUCER = ROOT / "scripts/reduce_m6_remarriage_round3.py"
ROUND4_REDUCER = ROOT / "scripts/reduce_m6_remarriage_round4.py"
GROUP_IDENTITY_FIELDS = {"group", "origin", "age_band", "ysd_band"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _synthetic_sha256(label: str) -> str:
    return hashlib.sha256(f"round4 synthetic {label}".encode()).hexdigest()


def _projected(truth: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(truth)
    return {**projected, **reducer._ratio_fields(projected, truth)}


def _direct(truth: dict[str, Any]) -> dict[str, Any]:
    exposure = float(truth["exposure"])
    expected_numerator = exposure / 4.0
    weighted_deviance_numerator = exposure / 2.0
    return {
        "risk_rows": truth["risk_rows"],
        "event_rows": truth["event_rows"],
        "matchable_positive_weight_event_rows": truth["event_rows"],
        "matchable_event_rows": truth["event_rows"],
        "unmatched_same_year_event_rows": 0,
        "unmatched_same_year_event_weight": 0.0,
        "exposure": exposure,
        "deviance_exposure": exposure,
        "actual_numerator": truth["numerator"],
        "matchable_numerator": truth["numerator"],
        "actual_rate": truth["rate"],
        "row_expected_numerator": expected_numerator,
        "expected_numerator": expected_numerator,
        "expected_rate": 0.25,
        "qdir": 0.25,
        "weighted_deviance_numerator": weighted_deviance_numerator,
        "weighted_bernoulli_deviance": 0.5,
    }


def _group_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "group": group,
            "origin": origin,
            "age_band": list(age),
            "ysd_band": list(ysd),
            **copy.deepcopy(payload),
        }
        for group, origin, age, ysd in reducer.EXPECTED_GROUPS
    ]


def _projected_groups(
    truth_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = []
    for truth in truth_groups:
        payload = {
            key: value
            for key, value in truth.items()
            if key not in GROUP_IDENTITY_FIELDS
        }
        groups.append(
            {
                **{key: truth[key] for key in GROUP_IDENTITY_FIELDS},
                **_projected(payload),
            }
        )
    return groups


def _direct_groups(
    truth_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = []
    for truth in truth_groups:
        payload = {
            key: value
            for key, value in truth.items()
            if key not in GROUP_IDENTITY_FIELDS
        }
        groups.append(
            {
                **{key: truth[key] for key in GROUP_IDENTITY_FIELDS},
                **_direct(payload),
            }
        )
    return groups


def _fit_validation(
    preflight: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    fits: dict[str, Any] = {}
    for boundary in reducer.BOUNDARIES:
        preflight_fit = preflight["fit_validation"][boundary]
        frozen = validation["boundaries"][boundary]
        references: dict[str, Any] = {}
        for origin in reducer.ORIGINS:
            audit = copy.deepcopy(frozen["reference_spells"][origin])
            audit.update(
                {
                    "duplicate_spells_checksum_sha256": _synthetic_sha256(
                        f"{boundary} {origin} duplicate spells"
                    ),
                    "same_year_remarriage_spells_checksum_sha256": (
                        _synthetic_sha256(
                            f"{boundary} {origin} same-year spells"
                        )
                    ),
                    "missing_required_or_nonpositive_weight_spells_excluded_weight": 0.0,
                    "missing_required_or_nonpositive_weight_checksum_sha256": (
                        _synthetic_sha256(
                            f"{boundary} {origin} missing spells"
                        )
                    ),
                    "no_potential_path_spells_excluded_weight": 0.0,
                    "no_potential_path_spells_checksum_sha256": (
                        _synthetic_sha256(
                            f"{boundary} {origin} no-path spells"
                        )
                    ),
                    "included_spells_checksum_sha256": _synthetic_sha256(
                        f"{boundary} {origin} included spells"
                    ),
                }
            )
            if set(audit) != reducer.REFERENCE_AUDIT_FIELDS:
                raise ValueError("synthetic reference audit is not 21-key")
            references[origin] = audit
        fit = {
            "fit_max_year": frozen["fit_max_year"],
            "fit_person_year_rows": 1,
            "fit_event_rows": 1,
            "dissolved_rows": 1,
            "remarriage_events": 1,
            "wbar": 1.0,
            "incumbent_table_sha256": frozen["incumbent_table_sha256"],
            "fit_support_by_origin_ysd": copy.deepcopy(
                frozen["fit_support_by_origin_ysd"]
            ),
            "fit_exposure_center": 0.0,
            "centered_contrast": [1.0, 0.0, -1.0],
            "reference_spells": references,
            "reference_exclusion_category_hashes": {
                origin: {
                    key: references[origin][key]
                    for key in (
                        reducer.REFERENCE_EXCLUSION_CATEGORY_HASH_FIELDS
                    )
                }
                for origin in reducer.ORIGINS
            },
            "divorced_calibration": copy.deepcopy(
                frozen["divorced_calibration"]
            ),
            "widowed_targets": copy.deepcopy(frozen["widowed_targets"]),
            "support_struck_named_laws": copy.deepcopy(
                frozen["support_struck_named_laws"]
            ),
            "validation_match": True,
            "field_boundary_audit": {
                "boundary": int(boundary),
                "field_maxima": {},
                "post_boundary_end_and_separation_fields_nulled": True,
                "marriage_counts_recomputed": True,
                "every_fit_frame_field_asserted": True,
            },
            "laws": copy.deepcopy(preflight_fit["laws"]),
        }
        fits[boundary] = fit
    return fits


def _truth(locked: dict[str, Any]) -> dict[str, Any]:
    group_truth = {
        "risk_rows": 1,
        "event_rows": 1,
        "exposure": 2.0,
        "numerator": 1.0,
        "rate": 0.5,
    }
    return {
        "pooled": copy.deepcopy(locked["truth"]),
        "origin": copy.deepcopy(locked["truth_origin"]),
        "groups": _group_records(group_truth),
    }


def _seed_row(
    *,
    boundary: str,
    law: str,
    seed: int,
    truth: dict[str, Any],
    boundary_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    is_r0 = law == "R0"
    return {
        "seed": seed,
        "pooled": _projected(truth["pooled"]),
        "origin": {
            origin: _projected(truth["origin"][origin])
            for origin in reducer.ORIGINS
        },
        "publication_groups": _projected_groups(truth["groups"]),
        "carrier_checks": {
            "expected_entry_dissolved_carriers": boundary_record[
                "entry_dissolved_carriers"
            ],
            "verified_entry_dissolved_carriers": boundary_record[
                "entry_dissolved_carriers"
            ],
            "exact": True,
        },
        "support_checks": {
            "exact_truth_keys": True,
            "truth_key_count": boundary_record["truth_support_rows"],
            "projected_key_count": boundary_record["truth_support_rows"],
            "truth_key_sha256": boundary_record["truth_support_key_sha256"],
            "projected_key_sha256": boundary_record[
                "truth_support_key_sha256"
            ],
            "weighted_support_sha256": boundary_record["support_checksum"],
            "weighted_support_exact_truth": True,
            "event_semantics": {
                "remarriage_event_rows": truth["pooled"]["event_rows"],
                "origins_valid": True,
                "years_since_dissolution_defined": True,
                "unique_person_year": True,
                "matchable_origin_and_ysd_exact_risk_row": True,
                "unmatched_same_year_event_rows": 0,
            },
        },
        "uniform_checks": {
            "draw_index": seed - config["rng"]["seed_root"],
            "n_periods": config["protocol"]["n_periods"][boundary],
            "transition_address": copy.deepcopy(
                config["rng"]["transition_uniform_address"]
            ),
            "spouse_gap_address": copy.deepcopy(
                config["rng"]["spouse_gap_address"]
            ),
            "transition_uniform_sha256": _synthetic_sha256(
                f"{boundary} {seed} transition uniform"
            ),
            "exact_R0_stream": True,
        },
        "downstream": {
            "person_year_rows": 1,
            "event_rows": 1,
            "birth_rows": 0,
            "person_years_sha256": _synthetic_sha256(
                f"{boundary} {law} {seed} person years"
            ),
            "events_sha256": _synthetic_sha256(
                f"{boundary} {law} {seed} events"
            ),
            "births_sha256": _synthetic_sha256(
                f"{boundary} {law} {seed} births"
            ),
            "spouse_gap_consumption": {"calls": 0, "draws": 0},
            "spouse_gap_consumption_difference_from_R0": {
                "calls": 0,
                "draws": 0,
            },
            "R0_spouse_gap_consumption_exact_incumbent": (
                True if is_r0 else None
            ),
            "R0_projection_exact_incumbent": True if is_r0 else None,
        },
    }


def _law_record(
    *,
    boundary: str,
    law: str,
    truth: dict[str, Any],
    boundary_record: dict[str, Any],
    fit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        _seed_row(
            boundary=boundary,
            law=law,
            seed=seed,
            truth=truth,
            boundary_record=boundary_record,
            config=config,
        )
        for seed in reducer.SEEDS
    ]
    direct = {
        "pooled": _direct(truth["pooled"]),
        "origin": {
            origin: _direct(truth["origin"][origin])
            for origin in reducer.ORIGINS
        },
        "groups": _direct_groups(truth["groups"]),
    }
    is_r0 = law == "R0"
    return {
        "construction": copy.deepcopy(fit["laws"][law]),
        "direct": direct,
        "g_widowed_log_qdir_ratio": 0.0,
        "per_seed": rows,
        "mean": reducer._aggregate_projection(rows, truth, reducer.SEEDS),
        "blocks": {
            f"block_{index}": reducer._aggregate_projection(rows, truth, seeds)
            for index, seeds in enumerate(reducer.BLOCKS, start=1)
        },
        "carrier_conformance_all_draws": True,
        "support_exact_all_draws": True,
        "uniform_exact_all_draws": True,
        "non_remarriage_components_exact_R0": True,
        "R0_projection_exact_incumbent_all_draws": True if is_r0 else None,
        "R0_spouse_gap_consumption_exact_incumbent_all_draws": (
            True if is_r0 else None
        ),
    }


def _boundary_record(
    *, boundary: str, fit: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    locked = config["expected_pseudo_holdouts"][boundary]
    truth = _truth(locked)
    truth_support_key_sha256 = _synthetic_sha256(
        f"{boundary} truth support keys"
    )
    record = {
        "boundary": int(boundary),
        "anchor_waves": copy.deepcopy(
            config["pseudo_holdouts"]["boundaries"][boundary]["anchor_waves"]
        ),
        "evaluation_years": copy.deepcopy(
            config["pseudo_holdouts"]["boundaries"][boundary][
                "evaluation_years"
            ]
        ),
        **{
            key: copy.deepcopy(locked[key])
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
            )
        },
        "truth_origin": copy.deepcopy(truth["origin"]),
        "truth": truth,
        "truth_support_key_sha256": truth_support_key_sha256,
        "pseudo_input_hashes": {
            "anchor_sha256": _synthetic_sha256(f"{boundary} anchor"),
            "seed_attrs_sha256": _synthetic_sha256(
                f"{boundary} seed attributes"
            ),
            "seed_entry_person_years_sha256": _synthetic_sha256(
                f"{boundary} seed entry person years"
            ),
            "truth_support_key_sha256": truth_support_key_sha256,
            "truth_support_weighted_sha256": locked["support_checksum"],
        },
        "fit": fit,
        "laws": {},
    }
    record["laws"] = {
        law: _law_record(
            boundary=boundary,
            law=law,
            truth=truth,
            boundary_record=record,
            fit=fit,
            config=config,
        )
        for law in reducer.LAWS
    }
    return record


def _input_audit(config: dict[str, Any]) -> dict[str, Any]:
    authority_reads = (
        "scripts/m6_remarriage_round3_selector_config.json",
        "docs/design/m6_remarriage_learning_plan_round2.md",
        "docs/design/m6_remarriage_learning_plan_round2_validation.json",
        "docs/analysis/m6_remarriage_train_only_delta_results.json",
    )
    observed_paths = sorted(
        str((ROOT / path).resolve()) for path in authority_reads
    )
    return {
        "selection_relevant_reads": [
            *authority_reads,
            "sanitized staged PSID sources through the frozen round-1 chassis",
        ],
        "maximum_information_year": 2014,
        "source_audit": copy.deepcopy(config["expected_source_audit"]),
        "dynamic_open_audit_installed_before_selection_reads": True,
        "observed_open_path_count": len(observed_paths),
        "observed_open_paths": observed_paths,
        "gates_yaml_read": False,
        "runs_artifact_read": False,
        "M6_scorer_imported": False,
        "post_2014_selection_data_read": False,
        "helper_wrote_files": False,
        "stdout_machine_documents": 1,
    }


def _fixture() -> bytes:
    config = _load_json(CONFIG_PATH)
    preflight = _load_json(PREFLIGHT_PATH)
    validation = _load_json(VALIDATION_PATH)
    if preflight.get("candidate_outcome_contact") is not False:
        raise ValueError("fixture source preflight contacted an outcome")
    if preflight.get("pseudo_holdout_truth_constructed") is not False:
        raise ValueError("fixture source preflight constructed pseudo-truth")

    fit_validation = _fit_validation(preflight, validation)
    boundaries = {
        boundary: _boundary_record(
            boundary=boundary,
            fit=fit_validation[boundary],
            config=config,
        )
        for boundary in reducer.BOUNDARIES
    }
    selection = reducer._independent_selection(boundaries, config)
    if selection["selected_law"] != "R0":
        raise ValueError("synthetic equal-law cube did not select R0")
    ledger = {
        "schema": reducer.RAW_SCHEMA,
        "status": "SELECTION_COMPLETE",
        "candidate_outcome_contact": True,
        "authority": copy.deepcopy(config["authority"]),
        "freeze": copy.deepcopy(preflight["freeze"]),
        "runtime": copy.deepcopy(preflight["runtime"]),
        "protocol": copy.deepcopy(config["protocol"]),
        "input_and_file_open_audit": _input_audit(config),
        "fit_validation": fit_validation,
        "boundaries": boundaries,
        "selection": selection,
        "final_information_fit": {
            "boundary": 2014,
            "selected_law": "R0",
            "status": "NOT_RUN_R0_SELECTED",
            "construction_pass": True,
            "designed_pause_continues": True,
        },
        "publication": {
            "full_stdout_path": config["output"]["full_stdout_json"],
            "reduced_findings_path": config["output"]["findings_json"],
            "report_path": config["output"]["findings_report"],
            "publish_regardless_of_outcome": True,
            "cumulative_nonzero_laws_two_rounds": 7,
        },
    }
    return json.dumps(
        ledger,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _run_reducer(path: Path, raw: bytes) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        env=environment,
        input=raw,
        capture_output=True,
        check=False,
    )


def main() -> int:
    raw = _fixture()
    round3 = _run_reducer(ROUND3_REDUCER, raw)
    round3_error = round3.stderr.decode().strip().splitlines()[-1]
    expected_error = (
        "ValueError: fit_validation.2006.reference_spells does not match "
        "its independent recomputation"
    )
    if round3.returncode != 1 or round3_error != expected_error:
        raise ValueError(
            "round-3 reducer did not fail at the expected leg-A comparison"
        )
    print(
        "ROUND3_REDUCER_SYNTHETIC_SMOKE=EXPECTED_FAIL "
        f'exit=1 error="{round3_error}"'
    )

    round4 = _run_reducer(ROUND4_REDUCER, raw)
    if round4.returncode != 0:
        raise ValueError(round4.stderr.decode())
    findings = json.loads(round4.stdout)
    removed = findings["reducer"]["removed_publication_groups_arrays"]
    if findings["selection"]["selected_law"] != "R0" or removed != 600:
        raise ValueError("round-4 reducer did not traverse the full cube")
    print(
        "ROUND4_REDUCER_SYNTHETIC_SMOKE=PASS exit=0 "
        f"selected_law=R0 removed_arrays={removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
