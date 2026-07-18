#!/usr/bin/env python3
"""Exercise round-6 selection-shape completeness on synthetic fixtures.

The round-5 smoke supplies the candidate-blind synthetic cube and its frozen
four-generation reducer ladder.  The faithful fixture's selection object is
built independently by the frozen round-3 selector.  This script never opens
the pinned full selector stdout or any staged data source.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

round5_smoke = importlib.import_module("smoke_m6_remarriage_round5_reducer")
round5 = importlib.import_module("reduce_m6_remarriage_round5")
round6 = importlib.import_module("reduce_m6_remarriage_round6")
selector = importlib.import_module("select_m6_remarriage_round3")

EXPECTED_COMPARISON_ERROR = (
    "ValueError: selection and one-SE outcome does not match its independent "
    "recomputation"
)
EXTRA_KEY = "synthetic_unexpected_extra_key"
EXTRA_VALUE = "must_be_rejected"


def _dump(ledger: dict[str, Any]) -> bytes:
    return json.dumps(
        ledger,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _last_stderr_line(result: Any) -> str:
    lines = result.stderr.decode().strip().splitlines()
    return lines[-1] if lines else ""


def _require_error(result: Any, expected: str, *, where: str) -> None:
    observed = _last_stderr_line(result)
    if result.returncode != 1 or observed != expected:
        raise ValueError(
            f"{where} drifted: exit={result.returncode}, error={observed!r}"
        )


def _object_diffs(
    raw: Any,
    recomputed: Any,
    *,
    path: str = "selection",
) -> tuple[list[str], list[str]]:
    key_presence: list[str] = []
    values: list[str] = []
    if isinstance(raw, dict) and isinstance(recomputed, dict):
        raw_keys = set(raw)
        recomputed_keys = set(recomputed)
        key_presence.extend(
            f"{path}.{key}" for key in sorted(raw_keys ^ recomputed_keys)
        )
        for key in sorted(raw_keys & recomputed_keys):
            child_keys, child_values = _object_diffs(
                raw[key],
                recomputed[key],
                path=f"{path}.{key}",
            )
            key_presence.extend(child_keys)
            values.extend(child_values)
        return key_presence, values
    if isinstance(raw, list) and isinstance(recomputed, list):
        if len(raw) != len(recomputed):
            key_presence.append(f"{path}.length")
            return key_presence, values
        for index, (raw_item, recomputed_item) in enumerate(
            zip(raw, recomputed, strict=True)
        ):
            child_keys, child_values = _object_diffs(
                raw_item,
                recomputed_item,
                path=f"{path}[{index}]",
            )
            key_presence.extend(child_keys)
            values.extend(child_values)
        return key_presence, values
    if type(raw) is not type(recomputed) or raw != recomputed:
        values.append(path)
    return key_presence, values


def _branch(
    selection: dict[str, Any], *, law: str, boundary: str
) -> dict[str, Any]:
    return selection["eligibility"][law]["rules"]["rule_7_origin_protection"][
        "widowed_truth_defined_branches"
    ][boundary]


def _annotation_path(*, law: str, boundary: str) -> str:
    return (
        f"selection.eligibility.{law}.rules.rule_7_origin_protection."
        "widowed_truth_defined_branches."
        f"{boundary}.truth_branch_law_independent"
    )


def _expected_annotation_paths() -> list[str]:
    return sorted(
        _annotation_path(law=law, boundary=boundary)
        for law in round6.LAWS[1:]
        for boundary in round6.BOUNDARIES
    )


def _faithful_fixture(legacy_raw: bytes) -> bytes:
    ledger = json.loads(legacy_raw)
    config = round5_smoke._load_json(round5_smoke.CONFIG_PATH)
    selector_selection = selector._select_law(ledger["boundaries"], config)
    recomputed_selection = round6._independent_selection(
        ledger["boundaries"], config
    )
    key_presence, values = _object_diffs(
        selector_selection, recomputed_selection
    )
    if key_presence or values:
        raise ValueError(
            "round-6 reconstruction does not exactly match the frozen "
            f"selector: key_presence={key_presence}, values={values}"
        )
    annotation_paths = []
    for law in round6.LAWS[1:]:
        for boundary in round6.BOUNDARIES:
            branch = _branch(
                selector_selection,
                law=law,
                boundary=boundary,
            )
            if branch.get("truth_branch_law_independent") is not True:
                raise ValueError("faithful annotation did not recompute true")
            annotation_paths.append(
                _annotation_path(law=law, boundary=boundary)
            )
    if sorted(annotation_paths) != _expected_annotation_paths():
        raise ValueError("faithful annotation path census drifted")
    ledger["selection"] = selector_selection
    return _dump(ledger)


def _with_extra_key(raw: bytes) -> tuple[bytes, str]:
    ledger = json.loads(raw)
    law = round6.LAWS[1]
    boundary = round6.BOUNDARIES[0]
    branch = _branch(ledger["selection"], law=law, boundary=boundary)
    branch[EXTRA_KEY] = EXTRA_VALUE
    path = (
        f"selection.eligibility.{law}.rules.rule_7_origin_protection."
        f"widowed_truth_defined_branches.{boundary}.{EXTRA_KEY}"
    )
    return _dump(ledger), path


def _machine_diff(stderr: bytes) -> dict[str, Any]:
    documents = []
    for line in stderr.decode().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == (
            "m6.remarriage.round6.selection_diff.v1"
        ):
            documents.append(value)
    if len(documents) != 1:
        raise ValueError(
            f"expected one machine-readable selection diff, got {documents}"
        )
    return documents[0]


def _exercise_four_generation_ladder(legacy_raw: bytes) -> None:
    round3_result = round5_smoke._run_reducer(
        round5_smoke.ROUND3_REDUCER, legacy_raw
    )
    _require_error(
        round3_result,
        "ValueError: fit_validation.2006.reference_spells does not match "
        "its independent recomputation",
        where="round-3 ladder rung",
    )

    prior_round4_result = round5_smoke._run_prior_round4_reducer(legacy_raw)
    _require_error(
        prior_round4_result,
        "ValueError: fit_validation.2006.divorced_calibration does not match "
        "its independent recomputation",
        where="prior round-4 ladder rung",
    )

    round4_result = round5_smoke._run_reducer(
        round5_smoke.ROUND4_REDUCER, legacy_raw
    )
    _require_error(
        round4_result,
        "ValueError: boundaries.2006.laws.R0.mean.groups[11].rate must be a "
        "JSON number",
        where="current round-4 ladder rung",
    )

    round5_result = round5_smoke._run_reducer(
        round5_smoke.ROUND5_REDUCER, legacy_raw
    )
    if round5_result.returncode != 0:
        raise ValueError(round5_result.stderr.decode())
    findings = json.loads(round5_result.stdout)
    if (
        findings["selection"]["selected_law"] != "R0"
        or findings["reducer"]["removed_publication_groups_arrays"] != 600
    ):
        raise ValueError("round-5 ladder rung did not traverse the full cube")

    print(
        "ROUND6_FOUR_GENERATION_LADDER=PASS "
        "round3=EXPECTED_FAIL_REFERENCE_SPELLS "
        "round4_a0c9d916=EXPECTED_FAIL_DIVORCED_CALIBRATION "
        "round4=EXPECTED_FAIL_NULL_RATE round5=PASS_R0"
    )


def _exercise_missing_reconstruction_key(
    legacy_raw: bytes, faithful_raw: bytes
) -> None:
    ledger = json.loads(faithful_raw)
    config = round5_smoke._load_json(round5_smoke.CONFIG_PATH)
    round5_selection = round5._independent_selection(
        ledger["boundaries"], config
    )
    key_presence, values = _object_diffs(ledger["selection"], round5_selection)
    if key_presence != _expected_annotation_paths() or values:
        raise ValueError(
            "round-5 synthetic selection diff did not isolate the 12 "
            f"annotations: key_presence={key_presence}, values={values}"
        )

    result = round5_smoke._run_reducer(
        round5_smoke.ROUND5_REDUCER, faithful_raw
    )
    _require_error(
        result,
        EXPECTED_COMPARISON_ERROR,
        where="round-5 missing-reconstruction-key probe",
    )
    if (
        round5_smoke._run_reducer(
            round5_smoke.ROUND5_REDUCER, legacy_raw
        ).returncode
        != 0
    ):
        raise ValueError("round-5 legacy control unexpectedly failed")

    print(
        "ROUND6_MISSING_RECONSTRUCTION_KEY_PROBE=EXPECTED_FAIL "
        "reducer=round5 exit=1 key_presence_diffs=12 value_diffs=0"
    )


def _exercise_faithful_fixture(faithful_raw: bytes) -> None:
    result = round5_smoke._run_reducer(
        round5_smoke.ROOT / "scripts/reduce_m6_remarriage_round6.py",
        faithful_raw,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.decode())
    findings = json.loads(result.stdout)
    if (
        findings["selection"]["selected_law"] != "R0"
        or findings["reducer"]["removed_publication_groups_arrays"] != 600
    ):
        raise ValueError("round-6 faithful fixture did not traverse full cube")
    print(
        "ROUND6_FAITHFUL_SELECTION_FIXTURE=PASS exit=0 "
        "selected_law=R0 annotations=12 removed_arrays=600"
    )


def _exercise_extra_key(faithful_raw: bytes) -> None:
    raw, extra_path = _with_extra_key(faithful_raw)
    ledger = json.loads(raw)
    config = round5_smoke._load_json(round5_smoke.CONFIG_PATH)
    recomputed = round6._independent_selection(ledger["boundaries"], config)
    key_presence, values = _object_diffs(ledger["selection"], recomputed)
    if key_presence != [extra_path] or values:
        raise ValueError(
            "extra-key fixture did not isolate one key-presence diff: "
            f"key_presence={key_presence}, values={values}"
        )

    result = round5_smoke._run_reducer(
        round5_smoke.ROOT / "scripts/reduce_m6_remarriage_round6.py",
        raw,
    )
    _require_error(
        result,
        EXPECTED_COMPARISON_ERROR,
        where="round-6 unexpected-extra-key probe",
    )
    expected_diff = {
        "key_presence_diffs": [
            {
                "path": extra_path,
                "raw": {"present": True, "value": EXTRA_VALUE},
                "recomputed": {"present": False},
            }
        ],
        "schema": "m6.remarriage.round6.selection_diff.v1",
        "value_diffs": [],
        "where": "selection and one-SE outcome",
    }
    observed_diff = _machine_diff(result.stderr)
    if observed_diff != expected_diff:
        raise ValueError(
            "round-6 extra-key diagnostic drifted: "
            f"observed={observed_diff}, expected={expected_diff}"
        )
    print(
        "ROUND6_UNEXPECTED_EXTRA_KEY_PROBE=EXPECTED_FAIL "
        "reducer=round6 exit=1 key_presence_diffs=1 value_diffs=0 "
        "machine_diff=PASS"
    )


def _exercise_false_annotation_conjunct(legacy_raw: bytes) -> None:
    ledger = json.loads(legacy_raw)
    config = round5_smoke._load_json(round5_smoke.CONFIG_PATH)
    boundaries = copy.deepcopy(ledger["boundaries"])
    law = round6.LAWS[1]
    boundary = round6.BOUNDARIES[0]
    direct = boundaries[boundary]["laws"][law]["direct"]["origin"]["widowed"]
    direct["matchable_positive_weight_event_rows"] += 1

    selector_selection = selector._select_law(boundaries, config)
    recomputed_selection = round6._independent_selection(boundaries, config)
    key_presence, values = _object_diffs(
        selector_selection, recomputed_selection
    )
    if key_presence or values:
        raise ValueError(
            "false-annotation reconstruction drifted from selector: "
            f"key_presence={key_presence}, values={values}"
        )
    branch = _branch(recomputed_selection, law=law, boundary=boundary)
    if (
        branch["truth_branch_law_independent"] is not False
        or branch["pass"] is not False
    ):
        raise ValueError(
            "law-dependent truth branch did not force annotation and pass false"
        )
    print(
        "ROUND6_FALSE_ANNOTATION_CONJUNCT_PROBE=PASS "
        f"law={law} boundary={boundary} annotation=false branch_pass=false "
        "exact_selector_match=PASS"
    )


def main() -> int:
    legacy_raw = round5_smoke._fixture()
    faithful_raw = _faithful_fixture(legacy_raw)
    _exercise_four_generation_ladder(legacy_raw)
    _exercise_missing_reconstruction_key(legacy_raw, faithful_raw)
    _exercise_faithful_fixture(faithful_raw)
    _exercise_extra_key(faithful_raw)
    _exercise_false_annotation_conjunct(legacy_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
