"""Prospective lock bindings for the closed-domain M6 floors-v4 ceremony.

ALWAYS-RUNNABLE (artifact tier): these tests read only the committed v3/v4
floor artifacts, the v4 environment sidecar, the v4 builder source, and
``gates.yaml``. They do not read PSID microdata, fit an earnings model, build a
projection, or inspect a candidate run.

The suite binds the new artifact before it can become operative: v3 remains the
live historical contract, while v4 reprices exactly the six earnings cells on
the F7 closed domain. It independently recomputes the six tolerances, the
faithful-candidate operating characteristic, and the candidate-blind numerical
core hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GATES_PATH = ROOT / "gates.yaml"
V3_PATH = ROOT / "runs" / "m6_holdout_floors_v3.json"
V4_PATH = ROOT / "runs" / "m6_holdout_floors_v4.json"
V4_SIDECAR_PATH = ROOT / "runs" / "m6_holdout_floors_v4.json.env.json"
BUILDER_PATH = ROOT / "scripts" / "build_m6_holdout_floors_v4.py"
AMENDMENT_PATH = (
    ROOT
    / "docs"
    / "amendments"
    / "gate_m6_amendment_1_closed_domain_floors.md"
)

V3_SHA256 = "e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77"
V4_SHA256 = "4cd2d01a9fd76064e701ae77a9226208cbae94d743f76f502d3d0a5f657d9523"
V4_SIDECAR_SHA256 = (
    "75ae57ceacad7abb0958d322ba52f72a7404e45001cdde463539b96a215b37e8"
)
V4_SOURCE_COMMIT = "2d1704ede69aea4cb1caf174d6dc40653e56d63a"
PROGRAM_COMMIT = "051b4494ecce9345da14d68488bb2833ed476d22"

EARNINGS_CELLS = frozenset(
    {
        "earn_autocorr_lag2",
        "earn_dlog_mean.prime",
        "earn_dlog_sd.older",
        "earn_mob_h1_diag",
        "earn_p10.prime",
        "earn_zero_rate.older",
    }
)
TOLERANCE_MOVEMENTS = {
    "earn_autocorr_lag2": (0.087, 0.087),
    "earn_dlog_mean.prime": (0.043, 0.043),
    "earn_dlog_sd.older": (0.269, 0.279),
    "earn_mob_h1_diag": (0.052, 0.054),
    "earn_p10.prime": (0.221, 0.284),
    "earn_zero_rate.older": (0.163, 0.168),
}
EXPECTED_OC = {
    "combined": (0.8889, 0.9018),
    "family_a_flows": (0.9559, 0.9822),
    "earnings_subfamily": (0.9300, 0.9575),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _v3() -> dict[str, Any]:
    return _load_json(V3_PATH)


def _v4() -> dict[str, Any]:
    return _load_json(V4_PATH)


def _sidecar() -> dict[str, Any]:
    return _load_json(V4_SIDECAR_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _oc(
    floor: dict[str, dict[str, float]],
    tolerances: dict[str, float],
    cells: list[str],
) -> tuple[float, float, dict[str, float]]:
    p_seed = 1.0
    per_cell: dict[str, float] = {}
    for cell in sorted(cells):
        sigma = floor[cell]["realized_sigma"]
        probability = (
            2.0 * _normal_cdf(tolerances[cell] / sigma) - 1.0
            if sigma > 0
            else 1.0
        )
        p_seed *= probability
        per_cell[cell] = round(probability, 6)
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return round(p_seed, 4), round(p_gate, 4), per_cell


def _derivation_core(v4: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact candidate-blind object hashed by the builder."""
    oc = v4["faithful_candidate_oc"]
    return {
        "domain_support": v4["domain_support"],
        "earnings_per_cell": v4["closed_domain_earnings"]["per_cell"],
        "floor_seed_disclosure": v4["closed_domain_earnings"][
            "all_100_floor_seeds"
        ],
        "floor_cells": v4["floor"]["cells"],
        "tolerances": v4["tolerances"],
        "faithful_candidate_oc": {
            "combined": oc["combined"],
            "family_a_flows": oc["family_a_flows"],
            "earnings_subfamily": oc["earnings_subfamily"],
        },
        "oc_comparison": v4["oc_comparison"],
        "two_directional_power_check": v4["two_directional_power_check"],
    }


def _named_values(node: Any, name: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == name:
                values.append(value)
            values.extend(_named_values(value, name))
    elif isinstance(node, list):
        for value in node:
            values.extend(_named_values(value, name))
    return values


def test_v4_primary_sidecar_and_frozen_v3_byte_hashes_are_fixed():
    assert _sha256(V3_PATH) == V3_SHA256
    assert _sha256(V4_PATH) == V4_SHA256
    assert _sha256(V4_SIDECAR_PATH) == V4_SIDECAR_SHA256


def test_live_gate_remains_bound_to_frozen_v3_before_ratification():
    gate_document = yaml.safe_load(GATES_PATH.read_text(encoding="utf-8"))
    gate = gate_document["gates"]["gate_m6"]
    assert gate["locked"] is True
    assert gate["floor_run"] == "runs/m6_holdout_floors_v3.json"
    assert gate["floor_run_sha256"] == V3_SHA256
    assert set(_named_values(gate, "floor_run")) == {
        "runs/m6_holdout_floors_v3.json"
    }
    assert "runs/m6_holdout_floors_v4.json" not in GATES_PATH.read_text(
        encoding="utf-8"
    )


def test_v4_schema_and_governance_are_prospective_only():
    v4 = _v4()
    assert v4["schema_version"] == "m6_holdout_floors.v4"
    assert v4["run"] == "m6_holdout_floors_v4"
    assert v4["revision_pins"] == {
        "artifact_schema_version": "m6_holdout_floors.v4",
        "floor_seed_range": [0, 99],
        "full_anchor_split_before_domain_intersection": True,
        "v3_sha256": V3_SHA256,
    }
    assert v4["lineage"]["program_commit"] == PROGRAM_COMMIT
    assert v4["lineage"]["verification_comment"] == "5001901052"
    assert v4["lineage"]["v3_read_only"] is True
    assert v4["governance"]["operative"] is False
    assert (
        v4["governance"]["status"]
        == "DERIVED_NOT_OPERATIVE_UNTIL_PROSPECTIVE_LOCK"
    )
    assert v4["governance"]["gates_yaml_edited_by_builder"] is False
    assert v4["governance"]["v3_historical_contract_unchanged"] is True
    assert v4["reported_truth_side_only"] is True
    assert v4["no_projection_no_candidate"] is True


def test_exactly_six_earnings_tolerances_move_from_v3_to_v4():
    v3 = _v3()
    v4 = _v4()
    assert (
        set(v4["partition"]["by_family_gated"]["earnings"]) == EARNINGS_CELLS
    )
    assert set(TOLERANCE_MOVEMENTS) == EARNINGS_CELLS
    for cell, (v3_tolerance, v4_tolerance) in TOLERANCE_MOVEMENTS.items():
        assert v3["tolerances"][cell] == v3_tolerance
        assert v4["tolerances"][cell] == v4_tolerance
        record = v4["closed_domain_earnings"]["per_cell"][cell]
        assert record["v3_tolerance"] == v3_tolerance
        assert record["v4_tolerance"] == v4_tolerance
        assert record["tolerance_delta"] == pytest.approx(
            v4_tolerance - v3_tolerance,
            abs=1e-15,
        )

    non_earnings_gated = set(v3["tolerances"]) - EARNINGS_CELLS
    assert non_earnings_gated == set(v4["tolerances"]) - EARNINGS_CELLS
    for cell in non_earnings_gated:
        assert v4["tolerances"][cell] == v3["tolerances"][cell]


def test_every_non_earnings_floor_record_is_byte_carried_from_v3():
    v3_cells = _v3()["floor"]["cells"]
    v4_cells = _v4()["floor"]["cells"]
    assert set(v4_cells) == set(v3_cells)
    carried_cells = set(v3_cells) - EARNINGS_CELLS
    assert len(carried_cells) == 14
    for cell in carried_cells:
        assert _canonical(v4_cells[cell]) == _canonical(v3_cells[cell]), cell
    for cell in EARNINGS_CELLS:
        assert _canonical(v4_cells[cell]) != _canonical(v3_cells[cell]), cell


def test_six_closed_domain_tolerances_recompute_with_registered_rule_and_caps():
    v4 = _v4()
    quantum = Decimal("0.001")
    for cell in sorted(EARNINGS_CELLS):
        record = v4["closed_domain_earnings"]["per_cell"][cell]
        floor = record["floor"]
        raw = (
            Decimal(str(floor["mean"]))
            + Decimal(record["k"]) * Decimal(str(floor["sd"]))
        ).quantize(quantum, rounding=ROUND_HALF_EVEN)
        cap = Decimal(str(record["metric_cap"]))
        capped = min(raw, cap)
        assert record["k"] == 3
        assert record["rounding"] == 3
        assert Decimal(str(record["v4_raw_tolerance"])) == raw
        assert Decimal(str(record["v4_tolerance"])) == capped
        assert v4["tolerances"][cell] == float(capped)
        assert record["at_metric_cap"] is (raw >= cap)


def test_v4_operating_characteristics_recompute_from_individual_cells():
    v4 = _v4()
    floor = v4["floor"]["cells"]
    tolerances = v4["tolerances"]
    groups = {
        "combined": list(v4["partition"]["gated"]),
        "family_a_flows": sorted(
            set(v4["partition"]["gated"]) - EARNINGS_CELLS
        ),
        "earnings_subfamily": sorted(EARNINGS_CELLS),
    }
    for group, cells in groups.items():
        p_seed, p_gate, per_cell = _oc(floor, tolerances, cells)
        record = v4["faithful_candidate_oc"][group]
        assert (p_seed, p_gate) == EXPECTED_OC[group]
        assert record["p_seed_pass"] == p_seed
        assert record["p_gate_pass_4_of_5"] == p_gate
        assert {
            cell: value["cell_pass_prob"]
            for cell, value in record["per_cell"].items()
        } == per_cell
    assert v4["faithful_candidate_oc"]["p_seed_pass"] == 0.8889
    assert v4["faithful_candidate_oc"]["p_gate_pass_4_of_5"] == 0.9018
    assert v4["oc_before_lock"]["p_gate_combined"] == 0.9018
    assert v4["oc_before_lock"]["p_gate_flows"] == 0.9822
    assert v4["oc_before_lock"]["p_gate_earnings"] == 0.9575


def test_all_100_seeds_bind_support_and_full_split_before_f7_intersection():
    v4 = _v4()
    support = v4["domain_support"]
    seeds = v4["closed_domain_earnings"]["all_100_floor_seeds"]
    assert support["n_full_anchor_persons"] == 29_792
    assert support["n_domain_persons"] == 13_561
    assert support["n_domain_earnings_rows"] == 45_606
    assert support["n_full_gated_earnings_persons"] == 13_163
    assert support["n_domain_gated_earnings_persons"] == 10_441
    assert support["n_later_entrant_gated_persons"] == 2_722
    assert support["later_entrant_persons_by_cohort"] == {
        "older": 590,
        "prime": 2_199,
    }
    assert support["domain_earnings_by_period"] == {
        "2014": {"n_persons": 10_982, "n_rows": 10_982},
        "2016": {"n_persons": 9_993, "n_rows": 9_993},
        "2018": {"n_persons": 9_179, "n_rows": 9_179},
        "2020": {"n_persons": 8_053, "n_rows": 8_053},
        "2022": {"n_persons": 7_399, "n_rows": 7_399},
    }
    assert "split the full anchor" in support["split_order"]
    assert "inside the reducer" in support["split_order"]
    assert v4["revision_pins"]["full_anchor_split_before_domain_intersection"]
    assert v4["domain_refit"]["provenance"]["boundary_year"] == 2014
    assert v4["domain_refit"]["provenance"]["max_year"] == {
        "earnings_reference_year": 2014
    }

    assert [seed["seed"] for seed in seeds] == list(range(100))
    for seed in seeds:
        assert (
            seed["full_anchor_persons_a"] + seed["full_anchor_persons_b"]
            == 29_792
        )
        assert seed["domain_persons_a"] + seed["domain_persons_b"] == 13_561
        assert (
            seed["domain_earnings_rows_a"] + seed["domain_earnings_rows_b"]
            == 45_606
        )
        assert seed["domain_persons_a"] <= seed["full_anchor_persons_a"]
        assert seed["domain_persons_b"] <= seed["full_anchor_persons_b"]
        assert set(seed["cells"]) == EARNINGS_CELLS

    for cell in EARNINGS_CELLS:
        weaker_half = min(
            min(
                seed["cells"][cell]["n_events_a"],
                seed["cells"][cell]["n_events_b"],
            )
            for seed in seeds
        )
        floor = v4["floor"]["cells"][cell]
        assert floor["n_defined_seeds"] == 100
        assert floor["min_events_weaker_half"] == weaker_half


def test_v4_clears_cap_support_and_two_directional_power_guards():
    v4 = _v4()
    power = v4["two_directional_power_check"]
    assert power == {
        "cells_below_minimum_support": [],
        "ceremony_may_proceed": True,
        "clears_flow_surface_power": True,
        "clears_minimum_support": True,
        "domain_tolerances_at_metric_cap": [],
        "minimum_weaker_half_support": 20,
        "near_tautological_oc": False,
        "near_unfailable_cells": [],
        "near_unpassable": False,
        "vacuity": False,
        "weak_power_floor": 0.9,
    }
    for cell in EARNINGS_CELLS:
        record = v4["closed_domain_earnings"]["per_cell"][cell]
        assert record["v4_tolerance"] < record["metric_cap"]
        assert record["floor"]["min_events_weaker_half"] >= 20
    before_lock = v4["oc_before_lock"]
    assert before_lock["n_gated_cells"] == 11
    assert before_lock["n_gated_flow_cells"] == 5
    assert before_lock["n_gated_earnings_cells"] == 6
    assert before_lock["n_gated_tolerances_at_cap"] == 0
    assert before_lock["n_domain_earnings_tolerances_at_cap"] == 0
    assert before_lock["clears_weak_power_threshold"] is True
    assert before_lock["ceremony_may_proceed"] is True


def test_candidate_blind_core_hash_excludes_s2_governance_disclosure():
    v4 = _v4()
    core = _derivation_core(v4)
    digest = _json_sha256(core)
    proof = v4["candidate_blindness_proof"]
    assert (
        digest
        == "b34ab1943f7793e41641819ff3482187e8f0acf492ea2cce7f0e694f9fd01cc0"
    )
    assert v4["derivation_core_sha256"] == digest
    assert proof["derivation_core_sha256"] == digest
    assert proof["candidate_artifact_or_projection_read"] is False
    assert proof["candidate_score_used_in_floor_tolerance_or_oc"] is False
    assert proof["resolution_b_s2_disclosure"]["used_in_derivation"] is False
    assert v4["input_boundary"]["candidate_artifacts_read"] is False
    assert v4["input_boundary"]["projection_outputs_read"] is False
    assert (
        v4["input_boundary"]["s2_movement_disclosure_used_in_derivation"]
        is False
    )

    disclosure_mutation = copy.deepcopy(v4)
    disclosure_mutation["candidate_blindness_proof"][
        "resolution_b_s2_disclosure"
    ]["authority"] = "deliberately mutated governance-only text"
    assert _json_sha256(_derivation_core(disclosure_mutation)) == digest
    assert b"resolution_b_s2_disclosure" not in _canonical(core)


def test_builder_embeds_no_candidate_run_path_or_historical_scores():
    source = BUILDER_PATH.read_text(encoding="utf-8")
    # Candidate terminology in explanatory declarations is expected. Numerical
    # dependence is forbidden: no candidate artifact path or S2 score vector
    # may be embedded in the numerical builder.
    forbidden_paths = (
        "runs/gate_m6_candidate",
        "gate_m6_candidate1_v1.json",
    )
    historical_candidate_scores = (
        "0.251827",
        "0.308309",
        "0.271703",
        "0.327174",
        "0.356735",
    )
    assert all(value not in source for value in forbidden_paths)
    assert all(value not in source for value in historical_candidate_scores)
    v4 = _v4()
    builder_key = "scripts/build_m6_holdout_floors_v4.py"
    assert v4["derivation_code_sha256"][builder_key] == _sha256(BUILDER_PATH)


def test_sidecar_binds_builder_source_commit_and_clean_fitting_stack():
    sidecar = _sidecar()
    v4 = _v4()
    assert sidecar["contract"] == {
        "blob_sha": "1efbf0958b722d8172697ac3f9a48c043de09bcf",
        "head_sha": V4_SOURCE_COMMIT,
        "path": "gates.yaml",
    }
    gates_bytes = GATES_PATH.read_bytes()
    git_blob = b"blob " + str(len(gates_bytes)).encode() + b"\0" + gates_bytes
    assert (
        hashlib.sha1(git_blob).hexdigest() == sidecar["contract"]["blob_sha"]
    )
    assert sidecar["environment"]["fitting_stack"] == {
        "populace_fit": {
            "git_rev": v4["fitting_stack_source"]["commit"],
            "version": "0.1.0",
        },
        "populace_frame": {
            "git_rev": v4["fitting_stack_source"]["commit"],
            "version": "0.1.0",
        },
    }
    assert v4["fitting_stack_source"]["package_sources_clean"] is True


def test_prospective_amendment_binds_lock_and_s2_disclosures():
    text = AMENDMENT_PATH.read_text(encoding="utf-8")
    compact = " ".join(text.split())
    required_bindings = (
        "DRAFT_NOT_OPERATIVE",
        V4_SOURCE_COMMIT,
        V4_SHA256,
        V4_SIDECAR_SHA256,
        "b34ab1943f7793e41641819ff3482187e8f0acf492ea2cce7f0e694f9fd01cc0",
        "0.8373",
        "0.8114",
        "0.8889",
        "0.9018",
        "[0.251827, 0.308309, 0.271703, 0.327174, 0.356735]",
        "0.221 → 0.284",
        "0.052 → 0.054",
        "0.087 → 0.087",
        "0.043 → 0.043",
        "earn_dlog_sd.older=0.269",
        "earn_zero_rate.older=0.163",
    )
    for binding in required_bindings:
        assert binding in text
    assert (
        "Applying v4 retrospectively to candidate 1 is prohibited" in compact
    )
    assert "this lane makes no `gates.yaml` edit" in compact
