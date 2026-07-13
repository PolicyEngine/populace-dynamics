"""Bind the gate_m6 floors artifacts to the gate_m6 block draft.

ALWAYS-RUNNABLE (artifact tier): reads only the committed
``runs/m6_holdout_floors_v1.json`` (the FROZEN pause evidence) and
``runs/m6_holdout_floors_v2.json`` (the redesigned surface) JSON artifacts, the
``docs/design/gate_m6_block_draft.yaml`` draft block (now v2), and the merged
design ``docs/design/m6_projection_engine.md`` -- no engine, no candidate, no
PSID micro-read.

Two binding layers:
  * v1 FROZEN-EVIDENCE bindings -- v1 records the OC-before-lock PAUSE (near-
    vacuous flow surface + combined p_gate 0.8449 < 0.90). These stay and bind
    the immutable pause evidence.
  * v2 bindings -- the block draft binds the redesigned surface string-for-
    string: the coarsening + earnings-decompounding ladders, the 11-cell gated
    registry, tolerance recompute from the v2 floor sigmas, the operating
    characteristic recompute, and the v1 pause-evidence reference.

>= 8 mutations are caught, including ladder-order drift and p_gate drift.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from populace_dynamics.evaluation import derive_tolerance

ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "runs" / "m6_holdout_floors_v1.json"
V2_PATH = ROOT / "runs" / "m6_holdout_floors_v2.json"
BLOCK_PATH = ROOT / "docs" / "design" / "gate_m6_block_draft.yaml"
DESIGN_PATH = ROOT / "docs" / "design" / "m6_projection_engine.md"

V1_SHA = "16c28d8cd9095e5233ab224c659c8d5b9eb1621099e2524455a3a8ff8e88d318"
LN_1_5 = math.log(1.5)
METRIC_CAP = {"log_ratio": LN_1_5, "abs_gap_log": LN_1_5, "abs_gap_corr": 0.15}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _v1() -> dict[str, Any]:
    return json.loads(V1_PATH.read_text(encoding="utf-8"))


def _v2() -> dict[str, Any]:
    return json.loads(V2_PATH.read_text(encoding="utf-8"))


def _block() -> dict[str, Any]:
    doc = yaml.safe_load(BLOCK_PATH.read_text(encoding="utf-8"))
    return doc["gates"]["gate_m6"]


def _block_tolerances(block: dict[str, Any]) -> dict[str, float]:
    tolerances: dict[str, float] = {}
    for view in block["views"].values():
        for cell, value in view["tolerances"].items():
            tolerances[cell] = value
    return tolerances


def _block_rules(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for view in block["views"].values():
        for cell, rule in view["derivations"]["rules"].items():
            rules[cell] = rule
    return rules


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _oc(
    floor: dict[str, dict[str, float]],
    tolerances: dict[str, float],
    cells: list[str],
) -> tuple[float, float]:
    p_seed = 1.0
    for key in sorted(cells):
        sigma = floor[key]["realized_sigma"]
        tol = tolerances[key]
        p = (2.0 * _normal_cdf(tol / sigma) - 1.0) if sigma > 0 else 1.0
        p_seed *= p
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return round(p_seed, 4), round(p_gate, 4)


# --------------------------------------------------------------------------
# Binding checks (raise AssertionError on drift; reused by the mutation test)
# --------------------------------------------------------------------------
def check_floor_run_sha(block: dict[str, Any]) -> None:
    committed = hashlib.sha256(V2_PATH.read_bytes()).hexdigest()
    assert block["floor_run_sha256"] == committed
    assert block["floor_run"] == "runs/m6_holdout_floors_v2.json"


def check_v1_pause_evidence(block: dict[str, Any], v1: dict[str, Any]) -> None:
    ev = block["v1_pause_evidence"]
    assert ev["sha256"] == V1_SHA
    assert ev["sha256"] == hashlib.sha256(V1_PATH.read_bytes()).hexdigest()
    assert ev["combined_p_gate"] == v1["oc_before_lock"]["p_gate_combined"]
    assert (
        ev["n_gated_flow_cells"] == v1["oc_before_lock"]["n_gated_flow_cells"]
    )
    # v1 stays FROZEN as a PAUSE
    assert v1["oc_before_lock"]["ceremony_may_proceed"] is False


def check_gated_cells(block: dict[str, Any], v2: dict[str, Any]) -> None:
    assert set(_block_tolerances(block)) == set(v2["partition"]["gated"])


def check_tolerances(block: dict[str, Any], v2: dict[str, Any]) -> None:
    block_tol = _block_tolerances(block)
    rules = _block_rules(block)
    floor = v2["floor"]["cells"]
    v2_tol = v2["tolerances"]
    for cell, tol in block_tol.items():
        assert tol == v2_tol[cell], cell
        rule = rules[cell]
        recomputed = float(
            derive_tolerance(
                Decimal(str(floor[cell]["mean"])),
                Decimal(str(floor[cell]["sd"])),
                rule["k"],
                rule["rounding"],
            )
        )
        assert recomputed == tol, (cell, recomputed, tol)
        assert tol <= METRIC_CAP[rule["metric"]] + 1e-12, cell


def check_partition_rollups(block: dict[str, Any], v2: dict[str, Any]) -> None:
    part = v2["partition"]
    roll = block["gated_roll_up"]
    assert roll["n_gated"] == part["n_gated"] == len(_block_tolerances(block))
    assert roll["n_gated_flow_cells"] == part["n_gated_flow_cells"]
    assert roll["n_gated_earnings_cells"] == part["n_gated_earnings_cells"]
    by_fam = Counter(v2["cell_meta"][k]["family"] for k in part["gated"])
    assert dict(roll["by_family"]) == dict(by_fam)


def check_oc(block: dict[str, Any], v2: dict[str, Any]) -> None:
    floor = v2["floor"]["cells"]
    tolerances = v2["tolerances"]
    gated = v2["partition"]["gated"]
    flow = [
        k
        for k in gated
        if v2["cell_meta"][k]["family"]
        in ("mortality", "marital", "disability")
    ]
    earn = [k for k in gated if v2["cell_meta"][k]["family"] == "earnings"]
    p_seed, p_gate = _oc(floor, tolerances, gated)
    _, p_gate_flow = _oc(floor, tolerances, flow)
    _, p_gate_earn = _oc(floor, tolerances, earn)
    combined = block["faithful_candidate_oc"]["combined"]
    assert combined["p_seed_pass"] == p_seed
    assert combined["p_gate_pass_4_of_5"] == p_gate
    assert (
        block["faithful_candidate_oc"]["family_a_flows"]["p_gate_pass_4_of_5"]
        == p_gate_flow
    )
    assert (
        block["faithful_candidate_oc"]["earnings_subfamily"][
            "p_gate_pass_4_of_5"
        ]
        == p_gate_earn
    )
    # block p_gate binds the artifact's oc_before_lock record
    assert (
        block["oc_before_lock"]["p_gate_combined"]
        == v2["oc_before_lock"]["p_gate_combined"]
        == p_gate
    )


def check_ladders(block: dict[str, Any], v2: dict[str, Any]) -> None:
    # coarsening: adopted rungs bind the artifact's ladder record
    block_rungs = block["coarsening_ladder"]["adopted_rungs"]
    art_rungs = {
        t: (rec["adopted_rung"] or "none_cleared_any_rung")
        for t, rec in v2["coarsening_ladder"]["ladders"].items()
    }
    assert block_rungs == art_rungs
    # earnings decompounding: retained set, prune count, and pinned-order head
    el = v2["earnings_decompounding_ladder"]
    assert block["earnings_decompounding_ladder"]["retained"] == el["retained"]
    assert block["earnings_decompounding_ladder"]["n_pruned"] == len(
        el["pruned"]
    )
    assert block["earnings_decompounding_ladder"]["pinned_order_head"] == [
        e["cell"] for e in el["pinned_order"][:3]
    ]


# --------------------------------------------------------------------------
# v1 FROZEN-EVIDENCE tests
# --------------------------------------------------------------------------
def test_v1_is_frozen_pause_evidence():
    v1 = _v1()
    assert hashlib.sha256(V1_PATH.read_bytes()).hexdigest() == V1_SHA
    ob = v1["oc_before_lock"]
    assert ob["ceremony_may_proceed"] is False
    assert ob["clears_weak_power_threshold"] is False
    assert ob["p_gate_combined"] == 0.8449
    assert ob["n_gated_flow_cells"] == 1
    assert v1["partition"]["n_gated"] == 17


def test_v1_shock_window_still_published():
    v1 = _v1()
    shock = v1["shock_window_diagnostics"]
    assert set(shock["floor"]) == set(shock["reference"])
    assert shock["machine_reason"] == "exogenous_shock_outside_model_class"
    # the shock axis never demotes a GATED-window cell (separate diagnostic)
    assert "exogenous_shock_outside_model_class" not in v1["report_reasons"]
    assert "exogenous_shock_outside_model_class" not in set(
        v1["partition"]["report_only"].values()
    )


# --------------------------------------------------------------------------
# v2 bindings
# --------------------------------------------------------------------------
def test_v2_floor_run_sha_binds():
    check_floor_run_sha(_block())


def test_v2_v1_pause_evidence_binds():
    check_v1_pause_evidence(_block(), _v1())


def test_v2_gated_cells_match():
    check_gated_cells(_block(), _v2())


def test_v2_tolerances_bind_and_recompute():
    check_tolerances(_block(), _v2())


def test_v2_partition_rollups_bind():
    check_partition_rollups(_block(), _v2())


def test_v2_operating_characteristic_recomputes():
    check_oc(_block(), _v2())


def test_v2_ladders_bind():
    check_ladders(_block(), _v2())


def test_v2_clears_weak_power_threshold():
    v2 = _v2()
    block = _block()
    ob = v2["oc_before_lock"]
    assert ob["weak_power_p_gate_floor"] == 0.90
    assert ob["ceremony_may_proceed"] is True
    assert block["oc_before_lock"]["ceremony_may_proceed"] is True
    # both directions clear
    assert ob["n_gated_flow_cells"] >= ob["min_gated_flow_cells"]
    assert ob["p_gate_combined"] >= ob["weak_power_p_gate_floor"]
    assert block["locked"] is False


def test_v2_earnings_keeps_one_per_concept_family():
    v2 = _v2()
    el = v2["earnings_decompounding_ladder"]
    covered = set(el["retained_by_concept"])
    assert covered == set(el["concept_families"])
    for concept, cells in el["retained_by_concept"].items():
        assert len(cells) >= 1, concept


def test_v2_coarsening_flow_surface_not_vacuous():
    v2 = _v2()
    # >= 4 gated FLOW cells across >= distinct transition types
    flow = [
        k
        for k in v2["partition"]["gated"]
        if v2["cell_meta"][k]["family"] in ("marital", "disability")
    ]
    assert len(flow) >= 4
    # mortality 85+ stays attrition report-only regardless of the ladder
    ro = v2["partition"]["report_only"]
    assert ro.get("death.85+|female") == "attrition_confounded_truth"
    assert ro.get("death.85+|male") == "attrition_confounded_truth"


def test_v2_is_candidate_blind_and_holds_invariants():
    v2 = _v2()
    block = _block()
    assert "truth-side power arithmetic" in v2["candidate_blind"]
    assert "candidate" in block["candidate_blind"]
    inv = v2["invariants_held_fixed"]
    assert inv["boundary_T_star"] == 2014
    assert inv["no_window_extension_this_round"] is True
    assert inv["flow_k"] == 3
    assert inv["weak_power_p_gate_floor"] == 0.90
    assert block["fit_holdout"]["no_window_extension_this_round"] is True


def test_v2_presence_f6_split_and_household_rule_declared():
    v2 = _v2()
    block = _block()
    assert "held FIXED" in v2["f6_weight_rule"]["definition"]
    assert "held FIXED" in block["f6_weight"]["definition"]
    assert "13" in str(v2["household_id_weight_carriage_rule"]["finding"])
    assert set(block["split_units"]["household_disjoint_families"]) == set(
        v2["split_units"]["household_disjoint_families"]
    )
    # the split-width diagnostic re-checked at the redesigned granularity
    diag = v2["split_width_diagnostic"]
    assert 0.8 <= diag["mean_household_over_person"] <= 1.25
    assert (
        block["split_width_diagnostic"]["mean_household_over_person"]
        == diag["mean_household_over_person"]
    )


def test_gates_yaml_untouched_and_design_amended():
    block = _block()
    assert block["gates_yaml_untouched_by_this_draft"] is True
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text(encoding="utf-8"))
    assert "gate_m6" not in gates.get("gates", {})
    text = DESIGN_PATH.read_text(encoding="utf-8")
    assert "4.10" in text and "Post-pause surface redesign" in text
    assert V1_SHA in text  # the pause evidence is cited in the design


# --------------------------------------------------------------------------
# Mutation guard: >= 8 mutations must be caught (incl. ladder-order, p_gate)
# --------------------------------------------------------------------------
def _deep(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def test_mutations_are_caught():
    v1, v2 = _v1(), _v2()

    # 1. block-draft drift: rename a gated cell -> gated-set mismatch
    m1 = _deep(_block())
    tol = m1["views"]["earnings_log_ratio"]["tolerances"]
    tol["earn_p10.DRIFTED"] = tol.pop("earn_p10.prime")
    with pytest.raises(AssertionError):
        check_gated_cells(m1, v2)

    # 2. tolerance digit: bump one gated tolerance by a least-significant digit
    m2 = _deep(_block())
    m2["views"]["marital_flows"]["tolerances"]["divorce.18-44"] = 0.380
    with pytest.raises(AssertionError):
        check_tolerances(m2, v2)

    # 3. partition count: change the gated roll-up count
    m3 = _deep(_block())
    m3["gated_roll_up"]["n_gated"] = 10
    with pytest.raises(AssertionError):
        check_partition_rollups(m3, v2)

    # 3b. partition family tally drift
    m3b = _deep(_block())
    m3b["gated_roll_up"]["by_family"]["earnings"] = 6
    with pytest.raises(AssertionError):
        check_partition_rollups(m3b, v2)

    # 4. OC drift: perturb the committed combined p_gate
    m4 = _deep(_block())
    m4["faithful_candidate_oc"]["combined"]["p_gate_pass_4_of_5"] = 0.95
    with pytest.raises(AssertionError):
        check_oc(m4, v2)

    # 4b. p_gate drift in the oc_before_lock record
    m4b = _deep(_block())
    m4b["oc_before_lock"]["p_gate_combined"] = 0.91
    with pytest.raises(AssertionError):
        check_oc(m4b, v2)

    # 5. LADDER-ORDER drift: change an adopted coarsening rung
    m5 = _deep(_block())
    m5["coarsening_ladder"]["adopted_rungs"]["divorce"] = "sex_pooled"
    with pytest.raises(AssertionError):
        check_ladders(m5, v2)

    # 5b. LADDER-ORDER drift: reorder the pinned earnings-prune head
    m5b = _deep(_block())
    m5b["earnings_decompounding_ladder"]["pinned_order_head"] = [
        "earn_p90.older",
        "earn_p10.older",
        "earn_p90.prime",
    ]
    with pytest.raises(AssertionError):
        check_ladders(m5b, v2)

    # 5c. retained-set drift
    m5c = _deep(_block())
    m5c["earnings_decompounding_ladder"]["retained"].append("earn_p50.older")
    with pytest.raises(AssertionError):
        check_ladders(m5c, v2)

    # 6. v1 pause-evidence drift: a stale v1 sha is caught
    m6 = _deep(_block())
    m6["v1_pause_evidence"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        check_v1_pause_evidence(m6, v1)

    # 7. v2 floor sha drift
    m7 = _deep(_block())
    m7["floor_run_sha256"] = "f" * 64
    with pytest.raises(AssertionError):
        check_floor_run_sha(m7)
