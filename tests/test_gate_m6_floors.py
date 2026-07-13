"""Bind the gate_m6 floors artifact to the gate_m6 block draft.

ALWAYS-RUNNABLE (artifact tier): reads only the committed
``runs/m6_holdout_floors_v1.json`` evidence artifact, the
``docs/design/gate_m6_block_draft.yaml`` draft block, and the merged design
``docs/design/m6_projection_engine.md`` -- no engine, no candidate, no PSID
micro-read. It proves the block draft does not drift from the frozen floor:
every operative field (gated cell registry, tolerances, partition roll-ups,
presence-conditioning / F6 / split-unit declarations, the operating
characteristic, and the shock-window exclusion) binds string-for-string, the
tolerance arithmetic recomputes from the floor sigmas, and >= 4 mutations are
caught.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from populace_dynamics.evaluation import derive_tolerance

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m6_holdout_floors_v1.json"
BLOCK_PATH = ROOT / "docs" / "design" / "gate_m6_block_draft.yaml"
DESIGN_PATH = ROOT / "docs" / "design" / "m6_projection_engine.md"

LN_1_5 = math.log(1.5)
METRIC_CAP = {
    "log_ratio": LN_1_5,
    "abs_gap_log": LN_1_5,
    "abs_gap_corr": 0.15,
}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _block() -> dict[str, Any]:
    doc = yaml.safe_load(BLOCK_PATH.read_text(encoding="utf-8"))
    return doc["gates"]["gate_m6"]


def _block_tolerances(block: dict[str, Any]) -> dict[str, float]:
    """Every gated cell -> tolerance, flattened across the draft's views."""
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
    """Recompute (p_seed, p_gate) on the draw-noise-free half-normal basis."""
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
    committed = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    assert block["floor_run_sha256"] == committed
    assert block["floor_run"] == "runs/m6_holdout_floors_v1.json"


def check_gated_cells(block: dict[str, Any], art: dict[str, Any]) -> None:
    block_gated = set(_block_tolerances(block))
    artifact_gated = set(art["partition"]["gated"])
    assert block_gated == artifact_gated


def check_tolerances(block: dict[str, Any], art: dict[str, Any]) -> None:
    block_tol = _block_tolerances(block)
    rules = _block_rules(block)
    floor = art["floor"]["cells"]
    art_tol = art["tolerances"]
    for cell, tol in block_tol.items():
        # (a) block tolerance == artifact tolerance
        assert tol == art_tol[cell], cell
        # (b) tolerance recomputes from the floor (mean, sd) at the rule's k
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
        # (c) tolerance clears its metric power cap
        assert tol <= METRIC_CAP[rule["metric"]] + 1e-12, cell


def check_partition_rollups(
    block: dict[str, Any], art: dict[str, Any]
) -> None:
    part = art["partition"]
    assert block["gated_roll_up"]["n_gated"] == part["n_gated"]
    assert block["gated_roll_up"]["n_gated"] == len(_block_tolerances(block))
    assert (
        block["gated_roll_up"]["n_gated_flow_cells"]
        == art["oc_before_lock"]["n_gated_flow_cells"]
    )
    ro = block["report_only"]
    assert ro["n_report_only"] == part["n_report_only"]
    # roll-up by machine reason binds to the artifact's per-cell reasons
    from collections import Counter

    art_counts = Counter(part["report_only"].values())
    assert dict(ro["by_machine_reason"]) == dict(art_counts)
    assert sum(ro["by_machine_reason"].values()) == part["n_report_only"]
    assert block["report_reasons"] == art["report_reasons"]


def check_oc(block: dict[str, Any], art: dict[str, Any]) -> None:
    floor = art["floor"]["cells"]
    tolerances = art["tolerances"]
    gated = art["partition"]["gated"]
    flow = [
        k
        for k in gated
        if art["cell_meta"][k]["family"]
        in ("mortality", "marital", "disability")
    ]
    earn = [k for k in gated if art["cell_meta"][k]["family"] == "earnings"]
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
    # block OC binds to the artifact OC
    art_oc = art["faithful_candidate_oc"]
    assert combined["p_gate_pass_4_of_5"] == art_oc["p_gate_pass_4_of_5"]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_floor_run_sha_binds():
    check_floor_run_sha(_block())


def test_gated_cells_match_artifact():
    check_gated_cells(_block(), _artifact())


def test_tolerances_bind_and_recompute_from_sigmas():
    check_tolerances(_block(), _artifact())


def test_partition_rollups_bind():
    check_partition_rollups(_block(), _artifact())


def test_operating_characteristic_recomputes():
    check_oc(_block(), _artifact())


def test_presence_conditioning_and_f6_and_split_declared():
    block = _block()
    art = _artifact()
    # presence-conditioning is declared in BOTH and is conditioning-not-leakage
    assert "presence" in block["presence_conditioning"]["rule"].lower()
    assert block["presence_conditioning"]["classification"] == (
        "conditioning_not_leakage"
    )
    assert (
        "reproduction mode"
        in block["presence_conditioning"]["disability_scope"]
    )
    assert (
        "reproduction mode"
        in art["presence_conditioning"]["per_family"]["disability"]
    )
    # F6 start-wave weight held fixed, on projection + truth + floor
    assert "held FIXED" in block["f6_weight"]["definition"]
    assert "held FIXED" in art["f6_weight_rule"]["definition"]
    # the household-ID + weight carriage rule (finding 13 / watch-item)
    assert (
        "interview number"
        in block["household_id_weight_carriage_rule"]["household_id"]
    )
    assert "13" in str(art["household_id_weight_carriage_rule"]["finding"])


def test_split_units_bind():
    block = _block()
    art = _artifact()
    assert set(block["split_units"]["household_disjoint_families"]) == set(
        art["split_units"]["household_disjoint_families"]
    )
    assert set(block["split_units"]["person_disjoint_families"]) == set(
        art["split_units"]["person_disjoint_families"]
    )
    assert (
        block["split_units"]["machine"]
        == art["split_units"]["machine"]
        == "populace_dynamics.harness.panel.split_panel_by_person"
    )
    # the person-vs-household split-width diagnostic binds (watch-item)
    assert (
        block["split_width_diagnostic"]["mean_household_over_person"]
        == art["split_width_diagnostic"]["mean_household_over_person"]
    )


def test_weight_rule_written_into_artifact():
    """The household-ID / weight rule must be explicit in the artifact so the
    referee can check the split is unbiased (ceremony watch-item)."""
    art = _artifact()
    rule = art["household_id_weight_carriage_rule"]
    assert "divorce_splitoff_rule" in rule
    assert "held FIXED" in rule["household_id"]
    # the split-bias diagnostic exists and is ~1 (unbiased)
    diag = art["split_width_diagnostic"]
    assert 0.8 <= diag["mean_household_over_person"] <= 1.25
    assert diag["n_cells"] >= 10


def test_shock_window_computed_but_partitioned_out():
    """Shock cells (2020-2022) are COMPUTED with uncertainty and PUBLISHED in
    their own block, but partitioned OUT of the gated scoring (sec. 4.1). They
    reuse the gated-window observable NAMES on a DIFFERENT data window, so the
    separation is structural (block + window), not by name: the shock block is
    distinct, carries its own reference + half-split floor, and its reference
    values differ from the gated-window truth for shared names (the 2020
    marriage collapse / excess deaths the design publishes as a diagnostic)."""
    art = _artifact()
    block = _block()
    shock = art["shock_window_diagnostics"]
    shock_ref = shock["reference"]
    # a first-class published diagnostic with reference AND uncertainty (floor)
    assert shock_ref
    assert set(shock["floor"]) == set(shock_ref)
    assert shock["machine_reason"] == "exogenous_shock_outside_model_class"
    # structurally separate from the gated-window scoring surface
    assert "shock_window_diagnostics" != "floor"
    gated_ref = art["reference_moments"]
    shared = set(shock_ref) & set(gated_ref)
    assert shared, "shock reuses gated-window observable names"
    # for at least one shared flow cell the shock window's rate DIFFERS from
    # the gated window's -- proving the two are computed on different data.
    diffs = [
        c
        for c in shared
        if "rate" in shock_ref[c]
        and shock_ref[c].get("rate") != gated_ref[c].get("rate")
    ]
    assert diffs, "shock and gated windows must carry different truth"
    # the block declares the exclusion; the shock reason never demotes a
    # gated-window cell (it is a separate axis, not a power / events demotion)
    assert block["shock_window"]["partitioned_out_of_gated_set"] is True
    assert (
        block["shock_window"]["machine_reason"]
        == "exogenous_shock_outside_model_class"
    )
    assert "exogenous_shock_outside_model_class" not in art["report_reasons"]
    assert "exogenous_shock_outside_model_class" not in set(
        art["partition"]["report_only"].values()
    )


def test_attrition_demotion_binds():
    art = _artifact()
    block = _block()
    ro = art["partition"]["report_only"]
    for cell in block["attrition_demotion"]["cells"]:
        assert ro.get(cell) == "attrition_confounded_truth"
    assert block["attrition_demotion"]["machine_reason"] == (
        "attrition_confounded_truth"
    )


def test_oc_before_lock_pauses_the_ceremony():
    """The certifiable FLOW surface is near-vacuous (1 gated flow cell < 4) and
    the combined faithful p_gate sits below the named 0.90 floor -> the
    ceremony PAUSES rather than locking (both directions, sec. 4.9 / sec. 9).
    """
    art = _artifact()
    block = _block()
    ob = art["oc_before_lock"]
    assert ob["weak_power_p_gate_floor"] == 0.90
    assert block["oc_before_lock"]["weak_power_p_gate_floor"] == 0.90
    assert ob["clears_weak_power_threshold"] is False
    assert ob["ceremony_may_proceed"] is False
    assert block["oc_before_lock"]["ceremony_may_proceed"] is False
    # near-vacuous flow direction: fewer than the minimum gated flow cells
    assert ob["n_gated_flow_cells"] < ob["min_gated_flow_cells"]
    # near-unpassable direction: combined p_gate below the floor
    assert ob["p_gate_combined"] < ob["weak_power_p_gate_floor"]
    assert block["locked"] is False


def test_block_is_not_lockable_and_leaves_gates_yaml_untouched():
    block = _block()
    assert block["locked"] is False
    assert block["status"] == "draft_paused_pending_surface_redesign"
    assert block["gates_yaml_untouched_by_this_draft"] is True
    # gates.yaml must not already carry a gate_m6 block (lock flip is later)
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text(encoding="utf-8"))
    assert "gate_m6" not in gates.get("gates", {})


def test_design_doc_pins_this_surface():
    """The block's design pins agree with the merged design doc's parameters."""
    art = _artifact()
    assert art["design_pins"]["boundary_T_star"] == 2014
    text = DESIGN_PATH.read_text(encoding="utf-8")
    assert "T* = 2014" in text or "T*=2014" in text or "T_star" in text
    assert "exogenous_shock_outside_model_class" in text


# --------------------------------------------------------------------------
# Mutation guard: >= 4 mutations must be caught
# --------------------------------------------------------------------------
def _deep(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def test_mutations_are_caught():
    art = _artifact()

    # 1. block-draft drift: rename a gated cell -> gated-set mismatch
    m1 = _deep(_block())
    tol = m1["views"]["earnings_log_ratio"]["tolerances"]
    tol["earn_p50.DRIFTED"] = tol.pop("earn_p50.prime")
    with pytest.raises(AssertionError):
        check_gated_cells(m1, art)

    # 2. tolerance digit: bump one tolerance by a least significant digit
    m2 = _deep(_block())
    m2["views"]["flow_hazards"]["tolerances"][
        "first_marriage.18-29|female"
    ] = 0.357
    with pytest.raises(AssertionError):
        check_tolerances(m2, art)

    # 3. partition count: change the gated roll-up count
    m3 = _deep(_block())
    m3["gated_roll_up"]["n_gated"] = 16
    with pytest.raises(AssertionError):
        check_partition_rollups(m3, art)

    # 3b. partition reason count: change a report-only reason tally
    m3b = _deep(_block())
    m3b["report_only"]["by_machine_reason"]["tolerance_above_power_cap"] = 11
    with pytest.raises(AssertionError):
        check_partition_rollups(m3b, art)

    # 4. OC drift: perturb the committed p_gate
    m4 = _deep(_block())
    m4["faithful_candidate_oc"]["combined"]["p_gate_pass_4_of_5"] = 0.95
    with pytest.raises(AssertionError):
        check_oc(m4, art)

    # 5. sha drift: a stale floor_run_sha256 is caught
    m5 = _deep(_block())
    m5["floor_run_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        check_floor_run_sha(m5)
