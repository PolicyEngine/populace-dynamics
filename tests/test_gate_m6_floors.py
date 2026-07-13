"""Bind the gate_m6 floors artifacts to the gate_m6 block draft.

ALWAYS-RUNNABLE (artifact tier): reads only the committed floors JSON artifacts
(``runs/m6_holdout_floors_v1.json`` = FROZEN pause evidence,
``runs/m6_holdout_floors_v2.json`` = FROZEN prior surface / lineage,
``runs/m6_holdout_floors_v3.json`` = the completed-ladder surface the block now
binds), the ``docs/design/gate_m6_block_draft.yaml`` draft block (now v3), and
the merged design ``docs/design/m6_projection_engine.md`` -- no engine, no
candidate, no PSID micro-read.

Three binding layers:
  * v1 FROZEN-EVIDENCE -- v1 records the OC-before-lock PAUSE (near-vacuous flow
    surface + combined p_gate 0.8449 < 0.90). Frozen; byte-hash asserted.
  * v2 FROZEN-LINEAGE -- v2 cleared the threshold but with an INCOMPLETE marital
    ladder enumeration (symmetric age-2 only, 4 flow cells). Frozen; byte-hash
    asserted. Read for no v3 choice.
  * v3 BINDINGS -- the block binds the completed surface string-for-string: the
    coarsening ladder WITH the asymmetric age-2 rung (referee amendment 2), the
    11-cell registry (5 flow + 6 earnings), tolerance recompute from the v3 floor
    sigmas, the OC recompute, the year pins (referee amendment 4), the
    not_certified declaration (referee amendment 1), and the SSA/NCHS anchor
    deliverable (referee amendment 3).

>= 8 mutations are caught, INCLUDING the referee's missed one (a year-pin drift).
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
V3_PATH = ROOT / "runs" / "m6_holdout_floors_v3.json"
BLOCK_PATH = ROOT / "docs" / "design" / "gate_m6_block_draft.yaml"
DESIGN_PATH = ROOT / "docs" / "design" / "m6_projection_engine.md"

# v1 / v2 are FROZEN: their byte-hashes are lineage evidence and never move.
V1_SHA = "16c28d8cd9095e5233ab224c659c8d5b9eb1621099e2524455a3a8ff8e88d318"
V2_SHA = "3f273d474692917b01055f85830cb982dfbe9e63070581c99975aa799759b9a0"
V3_SHA = "e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77"
LN_1_5 = math.log(1.5)
METRIC_CAP = {"log_ratio": LN_1_5, "abs_gap_log": LN_1_5, "abs_gap_corr": 0.15}

# The asymmetric age-2 marital rung the mortality ladder already used, and which
# v2 never enumerated -- the single change v3 makes (referee amendment 2).
ASYMMETRIC_AGE2_RUNG = "sex_pooled_age2p"
ASYMMETRIC_AGE2_BANDS = [[18, 64], [65, 120]]

# The not_certified margins the block MUST name (referee amendment 1); mortality
# drift is named FIRST.
REQUIRED_NOT_CERTIFIED = {
    "mortality_drift",
    "widowhood",
    "shock_window_2020_2022",
    "entrants_open_panel",
    "autocorrelation_lag5",
    "forward_projection_2100_extrapolation",
}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _v1() -> dict[str, Any]:
    return json.loads(V1_PATH.read_text(encoding="utf-8"))


def _v2() -> dict[str, Any]:
    return json.loads(V2_PATH.read_text(encoding="utf-8"))


def _v3() -> dict[str, Any]:
    return json.loads(V3_PATH.read_text(encoding="utf-8"))


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
    committed = hashlib.sha256(V3_PATH.read_bytes()).hexdigest()
    assert block["floor_run_sha256"] == committed == V3_SHA
    assert block["floor_run"] == "runs/m6_holdout_floors_v3.json"


def check_v1_pause_evidence(block: dict[str, Any], v1: dict[str, Any]) -> None:
    ev = block["v1_pause_evidence"]
    assert ev["sha256"] == V1_SHA
    assert ev["sha256"] == hashlib.sha256(V1_PATH.read_bytes()).hexdigest()
    assert ev["combined_p_gate"] == v1["oc_before_lock"]["p_gate_combined"]
    assert (
        ev["n_gated_flow_cells"] == v1["oc_before_lock"]["n_gated_flow_cells"]
    )
    assert ev["frozen"] is True
    # v1 stays FROZEN as a PAUSE
    assert v1["oc_before_lock"]["ceremony_may_proceed"] is False


def check_v2_lineage(block: dict[str, Any], v2: dict[str, Any]) -> None:
    lin = block["v2_lineage"]
    assert lin["sha256"] == V2_SHA
    assert lin["sha256"] == hashlib.sha256(V2_PATH.read_bytes()).hexdigest()
    assert lin["frozen"] is True
    # v2 cleared, but with 4 flow cells (incomplete marital enumeration)
    assert (
        lin["combined_p_gate"]
        == v2["oc_before_lock"]["p_gate_combined"]
        == 0.9067
    )
    assert (
        lin["n_gated_flow_cells"]
        == v2["oc_before_lock"]["n_gated_flow_cells"]
        == 4
    )
    assert v2["oc_before_lock"]["ceremony_may_proceed"] is True


def check_gated_cells(block: dict[str, Any], v3: dict[str, Any]) -> None:
    assert set(_block_tolerances(block)) == set(v3["partition"]["gated"])


def check_tolerances(block: dict[str, Any], v3: dict[str, Any]) -> None:
    block_tol = _block_tolerances(block)
    rules = _block_rules(block)
    floor = v3["floor"]["cells"]
    v3_tol = v3["tolerances"]
    for cell, tol in block_tol.items():
        assert tol == v3_tol[cell], cell
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


def check_partition_rollups(block: dict[str, Any], v3: dict[str, Any]) -> None:
    part = v3["partition"]
    roll = block["gated_roll_up"]
    assert roll["n_gated"] == part["n_gated"] == len(_block_tolerances(block))
    assert roll["n_gated_flow_cells"] == part["n_gated_flow_cells"] == 5
    assert (
        roll["n_gated_earnings_cells"] == part["n_gated_earnings_cells"] == 6
    )
    by_fam = Counter(v3["cell_meta"][k]["family"] for k in part["gated"])
    assert dict(roll["by_family"]) == dict(by_fam)
    assert dict(roll["by_family"]) == {
        "marital": 3,
        "disability": 2,
        "earnings": 6,
    }


def check_oc(block: dict[str, Any], v3: dict[str, Any]) -> None:
    floor = v3["floor"]["cells"]
    tolerances = v3["tolerances"]
    gated = v3["partition"]["gated"]
    flow = [
        k
        for k in gated
        if v3["cell_meta"][k]["family"]
        in ("mortality", "marital", "disability")
    ]
    earn = [k for k in gated if v3["cell_meta"][k]["family"] == "earnings"]
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
        == v3["oc_before_lock"]["p_gate_combined"]
        == p_gate
    )


def check_ladders(block: dict[str, Any], v3: dict[str, Any]) -> None:
    # coarsening: adopted rungs bind the artifact's ladder record
    block_rungs = block["coarsening_ladder"]["adopted_rungs"]
    art_rungs = {
        t: (rec["adopted_rung"] or "none_cleared_any_rung")
        for t, rec in v3["coarsening_ladder"]["ladders"].items()
    }
    assert block_rungs == art_rungs
    # earnings decompounding: retained set, prune count, and pinned-order head
    el = v3["earnings_decompounding_ladder"]
    assert block["earnings_decompounding_ladder"]["retained"] == el["retained"]
    assert block["earnings_decompounding_ladder"]["n_pruned"] == len(
        el["pruned"]
    )
    assert len(el["pruned"]) == 10  # one more than v2 (the forced prune)
    assert block["earnings_decompounding_ladder"]["pinned_order_head"] == [
        e["cell"] for e in el["pinned_order"][:3]
    ]


def check_ladder_completeness(
    block: dict[str, Any], v3: dict[str, Any]
) -> None:
    """Referee amendment 2: the marital ladder now enumerates the asymmetric
    age-2 rung the mortality ladder already used, remarriage adopts it, and
    widowhood is swept at it and still clears nothing."""
    comp = block["coarsening_ladder"]["marital_ladder_completion"]
    added = comp["added_rung"]
    assert added["label"] == ASYMMETRIC_AGE2_RUNG
    assert added["bands"] == ASYMMETRIC_AGE2_BANDS
    # the artifact enumerates the same completed marital rung vocabulary
    art_comp = v3["coarsening_ladder"]["marital_ladder_completion"]
    assert ASYMMETRIC_AGE2_RUNG in art_comp["marital_rungs_enumerated"]
    # remarriage clears ONLY at the asymmetric rung -> gates remarriage.18-64
    rem = v3["coarsening_ladder"]["ladders"]["remarriage"]
    assert rem["adopted_rung"] == ASYMMETRIC_AGE2_RUNG
    assert rem["gated"] == ["remarriage.18-64"]
    assert ASYMMETRIC_AGE2_RUNG in [s["rung"] for s in rem["steps"]]
    assert (
        block["coarsening_ladder"]["adopted_rungs"]["remarriage"]
        == ASYMMETRIC_AGE2_RUNG
    )
    assert "remarriage.18-64" in v3["partition"]["gated"]
    # widowhood swept at the asymmetric rung too, still clears nothing
    wid = v3["coarsening_ladder"]["ladders"]["widowhood"]
    assert wid["adopted_rung"] is None
    assert ASYMMETRIC_AGE2_RUNG in [s["rung"] for s in wid["steps"]]
    assert (
        block["coarsening_ladder"]["adopted_rungs"]["widowhood"]
        == "none_cleared_any_rung"
    )
    # first_marriage / divorce dispositions are UNCHANGED from v2
    assert (
        v3["coarsening_ladder"]["ladders"]["first_marriage"]["adopted_rung"]
        == "age_x_sex"
    )
    assert (
        v3["coarsening_ladder"]["ladders"]["divorce"]["adopted_rung"]
        == "sex_pooled_age2"
    )


def check_year_pins(block: dict[str, Any], v3: dict[str, Any]) -> None:
    """Referee amendment 4 (the missed mutation): the block's fit/holdout year
    pins, shock windows, and scoring seeds are string-bound to the v3 artifact's
    design_pins, so a year-pin mutation fails a test."""
    pins = v3["design_pins"]
    fh = block["fit_holdout"]
    assert fh["boundary_T_star"] == pins["boundary_T_star"] == 2014
    assert fh["seed_wave"] == pins["seed_wave"] == 2015
    assert fh["gated_flow_event_years"] == pins["gated_flow_event_years"]
    assert (
        fh["gated_earnings_reference_years"]
        == pins["gated_earnings_reference_years"]
        == [2016, 2018]
    )
    sw = block["shock_window"]
    assert (
        sw["earnings_shock_reference_years"]
        == pins["shock_earnings_reference_years"]
    )
    assert sw["flow_shock_event_years"] == pins["shock_flow_event_years"]
    sc = block["scoring"]
    assert sc["floor_seeds"] == pins["floor_seeds"]
    assert sc["gate_seeds"] == pins["gate_seeds"]
    assert sc["conjunction"] == pins["conjunction"]
    assert dict(sc["mixed_k"]) == dict(pins["mixed_k"])


def check_not_certified(block: dict[str, Any]) -> None:
    """Referee amendment 1: the negative surface is declared out loud at the
    same prominence as `covers`, mortality drift FIRST."""
    nc = block["not_certified"]
    margins = [entry["margin"] for entry in nc]
    assert margins[0] == "mortality_drift"  # named FIRST
    assert REQUIRED_NOT_CERTIFIED <= set(margins)
    for entry in nc:
        assert entry["detail"].strip(), entry["margin"]
    mort = next(e for e in nc if e["margin"] == "mortality_drift")
    assert "NOTHING" in mort["detail"]


def check_ssa_nchs_anchor(block: dict[str, Any]) -> None:
    """Referee amendment 3: the named report-only SSA/NCHS mortality anchor."""
    d = block["deliverables"]["ssa_nchs_life_table_mortality_anchor"]
    assert d["family"] == "B"
    assert d["status"] == "report_only"
    # |ln|-gating external LEVELS stays rejected (the W1 concept-bridge lesson)
    assert "REJECTED" in d["gating"]
    assert "circularity_disclosure" in d
    assert "NCHS" in d["what"] and "SSA" in d["what"]
    assert "M7" in d["required_before_lock_flip_for"]


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
# v2 FROZEN-LINEAGE tests
# --------------------------------------------------------------------------
def test_v2_is_frozen_lineage():
    v2 = _v2()
    assert hashlib.sha256(V2_PATH.read_bytes()).hexdigest() == V2_SHA
    # v2 cleared, but with the INCOMPLETE marital enumeration (4 flow cells)
    ob = v2["oc_before_lock"]
    assert ob["ceremony_may_proceed"] is True
    assert ob["p_gate_combined"] == 0.9067
    assert ob["n_gated_flow_cells"] == 4
    # v2's marital ladder never enumerated the asymmetric age-2 rung
    rem = v2["coarsening_ladder"]["ladders"]["remarriage"]
    assert rem["adopted_rung"] is None
    assert ASYMMETRIC_AGE2_RUNG not in [s["rung"] for s in rem["steps"]]


def test_v2_lineage_binds():
    check_v2_lineage(_block(), _v2())


# --------------------------------------------------------------------------
# v3 bindings
# --------------------------------------------------------------------------
def test_v3_floor_run_sha_binds():
    check_floor_run_sha(_block())


def test_v3_v1_pause_evidence_binds():
    check_v1_pause_evidence(_block(), _v1())


def test_v3_gated_cells_match():
    check_gated_cells(_block(), _v3())


def test_v3_tolerances_bind_and_recompute():
    check_tolerances(_block(), _v3())


def test_v3_partition_rollups_bind():
    check_partition_rollups(_block(), _v3())


def test_v3_operating_characteristic_recomputes():
    check_oc(_block(), _v3())


def test_v3_ladders_bind():
    check_ladders(_block(), _v3())


def test_v3_marital_ladder_completeness():
    check_ladder_completeness(_block(), _v3())


def test_v3_year_pins_bind():
    check_year_pins(_block(), _v3())


def test_v3_not_certified_declared():
    check_not_certified(_block())


def test_v3_ssa_nchs_anchor_deliverable():
    check_ssa_nchs_anchor(_block())


def test_v3_clears_weak_power_threshold():
    v3 = _v3()
    block = _block()
    ob = v3["oc_before_lock"]
    assert ob["weak_power_p_gate_floor"] == 0.90
    assert ob["ceremony_may_proceed"] is True
    assert block["oc_before_lock"]["ceremony_may_proceed"] is True
    # both directions clear
    assert ob["n_gated_flow_cells"] >= ob["min_gated_flow_cells"]
    assert ob["n_gated_flow_cells"] == 5
    assert ob["p_gate_combined"] >= ob["weak_power_p_gate_floor"]
    assert ob["p_gate_combined"] == 0.9087
    assert block["locked"] is False


def test_v3_third_vacuity_guard_restored():
    # the referee's minor observation: v2 dropped v1's third vacuity guard;
    # v3 restores it (n_at_cap = 0, so no effect, but registry-faithful).
    v3 = _v3()
    ob = v3["oc_before_lock"]
    assert ob["n_gated_tolerances_at_cap"] == 0
    assert ob["clears_not_all_tolerances_capped"] is True
    assert (
        _block()["oc_before_lock"]["clears_not_all_tolerances_capped"] is True
    )


def test_v3_earnings_keeps_one_per_concept_family():
    v3 = _v3()
    el = v3["earnings_decompounding_ladder"]
    covered = set(el["retained_by_concept"])
    assert covered == set(el["concept_families"])
    for concept, cells in el["retained_by_concept"].items():
        assert len(cells) >= 1, concept
    assert len(el["retained"]) == 6


def test_v3_coarsening_flow_surface_not_vacuous():
    v3 = _v3()
    # >= 5 gated FLOW cells across >= 3 distinct marital transition types + both
    # disability flows -- strictly larger than v2's 4.
    flow = [
        k
        for k in v3["partition"]["gated"]
        if v3["cell_meta"][k]["family"] in ("marital", "disability")
    ]
    assert len(flow) == 5
    marital = [k for k in flow if v3["cell_meta"][k]["family"] == "marital"]
    transition_types = {k.split(".")[0] for k in marital}
    assert transition_types == {"first_marriage", "divorce", "remarriage"}
    # mortality 85+ stays attrition report-only regardless of the ladder
    ro = v3["partition"]["report_only"]
    assert ro.get("death.85+|female") == "attrition_confounded_truth"
    assert ro.get("death.85+|male") == "attrition_confounded_truth"


def test_v3_is_candidate_blind_and_holds_invariants():
    v3 = _v3()
    block = _block()
    assert "truth-side power arithmetic" in v3["candidate_blind"]
    assert "candidate" in block["candidate_blind"]
    inv = v3["invariants_held_fixed"]
    assert inv["boundary_T_star"] == 2014
    assert inv["no_window_extension_this_round"] is True
    assert inv["flow_k"] == 3
    assert inv["weak_power_p_gate_floor"] == 0.90
    assert block["fit_holdout"]["no_window_extension_this_round"] is True


def test_v3_presence_f6_split_and_household_rule_declared():
    v3 = _v3()
    block = _block()
    assert "held FIXED" in v3["f6_weight_rule"]["definition"]
    assert "held FIXED" in block["f6_weight"]["definition"]
    assert "13" in str(v3["household_id_weight_carriage_rule"]["finding"])
    assert set(block["split_units"]["household_disjoint_families"]) == set(
        v3["split_units"]["household_disjoint_families"]
    )
    # the split-width diagnostic re-checked at the redesigned granularity
    diag = v3["split_width_diagnostic"]
    assert 0.8 <= diag["mean_household_over_person"] <= 1.25
    assert (
        block["split_width_diagnostic"]["mean_household_over_person"]
        == diag["mean_household_over_person"]
    )
    # now three marital household-split flow cells feed the diagnostic
    assert diag["n_household_gated_flow_cells"] == 3


def test_gates_yaml_untouched_and_design_amended():
    block = _block()
    assert block["gates_yaml_untouched_by_this_draft"] is True
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text(encoding="utf-8"))
    assert "gate_m6" not in gates.get("gates", {})
    text = DESIGN_PATH.read_text(encoding="utf-8")
    assert "4.10" in text and "Post-pause surface redesign" in text
    assert V1_SHA in text  # the pause evidence is cited in the design
    assert V3_SHA in text  # the binding v3 artifact is cited in the design
    # the v2->v3 correction and the negative surface are on the public record
    assert "v2→v3 correction" in text
    assert "not_certified" in text


# --------------------------------------------------------------------------
# Mutation guard: >= 8 mutations must be caught (incl. the year-pin miss)
# --------------------------------------------------------------------------
def _deep(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def test_mutations_are_caught():
    v1, v2, v3 = _v1(), _v2(), _v3()

    # 1. block-draft drift: rename a gated cell -> gated-set mismatch
    m1 = _deep(_block())
    tol = m1["views"]["earnings_log_ratio"]["tolerances"]
    tol["earn_p10.DRIFTED"] = tol.pop("earn_p10.prime")
    with pytest.raises(AssertionError):
        check_gated_cells(m1, v3)

    # 2. tolerance digit: bump one gated tolerance by a least-significant digit
    m2 = _deep(_block())
    m2["views"]["marital_flows"]["tolerances"]["remarriage.18-64"] = 0.404
    with pytest.raises(AssertionError):
        check_tolerances(m2, v3)

    # 3. partition count: change the gated roll-up count
    m3 = _deep(_block())
    m3["gated_roll_up"]["n_gated"] = 10
    with pytest.raises(AssertionError):
        check_partition_rollups(m3, v3)

    # 3b. partition family tally drift (v3 marital is 3, not 2)
    m3b = _deep(_block())
    m3b["gated_roll_up"]["by_family"]["marital"] = 2
    with pytest.raises(AssertionError):
        check_partition_rollups(m3b, v3)

    # 4. OC drift: perturb the committed combined p_gate
    m4 = _deep(_block())
    m4["faithful_candidate_oc"]["combined"]["p_gate_pass_4_of_5"] = 0.95
    with pytest.raises(AssertionError):
        check_oc(m4, v3)

    # 4b. p_gate drift in the oc_before_lock record
    m4b = _deep(_block())
    m4b["oc_before_lock"]["p_gate_combined"] = 0.91
    with pytest.raises(AssertionError):
        check_oc(m4b, v3)

    # 5. LADDER-ORDER drift: change the remarriage adopted rung
    m5 = _deep(_block())
    m5["coarsening_ladder"]["adopted_rungs"]["remarriage"] = "sex_pooled_age1"
    with pytest.raises(AssertionError):
        check_ladders(m5, v3)

    # 5b. LADDER-ORDER drift: reorder the pinned earnings-prune head
    m5b = _deep(_block())
    m5b["earnings_decompounding_ladder"]["pinned_order_head"] = [
        "earn_p90.older",
        "earn_p10.older",
        "earn_p90.prime",
    ]
    with pytest.raises(AssertionError):
        check_ladders(m5b, v3)

    # 5c. retained-set drift (v2's 7th cell must NOT be retained in v3)
    m5c = _deep(_block())
    m5c["earnings_decompounding_ladder"]["retained"].append(
        "earn_dlog_mean.older"
    )
    with pytest.raises(AssertionError):
        check_ladders(m5c, v3)

    # 5d. LADDER-COMPLETENESS drift: drop the asymmetric age-2 rung bands
    m5d = _deep(_block())
    m5d["coarsening_ladder"]["marital_ladder_completion"]["added_rung"][
        "bands"
    ] = [[18, 44], [45, 120]]
    with pytest.raises(AssertionError):
        check_ladder_completeness(m5d, v3)

    # 6. v1 pause-evidence drift: a stale v1 sha is caught
    m6 = _deep(_block())
    m6["v1_pause_evidence"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        check_v1_pause_evidence(m6, v1)

    # 6b. v2 lineage drift: a stale v2 sha is caught
    m6b = _deep(_block())
    m6b["v2_lineage"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        check_v2_lineage(m6b, v2)

    # 7. v3 floor sha drift
    m7 = _deep(_block())
    m7["floor_run_sha256"] = "f" * 64
    with pytest.raises(AssertionError):
        check_floor_run_sha(m7)

    # 8. THE REFEREE'S MISSED MUTATION (amendment 4): drift the gated earnings
    # reference years to [2016, 2020]. This passed all v2 tests; it must fail
    # now that the block's pins are bound to the artifact's design_pins.
    m8 = _deep(_block())
    m8["fit_holdout"]["gated_earnings_reference_years"] = [2016, 2020]
    with pytest.raises(AssertionError):
        check_year_pins(m8, v3)

    # 9. not_certified drift: mortality not named FIRST (referee amendment 1)
    m9 = _deep(_block())
    nc = m9["not_certified"]
    nc[0], nc[1] = nc[1], nc[0]
    with pytest.raises(AssertionError):
        check_not_certified(m9)

    # 10. SSA/NCHS anchor drift: promote external levels to gated
    m10 = _deep(_block())
    m10["deliverables"]["ssa_nchs_life_table_mortality_anchor"][
        "status"
    ] = "gated"
    with pytest.raises(AssertionError):
        check_ssa_nchs_anchor(m10)
