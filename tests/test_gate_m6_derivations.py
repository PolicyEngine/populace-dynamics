"""LOCKED-HOT bindings for the ratified gate_m6 block in ``gates.yaml``.

Companion to ``tests/test_gate_m6_floors.py`` (which binds the ``docs/design/
gate_m6_block_draft.yaml`` DRAFT to the floor artifacts). This file binds the
LIVE ``gates.gate_m6`` block the lock flip inserted -- the block draft carried
VERBATIM with exactly the lock-time deltas (``locked: true``, ``status:
locked``, and the ``history`` ceremony entry) -- string-for-string to the
FROZEN v3 floor ``runs/m6_holdout_floors_v3.json``.

The gate is LOCKED, so these bindings run UNCONDITIONALLY (the 2a lesson): no
skip guard, no env gate. ALWAYS-RUNNABLE (artifact tier): reads only the
committed ``runs/*.json`` floor artifacts and ``gates.yaml`` -- no engine, no
candidate, no PSID micro-read.

Binding layers:
  * lock deltas -- ``locked``/``status`` flipped, ``history`` records the
    ceremony (comment ids 4958425437 / 4958956779 / 4959310892, the v1/v2/v3
    shas, the PR #172 / squash 1818784 ratification).
  * 11-cell registry equality (5 flow + 6 earnings), each tolerance
    ``round(mean + k*sd, 3)`` recomputed from the v3 floor sigmas and capped at
    the metric cap, the faithful-candidate OC recomputed
    ``p_seed 0.8934 / p_gate 0.9087`` from the realized sigmas.
  * the not_certified negative surface (mortality drift FIRST), the year pins /
    shock-window exclusions bound to the artifact's design_pins, the v1/v2/v3
    lineage shas, the named SSA/NCHS mortality anchor, and the presence /
    F6 / split-unit declarations.

>= 6 mutations are caught (tolerance digit, cell add/drop, OC drift,
not_certified reorder, year pin, history sha).
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
GATES = ROOT / "gates.yaml"
V1_PATH = ROOT / "runs" / "m6_holdout_floors_v1.json"
V2_PATH = ROOT / "runs" / "m6_holdout_floors_v2.json"
V3_PATH = ROOT / "runs" / "m6_holdout_floors_v3.json"

# v1 / v2 are FROZEN lineage; v3 is the ratified frozen floor the lock binds.
V1_SHA = "16c28d8cd9095e5233ab224c659c8d5b9eb1621099e2524455a3a8ff8e88d318"
V2_SHA = "3f273d474692917b01055f85830cb982dfbe9e63070581c99975aa799759b9a0"
V3_SHA = "e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77"
LN_1_5 = math.log(1.5)
METRIC_CAP = {"log_ratio": LN_1_5, "abs_gap_log": LN_1_5, "abs_gap_corr": 0.15}

ASYMMETRIC_AGE2_RUNG = "sex_pooled_age2p"

# The design pin: PR #175 amended the draft and re-pinned design_commit off the
# stale 1d83a221 (PR #170) to PR #175's design commit; the flip FINALIZES that
# pin to a squash-merge per the draft's design_commit_note. It was first
# finalized to the #175 squash (ce9893b), then RE-FINALIZED to the #178 squash
# (4c6a0f6) once SS 2.7.6 COMPLETED the forward law the covers references.
DESIGN_PR = "175"
DESIGN_COMMIT_DRAFT = "d6abb16b0a034ca08a26e3eb8fc9211967c53259"  # PR #175 pin
DESIGN_COMMIT_175 = "ce9893b13e74a99f38d04ace2d278fac495012d0"  # interim #175
DESIGN_COMMIT_FINAL = "4c6a0f69f5637c6832659ab4dc8599b2c1a928b2"  # #178 squash
# PR #175's new not_certified margin: the gated {2016,2018} earnings cells ride
# the design 2.7 FORWARD earnings law, first-certified by gate_m6 (gate_1 does
# NOT cover it).
FORWARD_EARNINGS_MARGIN = "forward_earnings_law_not_gate1_certified"

# The not_certified margins the locked block MUST name (referee amendment 1 +
# the PR #175 forward-law margin); mortality drift is named FIRST.
REQUIRED_NOT_CERTIFIED = {
    "mortality_drift",
    "widowhood",
    "shock_window_2020_2022",
    "entrants_open_panel",
    "autocorrelation_lag5",
    "forward_projection_2100_extrapolation",
    FORWARD_EARNINGS_MARGIN,
}
N_NOT_CERTIFIED = 10  # the full margin count after the PR #175 amendment

# The ratifying-ceremony public record the history entry MUST carry.
CEREMONY_COMMENTS = ("4958425437", "4958956779", "4959310892")
RATIFYING_SQUASH = "1818784"


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _gate() -> dict[str, Any]:
    return yaml.safe_load(GATES.read_text(encoding="utf-8"))["gates"][
        "gate_m6"
    ]


def _v1() -> dict[str, Any]:
    return json.loads(V1_PATH.read_text(encoding="utf-8"))


def _v2() -> dict[str, Any]:
    return json.loads(V2_PATH.read_text(encoding="utf-8"))


def _v3() -> dict[str, Any]:
    return json.loads(V3_PATH.read_text(encoding="utf-8"))


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


def _history_entry(block: dict[str, Any]) -> dict[str, Any]:
    hist = block["history"]
    return next(e for e in hist if e["id"] == "2026-07-13-m6-gate-lock")


# --------------------------------------------------------------------------
# Binding checks (raise AssertionError on drift; reused by the mutation test)
# --------------------------------------------------------------------------
def check_lock_deltas(block: dict[str, Any]) -> None:
    """The lock-time deltas from the #175-amended draft: locked / status /
    history / the design_commit pin finalized to the #175 squash-merge."""
    assert block["locked"] is True
    assert block["status"] == "locked"
    assert "history" in block
    assert block["id"] == "m6_temporal_holdout_projection_drift"
    assert block["kind"] == "temporal_holdout"
    # the flip READS the frozen floor and rewrites nothing.
    assert block["floor_run"] == "runs/m6_holdout_floors_v3.json"
    # PR #175: design_pr re-pinned to 175, design_commit FINALIZED to the #175
    # squash-merge (not the stale 1d83a221 nor PR #175's own design commit).
    assert block["design_pr"] == DESIGN_PR
    assert block["design_commit"] == DESIGN_COMMIT_FINAL
    assert "design_commit_note" in block


def check_floor_run_sha(block: dict[str, Any]) -> None:
    committed = hashlib.sha256(V3_PATH.read_bytes()).hexdigest()
    assert block["floor_run_sha256"] == committed == V3_SHA


def check_gated_cells(block: dict[str, Any], v3: dict[str, Any]) -> None:
    assert set(_block_tolerances(block)) == set(v3["partition"]["gated"])
    assert len(_block_tolerances(block)) == 11


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
    assert combined["p_seed_pass"] == p_seed == 0.8934
    assert combined["p_gate_pass_4_of_5"] == p_gate == 0.9087
    assert (
        block["faithful_candidate_oc"]["family_a_flows"]["p_gate_pass_4_of_5"]
        == p_gate_flow
        == 0.9822
    )
    assert (
        block["faithful_candidate_oc"]["earnings_subfamily"][
            "p_gate_pass_4_of_5"
        ]
        == p_gate_earn
        == 0.9626
    )
    assert (
        block["oc_before_lock"]["p_gate_combined"]
        == v3["oc_before_lock"]["p_gate_combined"]
        == p_gate
    )


def check_not_certified(block: dict[str, Any]) -> None:
    nc = block["not_certified"]
    margins = [entry["margin"] for entry in nc]
    assert margins[0] == "mortality_drift"  # named FIRST
    assert REQUIRED_NOT_CERTIFIED <= set(margins)
    assert len(nc) == N_NOT_CERTIFIED  # the full PR #175-amended margin count
    for entry in nc:
        assert entry["detail"].strip(), entry["margin"]
    mort = next(e for e in nc if e["margin"] == "mortality_drift")
    assert "NOTHING" in mort["detail"]
    # the PR #175 forward-earnings-law margin: gate_1 does NOT certify it.
    fwd = next(e for e in nc if e["margin"] == FORWARD_EARNINGS_MARGIN)
    assert "gate_1" in fwd["detail"] and "forward" in fwd["detail"].lower()


def check_year_pins(block: dict[str, Any], v3: dict[str, Any]) -> None:
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


def check_shock_window_excluded(block: dict[str, Any]) -> None:
    """The 2020-2022 shock window is partitioned OUT of every gated set and is
    never a gated cell (2021 unobserved in the biennial panel)."""
    sw = block["shock_window"]
    assert sw["partitioned_out_of_gated_set"] is True
    assert sw["machine_reason"] == "exogenous_shock_outside_model_class"
    gated = set(_block_tolerances(block))
    shock_years = set(sw["earnings_shock_reference_years"]) | set(
        sw["flow_shock_event_years"]
    )
    # no gated cell references a shock year in its reference/event pins
    fh = block["fit_holdout"]
    assert not (set(fh["gated_earnings_reference_years"]) & shock_years)
    assert not (set(fh["gated_flow_event_years"]) & shock_years)
    # shock_window_2020_2022 is a report-only margin, never gated
    nc_margins = {e["margin"] for e in block["not_certified"]}
    assert "shock_window_2020_2022" in nc_margins
    assert not (gated & shock_years)


def check_lineage_shas(block: dict[str, Any]) -> None:
    v1 = block["v1_pause_evidence"]
    assert (
        v1["sha256"]
        == V1_SHA
        == hashlib.sha256(V1_PATH.read_bytes()).hexdigest()
    )
    assert v1["combined_p_gate"] == 0.8449
    assert v1["n_gated_flow_cells"] == 1
    assert v1["frozen"] is True
    v2 = block["v2_lineage"]
    assert (
        v2["sha256"]
        == V2_SHA
        == hashlib.sha256(V2_PATH.read_bytes()).hexdigest()
    )
    assert v2["combined_p_gate"] == 0.9067
    assert v2["n_gated_flow_cells"] == 4
    assert v2["frozen"] is True
    assert block["floor_run_sha256"] == V3_SHA


def check_ssa_nchs_anchor(block: dict[str, Any]) -> None:
    d = block["deliverables"]["ssa_nchs_life_table_mortality_anchor"]
    assert d["family"] == "B"
    assert d["status"] == "report_only"
    assert "REJECTED" in d["gating"]
    assert "circularity_disclosure" in d
    assert "NCHS" in d["what"] and "SSA" in d["what"]
    assert "M7" in d["required_before_lock_flip_for"]


def check_presence_f6_split(block: dict[str, Any], v3: dict[str, Any]) -> None:
    assert "held FIXED" in block["f6_weight"]["definition"]
    assert "13" in str(block["household_id_weight_carriage_rule"]["finding"])
    assert "presence" in block["presence_conditioning"]["rule"].lower()
    assert (
        block["presence_conditioning"]["classification"]
        == "conditioning_not_leakage"
    )
    assert set(block["split_units"]["household_disjoint_families"]) == set(
        v3["split_units"]["household_disjoint_families"]
    )
    assert set(block["split_units"]["person_disjoint_families"]) == set(
        v3["split_units"]["person_disjoint_families"]
    )
    # three marital household-split flow cells feed the split-width diagnostic
    diag = block["split_width_diagnostic"]
    assert (
        diag["mean_household_over_person"]
        == v3["split_width_diagnostic"]["mean_household_over_person"]
    )


def check_history(block: dict[str, Any]) -> None:
    """The ceremony record: comment ids, the v1/v2/v3 shas, the PR #172 /
    squash 1818784 ratification, flipped_live, and the narrative content."""
    entry = _history_entry(block)
    assert entry["flipped_live"] == "this pull request"
    assert str(entry["proposed"]) == "2026-07-13"
    rr = entry["referee_round"]
    blob = json.dumps(entry)
    # every ceremony comment id present
    for cid in CEREMONY_COMMENTS:
        assert cid in blob, cid
    assert "4958425437" in rr["review"]
    assert "4958956779" in rr["fixes"]
    assert "4959310892" in rr["verification"]
    # the v1/v2/v3 shas are recorded in the floor lineage narrative
    assert V3_SHA in rr["floor"]
    assert V1_SHA in rr["floor"]
    assert V2_SHA in rr["floor"]
    # ratified 2026-07-13 by merge of PR #172 (squash 1818784) under the
    # standing campaign directive of 2026-07-07, exercised after the ceremony
    assert RATIFYING_SQUASH in entry["ratified"]
    assert "PR #172" in entry["ratified"]
    assert "2026-07-07" in entry["ratified"]
    assert "no_self_rescue" in entry["ratified"]
    # the design_commit pin finalization is RECORDED in the history entry
    # (per the draft's design_commit_note): from PR #175's design commit to
    # the #175 squash-merge.
    fin = entry["design_commit_finalized"]
    assert fin["from"] == DESIGN_COMMIT_DRAFT
    assert fin["to"] == DESIGN_COMMIT_FINAL == block["design_commit"]
    # the RE-finalization: the interim #175 squash was superseded by the #178
    # squash once SS 2.7.6 completed the forward law (twice-verified).
    assert fin["refinalized_from"] == DESIGN_COMMIT_175
    assert "2.7.6" in fin["reason"] and "4960583620" in fin["reason"]
    # the ceremony narrative names the load-bearing facts
    content = entry["content"]
    assert "0.8449" in content  # v1 pause
    assert "0.9067" in content  # v2 cleared
    assert "0.8934" in content and "0.9087" in content  # v3 OC
    assert "mortality drift FIRST" in content
    assert "ZERO threshold movement" in content
    assert DESIGN_COMMIT_FINAL in content  # the finalization is narrated


# --------------------------------------------------------------------------
# LOCKED-HOT binding tests (unconditional)
# --------------------------------------------------------------------------
def test_gate_m6_locked_added_as_temporal_holdout_top_level_gate():
    block = _gate()
    check_lock_deltas(block)
    # a NEW top-level gate (the gate_w1 / gate_m4 pattern), sole new key vs the
    # locked siblings which stay untouched.
    gates = yaml.safe_load(GATES.read_text(encoding="utf-8"))["gates"]
    assert "gate_m6" in gates
    assert gates["gate_1"]["thresholds"]["locked"] is True
    assert gates["gate_2"]["thresholds"]["locked"] is True
    assert gates["gate_2"]["gate_2b"]["locked"] is True
    assert gates["gate_2"]["gate_2c"]["locked"] is True
    assert gates["gate_m4"]["locked"] is True
    assert gates["gate_w1"]["locked"] is True


def test_gate_m6_floor_run_sha_binds():
    check_floor_run_sha(_gate())


def test_gate_m6_gated_registry_11_cells():
    check_gated_cells(_gate(), _v3())


def test_gate_m6_tolerances_recompute_capped():
    check_tolerances(_gate(), _v3())


def test_gate_m6_partition_rollups_bind():
    check_partition_rollups(_gate(), _v3())


def test_gate_m6_operating_characteristic_recomputes():
    check_oc(_gate(), _v3())


def test_gate_m6_not_certified_mortality_first():
    check_not_certified(_gate())


def test_gate_m6_year_pins_bind_to_design_pins():
    check_year_pins(_gate(), _v3())


def test_gate_m6_shock_window_excluded_from_gated_set():
    check_shock_window_excluded(_gate())


def test_gate_m6_v1_v2_v3_lineage_shas():
    check_lineage_shas(_gate())


def test_gate_m6_ssa_nchs_anchor_deliverable():
    check_ssa_nchs_anchor(_gate())


def test_gate_m6_presence_f6_split_declared():
    check_presence_f6_split(_gate(), _v3())


def test_gate_m6_history_records_the_ceremony():
    check_history(_gate())


def test_gate_m6_remarriage_asymmetric_rung_and_zero_mortality_widowhood():
    """v3 gates remarriage.18-64 at the asymmetric age-2 rung; ZERO mortality
    and ZERO widowhood cells are gated (verified power grounds)."""
    block = _gate()
    assert (
        block["coarsening_ladder"]["adopted_rungs"]["remarriage"]
        == ASYMMETRIC_AGE2_RUNG
    )
    assert (
        block["coarsening_ladder"]["adopted_rungs"]["death"]
        == "none_cleared_any_rung"
    )
    assert (
        block["coarsening_ladder"]["adopted_rungs"]["widowhood"]
        == "none_cleared_any_rung"
    )
    gated = set(_block_tolerances(block))
    assert "remarriage.18-64" in gated
    # no mortality / widowhood cell is gated
    assert not any(
        k.startswith("death") or k.startswith("widowhood") for k in gated
    )


def test_gate_m6_zero_threshold_movement_vs_frozen_floor():
    """Every gated tolerance in the locked block equals the frozen floor's own
    recorded tolerance -- the flip moved no threshold."""
    block = _gate()
    v3 = _v3()
    for cell, tol in _block_tolerances(block).items():
        assert tol == v3["tolerances"][cell], cell
    assert block["oc_before_lock"]["ceremony_may_proceed"] is True
    assert block["oc_before_lock"]["n_gated_tolerances_at_cap"] == 0


# --------------------------------------------------------------------------
# Mutation guard: >= 6 mutations must each fail a binding
# --------------------------------------------------------------------------
def _deep(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def test_mutations_are_caught():
    v3 = _v3()

    # 1. tolerance DIGIT: bump one gated tolerance by a least-significant digit
    m1 = _deep(_gate())
    m1["views"]["marital_flows"]["tolerances"]["remarriage.18-64"] = 0.404
    with pytest.raises(AssertionError):
        check_tolerances(m1, v3)

    # 2. cell ADD/DROP: drop a gated earnings cell -> registry mismatch
    m2 = _deep(_gate())
    m2["views"]["earnings_log_ratio"]["tolerances"].pop("earn_p10.prime")
    with pytest.raises(AssertionError):
        check_gated_cells(m2, v3)
    # ... and ADD a spurious cell
    m2b = _deep(_gate())
    m2b["views"]["earnings_log_ratio"]["tolerances"]["earn_p10.ADDED"] = 0.2
    with pytest.raises(AssertionError):
        check_gated_cells(m2b, v3)

    # 3. OC DRIFT: perturb the committed combined p_gate
    m3 = _deep(_gate())
    m3["faithful_candidate_oc"]["combined"]["p_gate_pass_4_of_5"] = 0.95
    with pytest.raises(AssertionError):
        check_oc(m3, v3)

    # 4. not_certified REORDER: mortality no longer named FIRST
    m4 = _deep(_gate())
    nc = m4["not_certified"]
    nc[0], nc[1] = nc[1], nc[0]
    with pytest.raises(AssertionError):
        check_not_certified(m4)

    # 5. YEAR-PIN drift: shift a gated earnings reference year into the shock
    # window (the referee's amendment-4 missed mutation)
    m5 = _deep(_gate())
    m5["fit_holdout"]["gated_earnings_reference_years"] = [2016, 2020]
    with pytest.raises(AssertionError):
        check_year_pins(m5, v3)

    # 6. HISTORY SHA drift: a stale v3 floor sha in the history ceremony record
    m6 = _deep(_gate())
    m6["history"][0]["referee_round"]["floor"] = m6["history"][0][
        "referee_round"
    ]["floor"].replace(V3_SHA, "0" * 64)
    with pytest.raises(AssertionError):
        check_history(m6)

    # 6b. floor_run_sha256 drift
    m6b = _deep(_gate())
    m6b["floor_run_sha256"] = "f" * 64
    with pytest.raises(AssertionError):
        check_floor_run_sha(m6b)

    # 7. LOCK-DELTA drift: an unlocked block must fail the lock guard
    m7 = _deep(_gate())
    m7["locked"] = False
    with pytest.raises(AssertionError):
        check_lock_deltas(m7)

    # 8. partition family tally drift (v3 marital is 3, not 2)
    m8 = _deep(_gate())
    m8["gated_roll_up"]["by_family"]["marital"] = 2
    with pytest.raises(AssertionError):
        check_partition_rollups(m8, v3)

    # 9. DESIGN-COMMIT re-finalization drift: the interim #175 squash (not the
    # #178 squash that completes the forward law) must fail the lock-delta bind.
    m9 = _deep(_gate())
    m9["design_commit"] = DESIGN_COMMIT_175
    with pytest.raises(AssertionError):
        check_lock_deltas(m9)

    # 10. not_certified COUNT drift: dropping the PR #175 forward-earnings-law
    # margin fails the count + required-set binding.
    m10 = _deep(_gate())
    m10["not_certified"] = [
        e
        for e in m10["not_certified"]
        if e["margin"] != FORWARD_EARNINGS_MARGIN
    ]
    with pytest.raises(AssertionError):
        check_not_certified(m10)
