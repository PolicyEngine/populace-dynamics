"""Proposal-consistency bindings for gate_w1 amendment 1 (family-B DI bands).

The amendment-1 PROPOSAL is a DRAFT: it argues, from the W1 forensics 1 Q4
evidence base, that the 8 family-B DI age-composition prevalence bands are
unclearable by any contract-consistent candidate and must be demoted (option a)
rather than re-anchored now (option b). This module proves the proposal document
does not drift from the committed evidence: it parses the machine-readable
``amendment-consistency-ledger`` block embedded in
``docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`` and cross-checks
EVERY load-bearing figure against the frozen artifacts --
``runs/gate_w1_forensics1_v1.json`` (Q4), ``runs/gate_w1_floors_v1.json`` (the
family-A OC machinery + partition), ``runs/gate_w1_candidate1_v1.json`` (the
committed FAIL the amendment must not rescue), and ``gates.yaml`` (still the
LOCKED 10-cell surface -- the proposal moves no threshold).

ALWAYS-RUNNABLE: reads only committed ``runs/*.json`` + the doc + gates.yaml, no
data load, no h5, no PSID/PE-US checkout. Reproduces the draw-noise-free
half-normal OC exactly as tests/test_gate_w1_derivations.py does, so the
proposal's OC-invariance claim is bound to the same recomputation the lock used.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "amendments" / "gate_w1_amendment_1_family_b_di_bands.md"
FORENSICS = ROOT / "runs" / "gate_w1_forensics1_v1.json"
FLOORS = ROOT / "runs" / "gate_w1_floors_v1.json"
CANDIDATE1 = ROOT / "runs" / "gate_w1_candidate1_v1.json"
GATES = ROOT / "gates.yaml"

FLOOR_KEY = "noise_floor_seeds_0_99"
DI_BANDS = (
    "under30",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-fra",
)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _ledger() -> dict:
    """Parse the fenced ``amendment-consistency-ledger`` JSON block."""
    lines = _doc_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if (
            line.startswith("```json")
            and "amendment-consistency-ledger" in line
        ):
            start = i + 1
            break
    assert start is not None, "ledger fence not found in the proposal doc"
    end = next(
        i for i in range(start, len(lines)) if lines[i].strip() == "```"
    )
    return json.loads("\n".join(lines[start:end]))


def _forensics() -> dict:
    return json.loads(FORENSICS.read_text())


def _floors() -> dict:
    return json.loads(FLOORS.read_text())


def _candidate1() -> dict:
    return json.loads(CANDIDATE1.read_text())


def _gate_w1() -> dict:
    return yaml.safe_load(GATES.read_text())["gates"]["gate_w1"]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# The ledger is well formed and names the recommendation + machine reason
# --------------------------------------------------------------------------
def test_ledger_parses_and_headline_matches_prose():
    led = _ledger()
    assert led["amendment_id"] == "2026-07-12-w1-family-b-di-bands"
    assert led["recommendation"] == "option_a_demote"
    assert led["machine_reason"] == "concept_bridge_undefined_di_stock"
    # the machine reason + the demote recommendation appear in the prose, so a
    # ledger/prose divergence is caught.
    text = _doc_text()
    assert "concept_bridge_undefined_di_stock" in text
    assert "RECOMMENDED" in text
    assert "option (a)" in text
    assert led["gates_yaml_untouched_by_this_proposal"] is True


# --------------------------------------------------------------------------
# Q4 evidence base: every forensics figure the doc cites is bound
# --------------------------------------------------------------------------
def test_forensics_pointers_and_frame_side_flags():
    led = _ledger()["forensics"]
    art = _forensics()
    assert led["artifact"] == "runs/gate_w1_forensics1_v1.json"
    assert (
        led["registration"]
        == art["registration"]["comment_id"]
        == ("4951218279")
    )
    assert led["grading"] == "4951430002"  # the Q4-finding grading (task)
    assert art["reported_not_gated"] is True
    assert led["reported_not_gated"] is True
    assert art["protocol"]["train_frame_side_only"] is True
    assert led["train_frame_side_only"] is True
    assert art["protocol"]["publishes_regardless"] is True
    # the finding was computed against the CURRENTLY locked contract blob.
    assert art["revision_pins"]["gates_yaml_blob"] == (
        "cd6411d973c64209a38cc12c7dc33e02d4254d65"
    )


def test_q4_concept_delta_and_bridge_determination_bound():
    led = _ledger()["forensics"]
    q4 = _forensics()["q4_di_level_bridge"]
    assert (
        round(q4["concept_delta_dominant_share"], 3)
        == (led["concept_delta_dominant_share_round3"])
        == 0.595
    )
    det = q4["gate_design_determination"]
    assert det["is_gate_design_finding"] is True
    assert det["insured_denominator_available"] is False
    assert led["insured_denominator_available"] is False
    assert len(q4["m4_concept_deltas"]) == led["n_concept_deltas"] == 7
    assert q4["worst_band"]["band"] == led["worst_band"] == "60-fra"


def test_q4_all_eight_di_bands_fail_and_miss_ratios_bound():
    """Every DI band fails; the doc's 2.9x-21.9x miss-ratio envelope recomputes
    from anchor/deployed/tolerance in the forensics per-band block."""
    led = _ledger()["forensics"]
    pb = _forensics()["q4_di_level_bridge"]["per_band"]
    assert set(pb) == set(DI_BANDS)
    assert led["n_di_bands"] == 8
    ratios = []
    for band in DI_BANDS:
        row = pb[band]
        assert row["passes"] is False, band
        gap = abs(row["anchor_ssa_stock_pp"] - row["deployed_m4_stock_pp"])
        ratios.append(gap / row["tolerance_pp"])
    assert led["all_di_bands_fail"] is True
    assert round(min(ratios), 1) == led["miss_ratio_min_round1"] == 2.9
    assert round(max(ratios), 1) == led["miss_ratio_max_round1"] == 21.9


def test_q4_worst_band_duration_vs_shape_decomposition_bound():
    """60-FRA: duration (stock-vs-flow) +21.3pp dominates the M4 hazard shape
    +2.6pp -- the doc's headline decomposition."""
    led = _ledger()["forensics"]
    worst = _forensics()["q4_di_level_bridge"]["per_band"]["60-fra"]
    assert (
        round(worst["duration_concept_flow_to_stock"], 1)
        == (led["worst_band_duration_component_pp_round1"])
        == 21.3
    )
    assert (
        round(worst["m4_shape_component_flow_minus_deployed"], 1)
        == (led["worst_band_m4_shape_component_pp_round1"])
        == 2.6
    )


def test_supplement_4c2_is_still_wanted_in_the_archive():
    """The concept bridge's third leg (the insured denominator) is not archived
    -- the doc's evidence-base gap (section 10) is real."""
    prov = ROOT / "data" / "external" / "di_asr_2023" / "provenance.md"
    text = prov.read_text()
    assert "Still wanted" in text
    assert "4.C2" in text
    assert "insured denominator" in text


# --------------------------------------------------------------------------
# OC consequence: family-A OC recomputes to 0.922 / 0.9481 and is INVARIANT
# --------------------------------------------------------------------------
def test_family_a_oc_recomputes_and_is_invariant_to_demotion():
    """The doc's OC claim: p_seed 0.922 / p_gate 0.9481 recomputes from the 53
    locked family-A tolerances + the frozen floor sigmas (draw-noise-free
    half-normal), and NONE of the 8 DI cells is in the family-A OC machinery, so
    demoting them cannot move it."""
    led = _ledger()["family_a_oc"]
    floors = _floors()
    per = floors["faithful_candidate_oc"]["per_cell"]
    gate_eligible = floors["gate_partition"]["gate_eligible"]

    tolerances: dict = {}
    for view in _gate_w1()["thresholds"]["family_a"]["views"].values():
        tolerances.update(view["tolerances"])
    assert len(tolerances) == 53 == led["n_gated"]

    p_seed = 1.0
    for cell in gate_eligible:
        p_seed *= (
            2.0 * _normal_cdf(tolerances[cell] / per[cell]["realized_sigma"])
            - 1.0
        )
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    assert round(p_seed, 4) == led["p_seed"] == 0.922
    assert round(p_gate, 4) == led["p_gate"] == 0.9481
    # the contract stores the rounded characteristic (the lock's own value).
    assert round(p_seed, 4) == floors["faithful_candidate_oc"]["p_seed_pass"]
    assert round(p_gate, 4) == (
        floors["faithful_candidate_oc"]["p_gate_pass_4_of_5"]
    )

    # INVARIANCE: the 8 DI cells are family-B; they never appear in the
    # family-A OC per-cell machinery, so demotion leaves the OC byte-identical.
    di_cells = {f"di_prevalence.{b}" for b in DI_BANDS}
    assert di_cells.isdisjoint(per)
    assert di_cells.isdisjoint(gate_eligible)
    assert led["invariant_under_amendment"] is True


# --------------------------------------------------------------------------
# Partition consequence: family B 10 -> 2 gated; overall 65 -> 57 gated
# --------------------------------------------------------------------------
def test_family_b_partition_now_and_after_option_a():
    led = _ledger()["family_b_partition"]
    fb = _gate_w1()["thresholds"]["family_b"]
    gated = list(fb["gated_cells"])
    report_only = list(fb["report_only"])
    conv = [c for c in gated if "disability_conversion" in c]
    di = [c for c in gated if c.startswith("di_prevalence.")]

    # CURRENT locked surface (the proposal has NOT flipped it).
    assert len(gated) == led["gated_now"] == 10
    assert len(conv) == led["conversion"] == 2
    assert len(di) == led["di_bands"] == 8
    assert set(c.split(".", 1)[1] for c in di) == set(DI_BANDS)
    assert len(report_only) == led["report_only_now"] == 15

    # AFTER option (a): the 8 DI bands move to report-only, conversion stays.
    assert led["gated_after_option_a"] == len(conv) == 2
    assert (
        led["report_only_after_option_a"] == len(report_only) + len(di) == 23
    )


def test_overall_partition_65_to_57_gated():
    led = _ledger()["overall_partition"]
    # 53 family-A + 10 family-B + 2 family-C == 65 gated; option (a) -> 57.
    assert led["gated_now"] == 53 + 10 + 2 == 65
    assert led["gated_after_option_a"] == 53 + 2 + 2 == 57
    # 52 family-A + 15 family-B report-only == 67; option (a) -> 75.
    assert led["report_only_now"] == 52 + 15 == 67
    assert led["report_only_after_option_a"] == 52 + 23 == 75


# --------------------------------------------------------------------------
# No-self-rescue: candidate 1 stands FAIL under BOTH surfaces
# --------------------------------------------------------------------------
def test_no_self_rescue_candidate1_fails_all_three_families():
    led = _ledger()["no_self_rescue"]
    c1 = _candidate1()
    v = c1["verdict"]
    assert c1["run"] == led["candidate1_run"] == "gate_w1_candidate1_v1"
    for family in (
        "gate_pass",
        "family_a_pass",
        "family_b_pass",
        "family_c_pass",
    ):
        assert v[family] is False, family
        assert led[family] is False, family
    assert v["n_seed_pass_family_a"] == 0


def test_no_self_rescue_retained_conversion_cells_still_fail_for_candidate1():
    """Demoting the 8 DI bands cannot flip candidate 1: it ALSO misses the 2
    RETAINED conversion cells, so the amended surface fails it too."""
    led = _ledger()["no_self_rescue"]
    per_cell = _candidate1()["family_b"]["per_cell"]
    conv = [c for c in per_cell if "disability_conversion" in c]
    assert len(conv) == 2
    for cell in conv:
        row = per_cell[cell]
        assert row["pass"] is False, cell
        assert row["abs_dev_pp"] > row["tolerance_pp"], cell
    assert led["retained_conversion_cells_fail_for_candidate1"] is True
    assert led["candidate1_pr"] == 162


# --------------------------------------------------------------------------
# The proposal is a DRAFT: gates.yaml is untouched (no flip)
# --------------------------------------------------------------------------
def test_proposal_does_not_flip_gates_yaml():
    """gate_w1 in the working tree is byte-identical to origin/master: the
    proposal adds only the doc + this test, it moves no threshold. (The flip is a
    separate post-ratification PR.)"""
    try:
        master_text = subprocess.run(
            ["git", "show", "origin/master:gates.yaml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("origin/master gates.yaml unreachable")
    master = yaml.safe_load(master_text)["gates"]["gate_w1"]
    current = _gate_w1()
    assert current == master, "the proposal PR must not edit gate_w1"
    # and family B still gates the full 10 (DI bands NOT yet demoted).
    assert len(current["thresholds"]["family_b"]["gated_cells"]) == 10
