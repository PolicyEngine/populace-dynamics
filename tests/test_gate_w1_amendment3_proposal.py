"""Proposal-consistency bindings for gate_w1 amendment 3 (the hh_size residual +
the pair-scoped C2).

The amendment-3 PROPOSAL is a DRAFT: it argues, from W1 forensics 3 (Q10 the
cap_150k adjacency decomposition + the entailment, Q11 the hh_size residual
quantification + the lever-exhaustion probe), W1 forensics 2 Q7, the committed
candidate cubes (c1/c2/c3), and the ratified amendment-2 series forecast that
PRE-NAMED both surfaces before c3 ran, that (a) the four family-A household-size
cells hh_size_share.{1,3,4,5plus} must be DEMOTED to report-only (uniform machine
reason no_permitted_entry_state_lever_reaches_cell) and (b) the family-C gated
binary C2 must be RE-SCOPED from the full 4-element ordering to the pair-scoped
elim<->+2pp adjacent swap.

This module proves the proposal document does not drift from the committed
evidence: it parses the machine-readable ``amendment-consistency-ledger`` block in
``docs/amendments/gate_w1_amendment_3_hh_size_and_c2_pair.md`` and cross-checks
EVERY load-bearing figure against the frozen artifacts -- the forensics-3
shares/mirror/ceiling numbers, the per-cell pass records from ALL THREE candidate
cubes, the OC before/after (47->43 on the frozen floor sigmas), the partition
roll-ups (48->44 gated / 84->88 report-only), and the operative section-7 flip text
(cube shape [20,47,5]->[20,43,5]).

Because the amended surface would be PASSABLE BY THE ALREADY-COMMITTED c3 MODEL,
the whittling scrutiny is at its sharpest, so a dedicated mutation battery (>=6)
guards the two ways the answer could silently rot: a BLANKET-IMPOSSIBILITY sentence
re-appearing (false -- c3 clears size-2 5/5 and size-4 seed 1) and the WHITTLING
section (the c3-passable statement) being deleted.

ALWAYS-RUNNABLE (artifact tier): reads only committed ``runs/*.json`` + the doc +
gates.yaml + the amendment-2 doc, no data load, no h5, no PSID/PE-US checkout.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = (
    ROOT / "docs" / "amendments" / "gate_w1_amendment_3_hh_size_and_c2_pair.md"
)
A2_DOC = (
    ROOT
    / "docs"
    / "amendments"
    / "gate_w1_amendment_2_family_a_concept_cells.md"
)
FORENSICS3 = ROOT / "runs" / "gate_w1_forensics3_v1.json"
FORENSICS2 = ROOT / "runs" / "gate_w1_forensics2_v1.json"
FLOORS = ROOT / "runs" / "gate_w1_floors_v1.json"
CANDIDATE1 = ROOT / "runs" / "gate_w1_candidate1_v1.json"
CANDIDATE2 = ROOT / "runs" / "gate_w1_candidate2_v1.json"
CANDIDATE3 = ROOT / "runs" / "gate_w1_candidate3_v1.json"
GATES = ROOT / "gates.yaml"

# The four family-A cells the amendment demotes; hh_size_share.2 STAYS gated.
HH_DEMOTED = (
    "hh_size_share.1",
    "hh_size_share.3",
    "hh_size_share.4",
    "hh_size_share.5plus",
)
HH_RETAINED_GATED = "hh_size_share.2"
REASON_HH = "no_permitted_entry_state_lever_reaches_cell"
REASON_CAP = "anchor_ordering_internally_inconsistent_under_certified_swap"

# The 6 cells amendment 2 demoted (retained_tolerances in the LIVE gates.yaml).
A2_DEMOTED = {
    "earnings_participation.18-24|female",
    "earnings_participation.18-24|male",
    "marital_share.married.65+|female",
    "marital_share.married.65+|male",
    "coresident_spouse.65+|female",
    "coresident_spouse.65+|male",
}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _ledger_of(text: str) -> dict:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if (
            line.startswith("```json")
            and "amendment-consistency-ledger" in line
        ):
            start = i + 1
            break
    assert start is not None, "ledger fence not found"
    end = next(
        i for i in range(start, len(lines)) if lines[i].strip() == "```"
    )
    return json.loads("\n".join(lines[start:end]))


def _ledger() -> dict:
    return _ledger_of(_doc_text())


def _prose_without_ledger() -> str:
    lines = _doc_text().splitlines()
    out, in_ledger = [], False
    for line in lines:
        if (
            line.startswith("```json")
            and "amendment-consistency-ledger" in line
        ):
            in_ledger = True
            continue
        if in_ledger:
            if line.strip() == "```":
                in_ledger = False
            continue
        out.append(line)
    return "\n".join(out)


def _section(num) -> str:
    lines = _doc_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"## {num}."):
            start = i
            break
    assert start is not None, f"section {num} not found"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def _forensics3() -> dict:
    return json.loads(FORENSICS3.read_text())


def _forensics2() -> dict:
    return json.loads(FORENSICS2.read_text())


def _floors() -> dict:
    return json.loads(FLOORS.read_text())


def _candidate(n: int) -> dict:
    return json.loads(
        {1: CANDIDATE1, 2: CANDIDATE2, 3: CANDIDATE3}[n].read_text()
    )


def _gate_w1() -> dict:
    return yaml.safe_load(GATES.read_text())["gates"]["gate_w1"]


def _family_a_tolerances() -> dict:
    tol: dict = {}
    for view in _gate_w1()["thresholds"]["family_a"]["views"].values():
        tol.update(view["tolerances"])
    return tol


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _oc(cells, tol: dict, per: dict) -> tuple:
    p_seed = 1.0
    for cell in cells:
        p_seed *= (
            2.0 * _normal_cdf(tol[cell] / per[cell]["realized_sigma"]) - 1.0
        )
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return p_seed, p_gate


def _seed_pass_counts(cand: dict, cell: str) -> int:
    return sum(
        1 for s in cand["family_a"]["per_seed"] if s["per_cell"][cell]["pass"]
    )


# --------------------------------------------------------------------------
# Ledger: recommendation + the two changes
# --------------------------------------------------------------------------
def test_ledger_parses_and_headline_matches_prose():
    led = _ledger()
    assert led["amendment_id"] == "2026-07-13-w1-hh-size-and-c2-pair"
    assert led["recommendation"] == "demote_hh_size_quad_and_pair_scope_c2"
    assert led["n_cells_demoted"] == 4
    assert led["family_c_rescope"] == "pair_scoped_c2_elim_plus2pp_swap"
    reasons = led["cell_reasons"]
    assert set(reasons) == set(HH_DEMOTED) and len(reasons) == 4
    assert all(r == REASON_HH for r in reasons.values())
    assert led["cap_150k_leg_reason"] == REASON_CAP
    assert led["gates_yaml_untouched_by_this_proposal"] is True
    prose = _prose_without_ledger()
    assert "RECOMMENDED" in prose
    assert "pair-scoped" in prose


def test_machine_reasons_named_in_prose_outside_ledger():
    """De-vacuous: both machine reasons appear in the prose with the ledger fence
    removed (the amendment-1/2 fix-B standard)."""
    prose = _prose_without_ledger()
    assert '"cell_reasons"' not in prose and '"machine_reason"' not in prose
    assert REASON_HH in prose
    assert REASON_CAP in prose


# --------------------------------------------------------------------------
# Per-cell pass records from ALL THREE candidate cubes
# --------------------------------------------------------------------------
def test_per_cell_pass_records_bound_to_all_three_cubes():
    """The ledger's per-seed pass records recompute from c1/c2/c3. The demoted quad
    fails 5/5 for c1/c2; size-4 CLEARS c3 seed 1 (1/5); size-2 CLEARS c3 5/5 and is
    NOT demoted. No blanket impossibility is defensible."""
    rec = _ledger()["per_cell_pass_record"]
    for cell in (*HH_DEMOTED, HH_RETAINED_GATED):
        for n in (1, 2, 3):
            assert (
                _seed_pass_counts(_candidate(n), cell) == rec[cell][f"c{n}"]
            ), (cell, n)
    # sizes 1/3/5plus fail every seed for all three candidates.
    for cell in ("hh_size_share.1", "hh_size_share.3", "hh_size_share.5plus"):
        assert rec[cell] == {"c1": 0, "c2": 0, "c3": 0}
    # size-4 is NOT impossible: it clears c3 seed 1 (1/5).
    assert rec["hh_size_share.4"] == {"c1": 0, "c2": 0, "c3": 1}
    assert _seed_pass_counts(_candidate(3), "hh_size_share.4") == 1
    # size-2 clears c3 5/5 -> stays gated.
    assert rec[HH_RETAINED_GATED]["c3"] == 5
    assert _seed_pass_counts(_candidate(3), HH_RETAINED_GATED) == 5
    # forensics-3's own exact-quad n_seed_pass agrees for c3.
    quad = _forensics3()["q11_hhsize_residual"]["candidate_independence"][
        "exact_quad_failing_cells"
    ]
    assert quad["4"]["n_seed_pass"] == 1
    for k in ("1", "3", "5plus"):
        assert quad[k]["n_seed_pass"] == 0


def test_prose_states_size2_clears_and_size4_single_pass_plainly():
    """Candor (the amendment-2 blocker lesson): the doc states, in prose, that
    size-2 clears (5/5, stays gated) and size-4 clears one c3 seed -- so no blanket
    'the household-size family is unreachable' claim can substitute."""
    prose = _prose_without_ledger()
    assert "hh_size_share.2" in prose
    assert "5 of 5" in prose or "5/5" in prose
    assert "seed 1" in prose
    assert "1 of 5" in prose or "1/5" in prose


# --------------------------------------------------------------------------
# Q11 hh_size: channel attribution + mirror + roster ceiling probe
# --------------------------------------------------------------------------
def test_q11_channel_attribution_bound_to_forensics3():
    led = _ledger()["hh_size_channel_attribution"]
    attr = _forensics3()["q11_hhsize_residual"]["per_cell_attribution"]
    # size-1 mixed: fertility 0.54, coresidence main-effect 0.40, complement 0.46.
    a1 = attr["1"]
    assert (
        round(a1["fertility_share_of_moved"], 2)
        == led["1"]["fertility_share_round2"]
        == 0.54
    )
    assert (
        round(a1["coresidence_share_of_moved"], 2)
        == led["1"]["coresidence_main_effect_share_round2"]
        == 0.4
    )
    assert (
        round(1.0 - a1["fertility_share_of_moved"], 2)
        == led["1"]["coresidence_complement_round2"]
        == 0.46
    )
    assert a1["dominant_channel"] == "fertility" and led["1"]["mixed"] is True
    # size-3 coresidence 0.75 (dominant).
    assert (
        round(attr["3"]["coresidence_share_of_moved"], 2)
        == led["3"]["coresidence_share_round2"]
        == 0.75
    )
    assert attr["3"]["dominant_channel"] == "coresidence"
    # size-4 fertility 0.67; size-5plus fertility 0.78.
    assert (
        round(attr["4"]["fertility_share_of_moved"], 2)
        == led["4"]["fertility_share_round2"]
        == 0.67
    )
    assert (
        round(attr["5plus"]["fertility_share_of_moved"], 2)
        == led["5plus"]["fertility_share_round2"]
        == 0.78
    )
    assert attr["4"]["dominant_channel"] == "fertility"
    assert attr["5plus"]["dominant_channel"] == "fertility"
    # prose carries the four figures.
    prose = _prose_without_ledger()
    for tok in ("0.54", "0.75", "0.67", "0.78"):
        assert tok in prose, tok


def test_q11_mirror_bound_to_forensics3():
    led = _ledger()["hh_size_mirror"]
    ms = _forensics3()["q11_hhsize_residual"]["mirror_structure"]
    assert (
        round(ms["coresidence_mirror_ratio"], 2)
        == led["coresidence_mirror_ratio_round2"]
        == 0.98
    )
    assert (
        round(ms["coresidence_moves_size1"], 3)
        == led["coresidence_moves_size1_round3"]
        == -0.047
    )
    assert (
        round(ms["coresidence_moves_size3plus"], 3)
        == led["coresidence_moves_size3plus_round3"]
        == 0.046
    )
    assert ms["coresidence_is_mirror_structured"] is True
    assert led["coresidence_owns_size1_size3_mirror"] is True
    assert "mirror" in _prose_without_ledger().lower()


def test_q11_roster_ceiling_probe_bound_levers_exhausted():
    """The decisive probe: the roster seed pushed to and beyond the permitted
    maximum reaches terminal coresidence only 0.111 (hazard-capped) and does NOT
    close size-1 (0.179 vs frame 0.083). Levers exhausted, three candidates deep.
    """
    led = _ledger()["roster_ceiling_probe"]
    ci = _forensics3()["q11_hhsize_residual"]["candidate_independence"]
    pb = ci["untried_full_age_roster_probe"]
    assert (
        round(pb["terminal_coresident_parent_rate_at_ceiling"], 3)
        == led["terminal_coresidence_at_ceiling_round3"]
        == 0.111
    )
    assert (
        round(pb["size1_at_ceiling_seed"], 3)
        == led["size1_at_ceiling_round3"]
        == 0.179
    )
    assert (
        round(pb["frame_rate_a_size1"], 3)
        == led["frame_rate_a_size1_round3"]
        == 0.083
    )
    assert pb["closes_size1"] is False and led["closes_size1"] is False
    assert ci["verdict"] == "levers_exhausted"
    assert led["levers_exhausted"] is True and led["three_candidates_deep"]
    prose = _prose_without_ledger()
    assert "0.111" in prose and "certified parental-exit hazard" in prose
    assert "levers_exhausted" in prose or "levers are exhausted" in prose
    assert _ledger()["forensics3"]["q11_levers_exhausted"] is True
    assert _ledger()["forensics3"]["q11_attribution"] == "mixed_but_structured"


# --------------------------------------------------------------------------
# Q10 cap_150k: the entailment + the ledger arithmetic + restoration empty
# --------------------------------------------------------------------------
def test_q10_cap150k_arithmetic_bound_to_forensics3():
    led = _ledger()["forensics3"]
    q10 = _forensics3()["q10_cap150k_adjacency"]
    gd = q10["gap_decomposition"]
    ent = q10["entailment"]
    assert (
        round(gd["share_a_frame_above_cap"], 4)
        == led["q10_share_a_frame_above_cap_round4"]
        == 0.9925
    )
    assert (
        round(gd["share_bc_config_vintage"], 4)
        == led["q10_share_bc_config_vintage_round4"]
        == 0.0075
    )
    assert (
        round(ent["A_over_B_deployed_representative"], 3)
        == led["q10_deployed_A_over_B_round3"]
        == 1.694
    )
    assert (
        round(ent["c2_breakeven_A_over_B"], 4)
        == led["q10_c2_breakeven_A_over_B_round4"]
        == 0.1613
    )
    win = ent["smith_adjacency_window_A_over_B_at_f_psid"]
    assert (
        [round(win[0], 4), round(win[1], 3)]
        == led["q10_smith_adjacency_window"]
        == [0.1613, 0.184]
    )
    assert (
        round(ent["smith_adjacency_window_width_at_f_psid"], 4)
        == led["q10_smith_adjacency_window_width_round4"]
        == 0.0228
    )
    assert ent["deployed_cap_rank"] == led["q10_deployed_cap_rank"] == 2
    assert ent["deployed_breaks_adjacency"] is True
    assert ent["entailment_holds"] is True and led["q10_entailment_holds"]
    assert q10["construction_attempt"]["restoration_exists"] is False
    assert led["q10_permitted_lever_restoration_exists"] is False
    assert (
        round(
            q10["instrumentation_fidelity"]["committed_exhaustion_deltas"][
                "cap_150k"
            ],
            1,
        )
        == led["q10_cap_150k_deployed_years_round1"]
        == 16.7
    )
    assert gd["smith_cap_150k_years"] == led["q10_smith_cap_150k_years"] == 1
    # the arithmetic is stated in prose, and the impossibility is scoped to the
    # PINNED certified frame (finding 1: not a universal claim).
    s2 = " ".join(_section(2).split())
    assert "0.1613" in s2 and "0.184" in s2 and "1.694" in s2
    assert "0.9925" in s2
    assert "internally inconsistent" in s2
    assert "pinned" in s2.lower()
    assert (
        "on the pinned transport target" in s2 or "on the pinned frame" in s2
    )


def test_q10_reported_not_gated_and_pointers():
    led = _ledger()["forensics3"]
    art = _forensics3()
    assert led["artifact"] == "runs/gate_w1_forensics3_v1.json"
    assert (
        led["registration"]
        == art["registration"]["comment_id"]
        == "4959668253"
    )
    assert led["grading"] == "4960691167"
    assert art["reported_not_gated"] is True and led["reported_not_gated"]
    assert art["protocol"]["train_frame_side_only"] is True
    assert led["train_frame_side_only"] is True
    assert art["protocol"]["publishes_regardless"] is True


# --------------------------------------------------------------------------
# Finding 1 (fixes round): the C2 impossibility is PINNED-FRAME-scoped, not a
# universal claim -- forensics-3's own f-conditional table refutes the universal
# --------------------------------------------------------------------------
def test_q10_impossibility_scoped_to_pinned_frame_not_universal():
    """The referee's AMEND finding: the width-0.023 window is f-conditional
    (0.1613, 0.0806/f); at f~=0.22 it widens and forensics-3's own
    order_at_f_deployed_0p22 exhibits a representative frame that realises the swap
    AND holds the full 4-element Smith order (tau 1.0). So the impossibility is a
    property of the PINNED certified frame (which measures cap at rank 2
    bit-identically), NOT a universal claim about representative frames. Bound to the
    artifact so the universal phrasing cannot silently return."""
    imp = _ledger()["cap_150k_impossibility"]
    q10 = _forensics3()["q10_cap150k_adjacency"]
    ent = q10["entailment"]
    tfs = ent["true_frame_sensitivity"]
    rc = q10["revenue_components"]
    # the ledger records the pinned scope, not a universal.
    assert _ledger()["cap_150k_leg_reason_scope"] == "pinned_certified_frame"
    assert imp["scope"] == "pinned_certified_frame_only"
    assert imp["is_universal_claim"] is False
    # dispositive ground 1: the PINNED frame measures cap at rank 2 bit-identically.
    assert (
        round(rc["deployed_representative_frame"]["Aband_over_B"], 3)
        == imp["deployed_A_band_over_B_round3"]
        == 0.373
    )
    assert ent["deployed_cap_rank"] == imp["deployed_cap_rank"] == 2
    # dispositive ground 2: no published anchor for the full order on a rep frame.
    key = (
        "any_published_anchor_supports_full_4_element_on_representative_frame"
    )
    assert q10["pair_scoped_respec"][key] is False
    # the window is f-conditional: const = win_upper(f_psid) * f_psid = 0.0806.
    f_psid = rc["psid_anchor_frame"]["Aband_over_A"]
    f_dep = rc["deployed_representative_frame"]["Aband_over_A"]
    win = ent["smith_adjacency_window_A_over_B_at_f_psid"]
    const = win[1] * f_psid
    assert round(const, 4) == 0.0806  # the 0.0806/f numerator (0.01/r)
    # V2 (fixes round 2): the pinned frame's A/B is the binding constraint, not f.
    # The pinned frame lands in the order-survival window only for f < const/(A/B) =
    # 0.0476 (equivalently 0.01/d_bal_elim), far below the measured deployed f 0.22 --
    # this is the true scoped content the §2a restoration clause now states.
    ab_dep = rc["deployed_representative_frame"]["A_over_B"]
    f_needed = const / ab_dep
    assert (
        round(f_needed, 4)
        == imp["pinned_A_over_B_needs_f_below_round4"]
        == 0.0476
    )
    assert round(f_psid, 3) == imp["f_psid_round3"] == 0.438
    assert round(f_dep, 2) == imp["f_deployed_round2"] == 0.22
    assert round(win[1], 3) == imp["window_upper_at_f_psid_round3"] == 0.184
    win_upper_dep = const / f_dep
    assert (
        round(win_upper_dep, 3)
        == imp["window_upper_at_f_deployed_round3"]
        == 0.367
    )
    # THE REFUTATION: the SSA true frame (A/B ~= 0.22) is INSIDE the widened window
    # (breakeven, win_upper_deployed) AND its order is the full Smith order tau=1.0.
    ssa = tfs["ssa_true_frame_A_over_B"]
    assert round(ssa, 2) == imp["ssa_true_frame_A_over_B_round2"] == 0.22
    assert ent["c2_breakeven_A_over_B"] < ssa < win_upper_dep  # inside window
    required = _candidate(3)["family_c"]["fingerprints"]["c2"][
        "required_representative_order"
    ]
    assert (
        tfs["order_at_f_deployed_0p22"]
        == required
        == ["elimination", "payroll_plus_2pp", "payroll_plus_1pp", "cap_150k"]
    )
    assert imp["order_at_f_deployed_holds_full_smith"] is True
    # that frame realises the swap too (elim precedes +2pp).
    o = tfs["order_at_f_deployed_0p22"]
    assert o.index("elimination") < o.index("payroll_plus_2pp")
    assert imp["order_at_f_deployed_realises_swap"] is True
    assert imp["swap_consistent_full_order_frame_exists_at_other_f"] is True
    # PROSE: sections 2 and 4 scope the claim to the pinned frame; 2a cites the
    # f-conditional caveat by the artifact field.
    for sec in ("2", "4"):
        assert "pinned" in " ".join(_section(sec).split()).lower()
    s2 = " ".join(_section(2).split())
    assert "order_at_f_deployed_0p22" in s2
    assert "f-conditional" in s2 or "0.0806/f" in s2
    # V2 (fixes round 2): the caveat's existence sentence AND the not-a-universal
    # header must BOTH be present -- AND, not OR, so deleting either is caught (the
    # verifier's M2b soft spot: the OR fallback let the existence sentence be dropped).
    assert "not one no frame can" in s2
    assert "not a universal" in s2.lower()
    # V2: the §2a restoration clause is scoped to the pinned A/B (needs f < 0.0476)
    # and restores the artifact's (<0.5) gloss + the no-contract-permitted-lever
    # conjunct -- so it is no longer the bare frame-level universal the verifier
    # flagged (which the extended BANNED_UNIVERSAL_C2_PHRASINGS guard now also bans).
    assert "0.0476" in s2
    assert "(<0.5)" in s2
    assert "no contract-permitted lever" in s2.lower()


# The universal-impossibility phrasings the a2 defect class produces: each asserts
# the C2/cap impossibility over representative frames IN GENERAL (or over earnings
# distributions in general) rather than scoped to the pinned certified frame, and
# forensics-3's own order_at_f_deployed_0p22 makes each false. A true class guard is
# impossible for a string guard (synonyms bypass it), so the KNOWN slots are
# enumerated; matched case-insensitively against the flattened doc. Fixes round 2
# (verification comment 4962578395, V2) extended the list past the round-1 slots to
# the earnings-distribution / representative-frame synonym class -- the §2a
# restoration-clause universal + the verifier's M4/M4b mutations.
BANNED_UNIVERSAL_C2_PHRASINGS = (
    # round-1 slots (Finding 1)
    "internally inconsistent as a representative-frame transport target",
    "internally inconsistent on a representative frame",
    "a frame cannot both realise the certified swap and reproduce",
    "any swap-realising frame overshoots",
    "no representative frame can hold",
    "no frame can realise the swap and",
    "the incoherent target",
    # round-2 additions (V2): the §2a restoration-clause universal (soft spot 1)
    # and the M4/M4b earnings-distribution / representative-frame synonyms.
    "no unimodal earnings distribution makes it small enough",
    "no earnings distribution of any shape",
    "no representative frame whatsoever",
    "realise the swap while reproducing smith's cap rank",
)


def test_no_universal_c2_impossibility_phrasing():
    """Finding 1's mutation guard (extended in fixes round 2): the a2 defect class --
    a universal impossibility the forensics refutes -- must not return, in any of its
    known synonym slots. Each banned phrasing asserts the claim UNIVERSALLY (over
    representative frames, or over earnings distributions, in general) rather than
    scoped to the pinned certified frame; the artifact's order_at_f_deployed_0p22
    makes each false. Case-insensitive so a capitalised reintroduction is caught. The
    verifier's M4 ("no earnings distribution of any shape can realise the swap and
    hold the full order") and M4b ("no representative frame whatsoever can realise the
    swap while reproducing Smith's cap rank") synonyms are now enumerated slots.
    """
    flat = " ".join(_doc_text().split()).lower()
    for banned in BANNED_UNIVERSAL_C2_PHRASINGS:
        assert banned not in flat, banned
    # the artifact truth that makes any such universal false is still bound.
    tfs = _forensics3()["q10_cap150k_adjacency"]["entailment"][
        "true_frame_sensitivity"
    ]
    o = tfs["order_at_f_deployed_0p22"]
    assert o == [
        "elimination",
        "payroll_plus_2pp",
        "payroll_plus_1pp",
        "cap_150k",
    ]
    assert o.index("elimination") < o.index("payroll_plus_2pp")


def test_fertility_lever_terminology_consistent():
    """Finding 2 (fixes round): the fertility lever is MAXED (at its ceiling), not
    at a 'floor' -- 'support floor' reads as the opposite of 'maxed'. Use
    ceiling/maxed consistently; the lever-exhaustion claim is unaffected."""
    assert "support floor" not in _doc_text()
    prose = _prose_without_ledger()
    assert "maxed" in prose
    assert "support ceiling" in prose
    # the substantive exhaustion claim survives (bound elsewhere too).
    ci = _forensics3()["q11_hhsize_residual"]["candidate_independence"]
    assert ci["verdict"] == "levers_exhausted"


# --------------------------------------------------------------------------
# The pair-scoped re-spec: anchor-supported, realised 3/3
# --------------------------------------------------------------------------
def test_pair_scoped_c2_anchor_supported_and_realised_3_of_3():
    led = _ledger()["pair_scoped_c2"]
    pr = _forensics3()["q10_cap150k_adjacency"]["pair_scoped_respec"]
    # anchor basis: Smith p.3 elim +21 > +2pp +18 (from every candidate's cube).
    for n in (1, 2, 3):
        deltas = _candidate(n)["family_c"]["fingerprints"]["c2"][
            "provision_deltas"
        ]["smith_year_deltas"]
        assert deltas["elimination"] == led["anchor_basis_smith_elim_years"]
        assert (
            deltas["payroll_plus_2pp"]
            == led["anchor_basis_smith_plus2pp_years"]
        )
    assert led["anchor_basis_smith_elim_years"] == 21
    assert led["anchor_basis_smith_plus2pp_years"] == 18
    assert (
        led["anchor_basis_smith_elim_years"]
        > led["anchor_basis_smith_plus2pp_years"]
    )
    # the swap realised 3/3 -- in the forensics record AND every candidate cube.
    assert pr["swap_realised_by_candidate"] == {
        "c1": True,
        "c2": True,
        "c3": True,
    }
    assert pr["realised_3_of_3"] is True and led["realised_3_of_3"] is True
    for n in (1, 2, 3):
        fp = _candidate(n)["family_c"]["fingerprints"]["c2"]
        assert fp["required_swap_pair"] == led["required_swap_pair"]
        assert fp["required_swap_realised"] is True
        assert led[f"swap_realised_c{n}"] is True
    # no published anchor supports the full 4-element ordering.
    key = (
        "any_published_anchor_supports_full_4_element_on_representative_frame"
    )
    assert pr[key] is False
    assert led["any_published_anchor_supports_full_4_element"] is False
    # the doc names all three options; c-c (4-element) is the untenable one.
    s4 = _section(4)
    assert "RECOMMENDED" in s4
    assert "demote family C entirely" in s4 or "demote family C" in s4
    assert "Untenable" in s4 or "untenable" in s4


def test_c2_record_bound():
    led = _ledger()["c2_record"]
    fp = _candidate(3)["family_c"]["fingerprints"]["c2"]
    assert fp["deployed_order"] == led["deployed_order"]
    assert (
        fp["required_representative_order"]
        == led["required_representative_order"]
    )
    assert (
        round(fp["kendall_tau_vs_required"], 4)
        == led["kendall_tau_vs_required_round4"]
        == 0.3333
    )
    deltas = fp["provision_deltas"]["smith_year_deltas"]
    assert deltas == led["smith_year_deltas"]
    assert (
        round(fp["provision_deltas"]["our_exhaustion_deltas"]["cap_150k"], 1)
        == led["cap_150k_deployed_years_round1"]
        == 16.7
    )
    assert deltas["cap_150k"] == led["cap_150k_smith_rank4_years"] == 1


# --------------------------------------------------------------------------
# OC consequence: 0.9344/0.9623 (47) AND 0.9403/0.9684 (43) on frozen floor sigmas
# --------------------------------------------------------------------------
def test_family_a_oc_recomputes_before_and_after_on_frozen_floor():
    led = _ledger()["family_a_oc"]
    floors = _floors()
    per = floors["faithful_candidate_oc"]["per_cell"]
    gate_eligible = floors["gate_partition"]["gate_eligible"]  # 53
    # reconstruct the 53-cell basis: LIVE 43 gated tolerances + the 10
    # retained (6 a2 + 4 a3) -- post-flip polarity.
    live_tol = _family_a_tolerances()
    assert len(live_tol) == 43
    retained = _gate_w1()["thresholds"]["family_a"]["retained_tolerances"]
    assert set(retained) == A2_DEMOTED | set(HH_DEMOTED)
    tol = dict(live_tol)
    tol.update({c: retained[c]["tolerance"] for c in retained})
    assert len(tol) == len(gate_eligible) == 53

    # BEFORE (committed a2 state): 53 minus the 6 a2-demoted = 47 live gated.
    surv47 = [c for c in gate_eligible if c not in A2_DEMOTED]
    assert len(surv47) == 47 == led["n_gated_before"]
    p_seed_b, p_gate_b = _oc(surv47, tol, per)
    assert round(p_seed_b, 4) == led["p_seed_before"] == 0.9344
    assert round(p_gate_b, 4) == led["p_gate_before"] == 0.9623
    # the LIVE contract's committed faithful_candidate_oc carries the a3
    # (after) values post-flip.
    foc = _gate_w1()["thresholds"]["family_a"]["faithful_candidate_oc"]
    assert foc["p_seed_pass"] == 0.9403 and foc["p_gate_pass_4_of_5"] == 0.9684

    # AFTER (a3 state): 47 minus the 4 hh_size = 43.
    assert set(HH_DEMOTED) <= set(surv47)
    surv43 = [c for c in surv47 if c not in HH_DEMOTED]
    assert len(surv43) == 43 == led["n_gated_after"]
    p_seed_a, p_gate_a = _oc(surv43, tol, per)
    assert round(p_seed_a, 4) == led["p_seed_after"] == 0.9403
    assert round(p_gate_a, 4) == led["p_gate_after"] == 0.9684
    assert led["invariant_under_amendment"] is False
    assert p_seed_a > p_seed_b and p_gate_a > p_gate_b

    passprobs = [
        2.0 * _normal_cdf(tol[c] / per[c]["realized_sigma"]) - 1.0
        for c in HH_DEMOTED
    ]
    assert (
        round(min(passprobs), 3)
        == led["demoted_hh_size_faithful_passprob_min_round3"]
        == 0.998
    )
    assert (
        round(max(passprobs), 4)
        == led["demoted_hh_size_faithful_passprob_max_round4"]
        == 0.9989
    )
    assert "prices sampling noise" in _prose_without_ledger()


def test_oc_statement_after_is_0p9684_times_pair_swap():
    led = _ledger()
    assert led["oc_statement_after"] == "0.9684 * I(pair_swap)"
    prose = _prose_without_ledger()
    assert "0.9684 × I(pair-swap)" in prose
    assert "44-cell" in prose


# --------------------------------------------------------------------------
# Partition consequence: 48 -> 44 gated / 84 -> 88 report-only
# --------------------------------------------------------------------------
def test_partition_arithmetic_bound():
    led = _ledger()
    fa = led["family_a_partition"]
    fc = led["family_c_partition"]
    ov = led["overall_partition"]
    # family A: 47 -> 43 gated (the four hh_size cells move), 58 -> 62 report.
    assert fa["gated_now"] == 47 and fa["gated_after"] == 43
    assert fa["gated_now"] - fa["gated_after"] == 4
    assert fa["report_only_now"] == 58 and fa["report_only_after"] == 62
    assert fa["report_only_after"] - fa["report_only_now"] == 4
    # family C: unchanged count (re-scope, not demotion).
    assert fc["gated_now"] == fc["gated_after"] == 1
    assert fc["report_only_now"] == fc["report_only_after"] == 1
    # overall: 48 -> 44 gated / 84 -> 88 report-only, total conserved (132).
    assert ov["gated_now"] == 47 + 1 == 48
    assert ov["gated_after"] == 43 + 1 == 44
    assert ov["report_only_now"] == 84 and ov["report_only_after"] == 88
    assert ov["gated_after"] + 4 == ov["gated_now"]
    assert ov["report_only_after"] - 4 == ov["report_only_now"]
    assert ov["gated_now"] + ov["report_only_now"] == 132
    assert ov["gated_after"] + ov["report_only_after"] == 132
    assert led["cube_shape_before"] == [20, 47, 5]
    assert led["cube_shape_after"] == [20, 43, 5]


# --------------------------------------------------------------------------
# BIND the operative flip text (section 7)
# --------------------------------------------------------------------------
def test_section7_flip_lists_four_hh_cells_and_reasons():
    s7 = _section(7)
    hh = set(re.findall(r"hh_size_share\.(1|3|4|5plus)\b", s7))
    assert hh == {"1", "3", "4", "5plus"}
    # the retained gated cell is named as staying gated.
    assert "hh_size_share.2" in s7
    # both machine reasons appear in the flip section.
    assert REASON_HH in s7
    assert REASON_CAP in s7
    # each demoted cell maps to the hh reason.
    flat = " ".join(s7.split())
    for cell in HH_DEMOTED:
        assert re.search(
            re.escape(cell) + r"`?\s*→\s*`?" + re.escape(REASON_HH), flat
        ), cell


def test_section7_rollups_and_transitions_bound():
    led = _ledger()
    s7 = " ".join(_section(7).split())
    rollups = set(re.findall(r"\*\*(\d+) gated / (\d+) report-only\*\*", s7))
    fa = led["family_a_partition"]
    fc = led["family_c_partition"]
    ov = led["overall_partition"]
    expected = {
        (str(fa["gated_after"]), str(fa["report_only_after"])),
        (str(fc["gated_after"]), str(fc["report_only_after"])),
        (str(ov["gated_after"]), str(ov["report_only_after"])),
    }
    assert rollups == expected
    assert ("44", "88") in rollups and ("43", "62") in rollups
    # OC + cube transitions string-bound.
    s7_raw = _section(7)
    assert "47 → **43**" in s7_raw
    assert "0.9344 → **0.9403**" in s7_raw
    assert "0.9623 → **0.9684**" in s7_raw
    assert "[20, 47, 5]" in s7_raw and "[20, 43, 5]" in s7_raw


def test_section8_history_figures_bound():
    led = _ledger()
    s8 = " ".join(_section(8).split())
    ov = led["overall_partition"]
    assert "2026-07-13-w1-hh-size-and-c2-pair" in s8
    assert "0.9623 x I(C2) to 0.9684 x I(pair-swap)" in s8
    assert "48 -> 44 gated, 84 -> 88 report-only" in s8
    assert "[20,47,5] -> [20,43,5]" in s8
    assert f"{ov['gated_after']}-cell" in s8
    assert "0.9925" in s8
    assert "1.694" in s8
    assert "0.111" in s8
    assert "realised 3/3" in s8
    assert "STAND FAIL" in s8
    # the whittling answer is carried in the permanent record.
    assert "KNOWN PASSABLE" in s8
    assert "43/43" in s8
    assert "pre-named" in s8.lower()
    # the false blanket must NOT appear in the history.
    assert "the household-size family is unreachable" not in s8


# --------------------------------------------------------------------------
# The whittling section (its own section) -- c3-passable, pre-named, prospective
# --------------------------------------------------------------------------
def test_whittling_section_states_c3_passable_first_and_binds_cube():
    """Section 3 must (a) state plainly that the amended surface would be passable by
    the committed c3 model, and (b) that claim recomputes: c3 lands 43/43 of the
    surviving family-A cells in-band on ALL FIVE seeds. If the whittling section is
    deleted, this fails (mutation guard)."""
    # the DEDICATED whittling section must exist as its own heading (deleting or
    # renaming it away from "The whittling question" fails here, even if the body
    # text survives elsewhere -- the mutation guard the task names).
    assert re.search(
        r"##\s*\d+\.\s*The whittling question", _doc_text()
    ), "the dedicated whittling section heading is missing/renamed"
    s3 = _section(3)
    assert (
        s3.splitlines()[0].lower().startswith("## 3. the whittling question")
    ), "section 3 is not the whittling section"
    flat = " ".join(s3.split())
    assert "whittling" in flat.lower()
    assert (
        "passable by the already-committed candidate-3 model" in flat
        or "passable by the already-committed" in flat
    )
    assert "43/43" in flat or "43 of 43" in flat
    led = _ledger()["whittling"]
    assert led["post_amendment_surface_c3_passable"] is True
    assert led["c3_family_a_cells_in_band_5of5_after"] == 43
    assert led["surface_known_passable_by_committed_model"] is True
    assert led["first_for_this_gate"] is True
    # CUBE TRUTH: c3 passes all 43 surviving family-A cells on all 5 seeds.
    surv43 = set(_family_a_tolerances()) - set(HH_DEMOTED)
    assert len(surv43) == 43
    c3 = _candidate(3)
    for seed in c3["family_a"]["per_seed"]:
        pc = seed["per_cell"]
        assert all(pc[c]["pass"] for c in surv43)
    # and the pair-swap realised for c3.
    assert (
        c3["family_c"]["fingerprints"]["c2"]["required_swap_realised"] is True
    )
    assert led["c3_pair_swap_realised"] is True


def test_whittling_boundary_was_prenamed_in_amendment2_forecast():
    """The demotion boundary is the permitted-lever line applied PROSPECTIVELY: the
    ratified amendment-2 forecast named EXACTLY this set as amendment3_risks before
    c3 ran. Bind the a3 target set to the a2 ledger's amendment3_risks."""
    led = _ledger()
    a2 = _ledger_of(A2_DOC.read_text(encoding="utf-8"))
    a2_risks = set(a2["amendment3_risks"])
    # a3 acts on exactly the a2-forecast set: 4 hh_size cells + the cap_150k adjacency.
    a3_target = set(HH_DEMOTED) | {"c2_cap_150k_adjacency"}
    assert a2_risks == a3_target
    assert set(led["prenamed_amendment3_risks_from_a2"]) == a2_risks
    assert led["whittling"]["pre_named_in_amendment2_forecast"] is True
    assert led["whittling"]["prospective_only"] is True
    assert led["amendment2_doc"].endswith(
        "gate_w1_amendment_2_family_a_concept_cells.md"
    )
    prose = _prose_without_ledger()
    assert "pre-named" in prose.lower()
    assert "prospective" in prose.lower()


# --------------------------------------------------------------------------
# No-self-rescue: c1/c2/c3 all STAND FAIL; c3's FAIL was under the current contract
# --------------------------------------------------------------------------
def test_no_self_rescue_all_three_stand_fail():
    led = _ledger()["no_self_rescue"]
    assert led["candidate1_pr"] == 162 and led["candidate1_gate_pass"] is False
    assert led["candidate2_pr"] == 167 and led["candidate2_gate_pass"] is False
    assert led["candidate3_pr"] == 176 and led["candidate3_gate_pass"] is False
    # c1/c2 committed FAIL.
    for n in (1, 2):
        assert _candidate(n)["verdict"]["gate_pass"] is False
    # c3: committed gate FAIL under the CURRENT contract.
    c3 = _candidate(3)
    assert c3["run"] == led["candidate3_run"] == "gate_w1_candidate3_v1"
    assert c3["verdict"]["gate_pass"] is False
    assert c3["family_a"]["family_a_pass"] is False
    assert c3["family_c"]["family_c_pass"] is False
    # family-A on the SCORING-TIME 47 surface (post-flip: reconstruct as the
    # live 43 gated cells + the a3-retained hh_size quad).
    gated47 = set(_family_a_tolerances()) | set(HH_DEMOTED)
    assert len(gated47) == 47
    seeds_pass = 0
    fail_union = set()
    for seed in c3["family_a"]["per_seed"]:
        pc = seed["per_cell"]
        fails = {c for c in gated47 if not pc[c]["pass"]}
        fail_union |= fails
        if not fails:
            seeds_pass += 1
    assert seeds_pass == led["candidate3_family_a_seeds_pass_47_surface"] == 0
    assert fail_union == set(HH_DEMOTED)
    assert set(led["candidate3_family_a_fail_union_47_surface"]) == set(
        HH_DEMOTED
    )
    # family-C: c3 FAILED the 4-element reversal (the current gated check).
    assert c3["family_c"]["both_reverse"] is False
    assert c3["family_c"]["fingerprints"]["c2"]["reversed_to_anchor"] is False
    assert led["candidate3_family_c_pass_4element"] is False
    assert led["candidate3_c2_reversed_to_anchor"] is False
    # the pair-swap it realised does NOT rescue it under the 4-element rule.
    assert led["candidate3_pair_swap_realised"] is True
    assert led["rescue_set_empty"] is True
    assert led["prospective_binding_only"] is True
    # candidate-4 follow-up is the unchanged c3 model, gated by reproduction.
    c4 = _ledger()["candidate4_followup"]
    assert c4["same_model_as_c3"] is True
    assert c4["registration_after_flip"] is True
    assert c4["subject_to_bitexact_reproduction_addendum"] is True
    assert _ledger()["family_b_unchanged_report_only"] is True


# --------------------------------------------------------------------------
# MUTATION GUARD 1: no blanket-impossibility sentence may reappear
# --------------------------------------------------------------------------
def test_no_blanket_impossibility_reintroduction():
    """The size-4 (clears c3 seed 1) and size-2 (clears c3 5/5) truths forbid any
    blanket 'the hh_size family is unreachable' claim. Bound to the cube so the
    false blanket cannot silently return. Case-insensitive so a capitalised
    reintroduction ("The household-size family is unreachable") is caught too.
    """
    flat = " ".join(_doc_text().split()).lower()
    for banned in (
        "the household-size family is unreachable",
        "the hh_size family is unreachable",
        "no candidate can clear any hh_size",
        "no candidate clears any hh_size",
        "no candidate can clear any household-size",
        "all five hh_size cells are unreachable",
        "none of the hh_size cells can be cleared",
        "size-4 cannot be cleared",
        "size-2 cannot be cleared",
    ):
        assert banned not in flat, banned
    # cube truth that makes any such blanket false.
    assert _seed_pass_counts(_candidate(3), "hh_size_share.4") == 1
    assert _seed_pass_counts(_candidate(3), HH_RETAINED_GATED) == 5


# --------------------------------------------------------------------------
# MUTATION GUARD 2: the proposal is a DRAFT -- gates.yaml is UNTOUCHED
# --------------------------------------------------------------------------
def test_gates_yaml_flipped_per_section7_and_moves_no_sibling():
    """Post-flip polarity of the proposal's untouched-guard: gates.yaml now
    carries section 7 exactly (43/62 family A with the hh_size quad in
    retained_tolerances; C2 pair-scoped with the cap_150k/+1pp legs
    report-only; OC 0.9403/0.9684; cube [20,43,5]), the PROPOSAL itself
    touched nothing (the ledger records that), and the flip moves NO sibling
    gate (the section-7 subset master-compare, stable pre- and post-merge)."""
    gw1 = _gate_w1()
    fa = gw1["thresholds"]["family_a"]
    fc = gw1["thresholds"]["family_c"]
    tol = _family_a_tolerances()
    # the four demoted cells are OUT of the views; size-2 stays gated.
    for cell in HH_DEMOTED:
        assert cell not in tol, cell
        assert cell in fa["report_only"], cell
        assert (
            fa["report_reasons"][cell]
            == "no_permitted_entry_state_lever_reaches_cell"
        )
    assert HH_RETAINED_GATED in tol
    assert len(tol) == 43
    # both amendments' retained sets are present, byte-exact.
    assert set(fa["retained_tolerances"]) == A2_DEMOTED | set(HH_DEMOTED)
    assert {
        c: fa["retained_tolerances"][c]["tolerance"] for c in HH_DEMOTED
    } == {
        "hh_size_share.1": 0.191,
        "hh_size_share.3": 0.191,
        "hh_size_share.4": 0.174,
        "hh_size_share.5plus": 0.184,
    }
    # C2 stays gated, re-scoped to the pair; the legs publish report-only.
    assert fc["gate_partition"]["gate_eligible"] == [
        "fingerprint.elimination_plus2pp"
    ]
    assert fc["gate_partition"]["n_gate_eligible"] == 1
    c2 = fc["fingerprints"]["c2"]
    assert "required_swap_realised" in c2["gated_check_amendment_3"]
    assert c2["report_only_legs_amendment_3"]["legs"] == [
        "cap_150k",
        "payroll_plus_1pp",
    ]
    # OC + cube shape carry the ratified a3 values.
    assert fa["faithful_candidate_oc"]["n_gated_cells"] == 43
    assert fa["faithful_candidate_oc"]["p_gate_pass_4_of_5"] == 0.9684
    proto = gw1["thresholds"]["protocol"]["fresh_run_artifact_schema"]
    assert proto["per_draw_per_cell_rates"]["shape"] == [20, 43, 5]
    assert _ledger()["gates_yaml_untouched_by_this_proposal"] is True
    # the section-7 subset master-compare: no sibling gate moves.
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
    with open(os.path.join(ROOT, "gates.yaml")) as fh:
        live_doc = yaml.safe_load(fh)["gates"]
    master_doc = yaml.safe_load(master_text)["gates"]
    assert set(master_doc) <= set(live_doc)
    for name in sorted(master_doc):
        if name == "gate_w1":
            continue
        assert live_doc[name] == master_doc[name], name


# --------------------------------------------------------------------------
# MUTATION GUARD 3: the section skeleton mirrors amendment 2 (11 sections + ledger)
# --------------------------------------------------------------------------
def test_section_skeleton_and_ceremony_checklist_present():
    text = _doc_text()
    for n in range(1, 12):
        assert f"## {n}." in text, f"section {n} missing"
    # section 7 is the flip, section 8 the history (mirrors amendment 2).
    assert "## 7. The exact prospective flip" in text
    assert "## 8. `amendment_history` entry draft" in text
    assert "## 9. Certification-scope language after amendment 3" in text
    # ceremony checklist: proposal + referee round + fixes round 1 DONE (fixes
    # revision); verification round still PENDING.
    s11 = _section(11)
    assert "- [x] **Proposal**" in s11
    assert "- [x] **Adversarial referee round**" in s11
    assert "4962052314" in s11  # the referee round is cited
    assert "- [x] **Fixes round 1**" in s11
    assert "- [ ] **Verification round**" in s11
    # certification-scope de-vacuous: both machine reasons appear in section 9.
    s9 = _section(9)
    assert REASON_HH in s9 and REASON_CAP in s9
