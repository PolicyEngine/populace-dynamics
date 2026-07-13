"""Proposal-consistency bindings for gate_w1 amendment 2 (the family-A concept
cells + the C1 fingerprint).

The amendment-2 PROPOSAL is a DRAFT: it argues, from W1 forensics 2 (Q6 the 65+
divorce over-accumulation, Q9 the 18-24 concept gap + the C1 consolidation) and
W1 forensics 1 Q5 (the C1 non-reversal analytic), that THREE cell groups on the
amended (post-amendment-1) 55-cell surface are unclearable by any
contract-consistent candidate and must be demoted to report-only (the package)
rather than re-anchored/re-specified now. This module proves the proposal
document does not drift from the committed evidence: it parses the machine-readable
``amendment-consistency-ledger`` block embedded in
``docs/amendments/gate_w1_amendment_2_family_a_concept_cells.md`` and cross-checks
EVERY load-bearing figure against the frozen artifacts --
``runs/gate_w1_forensics2_v1.json`` (Q6/Q9), ``runs/gate_w1_forensics1_v1.json``
(Q5), ``runs/gate_w1_floors_v1.json`` (the family-A OC machinery + partition),
``runs/gate_w1_candidate1_v1.json`` + ``runs/gate_w1_candidate2_v1.json`` (the two
committed FAILs the amendment must not rescue), and ``gates.yaml`` (still the
LOCKED, post-amendment-1 55-cell surface -- this proposal moves no threshold).

It ALSO binds the OPERATIVE flip text (the amendment-1 referee fix-B standard, not
repeated here): the section-7 enumerated demotion list (all 7 cells), the three
section-7 partition roll-ups (family A, family C, overall), the OC transition, and
the section-8 history figures are parsed from the doc and cross-checked against the
ledger/artifacts so they cannot silently drift; the machine-reason prose
assertions are checked OUTSIDE the ledger fence (de-vacuoused).

Unlike amendment 1 (which recomputed a family-A OC that was INVARIANT because the
demoted cells sat outside the family-A machinery), amendment 2 removes SIX cells
from family A, so the family-A OC is RECOMPUTED on the residual 47-cell surface
(0.922/0.9481 -> 0.9344/0.9623) -- and this module reproduces both, the exact
draw-noise-free half-normal basis tests/test_gate_w1_derivations.py uses.

ALWAYS-RUNNABLE: reads only committed ``runs/*.json`` + the doc + gates.yaml, no
data load, no h5, no PSID/PE-US checkout.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = (
    ROOT
    / "docs"
    / "amendments"
    / "gate_w1_amendment_2_family_a_concept_cells.md"
)
FORENSICS2 = ROOT / "runs" / "gate_w1_forensics2_v1.json"
FORENSICS1 = ROOT / "runs" / "gate_w1_forensics1_v1.json"
FLOORS = ROOT / "runs" / "gate_w1_floors_v1.json"
CANDIDATE1 = ROOT / "runs" / "gate_w1_candidate1_v1.json"
CANDIDATE2 = ROOT / "runs" / "gate_w1_candidate2_v1.json"
GATES = ROOT / "gates.yaml"

# The three cell groups the amendment demotes.
C1_CELL = "fingerprint.ppi_nra"
PARTICIPATION_18_24 = (
    "earnings_participation.18-24|female",
    "earnings_participation.18-24|male",
)
MARITAL_CORESIDENT_65 = (
    "marital_share.married.65+|female",
    "marital_share.married.65+|male",
    "coresident_spouse.65+|female",
    "coresident_spouse.65+|male",
)
FAMILY_A_DEMOTED = set(PARTICIPATION_18_24) | set(MARITAL_CORESIDENT_65)  # 6
ALL_DEMOTED = FAMILY_A_DEMOTED | {C1_CELL}  # 7

REASON_C1 = "fingerprint_reversal_not_realized"
REASON_18_24 = "population_concept_delta_head_spouse_universe"
REASON_65 = "cohort_vintage_hazard_frame_mismatch"


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


def _prose_without_ledger() -> str:
    """Doc text with the fenced ledger JSON removed (de-vacuous prose checks)."""
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


def _section(num: int) -> str:
    """Return the text of ``## {num}.`` up to the next ``## `` header."""
    lines = _doc_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"## {num}."):
            start = i
            break
    assert start is not None, f"section {num} not found in the proposal doc"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def _forensics2() -> dict:
    return json.loads(FORENSICS2.read_text())


def _forensics1() -> dict:
    return json.loads(FORENSICS1.read_text())


def _floors() -> dict:
    return json.loads(FLOORS.read_text())


def _candidate(n: int) -> dict:
    return json.loads((CANDIDATE1 if n == 1 else CANDIDATE2).read_text())


def _gate_w1() -> dict:
    return yaml.safe_load(GATES.read_text())["gates"]["gate_w1"]


def _family_a_tolerances() -> dict:
    tol: dict = {}
    for view in _gate_w1()["thresholds"]["family_a"]["views"].values():
        tol.update(view["tolerances"])
    return tol


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _oc(cells, tol: dict, per: dict) -> tuple[float, float]:
    """Draw-noise-free half-normal OC over ``cells`` (the derivations basis)."""
    p_seed = 1.0
    for cell in cells:
        p_seed *= (
            2.0 * _normal_cdf(tol[cell] / per[cell]["realized_sigma"]) - 1.0
        )
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    return p_seed, p_gate


# --------------------------------------------------------------------------
# The ledger is well formed and names the recommendation + machine reasons
# --------------------------------------------------------------------------
def test_ledger_parses_and_headline_matches_prose():
    led = _ledger()
    assert led["amendment_id"] == "2026-07-12-w1-family-a-concept-cells"
    assert led["recommendation"] == "report_only_package_all_three_groups"
    assert led["n_cells_demoted"] == 7
    groups = led["groups"]
    assert set(groups["c1_fingerprint"]["cells"]) == {C1_CELL}
    assert set(groups["participation_18_24"]["cells"]) == set(
        PARTICIPATION_18_24
    )
    assert set(groups["marital_coresident_65plus"]["cells"]) == set(
        MARITAL_CORESIDENT_65
    )
    assert groups["c1_fingerprint"]["machine_reason"] == REASON_C1
    assert groups["participation_18_24"]["machine_reason"] == REASON_18_24
    assert groups["marital_coresident_65plus"]["machine_reason"] == REASON_65
    # union of the three groups is exactly the 7 demoted cells.
    union = (
        set(groups["c1_fingerprint"]["cells"])
        | set(groups["participation_18_24"]["cells"])
        | set(groups["marital_coresident_65plus"]["cells"])
    )
    assert union == ALL_DEMOTED and len(union) == 7
    # de-vacuoused: recommendation + package language appear in the PROSE.
    prose = _prose_without_ledger()
    assert "report-only package" in prose
    assert "RECOMMENDED" in prose
    assert led["gates_yaml_untouched_by_this_proposal"] is True


def test_machine_reasons_named_in_prose_outside_ledger():
    """De-vacuous: the ledger block holds these strings, so check them in the
    prose with the ledger fence removed (the amendment-1 fix-B standard)."""
    prose = _prose_without_ledger()
    assert (
        '"machine_reason"' not in prose
    )  # sanity: the ledger really was stripped
    assert REASON_C1 in prose
    assert REASON_18_24 in prose
    assert REASON_65 in prose


# --------------------------------------------------------------------------
# Q9 / Q5 evidence: the 18-24 concept gap + the C1 non-reversal are bound
# --------------------------------------------------------------------------
def test_forensics2_pointers_and_frame_side_flags():
    led = _ledger()["forensics2"]
    art = _forensics2()
    assert led["artifact"] == "runs/gate_w1_forensics2_v1.json"
    assert (
        led["registration"]
        == art["registration"]["comment_id"]
        == "4953088871"
    )
    assert led["grading"] == "4953311492"  # the forensics-2 grading (task)
    assert (
        art["reported_not_gated"] is True and led["reported_not_gated"] is True
    )
    assert art["protocol"]["train_frame_side_only"] is True
    assert led["train_frame_side_only"] is True
    assert art["protocol"]["publishes_regardless"] is True


def test_q9_concept_gap_18_24_bound():
    led = _ledger()["forensics2"]
    q9 = _forensics2()["q9_concept_cells"]["concept_gap_18_24_participation"]
    gap = q9["concept_gap_psid_minus_cps"]
    assert (
        round(q9["pooled_gap_pp"], 1)
        == led["concept_gap_18_24_pooled_pp_round1"]
    )
    assert led["concept_gap_18_24_pooled_pp_round1"] == 22.1
    assert (
        round(gap["female"], 3)
        == led["concept_gap_18_24_female_round3"]
        == 0.173
    )
    assert (
        round(gap["male"], 3) == led["concept_gap_18_24_male_round3"] == 0.276
    )
    assert (
        round(q9["psid_head_spouse_universe"]["pooled"], 3)
        == led["psid_head_spouse_pooled_round3"]
        == 0.865
    )
    assert (
        round(q9["cps_all_person_frame"]["pooled"], 3)
        == led["cps_all_person_pooled_round3"]
        == 0.644
    )
    assert q9["exceeds_15pp_amendment_threshold"] is True
    assert led["exceeds_15pp_amendment_threshold"] is True
    # the gap exceeds the male 18-24 tolerance -> structural fail (candidate-indep).
    assert gap["male"] > 0  # concept gap is positive (PSID over-participates)


def test_q6_65plus_divorce_over_accumulation_bound():
    led = _ledger()["forensics2"]
    comp = _forensics2()["q6_marital_calibration_frame"]["composition_65plus"]
    chan = _forensics2()["q6_marital_calibration_frame"][
        "dissolution_channel_65plus"
    ]
    # divorce is the realized channel, widowhood is NOT (the pre-reg was wrong).
    for sex in ("female", "male"):
        assert chan[sex]["dominant_excess_status"] == "divorced"
        assert chan[sex]["widowhood_channel_realized"] is False
    assert led["realized_65plus_channel"] == "divorce"
    assert led["widowhood_channel_realized_65plus"] is False
    # divorced excess and married deployed/frame levels.
    df = comp["female"]
    dm = comp["male"]
    assert (
        round(df["divorced"]["excess_deployed_minus_frame"], 3)
        == led["divorced_excess_65plus_female_round3"]
        == 0.12
    )
    assert (
        round(dm["divorced"]["excess_deployed_minus_frame"], 3)
        == led["divorced_excess_65plus_male_round3"]
        == 0.127
    )
    assert (
        round(df["married"]["deployed"], 3)
        == led["married_65plus_deployed_female_round3"]
        == 0.573
    )
    assert (
        round(df["married"]["frame"], 3)
        == led["married_65plus_frame_female_round3"]
        == 0.691
    )
    assert (
        round(dm["married"]["deployed"], 3)
        == led["married_65plus_deployed_male_round3"]
        == 0.707
    )
    assert (
        round(dm["married"]["frame"], 3)
        == led["married_65plus_frame_male_round3"]
        == 0.856
    )


def test_q6_contract_adjudication_says_65plus_unfixable_by_permitted_lever():
    """The load-bearing finding for group (c): a CPS-anchored ENTRY model is
    contract-permitted and fixes 25-34, but CANNOT fix the 65+ undershoot (a
    hazard-evolution miss); back-solving the entry is the identity in disguise.
    """
    det = _forensics2()["q6_marital_calibration_frame"][
        "contract_adjudication"
    ]["determination"]
    assert "CONTRACT-PERMITTED" in det
    assert "CANNOT fix the 65+ undershoot" in det
    assert "identity in disguise" in det
    # the doc's construction-attempt table names the permitted levers + failures.
    s4c = _section(4)
    for lever in (
        "CPS-anchored entry",
        "identity",
        "re-calibration",
        "widowhood",
    ):
        assert lever in s4c


def test_c1_non_reversal_bound_from_forensics1_q5_and_forensics2():
    led = _ledger()["forensics1_q5_c1"]
    q5 = _forensics1()["q5_tail_upper_read"]
    assert led["artifact"] == "runs/gate_w1_forensics1_v1.json"
    assert led["registration"] == "4951218279"
    # upper read (heaviest tail, most favourable to PPI): PPI << NRA.
    assert (
        round(q5["upper_read"]["ppi_savings_abs"], 4)
        == led["upper_read_ppi_savings_round4"]
        == 0.0169
    )
    assert (
        round(q5["upper_read"]["nra_savings_abs"], 4)
        == led["upper_read_nra_savings_round4"]
        == 0.2023
    )
    assert (
        round(q5["upper_read"]["ppi_minus_nra"], 4)
        == led["upper_read_ppi_minus_nra_round4"]
        == -0.1854
    )
    # corrected (realistic) tail moves PPI DOWN -> further from reversal.
    assert (
        round(q5["corrected_tail"]["ppi_savings_abs"], 4)
        == led["corrected_ppi_savings_round4"]
        == 0.0137
    )
    assert (
        round(q5["corrected_tail"]["nra_savings_abs"], 4)
        == led["corrected_nra_savings_round4"]
        == 0.2022
    )
    assert q5["upper_read"]["c1_reversed"] is False
    assert q5["corrected_tail"]["c1_reversed"] is False
    assert q5["c1_robustness_answer"]["answer_non_reversal_is_robust"] is True
    assert led["non_reversal_is_robust"] is True
    # cross-check the same figures live in forensics-2's c1 consolidation block.
    c1b = _forensics2()["q9_concept_cells"]["c1_binary"]
    assert c1b["forensics1_analytic"]["non_reversal_is_robust"] is True
    assert c1b["candidate1_empirical"]["c1_reversed"] is False
    assert led["candidate1_c1_reversed"] is False
    assert (
        round(c1b["candidate2_empirical"]["kendall_tau_vs_required"], 4)
        == led["candidate2_c1_kendall_tau_vs_required_round4"]
        == 0.6667
    )
    assert c1b["candidate2_empirical"]["c1_reversed_to_anchor"] is False
    assert led["candidate2_c1_reversed"] is False


# --------------------------------------------------------------------------
# OC consequence: family-A OC recomputes 0.922/0.9481 (53) AND 0.9344/0.9623 (47)
# --------------------------------------------------------------------------
def test_family_a_oc_recomputes_before_and_after_on_the_derivations_basis():
    """Amendment 2 differs from amendment 1: it removes 6 cells FROM family A, so
    the OC CHANGES. Reproduce the committed 53-cell 0.922/0.9481 AND the amended
    47-cell 0.9344/0.9623 from the frozen floor sigmas + the locked tolerances.
    """
    led = _ledger()["family_a_oc"]
    floors = _floors()
    per = floors["faithful_candidate_oc"]["per_cell"]
    gate_eligible = floors["gate_partition"]["gate_eligible"]
    tol = _family_a_tolerances()
    assert len(tol) == len(gate_eligible) == 53 == led["n_gated_before"]

    # BEFORE: the committed 53-cell characteristic (the lock's own value).
    p_seed_b, p_gate_b = _oc(gate_eligible, tol, per)
    assert round(p_seed_b, 4) == led["p_seed_before"] == 0.922
    assert round(p_gate_b, 4) == led["p_gate_before"] == 0.9481
    assert round(p_seed_b, 4) == floors["faithful_candidate_oc"]["p_seed_pass"]
    assert (
        round(p_gate_b, 4)
        == floors["faithful_candidate_oc"]["p_gate_pass_4_of_5"]
    )

    # the six demoted family-A cells are all currently gate-eligible.
    assert FAMILY_A_DEMOTED <= set(gate_eligible)
    surviving = [c for c in gate_eligible if c not in FAMILY_A_DEMOTED]
    assert len(surviving) == 47 == led["n_gated_after"]

    # AFTER: recompute on the residual 47-cell surface.
    p_seed_a, p_gate_a = _oc(surviving, tol, per)
    assert round(p_seed_a, 4) == led["p_seed_after"] == 0.9344
    assert round(p_gate_a, 4) == led["p_gate_after"] == 0.9623
    # NOT invariant (the amendment-1 contrast) -- removing cells raised p_seed.
    assert led["invariant_under_amendment"] is False
    assert p_seed_a > p_seed_b and p_gate_a > p_gate_b

    # the crux: each demoted family-A cell is ~trivially passable on the
    # SAMPLING-noise basis (0.997-0.998), yet unclearable on concept/vintage bias.
    passprobs = [
        2.0 * _normal_cdf(tol[c] / per[c]["realized_sigma"]) - 1.0
        for c in FAMILY_A_DEMOTED
    ]
    assert (
        round(min(passprobs), 3)
        == led["demoted_family_a_cell_faithful_passprob_min_round3"]
    )
    assert (
        round(max(passprobs), 3)
        == led["demoted_family_a_cell_faithful_passprob_max_round3"]
    )
    assert (
        min(passprobs) > 0.99
    )  # trivially passable under the half-normal model
    # the crux is stated in the prose (section 3/4).
    prose = _prose_without_ledger()
    assert "prices sampling noise, not concept" in prose or (
        "sampling noise" in prose and "concept/vintage bias" in prose
    )


def test_oc_statement_after_is_0p9623_times_c2():
    led = _ledger()
    assert led["oc_statement_after"] == "0.9623 * I(c2)"
    prose = _prose_without_ledger()
    assert "0.9623 × I(C2)" in prose
    assert "48-cell" in prose  # the residual gated surface size


# --------------------------------------------------------------------------
# Partition consequence: 55 -> 48 gated; family A 53 -> 47; family C 2 -> 1
# --------------------------------------------------------------------------
def test_partition_arithmetic_bound_to_current_gates_yaml():
    """The current (DRAFT) gates.yaml still gates the full 55 (family A 53 +
    family C 2). The ledger's after-values are the arithmetic of the demotion.
    """
    led = _ledger()
    fa = _gate_w1()["thresholds"]["family_a"]
    fc = _gate_w1()["thresholds"]["family_c"]

    # current family A: 53 gated (tolerances) / 52 report-only.
    n_gated_fa = len(_family_a_tolerances())
    n_ro_fa = len(fa["report_only"])
    assert n_gated_fa == led["family_a_partition"]["gated_now"] == 53
    assert n_ro_fa == led["family_a_partition"]["report_only_now"] == 52
    assert led["family_a_partition"]["gated_after"] == 53 - 6 == 47
    assert led["family_a_partition"]["report_only_after"] == 52 + 6 == 58

    # current family C: 2 gate-eligible / 0 report-only; C1 still gated.
    part = fc["gate_partition"]
    assert (
        part["n_gate_eligible"] == led["family_c_partition"]["gated_now"] == 2
    )
    assert (
        part["n_report_only"]
        == led["family_c_partition"]["report_only_now"]
        == 0
    )
    assert f"fingerprint.{fc['fingerprints']['c1']['id']}" == C1_CELL
    assert C1_CELL in part["gate_eligible"]  # DRAFT: C1 still gated
    assert led["family_c_partition"]["gated_after"] == 2 - 1 == 1
    assert led["family_c_partition"]["report_only_after"] == 0 + 1 == 1

    # overall: 55 -> 48 gated; 77 -> 84 report-only.
    ov = led["overall_partition"]
    assert (
        ov["gated_now"] == 53 + 0 + 2 == 55
    )  # family B gates 0 after amendment 1
    assert ov["gated_after"] == 47 + 0 + 1 == 48
    assert ov["report_only_now"] == 77
    assert ov["report_only_after"] == 84
    assert ov["gated_after"] + 7 == ov["gated_now"]  # 7 cells demoted
    assert ov["report_only_after"] - 7 == ov["report_only_now"]


# --------------------------------------------------------------------------
# BIND the operative flip text (section 7): the 7 cells + the 3 roll-ups
# --------------------------------------------------------------------------
def test_section7_flip_list_binds_all_seven_demoted_cells():
    """The section-7 flip must name EXACTLY the 7 cells: the 2 participation, the
    4 marital/coresident, and C1. Deleting any key now fails (fix-B standard).
    """
    s7 = _section(7)
    part = set(re.findall(r"earnings_participation\.18-24\|(female|male)", s7))
    marr = set(re.findall(r"marital_share\.married\.65\+\|(female|male)", s7))
    cores = set(re.findall(r"coresident_spouse\.65\+\|(female|male)", s7))
    has_c1 = "fingerprint.ppi_nra" in s7 or "ppi_nra" in s7
    parsed = (
        {f"earnings_participation.18-24|{s}" for s in part}
        | {f"marital_share.married.65+|{s}" for s in marr}
        | {f"coresident_spouse.65+|{s}" for s in cores}
        | ({C1_CELL} if has_c1 else set())
    )
    assert part == {"female", "male"}
    assert marr == {"female", "male"}
    assert cores == {"female", "male"}
    assert has_c1
    assert parsed == ALL_DEMOTED and len(parsed) == 7
    # each machine reason appears in the flip section, mapped to its group.
    assert REASON_C1 in s7 and REASON_18_24 in s7 and REASON_65 in s7


def test_section7_three_rollups_match_ledger_partition():
    """The three bolded section-7 roll-ups (family A 47/58, family C 1/1, overall
    48/84) are parsed and cross-checked against the ledger. Any mutation of any
    roll-up now fails."""
    led = _ledger()
    s7 = " ".join(
        _section(7).split()
    )  # collapse wrapped whitespace before match
    rollups = set(re.findall(r"\*\*(\d+) gated / (\d+) report-only\*\*", s7))
    fa = led["family_a_partition"]
    fc = led["family_c_partition"]
    ov = led["overall_partition"]
    expected = {
        (str(fa["gated_after"]), str(fa["report_only_after"])),  # 47 / 58
        (str(fc["gated_after"]), str(fc["report_only_after"])),  # 1 / 1
        (str(ov["gated_after"]), str(ov["report_only_after"])),  # 48 / 84
    }
    assert rollups == expected
    assert ("48", "84") in rollups  # the overall roll-up, explicitly


def test_section7_oc_and_cube_transitions_bound():
    """Section 7 item 1/2: the OC transition (53->47, 0.922->0.9344,
    0.9481->0.9623) and the committed-cube shape transition [20,53,5]->[20,47,5].
    """
    led = _ledger()["family_a_oc"]
    s7 = _section(7)
    assert (
        f"{led['n_gated_before']} → **{led['n_gated_after']}**" in s7
        or "53 → **47**" in s7
    )
    assert f"0.922 → **{led['p_seed_after']}**" in s7
    assert f"0.9481 → **{led['p_gate_after']}**" in s7
    assert "[20, 53, 5]" in s7 and "[20, 47, 5]" in s7


def test_section8_history_figures_bound():
    """The section-8 permanent history entry's figures are parsed and bound to the
    ledger. A drift of p_gate, the partition transition, or the concept gap fails.
    """
    led = _ledger()
    s8 = " ".join(_section(8).split())  # folded YAML -> collapse whitespace
    oc = led["family_a_oc"]
    ov = led["overall_partition"]
    f2 = led["forensics2"]
    c1 = led["forensics1_q5_c1"]
    # the recomputed OC transition carried verbatim.
    assert f"p_seed {oc['p_seed_after']} / p_gate {oc['p_gate_after']}" in s8
    assert "from 53 cells / 0.922 / 0.9481" in s8
    # the exact overall partition transition.
    transition = (
        f"{ov['gated_now']} -> {ov['gated_after']} gated, "
        f"{ov['report_only_now']} -> {ov['report_only_after']} report-only"
    )
    assert transition in s8
    assert "55 -> 48 gated, 77 -> 84 report-only" in s8
    assert f"{ov['gated_after']}-cell" in s8  # residual surface size
    # the three groups' headline evidence in the record.
    assert f"{f2['concept_gap_18_24_pooled_pp_round1']}pp pooled" in s8
    assert f"{c1['upper_read_ppi_savings_round4']}" in s8  # 0.0169
    assert f"{c1['upper_read_nra_savings_round4']}" in s8  # 0.2023
    assert (
        "divorce over-accumulation" in s8.lower()
        or "DIVORCE over-accumulation" in s8
    )


# --------------------------------------------------------------------------
# No-self-rescue: BOTH candidates STAND FAIL on the amended 48-cell surface
# --------------------------------------------------------------------------
def _seeds_pass_on_47_surface(cand: dict) -> int:
    """Count family-A seeds that pass on the residual 47-cell surface."""
    n_pass = 0
    for seed in cand["family_a"]["per_seed"]:
        per_cell = seed["per_cell"]
        surviving = [c for c in per_cell if c not in FAMILY_A_DEMOTED]
        assert len(surviving) == 47
        if all(per_cell[c]["pass"] for c in surviving):
            n_pass += 1
    return n_pass


@pytest.mark.parametrize("n", [1, 2])
def test_no_self_rescue_candidate_stands_fail_over_determined(n):
    """Demoting the 7 cells rescues NEITHER candidate: on the amended surface each
    still fails family A (0/5 seeds on the 47) AND family C (C2 does not reverse).
    Over-determined -- both surviving families fail."""
    led = _ledger()["no_self_rescue"]
    cand = _candidate(n)
    key = f"candidate{n}"
    assert cand["run"] == led[f"{key}_run"] == f"gate_w1_candidate{n}_v1"
    assert cand["verdict"]["gate_pass"] is False
    assert led[f"{key}_gate_pass"] is False

    # family A on the residual 47-cell surface: still 0/5 seeds.
    seeds = _seeds_pass_on_47_surface(cand)
    assert seeds == led[f"{key}_seeds_pass_47_surface"] == 0
    assert led[f"{key}_family_a_pass_47_surface"] is False

    # family C after C1 demoted = C2 only; C2 does not reverse.
    c2 = cand["family_c"]["fingerprints"]["c2"]
    assert c2["reversed_to_anchor"] is False
    assert led[f"{key}_c2_reversed"] is False

    # over-determined: BOTH surviving families fail -> gate fails regardless.
    assert (seeds >= 4) is False  # family A fails
    assert c2["reversed_to_anchor"] is False  # family C fails
    assert led["rescue_set_empty"] is True
    assert led["over_determined_both_families"] is True
    assert led[f"{key}_pr"] == (162 if n == 1 else 167)


def test_no_self_rescue_demoted_cells_were_failing_for_both_candidates():
    """The demoted cells were themselves failing for both candidates (the male
    18-24 and both 65+ male cells are hard 0/5 fails), so removing them cannot
    flip a pass -- the empirical leg of the candidate-independence argument."""
    hard_cells = (
        "earnings_participation.18-24|male",
        "marital_share.married.65+|male",
        "coresident_spouse.65+|male",
    )
    for n in (1, 2):
        cand = _candidate(n)
        for cell in hard_cells:
            passes = [
                s["per_cell"][cell]["pass"]
                for s in cand["family_a"]["per_seed"]
            ]
            assert sum(passes) == 0, (n, cell)  # 0/5 seeds pass
        # C1 never reverses for either candidate.
        assert (
            cand["family_c"]["fingerprints"]["c1"]["reversed_to_anchor"]
            is False
        )


# --------------------------------------------------------------------------
# The proposal is a DRAFT: gates.yaml is UNTOUCHED (no flip)
# --------------------------------------------------------------------------
def test_gates_yaml_untouched_all_55_cells_still_gated():
    """The PROPOSAL moves no threshold: gates.yaml still gates the full 55 (family
    A 53, family C 2), C1 and the 6 family-A cells are STILL gate-eligible, and
    gate_w1 is byte-identical to origin/master."""
    gw1 = _gate_w1()
    fa = gw1["thresholds"]["family_a"]
    fc = gw1["thresholds"]["family_c"]
    # the 6 family-A cells are still gated (present in a view's tolerances).
    tol = _family_a_tolerances()
    assert FAMILY_A_DEMOTED <= set(tol)
    # none of the 6 is in family-A report_only yet.
    assert set(fa["report_only"]).isdisjoint(FAMILY_A_DEMOTED)
    # C1 still gate-eligible; family C gates 2.
    assert C1_CELL in fc["gate_partition"]["gate_eligible"]
    assert fc["gate_partition"]["n_gate_eligible"] == 2
    # faithful_candidate_oc still records the 53-cell characteristic (unflipped).
    assert fa["faithful_candidate_oc"]["n_gated_cells"] == 53
    assert fa["faithful_candidate_oc"]["p_gate_pass_4_of_5"] == 0.9481
    assert _ledger()["gates_yaml_untouched_by_this_proposal"] is True

    # byte-identical to origin/master (a DRAFT edits no gate cell).
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
    assert gw1 == master  # gate_w1 unchanged by this proposal
