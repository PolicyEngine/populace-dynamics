"""Proposal-consistency bindings for gate_w1 amendment 2 (the family-A concept
cells + the C1 fingerprint).

The amendment-2 PROPOSAL is a DRAFT: it argues, from W1 forensics 2 (Q6 the 65+
divorce over-accumulation, Q9 the 18-24 concept gap + the C1 consolidation), W1
forensics 1 Q5/Q1 (the C1 non-reversal analytic + the 65+ equilibration ceiling),
and the adversarial referee round (PR #169 comment 4953763322), that THREE cell
groups on the amended (post-amendment-1) 55-cell surface cannot be cleared at the
gate-relevant >=4-of-5-seed level by any contract-consistent candidate and must be
demoted to report-only.

This module proves the proposal document does not drift from the committed
evidence: it parses the machine-readable ``amendment-consistency-ledger`` block in
``docs/amendments/gate_w1_amendment_2_family_a_concept_cells.md`` and cross-checks
EVERY load-bearing figure against the frozen artifacts. After the referee's AMEND
THE AMENDMENT round it ALSO binds the PER-CELL truth (finding 2): the demoted
cells' committed per-seed pass records (candidate 1 PASSED
``coresident_spouse.65+|female`` 4/5), the two distinct 65+ machine reasons
(ceiling-vs-window vs scored-duplicate), the ceiling-vs-window arithmetic, the
corrected sigma ranges (finding 5), and the retained-surface candor + series
forecast (finding 4: Q7 REFUTED, the C2 record).

It binds the OPERATIVE flip text (the amendment-1 fix-B standard): the section-7
demotion list (all 7 cells), the three partition roll-ups, the OC transition, and
the section-8 history figures are parsed and cross-checked so they cannot silently
drift; prose assertions are checked OUTSIDE the ledger fence (de-vacuoused).

ALWAYS-RUNNABLE (artifact tier): reads only committed ``runs/*.json`` + the doc +
gates.yaml, no data load, no h5, no PSID/PE-US checkout.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from statistics import mean

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

# Per-cell machine reasons (finding 2: per-cell truth, two reasons for the 65+
# quad -- ceiling-vs-window for the cells the stationary distribution cannot reach,
# scored-duplicate for the coresident cell candidate 1 clears 4/5).
REASON_C1 = "fingerprint_reversal_not_realized"
REASON_18_24 = "population_concept_delta_head_spouse_universe"
REASON_CEILING = "cohort_vintage_hazard_frame_mismatch"
REASON_DUPLICATE = "scored_duplicate_of_demoted_married_quantity"
CEILING_CELLS = {
    "marital_share.married.65+|female",
    "marital_share.married.65+|male",
    "coresident_spouse.65+|male",
}
DUPLICATE_CELL = "coresident_spouse.65+|female"


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _ledger() -> dict:
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
    """Return ``## {num}.`` up to the next ``## `` header (num may be '4' etc.)."""
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
# Ledger: recommendation + per-cell machine reasons (two for the 65+ quad)
# --------------------------------------------------------------------------
def test_ledger_parses_and_headline_matches_prose():
    led = _ledger()
    assert led["amendment_id"] == "2026-07-12-w1-family-a-concept-cells"
    assert led["recommendation"] == "report_only_package_all_three_groups"
    assert led["n_cells_demoted"] == 7
    assert led["referee_round_comment"] == "4953763322"
    reasons = led["cell_reasons"]
    assert set(reasons) == ALL_DEMOTED and len(reasons) == 7
    assert reasons[C1_CELL] == REASON_C1
    assert all(reasons[c] == REASON_18_24 for c in PARTICIPATION_18_24)
    assert all(reasons[c] == REASON_CEILING for c in CEILING_CELLS)
    assert reasons[DUPLICATE_CELL] == REASON_DUPLICATE
    prose = _prose_without_ledger()
    assert "report-only package" in prose
    assert "RECOMMENDED" in prose
    assert led["gates_yaml_untouched_by_this_proposal"] is True


def test_machine_reasons_named_in_prose_outside_ledger():
    """De-vacuous: all FOUR machine reasons appear in the prose with the ledger
    fence removed (the amendment-1 fix-B standard)."""
    prose = _prose_without_ledger()
    assert '"machine_reason"' not in prose and '"cell_reasons"' not in prose
    for reason in (REASON_C1, REASON_18_24, REASON_CEILING, REASON_DUPLICATE):
        assert reason in prose, reason


# --------------------------------------------------------------------------
# Finding 2 (per-cell truth): the committed per-seed pass records
# --------------------------------------------------------------------------
def test_per_cell_pass_records_bound_to_committed_artifacts():
    """The ledger's per-seed pass records recompute from the candidate cubes.
    This is the F1 correction: candidate 1 PASSED coresident_spouse.65+|female 4/5
    and marital_share.married.65+|female 2/5 -- the female cells are NOT
    unclearable, so 'no candidate can clear any of the 7' is false and forbidden.
    """
    led = _ledger()["per_cell_pass_record"]
    for cell in FAMILY_A_DEMOTED:
        assert (
            _seed_pass_counts(_candidate(1), cell) == led[cell]["candidate1"]
        ), cell
        assert (
            _seed_pass_counts(_candidate(2), cell) == led[cell]["candidate2"]
        ), cell
    # the specific facts the referee flagged.
    assert led["coresident_spouse.65+|female"]["candidate1"] == 4
    assert led["marital_share.married.65+|female"]["candidate1"] == 2
    assert led["earnings_participation.18-24|female"]["candidate2"] == 2
    # the MALE cells are the hard 0/5 that force gate-level unpassability.
    for cell in (
        "earnings_participation.18-24|male",
        "marital_share.married.65+|male",
        "coresident_spouse.65+|male",
    ):
        assert led[cell]["candidate1"] == 0 and led[cell]["candidate2"] == 0
    # the doc states the coresident-F 4/5 pass PLAINLY (prose, not only ledger).
    prose = _prose_without_ledger()
    assert "4 of 5 seeds" in prose or "4/5" in prose
    assert "candidate 1" in prose.lower() or "candidate 1" in prose


def test_65plus_two_machine_reasons_grounds_are_per_cell_true():
    """The 65+ quad carries TWO reasons, each true: ceiling-vs-window for the three
    cells the stationary distribution cannot reach at >=4/5 (married pair + male
    coresident), scored-duplicate for coresident-65+|female (candidate 1 clears
    4/5). Bound to the committed pass records so a mislabel fails."""
    led = _ledger()
    reasons = led["cell_reasons"]
    rec = led["per_cell_pass_record"]
    # every ceiling cell fails >=3/5 for BOTH candidates (cannot reach at >=4/5).
    for cell in CEILING_CELLS:
        assert reasons[cell] == REASON_CEILING
        assert rec[cell]["candidate1"] <= 2 and rec[cell]["candidate2"] <= 2
    # the duplicate cell is the one that CAN clear at >=4/5 (candidate 1 = 4).
    assert reasons[DUPLICATE_CELL] == REASON_DUPLICATE
    assert rec[DUPLICATE_CELL]["candidate1"] == 4
    # section 4c names both grounds explicitly.
    s4 = _section(4)
    assert "ceiling-vs-window" in s4
    assert "scored duplicate" in s4 or "scored_duplicate" in s4
    assert (
        "duplicate of the demoted married" in s4
        or "duplicate of a demoted" in s4
    )


def test_ceiling_vs_window_65plus_female_arithmetic_bound():
    """Married-65+|female caps 2-3/5 by ceiling-vs-window, NOT the +0.12 magnitude:
    the CANDIDATE_16 stationary ceiling ~0.577 (Q1 entry-0 0.5767, Q6 entry-0.58
    0.5728) sits at the per-seed lower window edges; and coresident==married on
    every seed (ratio 1.0) so the identical value straddles an easier window at
    4/5."""
    led = _ledger()["ceiling_vs_window_65plus_female"]
    f1 = _forensics1()
    f2 = _forensics2()
    # the two stationary anchors.
    q1 = f1["q1_marital_equilibration"]["per_band_sex"]["65+|female"]
    assert (
        round(q1["synthetic_equilibration_S"], 4)
        == led["q1_entry0_stationary_round4"]
        == 0.5767
    )
    q6 = f2["q6_marital_calibration_frame"]["per_band_sex"]["65+|female"]
    assert (
        round(q6["terminal_deployed_D"], 4)
        == led["q6_entry058_terminal_round4"]
        == 0.5728
    )
    # coresident IS regenerated as married: deployed ratio 1.0 on all seeds, both.
    for n in (1, 2):
        ps = _candidate(n)["family_a"]["per_seed"]
        for s in ps:
            m = s["per_cell"]["marital_share.married.65+|female"]["rbar"]
            c = s["per_cell"]["coresident_spouse.65+|female"]["rbar"]
            assert round(c / m, 6) == 1.0
    assert led["coresident_married_deployed_ratio"] == 1.0
    # the per-seed married lower window edges and pass flags (candidate 1).
    ps1 = _candidate(1)["family_a"]["per_seed"]
    edges, mflags, cflags = [], [], []
    for s in ps1:
        me = s["per_cell"]["marital_share.married.65+|female"]
        ce = s["per_cell"]["coresident_spouse.65+|female"]
        edges.append(round(me["rate_a"] * math.exp(-me["tolerance"]), 4))
        mflags.append("T" if me["pass"] else "F")
        cflags.append("T" if ce["pass"] else "F")
    assert edges == led["married_lower_window_edges_candidate1"]
    assert (
        mflags
        == led["married_pass_flags_candidate1"]
        == ["T", "F", "T", "F", "F"]
    )
    assert (
        cflags
        == led["coresident_pass_flags_candidate1"]
        == ["T", "T", "T", "T", "F"]
    )
    assert led["coresident_clears_4_of_5_candidate1"] is True


def test_section2c_no_blanket_65plus_impossibility_coresident_inside_window():
    """MM4 guard (verification round 4954135376, finding 1): section 2c must NOT
    re-blanket '65+ cells cannot be cleared by any permitted entry-state lever' --
    that is FALSE for coresident_spouse.65+|female, which candidate 1 clears 4/5
    because the ~0.577 stationary ceiling sits INSIDE that cell's (lower) window on
    4 of 5 seeds. Bound to the cube so the false blanket cannot return AND the
    per-cell correction cannot be silently dropped."""
    s2 = _section(2)
    flat = " ".join(s2.split())
    # the false blanket and the original-blocker paraphrases must be ABSENT.
    for banned in (
        "65+ cells cannot be cleared by any permitted entry-state lever",
        "the 65+ cells cannot be cleared",
        "cannot clear the 65+ quad",
        "clear any of the 7",
    ):
        assert banned not in flat, banned
    # the per-cell correction for coresident-65+|female must be PRESENT in 2c.
    assert (
        "coresident_spouse.65+|female" in flat
        or "coresident_spouse.65+| female" in flat
    )
    assert "inside" in flat and ("4 of 5" in flat or "4/5" in flat)
    # cube truth: the deployed value (== married rbar, ratio 1.0) sits INSIDE the
    # coresident-F window on 4/5 seeds; candidate 1 clears the cell 4/5.
    ps1 = _candidate(1)["family_a"]["per_seed"]
    cores_inside, cflags = 0, []
    for s in ps1:
        ce = s["per_cell"]["coresident_spouse.65+|female"]
        me = s["per_cell"]["marital_share.married.65+|female"]
        assert round(ce["rbar"] / me["rbar"], 6) == 1.0
        cores_le = ce["rate_a"] * math.exp(-ce["tolerance"])
        cores_inside += 1 if ce["rbar"] >= cores_le else 0
        cflags.append("T" if ce["pass"] else "F")
    assert cores_inside == 4
    assert cflags == ["T", "T", "T", "T", "F"]
    assert (
        _seed_pass_counts(_candidate(1), "coresident_spouse.65+|female") == 4
    )


# --------------------------------------------------------------------------
# Finding 5: the corrected sigma ranges (2.0-6.6, not 3-8)
# --------------------------------------------------------------------------
def test_sigma_range_corrected_2_to_6p6():
    """Section 3's systematic-miss range is the TRUE per-cell 2.0-6.6 sigma across
    all six cells and both candidates (0.11-0.44 ln vs sigma 0.027-0.072), NOT the
    3-8 sigma that fit only the 18-24 pair. The sub-3sigma end is the female cells
    that sometimes pass."""
    led = _ledger()["sigma_range"]
    per = _floors()["faithful_candidate_oc"]["per_cell"]
    scores, sigmas, perseed_mult, mean_mult = [], [], [], []
    for n in (1, 2):
        ps = _candidate(n)["family_a"]["per_seed"]
        for cell in FAMILY_A_DEMOTED:
            sig = per[cell]["realized_sigma"]
            sigmas.append(sig)
            cell_scores = [s["per_cell"][cell]["score"] for s in ps]
            scores += cell_scores
            perseed_mult += [sc / sig for sc in cell_scores]
            mean_mult.append(mean(cell_scores) / sig)
    assert round(min(scores), 2) == led["ln_miss_min_round2"] == 0.11
    assert round(max(scores), 2) == led["ln_miss_max_round2"] == 0.44
    assert round(min(sigmas), 3) == led["sigma_min_round3"] == 0.027
    assert round(max(sigmas), 3) == led["sigma_max_round3"] == 0.072
    # the low end of the multiple range is 2.0 (a 65+ female cell that clears).
    assert (
        round(min(perseed_mult), 1) == led["sigma_multiple_min_round1"] == 2.0
    )
    # the stated 6.6 upper is bracketed by the mean and per-seed maxima.
    assert led["sigma_multiple_stated_max"] == 6.6
    assert max(mean_mult) <= 6.6 <= max(perseed_mult)
    # the crux survives: draw noise (mean of K=20) is far below every miss.
    assert led["draw_noise_max_round3"] == 0.007
    # the doc states 2.0-6.6, not 3-8.
    s3 = _section(3)
    assert "2.0σ–6.6σ" in s3 or "2.0" in s3 and "6.6" in s3
    assert "3–8σ" not in s3 and "(3-8σ)" not in s3


def test_section3_1_5of5_fail_list_per_cell_true_excludes_never_married():
    """MM6 guard (verification round 4954135376, finding 2): section 3.1's '5/5 for
    both candidates' whittling list must be per-cell true. marital_share.
    never_married.25-34|male is NOT 5/5-fail (candidate 1 clears seed 1, 0.1707 vs
    0.192) and must not be reintroduced into that list; the concrete named cells
    recompute as genuine 5/5-fail from BOTH cubes; and the '5/5' count itself is
    pinned so a '4/5' downgrade fails too."""
    s3 = _section(3)
    flat = " ".join(s3.split())
    assert "5/5 for both candidates" in flat  # pins the count word.
    m = re.search(r"5/5 for both candidates\*{0,2}\s*—(.+?)—", flat)
    assert m, "the 5/5-fail cell list clause was not found in section 3.1"
    clause = m.group(1)
    nm = "marital_share.never_married.25-34|male"
    assert nm not in clause  # the false membership must stay OUT of the list.
    # recompute the genuine 5/5-fail RETAINED set from both committed cubes.
    per_cell = _candidate(1)["family_a"]["per_seed"][0]["per_cell"]
    genuine = {
        c
        for c in per_cell
        if c not in FAMILY_A_DEMOTED
        and _seed_pass_counts(_candidate(1), c) == 0
        and _seed_pass_counts(_candidate(2), c) == 0
    }
    assert nm not in genuine
    assert _seed_pass_counts(_candidate(1), nm) == 1  # clears seed 1.
    assert _seed_pass_counts(_candidate(2), nm) == 0
    # the clause names exactly the genuine set (hh_size shorthand + two concretes).
    assert "hh_size_share.{1..5plus}" in clause
    for k in ("1", "2", "3", "4", "5plus"):
        assert f"hh_size_share.{k}" in genuine, k
    for cell in (
        "marital_share.married.25-34|male",
        "earnings_participation.35-44|female",
    ):
        assert cell in clause and cell in genuine, cell
    # the corrected per-cell record of never_married is stated (retained a fortiori).
    assert nm in flat and ("1/5" in flat or "single loosest seed" in flat)


# --------------------------------------------------------------------------
# Q9 / Q5 evidence (unchanged figures) still bound
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
    assert led["grading"] == "4953311492"
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


def test_q6_65plus_divorce_over_accumulation_bound():
    led = _ledger()["forensics2"]
    comp = _forensics2()["q6_marital_calibration_frame"]["composition_65plus"]
    chan = _forensics2()["q6_marital_calibration_frame"][
        "dissolution_channel_65plus"
    ]
    for sex in ("female", "male"):
        assert chan[sex]["dominant_excess_status"] == "divorced"
        assert chan[sex]["widowhood_channel_realized"] is False
    assert led["realized_65plus_channel"] == "divorce"
    assert led["widowhood_channel_realized_65plus"] is False
    df, dm = comp["female"], comp["male"]
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
    assert round(df["married"]["deployed"], 3) == 0.573
    assert round(df["married"]["frame"], 3) == 0.691
    assert round(dm["married"]["deployed"], 3) == 0.707
    assert round(dm["married"]["frame"], 3) == 0.856


def test_q6_contract_adjudication_says_65plus_unfixable_by_permitted_lever():
    det = _forensics2()["q6_marital_calibration_frame"][
        "contract_adjudication"
    ]["determination"]
    assert "CONTRACT-PERMITTED" in det
    assert "CANNOT fix the 65+ undershoot" in det
    assert "identity in disguise" in det
    s4c = _section(4)
    s4c_low = s4c.lower()
    for lever in (
        "cps-anchored entry",
        "identity",
        "re-calibration",
        "widowhood",
    ):
        assert lever in s4c_low
    # the referee's R1-R3 construction-attempts are incorporated with the source.
    for token in ("Per-band entry", "Entry-55", "Max-entry", "4953763322"):
        assert token in s4c
    assert "0.4228" in s4c and "0.4951" in s4c  # R1/R2 landed values


def test_c1_non_reversal_bound_and_runs_bit_identical():
    led = _ledger()["forensics1_q5_c1"]
    q5 = _forensics1()["q5_tail_upper_read"]
    assert led["artifact"] == "runs/gate_w1_forensics1_v1.json"
    assert led["registration"] == "4951218279"
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
    assert (
        round(q5["corrected_tail"]["ppi_savings_abs"], 4)
        == led["corrected_ppi_savings_round4"]
        == 0.0137
    )
    assert q5["c1_robustness_answer"]["answer_non_reversal_is_robust"] is True
    assert led["non_reversal_is_robust"] is True
    # finding 9ii: the two candidate fingerprint runs are BIT-IDENTICAL.
    c1run = _candidate(1)["family_c"]["fingerprints"]
    c2run = _candidate(2)["family_c"]["fingerprints"]
    assert c1run["c1"] == c2run["c1"]
    assert led["candidate_fingerprint_runs_bit_identical"] is True
    assert (
        round(c1run["c1"]["kendall_tau_vs_required"], 4)
        == led["candidate2_c1_kendall_tau_vs_required_round4"]
        == 0.6667
    )
    # the doc qualifies the C1 empirical legs as one measurement.
    assert "bit-identical" in _prose_without_ledger()


# --------------------------------------------------------------------------
# Finding 4: retained-surface candor -- Q7 REFUTED + the C2 record
# --------------------------------------------------------------------------
def test_retained_surface_not_known_passable_candor():
    led = _ledger()
    prose = _prose_without_ledger()
    assert led["surface_not_known_passable"] is True
    # the plain "not known passable" sentence.
    assert "not known passable" in prose.lower()
    # Q7 REFUTED: hh_size joint clears only size-2 and all 5 fail 5/5 both cands.
    f2 = _forensics2()
    assert (
        f2["q7_coresident_parent_fertility"]["finding"][
            "pre_registration_proves_3_4_5plus"
        ]
        is False
    )
    assert led["forensics2"]["q7_pre_registration_proves_3_4_5plus"] is False
    assert led["forensics2"]["q7_joint_cells_cleared"] == ["2"]
    assert (
        f2["q7_coresident_parent_fertility"]["scoring"]["joint"][
            "joint_cells_cleared" if False else "n_cells_clear"
        ]
        == 1
    )
    for n in (1, 2):
        ps = _candidate(n)["family_a"]["per_seed"]
        for k in ("1", "2", "3", "4", "5plus"):
            cell = f"hh_size_share.{k}"
            assert sum(1 for s in ps if s["per_cell"][cell]["pass"]) == 0, (
                n,
                cell,
            )
    assert "REFUTED" in prose
    assert "not closable by these two entry-state levers alone" in prose


def test_c2_record_bound_and_amendment3_forecast():
    led = _ledger()["c2_record"]
    # both candidate C2 runs bit-identical, deployed order + tau.
    c2a = _candidate(1)["family_c"]["fingerprints"]["c2"]
    c2b = _candidate(2)["family_c"]["fingerprints"]["c2"]
    assert c2a == c2b
    assert led["candidate_runs_bit_identical"] is True
    assert c2a["deployed_order"] == led["deployed_order"]
    assert (
        round(c2a["kendall_tau_vs_required"], 4)
        == led["kendall_tau_vs_required_round4"]
        == 0.3333
    )
    # cap_150k deployed 16.7 years vs Smith rank-4 (1 year).
    deltas = c2a["provision_deltas"]["our_exhaustion_deltas"]
    assert (
        round(deltas["cap_150k"], 1)
        == led["cap_150k_deployed_years_round1"]
        == 16.7
    )
    assert (
        c2a["provision_deltas"]["smith_year_deltas"]["cap_150k"]
        == led["cap_150k_smith_rank4_years"]
        == 1
    )
    # corrected (realistic) tail moves C2 FURTHER away: tau -1/3.
    req = ["elimination", "payroll_plus_2pp", "payroll_plus_1pp", "cap_150k"]
    ct = _forensics1()["q5_tail_upper_read"]["corrected_tail"]["c2_order"]
    pos = {x: i for i, x in enumerate(req)}
    a = [pos[x] for x in ct]
    n = len(req)
    conc = sum(1 for i in range(n) for j in range(i + 1, n) if a[i] < a[j])
    disc = sum(1 for i in range(n) for j in range(i + 1, n) if a[i] > a[j])
    tau_ct = (conc - disc) / (n * (n - 1) / 2)
    assert (
        round(tau_ct, 4)
        == led["corrected_tail_tau_vs_required_round4"]
        == -0.3333
    )
    # the amendment-3 risks are pre-named in the ledger + prose.
    risks = _ledger()["amendment3_risks"]
    assert set(risks) == {
        "hh_size_share.1",
        "hh_size_share.3",
        "hh_size_share.4",
        "hh_size_share.5plus",
        "c2_cap_150k_adjacency",
    }
    prose = _prose_without_ledger()
    assert "cap_150k" in prose and "amendment-3" in prose.lower()
    assert "forensics-then-ceremony" in prose
    # verifier note N2: the amendment-3 hh_size risks are the FOUR cells Q7 leaves
    # out of tolerance (sizes 1/3/4/5+); size-2 is the one the joint lever clears.
    # All five hh_size cells still fail 5/5 (candor test) -- only four are risks.
    assert {r for r in risks if r.startswith("hh_size")} == {
        "hh_size_share.1",
        "hh_size_share.3",
        "hh_size_share.4",
        "hh_size_share.5plus",
    }
    assert "hh_size_share.2" not in risks
    q7j = _forensics2()["q7_coresident_parent_fertility"]["scoring"]["joint"]
    assert q7j["per_cell"]["2"]["clears"] is True
    for k in ("1", "3", "4", "5plus"):
        assert q7j["per_cell"][k]["clears"] is False
    # the prose risk-framing names the four sizes and flags size-2 as the cleared one.
    assert "1/3/4/5+" in prose and "size-2" in prose


# --------------------------------------------------------------------------
# OC consequence: 0.922/0.9481 (53) AND 0.9344/0.9623 (47) -- unchanged
# --------------------------------------------------------------------------
def test_family_a_oc_recomputes_before_and_after_on_the_derivations_basis():
    led = _ledger()["family_a_oc"]
    floors = _floors()
    per = floors["faithful_candidate_oc"]["per_cell"]
    gate_eligible = floors["gate_partition"]["gate_eligible"]
    # post-flip: the live contract carries 47 gated tolerances; the 6 demoted
    # cells' tolerances are RETAINED verbatim under retained_tolerances, so
    # the 53-cell pre-amendment basis reconstructs exactly.
    live_tol = _family_a_tolerances()
    assert len(live_tol) == 47
    retained = _gate_w1()["thresholds"]["family_a"]["retained_tolerances"]
    assert set(retained) == FAMILY_A_DEMOTED
    tol = dict(live_tol)
    tol.update({c: retained[c]["tolerance"] for c in FAMILY_A_DEMOTED})
    assert len(tol) == len(gate_eligible) == 53 == led["n_gated_before"]

    p_seed_b, p_gate_b = _oc(gate_eligible, tol, per)
    assert round(p_seed_b, 4) == led["p_seed_before"] == 0.922
    assert round(p_gate_b, 4) == led["p_gate_before"] == 0.9481
    assert round(p_seed_b, 4) == floors["faithful_candidate_oc"]["p_seed_pass"]
    assert (
        round(p_gate_b, 4)
        == floors["faithful_candidate_oc"]["p_gate_pass_4_of_5"]
    )

    assert FAMILY_A_DEMOTED <= set(gate_eligible)
    surviving = [c for c in gate_eligible if c not in FAMILY_A_DEMOTED]
    assert len(surviving) == 47 == led["n_gated_after"]

    p_seed_a, p_gate_a = _oc(surviving, tol, per)
    assert round(p_seed_a, 4) == led["p_seed_after"] == 0.9344
    assert round(p_gate_a, 4) == led["p_gate_after"] == 0.9623
    assert led["invariant_under_amendment"] is False
    assert p_seed_a > p_seed_b and p_gate_a > p_gate_b

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
    prose = _prose_without_ledger()
    assert "prices sampling noise, not concept" in prose


def test_oc_statement_after_is_0p9623_times_c2():
    led = _ledger()
    assert led["oc_statement_after"] == "0.9623 * I(c2)"
    prose = _prose_without_ledger()
    assert "0.9623 × I(C2)" in prose
    assert "48-cell" in prose


# --------------------------------------------------------------------------
# Partition consequence (unchanged): 55 -> 48 gated; A 53 -> 47; C 2 -> 1
# --------------------------------------------------------------------------
def test_partition_arithmetic_bound_to_current_gates_yaml():
    led = _ledger()
    fa = _gate_w1()["thresholds"]["family_a"]
    fc = _gate_w1()["thresholds"]["family_c"]

    # post-flip: the LIVE contract carries the ledger's "after" state; the
    # ledger's "now" figures remain the recorded pre-flip state.
    assert led["family_a_partition"]["gated_now"] == 53
    assert led["family_a_partition"]["report_only_now"] == 52
    assert (
        len(_family_a_tolerances())
        == led["family_a_partition"]["gated_after"]
        == 53 - 6
        == 47
    )
    assert (
        len(fa["report_only"])
        == led["family_a_partition"]["report_only_after"]
        == 52 + 6
        == 58
    )

    part = fc["gate_partition"]
    assert led["family_c_partition"]["gated_now"] == 2
    assert led["family_c_partition"]["report_only_now"] == 0
    assert f"fingerprint.{fc['fingerprints']['c1']['id']}" == C1_CELL
    assert C1_CELL not in part["gate_eligible"]
    assert C1_CELL in fc["report_only"]
    assert (
        part["n_gate_eligible"]
        == led["family_c_partition"]["gated_after"]
        == 2 - 1
        == 1
    )
    assert (
        part["n_report_only"]
        == led["family_c_partition"]["report_only_after"]
        == 0 + 1
        == 1
    )

    ov = led["overall_partition"]
    assert ov["gated_now"] == 53 + 0 + 2 == 55
    assert ov["gated_after"] == 47 + 0 + 1 == 48
    assert ov["report_only_now"] == 77
    assert ov["report_only_after"] == 84
    assert ov["gated_after"] + 7 == ov["gated_now"]
    assert ov["report_only_after"] - 7 == ov["report_only_now"]


# --------------------------------------------------------------------------
# BIND the operative flip text (section 7): the 7 cells + 4 reasons + roll-ups
# --------------------------------------------------------------------------
def test_section7_flip_list_binds_all_seven_demoted_cells_and_reasons():
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
    # all FOUR machine reasons appear in the flip section.
    for reason in (REASON_C1, REASON_18_24, REASON_CEILING, REASON_DUPLICATE):
        assert reason in s7, reason
    # the flip maps coresident-65+|female specifically to the DUPLICATE reason.
    assert re.search(
        r"coresident_spouse\.65\+\|female`?\s*→\s*`?"
        + re.escape(REASON_DUPLICATE),
        " ".join(s7.split()),
    )


def test_section7_three_rollups_match_ledger_partition():
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
    assert ("48", "84") in rollups


def test_section7_oc_and_cube_transitions_bound():
    led = _ledger()["family_a_oc"]
    s7 = _section(7)
    assert "53 → **47**" in s7
    assert f"0.922 → **{led['p_seed_after']}**" in s7
    assert f"0.9481 → **{led['p_gate_after']}**" in s7
    assert "[20, 53, 5]" in s7 and "[20, 47, 5]" in s7


def test_section8_history_figures_bound():
    led = _ledger()
    s8 = " ".join(_section(8).split())
    oc = led["family_a_oc"]
    ov = led["overall_partition"]
    f2 = led["forensics2"]
    c1 = led["forensics1_q5_c1"]
    assert f"p_seed {oc['p_seed_after']} / p_gate {oc['p_gate_after']}" in s8
    assert "from 53 cells / 0.922 / 0.9481" in s8
    assert "55 -> 48 gated, 77 -> 84 report-only" in s8
    assert f"{ov['gated_after']}-cell" in s8
    assert f"{f2['concept_gap_18_24_pooled_pp_round1']}pp pooled" in s8
    assert f"{c1['upper_read_ppi_savings_round4']}" in s8
    assert f"{c1['upper_read_nra_savings_round4']}" in s8
    # the corrected permanent record carries the per-cell truth + candor.
    assert "coresident_spouse.65+| female 4/5" in s8 or "4/5 seeds" in s8
    assert "NOT KNOWN PASSABLE" in s8
    assert "cap_150k" in s8 and "16.7 years" in s8
    assert "divorce over-accumulation" in s8.lower()
    # the false blanket claim must NOT reappear.
    assert "no contract-consistent candidate can clear any of the 7" not in s8


# --------------------------------------------------------------------------
# No-self-rescue: BOTH candidates STAND FAIL on the amended 48-cell surface
# --------------------------------------------------------------------------
def _seeds_pass_on_47_surface(cand: dict) -> int:
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
    led = _ledger()["no_self_rescue"]
    cand = _candidate(n)
    key = f"candidate{n}"
    assert cand["run"] == led[f"{key}_run"] == f"gate_w1_candidate{n}_v1"
    assert cand["verdict"]["gate_pass"] is False
    assert led[f"{key}_gate_pass"] is False

    seeds = _seeds_pass_on_47_surface(cand)
    assert seeds == led[f"{key}_seeds_pass_47_surface"] == 0
    assert led[f"{key}_family_a_pass_47_surface"] is False

    c2 = cand["family_c"]["fingerprints"]["c2"]
    assert c2["reversed_to_anchor"] is False
    assert led[f"{key}_c2_reversed"] is False

    assert (seeds >= 4) is False
    assert led["rescue_set_empty"] is True
    assert led["over_determined_both_families"] is True
    assert led[f"{key}_pr"] == (162 if n == 1 else 167)


def test_no_self_rescue_per_cell_record_is_the_corrected_finding2_wording():
    """Finding 2: the doc must NOT say 'the demoted cells were themselves failing
    for both candidates' (false). The male cells are 0/5; the female cells
    sometimes pass. Removing sometimes-passed cells cannot turn a 0/5 into >=4/5
    because OTHER retained cells fail every seed."""
    prose = _prose_without_ledger()
    assert "were themselves failing for both candidates" not in prose
    # the male hard-fails and female sometimes-passes are stated.
    for n in (1, 2):
        cand = _candidate(n)
        for cell in (
            "earnings_participation.18-24|male",
            "marital_share.married.65+|male",
            "coresident_spouse.65+|male",
        ):
            assert _seed_pass_counts(cand, cell) == 0
        assert (
            cand["family_c"]["fingerprints"]["c1"]["reversed_to_anchor"]
            is False
        )
    # candidate 2 passes the 18-24 female cell 2/5 -- the sometimes-pass fact.
    assert (
        _seed_pass_counts(_candidate(2), "earnings_participation.18-24|female")
        == 2
    )


# --------------------------------------------------------------------------
# The proposal is a DRAFT: gates.yaml is UNTOUCHED (no flip)
# --------------------------------------------------------------------------
def test_gates_yaml_flipped_per_section7_and_moves_no_sibling():
    """Post-flip polarity of the proposal's untouched-guard: gates.yaml now
    carries section 7 exactly (47/58 family A with retained tolerances; C1
    report-only; OC 0.9344/0.9623), the PROPOSAL itself touched nothing (the
    ledger records that), and the flip moves NO sibling gate (the section-7
    subset master-compare)."""
    gw1 = _gate_w1()
    fa = gw1["thresholds"]["family_a"]
    fc = gw1["thresholds"]["family_c"]
    tol = _family_a_tolerances()
    assert FAMILY_A_DEMOTED.isdisjoint(set(tol))
    assert FAMILY_A_DEMOTED <= set(fa["report_only"])
    assert set(fa["retained_tolerances"]) == FAMILY_A_DEMOTED
    assert C1_CELL not in fc["gate_partition"]["gate_eligible"]
    assert C1_CELL in fc["report_only"]
    assert fc["gate_partition"]["n_gate_eligible"] == 1
    assert fa["faithful_candidate_oc"]["n_gated_cells"] == 47
    assert fa["faithful_candidate_oc"]["p_gate_pass_4_of_5"] == 0.9623
    # zero threshold movement: retained == the ledger's demoted tolerances.
    assert {
        c: r["tolerance"] for c, r in fa["retained_tolerances"].items()
    } == {
        "earnings_participation.18-24|female": 0.211,
        "earnings_participation.18-24|male": 0.221,
        "marital_share.married.65+|female": 0.163,
        "marital_share.married.65+|male": 0.084,
        "coresident_spouse.65+|female": 0.168,
        "coresident_spouse.65+|male": 0.094,
    }
    assert _ledger()["gates_yaml_untouched_by_this_proposal"] is True

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
    # the section-7 subset master-compare: no locked sibling moves. Stated
    # master-side so it TOLERATES ADDITIONS -- a later locked-gate flip (the
    # gate_m6 lock, 2026-07-13) adds gate_m6, absent from master until that PR
    # merges; that is an addition, not a sibling move. Every gate on master is
    # still present-and-equal in live (except gate_w1, this flip's own gate);
    # no master gate is dropped.
    assert set(master_doc) <= set(live_doc)
    for name in sorted(master_doc):
        if name == "gate_w1":
            continue
        assert live_doc[name] == master_doc[name], name
