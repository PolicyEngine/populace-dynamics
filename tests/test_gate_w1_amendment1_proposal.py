"""Proposal-consistency bindings for gate_w1 amendment 1 (family-B DI bands +
conversion margins).

The amendment-1 PROPOSAL is a DRAFT: it argues, from the W1 forensics 1 Q4
evidence base (the 8 DI age-composition prevalence bands) and the adversarial
referee round (PR #164 comment 4951701300, the 2 disability-conversion margins),
that ALL 10 family-B gated cells are unclearable by any contract-consistent
candidate and must be demoted (option a) rather than re-anchored now (option b).
This module proves the proposal document does not drift from the committed
evidence: it parses the machine-readable ``amendment-consistency-ledger`` block
embedded in ``docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`` and
cross-checks EVERY load-bearing figure against the frozen artifacts --
``runs/gate_w1_forensics1_v1.json`` (Q4), ``runs/gate_w1_floors_v1.json`` (the
family-A OC machinery + partition), ``runs/gate_w1_candidate1_v1.json`` (the
committed FAIL the amendment must not rescue), ``runs/m4_disability_v1.json``
(the conversion ``conversion_validation`` evidence), and ``gates.yaml`` (still
the LOCKED 10-cell surface -- the proposal moves no threshold).

It ALSO binds the OPERATIVE flip text (referee fix B): the section-7 enumerated
demotion list, the section-7 partition roll-up, and the section-8 history entry
figures are parsed from the doc and cross-checked against the ledger/artifacts so
they cannot silently drift, and the machine-reason prose assertion is checked
OUTSIDE the ledger fence (de-vacuoused).

ALWAYS-RUNNABLE: reads only committed ``runs/*.json`` + the doc + gates.yaml, no
data load, no h5, no PSID/PE-US checkout. Reproduces the draw-noise-free
half-normal OC exactly as tests/test_gate_w1_derivations.py does, so the
proposal's OC-invariance claim is bound to the same recomputation the lock used.
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
DOC = ROOT / "docs" / "amendments" / "gate_w1_amendment_1_family_b_di_bands.md"
FORENSICS = ROOT / "runs" / "gate_w1_forensics1_v1.json"
FLOORS = ROOT / "runs" / "gate_w1_floors_v1.json"
CANDIDATE1 = ROOT / "runs" / "gate_w1_candidate1_v1.json"
M4 = ROOT / "runs" / "m4_disability_v1.json"
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
CONVERSION_CELLS = (
    "claim_age.disability_conversion|female",
    "claim_age.disability_conversion|male",
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


def _prose_without_ledger() -> str:
    """Doc text with the fenced ledger JSON removed.

    Used to de-vacuous prose assertions: a string that lives only inside the
    ledger block must NOT count as ``in`` the prose.
    """
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


def _forensics() -> dict:
    return json.loads(FORENSICS.read_text())


def _floors() -> dict:
    return json.loads(FLOORS.read_text())


def _candidate1() -> dict:
    return json.loads(CANDIDATE1.read_text())


def _m4_conversion_validation() -> dict:
    return json.loads(M4.read_text())["conversion_validation"]


def _gate_w1() -> dict:
    return yaml.safe_load(GATES.read_text())["gates"]["gate_w1"]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# The ledger is well formed and names the recommendation + machine reasons
# --------------------------------------------------------------------------
def test_ledger_parses_and_headline_matches_prose():
    led = _ledger()
    assert led["amendment_id"] == "2026-07-12-w1-family-b-di-bands"
    assert led["recommendation"] == "option_a_demote_all_family_b"
    assert led["machine_reason"] == "concept_bridge_undefined_di_stock"
    assert led["conversion_machine_reasons"] == [
        "concept_bridge_undefined_di_stock",
        "conversion_level_match_never_certified",
    ]
    # de-vacuoused: the recommendation + demote language appear in the PROSE
    # (ledger fence stripped), so a ledger/prose divergence is caught.
    prose = _prose_without_ledger()
    assert "RECOMMENDED" in prose
    assert "option (a)" in prose
    assert led["gates_yaml_untouched_by_this_proposal"] is True


def test_machine_reasons_named_in_prose_outside_ledger():
    """Referee fix B (de-vacuous): the ledger block held these strings, so the
    old ``in text`` assertion could not fail while the ledger held them. Check
    them in the prose with the ledger fence removed."""
    prose = _prose_without_ledger()
    # sanity: the ledger JSON really was stripped.
    assert '"machine_reason"' not in prose
    assert "concept_bridge_undefined_di_stock" in prose
    assert "conversion_level_match_never_certified" in prose


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


def test_q4_per_band_concept_share_range_bound():
    """Referee fix C: 0.595 is the AGGREGATE duration-concept share
    Sum|dur|/(Sum|dur|+Sum|shape|); per band it spans 0.023 (40-44, nearly all
    frozen-M4 shape) to 0.891 (60-fra, nearly all duration concept). The doc
    must state 0.595 as the aggregate with that per-band range."""
    led = _ledger()["forensics"]
    pb = _forensics()["q4_di_level_bridge"]["per_band"]
    shares: dict = {}
    s_dur = s_shape = 0.0
    for band in DI_BANDS:
        dur = abs(pb[band]["duration_concept_flow_to_stock"])
        shape = abs(pb[band]["m4_shape_component_flow_minus_deployed"])
        shares[band] = dur / (dur + shape)
        s_dur += dur
        s_shape += shape
    aggregate = s_dur / (s_dur + s_shape)
    assert round(aggregate, 3) == led["concept_delta_dominant_share_round3"]
    lo_band = min(shares, key=shares.get)
    hi_band = max(shares, key=shares.get)
    assert lo_band == led["concept_share_min_band"] == "40-44"
    assert hi_band == led["concept_share_max_band"] == "60-fra"
    assert (
        round(shares[lo_band], 3) == led["concept_share_min_round3"] == 0.023
    )
    assert (
        round(shares[hi_band], 3) == led["concept_share_max_round3"] == 0.891
    )


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
# Conversion cells (referee fix A): the committed grounds for demoting them
# --------------------------------------------------------------------------
def test_conversion_cells_committed_grounds_bound():
    """The 2 conversion cells fail the SAME candidate-independence test as the
    bands, on committed evidence: (i) the M4 evidence base pre-ruled the concept
    a level mismatch (ratio 0.267/0.322 from raw components, "never a level
    match"); (ii) the deployed numerator is the point-prevalence 5.79/6.55,
    constant on 60-66; (iii) the required pass-window floor 12.21/12.59 =
    anchor(14.1/14.5) - tolerance(1.89/1.91); (iv) c1's committed dev 8.30/8.07
    exceeds tolerance."""
    led = _ledger()["conversion_cells"]
    cv = _m4_conversion_validation()
    by_sex = cv["by_sex"]

    # (i) ratio from RAW components (matches the referee's 0.267/0.322; the
    # stored pre-rounded 0.3225 would mislead to 0.323).
    raw_f = (
        by_sex["female"]["psid_conversion_analog_pct"]
        / by_sex["female"]["admin_6b51_conversion_share_pct"]["mean_1998_2022"]
    )
    raw_m = (
        by_sex["male"]["psid_conversion_analog_pct"]
        / by_sex["male"]["admin_6b51_conversion_share_pct"]["mean_1998_2022"]
    )
    assert round(raw_f, 3) == led["ratio_psid_analog_to_admin_female_round3"]
    assert round(raw_m, 3) == led["ratio_psid_analog_to_admin_male_round3"]
    assert led["ratio_psid_analog_to_admin_female_round3"] == 0.267
    assert led["ratio_psid_analog_to_admin_male_round3"] == 0.322

    # "never a level match" -- the frozen adjudication the gate contradicts.
    assert "never a level match" in cv["interpretation"]
    assert led["never_a_level_match"] is True

    # (ii) deployed numerator = the point-prevalence, constant on 60-66.
    assert (
        by_sex["female"]["psid_disabled_prevalence_60_66_pct"]
        == led["prevalence_60_66_female_pp"]
        == 5.79
    )
    assert (
        round(by_sex["male"]["psid_disabled_prevalence_60_66_pct"], 2)
        == led["prevalence_60_66_male_pp_round2"]
        == 6.55
    )

    # (iii)+(iv) anchor / tolerance / window floor / dev from candidate 1.
    per_cell = _candidate1()["family_b"]["per_cell"]
    female = per_cell["claim_age.disability_conversion|female"]
    male = per_cell["claim_age.disability_conversion|male"]
    assert female["anchor_pp"] == led["anchor_female_pp"] == 14.1
    assert male["anchor_pp"] == led["anchor_male_pp"] == 14.5
    assert female["tolerance_pp"] == led["tolerance_female_pp"] == 1.89
    assert male["tolerance_pp"] == led["tolerance_male_pp"] == 1.91
    assert (
        round(female["anchor_pp"] - female["tolerance_pp"], 2)
        == led["required_window_floor_female_pp_round2"]
        == 12.21
    )
    assert (
        round(male["anchor_pp"] - male["tolerance_pp"], 2)
        == led["required_window_floor_male_pp_round2"]
        == 12.59
    )
    assert (
        round(female["abs_dev_pp"], 2)
        == led["candidate1_abs_dev_female_pp_round2"]
        == 8.3
    )
    assert (
        round(male["abs_dev_pp"], 2)
        == led["candidate1_abs_dev_male_pp_round2"]
        == 8.07
    )
    assert female["pass"] is False and male["pass"] is False
    assert led["candidate1_conversion_cells_fail"] is True


# --------------------------------------------------------------------------
# OC consequence: family-A OC recomputes to 0.922 / 0.9481 and is INVARIANT
# --------------------------------------------------------------------------
def test_family_a_oc_recomputes_and_is_invariant_to_demotion():
    """The doc's OC claim: p_seed 0.922 / p_gate 0.9481 recomputes from the 53
    locked family-A tolerances + the frozen floor sigmas (draw-noise-free
    half-normal), and NONE of the 10 family-B cells is in the family-A OC
    machinery, so demoting all of them cannot move it."""
    led = _ledger()["family_a_oc"]
    floors = _floors()
    per = floors["faithful_candidate_oc"]["per_cell"]
    gate_eligible = floors["gate_partition"]["gate_eligible"]

    fa = _gate_w1()["thresholds"]["family_a"]
    tolerances: dict = {}
    for view in fa["views"].values():
        tolerances.update(view["tolerances"])
    # amendment 2 (2026-07-12-w1-family-a-concept-cells) later demoted 6
    # family-A cells; their tolerances are RETAINED verbatim under
    # retained_tolerances, so the amendment-1-era 53-cell basis this ledger
    # binds reconstructs exactly (zero threshold movement).
    tolerances.update(
        {c: r["tolerance"] for c, r in fa["retained_tolerances"].items()}
    )
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

    # INVARIANCE: all 10 family-B cells (8 DI bands + 2 conversion) are absent
    # from the family-A OC per-cell machinery, so demotion leaves it identical.
    family_b_cells = {f"di_prevalence.{b}" for b in DI_BANDS} | set(
        CONVERSION_CELLS
    )
    assert family_b_cells.isdisjoint(per)
    assert family_b_cells.isdisjoint(gate_eligible)
    assert led["invariant_under_amendment"] is True


# --------------------------------------------------------------------------
# Partition consequence: family B 10 -> 0 gated; overall 65 -> 55 gated
# --------------------------------------------------------------------------
def test_family_b_partition_after_the_flip_realizes_option_a():
    """POST-RATIFICATION (this flip PR realized option a): family B now gates
    NOTHING and report-only is 25. The ledger's gated_now / report_only_now
    (10 / 15) describe the PRE-amendment surface the flip demoted; the flipped
    gates.yaml matches the ledger's after-values."""
    led = _ledger()["family_b_partition"]
    fb = _gate_w1()["thresholds"]["family_b"]
    gated = list(fb["gated_cells"])
    report_only = list(fb["report_only"])

    # the flip demoted all 10: family B gates nothing.
    assert gated == []
    assert led["gated_after_option_a"] == 0
    # the 10 demoted cells the ledger recorded as gated_now are now keyed in
    # report_reasons (the parallel machine-reason mapping the flip added).
    demoted = set(fb["report_reasons"])
    conv = [c for c in demoted if "disability_conversion" in c]
    di = [c for c in demoted if c.startswith("di_prevalence.")]
    assert len(demoted) == led["gated_now"] == 10
    assert len(conv) == led["conversion"] == 2
    assert len(di) == led["di_bands"] == 8
    assert set(c.split(".", 1)[1] for c in di) == set(DI_BANDS)
    assert set(conv) == set(CONVERSION_CELLS)
    # report-only grew 15 -> 25 and now contains the 10 demoted cells.
    assert len(report_only) == led["report_only_after_option_a"] == 25
    assert demoted <= set(report_only)
    assert led["report_only_now"] == 15
    assert led["report_only_now"] + len(demoted) == 25


def test_overall_partition_65_to_55_gated():
    led = _ledger()["overall_partition"]
    # 53 family-A + 10 family-B + 2 family-C == 65 gated; option (a) -> 55.
    assert led["gated_now"] == 53 + 10 + 2 == 65
    assert led["gated_after_option_a"] == 53 + 0 + 2 == 55
    # 52 family-A + 15 family-B report-only == 67; option (a) -> 77.
    assert led["report_only_now"] == 52 + 15 == 67
    assert led["report_only_after_option_a"] == 52 + 25 == 77


def test_oc_statement_after_option_a_is_0p9481_times_family_c():
    """After demoting all 10, family B contributes nothing gated, so the honest
    overall OC is 0.9481 x I(family C) -- stated in the prose, not only the
    ledger."""
    led = _ledger()
    assert led["oc_statement_after_option_a"] == "0.9481 * I(family_c)"
    prose = _prose_without_ledger()
    assert "0.9481 × I(family C)" in prose
    assert "55-cell" in prose  # the residual gated surface size


# --------------------------------------------------------------------------
# Referee fix B: BIND the operative flip text (section 7) + history (section 8)
# --------------------------------------------------------------------------
def test_section7_flip_list_binds_all_ten_family_b_cells():
    """The section-7 enumerated demotion list must name EXACTLY the 10 locked
    family-B gated cells. Deleting any key (the referee's '55-59 deleted still
    passes' mutation) now fails: the parsed set diverges from gates.yaml."""
    section7 = _section(7)
    bands = set(re.findall(r"di_prevalence\.([\w-]+)", section7))
    convs = set(re.findall(r"disability_conversion\|(\w+)", section7))
    parsed = {f"di_prevalence.{b}" for b in bands} | {
        f"claim_age.disability_conversion|{s}" for s in convs
    }
    fb = _gate_w1()["thresholds"]["family_b"]
    demoted = set(fb["report_reasons"])  # the 10 cells the flip demoted
    assert fb["gated_cells"] == {}  # post-ratification gated_cells is empty
    assert len(demoted) == 10
    assert bands == set(DI_BANDS)
    assert convs == {"female", "male"}
    assert parsed == demoted  # section 7 named exactly the 10 demoted cells
    assert _ledger()["family_b_partition"]["gated_after_option_a"] == 0


def test_section7_rollup_matches_ledger_partition():
    """The bolded section-7 roll-up '55 gated / 77 report-only' is parsed and
    cross-checked against the ledger. The referee's '57/75 -> 58/74 still
    passes' mutation now fails."""
    led = _ledger()["overall_partition"]
    section7 = _section(7)
    rollups = re.findall(r"\*\*(\d+) gated / (\d+) report-only\*\*", section7)
    assert rollups == [
        (
            str(led["gated_after_option_a"]),
            str(led["report_only_after_option_a"]),
        )
    ]
    assert rollups == [("55", "77")]


def test_section8_history_figures_bound():
    """The section-8 permanent history entry's figures are parsed and bound to
    the ledger. The referee's 'p_gate 0.9481 -> 0.9482 still passes' mutation
    now fails."""
    led = _ledger()
    # the history entry is a folded YAML scalar, so collapse whitespace runs
    # (newline + indent) before substring-matching wrapped phrases.
    section8 = " ".join(_section(8).split())
    part = led["overall_partition"]
    conv = led["conversion_cells"]
    # p_gate carried verbatim, bound to the LABELLED occurrence so a drift of
    # the reported p_gate to 0.9482 fails even though 0.9481 also appears in the
    # OC-statement clause.
    assert f"p_gate {led['family_a_oc']['p_gate']}" in section8
    # the exact partition transition, built from the ledger.
    transition = (
        f"{part['gated_now']} -> {part['gated_after_option_a']} gated, "
        f"{part['report_only_now']} -> "
        f"{part['report_only_after_option_a']} report-only"
    )
    assert transition in section8
    assert "65 -> 55 gated, 67 -> 77 report-only" in section8
    # residual surface size + the conversion grounds in the record.
    assert f"{part['gated_after_option_a']}-cell" in section8
    assert (
        f"{conv['ratio_psid_analog_to_admin_female_round3']}/"
        f"{conv['ratio_psid_analog_to_admin_male_round3']}"
    ) in section8
    assert "0.267/0.322" in section8


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


def test_no_self_rescue_all_ten_family_b_cells_fail_for_candidate1():
    """Demoting all 10 family-B cells rescues nothing: candidate 1 fails every
    one of them, AND fails family A and family C independently of the
    (family-B-only) amendment."""
    led = _ledger()["no_self_rescue"]
    c1 = _candidate1()
    per_cell = c1["family_b"]["per_cell"]
    conv = [c for c in per_cell if "disability_conversion" in c]
    di = [c for c in per_cell if c.startswith("di_prevalence.")]
    assert len(conv) == 2
    assert len(di) == 8
    for cell in conv + di:
        assert per_cell[cell]["pass"] is False, cell
    # the conversion pair in particular exceeds tolerance -- the "retained
    # cell" the referee overturned.
    for cell in conv:
        row = per_cell[cell]
        assert row["abs_dev_pp"] > row["tolerance_pp"], cell
    assert led["all_ten_family_b_cells_fail_for_candidate1"] is True
    # family A and family C fail independently of the amendment.
    v = c1["verdict"]
    assert v["family_a_pass"] is False and v["family_c_pass"] is False
    assert led["fails_family_a_and_c_independently_of_amendment"] is True
    assert led["candidate1_pr"] == 162


# --------------------------------------------------------------------------
# The proposal is a DRAFT: gates.yaml is untouched (no flip)
# --------------------------------------------------------------------------
def test_flip_demotes_all_ten_family_b_cells_vs_master():
    """Polarity flip (gate-2c floors-PR precedent, #130): the PROPOSAL PR guard
    asserted gate_w1 was byte-identical to origin/master and family B still
    gated 10; THIS flip PR inverts it -- family B now gates 0 (all 10 demoted to
    report-only) with ZERO THRESHOLD MOVEMENT on families A and C. Amendment 2
    (2026-07-12-w1-family-a-concept-cells) later RESCOPED six family-A cells
    and the C1 fingerprint to report-only, so byte identity to master no
    longer holds while that flip PR is open; the invariant that always holds
    is that no A/C VALUE moved -- every gated or retained tolerance equals
    master's value for that cell, and the fingerprint anchors/orders are
    unchanged. Robust to the eventual merge on both amendments."""
    c_fb = _gate_w1()["thresholds"]["family_b"]
    # ABSOLUTE post-flip state: family B gates nothing; 10 demoted with reasons.
    assert c_fb["gated_cells"] == {}
    assert len(c_fb["report_reasons"]) == 10
    assert len(c_fb["retained_anchors"]) == 10
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
    m_fb = master["thresholds"]["family_b"]
    # pre-merge master gates the full 10 (the surface the flip demotes);
    # post-merge it is already 0. Either way current is 0.
    assert len(m_fb["gated_cells"]) in (0, 10)
    if len(m_fb["gated_cells"]) == 10:
        assert current != master  # the flip edited gate_w1
        assert set(m_fb["gated_cells"]) == set(c_fb["report_reasons"])

    # family A / family C: ZERO THRESHOLD MOVEMENT vs master, always — no
    # gated-or-retained value moves, whichever side of either amendment flip
    # master sits on.
    def _view_tols(fa):
        out = {}
        for view in fa["views"].values():
            out.update(view["tolerances"])
        return out

    m_fa = master["thresholds"]["family_a"]
    c_fa = current["thresholds"]["family_a"]
    m_vals = _view_tols(m_fa)
    m_vals.update(
        {
            c: r["tolerance"]
            for c, r in m_fa.get("retained_tolerances", {}).items()
        }
    )
    for cell, tol in _view_tols(c_fa).items():
        assert m_vals.get(cell) == tol, cell
    for cell, r in c_fa.get("retained_tolerances", {}).items():
        assert m_vals.get(cell) == r["tolerance"], cell
    m_fc = master["thresholds"]["family_c"]
    c_fc = current["thresholds"]["family_c"]
    for cid in ("c1", "c2"):
        for field in (
            "anchor_values",
            "required_representative_order",
            "psid_frame_order",
            "swap_pair",
        ):
            assert (
                c_fc["fingerprints"][cid][field]
                == m_fc["fingerprints"][cid][field]
            ), (cid, field)
