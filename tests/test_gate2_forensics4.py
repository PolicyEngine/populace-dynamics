"""Tests for gate-2 forensics 4 (reported, not gated).

REPORTED-NOT-GATED. The consistency tests read only the forensics-4 artifact
(``runs/gate2_forensics4_v1.json``), the committed candidate-15 gate artifact
(``runs/gate2_hazard_v15.json``) and ``gates.yaml``; they never rerun the
diagnostic and need no PSID, so they run in CI. They audit that every headline
recomputes from the stored per-seed values: the Q8 completed_fertility.c1970s
parity-margin decomposition and its exact three-term split of seed 2's
published side-A score (systematic deficit + high-side reference draw + sim
draw), and the Q9 reachable-stock ledger (its four-bucket partition of the 75+
widowed stock, the inflow x exposure x yield per-onset-band table, the survival
curves, and the support-window-truncation yield attribution), plus the
published-outer context match.

One reproduction pin (``test_pin_*``) rebuilds the train-side inputs live and
matches seed 0's reference parity, reference ledger and its first candidate-15
simulation draw to float precision (skipped when the PSID marriage-history
files are absent).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_forensics4_v1.json"
CANDIDATE15 = ROOT / "runs" / "gate2_hazard_v15.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2_forensics4.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4928761676")
    assert a["registration_pointer"] == "4928761676"
    assert "candidate 15" in a["candidate_under_diagnosis"]
    for block in (
        "question_8_completed_fertility_c1970s",
        "question_9_reachable_stock_ledger",
    ):
        assert block in a


def test_protocol_is_train_side_only():
    p = _artifact()["protocol"]
    assert p["train_side_only"] is True
    assert "never" in p["outer_holdout_contact"].lower()
    assert p["gate_seeds"] == [0, 1, 2, 3, 4]
    assert p["n_draws"] == 20
    assert "5200" in p["draw_rng_rule"]
    # The Q9 boundary is the observed PSID support window.
    assert "first_wave" in p["q9_support_boundary"]
    # Reuses the forensics-1 and forensics-3 machinery.
    assert "gate2_forensics.py" in p["q8_functions_reused"]
    assert "gate2_forensics3.py" in p["q9_functions_reused"]


def test_no_gate_verdict_written_and_lock_untouched():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2_pass" not in a
    assert a["revision_pins"]["gates_yaml_locked"] is True


def test_no_nan_tokens_in_artifact():
    # NaN is not valid strict JSON; the runner emits null (None) instead.
    assert "NaN" not in ARTIFACT.read_text()


def test_candidate_16_registers_only_on_this_evidence():
    a = _artifact()
    imp = a["candidate_16_implications"]
    assert "candidate 16" in imp
    assert "one-shot" in imp.lower()
    # It leads with the Q8 classification and the Q9 support-truncation driver.
    assert "SPLIT ARTIFACT" in imp
    assert "SUPPORT-WINDOW TRUNCATION" in imp


# --------------------------------------------------------------------------
# Question 8 -- completed_fertility.c1970s, seed-resolved
# --------------------------------------------------------------------------
def test_q8_three_margin_identity_recomputes():
    q8 = _artifact()["question_8_completed_fertility_c1970s"]
    for b in q8["per_seed"]:
        ref_m = b["reference_margins_sideB"]
        sim_m = b["simulated_margins_sideB"]
        # The three margins sum to the mean parity (exact additive identity).
        assert sum(ref_m.values()) == pytest.approx(
            b["reference_mean_parity_sideB"], abs=1e-9
        )
        assert sum(sim_m.values()) == pytest.approx(
            b["simulated_mean_parity_sideB"], abs=1e-9
        )
        # The margin gap is ref minus sim per margin.
        for m in ("0_to_1", "1_to_2", "2_to_3plus"):
            assert b["margin_gap_sideB"][m] == pytest.approx(
                ref_m[m] - sim_m[m], abs=1e-9
            )


def test_q8_seed2_score_splits_exactly_into_three_log_terms():
    q8 = _artifact()["question_8_completed_fertility_c1970s"]
    s2 = next(b for b in q8["per_seed"] if b["seed"] == 2)
    sa = s2["score_split_attribution"]
    # ln(rate_a/rbar) == high_ref + systematic + sim_draw, reconstructing the
    # published side-A gate score exactly.
    assert sa["reconstructed_score"] == pytest.approx(
        s2["published_sideA"]["score"], abs=1e-6
    )
    total = (
        sa["high_side_reference_draw"]
        + sa["systematic_sim_deficit"]
        + sa["seed_sim_draw"]
    )
    assert abs(total) == pytest.approx(
        s2["published_sideA"]["score"], abs=1e-6
    )


def test_q8_seed2_is_the_only_failing_seed():
    q8 = _artifact()["question_8_completed_fertility_c1970s"]
    failing = [
        b["seed"]
        for b in q8["per_seed"]
        if b["published_sideA"]["pass"] is False
    ]
    assert failing == [2]
    # Its excess over tolerance is the registered +0.011 need.
    s2 = next(b for b in q8["per_seed"] if b["seed"] == 2)
    assert s2["published_sideA"]["excess_over_tolerance"] == pytest.approx(
        0.011, abs=0.001
    )


def test_q8_seed2_classification_is_both():
    q8 = _artifact()["question_8_completed_fertility_c1970s"]
    assert q8["seed2_classification"] == "both"
    s2 = next(b for b in q8["per_seed"] if b["seed"] == 2)
    sa = s2["score_split_attribution"]
    # "Both" means seed 2 is the high-reference-draw MAX and the low-sim MIN,
    # while the systematic deficit alone is inside tolerance.
    assert sa["rate_a_is_max_over_seeds"] is True
    assert sa["rbar_is_min_over_seeds"] is True
    assert (
        abs(sa["systematic_sim_deficit"]) < s2["published_sideA"]["tolerance"]
    )
    # Either split excursion regressing to typical clears the cell.
    assert (
        sa["counterfactual_ref_to_full_panel_score"]
        < s2["published_sideA"]["tolerance"]
    )
    assert (
        sa["counterfactual_sim_to_seed_mean_score"]
        < s2["published_sideA"]["tolerance"]
    )


def test_q8_deficit_is_progression_rates_not_cohort_exposure():
    q8 = _artifact()["question_8_completed_fertility_c1970s"]
    txt = q8["cohort_exposure_vs_progression"]
    assert "PARITY-PROGRESSION RATES" in txt
    # The completed-women denominator is identical sim vs ref within a split.
    for s in _artifact()["per_seed"]:
        assert s["ref_censoring"]["shared"] is True


def test_q8_margin_gap_is_broad_none_dominant():
    q8 = _artifact()["question_8_completed_fertility_c1970s"]
    gap = q8["seed_mean_sideB_margins"]["margin_gap"]
    # All three margins under-produce (broad), and no single margin carries a
    # majority of the mean gap.
    assert all(v > 0 for v in gap.values())
    total = sum(gap.values())
    assert max(gap.values()) / total < 0.6


# --------------------------------------------------------------------------
# Question 9 -- the reachable-stock ledger
# --------------------------------------------------------------------------
def test_q9_ledger_identity_per_seed():
    for s in _artifact()["per_seed"]:
        led = s["ref_ledger"]
        # Four buckets partition the full 75+ widowed stock.
        assert led["reconciliation_remainder"] == pytest.approx(0.0, abs=1e-2)
        # The banded inflow x yield reconstructs the reachable weight.
        assert led["ledger_reachable_from_bands"] == pytest.approx(
            led["W_reachable"], rel=1e-6
        )


def test_q9_stock_ratio_matches_grading():
    q9 = _artifact()["question_9_reachable_stock_ledger"]
    # The seed-mean sim/ref stock ratio is the ~0.84 the candidate-15 grading
    # reported ("the stock moved only 0.838->0.841").
    assert q9["stock_share"]["sim_over_ref"] == pytest.approx(0.84, abs=0.02)
    # The systematic stock deficit is the ~17% leak in |ln| units.
    assert q9["systematic_stock_deficit_abs_ln"] == pytest.approx(
        0.17, abs=0.02
    )


def test_q9_dominant_term_is_yield_via_support_truncation():
    q9 = _artifact()["question_9_reachable_stock_ledger"]
    assert q9["dominant_term"] == "survival_to_75plus_yield"
    assert q9["dominant_term_within_yield"] == "support_window_truncation"
    sub = q9["yield_sub_attribution"]
    # Survival-in-widowhood tracks (not over-remarriage): the 50-64 RMST gap is
    # far milder than the registered -1.0y candidate.
    assert abs(sub["survival_in_widowhood_sim_minus_ref_rmst_50_64"]) < 1.0
    # The reachable 50-64-onset window reaches age 75 much less often in sim.
    assert (
        sub["window_reaches_75_sim_50_64"] < sub["window_reaches_75_ref_50_64"]
    )


def test_q9_inflow_at_or_above_reference_and_carried_over_produced():
    q9 = _artifact()["question_9_reachable_stock_ledger"]
    bands = q9["ledger_by_onset_band"]
    # Candidate 15's trend removal lifted incidence: the elderly inflow rate is
    # at or above reference, so inflow is NOT the leak.
    for b in ("65-74", "75+"):
        assert bands[b]["inflow_rate_b"]["sim_over_ref"] >= 0.95
    # The carried mass is OVER-produced (an offset, not a leak).
    assert q9["buckets_weight"]["W_carried"]["sim_over_ref"] > 1.0


def test_q9_yield_leaks_and_reachable_is_the_shortfall():
    q9 = _artifact()["question_9_reachable_stock_ledger"]
    # The reachable stock under-produces while the total is ~0.84; the yield
    # (not inflow) carries it.
    assert q9["buckets_weight"]["W_reachable"]["sim_over_ref"] < 0.85
    for b in ("50-64", "65-74"):
        assert q9["ledger_by_onset_band"][b]["yield_b"]["sim_over_ref"] < 0.9


def test_q9_seed3_sizing_over_clears_the_need():
    q9 = _artifact()["question_9_reachable_stock_ledger"]
    sz = q9["seed3_sizing"]
    # Seed 3's published stock failure needs +0.023.
    assert sz["need"] == pytest.approx(0.023, abs=0.001)
    # Closing the mid-age-onset yield reduces the |ln| distance by far more.
    assert sz["abs_ln_reduction"] == pytest.approx(
        sz["sideB_abs_ln"] - sz["counterfactual_abs_ln_yield_fix"], abs=1e-9
    )
    assert sz["abs_ln_reduction"] > sz["need"]


def test_q9_survival_curves_are_valid_km():
    surv = _artifact()["question_9_reachable_stock_ledger"][
        "survival_in_widowhood_female"
    ]
    for label in ("50-64", "65+"):
        block = surv[label]
        for side in ("reference_survival_curve", "simulated_survival_curve"):
            curve = block[side]
            assert curve[0] == pytest.approx(1.0, abs=1e-12)
            for i in range(1, len(curve)):
                assert curve[i] <= curve[i - 1] + 1e-12
        assert block["sim_minus_ref_rmst"] == pytest.approx(
            block["simulated_restricted_mean_survival_years"]
            - block["reference_restricted_mean_survival_years"],
            abs=1e-9,
        )


def test_q9_band_ratios_recompute():
    bands = _artifact()["question_9_reachable_stock_ledger"][
        "ledger_by_onset_band"
    ]
    for row in bands.values():
        for factor in row.values():
            if factor["reference"] > 0 and factor["sim_over_ref"] is not None:
                assert factor["sim_over_ref"] == pytest.approx(
                    factor["simulated"] / factor["reference"], rel=1e-9
                )


def test_q9_sim_share_is_mean_over_20_draws():
    for s in _artifact()["per_seed"]:
        draws = s["per_draw_share_widowed_75plus"]
        assert len(draws) == 20
        assert s["sim_ledger_mean"]["share_widowed_75plus"] == pytest.approx(
            float(np.mean(draws)), abs=1e-9
        )


# --------------------------------------------------------------------------
# Published-outer context (candidate 15, never re-simulated)
# --------------------------------------------------------------------------
def test_published_outer_matches_committed_candidate15():
    committed = json.loads(CANDIDATE15.read_text())
    by_seed = {s["seed"]: s for s in committed["per_seed"]}
    ctx = _artifact()["published_outer_context"]
    for cell in ("completed_fertility.c1970s", "share_widowed.75+|female"):
        for seed_str, rec in ctx[cell].items():
            gc = by_seed[int(seed_str)]["gated_cells"][cell]
            assert rec["rate_a"] == pytest.approx(gc["rate_a"], abs=1e-9)
            assert rec["rbar"] == pytest.approx(gc["rbar"], abs=1e-9)
            assert rec["score"] == pytest.approx(gc["score"], abs=1e-9)
            assert rec["pass"] == gc["pass"]
    # Both failing seeds {2, 3} clip the stock; only seed 2 clips fertility.
    stock = ctx["share_widowed.75+|female"]
    fert = ctx["completed_fertility.c1970s"]
    assert {s for s, r in stock.items() if r["pass"] is False} == {"2", "3"}
    assert {s for s, r in fert.items() if r["pass"] is False} == {"2"}


def test_q8_per_seed_published_matches_committed():
    committed = json.loads(CANDIDATE15.read_text())
    by_seed = {s["seed"]: s for s in committed["per_seed"]}
    for b in _artifact()["question_8_completed_fertility_c1970s"]["per_seed"]:
        gc = by_seed[b["seed"]]["gated_cells"]["completed_fertility.c1970s"]
        assert b["published_sideA"]["rbar"] == pytest.approx(
            gc["rbar"], abs=1e-9
        )
        assert b["published_sideA"]["rate_a"] == pytest.approx(
            gc["rate_a"], abs=1e-9
        )


# --------------------------------------------------------------------------
# Reproduction pin (needs the staged PSID marriage-history files)
# --------------------------------------------------------------------------
@needs_psid
def test_pin_seed0_reference_and_first_draw_reproduce():
    """Seed 0's reference parity, reference ledger and first draw reproduce.

    Rebuilds the panel, computes the seed-0 train-half reference parity and
    reachable-stock ledger, fits candidate 15 and matches the recorded values
    and the first simulation draw (5200) to float precision -- the train-side
    pipeline pinned bit-for-bit.
    """
    sys.path.insert(0, str(SCRIPTS))
    import gate2_forensics as gf1
    import gate2_forensics4 as gf4
    import run_gate2_candidate15 as c15

    from populace_dynamics.data import transitions
    from populace_dynamics.harness import panel as hpanel

    data = gf1._load_inputs()
    panel = data["panel"]
    fert = data["fert"]
    support = gf4.gf3.observed_support(data["demo"])
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}

    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)

    # Reference parity (deterministic side-B observed).
    ref_parity = gf1.parity_distribution(fert, ids_b, gf4.Q8_DECADE)
    assert ref_parity["mean_parity"] == pytest.approx(
        recorded["ref_parity"]["mean_parity"], abs=1e-9
    )

    # Reference ledger (deterministic side-B observed).
    ref_led = gf4.stock_ledger(panel, ids_b, support)
    assert ref_led["W_total"] == pytest.approx(
        recorded["ref_ledger"]["W_total"], rel=1e-9
    )
    assert ref_led["W_reachable"] == pytest.approx(
        recorded["ref_ledger"]["W_reachable"], rel=1e-9
    )
    assert ref_led["bands"]["50-64"]["yield_b"] == pytest.approx(
        recorded["ref_ledger"]["bands"]["50-64"]["yield_b"], rel=1e-9
    )

    # First candidate-15 simulation draw (5200) reproduces the stock share.
    components = c15.fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )
    sim_panel, sim_births = c15.simulate_holdout(
        panel, ids_b, components, gf4.DRAW_SEED_BASE
    )
    sim_led = gf4.stock_ledger(sim_panel, ids_b, support)
    assert sim_led["share_widowed_75plus"] == pytest.approx(
        recorded["per_draw_share_widowed_75plus"][0], abs=1e-9
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    sim_parity = gf1.parity_distribution(sim_fert, ids_b, gf4.Q8_DECADE)
    # The first draw's mean parity is one of the 20 averaged into the record.
    assert not math.isnan(sim_parity["mean_parity"])
    assert 1.5 < sim_parity["mean_parity"] < 2.5
