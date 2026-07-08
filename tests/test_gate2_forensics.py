"""Tests for the gate-2 chronic-cell forensics (reported, not gated).

REPORTED-NOT-GATED. The consistency tests read only the forensics artifact
(``runs/gate2_forensics_v1.json``), the committed candidate-8 gate artifact
(``runs/gate2_hazard_v8.json``), and ``gates.yaml``; they never rerun the
diagnostic and need no PSID, so they run in CI. They audit that every headline
recomputes from the stored per-seed values: the Q1 marriage-count deficit and
its residual reconciliation, the Q2 parity/censoring decomposition, and the Q3
train-side draw distributions, verdicts, tolerances and the published outer
scores.

Two reproduction pins (``test_pin_*``) rebuild the train-side inputs live and
match one small block -- seed 0's train-side reference rates and the first
simulation draw -- to float precision (skipped when the PSID marriage-history
files are absent). They pin the train-side pipeline exactly as the candidate-8
run pins its own scoring path.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_forensics_v1.json"
CANDIDATE8 = ROOT / "runs" / "gate2_hazard_v8.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

FOCAL_SIX = (
    "share_divorced.45-54|female",
    "widowhood.75+|female",
    "share_widowed.75+|female",
    "mean_lifetime_marriages|female",
    "mean_lifetime_marriages|male",
    "completed_fertility.c1970s",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate2_tolerances() -> dict[str, float]:
    sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate1 as c1

    return c1.gated_tolerances(c1.load_gate2_thresholds())


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2_forensics.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4913512779")
    assert "candidate 8" in a["candidate_under_diagnosis"]
    for block in (
        "question_1_marriage_pathway",
        "question_2_completed_fertility_c1970s",
        "question_3_rng_stability",
    ):
        assert block in a


def test_protocol_is_train_side_only():
    p = _artifact()["protocol"]
    assert p["train_side_only"] is True
    assert "never" in p["outer_holdout_contact"].lower()
    assert p["gate_seeds"] == [0, 1, 2, 3, 4]


def test_no_gate_verdict_written_and_lock_untouched():
    a = _artifact()
    # A diagnostic, not a gate run: no gate pass/fail verdict is emitted.
    assert "verdict" not in a
    assert "gate_2_pass" not in a
    assert a["revision_pins"]["gates_yaml_locked"] is True


# --------------------------------------------------------------------------
# Question 1 -- marriage-count pathway decomposition
# --------------------------------------------------------------------------
def test_q1_deficit_recomputes_from_per_seed():
    a = _artifact()
    per_seed = a["per_seed"]
    for sex in ("male", "female"):
        blk = a["question_1_marriage_pathway"]["by_sex"][sex]
        ref = np.mean(
            [
                s["ref_pathway"][sex]["mean_lifetime_marriages"]
                for s in per_seed
            ]
        )
        sim = np.mean([s["sim_nmarr_mean"][sex] for s in per_seed])
        assert blk["mean_lifetime_marriages_reference"] == pytest.approx(
            ref, abs=1e-9
        )
        assert blk["mean_lifetime_marriages_simulated"] == pytest.approx(
            sim, abs=1e-9
        )
        assert blk["deficit"] == pytest.approx(ref - sim, abs=1e-9)


def test_q1_residual_reconciles_exactly():
    a = _artifact()
    for sex in ("male", "female"):
        blk = a["question_1_marriage_pathway"]["by_sex"][sex]
        residual = blk["reference_residual_per_person"]
        # The reference residual is the reference count not surfacing as an
        # in-exposure event: mean_lifetime_marriages minus in-exposure count.
        assert residual == pytest.approx(
            blk["mean_lifetime_marriages_reference"]
            - blk["in_exposure_marriages_per_person_reference"],
            abs=1e-9,
        )
        # The named buckets + remainder sum to the reference residual.
        bd = blk["reference_residual_breakdown"]
        assert sum(bd.values()) == pytest.approx(residual, abs=1e-9)
        # The reconciliation remainder is negligible (the buckets localise it).
        assert abs(bd["reconciliation_remainder"]) < 5e-3
        # The deficit that the cell shows decomposes as: reference residual
        # (the simulation cannot generate it) minus the simulation's own tiny
        # residual minus the in-exposure surplus. Equivalently the in-exposure
        # deficit is at most the total deficit (the residual carries the rest).
        assert blk["in_exposure_deficit"] <= blk["deficit"] + 1e-9


def test_q1_pathway_table_deficit_is_reference_minus_simulated():
    tbl = _artifact()["question_1_marriage_pathway"]["by_sex"]["male"][
        "pathway_cells"
    ]
    for rec in tbl.values():
        assert rec["deficit"] == pytest.approx(
            rec["reference"] - rec["simulated"], abs=1e-9
        )


def test_q1_first_marriage_is_intensive_margin_near_one():
    # Every conditioned (ever-married) person has exactly one first marriage,
    # so the "first" pathway per-person count is ~1 on both sides.
    tbl = _artifact()["question_1_marriage_pathway"]["by_sex"]["male"][
        "pathway_cells"
    ]
    assert tbl["first"]["reference"] == pytest.approx(1.0, abs=0.05)
    assert tbl["first"]["simulated"] == pytest.approx(1.0, abs=0.05)


def test_q1_finding_localises_deficit_to_residual():
    # The headline claim -- the deficit is the residual, not the in-exposure
    # pathways -- must be backed by the stored numbers.
    male = _artifact()["question_1_marriage_pathway"]["by_sex"]["male"]
    assert male["deficit"] > 0  # simulation under-produces the count
    # In-exposure the simulation matches or exceeds the reference.
    assert male["in_exposure_deficit"] <= male["deficit"]
    # The residual is the dominant part of the deficit.
    assert male["reference_residual_per_person"] >= male["deficit"]


# --------------------------------------------------------------------------
# Question 2 -- completed-fertility c1970s decomposition
# --------------------------------------------------------------------------
def test_q2_gap_recomputes_and_direction():
    a = _artifact()["question_2_completed_fertility_c1970s"]
    ref = a["mean_completed_parity_reference"]
    sim = a["mean_completed_parity_simulated"]
    assert a["gap_reference_minus_simulated"] == pytest.approx(
        ref - sim, abs=1e-9
    )
    # Direction string agrees with the sign of the gap.
    if a["gap_reference_minus_simulated"] > 0:
        assert "UNDER" in a["gap_direction"]


def test_q2_margin_deltas_recompute_and_sum_to_gap():
    a = _artifact()["question_2_completed_fertility_c1970s"]
    pd_ = a["parity_distribution"]
    ref, sim = pd_["reference"], pd_["simulated"]
    md = pd_["margin_deltas_reference_minus_simulated"]
    assert md["ge_1"] == pytest.approx(
        ref["share_ge_1"] - sim["share_ge_1"], abs=1e-9
    )
    assert md["ge_2"] == pytest.approx(
        ref["share_ge_2"] - sim["share_ge_2"], abs=1e-9
    )
    # mean parity == sum_k P(parity>=k): the margin deltas reconstruct the gap
    # (to the truncation at 4+, so within a small tail).
    approx_gap = md["ge_1"] + md["ge_2"] + md["ge_3plus"] + md["ge_4plus"]
    assert approx_gap == pytest.approx(
        a["gap_reference_minus_simulated"], abs=0.02
    )


def test_q2_parity_survival_shares_monotone():
    for side in ("reference", "simulated"):
        p = _artifact()["question_2_completed_fertility_c1970s"][
            "parity_distribution"
        ][side]
        assert p["share_ge_1"] >= p["share_ge_2"] >= p["share_ge_3plus"]
        assert p["share_ge_3plus"] >= p["share_ge_4plus"]


def test_q2_censoring_convention_nearly_shared():
    c = _artifact()["question_2_completed_fertility_c1970s"][
        "censoring_convention"
    ]
    # Only a negligible share of reference cohort births fall outside [15,49],
    # so the reference's all-age convention is effectively the sim's window.
    assert c["share_outside_15_49"] < 0.02
    assert c["shared"] is True


# --------------------------------------------------------------------------
# Question 3 -- RNG stability
# --------------------------------------------------------------------------
def test_q3_cells_and_draw_seeds():
    q3 = _artifact()["question_3_rng_stability"]
    assert q3["draw_seeds"] == list(range(5200, 5220))
    assert set(q3["registration_named_cells"]) == {
        "share_divorced.45-54|female",
        "widowhood.75+|female",
        "share_widowed.75+|female",
    }
    assert "mean_lifetime_marriages|female" in q3["focal_four"]
    assert set(q3["all_failing_cells_reported"]) == set(FOCAL_SIX)


def test_q3_tolerances_match_locked_gates_yaml():
    tol = _gate2_tolerances()
    cells = _artifact()["question_3_rng_stability"]["cells"]
    for cell, blk in cells.items():
        assert blk["tolerance"] == pytest.approx(tol[cell], abs=1e-12)


def test_q3_outer_scores_match_committed_candidate8():
    committed = json.loads(CANDIDATE8.read_text())
    by_seed = {s["seed"]: s for s in committed["per_seed"]}
    cells = _artifact()["question_3_rng_stability"]["cells"]
    for cell, blk in cells.items():
        for seed_str, rec in blk["per_seed"].items():
            gc = by_seed[int(seed_str)]["gated_cells"][cell]
            assert rec["outer_score"] == pytest.approx(gc["score"], abs=1e-9)
            assert rec["outer_rate_a"] == pytest.approx(gc["rate_a"], abs=1e-9)
            assert rec["outer_pass"] == gc["pass"]


def test_q3_train_distributions_recompute_from_draws():
    a = _artifact()
    per_seed = {s["seed"]: s for s in a["per_seed"]}
    cells = a["question_3_rng_stability"]["cells"]
    for cell, blk in cells.items():
        for seed_str, rec in blk["per_seed"].items():
            scores = per_seed[int(seed_str)]["draw_scores"][cell]
            assert len(scores) == 20
            assert rec["train_score_mean"] == pytest.approx(
                float(np.mean(scores)), abs=1e-9
            )
            assert rec["train_score_sd"] == pytest.approx(
                float(np.std(scores, ddof=1)), abs=1e-9
            )
            rates = per_seed[int(seed_str)]["draw_rates"][cell]
            rate_b = rec["rate_b_train_reference"]
            recomputed = [
                abs(math.log(r / rate_b))
                for r in rates
                if r > 0 and rate_b > 0
            ]
            assert rec["train_score_mean"] == pytest.approx(
                float(np.mean(recomputed)), abs=1e-9
            )


def test_q3_verdict_follows_level_and_clip_probability():
    # Graded verdict: LEVEL if the systematic offset alone clips the tolerance;
    # else BOUNDARY if a fresh train draw clips >= 25% of the time; else
    # NOISE-DOMINATED. Recompute from the stored summary fields.
    cells = _artifact()["question_3_rng_stability"]["cells"]
    for blk in cells.values():
        summ = blk["summary"]
        level = abs(summ["mean_train_signed_offset_over_seeds"])
        clip_p = summ["mean_prob_train_draw_clips_tolerance"]
        if level >= blk["tolerance"]:
            expected = "LEVEL"
        elif clip_p >= 0.25:
            expected = "BOUNDARY"
        else:
            expected = "NOISE-DOMINATED"
        assert summ["verdict"] == expected
        assert summ["systematic_offset_clips_tolerance"] == (
            level >= blk["tolerance"]
        )


def test_q3_verdict_partition_covers_all_cells():
    q3 = _artifact()["question_3_rng_stability"]
    part = q3["verdict_partition"]
    all_partitioned = (
        part["LEVEL"] + part["BOUNDARY"] + part["NOISE-DOMINATED"]
    )
    assert set(all_partitioned) == set(q3["all_failing_cells_reported"])
    assert len(all_partitioned) == len(set(all_partitioned))  # disjoint
    # The registration's headline reproduces: the persistent 75+ widowed-stock
    # under-production and the male marriage-count boundary are the near-level
    # targets; the knife-edge share_divorced.45-54 is noise-dominated.
    targets = q3["candidate_9_level_targets"]
    assert "share_widowed.75+|female" in targets
    assert "mean_lifetime_marriages|male" in targets
    assert "share_divorced.45-54|female" in part["NOISE-DOMINATED"]


def test_q3_outer_clip_decomposition_recomputes():
    # Each outer clip decomposes into level + split/draw excess, and the level
    # component equals |mean signed train offset|.
    cells = _artifact()["question_3_rng_stability"]["cells"]
    for blk in cells.values():
        level = abs(blk["summary"]["mean_train_signed_offset_over_seeds"])
        for dec in blk["outer_clip_decomposition"]:
            assert dec["level_component"] == pytest.approx(level, abs=1e-9)
            assert dec["split_draw_excess"] == pytest.approx(
                dec["outer_score"] - level, abs=1e-9
            )


def test_q3_prob_clip_in_unit_interval():
    cells = _artifact()["question_3_rng_stability"]["cells"]
    for blk in cells.values():
        for rec in blk["per_seed"].values():
            assert 0.0 <= rec["prob_train_draw_clips_tolerance"] <= 1.0


# --------------------------------------------------------------------------
# Reproduction pins (need the staged PSID marriage-history files)
# --------------------------------------------------------------------------
@needs_psid
def test_pin_seed0_train_reference_reproduces():
    """Seed 0's train-side reference rates reproduce the recorded values."""
    sys.path.insert(0, str(SCRIPTS))
    import gate2_forensics as gf

    from populace_dynamics.data import transitions
    from populace_dynamics.harness import panel as hpanel

    data = gf._load_inputs()
    panel, fert = data["panel"], data["fert"]
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    rate_b = transitions.reference_moments(panel, fert, ids_b, weighted=True)
    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    for cell in FOCAL_SIX:
        assert float(rate_b[cell]["rate"]) == pytest.approx(
            recorded["rate_b"][cell], abs=1e-9
        )


@needs_psid
def test_pin_seed0_first_draw_reproduces():
    """Seed 0's first simulation-RNG draw (5200) reproduces the recorded rate.

    Rebuilds the fitted components and simulates the train half at draw seed
    5200, matching the recorded ``draw_rates`` for the focal cells -- the
    train-side pipeline pinned bit-for-bit, as the candidate-8 run pins its
    own seed-0 scoring.
    """
    sys.path.insert(0, str(SCRIPTS))
    import gate2_forensics as gf
    import run_gate2_candidate8 as c8

    from populace_dynamics.data import transitions
    from populace_dynamics.harness import panel as hpanel

    data = gf._load_inputs()
    panel = data["panel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    components = c8.fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )
    sim_panel, sim_births = c8.simulate_holdout(
        panel, ids_b, components, gf.DRAW_SEED_BASE
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    cand = transitions.reference_moments(
        sim_panel, sim_fert, ids_b, weighted=True
    )
    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    for cell in FOCAL_SIX:
        assert float(cand[cell]["rate"]) == pytest.approx(
            recorded["draw_rates"][cell][0], abs=1e-9
        )
