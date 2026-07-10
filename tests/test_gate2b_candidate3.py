"""Tests for the gate-2b candidate-3 one-shot scored run.

Candidate 3 (issue #42 comment 4939960467) is candidate 2 with EXACTLY THREE
frozen deltas, each feeding a disjoint family the candidate-2 grading
(4939958136) isolated:

* **delta 1 -- custodial paternal conditioning** (``coresident_child``): a
  father-linked child counts as coresident in a wave only with the
  train-fitted ``P(coresident | child age band x father marital state)``;
* **delta 2 -- household bridge** (``hh_size``): a train-fitted non-family
  member count (0/1/2+) added to ``hh_size`` only;
* **delta 3 -- skipped-generation coresidence** (``coresident_grandchild``):
  train entry/exit hazards for ``coresident_grandchild AND NOT multigen``
  unioned into the composed grandchild only (never ``multigen``).

Every candidate-2 family that cleared or carried is carried byte-faithfully:
the carried cells' per-seed scores are IDENTICAL to candidate 2's to bit
precision (the three deltas draw from an isolated ``0xC3`` stream). The
one-shot outcome (published REGARDLESS of verdict) is pinned below from the
committed artifact ``runs/gate2b_hazard_v3.json``: **FAIL 0/5** -- the gate is
capped by the carried ``coresident_spouse`` family (byte-identical to candidate
2, failing on every seed), while all three deltas improve their target
families.

Always runnable: it inspects the committed artifact, binds the stored
tolerances to the ratified floor and the locked gates.yaml block, proves the
[20, 46, 5] cube reproduces every score, binds the cleared-family regression
and the byte-identical carried-score check, the c1->c2->c3 progression, and
unit-tests the three pure delta derivations. The reproduction pin (including
the byte-identical carried-family proof on real data) lives in
``tests/test_gate2b_candidate3_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from populace_dynamics.models import household_composition_sim_v3 as hcs3

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v3.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4939960467"
CANDIDATE2_CLEARED_FAMILIES = (
    "coresident_parent",
    "multigen_stock",
    "multigen_transition",
    "parental_home_exit",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _floor() -> dict:
    return json.loads(FLOOR.read_text())


def _gate2b_thresholds() -> dict:
    gates = yaml.safe_load(GATES.read_text())
    return gates["gates"]["gate_2"]["gate_2b"]["thresholds"]


def _gate2b_tolerances() -> dict[str, float]:
    tol: dict[str, float] = {}
    for view in _gate2b_thresholds()["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


# --------------------------------------------------------------------------
# Presence, registration, one-shot verdict pin
# --------------------------------------------------------------------------
def test_artifact_present_and_identity():
    a = _artifact()
    assert a["schema_version"] == "gate2b_hazard.v3"
    assert a["run"] == "gate2b_hazard_v3"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 3"
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["candidate2_registration_pointer"] == "4939456379"
    assert a["candidate1_registration_pointer"] == "4938726107"


def test_three_deltas_declared_and_mapped_to_disjoint_families():
    a = _artifact()
    deltas = a["deltas_vs_candidate_2"]
    assert len(deltas) == 3
    assert any("custodial" in d for d in deltas)
    assert any("household bridge" in d for d in deltas)
    assert any("skipped-generation" in d for d in deltas)
    mapping = a["per_delta_target_family"]
    assert mapping == {
        "delta_1_custodial_paternal_conditioning": "coresident_child",
        "delta_2_household_bridge": "hh_size",
        "delta_3_skipped_generation_coresidence": "coresident_grandchild",
    }
    # The three target families are disjoint.
    assert len(set(mapping.values())) == 3


def test_one_shot_verdict_pinned_fail_0_of_5():
    """The committed one-shot outcome: gate FAIL, 0 of 5 seeds pass."""
    v = _artifact()["verdict"]
    assert v["gate_2b_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert v["n_gate_seeds"] == 5
    assert v["n_gated_cells"] == N_GATED
    assert all(p is False for p in v["seed_pass"].values())


def test_gate_ceiling_is_carried_spouse_every_seed():
    """Every seed fails at least one carried coresident_spouse cell, so the
    byte-identically-carried spouse family caps the gate at 0/5 independent of
    the three deltas (each of which targets a non-spouse family)."""
    a = _artifact()
    for s in a["per_seed"]:
        spouse_fail = [
            c
            for c, rec in s["gated_cells"].items()
            if c.startswith("coresident_spouse.") and not rec["pass"]
        ]
        assert spouse_fail, s["seed"]


def test_forecast_recorded_and_not_graded_here():
    f = _artifact()["pre_registered_forecast"]
    assert f["p_gate_pass_4_of_5"] == "0.40-0.55"
    assert "grading_note" in f
    assert "orchestrator" in f["grading_note"]


def test_spec_resolution_notes_present():
    notes = _artifact()["spec_resolution_notes"]
    for key in (
        "rng_two_stream",
        "rng_byte_identical_carried_families",
        "observed_initial_states_are_the_holdout_persons_own",
        "custodial_paternal_conditioning",
        "household_bridge_nonfamily_members",
        "skipped_generation_coresidence",
        "coresident_children_maternal_side_untouched",
        "household_size_composition",
        "coresident_grandchild_composed_plus_skipgen",
    ):
        assert key in notes and notes[key]


# --------------------------------------------------------------------------
# Precheck + cell set
# --------------------------------------------------------------------------
def test_precheck_reproduced_exactly():
    p = _artifact()["precheck"]
    assert p["all_reproduced_exactly"] is True
    assert p["reference_moments_max_abs_deviation"] == 0.0
    assert p["rate_a_max_abs_deviation"] == 0.0
    assert p["holdout_sha256_all_match"] is True


def test_gated_cells_match_floor_gate_partition():
    a = _artifact()
    gated = set(_floor()["gate_partition"]["gate_eligible"])
    assert len(gated) == N_GATED
    assert set(_gate2b_tolerances()) == gated
    for s in a["per_seed"]:
        assert set(s["gated_cells"]) == gated


def test_report_only_cells_match_gates_yaml_and_never_gate():
    a = _artifact()
    report_only = set(_gate2b_thresholds()["report_only"])
    assert len(report_only) == 47
    for s in a["per_seed"]:
        assert set(s["report_only_cells"]) == report_only
        for rec in s["report_only_cells"].values():
            assert rec["gated"] is False


# --------------------------------------------------------------------------
# Fresh-run artifact-schema conformance ([20, 46, 5] + undefined + dispersion)
# --------------------------------------------------------------------------
def test_per_draw_per_cell_rates_shape_and_index():
    pc = _artifact()["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    assert pc["shape"] == [N_DRAWS, N_GATED, len(GATE_SEEDS)]
    assert pc["cell_index"] == sorted(_gate2b_tolerances())
    assert pc["seed_index"] == GATE_SEEDS
    assert pc["k_index_draw_seeds"] == [
        DRAW_SEED_BASE + k for k in range(N_DRAWS)
    ]
    rates = pc["rates"]
    assert len(rates) == N_DRAWS
    assert all(len(r) == N_GATED for r in rates)
    assert all(len(c) == len(GATE_SEEDS) for r in rates for c in r)


def test_per_draw_cube_matches_per_seed_records():
    a = _artifact()
    pc = a["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    ci, si = pc["cell_index"], pc["seed_index"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for k in range(N_DRAWS):
        for c_idx, cell in enumerate(ci):
            for s_idx, seed in enumerate(si):
                cube = pc["rates"][k][c_idx][s_idx]
                stored = by_seed[seed]["gated_cells"][cell]["per_draw_rate"][k]
                assert cube == pytest.approx(stored, abs=1e-15)


def test_rbar_recomputes_from_per_draw_rates_and_scores():
    for s in _artifact()["per_seed"]:
        for rec in s["gated_cells"].values():
            rates = rec["per_draw_rate"]
            assert len(rates) == N_DRAWS
            rbar = float(np.mean(rates))
            assert rbar == pytest.approx(rec["rbar"], abs=1e-12)
            assert rec["r_candidate"] == pytest.approx(rec["rbar"], abs=1e-15)
            rate_a = rec["rate_a"]
            if rbar > 0 and rate_a > 0:
                expected = abs(math.log(rbar / rate_a))
                assert rec["score"] == pytest.approx(expected, abs=1e-12)


def test_undefined_draw_rule_not_triggered_and_run_valid():
    u = _artifact()["fresh_run_artifact_schema"]["undefined_draw_rule"]
    assert u["required"] is True
    assert u["run_invalidated"] is False
    assert u["n_undefined_gated_draws"] == 0
    for s in _artifact()["per_seed"]:
        assert s["undefined_gated_draws"] == []
        for rec in s["gated_cells"].values():
            assert rec["n_draws_defined"] == N_DRAWS


# --------------------------------------------------------------------------
# Seed conjunction + verdict recompute from the cells
# --------------------------------------------------------------------------
def test_every_gated_pass_recomputes_from_score():
    for s in _artifact()["per_seed"]:
        for rec in s["gated_cells"].values():
            assert rec["pass"] == (rec["score"] <= rec["tolerance"])


def test_seed_pass_recomputes_from_all_gated_cells():
    for s in _artifact()["per_seed"]:
        n_pass = sum(rec["pass"] for rec in s["gated_cells"].values())
        assert n_pass == s["n_gated_pass"]
        assert s["seed_pass"] == (n_pass == N_GATED)


def test_verdict_recomputes_from_seed_conjunction():
    a = _artifact()
    n_pass = sum(s["seed_pass"] for s in a["per_seed"])
    assert a["verdict"]["n_seeds_pass"] == n_pass
    assert a["verdict"]["gate_2b_pass"] == (n_pass >= 4)


def test_stored_tolerances_match_locked_gates_yaml():
    tol = _gate2b_tolerances()
    for s in _artifact()["per_seed"]:
        for cell, rec in s["gated_cells"].items():
            assert rec["tolerance"] == pytest.approx(tol[cell], abs=1e-12)


# --------------------------------------------------------------------------
# Byte-identical carried families + cleared regression + progression
# --------------------------------------------------------------------------
def test_cleared_families_still_clear():
    """The four candidate-2 cleared families still clear at 1.00."""
    chk = _artifact()["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    assert set(chk["families"]) == set(CANDIDATE2_CLEARED_FAMILIES)
    for fam in CANDIDATE2_CLEARED_FAMILIES:
        assert chk["detail"][fam]["candidate3_pass_rate"] == 1.0
        assert chk["detail"][fam]["still_clears"] is True


def test_carried_family_scores_byte_identical_to_candidate_2():
    """Every carried cell's per-seed gated score equals candidate 2's, cell
    for cell and seed for seed -- the strong regression proof (the three
    deltas draw from an isolated stream and cannot perturb the carried
    families, including the carried-but-uncleared coresident_spouse)."""
    a = _artifact()
    byt = a["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["byte_identical"] is True
    assert byt["max_abs_score_deviation_vs_candidate2"] == 0.0
    # Cross-check directly against candidate 2's per-seed scores.
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())
    by_seed_3 = {s["seed"]: s for s in a["per_seed"]}
    by_seed_2 = {s["seed"]: s for s in c2["per_seed"]}
    carried_cells = [
        c
        for c in _gate2b_tolerances()
        if c.startswith(
            (
                "coresident_parent.",
                "multigen.",
                "parental_home_",
                "coresident_spouse.",
            )
        )
        or c in ("multigen_entry", "multigen_exit")
    ]
    assert set(carried_cells) == set(byt["carried_cells"])
    for seed in GATE_SEEDS:
        for cell in carried_cells:
            s3 = by_seed_3[seed]["gated_cells"][cell]["score"]
            s2 = by_seed_2[seed]["gated_cells"][cell]["score"]
            assert s3 == pytest.approx(s2, abs=1e-12), (seed, cell)


def test_spouse_family_byte_identical_to_candidate_2():
    """coresident_spouse is carried unchanged (no delta touches it): its
    per-family pass rate equals candidate 2's exactly. This is why the gate
    is capped at 0/5 regardless of the three deltas."""
    a = _artifact()
    prog = a["comparison_across_candidates"]["per_family_progression"]
    assert (
        prog["coresident_spouse"]["candidate3_pass_rate"]
        == prog["coresident_spouse"]["candidate2_pass_rate"]
    )
    assert prog["coresident_spouse"]["delta_c2_to_c3"] == 0.0


def test_progression_recomputes_and_targets_improve_over_candidate_2():
    a = _artifact()
    prog = a["comparison_across_candidates"]["per_family_progression"]
    decomp = a["per_family_decomposition"]
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())[
        "per_family_decomposition"
    ]
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())[
        "per_family_decomposition"
    ]
    for fam, block in prog.items():
        assert (
            block["candidate3_pass_rate"] == decomp[fam]["cell_seed_pass_rate"]
        )
        assert block["candidate1_pass_rate"] == c1[fam]["cell_seed_pass_rate"]
        assert block["candidate2_pass_rate"] == c2[fam]["cell_seed_pass_rate"]
        expected = round(
            block["candidate3_pass_rate"] - block["candidate2_pass_rate"], 4
        )
        assert block["delta_c2_to_c3"] == pytest.approx(expected, abs=1e-9)
    # Each delta's target family improves materially over candidate 2.
    assert prog["coresident_child"]["delta_c2_to_c3"] > 0
    assert prog["hh_size"]["delta_c2_to_c3"] > 0
    assert prog["coresident_grandchild"]["delta_c2_to_c3"] > 0


def test_delta_stats_recorded_for_all_three_deltas():
    for s in _artifact()["per_seed"]:
        ds = s["delta_stats"]
        d1 = ds["delta_1_custodial"]
        assert d1["custodial_n_train_exposure"] > 0
        assert d1["custodial_n_train_coresident"] > 0
        # 5 child age bands x 2 marital states = 10 custodial probabilities.
        assert len(d1["custodial_probability_by_band_marital"]) == 10
        for p in d1["custodial_probability_by_band_marital"].values():
            assert 0.0 <= p <= 1.0
        d2 = ds["delta_2_household_bridge"]
        p0, p1, p2 = d2["nonfamily_train_overall_p0_p1_p2plus"]
        assert p0 == pytest.approx(1.0 - p1 - p2, abs=1e-4)
        assert d2["mean_nonfamily_count_simulated"] >= 0.0
        d3 = ds["delta_3_skipgen"]
        assert 0.0 <= d3["skipgen_entry_overall"] <= 1.0
        assert 0.0 <= d3["skipgen_exit_overall"] <= 1.0
        assert d3["skipgen_train_person_waves"] > 0


def test_grandchild_55plus_female_improves_over_candidate_2():
    """Delta 3 materially reduces the worst grandchild cell (55+|female),
    the mechanism the candidate-2 grading isolated as flat and unaddressed."""
    a = _artifact()
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())
    cell = "coresident_grandchild.55+|female"
    m3 = np.mean([s["gated_cells"][cell]["score"] for s in a["per_seed"]])
    m2 = np.mean([s["gated_cells"][cell]["score"] for s in c2["per_seed"]])
    assert m3 < m2


def test_father_link_coverage_recorded_and_consistent():
    for s in _artifact()["per_seed"]:
        c = s["father_link_coverage"]
        assert c["n_linked_fathers"] + c["n_unlinked_men"] == c["n_side_a_men"]
        assert 0.0 <= c["coverage_fraction_men"] <= 1.0
        assert c["n_linked_fathers"] > 0
        assert c["n_unlinked_men"] > 0


def test_per_family_decomposition_covers_all_gated_cells():
    a = _artifact()
    decomp = a["per_family_decomposition"]
    covered: set[str] = set()
    for fam in decomp.values():
        covered.update(fam["cells"])
        assert fam["mechanism"]
    assert covered == set(_gate2b_tolerances())


def test_model_records_certified_registry_and_delta_modules():
    a = _artifact()
    assert a["model"]["family_transitions_spec"] == "candidate16_registry_v1"
    assert (
        a["model"]["delta_module"]
        == "populace_dynamics.models.household_composition_sim_v3"
    )
    assert a["model"]["custodial_child_age_bands"] == [
        "0-4",
        "5-12",
        "13-17",
        "18-24",
        "25-60",
    ]
    assert a["model"]["nonfamily_classes"] == ["0", "1", "2+"]
    sha = a["model"]["family_transitions_spec_sha256"]
    for s in a["per_seed"]:
        assert s["component_meta"]["family_transitions_spec_sha256"] == sha


# --------------------------------------------------------------------------
# Pure delta derivations (synthetic frames; no PSID needed)
# --------------------------------------------------------------------------
def test_custodial_child_age_bands_a_priori():
    assert hcs3.CUSTODIAL_CHILD_AGE_BANDS == (
        (0, 4),
        (5, 12),
        (13, 17),
        (18, 24),
        (25, 60),
    )
    assert hcs3.CHILD_CORESIDENCE_MAX_AGE == 60


def test_father_link_births_with_child_keeps_joinable_children_only():
    """Delta-1 fit input: male biological births with a joinable child id and
    a birth year; female parents, adoptions, denial rows, and births with no
    joinable child id are dropped."""
    records = pd.DataFrame(
        {
            "parent_person_id": [1, 2, 3, 4, 5, 6],
            "parent_sex": ["male", "female", "male", "male", "male", "male"],
            "record_type": [
                "birth",
                "birth",
                "adoption",
                "birth",
                "birth",
                "birth",
            ],
            "birth_year": pd.array(
                [1990, 1991, 1992, 1993, pd.NA, 1995], dtype="Int64"
            ),
            "child_person_id": pd.array(
                [1001, 1002, 1003, pd.NA, 1005, 1006], dtype="Int64"
            ),
            "birth_order": pd.array([1, 1, 1, 1, 1, 1], dtype="Int64"),
            "is_event": [True, True, True, True, True, True],
        }
    )
    out = hcs3.father_link_births_with_child(records)
    # only persons 1 and 6 survive (male, birth, has child id, has year).
    assert out["parent_person_id"].tolist() == [1, 6]
    assert out["child_person_id"].tolist() == [1001, 1006]
    assert out["birth_year"].tolist() == [1990, 1995]


def test_parent_child_coresidence_pairs_uses_parent_link_codes():
    """A father coded a parent (MX8 {50,53,55,56}) of an alter yields a
    coresidence pair; a non-parent link (e.g. sibling 40) does not."""
    rel_map = pd.DataFrame(
        {
            "interview_year": [2000, 2000, 2000],
            "ego_person_id": [10, 10, 20],
            "ego_rel_to_alter": [50, 40, 53],  # parent, sibling, step-parent
            "alter_person_id": [101, 102, 201],
        }
    )
    pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    got = set(
        map(
            tuple,
            pairs[["parent_person_id", "child_person_id", "year"]].to_numpy(),
        )
    )
    assert (10, 101, 2000) in got  # code 50 parent
    assert (20, 201, 2000) in got  # code 53 step-parent
    assert (10, 102, 2000) not in got  # code 40 sibling excluded


def test_family_unit_sizes_counts_self_plus_nuclear_links():
    """family_unit_size = 1 + spouse-links + child-links + parent-links; a
    sibling (40) and a non-relative (98) do NOT count (they are the
    non-family members delta 2 bridges)."""
    rel_map = pd.DataFrame(
        {
            "interview_year": [2000, 2000, 2000, 2000],
            "ego_person_id": [10, 10, 10, 10],
            # spouse(20), child(50), sibling(40), nonrelative(98)
            "ego_rel_to_alter": [20, 50, 40, 98],
            "alter_person_id": [11, 12, 13, 14],
        }
    )
    fu = hcs3.family_unit_sizes(rel_map).set_index(["person_id", "year"])
    # 1 self + 1 spouse + 1 child = 3 (sibling + nonrelative excluded).
    assert int(fu.loc[(10, 2000), "family_unit_size"]) == 3


def test_attach_skipgen_flags_grandchild_without_multigen():
    """The observed skipped-generation state is coresident_grandchild AND NOT
    multigen -- the mass the composed grandchild misses."""
    pw = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "year": [2000, 2002, 2000, 2000],
            "coresident_grandchild": [True, True, True, False],
            "multigen": [False, True, False, False],
        }
    )
    out = hcs3.attach_skipgen(pw).set_index(["person_id", "year"])
    assert bool(out.loc[(1, 2000), "skipgen"]) is True  # gc & ~multigen
    assert bool(out.loc[(1, 2002), "skipgen"]) is False  # gc & multigen
    assert bool(out.loc[(2, 2000), "skipgen"]) is True
    assert bool(out.loc[(3, 2000), "skipgen"]) is False  # no grandchild
    # next-state alignment within person 1.
    assert bool(out.loc[(1, 2000), "next_skipgen"]) is False


def test_nonfamily_classes_and_contribution():
    assert hcs3.NONFAMILY_CLASSES == ("0", "1", "2+")
    assert hcs3._NONFAMILY_CONTRIB == {"0": 0, "1": 1, "2+": 2}


def test_custodial_counts_gate_reduces_linked_children():
    """custodial_linked_child_counts gates each linked child per wave: at
    probability 1.0 every in-window child is counted; at 0.0 none is."""
    side_a_pw = pd.DataFrame(
        {
            "person_id": [1, 1],
            "year": [2000, 2010],
            "band": ["25-34", "35-44"],
            "sex": ["male", "male"],
        }
    )
    linked = pd.DataFrame(
        {"parent_person_id": [1], "birth_year": [1998]}
    )  # child age 2 in 2000, 12 in 2010
    marital = pd.DataFrame(
        {
            "person_id": [1, 1],
            "year": [2000, 2010],
            "marital": ["married", "married"],
        }
    )
    band_ok = {
        ("0-4", "married"): 1.0,
        ("5-12", "married"): 1.0,
    }
    full = {
        (hcs3.hc.band_label(lo, hi), m): band_ok.get(
            (hcs3.hc.band_label(lo, hi), m), 0.0
        )
        for lo, hi in hcs3.CUSTODIAL_CHILD_AGE_BANDS
        for m in ("married", "not_married")
    }
    rng = np.random.default_rng(0)
    counts_all = hcs3.custodial_linked_child_counts(
        linked, side_a_pw, marital, full, rng
    )
    assert counts_all.tolist() == [1, 1]  # both waves in-window at p=1
    none = {k: 0.0 for k in full}
    counts_none = hcs3.custodial_linked_child_counts(
        linked, side_a_pw, marital, none, np.random.default_rng(0)
    )
    assert counts_none.tolist() == [0, 0]
