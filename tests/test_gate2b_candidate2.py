"""Tests for the gate-2b candidate-2 one-shot scored run.

Candidate 2 (issue #42 comment 4939456379) is candidate 1 with EXACTLY TWO
frozen deltas -- a cohabiting-partner (MX8 code-22) occupancy overlay unioned
into ``coresident_spouse`` and paternal child attribution from observed
cah85_23 father->child links (shadow kernel retained for the unlinked
residual) -- and carries every candidate-1 family that cleared byte-faithfully
(the parental-home and multigen states come from candidate 1's
``simulate_draw`` unchanged). The one-shot outcome (published REGARDLESS of
verdict) is pinned below from the committed artifact
``runs/gate2b_hazard_v2.json``: **FAIL 0/5**, with the cleared families holding
and the two targeted families improving.

Always runnable: it inspects the committed artifact, binds the stored
tolerances to the ratified floor and the locked gates.yaml block, proves the
[20, 46, 5] per-draw cube reproduces every score, binds the cleared-family
regression check and the candidate-1 comparison, and unit-tests the two pure
delta derivations (the code-22 cohabitation flag and the father-link
resolver). The reproduction pin (including the byte-identical carried-family
proof) lives in ``tests/test_gate2b_candidate2_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from populace_dynamics.models import household_composition_sim_v2 as hcs2

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4939456379"
CANDIDATE1_CLEARED_FAMILIES = (
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
    assert a["schema_version"] == "gate2b_hazard.v2"
    assert a["run"] == "gate2b_hazard_v2"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 2"
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["candidate1_registration_pointer"] == "4938726107"


def test_two_deltas_declared():
    a = _artifact()
    deltas = a["deltas_vs_candidate_1"]
    assert len(deltas) == 2
    assert any("code-22" in d for d in deltas)
    assert any("father" in d for d in deltas)
    assert a["model"]["cohabitation_partner_code"] == 22


def test_one_shot_verdict_pinned_fail_0_of_5():
    """The committed one-shot outcome: gate FAIL, 0 of 5 seeds pass."""
    v = _artifact()["verdict"]
    assert v["gate_2b_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert v["n_gate_seeds"] == 5
    assert v["n_gated_cells"] == N_GATED
    assert all(p is False for p in v["seed_pass"].values())


def test_forecast_recorded_and_not_graded_here():
    f = _artifact()["pre_registered_forecast"]
    assert f["p_gate_pass_4_of_5"] == "0.35-0.50"
    assert "grading_note" in f
    assert "orchestrator" in f["grading_note"]


def test_spec_resolution_notes_present():
    notes = _artifact()["spec_resolution_notes"]
    for key in (
        "rng_two_stream",
        "rng_byte_identical_carried_families",
        "observed_initial_states_are_the_holdout_persons_own",
        "cohabitation_overlay_is_code_22_partner_only",
        "paternal_attribution_from_observed_father_links",
        "household_size_composition",
        "coresident_grandchild_composed_only",
    ):
        assert key in notes and notes[key]


# --------------------------------------------------------------------------
# Precheck: the committed artifact records a bit-exact floor reproduction
# --------------------------------------------------------------------------
def test_precheck_reproduced_exactly():
    p = _artifact()["precheck"]
    assert p["all_reproduced_exactly"] is True
    assert p["reference_moments_max_abs_deviation"] == 0.0
    assert p["rate_a_max_abs_deviation"] == 0.0
    assert p["holdout_sha256_all_match"] is True


# --------------------------------------------------------------------------
# Cell set: 46 gated == floor gate_partition; 47 report-only
# --------------------------------------------------------------------------
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
# Candidate-1 comparison + cleared-family regression check
# --------------------------------------------------------------------------
def test_cleared_families_still_clear_byte_faithfully():
    """The four candidate-1 families that cleared still clear at 1.00.

    Candidate 2 reads their states off candidate 1's simulate_draw unchanged,
    so byte-identical carry makes the regression check exact.
    """
    chk = _artifact()["comparison_to_candidate_1"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    assert set(chk["families"]) == set(CANDIDATE1_CLEARED_FAMILIES)
    for fam in CANDIDATE1_CLEARED_FAMILIES:
        assert chk["detail"][fam]["candidate2_pass_rate"] == 1.0
        assert chk["detail"][fam]["still_clears"] is True


def test_cleared_family_worst_scores_are_identical_to_candidate_1():
    """Byte-identical carry: each cleared family's per-seed gated scores equal
    candidate 1's, cell for cell and seed for seed (not just the pass rate)."""
    a = _artifact()
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())
    by_seed_2 = {s["seed"]: s for s in a["per_seed"]}
    by_seed_1 = {s["seed"]: s for s in c1["per_seed"]}
    carried_cells = [
        c
        for c in _gate2b_tolerances()
        if c.startswith(("coresident_parent.", "multigen.", "parental_home_"))
        or c in ("multigen_entry", "multigen_exit")
    ]
    assert carried_cells
    for seed in GATE_SEEDS:
        for cell in carried_cells:
            s2 = by_seed_2[seed]["gated_cells"][cell]["score"]
            s1 = by_seed_1[seed]["gated_cells"][cell]["score"]
            assert s2 == pytest.approx(s1, abs=1e-12), (seed, cell)


def test_comparison_deltas_recompute_and_targets_improve():
    a = _artifact()
    comp = a["comparison_to_candidate_1"]["per_family"]
    decomp = a["per_family_decomposition"]
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())[
        "per_family_decomposition"
    ]
    for fam, block in comp.items():
        assert (
            block["candidate2_pass_rate"] == decomp[fam]["cell_seed_pass_rate"]
        )
        assert block["candidate1_pass_rate"] == c1[fam]["cell_seed_pass_rate"]
        expected = round(
            block["candidate2_pass_rate"] - block["candidate1_pass_rate"], 4
        )
        assert block["delta"] == pytest.approx(expected, abs=1e-9)
    # The two targeted families improve materially over candidate 1.
    assert comp["coresident_spouse"]["delta"] > 0
    assert comp["coresident_child"]["delta"] > 0
    assert comp["hh_size"]["delta"] > 0


# --------------------------------------------------------------------------
# Father-link coverage split (delta 2) recorded and internally consistent
# --------------------------------------------------------------------------
def test_father_link_coverage_recorded_and_consistent():
    for s in _artifact()["per_seed"]:
        c = s["father_link_coverage"]
        assert c["n_linked_fathers"] + c["n_unlinked_men"] == c["n_side_a_men"]
        assert 0.0 <= c["coverage_fraction_men"] <= 1.0
        # Partial coverage: some linked, some shadow residual (the forecast).
        assert c["n_linked_fathers"] > 0
        assert c["n_unlinked_men"] > 0
        assert c["mean_paternal_linked_births"] > 0
        assert c["mean_paternal_shadow_births"] >= 0


# --------------------------------------------------------------------------
# Per-family decomposition covers all gated cells + records the delta mechanism
# --------------------------------------------------------------------------
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
        == "populace_dynamics.models.household_composition_sim_v2"
    )
    sha = a["model"]["family_transitions_spec_sha256"]
    for s in a["per_seed"]:
        assert s["component_meta"]["family_transitions_spec_sha256"] == sha


# --------------------------------------------------------------------------
# Pure delta derivations (synthetic frames; no PSID needed)
# --------------------------------------------------------------------------
def test_partner_code_is_mx8_22():
    assert hcs2.PARTNER_CODE == 22


def test_cohabitation_flag_selects_only_code_22():
    """A code-22 (partner) alter sets cohabiting; a code-20 (legal spouse)
    alter does not -- the overlay is the partner mass the registry omits."""
    rel_map = pd.DataFrame(
        {
            "interview_year": [2000, 2000, 2000, 2000, 2001],
            "interview_number": [1, 1, 2, 2, 3],
            "ego_person_id": [101, 202, 303, 404, 101],
            "ego_sequence": [1, 2, 1, 2, 1],
            "ego_rel_to_rp": [10, 22, 10, 20, 10],
            # ego 101 has a code-22 partner; ego 303 has a code-20 spouse.
            "ego_rel_to_alter": [22, 22, 20, 20, 10],
            "alter_person_id": [202, 101, 404, 303, 101],
            "alter_sequence": [2, 1, 2, 1, 1],
            "alter_rel_to_rp": [22, 10, 20, 10, 10],
        }
    )
    flag = hcs2.cohabitation_flag(rel_map).set_index(["person_id", "year"])[
        "cohabiting"
    ]
    assert bool(flag.loc[(101, 2000)]) is True  # code-22 partner
    assert bool(flag.loc[(202, 2000)]) is True  # code-22 partner
    assert (303, 2000) not in flag.index  # code-20 spouse -> not cohabiting
    assert (404, 2000) not in flag.index
    # the self-only wave (ego 101, 2001) carries no partner.
    assert (101, 2001) not in flag.index


def test_father_link_births_are_male_parent_biological_events():
    """father_link_births keeps male-parent biological births with a birth
    year; female parents, denial placeholders, and adoptions are dropped."""
    records = pd.DataFrame(
        {
            "parent_person_id": [1, 2, 3, 4, 5],
            "parent_sex": ["male", "female", "male", "male", "male"],
            "record_type": ["birth", "birth", "adoption", "birth", "birth"],
            "birth_year": pd.array(
                [1990, 1991, 1992, pd.NA, 1994], dtype="Int64"
            ),
            "birth_order": pd.array([1, 1, 1, 1, 1], dtype="Int64"),
            "is_event": [True, True, True, True, False],
        }
    )
    out = hcs2.father_link_births(records)
    # only person 1 (male, birth, has year) survives: person 2 female,
    # 3 adoption, 4 NA year, 5 denial placeholder.
    assert out["parent_person_id"].tolist() == [1]
    assert out["birth_year"].tolist() == [1990]
    assert out["parent_person_id"].dtype == np.dtype("int64")
    assert out["birth_year"].dtype == np.dtype("int64")


def test_attach_cohabitation_builds_next_state_within_person():
    pw = pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "year": [2000, 2002, 2000],
            "band": ["25-34", "25-34", "35-44"],
            "sex": ["female", "female", "male"],
            "weight": [1.0, 1.0, 1.0],
            "has_next": [True, False, False],
        }
    )
    cohab = pd.DataFrame(
        {"person_id": [1], "year": [2002], "cohabiting": [True]}
    )
    out = hcs2.attach_cohabitation(pw, cohab).set_index(["person_id", "year"])
    assert bool(out.loc[(1, 2000), "cohabiting"]) is False
    assert bool(out.loc[(1, 2002), "cohabiting"]) is True
    # person 1's 2000 wave sees its next wave (2002) cohabiting == True.
    assert bool(out.loc[(1, 2000), "next_cohabiting"]) is True
    # person 2 never cohabits and has no next wave.
    assert bool(out.loc[(2, 2000), "cohabiting"]) is False
    assert pd.isna(out.loc[(2, 2000), "next_cohabiting"])
