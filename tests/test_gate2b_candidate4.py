"""Tests for the gate-2b candidate-4 one-shot scored run.

Candidate 4 (issue #42 comment 4941160621) is candidate 3 with EXACTLY FOUR
frozen deltas, one per residual mechanism the gate-2b forensics-1 decomposition
(``runs/gate2b_forensics1_v1.json``) quantified:

* **delta 1 -- age-refined cohabitation overlay** (``coresident_spouse``):
  single-year (15-34) code-22 entry/exit hazards replacing the band pair;
* **delta 2 -- legal-spouse residual top-up** (``coresident_spouse``): an
  additive occupancy overlay sized to (ref_code20 - core_legal), 0xC4;
* **delta 3 -- custodial refit (single-year child age x era x father marital)
  + non-family 2+ tail spread** (``coresident_child`` / ``hh_size``);
* **delta 4 -- skip-gen level rebuild** (``coresident_grandchild``): 5-year
  (55+) entry/exit so the stationary stock tracks the raw age-graded stock.

The four candidate-3 cleared families are carried byte-faithfully: their
per-seed scores are IDENTICAL to candidate 3 to bit precision (the deltas
re-fit shape-preserving hazard tables or draw from an isolated 0xC4 stream).
The one-shot outcome (published REGARDLESS of verdict) is pinned below from the
committed artifact ``runs/gate2b_hazard_v4.json``: **FAIL 0/5** -- the spouse
family clears in both directions (0.74 -> 0.97) and hh_size improves
(0.48 -> 0.60), but the male ``coresident_child`` overshoot does NOT drain (the
single-year refit un-averages the band and faithfully reproduces the ~0.95-0.99
young-child coresidence the observable subset carries), ``coresident_grandchild
.55+|female`` is structurally bounded below the reference, and ``hh_size.3``
carries a family-core overshoot the 2+ tail spread does not touch.

Always runnable: it inspects the committed artifact, binds the stored
tolerances to the ratified floor and the locked gates.yaml block, proves the
[20, 46, 5] cube reproduces every score, binds the cleared-family regression,
the byte-identical carried-score check, the c1->c2->c3->c4 progression, the
fit-vs-raw gradient checks, and unit-tests the pure delta derivations. The
reproduction pin lives in ``tests/test_gate2b_candidate4_reproduction.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from populace_dynamics.models import household_composition_sim_v4 as hcs4

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_hazard_v4.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
CANDIDATE3_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v3.json"
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"

N_DRAWS = 20
N_GATED = 46
GATE_SEEDS = [0, 1, 2, 3, 4]
DRAW_SEED_BASE = 5200
REGISTRATION_POINTER = "4941160621"
CANDIDATE3_CLEARED_FAMILIES = (
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
    assert a["schema_version"] == "gate2b_hazard.v4"
    assert a["run"] == "gate2b_hazard_v4"
    assert a["gate"] == "gate_2b"
    assert a["candidate"] == "candidate 4"
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["candidate3_registration_pointer"] == "4939960467"
    assert a["candidate2_registration_pointer"] == "4939456379"
    assert a["candidate1_registration_pointer"] == "4938726107"
    assert a["forensics_artifact"] == "runs/gate2b_forensics1_v1.json"


def test_four_deltas_declared_and_mapped():
    a = _artifact()
    deltas = a["deltas_vs_candidate_3"]
    assert len(deltas) == 4
    assert any("cohabitation" in d for d in deltas)
    assert any("legal-spouse residual" in d for d in deltas)
    assert any("custodial" in d and "2+" in d for d in deltas)
    assert any("skip-gen" in d for d in deltas)
    mapping = a["per_delta_target_family"]
    assert mapping["delta_1_age_refined_cohabitation"] == "coresident_spouse"
    assert mapping["delta_2_legal_spouse_residual"] == "coresident_spouse"
    assert mapping["delta_3_custodial_refit_and_nonfamily_tail"] == [
        "coresident_child",
        "hh_size",
    ]
    assert mapping["delta_4_skipgen_level_rebuild"] == "coresident_grandchild"


def test_one_shot_verdict_pinned_fail_0_of_5():
    """The committed one-shot outcome: gate FAIL, 0 of 5 seeds pass."""
    v = _artifact()["verdict"]
    assert v["gate_2b_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert v["n_gate_seeds"] == 5
    assert v["n_gated_cells"] == N_GATED
    assert all(p is False for p in v["seed_pass"].values())


def test_per_seed_gated_pass_counts_pinned():
    """Committed per-seed gated pass counts (40/39/38/40/40)."""
    by_seed = {s["seed"]: s for s in _artifact()["per_seed"]}
    assert {s: by_seed[s]["n_gated_pass"] for s in GATE_SEEDS} == {
        0: 40,
        1: 39,
        2: 38,
        3: 40,
        4: 40,
    }


def test_chronic_failures_are_child_grandchild_and_hh_size_3():
    """Every seed fails coresident_child male, grandchild 55+|female, and
    hh_size.3 -- the residual mechanisms the four deltas do not resolve
    (delta 3a does not drain the observable-subset young-child coresidence;
    grandchild 55+|female is structurally bounded; hh_size.3 is a family-core
    overshoot). This caps the gate at 0/5 regardless of the spouse clear."""
    a = _artifact()
    for s in a["per_seed"]:
        fails = {c for c, rec in s["gated_cells"].items() if not rec["pass"]}
        assert "coresident_child.25-34|male" in fails, s["seed"]
        assert "coresident_child.35-44|male" in fails, s["seed"]
        assert "coresident_grandchild.55+|female" in fails, s["seed"]
        assert "hh_size.3" in fails, s["seed"]


def test_forecast_recorded_and_not_graded_here():
    f = _artifact()["pre_registered_forecast"]
    assert f["p_gate_pass_4_of_5"] == "0.50-0.65"
    assert "grading_note" in f
    assert "does NOT grade" in f["grading_note"]


def test_spec_resolution_notes_present():
    notes = _artifact()["spec_resolution_notes"]
    for key in (
        "rng_byte_identical_carried_families",
        "delta_1_age_refined_cohabitation",
        "delta_2_legal_spouse_residual",
        "delta_3a_custodial_refit",
        "delta_3b_nonfamily_tail_spread",
        "delta_4_skipgen_level_rebuild",
        "carried_families_byte_faithful",
        "observed_initial_states_are_the_holdout_persons_own",
    ):
        assert key in notes and notes[key]
    # The delta-3a resolution documents the honest negative (no drain).
    assert "does NOT drain" in notes["delta_3a_custodial_refit"]
    # The delta-4 resolution documents the structural bound.
    assert "structurally bounded" in notes["delta_4_skipgen_level_rebuild"]


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
    chk = _artifact()["comparison_across_candidates"][
        "cleared_family_regression_check"
    ]
    assert chk["all_cleared_families_still_clear"] is True
    assert set(chk["families"]) == set(CANDIDATE3_CLEARED_FAMILIES)
    for fam in CANDIDATE3_CLEARED_FAMILIES:
        assert chk["detail"][fam]["candidate4_pass_rate"] == 1.0
        assert chk["detail"][fam]["still_clears"] is True


def test_carried_family_scores_byte_identical_to_candidate_3():
    """Every carried cell's per-seed gated score equals candidate 3's, cell
    for cell and seed for seed. coresident_spouse is NOT carried (deltas 1 and
    2 target it), so it is excluded from the carried set."""
    a = _artifact()
    byt = a["comparison_across_candidates"][
        "byte_identical_carried_family_score_check"
    ]
    assert byt["byte_identical"] is True
    assert byt["max_abs_score_deviation_vs_candidate3"] == 0.0
    c3 = json.loads(CANDIDATE3_ARTIFACT.read_text())
    by_seed_4 = {s["seed"]: s for s in a["per_seed"]}
    by_seed_3 = {s["seed"]: s for s in c3["per_seed"]}
    carried_cells = [
        c
        for c in _gate2b_tolerances()
        if c.startswith(("coresident_parent.", "multigen.", "parental_home_"))
        or c in ("multigen_entry", "multigen_exit")
    ]
    assert set(carried_cells) == set(byt["carried_cells"])
    assert not any(c.startswith("coresident_spouse.") for c in carried_cells)
    for seed in GATE_SEEDS:
        for cell in carried_cells:
            s4 = by_seed_4[seed]["gated_cells"][cell]["score"]
            s3 = by_seed_3[seed]["gated_cells"][cell]["score"]
            assert s4 == pytest.approx(s3, abs=1e-12), (seed, cell)


def test_progression_recomputes_c1_to_c4():
    a = _artifact()
    prog = a["comparison_across_candidates"]["per_family_progression"]
    decomp = a["per_family_decomposition"]
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())[
        "per_family_decomposition"
    ]
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())[
        "per_family_decomposition"
    ]
    c3 = json.loads(CANDIDATE3_ARTIFACT.read_text())[
        "per_family_decomposition"
    ]
    for fam, block in prog.items():
        assert (
            block["candidate4_pass_rate"] == decomp[fam]["cell_seed_pass_rate"]
        )
        assert block["candidate1_pass_rate"] == c1[fam]["cell_seed_pass_rate"]
        assert block["candidate2_pass_rate"] == c2[fam]["cell_seed_pass_rate"]
        assert block["candidate3_pass_rate"] == c3[fam]["cell_seed_pass_rate"]
        expected = round(
            block["candidate4_pass_rate"] - block["candidate3_pass_rate"], 4
        )
        assert block["delta_c3_to_c4"] == pytest.approx(expected, abs=1e-9)


def test_spouse_family_improves_and_hh_size_improves_over_c3():
    """Deltas 1 and 2 lift the spouse family (0.74 -> 0.97); delta 3b lifts
    hh_size (0.48 -> 0.60). No target family regresses vs candidate 3."""
    prog = _artifact()["comparison_across_candidates"][
        "per_family_progression"
    ]
    assert prog["coresident_spouse"]["delta_c3_to_c4"] > 0
    assert prog["coresident_spouse"]["candidate4_pass_rate"] >= 0.9
    assert prog["hh_size"]["delta_c3_to_c4"] > 0
    # The unresolved families do not regress (honest flat, recorded).
    assert prog["coresident_child"]["delta_c3_to_c4"] >= 0
    assert prog["coresident_grandchild"]["delta_c3_to_c4"] >= 0


def test_spouse_older_male_cells_clear_via_legal_residual():
    """Delta 2 clears the 65+ male spouse cells the certified core
    under-produced (the forensics' older-male undershoot)."""
    a = _artifact()
    for cell in ("coresident_spouse.65-74|male", "coresident_spouse.75+|male"):
        passes = sum(s["gated_cells"][cell]["pass"] for s in a["per_seed"])
        assert passes >= 4, cell


# --------------------------------------------------------------------------
# Delta stats + fit-vs-raw checks recorded
# --------------------------------------------------------------------------
def test_delta_stats_recorded_for_all_four_deltas():
    for s in _artifact()["per_seed"]:
        ds = s["delta_stats"]
        assert set(ds) == {
            "delta_1_age_refined_cohabitation",
            "delta_2_legal_spouse_residual",
            "delta_3a_custodial_refit",
            "delta_3b_nonfamily_tail_spread",
            "delta_4_skipgen_level_rebuild",
        }
        d1 = ds["delta_1_age_refined_cohabitation"]
        assert d1["cohab_single_year_window"] == [15, 34]
        assert d1["n_cohab_person_waves_simulated"] > 0
        d2 = ds["delta_2_legal_spouse_residual"]
        assert d2["n_legal_residual_person_waves_simulated"] > 0
        assert d2["n_bands_active"] > 0
        d3a = ds["delta_3a_custodial_refit"]
        assert d3a["n_age_era_marital_cells"] > 0
        assert d3a["custodial_n_train_coresident"] > 0
        d3b = ds["delta_3b_nonfamily_tail_spread"]
        assert d3b["mean_nonfamily_count_within_2plus_simulated"] > 2.0
        d4 = ds["delta_4_skipgen_level_rebuild"]
        assert len(d4["skipgen_entry_5yr_female"]) == 6


def test_fit_vs_raw_checks_reproduce_forensics_gradients():
    fvr = _artifact()["fit_vs_raw_checks"]["checks"]
    # Delta 1: the age-graded overlay reproduces a STEEP within-15-24 gradient
    # (the band-constant hazard is ~1x); male >> female, matching the raw.
    d1 = fvr["delta_1_cohab_single_year_gradient"]
    assert d1["male"]["fitted_within_15_24_old_over_young_ratio"] > 20.0
    assert (
        d1["male"]["fitted_within_15_24_old_over_young_ratio"]
        > d1["female"]["fitted_within_15_24_old_over_young_ratio"]
    )
    assert d1["male"]["raw_within_15_24_old_over_young_ratio"] > 100.0
    # Delta 3b: the fitted 2+ mean matches the forensics train 2+ mean.
    d3b = fvr["delta_3b_nonfamily_2plus_mean"]
    assert d3b["fitted_mean_count_within_2plus"] == pytest.approx(
        d3b["forensics_mean_within_2plus_households"], abs=0.05
    )
    # Delta 4: the 5-year stationary stock peaks in the 65-69 band (tracking
    # the raw 65-74 peak), as the level rebuild intends.
    d4 = fvr["delta_4_skipgen_5yr_stationary"]["fitted_5yr_female"]
    assert (
        d4["65-69"]["fitted_stationary_stock"]
        > d4["55-59"]["fitted_stationary_stock"]
    )


def test_model_records_delta_module_and_windows():
    a = _artifact()
    assert a["model"]["family_transitions_spec"] == "candidate16_registry_v1"
    assert (
        a["model"]["delta_module"]
        == "populace_dynamics.models.household_composition_sim_v4"
    )
    assert a["model"]["cohab_single_year_window"] == [15, 34]
    assert a["model"]["custodial_era_slices"] == [
        "pre-1997",
        "1997-2009",
        "2010-2023",
    ]
    assert a["model"]["legal_spouse_code"] == 20


def test_per_family_decomposition_covers_all_gated_cells():
    a = _artifact()
    decomp = a["per_family_decomposition"]
    covered: set[str] = set()
    for fam in decomp.values():
        covered.update(fam["cells"])
        assert fam["mechanism"]
    assert covered == set(_gate2b_tolerances())


# --------------------------------------------------------------------------
# Pure delta derivations (synthetic frames; no PSID needed)
# --------------------------------------------------------------------------
def test_era_of_year_locked_floor_slices():
    assert hcs4.era_of_year(1980) == "pre-1997"
    assert hcs4.era_of_year(1996) == "pre-1997"
    assert hcs4.era_of_year(1997) == "1997-2009"
    assert hcs4.era_of_year(2009) == "1997-2009"
    assert hcs4.era_of_year(2010) == "2010-2023"
    assert hcs4.era_of_year(2023) == "2010-2023"
    assert hcs4.CUSTODIAL_ERA_SLICES == (
        "pre-1997",
        "1997-2009",
        "2010-2023",
    )


def test_legal_spouse_flag_uses_code_20_only():
    """The legal-spouse flag fires on MX8 code 20 (legal spouse) but NOT on
    code 22 (cohabiting partner) -- the cohabitation overlay carries that."""
    rel_map = pd.DataFrame(
        {
            "interview_year": [2000, 2000, 2000],
            "ego_person_id": [10, 20, 30],
            "ego_rel_to_alter": [20, 22, 40],  # legal, partner, sibling
            "alter_person_id": [11, 21, 31],
        }
    )
    flag = hcs4.legal_spouse_flag(rel_map).set_index(["person_id", "year"])
    assert bool(flag.loc[(10, 2000), "legal_spouse_obs"]) is True
    assert (20, 2000) not in flag.index  # code-22 partner excluded
    assert (30, 2000) not in flag.index  # sibling excluded


def test_fit_cohab_single_year_is_single_year_with_band_fallback():
    """A dense single-year age gets its own entry rate; a sparse one falls
    back to the carried band hazard."""
    n = 400
    pw = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "year": np.full(n, 2000),
            "age": np.full(n, 20),
            "band": np.full(n, "15-24"),
            "sex": np.full(n, "male"),
            "weight": np.ones(n),
            "has_next": np.ones(n, dtype=bool),
        }
    )
    # 400 age-20 males not cohabiting; half enter cohabitation next wave.
    cohab_flag = pd.DataFrame(
        {"person_id": [], "year": [], "cohabiting": []}
    ).astype({"person_id": "int64", "year": "int64", "cohabiting": "bool"})
    pw2 = pw.copy()
    pw2["cohabiting"] = False
    pw2 = pw2.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw2["next_cohabiting"] = np.concatenate(
        [np.ones(200, dtype=bool), np.zeros(200, dtype=bool)]
    )
    # Monkeypatch attach to return our frame (single wave per person).
    import populace_dynamics.models.household_composition_sim_v2 as hcs2

    orig = hcs2.attach_cohabitation
    hcs2.attach_cohabitation = lambda a, b: pw2
    try:
        entry, exit_ = hcs4.fit_cohab_single_year(
            pw, cohab_flag, set(range(n)), {("15-24", "male"): 0.9}, {}
        )
    finally:
        hcs2.attach_cohabitation = orig
    # age-20 male is dense (200 not-cohab at-risk) -> fitted ~0.5, not the 0.9
    # band fallback; a never-observed age (e.g. 30 outside 15-34) is absent.
    assert entry[(20, "male")] == pytest.approx(0.5, abs=1e-9)
    assert entry[(16, "male")] == pytest.approx(0.9, abs=1e-9)  # sparse -> fb
    assert (30, "male") in entry  # 15-34 window includes 30 (band fallback)
    assert (40, "male") not in entry  # outside the single-year window


def test_fit_nonfamily_2plus_distribution_conditions_on_ge2():
    """The 2+ count distribution is the weighted distribution of the actual
    non-family count among count>=2 waves (support 2, 3, 4, ...)."""
    pw = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "year": [2000] * 6,
            "band": ["25-34"] * 6,
            "sex": ["male"] * 6,
            "weight": [1.0] * 6,
            "hh_size": [1, 2, 3, 4, 5, 6],  # non-family = hh_size - 1
        }
    )
    fu = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "year": [2000] * 6,
            "family_unit_size": [1] * 6,
        }
    )
    dist, overall, diag = hcs4.fit_nonfamily_2plus(pw, fu, set(range(1, 7)))
    counts, cum = dist[("25-34", "male")]
    # residuals >=2 are {2,3,4,5} (persons 3-6), each weight 1 -> uniform.
    assert counts.tolist() == [2, 3, 4, 5]
    assert cum[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(np.concatenate([[0.0], cum])), 0.25)
    assert diag["true_weighted_mean_count_train"] == pytest.approx(
        (0 + 1 + 2 + 3 + 4 + 5) / 6, abs=1e-9
    )


def test_sample_nonfamily_v4_spreads_the_2plus_tail():
    """The 2+ class draws its count from the fitted 2+ distribution (mass
    beyond 2), while the 0/1 classes are unchanged."""
    pw = pd.DataFrame(
        {
            "band": ["25-34"] * 2000,
            "sex": ["male"] * 2000,
        }
    )

    class _M:
        base_v3 = type(
            "b", (), {"nonfamily": {("25-34", "male"): (0.0, 0.0, 1.0)}}
        )()
        # 2+ count is 3 or 5 with equal mass.
        nonfamily_2plus = {
            ("25-34", "male"): (
                np.array([3, 5]),
                np.cumsum([0.5, 0.5]),
            )
        }

    contrib = hcs4._sample_nonfamily_v4(
        pw, _M(), np.random.default_rng(0), np.random.default_rng(1)
    )
    # all in the 2+ class (q2=1.0) -> counts are 3 or 5, never the minimal 2.
    assert set(np.unique(contrib)) <= {3, 5}
    assert (contrib >= 3).all()
    assert contrib.mean() == pytest.approx(4.0, abs=0.2)


def test_skipgen_5yr_bands_a_priori():
    assert hcs4.SKIPGEN_AGE_BANDS_55PLUS == (
        (55, 59),
        (60, 64),
        (65, 69),
        (70, 74),
        (75, 79),
        (80, 120),
    )


def test_custodial_prob_graded_fallback():
    """The custodial lookup falls back (age, era, marital) -> (age, marital)
    -> (band, marital) -> overall."""

    class _M:
        custodial_era = {(2, "2010-2023", "married"): 0.9}
        custodial_age_marital = {(2, "married"): 0.7}
        custodial_band_marital = {("0-4", "married"): 0.5}
        custodial_overall = 0.3

    m = _M()
    assert hcs4._custodial_prob(m, 2, "2010-2023", "married") == 0.9
    assert hcs4._custodial_prob(m, 2, "1997-2009", "married") == 0.7  # age fb
    assert hcs4._custodial_prob(m, 3, "2010-2023", "married") == 0.5  # band fb
    assert (
        hcs4._custodial_prob(m, 3, "2010-2023", "not_married") == 0.3
    )  # overall
