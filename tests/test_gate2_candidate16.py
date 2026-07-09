"""Tests for the gate-2 candidate-16 (run 1) pre-registered run.

Candidate 16 is the SIXTEENTH pre-registered gate-2 candidate: candidate 15's
frozen spec (comment 4928232089, ``scripts/run_gate2_candidate15.py``, merged
#107 -- the untrended seven-band surviving-spouse widowhood level, the c13/c14
band structure) verbatim EXCEPT EXACTLY ONE delta, registered from gate-2
forensics 4 (#108) on issue #42 comment 4929419524:

* THE DELTA -- the surviving-spouse widowhood hazard (seven-band x sex) is
  additionally conditioned on the OBSERVED binary support-composition stratum:
  whether the person's observed support window extends to at least age 75
  (``min(last_wave, censor_year) >= birth_year + 75``, known ex ante). Both
  strata are train-estimated per age band x sex under the existing smoothing
  (``transitions._hazard_by_band`` weighted hazard, no add-one); the two strata
  recombine to candidate 15's band AGGREGATE by the exposure-weighted identity
  ``(den_0*rate_0 + den_1*rate_1)/(den_0+den_1) == (num_0+num_1)/(den_0+den_1)
  == rate_aggregate`` (the binary person-level stratum partitions the exposure
  and events exactly), so aggregate incidence is preserved while event
  composition matches the reference's observed-window correlation.

The delta threads a per-person conditioning covariate the widowhood hazard must
SEE at simulation time, so ``_widow_probs``, ``_build_sim_lookups`` and
``simulate_holdout`` DIVERGE (re-implemented) and ``fit_components`` fits the
stratified level; candidate 16 REUSES candidate 15's EXACT code objects for
``_draw_moments``, ``score_seed``, ``fit_remarriage_age_banded`` and
``_remarriage_probs_age_banded`` (rebound so the reused ``_draw_moments`` /
``score_seed`` call candidate 16's ``simulate_holdout`` / ``fit_components``).

Conditioning on the stratum is RNG-neutral (the uniform draw precedes
``_widow_probs``; only the competing-risk threshold value moves), so the
marital-state-independent cells (``asfr.*``, ``completed_fertility.*``,
``first_marriage.*``) are byte-identical to candidate 15 draw-by-draw
(``test_delta_untouched_cells_byte_identical_to_c15``). The stratum reshapes
every widowhood band's competition, so all widowhood-incidence and
widowed-stock cells move (``test_widowhood_trajectory_cells_moved_vs_c15``).

The one-shot outcome (published REGARDLESS of verdict) is pinned below from the
committed artifact: **PASS 4/5** -- the first passing gate-2 run in the ladder.
Frozen spec: issue #42 comment 4929419524.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_hazard_v16.json"
ARTIFACT_C15 = ROOT / "runs" / "gate2_hazard_v15.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
FORENSICS3 = ROOT / "runs" / "gate2_forensics3_v1.json"
FORENSICS4 = ROOT / "runs" / "gate2_forensics4_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4929419524"
)
SPEC_URL_C15 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4928232089"
)
REGISTRATION_POINTER = "4929419524"
GATE_SEEDS = [0, 1, 2, 3, 4]
N_DRAWS = 20
DRAW_SEED_BASE = 5200
N_GATED = 46
N_REPORT_ONLY = 16

COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
MODAL_CELL = "share_widowed.75+|female"
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
WIDOWHOOD_INCIDENCE_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "widowhood.75+|female",
    "widowhood.45+|male",
)

# The seven-band widowhood table (candidate 15's; UNCHANGED by candidate 16 --
# the delta is the support-composition stratum, not the bands).
WIDOW_BANDS = (
    (18, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 84),
    (85, 120),
)

# The committed NCHS betas (candidate 5's frozen values; documented, not
# applied -- candidate 15's trend removal inherited).
NCHS_BETA = {
    "female": -0.009234704865961198,
    "male": -0.010643975395626533,
}

# The reused candidate-15 code objects (byte-identity chain). ``_widow_probs``,
# ``_build_sim_lookups`` and ``simulate_holdout`` are NOT here -- they moved to
# the diverged set (the delta threads a per-person covariate).
REUSED_CODE_OBJECT_NAMES = (
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
# The RE-IMPLEMENTED functions (the delta): NOT candidate 15's bytecode.
DIVERGED_CODE_OBJECT_NAMES = (
    "_widow_probs",
    "_build_sim_lookups",
    "simulate_holdout",
    "fit_components",
)

# Cells the delta does not change (RNG-neutral): the widowhood threshold changes
# but the scored RNG is drawn over state-independent active/fertile populations
# and the never-married population's first-marriage timing is unaffected, so
# these are byte-identical to candidate 15 draw-by-draw.
DELTA_UNTOUCHED_PREFIXES = (
    "asfr.",
    "completed_fertility.",
    "first_marriage.",
)

# ==========================================================================
# One-shot outcome pins (from runs/gate2_hazard_v16.json; published REGARDLESS
# of verdict).
# ==========================================================================
# PASS 4/5 -- the FIRST passing gate-2 run in the candidate ladder. Conditioning
# the surviving-spouse widowhood hazard on the observed support-composition
# stratum closed the forensics-4 Q9 survival-to-75+ yield leak: the 75+ widowed
# stock lifted from 0.841 to 0.914 of reference (train yield 0.842 -> 0.911) and
# share_widowed.75+|female cleared ALL five seeds (candidate 15 failed seeds 2
# and 3; both flipped to pass). Both marriage counts held 5/5 with positive
# margin (the registered modal count re-clip did NOT materialise). The only
# failing cell is seed 2's completed_fertility.c1970s -- the RNG-isolated split
# artifact forensics 4 Q8 identified, byte-identical to candidate 15, unmovable
# -- so the gate passes 4/5 on the registered pass path seeds 0, 1, 3, 4.
EXPECTED_GATE_PASS = True
EXPECTED_N_SEEDS_PASS = 4
EXPECTED_SEED_PASS = {
    "0": True,
    "1": True,
    "2": False,
    "3": True,
    "4": True,
}
EXPECTED_SEED_FAILS = {
    0: set(),
    1: set(),
    2: {"completed_fertility.c1970s"},
    3: set(),
    4: set(),
}
# The 75+ widow-stock modal cleared 3/5 -> 5/5 (seeds 2 and 3 flipped to pass);
# both marriage counts held candidate 15's 5/5.
EXPECTED_MODAL_MOVEMENT = {"c15": 3, "c16": 5}
EXPECTED_STOCK_FLIPPED_TO_PASS = [2, 3]
EXPECTED_COUNT_MOVEMENT = {
    "mean_lifetime_marriages|male": {"c15": 5, "c16": 5},
    "mean_lifetime_marriages|female": {"c15": 5, "c16": 5},
}
# The share_widowed.75+|female cell tolerance and the per-seed scores -- every
# seed now well inside the 0.185 tolerance (candidate 15's seeds 2/3 cleared).
STOCK_TOLERANCE = 0.185
STOCK_SCORES = {
    0: 0.08952,
    1: 0.05746,
    2: 0.10706,
    3: 0.11097,
    4: 0.08691,
}
# The designed lift (seed-mean sim/ref vs candidate 15): incidence preserved
# (the recombination identity holds the band aggregate); the stock lifts toward
# reference.
EXPECTED_INCIDENCE_SIM_REF = {"c15": 1.0599, "c16": 1.0614}
EXPECTED_STOCK_SIM_REF = {"c15": 0.8411, "c16": 0.9137}
# The forensics-4 yield before/after (train-side, side B): the aggregate 75+
# widowed-stock ratio and the 50-64-onset survival-to-75+ yield / window-
# reaches-75 share move toward reference.
EXPECTED_YIELD_AGG = {"before_c15": 0.8419, "after_c16": 0.9114}
EXPECTED_YIELD_5064 = {
    "yield_before": 0.581,
    "yield_after": 0.816,
    "win75_ref": 0.572,
    "win75_before": 0.391,
    "win75_after": 0.551,
}
# The support-composition stratum split (seed-independent; from the seed-0 train
# fit's stored provenance).
EXPECTED_STRATUM = {
    "n_window_reaches_75plus": 3147,
    "n_window_below_75": 38262,
    "n_no_observed_support": 0,
}
# completed_fertility.c1970s: seed 2 fails (0.1816 > 0.171); all others pass.
FERT_C1970S_SEED2_SCORE = 0.18155
FERT_TOLERANCE = 0.171
# Seed-0 draw-0 (default_rng(5200)) single-draw rates, live-reproducible on seed
# 0's side A. first_marriage.25-34|female is byte-identical to candidate 15 (the
# never-married population is unaffected); the marriage counts, remarriage and
# widowhood.75+|female MOVE off candidate 15 (the stratum reshaped every
# widowhood band).
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.4200975724149578,
    "mean_lifetime_marriages|female": 1.409717269294589,
    "remarriage.after_divorce": 0.06013215598787471,
    "first_marriage.25-34|female": 0.07771323726223815,
    "widowhood.75+|female": 0.06880740391315175,
}
# candidate 15's committed seed-0 draw-0 75+ widowhood incidence (the stratum
# moved it -- the clearest live proof the delta is non-inert).
SEED0_DRAW0_C15_WIDOW75 = 0.06859212436250475
# The exact set of cells that moved off candidate 15 (any seed, any draw). Like
# candidate 15's trend removal, the stratum touches every widowhood band, so the
# widowhood incidence, the widowed / divorced stocks, the remarriage flows and
# the marriage counts move -- 31 cells; the state-independent cells do NOT.
EXPECTED_MOVED_CELLS = {
    "divorce.dur0-4",
    "divorce.dur10-19",
    "divorce.dur20+",
    "divorce.dur5-9",
    "mean_lifetime_marriages|female",
    "mean_lifetime_marriages|male",
    "remarriage.after_divorce",
    "remarriage.after_widowhood",
    "remarriage.widowed_60plus",
    "remarriage.widowed_under60",
    "remarriage.ysd0-4",
    "remarriage.ysd10+",
    "remarriage.ysd5-9",
    "share_divorced.45-54|female",
    "share_divorced.45-54|male",
    "share_divorced.55-64|female",
    "share_divorced.55-64|male",
    "share_widowed.65-74|female",
    "share_widowed.65-74|male",
    "share_widowed.75+|female",
    "share_widowed.75+|male",
    "widowhood.45+|male",
    "widowhood.45-54|female",
    "widowhood.45-54|male",
    "widowhood.45-64|female",
    "widowhood.55-64|female",
    "widowhood.55-64|male",
    "widowhood.65-74|female",
    "widowhood.65-74|male",
    "widowhood.75+|female",
    "widowhood.75+|male",
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c15() -> dict:
    return json.loads(ARTIFACT_C15.read_text())


def _gate2_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_2"]["thresholds"]


def _gate2_tolerances() -> dict[str, float]:
    tol: dict[str, float] = {}
    for view in _gate2_thresholds()["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate16 as runner

    return runner


def _import_c15():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate15 as c15

    return c15


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    runner = _import_runner()
    assert runner.SPEC_REGISTRATION == SPEC_URL
    assert runner.REGISTRATION_POINTER == REGISTRATION_POINTER
    assert runner.CANDIDATE15_REGISTRATION == SPEC_URL_C15
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    assert runner.SIM_SEED_BASE == 4200
    assert runner.SUPPORT_STRATUM_AGE == 75
    assert runner.SUPPORT_STRATA == (0, 1)
    # The widowhood band table is candidate 15's, UNCHANGED (the delta is the
    # support-composition stratum, not the bands).
    assert runner.WIDOW_BANDS == WIDOW_BANDS
    assert runner.WIDOW_BANDS == _import_c15().WIDOW_BANDS
    # the committed NCHS betas are retained (documented, not applied).
    assert runner.NCHS_BETA_BY_SEX_COMMITTED == NCHS_BETA
    # the 5-band remarriage table and delta-1 count are candidate 15's,
    # inherited verbatim.
    assert runner.REM_AGE_BANDS == _import_c15().REM_AGE_BANDS
    assert (
        runner.observed_residual_counts
        is _import_c15().observed_residual_counts
    )


def test_reused_and_diverged_code_objects_vs_c15():
    """The reused chain is candidate 15's exact bytecode; _widow_probs,
    _build_sim_lookups, simulate_holdout and fit_components diverge.

    The crux structural fact: ``simulate_holdout`` / ``_build_sim_lookups`` were
    REUSED in candidate 15 (the trend removal lived inside ``_widow_probs``) but
    DIVERGE in candidate 16 -- the support-composition stratum is a per-person
    covariate the widowhood hazard must SEE at simulation time, so it is
    threaded through ``simulate_holdout`` into ``_widow_probs`` and a stratum
    axis through ``_build_sim_lookups``. The reused ``_draw_moments`` /
    ``score_seed`` stay candidate 15's exact code objects and call candidate
    16's ``simulate_holdout`` / ``fit_components`` by global name.
    """
    runner = _import_runner()
    c15 = _import_c15()
    for name in REUSED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is getattr(c15, name).__code__
        ), f"{name} must reuse candidate 15's exact code object"
        assert getattr(runner, name).__globals__ is vars(runner)
    for name in DIVERGED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is not getattr(c15, name).__code__
        ), f"{name} must be re-implemented for the candidate-16 delta"
    # THE FLIP: simulate_holdout / _build_sim_lookups were reused in candidate
    # 15, diverge in candidate 16.
    assert "simulate_holdout" in c15.REUSED_CODE_OBJECT_NAMES
    assert "_build_sim_lookups" in c15.REUSED_CODE_OBJECT_NAMES
    assert "simulate_holdout" not in runner.REUSED_CODE_OBJECT_NAMES
    assert "simulate_holdout" in runner.DIVERGED_CODE_OBJECT_NAMES
    assert "_build_sim_lookups" in runner.DIVERGED_CODE_OBJECT_NAMES
    # the schema blocks are import-bound from candidate 15 unchanged.
    for name in (
        "_per_draw_per_cell_rates_block",
        "_undefined_draw_block",
        "_per_draw_dispersion_block",
    ):
        assert getattr(runner, name) is getattr(c15, name)
    pins = _artifact()["revision_pins"]
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )
    assert all(
        pins["diverged_code_objects_vs_candidate15"][n]
        for n in DIVERGED_CODE_OBJECT_NAMES
    )
    assert pins["nchs_trend_applied_candidate16"] is False
    assert pins["support_stratum_threshold_age"] == 75


def test_widow_probs_stratum_conditioned_and_year_invariant():
    """Structural: candidate 16's _widow_probs distinguishes the support strata
    and is year-invariant (no trend; candidate 15's removal inherited).

    Always-runnable (synthetic [band, sex, stratum] mort_arr -- no PSID). The
    returned rate depends on the ``stratum`` index; stratum 1 (window reaches
    75) selects the higher-hazard slab; ``year`` / ``beta_arr`` do not enter.
    The single, precise proof that the one delta is the support-composition
    conditioning.
    """
    runner = _import_runner()
    mort = np.zeros((7, 2, 2), dtype=np.float64)
    mort[:, :, 0] = 0.01
    mort[:, :, 1] = 0.05
    beta = np.array([NCHS_BETA["female"], NCHS_BETA["male"]], dtype=np.float64)
    age = np.array([40.0, 80.0, 90.0])
    egom = np.array([0.0, 0.0, 1.0])
    sp = np.array([42.0, 82.0, 88.0])
    spm = 1.0 - egom
    s0 = np.zeros(3, dtype=np.int64)
    s1 = np.ones(3, dtype=np.int64)
    p_s0 = runner._widow_probs(age, egom, sp, spm, 2020, mort, beta, s0)
    p_s1 = runner._widow_probs(age, egom, sp, spm, 2020, mort, beta, s1)
    # the stratum selects the slab: stratum 0 -> 0.01, stratum 1 -> 0.05.
    assert np.allclose(p_s0, 0.01)
    assert np.allclose(p_s1, 0.05)
    assert not np.allclose(p_s0, p_s1)  # non-inert
    # year-invariant (no trend enters the lookup).
    p_anchor = runner._widow_probs(age, egom, sp, spm, 1995, mort, beta, s1)
    assert np.array_equal(p_s1, p_anchor)
    # beta_arr does not enter either.
    p_zero_beta = runner._widow_probs(
        age, egom, sp, spm, 2020, mort, np.zeros(2), s1
    )
    assert np.array_equal(p_s1, p_zero_beta)


def test_delta_string_names_the_one_delta():
    runner = _import_runner()
    d = runner.DELTA_VS_CANDIDATE15.lower()
    assert "one delta" in d or "exactly one" in d
    assert "support" in d and "stratum" in d
    assert "age 75" in d or "birth_year + 75" in d
    assert "byte-identical" in d
    assert "candidate 15" in d
    assert "recombine" in d or "recombination" in d


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v16"
    assert a["candidate"] == "candidate 16"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_delta_recorded():
    a = _artifact()
    assert a["candidate15_registration"] == SPEC_URL_C15
    assert a["forensics4_registration"].endswith("4928761676")
    model = a["model"]
    comp = model["components"]
    assert "stratum" in comp["widowhood"].lower()
    assert "support" in comp["widowhood"].lower()
    assert "byte-identical" in comp["fertility"].lower()
    assert "byte-identical" in comp["remarriage"].lower()
    assert "untouched" in comp["spousal_age_gap"].lower()
    assert "byte-identical" in comp["entry_widowed_initial_state"].lower()
    assert "support_composition_stratum" in comp


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.6-0.7"
    assert f["pass_path_seeds"] == [0, 1, 3, 4]
    assert f["registration"] == SPEC_URL


def test_precheck_reproduced_exactly():
    a = _artifact()
    pre = a["precheck"]
    assert pre["all_reproduced_exactly"] is True
    assert pre["reference_moments_exact"] is True
    assert pre["rate_a_exact"] is True
    assert pre["holdout_sha256_all_match"] is True
    assert pre["reference_moments_max_abs_deviation"] <= 1e-12
    assert pre["rate_a_max_abs_deviation"] <= 1e-12


def test_delta1_count_reconciliation_recorded():
    a = _artifact()
    rec = a["delta1_reconciliation"]
    assert rec["reconciled"] is True
    assert rec["per_person_identity_max_abs_residual"] <= 1e-12
    assert rec["aggregate_reconciliation_max_abs_remainder"] <= 1e-9
    assert rec["residual_nonnegative"] is True
    assert len(rec["per_seed"]) == len(GATE_SEEDS)


def test_entry_widowed_reconciliation_to_forensics3():
    """Candidate 12's entry-widowed classification (inherited) reconciles."""
    a = _artifact()
    rec = a["entry_widowed_reconciliation"]
    assert rec["reconciled"] is True
    assert rec["max_abs_remainder"] <= 1e-9
    committed = {
        s["seed"]: s["ref_support_taxonomy"]["initial_state_fixable_share"]
        for s in json.loads(FORENSICS3.read_text())["per_seed"]
    }
    for row in rec["per_seed"]:
        assert row["forensics3_committed_fixable_share"] == pytest.approx(
            committed[row["seed"]], abs=1e-12
        )


# --------------------------------------------------------------------------
# THE DELTA: the support-composition stratum + the recombination identity
# --------------------------------------------------------------------------
def test_recombination_identity_reconciles_to_zero():
    """The exposure-weighted recombination identity is the hard structural gate.

    The binary person-level stratum partitions the married person-year exposure
    and the widowhood events exactly, so within each band x sex the two strata
    recombine to candidate 15's band AGGREGATE rate: ``(den_0*rate_0 +
    den_1*rate_1)/(den_0+den_1) == (num_0+num_1)/(den_0+den_1) ==
    rate_aggregate``. The max residual (vs the aggregate AND vs candidate 15's
    applied rate) must close to 0.0. Recomputes the identity cell-by-cell from
    the stored strata rates and exposures.
    """
    a = _artifact()
    recomb = a["recombination_identity"]
    assert recomb["reconciled"] is True
    assert recomb["n_cells"] == 14  # 7 bands x 2 sexes
    assert recomb["max_abs_residual_recombined_vs_aggregate"] <= 1e-9
    assert recomb["max_abs_residual_recombined_vs_candidate15_applied"] <= 1e-9
    # Recompute the exposure-weighted recombination from the strata rates /
    # exposures, cell by cell.
    for key, cell in recomb["cells"].items():
        s0 = cell["stratum0_window_below_75"]
        s1 = cell["stratum1_window_reaches_75plus"]
        den = s0["den_wt"] + s1["den_wt"]
        recombined = (
            (s0["den_wt"] * s0["rate"] + s1["den_wt"] * s1["rate"]) / den
            if den > 0
            else 0.0
        )
        assert recombined == pytest.approx(
            cell["aggregate_rate"], abs=1e-9
        ), key
        assert recombined == pytest.approx(
            cell["candidate15_applied_rate"], abs=1e-9
        ), key
        # num_wt / den_wt partition exactly (the two strata sum to the
        # aggregate).
        assert (s0["num_wt"] + s1["num_wt"]) == pytest.approx(
            cell["aggregate_rate"] * den, abs=1e-6
        )


def test_recombination_matches_candidate15_aggregate_level():
    """Every recombined rate equals candidate 15's committed widowhood level.

    Candidate 15's applied aggregate widowhood level is the recombination
    target; the artifact's per-cell ``candidate15_applied_rate`` must match
    candidate 15's committed ``mortality_level_new_widowhood`` (fitted on the
    same seed-0 train complement).
    """
    a = _artifact()
    c15 = _artifact_c15()
    c15_level = c15["per_seed"][0]["component_meta"][
        "mortality_level_new_widowhood"
    ]
    recomb = a["recombination_identity"]
    for key, cell in recomb["cells"].items():
        assert cell["candidate15_applied_rate"] == pytest.approx(
            float(c15_level[key]), abs=1e-12
        ), key


def test_support_stratum_rate_table_and_provenance():
    """The two-strata rate table + the observed-window stratum split.

    The stratum-1 (observed window reaches 75) hazard is >= the stratum-0
    (window below 75) hazard at every band (long observed support correlates
    with realized widowhood -- the forensics-4 Q9 mechanism), and the 75-84 /
    85+ bands sit entirely in stratum 1 (any ego observed AT 75+ necessarily has
    a window reaching 75). The observed stratum split is pinned.
    """
    a = _artifact()
    sd = a["support_stratum_diagnostic"]
    prov = sd["support_stratum_provenance"]
    assert prov["threshold_age"] == 75
    assert (
        prov["n_no_observed_support"]
        == EXPECTED_STRATUM["n_no_observed_support"]
    )
    assert (
        prov["n_window_reaches_75plus"]
        == EXPECTED_STRATUM["n_window_reaches_75plus"]
    )
    assert prov["n_window_below_75"] == EXPECTED_STRATUM["n_window_below_75"]
    assert prov["n_persons"] == (
        EXPECTED_STRATUM["n_window_reaches_75plus"]
        + EXPECTED_STRATUM["n_window_below_75"]
    )
    rows = {(r["band"], r["sex"]): r for r in sd["rate_table"]}
    assert len(rows) == 14
    for (band, _sex), r in rows.items():
        assert (
            r["stratum1_window_reaches_75plus_rate"]
            >= r["stratum0_window_below_75_rate"] - 1e-12
        ), band
        # the 75-84 and 85+ egos are all in stratum 1 (window reaches 75).
        if band in ("75-84", "85+"):
            assert r["stratum0_window_below_75_rate"] == 0.0
            assert r["stratum1_exposure_share"] == pytest.approx(1.0, abs=1e-9)


@needs_psid
def test_recombination_identity_live_seed0(_live_seed0):
    """Live: the seed-0 train fit's two strata recombine to the aggregate.

    Fits candidate 16 on seed 0's side B and recomputes the exposure-weighted
    recombination directly from the stratified train cells against candidate
    15's aggregate widowhood level (``components.mortality``), to <= 1e-9.
    """
    runner, panel, _demo, _ids_a, ids_b, components = _live_seed0
    strat = runner._widowhood_hazard_cells_by_stratum(
        panel, ids_b, components.support_stratum_series
    )
    for key, agg_rate in components.mortality.items():
        num = strat[0][key]["num_wt"] + strat[1][key]["num_wt"]
        den = strat[0][key]["den_wt"] + strat[1][key]["den_wt"]
        recombined = num / den if den > 0 else 0.0
        assert recombined == pytest.approx(agg_rate, abs=1e-9), key


# --------------------------------------------------------------------------
# Fresh-run artifact-schema conformance (amendment 1)
# --------------------------------------------------------------------------
def test_per_draw_per_cell_rates_shape_and_index():
    a = _artifact()
    pc = a["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    assert pc["shape"] == [N_DRAWS, N_GATED, len(GATE_SEEDS)]
    assert pc["cell_index"] == sorted(_gate2_tolerances())
    assert len(pc["cell_index"]) == N_GATED
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
    ci = pc["cell_index"]
    si = pc["seed_index"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for k in range(N_DRAWS):
        for c_idx, cell in enumerate(ci):
            for s_idx, seed in enumerate(si):
                cube = pc["rates"][k][c_idx][s_idx]
                stored = by_seed[seed]["gated_cells"][cell]["per_draw_rate"][k]
                assert cube == pytest.approx(stored, abs=1e-15)


def test_rbar_recomputes_from_per_draw_rates_and_scores():
    a = _artifact()
    for s in a["per_seed"]:
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


def test_undefined_draw_rule_not_triggered():
    a = _artifact()
    u = a["fresh_run_artifact_schema"]["undefined_draw_rule"]
    assert u["required"] is True
    assert u["pre_specified"] is True
    assert u["run_invalidated"] is False
    assert u["n_undefined_gated_draws"] == 0
    for s in a["per_seed"]:
        assert s["undefined_gated_draws"] == []
        for rec in s["gated_cells"].values():
            assert rec["n_draws_defined"] == N_DRAWS


def test_per_draw_dispersion_disclosure_report_only():
    a = _artifact()
    d = a["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    assert d["gated"] is False
    assert d["report_only"] is True
    cells = sorted(_gate2_tolerances())
    assert sorted(d["per_cell_per_draw_sd"]) == cells
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    cell = "mean_lifetime_marriages|male"
    for seed in GATE_SEEDS:
        rates = by_seed[seed]["gated_cells"][cell]["per_draw_rate"]
        expected = float(np.std(rates, ddof=1))
        assert d["per_cell_per_draw_sd"][cell][str(seed)] == pytest.approx(
            expected, abs=1e-12
        )


# --------------------------------------------------------------------------
# Structural delta checks: byte-identity vs c15 + the stratum recomposition
# --------------------------------------------------------------------------
def _has_prefix(cell: str, prefixes: tuple[str, ...]) -> bool:
    return any(cell.startswith(p) for p in prefixes)


def test_delta_untouched_cells_byte_identical_to_c15():
    """asfr / completed_fertility / first_marriage equal c15 draw-by-draw.

    Conditioning on the stratum changes only the married ego's competing-risk
    widowhood threshold; the scored RNG is drawn over state-independent
    active/fertile populations and the never-married population's first-marriage
    timing is unaffected, so these cells are byte-identical to candidate 15
    across every draw. A strong, always-runnable proof that the stratum is
    RNG-neutral -- and why seed 2's completed_fertility.c1970s stays failing.
    """
    a = _artifact()
    a15 = _artifact_c15()
    by16 = {s["seed"]: s for s in a["per_seed"]}
    by15 = {s["seed"]: s for s in a15["per_seed"]}
    n_checked = 0
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by16[seed][block].items():
                if not _has_prefix(cell, DELTA_UNTOUCHED_PREFIXES):
                    continue
                r15 = by15[seed][block][cell]["per_draw_rate"]
                r16 = rec["per_draw_rate"]
                assert len(r16) == len(r15) == N_DRAWS
                for k in range(N_DRAWS):
                    assert r16[k] == pytest.approx(
                        r15[k], abs=1e-12
                    ), f"{cell} seed {seed} draw {k} moved vs c15"
                n_checked += 1
    assert n_checked >= (7 + 5 + 10) * len(GATE_SEEDS)


def test_widowhood_trajectory_cells_moved_vs_c15():
    """Every widowhood band moves off candidate 15; the exact move-set is
    pinned.

    The stratum reshapes the widowhood hazard at EVERY band (the composition of
    who widows shifts toward long-observed-support persons), so all
    widowhood-incidence and widowed-stock cells, the divorced stocks, the
    remarriage flows and the marriage counts move -- 31 cells. The
    state-independent cells do NOT.
    """
    a = _artifact()
    a15 = _artifact_c15()
    by16 = {s["seed"]: s for s in a["per_seed"]}
    by15 = {s["seed"]: s for s in a15["per_seed"]}
    moved: set[str] = set()
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by16[seed][block].items():
                r15 = by15[seed][block][cell]["per_draw_rate"]
                r16 = rec["per_draw_rate"]
                if any(
                    abs(x - y) > 1e-12 for x, y in zip(r16, r15, strict=True)
                ):
                    moved.add(cell)
    assert moved == EXPECTED_MOVED_CELLS
    assert "widowhood.75+|female" in moved
    assert "share_widowed.75+|female" in moved
    # the state-independent cells did NOT move.
    assert not any(_has_prefix(c, DELTA_UNTOUCHED_PREFIXES) for c in moved)


def test_75plus_incidence_preserved_stock_lifted_vs_c15():
    """The stratum preserves the 75+ incidence (recombination identity) and
    lifts the stock across the tolerance.

    The recombination identity holds the band-aggregate incidence, so the 75+
    widowhood incidence is essentially unchanged (seed-mean sim/ref 1.060 ->
    1.061); the recomposed widowhood events lift the 75+ widowed stock from
    ~0.841 to ~0.914 of reference -- clearing the cell on every seed.
    """
    a = _artifact()
    e = a["elderly_75plus_diagnostic"]
    inc = e["cells"]["widowhood.75+|female"]
    stock = e["cells"]["share_widowed.75+|female"]
    assert inc["c15_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_INCIDENCE_SIM_REF["c15"], abs=1e-3
    )
    assert inc["c16_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_INCIDENCE_SIM_REF["c16"], abs=1e-3
    )
    assert stock["c15_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_STOCK_SIM_REF["c15"], abs=1e-3
    )
    assert stock["c16_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_STOCK_SIM_REF["c16"], abs=1e-3
    )
    summ = e["summary"]
    assert summ["incidence_sim_over_ref"]["preserved"] is True
    assert summ["stock_sim_over_ref"]["lifted_toward_reference"] is True


def test_yield_before_after_closes_the_leak():
    """The forensics-4 survival-to-75+ yield leak closes (train-side).

    The delta conditions the hazard on exactly the observed-window covariate
    forensics 4 Q9 isolated, so the train-side aggregate 75+ widowed-stock ratio
    lifts from ~0.842 (candidate 15) to ~0.911 (candidate 16), and the dominant
    50-64-onset survival-to-75+ yield / window-reaches-75 share move toward
    reference. The recomputed reference reconciles to forensics 4's committed
    reference.
    """
    a = _artifact()
    yb = a["yield_before_after"]
    agg = yb["aggregate_stock_share_75plus"]
    assert agg["before_sim_over_ref"] == pytest.approx(
        EXPECTED_YIELD_AGG["before_c15"], abs=2e-3
    )
    assert agg["after_sim_over_ref"] == pytest.approx(
        EXPECTED_YIELD_AGG["after_c16"], abs=2e-3
    )
    assert agg["after_sim_over_ref"] > agg["before_sim_over_ref"]
    assert yb["reference_reconciliation_vs_forensics4"]["reconciled"] is True
    b = yb["by_onset_band"]["50-64"]
    assert b["yield_b"]["before_sim_over_ref"] == pytest.approx(
        EXPECTED_YIELD_5064["yield_before"], abs=5e-3
    )
    assert b["yield_b"]["after_sim_over_ref"] == pytest.approx(
        EXPECTED_YIELD_5064["yield_after"], abs=5e-3
    )
    assert (
        b["yield_b"]["after_sim_over_ref"]
        > b["yield_b"]["before_sim_over_ref"]
    )
    w = b["share_window_reaches_75"]
    assert w["after_candidate16"] > w["before_candidate15"]
    # the after window-reaches-75 share moves toward the reference.
    assert abs(w["after_candidate16"] - w["reference"]) < abs(
        w["before_candidate15"] - w["reference"]
    )


# --------------------------------------------------------------------------
# Verdict / per-seed / per-block consistency (always runnable)
# --------------------------------------------------------------------------
def test_stored_tolerances_match_locked_gates_yaml():
    a = _artifact()
    tol = _gate2_tolerances()
    floor = json.loads(FLOOR.read_text())
    assert set(tol) == set(floor["gate_partition"]["gate_eligible"])
    for s in a["per_seed"]:
        assert set(s["gated_cells"]) == set(tol)
        for cell, rec in s["gated_cells"].items():
            assert rec["tolerance"] == pytest.approx(tol[cell], abs=1e-12)


def test_report_only_cells_match_gates_yaml_and_never_gate():
    a = _artifact()
    report_only = set(_gate2_thresholds()["report_only"])
    assert len(report_only) == N_REPORT_ONLY
    for s in a["per_seed"]:
        assert set(s["report_only_cells"]) == report_only
        for rec in s["report_only_cells"].values():
            assert rec["gated"] is False
    assert set(a["report_only"]["cells"]) == report_only


def test_every_gated_pass_recomputes_from_score():
    a = _artifact()
    for s in a["per_seed"]:
        n_pass = 0
        for rec in s["gated_cells"].values():
            recomputed = rec["score"] <= rec["tolerance"]
            assert rec["pass"] == recomputed
            n_pass += rec["pass"]
        assert s["n_gated_pass"] == n_pass
        assert s["n_gated"] == N_GATED
        assert s["n_gated_fail"] == N_GATED - n_pass


def test_seed_pass_recomputes_from_all_gated_cells():
    a = _artifact()
    for s in a["per_seed"]:
        expected = all(rec["pass"] for rec in s["gated_cells"].values())
        assert s["seed_pass"] == expected


def test_verdict_recomputes_from_seed_conjunction():
    a = _artifact()
    v = a["verdict"]
    n_pass = sum(1 for s in a["per_seed"] if s["seed_pass"])
    assert v["n_seeds_pass"] == n_pass
    assert v["gate_2_pass"] == (n_pass >= 4)
    for s in a["per_seed"]:
        assert v["seed_pass"][str(s["seed"])] == s["seed_pass"]


def test_verdict_per_block_counts_consistent():
    a = _artifact()
    v = a["verdict"]
    total_cells = sum(b["n_cells"] for b in v["per_block"].values())
    assert total_cells == N_GATED
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for blk in v["per_block"].values():
        for seed, rec in blk["per_seed_pass"].items():
            npass = sum(
                by_seed[int(seed)]["gated_cells"][c]["pass"]
                for c in blk["cells"]
            )
            assert rec["n_pass"] == npass


def test_all_failing_gated_cells_are_real_failures():
    a = _artifact()
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    seen = set()
    for f in a["verdict"]["all_failing_gated_cells"]:
        rec = by_seed[f["seed"]]["gated_cells"][f["cell"]]
        assert rec["pass"] is False
        assert rec["score"] > rec["tolerance"]
        seen.add((f["seed"], f["cell"]))
    expected = {
        (seed, cell)
        for seed, cells in EXPECTED_SEED_FAILS.items()
        for cell in cells
    }
    assert seen == expected


def test_candidate15_comparison_recomputes():
    a = _artifact()
    a15 = _artifact_c15()
    by15 = {s["seed"]: s for s in a15["per_seed"]}
    by16 = {s["seed"]: s for s in a["per_seed"]}
    comp = a["candidate15_comparison"]
    for cell in (MODAL_CELL,) + COUNT_CELLS + REMARRIAGE_GATED_CELLS:
        block = (
            comp["modal_cell"]
            if cell == MODAL_CELL
            else (
                comp["count_cells"]
                if cell in COUNT_CELLS
                else comp["remarriage_gated_cells"]
            )
        )
        rec = block[cell]
        c16_np = sum(
            1 for s in GATE_SEEDS if by16[s]["gated_cells"][cell]["pass"]
        )
        c15_np = sum(
            1 for s in GATE_SEEDS if by15[s]["gated_cells"][cell]["pass"]
        )
        assert rec["c16_n_seeds_pass"] == c16_np
        assert rec["c15_n_seeds_pass"] == c15_np
    # the widowhood aggregate is preserved by the recombination identity.
    agg = comp["widowhood_aggregate_preserved"]
    assert agg["reconciled"] is True
    assert agg["max_abs_residual_recombined_vs_aggregate"] <= 1e-9


def test_count_cells_held_5_of_5_with_margin():
    """Both marriage counts held candidate 15's 5/5 -- the registered modal
    (count re-clip) did NOT materialise.

    The registration feared the recomposed widowed exposure would clip the
    counts; measured, both counts held 5/5 with positive margin.
    """
    a = _artifact()
    cm = a["count_cell_margins"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = cm["cells"][cell]
        assert b["candidate15_n_seeds_pass"] == exp["c15"]
        assert b["n_seeds_pass"] == exp["c16"]
        assert b["held_vs_c15"] is True
        assert b["min_margin"] > 0.0
    assert cm["count_cells_hold"] is True


def test_incidence_headroom_all_pass():
    """The four gated widowhood-incidence cells hold on every seed.

    The recombination identity preserves each band's aggregate incidence, so the
    incidence cells stay inside tolerance on all five seeds.
    """
    a = _artifact()
    ih = a["incidence_headroom"]
    for cell in WIDOWHOOD_INCIDENCE_CELLS:
        b = ih["cells"][cell]
        assert b["n_seeds_pass"] == 5
        assert b["min_margin"] > 0.0


def test_modal_and_decider_recompute():
    a = _artifact()
    m = a["modal_failure_materialized"]
    assert list(m["modal_cells"]) == list(COUNT_CELLS)
    assert list(m["secondary_cells"]) == ["share_widowed.75+|female"]
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    # the registered modal (count re-clip) did NOT materialise; the gate PASSED.
    assert m["modal_materialized"] is False
    assert m["modal_failed_seeds"] == []
    assert m["secondary_failed_seeds"] == []
    assert dec["decider"] == "none (gate passed)"


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v16"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 9, 11, 12, 13, 14, 15):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate15_artifact_sha256"]) == 64
    assert len(pins["forensics4_artifact_sha256"]) == 64
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["candidate15_artifact"] == "runs/gate2_hazard_v15.json"
    assert fp["forensics4_artifact"] == "runs/gate2_forensics4_v1.json"
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 < fp["faithful_candidate_oc"] <= 1.0


# --------------------------------------------------------------------------
# Pinned one-shot outcome (published REGARDLESS of verdict) -- PASS 4/5
# --------------------------------------------------------------------------
def test_verdict_pinned():
    a = _artifact()
    v = a["verdict"]
    assert v["gate_2_pass"] is EXPECTED_GATE_PASS
    assert v["n_seeds_pass"] == EXPECTED_N_SEEDS_PASS
    assert v["seed_pass"] == EXPECTED_SEED_PASS


def test_seed_fails_pinned():
    a = _artifact()
    for s in a["per_seed"]:
        fails = {c for c, rec in s["gated_cells"].items() if not rec["pass"]}
        assert fails == set(EXPECTED_SEED_FAILS[s["seed"]])


def test_pass_path_is_registered_seeds():
    """The gate passes on exactly the registered pass path (seeds 0, 1, 3, 4).

    Seed 2 fails only on completed_fertility.c1970s -- the RNG-isolated split
    artifact forensics 4 Q8 identified (byte-identical to candidate 15).
    """
    a = _artifact()
    passing = sorted(int(k) for k, v in a["verdict"]["seed_pass"].items() if v)
    assert passing == [0, 1, 3, 4]
    assert a["pre_registered_forecast"]["pass_path_seeds"] == [0, 1, 3, 4]


def test_stock_cell_cleared_all_seeds_pinned():
    """The 75+ widowed-stock cell cleared on ALL five seeds (candidate 15 failed
    seeds 2 and 3).

    Pins the exact scores against the 0.185 tolerance; every seed is now well
    inside it (candidate 15's razor-edge seeds 2 and 3 flipped to pass).
    """
    a = _artifact()
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for seed, score in STOCK_SCORES.items():
        rec = by_seed[seed]["gated_cells"]["share_widowed.75+|female"]
        assert rec["tolerance"] == pytest.approx(STOCK_TOLERANCE, abs=1e-9)
        assert rec["score"] == pytest.approx(score, abs=1e-4)
        assert rec["pass"] is True
    sv = a["stock_per_seed_vs_candidate15"]
    assert sv["c15_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c15"]
    assert sv["c16_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c16"]
    assert sv["seeds_flipped_to_pass"] == EXPECTED_STOCK_FLIPPED_TO_PASS
    assert sv["seeds_flipped_to_fail"] == []


def test_completed_fertility_c1970s_seed2_only_failure_byte_identical_c15():
    """Seed 2's completed_fertility.c1970s is the only failure -- byte-identical
    to candidate 15 (the RNG-isolated split artifact, unmovable by the delta).
    """
    a = _artifact()
    a15 = _artifact_c15()
    by16 = {s["seed"]: s for s in a["per_seed"]}
    by15 = {s["seed"]: s for s in a15["per_seed"]}
    for seed in GATE_SEEDS:
        r16 = by16[seed]["gated_cells"]["completed_fertility.c1970s"]
        r15 = by15[seed]["gated_cells"]["completed_fertility.c1970s"]
        # byte-identical to candidate 15 draw-by-draw (RNG-neutral).
        for k in range(N_DRAWS):
            assert r16["per_draw_rate"][k] == pytest.approx(
                r15["per_draw_rate"][k], abs=1e-12
            )
        if seed == 2:
            assert r16["pass"] is False
            assert r16["score"] == pytest.approx(
                FERT_C1970S_SEED2_SCORE, abs=1e-4
            )
            assert r16["tolerance"] == pytest.approx(FERT_TOLERANCE, abs=1e-9)
        else:
            assert r16["pass"] is True


def test_target_movement_pinned():
    a = _artifact()
    comp = a["candidate15_comparison"]
    mc = comp["modal_cell"][MODAL_CELL]
    assert mc["c15_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c15"]
    assert mc["c16_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c16"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = comp["count_cells"][cell]
        assert b["c15_n_seeds_pass"] == exp["c15"]
        assert b["c16_n_seeds_pass"] == exp["c16"]


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 16 on seed 0's side B (train complement), once."""
    runner = _import_runner()
    import build_gate2_floors as g2f
    import run_gate2_candidate1 as c1

    from populace_dynamics.data import marriage
    from populace_dynamics.harness import panel as hpanel

    mh = marriage.marriage_history()
    birth = g2f.births.birth_history()
    death = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, _fert, _meta = g2f.load_panels()
    order_map = c1._order_map(mh)
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    components = runner.fit_components(
        panel, demo, death, mh, birth, order_map, ids_b
    )
    return runner, panel, demo, ids_a, ids_b, components


@needs_psid
def test_seed0_single_draw_pin(_live_seed0):
    """One draw at default_rng(5200) reproduces the committed draw-0 rate."""
    runner, panel, _demo, ids_a, _ids_b, components = _live_seed0
    cand = runner._draw_moments(panel, ids_a, components, DRAW_SEED_BASE)
    for cell, expected in SEED0_DRAW0.items():
        assert cand[cell]["rate"] == pytest.approx(expected, abs=1e-12)
    a = _artifact()
    pc = a["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    ci = pc["cell_index"]
    for cell, expected in SEED0_DRAW0.items():
        if cell in ci:
            cube = pc["rates"][0][ci.index(cell)][0]
            assert cube == pytest.approx(expected, abs=1e-12)


@needs_psid
def test_widowhood_level_aggregate_matches_c15_fit_live(_live_seed0):
    """The stratified widowhood level recombines to candidate 15's aggregate
    fit on the same train set (the delta is the conditioning, not the aggregate).
    """
    runner, _panel, _demo, _ids_a, ids_b, components = _live_seed0
    assert len(components.mortality) == 14  # 7 bands x 2 sexes
    c15 = _import_c15()
    import build_gate2_floors as g2f
    import run_gate2_candidate1 as c1

    from populace_dynamics.data import marriage

    mh = marriage.marriage_history()
    birth = g2f.births.birth_history()
    death = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    order_map = c1._order_map(mh)
    c15_components = c15.fit_components(
        _panel, demo, death, mh, birth, order_map, ids_b
    )
    # candidate 16's aggregate widowhood level (base.mortality) is candidate
    # 15's, bit-identical -- the delta is the stratum, not the aggregate.
    for key, rate in components.mortality.items():
        assert rate == pytest.approx(
            c15_components.mortality[key], abs=1e-12
        ), f"{key} aggregate widowhood level moved vs candidate 15's fit"
    # the two-stratum level is present and non-degenerate.
    assert set(components.widow_level_by_stratum) == {0, 1}
    assert components.meta["nchs_trend_applied_in_gate"] is False


@needs_psid
def test_delta_is_live_stratum_conditioned(_live_seed0):
    """The delta is live: the fitted stratum lookup distinguishes the strata and
    the seed-0 draw-0 75+ incidence moved off candidate 15.

    The fitted [band, sex, stratum] lookup gives stratum 1 (window reaches 75) a
    higher widowhood hazard than stratum 0 at the 65-74 band, and the seed-0
    draw-0 75+ widowhood incidence moved off candidate 15 (the stratum reshaped
    the widowhood composition).
    """
    runner, _panel, _demo, _ids_a, _ids_b, components = _live_seed0
    lk = runner._build_sim_lookups(components)
    assert lk.mort_arr.shape == (7, 2, 2)
    # 65-74 female (band index 4, sex 0): stratum 1 hazard > stratum 0.
    assert lk.mort_arr[4, 0, 1] > lk.mort_arr[4, 0, 0]
    # year-invariant (no trend); stratum-conditioned.
    fem = np.array([0.0])
    sp = np.array([80.0])
    male_opp = np.array([1.0])
    ego = np.array([70.0])  # 65-74 band
    s0 = np.array([0], dtype=np.int64)
    s1 = np.array([1], dtype=np.int64)
    p_s0 = runner._widow_probs(
        ego, fem, sp, male_opp, 2020, lk.mort_arr, lk.beta_arr, s0
    )
    p_s1 = runner._widow_probs(
        ego, fem, sp, male_opp, 2020, lk.mort_arr, lk.beta_arr, s1
    )
    assert p_s1[0] > p_s0[0]
    p_s1_1995 = runner._widow_probs(
        ego, fem, sp, male_opp, 1995, lk.mort_arr, lk.beta_arr, s1
    )
    assert np.array_equal(p_s1, p_s1_1995)  # year-invariant
    # the seed-0 draw-0 75+ widowhood incidence moved off candidate 15.
    assert (
        abs(SEED0_DRAW0["widowhood.75+|female"] - SEED0_DRAW0_C15_WIDOW75)
        > 1e-4
    )
