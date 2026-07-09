"""Tests for the gate-2 candidate-15 (run 1) pre-registered run.

Candidate 15 is the fifteenth pre-registered gate-2 candidate: candidate 14's
frozen spec (comment 4927236029, ``scripts/run_gate2_candidate14.py``, merged
#105) verbatim EXCEPT EXACTLY ONE delta, registered from candidate 14's grading
(issue #42 comment 4928232089):

* THE DELTA -- the NCHS period-trend multiplier ``exp(beta_sex * (year -
  1995))`` is REMOVED from the surviving-spouse widowhood hazard. Candidate 14
  applied ``rate = widow_level(ego_band, ego_sex) * exp(beta_sex * (year -
  1995))`` at simulation time; candidate 15 applies ``rate =
  widow_level(ego_band, ego_sex)`` -- the source-aligned train-empirical
  seven-band x sex level, period-pooled, with the SAME smoothing. The committed
  NCHS beta values stay documented (component_meta) for deployment-time use.

Candidate 14 was a DATA delta (the band table), so every compute code object --
including ``_widow_probs`` -- was reused. Candidate 15 is a COMPUTE delta: the
whole change lives in ``_widow_probs`` (it drops the ``* trend`` factor), so
``_widow_probs`` MOVES from the reused set into the DIVERGED set. Candidate 15
RE-IMPLEMENTS ``_widow_probs`` (returning the level only) and REUSES candidate
14's EXACT code objects for ``_build_sim_lookups``, ``simulate_holdout``,
``_draw_moments``, ``score_seed``, ``fit_remarriage_age_banded`` and
``_remarriage_probs_age_banded`` (rebound so the byte-identical
``simulate_holdout`` calls candidate 15's untrended ``_widow_probs``).
``fit_components`` wraps :func:`candidate14.fit_components` (the FIT is
byte-identical; the wrapper records the trend-removal provenance) and is pinned
DIVERGED from candidate 14's bytecode too.

Removing ``* trend`` is RNG-neutral (the uniform draw precedes ``_widow_probs``;
only the competing-risk threshold value moves), so the marital-state-
independent cells (``asfr.*``, ``completed_fertility.*``, ``first_marriage.*``)
are byte-identical to candidate 14 draw-by-draw
(``test_delta_untouched_cells_byte_identical_to_c14``). Unlike candidate 14's
split (localised above age 75), the trend removal touches EVERY widowhood band,
so all widowhood-incidence and widowed-stock cells move
(``test_widowhood_trajectory_cells_moved_vs_c14``).

The amended estimator (inherited, unchanged): per cell ``rbar_candidate,s`` is
the mean over K=20 draws (``default_rng(5200 + k)``, k=0..19) of the cell rate;
score ``|ln(rbar / rate_a,s)|`` scored once. The artifact conforms to
``fresh_run_artifact_schema``: the [20, 46, 5] per-draw per-cell rate cube,
undefined-draw invalidation, report-only dispersion. Frozen spec: issue #42
comment 4928232089.

The one-shot outcome (published REGARDLESS of verdict) is pinned below from the
committed artifact.
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v15.json"
ARTIFACT_C14 = ROOT / "runs" / "gate2_hazard_v14.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
FORENSICS3 = ROOT / "runs" / "gate2_forensics3_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4928232089"
)
SPEC_URL_C14 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4927236029"
)
REGISTRATION_POINTER = "4928232089"
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

# The seven-band widowhood table (candidate 14's; UNCHANGED by candidate 15).
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
# applied at gate time).
NCHS_BETA = {
    "female": -0.009234704865961198,
    "male": -0.010643975395626533,
}

# The reused candidate-14 code objects (byte-identity chain). ``_widow_probs``
# is NOT here -- it moved to the diverged set (the one delta).
REUSED_CODE_OBJECT_NAMES = (
    "_build_sim_lookups",
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
# The RE-IMPLEMENTED functions (the delta): NOT candidate 14's bytecode.
DIVERGED_CODE_OBJECT_NAMES = ("_widow_probs", "fit_components")

# Cells the delta does not change: the widowhood threshold changes but the
# scored RNG is drawn over state-independent active/fertile populations and the
# never-married population's first-marriage timing is unaffected by widowhood,
# so these are byte-identical to candidate 14 draw-by-draw.
DELTA_UNTOUCHED_PREFIXES = (
    "asfr.",
    "completed_fertility.",
    "first_marriage.",
)

# ==========================================================================
# One-shot outcome pins (from runs/gate2_hazard_v15.json; published REGARDLESS
# of verdict).
# ==========================================================================
# FAIL 3/5 -- up from candidate 14's 2/5. Removing the NCHS trend lifted the
# ELDERLY 75+ widowhood inflow (the 75+ exposure sits at late panel years, past
# the 1995 anchor, where the trend multiplier was ~0.88-0.90), so the 75+
# incidence rose (seed-mean sim/ref 0.952 -> 1.060, OVERSHOOTING reference) and
# seed 0's 75+ widowed stock cleared (share_widowed.75+|female flipped to
# pass). But the aggregate stock barely moved (0.838 -> 0.841) and the stock
# cell sits on the tolerance edge: seed 0 passes by 0.0003, seed 2 FAILS by
# 0.0001, seed 3 fails by 0.023 -- so the failing seeds shifted {0,3} -> {2,3}
# and the cell still passes only 3/5. Seed 2 still fails
# completed_fertility.c1970s (byte-identical to candidate 14). The registered
# female-count re-clip did NOT materialise (both counts held 5/5 with margin,
# because young-widow inflow FELL slightly -- early panel years, multiplier >
# 1 -- rather than rising ~7% as the registration hypothesised). The gate is
# 3/5: seeds 0, 1, 4 pass.
EXPECTED_GATE_PASS = False
EXPECTED_N_SEEDS_PASS = 3
EXPECTED_SEED_PASS = {
    "0": True,
    "1": True,
    "2": False,
    "3": False,
    "4": True,
}
EXPECTED_SEED_FAILS = {
    0: set(),
    1: set(),
    2: {"completed_fertility.c1970s", "share_widowed.75+|female"},
    3: {"share_widowed.75+|female"},
    4: set(),
}
# The 75+ widow-stock modal held 3/5 -> 3/5 but the failing seeds shifted
# {0,3} -> {2,3}; both marriage counts held candidate 14's 5/5.
EXPECTED_MODAL_MOVEMENT = {"c14": 3, "c15": 3}
EXPECTED_COUNT_MOVEMENT = {
    "mean_lifetime_marriages|male": {"c14": 5, "c15": 5},
    "mean_lifetime_marriages|female": {"c14": 5, "c15": 5},
}
# The share_widowed.75+|female cell tolerance and the razor-edge per-seed
# scores (seeds 0 and 2 straddle the tolerance).
STOCK_TOLERANCE = 0.185
STOCK_SCORES = {
    0: 0.18474,
    1: 0.13382,
    2: 0.18513,
    3: 0.208,
    4: 0.15496,
}
# The designed lift (seed-mean sim/ref vs candidate 14): incidence overshoots
# reference; the stock lifts marginally (still ~0.84).
EXPECTED_INCIDENCE_SIM_REF = {"c14": 0.9519, "c15": 1.0599}
EXPECTED_STOCK_SIM_REF = {"c14": 0.8382, "c15": 0.8411}
# The exposure-weighted trend multiplier candidate 15 removes, by slice
# (committed betas, full panel). The registration's ~0.92-0.95 matches the
# ELDERLY 75+ slices; the all-ages married-PY aggregate is > 1 (young exposure,
# early panel years).
EXPECTED_TREND_MULT = {
    "all_ages_married_person_years": 1.0254,
    "all_ages_widowhood_events": 0.9915,
    "elderly_75plus_married_person_years": 0.8834,
    "elderly_75plus_widowhood_events": 0.9012,
}
EXPECTED_TREND_HEADLINE = 1.0254
# Seed-0 draw-0 (default_rng(5200)) single-draw rates, live-reproducible on
# seed 0's side A. first_marriage.25-34|female is byte-identical to candidate
# 14 (the never-married population is unaffected); the marriage counts,
# remarriage and widowhood.75+|female MOVE off candidate 14 (the trend removal
# reshaped every widowhood band).
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.4184318661106998,
    "mean_lifetime_marriages|female": 1.4137686437341328,
    "remarriage.after_divorce": 0.06005230264954468,
    "first_marriage.25-34|female": 0.07771323726223815,
    "widowhood.75+|female": 0.06859212436250475,
}
# candidate 14's committed seed-0 draw-0 75+ widowhood incidence (the trend
# removal moved it up -- the clearest live proof the delta is non-inert).
SEED0_DRAW0_C14_WIDOW75 = 0.06143623110521258
# The exact set of cells that moved off candidate 14 (any seed, any draw).
# Unlike candidate 14 (17 cells, localised above 75), the trend removal touches
# every widowhood band, so the sub-75 widowhood cells and the divorced stocks
# move too (31 cells).
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


def _artifact_c14() -> dict:
    return json.loads(ARTIFACT_C14.read_text())


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
    import run_gate2_candidate15 as runner

    return runner


def _import_c14():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate14 as c14

    return c14


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    runner = _import_runner()
    assert runner.SPEC_REGISTRATION == SPEC_URL
    assert runner.REGISTRATION_POINTER == REGISTRATION_POINTER
    assert runner.CANDIDATE14_REGISTRATION == SPEC_URL_C14
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    assert runner.SIM_SEED_BASE == 4200
    # The widowhood band table is candidate 14's, UNCHANGED (the delta is the
    # trend removal, not the bands).
    assert runner.WIDOW_BANDS == WIDOW_BANDS
    assert runner.WIDOW_BANDS == _import_c14().WIDOW_BANDS
    # the committed NCHS betas are retained (documented, not applied).
    assert runner.NCHS_BETA_BY_SEX_COMMITTED == NCHS_BETA
    # the 5-band remarriage table and delta-1 count are candidate 14's,
    # inherited verbatim.
    assert runner.REM_AGE_BANDS == _import_c14().REM_AGE_BANDS
    assert (
        runner.observed_residual_counts
        is _import_c14().observed_residual_counts
    )


def test_reused_and_diverged_code_objects_vs_c14():
    """The reused chain is candidate 14's exact bytecode; _widow_probs and
    fit_components diverge.

    The crux structural fact: ``_widow_probs`` was REUSED in candidate 14 (a
    data delta) but DIVERGES in candidate 15 (a compute delta -- the trend is
    removed inside it). ``simulate_holdout`` stays candidate 14's exact code
    object and calls candidate 15's ``_widow_probs`` by global name.
    """
    runner = _import_runner()
    c14 = _import_c14()
    for name in REUSED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is getattr(c14, name).__code__
        ), f"{name} must reuse candidate 14's exact code object"
        assert getattr(runner, name).__globals__ is vars(runner)
    for name in DIVERGED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is not getattr(c14, name).__code__
        ), f"{name} must be re-implemented for the candidate-15 delta"
    # THE FLIP: _widow_probs was reused in candidate 14, diverges in candidate
    # 15.
    assert "_widow_probs" in c14.REUSED_CODE_OBJECT_NAMES
    assert "_widow_probs" not in runner.REUSED_CODE_OBJECT_NAMES
    assert "_widow_probs" in runner.DIVERGED_CODE_OBJECT_NAMES
    # the schema blocks are import-bound from candidate 14 unchanged.
    for name in (
        "_per_draw_per_cell_rates_block",
        "_undefined_draw_block",
        "_per_draw_dispersion_block",
    ):
        assert getattr(runner, name) is getattr(c14, name)
    pins = _artifact()["revision_pins"]
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )
    assert all(
        pins["diverged_code_objects_vs_candidate14"][n]
        for n in DIVERGED_CODE_OBJECT_NAMES
    )
    assert pins["nchs_trend_applied_candidate14"] is True
    assert pins["nchs_trend_applied_candidate15"] is False


def test_no_year_multiplier_enters_widowhood_lookup():
    """Structural: candidate 15's _widow_probs is year-invariant and differs
    from candidate 14's by exactly the removed trend factor.

    Always-runnable (synthetic mort_arr / betas -- no PSID). The returned rate
    does not depend on ``year`` or ``beta_arr`` (no trend); it equals candidate
    14's at the 1995 anchor (where the trend factor is 1) and candidate 14's
    equals candidate 15's times ``exp(beta * (year - 1995))`` away from it. The
    single, precise proof that the one delta is the removed multiplier.
    """
    runner = _import_runner()
    c14 = _import_c14()
    anchor = int(runner.TREND_ANCHOR_YEAR)
    mort = np.array(
        [
            [0.010, 0.005],
            [0.020, 0.010],
            [0.030, 0.015],
            [0.040, 0.020],
            [0.050, 0.025],
            [0.060, 0.030],
            [0.090, 0.045],
        ],
        dtype=np.float64,
    )
    beta = np.array([NCHS_BETA["female"], NCHS_BETA["male"]], dtype=np.float64)
    age = np.array([40.0, 80.0, 90.0])
    egom = np.array([0.0, 0.0, 1.0])
    sp = np.array([42.0, 82.0, 88.0])
    spm = 1.0 - egom
    p_anchor = runner._widow_probs(age, egom, sp, spm, anchor, mort, beta)
    p_late = runner._widow_probs(age, egom, sp, spm, 2020, mort, beta)
    # year-invariant (no trend enters the lookup).
    assert np.array_equal(p_anchor, p_late)
    # beta_arr does not enter either.
    p_zero_beta = runner._widow_probs(
        age, egom, sp, spm, 2020, mort, np.zeros(2)
    )
    assert np.array_equal(p_late, p_zero_beta)
    # equals candidate 14 at the anchor (trend factor 1).
    c14_anchor = c14._widow_probs(age, egom, sp, spm, anchor, mort, beta)
    assert np.allclose(p_anchor, c14_anchor)
    # candidate 14 away from the anchor = candidate 15 * the removed trend.
    c14_late = c14._widow_probs(age, egom, sp, spm, 2020, mort, beta)
    sidx = egom.astype(np.int64)
    trend = np.exp(beta[sidx] * (2020 - anchor))
    assert np.allclose(c14_late, p_late * trend)
    assert not np.allclose(c14_late, p_late)  # non-inert


def test_delta_string_names_the_one_delta():
    runner = _import_runner()
    d = runner.DELTA_VS_CANDIDATE14.lower()
    assert "one delta" in d or "exactly one" in d
    assert "trend" in d and "removed" in d
    assert "exp(beta_sex * (year - 1995))" in d
    assert "byte-identical" in d
    assert "candidate 14" in d


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v15"
    assert a["candidate"] == "candidate 15"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_delta_recorded():
    a = _artifact()
    assert a["candidate14_registration"] == SPEC_URL_C14
    model = a["model"]
    comp = model["components"]
    assert "removed" in comp["widowhood"].lower()
    assert "trend" in comp["widowhood"].lower()
    assert "byte-identical" in comp["fertility"].lower()
    assert "byte-identical" in comp["remarriage"].lower()
    assert "untouched" in comp["spousal_age_gap"].lower()
    assert "byte-identical" in comp["entry_widowed_initial_state"].lower()


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.65-0.75"
    assert "count" in f["modal_failure"].lower()
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
        assert row["forensics3_live_fixable_share"] == pytest.approx(
            committed[row["seed"]], abs=1e-12
        )


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
# Structural delta checks: byte-identity vs c14 + the trend removed
# --------------------------------------------------------------------------
def _has_prefix(cell: str, prefixes: tuple[str, ...]) -> bool:
    return any(cell.startswith(p) for p in prefixes)


def test_delta_untouched_cells_byte_identical_to_c14():
    """asfr / completed_fertility / first_marriage equal c14 draw-by-draw.

    Removing the trend changes only the married ego's competing-risk widowhood
    threshold; the scored RNG is drawn over state-independent active/fertile
    populations and the never-married population's first-marriage timing is
    unaffected, so these cells are byte-identical to candidate 14 across every
    draw. A strong, always-runnable proof that the removal is RNG-neutral.
    """
    a = _artifact()
    a14 = _artifact_c14()
    by15 = {s["seed"]: s for s in a["per_seed"]}
    by14 = {s["seed"]: s for s in a14["per_seed"]}
    n_checked = 0
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by15[seed][block].items():
                if not _has_prefix(cell, DELTA_UNTOUCHED_PREFIXES):
                    continue
                r14 = by14[seed][block][cell]["per_draw_rate"]
                r15 = rec["per_draw_rate"]
                assert len(r15) == len(r14) == N_DRAWS
                for k in range(N_DRAWS):
                    assert r15[k] == pytest.approx(
                        r14[k], abs=1e-12
                    ), f"{cell} seed {seed} draw {k} moved vs c14"
                n_checked += 1
    # asfr (7) + completed_fertility (5) + first_marriage (10, gated+report)
    # over 5 seeds.
    assert n_checked >= (7 + 5 + 10) * len(GATE_SEEDS)


def test_widowhood_trajectory_cells_moved_vs_c14():
    """Every widowhood band moves off candidate 14; the exact move-set is
    pinned.

    Unlike candidate 14's split (localised above 75), the trend removal touches
    the widowhood hazard at EVERY band, so all widowhood-incidence and
    widowed-stock cells, the divorced stocks, the remarriage flows and the
    marriage counts move -- 31 cells. The state-independent cells do NOT.
    """
    a = _artifact()
    a14 = _artifact_c14()
    by15 = {s["seed"]: s for s in a["per_seed"]}
    by14 = {s["seed"]: s for s in a14["per_seed"]}
    moved: set[str] = set()
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by15[seed][block].items():
                r14 = by14[seed][block][cell]["per_draw_rate"]
                r15 = rec["per_draw_rate"]
                if any(
                    abs(x - y) > 1e-12 for x, y in zip(r15, r14, strict=True)
                ):
                    moved.add(cell)
    assert moved == EXPECTED_MOVED_CELLS
    # the sub-75 widowhood bands moved too (unlike candidate 14).
    assert "widowhood.45-54|female" in moved
    assert "widowhood.55-64|female" in moved
    assert "widowhood.75+|female" in moved
    assert "share_widowed.75+|female" in moved
    # the state-independent cells did NOT move.
    assert not any(_has_prefix(c, DELTA_UNTOUCHED_PREFIXES) for c in moved)


def test_widowhood_level_identical_to_c14():
    """The fitted seven-band widowhood LEVEL is byte-identical to candidate 14.

    The delta is the APPLICATION (the trend), not the fit: candidate 15 wraps
    candidate 14's fit_components, so the fitted level cells are bit-identical.
    The only moving part is the removed multiplier.
    """
    a = _artifact()
    lc = a["candidate14_comparison"]["widowhood_level_identical_to_c14"]
    assert lc["all_bit_identical"] is True
    assert len(lc["cells"]) == 14  # 7 bands x 2 sexes
    for cell in lc["cells"].values():
        assert cell["bit_identical"] is True
        assert cell["candidate14_rate"] == pytest.approx(
            cell["candidate15_rate"], abs=1e-12
        )


def test_trend_multiplier_removed_slices():
    """The exposure-weighted trend multiplier the removal cancels, by slice.

    The registration hypothesised ~0.92-0.95 and a ~7% inflow lift ACROSS ALL
    BANDS. Measured: the ~0.92-0.95 matches the ELDERLY 75+ slices (0.88-0.90,
    late panel years past the 1995 anchor); the all-ages married-PY aggregate
    is 1.025 (> 1: young exposure, early years). This is why removing the trend
    lifts elderly inflow but slightly lowers young inflow.
    """
    a = _artifact()
    t = a["trend_multiplier_removed"]
    assert t["anchor_year"] == 1995.0
    assert t["beta_by_sex_committed"] == NCHS_BETA
    for slice_name, expected in EXPECTED_TREND_MULT.items():
        got = t["slices"][slice_name]["pooled"]["multiplier"]
        assert got == pytest.approx(expected, abs=1e-3), slice_name
    assert t["pooled_exposure_weighted_multiplier"] == pytest.approx(
        EXPECTED_TREND_HEADLINE, abs=1e-3
    )
    # the elderly slice is below 1 (trend suppressed elderly inflow); the
    # all-ages married-PY slice is above 1 (young exposure, early years).
    assert (
        t["slices"]["elderly_75plus_married_person_years"]["pooled"][
            "multiplier"
        ]
        < 1.0
    )
    assert (
        t["slices"]["all_ages_married_person_years"]["pooled"]["multiplier"]
        > 1.0
    )
    # weighted-mean years: elderly late (past anchor), all-ages early.
    assert (
        t["slices"]["elderly_75plus_married_person_years"]["pooled"][
            "weighted_mean_year"
        ]
        > 1995.0
    )
    assert (
        t["slices"]["all_ages_married_person_years"]["pooled"][
            "weighted_mean_year"
        ]
        < 1995.0
    )
    assert "across all bands" in t["registration_reconciliation"].lower()


def test_75plus_incidence_overshot_stock_lifted_vs_c14():
    """The trend removal lifts the 75+ incidence (overshooting reference) and
    marginally lifts the stock.

    The 75+ exposure sits at late panel years where the trend was ~0.88-0.90,
    so removing it lifts elderly incidence past reference (seed-mean sim/ref
    0.952 -> 1.060). The stock lifts only marginally (0.838 -> 0.841) and stays
    ~0.84 of reference -- the registered modal failure (the stock not clearing
    on every seed) materialised.
    """
    a = _artifact()
    e = a["elderly_75plus_diagnostic"]
    inc = e["cells"]["widowhood.75+|female"]
    stock = e["cells"]["share_widowed.75+|female"]
    assert inc["c14_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_INCIDENCE_SIM_REF["c14"], abs=1e-3
    )
    assert inc["c15_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_INCIDENCE_SIM_REF["c15"], abs=1e-3
    )
    assert stock["c14_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_STOCK_SIM_REF["c14"], abs=1e-3
    )
    assert stock["c15_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_STOCK_SIM_REF["c15"], abs=1e-3
    )
    summ = e["summary"]
    # incidence OVERSHOT (moved past 1.0, so not "toward" reference); the stock
    # lifted (marginally).
    assert summ["incidence_sim_over_ref"]["moved_toward_reference"] is False
    assert summ["stock_sim_over_ref"]["lifted_toward_reference"] is True
    assert inc["c15_sim_over_ref_mean"] > 1.0  # overshoot


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


def test_candidate14_comparison_recomputes():
    a = _artifact()
    a14 = _artifact_c14()
    by14 = {s["seed"]: s for s in a14["per_seed"]}
    by15 = {s["seed"]: s for s in a["per_seed"]}
    comp = a["candidate14_comparison"]
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
        c15_np = sum(
            1 for s in GATE_SEEDS if by15[s]["gated_cells"][cell]["pass"]
        )
        c14_np = sum(
            1 for s in GATE_SEEDS if by14[s]["gated_cells"][cell]["pass"]
        )
        assert rec["c15_n_seeds_pass"] == c15_np
        assert rec["c14_n_seeds_pass"] == c14_np


def test_count_cells_held_5_of_5_with_margin():
    """Both marriage counts held candidate 14's 5/5 -- the registered modal
    (female count re-clip) did NOT materialise.

    The registration feared the ~7% young-widow inflow rise would re-clip the
    female count; measured, young inflow slightly FELL (early panel years,
    multiplier > 1), so both counts held 5/5 with positive margin.
    """
    a = _artifact()
    cm = a["count_cell_margins"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = cm["cells"][cell]
        assert b["candidate14_n_seeds_pass"] == exp["c14"]
        assert b["n_seeds_pass"] == exp["c15"]
        assert b["held_vs_c14"] is True
        assert b["min_margin"] > 0.0
    assert cm["count_cells_hold"] is True


def test_incidence_headroom_all_pass():
    """The four gated widowhood-incidence cells hold on every seed.

    The ~7% inflow lift moves every incidence cell but the ln(1.5)-scale
    tolerances absorb it -- even widowhood.75+|female (which overshoots to
    ~1.06 sim/ref) keeps positive margin.
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
    assert list(m["modal_cells"]) == ["mean_lifetime_marriages|female"]
    assert list(m["secondary_cells"]) == ["share_widowed.75+|female"]
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    # the registered modal (female count re-clip) did NOT materialise.
    assert m["modal_materialized"] is False
    assert m["modal_failed_seeds"] == []
    # the registered secondary (75+ stock) DID: seeds 2 and 3 fail it, and
    # forgiving it flips >= 4 seeds to pass, so it is the decider.
    assert m["secondary_failed_seeds"] == [2, 3]
    assert m["secondary_seed3_failed"] is True
    assert "secondary" in dec["decider"]
    assert dec["n_seeds_pass_if_secondary_forgiven"] >= 4


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v15"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate14_artifact_sha256"]) == 64
    assert len(pins["forensics3_artifact_sha256"]) == 64
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["candidate14_artifact"] == "runs/gate2_hazard_v14.json"
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 < fp["faithful_candidate_oc"] <= 1.0


# --------------------------------------------------------------------------
# Pinned one-shot outcome (published REGARDLESS of verdict)
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


def test_stock_cell_razor_edge_pinned():
    """The 75+ widowed-stock cell straddles its tolerance: seed 0 passes by
    0.0003, seed 2 fails by 0.0001, seed 3 fails by 0.023.

    The gate improved 2/5 -> 3/5 because seed 0's stock cleared; the failing
    seeds shifted {0,3} (candidate 14) -> {2,3} (candidate 15). Pins the exact
    scores against the 0.185 tolerance.
    """
    a = _artifact()
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for seed, score in STOCK_SCORES.items():
        rec = by_seed[seed]["gated_cells"]["share_widowed.75+|female"]
        assert rec["tolerance"] == pytest.approx(STOCK_TOLERANCE, abs=1e-9)
        assert rec["score"] == pytest.approx(score, abs=1e-4)
        assert rec["pass"] == (rec["score"] <= rec["tolerance"])
    # the razor edge: seeds 0 and 2 straddle the tolerance.
    assert (
        by_seed[0]["gated_cells"]["share_widowed.75+|female"]["pass"] is True
    )
    assert (
        by_seed[2]["gated_cells"]["share_widowed.75+|female"]["pass"] is False
    )


def test_target_movement_pinned():
    a = _artifact()
    comp = a["candidate14_comparison"]
    mc = comp["modal_cell"][MODAL_CELL]
    assert mc["c14_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c14"]
    assert mc["c15_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c15"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = comp["count_cells"][cell]
        assert b["c14_n_seeds_pass"] == exp["c14"]
        assert b["c15_n_seeds_pass"] == exp["c15"]


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 15 on seed 0's side B (train complement), once."""
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
def test_widowhood_level_matches_c14_fit_live(_live_seed0):
    """The fitted widowhood level has 14 cells, all bit-identical to candidate
    14's fit on the same train set (the delta is the application, not the fit).
    """
    runner, _panel, _demo, _ids_a, ids_b, components = _live_seed0
    assert len(components.mortality) == 14  # 7 bands x 2 sexes
    c14 = _import_c14()
    import build_gate2_floors as g2f
    import run_gate2_candidate1 as c1

    from populace_dynamics.data import marriage

    mh = marriage.marriage_history()
    birth = g2f.births.birth_history()
    death = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    order_map = c1._order_map(mh)
    c14_components = c14.fit_components(
        _panel, demo, death, mh, birth, order_map, ids_b
    )
    for key, rate in components.mortality.items():
        assert rate == pytest.approx(
            c14_components.mortality[key], abs=1e-12
        ), f"{key} widowhood level moved vs candidate 14's fit"
    # the committed betas are still present in the fitted components' meta.
    assert components.meta["mortality_beta_by_sex"] == NCHS_BETA
    assert components.meta["nchs_trend_applied_in_gate"] is False


@needs_psid
def test_delta_is_live_trend_removed(_live_seed0):
    """The delta is live: _widow_probs is year-invariant on the fitted table
    and its seed-0 draw-0 75+ incidence moved up off candidate 14.

    Candidate 14 applied exp(beta * (year - 1995)); candidate 15 does not, so
    the fitted-table widowhood probabilities are identical across years, equal
    candidate 14's at the 1995 anchor, and the seed-0 draw-0 75+ widowhood
    incidence rose off candidate 14 (the trend suppressed elderly inflow).
    """
    runner, _panel, _demo, _ids_a, _ids_b, components = _live_seed0
    c14 = _import_c14()
    lk = runner._build_sim_lookups(components)
    assert lk.mort_arr.shape == (7, 2)
    fem = np.array([0.0])
    sp = np.array([80.0])
    male_opp = np.array([1.0])
    ego = np.array([90.0])  # 85+ band
    p_1995 = runner._widow_probs(
        ego, fem, sp, male_opp, 1995, lk.mort_arr, lk.beta_arr
    )
    p_2020 = runner._widow_probs(
        ego, fem, sp, male_opp, 2020, lk.mort_arr, lk.beta_arr
    )
    assert np.array_equal(p_1995, p_2020)  # year-invariant (no trend)
    c14_1995 = c14._widow_probs(
        ego, fem, sp, male_opp, 1995, lk.mort_arr, lk.beta_arr
    )
    c14_2020 = c14._widow_probs(
        ego, fem, sp, male_opp, 2020, lk.mort_arr, lk.beta_arr
    )
    assert np.allclose(p_1995, c14_1995)  # equal at the anchor
    assert not np.allclose(p_2020, c14_2020)  # differ away from it
    # the seed-0 draw-0 75+ widowhood incidence moved up off candidate 14.
    assert SEED0_DRAW0["widowhood.75+|female"] > SEED0_DRAW0_C14_WIDOW75
    assert (
        abs(SEED0_DRAW0["widowhood.75+|female"] - SEED0_DRAW0_C14_WIDOW75)
        > 1e-3
    )
