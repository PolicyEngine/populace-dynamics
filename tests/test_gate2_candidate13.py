"""Tests for the gate-2 candidate-13 (run 1) pre-registered run.

Candidate 13 is the thirteenth pre-registered gate-2 candidate: candidate 12's
frozen spec (comment 4925020986, ``scripts/run_gate2_candidate12.py``, merged
#103) verbatim EXCEPT EXACTLY ONE delta, registered from candidate 12's grading
(issue #42 comment 4925748151):

* THE DELTA -- the surviving-spouse widowhood hazard table gains younger bands,
  18-34 and 35-44. Candidate 12's four-band table {45-54, 55-64, 65-74, 75+} x
  sex clipped married egos below 45 into the 45-54 rate (~0.005/yr female, an
  order above the true young-widowhood risk); candidate 13's six-band table
  {18-34, 35-44, 45-54, 55-64, 65-74, 75+} x sex is train-estimated from
  mh85_23 spouse-death endings over married person-year exposure with the
  existing smoothing convention (``transitions._hazard_by_band`` weighted
  hazard, no add-one), the NCHS trend multiplier unchanged. The four inherited
  bands are bit-identical to candidate 12; the two new bands deflate the young
  widowed pool.

Everything else is byte-identical to candidate 12, and this is machine-checked:
candidate 13 REUSES candidate 12's EXACT code objects for the widowhood-band-
dependent compute (``_widow_probs``, ``_build_sim_lookups``,
``simulate_holdout``, ``_draw_moments``, ``score_seed``,
``fit_remarriage_age_banded``, ``_remarriage_probs_age_banded``), rebound to
candidate 13's module globals so the byte-identical simulation reads candidate
13's six-band widowhood table. Only ``fit_components`` is RE-IMPLEMENTED (to
install the six-band level) and pinned as DIVERGED from candidate 12's bytecode.
The vestigial spousal-age-gap machinery (candidate 12's delta 2, proven inert)
is left untouched per byte-minimality.

The empirically decisive structural fact
(``test_delta_untouched_cells_byte_identical_to_c12``): the delta changes only
the surviving-spouse widowhood competing-risk threshold, and the scored RNG
stream (per-year competing-risk uniforms, then fertility uniforms) is drawn over
state-independent active/fertile populations, so the marital-state-independent
cells -- ``asfr.*``, ``completed_fertility.*`` and ``first_marriage.*`` (the
never-married population's first-marriage timing is unaffected by widowhood) --
are byte-identical to candidate 12 draw-by-draw. Every widowed-trajectory cell
(divorce, remarriage, widowhood, the widowed/divorced stock and the marriage
counts) moves.

The amended estimator (inherited, unchanged): per cell ``rbar_candidate,s`` is
the mean over K=20 draws (``default_rng(5200 + k)``, k=0..19) of the cell rate;
score ``|ln(rbar / rate_a,s)|`` scored once. The artifact conforms to
``fresh_run_artifact_schema``: the [20, 46, 5] per-draw per-cell rate cube,
undefined-draw invalidation, report-only dispersion. Frozen spec: issue #42
comment 4925748151.

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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v13.json"
ARTIFACT_C12 = ROOT / "runs" / "gate2_hazard_v12.json"
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
    "#issuecomment-4925748151"
)
SPEC_URL_C12 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4925020986"
)
REGISTRATION_POINTER = "4925748151"
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

# The six-band widowhood table (the delta) and candidate 12's four-band table.
WIDOW_BANDS_C13 = ((18, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 120))
WIDOW_BANDS_C12 = ((45, 54), (55, 64), (65, 74), (75, 120))

# The reused candidate-12 code objects (byte-identity chain).
REUSED_CODE_OBJECT_NAMES = (
    "_widow_probs",
    "_build_sim_lookups",
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
# The RE-IMPLEMENTED function (the delta): NOT candidate 12's bytecode.
DIVERGED_CODE_OBJECT_NAMES = ("fit_components",)

# Cells the delta does not change: the widowhood threshold moves only married
# egos' state transitions, and the scored RNG is drawn over state-independent
# active/fertile populations, so fertility and the never-married population's
# first-marriage timing are byte-identical to candidate 12 draw-by-draw.
DELTA_UNTOUCHED_PREFIXES = (
    "asfr.",
    "completed_fertility.",
    "first_marriage.",
)
# Cells the widowed trajectory feeds (must move off candidate 12).
MOVED_PREFIXES = (
    "divorce.",
    "remarriage.",
    "widowhood.",
    "share_widowed.",
    "share_divorced.",
    "mean_lifetime_marriages|",
)

# ==========================================================================
# One-shot outcome pins (from runs/gate2_hazard_v13.json; published REGARDLESS
# of verdict).
# ==========================================================================
# FAIL 2/5 (seeds 1 and 4 clear all 46). The young-band delta cleared BOTH
# marriage-count cells (female 2/5 -> 5/5, male 4/5 -> 5/5) and deflated the
# young widowed pool (15-49 3.69x -> 1.70x; 50-64 1.63x -> 1.25x), exactly as
# designed. But the pre-registered SECONDARY risk materialised: deflating the
# young-widow inflow starved the elderly widowed stock, and
# share_widowed.75+|female regressed from candidate 12's 5/5 to 3/5 (newly
# failing seeds 0 and 3). Seed 2 still fails completed_fertility.c1970s (the
# persistent fertility tilt, byte-identical to candidate 12). The gate holds at
# 2/5.
EXPECTED_GATE_PASS = False
EXPECTED_N_SEEDS_PASS = 2
EXPECTED_SEED_PASS = {
    "0": False,
    "1": True,
    "2": False,
    "3": False,
    "4": True,
}
EXPECTED_SEED_FAILS = {
    0: {"share_widowed.75+|female"},
    1: set(),
    2: {"completed_fertility.c1970s"},
    3: {"share_widowed.75+|female"},
    4: set(),
}
# The widow-stock modal regressed (5/5 -> 3/5); both marriage counts cleared
# (male 4/5 -> 5/5, female 2/5 -> 5/5).
EXPECTED_MODAL_MOVEMENT = {"c12": 5, "c13": 3}
EXPECTED_COUNT_MOVEMENT = {
    "mean_lifetime_marriages|male": {"c12": 4, "c13": 5},
    "mean_lifetime_marriages|female": {"c12": 2, "c13": 5},
}
# The two new young bands (seed-0 train fit) sit well below the 45-54 rate their
# egos clipped into under candidate 12; the four inherited bands are
# bit-identical to candidate 12.
EXPECTED_YOUNG_BAND_RATES = {
    "18-34|female": 0.0015609573,
    "35-44|female": 0.0024943013,
    "18-34|male": 0.0008407511,
    "35-44|male": 0.0005961594,
}
EXPECTED_45_54_RATE = {"female": 0.0050488469, "male": 0.0021092738}
# The young-pool deflation (seed-mean sim/ref) the delta is designed to produce.
EXPECTED_YOUNG_POOL = {
    "15-49": {"c12": 3.692, "c13": 1.703},
    "50-64": {"c12": 1.633, "c13": 1.245},
}
# Seed-0 draw-0 (default_rng(5200)) single-draw rates, live-reproducible on seed
# 0's side B. first_marriage.25-34|female is byte-identical to candidate 12 (the
# never-married population is unaffected); the marriage counts and remarriage
# MOVE off candidate 12 (the widowed trajectory changed).
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.4200231819486067,
    "mean_lifetime_marriages|female": 1.4159320919404537,
    "remarriage.after_divorce": 0.05977159387794153,
    "first_marriage.25-34|female": 0.07771323726223815,
}
# candidate 12's committed seed-0 draw-0 male count (the delta moved it).
SEED0_DRAW0_C12_MALE = 1.4395857412139383


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c12() -> dict:
    return json.loads(ARTIFACT_C12.read_text())


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
    import run_gate2_candidate13 as runner

    return runner


def _import_c12():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate12 as c12

    return c12


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    runner = _import_runner()
    assert runner.SPEC_REGISTRATION == SPEC_URL
    assert runner.REGISTRATION_POINTER == REGISTRATION_POINTER
    assert runner.CANDIDATE12_REGISTRATION == SPEC_URL_C12
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    assert runner.SIM_SEED_BASE == 4200
    # THE DELTA: the six-band widowhood table.
    assert runner.WIDOW_BANDS == WIDOW_BANDS_C13
    assert list(runner.WIDOW_LOWERS) == [18, 35, 45, 55, 65, 75]
    # candidate 12's four inherited bands are unchanged (only two young bands
    # are added below them).
    assert tuple(runner.WIDOW_BANDS[2:]) == WIDOW_BANDS_C12
    assert _import_c12().WIDOW_BANDS == WIDOW_BANDS_C12
    # the 5-band remarriage table and delta-1 count are candidate 12's,
    # inherited verbatim.
    assert runner.REM_AGE_BANDS == _import_c12().REM_AGE_BANDS
    assert (
        runner.observed_residual_counts
        is _import_c12().observed_residual_counts
    )


def test_reused_code_objects_are_candidate12_bytecode():
    """The reused chain is candidate 12's exact bytecode, rebound to c13."""
    runner = _import_runner()
    c12 = _import_c12()
    for name in REUSED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is getattr(c12, name).__code__
        ), f"{name} must reuse candidate 12's exact code object"
        assert getattr(runner, name).__globals__ is vars(runner)
    # the one delta'd function is RE-IMPLEMENTED (diverged bytecode).
    for name in DIVERGED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is not getattr(c12, name).__code__
        ), f"{name} must be re-implemented for the candidate-13 delta"
    # the schema blocks are import-bound from candidate 12 unchanged.
    for name in (
        "_per_draw_per_cell_rates_block",
        "_undefined_draw_block",
        "_per_draw_dispersion_block",
    ):
        assert getattr(runner, name) is getattr(c12, name)
    pins = _artifact()["revision_pins"]
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )
    assert all(
        pins["diverged_code_objects_vs_candidate12"][n]
        for n in DIVERGED_CODE_OBJECT_NAMES
    )
    assert pins["widowhood_bands_candidate12"] == [
        list(b) for b in WIDOW_BANDS_C12
    ]
    assert pins["widowhood_bands_candidate13"] == [
        list(b) for b in WIDOW_BANDS_C13
    ]


def test_delta_string_names_the_one_delta():
    runner = _import_runner()
    d = runner.DELTA_VS_CANDIDATE12.lower()
    assert "one delta" in d or "exactly one" in d
    assert "18-34" in d and "35-44" in d
    assert "widowhood" in d
    assert "byte-identical" in d
    assert "candidate 12" in d


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v13"
    assert a["candidate"] == "candidate 13"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_delta_recorded():
    a = _artifact()
    assert a["candidate12_registration"] == SPEC_URL_C12
    model = a["model"]
    comp = model["components"]
    assert "18-34" in comp["widowhood"] and "35-44" in comp["widowhood"]
    assert "byte-identical" in comp["fertility"].lower()
    assert "byte-identical" in comp["remarriage"].lower()
    assert "untouched" in comp["spousal_age_gap"].lower()
    assert "byte-identical" in comp["entry_widowed_initial_state"].lower()


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.6-0.7"
    assert "count" in f["modal_failure"].lower()
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


def test_entry_widowed_seed_counts_present():
    a = _artifact()
    ec = a["entry_widowed_seed_counts"]
    assert ec["panel_total_entry_widowed_persons"] > 0
    assert len(ec["per_seed"]) == len(GATE_SEEDS)
    for row in ec["per_seed"]:
        assert row["n_entry_widowed_persons"] > 0
        assert (
            row["n_injected_widowed_person_years"]
            >= row["n_entry_widowed_persons"]
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
# Structural delta checks: byte-identity vs c12 + the young-band table
# --------------------------------------------------------------------------
def _has_prefix(cell: str, prefixes: tuple[str, ...]) -> bool:
    return any(cell.startswith(p) for p in prefixes)


def test_delta_untouched_cells_byte_identical_to_c12():
    """asfr / completed_fertility / first_marriage equal c12 draw-by-draw.

    The widowhood delta changes only married egos' competing-risk threshold;
    the scored RNG is drawn over state-independent active/fertile populations
    and the never-married population's first-marriage timing is unaffected by
    widowhood, so these cells are byte-identical to candidate 12 across every
    draw. A strong, always-runnable proof that only the widowed-trajectory
    cells moved.
    """
    a = _artifact()
    a12 = _artifact_c12()
    by13 = {s["seed"]: s for s in a["per_seed"]}
    by12 = {s["seed"]: s for s in a12["per_seed"]}
    n_checked = 0
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by13[seed][block].items():
                if not _has_prefix(cell, DELTA_UNTOUCHED_PREFIXES):
                    continue
                r12 = by12[seed][block][cell]["per_draw_rate"]
                r13 = rec["per_draw_rate"]
                assert len(r13) == len(r12) == N_DRAWS
                for k in range(N_DRAWS):
                    assert r13[k] == pytest.approx(
                        r12[k], abs=1e-12
                    ), f"{cell} seed {seed} draw {k} moved vs c12"
                n_checked += 1
    # asfr (6 gated) + completed_fertility (5 gated) + first_marriage (6 gated)
    # over 5 seeds, plus report-only fertility/first-marriage cells.
    assert n_checked >= (6 + 5 + 6) * len(GATE_SEEDS)


def test_widowed_trajectory_cells_moved_vs_c12():
    """Every widowed-trajectory cell moves off candidate 12 on some draw.

    Divorce, remarriage, widowhood, the widowed/divorced stock and the marriage
    counts all ride on the surviving-spouse widowhood transition, so the
    young-band delta must perturb them (unlike candidate 12's inert delta 2).
    """
    a = _artifact()
    a12 = _artifact_c12()
    by13 = {s["seed"]: s for s in a["per_seed"]}
    by12 = {s["seed"]: s for s in a12["per_seed"]}
    moved_prefixes: set[str] = set()
    for seed in GATE_SEEDS:
        for cell, rec in by13[seed]["gated_cells"].items():
            if not _has_prefix(cell, MOVED_PREFIXES):
                continue
            r12 = by12[seed]["gated_cells"][cell]["per_draw_rate"]
            r13 = rec["per_draw_rate"]
            if any(abs(x - y) > 1e-12 for x, y in zip(r13, r12, strict=True)):
                moved_prefixes.add(cell.split(".")[0].split("|")[0])
    # divorce, remarriage, widowhood, share_widowed, share_divorced and the
    # marriage counts each move on at least one gated cell.
    for pref in (
        "divorce",
        "remarriage",
        "widowhood",
        "share_widowed",
        "share_divorced",
        "mean_lifetime_marriages",
    ):
        assert pref in moved_prefixes, f"{pref} did not move vs c12"


def test_widowhood_band_rate_table():
    """The two young bands sit below 45-54; the four inherited bands match c12.

    The delta's central quantity: candidate 12's youngest band (45-54) is the
    rate its under-45 married egos clipped into; candidate 13's 18-34 and 35-44
    rates are a fraction of it, and the four inherited bands are bit-identical.
    """
    a = _artifact()
    table = a["candidate12_comparison"]["widowhood_band_rate_table_seed0"]
    cells = table["cells"]
    # The four inherited bands x 2 sexes are bit-identical to candidate 12.
    for band in ("45-54", "55-64", "65-74", "75+"):
        for sex in ("female", "male"):
            c = cells[f"{band}|{sex}"]
            assert c["new_band"] is False
            assert c["bit_identical"] is True
            assert c["candidate12_rate"] == pytest.approx(
                c["candidate13_rate"], abs=1e-12
            )
    # The two new young bands sit below the 45-54 rate their egos clipped into.
    for key, expected in EXPECTED_YOUNG_BAND_RATES.items():
        c = cells[key]
        assert c["new_band"] is True
        sex = key.split("|")[1]
        assert c["candidate12_applied_rate"] == pytest.approx(
            EXPECTED_45_54_RATE[sex], abs=1e-7
        )
        assert c["candidate13_rate"] == pytest.approx(expected, abs=1e-7)
        assert c["candidate13_rate"] < c["candidate12_applied_rate"]
        assert 0.0 < c["ratio_c13_over_c12_applied"] < 1.0


def test_young_pool_deflated_vs_c12():
    """The young widowed pools deflate toward reference vs candidate 12."""
    a = _artifact()
    summ = a["young_pool_diagnostic"]["seed_mean_sim_over_ref"]
    for band, exp in EXPECTED_YOUNG_POOL.items():
        c12 = summ[band]["c12_sim_over_ref_mean"]
        c13 = summ[band]["c13_sim_over_ref_mean"]
        assert c12 == pytest.approx(exp["c12"], abs=0.02)
        assert c13 == pytest.approx(exp["c13"], abs=0.02)
        # candidate 13 deflates the pool toward reference (ratio -> 1).
        assert c13 < c12
        assert abs(c13 - 1.0) < abs(c12 - 1.0)


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
    # exactly the pinned failing (seed, cell) pairs, nothing else.
    expected = {
        (seed, cell)
        for seed, cells in EXPECTED_SEED_FAILS.items()
        for cell in cells
    }
    assert seen == expected


def test_candidate12_comparison_recomputes():
    a = _artifact()
    a12 = _artifact_c12()
    by12 = {s["seed"]: s for s in a12["per_seed"]}
    by13 = {s["seed"]: s for s in a["per_seed"]}
    comp = a["candidate12_comparison"]
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
        c13_np = sum(
            1 for s in GATE_SEEDS if by13[s]["gated_cells"][cell]["pass"]
        )
        c12_np = sum(
            1 for s in GATE_SEEDS if by12[s]["gated_cells"][cell]["pass"]
        )
        assert rec["c13_n_seeds_pass"] == c13_np
        assert rec["c12_n_seeds_pass"] == c12_np


def test_count_cells_cleared_vs_c12():
    a = _artifact()
    comp = a["candidate12_comparison"]["count_cells"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = comp[cell]
        assert b["c12_n_seeds_pass"] == exp["c12"]
        assert b["c13_n_seeds_pass"] == exp["c13"]
    # both marriage counts cleared 5/5 -- the delta's designed effect.
    assert a["count_cell_tilt"]["count_cells_cleared"] is True


def test_modal_and_decider_recompute():
    a = _artifact()
    m = a["modal_failure_materialized"]
    assert list(m["modal_cells"]) == ["mean_lifetime_marriages|female"]
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    # the female count cleared 5/5, so the registered modal did NOT materialise;
    # the decider is the delta-targeted cells (the regressed widow stock).
    assert m["modal_materialized"] is False
    assert m["modal_failed_seeds"] == []
    assert "delta-targeted" in dec["decider"]


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v13"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 7, 8, 9, 10, 11, 12):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate12_artifact_sha256"]) == 64
    assert len(pins["forensics3_artifact_sha256"]) == 64
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["candidate12_artifact"] == "runs/gate2_hazard_v12.json"
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


def test_target_movement_pinned():
    a = _artifact()
    comp = a["candidate12_comparison"]
    mc = comp["modal_cell"][MODAL_CELL]
    assert mc["c12_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c12"]
    assert mc["c13_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c13"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = comp["count_cells"][cell]
        assert b["c12_n_seeds_pass"] == exp["c12"]
        assert b["c13_n_seeds_pass"] == exp["c13"]


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 13 on seed 0's side B (train complement), once."""
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
def test_widowhood_level_is_six_band_live(_live_seed0):
    """The fitted widowhood level has 12 cells; the 45+ bands match c12's fit.

    The four inherited 45+ bands are re-band-invariant: fitting candidate 12's
    four-band table on the SAME train set gives bit-identical rates (the band
    edges at 45+ are unchanged). The two young bands are new and lower.
    """
    runner, _panel, _demo, _ids_a, ids_b, components = _live_seed0
    assert len(components.mortality) == 12  # 6 bands x 2 sexes
    # candidate 13's own four-band re-fit (WIDOW_BANDS[2:]) on the same train
    # set: the gate reference's machinery banded on candidate 12's four bands.
    import populace_dynamics.data.transitions as transitions

    ev = _panel.events
    py = _panel.person_years
    ev = ev[ev["person_id"].isin(ids_b)]
    py = py[py["person_id"].isin(ids_b)]
    c12_cells = transitions._hazard_by_band(
        ev[ev["transition"] == "widowhood"],
        py[py["marital_state"] == "married"],
        "age",
        WIDOW_BANDS_C12,
        prefix="widowhood",
        by_sex=True,
        weighted=True,
    )
    for band in ("45-54", "55-64", "65-74", "75+"):
        for sex in ("female", "male"):
            key = f"{band}|{sex}"
            assert components.mortality[key] == pytest.approx(
                c12_cells[f"widowhood.{key}"]["rate"], abs=1e-12
            ), f"{key} widowhood level moved vs the four-band fit"
    # the two young bands exist and are below the 45-54 rate.
    for sex in ("female", "male"):
        assert (
            components.mortality[f"18-34|{sex}"]
            < components.mortality[f"45-54|{sex}"]
        )
        assert (
            components.mortality[f"35-44|{sex}"]
            < components.mortality[f"45-54|{sex}"]
        )


@needs_psid
def test_delta_is_live_widow_prob_deflated(_live_seed0):
    """The delta is live: young married egos widow below candidate 12's rate.

    Candidate 12 clips a married ego below 45 into the 45-54 widowhood rate;
    candidate 13 gives it the true 18-34 / 35-44 rate, an order lower. The male
    seed-0 draw-0 count moved off candidate 12 as a consequence (the delta is
    NOT inert, unlike candidate 12's spousal-gap draw).
    """
    runner, _panel, _demo, _ids_a, _ids_b, components = _live_seed0
    c12 = _import_c12()
    lk13 = runner._build_sim_lookups(components)
    assert lk13.mort_arr.shape == (6, 2)
    age30 = np.array([30.0])
    fem = np.array([0.0])
    male_opp = np.array([1.0])
    sp = np.array([55.0])
    p13 = runner._widow_probs(
        age30, fem, sp, male_opp, 2000, lk13.mort_arr, lk13.beta_arr
    )
    # candidate 12's clip gives the 45-54 rate (mort_arr row 0 is 45-54).
    p12_clip = float(components.mortality["45-54|female"]) * math.exp(
        lk13.beta_arr[0] * (2000.0 - runner.TREND_ANCHOR_YEAR)
    )
    assert p13[0] < p12_clip
    assert p13[0] / p12_clip < 0.5
    # the seed-0 draw-0 male count moved off candidate 12 (delta non-inert).
    assert (
        abs(SEED0_DRAW0["mean_lifetime_marriages|male"] - SEED0_DRAW0_C12_MALE)
        > 1e-4
    )
    assert c12 is not None
