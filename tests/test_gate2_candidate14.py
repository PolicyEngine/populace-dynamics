"""Tests for the gate-2 candidate-14 (run 1) pre-registered run.

Candidate 14 is the fourteenth pre-registered gate-2 candidate: candidate 13's
frozen spec (comment 4925748151, ``scripts/run_gate2_candidate13.py``, merged
#104) verbatim EXCEPT EXACTLY ONE delta, registered from candidate 13's grading
(issue #42 comment 4927236029):

* THE DELTA -- the surviving-spouse widowhood hazard's oldest band, 75+, splits
  into 75-84 and 85+. Candidate 13's six-band table {18-34, 35-44, 45-54,
  55-64, 65-74, 75+} x sex pooled ages 75-120 into one rate (~0.056/yr female),
  so an 85-year-old married ego widowed at the same rate as a 76-year-old;
  candidate 14's seven-band table {18-34, 35-44, 45-54, 55-64, 65-74, 75-84,
  85+} x sex is train-estimated from mh85_23 spouse-death endings over married
  person-year exposure with the existing smoothing convention
  (``transitions._hazard_by_band`` weighted hazard, no add-one), the NCHS trend
  multiplier unchanged. The five inherited bands (18-34 ... 65-74) are
  bit-identical to candidate 13; the pooled 75+ band splits in two, letting 85+
  egos widow at their own higher hazard.

Everything else is byte-identical to candidate 13, and this is machine-checked:
candidate 14 REUSES candidate 13's EXACT code objects for the widowhood-band-
dependent compute (``_widow_probs``, ``_build_sim_lookups``,
``simulate_holdout``, ``_draw_moments``, ``score_seed``,
``fit_remarriage_age_banded``, ``_remarriage_probs_age_banded`` -- themselves
candidate 12's, threaded through), rebound to candidate 14's module globals so
the byte-identical simulation reads candidate 14's seven-band widowhood table.
Only ``fit_components`` is RE-IMPLEMENTED (to install the seven-band level) and
pinned as DIVERGED from candidate 13's bytecode.

The empirically decisive structural fact
(``test_delta_untouched_cells_byte_identical_to_c13``): the delta changes only
the surviving-spouse widowhood competing-risk threshold for married egos aged
75+, and the scored RNG stream is drawn over state-independent active/fertile
populations, so the marital-state-independent cells -- ``asfr.*``,
``completed_fertility.*`` and ``first_marriage.*`` -- are byte-identical to
candidate 13 draw-by-draw, AND the split is localised above 75: the sub-75
widowhood cells, the divorced stock and the 65-74 widowed stock are byte-
identical too. Only the 75+ widowed-trajectory cells move.

The amended estimator (inherited, unchanged): per cell ``rbar_candidate,s`` is
the mean over K=20 draws (``default_rng(5200 + k)``, k=0..19) of the cell rate;
score ``|ln(rbar / rate_a,s)|`` scored once. The artifact conforms to
``fresh_run_artifact_schema``: the [20, 46, 5] per-draw per-cell rate cube,
undefined-draw invalidation, report-only dispersion. Frozen spec: issue #42
comment 4927236029.

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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v14.json"
ARTIFACT_C13 = ROOT / "runs" / "gate2_hazard_v13.json"
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
    "#issuecomment-4927236029"
)
SPEC_URL_C13 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4925748151"
)
REGISTRATION_POINTER = "4927236029"
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

# The seven-band widowhood table (the delta) and candidate 13's six-band table.
WIDOW_BANDS_C14 = (
    (18, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 84),
    (85, 120),
)
WIDOW_BANDS_C13 = ((18, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 120))

# The reused candidate-13 code objects (byte-identity chain).
REUSED_CODE_OBJECT_NAMES = (
    "_widow_probs",
    "_build_sim_lookups",
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
# The RE-IMPLEMENTED function (the delta): NOT candidate 13's bytecode.
DIVERGED_CODE_OBJECT_NAMES = ("fit_components",)

# Cells the delta does not change: the widowhood threshold moves only married
# egos aged 75+, and the scored RNG is drawn over state-independent
# active/fertile populations, so fertility and the never-married population's
# first-marriage timing are byte-identical to candidate 13 draw-by-draw.
DELTA_UNTOUCHED_PREFIXES = (
    "asfr.",
    "completed_fertility.",
    "first_marriage.",
)
# The split is localised above 75: these cells never see a 75+ hazard change,
# so they are byte-identical to candidate 13 too.
LOCALISED_IDENTICAL_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "share_widowed.65-74|female",
    "share_widowed.65-74|male",
    "share_divorced.45-54|female",
    "share_divorced.45-54|male",
    "share_divorced.55-64|female",
    "share_divorced.55-64|male",
)

# ==========================================================================
# One-shot outcome pins (from runs/gate2_hazard_v14.json; published REGARDLESS
# of verdict).
# ==========================================================================
# FAIL 2/5 -- the SAME 2/5 as candidate 13, with the SAME failing cells. The
# split recovered the 75+ widowhood INCIDENCE toward reference (seed-mean
# sim/ref 0.929 -> 0.952) exactly as designed, but because the reallocation is
# exposure-preserving (the split bands' events/exposure sum to candidate 13's
# pooled 75+) and the 75-84 band dominates the married 75+ exposure (~56M vs
# ~9M person-years), the aggregate 75+ widowed STOCK stayed flat (seed-mean
# sim/ref 0.841 -> 0.838), so share_widowed.75+|female still failed seeds 0 and
# 3 (the registered modal failure). Seed 2 still fails completed_fertility.c1970s
# (the persistent fertility tilt, byte-identical to candidate 13). The gate holds
# at 2/5.
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
# The 75+ widow-stock modal held (3/5 -> 3/5, the stock did not lift); both
# marriage counts held candidate 13's 5/5 (insulated).
EXPECTED_MODAL_MOVEMENT = {"c13": 3, "c14": 3}
EXPECTED_COUNT_MOVEMENT = {
    "mean_lifetime_marriages|male": {"c13": 5, "c14": 5},
    "mean_lifetime_marriages|female": {"c13": 5, "c14": 5},
}
# The split bands (seed-0 train fit): 85+ widows at ~1.7x the pooled 75+ rate,
# 75-84 at ~0.83-0.89x; the reallocation is exposure-preserving.
EXPECTED_SPLIT_BAND_RATES = {
    "75-84|female": 0.04983333,
    "85+|female": 0.09496430,
    "75-84|male": 0.02147069,
    "85+|male": 0.04555769,
}
EXPECTED_POOLED_75PLUS_RATE = {"female": 0.05584759, "male": 0.02583875}
# The designed recovery (seed-mean sim/ref vs candidate 13): incidence lifts
# toward reference; the stock does NOT (the registered modal failure).
EXPECTED_INCIDENCE_SIM_REF = {"c13": 0.9287, "c14": 0.9519}
EXPECTED_STOCK_SIM_REF = {"c13": 0.8410, "c14": 0.8382}
# Seed-0 draw-0 (default_rng(5200)) single-draw rates, live-reproducible on seed
# 0's side A. first_marriage.25-34|female is byte-identical to candidate 13 (the
# never-married population is unaffected); widowhood.75+|female and the male
# marriage count MOVE off candidate 13 (the 75+ widowed trajectory changed).
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.420002975932196,
    "mean_lifetime_marriages|female": 1.4159320919404537,
    "remarriage.after_divorce": 0.05977159715071064,
    "first_marriage.25-34|female": 0.07771323726223815,
    "widowhood.75+|female": 0.06143623110521258,
}
# candidate 13's committed seed-0 draw-0 75+ widowhood incidence (the split
# moved it -- the clearest live proof the delta is non-inert).
SEED0_DRAW0_C13_WIDOW75 = 0.05287845759306159
# The exact set of cells that moved off candidate 13 (any seed, any draw). The
# split is localised to the 75+ widowed trajectory and what it feeds.
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
    "remarriage.ysd0-4",
    "remarriage.ysd10+",
    "remarriage.ysd5-9",
    "share_widowed.75+|female",
    "share_widowed.75+|male",
    "widowhood.45+|male",
    "widowhood.75+|female",
    "widowhood.75+|male",
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c13() -> dict:
    return json.loads(ARTIFACT_C13.read_text())


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
    import run_gate2_candidate14 as runner

    return runner


def _import_c13():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate13 as c13

    return c13


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    runner = _import_runner()
    assert runner.SPEC_REGISTRATION == SPEC_URL
    assert runner.REGISTRATION_POINTER == REGISTRATION_POINTER
    assert runner.CANDIDATE13_REGISTRATION == SPEC_URL_C13
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    assert runner.SIM_SEED_BASE == 4200
    # THE DELTA: the seven-band widowhood table (75+ split into 75-84 and 85+).
    assert runner.WIDOW_BANDS == WIDOW_BANDS_C14
    assert list(runner.WIDOW_LOWERS) == [18, 35, 45, 55, 65, 75, 85]
    # candidate 13's five bands below 75 are unchanged; only the pooled 75+
    # band splits.
    assert tuple(runner.WIDOW_BANDS[:5]) == WIDOW_BANDS_C13[:5]
    assert runner.WIDOW_BANDS[5:] == ((75, 84), (85, 120))
    assert runner.SPLIT_WIDOW_BAND_INDICES == (5, 6)
    assert _import_c13().WIDOW_BANDS == WIDOW_BANDS_C13
    assert runner.WIDOW_BANDS[5:][0] != _import_c13().WIDOW_BANDS[5]
    # the 5-band remarriage table and delta-1 count are candidate 13's,
    # inherited verbatim.
    assert runner.REM_AGE_BANDS == _import_c13().REM_AGE_BANDS
    assert (
        runner.observed_residual_counts
        is _import_c13().observed_residual_counts
    )


def test_reused_code_objects_are_candidate13_bytecode():
    """The reused chain is candidate 13's exact bytecode, rebound to c14."""
    runner = _import_runner()
    c13 = _import_c13()
    for name in REUSED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is getattr(c13, name).__code__
        ), f"{name} must reuse candidate 13's exact code object"
        assert getattr(runner, name).__globals__ is vars(runner)
    # the one delta'd function is RE-IMPLEMENTED (diverged bytecode).
    for name in DIVERGED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is not getattr(c13, name).__code__
        ), f"{name} must be re-implemented for the candidate-14 delta"
    # the schema blocks are import-bound from candidate 13 unchanged.
    for name in (
        "_per_draw_per_cell_rates_block",
        "_undefined_draw_block",
        "_per_draw_dispersion_block",
    ):
        assert getattr(runner, name) is getattr(c13, name)
    pins = _artifact()["revision_pins"]
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )
    assert all(
        pins["diverged_code_objects_vs_candidate13"][n]
        for n in DIVERGED_CODE_OBJECT_NAMES
    )
    assert pins["widowhood_bands_candidate13"] == [
        list(b) for b in WIDOW_BANDS_C13
    ]
    assert pins["widowhood_bands_candidate14"] == [
        list(b) for b in WIDOW_BANDS_C14
    ]


def test_delta_string_names_the_one_delta():
    runner = _import_runner()
    d = runner.DELTA_VS_CANDIDATE13.lower()
    assert "one delta" in d or "exactly one" in d
    assert "75-84" in d and "85+" in d
    assert "widowhood" in d
    assert "byte-identical" in d
    assert "candidate 13" in d


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v14"
    assert a["candidate"] == "candidate 14"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_delta_recorded():
    a = _artifact()
    assert a["candidate13_registration"] == SPEC_URL_C13
    model = a["model"]
    comp = model["components"]
    assert "75-84" in comp["widowhood"] and "85+" in comp["widowhood"]
    assert "byte-identical" in comp["fertility"].lower()
    assert "byte-identical" in comp["remarriage"].lower()
    assert "untouched" in comp["spousal_age_gap"].lower()
    assert "byte-identical" in comp["entry_widowed_initial_state"].lower()


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.6-0.7"
    assert "stock" in f["modal_failure"].lower()
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
# Structural delta checks: byte-identity vs c13 + the split-band table
# --------------------------------------------------------------------------
def _has_prefix(cell: str, prefixes: tuple[str, ...]) -> bool:
    return any(cell.startswith(p) for p in prefixes)


def test_delta_untouched_cells_byte_identical_to_c13():
    """asfr / completed_fertility / first_marriage equal c13 draw-by-draw.

    The widowhood delta changes only married egos aged 75+ competing-risk
    threshold; the scored RNG is drawn over state-independent active/fertile
    populations and the never-married population's first-marriage timing is
    unaffected by widowhood, so these cells are byte-identical to candidate 13
    across every draw. A strong, always-runnable proof that only the 75+
    widowed-trajectory cells moved.
    """
    a = _artifact()
    a13 = _artifact_c13()
    by14 = {s["seed"]: s for s in a["per_seed"]}
    by13 = {s["seed"]: s for s in a13["per_seed"]}
    n_checked = 0
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by14[seed][block].items():
                if not _has_prefix(cell, DELTA_UNTOUCHED_PREFIXES):
                    continue
                r13 = by13[seed][block][cell]["per_draw_rate"]
                r14 = rec["per_draw_rate"]
                assert len(r14) == len(r13) == N_DRAWS
                for k in range(N_DRAWS):
                    assert r14[k] == pytest.approx(
                        r13[k], abs=1e-12
                    ), f"{cell} seed {seed} draw {k} moved vs c13"
                n_checked += 1
    # asfr (7) + completed_fertility (5) + first_marriage (10, gated+report)
    # over 5 seeds.
    assert n_checked >= (7 + 5 + 10) * len(GATE_SEEDS)


def test_split_localised_above_75_byte_identical_to_c13():
    """The sub-75 widowhood, divorced stock and 65-74 widowed stock hold c13.

    The split touches only the 75+ widowhood hazard, so no cell that never
    sees a 75+ hazard change may move: widowhood.45-64|female /
    widowhood.65-74|female, share_widowed.65-74 and the 45-64 divorced stock
    are byte-identical to candidate 13 draw-by-draw. A surgical proof of the
    delta's locality above age 75.
    """
    a = _artifact()
    a13 = _artifact_c13()
    by14 = {s["seed"]: s for s in a["per_seed"]}
    by13 = {s["seed"]: s for s in a13["per_seed"]}
    for seed in GATE_SEEDS:
        for cell in LOCALISED_IDENTICAL_CELLS:
            block = (
                "gated_cells"
                if cell in by14[seed]["gated_cells"]
                else "report_only_cells"
            )
            r14 = by14[seed][block][cell]["per_draw_rate"]
            r13 = by13[seed][block][cell]["per_draw_rate"]
            for k in range(N_DRAWS):
                assert r14[k] == pytest.approx(
                    r13[k], abs=1e-12
                ), f"{cell} seed {seed} draw {k} moved vs c13"


def test_widowed_trajectory_cells_moved_vs_c13():
    """The 75+ widowed-trajectory cells move off candidate 13; the exact
    move-set is pinned.

    The split changes only married egos aged 75+ widowhood, so the 75+
    incidence/stock, the elderly widowed remarriage exposure, the long-duration
    divorce exposure and the marriage counts move -- but the divorced stock and
    the sub-75 widowhood do NOT. The exact set is pinned from the artifact.
    """
    a = _artifact()
    a13 = _artifact_c13()
    by14 = {s["seed"]: s for s in a["per_seed"]}
    by13 = {s["seed"]: s for s in a13["per_seed"]}
    moved: set[str] = set()
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by14[seed][block].items():
                r13 = by13[seed][block][cell]["per_draw_rate"]
                r14 = rec["per_draw_rate"]
                if any(
                    abs(x - y) > 1e-12 for x, y in zip(r14, r13, strict=True)
                ):
                    moved.add(cell)
    assert moved == EXPECTED_MOVED_CELLS
    # the two 75+ targets are the point of the delta and must move.
    assert "widowhood.75+|female" in moved
    assert "share_widowed.75+|female" in moved
    # the divorced stock and sub-75 widowhood did NOT move (localised delta).
    assert not (moved & set(LOCALISED_IDENTICAL_CELLS))


def test_widowhood_split_band_table():
    """The pooled 75+ splits into 75-84 (lower) and 85+ (higher); the five
    inherited bands match candidate 13 and the reallocation is
    exposure-preserving.

    The delta's central quantity: candidate 13's pooled 75+ rate is what its
    75-84 and 85+ egos shared; candidate 14 gives 85+ its own higher hazard
    (~1.7x the pool) and 75-84 a lower one, and the split bands' weighted
    events/exposure sum to candidate 13's pooled 75+ (same events, re-banded).
    """
    a = _artifact()
    table = a["candidate13_comparison"]["widowhood_split_band_table_seed0"]
    cells = table["cells"]
    assert table["pooled_label"] == "75+"
    # The five inherited bands x 2 sexes are bit-identical to candidate 13.
    for band in ("18-34", "35-44", "45-54", "55-64", "65-74"):
        for sex in ("female", "male"):
            c = cells[f"{band}|{sex}"]
            assert c["split_band"] is False
            assert c["bit_identical"] is True
            assert c["candidate13_rate"] == pytest.approx(
                c["candidate14_rate"], abs=1e-12
            )
    # The two split bands: 75-84 below the pool, 85+ well above it.
    c13art = _artifact_c13()
    c13_diag = c13art["per_seed"][0]["component_meta"][
        "mortality_level_diagnostics"
    ]["cells"]
    for sex in ("female", "male"):
        lo = cells[f"75-84|{sex}"]
        hi = cells[f"85+|{sex}"]
        pooled = EXPECTED_POOLED_75PLUS_RATE[sex]
        assert lo["split_band"] is True and hi["split_band"] is True
        assert lo["candidate13_pooled_rate"] == pytest.approx(pooled, abs=1e-6)
        assert hi["candidate13_pooled_rate"] == pytest.approx(pooled, abs=1e-6)
        assert lo["candidate14_rate"] == pytest.approx(
            EXPECTED_SPLIT_BAND_RATES[f"75-84|{sex}"], abs=1e-6
        )
        assert hi["candidate14_rate"] == pytest.approx(
            EXPECTED_SPLIT_BAND_RATES[f"85+|{sex}"], abs=1e-6
        )
        # the gradient the pooled band averaged away: 75-84 < pooled < 85+.
        assert lo["candidate14_rate"] < pooled < hi["candidate14_rate"]
        assert hi["candidate14_rate"] / lo["candidate14_rate"] > 1.5
        # exposure preservation: the split bands' weighted events and exposure
        # sum EXACTLY to candidate 13's pooled 75+ (same events, re-banded).
        pooled_cell = c13_diag[f"75+|{sex}"]
        assert lo["num_wt"] + hi["num_wt"] == pytest.approx(
            pooled_cell["num_wt"], rel=1e-9
        )
        assert lo["den_wt"] + hi["den_wt"] == pytest.approx(
            pooled_cell["den_wt"], rel=1e-9
        )
        assert lo["n_events"] + hi["n_events"] == pooled_cell["n_events"]


def test_75plus_incidence_recovered_stock_flat_vs_c13():
    """The split lifts the 75+ incidence toward reference; the stock stays flat.

    The designed effect materialised on incidence (seed-mean sim/ref 0.929 ->
    0.952, toward reference) but the registered modal failure materialised on
    the stock (0.841 -> 0.838, NOT lifted): the exposure-preserving split
    reweighted the elderly hazard by age without raising the aggregate, so the
    75+ widowed stock -- dominated by the slower-widowing 75-84 band -- did not
    recover, and share_widowed.75+|female still failed seeds 0 and 3.
    """
    a = _artifact()
    e = a["elderly_75plus_diagnostic"]
    inc = e["cells"]["widowhood.75+|female"]
    stock = e["cells"]["share_widowed.75+|female"]
    assert inc["c13_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_INCIDENCE_SIM_REF["c13"], abs=1e-3
    )
    assert inc["c14_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_INCIDENCE_SIM_REF["c14"], abs=1e-3
    )
    assert stock["c13_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_STOCK_SIM_REF["c13"], abs=1e-3
    )
    assert stock["c14_sim_over_ref_mean"] == pytest.approx(
        EXPECTED_STOCK_SIM_REF["c14"], abs=1e-3
    )
    summ = e["summary"]
    # incidence recovered toward reference; the stock did not lift.
    assert summ["incidence_sim_over_ref"]["recovered_toward_reference"] is True
    assert summ["stock_sim_over_ref"]["lifted_toward_reference"] is False
    assert e["forensics3_incidence_shortfall"] == 0.91


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


def test_candidate13_comparison_recomputes():
    a = _artifact()
    a13 = _artifact_c13()
    by13 = {s["seed"]: s for s in a13["per_seed"]}
    by14 = {s["seed"]: s for s in a["per_seed"]}
    comp = a["candidate13_comparison"]
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
        c14_np = sum(
            1 for s in GATE_SEEDS if by14[s]["gated_cells"][cell]["pass"]
        )
        c13_np = sum(
            1 for s in GATE_SEEDS if by13[s]["gated_cells"][cell]["pass"]
        )
        assert rec["c14_n_seeds_pass"] == c14_np
        assert rec["c13_n_seeds_pass"] == c13_np


def test_count_cells_stable_vs_c13():
    a = _artifact()
    cs = a["count_cell_stability"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = cs["cells"][cell]
        assert b["candidate13_n_seeds_pass"] == exp["c13"]
        assert b["n_seeds_pass"] == exp["c14"]
        assert b["stable_vs_c13"] is True
    # both marriage counts held candidate 13's 5/5 -- insulated by construction.
    assert cs["count_cells_hold"] is True
    assert cs["count_cells_stable_vs_c13"] is True


def test_modal_and_decider_recompute():
    a = _artifact()
    m = a["modal_failure_materialized"]
    assert list(m["modal_cells"]) == ["share_widowed.75+|female"]
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    # the 75+ stock still failed seeds 0 and 3, so the registered modal DID
    # materialise; forgiving it flips >= 4 seeds to pass, so it is the decider.
    assert m["modal_materialized"] is True
    assert m["modal_failed_seeds"] == [0, 3]
    assert m["modal_seed3_failed"] is True
    assert "registered modal" in dec["decider"]
    assert dec["n_seeds_pass_if_modal_forgiven"] >= 4


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v14"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate13_artifact_sha256"]) == 64
    assert len(pins["forensics3_artifact_sha256"]) == 64
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["candidate13_artifact"] == "runs/gate2_hazard_v13.json"
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
    comp = a["candidate13_comparison"]
    mc = comp["modal_cell"][MODAL_CELL]
    assert mc["c13_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c13"]
    assert mc["c14_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c14"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = comp["count_cells"][cell]
        assert b["c13_n_seeds_pass"] == exp["c13"]
        assert b["c14_n_seeds_pass"] == exp["c14"]


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 14 on seed 0's side B (train complement), once."""
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
def test_widowhood_level_is_seven_band_live(_live_seed0):
    """The fitted widowhood level has 14 cells; the five sub-75 bands match
    candidate 13's fit, and the pooled 75+ rate lies between the split rates.

    The five inherited bands are re-band-invariant: fitting candidate 13's
    six-band table on the SAME train set gives bit-identical rates below 75.
    The pooled 75+ rate (candidate 13's) sits strictly between candidate 14's
    75-84 and 85+ rates -- the gradient the pool averaged away.
    """
    runner, _panel, _demo, _ids_a, ids_b, components = _live_seed0
    assert len(components.mortality) == 14  # 7 bands x 2 sexes
    import populace_dynamics.data.transitions as transitions

    ev = _panel.events
    py = _panel.person_years
    ev = ev[ev["person_id"].isin(ids_b)]
    py = py[py["person_id"].isin(ids_b)]
    c13_cells = transitions._hazard_by_band(
        ev[ev["transition"] == "widowhood"],
        py[py["marital_state"] == "married"],
        "age",
        WIDOW_BANDS_C13,
        prefix="widowhood",
        by_sex=True,
        weighted=True,
    )
    # the five inherited bands (below 75) are bit-identical to the six-band fit.
    for band in ("18-34", "35-44", "45-54", "55-64", "65-74"):
        for sex in ("female", "male"):
            key = f"{band}|{sex}"
            assert components.mortality[key] == pytest.approx(
                c13_cells[f"widowhood.{key}"]["rate"], abs=1e-12
            ), f"{key} widowhood level moved vs the six-band fit"
    # the pooled 75+ rate lies strictly between the split 75-84 and 85+ rates.
    for sex in ("female", "male"):
        pooled = c13_cells[f"widowhood.75+|{sex}"]["rate"]
        lo = components.mortality[f"75-84|{sex}"]
        hi = components.mortality[f"85+|{sex}"]
        assert lo < pooled < hi


@needs_psid
def test_delta_is_live_widow_prob_split(_live_seed0):
    """The delta is live: an 85+ married ego widows above a 75-84 one.

    Candidate 13 pools ages 75-120 into one 75+ rate, so an 85-year-old and a
    78-year-old widow identically; candidate 14 gives the 85+ ego its own
    higher hazard (~1.7x the pool). The seed-0 draw-0 75+ widowhood incidence
    moved off candidate 13 as a consequence (the delta is NOT inert).
    """
    runner, _panel, _demo, _ids_a, _ids_b, components = _live_seed0
    lk14 = runner._build_sim_lookups(components)
    assert lk14.mort_arr.shape == (7, 2)
    fem = np.array([0.0])
    sp = np.array([80.0])
    male_opp = np.array([1.0])
    # a 78-year-old ego (75-84 band) vs a 90-year-old ego (85+ band), female.
    p_75_84 = runner._widow_probs(
        np.array([78.0]), fem, sp, male_opp, 2000, lk14.mort_arr, lk14.beta_arr
    )
    p_85plus = runner._widow_probs(
        np.array([90.0]), fem, sp, male_opp, 2000, lk14.mort_arr, lk14.beta_arr
    )
    assert p_85plus[0] > p_75_84[0]
    assert p_85plus[0] / p_75_84[0] > 1.5
    # the seed-0 draw-0 75+ widowhood incidence moved off candidate 13.
    assert (
        abs(SEED0_DRAW0["widowhood.75+|female"] - SEED0_DRAW0_C13_WIDOW75)
        > 1e-4
    )
