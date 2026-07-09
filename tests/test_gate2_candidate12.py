"""Tests for the gate-2 candidate-12 (run 1) pre-registered run.

Candidate 12 is the twelfth pre-registered gate-2 candidate: candidate 11's
frozen spec (comment 4919417729, ``scripts/run_gate2_candidate11.py``, merged
#101) verbatim EXCEPT EXACTLY TWO deltas, registered from gate-2 forensics 3
(#102):

* DELTA 1 -- entry-widowed observed initial state: persons observed
  already-widowed at their first observed PSID wave (all sexes, all ages) enter
  the simulation widowed. The reference-carried widowed person-years (widowhood
  onset year < first observed wave) are injected onto the simulated panel
  post-assembly, an RNG-neutral state injection reusing forensics 3's
  carried-status classification (the candidate-9 delta-1 precedent).
* DELTA 2 -- the spousal-age-gap draw is conditioned on the ego's age band at
  marriage (18-34/35-49/50-64/65+, 1-year gap bins, <200-weighted-couple
  fallback to the adjacent pooled band) instead of a single pooled distribution.

Everything else is byte-identical to candidate 11, and this is machine-checked:
candidate 12 REUSES candidate 11's (== candidate 10's) EXACT code objects for
the un-delta'd chain (``_draw_moments``, ``score_seed``,
``fit_remarriage_age_banded``, ``_build_sim_lookups``,
``_remarriage_probs_age_banded``), rebound to candidate 12's module globals so
the reused ``_draw_moments`` / ``score_seed`` call candidate 12's own
``simulate_holdout`` / ``fit_components``. The two delta'd functions
(``simulate_holdout``, ``fit_components``) are RE-IMPLEMENTED and pinned as
DIVERGED from candidate 11's bytecode.

The empirically decisive structural fact (``test_delta2_is_inert_live``): the
composed surviving-spouse widowhood hazard (candidate 6's ``_widow_probs``)
looks the level up by the married EGO's own ``(age, sex)`` and does NOT use the
imputed spouse age -- so the spousal-age-gap draw is VESTIGIAL, and DELTA 2 is
provably inert (candidate-12 with the age-band gap draw but the entry-widowed
injection disabled is bit-identical to candidate 11). All candidate-12 movement
is DELTA 1's injection.

The amended estimator (inherited, unchanged): per cell ``rbar_candidate,s`` is
the mean over K=20 draws (``default_rng(5200 + k)``, k=0..19) of the cell rate;
score ``|ln(rbar / rate_a,s)|`` scored once. The artifact conforms to
``fresh_run_artifact_schema``: the [20, 46, 5] per-draw per-cell rate cube,
undefined-draw invalidation, report-only dispersion. Frozen spec: issue #42
comment 4925020986.

Tiers:

* the always-runnable consistency + schema-conformance suite (touches only the
  committed candidate-12 / candidate-11 artifacts and ``gates.yaml``);
* the structural delta checks: the reused-code-object byte-identity attestation,
  the two diverged functions, the gap-band footprint / fallback, the
  entry-widowed carried reconciliation, and the draw-by-draw byte-identity of
  the delta-1-untouched cells (fertility + marriage counts) vs candidate 11;
* the live checks (skipped when the PSID history files are absent): the seed-0
  single-draw pin, the delta-2-inert attestation, and the delta-1 injection
  footprint.

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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v12.json"
ARTIFACT_C11 = ROOT / "runs" / "gate2_hazard_v11.json"
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
    "#issuecomment-4925020986"
)
SPEC_URL_C11 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4919417729"
)
REGISTRATION_POINTER = "4925020986"
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

# The reused candidate-11 (== candidate-10) code objects (byte-identity chain).
REUSED_CODE_OBJECT_NAMES = (
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_build_sim_lookups",
    "_remarriage_probs_age_banded",
)
# The two RE-IMPLEMENTED functions (the deltas): NOT candidate 11's bytecode.
DIVERGED_CODE_OBJECT_NAMES = ("simulate_holdout", "fit_components")

# Cells neither delta changes: fertility is marital-state-independent and the
# entry-widowed state injection touches no birth and no marriage-count event, so
# fertility and the marriage-count cells are byte-identical to candidate 11.
# (Delta 2 is inert; delta 1 is a state-only injection that leaves n_marriages,
# events and births untouched.)
DELTA_UNTOUCHED_PREFIXES = (
    "asfr.",
    "completed_fertility.",
    "mean_lifetime_marriages|",
)

# GAP ego-age bands (delta 2).
GAP_AGE_BANDS = [[18, 34], [35, 49], [50, 64], [65, 120]]

# ==========================================================================
# One-shot outcome pins (from runs/gate2_hazard_v12.json; published REGARDLESS
# of verdict).
# ==========================================================================
# FAIL 2/5 (seeds 0 and 1 clear all 46). Delta 1's entry-widowed injection
# cleared the modal share_widowed.75+|female (candidate 11's 1/5 -> 5/5), which
# flipped seed 0 to a pass; but DELTA 2 IS INERT (the spousal-age gap does not
# enter the composed widowhood hazard), so the female marriage count did not
# recover (still 2/5, bit-identical to candidate 11) and holds the gate below
# 4/5 on seeds 2, 3 and 4 (with seed 2 also failing completed_fertility.c1970s
# and the male count, both unchanged from candidate 11).
EXPECTED_GATE_PASS = False
EXPECTED_N_SEEDS_PASS = 2
EXPECTED_SEED_PASS = {
    "0": True,
    "1": True,
    "2": False,
    "3": False,
    "4": False,
}
EXPECTED_SEED_FAILS = {
    0: set(),
    1: set(),
    2: {
        "completed_fertility.c1970s",
        "mean_lifetime_marriages|female",
        "mean_lifetime_marriages|male",
    },
    3: {"mean_lifetime_marriages|female"},
    4: {"mean_lifetime_marriages|female"},
}
# The modal widow-stock cell cleared entirely (delta 1); the marriage counts did
# not move (delta 2 inert).
EXPECTED_MODAL_MOVEMENT = {"c11": 1, "c12": 5}
EXPECTED_COUNT_MOVEMENT = {
    "mean_lifetime_marriages|male": {"c11": 4, "c12": 4},
    "mean_lifetime_marriages|female": {"c11": 2, "c12": 2},
}
# Seed-0 draw-0 (default_rng(5200)) single-draw rates, live-reproducible on seed
# 0's side B. mean_lifetime_marriages|male is byte-identical to candidate 11
# (delta 2 inert, delta 1 count-neutral); first_marriage and remarriage move
# slightly under delta 1's state injection.
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.4395857412139383,
    "remarriage.after_divorce": 0.05887147518664376,
    "first_marriage.25-34|female": 0.07771323726223815,
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c11() -> dict:
    return json.loads(ARTIFACT_C11.read_text())


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
    import run_gate2_candidate12 as runner

    return runner


def _import_c11():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate11 as c11

    return c11


def _import_c9():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate9 as c9

    return c9


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    runner = _import_runner()
    assert runner.SPEC_REGISTRATION == SPEC_URL
    assert runner.REGISTRATION_POINTER == REGISTRATION_POINTER
    assert runner.CANDIDATE11_REGISTRATION == SPEC_URL_C11
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    assert runner.SIM_SEED_BASE == 4200
    # DELTA 2 bands (ego age at marriage).
    assert runner.GAP_AGE_BANDS == ((18, 34), (35, 49), (50, 64), (65, 120))
    assert runner.FALLBACK_MIN_WEIGHTED_COUPLES == 200.0
    # The 5-band remarriage table is UNCHANGED from candidate 11.
    assert runner.REM_AGE_BANDS == _import_c11().REM_AGE_BANDS
    # delta 1 (count) is candidate 9's, inherited, reused verbatim.
    assert (
        runner.observed_residual_counts
        is _import_c9().observed_residual_counts
    )


def test_reused_code_objects_are_candidate11_bytecode():
    """The reused chain is candidate 11's (== candidate 10's) exact bytecode."""
    runner = _import_runner()
    c11 = _import_c11()
    for name in REUSED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is getattr(c11, name).__code__
        ), f"{name} must reuse candidate 11's exact code object"
        assert getattr(runner, name).__globals__ is vars(runner)
    # the two delta'd functions are RE-IMPLEMENTED (diverged bytecode).
    for name in DIVERGED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is not getattr(c11, name).__code__
        ), f"{name} must be re-implemented for a candidate-12 delta"
    # the schema blocks are import-bound from candidate 10 unchanged.
    import run_gate2_candidate10 as c10

    for name in (
        "_per_draw_per_cell_rates_block",
        "_undefined_draw_block",
        "_per_draw_dispersion_block",
    ):
        assert getattr(runner, name) is getattr(c10, name)
    pins = _artifact()["revision_pins"]
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )
    assert all(
        pins["diverged_code_objects_vs_candidate11"][n]
        for n in DIVERGED_CODE_OBJECT_NAMES
    )


def test_delta_strings_name_the_two_deltas():
    runner = _import_runner()
    d = runner.DELTAS_VS_CANDIDATE11.lower()
    assert "two delta" in d or "exactly two" in d
    assert "entry-widowed" in d or "already-widowed" in d
    assert "age band" in d or "age-band" in d
    assert "byte-identical" in d
    assert "candidate 11" in d


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v12"
    assert a["candidate"] == "candidate 12"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_deltas_recorded():
    a = _artifact()
    assert a["candidate11_registration"] == SPEC_URL_C11
    assert "gate2_forensics3_v1.json" in a["forensics3_diagnostic"]
    model = a["model"]
    comp = model["components"]
    assert "band" in comp["spousal_age_gap"].lower()
    assert "18-34" in comp["spousal_age_gap"]
    assert (
        "widowed" in comp["entry_widowed_initial_state"].lower()
        and "inject" in comp["entry_widowed_initial_state"].lower()
    )
    assert "byte-identical" in comp["fertility"].lower()
    assert "byte-identical" in comp["remarriage"].lower()


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.5-0.6"
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
    """Delta 1's carried classification reproduces forensics 3's Q6 exactly."""
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
        assert row["candidate12_carried_cells_fixable_share"] == pytest.approx(
            committed[row["seed"]], abs=1e-9
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
        assert row["weighted_injected_widowed_py"] > 0.0


def test_gap_band_distribution_footprint():
    a = _artifact()
    for s in a["per_seed"]:
        gbd = s["component_meta"]["delta2_gap_band_distribution"]
        for sex in ("female", "male"):
            bands = gbd[sex]["bands"]
            assert set(bands) == {"18-34", "35-49", "50-64", "65+"}
            # the youngest band is populous and never falls back.
            assert bands["18-34"]["fell_back"] is False
            # every band's used array has a defined gap distribution.
            for cell in bands.values():
                assert cell["used_n"] > 0
        assert s["component_meta"]["delta2_gap_age_bands"] == GAP_AGE_BANDS


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
    assert sorted(d["max_per_draw_abs_ln_per_cell"]) == cells
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    cell = "mean_lifetime_marriages|male"
    for seed in GATE_SEEDS:
        rates = by_seed[seed]["gated_cells"][cell]["per_draw_rate"]
        expected = float(np.std(rates, ddof=1))
        assert d["per_cell_per_draw_sd"][cell][str(seed)] == pytest.approx(
            expected, abs=1e-12
        )


# --------------------------------------------------------------------------
# Structural delta checks: byte-identity of the delta-untouched cells vs c11
# --------------------------------------------------------------------------
def _is_delta_untouched(cell: str) -> bool:
    return any(cell.startswith(p) for p in DELTA_UNTOUCHED_PREFIXES)


def test_delta_untouched_cells_byte_identical_to_c11():
    """Fertility and the marriage counts equal candidate 11 draw-by-draw.

    Delta 2 (age-band gap) is inert -- the vestigial spouse-age gap does not
    enter the composed widowhood hazard -- and delta 1 (entry-widowed) injects
    only marital STATE (no birth, no marriage-count event), so fertility and the
    marriage-count cells are byte-identical to candidate 11 across every draw. A
    strong, always-runnable proof that only the widowed-state-derived cells
    moved.
    """
    a = _artifact()
    a11 = _artifact_c11()
    by12 = {s["seed"]: s for s in a["per_seed"]}
    by11 = {s["seed"]: s for s in a11["per_seed"]}
    n_checked = 0
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by12[seed][block].items():
                if not _is_delta_untouched(cell):
                    continue
                r11 = by11[seed][block][cell]["per_draw_rate"]
                r12 = rec["per_draw_rate"]
                assert len(r12) == len(r11) == N_DRAWS
                for k in range(N_DRAWS):
                    assert r12[k] == pytest.approx(
                        r11[k], abs=1e-12
                    ), f"{cell} seed {seed} draw {k} moved vs c11"
                n_checked += 1
    # asfr (6) + completed_fertility (5) gated + counts (2), over 5 seeds,
    # plus the report-only fertility cells.
    assert n_checked >= (6 + 5 + 2) * len(GATE_SEEDS)


def test_widow_stock_cells_moved_vs_c11():
    """Delta 1's injection lifts the widowed-stock cells off candidate 11."""
    a = _artifact()
    a11 = _artifact_c11()
    by12 = {s["seed"]: s for s in a["per_seed"]}
    by11 = {s["seed"]: s for s in a11["per_seed"]}
    moved = 0
    for seed in GATE_SEEDS:
        for cell in ("share_widowed.75+|female", "share_widowed.65-74|female"):
            r12 = by12[seed]["gated_cells"][cell]["rbar"]
            r11 = by11[seed]["gated_cells"][cell]["rbar"]
            # the injection ADDS carried widowed person-years (rbar rises).
            if r12 > r11 + 1e-9:
                moved += 1
    assert moved >= len(GATE_SEEDS)


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
    for f in a["verdict"]["all_failing_gated_cells"]:
        rec = by_seed[f["seed"]]["gated_cells"][f["cell"]]
        assert rec["pass"] is False
        assert rec["score"] > rec["tolerance"]
        assert f["score"] == pytest.approx(rec["score"], abs=1e-12)


# --------------------------------------------------------------------------
# Candidate-11 comparison + count tilt + young pool + modal (always runnable)
# --------------------------------------------------------------------------
def test_candidate11_comparison_recomputes():
    a = _artifact()
    a11 = _artifact_c11()
    by12 = {s["seed"]: s for s in a["per_seed"]}
    by11 = {s["seed"]: s for s in a11["per_seed"]}
    comp = a["candidate11_comparison"]
    blocks = {MODAL_CELL: comp["modal_cell"][MODAL_CELL]}
    for c in COUNT_CELLS:
        blocks[c] = comp["count_cells"][c]
    for cell, block in blocks.items():
        c12_np = 0
        c11_np = 0
        for row in block["per_seed"]:
            r12 = by12[row["seed"]]["gated_cells"][cell]
            r11 = by11[row["seed"]]["gated_cells"][cell]
            assert row["c12_score"] == pytest.approx(r12["score"], abs=1e-12)
            assert row["c11_score"] == pytest.approx(r11["score"], abs=1e-12)
            assert row["c12_rbar"] == pytest.approx(r12["rbar"], abs=1e-12)
            c12_np += r12["pass"]
            c11_np += r11["pass"]
        assert block["c12_n_seeds_pass"] == c12_np
        assert block["c11_n_seeds_pass"] == c11_np


def test_count_cells_identical_to_c11_in_comparison():
    """The marriage counts did not move vs candidate 11 (delta 2 inert)."""
    a = _artifact()
    comp = a["candidate11_comparison"]["count_cells"]
    for cell in COUNT_CELLS:
        for row in comp[cell]["per_seed"]:
            assert row["c12_rbar"] == pytest.approx(row["c11_rbar"], abs=1e-9)
            assert row["c12_score"] == pytest.approx(
                row["c11_score"], abs=1e-9
            )
        assert comp[cell]["c12_n_seeds_pass"] == comp[cell]["c11_n_seeds_pass"]


def test_young_pool_diagnostic_recomputes_and_c11_matches_forensics3():
    a = _artifact()
    yp = a["young_pool_diagnostic"]
    f3 = {s["seed"]: s for s in json.loads(FORENSICS3.read_text())["per_seed"]}
    for row in yp["per_seed"]:
        s = f3[row["seed"]]
        for lab, cell in row["bands"].items():
            key = f"{lab}|female"
            c11_ref = s["ref_widowed_by_age"][key]["widowed_share"]
            c11_sim = s["sim_widowed_by_age_mean"][key]["widowed_share"]
            # the reference share is side B's own panel (deterministic).
            assert cell["ref_widowed_share"] == pytest.approx(
                c11_ref, abs=1e-12
            )
            if c11_ref > 0:
                assert cell["c11_sim_over_ref"] == pytest.approx(
                    c11_sim / c11_ref, abs=1e-9
                )


def test_gap_band_table_present_in_comparison():
    a = _artifact()
    tbl = a["candidate11_comparison"]["gap_band_table_seed0"]["cells"]
    for sex in ("female", "male"):
        assert set(tbl[sex]["new_by_band"]) == {
            "18-34",
            "35-49",
            "50-64",
            "65+",
        }
        assert "old_pooled" in tbl[sex]


def test_modal_and_decider_recompute():
    a = _artifact()
    m = a["modal_failure_materialized"]
    assert list(m["modal_cells"]) == list(COUNT_CELLS)
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    female_count = "mean_lifetime_marriages|female"
    v_failed = sorted(
        f["seed"]
        for f in v["all_failing_gated_cells"]
        if f["cell"] == female_count
    )
    assert sorted(m["modal_failed_seeds"]) == v_failed


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v12"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 7, 8, 9, 10, 11):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate11_artifact_sha256"]) == 64
    assert len(pins["forensics3_artifact_sha256"]) == 64
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["candidate11_artifact"] == "runs/gate2_hazard_v11.json"
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
    comp = a["candidate11_comparison"]
    mc = comp["modal_cell"][MODAL_CELL]
    assert mc["c11_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c11"]
    assert mc["c12_n_seeds_pass"] == EXPECTED_MODAL_MOVEMENT["c12"]
    for cell, exp in EXPECTED_COUNT_MOVEMENT.items():
        b = comp["count_cells"][cell]
        assert b["c11_n_seeds_pass"] == exp["c11"]
        assert b["c12_n_seeds_pass"] == exp["c12"]


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 12 on seed 0's side B (train complement), once."""
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
def test_delta2_is_inert_live(_live_seed0):
    """DELTA 2 is provably inert: banded gap == pooled gap at every cell.

    The composed surviving-spouse widowhood hazard does not use the imputed
    spouse age, so conditioning the (vestigial) spousal-gap draw on the ego's
    age band changes NO moment. With the delta-1 injection disabled, candidate
    12's draw is bit-identical to candidate 11's at a shared draw seed.
    """
    runner, panel, _demo, ids_a, ids_b, components = _live_seed0
    c11 = _import_c11()
    import build_gate2_floors as g2f
    import run_gate2_candidate1 as c1

    from populace_dynamics.data import marriage
    from populace_dynamics.harness import panel as hpanel

    mh = marriage.marriage_history()
    birth = g2f.births.birth_history()
    death = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    order_map = c1._order_map(mh)
    _sa, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b2 = set(int(x) for x in side_b.person_id.unique())
    comp11 = c11.fit_components(
        panel, demo, death, mh, birth, order_map, ids_b2
    )
    orig = runner._inject_entry_widowed
    runner._inject_entry_widowed = lambda sp, cc: None
    try:
        m12 = runner._draw_moments(panel, ids_a, components, DRAW_SEED_BASE)
    finally:
        runner._inject_entry_widowed = orig
    m11 = c11._draw_moments(panel, ids_a, comp11, DRAW_SEED_BASE)
    assert ids_b == ids_b2
    for cell in m11:
        assert float(m12[cell]["rate"]) == pytest.approx(
            float(m11[cell]["rate"]), abs=1e-12
        ), f"{cell}: delta 2 is not inert (gap conditioning changed a moment)"


@needs_psid
def test_delta1_injection_lifts_widow_stock_live(_live_seed0):
    """Delta 1's injection raises the simulated 75+ female widowed stock."""
    runner, panel, _demo, ids_a, _ids_b, components = _live_seed0
    import run_gate2_candidate11 as c11

    sim12, _b12 = runner.simulate_holdout(
        panel, ids_a, components, DRAW_SEED_BASE
    )
    # with the injection disabled the stock is candidate 11's.
    orig = runner._inject_entry_widowed
    runner._inject_entry_widowed = lambda sp, cc: None
    try:
        sim_noinj, _bn = runner.simulate_holdout(
            panel, ids_a, components, DRAW_SEED_BASE
        )
    finally:
        runner._inject_entry_widowed = orig
    import populace_dynamics.data.transitions as transitions

    stock12 = transitions.stock_occupancy_cells(sim12, ids_a, weighted=True)
    stock_noinj = transitions.stock_occupancy_cells(
        sim_noinj, ids_a, weighted=True
    )
    r12 = stock12["share_widowed.75+|female"]["rate"]
    r0 = stock_noinj["share_widowed.75+|female"]["rate"]
    assert r12 > r0
    assert c11 is not None


@needs_psid
def test_entry_widowed_carried_reuses_forensics3_mask_live(_live_seed0):
    """Candidate 12's carried classification == forensics 3's, generalised."""
    runner, panel, demo, _ids_a, ids_b, _components = _live_seed0
    import gate2_forensics3 as f3

    support = f3.observed_support(demo)
    tax = f3.widowed_75plus_support_taxonomy(panel, support, ids_b)
    carried = runner.entry_widowed_carried_cells(panel, demo)
    # candidate 12's carried set restricted to 75+ female / side B reproduces
    # forensics 3's initial_state_fixable share.
    py = panel.person_years
    fem75 = py[
        py["person_id"].isin(ids_b)
        & (py["sex"] == "female")
        & (py["age"] >= 75)
    ]
    wid75 = fem75[fem75["marital_state"] == "widowed"]
    key = wid75["person_id"].to_numpy(dtype="int64") * 10000 + wid75[
        "year"
    ].to_numpy(dtype="int64")
    ks = carried["key_sorted"]
    pos = np.clip(np.searchsorted(ks, key), 0, max(ks.size - 1, 0))
    is_carried = (ks.size > 0) & (ks[pos] == key)
    w = wid75["weight"].to_numpy(dtype="float64")
    den = float(wid75["weight"].sum())
    share = float(w[is_carried].sum()) / den if den > 0 else 0.0
    assert share == pytest.approx(tax["initial_state_fixable_share"], abs=1e-9)
