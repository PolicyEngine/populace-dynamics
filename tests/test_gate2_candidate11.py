"""Tests for the gate-2 candidate-11 (run 1) pre-registered run.

Candidate 11 is the eleventh pre-registered gate-2 candidate: candidate 10's
frozen spec (comment 4917059482, ``scripts/run_gate2_candidate10.py``, merged
#98) verbatim EXCEPT EXACTLY ONE delta, registered from gate-2 forensics 2
(#99):

* THE DELTA -- the remarriage current-age conditioning splits candidate 10's
  pooled 50+ band into **50-64 / 65-74 / 75+**. Full structure:
  ``{18-34, 35-49, 50-64, 65-74, 75+}`` x years-since-dissolution band x origin
  (divorced/widowed) x sex, SAME add-one smoothing (``wbar_diss``) at the same
  weight convention as candidate 1. No other change.

Everything else is byte-identical to candidate 10, and this is machine-checked:
candidate 11 REUSES candidate 10's EXACT code objects for the band-dependent
compute chain (``simulate_holdout``, ``_draw_moments``, ``score_seed``,
``fit_remarriage_age_banded``, ``_build_sim_lookups``,
``_remarriage_probs_age_banded``), rebound to candidate 11's module globals so
the reused code reads the 5-band constants. ``test_reused_code_objects_are_c10``
asserts ``candidate11.f.__code__ is candidate10.f.__code__`` for each; the
byte-identity of every remarriage-independent gated cell is then confirmed
draw-by-draw against candidate 10's committed cube.

The amended estimator (inherited from candidate 10, unchanged): per cell
``rbar_candidate,s`` is the mean over K=20 draws (``default_rng(5200 + k)``,
k=0..19) of the cell rate; score ``|ln(rbar / rate_a,s)|`` scored once. The
artifact conforms to ``fresh_run_artifact_schema``: the [20, 46, 5] per-draw
per-cell rate cube, undefined-draw invalidation, report-only dispersion.
Frozen spec: issue #42 comment 4919417729.

Tiers:

* the always-runnable consistency + schema-conformance suite (touches only the
  committed candidate-11 / candidate-10 artifacts and ``gates.yaml``): the spec
  URLs and recorded delta, the amended estimator, the bit-exact precheck, the
  delta-1 reconciliation record, the [20, 46, 5] cube shape/index, ``rbar``
  recomputing cell-by-cell (and the score from ``rbar``), the undefined-draw
  check, the report-only dispersion, tolerances equal the locked gates.yaml,
  every stored gated-cell pass / seed pass / verdict / per-block count
  recomputing, the count-cell tilt and the candidate-10 comparison recomputing,
  and the modal / decider;
* the structural delta checks: the 60-cell age-banded footprint (5 bands
  18-34/35-49/50-64/65-74/75+), the reused-code-object byte-identity
  attestation, and the draw-by-draw byte-identity of the remarriage-independent
  cells vs candidate 10's committed cube;
* the live checks (skipped when the PSID history files are absent): the seed-0
  single-draw pin (one draw at 5200 reproduces the committed draw-0 rate) and
  the faithful-copy attestation (at a shared draw seed the remarriage-
  independent cells match candidate 10 exactly while the remarriage-driven
  cells move).

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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v11.json"
ARTIFACT_C10 = ROOT / "runs" / "gate2_hazard_v10.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4919417729"
)
SPEC_URL_C10 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4917059482"
)
REGISTRATION_POINTER = "4919417729"
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
THREE_TARGET_CELLS = (MODAL_CELL,) + COUNT_CELLS
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)

# THE DELTA footprint: 5 age bands x 3 ysd x 2 origin x 2 sex = 60 cells.
N_REMARRIAGE_AGE_BANDED_CELLS = 60
REM_AGE_BANDS = [[18, 34], [35, 49], [50, 64], [65, 74], [75, 120]]

# The reused candidate-10 code objects (the one-delta byte-identity contract).
REUSED_CODE_OBJECT_NAMES = (
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_build_sim_lookups",
    "_remarriage_probs_age_banded",
)

# Cells provably independent of the remarriage trajectory (never-married and
# fertility dynamics and ever-married status): byte-identical to candidate 10
# at a shared draw seed. First marriage never re-enters the never-married pool,
# fertility is marital-state-independent with the same draw block, and
# ever-married status is set once at first marriage.
BYTE_IDENTICAL_PREFIXES = (
    "first_marriage.",
    "ever_married_by_40|",
    "ever_married_by_60|",
    "ever_married_by_40.c",
    "asfr.",
    "completed_fertility.",
)

# ==========================================================================
# One-shot outcome pins (filled from runs/gate2_hazard_v11.json after the run;
# published REGARDLESS of verdict).
# ==========================================================================
# FAIL 1/5 (only seed 1 clears all 46); the registered modal
# (share_widowed.75+|female) materialised on seeds 0, 2, 3, 4 via the untouched
# inflow shortfall, and the female marriage count regressed 3/5 -> 2/5.
EXPECTED_GATE_PASS = False
EXPECTED_N_SEEDS_PASS = 1
EXPECTED_SEED_PASS = {
    "0": False,
    "1": True,
    "2": False,
    "3": False,
    "4": False,
}
EXPECTED_SEED_FAILS = {
    0: {"share_widowed.75+|female"},
    1: set(),
    2: {
        "completed_fertility.c1970s",
        "mean_lifetime_marriages|female",
        "mean_lifetime_marriages|male",
        "share_widowed.75+|female",
    },
    3: {"mean_lifetime_marriages|female", "share_widowed.75+|female"},
    4: {"mean_lifetime_marriages|female", "share_widowed.75+|female"},
}
# Count-cell tilt vs candidate 10 (mean signed ln + seeds passing): the male
# count held (net +0.025 ln, 4/5), the female count regressed (net +0.051 ln,
# 2/5 vs candidate 10's +0.044 ln, 3/5) -- the elderly split raised 50-64
# remarriage, adding to the count rather than removing over-production.
EXPECTED_COUNT_TILT = {
    "mean_lifetime_marriages|male": {"mean_signed": 0.0246, "n_pass": 4},
    "mean_lifetime_marriages|female": {"mean_signed": 0.0507, "n_pass": 2},
}
# Seed-0 draw-0 (default_rng(5200)) single-draw rates, pinned to float
# precision (live-reproducible on seed 0's side B). first_marriage.25-34|female
# is byte-identical to candidate 10's committed value.
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.4395857412139383,
    "remarriage.after_divorce": 0.058026332367248766,
    "first_marriage.25-34|female": 0.07753957436481304,
}
# Movement of the three registered target cells vs candidate 10 (n seeds pass):
# the widow stock and the male count held, the female count regressed.
EXPECTED_TARGET_MOVEMENT = {
    "share_widowed.75+|female": {"c10": 1, "c11": 1},
    "mean_lifetime_marriages|male": {"c10": 4, "c11": 4},
    "mean_lifetime_marriages|female": {"c10": 3, "c11": 2},
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c10() -> dict:
    return json.loads(ARTIFACT_C10.read_text())


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
    import run_gate2_candidate11 as runner

    return runner


def _import_c10():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate10 as c10

    return c10


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
    assert runner.CANDIDATE10_REGISTRATION == SPEC_URL_C10
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    assert runner.SIM_SEED_BASE == 4200
    # THE ONE DELTA: the pooled 50+ band split into 50-64 / 65-74 / 75+.
    assert runner.REM_AGE_BANDS == (
        (18, 34),
        (35, 49),
        (50, 64),
        (65, 74),
        (75, 120),
    )
    assert runner._REM_AGE_LABEL == {
        0: "18-34",
        1: "35-49",
        2: "50-64",
        3: "65-74",
        4: "75+",
    }
    assert runner.COUNT_CELLS == COUNT_CELLS
    assert runner.MODAL_CELL == MODAL_CELL
    assert runner.REGISTERED_MODAL_CELLS == (MODAL_CELL,)
    # delta 1 is candidate 9's, inherited via candidate 10, reused verbatim.
    assert (
        runner.observed_residual_counts
        is _import_c9().observed_residual_counts
    )
    assert (
        runner.observed_residual_counts
        is _import_c10().observed_residual_counts
    )


def test_reused_code_objects_are_candidate10_bytecode():
    """The one-delta contract: everything but the age bands is c10's bytecode.

    Candidate 11 reuses candidate 10's EXACT code objects for the whole
    band-dependent compute chain, rebound to candidate 11's globals. Sharing
    the code object makes ``everything else byte-identical`` machine-checkable.
    """
    runner = _import_runner()
    c10 = _import_c10()
    for name in REUSED_CODE_OBJECT_NAMES:
        assert (
            getattr(runner, name).__code__ is getattr(c10, name).__code__
        ), f"{name} must reuse candidate 10's exact code object"
        # rebound to candidate 11's module globals (so it reads the 5 bands).
        assert getattr(runner, name).__globals__ is vars(runner)
    # the schema blocks are import-bound from candidate 10 unchanged.
    for name in (
        "_per_draw_per_cell_rates_block",
        "_undefined_draw_block",
        "_per_draw_dispersion_block",
    ):
        assert getattr(runner, name) is getattr(c10, name)
    # and the artifact records the machine-check.
    pins = _artifact()["revision_pins"]["byte_identity_code_objects"]
    assert all(pins[name] for name in REUSED_CODE_OBJECT_NAMES)


def test_delta_string_names_the_one_delta():
    runner = _import_runner()
    d = runner.DELTA_VS_CANDIDATE10.lower()
    assert "one delta" in d or "exactly one" in d
    assert "50-64" in d and "65-74" in d and "75+" in d
    assert "byte-identical" in d
    assert "candidate 10" in d


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v11"
    assert a["candidate"] == "candidate 11"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_delta_recorded():
    a = _artifact()
    assert a["candidate10_registration"] == SPEC_URL_C10
    assert "gate2_forensics2_v1.json" in a["forensics2_diagnostic"]
    model = a["model"]
    rem = model["components"]["remarriage"].lower()
    assert "50-64" in rem and "65-74" in rem and "75+" in rem
    assert "split" in rem
    assert "byte-identical" in model["components"]["fertility"].lower()
    assert (
        "delta 1"
        in model["components"]["lifetime_marriage_count_initial_state"].lower()
        or "candidate 9"
        in model["components"]["lifetime_marriage_count_initial_state"].lower()
    )


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.55-0.65"
    assert "share_widowed.75+|female" in f["modal_failure"]
    assert "inflow" in f["modal_failure"]
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


def test_delta1_reconciliation_recorded():
    a = _artifact()
    rec = a["delta1_reconciliation"]
    assert rec["reconciled"] is True
    assert rec["per_person_identity_max_abs_residual"] <= 1e-12
    assert rec["aggregate_reconciliation_max_abs_remainder"] <= 1e-9
    assert rec["residual_nonnegative"] is True
    assert len(rec["per_seed"]) == len(GATE_SEEDS)


def test_delta_age_banded_remarriage_footprint():
    a = _artifact()
    for s in a["per_seed"]:
        ab = s["component_meta"]["remarriage_age_banded"]
        assert ab["n_cells"] == N_REMARRIAGE_AGE_BANDED_CELLS
        assert ab["age_bands"] == REM_AGE_BANDS
        assert "split" in ab["representation"].lower()
        assert "50-64" in ab["representation"]
        # candidate 8's order-split / rescale meta stays scrubbed.
        assert "remarriage_rescale" not in s["component_meta"]
        assert "remarriage_order_diagnostics" not in s["component_meta"]


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
    for c in cells:
        assert sorted(d["per_cell_per_draw_sd"][c]) == [
            str(s) for s in GATE_SEEDS
        ]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    cell = "mean_lifetime_marriages|male"
    for seed in GATE_SEEDS:
        rates = by_seed[seed]["gated_cells"][cell]["per_draw_rate"]
        expected = float(np.std(rates, ddof=1))
        assert d["per_cell_per_draw_sd"][cell][str(seed)] == pytest.approx(
            expected, abs=1e-12
        )


def test_dispersion_max_per_draw_abs_ln_recomputes():
    a = _artifact()
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    d = a["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    cell = "widowhood.75+|female"
    for seed in GATE_SEEDS:
        rec = by_seed[seed]["gated_cells"][cell]
        rate_a = rec["rate_a"]
        rates = rec["per_draw_rate"]
        excursions = [
            abs(math.log(r / rate_a)) for r in rates if r > 0 and rate_a > 0
        ]
        expected = max(excursions)
        assert d["max_per_draw_abs_ln_per_cell"][cell][
            str(seed)
        ] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# Byte-identity of the remarriage-independent cells vs candidate 10
# --------------------------------------------------------------------------
def _is_byte_identical(cell: str) -> bool:
    return any(cell.startswith(p) for p in BYTE_IDENTICAL_PREFIXES)


def test_remarriage_independent_cells_byte_identical_to_c10():
    """Every draw of every remarriage-independent gated cell equals c10's.

    The reused simulation is candidate 10's exact bytecode; only the remarriage
    threshold moves (THE DELTA). First marriage, fertility and ever-married
    status are untouched, so their per-draw rates equal candidate 10's committed
    cube to the bit -- a strong, always-runnable proof that only the remarriage
    trajectory moved.
    """
    a = _artifact()
    a10 = _artifact_c10()
    by11 = {s["seed"]: s for s in a["per_seed"]}
    by10 = {s["seed"]: s for s in a10["per_seed"]}
    n_checked = 0
    for seed in GATE_SEEDS:
        for block in ("gated_cells", "report_only_cells"):
            for cell, rec in by11[seed][block].items():
                if not _is_byte_identical(cell):
                    continue
                r10 = by10[seed][block][cell]["per_draw_rate"]
                r11 = rec["per_draw_rate"]
                assert len(r11) == len(r10) == N_DRAWS
                for k in range(N_DRAWS):
                    assert r11[k] == pytest.approx(
                        r10[k], abs=1e-12
                    ), f"{cell} seed {seed} draw {k} moved vs c10"
                n_checked += 1
    # first_marriage (6) + fertility (11) + nuptiality_cohort (5) +
    # ever_married occupancy (4) gated, plus report-only, over 5 seeds.
    assert n_checked >= (6 + 11 + 5 + 4) * len(GATE_SEEDS)


def test_remarriage_and_count_cells_moved_vs_c10():
    """The delta's targets moved off candidate 10 (the fit really changed)."""
    a = _artifact()
    a10 = _artifact_c10()
    by11 = {s["seed"]: s for s in a["per_seed"]}
    by10 = {s["seed"]: s for s in a10["per_seed"]}
    moved = 0
    for seed in GATE_SEEDS:
        for cell in REMARRIAGE_GATED_CELLS + COUNT_CELLS + (MODAL_CELL,):
            r11 = by11[seed]["gated_cells"][cell]["rbar"]
            r10 = by10[seed]["gated_cells"][cell]["rbar"]
            if abs(r11 - r10) > 1e-9:
                moved += 1
    # the elderly split reshapes remarriage on the 50+ exposure across seeds.
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


# --------------------------------------------------------------------------
# Candidate-10 comparison + count tilt + modal (always runnable)
# --------------------------------------------------------------------------
def test_candidate10_comparison_recomputes():
    a = _artifact()
    a10 = _artifact_c10()
    by11 = {s["seed"]: s for s in a["per_seed"]}
    by10 = {s["seed"]: s for s in a10["per_seed"]}
    comp = a["candidate10_comparison"]["three_target_cells"]
    for cell in THREE_TARGET_CELLS:
        block = comp[cell]
        c11_np = 0
        c10_np = 0
        for row in block["per_seed"]:
            r11 = by11[row["seed"]]["gated_cells"][cell]
            r10 = by10[row["seed"]]["gated_cells"][cell]
            assert row["c11_score"] == pytest.approx(r11["score"], abs=1e-12)
            assert row["c10_score"] == pytest.approx(r10["score"], abs=1e-12)
            assert row["c11_rbar"] == pytest.approx(r11["rbar"], abs=1e-12)
            c11_np += r11["pass"]
            c10_np += r10["pass"]
        assert block["c11_n_seeds_pass"] == c11_np
        assert block["c10_n_seeds_pass"] == c10_np


def test_target_movement_pinned():
    a = _artifact()
    comp = a["candidate10_comparison"]["three_target_cells"]
    for cell, exp in EXPECTED_TARGET_MOVEMENT.items():
        block = comp[cell]
        assert block["c10_n_seeds_pass"] == exp["c10"]
        assert block["c11_n_seeds_pass"] == exp["c11"]


def test_elderly_band_rate_table_75plus_below_pooled():
    """The 75+ widowed remarriage rate falls well below c10's pooled 50+ rate.

    The mechanism forensics 2 (#99) located: candidate 10's pooled 50+ rate is
    ~9x too high for 75+ widows. The split gives 75+ its own near-zero rate.
    """
    a = _artifact()
    table = a["candidate10_comparison"]["elderly_band_rate_table_seed0"][
        "cells"
    ]
    checked = 0
    for b in range(3):
        rec = table[f"widowed|female|ysd{b}"]
        pooled = rec["c10_pooled_50plus_hazard"]
        r75 = rec["c11_75plus_hazard"]
        if pooled is None or r75 is None:
            continue
        # the 75+ widowed-female rate is strictly below the pooled 50+ rate.
        assert r75 < pooled
        checked += 1
    assert checked >= 1


def test_count_cell_tilt_recomputes():
    a = _artifact()
    ct = a["count_cell_tilt"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    for cell in COUNT_CELLS:
        block = ct["cells"][cell]
        signed = []
        for row in block["per_seed"]:
            rec = by_seed[row["seed"]]["gated_cells"][cell]
            assert row["rbar"] == pytest.approx(rec["rbar"], abs=1e-15)
            expected_tilt = math.log(rec["rbar"] / rec["rate_a"])
            assert row["signed_ln_tilt"] == pytest.approx(
                expected_tilt, abs=1e-12
            )
            assert row["score_abs_ln"] == pytest.approx(
                rec["score"], abs=1e-12
            )
            signed.append(expected_tilt)
        assert block["mean_signed_ln_tilt"] == pytest.approx(
            float(np.mean(signed)), abs=1e-12
        )
        assert block["n_seeds_pass"] == sum(
            r["pass"] for r in block["per_seed"]
        )


def test_count_cell_tilt_pinned():
    a = _artifact()
    ct = a["count_cell_tilt"]
    assert ct["candidate10_female_net_ln"] == pytest.approx(0.044, abs=1e-9)
    assert ct["candidate10_male_net_ln"] == pytest.approx(0.025, abs=1e-9)
    for cell, exp in EXPECTED_COUNT_TILT.items():
        block = ct["cells"][cell]
        assert block["mean_signed_ln_tilt"] == pytest.approx(
            exp["mean_signed"], abs=5e-4
        )
        assert block["n_seeds_pass"] == exp["n_pass"]


def test_modal_and_decider_recompute():
    a = _artifact()
    m = a["modal_failure_materialized"]
    assert list(m["modal_cells"]) == [MODAL_CELL]
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    # the registered modal materialises iff share_widowed.75+|female fails >=2.
    failed = sorted(m["modal_failed_seeds"])
    assert m["modal_materialized"] == (len(failed) >= 2)
    # consistency with the verdict's failing-cell list.
    v_failed = sorted(
        f["seed"]
        for f in v["all_failing_gated_cells"]
        if f["cell"] == MODAL_CELL
    )
    assert failed == v_failed


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v11"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 7, 8, 9, 10):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate10_artifact_sha256"]) == 64
    assert len(pins["forensics2_artifact_sha256"]) == 64
    assert all(
        pins["byte_identity_code_objects"][n] for n in REUSED_CODE_OBJECT_NAMES
    )


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["candidate10_artifact"] == "runs/gate2_hazard_v10.json"
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 < fp["faithful_candidate_oc"] <= 1.0


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 11 on seed 0's side B (train complement), once."""
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
    return runner, panel, ids_a, components


@needs_psid
def test_seed0_single_draw_pin(_live_seed0):
    """One draw at default_rng(5200) reproduces the committed draw-0 rate."""
    runner, panel, ids_a, components = _live_seed0
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
def test_fit_footprint_is_60_cells_live(_live_seed0):
    """The live fit yields the 60-cell 5-band remarriage table."""
    _runner, _panel, _ids_a, components = _live_seed0
    ab = components.meta["remarriage_age_banded"]
    assert ab["n_cells"] == N_REMARRIAGE_AGE_BANDED_CELLS
    assert ab["age_bands"] == REM_AGE_BANDS
    # both the 65-74 and 75+ widowed-female cells exist (the split really fit).
    labels = set(ab["cells"])
    assert any("age65-74|" in c and "widowed|female" in c for c in labels)
    assert any("age75+|" in c and "widowed|female" in c for c in labels)


@needs_psid
def test_faithful_copy_of_candidate10_at_shared_seed(_live_seed0):
    """At a shared draw seed the remarriage-independent cells match c10.

    First marriage, the ever-married shares and fertility are independent of
    remarriage and the per-year RNG consumption is state-independent, so those
    cells are byte-identical to candidate 10; the remarriage-driven cells move.
    """
    runner, panel, ids_a, components = _live_seed0
    c10 = _import_c10()
    import build_gate2_floors as g2f
    import run_gate2_candidate1 as c1

    from populace_dynamics.data import marriage
    from populace_dynamics.harness import panel as hpanel

    mh = marriage.marriage_history()
    birth = g2f.births.birth_history()
    death = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    order_map = c1._order_map(mh)
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())
    comp10 = c10.fit_components(
        panel, demo, death, mh, birth, order_map, ids_b
    )

    c10m = c10._draw_moments(panel, ids_a, comp10, DRAW_SEED_BASE)
    c11m = runner._draw_moments(panel, ids_a, components, DRAW_SEED_BASE)

    n_identical = 0
    n_moved = 0
    for cell in c10m:
        d = abs(float(c10m[cell]["rate"]) - float(c11m[cell]["rate"]))
        if _is_byte_identical(cell):
            assert d <= 1e-12, f"{cell} should be byte-identical to c10"
            n_identical += 1
        elif cell.startswith("remarriage.") or cell.startswith(
            "mean_lifetime_marriages"
        ):
            if d > 1e-9:
                n_moved += 1
    assert n_identical >= 25
    assert n_moved >= 1


@needs_psid
def test_delta1_count_add_present_live(_live_seed0):
    """Delta 1 (inherited) adds the observed residual to marriage counts."""
    runner, panel, ids_a, _components = _live_seed0
    residual = runner.observed_residual_counts(panel)
    holdout_res = [residual.get(pid, 0.0) for pid in ids_a]
    assert min(holdout_res) >= 0.0
    assert max(holdout_res) > 0.0
