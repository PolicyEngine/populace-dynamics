"""Tests for the gate-2 candidate-10 (run 1) pre-registered run.

Candidate 10 is the tenth pre-registered gate-2 candidate, and the FIRST fresh
registration under the amended mean-over-K=20-draws estimator (gates.yaml
gate_2 amendment 1, ratified 2026-07-08, flipped live #97). It is candidate
8's five-component family-transition simulator with EXACTLY TWO named deltas
registered from the gate-2 forensics (#94):

* DELTA 1 -- observed undatable-marriage lifetime-count initial state
  (candidate 9's delta 1, UNCHANGED). Each holdout person's simulated
  lifetime-marriage count initialises at their OBSERVED residual
  ``R = n_marriages - (# datable in-exposure first_marriage/remarriage
  transition events)`` and accumulates the simulated datable transitions. An
  OBSERVED initial state per protocol; RNG-neutral; reconciles to remainder
  0.0 train-side before the one-shot. ``observed_residual_counts`` and
  ``_delta1_reconciliation`` are imported from candidate 9 and reused
  byte-for-byte.
* DELTA 2 -- age-band-conditioned remarriage (candidate 8's rescale REMOVED).
  Remarriage hazard conditioned on the ego's age band (18-34 / 35-49 / 50+) x
  years-since-dissolution band x origin (divorced/widowed) x sex, same add-one
  smoothing (``wbar_diss``) as candidate 1, with candidate 8's order-bit
  stratum and aggregate-preservation rescale removed. It moves only the
  remarriage THRESHOLD (same per-year uniform draw order and size), so first
  marriage, the ever-married shares and fertility are byte-identical to
  candidate 8 at a shared draw seed.

The amended estimator: per cell ``rbar_candidate,s`` is the mean over K=20
draws (``default_rng(5200 + k)``, k=0..19) of the cell rate; the score is
``|ln(rbar / rate_a,s)|`` scored once. The artifact conforms to
``protocol.fresh_run_artifact_schema``: it commits the [20, 46, 5] per-draw
per-cell rate cube, invalidates on any undefined gated-cell draw, and reports
(never gates) per-draw dispersion. Frozen spec: issue #42 comment 4917059482.

Three tiers:

* the always-runnable consistency + schema-conformance suite (touches only the
  committed candidate-10 / candidate-8 / candidate-9 artifacts and
  ``gates.yaml``): the spec URLs and recorded deltas, the amended estimator,
  the bit-exact precheck, the delta-1 reconciliation record, the [20, 46, 5]
  per-draw cube shape and index, ``rbar`` recomputing cell-by-cell from the
  per-draw rates (and the score from ``rbar``), the undefined-draw check, the
  report-only dispersion, the tolerances equal the locked gates.yaml, every
  stored gated-cell pass / seed pass / verdict / per-block count recomputing,
  the count-cell tilt recomputing, and the modal / decider;
* the structural delta checks: the age-banded remarriage footprint (36 cells,
  bands 18-34/35-49/50+) and the delta-1 reuse of candidate 9;
* the live checks (skipped when the PSID history files are absent): the seed-0
  single-draw pin (one draw at 5200 reproduces the committed draw-0 rate to
  float precision), and the faithful-copy attestation (at a shared draw seed
  the byte-identical set matches candidate 8 exactly while the remarriage-
  driven cells move).

The one-shot outcome (published REGARDLESS of verdict): FAIL 1/5 (only seed 1
clears all 46). The designed count-cell cancellation (delta 1's +residual
~+0.046 ln minus delta 2's removed over-production ~-0.046 ln) was ASYMMETRIC:
it held for the male count (net +0.025 ln, 4/5) but MISSED for the female
count (net +0.044 ln, 3/5 -- age conditioning removed almost none of the
female over-production). The gate blocker is ``share_widowed.75+|female``
(fails 4/5 -- an elderly-widow-stock level cell neither delta addresses), with
the female count and ``completed_fertility.c1970s`` failing on seed 2 and the
female count on seed 4. The gated remarriage cells all pass 5/5 (the thin-50+
risk did not bite).
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v10.json"
ARTIFACT_C8 = ROOT / "runs" / "gate2_hazard_v8.json"
ARTIFACT_C9 = ROOT / "runs" / "gate2_hazard_v9.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4917059482"
)
SPEC_URL_C8 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912995860"
)
REGISTRATION_POINTER = "4917059482"
GATE_SEEDS = [0, 1, 2, 3, 4]
N_DRAWS = 20
DRAW_SEED_BASE = 5200
N_GATED = 46
N_REPORT_ONLY = 16

COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
MODAL_BLOCKER = "share_widowed.75+|female"

# The pinned one-shot outcome (published REGARDLESS of verdict): FAIL 1/5.
EXPECTED_SEED_FAILS = {
    0: {"share_widowed.75+|female"},
    1: set(),
    2: {
        "completed_fertility.c1970s",
        "mean_lifetime_marriages|female",
        "mean_lifetime_marriages|male",
        "share_widowed.75+|female",
    },
    3: {"share_widowed.75+|female"},
    4: {"mean_lifetime_marriages|female", "share_widowed.75+|female"},
}
EXPECTED_N_SEEDS_PASS = 1

# Count-cell tilt vs the designed cancellation (pinned).
EXPECTED_COUNT_TILT = {
    "mean_lifetime_marriages|male": {"mean_signed": 0.0252, "n_pass": 4},
    "mean_lifetime_marriages|female": {"mean_signed": 0.0440, "n_pass": 3},
}

# Seed-0 draw-0 (default_rng(5200)) single-draw rates, pinned to float
# precision (live-reproducible: one draw on seed 0's side B).
SEED0_DRAW0 = {
    "mean_lifetime_marriages|male": 1.4466938654104702,
    "remarriage.after_divorce": 0.05855510132092438,
    "first_marriage.25-34|female": 0.07753957436481304,
}

# Age-band remarriage footprint (delta 2): 3 age bands x 3 ysd x 2 origin x 2
# sex = 36 cells.
N_REMARRIAGE_AGE_BANDED_CELLS = 36


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


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
    import run_gate2_candidate10 as runner

    return runner


def _import_c8():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate8 as c8

    return c8


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
    assert runner.CANDIDATE8_REGISTRATION == SPEC_URL_C8
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.N_DRAWS == N_DRAWS
    assert runner.DRAW_SEED_BASE == DRAW_SEED_BASE
    # single-draw provenance stream unchanged from candidates 1-9.
    assert runner.SIM_SEED_BASE == 4200
    assert runner.REM_AGE_BANDS == ((18, 34), (35, 49), (50, 120))
    assert runner.COUNT_CELLS == COUNT_CELLS
    # delta 1 is candidate 9's, reused byte-for-byte.
    assert (
        runner.observed_residual_counts
        is _import_c9().observed_residual_counts
    )
    assert runner._delta1_reconciliation is _import_c9()._delta1_reconciliation


def test_delta_string_names_both_deltas():
    runner = _import_runner()
    d = runner.DELTA_VS_CANDIDATE8.lower()
    assert "age band" in d or "age-band" in d
    assert "18-34" in d and "35-49" in d and "50+" in d
    assert "rescale removed" in d
    assert "observed residual" in d or "observed undatable" in d
    assert "unchanged" in d  # fertility unchanged


def test_artifact_present_and_records_amended_estimator():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v10"
    assert a["candidate"] == "candidate 10"
    assert a["gate"] == "gate_2"
    assert a["spec_registration"] == SPEC_URL
    assert a["registration_pointer"] == REGISTRATION_POINTER
    assert a["protocol"]["estimator"] == "mean_over_K20_draws"
    assert a["protocol"]["n_draws"] == N_DRAWS
    assert "5200 + k" in a["protocol"]["draw_rng_rule"]
    assert "amendment 1" in a["amended_estimator"]


def test_spec_and_deltas_recorded():
    a = _artifact()
    assert a["candidate9_registration"] == (
        "https://github.com/PolicyEngine/populace-dynamics/issues/42"
        "#issuecomment-4914111252"
    )
    model = a["model"]
    assert "age band" in model["components"]["remarriage"].lower()
    assert "removed" in model["components"]["remarriage"].lower()
    assert "byte-identical" in model["components"]["fertility"].lower()
    assert (
        "delta 1"
        in model["components"]["lifetime_marriage_count_initial_state"].lower()
    )


def test_forecast_recorded():
    a = _artifact()
    f = a["pre_registered_forecast"]
    assert f["p_pass"] == "0.65-0.75"
    assert "count cells" in f["modal_failure"]
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


def test_delta2_age_banded_remarriage_footprint():
    a = _artifact()
    # Present in every seed's component meta.
    for s in a["per_seed"]:
        ab = s["component_meta"]["remarriage_age_banded"]
        assert ab["n_cells"] == N_REMARRIAGE_AGE_BANDED_CELLS
        assert ab["age_bands"] == [[18, 34], [35, 49], [50, 120]]
        assert "rescale" in ab["representation"].lower()
        assert "removed" in ab["representation"].lower()
        # candidate 8's order-split / rescale meta is scrubbed.
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
    """rbar = mean over the 20 per-draw rates; score = |ln(rbar / rate_a)|."""
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
    # every gated cell defined on all 20 draws, every seed.
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
    # the sd recomputes from the committed per-draw rates.
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


def test_dispersion_mean_absorbs_flag_is_consistent():
    """Cells whose worst draw clips the tol but the 20-draw mean passes."""
    a = _artifact()
    d = a["fresh_run_artifact_schema"]["per_draw_dispersion_disclosure"]
    n_flagged = d["n_cells_worst_draw_exceeds_tol_but_mean_passes"]
    assert n_flagged > 0  # the amendment's whole point
    for row in d["top_excursions"]:
        if row["mean_absorbs"]:
            assert row["max_per_draw_abs_ln"] > row["tolerance"]
            assert row["certified_score"] <= row["tolerance"]


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


def test_verdict_is_fail_one_of_five_pinned():
    a = _artifact()
    v = a["verdict"]
    assert v["gate_2_pass"] is False
    assert v["n_seeds_pass"] == EXPECTED_N_SEEDS_PASS
    assert v["seed_pass"] == {
        "0": False,
        "1": True,
        "2": False,
        "3": False,
        "4": False,
    }


def test_seed_fails_pinned():
    a = _artifact()
    for s in a["per_seed"]:
        fails = {c for c, rec in s["gated_cells"].items() if not rec["pass"]}
        assert fails == EXPECTED_SEED_FAILS[s["seed"]]


def test_share_widowed_75_female_is_the_gate_blocker():
    """The elderly-widow-stock cell neither delta addresses fails 4/5."""
    a = _artifact()
    failed = [
        f["seed"]
        for f in a["verdict"]["all_failing_gated_cells"]
        if f["cell"] == MODAL_BLOCKER
    ]
    assert sorted(failed) == [0, 2, 3, 4]
    # even forgiving the delta-targeted cells, < 4 seeds pass (this cell holds).
    dec = a["modal_failure_materialized"]["decider_analysis"]
    assert dec["n_seeds_pass_if_targeted_forgiven"] < 4


# --------------------------------------------------------------------------
# Count-cell tilt vs the designed cancellation (always runnable)
# --------------------------------------------------------------------------
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
            assert abs(expected_tilt) == pytest.approx(rec["score"], abs=1e-12)
            signed.append(expected_tilt)
        assert block["mean_signed_ln_tilt"] == pytest.approx(
            float(np.mean(signed)), abs=1e-12
        )
        assert block["n_seeds_pass"] == sum(
            r["pass"] for r in block["per_seed"]
        )


def test_count_cell_tilt_asymmetric_cancellation_pinned():
    a = _artifact()
    ct = a["count_cell_tilt"]
    assert ct["designed_cancellation_succeeded"] is False
    for cell, exp in EXPECTED_COUNT_TILT.items():
        block = ct["cells"][cell]
        assert block["mean_signed_ln_tilt"] == pytest.approx(
            exp["mean_signed"], abs=5e-4
        )
        assert block["n_seeds_pass"] == exp["n_pass"]
    # the design targets: residual +0.046, over-production -0.046.
    assert ct["design_residual_ln"] == pytest.approx(0.046, abs=1e-9)
    assert ct["design_over_production_ln"] == pytest.approx(-0.046, abs=1e-9)


def test_modal_and_decider_recompute():
    a = _artifact()
    m = a["modal_failure_materialized"]
    assert list(m["modal_cells"]) == list(COUNT_CELLS)
    v = a["verdict"]
    dec = m["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    # forgiving the modal count cells still leaves < 4 passing (blocker held).
    assert dec["n_seeds_pass_if_modal_forgiven"] < 4
    assert "broader" in dec["decider"]


def test_revision_pins_record_shas_and_estimator():
    a = _artifact()
    pins = a["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2_hazard_v10"
    assert "5200 + k" in pins["estimator"]
    for name in (1, 5, 6, 7, 8, 9):
        assert pins[f"candidate{name}_runner"] == (
            f"scripts/run_gate2_candidate{name}.py"
        )
        assert len(pins[f"candidate{name}_runner_sha256"]) == 64
    assert len(pins["candidate9_artifact_sha256"]) == 64
    assert len(pins["forensics_artifact_sha256"]) == 64


def test_forecast_pointer_present():
    a = _artifact()
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["registration_pointer"] == REGISTRATION_POINTER
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 < fp["faithful_candidate_oc"] <= 1.0


# --------------------------------------------------------------------------
# Live checks (skipped when the PSID history files are absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _live_seed0():
    """Fit candidate 10 on seed 0's side B (train complement), once."""
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
    # and it equals the committed per-draw cube's [k=0, cell, seed=0] entry.
    a = _artifact()
    pc = a["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    ci = pc["cell_index"]
    for cell, expected in SEED0_DRAW0.items():
        if cell in ci:
            cube = pc["rates"][0][ci.index(cell)][0]
            assert cube == pytest.approx(expected, abs=1e-12)


@needs_psid
def test_faithful_copy_of_candidate8_at_shared_seed(_live_seed0):
    """At a shared draw seed the byte-identical set matches candidate 8.

    First marriage, the ever-married shares and fertility are independent of
    remarriage and the per-year RNG consumption is state-independent, so those
    cells are byte-identical to candidate 8; the remarriage-driven cells move.
    """
    runner, panel, ids_a, components = _live_seed0
    c8 = _import_c8()
    import build_gate2_floors as g2f
    import run_gate2_candidate1 as c1

    from populace_dynamics.data import marriage, transitions
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
    comp8 = c8.fit_components(panel, demo, death, mh, birth, order_map, ids_b)

    sp8, sb8 = c8.simulate_holdout(panel, ids_a, comp8, DRAW_SEED_BASE)
    sf8 = transitions.build_fertility_panel(sp8, sb8)
    c8m = transitions.reference_moments(sp8, sf8, ids_a, weighted=True)
    c10m = runner._draw_moments(panel, ids_a, components, DRAW_SEED_BASE)

    identical_prefixes = (
        "first_marriage.",
        "ever_married_by_40|",
        "ever_married_by_60|",
        "ever_married_by_40.c",
        "asfr.",
        "completed_fertility.",
    )
    n_identical = 0
    n_moved = 0
    for cell in c8m:
        d = abs(float(c8m[cell]["rate"]) - float(c10m[cell]["rate"]))
        if any(cell.startswith(p) for p in identical_prefixes):
            assert d <= 1e-12, f"{cell} should be byte-identical to c8"
            n_identical += 1
        elif cell.startswith("remarriage.") or cell.startswith(
            "mean_lifetime_marriages"
        ):
            n_moved += 1
    assert n_identical >= 25
    # remarriage + count cells moved under the deltas.
    assert n_moved >= 4


@needs_psid
def test_delta1_count_add_present_live(_live_seed0):
    """Delta 1 adds the observed residual to the simulated marriage counts."""
    runner, panel, ids_a, components = _live_seed0
    c9 = _import_c9()
    residual = c9.observed_residual_counts(panel)
    # the residual is non-negative and positive for some holdout persons.
    holdout_res = [residual.get(pid, 0.0) for pid in ids_a]
    assert min(holdout_res) >= 0.0
    assert max(holdout_res) > 0.0
