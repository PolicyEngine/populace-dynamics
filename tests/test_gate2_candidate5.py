"""Tests for the gate-2 candidate-5 (run 1) pre-registered run.

Candidate 5 is the fifth pre-registered gate-2 candidate: candidate 4's
five-component family-transition simulator with EXACTLY three named deltas --
(1) the spouse-death mortality *trend* ``beta_sex`` is fixed from EXTERNAL
NCHS life tables (the log-linear slope of the age-45-84-band-average ``qx``
across three US life-table vintages, 2000/2010/2023) rather than candidate
4's train-fit Poisson slope, applied with candidate 4's unchanged form
``rate(age, sex, year) = pooled_PSID(age, sex) * exp(beta_sex *
(year - 1995))``; (2) fertility is estimated at single-year-of-age within
parity x cohort and triangular-kernel smoothed over age (bandwidth 3),
replacing the 5-year ASFR band table; and (3) remarriage duration-band
hazards estimated separately by origin -- candidate 1's remarriage table,
inherited byte-identically through candidates 2-4, is ALREADY the origin-
split ``(ysd_band, origin, sex)`` construction, so this delta pins it
explicitly (the remarriage hazard is byte-identical to candidate 4).
Everything else is byte-identical to candidate 4. Frozen spec: issue #42
comment 4911788302.

Three tiers, mirroring the candidate-4 suite:

* the always-runnable consistency suite (touches only the committed
  artifacts, the committed NCHS references, and ``gates.yaml``): schema and
  spec URLs, the three recorded deltas, the bit-exact reproduction precheck
  attestation, every stored gated-cell pass recomputes from its score against
  its stored (locked) tolerance, the stored tolerances equal the locked
  gates.yaml, each seed's pass recomputes, the verdict and per-block counts
  recompute, the report-only cells never gate, the registered modal
  (``mean_lifetime_marriages|male``), the targeted cells, the decider
  analysis, the candidate-4 movement and the NCHS-vs-train-fit beta comparison
  all recompute, fertility is NOT byte-identical to candidate 1 (delta 2), and
  the forecast / registration / revision pins are carried;
* structural delta checks: delta 1 records the external-NCHS trend, its
  per-sex betas and the three-vintage band-average qx diagnostics (and
  retains candidate 4's train-fit betas for the comparison); delta 2 records
  the single-year triangular-kernel fertility construction; delta 3 records
  the explicit origin split; the first-marriage fitter is candidate 3's
  object (reused by identity); and the two committed historical NCHS
  references (2000, 2010) pass the standard life-table provenance / sanity
  checks;
* :func:`test_seed0_reproduces_committed_artifact` and the live delta checks
  (skipped when the PSID history files are absent) rerun seed 0 end-to-end and
  pin the committed seed-0 block to float precision, confirm the NCHS betas
  reproduce and equal the external slope computed from the committed
  references, confirm the non-delta components (remarriage, divorce, gap,
  first marriage) are byte-identical to candidate 4, and confirm the NEW
  single-year fertility construction reproduces its own seed-0 outcome and
  differs from candidate 1's band fertility.
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
ARTIFACT_C1 = ROOT / "runs" / "gate2_hazard_v1.json"
ARTIFACT_C4 = ROOT / "runs" / "gate2_hazard_v4.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"
NCHS_DIR = ROOT / "data" / "external"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911788302"
)
SPEC_URL_C4 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911532899"
)
SPEC_URL_C1 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4910914098"
)
GATE_SEEDS = [0, 1, 2, 3, 4]
N_GATED = 46
N_REPORT_ONLY = 16

MODAL_CELL = "mean_lifetime_marriages|male"
TARGETED_CELLS = (
    "share_widowed.75+|female",
    "asfr.20-24",
    "completed_fertility.c1970s",
    "remarriage.after_divorce",
)
NCHS_VINTAGES = (2000, 2010, 2023)
NCHS_PUBLISHED_EX0 = {
    2000: {"total": 76.9, "male": 74.1, "female": 79.5},
    2010: {"total": 78.7, "male": 76.2, "female": 81.0},
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c1() -> dict:
    return json.loads(ARTIFACT_C1.read_text())


def _artifact_c4() -> dict:
    return json.loads(ARTIFACT_C4.read_text())


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
    import run_gate2_candidate5 as runner

    return runner


def _import_c3():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate3 as c3

    return c3


def _import_c4():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate4 as c4

    return c4


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses candidate 1's dials.

    The frozen dials are IMPORTED from candidate 1, the trend anchor (1995)
    from candidate 4, and the first-marriage fitter from candidate 3 -- the
    provenance of "candidate 4 verbatim except the three deltas".
    """
    runner = _import_runner()
    c3 = _import_c3()
    c4 = _import_c4()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.SIM_SEED_BASE == 4200
    assert runner.TREND_ANCHOR_YEAR == 1995.0
    assert runner.TREND_ANCHOR_YEAR is c4.TREND_ANCHOR_YEAR
    # First marriage is candidate 3's, reused unchanged (knot-at-22 spline).
    assert runner.SPLINE_KNOTS_C3 == (20.0, 22.0, 25.0, 30.0, 40.0)
    assert runner.fit_first_marriage is c3.fit_first_marriage
    assert runner.FirstMarriageModelC5 is c3.FirstMarriageModelC3
    # DELTA 1 external-NCHS constants.
    assert runner.NCHS_VINTAGE_YEARS == (2000, 2010, 2023)
    assert (runner.NCHS_BAND_AGE_LO, runner.NCHS_BAND_AGE_HI) == (45, 84)
    # DELTA 2 single-year fertility constants.
    assert (runner.FERT_AGE_LO, runner.FERT_AGE_HI) == (15, 49)
    assert runner.FERT_KERNEL_BANDWIDTH == 3
    # The widow-prob helper IS candidate 4's (unchanged functional form).
    assert runner._widow_probs is c4._widow_probs
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v5"
    assert "4911788302" in runner.SPEC_REGISTRATION
    assert "4911532899" in runner.CANDIDATE4_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert len(runner.DELTAS_VS_CANDIDATE4) == 3


def test_triangular_kernel_weights():
    """The pre-registered triangular kernel: bandwidth 3, support |d| <= 2."""
    runner = _import_runner()
    w = runner._triangular_kernel_weights()
    assert set(w) == {-2, -1, 0, 1, 2}
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(2 / 3) and w[-1] == pytest.approx(2 / 3)
    assert w[2] == pytest.approx(1 / 3) and w[-2] == pytest.approx(1 / 3)


def test_ols_slope_matches_numpy():
    """The closed-form OLS slope equals numpy's polyfit (deterministic)."""
    runner = _import_runner()
    x = np.array([2000.0, 2010.0, 2023.0])
    y = np.array([-3.90, -4.07, -4.12])
    assert runner._ols_slope(x, y) == pytest.approx(
        float(np.polyfit(x, y, 1)[0]), abs=1e-12
    )


# --------------------------------------------------------------------------
# Delta 1: external NCHS mortality trend
# --------------------------------------------------------------------------
def test_delta1_external_nchs_trend_recorded():
    """Every seed records the external-NCHS trend, its betas and diagnostics.

    Candidate 5 keeps candidate 4's functional form and 1995 anchor but sets
    ``beta_sex`` from the external NCHS band-average-qx slope; the train-fit
    betas are retained for the comparison, and the three-vintage diagnostics
    are carried.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert meta["mortality_trend_anchor_year"] == 1995.0
        assert meta["mortality_trend_source"] == (
            "external NCHS US life tables (2000, 2010, 2023)"
        )
        assert "OLS slope" in meta["mortality_trend_estimator"]
        assert meta["mortality_cells"] == 14  # 7 bands x 2 sexes, unchanged
        beta = meta["mortality_beta_by_sex"]
        assert set(beta) == {"female", "male"}
        # NCHS betas are negative (mortality declines) and constant per seed.
        assert beta["female"] < 0 and beta["male"] < 0
        # Candidate 4's train-fit betas retained for the comparison.
        tf = meta["mortality_beta_by_sex_candidate4_trainfit"]
        assert set(tf) == {"female", "male"}
        diag = meta["mortality_trend_diagnostics_nchs"]
        assert diag["vintages"] == [2000, 2010, 2023]
        assert (diag["band_age_lo"], diag["band_age_hi"]) == (45, 84)
        for sex in ("female", "male"):
            ps = diag["per_sex"][sex]
            assert ps["beta"] == pytest.approx(beta[sex], abs=0)
            qbar = ps["band_average_qx_by_vintage"]
            assert set(qbar) == {"2000", "2010", "2023"}
            # Mortality declined 2000 -> 2023 (band-average qx falls).
            assert qbar["2000"] > qbar["2010"] > qbar["2023"]


def test_delta1_betas_equal_external_reference_slope():
    """The recorded betas equal the OLS slope of the committed NCHS references.

    Recompute ``beta_sex`` directly from the three committed
    ``nchs_life_tables_*.json`` references and confirm it matches the value the
    runner stored -- the external trend is reproducible from the committed
    provenance alone (no PSID).
    """
    a = _artifact()
    beta = a["per_seed"][0]["component_meta"]["mortality_beta_by_sex"]
    years = np.array(NCHS_VINTAGES, dtype=float)
    for sex in ("female", "male"):
        qbar = []
        for y in NCHS_VINTAGES:
            ref = json.loads(
                (NCHS_DIR / f"nchs_life_tables_{y}.json").read_text()
            )
            rows = {r["age"]: r for r in ref["tables"][sex]}
            qbar.append(np.mean([rows[age]["qx"] for age in range(45, 85)]))
        slope = float(np.polyfit(years, np.log(qbar), 1)[0])
        assert beta[sex] == pytest.approx(slope, abs=1e-9)


def test_beta_comparison_block_recorded():
    """The NCHS-vs-train-fit beta comparison is carried and consistent."""
    a = _artifact()
    cmp = a["mortality_trend_beta_comparison"]
    nchs = cmp["beta_sex_nchs"]
    assert set(nchs) == {"female", "male"}
    per_seed_tf = cmp["beta_sex_candidate4_trainfit_per_seed"]
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert nchs == meta["mortality_beta_by_sex"]
        assert per_seed_tf[str(seed["seed"])] == (
            meta["mortality_beta_by_sex_candidate4_trainfit"]
        )


# --------------------------------------------------------------------------
# Delta 2: single-year-of-age triangular-kernel fertility
# --------------------------------------------------------------------------
def test_delta2_single_year_fertility_recorded():
    """Every seed records the single-year triangular-kernel fertility build."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert "single-year-of-age" in meta["fertility_representation"]
        assert "triangular" in meta["fertility_representation"]
        assert meta["fertility_kernel"] == "triangular"
        assert meta["fertility_kernel_bandwidth"] == 3
        summ = meta["fertility_single_year_summary"]
        assert summ["age_resolution"] == "single_year_of_age"
        assert summ["age_range"] == [15, 49]
        assert summ["n_ages"] == 35
        assert summ["parity_bands"] == [0, 1, 2, 3]


def test_delta2_fertility_not_byte_identical_to_candidate1():
    """Fertility byte-identity vs candidate 1 no longer holds (by design).

    Delta 2 replaces the 5-year band table with a single-year kernel-smoothed
    one, so at least one committed ``asfr`` / ``completed_fertility`` cell must
    move off candidate 1's value -- the pinned "no longer byte-identical"
    property. (The marital cells remain unperturbed, tested live.)
    """
    a5 = _artifact()
    a1 = _artifact_c1()
    by1 = {s["seed"]: s for s in a1["per_seed"]}
    by5 = {s["seed"]: s for s in a5["per_seed"]}
    moved = 0
    for seed in GATE_SEEDS:
        s1, s5 = by1[seed], by5[seed]
        for cell, rec in s5["gated_cells"].items():
            if _is_fertility(cell):
                if not math.isclose(
                    rec["r_candidate"],
                    s1["gated_cells"][cell]["r_candidate"],
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    moved += 1
    assert moved > 0, "delta 2 must move at least one fertility cell vs c1"


# --------------------------------------------------------------------------
# Delta 3: origin-split remarriage (explicit; byte-identical hazard)
# --------------------------------------------------------------------------
def test_delta3_origin_split_remarriage_recorded():
    """Every seed records the explicit origin-split remarriage construction."""
    a = _artifact()
    for seed in a["per_seed"]:
        note = seed["component_meta"]["remarriage_origin_split"]
        assert "separately by origin" in note
        assert "(ysd_band, origin, sex)" in note
        assert "byte-identical to candidate 4" in note


# --------------------------------------------------------------------------
# Artifact presence, spec, lock (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v5"
    assert a["run"] == "gate2_hazard_v5"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 5"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_deltas_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert a["candidate4_registration"] == SPEC_URL_C4
    assert a["candidate1_registration"] == SPEC_URL_C1
    deltas = a["deltas_vs_candidate4"]
    assert len(deltas) == 3
    joined = " ".join(deltas).lower()
    assert "nchs" in joined
    assert "exp(beta_sex" in joined
    assert "1995" in joined
    assert "single-year-of-age" in joined
    assert "triangular" in joined
    assert "origin" in joined
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate4_registration"] == SPEC_URL_C4


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.35-0.45"
    assert fc["registration"] == SPEC_URL
    assert MODAL_CELL in fc["modal_failure"]
    assert len(fc["deltas_vs_candidate4"]) == 3
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
    }
    assert "20/22/25/30/40" in model["components"]["first_marriage"]
    wid = model["components"]["widowhood"].lower()
    assert "delta 1" in wid and "nchs" in wid and "1995" in wid
    fertm = model["components"]["fertility"].lower()
    assert "delta 2" in fertm and "single-year" in fertm
    rem = model["components"]["remarriage"].lower()
    assert "delta 3" in rem and "origin" in rem
    assert len(model["deltas_vs_candidate4"]) == 3


def test_precheck_reproduced_exactly():
    """The scoring path reproduced the committed floor bit-for-bit."""
    pc = _artifact()["precheck"]
    assert pc["all_reproduced_exactly"] is True
    assert pc["reference_moments_exact"] is True
    assert pc["rate_a_exact"] is True
    assert pc["holdout_sha256_all_match"] is True
    assert pc["reference_moments_max_abs_deviation"] == 0.0
    assert pc["rate_a_max_abs_deviation"] == 0.0
    for row in pc["per_seed"]:
        assert row["holdout_sha256_match"] is True


# --------------------------------------------------------------------------
# Tolerances match the locked gates.yaml (always runnable)
# --------------------------------------------------------------------------
def test_stored_tolerances_match_locked_gates_yaml():
    a = _artifact()
    locked = _gate2_tolerances()
    assert len(locked) == N_GATED
    floor = json.loads(FLOOR.read_text())
    assert set(locked) == set(floor["gate_partition"]["gate_eligible"])
    for seed in a["per_seed"]:
        stored = {k: v["tolerance"] for k, v in seed["gated_cells"].items()}
        assert set(stored) == set(locked)
        for cell, value in stored.items():
            assert value == pytest.approx(locked[cell], abs=0)


def test_report_only_cells_match_gates_yaml_and_never_gate():
    a = _artifact()
    locked_report = set(_gate2_thresholds()["report_only"])
    locked_tol = set(_gate2_tolerances())
    assert len(locked_report) == N_REPORT_ONLY
    assert locked_report.isdisjoint(locked_tol)
    for seed in a["per_seed"]:
        report = seed["report_only_cells"]
        assert set(report) == locked_report
        for cell, rec in report.items():
            assert rec["gated"] is False
            assert "tolerance" not in rec
            assert "pass" not in rec
            assert cell not in seed["gated_cells"]


# --------------------------------------------------------------------------
# Every stored pass recomputes from its score + tolerance (always runnable)
# --------------------------------------------------------------------------
def test_every_gated_pass_recomputes_from_score():
    a = _artifact()
    for seed in a["per_seed"]:
        n_pass = 0
        for cell, rec in seed["gated_cells"].items():
            recomputed = rec["score"] <= rec["tolerance"]
            assert recomputed == rec["pass"], (seed["seed"], cell)
            if rec["r_candidate"] > 0 and rec["rate_a"] > 0:
                expected = abs(math.log(rec["r_candidate"] / rec["rate_a"]))
                assert rec["score"] == pytest.approx(expected, abs=1e-12)
            n_pass += rec["pass"]
        assert seed["n_gated"] == N_GATED
        assert seed["n_gated_pass"] == n_pass
        assert seed["n_gated_fail"] == N_GATED - n_pass


def test_seed_pass_recomputes_from_all_gated_cells():
    a = _artifact()
    for seed in a["per_seed"]:
        all_pass = all(rec["pass"] for rec in seed["gated_cells"].values())
        assert seed["seed_pass"] == all_pass
        assert seed["seed_pass"] == (seed["n_gated_pass"] == N_GATED)


def test_verdict_recomputes_from_seed_conjunction():
    a = _artifact()
    v = a["verdict"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    n_pass = 0
    for row in a["seed_conjunction"]:
        s = by_seed[row["seed"]]
        assert row["seed_pass"] == s["seed_pass"]
        assert row["n_gated_pass"] == s["n_gated_pass"]
        n_pass += row["seed_pass"]
    assert v["n_seeds_pass"] == n_pass
    assert v["n_gate_seeds"] == len(GATE_SEEDS)
    assert v["gate_2_pass"] == (n_pass >= 4)
    assert v["seed_pass"] == {
        str(s): by_seed[s]["seed_pass"] for s in GATE_SEEDS
    }


def test_verdict_per_block_counts_consistent():
    a = _artifact()
    v = a["verdict"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    all_block_cells: set[str] = set()
    for _fam, blk in v["per_block"].items():
        all_block_cells |= set(blk["cells"])
        for seed_key, d in blk["per_seed_pass"].items():
            seed = by_seed[int(seed_key)]
            recomputed = sum(
                seed["gated_cells"][c]["pass"] for c in blk["cells"]
            )
            assert d["n_pass"] == recomputed
            assert d["n_cells"] == len(blk["cells"])
    assert all_block_cells == set(_gate2_tolerances())


def test_all_failing_gated_cells_are_real_failures():
    a = _artifact()
    v = a["verdict"]
    by_seed = {s["seed"]: s for s in a["per_seed"]}
    listed = {(f["cell"], f["seed"]) for f in v["all_failing_gated_cells"]}
    actual = {
        (cell, s["seed"])
        for s in a["per_seed"]
        for cell, rec in s["gated_cells"].items()
        if not rec["pass"]
    }
    assert listed == actual
    for f in v["all_failing_gated_cells"]:
        rec = by_seed[f["seed"]]["gated_cells"][f["cell"]]
        assert f["score"] == pytest.approx(rec["score"], abs=0)
        assert rec["score"] > rec["tolerance"]


# --------------------------------------------------------------------------
# Modal, targeted cells, decider, candidate-4 movement (always runnable)
# --------------------------------------------------------------------------
def test_modal_and_targeted_tracks_recompute():
    a = _artifact()
    v = a["verdict"]
    modal = a["modal_failure_materialized"]
    fails: dict[str, list[int]] = {}
    for f in v["all_failing_gated_cells"]:
        fails.setdefault(f["cell"], []).append(f["seed"])

    def check_track(cell: str, track: dict) -> None:
        for s in a["per_seed"]:
            assert track["per_seed_score"][str(s["seed"])] == pytest.approx(
                s["gated_cells"][cell]["score"], abs=0
            )
            assert (
                track["per_seed_pass"][str(s["seed"])]
                == s["gated_cells"][cell]["pass"]
            )
        assert track["failed_seeds"] == sorted(fails.get(cell, []))

    assert modal["modal_cell"] == MODAL_CELL
    check_track(MODAL_CELL, modal["modal_track"])
    assert set(modal["targeted_cells"]) == set(TARGETED_CELLS)
    for cell, track in modal["targeted_cells_track"].items():
        check_track(cell, track)


def test_decider_analysis_recomputes():
    """The counterfactual decider analysis recomputes from the gated cells."""
    a = _artifact()
    v = a["verdict"]
    dec = a["modal_failure_materialized"]["decider_analysis"]

    def seeds_pass_if_forgiven(forgiven: set[str]) -> int:
        return sum(
            all(
                rec["pass"]
                for cell, rec in s["gated_cells"].items()
                if cell not in forgiven
            )
            for s in a["per_seed"]
        )

    assert dec["n_seeds_pass_actual"] == v["n_seeds_pass"]
    assert dec["n_seeds_pass_if_modal_forgiven"] == seeds_pass_if_forgiven(
        {MODAL_CELL}
    )
    assert dec["n_seeds_pass_if_targeted_forgiven"] == seeds_pass_if_forgiven(
        set(TARGETED_CELLS)
    )
    # "modal_decided" iff the gate failed and forgiving the modal alone flips
    # it to >= 4 passing seeds.
    modal_decided = (not v["gate_2_pass"]) and (
        dec["n_seeds_pass_if_modal_forgiven"] >= 4
    )
    assert dec["modal_decided"] == modal_decided


def test_candidate4_comparison_movement_recomputes():
    """The vs-candidate-4 movement block recomputes from both artifacts."""
    a = _artifact()
    cmp = a["candidate4_comparison"]
    assert cmp["available"] is True
    a4 = _artifact_c4()
    by4 = {s["seed"]: s for s in a4["per_seed"]}
    by5 = {s["seed"]: s for s in a["per_seed"]}
    tracked = {MODAL_CELL, *TARGETED_CELLS}
    assert set(cmp["cells"]) == tracked
    for cell, d in cmp["cells"].items():
        c4_pass = sum(by4[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c5_pass = sum(by5[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate4_n_seeds_pass"] == c4_pass
        assert d["candidate5_n_seeds_pass"] == c5_pass
        for s in GATE_SEEDS:
            assert d["candidate4_per_seed_score"][str(s)] == pytest.approx(
                by4[s]["gated_cells"][cell]["score"], abs=0
            )
            assert d["candidate5_per_seed_score"][str(s)] == pytest.approx(
                by5[s]["gated_cells"][cell]["score"], abs=0
            )


def test_revision_pins_record_runner_and_nchs_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v5"
    assert pins["sklearn_version"].startswith("1.9")
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    for n in (1, 2, 3, 4):
        assert pins[f"candidate{n}_runner"] == (
            f"scripts/run_gate2_candidate{n}.py"
        )
        assert len(pins[f"candidate{n}_runner_sha256"]) == 64
    assert pins["nchs_fetch_script"] == (
        "scripts/fetch_nchs_life_tables_historical.py"
    )
    assert len(pins["nchs_fetch_script_sha256"]) == 64
    refs = pins["nchs_life_table_references"]
    assert set(refs) == {"2000", "2010", "2023"}
    for meta in refs.values():
        assert len(meta["sha256"]) == 64
        assert meta["path"].startswith("data/external/nchs_life_tables_")


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate4_registration"] == SPEC_URL_C4
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


# --------------------------------------------------------------------------
# NCHS historical references: provenance + life-table sanity (always runnable)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("vintage", [2000, 2010])
def test_nchs_reference_schema_and_provenance(vintage):
    ref = json.loads(
        (NCHS_DIR / f"nchs_life_tables_{vintage}.json").read_text()
    )
    assert ref["schema_version"] == "nchs_life_tables.v1"
    assert ref["vintage_year"] == vintage
    assert set(ref["tables"]) == {"total", "male", "female"}
    report = ref["report"]
    assert report["title"] == f"United States Life Tables, {vintage}"
    assert "National Center for Health Statistics" in report["publisher"]
    assert report["report_pdf_url"].startswith("https://www.cdc.gov/")
    fetch = ref["fetch"]
    assert fetch["fetched_by"] == (
        "scripts/fetch_nchs_life_tables_historical.py"
    )
    assert fetch["fetched_utc"]
    assert set(fetch["source_files"]) == {"total", "male", "female"}
    for meta in fetch["source_files"].values():
        import re

        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"])
        assert meta["n_bytes"] > 0
    if vintage == 2010:
        assert "xlsx" in fetch["source_format"]
        for meta in fetch["source_files"].values():
            assert meta["url"].endswith(".xlsx")
    else:
        assert "pdf" in fetch["source_format"]
        for meta in fetch["source_files"].values():
            assert meta["url"].endswith(".pdf")
            assert isinstance(meta["pdf_table_number"], int)


@pytest.mark.parametrize("vintage", [2000, 2010])
@pytest.mark.parametrize("population", ["total", "male", "female"])
def test_nchs_reference_life_table_sanity(vintage, population):
    ref = json.loads(
        (NCHS_DIR / f"nchs_life_tables_{vintage}.json").read_text()
    )
    rows = ref["tables"][population]
    ages = [r["age"] for r in rows]
    assert ages == list(range(0, ref["terminal_age"] + 1))
    assert round(rows[0]["lx"]) == 100000
    assert round(rows[0]["ex"], 1) == NCHS_PUBLISHED_EX0[vintage][population]
    by_age = {r["age"]: r for r in rows}
    for age in range(40, ref["terminal_age"]):
        assert by_age[age]["qx"] <= by_age[age + 1]["qx"] + 1e-9, age
    assert round(by_age[ref["terminal_age"]]["qx"], 5) == 1.0


# --------------------------------------------------------------------------
# Live seed-0 reproduction + delta checks (needs the staged PSID files)
# --------------------------------------------------------------------------
@needs_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 end-to-end and match the committed seed-0 block."""
    runner = _import_runner()
    from populace_dynamics.data import marriage

    thresholds = runner.c1.load_gate2_thresholds()
    tol = runner.c1.gated_tolerances(thresholds)
    report_only = list(thresholds["report_only"])
    floor = json.loads(FLOOR.read_text())

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, fert, _ = runner.g2f.load_panels()
    order_map = runner.c1._order_map(mh)

    result = runner.score_seed(
        0,
        panel,
        fert,
        demo,
        dr,
        mh,
        bh,
        order_map,
        floor,
        tol,
        report_only,
        False,
    )
    committed = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    assert result["n_gated_pass"] == committed["n_gated_pass"]
    assert result["seed_pass"] == committed["seed_pass"]
    for cell, rec in committed["gated_cells"].items():
        got = result["gated_cells"][cell]
        assert got["r_candidate"] == pytest.approx(
            rec["r_candidate"], abs=1e-12
        ), cell
        assert got["score"] == pytest.approx(rec["score"], abs=1e-12), cell
        assert got["pass"] == rec["pass"], cell
    for cell, rec in committed["report_only_cells"].items():
        got = result["report_only_cells"][cell]
        assert got["r_candidate"] == pytest.approx(
            rec["r_candidate"], abs=1e-12
        ), cell
    meta = result["component_meta"]
    for sex in ("female", "male"):
        assert meta["mortality_beta_by_sex"][sex] == pytest.approx(
            committed["component_meta"]["mortality_beta_by_sex"][sex],
            abs=1e-12,
        )


@needs_psid
def test_seed0_deltas_against_candidate4_live():
    """Delta structure on seed 0's train complement, live.

    (i) The NCHS betas equal the external slope from the committed references;
    (ii) the mortality anchor and the non-delta components (remarriage,
    divorce, spousal-gap mean+distribution, first marriage) are byte-identical
    to candidate 4; (iii) the fertility table is single-year-of-age and
    differs from candidate 1's band fertility.
    """
    runner = _import_runner()
    c4 = _import_c4()
    from populace_dynamics.data import marriage
    from populace_dynamics.harness import panel as hpanel

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, _fert, _ = runner.g2f.load_panels()
    order_map = runner.c1._order_map(mh)

    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())
    comp5 = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    comp4 = c4.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    comp1 = runner.c1.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # (i) NCHS betas = external slope from committed references.
    beta_nchs, _ = runner.fit_mortality_trend_nchs()
    assert comp5.meta["mortality_beta_by_sex"] == beta_nchs
    assert comp5.meta["mortality_beta_by_sex_candidate4_trainfit"] == (
        comp4.meta["mortality_beta_by_sex"]
    )

    # (ii) Anchor + non-delta components byte-identical to candidate 4.
    assert comp5.mortality == comp4.mortality
    assert comp5.remarriage == comp4.remarriage  # delta 3 is byte-identical
    assert np.array_equal(comp5.divorce, comp4.divorce)
    assert comp5.gap_by_sex == comp4.gap_by_sex
    for sex in ("female", "male"):
        assert np.array_equal(
            comp5.gap_dist_by_sex[sex], comp4.gap_dist_by_sex[sex]
        )
    assert comp5.first_marriage.knots == (20.0, 22.0, 25.0, 30.0, 40.0)

    # (iii) Fertility is single-year and differs from candidate 1's bands.
    ages = {a for (a, _p, _d) in comp5.fertility}
    assert min(ages) == 15 and max(ages) == 49
    assert comp5.fertility != comp1.fertility


@needs_psid
def test_seed0_fertility_delta_does_not_perturb_marital_cells():
    """Delta 2 (fertility) is RNG-isolated from the marital process.

    Re-simulating seed 0's holdout with the fertility table zeroed (so every
    ``p_birth`` is 0 and no birth occurs) draws the SAME per-year uniform
    blocks in the same order and size -- only the fertility threshold changes
    -- so every non-fertility (marital / widowhood / remarriage) reference cell
    is byte-identical to the single-year run, while births vanish. The direct
    generative proof that changing the fertility model cannot move the marital
    cells.
    """
    import copy

    runner = _import_runner()
    from populace_dynamics.data import marriage, transitions
    from populace_dynamics.harness import panel as hpanel

    mh = marriage.marriage_history()
    bh = runner.g2f.births.birth_history()
    dr = runner.g2f.deaths.read_death_records()
    demo = runner.g2f.panels.demographic_panel()
    panel, _fert, _ = runner.g2f.load_panels()
    order_map = runner.c1._order_map(mh)

    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    comp5 = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    sim5, b5 = runner.simulate_holdout(
        panel, ids_a, comp5, runner.SIM_SEED_BASE
    )
    comp0 = copy.copy(comp5)
    comp0.fertility = {}  # zero fertility -> every p_birth is 0
    sim0, b0 = runner.simulate_holdout(
        panel, ids_a, comp0, runner.SIM_SEED_BASE
    )

    m5 = transitions.reference_moments(
        sim5, transitions.build_fertility_panel(sim5, b5), ids_a, weighted=True
    )
    m0 = transitions.reference_moments(
        sim0, transitions.build_fertility_panel(sim0, b0), ids_a, weighted=True
    )
    for cell in m5:
        if not _is_fertility(cell):
            assert m0[cell]["rate"] == pytest.approx(
                m5[cell]["rate"], abs=1e-12
            ), cell
    assert len(b0) == 0  # zeroed fertility -> no simulated births
    assert len(b5) > 0


# --------------------------------------------------------------------------
# Pinned outcome of the one-shot run (FAIL 0/5; the published finding)
# --------------------------------------------------------------------------
def test_verdict_is_fail_zero_of_five():
    """The pre-registered outcome: FAIL, 0/5 seeds pass (published)."""
    v = _artifact()["verdict"]
    assert v["gate_2_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert all(v["seed_pass"][str(s)] is False for s in GATE_SEEDS)


def test_seed0_pins():
    """Pin the committed seed-0 outcome: 45/46, the modal the sole failure."""
    a = _artifact()
    s0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    assert s0["n_gated_pass"] == 45
    assert s0["seed_pass"] is False
    fails = {c for c, r in s0["gated_cells"].items() if not r["pass"]}
    assert fails == {"mean_lifetime_marriages|male"}
    meta = s0["component_meta"]
    assert meta["mortality_trend_source"] == (
        "external NCHS US life tables (2000, 2010, 2023)"
    )
    assert meta["fertility_kernel"] == "triangular"


def test_registered_modal_materialized_but_not_decisive():
    """mean_lifetime_marriages|male failed 4 seeds but did NOT decide the gate.

    The registered modal materialized (seeds 0,1,3,4) yet forgiving it alone
    lifts only one seed to passing; even forgiving both the modal and every
    targeted cell leaves the gate below four passing seeds -- the failure is
    broader than the modal, the honest published finding.
    """
    a = _artifact()
    modal = a["modal_failure_materialized"]
    assert modal["modal_cell"] == MODAL_CELL
    assert modal["modal_failed"] is True
    assert modal["modal_failed_seeds"] == [0, 1, 3, 4]
    assert modal["modal_is_sole_failing_cell"] is False
    dec = modal["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == 0
    assert dec["n_seeds_pass_if_modal_forgiven"] == 1
    assert dec["n_seeds_pass_if_targeted_forgiven"] == 0
    assert dec["n_seeds_pass_if_both_forgiven"] == 2
    assert dec["modal_decided"] is False
    assert "broader" in dec["decider"]
    for s in a["per_seed"]:
        expect_fail = s["seed"] in (0, 1, 3, 4)
        assert s["gated_cells"][MODAL_CELL]["pass"] is (not expect_fail)


def test_targeted_cells_movement_pins():
    """How the four targeted cells moved vs candidate 4 (pinned).

    Delta 2 fixed asfr.20-24 (4/5 -> 5/5); completed_fertility.c1970s stayed
    4/5 (mean score improved, seed 2 still clips); delta 1's NCHS trend left
    share_widowed.75+|female at 1/5 (a level gap, not a trend one); and delta
    3's byte-identical origin-split remarriage left remarriage.after_divorce
    at 4/5 (the seed-1 clip persists).
    """
    a = _artifact()
    cells = a["candidate4_comparison"]["cells"]
    move = {
        c: (d["candidate4_n_seeds_pass"], d["candidate5_n_seeds_pass"])
        for c, d in cells.items()
    }
    assert move["asfr.20-24"] == (4, 5)
    assert move["completed_fertility.c1970s"] == (4, 4)
    assert move["share_widowed.75+|female"] == (1, 1)
    assert move["remarriage.after_divorce"] == (4, 4)
    assert move[MODAL_CELL] == (1, 1)
    # asfr.20-24 (the headline delta-2 target) now passes every seed.
    for s in a["per_seed"]:
        assert s["gated_cells"]["asfr.20-24"]["pass"] is True
