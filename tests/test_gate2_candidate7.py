"""Tests for the gate-2 candidate-7 (run 1) pre-registered run.

Candidate 7 is the seventh pre-registered gate-2 candidate: candidate 6's
five-component family-transition simulator with EXACTLY TWO named structural
deltas -- (1) the remarriage hazard is additionally conditioned on the order
of the marriage being entered (2nd vs 3rd+), with the same YSD band x origin
x sex construction and add-one smoothing; (2) the single-year-of-age
fertility rate is additionally conditioned on the woman's current simulated
marital status (married vs not), with the same triangular kernel within each
parity x cohort x marital stratum. Everything else -- the source-aligned
surviving-spouse marriage-history widowhood level, the committed NCHS betas,
the knot-at-22 first-marriage spline, divorce, the spousal-gap draw, the RNG
rule -- is byte-identical to candidate 6. Frozen spec: issue #42 comment
4912542742.

Three tiers, mirroring the candidate-6 suite:

* the always-runnable consistency suite (touches only the committed
  candidate-7 / candidate-6 artifacts and ``gates.yaml``): schema and spec
  URLs, the two recorded deltas, the bit-exact reproduction precheck
  attestation, every stored gated-cell pass recomputes from its score against
  its stored (locked) tolerance, the stored tolerances equal the locked
  gates.yaml, each seed's pass recomputes, the verdict and per-block counts
  recompute, the report-only cells never gate, the registered modal
  (``mean_lifetime_marriages|male``), the targeted cells, the seed-0
  regression analysis, the decider analysis and the candidate-6 movement all
  recompute, the first-marriage and ever-married cells are byte-identical to
  candidate 6 (the deltas move only fertility, remarriage and the states they
  feed) while the fertility cells MOVED, and the forecast / registration /
  revision pins are carried;
* structural delta checks: the remarriage table is the order-conditioned
  24-cell construction (3 YSD bands x 2 origins x 2 sexes x 2 order bits, not
  the 12-cell candidate-6 table) and the fertility table gains the married
  stratum;
* :func:`test_seed0_reproduces_committed_artifact` and the live delta checks
  (skipped when the PSID history files are absent) rerun seed 0 end-to-end and
  pin the committed seed-0 block to float precision, confirm the two delta'd
  component tables derive as specified, confirm the non-delta components are
  byte-identical to candidate 6, and confirm the first-marriage and
  ever-married reference cells are byte-identical to candidate 6's simulation
  while the fertility cells move.

The one-shot outcome (published REGARDLESS of verdict): FAIL 0/5. Both deltas
fired but regressed candidate 6's only passing seed -- seed 0 fell 46/46 ->
44/46 (the registered PRIMARY risk): marital-conditioned fertility
over-produces asfr.15-19 and the order split's lower higher-order remarriage
lowers the already-low lifetime marriage counts the wrong way, moving
share_divorced.45-54|female off. remarriage.after_divorce did improve (4/5 ->
5/5) and completed_fertility.c1970s stayed a single-seed clip, but no seed
clears all 46 gated cells.
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
ARTIFACT_C6 = ROOT / "runs" / "gate2_hazard_v6.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912542742"
)
SPEC_URL_C6 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912170754"
)
SPEC_URL_C5 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911788302"
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
    "mean_lifetime_marriages|male",
    "completed_fertility.c1970s",
    "remarriage.after_divorce",
)

# The pinned one-shot outcome (published REGARDLESS of verdict): FAIL 0/5.
EXPECTED_SEED_FAILS = {
    0: {"asfr.15-19", "share_divorced.45-54|female"},
    1: {"mean_lifetime_marriages|female", "share_widowed.75+|female"},
    2: {"completed_fertility.c1970s"},
    3: {
        "asfr.15-19",
        "mean_lifetime_marriages|male",
        "widowhood.75+|female",
    },
    4: {"asfr.15-19", "mean_lifetime_marriages|male"},
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c6() -> dict:
    return json.loads(ARTIFACT_C6.read_text())


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
    import run_gate2_candidate7 as runner

    return runner


def _import_c6():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate6 as c6

    return c6


def _is_first_marriage_or_ever_married(cell: str) -> bool:
    return cell.startswith("first_marriage.") or cell.startswith(
        "ever_married_by_"
    )


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses candidate 6's dials.

    The frozen dials are IMPORTED from candidate 1, and the widowhood level,
    the committed NCHS betas and the first-marriage machinery from candidate 6
    -- the provenance of "candidate 6 verbatim except the two deltas". The
    order-conditioned remarriage and marital-conditioned fertility helpers are
    candidate 7's OWN (the deltas).
    """
    runner = _import_runner()
    c6 = _import_c6()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.SIM_SEED_BASE == 4200
    assert runner.TREND_ANCHOR_YEAR == 1995.0
    # First marriage + widowhood level are candidate 6's, reused unchanged.
    assert runner.fit_first_marriage is c6.fit_first_marriage
    assert runner.FirstMarriageModelC7 is c6.FirstMarriageModelC6
    assert runner._widow_probs is c6._widow_probs
    assert runner.WIDOW_BANDS == ((45, 54), (55, 64), (65, 74), (75, 120))
    assert runner.YSD_BANDS == ((0, 4), (5, 9), (10, 120))
    assert (runner.FERT_AGE_LO, runner.FERT_AGE_HI) == (15, 49)
    # DELTA helpers are candidate 7's OWN (order / marital conditioned).
    assert hasattr(runner, "_remarriage_probs_ordered")
    assert hasattr(runner, "_fertility_probs_single_marital")
    assert hasattr(runner, "fit_remarriage_ordered")
    assert hasattr(runner, "fit_fertility_single_marital")
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v7"
    assert "4912542742" in runner.SPEC_REGISTRATION
    assert "4912170754" in runner.CANDIDATE6_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert runner.REGISTERED_MODAL_CELL == MODAL_CELL
    assert set(runner.TARGETED_CELLS) == set(TARGETED_CELLS)


# --------------------------------------------------------------------------
# Artifact presence, spec, lock, the two deltas (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v7"
    assert a["run"] == "gate2_hazard_v7"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 7"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_deltas_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert a["candidate6_registration"] == SPEC_URL_C6
    assert a["candidate5_registration"] == SPEC_URL_C5
    assert a["candidate1_registration"] == SPEC_URL_C1
    d1 = a["delta1_vs_candidate6"].lower()
    assert "marriage" in d1 and "order" in d1
    assert "2nd" in d1 and "3rd" in d1
    d2 = a["delta2_vs_candidate6"].lower()
    assert "marital status" in d2
    assert "married" in d2
    assert "single-year" in d2 or "single year" in d2
    assert a["deltas_vs_candidate6"] == [
        a["delta1_vs_candidate6"],
        a["delta2_vs_candidate6"],
    ]
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate6_registration"] == SPEC_URL_C6


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.5-0.6"
    assert fc["registration"] == SPEC_URL
    assert "seed-0" in fc["modal_failure"] or "seed 0" in fc["modal_failure"]
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
    }
    # First marriage is byte-identical to candidate 6.
    fm = model["components"]["first_marriage"].lower()
    assert "byte-identical to candidate 6" in fm
    # Remarriage carries DELTA 1 (marriage order); fertility carries DELTA 2.
    rem = model["components"]["remarriage"].lower()
    assert "delta 1" in rem and "order" in rem
    fert = model["components"]["fertility"].lower()
    assert "delta 2" in fert and "marital status" in fert


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
# Structural: the two delta'd component tables (always runnable)
# --------------------------------------------------------------------------
def test_delta1_remarriage_order_structure():
    """Every seed records the 24-cell order-conditioned remarriage table."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        diag = meta["remarriage_order_diagnostics"]
        assert diag["n_cells"] == 24  # 3 ysd x 2 origin x 2 sex x 2 order
        assert diag["ysd_bands"] == [[0, 4], [5, 9], [10, 120]]
        assert set(diag["order_bits"]) == {"0", "1"}
        # Every cell key carries the 2nd / 3rd+ order label.
        for key in diag["cells"]:
            assert key.endswith("|2nd") or key.endswith("|3plus")
        # Candidate 6's 12-cell origin-split table is retained for reference.
        assert len(meta["remarriage_candidate6"]) == 12
        assert "marriage-order" in meta["remarriage_representation"].lower()


def test_delta2_fertility_marital_structure():
    """Every seed records the marital-conditioned single-year fertility."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        diag = meta["fertility_marital_diagnostics"]
        assert diag["married_bits"] == [0, 1]
        assert diag["age_range"] == [15, 49]
        assert diag["parity_bands"] == [0, 1, 2, 3]
        # The marital split at least doubles the cell count vs the single
        # marital-independent table (candidate 6's).
        assert diag["n_cells"] > meta["fertility_candidate6_n_cells"]
        # Every birth's marital state joined (a clean fit; the diagnostic).
        assert diag["n_births_unjoined_marital_state"] == 0
        assert meta["n_births_unjoined_marital_state"] == 0
        assert "marital status" in meta["fertility_representation"].lower()


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
# First-marriage + ever-married byte-identity; fertility moved (always run)
# --------------------------------------------------------------------------
def test_first_marriage_ever_married_byte_identical_to_candidate6():
    """Neither delta touches first marriage or whether a person ever married.

    First marriage is marital-state-independent under the shared RNG stream;
    remarriage adds marriages but no first marriages, and fertility touches no
    marital state -- so every ``first_marriage.*`` and ``ever_married_by_*``
    gated cell must carry candidate 6's exact ``r_candidate``.
    """
    a = _artifact()
    fi = a["first_marriage_identity_vs_candidate6"]
    assert fi["available"] is True
    assert fi["byte_identical"] is True
    assert fi["max_abs_r_candidate_deviation_vs_candidate6"] == 0.0

    a6 = _artifact_c6()
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    n_checked = 0
    for seed in a["per_seed"]:
        s6 = by6[seed["seed"]]
        for cell, rec in seed["gated_cells"].items():
            if _is_first_marriage_or_ever_married(cell):
                assert rec["r_candidate"] == pytest.approx(
                    s6["gated_cells"][cell]["r_candidate"], abs=0
                ), (seed["seed"], cell)
                n_checked += 1
    assert n_checked == fi["n_cells_checked"]


def test_fertility_moved_vs_candidate6():
    """Delta 2 is active: fertility cells are NOT byte-identical to c6.

    Unlike every prior candidate (whose fertility was marital-state-
    independent), candidate 7 conditions fertility on marital status, so the
    ``asfr.*`` and ``completed_fertility.*`` reference cells move.
    """
    a = _artifact()
    fm = a["fertility_movement_vs_candidate6"]
    assert fm["available"] is True
    assert fm["delta2_active"] is True
    assert fm["max_abs_r_candidate_deviation_vs_candidate6"] > 0.0
    assert fm["n_cells_moved"] > 0

    a6 = _artifact_c6()
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    moved = 0
    for seed in a["per_seed"]:
        s6 = by6[seed["seed"]]
        for cell, rec in seed["gated_cells"].items():
            if _is_fertility(cell):
                if (
                    abs(
                        rec["r_candidate"]
                        - s6["gated_cells"][cell]["r_candidate"]
                    )
                    > 0.0
                ):
                    moved += 1
    assert moved == fm["n_cells_moved"]


# --------------------------------------------------------------------------
# Modal, seed-0 regression, decider, candidate-6 movement (always runnable)
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


def test_seed0_regression_analysis_recomputes():
    """The seed-0 regression analysis recomputes vs the candidate-6 artifact."""
    a = _artifact()
    s0 = a["modal_failure_materialized"]["seed0_analysis"]
    seed0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    c6_seed0 = next(s for s in _artifact_c6()["per_seed"] if s["seed"] == 0)[
        "gated_cells"
    ]
    regressed = sorted(
        cell
        for cell, rec in seed0["gated_cells"].items()
        if c6_seed0[cell]["pass"] and not rec["pass"]
    )
    assert s0["seed0_held_all_gated"] == seed0["seed_pass"]
    assert s0["seed0_n_gated_pass"] == seed0["n_gated_pass"]
    assert s0["seed0_regressed_cells_vs_candidate6"] == regressed
    rem_fert = sorted(
        c for c in regressed if c.startswith("remarriage.") or _is_fertility(c)
    )
    assert s0["seed0_regressed_remarriage_or_fertility"] == rem_fert


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
    modal_decided = (not v["gate_2_pass"]) and (
        dec["n_seeds_pass_if_modal_forgiven"] >= 4
    )
    assert dec["modal_decided"] == modal_decided


def test_candidate6_comparison_movement_recomputes():
    """The vs-candidate-6 movement block recomputes from both artifacts."""
    a = _artifact()
    cmp = a["candidate6_comparison"]
    assert cmp["available"] is True
    a6 = _artifact_c6()
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    by7 = {s["seed"]: s for s in a["per_seed"]}
    for cell, d in cmp["cells"].items():
        c6_pass = sum(by6[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c7_pass = sum(by7[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate6_n_seeds_pass"] == c6_pass
        assert d["candidate7_n_seeds_pass"] == c7_pass
        for s in GATE_SEEDS:
            assert d["candidate6_per_seed_score"][str(s)] == pytest.approx(
                by6[s]["gated_cells"][cell]["score"], abs=0
            )
            assert d["candidate7_per_seed_score"][str(s)] == pytest.approx(
                by7[s]["gated_cells"][cell]["score"], abs=0
            )


def test_seed0_full_movement_recomputes():
    """The seed-0 full 46-cell movement block recomputes from both artifacts."""
    a = _artifact()
    m = a["seed0_full_movement"]
    assert m["available"] is True
    seed0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    c6_seed0 = next(s for s in _artifact_c6()["per_seed"] if s["seed"] == 0)[
        "gated_cells"
    ]
    n_regressed = sum(
        1
        for cell, rec in seed0["gated_cells"].items()
        if c6_seed0[cell]["pass"] and not rec["pass"]
    )
    n_improved = sum(
        1
        for cell, rec in seed0["gated_cells"].items()
        if (not c6_seed0[cell]["pass"]) and rec["pass"]
    )
    assert m["n_cells"] == N_GATED
    assert m["n_regressed"] == n_regressed
    assert m["n_improved"] == n_improved
    assert m["seed0_held_all_gated"] == seed0["seed_pass"]


def test_revision_pins_record_runner_and_candidate6_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v7"
    assert pins["sklearn_version"].startswith("1.9")
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    for n in (1, 2, 3, 4, 5, 6):
        assert pins[f"candidate{n}_runner"] == (
            f"scripts/run_gate2_candidate{n}.py"
        )
        assert len(pins[f"candidate{n}_runner_sha256"]) == 64
    assert pins["candidate6_artifact"] == "runs/gate2_hazard_v6.json"
    assert len(pins["candidate6_artifact_sha256"]) == 64


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate6_registration"] == SPEC_URL_C6
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


# --------------------------------------------------------------------------
# Pinned outcome of the one-shot run (FAIL 0/5; the published finding)
# --------------------------------------------------------------------------
def test_verdict_is_fail_zero_of_five():
    """The pre-registered outcome: FAIL, 0/5 seeds pass (published)."""
    v = _artifact()["verdict"]
    assert v["gate_2_pass"] is False
    assert v["n_seeds_pass"] == 0
    assert v["seed_pass"] == {
        "0": False,
        "1": False,
        "2": False,
        "3": False,
        "4": False,
    }


def test_seed_fail_sets_pinned():
    """Each seed's failing gated-cell set is pinned (the published finding).

    No seed clears all 46 gated cells. Seed 0 -- candidate 6's only passing
    seed -- regressed on the marital-conditioned teen fertility (asfr.15-19)
    and the divorced-stock cell (share_divorced.45-54|female); seed 2 keeps
    the single completed_fertility.c1970s clip; seeds 3 and 4 add the teen
    fertility miss to the mean_lifetime_marriages|male boundary.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        fails = {c for c, r in seed["gated_cells"].items() if not r["pass"]}
        assert fails == EXPECTED_SEED_FAILS[seed["seed"]], seed["seed"]
    # Every seed lands 43-45 of 46 (close, but none clears).
    for s in a["per_seed"]:
        assert 43 <= s["n_gated_pass"] <= 45
        assert s["seed_pass"] is False


def test_seed0_regressed_the_primary_registered_risk():
    """Seed 0 fell 46/46 -> 44/46 -- the registered PRIMARY failure mode.

    The registration named a seed-0 regression on a previously-passing
    remarriage or fertility cell as the primary risk (change-what-works). It
    materialized: seed 0 regressed on asfr.15-19 (fertility, delta 2) and
    share_divorced.45-54|female (delta 1's marital-exposure shift), with no
    offsetting improvement.
    """
    a = _artifact()
    m = a["seed0_full_movement"]
    assert m["seed0_held_all_gated"] is False
    assert m["n_regressed"] == 2
    assert m["n_improved"] == 0
    regressed = {
        c
        for c, d in m["cells"].items()
        if d["candidate6_pass"] and not d["candidate7_pass"]
    }
    assert regressed == {"asfr.15-19", "share_divorced.45-54|female"}
    s0 = a["modal_failure_materialized"]["seed0_analysis"]
    assert s0["seed0_regressed_remarriage_or_fertility"] == ["asfr.15-19"]


def test_key_cells_movement_pins():
    """How the two deltas' key cells moved vs candidate 6 (pinned).

    Delta 1 fixed the after-divorce remarriage flow (remarriage.after_divorce
    4/5 -> 5/5) but lowered the already-low lifetime marriage counts the wrong
    way, so mean_lifetime_marriages|male held at 3/5 (its boundary seeds 3/4
    unfixed) and |female held at 4/5; delta 2 did NOT fix the c1970s
    completed-fertility clip (4/5 -> 4/5, mean score WORSE, seed 2 still
    clips).
    """
    a = _artifact()
    cells = a["candidate6_comparison"]["cells"]
    move = {
        c: (d["candidate6_n_seeds_pass"], d["candidate7_n_seeds_pass"])
        for c, d in cells.items()
    }
    assert move["remarriage.after_divorce"] == (4, 5)
    assert move["mean_lifetime_marriages|male"] == (3, 3)
    assert move["mean_lifetime_marriages|female"] == (4, 4)
    assert move["completed_fertility.c1970s"] == (4, 4)
    # The c1970s clip got WORSE under delta 2, not better.
    clip = cells["completed_fertility.c1970s"]
    assert clip["candidate7_mean_score"] > clip["candidate6_mean_score"]
    # The registered modal materialized on seeds 3 and 4 (secondary).
    modal = a["modal_failure_materialized"]
    assert modal["modal_failed"] is True
    assert modal["modal_failed_seeds"] == [3, 4]


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


@needs_psid
def test_deltas_derive_from_train_and_nondelta_byte_identical():
    """The two deltas derive as specified; non-delta components match c6.

    On seed 0's train complement: the remarriage table is the 24-cell
    order-conditioned build (3 YSD x 2 origin x 2 sex x 2 order); the
    fertility table gains the married stratum; and every non-delta component
    (first marriage, divorce, the surviving-spouse widowhood level, the
    committed betas, the spousal gap) is byte-identical to candidate 6.
    """
    runner = _import_runner()
    c6 = _import_c6()
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
    c7c = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    c6c = c6.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # (i) DELTA 1: remarriage is the 24-cell order-conditioned table.
    assert len(c7c.remarriage) == 24
    assert all(len(k) == 4 for k in c7c.remarriage)
    assert {k[3] for k in c7c.remarriage} == {0, 1}
    assert len(c6c.remarriage) == 12
    # Higher-order remarriage rates are lower in the data (the mechanism).
    lower = sum(
        c7c.remarriage[(b, o, s, 1)] < c7c.remarriage[(b, o, s, 0)]
        for b in range(len(runner.YSD_BANDS))
        for o in ("divorced", "widowed")
        for s in ("female", "male")
        if (b, o, s, 0) in c7c.remarriage
    )
    assert lower >= 6  # a clear majority of the 12 (band,origin,sex) cells

    # (ii) DELTA 2: fertility gains the married stratum.
    assert all(len(k) == 4 for k in c7c.fertility)
    assert {k[3] for k in c7c.fertility} == {0, 1}
    assert len(c7c.fertility) > len(c6c.fertility)

    # (iii) Every non-delta component is byte-identical to candidate 6.
    assert np.array_equal(c7c.divorce, c6c.divorce)
    assert c7c.mortality == c6c.mortality
    assert c7c.gap_by_sex == c6c.gap_by_sex
    for sex in ("female", "male"):
        assert np.array_equal(
            c7c.gap_dist_by_sex[sex], c6c.gap_dist_by_sex[sex]
        )
    assert (
        c7c.meta["mortality_beta_by_sex"] == c6c.meta["mortality_beta_by_sex"]
    )
    assert c7c.first_marriage.knots == (20.0, 22.0, 25.0, 30.0, 40.0)
    # The add-one smoothing constant is candidate 1's exactly.
    assert c7c.meta["remarriage_mean_dissolved_weight_check"] == (
        c6c.meta["remarriage_mean_dissolved_weight"]
    )


@needs_psid
def test_seed0_first_marriage_ever_married_byte_identical_live():
    """Live proof: first-marriage + ever-married cells match c6 exactly.

    Simulating seed 0's holdout under candidate 7 and candidate 6 draws the
    SAME per-year uniform blocks (both deltas move only thresholds), so every
    ``first_marriage.*`` and ``ever_married_by_*`` reference cell is byte-
    identical -- while the fertility cells move (delta 2 active).
    """
    runner = _import_runner()
    c6 = _import_c6()
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

    c7c = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    c6c = c6.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    sim7, b7 = runner.simulate_holdout(panel, ids_a, c7c, runner.SIM_SEED_BASE)
    sim6, b6 = c6.simulate_holdout(panel, ids_a, c6c, c6.SIM_SEED_BASE)

    m7 = transitions.reference_moments(
        sim7, transitions.build_fertility_panel(sim7, b7), ids_a, weighted=True
    )
    m6 = transitions.reference_moments(
        sim6, transitions.build_fertility_panel(sim6, b6), ids_a, weighted=True
    )
    n_identical = 0
    n_fert_moved = 0
    for cell in m7:
        if _is_first_marriage_or_ever_married(cell):
            assert m7[cell]["rate"] == pytest.approx(
                m6[cell]["rate"], abs=1e-12
            ), cell
            n_identical += 1
        elif _is_fertility(cell):
            if abs(m7[cell]["rate"] - m6[cell]["rate"]) > 1e-12:
                n_fert_moved += 1
    assert n_identical > 0
    assert n_fert_moved > 0  # delta 2 is active
