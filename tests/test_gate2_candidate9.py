"""Tests for the gate-2 candidate-9 (run 1) pre-registered run.

Candidate 9 is the ninth pre-registered gate-2 candidate: candidate 8's five-
component family-transition simulator with EXACTLY TWO named deltas registered
from the gate-2 forensics (#94):

* DELTA 1 -- observed undatable-marriage lifetime-count initial state. Each
  holdout person's simulated lifetime-marriage count initializes at their
  OBSERVED residual ``R = n_marriages - (# datable in-exposure first_marriage
  and remarriage transition events)`` -- the forensics Q1 residual (undatable
  start/dissolution, out-of-window/underage, separation-origin and MH-count-
  excess marriages the hazards cannot generate) at per-person resolution --
  and accumulates the simulated datable transitions as candidate 8 does. It is
  an OBSERVED initial state per protocol (like the entry marital state); it
  reconciles to remainder 0.0 (``R + datable == n_marriages`` per person, and
  aggregated over the ever-married denominator equals the forensics reference
  residual and its five buckets); it perturbs NO RNG draw (a post-assembly
  count add) and leaves the hazards untouched.
* DELTA 2 -- unsmoothed low-parity peak fertility. Parities 0->1 and 1->2
  (parity bands 0, 1) use the EXACT empirical single-year rate (no kernel) at
  ages 22-38 where the single-year cell exposure is >= 200 weighted person-
  years; the triangular kernel is retained everywhere else and higher parities
  are unchanged. It moves only the fertility THRESHOLD (same per-year uniform
  draw order and size), so the marriage draw stream is byte-identical to
  candidate 8.

Because delta 1 is an RNG-neutral count add on ``n_marriages`` alone and delta
2 moves only the fertility threshold, EVERY gated cell that is neither a
fertility cell (``asfr.*``, ``completed_fertility.*``) nor a lifetime-marriage
count (``mean_lifetime_marriages|*``) is BYTE-IDENTICAL to candidate 8. Frozen
spec: issue #42 comment 4914111252.

Three tiers, mirroring the candidate-8 suite:

* the always-runnable consistency suite (touches only the committed
  candidate-9 / candidate-8 artifacts and ``gates.yaml``): schema and spec
  URLs, the two recorded deltas, the bit-exact reproduction precheck, the
  delta-1 reconciliation record, every stored gated-cell pass recomputes, the
  tolerances equal the locked gates.yaml, each seed's pass and the verdict and
  per-block counts recompute, the report-only cells never gate, the registered
  modal (``share_widowed.75+|female``) and decider recompute, the candidate-8
  movement recomputes, and the non-fertility non-count cells are byte-identical
  to candidate 8;
* the structural delta checks: the fertility table overrides exactly the low-
  parity peak cells (parities 0/1, ages 22-38, exposure >= 200) with num/den
  and is otherwise candidate 5's kernel, and the observed residual is the
  exact complement of the datable event count;
* :func:`test_seed0_reproduces_committed_artifact` and the live delta checks
  (skipped when the PSID history files are absent) rerun seed 0 end-to-end and
  pin the committed seed-0 block to float precision, confirm the fertility
  delta derives as specified and the non-delta components are byte-identical to
  candidate 8, and confirm the count add moves ONLY the lifetime-marriage cells
  while every non-fertility non-count cell is byte-identical to candidate 8's
  simulation.

The one-shot outcome (published REGARDLESS of verdict): FAIL 0/5. Delta 1
fixed the male marriage-count boundary on seeds 3-4 (3/5 -> 4/5) but OVERSHOT
on seed 2 (the sim in-exposure over-production the forensics flagged), pushing
mean_lifetime_marriages|female to fail on seeds 2 and 4 (4/5 -> 3/5); delta 2
did not clear completed_fertility.c1970s (seed 2, worse) and broke asfr.20-24
on seed 0 (5/5 -> 4/5); and the byte-identical-to-candidate-8 marital cells
(share_widowed.75+|female s1, share_divorced.45-54|female s0, widowhood.75+|
female s3) held exactly where candidate 8 failed them. No seed clears all 46.
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
ARTIFACT_C8 = ROOT / "runs" / "gate2_hazard_v8.json"
FORENSICS = ROOT / "runs" / "gate2_forensics_v1.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4914111252"
)
SPEC_URL_C8 = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912995860"
)
GATE_SEEDS = [0, 1, 2, 3, 4]
N_GATED = 46
N_REPORT_ONLY = 16

MODAL_CELL = "share_widowed.75+|female"
TARGETED_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
    "completed_fertility.c1970s",
)

# The pinned one-shot outcome (published REGARDLESS of verdict): FAIL 0/5.
EXPECTED_SEED_FAILS = {
    0: {"asfr.20-24", "share_divorced.45-54|female"},
    1: {"share_widowed.75+|female"},
    2: {
        "completed_fertility.c1970s",
        "mean_lifetime_marriages|female",
        "mean_lifetime_marriages|male",
    },
    3: {"widowhood.75+|female"},
    4: {"mean_lifetime_marriages|female"},
}

# Cell movement vs candidate 8 (n_seeds_pass), pinned.
EXPECTED_MOVEMENT = {
    "mean_lifetime_marriages|male": (3, 4),
    "mean_lifetime_marriages|female": (4, 3),
    "completed_fertility.c1970s": (4, 4),
    "asfr.15-19": (5, 5),
    "asfr.20-24": (5, 4),
    "share_divorced.45-54|female": (4, 4),
    "share_widowed.75+|female": (4, 4),
    "widowhood.75+|female": (4, 4),
}

# Delta-2 fertility exact-rate footprint. The number of overridden low-parity
# peak cells depends on the train split (which single-year cells clear the 200-
# py exposure floor), so it is pinned per seed; seed 0's is the live-
# reproducible value the live tests fit on side B.
N_FERT_EXACT_OVERRIDE_PER_SEED = {0: 362, 1: 348, 2: 353, 3: 368, 4: 368}
N_FERT_EXACT_OVERRIDE = N_FERT_EXACT_OVERRIDE_PER_SEED[0]
N_FERT_EXACT_BY_PB = {"0": 178, "1": 184}

# Seed-0 lifetime-marriage r_candidate, pinned to float precision (live-
# reproducible): the observed residual added to candidate 8's simulated count.
SEED0_MLM_MALE = 1.4820847342635983
SEED0_MLM_FEMALE = 1.4392059451165762


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c8() -> dict:
    return json.loads(ARTIFACT_C8.read_text())


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
    import run_gate2_candidate9 as runner

    return runner


def _import_c8():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate8 as c8

    return c8


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


def _is_count(cell: str) -> bool:
    return cell.startswith("mean_lifetime_marriages")


def _is_byte_identical_to_c8(cell: str) -> bool:
    return not (_is_fertility(cell) or _is_count(cell))


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses c8 machinery."""
    runner = _import_runner()
    c8 = _import_c8()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.SIM_SEED_BASE == 4200
    assert (runner.FERT_AGE_LO, runner.FERT_AGE_HI) == (15, 49)
    # Delta 2 dials.
    assert runner.FERT_EXACT_PARITY_BANDS == (0, 1)
    assert (runner.FERT_EXACT_AGE_LO, runner.FERT_EXACT_AGE_HI) == (22, 38)
    assert runner.FERT_EXACT_MIN_EXPOSURE == 200.0
    # Candidate 9's OWN additions: the residual + low-parity-exact fertility.
    assert hasattr(runner, "observed_residual_counts")
    assert hasattr(runner, "fit_fertility_low_parity_exact")
    assert hasattr(runner, "_delta1_reconciliation")
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v9"
    assert "4914111252" in runner.SPEC_REGISTRATION
    assert runner.CANDIDATE8_REGISTRATION == c8.SPEC_REGISTRATION
    assert runner.REGISTERED_MODAL_CELL == MODAL_CELL
    assert set(runner.TARGETED_CELLS) == set(TARGETED_CELLS)


# --------------------------------------------------------------------------
# Artifact presence, spec, lock, the two deltas (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v9"
    assert a["run"] == "gate2_hazard_v9"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 9"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_deltas_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert a["candidate8_registration"] == SPEC_URL_C8
    d = a["delta_vs_candidate8"].lower()
    # Delta 1 language.
    assert "initial state" in d or "initializes" in d
    assert "residual" in d and "datable" in d
    # Delta 2 language.
    assert "exact" in d and "kernel" in d and "22-38" in d
    assert ">= 200" in d or "200 weighted" in d
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate8_registration"] == SPEC_URL_C8


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.40-0.50"
    assert fc["registration"] == SPEC_URL
    assert "share_widowed.75+|female" in fc["modal_failure"].lower()
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
        "lifetime_marriage_count_initial_state",
    }
    # First marriage / divorce / widowhood / remarriage byte-identical to c8.
    for comp in ("first_marriage", "divorce", "widowhood"):
        assert "byte-identical to candidate 8" in (
            model["components"][comp].lower()
        )
    # Remarriage is candidate 8's own delta, byte-identical here.
    assert "byte-identical" in model["components"]["remarriage"].lower()
    # Fertility carries DELTA 2, the count init carries DELTA 1.
    assert "delta 2" in model["components"]["fertility"].lower()
    assert "delta 1" in (
        model["components"]["lifetime_marriage_count_initial_state"].lower()
    )
    res = model["registered_ambiguity_resolutions"]
    assert "observed_residual_definition" in res
    assert "low_parity_exact_fertility" in res
    assert "rng_neutrality" in res


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
# Delta 1: the reconciliation record (always runnable)
# --------------------------------------------------------------------------
def test_delta1_reconciliation_recorded():
    """R + datable == n_marriages (remainder 0.0) and R aggregates to the
    forensics reference residual and its five buckets."""
    rec = _artifact()["delta1_reconciliation"]
    assert rec["reconciled"] is True
    assert rec["per_person_identity_max_abs_residual"] == 0.0
    assert rec["aggregate_reconciliation_max_abs_remainder"] <= 1e-9
    assert rec["residual_nonnegative"] is True
    assert rec["residual_min_over_defined_persons"] == 0.0
    for row in rec["per_seed"]:
        assert row["seed"] in GATE_SEEDS
        for sex in ("male", "female"):
            b = row["by_sex"][sex]
            # R aggregate == forensics reference residual == bucket sum.
            assert b["residual_agg_per_person"] == pytest.approx(
                b["forensics_reference_residual"], abs=1e-9
            )
            assert b["forensics_bucket_sum"] == pytest.approx(
                b["forensics_reference_residual"], abs=1e-9
            )
            assert abs(b["remainder_agg_minus_reference"]) <= 1e-9
            assert abs(b["remainder_reference_minus_buckets"]) <= 1e-9
            # The five forensics buckets are all present.
            assert set(b["buckets"]) == {
                "undatable_na_start",
                "out_of_window_or_underage",
                "remarriage_prior_dissolution_year_undatable",
                "remarriage_separation_other_unknown_origin",
                "n_marriages_minus_datable_episodes",
            }


def test_delta1_reconciliation_matches_committed_forensics():
    """The reconciled residual matches the committed forensics per-seed
    reference-residual buckets bit-for-bit."""
    rec = _artifact()["delta1_reconciliation"]
    forensics = json.loads(FORENSICS.read_text())
    for_by_seed = {s["seed"]: s for s in forensics["per_seed"]}
    for row in rec["per_seed"]:
        f = for_by_seed[row["seed"]]
        for sex in ("male", "female"):
            got = row["by_sex"][sex]["buckets"]
            want = f["ref_residual"][sex]
            for key in want:
                assert got[key] == pytest.approx(want[key], abs=1e-9), (
                    row["seed"],
                    sex,
                    key,
                )


# --------------------------------------------------------------------------
# Delta 2: the fertility exact-rate footprint (always runnable)
# --------------------------------------------------------------------------
def test_delta2_fertility_footprint_recorded():
    """Every seed records the low-parity exact-rate footprint.

    The count of overridden cells depends on the train split (which single-
    year cells clear the 200-py exposure floor), so it is pinned per seed; the
    parity bands, age window and exposure floor are fixed, and every override
    is a parity-0/1 cell inside ages 22-38.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        m = seed["component_meta"]["fertility_low_parity_exact"]
        assert m["exact_parity_bands"] == [0, 1]
        assert m["exact_age_range"] == [22, 38]
        assert m["exact_min_exposure_weighted_person_years"] == 200.0
        assert (
            m["n_cells_exact_override"]
            == N_FERT_EXACT_OVERRIDE_PER_SEED[seed["seed"]]
        )
        # Split-dependent but bounded, and split into the two low parities.
        by_pb = m["n_exact_override_by_parity_band"]
        assert set(by_pb) == {"0", "1"}
        assert by_pb["0"] + by_pb["1"] == m["n_cells_exact_override"]
        assert 340 <= m["n_cells_exact_override"] <= 375
        # Every overridden age is inside the window.
        assert all(22 <= age <= 38 for age in m["override_ages"])
        # Representation names the delta.
        assert (
            "exact"
            in seed["component_meta"]["fertility_representation"].lower()
        )


# --------------------------------------------------------------------------
# Byte-identity to candidate 8 (always runnable)
# --------------------------------------------------------------------------
def test_non_fertility_non_count_cells_byte_identical_to_candidate8():
    """The core structural property: every gated cell that is neither a
    fertility cell nor a lifetime-marriage count is byte-identical to
    candidate 8 (delta 1 is RNG-neutral, delta 2 moves only fertility)."""
    a = _artifact()
    idb = a["identity_vs_candidate8"]
    assert idb["available"] is True
    assert idb["byte_identical"] is True
    assert idb["max_abs_r_candidate_deviation_vs_candidate8"] == 0.0
    assert idb["delta_active"] is True
    assert idb["n_lifetime_marriage_cell_movements"] > 0
    assert idb["n_fertility_cell_movements"] > 0

    a8 = _artifact_c8()
    by8 = {s["seed"]: s for s in a8["per_seed"]}
    n_checked = 0
    for seed in a["per_seed"]:
        s8 = by8[seed["seed"]]
        for cell, rec in seed["gated_cells"].items():
            if _is_byte_identical_to_c8(cell):
                assert rec["r_candidate"] == pytest.approx(
                    s8["gated_cells"][cell]["r_candidate"], abs=0
                ), (seed["seed"], cell)
                n_checked += 1
    assert n_checked == idb["n_cells_checked"]
    # 33 byte-identical cells (46 - 11 fertility - 2 count) x 5 seeds.
    assert n_checked == 33 * 5


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
# Every stored pass recomputes (always runnable)
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
# Modal, decider, candidate-8 movement (always runnable)
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
    assert modal["modal_failed_seeds"] == sorted(fails.get(MODAL_CELL, []))
    check_track(MODAL_CELL, modal["modal_track"])
    assert set(modal["targeted_cells"]) == set(TARGETED_CELLS)
    for cell, track in modal["targeted_cells_track"].items():
        check_track(cell, track)


def test_decider_analysis_recomputes():
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
    assert dec[
        "n_seeds_pass_if_modal_and_targeted_forgiven"
    ] == seeds_pass_if_forgiven(set(TARGETED_CELLS) | {MODAL_CELL})


def test_candidate8_comparison_movement_recomputes():
    """The vs-candidate-8 movement block recomputes from both artifacts."""
    a = _artifact()
    cmp = a["candidate8_comparison"]
    assert cmp["available"] is True
    a8 = _artifact_c8()
    by8 = {s["seed"]: s for s in a8["per_seed"]}
    by9 = {s["seed"]: s for s in a["per_seed"]}
    for cell, d in cmp["cells"].items():
        c8_pass = sum(by8[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c9_pass = sum(by9[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate8_n_seeds_pass"] == c8_pass
        assert d["candidate9_n_seeds_pass"] == c9_pass
        for s in GATE_SEEDS:
            assert d["candidate8_per_seed_score"][str(s)] == pytest.approx(
                by8[s]["gated_cells"][cell]["score"], abs=0
            )
            assert d["candidate9_per_seed_score"][str(s)] == pytest.approx(
                by9[s]["gated_cells"][cell]["score"], abs=0
            )


def test_revision_pins_record_runner_and_candidate_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v9"
    assert pins["sklearn_version"].startswith("1.9")
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        assert pins[f"candidate{n}_runner"] == (
            f"scripts/run_gate2_candidate{n}.py"
        )
        assert len(pins[f"candidate{n}_runner_sha256"]) == 64
    assert pins["candidate8_artifact"] == "runs/gate2_hazard_v8.json"
    assert len(pins["candidate8_artifact_sha256"]) == 64
    assert pins["forensics_artifact"] == "runs/gate2_forensics_v1.json"
    assert len(pins["forensics_artifact_sha256"]) == 64


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate8_registration"] == SPEC_URL_C8
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

    No seed clears all 46. Seed 0 loses asfr.20-24 to delta 2 and keeps the
    byte-identical share_divorced.45-54|female; seed 1 keeps only the byte-
    identical share_widowed.75+|female (delta 1 cured its count cell); seed 2
    keeps completed_fertility.c1970s AND is pushed to fail BOTH count cells by
    delta 1's overshoot; seed 3 keeps the byte-identical widowhood.75+|female
    (delta 1 cured its male count cell); seed 4 is pushed to fail the female
    count cell by delta 1's overshoot.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        fails = {c for c, r in seed["gated_cells"].items() if not r["pass"]}
        assert fails == EXPECTED_SEED_FAILS[seed["seed"]], seed["seed"]
    for s in a["per_seed"]:
        assert 43 <= s["n_gated_pass"] <= 45
        assert s["seed_pass"] is False


def test_cell_movement_vs_candidate8_pinned():
    """How the delta's key cells moved vs candidate 8 (pinned).

    Delta 1 raised the male marriage-count boundary (3/5 -> 4/5, curing seeds
    3-4) but OVERSHOT on seed 2 and dragged the female count down (4/5 -> 3/5,
    failing seeds 2 and 4); delta 2 did not clear completed_fertility.c1970s
    (4/5 held, seed-2 score worse) and broke asfr.20-24 (5/5 -> 4/5); the
    byte-identical marital cells held exactly (share_widowed.75+|female,
    share_divorced.45-54|female, widowhood.75+|female all 4/5, unchanged).
    """
    a = _artifact()
    cells = a["candidate8_comparison"]["cells"]
    for cell, (c8_pass, c9_pass) in EXPECTED_MOVEMENT.items():
        assert cells[cell]["candidate8_n_seeds_pass"] == c8_pass, cell
        assert cells[cell]["candidate9_n_seeds_pass"] == c9_pass, cell
    # The byte-identical marital cells did not move at all (scores equal).
    for cell in (
        "share_widowed.75+|female",
        "share_divorced.45-54|female",
        "widowhood.75+|female",
    ):
        d = cells[cell]
        for s in GATE_SEEDS:
            assert d["candidate9_per_seed_score"][str(s)] == pytest.approx(
                d["candidate8_per_seed_score"][str(s)], abs=0
            )
    # The registered modal (share_widowed.75+|female) failed exactly seed 1.
    modal = a["modal_failure_materialized"]
    assert modal["modal_failed"] is True
    assert modal["modal_failed_seeds"] == [1]


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
    # The seed-0 lifetime-marriage counts reproduce to float precision.
    assert result["gated_cells"]["mean_lifetime_marriages|male"][
        "r_candidate"
    ] == pytest.approx(SEED0_MLM_MALE, abs=1e-12)
    assert result["gated_cells"]["mean_lifetime_marriages|female"][
        "r_candidate"
    ] == pytest.approx(SEED0_MLM_FEMALE, abs=1e-12)


@needs_psid
def test_delta_derives_and_reconciles_live():
    """Both deltas derive as specified on seed 0's train complement.

    DELTA 2: the fertility table overrides exactly the low-parity peak cells
    (parities 0/1, ages 22-38, exposure >= 200) with num/den and is otherwise
    byte-identical to candidate 5's kernel (candidate 8's fertility); every
    other component is byte-identical to candidate 8. DELTA 1: the observed
    residual reconciles (R + datable == n_marriages per person; R aggregated
    over the ever-married denominator equals the forensics reference residual).
    """
    runner = _import_runner()
    c8 = _import_c8()
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
    c9c = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    c8c = c8.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # DELTA 2: same key set; low-parity peak cells = num/den, rest = kernel.
    assert set(c9c.fertility) == set(c8c.fertility)
    attr_by = panel.attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")
    den, num = runner._fertility_single_year_counts(
        panel, bh, ids_b, birth_decade
    )
    n_override = 0
    for key, rate in c9c.fertility.items():
        age, pb, dec = key
        fire = (
            pb in runner.FERT_EXACT_PARITY_BANDS
            and runner.FERT_EXACT_AGE_LO <= age <= runner.FERT_EXACT_AGE_HI
            and float(den.get(key, 0.0)) >= runner.FERT_EXACT_MIN_EXPOSURE
        )
        if fire:
            assert rate == pytest.approx(
                float(num.get(key, 0.0)) / float(den[key]), abs=1e-15
            ), key
            n_override += 1
        else:
            # non-firing cells are candidate 8's kernel value exactly
            assert rate == c8c.fertility[key], key
    assert n_override == N_FERT_EXACT_OVERRIDE
    meta = c9c.meta["fertility_low_parity_exact"]
    assert meta["n_cells_exact_override"] == N_FERT_EXACT_OVERRIDE

    # Every other component byte-identical to candidate 8.
    assert np.array_equal(c9c.divorce, c8c.divorce)
    assert c9c.remarriage == c8c.remarriage
    assert c9c.mortality == c8c.mortality
    assert c9c.gap_by_sex == c8c.gap_by_sex
    assert (
        c9c.meta["mortality_beta_by_sex"] == c8c.meta["mortality_beta_by_sex"]
    )
    assert len(c9c.remarriage) == 24

    # DELTA 1: reconciliation via the runner's own hard-gate function.
    rec = runner._delta1_reconciliation(panel, mh, (0,))
    assert rec["reconciled"] is True
    assert rec["per_person_identity_max_abs_residual"] == 0.0
    assert rec["residual_nonnegative"] is True


@needs_psid
def test_count_add_moves_only_lifetime_marriages_live():
    """Live proof: delta 1 moves ONLY the lifetime-marriage cells vs c8.

    Simulating seed 0's holdout under candidate 9 and candidate 8 draws the
    SAME per-year uniform blocks (delta 2 moves only the fertility threshold),
    so every marital cell is byte-identical; the observed-residual count add
    moves ONLY mean_lifetime_marriages|{male,female}, and the fertility cells
    move through the exact-rate threshold.
    """
    runner = _import_runner()
    c8 = _import_c8()
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

    c9c = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    c8c = c8.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    sim9, b9 = runner.simulate_holdout(panel, ids_a, c9c, runner.SIM_SEED_BASE)
    sim8, b8 = c8.simulate_holdout(panel, ids_a, c8c, c8.SIM_SEED_BASE)

    m9 = transitions.reference_moments(
        sim9, transitions.build_fertility_panel(sim9, b9), ids_a, weighted=True
    )
    m8 = transitions.reference_moments(
        sim8, transitions.build_fertility_panel(sim8, b8), ids_a, weighted=True
    )
    n_marital_identical = 0
    n_count_moved = 0
    for cell in m8:
        if _is_byte_identical_to_c8(cell):
            assert m9[cell]["rate"] == pytest.approx(
                m8[cell]["rate"], abs=1e-12
            ), cell
            n_marital_identical += 1
        elif _is_count(cell):
            if abs(m9[cell]["rate"] - m8[cell]["rate"]) > 1e-9:
                n_count_moved += 1
    assert n_marital_identical > 0
    assert n_count_moved == 2  # both lifetime-marriage counts moved
    # The seed-0 male count is candidate 8's simulated count plus the residual.
    assert m9["mean_lifetime_marriages|male"]["rate"] == pytest.approx(
        SEED0_MLM_MALE, abs=1e-9
    )
