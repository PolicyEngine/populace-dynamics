"""Tests for the gate-2 candidate-6 (run 1) pre-registered run.

Candidate 6 is the sixth pre-registered gate-2 candidate: candidate 5's
five-component family-transition simulator with EXACTLY ONE named delta --
the spouse-death hazard LEVEL is source-aligned to the gate reference. The
level is estimated from the train half's marriage histories (``mh85_23``
how-ended = spouse death over married person-year exposure, by age band x
sex of the SURVIVING spouse -- ``transitions.hazard_cells`` widowhood cells
over ``WIDOWHOOD_AGE_BANDS``, the gate reference's own construction),
replacing candidate 1's ``ind2023er`` death-record central-rate table. The
NCHS period-trend multiplier ``exp(beta_sex * (year - 1995))`` with
candidate 5's COMMITTED beta values (read from ``runs/gate2_hazard_v5.json``,
NOT re-fit) applies unchanged, looked up by the married ego's own (age, sex).
Everything else is byte-identical to candidate 5. Frozen spec: issue #42
comment 4912170754.

Three tiers, mirroring the candidate-5 suite:

* the always-runnable consistency suite (touches only the committed
  candidate-6 / candidate-5 artifacts and ``gates.yaml``): schema and spec
  URLs, the recorded delta, the bit-exact reproduction precheck attestation,
  every stored gated-cell pass recomputes from its score against its stored
  (locked) tolerance, the stored tolerances equal the locked gates.yaml, each
  seed's pass recomputes, the verdict and per-block counts recompute, the
  report-only cells never gate, the registered modal
  (``mean_lifetime_marriages|male``), the targeted cells, the decider
  analysis and the candidate-5 movement all recompute, the first-marriage and
  fertility cells are byte-identical to candidate 5 (the delta only moves
  widowhood and the states it feeds), and the forecast / registration /
  revision pins are carried;
* structural delta checks: the mortality LEVEL is recorded as the
  surviving-spouse marriage-history widowhood construction over
  ``WIDOWHOOD_AGE_BANDS`` (8 cells, not the 14-cell death-record table); the
  applied beta is candidate 5's committed value (read, not re-fit); and the
  old-vs-new-by-band level comparison is carried;
* :func:`test_seed0_reproduces_committed_artifact` and the live delta checks
  (skipped when the PSID history files are absent) rerun seed 0 end-to-end and
  pin the committed seed-0 block to float precision, confirm the new level
  table derives EXACTLY from the gate reference's marriage-history widowhood
  endings, confirm the non-delta components are byte-identical to candidate 5,
  and confirm the first-marriage and fertility reference cells are
  byte-identical to candidate 5's simulation.
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
ARTIFACT_C5 = ROOT / "runs" / "gate2_hazard_v5.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
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
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)
BETA_V5 = {
    "female": -0.009234704865961198,
    "male": -0.010643975395626533,
}

# The pinned one-shot outcome (published REGARDLESS of verdict).
EXPECTED_SEED_FAILS = {
    0: set(),
    1: {
        "mean_lifetime_marriages|female",
        "remarriage.after_divorce",
        "share_widowed.75+|female",
    },
    2: {"completed_fertility.c1970s"},
    3: {"mean_lifetime_marriages|male"},
    4: {"mean_lifetime_marriages|male"},
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c5() -> dict:
    return json.loads(ARTIFACT_C5.read_text())


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
    import run_gate2_candidate6 as runner

    return runner


def _import_c5():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate5 as c5

    return c5


def _is_first_marriage_or_fertility(cell: str) -> bool:
    return (
        cell.startswith("first_marriage.")
        or cell.startswith("asfr.")
        or cell.startswith("completed_fertility.")
    )


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses candidate 5's dials.

    The frozen dials are IMPORTED from candidate 1, the trend anchor (1995)
    from candidate 4, and the single-year fertility + first-marriage machinery
    from candidate 5 -- the provenance of "candidate 5 verbatim except the one
    delta". The widow-prob helper is candidate 6's OWN (the delta), not
    candidate 4's/5's.
    """
    runner = _import_runner()
    c5 = _import_c5()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.SIM_SEED_BASE == 4200
    assert runner.TREND_ANCHOR_YEAR == 1995.0
    # First marriage + single-year fertility are candidate 5's, reused.
    assert runner.fit_first_marriage is c5.fit_first_marriage
    assert runner.FirstMarriageModelC6 is c5.FirstMarriageModelC5
    assert runner._fertility_probs_single is c5._fertility_probs_single
    assert (runner.FERT_AGE_LO, runner.FERT_AGE_HI) == (15, 49)
    # DELTA: surviving-spouse widowhood level bands (the gate reference's).
    assert runner.WIDOW_BANDS == ((45, 54), (55, 64), (65, 74), (75, 120))
    assert list(runner.WIDOW_LOWERS) == [45, 55, 65, 75]
    # The widow-prob helper is candidate 6's OWN (surviving-spouse indexed).
    assert runner._widow_probs is not c5._widow_probs
    # The committed candidate-5 NCHS betas, applied unchanged (read, not fit).
    assert runner._committed_beta_v5() == BETA_V5
    assert runner._BETA_V5_EXPECTED == BETA_V5
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v6"
    assert "4912170754" in runner.SPEC_REGISTRATION
    assert "4911788302" in runner.CANDIDATE5_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert runner.REGISTERED_MODAL_CELL == MODAL_CELL
    assert set(runner.TARGETED_CELLS) == set(TARGETED_CELLS)


def test_committed_beta_reads_from_candidate5_artifact():
    """The applied beta is candidate 5's committed value, read (not re-fit)."""
    runner = _import_runner()
    a5 = _artifact_c5()
    committed = a5["mortality_trend_beta_comparison"]["beta_sex_nchs"]
    assert runner._committed_beta_v5() == {
        "female": float(committed["female"]),
        "male": float(committed["male"]),
    }


# --------------------------------------------------------------------------
# Artifact presence, spec, lock, the one delta (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v6"
    assert a["run"] == "gate2_hazard_v6"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 6"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_delta_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert a["candidate5_registration"] == SPEC_URL_C5
    assert a["candidate1_registration"] == SPEC_URL_C1
    delta = a["delta_vs_candidate5"].lower()
    assert "marriage histor" in delta
    assert "mh85_23" in delta
    assert "surviving spouse" in delta
    assert "exp(beta_sex" in delta
    assert "1995" in delta
    assert "death-record" in delta
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate5_registration"] == SPEC_URL_C5


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.5-0.6"
    assert fc["registration"] == SPEC_URL
    assert MODAL_CELL in fc["modal_failure"]
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
    }
    wid = model["components"]["widowhood"].lower()
    assert "delta" in wid and "surviving-spouse" in wid
    assert "marriage-history" in wid or "marriage history" in wid
    # First marriage + fertility are byte-identical to candidate 5.
    fm = model["components"]["first_marriage"].lower()
    assert "byte-identical to candidate 5" in fm
    fert = model["components"]["fertility"].lower()
    assert "byte-identical to candidate 5" in fert


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
# Structural: the delta'd surviving-spouse widowhood LEVEL (always runnable)
# --------------------------------------------------------------------------
def test_delta_level_source_recorded():
    """Every seed records the surviving-spouse marriage-history level source."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        assert "mh85_23" in meta["mortality_level_source"]
        assert "surviving spouse" in meta["mortality_level_source"]
        assert "death-record" in meta["mortality_level_source"]
        assert meta["mortality_cells"] == 8  # 4 WIDOWHOOD bands x 2 sexes
        assert meta["mortality_level_bands"] == [
            [45, 54],
            [55, 64],
            [65, 74],
            [75, 120],
        ]
        # The new level table keys are the widowhood bands x surviving sex.
        new = meta["mortality_level_new_widowhood"]
        assert set(new) == {
            f"{band}|{sex}"
            for band in ("45-54", "55-64", "65-74", "75+")
            for sex in ("female", "male")
        }
        # Candidate 5's death-record level is retained for the comparison.
        old = meta["mortality_level_candidate5_deathrecord"]
        assert len(old) == 14  # 7 MORT bands x 2 sexes
        # The applied beta is candidate 5's committed value, unchanged.
        assert meta["mortality_beta_by_sex"] == BETA_V5
        assert (
            meta["mortality_beta_by_sex"]
            == meta["mortality_beta_by_sex_candidate5_recomputed"]
        )
        diag = meta["mortality_level_diagnostics"]
        assert diag["n_cells"] == 8
        assert set(diag["cells"]) == set(new)
        for key, cell in diag["cells"].items():
            assert cell["rate"] == pytest.approx(new[key], abs=0)


def test_mortality_level_comparison_carried():
    """The old-vs-new-by-band LEVEL comparison recomputes from the metas."""
    a = _artifact()
    cmp = a["mortality_level_comparison"]
    assert cmp["old_death_record_bands"] == [
        [25, 34],
        [35, 44],
        [45, 54],
        [55, 64],
        [65, 74],
        [75, 84],
        [85, 120],
    ]
    assert cmp["new_widowhood_level_bands"] == [
        [45, 54],
        [55, 64],
        [65, 74],
        [75, 120],
    ]
    # The cross-seed means recompute from the per-seed metas.
    by_seed = a["per_seed"]

    def mean(getter, key):
        return float(np.mean([getter(s)[key] for s in by_seed]))

    for key, val in cmp["new_widowhood_level_mean"].items():
        assert val == pytest.approx(
            mean(
                lambda s: s["component_meta"]["mortality_level_new_widowhood"],
                key,
            ),
            abs=1e-12,
        )
    # Every by-band row is consistent with the means; the overlapping
    # single-band ratios are new/old at the same (band, sex) label.
    old_mean = cmp["old_death_record_level_mean"]
    new_mean = cmp["new_widowhood_level_mean"]
    for row in cmp["by_band"]:
        key = f"{row['band']}|{row['sex']}"
        assert row["new_widowhood_level_mean"] == pytest.approx(
            new_mean[key], abs=0
        )
        if key in old_mean:
            assert row["ratio_new_over_old"] == pytest.approx(
                new_mean[key] / old_mean[key], abs=1e-12
            )


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
# First-marriage + fertility byte-identity to candidate 5 (always runnable)
# --------------------------------------------------------------------------
def test_first_marriage_fertility_byte_identical_to_candidate5():
    """The delta only moves widowhood and the states it feeds.

    First marriage and fertility are marital-state-independent under the
    shared RNG stream, so every ``first_marriage.*``, ``asfr.*`` and
    ``completed_fertility.*`` gated cell must carry candidate 5's exact
    ``r_candidate``. The recorded attestation, and a direct recompute against
    the committed candidate-5 artifact.
    """
    a = _artifact()
    fi = a["first_marriage_fertility_identity_vs_candidate5"]
    assert fi["available"] is True
    assert fi["byte_identical"] is True
    assert fi["max_abs_r_candidate_deviation_vs_candidate5"] == 0.0

    a5 = _artifact_c5()
    by5 = {s["seed"]: s for s in a5["per_seed"]}
    n_checked = 0
    for seed in a["per_seed"]:
        s5 = by5[seed["seed"]]
        for cell, rec in seed["gated_cells"].items():
            if _is_first_marriage_or_fertility(cell):
                assert rec["r_candidate"] == pytest.approx(
                    s5["gated_cells"][cell]["r_candidate"], abs=0
                ), (seed["seed"], cell)
                n_checked += 1
    assert n_checked == fi["n_cells_checked"]


# --------------------------------------------------------------------------
# Modal, targeted cells, decider, candidate-5 movement (always runnable)
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
    modal_decided = (not v["gate_2_pass"]) and (
        dec["n_seeds_pass_if_modal_forgiven"] >= 4
    )
    assert dec["modal_decided"] == modal_decided


def test_candidate5_comparison_movement_recomputes():
    """The vs-candidate-5 movement block recomputes from both artifacts."""
    a = _artifact()
    cmp = a["candidate5_comparison"]
    assert cmp["available"] is True
    a5 = _artifact_c5()
    by5 = {s["seed"]: s for s in a5["per_seed"]}
    by6 = {s["seed"]: s for s in a["per_seed"]}
    for cell, d in cmp["cells"].items():
        c5_pass = sum(by5[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c6_pass = sum(by6[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate5_n_seeds_pass"] == c5_pass
        assert d["candidate6_n_seeds_pass"] == c6_pass
        for s in GATE_SEEDS:
            assert d["candidate5_per_seed_score"][str(s)] == pytest.approx(
                by5[s]["gated_cells"][cell]["score"], abs=0
            )
            assert d["candidate6_per_seed_score"][str(s)] == pytest.approx(
                by6[s]["gated_cells"][cell]["score"], abs=0
            )


def test_revision_pins_record_runner_and_candidate5_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v6"
    assert pins["sklearn_version"].startswith("1.9")
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    for n in (1, 2, 3, 4, 5):
        assert pins[f"candidate{n}_runner"] == (
            f"scripts/run_gate2_candidate{n}.py"
        )
        assert len(pins[f"candidate{n}_runner_sha256"]) == 64
    assert pins["candidate5_artifact"] == "runs/gate2_hazard_v5.json"
    assert len(pins["candidate5_artifact_sha256"]) == 64


def test_forecast_pointer_present():
    fp = _artifact()["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate5_registration"] == SPEC_URL_C5
    assert fp["floor_run"] == "runs/gate2_floors_v2.json"
    assert 0.0 <= float(fp["faithful_candidate_oc"]) <= 1.0


# --------------------------------------------------------------------------
# Pinned outcome of the one-shot run (FAIL 1/5; the published finding)
# --------------------------------------------------------------------------
def test_verdict_is_fail_one_of_five():
    """The pre-registered outcome: FAIL, 1/5 seeds pass (published)."""
    v = _artifact()["verdict"]
    assert v["gate_2_pass"] is False
    assert v["n_seeds_pass"] == 1
    assert v["seed_pass"] == {
        "0": True,
        "1": False,
        "2": False,
        "3": False,
        "4": False,
    }


def test_seed_fail_sets_pinned():
    """Each seed's failing gated-cell set is pinned (the published finding).

    Seed 0 clears every gated cell (46/46 -- the first gate-2 seed to pass);
    seed 1 clips the female widowed stock, its remarriage and mean lifetime
    marriages; seed 2 clips the untouched c1970s completed-fertility cell; and
    seeds 3 and 4 clip only the registered modal.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        fails = {c for c, r in seed["gated_cells"].items() if not r["pass"]}
        assert fails == EXPECTED_SEED_FAILS[seed["seed"]], seed["seed"]
    s0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    assert s0["n_gated_pass"] == 46
    assert s0["seed_pass"] is True


def test_registered_modal_materialized_but_not_decisive():
    """mean_lifetime_marriages|male failed 2 seeds but did NOT decide the gate.

    The registered modal materialized (seeds 3, 4) yet forgiving it alone
    lifts only to three passing seeds -- below the four the gate needs -- so
    the failure is broader than the modal, the honest published finding.
    """
    a = _artifact()
    modal = a["modal_failure_materialized"]
    assert modal["modal_cell"] == MODAL_CELL
    assert modal["modal_failed"] is True
    assert modal["modal_failed_seeds"] == [3, 4]
    assert modal["modal_is_sole_failing_cell"] is False
    dec = modal["decider_analysis"]
    assert dec["n_seeds_pass_actual"] == 1
    assert dec["n_seeds_pass_if_modal_forgiven"] == 3
    assert dec["n_seeds_pass_if_targeted_forgiven"] == 1
    assert dec["n_seeds_pass_if_both_forgiven"] == 3
    assert dec["modal_decided"] is False
    assert "broader" in dec["decider"]


def test_key_cells_movement_pins():
    """How the delta's key cells moved vs candidate 5 (pinned).

    The source alignment fixed the female widowhood FLOW
    (widowhood.45-64|female 3/5 -> 5/5) and lifted the elderly female widowed
    STOCK (share_widowed.75+|female 1/5 -> 4/5, only seed 1 still clips); the
    registered modal eased (mean_lifetime_marriages|male 1/5 -> 3/5); and the
    untouched completed_fertility.c1970s clip persisted byte-identically
    (4/5 -> 4/5, identical mean score).
    """
    a = _artifact()
    cells = a["candidate5_comparison"]["cells"]
    move = {
        c: (d["candidate5_n_seeds_pass"], d["candidate6_n_seeds_pass"])
        for c, d in cells.items()
    }
    assert move["share_widowed.75+|female"] == (1, 4)
    assert move["widowhood.45-64|female"] == (3, 5)
    assert move[MODAL_CELL] == (1, 3)
    assert move["completed_fertility.c1970s"] == (4, 4)
    # The untouched fertility clip's score is byte-identical to candidate 5.
    clip = cells["completed_fertility.c1970s"]
    assert clip["candidate6_mean_score"] == pytest.approx(
        clip["candidate5_mean_score"], abs=0
    )
    # Every widowhood FLOW cell now passes every seed (the delta's target).
    widow_cells = a["verdict"]["per_block"]["widowhood"]["cells"]
    assert widow_cells  # the widowhood family is gate-eligible
    for s in a["per_seed"]:
        for cell in widow_cells:
            assert s["gated_cells"][cell]["pass"] is True, (s["seed"], cell)


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
    assert meta["mortality_beta_by_sex"] == BETA_V5
    assert meta["mortality_cells"] == 8


@needs_psid
def test_new_level_derives_from_marriage_history_endings():
    """The delta: the new LEVEL table IS the gate reference's widowhood build.

    On seed 0's train complement, the fitted mortality level equals -- cell
    for cell -- ``transitions.hazard_cells`` widowhood endings (mh85_23 spouse
    death over married exposure, by surviving-spouse age band x sex), it is NOT
    candidate 5's death-record table, and every non-delta component (divorce,
    remarriage, spousal gap, single-year fertility, first marriage) is
    byte-identical to candidate 5.
    """
    runner = _import_runner()
    c5 = _import_c5()
    from populace_dynamics.data import marriage, transitions
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
    comp6 = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    comp5 = c5.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # (i) The level table IS the gate reference's widowhood construction.
    cells = transitions.hazard_cells(panel, ids_b, weighted=True)
    widow = {
        key[len("widowhood.") :]: float(cell["rate"])
        for key, cell in cells.items()
        if key.startswith("widowhood.")
    }
    assert comp6.mortality == widow
    assert set(comp6.mortality) == {
        f"{transitions.band_label(lo, hi)}|{sex}"
        for (lo, hi) in transitions.WIDOWHOOD_AGE_BANDS
        for sex in ("female", "male")
    }
    assert len(comp6.mortality) == 8

    # (ii) It is NOT candidate 5's death-record level (the delta).
    assert comp6.mortality != comp5.mortality

    # (iii) Every non-delta component is byte-identical to candidate 5.
    assert comp6.remarriage == comp5.remarriage
    assert np.array_equal(comp6.divorce, comp5.divorce)
    assert comp6.gap_by_sex == comp5.gap_by_sex
    for sex in ("female", "male"):
        assert np.array_equal(
            comp6.gap_dist_by_sex[sex], comp5.gap_dist_by_sex[sex]
        )
    assert comp6.fertility == comp5.fertility
    assert comp6.first_marriage.knots == (20.0, 22.0, 25.0, 30.0, 40.0)

    # (iv) The applied beta is candidate 5's committed value (read, not fit).
    assert comp6.meta["mortality_beta_by_sex"] == runner._committed_beta_v5()


@needs_psid
def test_seed0_first_marriage_fertility_byte_identical_live():
    """Live proof: first-marriage + fertility cells match candidate 5 exactly.

    Simulating seed 0's holdout under candidate 6 and candidate 5 with their
    respective components draws the SAME per-year uniform blocks (widowhood
    only leaves the married state; first marriage and fertility are
    marital-state-independent), so every ``first_marriage.*``, ``asfr.*`` and
    ``completed_fertility.*`` reference cell is byte-identical and the same
    births are produced.
    """
    runner = _import_runner()
    c5 = _import_c5()
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

    comp6 = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    comp5 = c5.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    sim6, b6 = runner.simulate_holdout(
        panel, ids_a, comp6, runner.SIM_SEED_BASE
    )
    sim5, b5 = c5.simulate_holdout(panel, ids_a, comp5, c5.SIM_SEED_BASE)

    m6 = transitions.reference_moments(
        sim6, transitions.build_fertility_panel(sim6, b6), ids_a, weighted=True
    )
    m5 = transitions.reference_moments(
        sim5, transitions.build_fertility_panel(sim5, b5), ids_a, weighted=True
    )
    n_checked = 0
    for cell in m6:
        if _is_first_marriage_or_fertility(cell):
            assert m6[cell]["rate"] == pytest.approx(
                m5[cell]["rate"], abs=1e-12
            ), cell
            n_checked += 1
    assert n_checked > 0
    assert len(b6) == len(b5)  # identical births
