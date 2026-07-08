"""Tests for the gate-2 candidate-8 (run 1) pre-registered run.

Candidate 8 is the eighth pre-registered gate-2 candidate: candidate 6's
five-component family-transition simulator with EXACTLY ONE named delta --
the remarriage hazard is additionally conditioned on the order of the
marriage being entered (2nd vs 3rd+), with candidate 7's YSD band x origin x
sex construction and add-one smoothing (candidate 7's estimator, reused byte-
for-byte), THEN both order-specific tables are multiplied by a single train-
side scalar that makes the order-split table's exposure-weighted aggregate
remarriage rate over the train dissolved person-years equal the unsplit
candidate-6 table's aggregate over the same exposure (one scalar per seed,
computed on train only, recorded in the artifact). There is NO fertility delta
(candidate 7's marital-status delta 2 was falsified), so fertility is byte-
identical to candidate 6. Everything else -- the surviving-spouse widowhood
level, the committed NCHS betas, the knot-at-22 first-marriage spline,
divorce, the spousal-gap draw, the RNG rule -- is byte-identical to candidate
6. Frozen spec: issue #42 comment 4912995860.

Three tiers, mirroring the candidate-6/7 suites:

* the always-runnable consistency suite (touches only the committed
  candidate-8 / candidate-6 / candidate-7 artifacts and ``gates.yaml``):
  schema and spec URLs, the recorded delta, the bit-exact reproduction
  precheck attestation, every stored gated-cell pass recomputes from its score
  against its stored (locked) tolerance, the stored tolerances equal the
  locked gates.yaml, each seed's pass recomputes, the verdict and per-block
  counts recompute, the report-only cells never gate, the registered modal
  (``mean_lifetime_marriages|male``), the targeted cells, the seed-0 analysis,
  the decider analysis, the candidate-6 and candidate-7 movement all
  recompute, the first-marriage/ever-married AND fertility cells are byte-
  identical to candidate 6 (the delta moves only remarriage), and the per-seed
  rescale scalars recompute and preserve the aggregate;
* the structural delta check: the remarriage table is the order-conditioned
  24-cell construction (3 YSD bands x 2 origins x 2 sexes x 2 order bits, not
  the 12-cell candidate-6 table), rescaled by the recorded scalar;
* :func:`test_seed0_reproduces_committed_artifact` and the live delta checks
  (skipped when the PSID history files are absent) rerun seed 0 end-to-end and
  pin the committed seed-0 block to float precision, confirm the order split
  derives and rescales as specified, confirm the non-delta components AND
  fertility are byte-identical to candidate 6, and confirm the first-marriage,
  ever-married and fertility reference cells are byte-identical to candidate
  6's simulation while remarriage and the marriage counts move.

The one-shot outcome (published REGARDLESS of verdict): FAIL 0/5. The rescaled
order split KEPT candidate 7's after-divorce compositional fix
(remarriage.after_divorce 4/5 -> 5/5, held at 5/5) and reverting the fertility
delta returned asfr.15-19 (2/5 -> 5/5, byte-identical to candidate 6), but the
aggregate-preserving scalar (~0.996) only PROTECTED the male marriage-count
boundary -- it did not raise it -- so mean_lifetime_marriages|male stayed 3/5
(the registered modal, failing seeds 3 and 4), and seed 0 did not return to
46/46: it stayed 45/46, regressing on share_divorced.45-54|female (the order
split's married-state exposure shift, not the reverted fertility). No seed
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
ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
ARTIFACT_C6 = ROOT / "runs" / "gate2_hazard_v6.json"
ARTIFACT_C7 = ROOT / "runs" / "gate2_hazard_v7.json"
FLOOR = ROOT / "runs" / "gate2_floors_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912995860"
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
    "remarriage.after_divorce",
)

# The pinned one-shot outcome (published REGARDLESS of verdict): FAIL 0/5.
EXPECTED_SEED_FAILS = {
    0: {"share_divorced.45-54|female"},
    1: {"mean_lifetime_marriages|female", "share_widowed.75+|female"},
    2: {"completed_fertility.c1970s"},
    3: {"mean_lifetime_marriages|male", "widowhood.75+|female"},
    4: {"mean_lifetime_marriages|male"},
}

# Seed 0's rescale scalar, pinned to float precision (live-reproducible).
SEED0_SCALAR = 0.9960211021340246


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _artifact_c6() -> dict:
    return json.loads(ARTIFACT_C6.read_text())


def _artifact_c7() -> dict:
    return json.loads(ARTIFACT_C7.read_text())


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
    import run_gate2_candidate8 as runner

    return runner


def _import_c6():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate6 as c6

    return c6


def _import_c7():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate2_candidate7 as c7

    return c7


def _is_first_marriage_or_ever_married(cell: str) -> bool:
    return cell.startswith("first_marriage.") or cell.startswith(
        "ever_married_by_"
    )


def _is_fertility(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


def _is_byte_identical_cell(cell: str) -> bool:
    return _is_first_marriage_or_ever_married(cell) or _is_fertility(cell)


# --------------------------------------------------------------------------
# Runner import + module constants (always runnable)
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and reuses c6 + c7 machinery.

    The frozen dials are IMPORTED from candidate 1; the widowhood level, the
    committed NCHS betas and the first-marriage machinery from candidate 6; and
    the order-split remarriage estimator + lookup from candidate 7 (reused
    byte-for-byte). Candidate 8's OWN additions are the train-side rescale
    helpers.
    """
    runner = _import_runner()
    c6 = _import_c6()
    c7 = _import_c7()
    assert runner.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert runner.SIM_SEED_BASE == 4200
    assert runner.TREND_ANCHOR_YEAR == 1995.0
    # First marriage + widowhood level are candidate 6's, reused unchanged.
    assert runner.fit_first_marriage is c6.fit_first_marriage
    assert runner.FirstMarriageModelC8 is c6.FirstMarriageModelC6
    assert runner._widow_probs is c6._widow_probs
    assert runner._committed_beta_v5 is c6._committed_beta_v5
    assert runner.WIDOW_BANDS == ((45, 54), (55, 64), (65, 74), (75, 120))
    assert runner.YSD_BANDS == ((0, 4), (5, 9), (10, 120))
    assert (runner.FERT_AGE_LO, runner.FERT_AGE_HI) == (15, 49)
    # The order-split estimator + lookup are candidate 7's, reused verbatim.
    assert runner.fit_remarriage_ordered is c7.fit_remarriage_ordered
    assert runner._remarriage_probs_ordered is c7._remarriage_probs_ordered
    # Fertility lookup is candidate 6's (no delta); candidate 8 does NOT
    # define candidate 7's falsified marital-fertility helpers.
    assert not hasattr(runner, "fit_fertility_single_marital")
    assert not hasattr(runner, "_fertility_probs_single_marital")
    # Candidate 8's OWN additions: the train-side rescale.
    assert hasattr(runner, "_remarriage_train_exposure")
    assert hasattr(runner, "_aggregate_preserving_scalar")
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate2_hazard_v8"
    assert "4912995860" in runner.SPEC_REGISTRATION
    assert "4912170754" in runner.CANDIDATE6_REGISTRATION
    assert "4910914098" in runner.CANDIDATE1_REGISTRATION
    assert runner.REGISTERED_MODAL_CELL == MODAL_CELL
    assert set(runner.TARGETED_CELLS) == set(TARGETED_CELLS)


# --------------------------------------------------------------------------
# Artifact presence, spec, lock, the one delta (always runnable)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    a = _artifact()
    assert a["schema_version"] == "gate2_hazard_v8"
    assert a["run"] == "gate2_hazard_v8"
    assert a["gate"] == "gate_2"
    assert a["candidate"] == "candidate 8"
    assert a["revision_pins"]["gates_yaml_locked"] is True
    assert _gate2_thresholds()["locked"] is True


def test_spec_registration_and_delta_recorded():
    a = _artifact()
    assert a["spec_registration"] == SPEC_URL
    assert a["candidate6_registration"] == SPEC_URL_C6
    assert a["candidate5_registration"] == SPEC_URL_C5
    assert a["candidate1_registration"] == SPEC_URL_C1
    d = a["delta_vs_candidate6"].lower()
    assert "order" in d and "2nd" in d and "3rd" in d
    # The delta names the aggregate-preservation rescale and the NO-fertility.
    assert "scalar" in d and "aggregate" in d
    assert "no fertility delta" in d
    fp = a["forecast_pointer"]
    assert fp["registration"] == SPEC_URL
    assert fp["candidate6_registration"] == SPEC_URL_C6


def test_forecast_and_model_recorded():
    a = _artifact()
    fc = a["pre_registered_forecast"]
    assert fc["p_pass"] == "0.35-0.45"
    assert fc["registration"] == SPEC_URL
    mf = fc["modal_failure"].lower()
    assert "mean_lifetime_marriages|male" in mf and "seeds 3-4" in mf
    model = a["model"]
    assert model["populace_fit_used"] is False
    assert set(model["components"]) == {
        "first_marriage",
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
    }
    # First marriage AND fertility are byte-identical to candidate 6.
    assert "byte-identical to candidate 6" in (
        model["components"]["first_marriage"].lower()
    )
    assert "byte-identical to candidate 6" in (
        model["components"]["fertility"].lower()
    )
    # Remarriage carries the one DELTA (order split + train-side scalar).
    rem = model["components"]["remarriage"].lower()
    assert "delta" in rem and "order" in rem and "scalar" in rem
    # The ambiguity resolutions name the aggregate-preservation scalar.
    res = model["registered_ambiguity_resolutions"]
    assert "aggregate_preservation_scalar" in res
    assert "no_fertility_delta" in res


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
# Structural: the delta'd remarriage table (always runnable)
# --------------------------------------------------------------------------
def test_delta_remarriage_order_structure():
    """Every seed records the 24-cell rescaled order-conditioned remarriage."""
    a = _artifact()
    for seed in a["per_seed"]:
        meta = seed["component_meta"]
        diag = meta["remarriage_order_diagnostics"]
        assert diag["n_cells"] == 24  # 3 ysd x 2 origin x 2 sex x 2 order
        assert diag["ysd_bands"] == [[0, 4], [5, 9], [10, 120]]
        assert set(diag["order_bits"]) == {"0", "1"}
        for key in diag["cells"]:
            assert key.endswith("|2nd") or key.endswith("|3plus")
        # Candidate 6's 12-cell origin-split table is retained for reference.
        assert len(meta["remarriage_candidate6"]) == 12
        # The pre-rescale order-split cells are retained (24 of them).
        assert len(meta["remarriage_order_prerescale_cells"]) == 24
        assert "marriage-order" in meta["remarriage_representation"].lower()
        # The add-one smoothing constant is candidate 1's exactly.
        assert "remarriage_mean_dissolved_weight_check" in meta


def test_no_fertility_delta_fertility_byte_identical_to_candidate6():
    """No fertility delta: the fertility cells match candidate 6 exactly.

    Unlike candidate 7 (whose delta 2 moved fertility and regressed
    asfr.15-19), candidate 8 leaves fertility byte-identical to candidate 6, so
    every ``asfr.*`` and ``completed_fertility.*`` reference cell carries
    candidate 6's exact ``r_candidate`` -- the identity_vs_candidate6 block
    covers first marriage, ever-married AND fertility.
    """
    a = _artifact()
    idb = a["identity_vs_candidate6"]
    assert idb["available"] is True
    assert idb["byte_identical"] is True
    assert idb["max_abs_r_candidate_deviation_vs_candidate6"] == 0.0
    assert idb["n_fertility_cells_checked"] > 0

    a6 = _artifact_c6()
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    n_fert = 0
    for seed in a["per_seed"]:
        s6 = by6[seed["seed"]]
        for cell, rec in seed["gated_cells"].items():
            if _is_fertility(cell):
                assert rec["r_candidate"] == pytest.approx(
                    s6["gated_cells"][cell]["r_candidate"], abs=0
                ), (seed["seed"], cell)
                n_fert += 1
    assert n_fert == idb["n_fertility_cells_checked"]


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
# First-marriage + ever-married + fertility byte-identity (always runnable)
# --------------------------------------------------------------------------
def test_first_marriage_ever_married_and_fertility_byte_identical():
    """The one delta touches only remarriage.

    First marriage adds no marriages the delta changes, whether a person EVER
    married is first-marriage-driven, and fertility is candidate 6's marital-
    state-independent table (NO delta) -- all under the shared RNG stream. So
    every ``first_marriage.*``, ``ever_married_by_*``, ``asfr.*`` and
    ``completed_fertility.*`` gated cell must carry candidate 6's exact
    ``r_candidate``.
    """
    a = _artifact()
    fi = a["identity_vs_candidate6"]
    assert fi["available"] is True
    assert fi["byte_identical"] is True
    assert fi["max_abs_r_candidate_deviation_vs_candidate6"] == 0.0

    a6 = _artifact_c6()
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    n_checked = 0
    for seed in a["per_seed"]:
        s6 = by6[seed["seed"]]
        for cell, rec in seed["gated_cells"].items():
            if _is_byte_identical_cell(cell):
                assert rec["r_candidate"] == pytest.approx(
                    s6["gated_cells"][cell]["r_candidate"], abs=0
                ), (seed["seed"], cell)
                n_checked += 1
    assert n_checked == fi["n_cells_checked"]


# --------------------------------------------------------------------------
# The rescale scalars recompute and preserve the aggregate (always runnable)
# --------------------------------------------------------------------------
def test_remarriage_rescale_recomputes_and_preserves_aggregate():
    """Each seed's train-side scalar recomputes and preserves the aggregate.

    The scalar = (unsplit candidate-6 expected remarriages over the train
    dissolved exposure) / (order-split expected remarriages over the same
    exposure), and multiplying the order split by it makes its exposure-
    weighted aggregate remarriage rate equal candidate 6's over that exposure
    -- the property the rescale is defined to enforce, checked to float
    precision. Computed on TRAIN only.
    """
    a = _artifact()
    rb = a["remarriage_rescale"]
    assert rb["aggregate_preserved"] is True
    assert rb["aggregate_preservation_max_abs_residual"] == 0.0
    for seed in a["per_seed"]:
        rc = seed["component_meta"]["remarriage_rescale"]
        # scalar = expected_c6 / expected_split_prerescale
        recomputed = (
            rc["expected_remarriages_candidate6"]
            / rc["expected_remarriages_order_split_prerescale"]
        )
        assert rc["scalar"] == pytest.approx(recomputed, rel=1e-12)
        # Aggregate preserved: rescaled aggregate == candidate-6 aggregate.
        assert rc["order_split_train_aggregate_rescaled"] == pytest.approx(
            rc["candidate6_unsplit_train_aggregate"], abs=1e-12
        )
        # Expected rescaled count == candidate-6 expected count.
        assert rc["expected_remarriages_order_split_rescaled"] == (
            pytest.approx(rc["expected_remarriages_candidate6"], rel=1e-12)
        )
        # Aggregate = expected / exposure (both train-only).
        tot = rc["train_dissolved_exposure_weight"]
        assert rc["candidate6_unsplit_train_aggregate"] == pytest.approx(
            rc["expected_remarriages_candidate6"] / tot, rel=1e-12
        )
        # The block's per-seed scalar matches the component meta.
        assert rb["per_seed"][str(seed["seed"])]["scalar"] == rc["scalar"]
    # Every scalar is a small (~0.4%) level correction near 1.
    for scalar in rb["scalar_per_seed"].values():
        assert 0.99 < scalar < 1.0


# --------------------------------------------------------------------------
# Modal, seed-0, decider, candidate-6/7 movement (always runnable)
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


def test_seed0_analysis_recomputes():
    """The seed-0 return analysis recomputes vs the candidate-6 artifact."""
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
    by8 = {s["seed"]: s for s in a["per_seed"]}
    for cell, d in cmp["cells"].items():
        c6_pass = sum(by6[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c8_pass = sum(by8[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate6_n_seeds_pass"] == c6_pass
        assert d["candidate8_n_seeds_pass"] == c8_pass
        for s in GATE_SEEDS:
            assert d["candidate6_per_seed_score"][str(s)] == pytest.approx(
                by6[s]["gated_cells"][cell]["score"], abs=0
            )
            assert d["candidate8_per_seed_score"][str(s)] == pytest.approx(
                by8[s]["gated_cells"][cell]["score"], abs=0
            )


def test_candidate7_comparison_movement_recomputes():
    """The vs-candidate-7 movement block recomputes from both artifacts."""
    a = _artifact()
    cmp = a["candidate7_comparison"]
    assert cmp["available"] is True
    a7 = _artifact_c7()
    by7 = {s["seed"]: s for s in a7["per_seed"]}
    by8 = {s["seed"]: s for s in a["per_seed"]}
    for cell, d in cmp["cells"].items():
        c7_pass = sum(by7[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        c8_pass = sum(by8[s]["gated_cells"][cell]["pass"] for s in GATE_SEEDS)
        assert d["candidate7_n_seeds_pass"] == c7_pass
        assert d["candidate8_n_seeds_pass"] == c8_pass


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


def test_revision_pins_record_runner_and_candidate_shas():
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate2_hazard_v8"
    assert pins["sklearn_version"].startswith("1.9")
    assert pins["gate2_floor_run"] == "runs/gate2_floors_v2.json"
    for n in (1, 2, 3, 4, 5, 6, 7):
        assert pins[f"candidate{n}_runner"] == (
            f"scripts/run_gate2_candidate{n}.py"
        )
        assert len(pins[f"candidate{n}_runner_sha256"]) == 64
    assert pins["candidate6_artifact"] == "runs/gate2_hazard_v6.json"
    assert len(pins["candidate6_artifact_sha256"]) == 64
    assert pins["candidate7_artifact"] == "runs/gate2_hazard_v7.json"
    assert len(pins["candidate7_artifact_sha256"]) == 64


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
    seed -- stayed 45/46, regressing on share_divorced.45-54|female (the order
    split's married-state exposure shift); seeds 3 and 4 fail the
    mean_lifetime_marriages|male boundary (the registered modal), seed 3 also
    the elderly-widow flow; seed 1 fails two independent cells; seed 2 keeps
    the single completed_fertility.c1970s clip.
    """
    a = _artifact()
    for seed in a["per_seed"]:
        fails = {c for c, r in seed["gated_cells"].items() if not r["pass"]}
        assert fails == EXPECTED_SEED_FAILS[seed["seed"]], seed["seed"]
    # Every seed lands 44-45 of 46 (close, but none clears).
    for s in a["per_seed"]:
        assert 44 <= s["n_gated_pass"] <= 45
        assert s["seed_pass"] is False


def test_seed0_did_not_return_but_fertility_regressor_cured():
    """Seed 0 stayed 45/46 -- fertility regressor cured, divorce-stock not.

    The registration expected seed 0 to return to 46/46. It did NOT: it stayed
    45/46. But the split of candidate 7's two seed-0 regressions is the finding
    -- asfr.15-19 (candidate 7's fertility/delta-2 regression) RETURNED to
    passing (fertility is byte-identical to candidate 6), while
    share_divorced.45-54|female (the order split's married-state exposure
    shift, delta 1) persisted as the single seed-0 failure.
    """
    a = _artifact()
    m = a["seed0_full_movement"]
    assert m["seed0_held_all_gated"] is False
    assert m["n_regressed"] == 1
    assert m["n_improved"] == 0
    regressed = {
        c
        for c, d in m["cells"].items()
        if d["candidate6_pass"] and not d["candidate8_pass"]
    }
    assert regressed == {"share_divorced.45-54|female"}
    s0 = a["modal_failure_materialized"]["seed0_analysis"]
    assert s0["seed0_regressed_cells_vs_candidate6"] == [
        "share_divorced.45-54|female"
    ]
    # asfr.15-19 -- candidate 7's OTHER seed-0 regressor -- passes on seed 0.
    seed0 = next(s for s in a["per_seed"] if s["seed"] == 0)
    assert seed0["gated_cells"]["asfr.15-19"]["pass"] is True


def test_key_cells_movement_pins():
    """How the delta's key cells moved (pinned).

    remarriage.after_divorce kept candidate 7's fix (4/5 -> 5/5, held at 5/5);
    the reverted fertility returned asfr.15-19 (candidate 7 2/5 -> candidate 8
    5/5, byte-identical to candidate 6's 5/5); but the aggregate-preserving
    scalar only PROTECTED the male marriage-count boundary --
    mean_lifetime_marriages|male stayed 3/5, failing seeds 3 and 4 (the
    registered modal).
    """
    a = _artifact()
    c6 = a["candidate6_comparison"]["cells"]
    move6 = {
        c: (d["candidate6_n_seeds_pass"], d["candidate8_n_seeds_pass"])
        for c, d in c6.items()
    }
    assert move6["remarriage.after_divorce"] == (4, 5)
    assert move6["mean_lifetime_marriages|male"] == (3, 3)
    assert move6["mean_lifetime_marriages|female"] == (4, 4)
    assert move6["completed_fertility.c1970s"] == (4, 4)
    assert move6["asfr.15-19"] == (5, 5)  # fertility byte-identical to c6
    assert move6["share_divorced.45-54|female"] == (5, 4)

    c7 = a["candidate7_comparison"]["cells"]
    move7 = {
        c: (d["candidate7_n_seeds_pass"], d["candidate8_n_seeds_pass"])
        for c, d in c7.items()
    }
    # Reverting delta 2 restored teen fertility that candidate 7 broke.
    assert move7["asfr.15-19"] == (2, 5)
    # The order split's after-divorce fix survived the rescale.
    assert move7["remarriage.after_divorce"] == (5, 5)

    # The registered modal materialized on seeds 3 and 4.
    modal = a["modal_failure_materialized"]
    assert modal["modal_failed"] is True
    assert modal["modal_failed_seeds"] == [3, 4]


def test_rescale_scalars_pinned():
    """The per-seed train-side rescale scalars are pinned (the published set).

    All five are ~0.996 (a small down-correction: the order split's train-fit
    exposure-weighted aggregate is slightly above candidate 6's, so the scalar
    nudges it back). Seed 0's scalar is pinned to float precision (live-
    reproducible).
    """
    a = _artifact()
    scalars = a["remarriage_rescale"]["scalar_per_seed"]
    assert scalars["0"] == SEED0_SCALAR
    for s in ("0", "1", "2", "3", "4"):
        assert 0.9960 <= scalars[s] <= 0.9965
    assert a["remarriage_rescale"]["scalar_min"] == pytest.approx(
        min(scalars.values()), abs=0
    )
    assert a["remarriage_rescale"]["scalar_max"] == pytest.approx(
        max(scalars.values()), abs=0
    )


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
    # The seed-0 rescale scalar reproduces to float precision.
    assert result["component_meta"]["remarriage_rescale"]["scalar"] == (
        pytest.approx(SEED0_SCALAR, abs=1e-15)
    )


@needs_psid
def test_delta_derives_and_nondelta_and_fertility_byte_identical():
    """The delta derives + rescales as specified; non-delta AND fertility = c6.

    On seed 0's train complement: the remarriage table is the 24-cell order-
    conditioned build rescaled by the aggregate-preserving scalar (its
    exposure-weighted train aggregate equals candidate 6's), and EVERY other
    component -- first marriage, divorce, the surviving-spouse widowhood level,
    the committed betas, the spousal gap AND fertility -- is byte-identical to
    candidate 6.
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
    c8c = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    c6c = c6.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)

    # (i) DELTA: remarriage is the 24-cell order-conditioned table.
    assert len(c8c.remarriage) == 24
    assert all(len(k) == 4 for k in c8c.remarriage)
    assert {k[3] for k in c8c.remarriage} == {0, 1}
    assert len(c6c.remarriage) == 12

    # (ii) The rescale is a single scalar: rescaled == scalar * pre-rescale.
    rc = c8c.meta["remarriage_rescale"]
    scalar = rc["scalar"]
    pre = c8c.meta["remarriage_order_prerescale_cells"]
    diag = c8c.meta["remarriage_order_diagnostics"]["cells"]
    for key, rate in diag.items():
        assert rate == pytest.approx(scalar * pre[key], abs=1e-15)
    # The compositional 3rd+/2nd RATIO is preserved (scalar cancels): the
    # rescaled ratio equals the pre-rescale ratio per (band, origin, sex).
    for b in range(len(runner.YSD_BANDS)):
        for o in ("divorced", "widowed"):
            for s in ("female", "male"):
                r_pre = (
                    pre[f"ysd{b}|{o}|{s}|3plus"] / pre[f"ysd{b}|{o}|{s}|2nd"]
                )
                r_post = (
                    diag[f"ysd{b}|{o}|{s}|3plus"] / diag[f"ysd{b}|{o}|{s}|2nd"]
                )
                assert r_post == pytest.approx(r_pre, rel=1e-12)

    # (iii) Aggregate preserved on train (the defining property).
    exposure = runner._remarriage_train_exposure(panel, order_map, ids_b)
    ordered = {k: v / scalar for k, v in c8c.remarriage.items()}
    s2, agg2 = runner._aggregate_preserving_scalar(
        exposure, c6c.remarriage, ordered
    )
    assert s2 == pytest.approx(scalar, rel=1e-12)
    assert agg2["order_split_train_aggregate_rescaled"] == pytest.approx(
        agg2["candidate6_unsplit_train_aggregate"], abs=1e-12
    )

    # (iv) Fertility is BYTE-IDENTICAL to candidate 6 (no delta).
    assert c8c.fertility == c6c.fertility
    assert all(len(k) == 3 for k in c8c.fertility)

    # (v) Every other component is byte-identical to candidate 6.
    assert np.array_equal(c8c.divorce, c6c.divorce)
    assert c8c.mortality == c6c.mortality
    assert c8c.gap_by_sex == c6c.gap_by_sex
    for sex in ("female", "male"):
        assert np.array_equal(
            c8c.gap_dist_by_sex[sex], c6c.gap_dist_by_sex[sex]
        )
    assert (
        c8c.meta["mortality_beta_by_sex"] == c6c.meta["mortality_beta_by_sex"]
    )
    assert c8c.first_marriage.knots == (20.0, 22.0, 25.0, 30.0, 40.0)
    # The add-one smoothing constant is candidate 1's exactly.
    assert c8c.meta["remarriage_mean_dissolved_weight_check"] == (
        c6c.meta["remarriage_mean_dissolved_weight"]
    )


@needs_psid
def test_seed0_first_marriage_and_fertility_byte_identical_live():
    """Live proof: first-marriage, ever-married AND fertility cells match c6.

    Simulating seed 0's holdout under candidate 8 and candidate 6 draws the
    SAME per-year uniform blocks (the delta moves only the remarriage
    threshold), so every ``first_marriage.*``, ``ever_married_by_*``, ``asfr.*``
    and ``completed_fertility.*`` reference cell is byte-identical -- while
    remarriage and the lifetime-marriage cells move (delta active).
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

    c8c = runner.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    c6c = c6.fit_components(panel, demo, dr, mh, bh, order_map, ids_b)
    sim8, b8 = runner.simulate_holdout(panel, ids_a, c8c, runner.SIM_SEED_BASE)
    sim6, b6 = c6.simulate_holdout(panel, ids_a, c6c, c6.SIM_SEED_BASE)

    m8 = transitions.reference_moments(
        sim8, transitions.build_fertility_panel(sim8, b8), ids_a, weighted=True
    )
    m6 = transitions.reference_moments(
        sim6, transitions.build_fertility_panel(sim6, b6), ids_a, weighted=True
    )
    n_identical = 0
    n_rem_moved = 0
    for cell in m8:
        if _is_byte_identical_cell(cell):
            assert m8[cell]["rate"] == pytest.approx(
                m6[cell]["rate"], abs=1e-12
            ), cell
            n_identical += 1
        elif cell.startswith("remarriage.") or cell.startswith(
            "mean_lifetime"
        ):
            if abs(m8[cell]["rate"] - m6[cell]["rate"]) > 1e-12:
                n_rem_moved += 1
    assert n_identical > 0
    assert n_rem_moved > 0  # the delta is active
