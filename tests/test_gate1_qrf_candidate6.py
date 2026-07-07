"""Tests for the gate-1 candidate-6 anchored rank-transition-kernel run.

Mirrors the prior runs' tests (``test_gate1_qrf_candidate5b`` and its
predecessors), adapted for the counting-estimator candidate:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent, and ``importorskip("populace.fit")`` because
  the participation gate needs it) reruns seed 0 through the candidate-6
  kernel estimation + backward generation and pins the committed
  artifact's seed-0 geometry and battery values to float precision. The
  candidate-5b run proved this catches signature drift, so it is run live
  in the dedicated gate venv before the artifact is committed.
* :func:`test_kernel_counting_on_synthetic_pool` (always runnable; no
  PSID, no populace-fit) drives the kernel counting, add-one smoothing,
  row normalization, and re-entry counting on a hand-built train panel,
  checking each frozen rule directly.
* :func:`test_draw_bin_convention` (always runnable) pins the inverse-CDF
  bin draw to the gate's exact ``(cumulative >= u).argmax`` convention.
* The always-runnable consistency tests touch only the committed artifact
  and ``gates.yaml``: the artifact schema and spec URL are recorded, the
  battery reference reproduces exactly, every reported pass/fail
  recomputes from its own stored score against its stored threshold, the
  stored thresholds equal the locked ones in ``gates.yaml``, the
  zero-persistence identity holds, the gate verdict recomputes from the
  seed-conjunction table, and the kernel diagnostics are
  reported-not-gated (each kernel row and re-entry row is a proper
  distribution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_rank_kernel_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4895401373"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate6 as runner

    return runner


# --------------------------------------------------------------------------
# Reproduction (needs the staged PSID family files AND populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 and match the committed artifact to float precision."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    runner = _import_runner()

    artifact = _artifact()
    seed0 = next(s for s in artifact["per_seed"] if s["seed"] == 0)
    thresholds = runner.load_gate1_thresholds()
    views_cfg = thresholds["views"]
    battery_reference = json.loads(
        (runner.ROOT / runner.BATTERY_REFERENCE_RUN).read_text()
    )["battery_reference"]
    panel = runner.load_filtered_panel()
    all_anchor = runner.anchor_rows(panel)
    view_specs = {
        "psid_family_earnings_pairs": runner.build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": runner.build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }
    result = runner.run_seed(
        0,
        panel,
        all_anchor,
        view_specs,
        views_cfg,
        battery_reference,
        battery_tol,
        False,
    )
    for view, block in seed0["geometry"].items():
        for metric, stored in block["scores"].items():
            assert result["geometry"][view]["scores"][metric] == pytest.approx(
                stored, abs=1e-12
            ), f"{view}.{metric}"
    for stat, stored in seed0["battery_values"].items():
        assert result["battery_values"][stat] == pytest.approx(
            stored, abs=1e-12
        ), stat
    # The counted kernel is deterministic given the split: pair counts
    # reproduce exactly.
    assert (
        result["kernel"]["n_kernel_pairs"] == seed0["kernel"]["n_kernel_pairs"]
    )
    assert (
        result["kernel"]["n_reentry_pairs"]
        == seed0["kernel"]["n_reentry_pairs"]
    )


# --------------------------------------------------------------------------
# Kernel counting on a hand-built pool (always runnable; no populace-fit)
# --------------------------------------------------------------------------
def test_kernel_counting_on_synthetic_pool():
    """Kernel + re-entry counting, add-one smoothing, and normalization."""
    runner = _import_runner()

    # Two persons, biennial. Person 1 anchors positive; person 2 too.
    # Build so that person 1 contributes one both-positive backward pair
    # and person 2 contributes one re-entry pair (later zero, earlier
    # positive). Cells are (age-bin, period); pick ages in one bin.
    rows = [
        # person 1: 2018 pos, 2020 pos (anchor). backward pair (2018->2020).
        (1, 2018, 30000.0, 40, 1.0),
        (1, 2020, 50000.0, 42, 1.0),
        # person 2: 2018 pos, 2020 zero (anchor zero). re-entry pair.
        (2, 2018, 20000.0, 40, 2.0),
        (2, 2020, 0.0, 42, 2.0),
        # a few extra positives to populate cell quantiles / ranks.
        (3, 2018, 45000.0, 41, 1.0),
        (3, 2020, 47000.0, 43, 1.0),
        (4, 2018, 10000.0, 40, 1.0),
        (4, 2020, 12000.0, 42, 1.0),
    ]
    panel = pd.DataFrame(
        rows, columns=["person_id", "period", "earnings", "age", "weight"]
    )
    all_anchor = runner.anchor_rows(panel)
    marginals = runner.fit_cell_marginals(panel)
    kernel = runner.estimate_kernel(panel, all_anchor, marginals)

    P = kernel["P"]
    R = kernel["R"]
    assert P.shape == (5, 20, 20)
    assert R.shape == (5, 20)

    # Every kernel row is a proper distribution (Laplace makes each row
    # strictly positive and normalized).
    row_sums = P.sum(axis=2)
    assert np.allclose(row_sums, 1.0)
    assert (P > 0).all()
    re_sums = R.sum(axis=1)
    assert np.allclose(re_sums, 1.0)
    assert (R > 0).all()

    # Persons 1, 3, 4 give both-positive backward pairs (3 kernel pairs);
    # person 2 gives one re-entry pair. Person 2's anchor is zero, so its
    # backward pair is NOT a kernel pair.
    assert kernel["n_kernel_pairs"] == 3
    assert kernel["n_reentry_pairs"] == 1
    assert int(kernel["kernel_pairs_per_bin"].sum()) == 3
    assert int(kernel["reentry_pairs_per_bin"].sum()) == 1

    # Raw weighted counts: person 2's re-entry pair carries its earlier
    # (2018) weight of 2.0.
    assert kernel["R_raw"].sum() == pytest.approx(2.0, abs=0)
    # Kernel raw weight = sum of the three positive earlier weights (all 1).
    assert kernel["N_raw"].sum() == pytest.approx(3.0, abs=0)


def test_draw_bin_convention():
    """The bin draw is the gate's exact ``(cumulative >= u).argmax``."""
    runner = _import_runner()
    # A pmf concentrated on bins 0, 5, 19.
    pmf = np.zeros(20)
    pmf[0], pmf[5], pmf[19] = 0.25, 0.25, 0.5
    rows = np.tile(pmf, (5, 1))
    u = np.array([0.0, 0.24, 0.26, 0.5, 1.0 - 1e-12])
    got = runner._draw_bin(rows, u)
    # u=0 -> first bin whose inclusive cumulative >= 0 is bin 0.
    # u=0.24 -> still within bin 0's mass (0.25) -> bin 0.
    # u=0.26 -> cumulative reaches 0.5 at bin 5 -> bin 5.
    # u=0.5 -> cumulative 0.5 first reached at bin 5 -> bin 5.
    # u~=1 -> bin 19.
    assert list(got) == [0, 0, 5, 5, 19]


# --------------------------------------------------------------------------
# Always-runnable consistency tests (artifact + gates.yaml only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_rank_kernel.v1"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen-spec issue-comment URL is carried in the artifact."""
    assert _artifact()["spec_registration"] == SPEC_URL


def test_no_calibration_stage_declared():
    """Candidate 6 has no calibration stage (zero free parameters)."""
    artifact = _artifact()
    assert "calibration" in artifact["model"]
    assert artifact["model"]["calibration"].startswith("none")
    # No SMM block (that was candidate 5b); a kernel block instead.
    assert "smm" not in artifact["model"]
    assert "kernel" in artifact["model"]


def test_battery_reference_reproduced_exactly_in_artifact():
    """The stored reproduction block must attest exact float matches."""
    repro = _artifact()["battery_reference_reproduction"]
    assert repro["all_committed_values_reproduced_exactly"] is True
    committed = json.loads(
        (ROOT / _artifact()["battery_reference_run"]).read_text()
    )["battery_reference"]
    for name, chk in repro["checks"].items():
        assert chk["exact_float_match"] is True
        assert chk["committed"] == pytest.approx(committed[name], abs=0)
        assert chk["recomputed"] == pytest.approx(committed[name], abs=0)


def test_stored_thresholds_match_locked_gates_yaml():
    """Every stored geometry threshold equals the locked one."""
    artifact = _artifact()
    views_cfg = _gate1_thresholds()["views"]
    for seed in artifact["per_seed"]:
        for vname, view in seed["geometry"].items():
            stored = dict(view["thresholds"])
            # Ratified amendments may demote a metric after this run
            # published (gates.yaml amendment_history); the stored
            # thresholds remain the correct record of the gate AS RUN.
            demoted = views_cfg[vname].get(
                "reported_not_gated", []
            ) + views_cfg[vname].get("per_seed_rule_superseded", [])
            for metric in demoted:
                stored.pop(f"{metric}_max", None)
                stored.pop(f"{metric}_range", None)
                stored.pop(f"{metric}_min", None)
            assert stored == views_cfg[vname]["geometry"]

    # Battery tolerances stored per check must equal the locked ones.
    battery_tol = {
        k[: -len("_tolerance")]: v
        for k, v in _gate1_thresholds()["battery"].items()
        if k.endswith("_tolerance")
    }
    for seed in artifact["per_seed"]:
        for stat, chk in seed["battery_checks"].items():
            assert chk["tolerance"] == pytest.approx(battery_tol[stat], abs=0)


def _recompute_geometry_pass(check: dict) -> bool:
    comp = check["comparison"]
    score = check["score"]
    thr = check["threshold"]
    if comp == "<=":
        return score <= thr
    if comp == ">=":
        return score >= thr
    if comp == "in":
        lo, hi = thr
        return (score >= lo) and (score <= hi)
    raise AssertionError(f"unknown comparison {comp!r}")


def test_every_geometry_pass_recomputes_from_stored_score():
    """Recompute each geometry pass/fail from its stored score+threshold."""
    artifact = _artifact()
    for seed in artifact["per_seed"]:
        seed_geometry_pass = True
        for vname, view in seed["geometry"].items():
            view_pass = True
            for tname, chk in view["checks"].items():
                recomputed = _recompute_geometry_pass(chk)
                assert recomputed == chk["pass"], (
                    f"seed {seed['seed']} {vname}.{tname}: "
                    f"stored pass={chk['pass']} recomputed={recomputed}"
                )
                view_pass = view_pass and chk["pass"]
            assert view["view_pass"] == view_pass
            seed_geometry_pass = seed_geometry_pass and view_pass
        assert seed["geometry_pass"] == seed_geometry_pass


def test_every_battery_pass_recomputes_from_stored_value():
    """Recompute each battery pass/fail from stored value/ref/tolerance."""
    artifact = _artifact()
    for seed in artifact["per_seed"]:
        seed_battery_pass = True
        for stat, chk in seed["battery_checks"].items():
            deviation = abs(chk["value"] - chk["reference"])
            assert deviation == pytest.approx(chk["deviation"], abs=1e-12)
            recomputed = deviation <= chk["tolerance"]
            assert recomputed == chk["pass"], (
                f"seed {seed['seed']} battery {stat}: "
                f"stored pass={chk['pass']} recomputed={recomputed}"
            )
            seed_battery_pass = seed_battery_pass and chk["pass"]
        assert seed["battery_pass"] == seed_battery_pass


def test_battery_references_match_committed_floor():
    """Every stored battery reference equals the committed floor value."""
    artifact = _artifact()
    committed = json.loads(
        (ROOT / artifact["battery_reference_run"]).read_text()
    )["battery_reference"]
    alias = {"mobility_diagonal": "mobility_diagonal_mean"}
    for seed in artifact["per_seed"]:
        for stat, chk in seed["battery_checks"].items():
            ref_key = alias.get(stat, stat)
            assert chk["reference"] == pytest.approx(committed[ref_key], abs=0)


def test_zero_persistence_identity_holds_in_every_seed():
    """The lock pins zero_persistence == 1 - exit_rate."""
    for seed in _artifact()["per_seed"]:
        zp = seed["battery_values"]["zero_persistence"]
        ex = seed["battery_values"]["exit_rate"]
        assert zp == pytest.approx(1.0 - ex, abs=1e-12)


def test_verdict_recomputes_from_seed_conjunction():
    """The gate verdict recomputes from the seed-conjunction table."""
    artifact = _artifact()
    table = artifact["seed_conjunction"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in table:
        assert row["geometry_pass"] == by_seed[row["seed"]]["geometry_pass"]
        assert row["battery_pass"] == by_seed[row["seed"]]["battery_pass"]

    n_geo = sum(1 for r in table if r["geometry_pass"])
    n_bat = sum(1 for r in table if r["battery_pass"])
    verdict = artifact["verdict"]
    assert verdict["n_geometry_pass"] == n_geo
    assert verdict["n_battery_pass"] == n_bat
    assert verdict["geometry_gate_pass"] == (n_geo >= 4)
    assert verdict["battery_gate_pass"] == (n_bat >= 4)
    assert verdict["gate_1_pass"] == ((n_geo >= 4) and (n_bat >= 4))


def test_kernel_diagnostics_are_proper_distributions():
    """Reported-not-gated: every re-entry distribution is normalized.

    The re-entry rows are published per anchor bin as distributions over
    the twenty rank bins; each must sum to one and be non-negative (the
    add-one smoothing guarantees positivity). The kernel diagonal and
    corner masses are probabilities in ``[0, 1]``.
    """
    artifact = _artifact()
    for seed in artifact["kernel_context"]["per_seed"]:
        for A, dist in seed["reentry_distributions"].items():
            assert len(dist) == 20
            assert all(v >= 0 for v in dist)
            assert sum(dist) == pytest.approx(1.0, abs=1e-9), A
        diag = seed["kernel_diagonal_mass_by_anchor_bin"]
        for v in diag.values():
            assert 0.0 <= v <= 1.0
        for corner in ("top_to_top", "bottom_to_bottom"):
            for v in seed["kernel_corner_mass"][corner].values():
                assert 0.0 <= v <= 1.0


def test_candidate_panel_pin_metadata_consistent():
    """Each seed's window counts are positive and pairs >= runs.

    The candidate panel holds exactly the holdout persons on exactly
    their observed periods, so its projected window counts are a property
    of the holdout support; the stored counts are recorded per view and
    must be positive and consistent with the person count.
    """
    for seed in _artifact()["per_seed"]:
        assert seed["n_persons"] > 0
        assert seed["n_person_periods"] >= seed["n_persons"]
        assert set(seed["n_windows"]) == {
            "psid_family_earnings_pairs",
            "psid_family_earnings_runs",
        }
        for vname, n in seed["n_windows"].items():
            assert n > 0, f"{vname} has no windows"
        # window-2 (pairs) yields at least as many windows as window-3.
        assert (
            seed["n_windows"]["psid_family_earnings_pairs"]
            >= seed["n_windows"]["psid_family_earnings_runs"]
        )
