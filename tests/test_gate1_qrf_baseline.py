"""Tests for the gate-1 chained-weighted-QRF baseline run.

Two guarantees:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the
  PSID family files are absent) reruns seed 0 through the runner's
  building blocks and pins the committed artifact's seed-0 geometry and
  battery values to float precision — the run is reproducible from the
  seeds alone.
* The always-runnable consistency tests touch only the committed
  artifact and ``gates.yaml``: every reported pass/fail recomputes from
  its own stored score against its stored threshold, the stored
  thresholds equal the locked ones in ``gates.yaml``, and the gate
  verdict recomputes from the seed-conjunction table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_qrf_baseline_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


# --------------------------------------------------------------------------
# Reproduction (needs the staged PSID family files)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 and match the committed artifact to float precision."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (it pins scikit-learn<1.9, "
        "so gate runs use a dedicated venv)",
    )
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_baseline as runner

    artifact = _artifact()
    seed0 = next(s for s in artifact["per_seed"] if s["seed"] == 0)

    panel = runner.load_filtered_panel()

    # The battery reference must still reproduce exactly.
    repro = runner.reproduce_battery_reference(panel)
    assert repro["all_committed_values_reproduced_exactly"] is True

    holdout, train = runner.split_holdout_train(panel, 0)
    assert int(holdout.person_id.nunique()) == seed0["n_persons"]
    assert int(len(holdout)) == seed0["n_person_periods"]
    assert int(train.person_id.nunique()) == seed0["n_train_persons"]

    pairs = runner.build_backward_pairs(train)
    assert int(len(pairs)) == seed0["n_train_pairs"]

    fitted = runner.fit_backward_model(pairs, 0)
    candidate = runner.generate_candidate(fitted, holdout)

    # Geometry: rescore both locked views and match every stored score.
    views_cfg = _gate1_thresholds()["views"]
    view_specs = {
        "psid_family_earnings_pairs": runner.build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": runner.build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }
    from populace_dynamics.harness import panel as hpanel

    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(candidate, holdout, view, seed=0)
        stored = seed0["geometry"][vname]["scores"]
        for key, stored_value in stored.items():
            assert scores[key] == pytest.approx(
                stored_value, rel=0, abs=1e-12
            ), f"{vname}.{key}: {scores[key]} != {stored_value}"
        # And the per-threshold pass/fail recomputes identically.
        checks = runner.check_geometry(scores, views_cfg[vname]["geometry"])
        for tname, chk in checks.items():
            assert (
                chk["pass"]
                == seed0["geometry"][vname]["checks"][tname]["pass"]
            )

    # Battery: recompute on the candidate and match every stored value.
    battery = runner.compute_battery(candidate)
    for stat, stored_value in seed0["battery_values"].items():
        assert battery[stat] == pytest.approx(
            stored_value, rel=0, abs=1e-12
        ), f"battery {stat}: {battery[stat]} != {stored_value}"


# --------------------------------------------------------------------------
# Internal consistency (always runnable; committed files only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_qrf_baseline.v1"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


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
            for metric in views_cfg[vname].get("reported_not_gated", []):
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
    alias = {"mobility_diagonal": "mobility_diagonal"}
    for seed in artifact["per_seed"]:
        for stat, chk in seed["battery_checks"].items():
            locked_stat = alias.get(stat, stat)
            assert chk["tolerance"] == pytest.approx(
                battery_tol[locked_stat], abs=0
            )


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
    # The conjunction table agrees with per_seed.
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


def test_candidate_panel_pin_metadata_consistent():
    """Each seed's candidate window counts equal the holdout window counts.

    The candidate panel holds exactly the holdout persons on exactly
    their observed periods, so its projected window counts are a
    property of the holdout support; the stored counts are recorded per
    view and must be positive and consistent with the person count.
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
