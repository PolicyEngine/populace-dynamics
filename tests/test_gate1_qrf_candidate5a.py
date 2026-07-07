"""Tests for the gate-1 candidate-5a MINT-style donor-splicing run.

Mirrors the prior runs' tests (``test_gate1_qrf_candidate4`` and its
predecessors), adapted for a DETERMINISTIC candidate that fits no model
and needs no populace-fit:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent) reruns seed 0 through the candidate-5a
  matching + splicing and pins the committed artifact's seed-0 splice
  diagnostics, geometry, and battery values to float precision. There is
  NO populace-fit gate on this test -- candidate 5a is fully
  deterministic donor matching + splicing under the repo ``.venv`` -- and
  no populace-fit importorskip, per the frozen spec (the gate seed enters
  only through the split).
* The always-runnable consistency tests touch only the committed artifact
  and ``gates.yaml``: every reported pass/fail recomputes from its own
  stored score against its stored threshold, the stored thresholds equal
  the locked ones in ``gates.yaml``, the gate verdict recomputes from the
  seed-conjunction table, and the splice diagnostics are
  reported-not-gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_splice_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4891949761"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


# --------------------------------------------------------------------------
# Reproduction (needs the staged PSID family files; NO populace-fit needed)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 and match the committed artifact to float precision.

    Candidate 5a is deterministic (no RNG, no model fit, no populace-fit),
    so seed 0 must reproduce exactly under the repo ``.venv``. The gate
    seed enters only through ``split_holdout_train``.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate5a as runner

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

    # Anchors on the full filtered panel, sliced per split.
    all_anchor = runner.anchor_rows(panel)
    candidate, diagnostics = runner.generate_candidate(
        holdout, train, all_anchor
    )

    # Splice diagnostics reproduce exactly (deterministic). JSON coerces
    # integer dict keys to strings, so compare through a round-trip.
    assert json.loads(json.dumps(diagnostics)) == seed0["splice_diagnostics"]

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


@needs_real_family
def test_candidate_is_deterministic_and_pins_the_panel():
    """Two generations agree bit-for-bit; the candidate panel is pinned.

    The candidate holds exactly the holdout persons on exactly their
    observed periods (only earnings generated); the anchor keeps its real
    value. Both are structural guarantees of the frozen spec, so they are
    checked directly (no populace-fit needed).
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    import run_gate1_candidate5a as runner

    panel = runner.load_filtered_panel()
    all_anchor = runner.anchor_rows(panel)
    holdout, train = runner.split_holdout_train(panel, 0)
    # A subsample keeps the test quick while exercising the full path.
    sub = np.sort(holdout.person_id.unique())[:400]
    h = holdout[holdout.person_id.isin(sub)].copy()

    c1, d1 = runner.generate_candidate(h, train, all_anchor)
    c2, d2 = runner.generate_candidate(h, train, all_anchor)
    c1s = c1.sort_values(["person_id", "period"]).reset_index(drop=True)
    c2s = c2.sort_values(["person_id", "period"]).reset_index(drop=True)
    assert (c1s["earnings"].values == c2s["earnings"].values).all()
    assert d1 == d2

    # Panel pin: same rows, same person/period/age/weight as the holdout.
    hs = h.sort_values(["person_id", "period"]).reset_index(drop=True)
    assert (c1s["person_id"].values == hs["person_id"].values).all()
    assert (c1s["period"].values == hs["period"].values).all()
    assert (c1s["age"].values == hs["age"].values).all()
    assert np.allclose(c1s["weight"].values, hs["weight"].values)

    # Anchor kept at its real value.
    anc = all_anchor[all_anchor.person_id.isin(sub)]
    merged = anc.merge(
        c1s, on=["person_id", "period"], suffixes=("_real", "_gen")
    )
    assert np.allclose(merged["earnings_real"], merged["earnings_gen"])


@needs_real_family
def test_nearest_age_fallback_ties_break_to_younger():
    """The nearest-observed-age rule breaks age ties toward the younger."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    import run_gate1_candidate5a as runner

    ages = np.array([25.0, 27.0, 29.0])
    earn = np.array([100.0, 200.0, 300.0])
    # Equidistant between 25 and 27 -> younger (25).
    assert runner.donor_earnings_at_age(ages, earn, 26.0) == (100.0, True)
    # Equidistant between 27 and 29 -> younger (27).
    assert runner.donor_earnings_at_age(ages, earn, 28.0) == (200.0, True)
    # Exact age hit -> no fallback.
    assert runner.donor_earnings_at_age(ages, earn, 27.0) == (200.0, False)
    # Beyond the observed range -> nearest end, fallback.
    assert runner.donor_earnings_at_age(ages, earn, 40.0) == (300.0, True)


# --------------------------------------------------------------------------
# Internal consistency (always runnable; committed files only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_splice.v1"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen-spec issue-comment URL is carried in the artifact."""
    assert _artifact()["spec_registration"] == SPEC_URL


def test_candidate_uses_no_populace_fit():
    """The artifact attests the deterministic, no-populace-fit design."""
    model = _artifact()["model"]
    assert model["stochastic"] is False
    assert model["populace_fit_used"] is False


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


def test_splice_diagnostics_reported_not_gated_and_consistent():
    """The splice diagnostics are recorded per seed and never gated.

    Each seed carries an age-window-widening distribution, a nearest-age
    fallback rate, a scaling-clip rate, and a donor-reuse distribution.
    The top-level context block mirrors the per-seed diagnostics. None of
    them enters the geometry or battery pass/fail; the verdict rule names
    only those two families.
    """
    artifact = _artifact()
    ctx = artifact["splice_diagnostics_context"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in ctx["per_seed"]:
        diag = by_seed[row["seed"]]["splice_diagnostics"]
        assert row["age_window_widening"] == diag["age_window_widening"]
        assert row["nearest_age_fallback_rate"] == pytest.approx(
            diag["nearest_age_fallback"]["rate"], abs=0
        )
        assert row["scaling_clip_rate_over_holdout"] == pytest.approx(
            diag["scaling_clip"]["rate_over_holdout"], abs=0
        )
        assert row["donor_reuse"] == diag["donor_reuse"]

    # Internal arithmetic of each diagnostic is self-consistent.
    for seed in artifact["per_seed"]:
        diag = seed["splice_diagnostics"]
        n = diag["n_holdout_persons"]
        assert n == seed["n_persons"]

        fb = diag["nearest_age_fallback"]
        assert (
            0 <= fb["n_fallback_observations"] <= fb["n_spliced_observations"]
        )
        if fb["n_spliced_observations"] > 0:
            assert fb["rate"] == pytest.approx(
                fb["n_fallback_observations"] / fb["n_spliced_observations"],
                abs=1e-12,
            )

        sc = diag["scaling_clip"]
        assert 0 <= sc["n_clipped_persons"] <= sc["n_scaled_persons"] <= n
        if sc["n_scaled_persons"] > 0:
            assert sc["rate_over_scaled"] == pytest.approx(
                sc["n_clipped_persons"] / sc["n_scaled_persons"], abs=1e-12
            )
        assert sc["clip_bounds"] == [0.2, 5.0]

        reuse = diag["donor_reuse"]
        # Every holdout person has exactly one matched donor, so the
        # reuse counts partition the holdout persons.
        total = sum(
            int(k) * v for k, v in reuse["reuse_count_distribution"].items()
        )
        assert total == n
        assert reuse["n_distinct_donors"] == sum(
            reuse["reuse_count_distribution"].values()
        )

        widen = diag["age_window_widening"]
        assert sum(widen["distribution"].values()) == n

    # The gate rule is the seed-level conjunction over geometry AND
    # battery only; no splice diagnostic is in it.
    assert artifact["verdict"]["rule"] == (
        ">=4/5 seeds geometry AND >=4/5 seeds battery"
    )
