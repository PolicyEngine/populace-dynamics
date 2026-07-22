"""Tests for the gate-1 candidate-5a' segment-splicing run.

Mirrors the prior runs' tests (``test_gate1_qrf_candidate5a`` and its
predecessors), adapted for a DETERMINISTIC candidate that fits no model
and needs no populace-fit:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files or the registered scikit-learn 1.9.0 scoring environment
  are absent) reruns seed 0 through the candidate-5a' segmentation +
  splicing and pins the committed artifact's seed-0 splice diagnostics,
  geometry, and battery values to float precision. There is NO populace-fit
  gate on this test -- candidate 5a' is fully deterministic segment matching
  + splicing -- but its C2ST scorer is scikit-learn-version-sensitive.
* :func:`test_segmentation_and_cascade_on_synthetic_pool` (always
  runnable; no PSID, no populace-fit) drives the segmentation, boundary,
  period-indexed match, and the widen-then-shorten fallback cascade on a
  hand-built donor pool, checking each frozen rule directly.
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
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_splice_v2.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)
needs_registered_scoring_environment = pytest.mark.skipif(
    sklearn.__version__ != "1.9.0",
    reason=(
        "exact Gate-1 scoring reproduction requires the registered "
        "scikit-learn 1.9.0 environment "
        f"(running {sklearn.__version__})"
    ),
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4892604375"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate5a2 as runner

    return runner


def _earn(cand, period: int) -> float:
    """The generated earnings at ``period`` for a one-person candidate."""
    return float(cand.loc[cand.period == period, "earnings"].iloc[0])


# --------------------------------------------------------------------------
# Reproduction (needs staged PSID + registered scorer; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
@needs_registered_scoring_environment
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 and match the committed artifact to float precision.

    Candidate 5a' is deterministic (no RNG, no model fit, no
    populace-fit), so seed 0 must reproduce exactly under the repo
    ``.venv``. The gate seed enters only through ``split_holdout_train``.
    """
    runner = _import_runner()

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
    import numpy as np

    runner = _import_runner()

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
    assert json.loads(json.dumps(d1)) == json.loads(json.dumps(d2))

    # Panel pin: same rows, same person/period/age/weight as the holdout.
    hs = h.sort_values(["person_id", "period"]).reset_index(drop=True)
    assert (c1s["person_id"].values == hs["person_id"].values).all()
    assert (c1s["period"].values == hs["period"].values).all()
    assert (c1s["age"].values == hs["age"].values).all()
    assert np.allclose(c1s["weight"].values, hs["weight"].values)
    # No negative earnings are ever emitted (donor earnings are >= 0 and
    # the scale factor is positive).
    assert (c1s["earnings"].values >= 0).all()

    # Anchor kept at its real value.
    anc = all_anchor[all_anchor.person_id.isin(sub)]
    merged = anc.merge(
        c1s, on=["person_id", "period"], suffixes=("_real", "_gen")
    )
    assert np.allclose(merged["earnings_real"], merged["earnings_gen"])


# --------------------------------------------------------------------------
# Segmentation + boundary + cascade (always runnable; synthetic pool)
# --------------------------------------------------------------------------
def test_segmentation_boundary_and_scaling_backward_chain():
    """A full-donor case: one segment, boundary = anchor, scale to it.

    Observed periods [2000, 2002, 2004, 2006] with anchor 2006. The three
    pre-anchor periods form ONE segment [2000, 2002, 2004] whose boundary
    is the anchor 2006. A single donor covers all four periods; the
    segment's donor earnings are scaled by anchor_value /
    donor_earnings_at_2006.
    """
    import pandas as pd

    runner = _import_runner()

    def rows(recs):
        return pd.DataFrame(
            recs,
            columns=["person_id", "period", "earnings", "age", "weight"],
        )

    # Donor covers 2000..2006 with earnings 90/95/100/105 at ages 54..60.
    train = rows(
        [
            (1, 2000, 90.0, 54, 1.0),
            (1, 2002, 95.0, 56, 1.0),
            (1, 2004, 100.0, 58, 1.0),
            (1, 2006, 105.0, 60, 1.0),
        ]
    )
    holdout = rows(
        [
            (9, 2000, 500.0, 54, 1.0),
            (9, 2002, 600.0, 56, 1.0),
            (9, 2004, 700.0, 58, 1.0),
            (9, 2006, 210.0, 60, 1.0),  # anchor kept real
        ]
    )
    all_anchor = runner.anchor_rows(holdout)
    cand, diag = runner.generate_candidate(holdout, train, all_anchor)
    cand = cand.sort_values("period").reset_index(drop=True)

    # One segment of length 3; boundary is the anchor.
    assert diag["n_segments"] == 1
    assert diag["segment_length"]["distribution"] == {3: 1}
    assert diag["segment_count"]["distribution"] == {1: 1}
    # scale = anchor 210 / donor@2006 105 = 2.0 (inside [0.2, 5], not
    # clipped).
    assert diag["scaling_clip"]["n_segments_scaled"] == 1
    assert diag["scaling_clip"]["n_segments_clipped"] == 0
    # anchor unchanged; pre-anchor periods scaled by 2.0.
    assert _earn(cand, 2006) == 210.0
    assert _earn(cand, 2004) == 200.0
    assert _earn(cand, 2002) == 190.0
    assert _earn(cand, 2000) == 180.0


def test_scaling_clip_upper_bound_binds():
    """A donor far below the boundary forces the ratio to clip at 5."""
    import pandas as pd

    runner = _import_runner()

    train = pd.DataFrame(
        [
            (1, 2004, 100.0, 58, 1.0),
            (1, 2006, 100.0, 60, 1.0),
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    holdout = pd.DataFrame(
        [
            (9, 2004, 700.0, 58, 1.0),
            (9, 2006, 1000.0, 60, 1.0),  # anchor; ratio 1000/100 = 10 -> 5
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    all_anchor = runner.anchor_rows(holdout)
    cand, diag = runner.generate_candidate(holdout, train, all_anchor)
    cand = cand.sort_values("period").reset_index(drop=True)
    assert diag["scaling_clip"]["n_segments_clipped"] == 1
    assert diag["scaling_clip"]["raw_ratio_max"] == pytest.approx(10.0)
    # 100 * clip(10, 0.2, 5) = 500.
    assert _earn(cand, 2004) == 500.0


def test_widen_then_shorten_cascade():
    """No 3-period donor forces a shorten; the boundary is preserved.

    Observed [2000, 2002, 2004, 2006], anchor 2006. No train donor is
    observed at all of 2000/2002/2004/2006 together, so the length-3
    segment shortens to [2002, 2004] (a donor covers those plus the
    boundary 2006). The dropped 2000 re-groups into a following one-period
    segment [2000] whose boundary is the just-generated 2002; no donor
    covers 2000+2002, so that one-period segment stays unmatched (keeps
    the holdout value) -- which the diagnostics record.
    """
    import pandas as pd

    runner = _import_runner()

    train = pd.DataFrame(
        [
            # Donor B covers 2002, 2004, 2006 (not 2000).
            (2, 2002, 100.0, 56, 1.0),
            (2, 2004, 110.0, 58, 1.0),
            (2, 2006, 120.0, 60, 1.0),
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    holdout = pd.DataFrame(
        [
            (9, 2000, 500.0, 54, 1.0),
            (9, 2002, 600.0, 56, 1.0),
            (9, 2004, 700.0, 58, 1.0),
            (9, 2006, 1000.0, 60, 1.0),  # anchor
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    all_anchor = runner.anchor_rows(holdout)
    cand, diag = runner.generate_candidate(holdout, train, all_anchor)
    cand = cand.sort_values("period").reset_index(drop=True)

    # One shorten event (3 -> 2); two segments result (a length-2 and a
    # length-1); the length-1 segment is unmatched.
    assert diag["segment_shortening"]["shorten_events"] == 1
    assert diag["segment_shortening"]["n_unmatched_one_period_segments"] == 1
    assert diag["segment_length"]["distribution"] == {1: 1, 2: 1}
    # [2002, 2004] scaled to boundary 1000 via donor@2006 120: ratio 8.33
    # -> clip 5. 100*5 = 500 at 2002, 110*5 = 550 at 2004.
    assert _earn(cand, 2004) == 550.0
    assert _earn(cand, 2002) == 500.0
    # 2000 kept its holdout value (no period-matched donor).
    assert _earn(cand, 2000) == 500.0
    # anchor kept real.
    assert _earn(cand, 2006) == 1000.0


def test_age_window_widens_before_shortening():
    """The +/-2 window widens (up to +/-10) before any shortening.

    A donor covers all segment periods but sits 6 years off the target's
    boundary age; the match must widen the window to +/-6 (3 steps) rather
    than shorten.
    """
    import pandas as pd

    runner = _import_runner()

    # Donor covers 2004, 2006 but boundary (2006) age is 66 vs target 60.
    train = pd.DataFrame(
        [
            (1, 2004, 100.0, 64, 1.0),
            (1, 2006, 105.0, 66, 1.0),
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    holdout = pd.DataFrame(
        [
            (9, 2004, 700.0, 58, 1.0),
            (9, 2006, 210.0, 60, 1.0),  # anchor; boundary age 60
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    all_anchor = runner.anchor_rows(holdout)
    cand, diag = runner.generate_candidate(holdout, train, all_anchor)
    # |66 - 60| = 6 -> widen from 2 to 6 is 2 additional steps (4, 6).
    assert diag["age_window_widening"]["n_widened_segments"] == 1
    assert diag["age_window_widening"]["max_widen_steps"] == 2
    assert diag["segment_shortening"]["shorten_events"] == 0


def test_donor_zeros_and_zero_boundary_copy_unscaled():
    """A zero boundary copies the donor unscaled; donor zeros stay zero."""
    import pandas as pd

    runner = _import_runner()

    train = pd.DataFrame(
        [
            (1, 2004, 0.0, 58, 1.0),  # donor zero at 2004
            (1, 2006, 50.0, 60, 1.0),
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    holdout = pd.DataFrame(
        [
            (9, 2004, 700.0, 58, 1.0),
            (9, 2006, 0.0, 60, 1.0),  # anchor is zero -> boundary value 0
        ],
        columns=["person_id", "period", "earnings", "age", "weight"],
    )
    all_anchor = runner.anchor_rows(holdout)
    cand, diag = runner.generate_candidate(holdout, train, all_anchor)
    cand = cand.sort_values("period").reset_index(drop=True)
    # Zero boundary -> unscaled; donor value at 2004 is 0 -> copies 0.
    assert diag["scaling_clip"]["n_segments_unscaled"] == 1
    assert diag["scaling_clip"]["n_segments_scaled"] == 0
    assert _earn(cand, 2004) == 0.0
    assert _earn(cand, 2006) == 0.0


# --------------------------------------------------------------------------
# Internal consistency (always runnable; committed files only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_splice.v2"
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

    Each seed carries segment count/length distributions, age-window
    widening and segment shortening rates, a scaling-clip rate, a
    boundary match-error distribution, and a donor-reuse distribution.
    The top-level context block mirrors the per-seed diagnostics. None of
    them enters the geometry or battery pass/fail; the verdict rule names
    only those two families.
    """
    artifact = _artifact()
    ctx = artifact["splice_diagnostics_context"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in ctx["per_seed"]:
        diag = by_seed[row["seed"]]["splice_diagnostics"]
        assert row["segment_count"] == diag["segment_count"]
        assert row["segment_length"] == diag["segment_length"]
        assert row["age_window_widening"] == diag["age_window_widening"]
        assert row["segment_shortening"] == diag["segment_shortening"]
        assert row["scaling_clip_rate_over_segments"] == pytest.approx(
            diag["scaling_clip"]["rate_over_segments"], abs=0
        )
        assert row["boundary_match_error"] == diag["boundary_match_error"]
        assert row["donor_reuse"] == diag["donor_reuse"]

    # Internal arithmetic of each diagnostic is self-consistent.
    for seed in artifact["per_seed"]:
        diag = seed["splice_diagnostics"]
        n = diag["n_holdout_persons"]
        assert n == seed["n_persons"]

        # Segment-count distribution partitions the holdout persons.
        sc = diag["segment_count"]
        assert sum(sc["distribution"].values()) == n

        # Segment-length distribution partitions the segments.
        sl = diag["segment_length"]
        assert sum(sl["distribution"].values()) == diag["n_segments"]
        # No segment exceeds the frozen cap of 3.
        assert all(int(k) <= 3 for k in sl["distribution"])
        assert all(int(k) >= 1 for k in sl["distribution"])

        # Widening / shortening rates are over the segment count.
        aw = diag["age_window_widening"]
        assert aw["n_segments"] == diag["n_segments"]
        if diag["n_segments"] > 0:
            assert aw["rate_over_segments"] == pytest.approx(
                aw["n_widened_segments"] / diag["n_segments"], abs=1e-12
            )
        ss = diag["segment_shortening"]
        assert ss["n_segments"] == diag["n_segments"]

        # Scaling: clipped <= scaled <= segments; unscaled accounts for
        # the rest of the matched segments.
        sk = diag["scaling_clip"]
        assert (
            0
            <= sk["n_segments_clipped"]
            <= sk["n_segments_scaled"]
            <= diag["n_segments"]
        )
        if sk["n_segments_scaled"] > 0:
            assert sk["rate_over_scaled"] == pytest.approx(
                sk["n_segments_clipped"] / sk["n_segments_scaled"], abs=1e-12
            )
        assert sk["clip_bounds"] == [0.2, 5.0]

        # Donor reuse: the holdouts-per-donor distribution and the
        # distinct-donors-per-person distribution count the SAME
        # person-donor incidences, and the distinct-donor count matches
        # the holdouts-per-donor support.
        reuse = diag["donor_reuse"]
        hpd = reuse["holdouts_per_donor"]["distribution"]
        assert reuse["n_distinct_donors"] == sum(hpd.values())
        inc_donor_side = sum(int(k) * v for k, v in hpd.items())
        dpp = reuse["distinct_donors_per_person"]["distribution"]
        inc_person_side = sum(int(k) * v for k, v in dpp.items())
        assert inc_donor_side == inc_person_side
        # The distinct-donors-per-person distribution partitions holdouts.
        assert sum(dpp.values()) == n

    # The gate rule is the seed-level conjunction over geometry AND
    # battery only; no splice diagnostic is in it.
    assert artifact["verdict"]["rule"] == (
        ">=4/5 seeds geometry AND >=4/5 seeds battery"
    )
