"""Tests for the gate-1 candidate-7 k-NN conditional-rank-bootstrap run.

Mirrors the prior runs' tests (``test_gate1_qrf_candidate6`` and its
predecessors), adapted for the continuous k-NN bootstrap candidate:

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent, and ``importorskip("populace.fit")`` because the
  participation gate needs it) reruns seed 0 through the candidate-7 donor
  pools + backward k-NN generation and pins the committed artifact's seed-0
  geometry and battery values to float precision. It is run live in the
  dedicated gate venv before the artifact is committed.
* :func:`test_donor_pools_on_synthetic_pool` (always runnable; no PSID, no
  populace-fit) drives the pair/triple/re-entry pool construction and the
  pinned tie-break sort on a hand-built train panel, checking each frozen
  rule directly.
* :func:`test_knn_draw_weighted_and_tie_break` (always runnable) pins the
  k-NN weighted single-record draw and its k=25 nearest selection with the
  record-order tie-break.
* The always-runnable consistency tests touch only the committed artifact
  and ``gates.yaml``: the artifact schema and spec URL are recorded, the
  battery reference reproduces exactly, every reported pass/fail recomputes
  from its own stored score against its stored threshold, the stored
  thresholds equal the locked ones in ``gates.yaml``, the zero-persistence
  identity holds, the gate verdict recomputes from the seed-conjunction
  table, and the k-NN diagnostics are reported-not-gated (usage shares and
  corner masses in range).
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
ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4896132094"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate7 as runner

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
    # The donor pools are deterministic given the split: pool sizes
    # reproduce exactly.
    assert result["pools"]["n_pairs"] == seed0["pools"]["n_pairs"]
    assert result["pools"]["n_triples"] == seed0["pools"]["n_triples"]
    assert result["pools"]["n_reentry"] == seed0["pools"]["n_reentry"]


# --------------------------------------------------------------------------
# Donor pools on a hand-built panel (always runnable; no populace-fit)
# --------------------------------------------------------------------------
def test_donor_pools_on_synthetic_pool():
    """Pair / triple / re-entry pool construction and the pinned sort."""
    runner = _import_runner()

    # Persons designed to cover each pool exactly:
    # - person 1: three consecutive positives 2018/2020/2022 (anchor 2022)
    #   -> two backward pairs (2018->2020, 2020->2022) and one triple
    #   (u_prev=2018, u_next=2020, u_next2=2022).
    # - person 2: positive 2018, zero 2020 (anchor zero) -> one re-entry
    #   pair; its backward pair is NOT a positive pair.
    # - persons 3, 4: one positive pair each (no triple, no re-entry).
    rows = [
        (1, 2018, 30000.0, 40, 1.0),
        (1, 2020, 40000.0, 42, 1.0),
        (1, 2022, 50000.0, 44, 1.0),
        (2, 2018, 20000.0, 40, 2.0),
        (2, 2020, 0.0, 42, 2.0),
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
    pools = runner.build_donor_pools(panel, all_anchor, marginals)

    # Positive backward pairs: person 1 gives two (2018->2020, 2020->2022);
    # persons 3, 4 give one each; person 2's pair is a re-entry, not a pair.
    assert pools["n_pairs"] == 4
    # Triples: only person 1 has three consecutive positives -> one triple
    # (u_prev at 2018, u_next at 2020, u_next2 at 2022).
    assert pools["n_triples"] == 1
    # Re-entry: person 2 (later zero, earlier positive).
    assert pools["n_reentry"] == 1

    # Every pool carries u_A and weight per record.
    for name in ("pairs", "triples", "reentry"):
        p = pools[name]
        assert len(p["u_A"]) == len(p["u_prev"])
        assert len(p["weight"]) == len(p["u_prev"])

    # The single triple's u_next2 is a valid rank in [0.001, 0.999].
    tri = pools["triples"]
    assert 0.001 - 1e-9 <= float(tri["u_next2"][0]) <= 0.999 + 1e-9

    # The re-entry record carries person 2's earlier (2018) weight 2.0.
    assert float(pools["reentry"]["weight"][0]) == pytest.approx(2.0, abs=0)

    # Pools are pinned in ascending (person_id, period_prev) order.
    for name in ("pairs", "triples", "reentry"):
        p = pools[name]
        key = p["person_id"].astype(np.int64) * 100000 + p[
            "period_prev"
        ].astype(np.int64)
        assert np.all(np.diff(key) >= 0), f"{name} not in pinned order"


def test_knn_draw_weighted_and_tie_break():
    """The k-NN draw takes the k nearest and draws one by weight.

    With fewer than ``k`` donors the draw uses all of them; the weighted
    inverse-CDF picks the record whose cumulative weight first reaches the
    uniform draw. A distance array with a clear ordering pins the nearest
    selection; equal-distance donors resolve by record (index) order.
    """
    runner = _import_runner()

    # Four donors; distances chosen so donor order by distance is 2,0,3,1.
    dist = np.array([[0.20, 0.40, 0.05, 0.30]], dtype=np.float64)
    weight = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    u_prev = np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float64)
    # k defaults to 25 but there are only 4 donors -> all four are neighbors.
    # u_draw = 0.0 -> the first donor in cumulative order (nearest-first?
    # no: the draw order follows the SELECTED order, which is by
    # (distance, index)); cumulative weight reaches 0 at the first selected.
    drawn, kth = runner._knn_draw(dist, weight, u_prev, np.array([0.0]))
    # Selected order by (distance, index): [2, 0, 3, 1]; u_draw=0 picks the
    # first -> donor 2 -> u_prev 0.33.
    assert float(drawn[0]) == pytest.approx(0.33, abs=0)
    # k-th (last selected) distance is the max among the four = 0.40.
    assert float(kth[0]) == pytest.approx(0.40, abs=0)

    # Weighted draw: donor 2 has weight 100, others 1; u_draw just above 0
    # still lands on donor 2 (its cumulative mass dominates).
    weight_w = np.array([1.0, 1.0, 100.0, 1.0], dtype=np.float64)
    drawn_w, _ = runner._knn_draw(dist, weight_w, u_prev, np.array([0.5]))
    assert float(drawn_w[0]) == pytest.approx(0.33, abs=0)

    # Tie-break by record order: two donors at equal distance resolve by
    # index (the pinned pool order). Here donors 0 and 1 tie at 0.10; the
    # nearest is donor 2 (0.05), then donor 0 then donor 1.
    dist_tie = np.array([[0.10, 0.10, 0.05, 0.30]], dtype=np.float64)
    drawn_t, _ = runner._knn_draw(dist_tie, weight, u_prev, np.array([0.34]))
    # Selected order [2, 0, 1, 3] (0 before 1 by index); cumulative
    # 0.25/0.50/0.75/1.0; u=0.34 -> second selected -> donor 0 -> 0.11.
    assert float(drawn_t[0]) == pytest.approx(0.11, abs=0)


# --------------------------------------------------------------------------
# Always-runnable consistency tests (artifact + gates.yaml only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_rank_knn.v1"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen-spec issue-comment URL is carried in the artifact."""
    assert _artifact()["spec_registration"] == SPEC_URL


def test_no_calibration_stage_declared():
    """Candidate 7 has no calibration stage (zero free parameters)."""
    artifact = _artifact()
    assert "calibration" in artifact["model"]
    assert artifact["model"]["calibration"].startswith("none")
    # No SMM block (that was candidate 5b) and no kernel block (that was
    # candidate 6); a donor_pools + knn block instead.
    assert "smm" not in artifact["model"]
    assert "kernel" not in artifact["model"]
    assert "donor_pools" in artifact["model"]
    assert "knn" in artifact["model"]


def test_knn_constants_recorded():
    """The frozen k and distance weights are carried in the artifact."""
    knn = _artifact()["model"]["knn"]
    assert knn["k"] == 25
    assert knn["weights"] == {"w_next": 1.0, "w_next2": 0.5, "w_anchor": 0.25}


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


def test_knn_diagnostics_reported_not_gated():
    """Reported-not-gated: usage shares and corner masses are in range.

    The triple/pair/re-entry usage share sums the three draw counts; the
    drawn corner masses by anchor quintile are shares in ``[0, 1]``; the
    neighbor-distance percentiles are non-negative and monotone.
    """
    artifact = _artifact()
    for seed in artifact["knn_context"]["per_seed"]:
        usage = seed["triple_pair_usage"]
        assert usage["n_triple_draws"] >= 0
        assert usage["n_pair_draws"] >= 0
        assert usage["n_reentry_draws"] >= 0
        assert 0.0 <= usage["triple_share_of_positive"] <= 1.0

        for block in seed["drawn_corner_mass_by_anchor_quintile"].values():
            assert block["n"] >= 0
            assert 0.0 <= block["bottom_share"] <= 1.0
            assert 0.0 <= block["top_share"] <= 1.0

        nd = seed["neighbor_distance_distribution"]
        if nd:
            pcts = [nd[f"p{p}"] for p in (0, 10, 25, 50, 75, 90, 100)]
            assert all(v >= 0 for v in pcts)
            assert pcts == sorted(pcts), "neighbor-distance pcts not monotone"


def test_donor_reuse_reported_not_gated():
    """Reported-not-gated: donor-reuse record counts are positive."""
    for seed in _artifact()["knn_context"]["per_seed"]:
        reuse = seed["donor_reuse"]
        assert reuse["n_pair_records"] > 0
        assert reuse["n_triple_records"] > 0
        assert reuse["n_reentry_records"] > 0
        for key in (
            "pair_draws_per_record",
            "triple_draws_per_record",
            "reentry_draws_per_record",
        ):
            assert reuse[key] >= 0.0


def test_candidate_panel_pin_metadata_consistent():
    """Each seed's window counts are positive and pairs >= runs.

    The candidate panel holds exactly the holdout persons on exactly their
    observed periods, so its projected window counts are a property of the
    holdout support; the stored counts are recorded per view and must be
    positive and consistent with the person count.
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
        assert (
            seed["n_windows"]["psid_family_earnings_pairs"]
            >= seed["n_windows"]["psid_family_earnings_runs"]
        )
