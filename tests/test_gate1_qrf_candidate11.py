"""Tests for the gate-1 candidate-11 (run 13) amendment-2 re-registration.

Candidate 11's spec is BYTE-IDENTICAL to candidate 10 (the inner sweep's
V1-lam0.1 variant at outer scale): no modeling constant moves. The ONLY change
is the gate estimator, ratified amendment 2 (PR #67/#69): the pairs-view
``c2st_auc`` line moved from per-seed on the five locked seeds to a 20-seed
MEAN rule (mean over seeds 0-19 <= 0.53) plus a per-seed CAP (each of the 20
<= 0.554), with the per-seed pairs c2st retired from the geometry conjunction
(``per_seed_rule_superseded``). These tests target exactly the amended scoring
and the reproduction discipline:

* the artifact-consistency suite (always runnable; touches only the committed
  artifact, ``gates.yaml``, and the committed baselines): the schema and spec
  URL, the exact battery-reference reproduction, every stored geometry /
  battery / benefit-space pass recomputes from its own stored score against its
  stored (locked) threshold, the stored per-seed thresholds match the locked
  ``gates.yaml`` (popping ``reported_not_gated`` + ``per_seed_rule_superseded``),
  the pairs c2st is NOT gated per-seed but its score is still recorded, the
  20-seed mean/cap recompute from the 20 stored values against the gates.yaml
  ``value_max``, the AMENDED verdict recomputes from the seed conjunction PLUS
  the pooled Q0 gate PLUS the mean rule PLUS the cap, and the reproduction block
  is internally consistent (its stored max-deviations recompute from the stored
  per-seed values against the committed baselines);
* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID family
  files are absent, and ``importorskip("populace.fit")``) reruns seed 0 through
  the candidate-10 machinery the runner reuses and pins the committed v5 seed-0
  values to float precision -- run live in the dedicated gate venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v5.json"
RUN12_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v4.json"
DIAGNOSTICS_ARTIFACT = ROOT / "runs" / "c10_diagnostics_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4905323933"
)
PAIRS_VIEW = "psid_family_earnings_pairs"
ALL_SEEDS = list(range(20))
EXACT_ATOL = 1e-12


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _import_runner():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_gate1_candidate11 as runner

    return runner


# --------------------------------------------------------------------------
# Reproduction helpers -- MIRROR the runner's deviation logic exactly so the
# test recomputes the stored max-deviations from the stored per-seed values.
# --------------------------------------------------------------------------
def _abs_dev(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(float(a) - float(b))


def _block_max_dev(mine: dict, ref: dict) -> float:
    keys = set(mine) | set(ref)
    return max((_abs_dev(mine.get(k), ref.get(k)) for k in keys), default=0.0)


def _benefit_metric_values(seed_result: dict) -> dict:
    out: dict = {}
    for name, chk in (seed_result.get("benefit_space_checks") or {}).items():
        out[name] = chk.get("value")
    bs = seed_result.get("benefit_space")
    if bs is not None:
        q0 = bs["by_anchor_quintile"]["quintiles"].get("Q0", {})
        out["q0_mean_pct_diff"] = (
            q0["distribution"]["mean"]["pct_diff"]
            if q0.get("n_persons")
            else None
        )
    return out


def _geometry_score_values(seed_result: dict) -> dict:
    out: dict = {}
    for vname, block in seed_result["geometry"].items():
        for metric, score in block["scores"].items():
            out[f"{vname}.{metric}"] = score
    return out


# --------------------------------------------------------------------------
# Reproduction (needs the staged PSID family files AND populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun seed 0 and match the committed v5 artifact to float precision.

    Candidate 11 reuses candidate 10's ``run_seed`` verbatim for the locked
    seeds, so this pins the locked-seed scoring path (fixed lambda, geometry,
    battery, Q0 benefit) against the committed v5 seed-0 block.
    """
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_downstream_relevance as ds
    import run_gate1_candidate10 as c10

    from populace_dynamics.ss.params import load_ssa_parameters

    artifact = _artifact()
    seed0 = next(s for s in artifact["per_seed"] if s["seed"] == 0)
    thresholds = c10.load_gate1_thresholds()
    views_cfg = thresholds["views"]
    benefit_metrics_cfg = thresholds["benefit_space"]["metrics"]
    battery_reference = json.loads(
        (c10.ROOT / c10.BATTERY_REFERENCE_RUN).read_text()
    )["battery_reference"]
    panel = c10.load_filtered_panel()
    all_anchor = c10.anchor_rows(panel)
    params = load_ssa_parameters()
    cutpoints = ds.anchor_quintile_cutpoints(all_anchor)
    view_specs = {
        "psid_family_earnings_pairs": c10.build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": c10.build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }
    result = c10.run_seed(
        0,
        panel,
        all_anchor,
        view_specs,
        views_cfg,
        battery_reference,
        battery_tol,
        benefit_metrics_cfg,
        params,
        cutpoints,
        False,
    )
    assert result["lambda"] == 0.1
    assert result["lambda"] == seed0["lambda"]
    for view, block in seed0["geometry"].items():
        for metric, stored in block["scores"].items():
            assert result["geometry"][view]["scores"][metric] == pytest.approx(
                stored, abs=1e-12
            ), f"{view}.{metric}"
    for stat, stored in seed0["battery_values"].items():
        assert result["battery_values"][stat] == pytest.approx(
            stored, abs=1e-12
        ), stat
    for key in ("n_pairs", "n_triples", "n_reentry", "n_reentry_q0"):
        assert result["pools"][key] == seed0["pools"][key], key
    q0_stored = seed0["benefit_space"]["by_anchor_quintile"]["quintiles"][
        "Q0"
    ]["distribution"]["mean"]["pct_diff"]
    q0_got = result["benefit_space"]["by_anchor_quintile"]["quintiles"]["Q0"][
        "distribution"
    ]["mean"]["pct_diff"]
    assert q0_got == pytest.approx(q0_stored, abs=1e-9)


# --------------------------------------------------------------------------
# Always-runnable: runner import + module constants
# --------------------------------------------------------------------------
def test_runner_module_constants():
    """The runner imports under the plain venv and exposes the seed sets."""
    runner = _import_runner()
    assert runner.LOCKED_SEEDS == (0, 1, 2, 3, 4)
    assert runner.EXTENSION_SEEDS == tuple(range(5, 20))
    assert runner.ALL_SEEDS == tuple(range(20))
    assert runner.ARTIFACT_SCHEMA_VERSION == "gate1_rank_knn_v5"
    assert "4905323933" in runner.SPEC_REGISTRATION
    # The k-NN draw is candidate 7's, reached through the reused c10 module.
    dist = np.array([[0.20, 0.40, 0.05, 0.30]], dtype=np.float64)
    weight = np.ones(4, dtype=np.float64)
    u_prev = np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float64)
    drawn, kth = runner.c10._knn_draw(dist, weight, u_prev, np.array([0.0]))
    assert float(drawn[0]) == pytest.approx(0.33, abs=0)
    assert float(kth[0]) == pytest.approx(0.40, abs=0)


# --------------------------------------------------------------------------
# Always-runnable consistency tests (artifact + gates.yaml + baselines only)
# --------------------------------------------------------------------------
def test_artifact_present_and_locked():
    artifact = _artifact()
    assert artifact["schema_version"] == "gate1_rank_knn_v5"
    assert artifact["run"] == "gate1_rank_knn_v5"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The candidate-11 frozen-spec issue-comment URL is carried; provenance."""
    artifact = _artifact()
    assert artifact["spec_registration"] == SPEC_URL
    # The re-registered candidate-10 spec and the machinery provenance.
    assert "4902561460" in artifact["candidate10_spec_registration"]
    assert "4896132094" in artifact["base_registration"]
    assert "4897723604" in artifact["uw_registration"]
    assert "4898825218" in artifact["c9_registration"]


def test_amendment2_variant_and_forecast_recorded():
    """The artifact declares amendment 2 (mean rule + cap) and the forecast."""
    artifact = _artifact()
    variant = artifact["gate_variant"]
    assert "amendment 2" in variant
    assert "mean" in variant.lower() and "cap" in variant.lower()
    fc = artifact["pre_registered_forecast"]
    assert fc["p_pass"] == 0.97
    assert fc["registration"] == SPEC_URL
    # The spec is byte-identical to candidate 10; no dial moved.
    assert artifact["model"]["spec_identical_to_candidate_10"] is True
    assert artifact["model"]["modeling_constants_changed"] == "none"


def test_fixed_lambda_no_calibration():
    """Every locked seed carries the FIXED lambda 0.1; no calibration stage."""
    artifact = _artifact()
    model = artifact["model"]
    assert "none in this run" in model["calibration"]
    lam_by_seed = artifact["lambda_by_seed"]
    for s in artifact["per_seed"]:
        assert s["lambda"] == 0.1
        assert lam_by_seed[str(s["seed"])] == 0.1
        assert "lambda_calibration" not in s
    assert artifact["knn_context"]["lambda"] == 0.1
    assert artifact["knn_context"]["lambda_fixed"] is True
    knn = model["knn"]
    assert "0.1*u_w" in knn["distance_pairs_nonq0"]
    assert knn["distance_pairs_q0"] == "|u_next - v1| + 0.25|u_A - a|"
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
    """Every stored geometry threshold equals the locked one (both pops).

    Pops BOTH ``reported_not_gated`` (the runs-view c2st demotion, amendment 1)
    and ``per_seed_rule_superseded`` (the pairs-view c2st retirement, amendment
    2) before comparing -- the current pop pattern on origin/master.
    """
    artifact = _artifact()
    views_cfg = _gate1_thresholds()["views"]
    for seed in artifact["per_seed"]:
        for vname, view in seed["geometry"].items():
            stored = dict(view["thresholds"])
            demoted = views_cfg[vname].get(
                "reported_not_gated", []
            ) + views_cfg[vname].get("per_seed_rule_superseded", [])
            for metric in demoted:
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


def test_pairs_c2st_not_gated_per_seed_but_scored():
    """Amendment 2: pairs c2st is NOT a per-seed geometry gate, but scored.

    No locked seed's pairs geometry ``checks`` or ``thresholds`` carries a
    c2st entry (it was retired to the 20-seed mean rule + cap), yet every seed
    still records the pairs ``c2st_auc`` score (it feeds the 20-seed vector).
    """
    for seed in _artifact()["per_seed"]:
        pairs = seed["geometry"][PAIRS_VIEW]
        assert not any("c2st" in k for k in pairs["checks"])
        assert not any("c2st" in k for k in pairs["thresholds"])
        assert "c2st_auc" in pairs["scores"]
        assert 0.0 <= pairs["scores"]["c2st_auc"] <= 1.0


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


def test_every_geometry_threshold_pass_recomputes_from_stored_score():
    """Recompute each locked-geometry pass/fail from its stored score."""
    artifact = _artifact()
    for seed in artifact["per_seed"]:
        thresholds_pass = True
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
            thresholds_pass = thresholds_pass and view_pass
        assert seed["geometry_thresholds_pass"] == thresholds_pass


def test_benefit_space_per_seed_recomputes_from_stored_values():
    """Recompute each per-seed benefit-space gate from its stored value."""
    artifact = _artifact()
    bmetrics = _gate1_thresholds()["benefit_space"]["metrics"]
    mean_thr = float(bmetrics["abs_mean_pct_diff_max"]["value"])
    med_thr = float(bmetrics["abs_median_pct_diff_max"]["value"])
    dec_thr = float(bmetrics["decile_pct_diff_max"]["value"])
    dec_gated = bmetrics["decile_pct_diff_max"]["deciles_gated"]
    ks_thr = float(bmetrics["weighted_ks_max"]["value"])
    for seed in artifact["per_seed"]:
        checks = seed.get("benefit_space_checks")
        if checks is None:
            pytest.skip("benefit-space block absent (SSA oracle unavailable)")
        assert checks["abs_mean_pct_diff"]["threshold"] == pytest.approx(
            mean_thr, abs=0
        )
        assert checks["abs_median_pct_diff"]["threshold"] == pytest.approx(
            med_thr, abs=0
        )
        assert checks["weighted_ks"]["threshold"] == pytest.approx(
            ks_thr, abs=0
        )
        for dkey in dec_gated:
            assert checks[f"decile_{dkey}_pct_diff"][
                "threshold"
            ] == pytest.approx(dec_thr, abs=0)
        recomputed_all = True
        for name, chk in checks.items():
            val = chk["value"]
            thr = chk["threshold"]
            if val is None:
                recomputed = False
            elif chk["comparison"] == "|.| <=":
                recomputed = abs(float(val)) <= thr
            else:
                recomputed = float(val) <= thr
            assert recomputed == chk["pass"], (
                f"seed {seed['seed']} benefit_space {name}: "
                f"stored={chk['pass']} recomputed={recomputed}"
            )
            recomputed_all = recomputed_all and chk["pass"]
        assert seed["benefit_space_seed_pass"] == recomputed_all


def test_amended_geometry_verdict_conjoins_benefit_space():
    """Each seed's amended geometry verdict = thresholds AND benefit-space."""
    for seed in _artifact()["per_seed"]:
        bs_pass = seed.get("benefit_space_seed_pass")
        if bs_pass is None:
            assert seed["geometry_pass"] == seed["geometry_thresholds_pass"]
        else:
            assert seed["geometry_pass"] == (
                seed["geometry_thresholds_pass"] and bs_pass
            )


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


def test_pooled_q0_gate_recomputes_from_per_seed():
    """The pooled Q0 gate recomputes from the per-seed Q0 means."""
    artifact = _artifact()
    pooled = artifact["benefit_space_gated"]["pooled_q0_gate"]
    thr = float(
        _gate1_thresholds()["benefit_space"]["metrics"][
            "abs_q0_mean_pct_diff_max"
        ]["value"]
    )
    if pooled["pooled_q0_mean_pct_diff"] is None:
        pytest.skip("pooled Q0 unavailable (SSA oracle absent)")
    assert pooled["threshold"] == pytest.approx(thr, abs=0)
    vals = [
        r["q0_mean_pct_diff"]
        for r in pooled["per_seed_q0"]
        if r["q0_mean_pct_diff"] is not None
    ]
    recomputed = float(np.mean(vals)) if vals else None
    assert recomputed == pytest.approx(
        pooled["pooled_q0_mean_pct_diff"], abs=1e-9
    )
    assert pooled["pooled_q0_pass"] == (abs(recomputed) <= thr)


def test_pairs_c2st_seed_set_twenty_values():
    """The 20-seed pairs c2st vector: 20 seeds, sources, mean/max recompute."""
    artifact = _artifact()
    block = artifact["pairs_c2st_seed_set"]
    rows = sorted(block["per_seed"], key=lambda e: e["seed"])
    assert [r["seed"] for r in rows] == ALL_SEEDS
    for r in rows:
        assert 0.0 <= r["c2st_auc"] <= 1.0
        assert r["source"] == (
            "locked_full_scoring" if r["seed"] < 5 else "extension_pairs_only"
        )
    values = np.array([r["c2st_auc"] for r in rows], dtype=np.float64)
    assert block["mean"] == pytest.approx(float(values.mean()), abs=1e-12)
    assert block["max"] == pytest.approx(float(values.max()), abs=1e-12)
    assert block["n_seeds"] == 20
    # The locked seeds' vector value equals their full-scoring pairs c2st.
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for r in rows:
        if r["seed"] in by_seed:
            assert r["c2st_auc"] == pytest.approx(
                by_seed[r["seed"]]["geometry"][PAIRS_VIEW]["scores"][
                    "c2st_auc"
                ],
                abs=0,
            )


def test_c2st_mean_rule_and_cap_from_gates_yaml():
    """The mean rule + cap thresholds and pass flags match gates.yaml + values."""
    artifact = _artifact()
    pairs_cfg = _gate1_thresholds()["views"][PAIRS_VIEW]
    mean_max = float(pairs_cfg["c2st_mean_rule"]["value_max"])
    cap_max = float(pairs_cfg["c2st_per_seed_cap"]["value_max"])
    # seed_set in gates.yaml is exactly 0-19.
    assert sorted(int(s) for s in pairs_cfg["c2st_mean_rule"]["seed_set"]) == (
        ALL_SEEDS
    )
    block = artifact["pairs_c2st_seed_set"]
    assert block["mean_rule_value_max"] == pytest.approx(mean_max, abs=0)
    assert block["per_seed_cap_value_max"] == pytest.approx(cap_max, abs=0)
    values = np.array(
        [
            r["c2st_auc"]
            for r in sorted(block["per_seed"], key=lambda e: e["seed"])
        ],
        dtype=np.float64,
    )
    mean_20 = float(values.mean())
    max_20 = float(values.max())
    v = artifact["verdict"]
    assert v["c2st_mean_20"] == pytest.approx(mean_20, abs=1e-12)
    assert v["c2st_max_20"] == pytest.approx(max_20, abs=1e-12)
    assert v["c2st_mean_rule_value_max"] == pytest.approx(mean_max, abs=0)
    assert v["c2st_per_seed_cap_value_max"] == pytest.approx(cap_max, abs=0)
    assert v["c2st_mean_rule_pass"] == (mean_20 <= mean_max)
    assert v["c2st_per_seed_cap_pass"] == (max_20 <= cap_max)


def test_amended_verdict_recomputes_from_full_conjunction():
    """The verdict recomputes from geometry+battery+pooled-Q0+mean-rule+cap.

    The full amended conjunction (gates.yaml protocol.pass_rule): gate passes
    iff >= 4/5 geometry AND >= 4/5 battery AND pooled Q0 AND 20-seed mean <=
    0.53 AND 20-seed per-seed max <= 0.554.
    """
    artifact = _artifact()
    table = artifact["seed_conjunction"]
    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in table:
        assert row["geometry_pass"] == by_seed[row["seed"]]["geometry_pass"]
        assert row["battery_pass"] == by_seed[row["seed"]]["battery_pass"]
        assert row["lambda"] == 0.1

    n_geo = sum(1 for r in table if r["geometry_pass"])
    n_bat = sum(1 for r in table if r["battery_pass"])
    pooled_q0_pass = artifact["benefit_space_gated"]["pooled_q0_gate"][
        "pooled_q0_pass"
    ]
    values = np.array(
        [
            r["c2st_auc"]
            for r in sorted(
                artifact["pairs_c2st_seed_set"]["per_seed"],
                key=lambda e: e["seed"],
            )
        ],
        dtype=np.float64,
    )
    pairs_cfg = _gate1_thresholds()["views"][PAIRS_VIEW]
    mean_max = float(pairs_cfg["c2st_mean_rule"]["value_max"])
    cap_max = float(pairs_cfg["c2st_per_seed_cap"]["value_max"])
    mean_pass = float(values.mean()) <= mean_max
    cap_pass = float(values.max()) <= cap_max

    v = artifact["verdict"]
    assert v["n_geometry_pass"] == n_geo
    assert v["n_battery_pass"] == n_bat
    assert v["geometry_gate_pass"] == (n_geo >= 4)
    assert v["battery_gate_pass"] == (n_bat >= 4)
    assert v["pooled_q0_pass"] == pooled_q0_pass
    assert v["c2st_mean_rule_pass"] == mean_pass
    assert v["c2st_per_seed_cap_pass"] == cap_pass
    expected = (
        (n_geo >= 4)
        and (n_bat >= 4)
        and pooled_q0_pass
        and mean_pass
        and cap_pass
    )
    assert v["gate_1_pass"] == expected

    sv = v["sub_verdicts"]
    assert sv["a_geometry_4_of_5"]["pass"] == (n_geo >= 4)
    assert sv["b_battery_4_of_5"]["pass"] == (n_bat >= 4)
    assert sv["c_pooled_q0"]["pass"] == pooled_q0_pass
    assert sv["d_c2st_mean_rule"]["pass"] == mean_pass
    assert sv["e_c2st_per_seed_cap"]["pass"] == cap_pass
    assert v["gate_1_pass"] == all(sv[k]["pass"] for k in sv)


def test_reproduction_block_internally_consistent():
    """Stored max-deviations recompute from stored per-seed values vs baselines.

    The one-shot reproduction discipline: every stored deviation and
    exact-match flag recomputes from the artifact's own stored per-seed values
    against the committed baselines (run 12 for the locked five; the 20-seed
    diagnostics for pairs c2st).
    """
    artifact = _artifact()
    repro = artifact["reproduction"]

    diag = json.loads(DIAGNOSTICS_ARTIFACT.read_text())
    diag_by_seed = {
        int(r["seed"]): float(r["candidate_pairs_c2st"])
        for r in diag["diagnostic_1_seed_extension"]["per_seed"]
    }
    pblock = repro["pairs_c2st_20_seed"]
    pairs_max = 0.0
    for e in pblock["per_seed"]:
        committed = diag_by_seed[e["seed"]]
        dev = abs(e["this_run"] - committed)
        assert e["committed"] == pytest.approx(committed, abs=0)
        assert e["abs_deviation"] == pytest.approx(dev, abs=1e-15)
        pairs_max = max(pairs_max, dev)
    assert pblock["max_abs_deviation"] == pytest.approx(pairs_max, abs=1e-15)
    assert pblock["exact_match"] == (pairs_max <= EXACT_ATOL)

    run12 = json.loads(RUN12_ARTIFACT.read_text())
    v4_by_seed = {s["seed"]: s for s in run12["per_seed"]}
    a5_by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    lblock = repro["locked_seed_blocks"]
    bat_mx = ben_mx = geo_mx = 0.0
    for e in lblock["per_seed"]:
        mine = a5_by_seed[e["seed"]]
        ref = v4_by_seed[e["seed"]]
        b = _block_max_dev(mine["battery_values"], ref["battery_values"])
        n = _block_max_dev(
            _benefit_metric_values(mine), _benefit_metric_values(ref)
        )
        g = _block_max_dev(
            _geometry_score_values(mine), _geometry_score_values(ref)
        )
        assert e["battery"]["max_abs_deviation"] == pytest.approx(b, abs=1e-15)
        assert e["battery"]["exact_match"] == (b <= EXACT_ATOL)
        assert e["benefit_metrics"]["max_abs_deviation"] == pytest.approx(
            n, abs=1e-15
        )
        assert e["benefit_metrics"]["exact_match"] == (n <= EXACT_ATOL)
        assert e["geometry_scores"]["max_abs_deviation"] == pytest.approx(
            g, abs=1e-15
        )
        assert e["geometry_scores"]["exact_match"] == (g <= EXACT_ATOL)
        bat_mx = max(bat_mx, b)
        ben_mx = max(ben_mx, n)
        geo_mx = max(geo_mx, g)
    assert lblock["battery_max_abs_deviation"] == pytest.approx(
        bat_mx, abs=1e-15
    )
    assert lblock["benefit_metrics_max_abs_deviation"] == pytest.approx(
        ben_mx, abs=1e-15
    )
    assert lblock["geometry_scores_max_abs_deviation"] == pytest.approx(
        geo_mx, abs=1e-15
    )

    pq = lblock["pooled_q0"]
    ref_pooled = run12["verdict"]["pooled_q0_mean_pct_diff"]
    my_pooled = artifact["benefit_space_gated"]["pooled_q0_gate"][
        "pooled_q0_mean_pct_diff"
    ]
    assert pq["this_run"] == pytest.approx(my_pooled, abs=0)
    assert pq["committed"] == pytest.approx(ref_pooled, abs=0)
    pooled_dev = abs(my_pooled - ref_pooled)
    assert pq["abs_deviation"] == pytest.approx(pooled_dev, abs=1e-15)

    assert lblock["exact_match"] == (
        bat_mx <= EXACT_ATOL
        and ben_mx <= EXACT_ATOL
        and geo_mx <= EXACT_ATOL
        and pooled_dev <= EXACT_ATOL
    )
    assert repro["all_exact_match"] == (
        pblock["exact_match"] and lblock["exact_match"]
    )
    assert (
        artifact["verdict"]["reproduction_all_exact_match"]
        == repro["all_exact_match"]
    )


def test_q0_participation_diagnostics_present():
    """The Q0 participation diagnostics (generated vs real) are reported."""
    q0p = _artifact()["q0_participation_diagnostics"]
    assert "per_seed" in q0p
    for row in q0p["per_seed"]:
        gen = row["generated"]
        real = row["real"]
        assert 0.0 <= gen["all_zero_share"] <= 1.0
        assert 0.0 <= real["all_zero_share"] <= 1.0
        assert gen["mean_positive_periods"] >= 0.0
        assert real["mean_positive_periods"] >= 0.0


def test_knn_diagnostics_reported_not_gated():
    """Reported-not-gated: usage shares, corners, neighbor distances."""
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
    """Reported-not-gated: donor-reuse record counts are non-negative."""
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


def test_uw_fit_reported_per_seed():
    """Each seed carries the u_w decomposition (reported)."""
    for seed in _artifact()["knn_context"]["per_seed"]:
        fit = seed["uw_fit"]
        assert fit["sigma_hat_w"] >= 0.0
        assert 0.0 <= fit["implied_perm_share"] <= 1.0
        u = fit["u_w_distribution_positive_obs"]
        assert 0.0 <= u["min"] <= u["median"] <= u["max"] <= 1.0


def test_candidate_panel_pin_metadata_consistent():
    """Each seed's window counts are positive and pairs >= runs."""
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


def test_revision_pins_record_sklearn_version():
    """The classifier version pin: the artifact records its sklearn version."""
    pins = _artifact()["revision_pins"]
    assert pins["gates_yaml_locked"] is True
    assert pins["artifact_schema_version"] == "gate1_rank_knn_v5"
    assert "sklearn_version" in pins
    # The ratified pin is scikit-learn < 1.9 (the gate venv scores under 1.8.x).
    assert pins["sklearn_version"].startswith("1.8")
    assert pins.get("pe_us_revision") is not None
