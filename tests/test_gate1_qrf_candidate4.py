"""Tests for the gate-1 candidate-4 structural-generation run.

Mirrors the candidate-3 run's tests (``test_gate1_qrf_candidate4`` is a
one-for-one analogue of ``test_gate1_qrf_candidate3``):

* :func:`test_seed0_reproduces_committed_artifact` (skipped when the PSID
  family files are absent AND when populace-fit is not importable — the
  dedicated-venv pattern, since populace-fit pins scikit-learn < 1.9)
  reruns seed 0 through the candidate-4 pipeline (stages 0-1 imported
  from candidate 3; the new structural generation local) and pins the
  committed artifact's seed-0 stage-1 fit, person-effect bookkeeping,
  geometry, and battery values to float precision — the run is
  reproducible from the seeds alone.
* The always-runnable consistency tests touch only the committed artifact
  and ``gates.yaml``: every reported pass/fail recomputes from its own
  stored score against its stored threshold, the stored thresholds equal
  the locked ones in ``gates.yaml``, the gate verdict recomputes from the
  seed-conjunction table, and the structural generation diagnostics are
  reported-not-gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate1_qrf_structural_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SPEC_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4888341875"
)
C2_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886538087"
)
C3_URL = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886848510"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _gate1_thresholds() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


# --------------------------------------------------------------------------
# Reproduction (needs the staged PSID family files AND populace-fit)
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
    import run_gate1_candidate4 as runner

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

    # Anchors on the full filtered panel, sliced per split (imported).
    all_anchor = runner.anchor_rows(panel)
    train_ids = train["person_id"].unique()
    holdout_ids = holdout["person_id"].unique()
    train_anchor = all_anchor[all_anchor.person_id.isin(train_ids)]
    holdout_anchor = all_anchor[all_anchor.person_id.isin(holdout_ids)]

    # Stage 0 (imported): weighted OLS quadratic-in-age residualiser.
    beta = runner.fit_age_residualizer(train)
    resid_train = runner.add_residuals(train, beta)
    resid_pos_full = resid_train.loc[
        resid_train["is_pos"],
        ["person_id", "period", "age", "weight", "r"],
    ].copy()
    resid_pos = resid_pos_full[["person_id", "period", "r"]]

    # Stage 1a-1c (imported from candidate 3): persistence-aware decomp.
    gamma, gamma_counts, pooled_mean = runner.pooled_autocovariances(resid_pos)
    stored_fit = seed0["stage1_fit"]
    for k in range(6):
        assert gamma[k] == pytest.approx(
            stored_fit["gamma"][str(k)], rel=0, abs=1e-12
        ), f"gamma_{k}: {gamma[k]} != {stored_fit['gamma'][str(k)]}"
        assert gamma_counts[k] == stored_fit["gamma_pair_counts"][str(k)]
    assert pooled_mean == pytest.approx(
        stored_fit["pooled_residual_mean"], rel=0, abs=1e-12
    )

    fit = runner.fit_three_component(gamma)
    assert fit["rho"] == pytest.approx(stored_fit["rho"], rel=0, abs=1e-12)
    for key in ("sigma2_perm", "sigma2_trans", "sigma2_noise"):
        assert fit[key] == pytest.approx(
            stored_fit[key], rel=0, abs=1e-12
        ), f"{key}: {fit[key]} != {stored_fit[key]}"
    assert fit["sse"] == pytest.approx(
        stored_fit["moment_sse"], rel=0, abs=1e-12
    )
    assert fit["implied_perm_share"] == pytest.approx(
        stored_fit["implied_perm_share"], rel=0, abs=1e-12
    )

    perm_train = runner.person_effects_pa(
        resid_pos, fit, pooled_mean, train_ids
    )

    # --- Generation (structural): each fresh fit reproduces its draws. ---
    part_pairs = runner.build_backward_pairs(train)
    assert int(len(part_pairs)) == seed0["n_train_pairs"]
    fitted_part = runner.fit_participation_model(part_pairs, 0)

    fitted_perm = runner.fit_perm_model(train_anchor, perm_train, 0)
    m_person, m_diag = runner.assemble_person_effect(
        fitted_perm, train_anchor, holdout_anchor, fit, 0
    )
    stored_mdiag = seed0["person_effect_diagnostic"]
    for key, stored_value in stored_mdiag.items():
        assert m_diag[key] == pytest.approx(
            stored_value, rel=0, abs=1e-9
        ), f"person-effect {key}: {m_diag[key]} != {stored_value}"

    t_pairs = runner.build_transitory_pairs(resid_pos_full, perm_train)
    assert int(len(t_pairs)) == seed0["n_transitory_pairs"]
    fitted_t = runner.fit_transitory_model(t_pairs, 0)

    candidate = runner.generate_candidate_structural(
        holdout, beta, m_person, fitted_t, fitted_part, fit, 0
    )

    # Generated-log-variance and perm-share diagnostics reproduce.
    logvar = runner.generated_log_variance(candidate, train)
    stored_logvar = seed0["generated_log_variance"]
    for key, stored_value in stored_logvar.items():
        assert logvar[key] == pytest.approx(
            stored_value, rel=0, abs=1e-9
        ), f"log-variance {key}: {logvar[key]} != {stored_value}"

    diag = runner.perm_share_diagnostic(
        beta, train, m_person, holdout_anchor, fit
    )
    stored_diag = seed0["perm_share_diagnostic"]
    for key, stored_value in stored_diag.items():
        assert diag[key] == pytest.approx(
            stored_value, rel=0, abs=1e-9
        ), f"perm-share {key}: {diag[key]} != {stored_value}"

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
    assert artifact["schema_version"] == "gate1_qrf_structural.v1"
    assert artifact["gate"] == "gate_1"
    assert artifact["revision_pins"]["gates_yaml_locked"] is True
    assert _gate1_thresholds()["locked"] is True


def test_spec_registration_recorded():
    """The frozen candidate-4 spec URL (and the c2/c3 base) are recorded."""
    artifact = _artifact()
    assert artifact["spec_registration"] == SPEC_URL
    assert artifact["candidate2_registration"] == C2_URL
    assert artifact["candidate3_registration"] == C3_URL
    assert artifact["builds_on"] == (
        "stages 0-1 identical to candidate 3 (imported); generation "
        "replaced by structural three-component assembly"
    )


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
        assert (
            seed["n_windows"]["psid_family_earnings_pairs"]
            >= seed["n_windows"]["psid_family_earnings_runs"]
        )


def test_stage1_fit_is_reported_not_gated_and_consistent():
    """The persistence-aware stage-1 fit is recorded per seed, not gated.

    Stages 0-1 are candidate 3's exactly (imported), so every seed carries
    the same stage-1 estimates (gamma_k, chosen rho, the three component
    variances, the implied perm share); the top-level context block
    mirrors them and records the memo's 0.467 back-out and the candidate
    2/3 drawn ranges. These are context only — they never enter the
    geometry or battery pass/fail, and the verdict rule names only those
    two families.
    """
    artifact = _artifact()
    ctx = artifact["generation_diagnostics_context"]
    assert ctx["memo_backout_perm_share"] == pytest.approx(0.467, abs=0)
    assert ctx["candidate2_drawn_perm_share_range"] == [0.52, 0.55]
    assert ctx["candidate3_drawn_perm_share_range"] == [0.21, 0.27]

    by_seed = {s["seed"]: s for s in artifact["per_seed"]}
    for row in ctx["per_seed"]:
        seed = by_seed[row["seed"]]
        fit = seed["stage1_fit"]
        diag = seed["perm_share_diagnostic"]
        mdiag = seed["person_effect_diagnostic"]
        lvdiag = seed["generated_log_variance"]
        # The context mirrors the per-seed stage-1 fit.
        assert row["rho"] == pytest.approx(fit["rho"], abs=0)
        for key in ("sigma2_perm", "sigma2_trans", "sigma2_noise"):
            assert row[key] == pytest.approx(fit[key], abs=0)
        assert row["gamma"] == fit["gamma"]
        assert row["implied_perm_share"] == pytest.approx(
            fit["implied_perm_share"], abs=0
        )
        # The context mirrors the per-seed person-effect + log-var + share.
        assert row["var_mu_hat_train"] == pytest.approx(
            mdiag["var_mu_hat_train"], abs=0
        )
        assert row["eta_variance"] == pytest.approx(
            mdiag["eta_variance"], abs=0
        )
        assert row["realized_var_m"] == pytest.approx(
            mdiag["realized_var_m"], abs=0
        )
        assert row["var_log_earnings_generated"] == pytest.approx(
            lvdiag["var_log_earnings_generated"], abs=0
        )
        assert row["var_log_earnings_train"] == pytest.approx(
            lvdiag["var_log_earnings_train"], abs=0
        )
        assert row["ratio_generated_over_train"] == pytest.approx(
            lvdiag["ratio_generated_over_train"], abs=0
        )
        assert row["perm_share"] == pytest.approx(diag["perm_share"], abs=0)
        assert row["perm_share_weighted"] == pytest.approx(
            diag["perm_share_weighted"], abs=0
        )
        # Structural identities of the fit.
        total = fit["sigma2_perm"] + fit["sigma2_trans"] + fit["sigma2_noise"]
        assert total == pytest.approx(fit["gamma"]["0"], rel=1e-6)
        assert fit["implied_perm_share"] == pytest.approx(
            fit["sigma2_perm"] / fit["gamma"]["0"], rel=0, abs=1e-12
        )
        # rho stays interior to the locked grid, and the three variances
        # are non-negative (NNLS).
        assert fit["rho_at_grid_boundary"] is False
        assert 0.50 <= fit["rho"] <= 0.95
        assert fit["sigma2_perm"] >= 0.0
        assert fit["sigma2_trans"] >= 0.0
        assert fit["sigma2_noise"] >= 0.0
        assert diag["memo_backout_perm_share"] == pytest.approx(0.467, abs=0)

    # The gate rule is the seed-level conjunction over geometry AND
    # battery only; no diagnostic is in it.
    assert artifact["verdict"]["rule"] == (
        ">=4/5 seeds geometry AND >=4/5 seeds battery"
    )


def test_person_effect_variance_bookkeeping_consistent():
    """The eta variance is the residual after mu_hat, floored at zero.

    By construction eta_variance = max(0, sigma2_perm -
    Var_train[mu_hat(anchor)]) and eta_sd = sqrt(eta_variance); the
    reported realized cross-person var(m) is a positive number (the
    structural target sigma2_perm). Reported-not-gated bookkeeping.
    """
    for seed in _artifact()["per_seed"]:
        md = seed["person_effect_diagnostic"]
        expected_eta = max(0.0, md["sigma2_perm"] - md["var_mu_hat_train"])
        assert md["eta_variance"] == pytest.approx(expected_eta, rel=1e-9)
        assert md["eta_sd"] == pytest.approx(expected_eta**0.5, rel=1e-9)
        assert md["realized_var_m"] >= 0.0
        assert md["var_mu_hat_train"] >= 0.0
        assert md["var_mu_hat_holdout"] >= 0.0
        assert md["n_train_anchors"] > 0
        assert md["n_holdout_anchors"] > 0
        assert md["sigma2_perm"] == pytest.approx(
            seed["stage1_fit"]["sigma2_perm"], rel=0, abs=1e-12
        )


def test_stage1_gamma_monotone_nonincreasing_in_every_seed():
    """gamma_k is non-increasing in the lag k in every seed.

    A permanent + AR(1)-transitory + noise autocovariance sequence decays
    monotonically from gamma_0 (which additionally carries the noise and
    full transitory variance) down toward the permanent floor. This is a
    reported-not-gated sanity property of the measured autocovariances.
    """
    for seed in _artifact()["per_seed"]:
        g = seed["stage1_fit"]["gamma"]
        vals = [g[str(k)] for k in range(6)]
        for a, b in zip(vals, vals[1:], strict=False):
            assert b <= a + 1e-9, f"gamma not monotone in seed {seed['seed']}"
