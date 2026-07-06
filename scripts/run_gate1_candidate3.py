"""Gate-1 candidate 3: persistence-aware permanent-component decomposition.

The THIRD pre-registered model run of PolicyEngine/populace-dynamics. It
amends ONLY stage 1 of the candidate-2 registration; every other stage,
feature, default, seed, and the entire protocol are identical and are
IMPORTED from the merged candidate-2 runner (:mod:`run_gate1_candidate2`,
pull request 43), which itself imports the protocol machinery from the
baseline runner (:mod:`run_gate1_baseline`, pull request 40). The
candidate-3 amendment is registered, frozen before the run, in issue
#42's candidate-3 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4886848510);
it is implemented literally. No threshold is hardcoded, no model choice
is tuned against holdout scores, and the run is one shot. The outcome
publishes whether it passes or fails.

Why the amendment (from the candidate-3 registration). Candidate 2
published (pull request 43): a drawn permanent share of 0.52-0.55 vs the
memo's structural 0.467 back-out, and a 10-year autocorrelation
OVERSHOOT as the sole failing battery statistic on every seed
(``autocorr_log_10yr`` value ~0.61-0.67 vs the locked band centred at
0.539, tolerance 0.070). The diagnosis: candidate 2's white-noise
one-way random-effects decomposition over-attributed variance to the
permanent component because the transitory component is itself
persistent and does not average out over a person's few biennial
observations, so its persistence leaked into ``perm``. Candidate 3
replaces the white-noise MoM lambda/shrinkage with a persistence-aware
three-component decomposition (permanent + AR(1)-persistent transitory +
white noise), estimated on each seed's TRAIN split only (never from the
committed reference artifacts), so the persistent-transitory variance is
attributed correctly and only the truly permanent variance drives the
long-horizon autocorrelation floor.

The stage-1 amendment, per the frozen spec (implemented literally):

* **Stage 1a - within-person autocovariances (train split).** From the
  stage-0 residuals of positive-earnings rows, centred at the pooled
  mean, compute pooled UNWEIGHTED within-person autocovariances
  ``gamma_k`` for biennial lags ``k = 0..5``: for each lag, the mean of
  ``r_it * r_i,t+2k`` over all within-person pairs of observed periods
  exactly ``2k`` years apart (``gamma_0`` over all residual squares).
* **Stage 1b - three-component fit.** Fit ``gamma_0 = sigma2_perm +
  sigma2_trans + sigma2_noise`` and ``gamma_k = sigma2_perm +
  sigma2_trans * rho**k`` (``k = 1..5``): grid ``rho`` in
  ``{0.50, 0.51, ..., 0.95}``; at each ``rho`` solve the six moment
  equations for ``(sigma2_perm, sigma2_trans, sigma2_noise)`` by
  non-negative least squares (``scipy.optimize.nnls``); select the
  ``rho`` minimising the sum of squared moment residuals. Deterministic;
  no iterative starting-point sensitivity.
* **Stage 1c - person effects with correlated-noise shrinkage.**
  ``perm_hat_i = w_i * mean_i(r)`` with
  ``w_i = sigma2_perm / (sigma2_perm + V_i)`` and
  ``V_i = (1/n_i**2) * [sigma2_trans * sum_j sum_l rho**(|p_j - p_l|/2)
  + n_i * sigma2_noise]`` over person ``i``'s observed positive periods
  ``p`` (biennial lag distances); ``mean_i(r)`` is the person's mean of
  the pooled-mean-CENTRED residuals (the scale the decomposition lives
  on). Persons with no positive observations get ``perm_hat_i = 0``, as
  before.

Stages 0, 2, 3 are unchanged from the candidate-2 registration: the same
weighted-OLS quadratic-in-age residualiser, the same anchor definition,
the same ``RegimeGatedQRF`` perm-draw model (predictors = anchor
earnings level, anchor age), the same perm-conditioned backward biennial
chain (predictors = earnings at ``t``, age at ``t-2``, ``perm``) with the
same gap/zero rules, populace-fit defaults throughout, and every RNG
seeded from the gate seed. Those functions are imported, not
reimplemented, so they are byte-for-byte the candidate-2 code.

Determinism. Every model seed is the gate seed ``s``. Stages 0, 1a, 1b,
1c are deterministic given the split (the grid search and NNLS have no
random state). The Stage-2 perm draw is a fresh fitted model's first
``predict``, seeded from ``s``. The Stage-3 chain reuses the candidate-2
batched-by-step, ordered-by-``person_id`` generation. The run reproduces
from the seeds alone.

Run from the repository root with the PSID family files staged, using
the DEDICATED gate venv (populace-fit pins scikit-learn < 1.9, which the
repo's ``.venv`` violates; see populace #318):

    .venv-gate/bin/python scripts/run_gate1_candidate3.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The protocol mechanics and stages 0, 2, 3 are IMPORTED from the merged
# candidate-2 runner so that the split, residualiser, anchors, perm-draw
# model, perm-conditioned chain, view construction, battery definitions,
# threshold loading, and the battery-reference reproduction are
# byte-for-byte identical to candidate 2 (which imports them, in turn,
# from the baseline runner). ONLY stage 1 is overridden below.
import run_gate1_candidate2 as c2

# Importing populace.fit (transitively, via candidate 2) runs its
# import-time compat gate; the RegimeGatedQRF hyperparameter defaults are
# re-exported for the artifact's model block.
from populace.fit.qrf import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL
from run_gate1_candidate2 import (  # noqa: F401 (re-exported for tests)
    AGE_MAX,
    AGE_MIN,
    BATTERY_REFERENCE_RUN,
    PERIOD_MAX,
    PERIOD_MIN,
    PERIOD_STEP,
    SEEDS,
    add_residuals,
    anchor_rows,
    build_backward_pairs_with_perm,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    draw_holdout_perm,
    fit_age_residualizer,
    fit_backward_model_with_perm,
    fit_perm_model,
    generate_candidate_with_perm,
    load_filtered_panel,
    load_gate1_thresholds,
    reproduce_battery_reference,
    split_holdout_train,
)
from scipy.optimize import nnls

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_qrf_latent_perm_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate1_qrf_latent_perm.v2"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886848510"
)
#: The candidate-2 registration this amendment starts from (reported for
#: provenance; candidate 3 amends only its stage 1).
BASE_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886538087"
)

#: The perm-share the registration memo backs out of the committed
#: battery autocorrelations (permanent + AR(1) transitory + noise). This
#: is REPORTED as context against the measured perm share; not gated.
MEMO_PERM_SHARE_BACKOUT = c2.MEMO_PERM_SHARE_BACKOUT  # 0.467

#: The persistence-aware autocovariance lags: biennial k = 0..5, i.e. the
#: 0/2/4/6/8/10-year within-person lags.
GAMMA_LAGS = (0, 1, 2, 3, 4, 5)

#: The rho grid: 0.50..0.95 step 0.01 (inclusive on both ends). Built
#: once so the fit and its provenance echo the identical array.
RHO_GRID = tuple(round(0.50 + 0.01 * i, 2) for i in range(46))


# --------------------------------------------------------------------------
# Stage 1a - pooled within-person autocovariances (train split)
# --------------------------------------------------------------------------
def pooled_autocovariances(
    resid_pos: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, int], float]:
    """Pooled unweighted within-person autocovariances on train residuals.

    ``resid_pos`` carries ``person_id``, ``period``, ``r`` for the train
    split's positive-earnings rows. Residuals are centred at the pooled
    (unweighted) mean; then for each biennial lag ``k = 0..5``,

    * ``gamma_0`` = mean over ALL centred residual squares,
    * ``gamma_k`` = mean of ``rc_it * rc_i,t+2k`` over ALL within-person
      pairs of observed periods exactly ``2k`` years apart (each such
      unordered pair contributes once, via a self-merge that lifts the
      later row's period down by the lag onto the earlier row).

    Returns ``(gamma, counts, pooled_mean)`` where ``gamma`` and
    ``counts`` are keyed by lag ``k`` and ``pooled_mean`` is the centring
    constant.
    """
    df = resid_pos[["person_id", "period", "r"]].copy()
    pooled_mean = float(df["r"].to_numpy(dtype=np.float64).mean())
    df["rc"] = df["r"].to_numpy(dtype=np.float64) - pooled_mean

    gamma: dict[int, float] = {}
    counts: dict[int, int] = {}

    rc0 = df["rc"].to_numpy(dtype=np.float64)
    gamma[0] = float(np.mean(rc0**2))
    counts[0] = int(rc0.size)

    for k in range(1, 6):
        lag_years = PERIOD_STEP * k
        left = df.rename(columns={"period": "p_l", "rc": "rc_l"})[
            ["person_id", "p_l", "rc_l"]
        ]
        right = df.rename(columns={"period": "p_r", "rc": "rc_r"})
        # Lift the later row's period down by the lag so it inner-joins
        # the earlier row observed 2k years before it.
        right = right.assign(p_l=right["p_r"] - lag_years)
        merged = left.merge(
            right[["person_id", "p_l", "rc_r"]],
            on=["person_id", "p_l"],
            how="inner",
        )
        prods = (merged["rc_l"] * merged["rc_r"]).to_numpy(dtype=np.float64)
        gamma[k] = float(np.mean(prods)) if prods.size else float("nan")
        counts[k] = int(prods.size)

    return gamma, counts, pooled_mean


# --------------------------------------------------------------------------
# Stage 1b - three-component fit (grid rho, NNLS variances, min SSE)
# --------------------------------------------------------------------------
def _design_matrix(rho: float) -> np.ndarray:
    """The 6x3 moment design at a given ``rho``.

    Columns are ``[sigma2_perm, sigma2_trans, sigma2_noise]``; rows are
    the biennial lags ``k = 0..5``:

    * ``gamma_0 = perm + trans + noise``  -> row ``[1, 1, 1]``,
    * ``gamma_k = perm + trans * rho**k`` -> row ``[1, rho**k, 0]``.
    """
    A = np.zeros((6, 3), dtype=np.float64)
    A[:, 0] = 1.0  # perm loads on every equation
    A[0, 1] = 1.0  # trans at k=0 is rho**0 = 1
    A[0, 2] = 1.0  # noise loads only on gamma_0
    for k in range(1, 6):
        A[k, 1] = rho**k
    return A


def fit_three_component(gamma: dict[int, float]) -> dict[str, Any]:
    """Grid-search ``rho``, NNLS the three variances, minimise moment SSE.

    Solves the six moment equations for ``(sigma2_perm, sigma2_trans,
    sigma2_noise)`` at each ``rho`` on the locked grid by non-negative
    least squares (``scipy.optimize.nnls``), and selects the ``rho``
    giving the smallest sum of squared moment residuals. Deterministic.

    Returns a dict with ``rho``, the three component variances, ``sse``,
    the ``implied_perm_share`` (``sigma2_perm / gamma_0``), and the
    per-grid-point SSE trace for provenance.
    """
    g = np.array([gamma[k] for k in range(6)], dtype=np.float64)
    sse_trace: list[dict[str, float]] = []
    best: dict[str, Any] | None = None
    for rho in RHO_GRID:
        A = _design_matrix(rho)
        x, rnorm = nnls(A, g)
        sse = float(rnorm**2)
        sse_trace.append({"rho": float(rho), "sse": sse})
        if best is None or sse < best["sse"]:
            best = {
                "rho": float(rho),
                "sigma2_perm": float(x[0]),
                "sigma2_trans": float(x[1]),
                "sigma2_noise": float(x[2]),
                "sse": sse,
            }
    assert best is not None
    gamma_0 = float(gamma[0])
    best["gamma_0"] = gamma_0
    best["implied_perm_share"] = (
        best["sigma2_perm"] / gamma_0 if gamma_0 > 0 else 0.0
    )
    best["sse_trace"] = sse_trace
    return best


# --------------------------------------------------------------------------
# Stage 1c - person effects with correlated-noise shrinkage
# --------------------------------------------------------------------------
def person_effects_pa(
    resid_pos: pd.DataFrame,
    fit: dict[str, Any],
    pooled_mean: float,
    all_person_ids: np.ndarray,
) -> pd.DataFrame:
    """Correlated-noise shrunk person effects on the CENTRED residuals.

    For each person with at least one positive observation,

    * ``V_i = (1/n_i**2) * [sigma2_trans * sum_j sum_l rho**(|p_j-p_l|/2)
      + n_i * sigma2_noise]`` over the person's observed positive periods
      (the double sum runs over ALL ordered pairs, so its diagonal
      contributes ``n_i`` ones),
    * ``w_i = sigma2_perm / (sigma2_perm + V_i)``,
    * ``perm_hat_i = w_i * mean_i(rc)``, the person's mean of the
      pooled-mean-CENTRED residuals.

    Persons in ``all_person_ids`` with no positive observation get
    ``perm_hat_i = 0`` (the prior mean). Returns a frame with
    ``person_id``, ``perm``, ``n_pos``, and the reported-not-gated
    per-person ``w_i`` / ``V_i``.
    """
    s2p = float(fit["sigma2_perm"])
    s2t = float(fit["sigma2_trans"])
    s2n = float(fit["sigma2_noise"])
    rho = float(fit["rho"])

    df = resid_pos[["person_id", "period", "r"]].copy()
    df["rc"] = df["r"].to_numpy(dtype=np.float64) - pooled_mean

    pids: list[Any] = []
    perms: list[float] = []
    n_pos: list[float] = []
    w_list: list[float] = []
    v_list: list[float] = []
    for pid, sub in df.groupby("person_id", sort=True):
        periods = sub["period"].to_numpy(dtype=np.float64)
        rc = sub["rc"].to_numpy(dtype=np.float64)
        n_i = int(periods.size)
        mean_rc = float(rc.mean())
        # Full AR(1) correlation-matrix sum over the observed periods:
        # rho**(biennial lag distance) for every ordered pair (incl. j=l).
        lag_bien = np.abs(periods[:, None] - periods[None, :]) / PERIOD_STEP
        ar_sum = float(np.sum(rho**lag_bien))
        v_i = (1.0 / n_i**2) * (s2t * ar_sum + n_i * s2n)
        w_i = s2p / (s2p + v_i) if (s2p + v_i) > 0 else 0.0
        pids.append(pid)
        perms.append(w_i * mean_rc)
        n_pos.append(float(n_i))
        w_list.append(w_i)
        v_list.append(v_i)

    eff = pd.DataFrame(
        {
            "person_id": pids,
            "perm": np.asarray(perms, dtype=np.float64),
            "n_pos": np.asarray(n_pos, dtype=np.float64),
            "w_i": np.asarray(w_list, dtype=np.float64),
            "V_i": np.asarray(v_list, dtype=np.float64),
        }
    )
    perm = pd.DataFrame({"person_id": all_person_ids}).merge(
        eff, on="person_id", how="left"
    )
    perm["perm"] = perm["perm"].fillna(0.0)
    perm["n_pos"] = perm["n_pos"].fillna(0.0)
    return perm


# --------------------------------------------------------------------------
# Perm-share diagnostic (reported, not gated) - persistence-aware form
# --------------------------------------------------------------------------
def perm_share_diagnostic(
    beta: np.ndarray,
    train: pd.DataFrame,
    holdout_perm: pd.DataFrame,
    holdout_anchor: pd.DataFrame,
    fit: dict[str, Any],
) -> dict[str, float]:
    """Measure the variance share the DRAWN perm explains, plus the fit.

    Mirrors the candidate-2 diagnostic (drawn-perm variance over Stage-0
    train-residual variance, unweighted primary + anchor-weighted
    secondary) so the per-seed drawn-perm-share numbers are comparable to
    v1, and reports the persistence-aware structural share
    ``implied_perm_share = sigma2_perm / gamma_0`` in place of candidate
    2's white-noise ``mom_perm_share``. All variances are unweighted
    population variances, matching the decomposition. None is gated.
    """
    resid = add_residuals(train, beta)
    r_train = resid.loc[resid["is_pos"], "r"].to_numpy(dtype=np.float64)
    var_residual_train = float(np.var(r_train))

    drawn = holdout_perm.merge(
        holdout_anchor[["person_id", "weight"]], on="person_id", how="left"
    )
    perm = drawn["perm"].to_numpy(dtype=np.float64)
    w = drawn["weight"].to_numpy(dtype=np.float64)
    var_drawn_perm = float(np.var(perm))
    mean_w = float(np.average(perm, weights=w))
    var_drawn_perm_weighted = float(
        np.average((perm - mean_w) ** 2, weights=w)
    )

    perm_share = (
        var_drawn_perm / var_residual_train if var_residual_train > 0 else 0.0
    )
    perm_share_weighted = (
        var_drawn_perm_weighted / var_residual_train
        if var_residual_train > 0
        else 0.0
    )
    return {
        "perm_share": perm_share,
        "perm_share_weighted": perm_share_weighted,
        "var_drawn_perm": var_drawn_perm,
        "var_drawn_perm_weighted": var_drawn_perm_weighted,
        "var_residual_train": var_residual_train,
        "implied_perm_share": float(fit["implied_perm_share"]),
        "memo_backout_perm_share": MEMO_PERM_SHARE_BACKOUT,
    }


# --------------------------------------------------------------------------
# Driver - candidate 3 differs from candidate 2 ONLY in stage 1
# --------------------------------------------------------------------------
def run_seed(
    seed: int,
    panel: pd.DataFrame,
    view_specs: dict[str, Any],
    views_cfg: dict[str, Any],
    battery_reference: dict[str, float],
    battery_tol: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    """Fit, generate, and score candidate 3 for a single gate seed.

    Identical to :func:`run_gate1_candidate2.run_seed` except stage 1:
    the white-noise MoM lambda and empirical-Bayes shrinkage are replaced
    by the persistence-aware autocovariance decomposition (stages 1a-1c).
    All other stages call the imported candidate-2 functions.
    """
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Anchors on the FULL filtered panel (imported candidate-2 function).
    all_anchor = anchor_rows(panel)
    train_ids = train["person_id"].unique()
    holdout_ids = holdout["person_id"].unique()
    train_anchor = all_anchor[all_anchor.person_id.isin(train_ids)]
    holdout_anchor = all_anchor[all_anchor.person_id.isin(holdout_ids)]

    # Stage 0: weighted OLS quadratic-in-age residualiser (imported).
    beta = fit_age_residualizer(train)
    resid_train = add_residuals(train, beta)
    resid_pos = resid_train.loc[
        resid_train["is_pos"], ["person_id", "period", "r"]
    ]

    # Stage 1 (AMENDED): persistence-aware decomposition on the train
    # split, then correlated-noise shrunk person effects.
    gamma, gamma_counts, pooled_mean = pooled_autocovariances(resid_pos)
    fit = fit_three_component(gamma)
    perm_train = person_effects_pa(resid_pos, fit, pooled_mean, train_ids)

    # Stage 2: fit P(perm | anchor) on train, draw holdout perms (imported).
    perm_model = fit_perm_model(train_anchor, perm_train, seed)
    holdout_perm = draw_holdout_perm(perm_model, holdout_anchor)

    # Stage 3: perm-conditioned backward chain (imported).
    pairs = build_backward_pairs_with_perm(train, perm_train)
    fitted = fit_backward_model_with_perm(pairs, seed)
    candidate = generate_candidate_with_perm(fitted, holdout, holdout_perm)

    # --- geometry: score candidate vs holdout on both locked views ---
    geometry_by_view: dict[str, Any] = {}
    geometry_seed_pass = True
    n_windows: dict[str, int] = {}
    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(candidate, holdout, view, seed=seed)
        checks = check_geometry(scores, views_cfg[vname]["geometry"])
        view_pass = all(c["pass"] for c in checks.values())
        geometry_seed_pass = geometry_seed_pass and view_pass
        cand_windows, _ = hpanel.project_panel(candidate, view)
        n_windows[vname] = int(len(cand_windows))
        geometry_by_view[vname] = {
            "scores": {k: float(v) for k, v in scores.items()},
            "thresholds": views_cfg[vname]["geometry"],
            "checks": checks,
            "view_pass": bool(view_pass),
        }

    # --- battery: on the CANDIDATE panel, vs committed reference ---
    battery_values = compute_battery(candidate)
    battery_checks = check_battery(
        battery_values, battery_reference, battery_tol
    )
    battery_seed_pass = all(c["pass"] for c in battery_checks.values())

    # --- perm-share diagnostic (reported, not gated) ---
    perm_diag = perm_share_diagnostic(
        beta, train, holdout_perm, holdout_anchor, fit
    )

    # Stage-1 fit block (reported, not gated): the amendment's estimates.
    stage1_fit = {
        "gamma": {str(k): float(gamma[k]) for k in GAMMA_LAGS},
        "gamma_pair_counts": {
            str(k): int(gamma_counts[k]) for k in GAMMA_LAGS
        },
        "pooled_residual_mean": float(pooled_mean),
        "rho": float(fit["rho"]),
        "sigma2_perm": float(fit["sigma2_perm"]),
        "sigma2_trans": float(fit["sigma2_trans"]),
        "sigma2_noise": float(fit["sigma2_noise"]),
        "moment_sse": float(fit["sse"]),
        "implied_perm_share": float(fit["implied_perm_share"]),
        "rho_at_grid_boundary": bool(
            fit["rho"] <= RHO_GRID[0] or fit["rho"] >= RHO_GRID[-1]
        ),
    }

    result = {
        "seed": seed,
        "n_persons": int(holdout.person_id.nunique()),
        "n_person_periods": int(len(holdout)),
        "n_train_persons": int(train.person_id.nunique()),
        "n_train_pairs": int(len(pairs)),
        "n_windows": n_windows,
        "regimes": {
            "perm_model": perm_model.regimes(),
            "transition": fitted.regimes(),
        },
        "stage1_fit": stage1_fit,
        "perm_share_diagnostic": perm_diag,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if verbose:
        d = perm_diag
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"rho={fit['rho']:.2f} "
            f"perm_share(drawn)={d['perm_share']:.3f} "
            f"(implied {d['implied_perm_share']:.3f}) "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-3 run."""
    started = time.time()
    thresholds = load_gate1_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_1 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    views_cfg = thresholds["views"]
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }

    battery_ref_artifact = json.loads(
        (ROOT / BATTERY_REFERENCE_RUN).read_text()
    )
    battery_reference = battery_ref_artifact["battery_reference"]

    panel = load_filtered_panel()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    # Identical battery-reference bit-exact precheck as candidate 2 / the
    # baseline: the battery code path must reproduce every committed
    # reference value to float precision before any candidate is scored.
    repro = reproduce_battery_reference(panel)
    if verbose:
        print(
            "battery_reference reproduced exactly: "
            f"{repro['all_committed_values_reproduced_exactly']}"
        )
    if not repro["all_committed_values_reproduced_exactly"]:
        raise RuntimeError(
            "Battery code path does not reproduce the committed "
            "battery_reference to float precision; refusing to proceed "
            "with a divergent definition."
        )

    view_specs = {
        "psid_family_earnings_pairs": build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed.append(
            run_seed(
                seed,
                panel,
                view_specs,
                views_cfg,
                battery_reference,
                battery_tol,
                verbose,
            )
        )

    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    gate_pass = geometry_gate_pass and battery_gate_pass

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_qrf_latent_perm_v2",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "base_registration": BASE_REGISTRATION,
        "amends": "candidate 2 stage 1 only (all other stages identical)",
        "description": (
            "Gate-1 candidate 3: persistence-aware permanent-component "
            "decomposition. Amends ONLY stage 1 of candidate 2: the "
            "white-noise one-way random-effects lambda/shrinkage is "
            "replaced by a three-component decomposition (permanent + "
            "AR(1)-persistent transitory + white noise) of pooled "
            "within-person autocovariances of age-residualised log "
            "earnings, estimated on each seed's train split (grid rho, "
            "NNLS the three variances on the six biennial-lag moments, min "
            "SSE), with correlated-noise shrinkage for the person effect. "
            "Every other stage - the weighted-OLS age residualiser, the "
            "anchor definition, the RegimeGatedQRF perm-draw model, and "
            "the perm-conditioned backward biennial chain - is imported "
            "byte-for-byte from the candidate-2 runner (pull request 43), "
            "which imports its protocol machinery from the baseline runner "
            "(pull request 40). Registered frozen before the run in issue "
            "#42's candidate-3 comment (see spec_registration). Candidate "
            "scored against the held-out PSID family earnings panel "
            "geometry (two locked views) and the locked moment battery, "
            "per the locked seed-level conjunction in gates.yaml (pull "
            "request 39)."
        ),
        "model": {
            "class": "populace.fit.qrf.RegimeGatedQRF",
            "front_door": "plain DataFrame (explicit weight column)",
            "hyperparameters": {
                "n_estimators": DEFAULT_N_ESTIMATORS,
                "zero_atol": DEFAULT_ZERO_ATOL,
                "max_samples_leaf": None,
                "note": "populace-fit defaults; seed = gate seed s",
            },
            "stage_0_residualization": {
                "target": "log positive earnings (train split only)",
                "regressor": "weighted OLS quadratic in age [1, age, age^2]",
                "weights": "person-period weight",
                "fit": "pooled on the seed's 80% complement",
                "note": "unchanged from candidate 2 (imported)",
            },
            "stage_1_person_effect": {
                "amendment": (
                    "persistence-aware three-component decomposition "
                    "(replaces candidate 2's white-noise MoM lambda + "
                    "empirical-Bayes shrinkage)"
                ),
                "stage_1a_autocovariances": (
                    "pooled UNWEIGHTED within-person autocovariances "
                    "gamma_k for biennial lags k=0..5 on the train "
                    "residuals centred at the pooled mean; gamma_0 over all "
                    "residual squares, gamma_k = mean of r_it * r_i,t+2k "
                    "over within-person pairs exactly 2k years apart"
                ),
                "stage_1b_three_component_fit": (
                    "gamma_0 = sigma2_perm + sigma2_trans + sigma2_noise; "
                    "gamma_k = sigma2_perm + sigma2_trans * rho**k "
                    "(k=1..5); grid rho in {0.50..0.95 step 0.01}; at each "
                    "rho, non-negative least squares (scipy.optimize.nnls) "
                    "for the three variances on the six moments; select rho "
                    "minimising the sum of squared moment residuals"
                ),
                "stage_1c_correlated_noise_shrinkage": (
                    "perm_hat_i = w_i * mean_i(rc); w_i = sigma2_perm / "
                    "(sigma2_perm + V_i); V_i = (1/n_i**2) * [sigma2_trans "
                    "* sum_j sum_l rho**(|p_j-p_l|/2) + n_i * sigma2_noise] "
                    "over the person's observed positive periods; rc = "
                    "pooled-mean-centred residual; persons with no positive "
                    "observation get perm_hat_i = 0"
                ),
                "rho_grid": list(RHO_GRID),
                "gamma_lags_biennial": list(GAMMA_LAGS),
            },
            "stage_2_perm_draw_model": {
                "class": "populace.fit.qrf.RegimeGatedQRF (defaults)",
                "target": "perm_i",
                "predictors": ["anchor earnings level", "anchor age"],
                "anchor": (
                    "person's chronologically last observed period in the "
                    "filtered panel"
                ),
                "training_rows": "train persons' anchor rows",
                "sample_weight": "anchor row weight",
                "generation": (
                    "each holdout person's perm is a DRAW conditioned on "
                    "their anchor row (holdout perms never estimated from "
                    "holdout data), seeded from the gate seed"
                ),
                "note": "unchanged from candidate 2 (imported)",
            },
            "stage_3_chain": {
                "direction": "backward (one-step biennial)",
                "target": "earnings at period t-2",
                "predictors": [
                    "earnings at period t",
                    "age at period t-2",
                    "perm",
                ],
                "perm_source": (
                    "estimated perm_i for training pairs; drawn perm for "
                    "generation"
                ),
                "sample_weight": "earlier-period (t-2) person-period weight",
                "fit_pairs": "80% complement's adjacent 2-year pairs",
                "anchor": (
                    "chronologically last observed period held at real "
                    "earnings"
                ),
                "gap_rule": (
                    "gap (next observed period 4+ years later) applies the "
                    "one-step model once across the gap"
                ),
                "row_order": (
                    "batched by step-from-anchor; within a step, ordered by "
                    "person_id (deterministic)"
                ),
                "candidate_panel_pin": (
                    "exactly the holdout persons on exactly their observed "
                    "periods; only earnings generated; anchor keeps real "
                    "value; person_id/period/age/weight copied from holdout"
                ),
                "note": "unchanged from candidate 2 (imported)",
            },
        },
        "protocol": {
            "filter": (
                f"age {AGE_MIN}-{AGE_MAX}, reference years "
                f"{PERIOD_MIN}-{PERIOD_MAX}, positive weights (applied "
                "before the split)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel, 'person_id', fraction=0.2, seed=s); the drawn 20% "
                "is the holdout, the complement is the training set "
                "(imported from the baseline runner via candidate 2)"
            ),
            "seeds": list(SEEDS),
            "views": {
                "psid_family_earnings_pairs": {"window": 2, "period_step": 2},
                "psid_family_earnings_runs": {"window": 3, "period_step": 2},
            },
            "scoring": (
                "panel_scorecard(candidate, holdout, view, seed=s) per "
                "locked view; battery on the candidate panel vs committed "
                "battery_reference (imported from the baseline runner)"
            ),
            "pass_rule": (
                "seed passes geometry iff every locked threshold on every "
                "locked view holds; seed passes battery iff every locked "
                "tolerance holds; gate passes iff >=4/5 seeds pass geometry "
                "AND >=4/5 seeds pass battery"
            ),
        },
        "battery_reference_reproduction": repro,
        "battery_reference_run": BATTERY_REFERENCE_RUN,
        "per_seed": per_seed,
        "seed_conjunction": [
            {
                "seed": s["seed"],
                "geometry_pass": s["geometry_pass"],
                "battery_pass": s["battery_pass"],
            }
            for s in per_seed
        ],
        "stage1_fit_context": {
            "memo_backout_perm_share": MEMO_PERM_SHARE_BACKOUT,
            "candidate2_drawn_perm_share_range": [0.52, 0.55],
            "note": (
                "The persistence-aware stage 1 (candidate 3) attributes "
                "the persistent-transitory variance to an AR(1) component "
                "rather than the permanent one. Reported per seed: gamma_k "
                "(biennial lags 0..5), chosen rho, the three component "
                "variances, and implied_perm_share = sigma2_perm/gamma_0. "
                "perm_share = var(drawn holdout perms)/var(Stage-0 train "
                "residuals) is the drawn-perm-share analogue reported in "
                "candidate 2 (whose seed-0 drawn share was 0.527). NONE is "
                "gated; the gate rule names only geometry AND battery."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "rho": s["stage1_fit"]["rho"],
                    "sigma2_perm": s["stage1_fit"]["sigma2_perm"],
                    "sigma2_trans": s["stage1_fit"]["sigma2_trans"],
                    "sigma2_noise": s["stage1_fit"]["sigma2_noise"],
                    "gamma": s["stage1_fit"]["gamma"],
                    "implied_perm_share": s["stage1_fit"][
                        "implied_perm_share"
                    ],
                    "perm_share": s["perm_share_diagnostic"]["perm_share"],
                    "perm_share_weighted": s["perm_share_diagnostic"][
                        "perm_share_weighted"
                    ],
                }
                for s in per_seed
            ],
        },
        "verdict": {
            "n_seeds": len(SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "gate_1_pass": bool(gate_pass),
            "rule": ">=4/5 seeds geometry AND >=4/5 seeds battery",
        },
        "revision_pins": _revision_pins(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5)"
        )
    return artifact


def _revision_pins() -> dict[str, Any]:
    """Repo/populace SHAs and schema version for provenance."""
    import subprocess

    def _sha(cwd: Path) -> str | None:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode()
                .strip()
            )
        except Exception:
            return None

    populace_root = Path("~/PolicyEngine/populace").expanduser()
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "populace_repo_sha": _sha(populace_root),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml_locked": True,
    }


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
