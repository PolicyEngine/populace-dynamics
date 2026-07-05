"""Gate-1 candidate 2: the latent-permanent conditioned chained QRF.

The SECOND pre-registered model run of PolicyEngine/populace-dynamics.
Scores the candidate-2 earnings process — a chained weighted QRF whose
every transition conditions on a drawn latent PERMANENT person effect —
against the LOCKED gate-1 thresholds in ``gates.yaml`` (the commit that
flipped ``locked: true``, pull request 39). The candidate is registered,
frozen before the run, in issue #42's registration comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4886538087);
every modeling degree of freedom below is fixed there and implemented
literally. No threshold is hardcoded, no model choice is tuned against
holdout scores, and the run is one shot. The outcome publishes whether
it passes or fails.

Why a latent permanent state (from the registration memo). The committed
battery autocorrelations imply a permanent + AR(1)-transitory + noise
structure with a permanent share of 0.467; no finite-lag chain can hold
a ~0.47 floor on long-horizon autocorrelation, and the baseline run
(#40) measured exactly the shortfall (10yr autocorr 0.31-0.37 vs the
locked band [0.469, 0.609]). Conditioning every transition on a drawn
persistent state puts a floor under long-horizon autocorrelation equal
to the variance share the drawn perm explains.

The protocol mechanics — the filtered panel, the person-disjoint split,
refit per seed, the two locked views, ``panel_scorecard`` scoring, the
battery on the candidate panel vs the committed ``battery_reference``,
the seed-level conjunction, the thresholds read from ``gates.yaml`` at
runtime, and the battery-reference bit-exact reproduction precheck — are
IMPORTED from the baseline runner (:mod:`run_gate1_baseline`) so they are
byte-for-byte identical to the baseline run. Only the model changes.

The model, per the frozen spec (each stage implemented literally):

* **Stage 0 - residualization (train split only).** For train
  person-periods with positive earnings, ``r = log(earnings) - f(age)``,
  where ``f`` is a weighted OLS quadratic in age (weights = person-period
  weights) fit pooled on the seed's 80% complement.
* **Stage 1 - person effect (empirical Bayes).** ``perm_i = n_i *
  mean_i / (n_i + lambda)`` with ``lambda = sigma2_within / sigma2_perm``
  from an UNWEIGHTED one-way random-effects method-of-moments
  decomposition on the train residuals: persons with ``n_i >= 2``
  identify ``sigma2_within``; ``sigma2_perm = max(0, var(person means) -
  sigma2_within * mean(1/n_i))``. Persons with zero positive observations
  get ``perm_i = 0`` (the prior mean, ``n_i = 0``).
* **Stage 2 - perm draw model.** ``RegimeGatedQRF`` (populace-fit
  defaults), target ``perm_i``, predictors = (anchor-period earnings
  level, anchor-period age), where a person's anchor is their
  chronologically last observed period in the filtered panel; training
  rows = train persons' anchor rows, ``sample_weight`` = the anchor
  row's weight. At generation each holdout person's perm is a DRAW from
  this model conditioned on their anchor row (holdout perms are never
  estimated from holdout data), the draw seeded from the gate seed.
* **Stage 3 - chain.** Identical to the baseline runner's backward
  one-step biennial chain (anchor keeps its real earnings, one-step
  across observation gaps, regime gate for zeros, earlier-period weight
  as ``sample_weight``), with ONE change: the transition QRF's predictors
  are (earnings at ``t``, age at ``t-2``, ``perm``), where ``perm`` is
  the ESTIMATED ``perm_i`` for training pairs and the DRAWN perm for
  generation. populace-fit defaults throughout; all RNG seeded from the
  gate seed.

Determinism. Every model seed is the gate seed ``s``. Stages 0-1 are
deterministic given the split. The Stage-2 perm draw is a fresh fitted
model's first (and only) ``predict``, seeded from ``s``. The Stage-3
chain reuses the baseline's batched-by-step, ordered-by-``person_id``
generation, so with a freshly fitted transition model per seed every
draw is fixed. The run reproduces from the seeds alone.

Run from the repository root with the PSID family files staged, using
the DEDICATED gate venv (populace-fit pins scikit-learn < 1.9, which the
repo's ``.venv`` violates; see populace #318):

    .venv-gate/bin/python scripts/run_gate1_candidate2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The protocol mechanics are IMPORTED from the baseline runner so that
# the split, view construction, battery definitions, geometry/battery
# checks, threshold loading, and the battery-reference reproduction are
# byte-for-byte identical to the baseline run. Only the model differs.
import run_gate1_baseline as baseline

# Importing populace.fit runs its import-time compat gate.
from populace.fit.qrf import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_ZERO_ATOL,
    RegimeGatedQRF,
)
from run_gate1_baseline import (  # noqa: F401 (re-exported for tests)
    AGE_MAX,
    AGE_MIN,
    BATTERY_REFERENCE_RUN,
    PERIOD_MAX,
    PERIOD_MIN,
    PERIOD_STEP,
    SEEDS,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    load_filtered_panel,
    load_gate1_thresholds,
    reproduce_battery_reference,
    split_holdout_train,
)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_qrf_latent_perm_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate1_qrf_latent_perm.v1"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886538087"
)

#: The perm-share the registration memo backs out of the committed
#: battery autocorrelations (permanent + AR(1) transitory + noise). This
#: is REPORTED as context against the measured perm share; it is not a
#: gated quantity.
MEMO_PERM_SHARE_BACKOUT = 0.467


# --------------------------------------------------------------------------
# Stage 0 - residualization (weighted OLS quadratic in age; train only)
# --------------------------------------------------------------------------
def fit_age_residualizer(train: pd.DataFrame) -> np.ndarray:
    """Weighted OLS of log positive earnings on a quadratic in age.

    Fit pooled on the train split's positive-earnings person-periods,
    weighted by the person-period weight. Returns the coefficient vector
    ``beta`` for the design ``[1, age, age**2]``. The residualizer is a
    property of the training population only; holdout rows never enter
    the fit.
    """
    pos = train[train["earnings"] > 0]
    age = pos["age"].to_numpy(dtype=np.float64)
    y = np.log(pos["earnings"].to_numpy(dtype=np.float64))
    w = pos["weight"].to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones_like(age), age, age**2])
    # Weighted normal equations: (X' W X) beta = X' W y.
    wx = design * w[:, None]
    xtwx = design.T @ wx
    xtwy = wx.T @ y
    beta = np.linalg.solve(xtwx, xtwy)
    return beta


def age_prediction(age: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Evaluate the fitted age quadratic ``f(age)`` at ``age``."""
    age = np.asarray(age, dtype=np.float64)
    design = np.column_stack([np.ones_like(age), age, age**2])
    return design @ beta


def add_residuals(frame: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    """Attach the Stage-0 residual ``r`` on positive-earnings rows.

    ``r = log(earnings) - f(age)`` where earnings > 0; rows with zero
    earnings carry ``NaN`` residual (they contribute no positive
    observation to the person effect). Returns a copy with columns
    ``r`` and ``is_pos`` added.
    """
    out = frame.copy()
    is_pos = out["earnings"].to_numpy() > 0
    r = np.full(len(out), np.nan, dtype=np.float64)
    age = out["age"].to_numpy(dtype=np.float64)
    logy = np.where(
        is_pos, np.log(np.where(is_pos, out["earnings"], 1.0)), 0.0
    )
    r[is_pos] = logy[is_pos] - age_prediction(age[is_pos], beta)
    out["r"] = r
    out["is_pos"] = is_pos
    return out


# --------------------------------------------------------------------------
# Stage 1 - empirical-Bayes person effect (method-of-moments lambda)
# --------------------------------------------------------------------------
def method_of_moments_lambda(
    resid_train: pd.DataFrame,
) -> dict[str, float]:
    """Unweighted one-way random-effects MoM decomposition on residuals.

    Operates on the train residuals of positive-earnings rows
    (``resid_train`` carries ``person_id`` and ``r``). Persons with
    ``n_i >= 2`` identify the within-person variance ``sigma2_within``
    (the pooled within-person sum of squares over its degrees of
    freedom). The between-person variance is
    ``sigma2_perm = max(0, var(person means) - sigma2_within *
    mean(1/n_i))``, where the person-mean variance and ``mean(1/n_i)``
    average over ALL persons with at least one positive observation and
    ``var`` is the population (biased, ddof=0) variance of the person
    means. ``lambda = sigma2_within / sigma2_perm``.

    The decomposition is UNWEIGHTED per the registration: it estimates a
    variance-components ratio, a structural property of the residual
    process, not a population average.

    Returns a dict with ``lambda``, ``sigma2_within``, ``sigma2_perm``,
    ``mean_inv_n``, ``var_person_means``, ``n_persons`` (persons with
    >=1 positive obs), and ``n_persons_multi`` (persons with >=2).
    """
    grp = resid_train.groupby("person_id")["r"]
    n_i = grp.size().to_numpy(dtype=np.float64)
    mean_i = grp.mean().to_numpy(dtype=np.float64)
    # Within-person sum of squares, pooled over persons with n_i >= 2.
    ss_within = grp.apply(
        lambda s: float(((s - s.mean()) ** 2).sum())
    ).to_numpy(dtype=np.float64)
    multi = n_i >= 2
    dof_within = float((n_i[multi] - 1).sum())
    sigma2_within = (
        float(ss_within[multi].sum() / dof_within) if dof_within > 0 else 0.0
    )
    # Population variance of person means over all persons with >=1 obs.
    var_person_means = float(np.average((mean_i - mean_i.mean()) ** 2))
    mean_inv_n = float(np.average(1.0 / n_i))
    sigma2_perm = max(0.0, var_person_means - sigma2_within * mean_inv_n)
    lam = float(sigma2_within / sigma2_perm) if sigma2_perm > 0 else np.inf
    return {
        "lambda": lam,
        "sigma2_within": sigma2_within,
        "sigma2_perm": sigma2_perm,
        "mean_inv_n": mean_inv_n,
        "var_person_means": var_person_means,
        "n_persons": int(len(n_i)),
        "n_persons_multi": int(multi.sum()),
    }


def person_effects(
    resid: pd.DataFrame, lam: float, all_person_ids: np.ndarray
) -> pd.DataFrame:
    """Empirical-Bayes shrunk person effects over ``all_person_ids``.

    ``perm_i = n_i * mean_i / (n_i + lambda)`` from a person's positive
    residual observations in ``resid``; persons in ``all_person_ids``
    with zero positive observations get ``perm_i = 0`` (the prior mean,
    ``n_i = 0``). Returns a frame indexed positionally with columns
    ``person_id``, ``perm``, ``n_pos``.
    """
    grp = resid.groupby("person_id")["r"]
    n_i = grp.size()
    mean_i = grp.mean()
    perm_obs = (n_i * mean_i) / (n_i + lam)
    perm = pd.DataFrame({"person_id": all_person_ids})
    perm = perm.merge(
        pd.DataFrame(
            {
                "person_id": perm_obs.index.to_numpy(),
                "perm": perm_obs.to_numpy(dtype=np.float64),
                "n_pos": n_i.to_numpy(dtype=np.float64),
            }
        ),
        on="person_id",
        how="left",
    )
    perm["perm"] = perm["perm"].fillna(0.0)
    perm["n_pos"] = perm["n_pos"].fillna(0.0)
    return perm


# --------------------------------------------------------------------------
# Anchors (chronologically last observed period per person)
# --------------------------------------------------------------------------
def anchor_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per person: their chronologically LAST observed period.

    Anchor = the person's maximum ``period`` in the filtered panel, with
    that row's ``earnings``, ``age``, ``weight``. Deterministic (ties on
    ``period`` cannot occur — periods are unique per person-period).
    """
    idx = panel.groupby("person_id")["period"].idxmax()
    cols = ["person_id", "period", "earnings", "age", "weight"]
    return panel.loc[idx, cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage 2 - perm draw model (RegimeGatedQRF on anchor covariates)
# --------------------------------------------------------------------------
def fit_perm_model(
    train_anchor: pd.DataFrame, perm_train: pd.DataFrame, seed: int
) -> Any:
    """Fit ``RegimeGatedQRF`` (defaults) for P(perm | anchor covariates).

    Predictors = (anchor earnings level, anchor age); target = the
    estimated ``perm``; ``sample_weight`` = the anchor row's weight.
    Training rows are the train persons' anchor rows joined to their
    estimated person effect. Seeded from the gate seed.
    """
    frame = train_anchor.merge(
        perm_train[["person_id", "perm"]], on="person_id", how="inner"
    )
    frame = frame.rename(
        columns={"earnings": "anchor_earnings", "age": "anchor_age"}
    )
    model = RegimeGatedQRF(seed=seed)  # populace-fit defaults
    return model.fit(
        frame,
        predictors=["anchor_earnings", "anchor_age"],
        targets=["perm"],
        weights="weight",
    )


def draw_holdout_perm(
    perm_model: Any, holdout_anchor: pd.DataFrame
) -> pd.DataFrame:
    """Draw each holdout person's perm from the anchor-conditioned model.

    A fresh fitted model's first (and only) ``predict``, so the draw is
    seeded from the gate seed. Rows are ordered by ``person_id`` for a
    deterministic draw order. Returns ``person_id`` and drawn ``perm``.
    """
    ha = holdout_anchor.sort_values("person_id").reset_index(drop=True)
    feats = ha.rename(
        columns={"earnings": "anchor_earnings", "age": "anchor_age"}
    )[["anchor_earnings", "anchor_age"]]
    drawn = perm_model.predict(feats)["perm"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {"person_id": ha["person_id"].to_numpy(), "perm": drawn}
    )


# --------------------------------------------------------------------------
# Stage 3 - backward transition model conditioned on perm
# --------------------------------------------------------------------------
def build_backward_pairs_with_perm(
    train: pd.DataFrame, perm_train: pd.DataFrame
) -> pd.DataFrame:
    """Adjacent-period BACKWARD pairs with the ESTIMATED perm attached.

    Mirrors the baseline's :func:`build_backward_pairs` exactly (one row
    per (person, period ``t``) whose period ``t-2`` is also observed;
    predictors = earnings at ``t`` and age at ``t-2``; target = earnings
    at ``t-2``; ``sample_weight`` = the ``t-2`` person-period weight),
    then attaches the person's estimated ``perm`` (a person-level column,
    identical across the person's pairs) as an additional predictor.
    """
    pairs = baseline.build_backward_pairs(train)
    pairs = pairs.merge(
        perm_train[["person_id", "perm"]], on="person_id", how="left"
    )
    return pairs


def fit_backward_model_with_perm(pairs: pd.DataFrame, seed: int) -> Any:
    """Fit ``RegimeGatedQRF`` (defaults) on the perm-conditioned pairs.

    Plain-DataFrame front door with the explicit ``t-2`` weight column,
    exactly as the baseline, with ``perm`` added to the predictor list.
    """
    model = RegimeGatedQRF(seed=seed)  # default hyperparameters
    return model.fit(
        pairs,
        predictors=["earnings", "age_tm2", "perm"],
        targets=["earnings_tm2"],
        weights="weight_tm2",
    )


def generate_candidate_with_perm(
    fitted: Any, holdout: pd.DataFrame, holdout_perm: pd.DataFrame
) -> pd.DataFrame:
    """Backward-chained candidate panel conditioned on the DRAWN perm.

    Identical in structure to the baseline's
    :func:`generate_candidate` — anchor each person's chronologically
    last observed period at its real earnings, chain backward over
    consecutive observed periods drawing each earlier period from
    ``fitted`` conditioned on the next observed period's
    (generated/anchor) earnings and the earlier period's age, one-step
    across gaps, batched by step-from-anchor and ordered by
    ``person_id`` within a step — with the person's DRAWN ``perm`` added
    as a third predictor at every step. Returns exactly the holdout
    persons on exactly their observed periods; only ``earnings`` is
    generated (anchor rows keep the real value).
    """
    hp = holdout.sort_values(["person_id", "period"]).reset_index(drop=True)
    hp = hp.merge(
        holdout_perm[["person_id", "perm"]], on="person_id", how="left"
    )
    hp["rank_from_top"] = (
        hp.groupby("person_id")["period"].rank(ascending=False, method="first")
        - 1
    ).astype(int)
    hp["depth"] = (
        hp.groupby("person_id")["period"].transform("size").astype(int)
    )

    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    ages = hp["age"].to_numpy(dtype=np.float64)
    perms = hp["perm"].to_numpy(dtype=np.float64)
    pids = hp["person_id"].to_numpy()
    ranks = hp["rank_from_top"].to_numpy()
    pos_by_key = {
        (pid, r): i for i, (pid, r) in enumerate(zip(pids, ranks, strict=True))
    }
    max_depth = int(hp["depth"].max()) if len(hp) else 0

    for j in range(1, max_depth):
        earlier_positions = np.nonzero(ranks == j)[0]
        if earlier_positions.size == 0:
            continue
        # Canonical, deterministic row order within the step.
        order = np.argsort(pids[earlier_positions], kind="stable")
        earlier_positions = earlier_positions[order]
        next_positions = np.array(
            [pos_by_key[(pids[p], j - 1)] for p in earlier_positions]
        )
        feat = pd.DataFrame(
            {
                "earnings": gen_earn[next_positions],
                "age_tm2": ages[earlier_positions],
                "perm": perms[earlier_positions],
            }
        )
        drawn = fitted.predict(feat)["earnings_tm2"].to_numpy(dtype=np.float64)
        gen_earn[earlier_positions] = drawn

    out = hp.copy()
    out["earnings"] = gen_earn
    return out[["person_id", "period", "earnings", "age", "weight"]]


# --------------------------------------------------------------------------
# Perm-share diagnostic (reported, not gated)
# --------------------------------------------------------------------------
def perm_share_diagnostic(
    beta: np.ndarray,
    train: pd.DataFrame,
    holdout_perm: pd.DataFrame,
    holdout_anchor: pd.DataFrame,
    mom: dict[str, float],
) -> dict[str, float]:
    """Measure the variance share the DRAWN perm explains.

    The registration memo backs a permanent share of 0.467 out of the
    committed battery autocorrelations (permanent + AR(1) transitory +
    noise). The analogue this candidate realizes is the variance of the
    DRAWN holdout perms over the variance of the Stage-0 train residuals
    (both in log points, the residual's own scale). Reported as the
    primary ``perm_share`` alongside the MoM structural share
    ``sigma2_perm / (sigma2_perm + sigma2_within)`` for context; neither
    is gated.

    All variances are unweighted population variances, matching the MoM
    decomposition. The drawn-perm variance is weighted by the anchor
    weight as a secondary figure (``perm_share_weighted``), since the
    holdout population the draws describe is a weighted sample.
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
    denom = mom["sigma2_perm"] + mom["sigma2_within"]
    mom_perm_share = mom["sigma2_perm"] / denom if denom > 0 else 0.0
    return {
        "perm_share": perm_share,
        "perm_share_weighted": perm_share_weighted,
        "var_drawn_perm": var_drawn_perm,
        "var_drawn_perm_weighted": var_drawn_perm_weighted,
        "var_residual_train": var_residual_train,
        "mom_perm_share": mom_perm_share,
        "memo_backout_perm_share": MEMO_PERM_SHARE_BACKOUT,
    }


# --------------------------------------------------------------------------
# Driver
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
    """Fit, generate, and score candidate 2 for a single gate seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Anchors on the FULL filtered panel (a person's last observed period
    # is a property of the panel, computed once and sliced per split).
    all_anchor = anchor_rows(panel)
    train_ids = train["person_id"].unique()
    holdout_ids = holdout["person_id"].unique()
    train_anchor = all_anchor[all_anchor.person_id.isin(train_ids)]
    holdout_anchor = all_anchor[all_anchor.person_id.isin(holdout_ids)]

    # Stage 0: weighted OLS quadratic-in-age residualizer (train only).
    beta = fit_age_residualizer(train)
    resid_train = add_residuals(train, beta)
    resid_pos = resid_train.loc[resid_train["is_pos"], ["person_id", "r"]]

    # Stage 1: MoM lambda + empirical-Bayes person effects (train).
    mom = method_of_moments_lambda(resid_pos)
    perm_train = person_effects(resid_pos, mom["lambda"], train_ids)

    # Stage 2: fit P(perm | anchor) on train, draw holdout perms.
    perm_model = fit_perm_model(train_anchor, perm_train, seed)
    holdout_perm = draw_holdout_perm(perm_model, holdout_anchor)

    # Stage 3: perm-conditioned backward chain.
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
        beta, train, holdout_perm, holdout_anchor, mom
    )

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
        "lambda": mom["lambda"],
        "mom": mom,
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
            f"lambda={mom['lambda']:.3f} "
            f"perm_share={d['perm_share']:.3f} "
            f"(mom {d['mom_perm_share']:.3f}) "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-2 run."""
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

    # Identical battery-reference bit-exact precheck as the baseline: the
    # battery code path must reproduce every committed reference value to
    # float precision before any candidate is scored.
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
        "run": "gate1_qrf_latent_perm_v1",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "description": (
            "Gate-1 candidate 2: latent-permanent conditioned chained "
            "weighted QRF. A drawn latent PERMANENT person effect "
            "(empirical-Bayes shrunk within-person mean of "
            "age-residualized log earnings, drawn per holdout person from "
            "a RegimeGatedQRF conditioned on the person's anchor row) "
            "conditions every step of the baseline's backward one-step "
            "biennial chain. Registered frozen before the run in issue "
            "#42 (see spec_registration). Candidate scored against the "
            "held-out PSID family earnings panel geometry (two locked "
            "views) and the locked moment battery, per the locked "
            "seed-level conjunction in gates.yaml (pull request 39). "
            "Protocol mechanics imported byte-for-byte from the baseline "
            "runner (pull request 40)."
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
            },
            "stage_1_person_effect": {
                "estimator": (
                    "empirical-Bayes shrunk within-person mean of residuals: "
                    "perm_i = n_i * mean_i / (n_i + lambda)"
                ),
                "lambda": (
                    "sigma2_within / sigma2_perm from an UNWEIGHTED one-way "
                    "random-effects method-of-moments decomposition on train "
                    "residuals; persons with n_i >= 2 identify sigma2_within; "
                    "sigma2_perm = max(0, var(person means) - sigma2_within * "
                    "mean(1/n_i))"
                ),
                "zero_positive_obs": "perm_i = 0 (prior mean, n_i = 0)",
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
                "(imported from the baseline runner)"
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
        "perm_share_context": {
            "memo_backout_perm_share": MEMO_PERM_SHARE_BACKOUT,
            "note": (
                "perm_share = var(drawn holdout perms) / var(Stage-0 train "
                "residuals); mom_perm_share = sigma2_perm / (sigma2_perm + "
                "sigma2_within). Reported per seed as context against the "
                "memo's 0.467 back-out; NOT gated."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "lambda": s["lambda"],
                    "perm_share": s["perm_share_diagnostic"]["perm_share"],
                    "perm_share_weighted": s["perm_share_diagnostic"][
                        "perm_share_weighted"
                    ],
                    "mom_perm_share": s["perm_share_diagnostic"][
                        "mom_perm_share"
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
