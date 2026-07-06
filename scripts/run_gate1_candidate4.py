"""Gate-1 candidate 4: structural three-component generation.

The FOURTH pre-registered model run of PolicyEngine/populace-dynamics. It
keeps stages 0 and 1 of the candidate-3 registration verbatim (they are
IMPORTED from the merged candidate-3 runner, :mod:`run_gate1_candidate3`,
pull request 44, which imports stage 0 and the protocol machinery from
candidate 2 / the baseline) and replaces the GENERATION with a structural
three-component assembly on the log scale. The candidate-4 spec is
registered, frozen before the run, in issue #42's candidate-4 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4888341875);
it is implemented literally. No threshold is hardcoded, no model choice
is tuned against holdout scores, and the run is one shot. The outcome
publishes whether it passes or fails.

Why the structural generation (from the candidate-4 diagnosis).
Candidates 2 and 3 produced statistically identical autocorrelation
ladders despite a 2.3x difference in drawn-permanent variance, and their
shared 10-year floor (~0.648) equalled the raw train person-mean
variance share, not the decomposed permanent share. That is
errors-in-variables scaling: rescaling the person-effect FEATURE makes
the chained QRF rescale its response inversely, so generation transmits
the same total person-level variance either way. No feature-side dial
fixes it. Candidate 4 therefore sets the person-level variance and the
observation-noise layer STRUCTURALLY at generation, assembling each
positive log earnings from four explicit pieces (age fit, person effect,
persistent transitory chain, white noise) whose variances are the
stage-1 estimates by construction.

Generation, per the frozen spec (implemented literally):

* **Participation (zero vs positive):** unchanged from the prior
  candidates' machinery. A ``RegimeGatedQRF`` (populace-fit defaults,
  seeded from the gate seed) is fit on the 80% complement's adjacent
  backward biennial pairs with predictors = (next-period earnings level,
  earlier-period age) and target = earlier-period earnings, exactly the
  baseline transition. At each backward step its GATE decides zero vs
  positive conditioned on the next period's GENERATED level and the
  earlier period's age; the magnitude, when positive, is supplied by the
  structural assembly below (not by the forest's magnitude draw). This
  is the same regime-gated zero/positive channel the prior candidates
  passed on every geometry and spell check.
* **Magnitude (positive observations), log scale, assembled from
  components.** For each holdout person, with their anchor
  (chronologically last observed period) kept at its REAL value:

  1. **Person effect.** ``m_i = mu_hat(anchor_i) + eta_i``, where
     ``mu_hat`` is the conditional MEDIAN of ``QRF_perm`` (the candidate-2
     perm-draw model: target ``perm_hat_i``, predictors = anchor earnings
     level and anchor age, trained on train anchors, populace-fit
     defaults) evaluated at the person's anchor row, and
     ``eta_i ~ N(0, max(0, sigma2_perm - Var_train[mu_hat(anchor)]))``
     with ``Var_train`` the variance of ``mu_hat`` over the TRAIN
     persons' anchors. The cross-person variance of ``m`` thus equals
     ``sigma2_perm`` by construction; the anchor informs its location
     only.
  2. **Transitory chain.** ``t_hat_it = r_it - perm_hat_i`` on the train
     split (age-residual less the estimated person effect) defines
     training pairs; ``QRF_t`` (populace-fit defaults, seeded from the
     gate seed) fits ``t_prev | (t_next, age_prev)`` on the adjacent
     backward biennial pairs of positive train observations. At
     generation ``t_anchor,i = r_anchor,i - m_i`` (the anchor's observed
     residual less the drawn person effect; this start absorbs the
     anchor's observation noise, accepted and documented), then the
     one-step kernel chains backward over the person's observed periods,
     one step across gaps as before.
  3. **Observation noise.** ``eps_it ~ N(0, sigma2_noise)`` i.i.d. per
     generated positive observation (NONE at the anchor, which keeps its
     real value).
  4. **Assembly.** generated positive earnings =
     ``exp(f_hat(age_it) + m_i + t_it + eps_it)``, where ``f_hat`` is the
     stage-0 age fit. The anchor keeps its real earnings.

All RNG is seeded from the gate seed; populace-fit defaults throughout;
thresholds are read from ``gates.yaml`` at runtime; the paired splits,
the two locked views, ``panel_scorecard`` scoring, the battery vs the
committed ``battery_reference``, the seed-level conjunction (>=4/5 both
blocks), and the battery-reference bit-exact precheck are the imported
protocol machinery, byte-for-byte the prior runs'. One shot; publishes
pass or fail.

Determinism. Every model seed is the gate seed ``s``. Stages 0 and 1 are
deterministic given the split (imported from candidate 3). The
conditional median is a deterministic function of the fitted
``QRF_perm`` and the anchor rows (a mixture-CDF inversion at 0.5, no
RNG). ``QRF_t`` and the participation gate are freshly fitted models
seeded from ``s``; their draws are read in the canonical
batched-by-step, ordered-by-``person_id`` order the baseline chain uses.
The Gaussian ``eta`` (one per holdout person, ordered by ``person_id``)
and ``eps`` (one per generated positive observation, drawn in the same
step/person order as the chain) come from a single generator seeded from
``s``, drawn in a fixed order. The run reproduces from the seeds alone.

Run from the repository root with the PSID family files staged, using
the DEDICATED gate venv (populace-fit pins scikit-learn < 1.9, which the
repo's ``.venv`` violates; see populace #318):

    .venv-gate/bin/python scripts/run_gate1_candidate4.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Stages 0 and 1 and the protocol machinery are IMPORTED from the merged
# candidate-3 runner (which imports stage 0 and the protocol from
# candidate 2 / the baseline), so the split, residualiser, anchors,
# persistence-aware decomposition, view construction, battery
# definitions, threshold loading, and the battery-reference reproduction
# are byte-for-byte identical to the prior runs. ONLY the generation
# changes below.
import run_gate1_candidate3 as c3

# Importing populace.fit (transitively) runs its import-time compat gate;
# the RegimeGatedQRF hyperparameter defaults and the fine quantile grid
# are re-exported for the assembly and the artifact's model block.
from populace.fit.qrf import (
    _QUANTILE_GRID,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_ZERO_ATOL,
    RegimeGatedQRF,
)

# The baseline transition's backward-pair builder is the participation
# channel's fit data; it lives in the baseline runner (candidates 2/3
# import it there but do not re-export it), so import it directly.
from run_gate1_baseline import build_backward_pairs

# ``age_prediction`` (the stage-0 age fit's evaluator) is defined in the
# candidate-2 runner and used by ``add_residuals`` there, but candidate 3
# never re-exported it into its namespace, so it must be imported from its
# actual definition site (byte-for-byte the same stage-0 function the whole
# chain uses; the structural assembly evaluates f_hat(age) with it).
from run_gate1_candidate2 import age_prediction
from run_gate1_candidate3 import (  # noqa: F401 (re-exported for tests)
    AGE_MAX,
    AGE_MIN,
    BATTERY_REFERENCE_RUN,
    GAMMA_LAGS,
    PERIOD_MAX,
    PERIOD_MIN,
    PERIOD_STEP,
    RHO_GRID,
    SEEDS,
    add_residuals,
    anchor_rows,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    fit_age_residualizer,
    fit_perm_model,
    fit_three_component,
    load_filtered_panel,
    load_gate1_thresholds,
    person_effects_pa,
    pooled_autocovariances,
    reproduce_battery_reference,
    split_holdout_train,
)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_qrf_structural_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate1_qrf_structural.v1"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4888341875"
)
#: The candidate-2 and candidate-3 registrations this run builds on
#: (stages 0-1 are candidate 3's exactly; candidate 3 amends candidate 2's
#: stage 1; candidate 2 fixes the protocol). Reported for provenance.
CANDIDATE2_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886538087"
)
CANDIDATE3_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4886848510"
)

#: The perm-share the registration memo backs out of the committed
#: battery autocorrelations (permanent + AR(1) transitory + noise);
#: REPORTED as context, not gated (kept for comparability with v1/v2).
MEMO_PERM_SHARE_BACKOUT = c3.MEMO_PERM_SHARE_BACKOUT  # 0.467


# --------------------------------------------------------------------------
# Conditional median of QRF_perm (deterministic mixture-CDF inversion)
# --------------------------------------------------------------------------
def _forest_value_grid(forest: Any, x: np.ndarray) -> np.ndarray:
    """Per-row conditional quantile VALUES of a sign forest on the grid.

    Queries the fitted quantile forest at the shared fine quantile grid
    ``_QUANTILE_GRID`` (the same grid the model's own draw reads), giving
    a ``(rows, len(grid))`` matrix of quantile values, ascending along the
    grid axis. This is the sign-conditional quantile function used to
    build the mixture CDF.
    """
    preds = np.asarray(forest.model.predict(x, quantiles=list(_QUANTILE_GRID)))
    return preds.reshape(len(x), len(_QUANTILE_GRID))


def conditional_median(fitted_perm: Any, anchors: pd.DataFrame) -> np.ndarray:
    """Deterministic conditional median of ``QRF_perm`` at each anchor.

    ``QRF_perm`` is a regime-gated mixture: the gate assigns sign-class
    probabilities ``(p_neg, p_zero, p_pos)`` and a per-sign quantile
    forest gives the magnitude conditional. The conditional median is the
    0.5 quantile of the full mixture CDF

        ``F(x) = p_neg * F_neg(x) + p_zero * 1{x >= 0} + p_pos * F_pos(x)``

    inverted per row. ``F_neg`` / ``F_pos`` are read by interpolating the
    probe value into that row's ascending quantile-value grid (value ->
    quantile). The probe set per row is the union of the sign grids and
    ``0`` (the only atom the zero class contributes); the median is the
    smallest probe at which ``F >= 0.5``. Ungated single-sign regimes
    reduce to that sign's 0.5-quantile value; a degenerate-zero target
    gives ``0``. No RNG: a deterministic function of the fit and the
    anchor rows, so it reproduces exactly and supports the variance
    bookkeeping in the person-effect assembly.

    ``anchors`` carries ``earnings`` and ``age`` (the anchor row); they
    are renamed to the model's predictor columns. Rows are taken in the
    given order (the caller orders by ``person_id``). Returns one median
    per anchor row, positionally aligned with ``anchors``.
    """
    tm = fitted_perm._target_models[fitted_perm.targets[0]]
    feat = anchors.rename(
        columns={"earnings": "anchor_earnings", "age": "anchor_age"}
    )
    x = feat.loc[:, list(tm.columns)].to_numpy(dtype=np.float64)
    n = len(x)

    # Gate probabilities per sign class (or a point mass for single-sign).
    if tm.gate is not None:
        classes = np.asarray(tm.gate.classes_)
        proba = np.asarray(tm.gate.predict_proba(x))
        pmap = {int(c): proba[:, i] for i, c in enumerate(classes)}
    elif tm.positive is not None:
        pmap = {1: np.ones(n)}
    elif tm.negative is not None:
        pmap = {-1: np.ones(n)}
    else:  # degenerate zero
        return np.zeros(n, dtype=np.float64)
    p_neg = pmap.get(-1, np.zeros(n))
    p_zero = pmap.get(0, np.zeros(n))
    p_pos = pmap.get(1, np.zeros(n))

    neg_v = (
        _forest_value_grid(tm.negative, x) if tm.negative is not None else None
    )
    pos_v = (
        _forest_value_grid(tm.positive, x) if tm.positive is not None else None
    )
    grid = _QUANTILE_GRID

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        probes = [0.0]
        if neg_v is not None:
            probes.extend(neg_v[i].tolist())
        if pos_v is not None:
            probes.extend(pos_v[i].tolist())
        xs = np.unique(np.asarray(probes, dtype=np.float64))
        f = p_zero[i] * (xs >= 0.0).astype(np.float64)
        if neg_v is not None:
            f = f + p_neg[i] * np.interp(
                xs, neg_v[i], grid, left=0.0, right=1.0
            )
        if pos_v is not None:
            f = f + p_pos[i] * np.interp(
                xs, pos_v[i], grid, left=0.0, right=1.0
            )
        idx = int(np.searchsorted(f, 0.5, side="left"))
        idx = min(idx, len(xs) - 1)
        out[i] = xs[idx]
    return out


# --------------------------------------------------------------------------
# Transitory chain model (QRF_t on t = r - perm_hat over positive pairs)
# --------------------------------------------------------------------------
def build_transitory_pairs(
    resid_pos_full: pd.DataFrame, perm_train: pd.DataFrame
) -> pd.DataFrame:
    """Adjacent backward biennial pairs of the transitory residual ``t``.

    ``t_it = r_it - perm_hat_i`` on the train split's positive rows (the
    age-residual less the estimated person effect). ``resid_pos_full``
    carries ``person_id``, ``period``, ``age``, ``weight``, ``r`` for the
    positive train rows. One row per (person, period ``t``) whose period
    ``t-2`` is also an observed positive row:

    * ``t_next`` = ``t`` at period ``t``       (predictor),
    * ``age_prev`` = age at period ``t-2``     (predictor),
    * ``t_prev`` = ``t`` at period ``t-2``     (target),
    * ``weight_prev`` = person-period weight at ``t-2`` (sample_weight).

    Only exact 2-year adjacencies of positive observations form a pair
    (the transitory residual exists only where earnings are positive), so
    the fitted kernel is the one-step biennial backward transitory step.
    ``perm_hat`` lives on the residual's own scale (candidate 3 stage 1c),
    so ``t = r - perm_hat`` reconstructs the age-residual as perm + t.
    """
    df = resid_pos_full.merge(
        perm_train[["person_id", "perm"]], on="person_id", how="left"
    )
    df = df.assign(
        t=df["r"].to_numpy(dtype=np.float64)
        - df["perm"].to_numpy(dtype=np.float64)
    )
    base = df[["person_id", "period", "age", "t", "weight"]].copy()
    earlier = base.rename(
        columns={
            "t": "t_prev",
            "age": "age_prev",
            "weight": "weight_prev",
        }
    )
    earlier = earlier.assign(period=earlier["period"] + PERIOD_STEP)
    pairs = base.rename(columns={"t": "t_next"}).merge(
        earlier[["person_id", "period", "t_prev", "age_prev", "weight_prev"]],
        on=["person_id", "period"],
        how="inner",
    )
    return pairs[
        ["person_id", "period", "t_next", "age_prev", "t_prev", "weight_prev"]
    ]


def fit_transitory_model(pairs: pd.DataFrame, seed: int) -> Any:
    """Fit ``RegimeGatedQRF`` (defaults) for ``t_prev | (t_next, age_prev)``.

    Plain-DataFrame front door with the explicit earlier-period weight
    column, seeded from the gate seed. The transitory residual is a
    real-valued (sign-mixed) target; the regime gate handles whatever
    sign support it carries.
    """
    model = RegimeGatedQRF(seed=seed)  # populace-fit defaults
    return model.fit(
        pairs,
        predictors=["t_next", "age_prev"],
        targets=["t_prev"],
        weights="weight_prev",
    )


# --------------------------------------------------------------------------
# Participation gate (baseline transition; only its zero/positive gate used)
# --------------------------------------------------------------------------
def fit_participation_model(pairs: pd.DataFrame, seed: int) -> Any:
    """Fit the baseline transition ``RegimeGatedQRF`` (defaults), seed s.

    Predictors = (earnings, age_tm2), target = earnings_tm2, weights =
    weight_tm2 — the baseline transition. Only the zero/positive gate is
    consumed by generation.
    """
    model = RegimeGatedQRF(seed=seed)  # populace-fit defaults
    return model.fit(
        pairs,
        predictors=["earnings", "age_tm2"],
        targets=["earnings_tm2"],
        weights="weight_tm2",
    )


def _gate_positive_prob(fitted_part: Any, x: np.ndarray) -> np.ndarray:
    """P(positive) from the participation model's gate for feature rows.

    The baseline transition's target ``earnings_tm2`` is
    zero-inflated-positive, so its gate's classes are ``{0, 1}``. Returns
    the per-row probability of the positive class (for the stochastic
    zero/positive draw). If the target were degenerate the model has no
    gate; then P(positive)=1 for a positive-only target and 0 for a
    degenerate zero.
    """
    tm = fitted_part._target_models[fitted_part.targets[0]]
    n = len(x)
    if tm.gate is None:
        if tm.positive is not None:
            return np.ones(n, dtype=np.float64)
        return np.zeros(n, dtype=np.float64)
    classes = np.asarray(tm.gate.classes_)
    proba = np.asarray(tm.gate.predict_proba(x))
    pos_cols = np.nonzero(classes == 1)[0]
    if pos_cols.size == 0:
        return np.zeros(n, dtype=np.float64)
    return proba[:, int(pos_cols[0])].astype(np.float64)


# --------------------------------------------------------------------------
# Structural generation (participation gate + assembled magnitude)
# --------------------------------------------------------------------------
def generate_candidate_structural(
    holdout: pd.DataFrame,
    beta: np.ndarray,
    m_person: pd.DataFrame,
    fitted_t: Any,
    fitted_part: Any,
    fit: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Structural three-component backward-chained candidate panel.

    For each holdout person, anchor the chronologically last observed
    period at its REAL earnings, then chain BACKWARD over the person's
    observed periods (batched by step-from-anchor, ordered by
    ``person_id`` within a step — the baseline chain order). At each
    backward step:

    * the participation gate (from ``fitted_part``) draws zero vs positive
      conditioned on the next period's GENERATED earnings level and the
      earlier period's age;
    * the transitory residual ``t`` is drawn from ``fitted_t`` conditioned
      on the next period's transitory residual ``t`` and the earlier
      period's age (chaining ``t`` backward from ``t_anchor,i =
      r_anchor,i - m_i``, one step across gaps);
    * where positive, the assembled earnings are
      ``exp(f_hat(age) + m_i + t_it + eps_it)`` with
      ``eps_it ~ N(0, sigma2_noise)`` drawn i.i.d. per positive
      generated observation; where the gate draws zero, earnings = 0.

    The anchor keeps its real earnings and contributes no ``eps``. The
    ``eta`` person draws are already folded into ``m_person`` (drawn by
    the caller, ordered by ``person_id``); the ``eps`` draws and the gate
    / transitory / gate-sign RNG are seeded from the gate seed and read in
    the canonical step/person order, so the panel reproduces exactly.

    Returns exactly the holdout persons on exactly their observed periods
    (``person_id`` / ``period`` / ``age`` / ``weight`` copied from the
    holdout, only ``earnings`` generated; the anchor keeps the real
    value).
    """
    s2_noise = float(fit["sigma2_noise"])
    sd_noise = float(np.sqrt(max(0.0, s2_noise)))

    hp = holdout.sort_values(["person_id", "period"]).reset_index(drop=True)
    hp = hp.merge(m_person[["person_id", "m"]], on="person_id", how="left")
    hp["rank_from_top"] = (
        hp.groupby("person_id")["period"].rank(ascending=False, method="first")
        - 1
    ).astype(int)
    hp["depth"] = (
        hp.groupby("person_id")["period"].transform("size").astype(int)
    )

    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    ages = hp["age"].to_numpy(dtype=np.float64)
    m_i = hp["m"].to_numpy(dtype=np.float64)
    pids = hp["person_id"].to_numpy()
    ranks = hp["rank_from_top"].to_numpy()

    # f_hat(age) on every row (the stage-0 age fit).
    f_age = age_prediction(ages, beta)

    # Transitory residual carried down the chain. At the anchor
    # (rank 0), t_anchor = r_anchor - m_i where r_anchor is the anchor's
    # observed age-residual (positive anchors only; a zero anchor has no
    # residual, so its t starts at 0 and is never assembled). The chain
    # overwrites earlier ranks in place.
    is_anchor = ranks == 0
    anchor_pos = is_anchor & (gen_earn > 0)
    r_anchor = np.zeros(len(hp), dtype=np.float64)
    r_anchor[anchor_pos] = np.log(gen_earn[anchor_pos]) - f_age[anchor_pos]
    t_chain = np.zeros(len(hp), dtype=np.float64)
    t_chain[is_anchor] = r_anchor[is_anchor] - m_i[is_anchor]

    pos_by_key = {
        (pid, r): i for i, (pid, r) in enumerate(zip(pids, ranks, strict=True))
    }
    max_depth = int(hp["depth"].max()) if len(hp) else 0

    rng = np.random.default_rng(seed)
    for j in range(1, max_depth):
        earlier_positions = np.nonzero(ranks == j)[0]
        if earlier_positions.size == 0:
            continue
        order = np.argsort(pids[earlier_positions], kind="stable")
        earlier_positions = earlier_positions[order]
        next_positions = np.array(
            [pos_by_key[(pids[p], j - 1)] for p in earlier_positions]
        )

        # Transitory draw: t_prev | (t_next, age_prev).
        t_feat = pd.DataFrame(
            {
                "t_next": t_chain[next_positions],
                "age_prev": ages[earlier_positions],
            }
        )
        t_prev = fitted_t.predict(t_feat)["t_prev"].to_numpy(dtype=np.float64)
        t_chain[earlier_positions] = t_prev

        # Participation gate: P(positive | next generated level, age_prev).
        part_x = np.column_stack(
            [gen_earn[next_positions], ages[earlier_positions]]
        )
        p_pos = _gate_positive_prob(fitted_part, part_x)
        u = rng.random(len(earlier_positions))
        positive = u < p_pos

        # eps ~ N(0, sigma2_noise) i.i.d. per generated positive obs.
        eps = rng.normal(0.0, sd_noise, size=len(earlier_positions))

        log_earn = (
            f_age[earlier_positions] + m_i[earlier_positions] + t_prev + eps
        )
        drawn = np.where(positive, np.exp(log_earn), 0.0)
        gen_earn[earlier_positions] = drawn

    out = hp.copy()
    out["earnings"] = gen_earn
    return out[["person_id", "period", "earnings", "age", "weight"]]


# --------------------------------------------------------------------------
# Person-effect assembly (m_i = mu_hat(anchor) + eta_i)
# --------------------------------------------------------------------------
def assemble_person_effect(
    fitted_perm: Any,
    train_anchor: pd.DataFrame,
    holdout_anchor: pd.DataFrame,
    fit: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """``m_i = mu_hat(anchor_i) + eta_i`` for each holdout person.

    ``mu_hat`` is the conditional median of ``QRF_perm`` at the anchor
    (deterministic). ``Var_train[mu_hat(anchor)]`` is the variance of
    ``mu_hat`` over the TRAIN persons' anchors. ``eta_i`` is drawn
    ``N(0, max(0, sigma2_perm - Var_train[mu_hat(anchor)]))`` per holdout
    person, ordered by ``person_id``, from a generator seeded from the
    gate seed. Returns ``(m_person, diagnostics)`` where ``m_person`` has
    ``person_id`` and ``m`` (ordered by ``person_id``) and the
    diagnostics report the variance bookkeeping (all reported-not-gated).
    """
    s2_perm = float(fit["sigma2_perm"])

    ta = train_anchor.sort_values("person_id").reset_index(drop=True)
    mu_train = conditional_median(fitted_perm, ta)
    var_mu_train = float(np.var(mu_train))
    eta_var = max(0.0, s2_perm - var_mu_train)
    eta_sd = float(np.sqrt(eta_var))

    ha = holdout_anchor.sort_values("person_id").reset_index(drop=True)
    mu_hold = conditional_median(fitted_perm, ha)

    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, eta_sd, size=len(ha))
    m = mu_hold + eta

    m_person = pd.DataFrame({"person_id": ha["person_id"].to_numpy(), "m": m})
    diagnostics = {
        "sigma2_perm": s2_perm,
        "var_mu_hat_train": var_mu_train,
        "eta_variance": eta_var,
        "eta_sd": eta_sd,
        "var_mu_hat_holdout": float(np.var(mu_hold)),
        "mean_mu_hat_train": float(np.mean(mu_train)),
        "mean_mu_hat_holdout": float(np.mean(mu_hold)),
        "realized_var_m": float(np.var(m)),
        "n_train_anchors": int(len(ta)),
        "n_holdout_anchors": int(len(ha)),
    }
    return m_person, diagnostics


# --------------------------------------------------------------------------
# Generated-vs-train log-variance diagnostic (reported, not gated)
# --------------------------------------------------------------------------
def generated_log_variance(
    candidate: pd.DataFrame,
    train: pd.DataFrame,
) -> dict[str, float]:
    """Realized generated log-earnings variance vs the train log variance.

    Unweighted population variance of log positive earnings on the
    generated candidate panel and on the train split (both among positive
    observations). Reported for comparability; not gated.
    """
    cg = candidate.loc[candidate["earnings"] > 0, "earnings"].to_numpy(
        dtype=np.float64
    )
    tr = train.loc[train["earnings"] > 0, "earnings"].to_numpy(
        dtype=np.float64
    )
    var_gen = float(np.var(np.log(cg))) if cg.size else float("nan")
    var_train = float(np.var(np.log(tr))) if tr.size else float("nan")
    return {
        "var_log_earnings_generated": var_gen,
        "var_log_earnings_train": var_train,
        "ratio_generated_over_train": (
            var_gen / var_train if var_train > 0 else float("nan")
        ),
        "n_positive_generated": int(cg.size),
        "n_positive_train": int(tr.size),
    }


def perm_share_diagnostic(
    beta: np.ndarray,
    train: pd.DataFrame,
    m_person: pd.DataFrame,
    holdout_anchor: pd.DataFrame,
    fit: dict[str, Any],
) -> dict[str, float]:
    """Cross-person ``var(m)`` over the Stage-0 train-residual variance.

    The candidate-4 analogue of the prior candidates' drawn-perm-share
    diagnostic: the realized cross-person variance of the assembled person
    effect ``m`` (unweighted primary, anchor-weighted secondary) over the
    variance of the Stage-0 train residuals, reported alongside the
    structural ``implied_perm_share = sigma2_perm / gamma_0`` for
    comparability with v1/v2. By construction ``var(m)`` targets
    ``sigma2_perm``, so ``perm_share`` here approximates
    ``sigma2_perm / var_residual_train``. None is gated.
    """
    resid = add_residuals(train, beta)
    r_train = resid.loc[resid["is_pos"], "r"].to_numpy(dtype=np.float64)
    var_residual_train = float(np.var(r_train))

    mm = m_person.merge(
        holdout_anchor[["person_id", "weight"]], on="person_id", how="left"
    )
    m = mm["m"].to_numpy(dtype=np.float64)
    w = mm["weight"].to_numpy(dtype=np.float64)
    var_m = float(np.var(m))
    mean_w = float(np.average(m, weights=w))
    var_m_weighted = float(np.average((m - mean_w) ** 2, weights=w))

    perm_share = var_m / var_residual_train if var_residual_train > 0 else 0.0
    perm_share_weighted = (
        var_m_weighted / var_residual_train if var_residual_train > 0 else 0.0
    )
    return {
        "perm_share": perm_share,
        "perm_share_weighted": perm_share_weighted,
        "var_drawn_perm": var_m,
        "var_drawn_perm_weighted": var_m_weighted,
        "var_residual_train": var_residual_train,
        "implied_perm_share": float(fit["implied_perm_share"]),
        "memo_backout_perm_share": MEMO_PERM_SHARE_BACKOUT,
    }


# --------------------------------------------------------------------------
# Driver - candidate 4 keeps stages 0-1, replaces generation
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
    """Fit, generate, and score candidate 4 for a single gate seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    all_anchor = anchor_rows(panel)
    train_ids = train["person_id"].unique()
    holdout_ids = holdout["person_id"].unique()
    train_anchor = all_anchor[all_anchor.person_id.isin(train_ids)]
    holdout_anchor = all_anchor[all_anchor.person_id.isin(holdout_ids)]

    # Stage 0 (imported): weighted OLS quadratic-in-age residualiser.
    beta = fit_age_residualizer(train)
    resid_train = add_residuals(train, beta)
    # resid_pos carries person_id, period, r for stage 1; augment with age
    # and weight for the transitory pairs.
    resid_pos_full = resid_train.loc[
        resid_train["is_pos"],
        ["person_id", "period", "age", "weight", "r"],
    ].copy()
    resid_pos = resid_pos_full[["person_id", "period", "r"]]

    # Stage 1 (imported from candidate 3): persistence-aware decomposition.
    gamma, gamma_counts, pooled_mean = pooled_autocovariances(resid_pos)
    fit = fit_three_component(gamma)
    perm_train = person_effects_pa(resid_pos, fit, pooled_mean, train_ids)

    # --- Generation (NEW): structural three-component assembly. ---
    # Participation gate: baseline transition model, gate seed s.
    part_pairs = build_backward_pairs(train)
    fitted_part = fit_participation_model(part_pairs, seed)

    # Person effect: QRF_perm (candidate-2 perm model) conditional median
    # plus eta so cross-person var(m) == sigma2_perm.
    fitted_perm = fit_perm_model(train_anchor, perm_train, seed)
    m_person, m_diag = assemble_person_effect(
        fitted_perm, train_anchor, holdout_anchor, fit, seed
    )

    # Transitory chain: QRF_t on t = r - perm_hat over positive pairs.
    t_pairs = build_transitory_pairs(resid_pos_full, perm_train)
    fitted_t = fit_transitory_model(t_pairs, seed)

    candidate = generate_candidate_structural(
        holdout, beta, m_person, fitted_t, fitted_part, fit, seed
    )

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

    # --- diagnostics (reported, not gated) ---
    perm_diag = perm_share_diagnostic(
        beta, train, m_person, holdout_anchor, fit
    )
    logvar_diag = generated_log_variance(candidate, train)

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
        "n_train_pairs": int(len(part_pairs)),
        "n_transitory_pairs": int(len(t_pairs)),
        "n_windows": n_windows,
        "regimes": {
            "perm_model": fitted_perm.regimes(),
            "participation": fitted_part.regimes(),
            "transitory": fitted_t.regimes(),
        },
        "stage1_fit": stage1_fit,
        "person_effect_diagnostic": m_diag,
        "generated_log_variance": logvar_diag,
        "perm_share_diagnostic": perm_diag,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if verbose:
        d = perm_diag
        b = battery_values
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"rho={fit['rho']:.2f} "
            f"var(m)={d['var_drawn_perm']:.3f} "
            f"(sig2p {fit['sigma2_perm']:.3f}) "
            f"ac10={b['autocorr_log_10yr']:.3f} "
            f"ac4={b['autocorr_log_4yr']:.3f} "
            f"ac2={b['autocorr_log_2yr']:.3f} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-4 run."""
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

    # Identical battery-reference bit-exact precheck as the prior runs.
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
        "run": "gate1_qrf_structural_v1",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "candidate2_registration": CANDIDATE2_REGISTRATION,
        "candidate3_registration": CANDIDATE3_REGISTRATION,
        "builds_on": (
            "stages 0-1 identical to candidate 3 (imported); generation "
            "replaced by structural three-component assembly"
        ),
        "description": (
            "Gate-1 candidate 4: structural three-component generation. "
            "Keeps candidate 3's stages 0-1 verbatim (imported): the "
            "weighted-OLS age residualiser and the persistence-aware "
            "permanent-component decomposition (permanent + AR(1)-"
            "persistent transitory + white noise) of pooled within-person "
            "autocovariances, estimated on each seed's train split. The "
            "GENERATION is new and structural: each positive log earnings "
            "is assembled as f_hat(age) + m_i + t_it + eps_it, where m_i "
            "is the QRF_perm conditional median at the anchor plus Gaussian "
            "eta calibrated so cross-person var(m) equals sigma2_perm; "
            "t_it is a backward one-step QRF_t chain of the transitory "
            "residual t = r - perm_hat started at t_anchor = r_anchor - "
            "m_i; and eps_it is i.i.d. N(0, sigma2_noise) per generated "
            "positive observation (none at the anchor). Participation "
            "(zero vs positive) is the baseline transition's regime gate "
            "conditioned on the next generated level and age, unchanged "
            "from the prior candidates' machinery. The anchor keeps its "
            "real value. Registered frozen before the run in issue #42's "
            "candidate-4 comment (see spec_registration). Candidate scored "
            "against the held-out PSID family earnings panel geometry (two "
            "locked views) and the locked moment battery, per the locked "
            "seed-level conjunction in gates.yaml (pull request 39). The "
            "protocol machinery is imported byte-for-byte from the "
            "candidate-3 runner (pull request 44), which imports it from "
            "candidate 2 (pull request 43) / the baseline (pull request "
            "40)."
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
                "note": "unchanged from candidate 3 (imported)",
            },
            "stage_1_person_effect_decomposition": {
                "note": (
                    "persistence-aware three-component decomposition, "
                    "identical to candidate 3 (imported): pooled unweighted "
                    "within-person autocovariances gamma_k (biennial lags "
                    "0..5) on the train residuals centred at the pooled "
                    "mean; gamma_0 = sigma2_perm + sigma2_trans + "
                    "sigma2_noise, gamma_k = sigma2_perm + sigma2_trans * "
                    "rho**k; grid rho in {0.50..0.95 step 0.01}, NNLS the "
                    "three variances at each rho, min moment SSE; person "
                    "effect perm_hat_i = w_i * mean_i(rc) with "
                    "correlated-noise shrinkage. Only the decomposition "
                    "and perm_hat are reused; the generation is new."
                ),
                "rho_grid": list(RHO_GRID),
                "gamma_lags_biennial": list(GAMMA_LAGS),
            },
            "generation_participation": {
                "class": "populace.fit.qrf.RegimeGatedQRF (defaults)",
                "channel": "zero vs positive (the regime gate only)",
                "fit_model": (
                    "baseline transition: target earnings_tm2, predictors "
                    "(earnings at t, age at t-2), sample_weight = t-2 "
                    "person-period weight, fit on the 80% complement's "
                    "adjacent 2-year pairs"
                ),
                "draw": (
                    "per backward step, the gate draws zero/positive "
                    "conditioned on the next period's GENERATED earnings "
                    "level and the earlier period's age; only the gate is "
                    "used, not the magnitude forests"
                ),
                "note": "unchanged from the prior candidates' machinery",
            },
            "generation_magnitude": {
                "scale": "log",
                "assembly": "exp(f_hat(age) + m_i + t_it + eps_it)",
                "person_effect_m": {
                    "location": (
                        "mu_hat = conditional MEDIAN of QRF_perm at the "
                        "anchor row (deterministic mixture-CDF inversion "
                        "at 0.5)"
                    ),
                    "qrf_perm": (
                        "candidate-2 perm-draw model: target perm_hat_i, "
                        "predictors (anchor earnings level, anchor age), "
                        "trained on train anchors, populace-fit defaults, "
                        "seed = gate seed"
                    ),
                    "eta": (
                        "eta_i ~ N(0, max(0, sigma2_perm - "
                        "Var_train[mu_hat(anchor)])); Var_train over train "
                        "persons' anchors; one eta per holdout person, "
                        "ordered by person_id"
                    ),
                    "construction": (
                        "cross-person var(m) equals sigma2_perm by "
                        "construction; the anchor informs the location only"
                    ),
                },
                "transitory_t": {
                    "definition": "t = r - perm_hat on train positive rows",
                    "qrf_t": (
                        "RegimeGatedQRF (defaults, seed = gate seed) fitting "
                        "t_prev | (t_next, age_prev) on adjacent backward "
                        "biennial pairs of positive train observations, "
                        "sample_weight = earlier-period weight"
                    ),
                    "chain": (
                        "t_anchor,i = r_anchor,i - m_i (absorbs the anchor's "
                        "observation noise, documented); chain backward "
                        "one step over observed periods, one step across "
                        "gaps"
                    ),
                },
                "noise_eps": (
                    "eps_it ~ N(0, sigma2_noise) i.i.d. per generated "
                    "positive observation; NONE at the anchor (which keeps "
                    "its real value)"
                ),
                "anchor": (
                    "chronologically last observed period held at real "
                    "earnings; contributes no eps"
                ),
            },
            "generation_order": {
                "row_order": (
                    "batched by step-from-anchor; within a step ordered by "
                    "person_id (the baseline chain order)"
                ),
                "rng": (
                    "all RNG seeded from the gate seed s: eta (one per "
                    "holdout person, person_id order) and eps (per positive "
                    "generated obs, step/person order) from one generator; "
                    "QRF_t / participation-gate draws from their freshly "
                    "fitted models seeded from s"
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
                "(imported from the baseline runner via candidates 2/3)"
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
        "generation_diagnostics_context": {
            "memo_backout_perm_share": MEMO_PERM_SHARE_BACKOUT,
            "candidate2_drawn_perm_share_range": [0.52, 0.55],
            "candidate3_drawn_perm_share_range": [0.21, 0.27],
            "note": (
                "Reported-not-gated generation diagnostics. Per seed: the "
                "stage-1 fit (gamma_k, chosen rho, the three component "
                "variances, implied_perm_share = sigma2_perm/gamma_0), the "
                "person-effect variance bookkeeping (Var_train[mu_hat(anchor"
                ")], eta variance, realized cross-person var(m)), the "
                "realized generated log-variance vs train, and the "
                "perm_share = var(m)/var(Stage-0 train residuals) analogue "
                "reported for candidates 2 and 3 (whose drawn shares were "
                "0.52-0.55 and 0.21-0.27). NONE is gated; the gate rule "
                "names only geometry AND battery."
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
                    "var_mu_hat_train": s["person_effect_diagnostic"][
                        "var_mu_hat_train"
                    ],
                    "eta_variance": s["person_effect_diagnostic"][
                        "eta_variance"
                    ],
                    "realized_var_m": s["person_effect_diagnostic"][
                        "realized_var_m"
                    ],
                    "var_log_earnings_generated": s["generated_log_variance"][
                        "var_log_earnings_generated"
                    ],
                    "var_log_earnings_train": s["generated_log_variance"][
                        "var_log_earnings_train"
                    ],
                    "ratio_generated_over_train": s["generated_log_variance"][
                        "ratio_generated_over_train"
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
