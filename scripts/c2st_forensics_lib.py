"""Forensics for the gate's C2ST classifier: what it reads at 0.547.

Reported-not-gated, NO holdout contact. This library never touches the
holdout: every classifier here compares a candidate panel against the
seed-0 TRAIN persons' real windows (or one candidate against another, or
two disjoint train halves). The locked gate is untouched.

The gate's distinguishability metric is
:func:`populace_dynamics.harness.metrics.classifier_two_sample_auc`: a
``HistGradientBoostingClassifier(random_state=seed)`` trained to tell real
window rows from candidate window rows, scored by weighted 5-fold
stratified-CV ROC AUC, with sample weights normalized so each side carries
equal total mass. The features are the projected window columns in
:attr:`PanelView.dimension_names` order -- for the window=2 pairs view
``(earnings_t0, earnings_t1, age)`` (``t0`` is the LATER period, ``t1`` the
EARLIER; the harness lifts ahead rows), for window=3 runs
``(earnings_t0, earnings_t1, earnings_t2, age)``. Weights are not features.

Every classifier in this module is that classifier, mirrored bit for bit
(same estimator, same defaults, same fold seed, same equal-mass weighting,
same real-first / candidate-second argument order the harness uses in
``score_view``). The AUC is NOT symmetric in that order, so the order is
pinned everywhere: ``real`` is class 0, ``candidate`` is class 1.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from populace_dynamics.harness import panel as hpanel

# Round-number ladders probed for the PSID whole-dollar lattice.
ROUND_BASES: tuple[int, ...] = (100, 1000, 5000)


# --------------------------------------------------------------------------
# The mirrored classifier (harness-exact) + projection helpers
# --------------------------------------------------------------------------
def project(
    table: pd.DataFrame, view: hpanel.PanelView
) -> tuple[np.ndarray, np.ndarray]:
    """Project a person-period table into the view's window matrix.

    Thin wrapper on :func:`populace_dynamics.harness.panel.project_panel`
    so callers read points/weights in ``view.dimension_names`` order.
    """
    return hpanel.project_panel(table, view)


def _cv_auc_on_columns(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    columns: list[int],
    *,
    seed: int = 0,
) -> float:
    """Weighted 5-fold stratified-CV AUC on a column subset of ``x``.

    Identical to the gate's inner loop
    (:func:`classifier_two_sample_auc`): same estimator and defaults, same
    fold seed, same weighted ``roc_auc_score``. ``columns`` restricts the
    feature set so the same routine yields the full-feature AUC, a marginal
    (single column), or an interaction (a column pair).
    """
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in folds.split(x, y):
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(
            x[np.ix_(train_idx, columns)],
            y[train_idx],
            sample_weight=w[train_idx],
        )
        scores = model.predict_proba(x[np.ix_(test_idx, columns)])[:, 1]
        aucs.append(
            float(
                roc_auc_score(y[test_idx], scores, sample_weight=w[test_idx])
            )
        )
    return float(np.mean(aucs))


def _stack(
    real_points: np.ndarray,
    real_weights: np.ndarray,
    cand_points: np.ndarray,
    cand_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack real (class 0) over candidate (class 1), equal class mass.

    Mirrors the gate exactly: real first, candidate second, each side's
    weights renormalized to sum to one so the test scores shape, not the
    2048-vs-N size difference between a train pool and a candidate.
    """
    x = np.vstack([real_points, cand_points])
    y = np.r_[np.zeros(len(real_points)), np.ones(len(cand_points))]
    w = np.r_[
        real_weights / real_weights.sum(),
        cand_weights / cand_weights.sum(),
    ]
    return x, y, w


def c2st_auc(
    real: pd.DataFrame,
    candidate: pd.DataFrame,
    view: hpanel.PanelView,
    *,
    seed: int = 0,
) -> float:
    """Gate-exact C2ST AUC of ``candidate`` vs ``real`` on one view.

    Projects both frames through ``view`` and calls the mirrored inner
    loop on all columns. Equal to
    ``metrics.classifier_two_sample_auc(real_points, cand_points, ...)``
    (verified against the harness in the driver's sanity anchor).
    """
    rp, rw = project(real, view)
    cp, cw = project(candidate, view)
    x, y, w = _stack(rp, rw, cp, cw)
    return _cv_auc_on_columns(x, y, w, list(range(x.shape[1])), seed=seed)


# --------------------------------------------------------------------------
# Analysis 2: feature attribution (marginal + pairwise-interaction AUCs)
# --------------------------------------------------------------------------
def feature_attribution(
    real: pd.DataFrame,
    candidate: pd.DataFrame,
    view: hpanel.PanelView,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Full, per-feature marginal, and per-pair AUCs on one view.

    Trains the mirrored classifier on the full feature set, then on each
    single feature (its marginal separation power), then on each feature
    pair (interaction). Also reports the earnings-only subset (all
    ``earnings_t*`` columns, dropping age) so a joint-in-earnings signal is
    visible without the age axis. Returns a dict keyed by feature/pair
    name; every value is an AUC on the same stacked, equal-mass problem.
    """
    dims = view.dimension_names
    rp, rw = project(real, view)
    cp, cw = project(candidate, view)
    x, y, w = _stack(rp, rw, cp, cw)
    n = len(dims)

    full = _cv_auc_on_columns(x, y, w, list(range(n)), seed=seed)
    marginals = {
        dims[i]: _cv_auc_on_columns(x, y, w, [i], seed=seed) for i in range(n)
    }
    pairs = {
        f"{dims[a]}+{dims[b]}": _cv_auc_on_columns(x, y, w, [a, b], seed=seed)
        for a, b in itertools.combinations(range(n), 2)
    }
    earn_cols = [i for i, d in enumerate(dims) if d.startswith("earnings_t")]
    earnings_only = (
        _cv_auc_on_columns(x, y, w, earn_cols, seed=seed)
        if len(earn_cols) >= 2
        else float("nan")
    )
    return {
        "dimension_names": list(dims),
        "n_windows_real": int(len(rp)),
        "n_windows_candidate": int(len(cp)),
        "full_auc": full,
        "marginal_auc": marginals,
        "pairwise_auc": pairs,
        "earnings_only_auc": earnings_only,
    }


# --------------------------------------------------------------------------
# Analysis 3/4: distributional forensics (levels, joint, round numbers)
# --------------------------------------------------------------------------
def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Weighted quantiles (harness convention, midpoint plotting position)."""
    order = np.argsort(values)
    v = values[order]
    ww = weights[order]
    cumulative = np.cumsum(ww)
    positions = (cumulative - 0.5 * ww) / cumulative[-1]
    return np.interp(q, positions, v)


def round_number_signature(
    values: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Round-number footprint of a positive-earnings sample, weighted.

    PSID earnings are whole dollars, heavily clustered on round numbers;
    the generated columns need not be. Reports the weighted share of
    positive values that are non-integer and that are multiples of each
    :data:`ROUND_BASES` ladder, plus the distinct-value ratio (unweighted;
    generation that interpolates inflates the count of distinct values).
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    pos = v > 0
    v = v[pos]
    w = w[pos]
    if v.size == 0:
        return {"n_positive": 0}
    total = w.sum()
    out: dict[str, float] = {
        "n_positive": int(v.size),
        "non_integer_share": float(w[np.mod(v, 1) != 0].sum() / total),
        "distinct_ratio": float(np.unique(v).size / v.size),
    }
    for base in ROUND_BASES:
        out[f"mult_{base}_share"] = float(
            w[np.mod(v, base) == 0].sum() / total
        )
    return out


def level_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    q: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99),
) -> dict[str, float]:
    """Weighted quantiles of an earnings column (marginal shape)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    qq = _weighted_quantile(v, w, np.asarray(q))
    return {
        f"q{int(round(level * 100)):02d}": float(val)
        for level, val in zip(q, qq, strict=True)
    }


def joint_dependence(
    points: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Persistence of the (later, earlier) earnings pair, weighted.

    On the both-positive windows: the weighted correlation of
    ``log(earnings_t0)`` and ``log(earnings_t1)`` (year-to-year
    persistence), the mean and sd of the log change ``log(t0/t1)``, and
    the both-positive share. A rank-transition kernel that flattens the
    fine joint shows a persistence correlation below the real one while
    matching the marginals; a splice that rescales donor segments can
    change the log-change dispersion and the tail. ``points`` columns are
    ``(t0, t1, age)`` (later, earlier, covariate).
    """
    t0 = points[:, 0]
    t1 = points[:, 1]
    both = (t0 > 0) & (t1 > 0)
    w = weights[both]
    out: dict[str, float] = {
        "both_positive_share": float(weights[both].sum() / weights.sum()),
    }
    if w.sum() <= 0 or both.sum() < 2:
        return out
    lt0 = np.log(t0[both])
    lt1 = np.log(t1[both])
    change = lt0 - lt1
    m = np.average(change, weights=w)
    out["logchange_mean"] = float(m)
    out["logchange_sd"] = float(
        np.sqrt(np.average((change - m) ** 2, weights=w))
    )
    m0 = np.average(lt0, weights=w)
    m1 = np.average(lt1, weights=w)
    cov = np.average((lt0 - m0) * (lt1 - m1), weights=w)
    s0 = np.sqrt(np.average((lt0 - m0) ** 2, weights=w))
    s1 = np.sqrt(np.average((lt1 - m1) ** 2, weights=w))
    out["log_persistence_corr"] = float(cov / (s0 * s1))
    return out


def distributional_forensics(
    real: pd.DataFrame,
    candidate: pd.DataFrame,
    view: hpanel.PanelView,
) -> dict[str, Any]:
    """Distribution-side forensics for one candidate vs the train windows.

    Bundles the round-number signature and marginal quantiles of the
    earlier-period earnings column, plus the joint-dependence summary, for
    both the real (train) and the candidate projections -- so the report
    reads the defect off matched numbers, not off the AUC alone.
    """
    rp, rw = project(real, view)
    cp, cw = project(candidate, view)
    # Earlier-period earnings column index (t1 for pairs; the last-but-one
    # offset is the generated one nearest the anchor). Use t1 uniformly.
    dims = list(view.dimension_names)
    t1_idx = dims.index("earnings_t1")
    return {
        "earlier_col": "earnings_t1",
        "round_number": {
            "train": round_number_signature(rp[:, t1_idx], rw),
            "candidate": round_number_signature(cp[:, t1_idx], cw),
        },
        "level_quantiles": {
            "train": level_quantiles(rp[:, t1_idx], rw),
            "candidate": level_quantiles(cp[:, t1_idx], cw),
        },
        "joint_dependence": {
            "train": joint_dependence(rp, rw),
            "candidate": joint_dependence(cp, cw),
        },
    }


def rounding_repair_ablation(
    real: pd.DataFrame,
    candidate: pd.DataFrame,
    view: hpanel.PanelView,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Does snapping candidate earnings to a round lattice move the AUC?

    Rebuilds the candidate with positive earnings snapped to the nearest
    multiple of each :data:`ROUND_BASES` base and re-scores C2ST vs the
    train windows. If the round-number break were what the classifier
    reads, snapping would collapse the AUC toward the noise floor; a
    tree-based classifier that splits on thresholds is largely blind to
    the lattice, so this ablation quantifies how much (little) of 0.547 is
    the round-number footprint.
    """
    out = {"raw": c2st_auc(real, candidate, view, seed=seed)}
    for base in ROUND_BASES:
        snapped = candidate.copy()
        e = snapped["earnings"].to_numpy(dtype=np.float64).copy()
        pos = e > 0
        e[pos] = np.round(e[pos] / base) * base
        snapped["earnings"] = e
        out[f"snap_{base}"] = c2st_auc(real, snapped, view, seed=seed)
    return out


def decision_region_probe(
    real: pd.DataFrame,
    candidate: pd.DataFrame,
    view: hpanel.PanelView,
    *,
    top_decile: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    """Characterize the windows the classifier is most sure are candidate.

    Trains the mirrored classifier on the full stacked problem (single
    fit, no CV -- this is description, not scoring), then takes the
    candidate windows in the top ``top_decile`` of predicted
    candidate-probability and contrasts them with the rest of the
    candidate windows and with the train windows on the interpretable
    coordinates: earlier-period earnings quantiles, both-positive share,
    log persistence, and the round-number signature. The confident region
    is where the residual signal concentrates, so its profile names the
    signal.
    """
    dims = list(view.dimension_names)
    t1_idx = dims.index("earnings_t1")
    rp, rw = project(real, view)
    cp, cw = project(candidate, view)
    x, y, w = _stack(rp, rw, cp, cw)
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(x, y, sample_weight=w)
    cand_scores = model.predict_proba(cp)[:, 1]
    thr = np.quantile(cand_scores, 1.0 - top_decile)
    top = cand_scores >= thr
    rest = ~top

    def _profile(pts: np.ndarray, ww: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(len(pts)),
            "earlier_quantiles": level_quantiles(pts[:, t1_idx], ww),
            "round_number": round_number_signature(pts[:, t1_idx], ww),
            "joint_dependence": joint_dependence(pts, ww),
        }

    return {
        "top_decile_threshold": float(thr),
        "top_decile_mean_prob": float(cand_scores[top].mean()),
        "candidate_top_decile": _profile(cp[top], cw[top]),
        "candidate_rest": _profile(cp[rest], cw[rest]),
        "train_reference": _profile(rp, rw),
    }


# --------------------------------------------------------------------------
# Analysis 5: shared-vs-distinct signal + noise anchor
# --------------------------------------------------------------------------
def shared_signal_test(
    splice: pd.DataFrame,
    kernel: pd.DataFrame,
    train: pd.DataFrame,
    view: hpanel.PanelView,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Is the two candidates' residual the SAME signal or coincident AUC?

    * ``splice_vs_kernel``: the mirrored classifier trained to tell the two
      candidates apart (splice as class 0, kernel as class 1 -- pinned).
      Near 0.5 means the residual each leaves relative to real is the same
      direction (a shared signal); well above 0.5 means they are distinct
      defects that merely land at similar candidate-vs-real AUC.
    * ``noise_floor``: two person-disjoint halves of the TRAIN panel scored
      against each other, the real-vs-real sampling-noise anchor (the same
      split routine the harness's ``noise_floor`` uses, but on train, never
      the holdout).
    * ``splice_vs_train`` / ``kernel_vs_train``: each candidate's own
      separation from the train windows, for comparison.
    """
    left, right = hpanel.split_panel_by_person(
        train, "person_id", fraction=0.5, seed=seed
    )
    return {
        "splice_vs_kernel": c2st_auc(splice, kernel, view, seed=seed),
        "noise_floor": c2st_auc(left, right, view, seed=seed),
        "splice_vs_train": c2st_auc(train, splice, view, seed=seed),
        "kernel_vs_train": c2st_auc(train, kernel, view, seed=seed),
    }
