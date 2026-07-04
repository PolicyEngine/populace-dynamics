"""Weighted evaluation metrics for imputed target distributions.

Adapted from PolicyEngine/imputation-paper
(src/imputation_paper/experiments/metrics.py), the population-view harness
described in that paper; extended here for longitudinal views.

An imputation is a draw from an estimated conditional distribution, so the
metrics score *distributional* fidelity, not point accuracy, and they do so
under survey weights -- the property the paper argues common practice neglects.

Marginal metrics (scored per target):

* :func:`weighted_pinball_loss` -- weighted quantile (pinball) loss over a
  quantile grid, the primary distributional-calibration metric.
* :func:`weighted_wasserstein1` -- weighted Wasserstein-1 distance from the
  imputed sample to a donor sample, a single-number marginal-fit summary.
* :func:`zero_share_error` -- absolute error in the weighted share of exact
  zeros, the diagnostic for zero-inflated targets.

Joint metrics (scored on the multivariate imputed block; the population-view
harness in :mod:`populace_dynamics.harness.views` runs on these):

* :func:`energy_distance` -- weighted energy distance, a strictly proper
  sample-based score for the joint: hedging toward modal households is
  penalized by construction.
* :func:`prdc` -- precision/recall/density/coverage (Naeem et al. 2020);
  coverage is the explicit anti-mode-collapse axis, and its support-based
  variant is invariant to any reweighting of the candidate.
* :func:`classifier_two_sample_auc` -- weighted classifier two-sample test,
  the omnibus "can anything tell the files apart" check (0.5 = indistinguishable).

Stress metric:

* :func:`reweight_fragility` -- worst-case single-record share of a population
  aggregate over a bounded family of reweightings (the "landmine" diagnostic).

All weighted reductions accept ``weights=None`` for the unweighted case, so the
same code paths serve the weighted/unweighted ablation.
"""

from __future__ import annotations

import numpy as np

#: Default quantile grid for :func:`weighted_pinball_loss`. Deciles avoid the
#: exact 0/1 endpoints (where pinball loss is dominated by a single tail order
#: statistic) while spanning the distribution.
DEFAULT_QUANTILE_GRID: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


def _as_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    """Return a validated non-negative weight vector of length ``n``.

    ``None`` becomes uniform ones (the unweighted case). Raises on a length
    mismatch, negative weights, or an all-zero vector, so a metric never
    silently divides by a zero total.
    """
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must not sum to zero.")
    return w


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Weighted quantiles of ``values`` at levels ``q``.

    Uses the standard cumulative-weight interpolation: sort by value, form the
    normalized cumulative weight at the midpoint of each atom, and linearly
    interpolate the value at each requested level. Reduces to the usual linear
    (``numpy.quantile`` "linear") quantile when weights are equal.
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    # Midpoint (type-7-like) plotting positions on the weighted CDF.
    positions = (cumulative - 0.5 * w) / total
    return np.interp(q, positions, v)


def weighted_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILE_GRID,
) -> float:
    """Weighted pinball (quantile) loss, averaged over a quantile grid.

    For each level ``tau`` the loss compares the ``tau``-quantile of the
    *predicted* (imputed) sample against every held-out truth value with the
    asymmetric check function ``rho_tau(u) = u * (tau - 1{u < 0})``, weighted by
    the receiver weights, then averages the per-level weighted means over the
    grid. Lower is better.

    This scores the imputed *distribution*: a method that recovers the receiver
    population's conditional quantiles scores low even if no single row is
    predicted exactly. Because a single stochastic draw does not itself carry
    quantiles, the predicted quantiles are read from the pooled draw here; the
    per-method sweep can instead pass a method's own predicted-quantile columns.

    Args:
        y_true: Held-out truth values (receiver), shape ``(n,)``.
        y_pred: Imputed values whose quantiles are evaluated, shape ``(m,)``.
        weights: Receiver weights aligned to ``y_true`` (``None`` for uniform).
        quantiles: Quantile levels to score.

    Returns:
        The grid-averaged weighted pinball loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = _as_weights(weights, y_true.shape[0])
    q = np.asarray(quantiles, dtype=np.float64)
    pred_quantiles = np.quantile(y_pred, q)

    total = w.sum()
    losses = np.empty(q.shape[0], dtype=np.float64)
    for i, (tau, q_hat) in enumerate(zip(q, pred_quantiles, strict=True)):
        residual = y_true - q_hat
        check = residual * (tau - (residual < 0.0))
        losses[i] = float(np.dot(w, check) / total)
    return float(losses.mean())


def weighted_wasserstein1(
    imputed: np.ndarray,
    donor: np.ndarray,
    *,
    imputed_weights: np.ndarray | None = None,
    donor_weights: np.ndarray | None = None,
    n_grid: int = 1000,
) -> float:
    """Weighted Wasserstein-1 (earth-mover) distance between two 1-D samples.

    Computes ``W_1(P, Q) = integral over u in (0,1) of |F_P^{-1}(u) -
    F_Q^{-1}(u)| du`` by evaluating both weighted quantile functions on a shared
    grid of ``u`` levels and integrating their absolute difference (trapezoidal).
    This is the weighted analogue of the L1 distance between inverse CDFs and
    reduces to :func:`scipy.stats.wasserstein_distance` when both weight vectors
    are uniform. Lower is better.

    Args:
        imputed: Imputed target values.
        donor: Reference (donor) target values.
        imputed_weights: Weights for ``imputed`` (``None`` for uniform).
        donor_weights: Weights for ``donor`` (``None`` for uniform).
        n_grid: Number of interior ``u`` levels to integrate over.

    Returns:
        The weighted Wasserstein-1 distance.
    """
    imputed = np.asarray(imputed, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    wi = _as_weights(imputed_weights, imputed.shape[0])
    wd = _as_weights(donor_weights, donor.shape[0])
    # Interior grid avoids the degenerate exact-0/1 endpoints of the inverse CDF.
    u = (np.arange(n_grid, dtype=np.float64) + 0.5) / n_grid
    qi = _weighted_quantile(imputed, wi, u)
    qd = _weighted_quantile(donor, wd, u)
    return float(np.trapezoid(np.abs(qi - qd), u))


def zero_share_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    true_weights: np.ndarray | None = None,
    pred_weights: np.ndarray | None = None,
    zero_atol: float = 1e-6,
) -> float:
    """Absolute error in the weighted share of exact zeros.

    Zero-inflated targets (income components, credits, gains) carry a mass at
    zero that a method must reproduce: too few zeros inflates a program's
    caseload, too many erases it. This returns ``|zero_share(pred) -
    zero_share(true)|`` under weights, where a value is a zero when its magnitude
    is at or below ``zero_atol``. Lower is better; ``0`` means the imputed and
    true zero masses match.

    Args:
        y_true: Held-out truth values.
        y_pred: Imputed values.
        true_weights: Weights for ``y_true`` (``None`` for uniform).
        pred_weights: Weights for ``y_pred`` (``None`` for uniform).
        zero_atol: Magnitudes at or below this count as zeros.

    Returns:
        The absolute weighted zero-share error, in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    wt = _as_weights(true_weights, y_true.shape[0])
    wp = _as_weights(pred_weights, y_pred.shape[0])
    true_share = float(np.dot(wt, np.abs(y_true) <= zero_atol) / wt.sum())
    pred_share = float(np.dot(wp, np.abs(y_pred) <= zero_atol) / wp.sum())
    return abs(pred_share - true_share)


def _as_points(values: np.ndarray) -> np.ndarray:
    """Coerce a 1-D or 2-D array to float64 points of shape ``(n, d)``."""
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2:
        raise ValueError(
            f"Points must be 1-D or 2-D, got shape {points.shape}."
        )
    return points


def _standardize_by(
    points: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardize columns by given moments; constant columns pass through."""
    safe_std = np.where(std > 0, std, 1.0)
    return (points - mean) / safe_std


def _weighted_moments(
    points: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted per-column mean and standard deviation."""
    total = weights.sum()
    mean = (weights[:, None] * points).sum(axis=0) / total
    variance = (weights[:, None] * (points - mean) ** 2).sum(axis=0) / total
    return mean, np.sqrt(variance)


def _resample_to_cap(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Weight-proportional resample down to ``cap`` points (uniform weights out).

    Pairwise-distance metrics are O(n*m) in memory; above the cap we draw a
    weight-proportional sample with replacement, which converts the weighted
    empirical measure into a uniform one of manageable size. Below the cap the
    exact weighted computation is used (no Monte Carlo noise).
    """
    if len(points) <= cap:
        return points, weights
    picks = rng.choice(
        len(points), size=cap, replace=True, p=weights / weights.sum()
    )
    return points[picks], np.ones(cap, dtype=np.float64)


def energy_distance(
    imputed: np.ndarray,
    holdout: np.ndarray,
    *,
    imputed_weights: np.ndarray | None = None,
    holdout_weights: np.ndarray | None = None,
    standardize: bool = True,
    max_points: int = 2048,
    seed: int = 0,
) -> float:
    """Weighted squared energy distance between two multivariate samples.

    ``D^2(P, Q) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` with expectations
    under the weighted empirical measures. It is non-negative and zero iff the
    distributions coincide, and the corresponding scoring rule is strictly
    proper -- a candidate concentrated on modal households scores worse than one
    matching the full distribution, which is exactly the anti-mode-collapse
    property the population-view harness needs. Columns are standardized by the
    *holdout's* weighted moments so the value is comparable across views with
    different variable scales. Lower is better.

    Args:
        imputed: Candidate points, shape ``(m,)`` or ``(m, d)``.
        holdout: Held-out reference points, shape ``(n,)`` or ``(n, d)``.
        imputed_weights: Candidate weights (``None`` for uniform).
        holdout_weights: Holdout weights (``None`` for uniform).
        standardize: Standardize columns by the holdout's weighted moments.
        max_points: Cap on points per side; above it a weight-proportional
            resample (seeded) bounds the pairwise-distance cost.
        seed: Seed for the resample.

    Returns:
        The squared energy distance in standardized units.
    """
    from scipy.spatial.distance import cdist

    x = _as_points(imputed)
    y = _as_points(holdout)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: imputed has {x.shape[1]} columns, holdout "
            f"has {y.shape[1]}."
        )
    wx = _as_weights(imputed_weights, x.shape[0])
    wy = _as_weights(holdout_weights, y.shape[0])
    if standardize:
        mean, std = _weighted_moments(y, wy)
        x = _standardize_by(x, mean, std)
        y = _standardize_by(y, mean, std)

    rng = np.random.default_rng(seed)
    x, wx = _resample_to_cap(x, wx, cap=max_points, rng=rng)
    y, wy = _resample_to_cap(y, wy, cap=max_points, rng=rng)
    px = wx / wx.sum()
    py = wy / wy.sum()

    cross = float(px @ cdist(x, y) @ py)
    within_x = float(px @ cdist(x, x) @ px)
    within_y = float(py @ cdist(y, y) @ py)
    # Clip tiny negative values from floating-point cancellation.
    return max(2.0 * cross - within_x - within_y, 0.0)


def prdc(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    k: int = 5,
    real_weights: np.ndarray | None = None,
    synthetic_weights: np.ndarray | None = None,
    max_points: int = 2048,
    seed: int = 0,
) -> dict[str, float]:
    """Precision, recall, density, and coverage (Naeem et al. 2020), weighted.

    Neighbourhood radii are k-th-nearest-neighbour distances within each sample
    (support geometry, unweighted); the *averages* over points are weighted, so
    e.g. coverage is the weighted fraction of real points with at least one
    synthetic neighbour inside their radius. Columns are standardized by the
    real sample's weighted moments.

    Coverage is the harness's anti-mode-collapse axis: a candidate concentrated
    on modal households leaves most of the real manifold uncovered. Because
    coverage depends on the candidate only through its *support* (which records
    exist, not how they are weighted), it is invariant to any reweighting of
    the candidate -- the calibration-blind block of the scorecard.

    Args:
        real: Held-out reference points, shape ``(n,)`` or ``(n, d)``.
        synthetic: Candidate points, shape ``(m,)`` or ``(m, d)``.
        k: Neighbour rank defining each point's local radius.
        real_weights: Real-point weights (``None`` for uniform).
        synthetic_weights: Candidate weights, used for the precision/density
            averages and the size cap (``None`` for uniform).
        max_points: Cap per side via seeded weight-proportional resample.
        seed: Seed for the resample.

    Returns:
        ``{"precision", "recall", "density", "coverage"}``, each in
        ``[0, 1]`` except density (which can exceed 1).
    """
    from scipy.spatial.distance import cdist

    r = _as_points(real)
    s = _as_points(synthetic)
    if r.shape[1] != s.shape[1]:
        raise ValueError(
            f"Dimension mismatch: real has {r.shape[1]} columns, synthetic "
            f"has {s.shape[1]}."
        )
    wr = _as_weights(real_weights, r.shape[0])
    ws = _as_weights(synthetic_weights, s.shape[0])
    mean, std = _weighted_moments(r, wr)
    r = _standardize_by(r, mean, std)
    s = _standardize_by(s, mean, std)

    rng = np.random.default_rng(seed)
    r, wr = _resample_to_cap(r, wr, cap=max_points, rng=rng)
    s, ws = _resample_to_cap(s, ws, cap=max_points, rng=rng)
    if min(len(r), len(s)) <= k:
        raise ValueError(
            f"PRDC needs more than k={k} points per side after capping; got "
            f"{len(r)} real and {len(s)} synthetic."
        )
    pr = wr / wr.sum()
    ps = ws / ws.sum()

    def _knn_radii(points: np.ndarray) -> np.ndarray:
        distances = cdist(points, points)
        # Column k of the row-sorted matrix is the k-th neighbour excluding
        # self (column 0 is the zero self-distance).
        return np.sort(distances, axis=1)[:, k]

    radii_real = _knn_radii(r)
    radii_synth = _knn_radii(s)
    cross = cdist(s, r)  # (m, n): synthetic rows, real columns

    inside_real_ball = cross <= radii_real[None, :]
    precision = float(ps @ inside_real_ball.any(axis=1))
    density = float(ps @ inside_real_ball.sum(axis=1)) / k
    recall = float(pr @ (cross <= radii_synth[:, None]).any(axis=0))
    coverage = float(pr @ (cross.min(axis=0) <= radii_real))
    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


def classifier_two_sample_auc(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    real_weights: np.ndarray | None = None,
    synthetic_weights: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Weighted classifier two-sample test: cross-validated AUC, 0.5 is best.

    Trains a gradient-boosted classifier to distinguish real from synthetic
    rows, with sample weights normalized so each side carries equal total mass,
    and reports the stratified cross-validated AUC. An AUC of 0.5 means nothing
    the classifier can find separates the candidate from the holdout -- the
    omnibus complement to the per-axis metrics.

    Args:
        real: Held-out reference points, shape ``(n,)`` or ``(n, d)``.
        synthetic: Candidate points, shape ``(m,)`` or ``(m, d)``.
        real_weights: Real-side weights (``None`` for uniform).
        synthetic_weights: Candidate weights (``None`` for uniform).
        n_splits: Stratified CV folds.
        seed: Seed for fold assignment and the classifier.

    Returns:
        Mean out-of-fold AUC in ``[0, 1]``; 0.5 indicates indistinguishability.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    r = _as_points(real)
    s = _as_points(synthetic)
    if r.shape[1] != s.shape[1]:
        raise ValueError(
            f"Dimension mismatch: real has {r.shape[1]} columns, synthetic "
            f"has {s.shape[1]}."
        )
    wr = _as_weights(real_weights, r.shape[0])
    ws = _as_weights(synthetic_weights, s.shape[0])

    x = np.vstack([r, s])
    y = np.r_[np.zeros(len(r)), np.ones(len(s))]
    # Equal total mass per class, so the test is about shape, not sample size.
    w = np.r_[wr / wr.sum(), ws / ws.sum()]

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in folds.split(x, y):
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(x[train_idx], y[train_idx], sample_weight=w[train_idx])
        scores = model.predict_proba(x[test_idx])[:, 1]
        aucs.append(
            float(
                roc_auc_score(y[test_idx], scores, sample_weight=w[test_idx])
            )
        )
    return float(np.mean(aucs))


def reweight_fragility(
    imputed: np.ndarray,
    weights: np.ndarray,
    *,
    kappa: float = 5.0,
) -> float:
    """Worst-case single-record share of an aggregate under bounded reweighting.

    After imputation, downstream users re-weight the file (a new calibration, a
    subpopulation zoom, a stress scenario). A record carrying an extreme value
    at low weight is invisible in the shipped aggregates but can dominate one
    after reweighting -- the "landmine" failure mode. This diagnostic bounds the
    exposure over the admissible family of *bounded multiplicative*
    reweightings ``w_i -> m_i * w_i`` with ``m_i in [1/kappa, kappa]`` (the
    hard weight-ratio bounds used by production calibration guards).

    The worst case has a closed form: the adversary sets ``m = kappa`` on the
    record with the largest baseline contribution ``c_i = w_i * |a_i|`` and
    ``m = 1/kappa`` everywhere else, giving

    ``fragility = kappa^2 c* / (kappa^2 c* + (S - c*))``

    where ``c*`` is the largest contribution and ``S`` their sum. ``kappa = 1``
    reduces to the baseline maximum share. Contributions use ``|a_i|`` so
    sign-mixed components measure magnitude exposure. Higher is more fragile;
    a robust file keeps this near ``kappa^2 / n`` (the uniform-contribution
    value), a landmined file approaches 1.

    Args:
        imputed: Imputed values of the aggregated quantity, shape ``(n,)``.
        weights: Baseline weights aligned to ``imputed``.
        kappa: Multiplicative weight-ratio bound of the admissible family.

    Returns:
        The worst-case single-record share, in ``[0, 1]``.
    """
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1, got {kappa}.")
    values = np.asarray(imputed, dtype=np.float64)
    w = _as_weights(weights, values.shape[0])
    contributions = w * np.abs(values)
    total = float(contributions.sum())
    if total <= 0.0:
        return 0.0
    largest = float(contributions.max())
    boosted = kappa * kappa * largest
    return boosted / (boosted + (total - largest))
