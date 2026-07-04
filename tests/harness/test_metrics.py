"""Weighted evaluation metrics: identities, orderings, and closed forms.

Adapted from PolicyEngine/imputation-paper (tests/test_smoke.py), the
population-view harness described in that paper; extended here for
longitudinal views.

These tests were extracted from the imputation paper's CI smoke suite, which
exercised the metrics alongside the sweep/registry/CLI plumbing that lives
only in that repo. Only the metric-focused cases are ported here: cheap
identities on identical inputs, correct orderings under known distributional
shifts (energy distance, coverage, classifier AUC), and a closed-form check
for the reweight-fragility diagnostic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from populace_dynamics.harness import metrics


def test_metric_identities() -> None:
    """Cheap invariants: zero distance/error on identical inputs, finite loss."""
    rng = np.random.default_rng(0)
    values = rng.lognormal(1.0, 0.5, 300)
    weights = rng.uniform(1.0, 5.0, 300)
    assert metrics.weighted_wasserstein1(
        values, values, imputed_weights=weights, donor_weights=weights
    ) == pytest.approx(0.0, abs=1e-9)
    assert metrics.zero_share_error(values, values) == 0.0
    loss = metrics.weighted_pinball_loss(values, values, weights=weights)
    assert math.isfinite(loss) and loss >= 0.0
    with pytest.raises(ValueError, match="non-negative"):
        metrics.weighted_pinball_loss(values, values, weights=-weights)


def test_energy_distance_is_zero_iff_same_and_orders_shifts() -> None:
    """Identical weighted samples score 0; larger shifts score strictly worse."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, (500, 3))
    weights = rng.uniform(1.0, 4.0, 500)
    assert metrics.energy_distance(
        base, base, imputed_weights=weights, holdout_weights=weights
    ) == pytest.approx(0.0, abs=1e-9)
    small = metrics.energy_distance(base + 0.3, base, holdout_weights=weights)
    large = metrics.energy_distance(base + 1.5, base, holdout_weights=weights)
    assert 0.0 < small < large


def test_prdc_coverage_detects_mode_collapse() -> None:
    """A modal-point candidate scores near-zero coverage; a true sample doesn't.

    This is the harness's reason for carrying coverage alongside marginal
    distances: a candidate collapsed onto the modal household can look tolerable
    on a marginal metric while covering none of the real manifold.
    """
    rng = np.random.default_rng(2)
    real = rng.normal(0.0, 1.0, (600, 2))
    faithful = rng.normal(0.0, 1.0, (600, 2))
    modal = np.tile(np.median(real, axis=0), (600, 1))

    good = metrics.prdc(real, faithful, seed=0)
    collapsed = metrics.prdc(real, modal, seed=0)
    assert good["coverage"] > 0.7
    assert collapsed["coverage"] < 0.1
    assert good["recall"] > collapsed["recall"]
    for value in (*good.values(), *collapsed.values()):
        assert math.isfinite(value) and value >= 0.0


def test_c2st_auc_separates_shifted_from_identical() -> None:
    """Same distribution scores near 0.5; a strongly shifted one near 1."""
    rng = np.random.default_rng(3)
    real = rng.normal(0.0, 1.0, (400, 2))
    same = rng.normal(0.0, 1.0, (400, 2))
    shifted = rng.normal(3.0, 1.0, (400, 2))
    assert (
        abs(metrics.classifier_two_sample_auc(real, same, seed=0) - 0.5) < 0.12
    )
    assert metrics.classifier_two_sample_auc(real, shifted, seed=0) > 0.9


def test_reweight_fragility_closed_form_and_landmine() -> None:
    """Uniform contributions match the closed form; a landmine approaches 1."""
    n = 100
    uniform = metrics.reweight_fragility(np.ones(n), np.ones(n), kappa=1.0)
    assert uniform == pytest.approx(1.0 / n)
    # Equal contributions at kappa=5: k^2*c / (k^2*c + (n-1)*c) = 25/124.
    boosted = metrics.reweight_fragility(np.ones(n), np.ones(n), kappa=5.0)
    assert boosted == pytest.approx(25.0 / 124.0)
    # One record carrying 100x the contribution of each other record.
    landmine = metrics.reweight_fragility(
        np.r_[np.ones(n - 1), 100.0], np.ones(n), kappa=5.0
    )
    assert landmine > 0.9
    assert metrics.reweight_fragility(np.zeros(4), np.ones(4)) == 0.0
    with pytest.raises(ValueError, match="kappa"):
        metrics.reweight_fragility(np.ones(4), np.ones(4), kappa=0.5)
