"""Donor/receiver holdout splits: determinism, completeness, and validation.

Adapted from PolicyEngine/imputation-paper (tests/test_smoke.py), the
population-view harness described in that paper; extended here for
longitudinal views.

The source test (``test_paired_splits_are_deterministic_partitions``) built
its toy frame via ``imputation_paper.smoke.make_toy_dataset``, a fixture
specific to that repo's CI smoke suite. It is not part of the ported harness
(metrics.py, holdout.py, views.py), so this version builds an equivalent
minimal weighted frame directly with pandas instead of depending on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.harness.holdout import paired_splits, split_frame


def _toy_frame(seed: int, n: int = 400) -> pd.DataFrame:
    """A minimal weighted frame, standing in for imputation-paper's smoke fixture."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, n),
            "weight": rng.uniform(1.0, 5.0, n),
        }
    )


def test_paired_splits_are_deterministic_partitions() -> None:
    """Same seed => same split; every split is a disjoint, complete partition."""
    frame = _toy_frame(seed=1, n=400)
    first = split_frame(frame, seed=7)
    second = split_frame(frame, seed=7)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
    splits = list(paired_splits(frame, seeds=(0, 1)))
    assert [s.seed for s in splits] == [0, 1]
    for split in splits:
        assert len(split.train) + len(split.test) == len(frame)


def test_split_frame_respects_holdout_fraction() -> None:
    """The receiver split's size matches the requested fraction, rounded."""
    frame = _toy_frame(seed=2, n=500)
    split = split_frame(frame, holdout_frac=0.2, seed=0)
    assert len(split.test) == round(500 * 0.2)
    assert len(split.train) == 500 - len(split.test)
    # Train and test are disjoint and jointly exhaustive over the row values.
    combined = pd.concat([split.train, split.test]).sort_values("a")
    expected = frame.sort_values("a").reset_index(drop=True)
    pd.testing.assert_series_equal(
        combined["a"].reset_index(drop=True), expected["a"]
    )


def test_split_frame_rejects_holdout_frac_out_of_range() -> None:
    """holdout_frac must be a proper fraction in (0, 1), not 0, 1, or beyond."""
    frame = _toy_frame(seed=3, n=50)
    with pytest.raises(ValueError, match="holdout_frac"):
        split_frame(frame, holdout_frac=0.0)
    with pytest.raises(ValueError, match="holdout_frac"):
        split_frame(frame, holdout_frac=1.0)
    with pytest.raises(ValueError, match="holdout_frac"):
        split_frame(frame, holdout_frac=1.5)


def test_paired_splits_default_seeds_match_the_papers_ten_repeats() -> None:
    """The default seed grid is the paper's ten repeated 80/20 splits."""
    frame = _toy_frame(seed=4, n=100)
    splits = list(paired_splits(frame))
    assert [s.seed for s in splits] == list(range(10))
