"""Donor/receiver holdout splits for evaluating imputation out of sample.

Adapted from PolicyEngine/imputation-paper
(src/imputation_paper/experiments/holdout.py), the population-view harness
described in that paper; extended here for longitudinal views.

An imputer is fit on the donor rows and scored on receiver rows whose targets it
never saw. The paper's protocol is an 80/20 donor/receiver split repeated over
several seeds, with the *same* split paired across methods (so a method never
wins by drawing an easier receiver). :func:`paired_splits` yields exactly those
paired splits; :func:`split_frame` is the single-split primitive.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """One donor/receiver split.

    Attributes:
        seed: The seed that produced this split.
        train: Donor rows (targets observed, used to fit).
        test: Receiver rows (targets held out, used to score).
    """

    seed: int
    train: pd.DataFrame
    test: pd.DataFrame


def split_frame(
    frame: pd.DataFrame, *, holdout_frac: float = 0.2, seed: int = 0
) -> Split:
    """Split ``frame`` into donor/receiver by fraction, deterministically.

    Args:
        frame: The full observed table.
        holdout_frac: Fraction of rows placed in the receiver (test) split.
        seed: Seed for the permutation.

    Returns:
        A :class:`Split`.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(
            f"holdout_frac must be in (0, 1), got {holdout_frac}."
        )
    rng = np.random.default_rng(seed)
    n = len(frame)
    order = rng.permutation(n)
    n_holdout = int(round(n * holdout_frac))
    test_idx = np.sort(order[:n_holdout])
    train_idx = np.sort(order[n_holdout:])
    return Split(
        seed=seed,
        train=frame.iloc[train_idx].reset_index(drop=True),
        test=frame.iloc[test_idx].reset_index(drop=True),
    )


def paired_splits(
    frame: pd.DataFrame,
    *,
    holdout_frac: float = 0.2,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> Iterator[Split]:
    """Yield one donor/receiver split per seed.

    Each split is a pure function of its seed, so every method evaluated on the
    same seed sees the *same* donor/receiver partition -- the paired design the
    paper uses to compare methods on identical held-out targets.

    Args:
        frame: The full observed table.
        holdout_frac: Receiver fraction (default 0.2, i.e. an 80/20 split).
        seeds: The seeds to iterate (default ten, matching the paper's
            ten repeated splits).

    Yields:
        A :class:`Split` per seed, in ``seeds`` order.
    """
    for seed in seeds:
        yield split_frame(frame, holdout_frac=holdout_frac, seed=seed)
