"""Tests for the held-out panel-moment battery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.harness import moments


def _panel(
    n_ids: int = 400,
    n_periods: int = 12,
    phi: float = 0.8,
    zero_rate: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """AR(1)-in-logs synthetic panel with optional zero spells."""
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(n_ids):
        log_y = rng.normal(10.0, 1.0)
        birth_year = int(rng.integers(1950, 1990))
        for period in range(n_periods):
            log_y = (
                phi * (log_y - 10.0)
                + 10.0
                + rng.normal(0.0, np.sqrt(1 - phi**2))
            )
            value = 0.0 if rng.random() < zero_rate else np.exp(log_y)
            rows.append(
                {
                    "person_id": person,
                    "period": 2000 + period,
                    "earnings": value,
                    "age": 2000 + period - birth_year,
                    "cohort": birth_year // 10 * 10,
                    "weight": rng.uniform(0.5, 2.0),
                }
            )
    return pd.DataFrame(rows)


KW = dict(
    id_col="person_id",
    period_col="period",
    value_col="earnings",
    weight_col="weight",
)


def test_weighted_moments_match_numpy_on_uniform_weights():
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 3.0, 5000)
    stats = moments.weighted_moments(x, np.ones_like(x))
    assert stats["mean"] == pytest.approx(x.mean(), abs=1e-9)
    assert stats["sd"] == pytest.approx(x.std(), abs=1e-9)
    assert abs(stats["skew"]) < 0.1
    assert abs(stats["kurtosis"]) < 0.2


def test_mobility_matrix_rows_normalize_and_persist():
    panel = _panel(phi=0.99, seed=2)
    matrix = moments.mobility_matrix(panel, n_bins=5, **KW)
    sums = matrix.groupby("origin")["probability"].sum()
    assert np.allclose(sums, 1.0)
    diagonal = matrix[matrix.origin == matrix.destination]
    stay = diagonal.probability.mean()
    assert stay > 0.5  # near-unit persistence concentrates on diagonal


def test_mobility_matrix_independent_panel_is_near_uniform():
    panel = _panel(phi=0.0, n_ids=2000, seed=3)
    matrix = moments.mobility_matrix(panel, n_bins=5, zero_bin=False, **KW)
    off = matrix[matrix.origin != matrix.destination].probability
    assert off.mean() == pytest.approx(0.2, abs=0.05)


def test_mobility_matrix_zero_bin_present_only_with_zeros():
    with_zeros = moments.mobility_matrix(_panel(zero_rate=0.2, seed=4), **KW)
    without = moments.mobility_matrix(_panel(seed=4), **KW)
    assert 0 in set(with_zeros.origin)
    assert 0 not in set(without.origin)


def test_change_moments_constant_growth_has_zero_sd():
    periods = np.arange(2000, 2010)
    panel = pd.DataFrame(
        {
            "person_id": np.repeat(np.arange(50), len(periods)),
            "period": np.tile(periods, 50),
        }
    )
    panel["earnings"] = np.exp(0.05 * (panel.period - 2000) + 10.0)
    panel["weight"] = 1.0
    stats = moments.change_moments(panel, **KW)
    mean = stats.loc[stats.moment == "mean", "value"].iloc[0]
    sd = stats.loc[stats.moment == "sd", "value"].iloc[0]
    assert mean == pytest.approx(0.05, abs=1e-9)
    assert sd == pytest.approx(0.0, abs=1e-9)


def test_change_moments_grouped_returns_group_keys():
    stats = moments.change_moments(_panel(seed=5), by=["cohort"], **KW)
    assert set(stats.columns) == {"cohort", "moment", "value"}
    assert stats.cohort.nunique() > 1


def test_autocorrelation_recovers_ar1_persistence():
    panel = _panel(phi=0.8, n_ids=3000, seed=6)
    result = moments.autocorrelation(panel, lags=(1, 2), **KW)
    lag1 = result.loc[result.lag == 1, "value"].iloc[0]
    lag2 = result.loc[result.lag == 2, "value"].iloc[0]
    assert lag1 == pytest.approx(0.8, abs=0.05)
    assert lag2 == pytest.approx(0.64, abs=0.06)


def test_age_profile_monotone_input_yields_monotone_mean():
    panel = _panel(seed=7)
    panel["earnings"] = panel["age"] * 1000.0
    profile = moments.age_profile(
        panel,
        age_col="age",
        value_col="earnings",
        weight_col="weight",
    )
    means = (
        profile[profile.statistic == "mean"]
        .sort_values("age")["value"]
        .to_numpy()
    )
    assert (np.diff(means) > 0).all()


def test_zero_spells_exact_on_constructed_panel():
    panel = pd.DataFrame(
        {
            "person_id": [1] * 6 + [2] * 6,
            "period": list(range(6)) * 2,
            "earnings": [5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 5],
            "weight": 1.0,
        }
    )
    stats = moments.zero_spells(panel, **KW)
    entry = stats.loc[stats.statistic == "entry_rate", "value"].iloc[0]
    exit_ = stats.loc[stats.statistic == "exit_rate", "value"].iloc[0]
    length = stats.loc[stats.statistic == "mean_spell_length", "value"].iloc[0]
    # Nonzero->zero pairs: person 1 has (5,0) at t0 and (5,0) at t4;
    # nonzero origins: person 1 t0,t3,t4 + person 2 t0..t4 = 8.
    assert entry == pytest.approx(2 / 8)
    # Zero origins: person 1 t1,t2; exits: (0,5) at t2 -> 1/2.
    assert exit_ == pytest.approx(1 / 2)
    # Spells: lengths 2 and 1.
    assert length == pytest.approx(1.5)


def test_transition_rates_recover_markov_chain():
    rng = np.random.default_rng(8)
    transition = {0: [0.9, 0.1], 1: [0.4, 0.6]}
    rows = []
    for person in range(2000):
        state = int(rng.integers(0, 2))
        for period in range(8):
            rows.append(
                {
                    "person_id": person,
                    "period": period,
                    "state": state,
                    "weight": 1.0,
                }
            )
            state = int(rng.random() > transition[state][0])
    panel = pd.DataFrame(rows)
    rates = moments.transition_rates(
        panel,
        id_col="person_id",
        period_col="period",
        state_col="state",
        weight_col="weight",
    )
    stay0 = rates[
        (rates.origin == 0) & (rates.destination == 0)
    ].probability.iloc[0]
    stay1 = rates[
        (rates.origin == 1) & (rates.destination == 1)
    ].probability.iloc[0]
    assert stay0 == pytest.approx(0.9, abs=0.02)
    assert stay1 == pytest.approx(0.6, abs=0.03)


def test_moment_distance_zero_on_identical_batteries():
    battery = moments.autocorrelation(_panel(seed=9), **KW)
    assert moments.moment_distance(battery, battery.copy()) == 0.0


def test_moment_distance_raises_on_misaligned_batteries():
    battery = moments.autocorrelation(_panel(seed=9), **KW)
    with pytest.raises(ValueError, match="do not align"):
        moments.moment_distance(battery, battery.iloc[:-1].copy())


def test_biennial_period_step_pairs_correctly():
    panel = _panel(seed=10)
    panel["period"] = 2000 + (panel["period"] - 2000) * 2
    biennial = moments.autocorrelation(panel, lags=(1,), period_step=2, **KW)
    assert not np.isnan(biennial.value.iloc[0])
    annual_spacing = moments.autocorrelation(
        panel, lags=(1,), period_step=1, **KW
    )
    assert np.isnan(annual_spacing.value.iloc[0])
