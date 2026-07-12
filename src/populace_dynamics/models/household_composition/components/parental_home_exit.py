"""Parental-home exit component for the resolved candidate 9.

The spline model and base fit are copied from
``household_composition_sim.py:65-72,134-216``.  The adult-child single-year
refit and its isolated draw are copied from
``household_composition_sim_v6.py:139-173,251-301,533-578``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from populace_dynamics.data import household_composition as hc
from populace_dynamics.models.household_composition.common import (
    restricted_cubic_basis,
    weighted_rate,
)

PARENTAL_EXIT_KNOTS: tuple[float, ...] = (16.0, 19.0, 23.0, 30.0, 45.0)
CHILD_MIN_LEAVE_AGE = hc.START_AGE
CHILD_MAX_LEAVE_AGE = 60
CHILD_EXIT_REFIT_LO = 18
CHILD_EXIT_REFIT_HI = 30
MIN_STRATUM_N = 20
DELTA_STREAM_TAG_V6 = 0xC6
REFIT_GRID_AGES: tuple[int, ...] = tuple(
    age
    for age in range(CHILD_MIN_LEAVE_AGE, CHILD_MAX_LEAVE_AGE, 2)
    if CHILD_EXIT_REFIT_LO <= age <= CHILD_EXIT_REFIT_HI
)


@dataclass
class ParentalExitModel:
    """Fitted logistic parental-home exit hazard.

    Copied from ``household_composition_sim.py:134-163``.
    """

    clf: LogisticRegression
    knots: tuple[float, ...]
    col_mean: np.ndarray
    col_sd: np.ndarray
    n_train_rows: int
    n_train_events: int
    converged: bool

    def _raw_design(self, age: np.ndarray, is_male: np.ndarray) -> np.ndarray:
        spline = restricted_cubic_basis(age, self.knots)
        male = is_male.astype(np.float64).reshape(-1, 1)
        return np.column_stack([spline, male, spline * male])

    def predict(self, age: np.ndarray, is_male: np.ndarray) -> np.ndarray:
        age = np.asarray(age, dtype=np.float64)
        if age.size == 0:
            return np.zeros(0, dtype=np.float64)
        raw = self._raw_design(age, np.asarray(is_male, dtype=np.float64))
        x = (raw - self.col_mean) / self.col_sd
        return self.clf.predict_proba(x)[:, 1]


def fit_parental_exit(train_pw: pd.DataFrame) -> ParentalExitModel:
    """Fit the base parental-home exit hazard.

    Copied from ``household_composition_sim.py:182-216``.
    """
    at_risk = train_pw[
        train_pw["coresident_parent"] & train_pw["has_next"]
    ].copy()
    age = at_risk["age"].to_numpy(dtype=np.float64)
    is_male = (at_risk["sex"].to_numpy() == "male").astype(np.float64)
    weight = at_risk["weight"].to_numpy(dtype=np.float64)
    event = (
        at_risk["next_coresident_parent"].eq(False).to_numpy(dtype=np.float64)
    )
    model = ParentalExitModel(
        clf=LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-6),
        knots=PARENTAL_EXIT_KNOTS,
        col_mean=np.zeros(1),
        col_sd=np.ones(1),
        n_train_rows=int(len(at_risk)),
        n_train_events=int(event.sum()),
        converged=False,
    )
    raw = model._raw_design(age, is_male)
    col_mean = raw.mean(axis=0)
    col_sd = raw.std(axis=0)
    col_sd = np.where(col_sd > 0, col_sd, 1.0)
    model.col_mean = col_mean
    model.col_sd = col_sd
    x = (raw - col_mean) / col_sd
    model.clf.fit(x, event, sample_weight=weight)
    model.converged = bool(int(np.max(model.clf.n_iter_)) < model.clf.max_iter)
    return model


def fit_child_exit_single_year(
    person_waves: pd.DataFrame,
    parental_exit: ParentalExitModel,
    train_ids: set[int],
) -> tuple[dict[tuple[int, str], float], dict[str, Any]]:
    """Fit the single-year adult-child exit refit.

    Copied from ``household_composition_sim_v6.py:251-301``.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)]
    at_risk = pw[pw["coresident_parent"] & pw["has_next"]]
    table: dict[tuple[int, str], float] = {}
    diag_emp: dict[str, dict[str, float | int]] = {}
    for age in range(CHILD_EXIT_REFIT_LO, CHILD_EXIT_REFIT_HI + 1):
        for sex in hc.SEXES:
            sub = at_risk[(at_risk["age"] == age) & (at_risk["sex"] == sex)]
            spline = float(
                parental_exit.predict(
                    np.array([float(age)]),
                    np.array([1.0 if sex == "male" else 0.0]),
                )[0]
            )
            if len(sub) >= MIN_STRATUM_N:
                event = (
                    sub["next_coresident_parent"]
                    .eq(False)
                    .to_numpy(np.float64)
                )
                rate = weighted_rate(sub, event)
            else:
                rate = spline
            table[(age, sex)] = rate
            diag_emp[f"{age}|{sex}"] = {
                "empirical": round(rate, 5),
                "spline": round(spline, 5),
                "n_atrisk": int(len(sub)),
            }
    return table, {
        "refit_age_range": [CHILD_EXIT_REFIT_LO, CHILD_EXIT_REFIT_HI],
        "grid_ages_biennial": list(REFIT_GRID_AGES),
        "single_year_hazard_vs_spline": diag_emp,
        "n_at_risk_waves_train": int(len(at_risk)),
    }


def child_leave_years(
    births: pd.DataFrame,
    parental_exit: ParentalExitModel,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw base leave years, including intentionally discarded draws.

    Copied from ``household_composition_sim.py:550-581``.
    """
    n = len(births)
    if n == 0:
        return births.assign(leave_year=np.array([], dtype=np.int64))
    child_male = rng.random(n) < 0.5
    leave_age = np.full(n, CHILD_MAX_LEAVE_AGE, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    ages = list(range(CHILD_MIN_LEAVE_AGE, CHILD_MAX_LEAVE_AGE, 2))
    for age in ages:
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        prob = parental_exit.predict(
            np.full(idx.size, float(age)), child_male[idx].astype(np.float64)
        )
        u = rng.random(idx.size)
        left = u < prob
        left_idx = idx[left]
        leave_age[left_idx] = age
        alive[left_idx] = False
    birth_year = births["birth_year"].to_numpy(dtype=np.int64)
    return births.assign(leave_year=birth_year + leave_age)


def child_leave_years_refit(
    births: pd.DataFrame,
    parental_exit: ParentalExitModel,
    child_exit_single_year: dict[tuple[int, str], float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw maternal leave years with the 18--30 refit.

    Copied from ``household_composition_sim_v6.py:533-578``.
    """
    n = len(births)
    if n == 0:
        return births.assign(leave_year=np.array([], dtype=np.int64))
    child_male = rng.random(n) < 0.5
    leave_age = np.full(n, CHILD_MAX_LEAVE_AGE, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    ages = list(range(CHILD_MIN_LEAVE_AGE, CHILD_MAX_LEAVE_AGE, 2))
    for age in ages:
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        males = child_male[idx]
        if CHILD_EXIT_REFIT_LO <= age <= CHILD_EXIT_REFIT_HI:
            prob = np.array(
                [
                    child_exit_single_year[(age, "male" if male else "female")]
                    for male in males
                ],
                dtype=np.float64,
            )
        else:
            prob = parental_exit.predict(
                np.full(idx.size, float(age)), males.astype(np.float64)
            )
        u = rng.random(idx.size)
        left = u < prob
        left_idx = idx[left]
        leave_age[left_idx] = age
        alive[left_idx] = False
    birth_year = births["birth_year"].to_numpy(dtype=np.int64)
    return births.assign(leave_year=birth_year + leave_age)
