"""Candidate-16 first-marriage component, free of frozen-script imports.

Candidate 16 selects this unchanged component at
``scripts/run_gate2_candidate16.py:214-238``.  The implementation below ports
the restricted-cubic-spline basis from
``scripts/run_gate2_candidate1.py:184-209``, the age-spline-by-sex and
age-spline-by-cohort design from
``scripts/run_gate2_candidate2.py:145-175``, and the resolved 20/22/25/30/40
knot fit from ``scripts/run_gate2_candidate3.py:184-238``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

__all__ = [
    "SPLINE_KNOTS",
    "FirstMarriageModel",
    "fit_first_marriage",
    "ncs_basis",
]

# Frozen candidate-3 knots; source: scripts/run_gate2_candidate3.py:157-158.
SPLINE_KNOTS: tuple[float, ...] = (20.0, 22.0, 25.0, 30.0, 40.0)


def ncs_basis(x: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Return the frozen restricted-cubic-spline basis.

    This preserves the operation order in
    ``scripts/run_gate2_candidate1.py:184-209``: one linear column followed
    by ``K - 2`` nonlinear columns, scaled by the squared boundary-knot span.
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    n_knots = len(k)
    t1, tkm1, tk = k[0], k[-2], k[-1]
    denom = tk - tkm1
    scale = (tk - t1) ** 2
    cols = [x.copy()]
    for j in range(n_knots - 2):
        tj = k[j]
        term = (
            np.maximum(x - tj, 0.0) ** 3
            - np.maximum(x - tkm1, 0.0) ** 3 * (tk - tj) / denom
            + np.maximum(x - tk, 0.0) ** 3 * (tkm1 - tj) / denom
        ) / scale
        cols.append(term)
    return np.column_stack(cols)


@dataclass
class FirstMarriageModel:
    """Fitted candidate-16 discrete-time first-marriage hazard.

    The fields, standardised design, and probability lookup port
    ``scripts/run_gate2_candidate1.py:215-261``.  ``_raw_design`` ports the
    candidate-2 interaction layout at
    ``scripts/run_gate2_candidate2.py:147-175`` exactly: spline, sex,
    spline-by-sex, cohort dummies, and spline-by-cohort.
    """

    clf: LogisticRegression
    cohort_levels: list[int]
    knots: tuple[float, ...]
    col_mean: np.ndarray
    col_sd: np.ndarray
    n_train_rows: int
    n_train_events: int
    n_iter: int
    converged: bool

    def _raw_design(
        self, age: np.ndarray, is_male: np.ndarray, decade: np.ndarray
    ) -> np.ndarray:
        spline = ncs_basis(age, self.knots)
        male = is_male.astype(np.float64).reshape(-1, 1)
        parts = [spline, male, spline * male]
        dummies = []
        for level in self.cohort_levels[1:]:
            dummies.append((decade == level).astype(np.float64))
        if dummies:
            dmat = np.column_stack(dummies)
            parts.append(dmat)
            for c in range(spline.shape[1]):
                parts.append(spline[:, [c]] * dmat)
        return np.column_stack(parts)

    def _design(
        self, age: np.ndarray, is_male: np.ndarray, decade: np.ndarray
    ) -> np.ndarray:
        raw = self._raw_design(age, is_male, decade)
        return (raw - self.col_mean) / self.col_sd

    def predict(
        self, age: np.ndarray, is_male: np.ndarray, decade: np.ndarray
    ) -> np.ndarray:
        """Return fitted first-marriage probabilities.

        Ported from ``scripts/run_gate2_candidate1.py:255-261`` without
        changing the design or ``predict_proba`` call order.
        """
        if age.size == 0:
            return np.zeros(0, dtype=np.float64)
        x = self._design(age, is_male, decade)
        return self.clf.predict_proba(x)[:, 1]


def fit_first_marriage(
    train_py: pd.DataFrame, event_years: set[tuple[int, int]]
) -> FirstMarriageModel:
    """Fit candidate 16's logistic first-marriage hazard.

    ``train_py`` is the never-married train person-year frame and must contain
    ``person_id``, ``year``, ``age``, ``sex``, ``weight``, and
    ``birth_decade``. ``event_years`` contains the ``(person_id, year)``
    first-marriage outcomes.  The estimator, row construction,
    standardisation, sample weighting, and convergence bookkeeping preserve
    ``scripts/run_gate2_candidate3.py:187-238`` exactly.
    """
    age = train_py["age"].to_numpy(dtype=np.float64)
    is_male = (train_py["sex"].to_numpy() == "male").astype(np.float64)
    decade = (train_py["birth_decade"].to_numpy()).astype(np.int64)
    weight = train_py["weight"].to_numpy(dtype=np.float64)
    pid = train_py["person_id"].to_numpy()
    yr = train_py["year"].to_numpy()
    y = np.fromiter(
        (
            (int(p), int(t)) in event_years
            for p, t in zip(pid, yr, strict=True)
        ),
        dtype=np.float64,
        count=len(train_py),
    )

    cohort_levels = sorted(int(d) for d in np.unique(decade))
    model = FirstMarriageModel(
        clf=LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-6),
        cohort_levels=cohort_levels,
        knots=SPLINE_KNOTS,
        col_mean=np.zeros(1),
        col_sd=np.ones(1),
        n_train_rows=int(len(train_py)),
        n_train_events=int(y.sum()),
        n_iter=0,
        converged=False,
    )
    raw = model._raw_design(age, is_male, decade)
    col_mean = raw.mean(axis=0)
    col_sd = raw.std(axis=0)
    col_sd = np.where(col_sd > 0, col_sd, 1.0)
    model.col_mean = col_mean
    model.col_sd = col_sd
    x = (raw - col_mean) / col_sd
    model.clf.fit(x, y, sample_weight=weight)
    n_iter = int(np.max(model.clf.n_iter_))
    model.n_iter = n_iter
    model.converged = n_iter < model.clf.max_iter
    return model
