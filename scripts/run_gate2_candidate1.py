"""Gate-2 candidate 1 (run 1): stratified empirical hazards with composed
widowhood.

The FIRST pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4910914098 (``SPEC_REGISTRATION``): a five-component
family-transition simulator scored under the LOCKED gate-2 protocol
(gates.yaml ``gate_2``, ratified PR #79 + flip #81). One-shot; no constant
moves after the registration comment.

The five components (frozen)
----------------------------
1. **First marriage** -- a discrete-time logistic hazard on train
   never-married person-years: age (natural cubic / restricted spline,
   knots 20/25/30/40 -> 3 basis columns), sex, birth-decade cohort (main
   effects + age-spline x cohort interaction). ``sklearn.linear_model
   .LogisticRegression`` (repo .venv has scikit-learn 1.9; recorded in
   revision_pins). "Discrete-time logistic hazard fit" denotes the MLE;
   the sklearn default L2 (C=1.0) is negligible at the train scale
   (hundreds of thousands of weighted person-years) so the fit is
   effectively the weighted MLE while staying numerically stable on thin
   cohorts. Fit with sample_weight = the person-year PSID weight (the
   weighted target). Design columns are standardised for lbfgs
   conditioning (MLE-invariant); the scaler is stored for prediction.
2. **Divorce** -- an empirical weighted hazard table by marriage-duration
   band (0-4/5-9/10-19/20+) x marriage order (1st vs 2+), train-estimated
   with add-one (Laplace) smoothing at the train mean married person-year
   weight.
3. **Widowhood** -- COMPOSED, not fit: the spouse dies via the
   train-estimated age-band x sex mortality hazard table (the
   mortality-foundation construction, scripts.build_mortality_floors
   .build_exposure_slices + weighted_hazards, restricted to the train
   persons). The spouse is the opposite sex; spousal age = the person's
   age + the train mean spousal age gap by the person's sex (from the
   marriage records' joinable spouse birth years). Widowhood is the
   surviving partner's induced transition.
4. **Remarriage** -- an empirical weighted hazard table by
   years-since-dissolution band (0-4/5-9/10+) x origin (divorced/widowed)
   x sex, train-estimated with add-one smoothing at the train mean
   dissolved person-year weight.
5. **Fertility** -- empirical weighted age-band (5-year ASFR) x parity
   (0/1/2/3+) rates by birth-decade cohort, train-estimated (no smoothing,
   per the registration).

Protocol (LOCKED, gates.yaml gate_2, read at runtime -- no threshold
hardcoded)
-----------------------------------------------------------------------
Option (a), the gate-1 mirror. Per gate seed s in {0,1,2,3,4}:
``populace_dynamics.harness.panel.split_panel_by_person(panel.attrs,
'person_id', fraction=0.5, seed=s)`` draws side A (the HOLDOUT) and side B
(the train complement). The five components are refit on side B (side A
excluded from all fitting). Each side-A person's family history is then
simulated annually over their observed panel support from their observed
initial state at entry, RNG ``numpy.random.default_rng(4200 + s)``, ONE
simulated sequence per person. The simulated panel is scored by the SAME
``transitions`` cell constructors as the reference; each cell's rate
``r_candidate`` is compared to side A's own empirical rate ``rate_a``
(committed per gate seed in ``runs/gate2_floors_v2.json``
``noise_floor_per_seed``) as ``|ln(r_candidate / rate_a)|`` against the
cell's locked tolerance. A seed passes iff every one of the 46 gated cells
holds; the gate passes iff >= 4 of the 5 gate seeds pass. The 16
report-only cells are scored and published, never gated.

Hard-stop precheck (mirrors gate-1's battery-reference reproduction): the
scoring path must reproduce, bit-for-bit, (i) every committed full-panel
reference moment, (ii) every committed per-gate-seed ``rate_a``, and (iii)
each gate seed's committed holdout-id sha256, BEFORE any candidate is
simulated. Any mismatch is a hard stop.

One-shot rule. Run ONCE. A per-seed cache OUTSIDE ``runs/`` lets a crash be
fixed and relaunched without re-scoring any already-scored seed (no scored
output influences a fix). Publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit). Run from the repository root with the PSID history files
staged::

    .venv/bin/python scripts/run_gate2_candidate1.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

# The gate-2 floor builder supplies the panel loader and the reference-moment
# cell constructors (reused verbatim -- the scoring path IS the committed
# floor's path). The mortality-foundation construction supplies the composed
# widowhood table. Both import cleanly under the repo .venv.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import build_mortality_floors as mort  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v1.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v1"
RUN_NAME = "gate2_hazard_v1"

#: This run's frozen-spec registration (issue #42, comment 4910914098).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4910914098"
)

#: The pinned gate-run seeds and the simulation RNG rule (registration).
GATE_SEEDS = (0, 1, 2, 3, 4)
SIM_SEED_BASE = 4200

#: First-marriage spline knots (natural cubic / restricted cubic spline).
SPLINE_KNOTS = (20.0, 25.0, 30.0, 40.0)

#: Bit-exact tolerance for the reproduction precheck.
EXACT_ATOL = 1e-12

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate1_run1_cache.json"
)


# --------------------------------------------------------------------------
# JSON / cache helpers
# --------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, default=_json_default) + "\n"
    )


# --------------------------------------------------------------------------
# gates.yaml gate-2 protocol (read at runtime; no threshold hardcoded)
# --------------------------------------------------------------------------
def load_gate2_thresholds() -> dict[str, Any]:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_2"]["thresholds"]


def gated_tolerances(thresholds: dict[str, Any]) -> dict[str, float]:
    """The 46 locked per-cell tolerances, merged across the gate-2 views."""
    tol: dict[str, float] = {}
    for view in thresholds["views"].values():
        for cell, value in view["tolerances"].items():
            tol[cell] = float(value)
    return tol


# --------------------------------------------------------------------------
# Natural cubic (restricted) spline basis
# --------------------------------------------------------------------------
def ncs_basis(x: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Restricted cubic spline basis (Harrell): K knots -> K-1 columns.

    Column 0 is the linear term; columns 1..K-2 are the natural-spline
    nonlinear terms (linear beyond the boundary knots), scaled by
    ``(t_K - t_1)**2``.
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


# --------------------------------------------------------------------------
# First-marriage logistic hazard model
# --------------------------------------------------------------------------
@dataclass
class FirstMarriageModel:
    """Fitted discrete-time logistic first-marriage hazard."""

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
        spline = ncs_basis(age, self.knots)  # (n, 3)
        n = spline.shape[0]
        parts = [spline, is_male.astype(np.float64).reshape(-1, 1)]
        # Cohort main-effect dummies (drop the first level = reference).
        dummies = []
        for level in self.cohort_levels[1:]:
            dummies.append((decade == level).astype(np.float64))
        if dummies:
            dmat = np.column_stack(dummies)  # (n, m-1)
            parts.append(dmat)
            # age-spline x cohort interaction.
            for c in range(spline.shape[1]):
                parts.append(spline[:, [c]] * dmat)
        else:
            dmat = np.zeros((n, 0))
        return np.column_stack(parts)

    def _design(
        self, age: np.ndarray, is_male: np.ndarray, decade: np.ndarray
    ) -> np.ndarray:
        raw = self._raw_design(age, is_male, decade)
        return (raw - self.col_mean) / self.col_sd

    def predict(
        self, age: np.ndarray, is_male: np.ndarray, decade: np.ndarray
    ) -> np.ndarray:
        if age.size == 0:
            return np.zeros(0, dtype=np.float64)
        x = self._design(age, is_male, decade)
        return self.clf.predict_proba(x)[:, 1]


def fit_first_marriage(
    train_py: pd.DataFrame, event_years: set[tuple[int, int]]
) -> FirstMarriageModel:
    """Fit the logistic first-marriage hazard on train never-married PY.

    ``train_py`` is the never-married person-year frame (columns
    ``person_id``, ``year``, ``age``, ``sex``, ``weight``) restricted to the
    train persons. ``event_years`` is the set of ``(person_id, year)`` that
    carry a first-marriage event (the person-year's binary outcome).
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
        # sklearn LogisticRegression at its default L2 regularisation (C=1.0);
        # negligible vs the weighted data term at train scale (effectively the
        # weighted MLE). penalty= is left at default (the 1.8 deprecation).
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


# --------------------------------------------------------------------------
# Fitted components bundle
# --------------------------------------------------------------------------
@dataclass
class Components:
    first_marriage: FirstMarriageModel
    divorce: np.ndarray  # (n_dur_bands, 2) indexed [dur_band, order>=2]
    remarriage: dict[tuple[int, int, str], float]  # (ysd_band, origin, sex)
    mortality: dict[str, float]  # "band|sex" -> central death rate
    fertility: dict[tuple[int, int, int], float]  # (age_band, parity, decade)
    gap_by_sex: dict[str, float]  # person sex -> mean (spouse_age - self_age)
    meta: dict[str, Any]


DIV_BANDS = transitions.DIVORCE_DURATION_BANDS
YSD_BANDS = transitions.REMARRIAGE_YSD_BANDS
ASFR_BANDS = transitions.ASFR_AGE_BANDS
MORT_BANDS = mort.BANDS

DIV_LOWERS = np.array([lo for lo, _ in DIV_BANDS], dtype=np.int64)
YSD_LOWERS = np.array([lo for lo, _ in YSD_BANDS], dtype=np.int64)
ASFR_LOWERS = np.array([lo for lo, _ in ASFR_BANDS], dtype=np.int64)
MORT_LOWERS = np.array([lo for lo, _ in MORT_BANDS], dtype=np.int64)
_ASFR_LO = min(b[0] for b in ASFR_BANDS)
_ASFR_HI = max(b[1] for b in ASFR_BANDS)


def _bands_vec(values: np.ndarray, lowers: np.ndarray, n: int) -> np.ndarray:
    """Vectorised band index: clip(searchsorted(lowers, v, 'right') - 1)."""
    return np.clip(np.searchsorted(lowers, values, side="right") - 1, 0, n - 1)


def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Fit all five components on the train complement (side B only)."""
    py = panel.person_years
    ev = panel.events
    attrs = panel.attrs
    train_py = py[py["person_id"].isin(train_ids)]
    train_ev = ev[ev["person_id"].isin(train_ids)]
    attr_by = attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")

    # ---- 1. first marriage (logistic) ----
    nm = train_py[train_py["marital_state"] == "never_married"][
        ["person_id", "year", "age", "sex", "weight"]
    ].copy()
    nm["birth_decade"] = nm["person_id"].map(birth_decade).to_numpy()
    fm_ev = train_ev[train_ev["transition"] == "first_marriage"]
    event_years = {
        (int(p), int(t))
        for p, t in zip(
            fm_ev["person_id"].to_numpy(),
            fm_ev["year"].to_numpy(),
            strict=True,
        )
    }
    fm_model = fit_first_marriage(nm, event_years)

    # ---- 2. divorce (weighted empirical, add-one) ----
    married = train_py[train_py["marital_state"] == "married"].copy()
    married = _attach_order(married, order_map, dur_col="marriage_duration")
    div_ev = train_ev[
        (train_ev["transition"] == "divorce")
        & train_ev["marriage_duration"].notna()
    ].copy()
    div_ev = _attach_order(div_ev, order_map, dur_col="marriage_duration")
    wbar_married = float(married["weight"].mean())
    married["dur_band"] = _bands_vec(
        married["marriage_duration"].astype("int64").to_numpy(),
        DIV_LOWERS,
        len(DIV_BANDS),
    )
    married["ord_bit"] = (married["order"] >= 2).astype(int)
    div_ev["dur_band"] = _bands_vec(
        div_ev["marriage_duration"].astype("int64").to_numpy(),
        DIV_LOWERS,
        len(DIV_BANDS),
    )
    div_ev["ord_bit"] = (div_ev["order"] >= 2).astype(int)
    div_den = married.groupby(["dur_band", "ord_bit"])["weight"].sum()
    div_num = div_ev.groupby(["dur_band", "ord_bit"])["weight"].sum()
    div_table = np.zeros((len(DIV_BANDS), 2), dtype=np.float64)
    for b in range(len(DIV_BANDS)):
        for o in (0, 1):
            wnum = float(div_num.get((b, o), 0.0))
            wden = float(div_den.get((b, o), 0.0))
            div_table[b, o] = (wnum + wbar_married) / (
                wden + 2.0 * wbar_married
            )

    # ---- 3. widowhood (composed: train mortality + spousal gap) ----
    slices = mort.build_exposure_slices(demo, death_records)
    slices = slices[slices["person_id"].isin(train_ids)]
    mortality = {
        key: cell["psid_m"]
        for key, cell in mort.weighted_hazards(slices).items()
    }
    gap_by_sex = _spousal_gap(mh_records, attr_by, train_ids)

    # ---- 4. remarriage (weighted empirical, add-one) ----
    diss = train_py[
        train_py["marital_state"].isin(("divorced", "widowed"))
        & train_py["years_since_dissolution"].notna()
    ].copy()
    rem_ev = train_ev[
        (train_ev["transition"] == "remarriage")
        & train_ev["years_since_dissolution"].notna()
    ].copy()
    wbar_diss = float(diss["weight"].mean()) if len(diss) else 1.0
    diss["ysd_band"] = _bands_vec(
        diss["years_since_dissolution"].astype("int64").to_numpy(),
        YSD_LOWERS,
        len(YSD_BANDS),
    )
    rem_ev["ysd_band"] = _bands_vec(
        rem_ev["years_since_dissolution"].astype("int64").to_numpy(),
        YSD_LOWERS,
        len(YSD_BANDS),
    )
    rem_den = diss.groupby(["ysd_band", "marital_state", "sex"])[
        "weight"
    ].sum()
    rem_num = rem_ev.groupby(["ysd_band", "origin", "sex"])["weight"].sum()
    remarriage: dict[tuple[int, str, str], float] = {}
    for b in range(len(YSD_BANDS)):
        for origin in ("divorced", "widowed"):
            for sex in ("female", "male"):
                wnum = float(rem_num.get((b, origin, sex), 0.0))
                wden = float(rem_den.get((b, origin, sex), 0.0))
                remarriage[(b, origin, sex)] = (wnum + wbar_diss) / (
                    wden + 2.0 * wbar_diss
                )

    # ---- 5. fertility (weighted empirical age x parity x cohort) ----
    fertility = _fit_fertility(panel, birth_records, train_ids, birth_decade)

    meta = {
        "first_marriage_train_rows": fm_model.n_train_rows,
        "first_marriage_train_events": fm_model.n_train_events,
        "first_marriage_lbfgs_n_iter": fm_model.n_iter,
        "first_marriage_converged": fm_model.converged,
        "first_marriage_n_cohort_levels": len(fm_model.cohort_levels),
        "divorce_mean_married_weight": wbar_married,
        "remarriage_mean_dissolved_weight": wbar_diss,
        "mortality_cells": len(mortality),
        "gap_by_sex": gap_by_sex,
        "n_train_persons": len(train_ids),
    }
    return Components(
        first_marriage=fm_model,
        divorce=div_table,
        remarriage=remarriage,
        mortality=mortality,
        fertility=fertility,
        gap_by_sex=gap_by_sex,
        meta=meta,
    )


def _attach_order(
    frame: pd.DataFrame, order_map: pd.DataFrame, *, dur_col: str
) -> pd.DataFrame:
    """Attach marriage order via current_start = year - duration."""
    frame = frame.copy()
    frame["current_start"] = (
        frame["year"].to_numpy() - frame[dur_col].astype("int64").to_numpy()
    )
    merged = frame.merge(
        order_map.rename(columns={"start_year": "current_start"}),
        on=["person_id", "current_start"],
        how="left",
    )
    merged["order"] = merged["order"].fillna(1).astype("int64")
    return merged


def _spousal_gap(
    mh_records: pd.DataFrame,
    attr_by: pd.DataFrame,
    train_ids: set[int],
) -> dict[str, float]:
    """Train mean (spouse_age - self_age) by the person's sex.

    ``spouse_age - self_age = self_birth_year - spouse_birth_year``. Uses
    train self-persons whose marriage record carries a joinable spouse with a
    known birth year.
    """
    person_birth = (
        mh_records.dropna(subset=["birth_year"])
        .groupby("person_id")["birth_year"]
        .first()
    )
    rec = mh_records[
        mh_records["is_marriage"]
        & mh_records["spouse_person_id"].notna()
        & mh_records["person_id"].isin(train_ids)
    ].copy()
    rec["self_birth"] = rec["person_id"].map(person_birth).astype("float64")
    rec["spouse_birth"] = (
        rec["spouse_person_id"].map(person_birth).astype("float64")
    )
    rec = rec[rec["self_birth"].notna() & rec["spouse_birth"].notna()]
    rec["gap"] = rec["self_birth"] - rec["spouse_birth"]
    gap: dict[str, float] = {}
    for sex in ("female", "male"):
        sub = rec[rec["sex"] == sex]["gap"]
        gap[sex] = float(sub.mean()) if len(sub) else 0.0
    return gap


def _fit_fertility(
    panel: transitions.MaritalPanel,
    birth_records: pd.DataFrame,
    train_ids: set[int],
    birth_decade: pd.Series,
) -> dict[tuple[int, int, int], float]:
    """Weighted ASFR age-band x parity x birth-decade rate table (train)."""
    py = panel.person_years
    attrs = panel.attrs
    women_ids = set(attrs[attrs["sex"] == "female"]["person_id"]) & train_ids
    lo, hi = _ASFR_LO, _ASFR_HI
    wy = py[
        py["person_id"].isin(women_ids) & (py["age"] >= lo) & (py["age"] <= hi)
    ][["person_id", "year", "age", "weight"]].copy()

    # Births to train women (running parity = births strictly before y).
    be = g2f.births.birth_events(birth_records)
    be = be[
        (be["record_type"] == "birth")
        & be["parent_person_id"].isin(women_ids)
        & be["birth_year"].notna()
    ].copy()
    be = be.rename(columns={"parent_person_id": "person_id"})
    be["birth_year"] = be["birth_year"].astype("int64")
    births_by = {
        int(p): np.sort(g["birth_year"].to_numpy())
        for p, g in be.groupby("person_id")
    }

    wy = wy.reset_index(drop=True)
    wy["parity"] = _parity_vec(
        wy["person_id"].to_numpy(), wy["year"].to_numpy(), births_by
    )
    wy["age_band"] = _bands_vec(
        wy["age"].to_numpy(dtype=np.int64), ASFR_LOWERS, len(ASFR_BANDS)
    )
    wy["decade"] = wy["person_id"].map(birth_decade).to_numpy()
    wy["parity_band"] = np.minimum(wy["parity"].to_numpy(), 3)

    attr_by = attrs.set_index("person_id")
    be["mother_birth"] = (
        be["person_id"].map(attr_by["birth_year"]).astype("float64")
    )
    be["mother_censor"] = (
        be["person_id"].map(attr_by["censor_year"]).astype("float64")
    )
    be["mother_age"] = be["birth_year"] - be["mother_birth"]
    be = be[
        (be["mother_age"] >= lo)
        & (be["mother_age"] <= hi)
        & (be["birth_year"] <= be["mother_censor"])
    ].reset_index(drop=True)
    be["age_band"] = _bands_vec(
        be["mother_age"].to_numpy(dtype=np.int64), ASFR_LOWERS, len(ASFR_BANDS)
    )
    be["decade"] = be["person_id"].map(birth_decade).to_numpy()
    be["weight"] = be["person_id"].map(attr_by["weight"]).to_numpy()
    be["parity"] = _parity_vec(
        be["person_id"].to_numpy(), be["birth_year"].to_numpy(), births_by
    )
    be["parity_band"] = np.minimum(be["parity"].to_numpy(), 3)

    den = (
        wy.groupby(["age_band", "parity_band", "decade"])["weight"]
        .sum()
        .to_dict()
    )
    num = (
        be.groupby(["age_band", "parity_band", "decade"])["weight"]
        .sum()
        .to_dict()
    )
    table: dict[tuple[int, int, int], float] = {}
    for key, dsum in den.items():
        nsum = float(num.get(key, 0.0))
        table[(int(key[0]), int(key[1]), int(key[2]))] = (
            (nsum / float(dsum)) if dsum > 0 else 0.0
        )
    return table


def _parity_vec(
    person_ids: np.ndarray, years: np.ndarray, births_by: dict[int, np.ndarray]
) -> np.ndarray:
    """Prior-birth count per (person, year), vectorised per person group."""
    out = np.zeros(len(person_ids), dtype=np.int64)
    order = np.argsort(person_ids, kind="stable")
    sp = person_ids[order]
    sy = years[order]
    starts = np.searchsorted(sp, np.unique(sp), side="left")
    bounds = np.append(starts, len(sp))
    for i in range(len(starts)):
        seg = slice(bounds[i], bounds[i + 1])
        pid = int(sp[bounds[i]])
        arr = births_by.get(pid)
        if arr is None:
            continue
        out[order[seg]] = np.searchsorted(arr, sy[seg], side="left")
    return out


# --------------------------------------------------------------------------
# Vectorised annual simulation of the holdout persons
# --------------------------------------------------------------------------
_STATE = {"never_married": 0, "married": 1, "divorced": 2, "widowed": 3}
_STATE_ABSORB = 4  # separated / other (no modelled transition)


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Simulate side-A persons' family histories; return sim panel + births.

    Annual steps over each person's observed panel support
    ``[start_exposure_year, censor_year]`` from their observed initial
    state at entry, ``numpy.random.default_rng(sim_seed)``, one sequence per
    person. Returns a ``MaritalPanel`` built by the SAME ``transitions``
    assembly and a ``birth_history``-shaped simulated-births frame.
    """
    attrs = panel.attrs[panel.attrs["person_id"].isin(holdout_ids)].copy()
    attrs = attrs.sort_values("person_id").reset_index(drop=True)
    n = len(attrs)
    pid = attrs["person_id"].to_numpy(dtype=np.int64)
    by = attrs["birth_year"].to_numpy(dtype=np.float64)
    sex = attrs["sex"].to_numpy()
    is_male = (sex == "male").astype(np.float64)
    sy = attrs["start_exposure_year"].to_numpy(dtype=np.int64)
    ey = attrs["censor_year"].to_numpy(dtype=np.int64)
    decade = (by // 10 * 10).astype(np.int64)

    # Observed initial state at entry (min-year person-year per person).
    py = panel.person_years
    entry = (
        py[py["person_id"].isin(holdout_ids)]
        .sort_values("year")
        .groupby("person_id", as_index=False)
        .first()
    )
    entry_state = (
        entry.set_index("person_id")["marital_state"].reindex(pid).to_numpy()
    )
    entry_dur = entry.set_index("person_id")["marriage_duration"].reindex(pid)
    entry_ysd = entry.set_index("person_id")[
        "years_since_dissolution"
    ].reindex(pid)

    state = np.zeros(n, dtype=np.int64)
    cur_start = np.full(n, -1, dtype=np.int64)
    order = np.zeros(n, dtype=np.int64)
    diss_year = np.full(n, -1, dtype=np.int64)
    parity = np.zeros(n, dtype=np.int64)
    open_start = np.full(n, -1, dtype=np.int64)
    open_order = np.zeros(n, dtype=np.int64)

    for i in range(n):
        st = entry_state[i]
        if pd.isna(st) or st == "never_married":
            state[i] = 0
        elif st == "married":
            state[i] = 1
            d = entry_dur.iloc[i]
            d0 = int(d) if not pd.isna(d) else 0
            cur_start[i] = int(sy[i]) - d0
            order[i] = 1
            open_start[i] = cur_start[i]
            open_order[i] = 1
        elif st in ("divorced", "widowed"):
            state[i] = _STATE[st]
            j = entry_ysd.iloc[i]
            j0 = int(j) if not pd.isna(j) else 0
            diss_year[i] = int(sy[i]) - j0
            order[i] = 1
        else:  # separated / other
            state[i] = _STATE_ABSORB

    gap_arr = np.where(
        is_male == 1,
        components.gap_by_sex["male"],
        components.gap_by_sex["female"],
    )
    opp_is_male = 1.0 - is_male  # spouse opposite sex

    lookups = _build_sim_lookups(components)
    fert_didx = np.array(
        [lookups.decade_map.get(int(d), -1) for d in decade], dtype=np.int64
    )

    ep_person: list[int] = []
    ep_order: list[int] = []
    ep_start: list[int] = []
    ep_end: list[Any] = []
    ep_how: list[str] = []
    bi_person: list[int] = []
    bi_year: list[int] = []
    bi_order: list[int] = []

    def close_ep(idx_arr: np.ndarray, how: str, end_year: int) -> None:
        for i in idx_arr:
            ep_person.append(int(pid[i]))
            ep_order.append(int(open_order[i]))
            ep_start.append(int(open_start[i]))
            ep_end.append(int(end_year))
            ep_how.append(how)

    rng = np.random.default_rng(sim_seed)
    y0, y1 = int(sy.min()), int(ey.max())
    for y in range(y0, y1 + 1):
        active = (sy <= y) & (y <= ey)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            continue
        age = y - by[idx]
        u = rng.random(idx.size)
        st = state[idx]

        nm = st == 0
        if nm.any():
            sub = idx[nm]
            p_fm = components.first_marriage.predict(
                age[nm], is_male[sub], decade[sub]
            )
            marry = u[nm] < p_fm
            gi = sub[marry]
            order[gi] += 1
            cur_start[gi] = y
            state[gi] = 1
            open_start[gi] = y
            open_order[gi] = order[gi]

        mar = st == 1
        if mar.any():
            sub = idx[mar]
            dur = (y - cur_start[sub]).astype(np.int64)
            p_div = _divorce_probs(dur, order[sub], components.divorce)
            sp_age = age[mar] + gap_arr[sub]
            p_wid = _widow_probs(sp_age, opp_is_male[sub], lookups.mort_arr)
            um = u[mar]
            div = um < p_div
            wid = (~div) & (um < p_div + p_wid)
            gdi = sub[div]
            close_ep(gdi, "divorce", y)
            state[gdi] = 2
            diss_year[gdi] = y
            gwi = sub[wid]
            close_ep(gwi, "widowhood", y)
            state[gwi] = 3
            diss_year[gwi] = y

        diss = (st == 2) | (st == 3)
        if diss.any():
            sub = idx[diss]
            ysd = (y - diss_year[sub]).astype(np.int64)
            origin = st[diss]  # 2 divorced, 3 widowed
            p_rm = _remarriage_probs(
                ysd, origin, is_male[sub], lookups.rem_arr
            )
            rm = u[diss] < p_rm
            gri = sub[rm]
            order[gri] += 1
            cur_start[gri] = y
            state[gri] = 1
            diss_year[gri] = -1
            open_start[gri] = y
            open_order[gri] = order[gri]

        # Fertility: women aged 15-49, any marital state (incl. absorbed).
        age_all = (y - by).astype(np.int64)
        fert = (
            active
            & (sex == "female")
            & (age_all >= _ASFR_LO)
            & (age_all <= _ASFR_HI)
        )
        fidx = np.nonzero(fert)[0]
        if fidx.size:
            uf = rng.random(fidx.size)
            fage = (y - by[fidx]).astype(np.int64)
            p_birth = _fertility_probs(
                fage, parity[fidx], fert_didx[fidx], lookups.fert_arr
            )
            born = uf < p_birth
            gbi = fidx[born]
            for i in gbi:
                bi_person.append(int(pid[i]))
                bi_year.append(int(y))
                bi_order.append(int(parity[i]) + 1)
            parity[gbi] += 1

    # Close still-open marriages at censor (intact).
    still = np.nonzero(state == 1)[0]
    for i in still:
        ep_person.append(int(pid[i]))
        ep_order.append(int(open_order[i]))
        ep_start.append(int(open_start[i]))
        ep_end.append(pd.NA)
        ep_how.append("intact")

    sim_panel = _assemble_sim_panel(
        attrs, ep_person, ep_order, ep_start, ep_end, ep_how
    )
    sim_births = pd.DataFrame(
        {
            "parent_person_id": np.array(bi_person, dtype=np.int64),
            "birth_year": pd.array(bi_year, dtype="Int64"),
            "birth_order": pd.array(bi_order, dtype="Int64"),
            "record_type": pd.array(
                ["birth"] * len(bi_person), dtype="string"
            ),
            "is_event": np.ones(len(bi_person), dtype=bool),
        }
    )
    return sim_panel, sim_births


@dataclass
class _SimLookups:
    mort_arr: np.ndarray  # [mort_band, sex(0=f,1=m)]
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex(0=f,1=m)]
    fert_arr: np.ndarray  # [asfr_band, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookups:
    mort_arr = np.zeros((len(MORT_BANDS), 2), dtype=np.float64)
    for b, (lo, hi) in enumerate(MORT_BANDS):
        band = mort.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            mort_arr[b, si] = components.mortality.get(f"{band}|{sex}", 0.0)

    rem_arr = np.zeros((len(YSD_BANDS), 2, 2), dtype=np.float64)
    for (b, origin, sex), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[b, oi, si] = v

    decades = sorted({d for (_, _, d) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    fert_arr = np.zeros(
        (len(ASFR_BANDS), 4, max(len(decades), 1)), dtype=np.float64
    )
    for (ab, pb, d), v in components.fertility.items():
        fert_arr[ab, pb, decade_map[d]] = v
    return _SimLookups(mort_arr, rem_arr, fert_arr, decade_map)


def _divorce_probs(
    dur: np.ndarray, order: np.ndarray, div_table: np.ndarray
) -> np.ndarray:
    bands = _bands_vec(dur, DIV_LOWERS, len(DIV_BANDS))
    ocol = (order >= 2).astype(np.int64)
    return div_table[bands, ocol]


def _widow_probs(
    spouse_age: np.ndarray, spouse_is_male: np.ndarray, mort_arr: np.ndarray
) -> np.ndarray:
    bands = _bands_vec(
        np.rint(spouse_age).astype(np.int64), MORT_LOWERS, len(MORT_BANDS)
    )
    return mort_arr[bands, spouse_is_male.astype(np.int64)]


def _remarriage_probs(
    ysd: np.ndarray,
    origin_state: np.ndarray,
    is_male: np.ndarray,
    rem_arr: np.ndarray,
) -> np.ndarray:
    bands = _bands_vec(ysd, YSD_LOWERS, len(YSD_BANDS))
    origin_idx = (origin_state == 3).astype(np.int64)  # 2 div -> 0, 3 wid -> 1
    return rem_arr[bands, origin_idx, is_male.astype(np.int64)]


def _fertility_probs(
    age: np.ndarray,
    parity: np.ndarray,
    didx: np.ndarray,
    fert_arr: np.ndarray,
) -> np.ndarray:
    ab = _bands_vec(age, ASFR_LOWERS, len(ASFR_BANDS))
    pb = np.minimum(parity, 3)
    safe = np.where(didx >= 0, didx, 0)
    vals = fert_arr[ab, pb, safe]
    return np.where(didx >= 0, vals, 0.0)


def _assemble_sim_panel(
    attrs: pd.DataFrame,
    ep_person: list[int],
    ep_order: list[int],
    ep_start: list[int],
    ep_end: list[Any],
    ep_how: list[str],
) -> transitions.MaritalPanel:
    """Build a MaritalPanel from simulated episodes via the SAME assembly."""
    n_marr = (
        pd.Series(ep_person).value_counts()
        if ep_person
        else pd.Series(dtype="int64")
    )
    sim_attrs = attrs.copy()
    sim_attrs["n_marriages"] = (
        sim_attrs["person_id"].map(n_marr).fillna(0).astype("float64")
    )
    episodes = pd.DataFrame(
        {
            "person_id": np.array(ep_person, dtype=np.int64),
            "marriage_order": pd.array(ep_order, dtype="Int64"),
            "start_year": pd.array(ep_start, dtype="Int64"),
            "start_month": pd.array([pd.NA] * len(ep_person), dtype="Int64"),
            "episode_end_year": pd.array(ep_end, dtype="Int64"),
            "how_ended": pd.array(ep_how, dtype="string"),
            "spouse_person_id": pd.array(
                [pd.NA] * len(ep_person), dtype="Int64"
            ),
            "last_known_status": pd.array(
                [pd.NA] * len(ep_person), dtype="string"
            ),
        }
    )
    episodes["episode_duration_years"] = (
        episodes["episode_end_year"] - episodes["start_year"]
    ).astype("Int64")
    changepoints = transitions._changepoints(episodes, sim_attrs)
    person_years = transitions._assign_state(
        transitions._person_years_frame(sim_attrs), changepoints
    )
    events = transitions._events_frame(episodes, sim_attrs)
    return transitions.MaritalPanel(
        person_years=person_years, events=events, attrs=sim_attrs
    )


# --------------------------------------------------------------------------
# Precheck (bit-exact reproduction of the committed scoring path)
# --------------------------------------------------------------------------
def run_precheck(
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
    floor: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce committed reference moments, per-seed rate_a, holdout ids."""
    ref_w = transitions.reference_moments(panel, fert, weighted=True)
    committed_ref = floor["reference_moments"]
    ref_devs = {
        key: abs(ref_w[key]["rate"] - committed_ref[key]["rate"])
        for key in committed_ref
    }
    ref_max = max(ref_devs.values())

    # Per gate-seed rate_a + holdout sha256.
    per_seed = []
    committed_ho = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    committed_ns = {p["seed"]: p for p in floor["noise_floor_per_seed"]}
    rate_a_max = 0.0
    sha_all_ok = True
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        sha_ok = digest == committed_ho[seed]["holdout_person_id_sha256"]
        sha_all_ok = sha_all_ok and sha_ok
        cells_a = transitions.reference_moments(
            panel, fert, set(ids), weighted=True
        )
        committed_cells = committed_ns[seed]["cells"]
        seed_dev = max(
            abs(cells_a[key]["rate"] - committed_cells[key]["rate_a"])
            for key in committed_cells
        )
        rate_a_max = max(rate_a_max, seed_dev)
        per_seed.append(
            {
                "seed": seed,
                "holdout_sha256_match": bool(sha_ok),
                "n_holdout": len(ids),
                "rate_a_max_abs_deviation": float(seed_dev),
            }
        )

    ok = bool(
        ref_max <= EXACT_ATOL and rate_a_max <= EXACT_ATOL and sha_all_ok
    )
    return {
        "note": (
            "Hard-stop precheck (gate-1 mirror): the scoring path must "
            "reproduce every committed full-panel reference moment and every "
            "committed per-gate-seed rate_a bit-for-bit, and each gate seed's "
            "holdout-id sha256, before any candidate is simulated."
        ),
        "reference_moments_max_abs_deviation": float(ref_max),
        "reference_moments_exact": bool(ref_max <= EXACT_ATOL),
        "n_reference_cells": len(committed_ref),
        "per_seed": per_seed,
        "rate_a_max_abs_deviation": float(rate_a_max),
        "rate_a_exact": bool(rate_a_max <= EXACT_ATOL),
        "holdout_sha256_all_match": bool(sha_all_ok),
        "all_reproduced_exactly": ok,
    }


# --------------------------------------------------------------------------
# Per-seed scoring
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    floor: dict[str, Any],
    tol: dict[str, float],
    report_only: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """Fit on side B, simulate side A, score every cell against rate_a."""
    t0 = time.time()
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    components = fit_components(
        panel, demo, death_records, mh_records, birth_records, order_map, ids_b
    )
    sim_panel, sim_births = simulate_holdout(
        panel, ids_a, components, SIM_SEED_BASE + seed
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    cand = transitions.reference_moments(
        sim_panel, sim_fert, ids_a, weighted=True
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]

    def score_cell(key: str) -> dict[str, Any]:
        rate_a = float(committed_cells[key]["rate_a"])
        r_cand = float(cand[key]["rate"])
        n_cand = int(cand[key]["n_events"])
        if r_cand > 0 and rate_a > 0:
            s = float(abs(math.log(r_cand / rate_a)))
        else:
            s = float("inf")
        return {
            "r_candidate": r_cand,
            "rate_a": rate_a,
            "n_events_candidate": n_cand,
            "log_ratio_abs": s if math.isfinite(s) else None,
            "score": s,
        }

    gated_cells: dict[str, Any] = {}
    n_gated_pass = 0
    for key in sorted(tol):
        rec = score_cell(key)
        rec["tolerance"] = float(tol[key])
        rec["pass"] = bool(rec["score"] <= tol[key])
        n_gated_pass += rec["pass"]
        gated_cells[key] = rec

    report_cells: dict[str, Any] = {}
    for key in sorted(report_only):
        rec = score_cell(key)
        rec["gated"] = False
        report_cells[key] = rec

    seed_pass = n_gated_pass == len(tol)
    elapsed = round(time.time() - t0, 1)
    if verbose:
        fails = [k for k, v in gated_cells.items() if not v["pass"]]
        print(
            f"seed {seed}: {n_gated_pass}/{len(tol)} gated pass "
            f"(seed_pass={seed_pass}); fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "sim_seed": SIM_SEED_BASE + seed,
        "component_meta": components.meta,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Verdict assembly (per-family / per-seed pass matrix)
# --------------------------------------------------------------------------
_FAMILY_PREFIX = {
    "first_marriage": lambda k: k.startswith("first_marriage"),
    "divorce": lambda k: k.startswith("divorce."),
    "widowhood": lambda k: k.startswith("widowhood."),
    "remarriage": lambda k: k.startswith("remarriage."),
    "occupancy": lambda k: k.startswith("ever_married_by_")
    and "|" in k
    or k.startswith("mean_lifetime_marriages"),
    "nuptiality_cohort": lambda k: k.startswith("ever_married_by_40.c"),
    "stock_occupancy": lambda k: k.startswith("share_"),
    "fertility": lambda k: k.startswith("asfr.")
    or k.startswith("completed_fertility."),
}


def _family_of(cell: str, tol: dict[str, float]) -> str:
    for fam, pred in _FAMILY_PREFIX.items():
        if pred(cell):
            return fam
    return "other"


def build_verdict(
    per_seed: list[dict[str, Any]], tol: dict[str, float]
) -> dict[str, Any]:
    seed_pass = {s["seed"]: s["seed_pass"] for s in per_seed}
    n_seeds_pass = sum(seed_pass.values())
    gate_pass = n_seeds_pass >= 4

    # Per-family per-seed pass counts + failing cells.
    families = sorted(set(_family_of(k, tol) for k in tol))
    per_block: dict[str, Any] = {}
    for fam in families:
        cells = sorted(k for k in tol if _family_of(k, tol) == fam)
        by_seed = {}
        failing = []
        for s in per_seed:
            npass = sum(s["gated_cells"][c]["pass"] for c in cells)
            by_seed[s["seed"]] = {"n_pass": npass, "n_cells": len(cells)}
            for c in cells:
                rec = s["gated_cells"][c]
                if not rec["pass"]:
                    failing.append(
                        {
                            "cell": c,
                            "seed": s["seed"],
                            "score": rec["score"],
                            "tolerance": rec["tolerance"],
                            "r_candidate": rec["r_candidate"],
                            "rate_a": rec["rate_a"],
                        }
                    )
        per_block[fam] = {
            "cells": cells,
            "n_cells": len(cells),
            "per_seed_pass": by_seed,
            "failing_cells": failing,
        }

    all_failing = [
        {
            "cell": c,
            "seed": s["seed"],
            "family": _family_of(c, tol),
            "score": s["gated_cells"][c]["score"],
            "tolerance": s["gated_cells"][c]["tolerance"],
            "r_candidate": s["gated_cells"][c]["r_candidate"],
            "rate_a": s["gated_cells"][c]["rate_a"],
        }
        for s in per_seed
        for c in sorted(tol)
        if not s["gated_cells"][c]["pass"]
    ]
    return {
        "n_gate_seeds": len(per_seed),
        "n_gated_cells": len(tol),
        "seed_pass": seed_pass,
        "n_seeds_pass": n_seeds_pass,
        "gate_2_pass": bool(gate_pass),
        "rule": (
            "A seed passes iff every one of the 46 gated cells holds "
            "(|ln(r_candidate / rate_a)| <= locked tolerance); the gate "
            "passes iff >= 4 of the 5 gate seeds pass."
        ),
        "per_block": per_block,
        "all_failing_gated_cells": all_failing,
    }


def report_only_summary(
    per_seed: list[dict[str, Any]], report_only: list[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cell in sorted(report_only):
        scores = [s["report_only_cells"][cell]["score"] for s in per_seed]
        finite = [x for x in scores if math.isfinite(x)]
        out[cell] = {
            "per_seed_score": {
                s["seed"]: s["report_only_cells"][cell]["score"]
                for s in per_seed
            },
            "mean_score": (float(np.mean(finite)) if finite else None),
            "max_score": (float(np.max(finite)) if finite else None),
        }
    return {
        "note": (
            "The 16 report-only cells (below the 20-event floor, above the "
            "T_max power cap, or superseded by a gating aggregate). Scored "
            "with the same |ln(r_candidate / rate_a)| statistic and published; "
            "never gated."
        ),
        "cells": out,
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _sha_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    import scipy
    import sklearn

    return {
        "populace_dynamics_sha": _git_sha(ROOT),
        "gate2_floor_run": "runs/gate2_floors_v2.json",
        "gate2_floor_sha256": _sha_of_file(FLOOR_RUN),
        "gates_yaml_locked": bool(thresholds.get("locked", False)),
        "gates_yaml_status": thresholds.get("status"),
        "sklearn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scipy_version": str(scipy.__version__),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _order_map(mh_records: pd.DataFrame) -> pd.DataFrame:
    """Per-person marriage order by start year (for the divorce order strat)."""
    ep = marriage.marriage_episodes(mh_records)
    ep = ep[ep["start_year"].notna()].copy()
    ep["start_year"] = ep["start_year"].astype("int64")
    ep = ep.sort_values(["person_id", "start_year"])
    ep["order"] = ep.groupby("person_id").cumcount() + 1
    return ep[["person_id", "start_year", "order"]].drop_duplicates(
        ["person_id", "start_year"]
    )


def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = _load_cache(cache_path)

    thresholds = load_gate2_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    tol = gated_tolerances(thresholds)
    if len(tol) != 46:
        raise RuntimeError(
            f"expected 46 gated tolerances, got {len(tol)} from gates.yaml."
        )
    report_only = list(thresholds["report_only"])

    floor = json.loads(FLOOR_RUN.read_text())
    gated_set = set(floor["gate_partition"]["gate_eligible"])
    if set(tol) != gated_set:
        raise RuntimeError(
            "gates.yaml gated tolerances do not match the floor's "
            "gate_partition; refusing to score a mismatched cell set."
        )

    mh_records = marriage.marriage_history()
    birth_records = g2f.births.birth_history()
    death_records = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, fert, data_meta = g2f.load_panels()
    order_map = _order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons"
        )

    # Hard-stop precheck BEFORE any candidate is simulated.
    precheck = run_precheck(panel, fert, floor)
    if verbose:
        print(
            "precheck all_reproduced_exactly="
            f"{precheck['all_reproduced_exactly']} "
            f"(ref dev={precheck['reference_moments_max_abs_deviation']:.2e}, "
            f"rate_a dev={precheck['rate_a_max_abs_deviation']:.2e}, "
            f"sha_all={precheck['holdout_sha256_all_match']})"
        )
    if not precheck["all_reproduced_exactly"]:
        raise RuntimeError(
            "Scoring path does not reproduce the committed gate-2 floor "
            "(reference moments / per-seed rate_a / holdout sha256) to bit "
            "precision; refusing to proceed."
        )

    per_seed: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"seed_{seed}"
        if key in cache:
            if verbose:
                print(f"seed {seed}: cached")
            per_seed.append(cache[key])
            continue
        result = score_seed(
            seed,
            panel,
            fert,
            demo,
            death_records,
            mh_records,
            birth_records,
            order_map,
            floor,
            tol,
            report_only,
            verbose,
        )
        cache[key] = json.loads(json.dumps(result, default=_json_default))
        _save_cache(cache_path, cache)
        per_seed.append(cache[key])

    verdict = build_verdict(per_seed, tol)
    report_block = report_only_summary(per_seed, report_only)
    seed_conjunction = [
        {
            "seed": s["seed"],
            "n_gated_pass": s["n_gated_pass"],
            "n_gated_fail": s["n_gated_fail"],
            "seed_pass": s["seed_pass"],
        }
        for s in per_seed
    ]

    modal = _modal_failure_check(verdict)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 1",
        "spec_registration": SPEC_REGISTRATION,
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 merge 82006877 + flip "
            "#81); protocol/views/tolerances read at runtime, no threshold "
            "moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.60-0.70",
            "conjunction_estimate": 0.65,
            "component_probabilities": {
                "hazard_cells": 0.95,
                "cohort_cells": 0.8,
                "stocks": 0.75,
                "fertility": 0.9,
            },
            "modal_failure": "stock occupancy cells (accumulation drift)",
            "secondary_failure": [
                "ever_married_by_40.c1980s (right-censoring x trend)",
                "remarriage.after_divorce (smoothing bias in thin late bands)",
            ],
            "registration": SPEC_REGISTRATION,
        },
        "model": _model_block(),
        "protocol": {
            "option": "a (gate-1 mirror; LOCKED gates.yaml gate_2)",
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "the holdout, side B = the train complement"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "sim_rng_rule": "numpy.random.default_rng(4200 + seed)",
            "one_sequence_per_person": True,
            "scored_against": (
                "side A's own empirical rate (rate_a in "
                "runs/gate2_floors_v2.json noise_floor_per_seed)"
            ),
            "statistic": "|ln(r_candidate / rate_a)| per cell",
            "conjunction": (
                "all 46 gated cells per seed AND >= 4 of 5 gate seeds"
            ),
            "weight_definition": (
                "person-constant most-recent positive PSID cross-sectional "
                "weight (populace_dynamics.data.panels.demographic_panel); "
                "every gated statistic weighted, none unweighted"
            ),
        },
        "data": data_meta,
        "precheck": precheck,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "floor_run": "runs/gate2_floors_v2.json",
            "faithful_candidate_oc": floor["faithful_candidate_oc"][
                "p_gate_pass_4_of_5"
            ],
        },
        "revision_pins": _revision_pins(thresholds),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_2_pass={v['gate_2_pass']} "
            f"({v['n_seeds_pass']}/5 seeds pass)"
        )
        print(f"seed_pass: {v['seed_pass']}")
        print(
            "modal failure (stocks/c1980s/remarriage.after_divorce) "
            f"materialized: {modal['any_materialized']}"
        )
    return artifact


def _modal_failure_check(verdict: dict[str, Any]) -> dict[str, Any]:
    """Did the registered modal / secondary failure cells fail any seed?"""
    fails_by_cell: dict[str, list[int]] = {}
    for f in verdict["all_failing_gated_cells"]:
        fails_by_cell.setdefault(f["cell"], []).append(f["seed"])
    stocks = [c for c in fails_by_cell if c.startswith("share_")]
    c1980s = "ever_married_by_40.c1980s" in fails_by_cell
    after_divorce = "remarriage.after_divorce" in fails_by_cell
    return {
        "stock_occupancy_failed": bool(stocks),
        "stock_cells_failed": {c: fails_by_cell[c] for c in stocks},
        "c1980s_failed": bool(c1980s),
        "c1980s_seeds": fails_by_cell.get("ever_married_by_40.c1980s", []),
        "remarriage_after_divorce_failed": bool(after_divorce),
        "remarriage_after_divorce_seeds": fails_by_cell.get(
            "remarriage.after_divorce", []
        ),
        "any_materialized": bool(stocks or c1980s or after_divorce),
    }


def _model_block() -> dict[str, Any]:
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic (restricted) "
                "spline on age, knots 20/25/30/40 (3 basis columns), sex, "
                "birth-decade cohort main effects + age-spline x cohort "
                "interaction; sklearn LogisticRegression(penalty='l2', C=1.0, "
                "lbfgs) fit with sample_weight = person-year PSID weight "
                "(effectively the weighted MLE at train scale); design "
                "standardised for conditioning"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x "
                "marriage order (1st vs 2+), add-one (Laplace) smoothed at the "
                "train mean married person-year weight"
            ),
            "widowhood": (
                "COMPOSED: train age-band x sex mortality hazard "
                "(build_mortality_floors construction, train persons) applied "
                "to the spouse (opposite sex, age = self age + train mean "
                "spousal age gap by sex); widowhood = induced transition"
            ),
            "remarriage": (
                "weighted empirical hazard by years-since-dissolution band x "
                "origin (divorced/widowed) x sex, add-one smoothed at the "
                "train mean dissolved person-year weight"
            ),
            "fertility": (
                "weighted empirical age-band x parity (0/1/2/3+) rates by "
                "birth-decade cohort, train-estimated (no smoothing)"
            ),
        },
        "registered_ambiguity_resolutions": {
            "logistic_estimator": (
                "'discrete-time logistic hazard fit' = MLE; sklearn "
                "LogisticRegression default L2 (C=1.0) is negligible vs the "
                "weighted data term at ~2-3e5 train person-years, so the fit "
                "is effectively the weighted MLE while remaining numerically "
                "stable on thin cohorts (statsmodels absent; sklearn 1.9)"
            ),
            "spline": (
                "natural cubic = restricted cubic spline (Harrell), 4 knots -> "
                "3 basis columns, linear beyond the boundary knots"
            ),
            "table_weighting": (
                "empirical tables estimated WEIGHTED (person-year PSID "
                "weight), consistent with the weighted gated statistics"
            ),
            "add_one_smoothing": (
                "Laplace: one pseudo-event + one pseudo-non-event at the "
                "train cell-universe mean person-year weight -- (Wnum + w) / "
                "(Wden + 2w); negligible on populated cells"
            ),
            "competing_risks": (
                "married-year single uniform: divorce if u < p_div, widowhood "
                "if p_div <= u < p_div + p_wid, else stays married"
            ),
            "spouse_sex": "opposite sex (heterosexual-marriage convention)",
            "initial_state": (
                "each holdout person's observed marital state at their entry "
                "person-year (41,346 never_married / 62 married / 1 separated "
                "at entry); simulation begins from it"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    artifact = run(verbose=True, cache_path=Path(args.cache))
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
