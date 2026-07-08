"""Gate-2 candidate 2 (run 1): candidate 1 + two named fixes.

The SECOND pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4911167286 (``SPEC_REGISTRATION``): candidate 1's frozen spec
(comment 4910914098) verbatim EXCEPT two named deltas. One-shot; no
constant moves after the registration comment.

The two deltas vs candidate 1 (everything else byte-identical)
--------------------------------------------------------------
1. **First marriage** -- the discrete-time logistic hazard adds an
   **age-spline x sex** interaction. The full design is age spline x sex +
   age spline x cohort + sex (same knots 20/25/30/40, same fitting rule as
   candidate 1: weighted MLE via ``sklearn.LogisticRegression`` at the
   default L2, standardised design). Candidate 1 gave sex a main effect
   only, forcing one age profile onto both sexes and missing the female
   early-marriage peak (``first_marriage.18-24|female`` failed all five
   seeds). This interaction identifies a separate female age profile.
2. **Widowhood composition** -- the spouse-death hazard table becomes
   **decade-period x age band x sex** (train-estimated, add-one / Laplace
   smoothed at the train mean slice weight, same mortality-foundation
   construction otherwise). Candidate 1 pooled all periods into one
   time-invariant table; real mortality declined across the panel's five
   decades, so integrating a pooled hazard drifted the widowed-stock
   accumulation (``share_widowed.75+|female`` / ``.65-74|female`` failed).
   Period stratification lets mortality decline enter the composition. The
   spousal-age-gap imputation is unchanged.

Everything else -- divorce, remarriage, fertility components, the
simulation loop, the RNG rule ``numpy.random.default_rng(4200 + seed)``,
one simulated sequence per person, and the LOCKED gate-2 protocol
(gates.yaml ``gate_2``, ratified PR #79 + flip #81) -- is byte-identical to
candidate 1. This runner IMPORTS candidate 1's machinery
(``run_gate2_candidate1``) and reuses every unchanged function: the
unchanged components come straight from ``candidate1.fit_components`` (so
they are provably identical), and only the two delta'd fields are
recomputed. The scoring path, precheck, verdict assembly, and report-only
handling are candidate 1's, imported unchanged.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit). Run from the repository root with the PSID history files
staged::

    .venv/bin/python scripts/run_gate2_candidate2.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Candidate 1 supplies ALL shared machinery (the floor loader + cell
# constructors it in turn reuses, the natural-cubic-spline basis, the
# divorce/remarriage/fertility/gap fitters, the vectorised simulation
# helpers, the precheck, the verdict assembly, and the report-only
# summary). The two deltas are the only functions re-implemented here; the
# unchanged components are taken directly from ``candidate1.fit_components``
# so they are byte-identical by construction. ``build_mortality_floors``
# supplies the mortality-foundation construction the widowhood composition
# reuses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import build_mortality_floors as mort  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v2.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v2"
RUN_NAME = "gate2_hazard_v2"

#: This run's frozen-spec registration (issue #42, comment 4911167286).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911167286"
)
#: The candidate-1 spec this build minimally deltas (comment 4910914098).
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The two named deltas (registration comment 4911167286).
DELTAS_VS_CANDIDATE1 = (
    "first-marriage logistic hazard adds an age-spline x sex interaction "
    "(full: age spline x sex + age spline x cohort + sex; same knots "
    "20/25/30/40, same weighted-MLE fitting rule)",
    "widowhood composition's spouse-death hazard table becomes "
    "decade-period x age band x sex (train-estimated, add-one smoothed at "
    "the train mean slice weight; same mortality-foundation construction "
    "otherwise); the spousal-age-gap imputation is unchanged",
)

# --- Frozen dials + band constants + pure helpers, reused from candidate 1
# (byte-identical; imported, never redefined). ---------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
SPLINE_KNOTS = c1.SPLINE_KNOTS
EXACT_ATOL = c1.EXACT_ATOL

DIV_BANDS = c1.DIV_BANDS
YSD_BANDS = c1.YSD_BANDS
ASFR_BANDS = c1.ASFR_BANDS
MORT_BANDS = c1.MORT_BANDS
DIV_LOWERS = c1.DIV_LOWERS
YSD_LOWERS = c1.YSD_LOWERS
ASFR_LOWERS = c1.ASFR_LOWERS
MORT_LOWERS = c1.MORT_LOWERS
_ASFR_LO = c1._ASFR_LO
_ASFR_HI = c1._ASFR_HI
_STATE = c1._STATE
_STATE_ABSORB = c1._STATE_ABSORB

_bands_vec = c1._bands_vec
_divorce_probs = c1._divorce_probs
_remarriage_probs = c1._remarriage_probs
_fertility_probs = c1._fertility_probs
_assemble_sim_panel = c1._assemble_sim_panel
Components = c1.Components

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate2_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA 1: first-marriage logistic hazard + age-spline x sex interaction
# --------------------------------------------------------------------------
class FirstMarriageModelC2(c1.FirstMarriageModel):
    """Candidate 1's first-marriage hazard + an age-spline x sex block.

    Inherits candidate 1's dataclass fields, ``_design``, and ``predict``
    (which call ``self._raw_design``); only the design construction changes.
    The added block is ``age-spline x sex`` -- three columns
    ``spline[:, c] * is_male`` -- giving the full design ``age spline + sex
    + age-spline x sex + cohort + age-spline x cohort``. The estimator,
    standardisation, weights, and knots are candidate 1's, unchanged.
    """

    def _raw_design(
        self, age: np.ndarray, is_male: np.ndarray, decade: np.ndarray
    ) -> np.ndarray:
        spline = c1.ncs_basis(age, self.knots)  # (n, 3)
        male = is_male.astype(np.float64).reshape(-1, 1)
        # DELTA 1: age-spline x sex interaction block appended to candidate
        # 1's [spline, sex] main effects (candidate 1 had no age x sex).
        parts = [spline, male, spline * male]
        dummies = []
        for level in self.cohort_levels[1:]:
            dummies.append((decade == level).astype(np.float64))
        if dummies:
            dmat = np.column_stack(dummies)  # (n, m-1)
            parts.append(dmat)
            # age-spline x cohort interaction (unchanged from candidate 1).
            for c in range(spline.shape[1]):
                parts.append(spline[:, [c]] * dmat)
        return np.column_stack(parts)


def fit_first_marriage(
    train_py: pd.DataFrame, event_years: set[tuple[int, int]]
) -> FirstMarriageModelC2:
    """Fit the candidate-2 first-marriage hazard.

    Byte-identical to ``candidate1.fit_first_marriage`` EXCEPT the model
    class is :class:`FirstMarriageModelC2` (whose design adds the
    age-spline x sex block). Same estimator, standardisation, weighting,
    knots, and convergence bookkeeping.
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
    model = FirstMarriageModelC2(
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
# DELTA 2: decade-period x age band x sex spouse-death hazard table
# --------------------------------------------------------------------------
def fit_period_mortality(
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    train_ids: set[int],
) -> dict[str, float]:
    """Train decade-period x age band x sex central death-rate table.

    The same mortality-foundation construction as candidate 1
    (:func:`build_mortality_floors.build_exposure_slices` and the weighted
    central rate ``m = sum(w*death) / sum(w*exposure)``), but (i) stratified
    by the slice's decade-period -- the wave-anchored ``start_wave`` decade
    the construction already tags every slice with, the module's own period
    concept -- and (ii) add-one (Laplace) smoothed at the train mean slice
    weight: one pseudo-death + one pseudo-survivor, ``(Wd + w) / (We + 2w)``,
    so thin period x band x sex cells stay finite (mirrors candidate 1's
    divorce/remarriage smoothing). Mortality decline across the panel's
    decades thus enters the widowhood composition. Keyed
    ``"period|band|sex"`` (e.g. ``"1990|75-84|female"``).
    """
    slices = mort.build_exposure_slices(demo, death_records)
    slices = slices[slices["person_id"].isin(train_ids)].copy()
    slices["period"] = (slices["start_wave"] // 10 * 10).astype(np.int64)
    wbar = float(slices["weight"].mean()) if len(slices) else 1.0
    slices["we"] = slices["weight"] * slices["exposure"]
    slices["wd"] = slices["weight"] * slices["death"]
    grouped = slices.groupby(["period", "band", "sex"], observed=True).agg(
        we=("we", "sum"),
        wd=("wd", "sum"),
    )
    mortality: dict[str, float] = {}
    for (period, band, sex), row in grouped.iterrows():
        mortality[f"{int(period)}|{band}|{sex}"] = (float(row.wd) + wbar) / (
            float(row.we) + 2.0 * wbar
        )
    return mortality


# --------------------------------------------------------------------------
# Fitted components (candidate 1's, with the two delta'd fields swapped)
# --------------------------------------------------------------------------
def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Fit all five components on side B, deltas 1 and 2 applied.

    The unchanged components (divorce, remarriage, fertility, spousal gap)
    are taken directly from :func:`candidate1.fit_components` -- so they are
    byte-identical to candidate 1 by construction, not by re-implementation.
    Only the two delta'd fields are recomputed: the first-marriage model
    (age x sex) and the spouse-death mortality table (decade-period).
    """
    base = c1.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    py = panel.person_years
    ev = panel.events
    attrs = panel.attrs
    train_py = py[py["person_id"].isin(train_ids)]
    train_ev = ev[ev["person_id"].isin(train_ids)]
    attr_by = attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")

    # DELTA 1: refit first marriage with the age-spline x sex interaction on
    # the same train never-married person-years candidate 1 used.
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

    # DELTA 2: period-varying add-one-smoothed spouse mortality.
    mortality = fit_period_mortality(demo, death_records, train_ids)

    base.first_marriage = fm_model
    base.mortality = mortality
    base.meta["first_marriage_train_rows"] = fm_model.n_train_rows
    base.meta["first_marriage_train_events"] = fm_model.n_train_events
    base.meta["first_marriage_lbfgs_n_iter"] = fm_model.n_iter
    base.meta["first_marriage_converged"] = fm_model.converged
    base.meta["first_marriage_n_cohort_levels"] = len(fm_model.cohort_levels)
    base.meta["first_marriage_design"] = (
        "age_spline + sex + age_spline:sex + cohort + age_spline:cohort"
    )
    base.meta["mortality_cells"] = len(mortality)
    base.meta["mortality_periods"] = sorted(
        {int(k.split("|")[0]) for k in mortality}
    )
    base.meta["mortality_stratification"] = "decade_period x band x sex"
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 1's, with period-aware widowhood)
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC2:
    mort_arr: np.ndarray  # [period_idx, mort_band, sex(0=f,1=m)]
    period_levels: list[int]
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex(0=f,1=m)]
    fert_arr: np.ndarray  # [asfr_band, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC2:
    """As candidate 1's, but ``mort_arr`` gains a leading decade-period axis.

    The remarriage and fertility lookups are built exactly as candidate 1
    does. The mortality lookup is indexed ``[period_idx, band, sex]`` from
    the ``"period|band|sex"`` keys of the delta-2 table.
    """
    period_levels = sorted(
        {int(k.split("|")[0]) for k in components.mortality}
    )
    if not period_levels:
        period_levels = [0]
    pidx_of = {p: i for i, p in enumerate(period_levels)}
    band_index = {
        mort.band_label(lo, hi): b for b, (lo, hi) in enumerate(MORT_BANDS)
    }
    mort_arr = np.zeros(
        (len(period_levels), len(MORT_BANDS), 2), dtype=np.float64
    )
    for key, v in components.mortality.items():
        ps, band, sex = key.split("|")
        mort_arr[
            pidx_of[int(ps)], band_index[band], 0 if sex == "female" else 1
        ] = v

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
    return _SimLookupsC2(
        mort_arr, period_levels, rem_arr, fert_arr, decade_map
    )


def _period_index(year: int, period_levels_arr: np.ndarray) -> int:
    """Index of the fitted decade-period nearest the year's calendar decade.

    The exact decade of ``year`` is present in the fitted table for every
    year in the panel's observation window; the nearest-decade fallback only
    engages for a simulated year outside the mortality panel's exposure
    span (well before 1969), where no period is identified.
    """
    dperiod = (year // 10) * 10
    return int(np.argmin(np.abs(period_levels_arr - dperiod)))


def _widow_probs(
    spouse_age: np.ndarray,
    spouse_is_male: np.ndarray,
    period_idx: int,
    mort_arr: np.ndarray,
) -> np.ndarray:
    """As candidate 1's widow probs, selecting the current period's slice."""
    bands = _bands_vec(
        np.rint(spouse_age).astype(np.int64), MORT_LOWERS, len(MORT_BANDS)
    )
    return mort_arr[period_idx, bands, spouse_is_male.astype(np.int64)]


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 1's simulation, with the delta-2 period-aware widowhood.

    Byte-identical to :func:`candidate1.simulate_holdout` EXCEPT the
    widowhood hazard is drawn from the decade-period slice of the current
    simulated year (delta 2). No RNG draw is added, removed, or resized --
    the per-year uniform blocks (``rng.random(n_active)`` then
    ``rng.random(n_fertile)``) are demographically sized exactly as in
    candidate 1, so the RNG-isolated fertility subprocess is bit-identical.
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
    period_levels_arr = np.asarray(lookups.period_levels, dtype=np.int64)
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
            # DELTA 2: spouse mortality of the current year's decade-period.
            period_idx = _period_index(y, period_levels_arr)
            p_wid = _widow_probs(
                sp_age, opp_is_male[sub], period_idx, lookups.mort_arr
            )
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


# --------------------------------------------------------------------------
# Per-seed scoring (candidate 1's, calling the candidate-2 fit + simulate)
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
    """Fit side B, simulate side A, score every cell against rate_a.

    Identical to :func:`candidate1.score_seed` except it calls the
    candidate-2 :func:`fit_components` and :func:`simulate_holdout` (the two
    deltas). The split, scoring statistic, gated/report partition, and
    per-seed record are candidate 1's.
    """
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
# Modal-failure check (registered c2 modal + candidate-1 killer movement)
# --------------------------------------------------------------------------
MALE_OCCUPANCY_CELLS = ("ever_married_by_40|male", "ever_married_by_60|male")
CANDIDATE1_KILLER_CELLS = (
    "first_marriage.18-24|female",
    "first_marriage.25-34|female",
    "first_marriage.35+|female",
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Did the registered modal failure materialize, and how did c1's move?

    Registered modal failure (comment 4911167286): male ever-married
    occupancy at its ln-scale-tight tolerance. Also tracks the candidate-1
    killers the two deltas target, so the artifact records their movement
    whether they pass or fail.
    """
    fails_by_cell: dict[str, list[int]] = {}
    for f in verdict["all_failing_gated_cells"]:
        fails_by_cell.setdefault(f["cell"], []).append(f["seed"])

    def track(cell: str) -> dict[str, Any]:
        return {
            "tolerance": per_seed[0]["gated_cells"][cell]["tolerance"],
            "per_seed_score": {
                s["seed"]: s["gated_cells"][cell]["score"] for s in per_seed
            },
            "per_seed_pass": {
                s["seed"]: s["gated_cells"][cell]["pass"] for s in per_seed
            },
            "failed_seeds": sorted(fails_by_cell.get(cell, [])),
        }

    male_occ_failed = {
        c: sorted(fails_by_cell[c])
        for c in MALE_OCCUPANCY_CELLS
        if c in fails_by_cell
    }
    any_male_occ = bool(male_occ_failed)
    return {
        "registered_modal": (
            "male ever-married occupancy (ever_married_by_*|male) at its "
            "ln-scale-tight tolerance"
        ),
        "male_ever_married_occupancy_failed": any_male_occ,
        "male_ever_married_occupancy_seeds": male_occ_failed,
        "male_ever_married_occupancy_track": {
            c: track(c) for c in MALE_OCCUPANCY_CELLS
        },
        "any_materialized": any_male_occ,
        "candidate1_killer_movement": {
            c: track(c) for c in CANDIDATE1_KILLER_CELLS
        },
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-2 schema + a c1-runner sha."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    pins["candidate1_runner"] = "scripts/run_gate2_candidate1.py"
    pins["candidate1_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "run_gate2_candidate1.py"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 1's model block, edited for the two deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 1 + two named "
            "fixes: age x sex first-marriage interaction; decade-period "
            "widowhood mortality)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "deltas_vs_candidate1": list(DELTAS_VS_CANDIDATE1),
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic (restricted) "
                "spline on age, knots 20/25/30/40 (3 basis columns), sex, "
                "birth-decade cohort; main effects + age-spline x SEX "
                "interaction (DELTA 1, new) + age-spline x cohort "
                "interaction; sklearn LogisticRegression(penalty='l2', "
                "C=1.0, lbfgs) fit with sample_weight = person-year PSID "
                "weight (effectively the weighted MLE at train scale); "
                "design standardised for conditioning"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x "
                "marriage order (1st vs 2+), add-one (Laplace) smoothed at "
                "the train mean married person-year weight (byte-identical "
                "to candidate 1)"
            ),
            "widowhood": (
                "COMPOSED: train DECADE-PERIOD x age-band x sex mortality "
                "hazard (DELTA 2; build_mortality_floors construction, train "
                "persons, add-one smoothed at the train mean slice weight) "
                "applied to the spouse (opposite sex, age = self age + train "
                "mean spousal age gap by sex) at the simulated year's "
                "calendar decade; widowhood = induced transition"
            ),
            "remarriage": (
                "weighted empirical hazard by years-since-dissolution band x "
                "origin (divorced/widowed) x sex, add-one smoothed at the "
                "train mean dissolved person-year weight (byte-identical to "
                "candidate 1)"
            ),
            "fertility": (
                "weighted empirical age-band x parity (0/1/2/3+) rates by "
                "birth-decade cohort, train-estimated (no smoothing; "
                "byte-identical to candidate 1, and RNG-isolated from the "
                "marital process so its per-seed outcomes reproduce "
                "candidate 1 bit-for-bit)"
            ),
        },
        "registered_ambiguity_resolutions": {
            "age_sex_interaction": (
                "full first-marriage design = age spline x sex + age spline "
                "x cohort + sex (the registered 'full' model); the delta "
                "over candidate 1 is exactly the age-spline x sex block"
            ),
            "period_definition": (
                "decade-period = the slice's wave-anchored start_wave decade "
                "(the mortality-foundation construction's own period "
                "concept, the only period tag it exposes per slice); the "
                "simulated spouse death in year y uses the calendar decade "
                "of y, nearest fitted period if outside the exposure span"
            ),
            "mortality_add_one": (
                "Laplace: one pseudo-death + one pseudo-survivor at the "
                "train mean slice weight -- (Wd + w) / (We + 2w) -- mirroring "
                "candidate 1's divorce/remarriage smoothing; stabilises the "
                "thinner period x band x sex cells"
            ),
            "everything_else": (
                "divorce, remarriage, fertility, the spousal-age-gap "
                "imputation, the simulation loop, the RNG rule "
                "default_rng(4200 + seed), one sequence per person, and the "
                "locked protocol are byte-identical to candidate 1"
            ),
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = c1._load_cache(cache_path)

    thresholds = c1.load_gate2_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    tol = c1.gated_tolerances(thresholds)
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
    order_map = c1._order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons"
        )

    # Hard-stop precheck BEFORE any candidate is simulated (candidate 1's).
    precheck = c1.run_precheck(panel, fert, floor)
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
        cache[key] = json.loads(json.dumps(result, default=c1._json_default))
        c1._save_cache(cache_path, cache)
        per_seed.append(cache[key])

    verdict = c1.build_verdict(per_seed, tol)
    report_block = c1.report_only_summary(per_seed, report_only)
    seed_conjunction = [
        {
            "seed": s["seed"],
            "n_gated_pass": s["n_gated_pass"],
            "n_gated_fail": s["n_gated_fail"],
            "seed_pass": s["seed_pass"],
        }
        for s in per_seed
    ]

    modal = _modal_failure_check(verdict, per_seed)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 2",
        "spec_registration": SPEC_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "deltas_vs_candidate1": list(DELTAS_VS_CANDIDATE1),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 merge 82006877 + flip "
            "#81); protocol/views/tolerances read at runtime, no threshold "
            "moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.55-0.65",
            "conjunction_estimate": 0.60,
            "component_probabilities": {
                "stocks_all_pass": 0.7,
                "male_occupancy_holds": 0.75,
                "thin_band_fertility_remarriage_clips": 0.9,
            },
            "modal_failure": (
                "male ever-married occupancy at its ln-scale-tight tolerance"
            ),
            "deltas_vs_candidate1": list(DELTAS_VS_CANDIDATE1),
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
            "candidate1_registration": CANDIDATE1_REGISTRATION,
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
            "registered modal failure (male ever-married occupancy) "
            f"materialized: {modal['any_materialized']}"
        )
    return artifact


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
