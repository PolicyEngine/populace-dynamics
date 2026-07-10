"""Gate-2b candidate 1: structural household composition from the certified core.

The gate-2b holdout basis is household composition -- who lives with whom
(:mod:`populace_dynamics.data.household_composition`, MX23REL). This module
composes each holdout person's household states from six structural
components (registration #42 comment 4938726107):

1. **coresident spouse** -- the certified tranche-2a family-transition
   registry (:data:`populace_dynamics.models.family_transitions.CANDIDATE_16`),
   per-seed refit on the train complement; the simulated marital state
   ``married`` maps to a coresident spouse/partner.
2. **coresident parent** -- a logistic parental-home *exit* hazard
   (restricted-cubic age spline x sex) fit on the train MX23REL person-waves,
   evolved forward from each holdout person's OBSERVED initial state at window
   entry (the tranche-2a initial-state lesson).
3. **multigenerational membership** -- train-fitted entry / exit hazards by
   age band x sex, carried forward from the observed initial state.
4. **coresident children** -- the certified tranche-2a fertility kernel
   (maternal births from the registry simulate; a paternal shadow process for
   married men at the spouse-age-gap-shifted kernel), aging out under the
   fitted parental-home-exit hazard.
5. **household size** -- COMPOSED, never separately fit: ``1 + coresident
   spouse + coresident children + coresident parents``.
6. **coresident grandchild** -- NOT separately modelled; a composed
   implication of the simulated states (top generation of a
   three-generation household).

The estimator is the ratified tranche-2a mean-over-K=20-draws statistic
(``gates.yaml`` gate_2b ``protocol``): each draw re-simulates the whole
composition at ``numpy.random.default_rng(5200 + k)`` and the score is
``|ln(rbar / rate_a)|`` on the 20-draw mean rate. Fitting is train-only
(side B); the holdout persons (side A) are simulated and scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models.family_transitions.components.fertility import (
    FERTILITY_AGE_HI,
    FERTILITY_AGE_LO,
    build_fertility_lookup,
    fertility_probabilities,
)

__all__ = [
    "PARENTAL_EXIT_KNOTS",
    "CHILD_MIN_LEAVE_AGE",
    "ParentalExitModel",
    "HouseholdCompositionModel",
    "fit_household_model",
    "simulate_draw",
    "restricted_cubic_basis",
    "compose_states",
]

#: Restricted-cubic-spline knots (ages) for the parental-home exit hazard.
PARENTAL_EXIT_KNOTS: tuple[float, ...] = (16.0, 19.0, 23.0, 30.0, 45.0)
#: A child never leaves the parental home before this age (the panel's
#: ``START_AGE``; the exit hazard is only fit on 15+ person-waves).
CHILD_MIN_LEAVE_AGE = hc.START_AGE
#: A child that has not left by this age is treated as still coresident (an
#: open top; rare elderly-parent coresidence).
CHILD_MAX_LEAVE_AGE = 60
#: MX8 ego-to-alter codes where ego is a child of the alter (a coresident
#: parent), reused from the moment module so the parent-count is on the same
#: inclusion rule as the coresident_parent flag.
_CHILD_LINK_CODES = hc.CORESIDENCE_LINKS["coresident_parent"]


# --------------------------------------------------------------------------
# Restricted cubic spline (Harrell), matching run_gate2_candidate1.ncs_basis
# --------------------------------------------------------------------------
def compose_states(
    spouse: np.ndarray,
    parent: np.ndarray,
    multigen: np.ndarray,
    child_counts: np.ndarray,
    parent_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Components 5 and 6: compose household size and grandchild coresidence.

    * ``coresident_child`` = any coresident child.
    * ``hh_size`` = ``1 + coresident spouse + coresident children + coresident
      parents`` (``parent_count`` coresident parents when a parent is present),
      the ego-centric family-unit size (component 5, never separately tuned).
    * ``coresident_grandchild`` = ``multigen AND coresident_child AND NOT
      coresident_parent`` -- the top generation of a three-generation
      household, a composed implication only (component 6).
    """
    spouse = np.asarray(spouse, dtype=bool)
    parent = np.asarray(parent, dtype=bool)
    multigen = np.asarray(multigen, dtype=bool)
    child_counts = np.asarray(child_counts, dtype=np.int64)
    coresident_child = child_counts > 0
    n_parents = np.where(parent, int(parent_count), 0)
    hh_size = (1 + spouse.astype(np.int64) + child_counts + n_parents).astype(
        np.int64
    )
    coresident_grandchild = multigen & coresident_child & (~parent)
    return coresident_child, coresident_grandchild, hh_size


def restricted_cubic_basis(
    x: np.ndarray, knots: tuple[float, ...]
) -> np.ndarray:
    """Restricted cubic spline basis: K knots -> K-1 columns (linear + K-2)."""
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
class ParentalExitModel:
    """Fitted discrete-time logistic parental-home exit hazard.

    ``predict(age, is_male)`` returns the per-wave-interval probability that a
    coresident-parent person-wave loses its coresident parent by the next
    observed wave. The design is the restricted-cubic age spline, a male main
    effect, and the age-spline x male interaction.
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


@dataclass
class HouseholdCompositionModel:
    """Every train-fitted / certified component the candidate composes from."""

    family_transitions: ft.FittedFamilyTransitions
    parental_exit: ParentalExitModel
    multigen_entry: dict[tuple[str, str], float]
    multigen_exit: dict[tuple[str, str], float]
    parent_count: int
    male_gap: float
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Fitters (train / side B only)
# --------------------------------------------------------------------------
def fit_parental_exit(train_pw: pd.DataFrame) -> ParentalExitModel:
    """Fit the logistic parental-home exit hazard on train at-risk waves.

    At-risk = ``coresident_parent`` True with an observed next wave
    (``has_next``); event = the next wave has no coresident parent. Weighted
    by the wave weight; the fit spans all ages (the child aging-out reuses it).
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


def _weighted_rate(frame: pd.DataFrame, event: np.ndarray) -> float:
    w = frame["weight"].to_numpy(dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    return float((w * event).sum() / total)


def fit_multigen_rates(
    train_pw: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Train multigen entry / exit hazards by age band x sex.

    Entry rate (among not-multigen at-risk waves) and exit rate (among
    multigen at-risk waves) of ``next_multigen``, weighted, per (band, sex).
    Empty strata fall back to the pooled sex rate, then the overall rate.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry_pool = has_next[~has_next["multigen"]]
    exit_pool = has_next[has_next["multigen"]]
    entry_overall = _weighted_rate(
        entry_pool, entry_pool["next_multigen"].to_numpy(dtype=np.float64)
    )
    exit_overall = _weighted_rate(
        exit_pool,
        exit_pool["next_multigen"].eq(False).to_numpy(dtype=np.float64),
    )
    entry: dict[tuple[str, str], float] = {}
    exit_: dict[tuple[str, str], float] = {}
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            e = entry_pool[
                (entry_pool["band"] == band) & (entry_pool["sex"] == sex)
            ]
            x = exit_pool[
                (exit_pool["band"] == band) & (exit_pool["sex"] == sex)
            ]
            entry[(band, sex)] = (
                _weighted_rate(
                    e, e["next_multigen"].to_numpy(dtype=np.float64)
                )
                if len(e)
                else entry_overall
            )
            exit_[(band, sex)] = (
                _weighted_rate(
                    x,
                    x["next_multigen"].eq(False).to_numpy(dtype=np.float64),
                )
                if len(x)
                else exit_overall
            )
    return entry, exit_


def fit_parent_count(train_pw: pd.DataFrame, rel_map: pd.DataFrame) -> int:
    """Modal integer number of coresident parents, for the hh-size composition.

    Counts, per train coresident-parent person-wave, the MX8 parent links
    (ego coded child of an alter) and rounds the weighted mean to an integer
    (typically 2, a two-parent home). Fit on train; never tuned to holdout.
    """
    nonself = rel_map[rel_map["ego_rel_to_alter"] != transitions_self()]
    links = nonself[nonself["ego_rel_to_alter"].isin(_CHILD_LINK_CODES)]
    counts = (
        links.groupby(["interview_year", "interview_number", "ego_person_id"])
        .size()
        .rename("parent_count")
        .reset_index()
        .rename(
            columns={
                "ego_person_id": "person_id",
                "interview_year": "year",
            }
        )
    )
    at_risk = train_pw[train_pw["coresident_parent"]][
        ["person_id", "year", "weight"]
    ].merge(counts, on=["person_id", "year"], how="left")
    at_risk = at_risk[at_risk["parent_count"].notna()]
    if not len(at_risk):
        return 2
    w = at_risk["weight"].to_numpy(dtype=np.float64)
    c = at_risk["parent_count"].to_numpy(dtype=np.float64)
    mean = float((w * c).sum() / w.sum()) if w.sum() > 0 else 2.0
    return max(1, int(round(mean)))


def transitions_self() -> int:
    """The MX8 self code (imported indirectly to avoid a hard dependency)."""
    from populace_dynamics.data import relmap

    return relmap.SELF


def fit_male_gap(fitted: ft.FittedFamilyTransitions) -> float:
    """Mean husband->wife age gap (spouse_age - self_age) for marriage-age men.

    Read from the certified spousal-age-gap component's youngest band (ages
    18-34, marriage formation); negative because husbands are typically older.
    Used to place the paternal shadow fertility at the wife's age.
    """
    gaps = fitted.spousal_age_gaps.get("male", {})
    band0 = gaps.get(0)
    if band0 is None or len(band0) == 0:
        allvals = [v for v in gaps.values() if v is not None and len(v)]
        if not allvals:
            return -2.0
        return float(np.concatenate(allvals).mean())
    return float(np.asarray(band0, dtype=np.float64).mean())


def fit_household_model(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    marriage_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    marriage_order_map: pd.DataFrame,
    rel_map: pd.DataFrame,
    train_ids: set[int],
) -> HouseholdCompositionModel:
    """Fit every component on the train complement (side B) only."""
    ctx = ft.FitContext(
        panel=mpanel,
        demographic_panel=demo,
        marriage_records=marriage_records,
        birth_records=birth_records,
        marriage_order_map=marriage_order_map,
        train_ids=frozenset(train_ids),
    )
    fitted_ft = ft.REGISTRY.fit(ft.CANDIDATE_16, ctx)

    train_pw = hh.person_waves[hh.person_waves["person_id"].isin(train_ids)]
    parental_exit = fit_parental_exit(train_pw)
    entry, exit_ = fit_multigen_rates(train_pw)
    parent_count = fit_parent_count(train_pw, rel_map)
    male_gap = fit_male_gap(fitted_ft)

    meta = {
        "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
        "n_train_persons": int(len(train_ids)),
        "parental_exit_n_train_rows": parental_exit.n_train_rows,
        "parental_exit_n_train_events": parental_exit.n_train_events,
        "parental_exit_converged": parental_exit.converged,
        "parent_count": int(parent_count),
        "male_spousal_gap": round(male_gap, 3),
        "multigen_entry_overall": round(
            float(np.mean(list(entry.values()))), 5
        ),
        "multigen_exit_overall": round(
            float(np.mean(list(exit_.values()))), 5
        ),
    }
    return HouseholdCompositionModel(
        family_transitions=fitted_ft,
        parental_exit=parental_exit,
        multigen_entry=entry,
        multigen_exit=exit_,
        parent_count=parent_count,
        male_gap=male_gap,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def _padded_person_matrices(
    side_a_pw: pd.DataFrame,
) -> dict[str, Any]:
    """Reshape the ordered side-A person-waves into [person, wave] matrices."""
    pw = side_a_pw.sort_values(["person_id", "year"]).reset_index(drop=True)
    pids = pw["person_id"].to_numpy()
    # wave index within person.
    order = np.arange(len(pw))
    first = np.concatenate([[True], pids[1:] != pids[:-1]])
    person_start = order[first]
    wave_idx = order - np.repeat(
        person_start, np.diff(np.concatenate([person_start, [len(pw)]]))
    )
    unique_pids = pids[first]
    n_persons = len(unique_pids)
    max_waves = int(wave_idx.max()) + 1
    row_of = np.full((n_persons, max_waves), -1, dtype=np.int64)
    person_of_row = np.repeat(
        np.arange(n_persons),
        np.diff(np.concatenate([person_start, [len(pw)]])),
    )
    row_of[person_of_row, wave_idx] = order
    return {
        "pw": pw,
        "unique_pids": unique_pids,
        "n_persons": n_persons,
        "max_waves": max_waves,
        "row_of": row_of,
        "person_of_row": person_of_row,
        "wave_idx": wave_idx,
    }


def _evolve_absorbing_exit(
    valid: np.ndarray,
    initial: np.ndarray,
    exit_prob: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve an absorbing True->False occupancy over [person, wave] columns.

    ``initial[:, 0]`` seeds the state; at wave w>=1 a still-True person exits
    with ``exit_prob[:, w-1]``. Invalid (padded) cells stay False.
    """
    n_persons, max_waves = valid.shape
    state = np.zeros((n_persons, max_waves), dtype=bool)
    cur = initial[:, 0] & valid[:, 0]
    state[:, 0] = cur
    for w in range(1, max_waves):
        u = rng.random(n_persons)
        exited = cur & (u < exit_prob[:, w - 1])
        cur = cur & ~exited & valid[:, w]
        state[:, w] = cur
    return state


def _evolve_two_state(
    valid: np.ndarray,
    initial: np.ndarray,
    entry_prob: np.ndarray,
    exit_prob: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve a two-state (enter/exit) occupancy over [person, wave] columns."""
    n_persons, max_waves = valid.shape
    state = np.zeros((n_persons, max_waves), dtype=bool)
    cur = initial[:, 0] & valid[:, 0]
    state[:, 0] = cur
    for w in range(1, max_waves):
        u = rng.random(n_persons)
        enters = (~cur) & (u < entry_prob[:, w - 1])
        exits = cur & (u < exit_prob[:, w - 1])
        cur = np.where(cur, cur & ~exits, enters) & valid[:, w]
        state[:, w] = cur
    return state


def _paternal_births(
    mpanel_sim_years: pd.DataFrame,
    attrs: pd.DataFrame,
    male_gap: float,
    fert_lookup: np.ndarray,
    fert_decade_map: dict[int, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shadow paternal births: married men parent children at the wife's age.

    For each married man-year the certified fertility kernel is evaluated at
    the wife's imputed age (man age + male_gap), the household parity, and the
    man's birth decade; a birth increments the household parity. Returns
    (parent_person_id, birth_year).
    """
    men = attrs[attrs["sex"] == "male"][["person_id", "birth_year"]]
    if not len(men):
        return pd.DataFrame({"parent_person_id": [], "birth_year": []}).astype(
            {"parent_person_id": "int64", "birth_year": "int64"}
        )
    married = mpanel_sim_years[mpanel_sim_years["marital_state"] == "married"][
        ["person_id", "year"]
    ]
    married = married[married["person_id"].isin(set(men["person_id"]))]
    if not len(married):
        return pd.DataFrame({"parent_person_id": [], "birth_year": []}).astype(
            {"parent_person_id": "int64", "birth_year": "int64"}
        )
    birth_year_by = men.set_index("person_id")["birth_year"]
    married = married.assign(
        birth_year_person=married["person_id"].map(birth_year_by)
    )
    married = married[married["birth_year_person"].notna()]
    married["wife_age"] = (
        married["year"] - married["birth_year_person"] + male_gap
    )
    married = married[
        (married["wife_age"] >= FERTILITY_AGE_LO)
        & (married["wife_age"] <= FERTILITY_AGE_HI)
    ].sort_values(["person_id", "year"])
    if not len(married):
        return pd.DataFrame({"parent_person_id": [], "birth_year": []}).astype(
            {"parent_person_id": "int64", "birth_year": "int64"}
        )

    married["decade_idx"] = (
        (married["birth_year_person"] // 10 * 10)
        .astype("int64")
        .map(lambda d: fert_decade_map.get(int(d), -1))
        .to_numpy()
    )
    married["wife_age_int"] = np.rint(married["wife_age"].to_numpy()).astype(
        np.int64
    )
    # Man -> dense index for a running household-parity array.
    man_ids = np.sort(married["person_id"].unique())
    man_index = {int(p): i for i, p in enumerate(man_ids)}
    married["man_idx"] = married["person_id"].map(man_index).to_numpy()
    parity = np.zeros(len(man_ids), dtype=np.int64)
    out_person: list[int] = []
    out_year: list[int] = []
    # Annual, vectorized over the men married that year (parity is sequential
    # across years, so the year loop is required; each year is vectorized).
    for year, g in married.groupby("year", sort=True):
        midx = g["man_idx"].to_numpy()
        prob = fertility_probabilities(
            g["wife_age_int"].to_numpy(),
            parity[midx],
            g["decade_idx"].to_numpy(),
            fert_lookup,
        )
        u = rng.random(len(g))
        born = u < prob
        born_idx = midx[born]
        out_person.extend(int(man_ids[i]) for i in born_idx)
        out_year.extend([int(year)] * int(born.sum()))
        np.add.at(parity, born_idx, 1)
    return pd.DataFrame(
        {
            "parent_person_id": np.array(out_person, dtype=np.int64),
            "birth_year": np.array(out_year, dtype=np.int64),
        }
    )


def _child_leave_years(
    births: pd.DataFrame,
    parental_exit: ParentalExitModel,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw each child's home-leaving year via the parental-exit hazard.

    A child stays until ``CHILD_MIN_LEAVE_AGE``, then at each biennial step
    exits with the fitted hazard at the child's age and a uniformly drawn sex;
    the first exit sets the leave age (open-topped at CHILD_MAX_LEAVE_AGE).
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


def _coresident_child_counts(
    child_leaves: pd.DataFrame,
    side_a_pw: pd.DataFrame,
) -> np.ndarray:
    """Per side-A person-wave, the count of the person's coresident children.

    A child (parent P, birth B, leave L) coresides at P's wave year Y iff
    ``B <= Y < L``. Returns an int array aligned to ``side_a_pw`` row order.
    """
    counts = np.zeros(len(side_a_pw), dtype=np.int64)
    if not len(child_leaves):
        return counts
    pw = side_a_pw.reset_index(drop=True)
    pw = pw.assign(_row=np.arange(len(pw), dtype=np.int64))
    children = child_leaves.rename(columns={"parent_person_id": "person_id"})[
        ["person_id", "birth_year", "leave_year"]
    ]
    merged = pw[["person_id", "year", "_row"]].merge(
        children, on="person_id", how="inner"
    )
    hit = (merged["year"] >= merged["birth_year"]) & (
        merged["year"] < merged["leave_year"]
    )
    agg = merged.loc[hit].groupby("_row").size()
    counts[agg.index.to_numpy()] = agg.to_numpy()
    return counts


def _spouse_from_marital(
    side_a_pw: pd.DataFrame,
    sim_years: pd.DataFrame,
) -> np.ndarray:
    """Map the simulated marital state onto side-A waves -> coresident spouse.

    ``married`` -> a coresident spouse. Where the certified simulation does
    not cover a (person, year) -- attrition gaps, or persons with no marriage
    record at all -- the person's OBSERVED initial coresident-spouse state
    (first wave) is carried; a within-person forward/back fill bridges gaps.
    """
    pw = side_a_pw.reset_index(drop=True)
    state = sim_years.set_index(["person_id", "year"])["marital_state"]
    idx = pd.MultiIndex.from_arrays(
        [pw["person_id"].to_numpy(), pw["year"].to_numpy()]
    )
    matched = state.reindex(idx)
    sim_spouse = (matched == "married").to_numpy()
    covered = matched.notna().to_numpy()
    # Fallback for uncovered (person, year) waves -- attrition gaps and
    # persons with no marriage-file record: carry the person's OBSERVED
    # initial (first-wave) coresident-spouse state (the tranche-2a
    # observed-initial lesson; captures cohabiting partners the legal
    # marriage file misses).
    obs_init = (
        pw.groupby("person_id")["coresident_spouse"]
        .transform("first")
        .to_numpy()
        .astype(bool)
    )
    return np.where(covered, sim_spouse, obs_init)


def simulate_draw(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModel,
    ids_a: set[int],
    draw_seed: int,
    occupancy_stream_tag: int = 0xB2B,
) -> hc.HouseholdCompositionPanel:
    """Simulate one draw of the side-A holdout households.

    ``draw_seed`` (= 5200 + k) seeds the certified tranche-2a simulation
    (marital core + maternal births); a distinct spawned stream seeds the
    paternal shadow births and the parental-home / multigen / child-aging
    occupancy overlays, so the two subsystems never share draws.
    """
    ft_rng_seed = draw_seed
    occ_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, occupancy_stream_tag])
    )

    # 1. certified marital core + maternal births.
    sim_panel, sim_births = ft.simulate(
        mpanel, ids_a, model.family_transitions, ft_rng_seed
    )
    sim_years = sim_panel.person_years

    side_a_pw = (
        hh.person_waves[hh.person_waves["person_id"].isin(ids_a)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )

    # 2. coresident spouse.
    spouse = _spouse_from_marital(side_a_pw, sim_years)

    # 3-4. child rosters -> leave years -> per-wave counts.
    maternal = sim_births.rename(
        columns={"parent_person_id": "parent_person_id"}
    )[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    lookup, decade_map = build_fertility_lookup(
        model.family_transitions.fertility
    )
    paternal = _paternal_births(
        sim_years,
        mpanel.attrs[mpanel.attrs["person_id"].isin(ids_a)],
        model.male_gap,
        lookup,
        decade_map,
        occ_rng,
    )
    all_births = pd.concat([maternal, paternal], ignore_index=True)
    all_births = all_births[all_births["parent_person_id"].isin(ids_a)]
    child_leaves = _child_leave_years(all_births, model.parental_exit, occ_rng)
    child_counts = _coresident_child_counts(child_leaves, side_a_pw)

    # 5-6. parental-home + multigen occupancy overlays.
    mats = _padded_person_matrices(side_a_pw)
    pw = mats["pw"]
    row_of = mats["row_of"]
    n_persons, max_waves = mats["n_persons"], mats["max_waves"]
    valid = row_of >= 0
    safe_row = np.where(valid, row_of, 0)

    age_mat = pw["age"].to_numpy()[safe_row]
    sex_mat = pw["sex"].to_numpy()[safe_row]
    is_male_mat = (sex_mat == "male").astype(np.float64)
    band_mat = pw["band"].to_numpy(dtype=object)[safe_row]
    obs_parent = pw["coresident_parent"].to_numpy(dtype=bool)[safe_row] & valid
    obs_multigen = pw["multigen"].to_numpy(dtype=bool)[safe_row] & valid

    # parental-home exit probability per [person, wave].
    exit_prob = model.parental_exit.predict(
        age_mat.reshape(-1), is_male_mat.reshape(-1)
    ).reshape(n_persons, max_waves)
    parent_state = _evolve_absorbing_exit(
        valid, obs_parent, exit_prob, occ_rng
    )

    # multigen entry/exit probability per [person, wave] from band x sex.
    entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    exitm_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.multigen_entry.items():
        mask = (band_mat == band) & (sex_mat == sex)
        entry_prob[mask] = rate
    for (band, sex), rate in model.multigen_exit.items():
        mask = (band_mat == band) & (sex_mat == sex)
        exitm_prob[mask] = rate
    multigen_state = _evolve_two_state(
        valid, obs_multigen, entry_prob, exitm_prob, occ_rng
    )

    # scatter [person, wave] states back to row order.
    parent_row = np.zeros(len(pw), dtype=bool)
    multigen_row = np.zeros(len(pw), dtype=bool)
    parent_row[row_of[valid]] = parent_state[valid]
    multigen_row[row_of[valid]] = multigen_state[valid]

    # 5-6. compose household size, coresident child and grandchild.
    coresident_child, coresident_grandchild, hh_size = compose_states(
        spouse, parent_row, multigen_row, child_counts, model.parent_count
    )

    sim_pw = pw.copy()
    sim_pw["coresident_spouse"] = spouse
    sim_pw["coresident_parent"] = parent_row
    sim_pw["coresident_child"] = coresident_child
    sim_pw["coresident_grandchild"] = coresident_grandchild
    sim_pw["multigen"] = multigen_row
    sim_pw["hh_size"] = hh_size.astype(np.int64)
    sim_pw = sim_pw.drop(
        columns=[
            "has_next",
            "next_coresident_parent",
            "next_coresident_spouse",
            "next_multigen",
        ]
    )
    sim_pw = hc._add_transitions(sim_pw)
    attrs = sim_pw[["person_id"]].drop_duplicates().reset_index(drop=True)
    return hc.HouseholdCompositionPanel(person_waves=sim_pw, attrs=attrs)
