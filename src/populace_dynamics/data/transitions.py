"""Person-period marital-transition and fertility panels from PSID histories.

Builds the gate-2 family-transition evidence base: an *annual* person-year
panel with marital state and the dated transitions (first marriage,
remarriage, divorce, widowhood) from the retrospective Marriage History
File (:mod:`populace_dynamics.data.marriage`), plus the fertility exposure
and birth events from the Childbirth & Adoption History File
(:mod:`populace_dynamics.data.births`). These are the transition-realism
moments a survivor/spousal/caregiver reform is scored on.

Construction (retrospective annual person-years)
------------------------------------------------
The marriage file is a single retrospective product: each sample member
carries their whole dated marriage history (start/end years, how each
marriage ended) as of their last interview. So marital state is
reconstructed **annually** across each person's life, not on the biennial
observation grid:

* **Person attributes.** One row per person from
  :func:`marriage.marriage_history`: ``sex``, ``birth_year``, the
  person-constant ``most_recent_report_year`` (``MH17``, the year the
  marital status was last confirmed -- verified person-constant), and
  ``n_marriages``. Persons missing ``birth_year`` or sex, or with no
  positive weight, drop out (documented coverage, below).
* **Censoring.** Each person is at risk from ``birth_year + START_AGE``
  (marriageable age) through ``censor_year = min(most_recent_report_year,
  exact death year, MAX_YEAR)`` -- own death (from
  :mod:`populace_dynamics.data.deaths`) ends the at-risk window, and the
  last-report year bounds the retrospective knowledge.
* **State machine.** From the ordered marriage episodes
  (:func:`marriage.marriage_episodes`): ``never_married`` before the first
  marriage start; ``married`` during ``[start, end]``; then ``divorced`` /
  ``widowed`` / ``separated`` / ``other`` after a dissolution until the
  next marriage. State at person-year ``t`` is the state entering year
  ``t`` (a transition *in* year ``t`` is the event, and year ``t`` carries
  the pre-transition at-risk state -- the standard discrete-time hazard
  convention, ``merge_asof(allow_exact_matches=False)``).
* **Transitions.** ``first_marriage`` (order-1 start), ``remarriage``
  (a later start out of a post-dissolution state, with years-since-
  dissolution), ``divorce`` and ``widowhood`` (dissolutions dated at the
  marriage end year), each carrying the age / duration / years-since-
  dissolution the gate-2 hazards band on.

Weights
-------
The marriage file carries no weight. A person-constant weight is attached
from :func:`populace_dynamics.data.panels.demographic_panel` -- the
person's most recent positive cross-sectional weight. Person-years are
weighted by it; persons with no positive PSID weight are excluded from the
weighted moments (the repo's ``weight > 0`` convention). Every builder
also exposes an unweighted count, and :func:`reference_moments` takes
``weighted=`` so the floor can document the weighted/unweighted gap.

Coverage caveats (reported, never tuned)
----------------------------------------
Retrospective histories are **left-truncated**: a person seen first at an
old age contributes reconstructed young-adult person-years only if they
survived and were sampled, so pooled young-age rates over deep history are
selected. Pooling all calendar years against a single recent national
anchor is a named period-concept delta. Both are reported honestly (the
external anchor is a shape/ratio report, not a level gate), exactly as the
mortality foundation reports the PSID mortality undercount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from populace_dynamics.data import births, marriage

__all__ = [
    "START_AGE",
    "MAX_YEAR",
    "OBSERVED_TO_AGES",
    "SEXES",
    "FIRST_MARRIAGE_AGE_BANDS",
    "WIDOWHOOD_AGE_BANDS",
    "DIVORCE_DURATION_BANDS",
    "REMARRIAGE_YSD_BANDS",
    "ASFR_AGE_BANDS",
    "COMPLETED_FERTILITY_AGE",
    "COHORT_DECADES",
    "MaritalPanel",
    "person_attributes",
    "build_marital_panel",
    "hazard_cells",
    "occupancy_cells",
    "FertilityPanel",
    "build_fertility_panel",
    "fertility_cells",
    "reference_moments",
    "band_label",
]

#: Marriageable age at which retrospective at-risk exposure starts.
START_AGE = 15
#: Last collection wave; no person-year is credited past it.
MAX_YEAR = 2023
SEXES: tuple[str, ...] = ("female", "male")

#: First-marriage hazard age bands (closed single-year bounds; top open).
FIRST_MARRIAGE_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (18, 24),
    (25, 34),
    (35, 44),
    (45, 120),
)
#: Widowhood-incidence age bands.
WIDOWHOOD_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 120),
)
#: Divorce-hazard marriage-duration bands (years since marriage start).
DIVORCE_DURATION_BANDS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 9),
    (10, 19),
    (20, 120),
)
#: Remarriage-hazard years-since-dissolution bands.
REMARRIAGE_YSD_BANDS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 9),
    (10, 120),
)
#: Age-specific fertility-rate bands (mother age; NCHS 5-year groups).
ASFR_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (15, 19),
    (20, 24),
    (25, 29),
    (30, 34),
    (35, 39),
    (40, 44),
    (45, 49),
)
#: Ages a person must be observed through for the occupancy shares.
OBSERVED_TO_AGES: tuple[int, ...] = (40, 60)
#: Age by which fertility is treated as complete (cohort mean parity).
COMPLETED_FERTILITY_AGE = 45
#: Woman birth decades reported for completed fertility (women who could
#: reach COMPLETED_FERTILITY_AGE by MAX_YEAR: born <= 1978).
COHORT_DECADES: tuple[int, ...] = (1930, 1940, 1950, 1960, 1970)

#: Post-dissolution states from which a person is at risk of remarriage.
_REMARRIAGE_RISK_STATES = ("divorced", "widowed")
#: How a dissolution's ``how_ended`` maps to the resulting marital state.
_DISSOLUTION_STATE = {
    "divorce": "divorced",
    "widowhood": "widowed",
    "separated": "separated",
    "other": "other",
    "unknown": "other",
}


def band_label(lo: int, hi: int, *, prefix: str = "") -> str:
    """Flat band label, e.g. ``"25-34"`` or open-topped ``"45+"``."""
    core = f"{lo}-{hi}" if hi < 120 else f"{lo}+"
    return f"{prefix}{core}"


def _band_of(value: object, bands: tuple[tuple[int, int], ...]) -> str | None:
    """Label of the band containing ``value`` (inclusive), or ``None``.

    Handles ``None``, ``float`` NaN, and pandas ``NA`` (nullable-Int64
    duration / years-since-dissolution columns carry the last two).
    """
    if pd.isna(value):
        return None
    v = float(value)
    for lo, hi in bands:
        if lo <= v <= hi:
            return band_label(lo, hi)
    return None


# --------------------------------------------------------------------------
# Person attributes and censoring
# --------------------------------------------------------------------------
def person_attributes(
    records: pd.DataFrame,
    death_records: pd.DataFrame,
    person_weight: pd.Series,
) -> pd.DataFrame:
    """One row per person: sex, birth_year, censor_year, weight, n_marriages.

    Pure transform over a :func:`marriage.marriage_history`-shaped frame, a
    :func:`populace_dynamics.data.deaths.read_death_records`-shaped frame,
    and a ``person_id -> weight`` series (so it is unit-testable on
    synthetic rows). ``most_recent_report_year`` is person-constant on the
    real file; the first value per person is taken. ``censor_year`` is the
    earliest of the last-report year, the exact death year, and
    :data:`MAX_YEAR`.
    """
    grouped = records.groupby("person_id", as_index=False).agg(
        sex=("sex", "first"),
        birth_year=("birth_year", "first"),
        most_recent_report_year=("most_recent_report_year", "min"),
        n_marriages=("n_marriages", "max"),
    )
    death_year = death_records.set_index("person_id")["death_year"]
    grouped["death_year"] = (
        grouped["person_id"].map(death_year).astype("Int64")
    )

    mrr = grouped["most_recent_report_year"].astype("float64")
    dyear = grouped["death_year"].astype("float64")
    censor = np.minimum(mrr.fillna(MAX_YEAR), np.float64(MAX_YEAR))
    censor = np.minimum(censor, dyear.fillna(np.inf))
    grouped["censor_year"] = censor.astype("float64")

    grouped["weight"] = (
        grouped["person_id"].map(person_weight).astype("float64").fillna(0.0)
    )
    grouped["birth_year"] = grouped["birth_year"].astype("float64")
    grouped["start_exposure_year"] = grouped["birth_year"] + START_AGE
    grouped["n_marriages"] = grouped["n_marriages"].astype("float64")
    return grouped


def _valid_persons(
    attrs: pd.DataFrame, *, require_weight: bool
) -> pd.DataFrame:
    """Persons with a usable birth year, sex, and positive exposure window."""
    ok = (
        attrs["birth_year"].notna()
        & attrs["sex"].isin(SEXES)
        & attrs["censor_year"].notna()
        & (attrs["start_exposure_year"] <= attrs["censor_year"])
    )
    if require_weight:
        ok &= attrs["weight"] > 0
    return attrs.loc[ok].reset_index(drop=True)


# --------------------------------------------------------------------------
# The marital person-year panel and its transition events
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MaritalPanel:
    """The person-period marital panel and its dated transitions.

    Attributes:
        person_years: One row per person-year with ``person_id``, ``year``,
            ``age``, ``sex``, ``weight``, ``marital_state``,
            ``marriage_duration`` (years since the current marriage start,
            NA off-marriage), and ``years_since_dissolution`` (NA outside a
            post-dissolution spell).
        events: One row per dated transition with ``person_id``, ``year``,
            ``age``, ``sex``, ``weight``, ``transition`` (``first_marriage``
            / ``remarriage`` / ``divorce`` / ``widowhood``),
            ``marriage_duration`` (divorce), and
            ``years_since_dissolution`` (remarriage).
        attrs: The valid-person attribute frame the panel was built from
            (one row per person), for occupancy statistics.
    """

    person_years: pd.DataFrame
    events: pd.DataFrame
    attrs: pd.DataFrame


def _changepoints(episodes: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """State change-points per person from ordered marriage episodes.

    A ``marry`` change-point at each ``start_year`` (state ``married``,
    carrying ``current_start``) and a ``dissolve`` change-point at each
    known ``episode_end_year`` (the dissolution state, carrying
    ``dissolution_year``). Sorted per person by year with dissolve before
    marry on ties, so a same-year dissolution-then-remarriage resolves in
    order.
    """
    valid = set(attrs["person_id"])
    ep = episodes[episodes["person_id"].isin(valid)].copy()
    ep = ep[ep["start_year"].notna()]

    marry = pd.DataFrame(
        {
            "person_id": ep["person_id"].to_numpy(),
            "year": ep["start_year"].astype("int64").to_numpy(),
            "new_state": "married",
            "current_start": ep["start_year"].astype("float64").to_numpy(),
            "dissolution_year": np.nan,
            "kind_order": 1,
        }
    )
    diss_src = ep[
        ep["episode_end_year"].notna()
        & ep["how_ended"].isin(list(_DISSOLUTION_STATE))
    ]
    dissolve = pd.DataFrame(
        {
            "person_id": diss_src["person_id"].to_numpy(),
            "year": diss_src["episode_end_year"].astype("int64").to_numpy(),
            "new_state": diss_src["how_ended"]
            .map(_DISSOLUTION_STATE)
            .to_numpy(),
            "current_start": np.nan,
            "dissolution_year": diss_src["episode_end_year"]
            .astype("float64")
            .to_numpy(),
            "kind_order": 0,
        }
    )
    cps = pd.concat([marry, dissolve], ignore_index=True)
    return cps.sort_values(["person_id", "year", "kind_order"]).reset_index(
        drop=True
    )


def _person_years_frame(attrs: pd.DataFrame) -> pd.DataFrame:
    """Explode valid persons into annual person-years (vectorized)."""
    start = attrs["start_exposure_year"].astype("int64").to_numpy()
    end = attrs["censor_year"].astype("int64").to_numpy()
    lengths = (end - start + 1).astype("int64")
    idx = np.repeat(np.arange(len(attrs)), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    year = start[idx] + offsets
    birth_year = attrs["birth_year"].to_numpy()[idx]
    return pd.DataFrame(
        {
            "person_id": attrs["person_id"].to_numpy()[idx],
            "year": year,
            "age": (year - birth_year).astype("int64"),
            "sex": attrs["sex"].to_numpy()[idx],
            "weight": attrs["weight"].to_numpy()[idx],
        }
    )


def _assign_state(
    person_years: pd.DataFrame, changepoints: pd.DataFrame
) -> pd.DataFrame:
    """Attach marital state entering each person-year via backward asof.

    ``allow_exact_matches=False``: a change-point in year ``t`` does not
    apply to person-year ``t``, so year ``t`` carries the state entering it
    and a transition in year ``t`` is the event (discrete-time hazard).
    """
    # merge_asof requires both frames sorted by the ``on`` key (year)
    # globally; the ``by`` key (person) is grouped internally.
    py = person_years.sort_values("year").reset_index(drop=True)
    if changepoints.empty:
        py["marital_state"] = "never_married"
        py["marriage_duration"] = pd.array([pd.NA] * len(py), dtype="Int64")
        py["years_since_dissolution"] = pd.array(
            [pd.NA] * len(py), dtype="Int64"
        )
        return py
    cps = changepoints.sort_values("year").reset_index(drop=True)
    merged = pd.merge_asof(
        py,
        cps[
            [
                "person_id",
                "year",
                "new_state",
                "current_start",
                "dissolution_year",
            ]
        ],
        on="year",
        by="person_id",
        direction="backward",
        allow_exact_matches=False,
    )
    merged["marital_state"] = merged["new_state"].fillna("never_married")
    dur = merged["year"] - merged["current_start"]
    merged["marriage_duration"] = dur.where(
        merged["marital_state"] == "married"
    ).astype("Int64")
    ysd = merged["year"] - merged["dissolution_year"]
    in_post = merged["marital_state"].isin(_REMARRIAGE_RISK_STATES)
    merged["years_since_dissolution"] = ysd.where(in_post).astype("Int64")
    return merged.drop(
        columns=["new_state", "current_start", "dissolution_year"]
    )


def _events_frame(episodes: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Dated transition events from ordered episodes, clipped to exposure."""
    valid = attrs.set_index("person_id")
    ep = episodes[episodes["person_id"].isin(valid.index)].copy()
    ep = ep.sort_values(["person_id", "start_year"]).reset_index(drop=True)
    ep["birth_year"] = ep["person_id"].map(valid["birth_year"])
    ep["sex"] = ep["person_id"].map(valid["sex"])
    ep["weight"] = ep["person_id"].map(valid["weight"])
    ep["censor_year"] = ep["person_id"].map(valid["censor_year"])
    ep["start_exposure_year"] = ep["person_id"].map(
        valid["start_exposure_year"]
    )
    ep["prev_end"] = ep.groupby("person_id")["episode_end_year"].shift(1)
    ep["prev_how_ended"] = ep.groupby("person_id")["how_ended"].shift(1)
    ep["order_rank"] = ep.groupby("person_id").cumcount()

    def _clip(frame: pd.DataFrame, year_col: str) -> pd.DataFrame:
        yr = frame[year_col].astype("float64")
        keep = (
            frame[year_col].notna()
            & (yr <= frame["censor_year"])
            & (yr >= frame["start_exposure_year"])
        )
        return frame.loc[keep]

    rows: list[pd.DataFrame] = []

    starts = ep[ep["start_year"].notna()].copy()
    first = _clip(starts[starts["order_rank"] == 0], "start_year")
    rows.append(
        pd.DataFrame(
            {
                "person_id": first["person_id"],
                "year": first["start_year"].astype("int64"),
                "age": (first["start_year"] - first["birth_year"]).astype(
                    "int64"
                ),
                "sex": first["sex"],
                "weight": first["weight"],
                "transition": "first_marriage",
                "marriage_duration": pd.NA,
                "years_since_dissolution": pd.NA,
            }
        )
    )

    # Remarriage risk is out of a divorced or widowed spell only; a later
    # start whose prior marriage ended in separation (still legally
    # married) has no post-dissolution exposure in the denominator, so it
    # is excluded here to keep numerator and denominator consistent.
    remar = starts[
        (starts["order_rank"] >= 1)
        & starts["prev_end"].notna()
        & starts["prev_how_ended"].isin(list(_DISSOLUTION_STATE))
        & starts["prev_how_ended"]
        .map(_DISSOLUTION_STATE)
        .isin(_REMARRIAGE_RISK_STATES)
    ]
    remar = _clip(remar, "start_year")
    rows.append(
        pd.DataFrame(
            {
                "person_id": remar["person_id"],
                "year": remar["start_year"].astype("int64"),
                "age": (remar["start_year"] - remar["birth_year"]).astype(
                    "int64"
                ),
                "sex": remar["sex"],
                "weight": remar["weight"],
                "transition": "remarriage",
                "marriage_duration": pd.NA,
                "years_since_dissolution": (
                    remar["start_year"] - remar["prev_end"]
                ).astype("Int64"),
            }
        )
    )

    for how, label in (("divorce", "divorce"), ("widowhood", "widowhood")):
        src = ep[(ep["how_ended"] == how) & ep["episode_end_year"].notna()]
        src = _clip(src, "episode_end_year")
        rows.append(
            pd.DataFrame(
                {
                    "person_id": src["person_id"],
                    "year": src["episode_end_year"].astype("int64"),
                    "age": (
                        src["episode_end_year"] - src["birth_year"]
                    ).astype("int64"),
                    "sex": src["sex"],
                    "weight": src["weight"],
                    "transition": label,
                    "marriage_duration": (
                        src["episode_duration_years"].astype("Int64")
                        if label == "divorce"
                        else pd.NA
                    ),
                    "years_since_dissolution": pd.NA,
                }
            )
        )

    events = pd.concat(rows, ignore_index=True)
    return events[events["age"] >= START_AGE].reset_index(drop=True)


def build_marital_panel(
    records: pd.DataFrame,
    death_records: pd.DataFrame,
    person_weight: pd.Series,
    *,
    require_weight: bool = True,
) -> MaritalPanel:
    """Assemble the marital person-year panel and its transition events.

    Pure over a :func:`marriage.marriage_history`-shaped frame, a death
    records frame, and a ``person_id -> weight`` series -- unit-testable on
    synthetic fixtures. With ``require_weight`` (default) only persons with
    a positive PSID weight are kept (the repo's weighted convention); pass
    ``False`` for the unweighted universe.
    """
    attrs = person_attributes(records, death_records, person_weight)
    attrs = _valid_persons(attrs, require_weight=require_weight)
    episodes = marriage.marriage_episodes(records)
    changepoints = _changepoints(episodes, attrs)
    person_years = _assign_state(_person_years_frame(attrs), changepoints)
    events = _events_frame(episodes, attrs)
    return MaritalPanel(person_years=person_years, events=events, attrs=attrs)


# --------------------------------------------------------------------------
# Hazard cells (person-subset aware, for the half-split floor)
# --------------------------------------------------------------------------
def _rate_cell(
    num_weight: float, den_weight: float, n_events: int
) -> dict[str, float]:
    return {
        "rate": float(num_weight / den_weight) if den_weight > 0 else 0.0,
        "num_wt": float(num_weight),
        "den_wt": float(den_weight),
        "n_events": int(n_events),
    }


def _hazard_by_band(
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    band_col: str,
    bands: tuple[tuple[int, int], ...],
    *,
    prefix: str,
    by_sex: bool,
    weighted: bool,
) -> dict[str, dict[str, float]]:
    """Weighted hazard = w-events / w-exposure per band (x sex)."""
    ev = events.copy()
    ex = exposure.copy()
    ev["band"] = ev[band_col].map(lambda v: _band_of(v, bands))
    ex["band"] = ex[band_col].map(lambda v: _band_of(v, bands))
    ev_w = ev["weight"] if weighted else 1.0
    ex_w = ex["weight"] if weighted else 1.0
    ev = ev.assign(_w=ev_w)
    ex = ex.assign(_w=ex_w)
    keys = ["band", "sex"] if by_sex else ["band"]
    num = ev.groupby(keys, observed=True).agg(
        num_wt=("_w", "sum"), n_events=("_w", "size")
    )
    den = ex.groupby(keys, observed=True)["_w"].sum()

    out: dict[str, dict[str, float]] = {}
    sexes = SEXES if by_sex else (None,)
    for lo, hi in bands:
        band = band_label(lo, hi)
        for sex in sexes:
            gk = (band, sex) if by_sex else band
            label = f"{prefix}.{band}|{sex}" if by_sex else f"{prefix}.{band}"
            num_wt = float(num["num_wt"].get(gk, 0.0))
            n_events = int(num["n_events"].get(gk, 0))
            den_wt = float(den.get(gk, 0.0))
            out[label] = _rate_cell(num_wt, den_wt, n_events)
    return out


def hazard_cells(
    panel: MaritalPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """All marital-hazard cells for a person subset (``None`` = everyone).

    Restricting by ``person_ids`` and recomputing is exactly the
    person-disjoint half-split the floor needs; the full-panel call is the
    committed reference moment.
    """
    py = panel.person_years
    ev = panel.events
    if person_ids is not None:
        py = py[py["person_id"].isin(person_ids)]
        ev = ev[ev["person_id"].isin(person_ids)]

    cells: dict[str, dict[str, float]] = {}
    cells.update(
        _hazard_by_band(
            ev[ev["transition"] == "first_marriage"],
            py[py["marital_state"] == "never_married"],
            "age",
            FIRST_MARRIAGE_AGE_BANDS,
            prefix="first_marriage",
            by_sex=True,
            weighted=weighted,
        )
    )
    cells.update(
        _hazard_by_band(
            ev[ev["transition"] == "divorce"],
            py[py["marital_state"] == "married"],
            "marriage_duration",
            DIVORCE_DURATION_BANDS,
            prefix="divorce",
            by_sex=False,
            weighted=weighted,
        )
    )
    cells.update(
        _hazard_by_band(
            ev[ev["transition"] == "widowhood"],
            py[py["marital_state"] == "married"],
            "age",
            WIDOWHOOD_AGE_BANDS,
            prefix="widowhood",
            by_sex=True,
            weighted=weighted,
        )
    )
    cells.update(
        _hazard_by_band(
            ev[ev["transition"] == "remarriage"],
            py[py["marital_state"].isin(_REMARRIAGE_RISK_STATES)],
            "years_since_dissolution",
            REMARRIAGE_YSD_BANDS,
            prefix="remarriage",
            by_sex=False,
            weighted=weighted,
        )
    )
    # Rename remarriage band labels to the ysd namespace for clarity.
    for lo, hi in REMARRIAGE_YSD_BANDS:
        old = f"remarriage.{band_label(lo, hi)}"
        new = f"remarriage.ysd{band_label(lo, hi)}"
        cells[new] = cells.pop(old)
    for lo, hi in DIVORCE_DURATION_BANDS:
        old = f"divorce.{band_label(lo, hi)}"
        new = f"divorce.dur{band_label(lo, hi)}"
        cells[new] = cells.pop(old)
    return cells


def occupancy_cells(
    panel: MaritalPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Ever-married-by-age shares and mean lifetime marriages, by sex.

    Occupancy is a per-person statistic (not a person-year hazard): among
    persons observed through a target age, the weighted share ever married
    by then; and among ever-married persons observed through
    :data:`COMPLETED_FERTILITY_AGE`, the weighted mean number of marriages.
    """
    attrs = panel.attrs
    ev = panel.events[panel.events["transition"] == "first_marriage"]
    if person_ids is not None:
        attrs = attrs[attrs["person_id"].isin(person_ids)]
        ev = ev[ev["person_id"].isin(person_ids)]
    first_age = ev.set_index("person_id")["age"]

    cells: dict[str, dict[str, float]] = {}
    for target in OBSERVED_TO_AGES:
        observed = attrs[attrs["censor_year"] >= attrs["birth_year"] + target]
        for sex in SEXES:
            grp = observed[observed["sex"] == sex]
            w = grp["weight"] if weighted else pd.Series(1.0, index=grp.index)
            ever = grp["person_id"].map(first_age)
            ever_by = (ever.notna() & (ever <= target)).to_numpy()
            den = float(w.sum())
            num = float(w[ever_by].sum())
            n = int(ever_by.sum())
            cells[f"ever_married_by_{target}|{sex}"] = _rate_cell(num, den, n)

    completed = attrs[
        attrs["censor_year"] >= attrs["birth_year"] + COMPLETED_FERTILITY_AGE
    ]
    for sex in SEXES:
        grp = completed[
            (completed["sex"] == sex) & (completed["n_marriages"] >= 1)
        ]
        w = grp["weight"] if weighted else pd.Series(1.0, index=grp.index)
        num = float((w * grp["n_marriages"]).sum())
        den = float(w.sum())
        cells[f"mean_lifetime_marriages|{sex}"] = _rate_cell(
            num, den, int(len(grp))
        )
    return cells


# --------------------------------------------------------------------------
# Fertility panel (woman-years + births) and cells
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FertilityPanel:
    """Woman-year exposure and dated maternal births for the ASFR moments.

    Attributes:
        woman_years: Female person-years aged :data:`ASFR_AGE_BANDS` bounds,
            with ``person_id``, ``year``, ``age``, ``weight``.
        births: Birth-type events to those women with ``person_id``
            (mother), ``year``, ``mother_age``, ``weight``.
        completed: Per-woman completed-parity frame (women observed through
            :data:`COMPLETED_FERTILITY_AGE`) with ``birth_decade``,
            ``n_births``, ``weight``.
    """

    woman_years: pd.DataFrame
    births: pd.DataFrame
    completed: pd.DataFrame


def build_fertility_panel(
    panel: MaritalPanel, birth_records: pd.DataFrame
) -> FertilityPanel:
    """Woman-year fertility exposure and maternal births from the panel.

    Reuses the marital panel's female person-years as the ASFR denominator
    (so the exposure base is identical to the marital moments) and counts
    birth-type events to those women, dated at the child's birth year and
    aged by the mother's own birth year. ``birth_records`` is a
    :func:`populace_dynamics.data.births.birth_history`-shaped frame.
    """
    attrs = panel.attrs
    women = attrs[attrs["sex"] == "female"]
    women_ids = set(women["person_id"])
    lo = min(b[0] for b in ASFR_AGE_BANDS)
    hi = max(b[1] for b in ASFR_AGE_BANDS)
    wy = panel.person_years[
        panel.person_years["person_id"].isin(women_ids)
        & (panel.person_years["age"] >= lo)
        & (panel.person_years["age"] <= hi)
    ][["person_id", "year", "age", "weight"]].reset_index(drop=True)

    birth_year = women.set_index("person_id")["birth_year"]
    censor = women.set_index("person_id")["censor_year"]
    weight = women.set_index("person_id")["weight"]

    be = births.birth_events(birth_records)
    be = be[
        (be["record_type"] == "birth")
        & be["parent_person_id"].isin(women_ids)
        & be["birth_year"].notna()
    ].copy()
    be["mother_birth_year"] = be["parent_person_id"].map(birth_year)
    be["mother_censor"] = be["parent_person_id"].map(censor)
    be["mother_age"] = (be["birth_year"] - be["mother_birth_year"]).astype(
        "Int64"
    )
    in_window = (
        (be["birth_year"] >= be["mother_birth_year"] + lo)
        & (be["birth_year"] <= be["mother_censor"])
        & (be["mother_age"] >= lo)
        & (be["mother_age"] <= hi)
    )
    be = be[in_window]
    births_frame = pd.DataFrame(
        {
            "person_id": be["parent_person_id"].to_numpy(),
            "year": be["birth_year"].astype("int64").to_numpy(),
            "mother_age": be["mother_age"].astype("int64").to_numpy(),
            "weight": be["parent_person_id"].map(weight).to_numpy(),
        }
    )

    # Completed parity: women observed through COMPLETED_FERTILITY_AGE, all
    # of their birth-type events (any age), by the woman's birth decade.
    completed_women = women[
        women["censor_year"] >= women["birth_year"] + COMPLETED_FERTILITY_AGE
    ].copy()
    all_births = births.birth_events(birth_records)
    all_births = all_births[
        (all_births["record_type"] == "birth")
        & all_births["parent_person_id"].isin(
            set(completed_women["person_id"])
        )
    ]
    parity = all_births.groupby("parent_person_id").size()
    completed_women["n_births"] = (
        completed_women["person_id"].map(parity).fillna(0).astype("int64")
    )
    completed_women["birth_decade"] = (
        completed_women["birth_year"] // 10 * 10
    ).astype("int64")
    completed = completed_women[
        ["person_id", "birth_decade", "n_births", "weight"]
    ].reset_index(drop=True)
    return FertilityPanel(
        woman_years=wy, births=births_frame, completed=completed
    )


def fertility_cells(
    fert: FertilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """ASFR-by-band and completed-fertility-by-cohort cells for a subset."""
    wy = fert.woman_years
    bi = fert.births
    comp = fert.completed
    if person_ids is not None:
        wy = wy[wy["person_id"].isin(person_ids)]
        bi = bi[bi["person_id"].isin(person_ids)]
        comp = comp[comp["person_id"].isin(person_ids)]

    cells: dict[str, dict[str, float]] = {}
    wy = wy.assign(band=wy["age"].map(lambda a: _band_of(a, ASFR_AGE_BANDS)))
    bi = bi.assign(
        band=bi["mother_age"].map(lambda a: _band_of(a, ASFR_AGE_BANDS))
    )
    wy_w = wy["weight"] if weighted else pd.Series(1.0, index=wy.index)
    bi_w = bi["weight"] if weighted else pd.Series(1.0, index=bi.index)
    den = wy.assign(_w=wy_w).groupby("band", observed=True)["_w"].sum()
    num = (
        bi.assign(_w=bi_w)
        .groupby("band", observed=True)
        .agg(num_wt=("_w", "sum"), n_events=("_w", "size"))
    )
    for lo, hi in ASFR_AGE_BANDS:
        band = band_label(lo, hi)
        cells[f"asfr.{band}"] = _rate_cell(
            float(num["num_wt"].get(band, 0.0)),
            float(den.get(band, 0.0)),
            int(num["n_events"].get(band, 0)),
        )

    cw = comp["weight"] if weighted else pd.Series(1.0, index=comp.index)
    comp = comp.assign(_w=cw)
    for decade in COHORT_DECADES:
        grp = comp[comp["birth_decade"] == decade]
        den_w = float(grp["_w"].sum())
        num_w = float((grp["_w"] * grp["n_births"]).sum())
        cells[f"completed_fertility.c{decade}s"] = _rate_cell(
            num_w, den_w, int(len(grp))
        )
    return cells


def reference_moments(
    panel: MaritalPanel,
    fert: FertilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Every gate-2 reference-moment cell for a person subset.

    The union of :func:`hazard_cells`, :func:`occupancy_cells`, and
    :func:`fertility_cells`. Calling it on ``person_ids=None`` gives the
    committed reference moments; calling it on each half of a
    person-disjoint split gives the noise-floor inputs.
    """
    cells: dict[str, dict[str, float]] = {}
    cells.update(hazard_cells(panel, person_ids, weighted=weighted))
    cells.update(occupancy_cells(panel, person_ids, weighted=weighted))
    cells.update(fertility_cells(fert, person_ids, weighted=weighted))
    return cells
