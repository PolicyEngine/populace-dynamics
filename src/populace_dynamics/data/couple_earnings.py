"""Marriage x earnings JOINT moments -- the gate-2c holdout surface.

The gate-2c holdout basis (``gates.yaml`` gate_2c, ``covers: the joint of
marriage and earnings (who marries whom)``; ``holdout_basis: mh85_23
marital histories crossed with the gate-1-certified earnings histories``)
is the JOINT of *who marries whom* and *what they earn* -- the surface on
which spousal / survivor benefit LEVELS depend, and the one the locked 2a
scope_note explicitly excludes (``marriage_x_earnings_joint = NOT
COVERED``). This module crosses the PSID Marriage History File
(:mod:`populace_dynamics.data.marriage`) with the family-file earnings
panel (:mod:`populace_dynamics.data.family`) and the certified AIME chain
(:mod:`populace_dynamics.ss.benefits`), producing the reference-moment
cells the gate-2c noise floor is measured on -- the analogue, one tranche
over, of what :mod:`populace_dynamics.data.household_composition` builds
for gate-2b and :mod:`populace_dynamics.data.transitions` for tranche 2a.

The earnings measure (load-bearing)
-----------------------------------
Each person's earnings capacity is summarised by their **AIME** (average
indexed monthly earnings, :func:`populace_dynamics.ss.benefits.aime`), the
SSA-relevant, wage-indexed quantity the spousal / survivor benefit is a
function of -- so terciles pool across cohorts on a comparable scale. It is
a **PSID-observed partial-career AIME proxy**, not the true SSA AIME: the
family earnings panel observes head/spouse labor income for reference years
1993-2022 only, so a person's covered history is truncated (top-35 of the
observed years, zeros filling the rest) and the LEVELS are lower bounds.
The terciles and within-couple ranks -- which the gated moments use -- are
far more robust to that truncation than the levels, but the truncation is a
disclosed wart. AIME is defined only where the age-60 indexing year lands
inside the NAWI series and at least :data:`MIN_EARNINGS_YEARS` positive
earnings years are observed.

The couple unit and the person split (load-bearing)
---------------------------------------------------
A **directed couple** is one marriage-history record whose spouse is a
joinable PSID sample-member person (``spouse_person_id`` populated, i.e.
``MH7`` in 1-9308 and ``MH8`` in 1-399) and where BOTH ego and spouse carry
a computable AIME. A marriage between two sample members yields two directed
records (ego A -> spouse B and ego B -> spouse A), so the tercile
contingency is symmetric by construction. The person-disjoint floor splits
by **ego person_id** (:func:`populace_dynamics.harness.panel.
split_panel_by_person`): every couple, marital event and earnings window of
an ego lands on one side. A couple whose two members split to opposite
sides contributes one directed record to EACH half, so the halves are
person-disjoint but NOT couple-disjoint -- a disclosed correlation wart
(it makes the halves marginally more alike, so the floor is if anything
conservative).

Statistic families
------------------
1. **assortative mating** -- own-AIME-tercile x spouse-AIME-tercile
   contingency shares over directed couples (``assort_mating.own{o}
   _spouse{u}``); the within-couple AIME rank correlation (Spearman),
   overall and by marriage-decade, is reported alongside REPORT-ONLY (a
   correlation is not a scale-free log-rate, so it is not gated);
2. **marriage hazards conditional on own earnings** --
   ``first_marriage_by_earnings.t{o}.{band}|{sex}`` (never-married
   person-years at risk) and ``remarriage_by_earnings.t{o}|{sex}``
   (post-dissolution person-years at risk), the tranche-2a marital hazards
   stratified by the person's fixed AIME tercile;
3. **earnings dynamics around marital events** --
   ``earnings_around_{marriage,divorce}.{all,female,male}``, the weighted
   median of each person's (post-event / pre-event) mean-earnings ratio
   over a +/-3-year window, on the events with observed earnings support;
4. **shared-earnings distribution shape** --
   ``shared_earnings_ratio.q{hi}_q{lo}``, adjacent-quintile cutpoint ratios
   of the couple's combined AIME (own + spouse), a scale-free shape moment.

The gate-eligible / report-only split follows 2a/2b exactly (>= 20 events
on the weaker half of the worst seed, defined on every seed, stabilised
tolerance <= T_max = ln(1.5)). There are NO coverage-recovery aggregates:
unlike 2a/2b (whose moderate-rate families had a sparse older-age tail
whose adjacent bands ALL failed the cap, so pooling them recovered coverage
without masking), 2c's sparse cells are the late first-marriage-by-earnings
cells -- pooling age bands would mask a standalone-gateable 35-44 cell
(round-1 finding 6b) and pooling terciles would destroy the earnings signal
the tranche exists to measure, so the sparse cells are honestly
report-only.

Weighting
---------
Each person carries the person-constant most-recent-positive PSID
cross-sectional weight (:func:`populace_dynamics.data.panels.
demographic_panel`), exactly tranche 2a's ``weight_definition``; there is no
unweighted gated statistic, and the unweighted rate is reported alongside.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from populace_dynamics.data import (
    deaths,
    family,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters, load_ssa_parameters

__all__ = [
    "SEXES",
    "TERCILES",
    "AIME_TERCILE_QUANTILES",
    "SHARED_QUINTILE_LEVELS",
    "SHARED_RATIO_PAIRS",
    "FIRST_MARRIAGE_AGE_BANDS",
    "MIN_EARNINGS_YEARS",
    "EVENT_WINDOW_YEARS",
    "MARRIAGE_DECADE_BANDS",
    "CoupleEarningsPanel",
    "aggregation_members",
    "person_earnings_histories",
    "aime_supply",
    "build_couple_panel",
    "reference_moments",
    "assortative_correlation",
]

#: The two coded sexes (na sex is dropped), matching tranche 2a.
SEXES: tuple[str, ...] = ("female", "male")

#: Earnings-tercile labels (1 = lowest AIME third).
TERCILES: tuple[int, ...] = (1, 2, 3)
#: The AIME tercile cut quantiles (over the AIME supply population).
AIME_TERCILE_QUANTILES: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)

#: Shared-earnings (combined AIME) quintile cut levels and the adjacent
#: cutpoint ratios reported as scale-free distribution-shape moments.
SHARED_QUINTILE_LEVELS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
SHARED_RATIO_PAIRS: tuple[tuple[str, float, float], ...] = (
    ("q40_q20", 0.4, 0.2),
    ("q60_q40", 0.6, 0.4),
    ("q80_q60", 0.8, 0.6),
    ("q80_q20", 0.8, 0.2),
)

#: First-marriage age bands (reused verbatim from tranche 2a so the
#: earnings-stratified hazard bands match the locked marital surface).
FIRST_MARRIAGE_AGE_BANDS: tuple[tuple[int, int], ...] = (
    transitions.FIRST_MARRIAGE_AGE_BANDS
)

#: Minimum observed positive-earnings years for a person's AIME proxy to be
#: admitted (reduces the worst partial-career truncation noise in the
#: tercile assignment; disclosed).
MIN_EARNINGS_YEARS = 5

#: Half-width (years) of the pre/post window for the earnings-around-event
#: ratio. The ratio is (mean post-window earnings) / (mean pre-window
#: earnings) over observed positive-earnings years in each side.
EVENT_WINDOW_YEARS = 3

#: Marriage-decade bands for the REPORT-ONLY by-decade Spearman correlation
#: (a couple's start-year decade); dense decades only.
MARRIAGE_DECADE_BANDS: tuple[int, ...] = (
    1940,
    1950,
    1960,
    1970,
    1980,
    1990,
    2000,
    2010,
)


def band_label(lo: int, hi: int) -> str:
    """Flat band label, e.g. ``"25-34"`` or open-topped ``"45+"``."""
    return f"{lo}-{hi}" if hi < 120 else f"{lo}+"


def _fm_band_of(age: int) -> str | None:
    for lo, hi in FIRST_MARRIAGE_AGE_BANDS:
        if lo <= age <= hi:
            return band_label(lo, hi)
    return None


def aggregation_members() -> dict[str, list[str]]:
    """No coverage-recovery aggregates for gate-2c (see the module
    docstring): pooling the sparse late-first-marriage age bands would mask
    a standalone-gateable 35-44 cell (round-1 finding 6b) or pool across the
    earnings terciles the tranche exists to resolve. Returned empty so the
    build script and tests bind the deliberate absence."""
    return {}


@dataclass(frozen=True)
class CoupleEarningsPanel:
    """The marriage x earnings panel and its per-person attribute frame.

    Attributes:
        couples: One row per directed couple (marriage-history record with a
            joinable, both-AIME spouse): ``person_id`` (ego),
            ``spouse_person_id``, ``sex``, ``weight``, ``aime_own``,
            ``aime_spouse``, ``own_tercile``, ``spouse_tercile``,
            ``shared`` (own + spouse AIME), ``start_year``, ``start_decade``.
        marital_events: AIME-supply marital transition events
            (``first_marriage`` / ``remarriage``) with ``person_id``,
            ``age``, ``sex``, ``weight``, ``transition``, ``tercile``,
            ``fm_band``.
        marital_exposure: AIME-supply person-years with ``person_id``,
            ``sex``, ``weight``, ``marital_state``, ``tercile``, ``age``,
            ``fm_band`` (the hazard denominators).
        event_windows: One row per marital event with observed earnings
            support: ``person_id``, ``sex``, ``weight``, ``event_type``
            (``marriage`` / ``divorce``), ``ratio`` (post/pre mean earnings).
        attrs: One row per person_id (the split / holdout unit).
        aime_tercile_cuts: The committed AIME tercile cut levels (t33, t67),
            fixed on the full AIME supply and applied to every half-split.
        meta: Provenance / coverage counts recorded in the artifact.
    """

    couples: pd.DataFrame
    marital_events: pd.DataFrame
    marital_exposure: pd.DataFrame
    event_windows: pd.DataFrame
    attrs: pd.DataFrame
    aime_tercile_cuts: tuple[float, float]
    meta: dict[str, object]


# --------------------------------------------------------------------------
# Earnings histories + the AIME proxy supply
# --------------------------------------------------------------------------
def person_earnings_histories(
    panel: pd.DataFrame,
) -> tuple[dict[int, dict[int, float]], dict[int, int], dict[int, float]]:
    """Per-person ``{year: earnings}`` history, implied birth year, weight.

    Pure transform over a :func:`populace_dynamics.data.family.
    family_earnings_panel`-shaped frame (``person_id``, ``period``,
    ``earnings``, ``age``, ``weight``). Birth year is the median implied
    ``period - age`` (the R7 convention); the anchor weight is the person's
    chronologically last observed weight.
    """
    p = panel[(panel["age"] >= 14) & (panel["age"] <= 90)].copy()
    p["implied_birth_year"] = p["period"] - p["age"]
    birth_year = (
        p.groupby("person_id")["implied_birth_year"]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )
    weight = (
        p.sort_values("period").groupby("person_id")["weight"].last().to_dict()
    )
    history: dict[int, dict[int, float]] = {}
    for pid, sub in p.groupby("person_id"):
        series = sub.groupby("period")["earnings"].sum()
        history[int(pid)] = {int(y): float(v) for y, v in series.items()}
    return (
        history,
        {int(k): int(v) for k, v in birth_year.items()},
        {int(k): float(v) for k, v in weight.items()},
    )


def aime_supply(
    history: dict[int, dict[int, float]],
    birth_year: dict[int, int],
    params: SSAParameters,
) -> dict[int, float]:
    """AIME proxy for every person whose age-60 indexing year lands in the
    NAWI series and who has >= :data:`MIN_EARNINGS_YEARS` positive-earnings
    years -- the certified-earnings-history supply the couples draw from."""
    min_nawi, max_nawi = min(params.nawi), max(params.nawi)
    supply: dict[int, float] = {}
    for pid, hist in history.items():
        b = birth_year.get(pid)
        if b is None or not (min_nawi <= b + 60 <= max_nawi):
            continue
        if sum(1 for v in hist.values() if v > 0) < MIN_EARNINGS_YEARS:
            continue
        supply[pid] = float(benefits.aime(hist, b, params))
    return supply


def _tercile_of(value: float, cuts: tuple[float, float]) -> int:
    return 1 + int(value >= cuts[0]) + int(value >= cuts[1])


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------
def _build_couples(
    episodes: pd.DataFrame,
    aime: dict[int, float],
    sex: dict[int, str],
    weight: dict[int, float],
    cuts: tuple[float, float],
) -> pd.DataFrame:
    """Directed both-AIME couples with terciles, shared AIME, decade."""
    ep = episodes[episodes["spouse_person_id"].notna()].copy()
    ep["spouse_person_id"] = ep["spouse_person_id"].astype(int)
    ep["aime_own"] = ep["person_id"].map(aime)
    ep["aime_spouse"] = ep["spouse_person_id"].map(aime)
    ep["sex"] = ep["person_id"].map(sex)
    ep = ep[
        ep["aime_own"].notna()
        & ep["aime_spouse"].notna()
        & ep["sex"].isin(SEXES)
        & ep["start_year"].notna()
    ].copy()
    ep["weight"] = ep["person_id"].map(weight).fillna(0.0).astype(float)
    ep = ep[ep["weight"] > 0]
    ep["own_tercile"] = ep["aime_own"].map(lambda v: _tercile_of(v, cuts))
    ep["spouse_tercile"] = ep["aime_spouse"].map(
        lambda v: _tercile_of(v, cuts)
    )
    ep["shared"] = ep["aime_own"].astype(float) + ep["aime_spouse"].astype(
        float
    )
    ep["start_year"] = ep["start_year"].astype(int)
    ep["start_decade"] = (ep["start_year"] // 10 * 10).astype(int)
    return ep[
        [
            "person_id",
            "spouse_person_id",
            "sex",
            "weight",
            "aime_own",
            "aime_spouse",
            "own_tercile",
            "spouse_tercile",
            "shared",
            "start_year",
            "start_decade",
        ]
    ].reset_index(drop=True)


def _build_marital(
    marital: transitions.MaritalPanel,
    aime: dict[int, float],
    cuts: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AIME-supply marital events + exposure, tercile- and band-tagged."""
    supply = set(aime)
    tmap = {pid: _tercile_of(v, cuts) for pid, v in aime.items()}

    ev = marital.events[
        marital.events["person_id"].isin(supply)
        & marital.events["transition"].isin(["first_marriage", "remarriage"])
    ].copy()
    ev["tercile"] = ev["person_id"].map(tmap)
    ev["fm_band"] = ev["age"].map(_fm_band_of)
    ev = ev[
        [
            "person_id",
            "age",
            "sex",
            "weight",
            "transition",
            "tercile",
            "fm_band",
        ]
    ]

    py = marital.person_years[
        marital.person_years["person_id"].isin(supply)
    ].copy()
    py["tercile"] = py["person_id"].map(tmap)
    py["fm_band"] = py["age"].map(_fm_band_of)
    py = py[
        [
            "person_id",
            "age",
            "sex",
            "weight",
            "marital_state",
            "tercile",
            "fm_band",
        ]
    ]
    return ev.reset_index(drop=True), py.reset_index(drop=True)


def _event_windows(
    episodes: pd.DataFrame,
    history: dict[int, dict[int, float]],
    sex: dict[int, str],
    weight: dict[int, float],
) -> pd.DataFrame:
    """Per marital event, the post/pre mean-earnings ratio where supported.

    Marriage events use the marriage ``start_year``; divorce events use the
    dissolution ``episode_end_year`` of divorce-ended episodes. An event
    contributes only when the person has at least one observed
    positive-earnings year in BOTH the pre window ``[y-w, y-1]`` and the
    post window ``[y+1, y+w]`` (``w`` = :data:`EVENT_WINDOW_YEARS`).
    """
    rows: list[dict[str, object]] = []
    plans = (
        ("marriage", episodes["start_year"], episodes),
        (
            "divorce",
            episodes["episode_end_year"],
            episodes[episodes["how_ended"] == "divorce"],
        ),
    )
    w = EVENT_WINDOW_YEARS
    for event_type, _years, src in plans:
        year_col = (
            "start_year" if event_type == "marriage" else "episode_end_year"
        )
        sub = src[src[year_col].notna()]
        for row in sub.itertuples(index=False):
            pid = int(row.person_id)
            hist = history.get(pid)
            s = sex.get(pid)
            if hist is None or s not in SEXES:
                continue
            year = int(getattr(row, year_col))
            pre = [
                hist[y]
                for y in range(year - w, year)
                if hist.get(y, 0.0) > 0.0
            ]
            post = [
                hist[y]
                for y in range(year + 1, year + w + 1)
                if hist.get(y, 0.0) > 0.0
            ]
            if not pre or not post:
                continue
            rows.append(
                {
                    "person_id": pid,
                    "sex": s,
                    "weight": float(weight.get(pid, 0.0)),
                    "event_type": event_type,
                    "ratio": float(np.mean(post) / np.mean(pre)),
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=["person_id", "sex", "weight", "event_type", "ratio"],
    )
    return frame[frame["weight"] > 0].reset_index(drop=True)


def build_couple_panel(
    *,
    data_dir: Path | None = None,
    params: SSAParameters | None = None,
    marriage_records: pd.DataFrame | None = None,
    earnings_panel: pd.DataFrame | None = None,
    death_records: pd.DataFrame | None = None,
    person_weight: pd.Series | None = None,
) -> CoupleEarningsPanel:
    """Assemble the marriage x earnings panel from the staged PSID products.

    Loads the marriage history (mh85_23), the family earnings panel, the
    demographic weights and the death records, computes the AIME proxy
    supply from the certified :func:`populace_dynamics.ss.benefits.aime`
    chain (SSA parameters loaded via :func:`load_ssa_parameters` unless
    injected), and builds the directed-couple, AIME-supply-marital and
    earnings-window frames. Source frames may be injected for synthetic
    unit tests (then no PSID / pe-us read happens).
    """
    if params is None:
        params = load_ssa_parameters()
    if marriage_records is None:
        marriage_records = marriage.marriage_history(data_dir=data_dir)
    if earnings_panel is None:
        earnings_panel = family.family_earnings_panel(data_dir=data_dir)
    if death_records is None:
        death_records = deaths.read_death_records(data_dir=data_dir)
    if person_weight is None:
        demo = panels.demographic_panel(data_dir=data_dir)
        demo_pos = demo[demo["weight"] > 0]
        person_weight = (
            demo_pos.sort_values("period")
            .groupby("person_id")
            .tail(1)
            .set_index("person_id")["weight"]
        )

    history, birth_year, earn_weight = person_earnings_histories(
        earnings_panel
    )
    aime = aime_supply(history, birth_year, params)
    cut_lo, cut_hi = (
        np.quantile(
            np.fromiter(aime.values(), dtype=float),
            AIME_TERCILE_QUANTILES,
        )
        if aime
        else (0.0, 0.0)
    )
    cuts = (float(cut_lo), float(cut_hi))

    sex = (
        marriage_records.drop_duplicates("person_id")
        .set_index("person_id")["sex"]
        .to_dict()
    )
    pw = {int(k): float(v) for k, v in person_weight.items()}

    episodes = marriage.marriage_episodes(marriage_records)
    couples = _build_couples(episodes, aime, sex, pw, cuts)

    marital = transitions.build_marital_panel(
        marriage_records, death_records, person_weight
    )
    marital_events, marital_exposure = _build_marital(marital, aime, cuts)

    event_windows = _event_windows(episodes, history, sex, earn_weight)

    person_ids = pd.unique(
        pd.concat(
            [
                couples["person_id"],
                marital_exposure["person_id"],
                event_windows["person_id"],
            ],
            ignore_index=True,
        )
    )
    attrs = pd.DataFrame({"person_id": np.sort(person_ids)})

    n_joinable = int(episodes["spouse_person_id"].notna().sum())
    ego_supply = (
        episodes.loc[episodes["spouse_person_id"].notna(), "person_id"]
        .isin(aime)
        .sum()
    )
    meta = {
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "nawi_year_range": [int(min(params.nawi)), int(max(params.nawi))],
        "n_aime_supply_persons": len(aime),
        "n_marriage_episodes_joinable": n_joinable,
        "n_directed_couples": int(len(couples)),
        "n_unordered_couple_pairs": (
            int(
                couples.apply(
                    lambda r: frozenset(
                        (r["person_id"], r["spouse_person_id"])
                    ),
                    axis=1,
                ).nunique()
            )
            if len(couples)
            else 0
        ),
        "join_coverage_both_over_joinable": (
            round(len(couples) / n_joinable, 4) if n_joinable else None
        ),
        "join_coverage_both_over_ego_supply": (
            round(float(len(couples)) / float(ego_supply), 4)
            if ego_supply
            else None
        ),
        "n_event_windows": int(len(event_windows)),
        "n_split_persons": int(len(attrs)),
        "min_earnings_years": MIN_EARNINGS_YEARS,
        "event_window_years": EVENT_WINDOW_YEARS,
    }

    return CoupleEarningsPanel(
        couples=couples,
        marital_events=marital_events,
        marital_exposure=marital_exposure,
        event_windows=event_windows,
        attrs=attrs,
        aime_tercile_cuts=cuts,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Reference moments
# --------------------------------------------------------------------------
def _rate_cell(
    num_weight: float, den_weight: float, n_events: int
) -> dict[str, float]:
    """The shared cell schema (mirrors transitions / household_composition).

    For the share and hazard families ``rate = num_wt / den_wt`` is the
    weighted share / hazard. For the median and cutpoint-ratio families the
    value has no natural numerator, so ``rate`` is stored directly and
    ``num_wt`` is defined as ``rate * den_wt`` to keep the ``rate ==
    num_wt / den_wt`` invariant uniform across every cell (``den_wt`` = the
    events' total weight); see :func:`_value_cell`.
    """
    return {
        "rate": float(num_weight / den_weight) if den_weight > 0 else 0.0,
        "num_wt": float(num_weight),
        "den_wt": float(den_weight),
        "n_events": int(n_events),
    }


def _value_cell(
    value: float, total_weight: float, n_events: int
) -> dict[str, float]:
    """A non-share cell (median ratio, cutpoint ratio): store the value as
    the rate, ``num_wt = value * den_wt`` so the uniform invariant holds."""
    return {
        "rate": float(value),
        "num_wt": float(value * total_weight),
        "den_wt": float(total_weight),
        "n_events": int(n_events),
    }


def _weight_col(frame: pd.DataFrame, weighted: bool) -> np.ndarray:
    if weighted:
        return frame["weight"].to_numpy(dtype=float)
    return np.ones(len(frame), dtype=float)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Lower weighted median (deterministic): smallest value whose
    cumulative weight reaches half the total."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    csum = np.cumsum(w)
    half = 0.5 * csum[-1]
    idx = int(np.searchsorted(csum, half, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """Weighted quantile by the cumulative-weight (lower) convention."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    csum = np.cumsum(w)
    target = q * csum[-1]
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _assortative_cells(
    couples: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """The 3x3 own-tercile x spouse-tercile contingency shares."""
    w = _weight_col(couples, weighted)
    total = float(w.sum())
    ot = couples["own_tercile"].to_numpy()
    st = couples["spouse_tercile"].to_numpy()
    cells: dict[str, dict[str, float]] = {}
    for o in TERCILES:
        for u in TERCILES:
            mask = (ot == o) & (st == u)
            cells[f"assort_mating.own{o}_spouse{u}"] = _rate_cell(
                float(w[mask].sum()), total, int(mask.sum())
            )
    return cells


def _first_marriage_cells(
    events: pd.DataFrame, exposure: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """First-marriage hazard by AIME tercile x age band x sex."""
    ev = events[events["transition"] == "first_marriage"]
    ex = exposure[exposure["marital_state"] == "never_married"]
    ev_w = _weight_col(ev, weighted)
    ex_w = _weight_col(ex, weighted)
    cells: dict[str, dict[str, float]] = {}
    for terc in TERCILES:
        for lo, hi in FIRST_MARRIAGE_AGE_BANDS:
            band = band_label(lo, hi)
            for sex in SEXES:
                em = (
                    (ev["tercile"].to_numpy() == terc)
                    & (ev["fm_band"].to_numpy() == band)
                    & (ev["sex"].to_numpy() == sex)
                )
                xm = (
                    (ex["tercile"].to_numpy() == terc)
                    & (ex["fm_band"].to_numpy() == band)
                    & (ex["sex"].to_numpy() == sex)
                )
                key = f"first_marriage_by_earnings.t{terc}.{band}|{sex}"
                cells[key] = _rate_cell(
                    float(ev_w[em].sum()),
                    float(ex_w[xm].sum()),
                    int(em.sum()),
                )
    return cells


def _remarriage_cells(
    events: pd.DataFrame, exposure: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """Remarriage hazard by AIME tercile x sex (pooled over age)."""
    ev = events[events["transition"] == "remarriage"]
    ex = exposure[
        exposure["marital_state"].isin(transitions._REMARRIAGE_RISK_STATES)
    ]
    ev_w = _weight_col(ev, weighted)
    ex_w = _weight_col(ex, weighted)
    cells: dict[str, dict[str, float]] = {}
    for terc in TERCILES:
        for sex in SEXES:
            em = (ev["tercile"].to_numpy() == terc) & (
                ev["sex"].to_numpy() == sex
            )
            xm = (ex["tercile"].to_numpy() == terc) & (
                ex["sex"].to_numpy() == sex
            )
            cells[f"remarriage_by_earnings.t{terc}|{sex}"] = _rate_cell(
                float(ev_w[em].sum()),
                float(ex_w[xm].sum()),
                int(em.sum()),
            )
    return cells


def _event_window_cells(
    windows: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """Median (post/pre) earnings ratio around marriage and divorce."""
    cells: dict[str, dict[str, float]] = {}
    for event_type in ("marriage", "divorce"):
        sub = windows[windows["event_type"] == event_type]
        for group in ("all", "female", "male"):
            g = sub if group == "all" else sub[sub["sex"] == group]
            key = f"earnings_around_{event_type}.{group}"
            if len(g) == 0:
                cells[key] = _value_cell(0.0, 0.0, 0)
                continue
            ratios = g["ratio"].to_numpy(dtype=float)
            w = _weight_col(g, weighted)
            cells[key] = _value_cell(
                _weighted_median(ratios, w), float(w.sum()), int(len(g))
            )
    return cells


def _shared_earnings_cells(
    couples: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """Adjacent-quintile cutpoint ratios of the couple's combined AIME."""
    cells: dict[str, dict[str, float]] = {}
    shared = couples["shared"].to_numpy(dtype=float)
    w = _weight_col(couples, weighted)
    total = float(w.sum())
    n = int(len(couples))
    cut = {
        level: _weighted_quantile(shared, w, level)
        for level in SHARED_QUINTILE_LEVELS
    }
    for name, hi, lo in SHARED_RATIO_PAIRS:
        denom = cut[lo]
        ratio = float(cut[hi] / denom) if denom > 0 else 0.0
        cells[f"shared_earnings_ratio.{name}"] = _value_cell(ratio, total, n)
    return cells


def reference_moments(
    panel: CoupleEarningsPanel,
    person_ids: Iterable[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Every gate-2c marriage x earnings reference-moment cell.

    The union of the assortative-mating tercile contingency shares, the
    earnings-stratified first-marriage and remarriage hazards, the
    around-event earnings-ratio medians, and the shared-earnings
    cutpoint-ratio shape moments. ``person_ids=None`` gives the committed
    reference moments; passing each half of a person-disjoint split (by ego
    ``person_id``) gives the noise-floor inputs. The AIME tercile cut levels
    are fixed on the full supply (``panel.aime_tercile_cuts``), so a half's
    contingency and hazard categories do not shift.
    """
    couples = panel.couples
    events = panel.marital_events
    exposure = panel.marital_exposure
    windows = panel.event_windows
    if person_ids is not None:
        ids = set(person_ids)
        couples = couples[couples["person_id"].isin(ids)]
        events = events[events["person_id"].isin(ids)]
        exposure = exposure[exposure["person_id"].isin(ids)]
        windows = windows[windows["person_id"].isin(ids)]

    cells: dict[str, dict[str, float]] = {}
    cells.update(_assortative_cells(couples, weighted=weighted))
    cells.update(_first_marriage_cells(events, exposure, weighted=weighted))
    cells.update(_remarriage_cells(events, exposure, weighted=weighted))
    cells.update(_event_window_cells(windows, weighted=weighted))
    cells.update(_shared_earnings_cells(couples, weighted=weighted))
    return cells


def assortative_correlation(
    panel: CoupleEarningsPanel,
    person_ids: Iterable[int] | None = None,
) -> dict[str, object]:
    """REPORT-ONLY within-couple AIME rank correlation (Spearman).

    Overall and by marriage-decade band. A correlation is not a scale-free
    log-rate, so it is NEVER gated (the assortative-mating GATE runs on the
    tercile contingency shares); this is the interpretable scalar summary
    reported alongside. Computed on the directed couples, so it is the rank
    correlation over marriage-person records.
    """
    couples = panel.couples
    if person_ids is not None:
        couples = couples[couples["person_id"].isin(set(person_ids))]

    def _spearman(frame: pd.DataFrame) -> float | None:
        if len(frame) < 3:
            return None
        rho = frame[["aime_own", "aime_spouse"]].corr("spearman").iloc[0, 1]
        return None if pd.isna(rho) else round(float(rho), 4)

    by_decade = {}
    for decade in MARRIAGE_DECADE_BANDS:
        sub = couples[couples["start_decade"] == decade]
        by_decade[str(decade)] = {
            "n_directed_couples": int(len(sub)),
            "spearman_aime": _spearman(sub),
        }
    return {
        "gated": False,
        "report_only": True,
        "statistic": "within-couple Spearman rank correlation of AIME",
        "overall": {
            "n_directed_couples": int(len(couples)),
            "spearman_aime": _spearman(couples),
        },
        "by_marriage_decade": by_decade,
        "note": (
            "REPORT-ONLY: a rank correlation is bounded in [-1, 1] and can "
            "cross zero, so it is not a scale-free |ln ratio| and is never "
            "gated. The assortative-mating GATE runs on the tercile "
            "contingency shares (assort_mating.own{o}_spouse{u}); this rho "
            "is the interpretable summary reported alongside."
        ),
    }
