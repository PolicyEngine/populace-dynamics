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

The earnings axis (load-bearing; fix C -- construct validity)
------------------------------------------------------------
Round-1 fix C. Each person's earnings axis is their **mean indexed
positive-year earnings** (:func:`indexed_earnings_supply`): the NAWI-indexed
family-file labor income averaged over the person's OBSERVED
positive-earnings years -- a per-year earnings-capacity measure, NOT a
career sum. The round-1 draft sorted couples on the **career-sum AIME
proxy** (top-35 indexed years / 420, zeros filling the unobserved rest);
the referee's decomposition showed that proxy's within-couple sorting
signal (Spearman rho 0.12) is dominated by OBSERVATION MECHANICS, not
earnings: the number of observed positive years anti-correlates within
couples (rho -0.095, single-earner-era head/wife role-years), the zeros-fill
rewards more-observed spouses, and cross-sex tercile pooling halves the
signal. On the SAME couples the per-positive-year earnings rank correlates
0.49 (literature-scale sorting; within-sex AIME 0.21), and the proxy's
split-half rank reliability is 0.98 -- so the weak career-AIME signal is
NOT sampling noise, it is what the career-sum proxy measures. The gate must
certify the marriage x earnings JOINT, so the earnings axis is the per-year
measure. The career-AIME proxy, the within-sex rank, the observed-years
count and the split-half reliability ride in
``assortative_correlation`` REPORT-ONLY as the decomposition, so the future
external-anchor bridge (EMZ/CPS/MINT positive sorting) does not read as a
contradiction.

Admission is unchanged from the draft (same SELECTED universe, 14,952
supply persons / 7,994 directed couples): a person's axis is defined only
where the age-60 NAWI indexing year lands inside the series and at least
:data:`MIN_EARNINGS_YEARS` positive-earnings years are observed. Only the
SCALAR the terciles / ranks / combined level read changed, not who is
admitted.

The observed earnings window (load-bearing; fix B -- estimand truth)
-------------------------------------------------------------------
Round-1 fix B. The committed family earnings panel spans income years
**1968-2022** (``family.FAMILY_WAVES``: annual 1968-1997, biennial
1999-2023; family-file income refers to the prior calendar year), NOT the
"1993-2022" the round-1 draft claimed in >=6 load-bearing places. Measured
on the committed supply: 69.0% of the 14,952 supply persons carry >=1
positive pre-1993 year (their pre-1993 nominal dollar share averages 68%);
observed positive years run median 13 / p90 28 / max 42 (a 1993-2022 window
would allow at most ~17). The partial-career truncation is therefore
COHORT-GRADED, not a blanket "levels are lower bounds": a 1940s-born ego is
observed near-fully (annual 1968-1996), a post-1970-born ego is the heavily
truncated one. The DATA is as committed; the round-1 description was false,
and every "1993-2022" statement is corrected to the true 1968-2022 surface.

The couple unit and the couple-disjoint split (load-bearing; fix A)
------------------------------------------------------------------
A **directed couple** is one marriage-history record whose spouse is a
joinable PSID sample-member person (``spouse_person_id`` populated, i.e.
``MH7`` in 1-9308 and ``MH8`` in 1-399) and where BOTH ego and spouse carry
a computable earnings axis. A marriage between two sample members yields two
directed records (ego A -> spouse B and ego B -> spouse A), so the tercile
contingency is symmetric by construction. Round-1 fix A: the noise floor
splits **couple-disjoint** by the connected component of the couple graph
(:func:`couple_components`), NOT by ego person_id. Under the round-1 ego
split, a couple whose two members split to opposite sides contributed one
directed record to EACH half (measured seed 0: 50.4% of mirrored pairs
straddled), so the two halves shared the very unit the tranche certifies --
understating the floor sigma 1.2-1.46x on the assortative diagonals and the
shared-earnings cells (where a straddling concordant couple's two records
land in the same cell on opposite sides). Splitting by component (13,275
components: 9,602 singletons, 3,519 pairs, 154 of size 3-4; every component
<= 4 persons) keeps both members of every couple on one side, so the halves
are genuinely independent. Person-level families (hazards, event windows)
are unaffected -- their observational units never straddled -- but the
single split object is component-disjoint everywhere so the floor split and
the candidate holdout commitment are the SAME object.

Statistic families
------------------
1. **assortative mating** -- own-tercile x spouse-tercile contingency shares
   over directed couples (``assort_mating.own{o}_spouse{u}``), on the
   per-year earnings axis; the within-couple earnings rank correlation
   (Spearman) with its construct-validity decomposition is reported
   alongside REPORT-ONLY (a correlation is not a scale-free log-rate);
2. **marriage hazards conditional on own earnings** --
   ``first_marriage_by_earnings.t{o}.{band}|{sex}`` (never-married
   person-years at risk) and ``remarriage_by_earnings.t{o}|{sex}``
   (post-dissolution person-years at risk), the tranche-2a marital hazards
   stratified by the person's fixed earnings tercile;
3. **earnings dynamics around marital events** --
   ``earnings_around_{marriage,divorce}.{all,female,male}``, the weighted
   median of each person's (post-event / pre-event) mean-earnings ratio over
   a +/-3-year window, DETRENDED by the age-agnostic placebo drift (fix E:
   NET of the non-event window drift deflator, so the gate certifies the
   event increment, not generic nominal life-cycle + inflation drift);
4. **shared-earnings distribution shape** --
   ``shared_earnings_ratio.q{hi}_q{lo}``, adjacent-quintile cutpoint ratios
   of the couple's combined earnings axis (own + spouse), a scale-free
   shape moment.

The gate-eligible / report-only split follows 2a/2b exactly (>= 20 events
on the weaker half of the worst seed, defined on every seed, stabilised
tolerance <= T_max = ln(1.5)). There are NO coverage-recovery aggregates:
unlike 2a/2b (whose moderate-rate families had a sparse older-age tail
whose adjacent bands ALL failed the cap, so pooling them recovered coverage
without masking), 2c's sparse cells are the late first-marriage-by-earnings
cells -- and no 35-44 pooled cell is standalone-gateable either (measured
100-seed tolerances 0.679-1.043, all above the cap; fix H record
correction), so the honest reason the sparse cells stay report-only is
POWER, not masking.

Weighting
---------
Each person carries the person-constant most-recent-positive PSID
cross-sectional weight (:func:`populace_dynamics.data.panels.
demographic_panel`), exactly tranche 2a's ``weight_definition`` -- for EVERY
gated family including the event windows (fix F: the round-1 draft used the
family-panel chronologically-last weight for the six event-window cells,
contradicting the stated definition; values move <= 0.0031 |ln|). There is
no unweighted gated statistic, and the unweighted rate is reported
alongside.
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
    "PLACEBO_MIN_EVENT_GAP",
    "MARRIAGE_DECADE_BANDS",
    "CoupleEarningsPanel",
    "aggregation_members",
    "person_earnings_histories",
    "indexed_earnings_supply",
    "career_aime_supply",
    "observed_positive_years",
    "couple_components",
    "placebo_drift_deflators",
    "build_couple_panel",
    "reference_moments",
    "assortative_correlation",
]

#: The two coded sexes (na sex is dropped), matching tranche 2a.
SEXES: tuple[str, ...] = ("female", "male")

#: Earnings-tercile labels (1 = lowest earnings third).
TERCILES: tuple[int, ...] = (1, 2, 3)
#: The earnings tercile cut quantiles (over the earnings-axis supply). Named
#: AIME_TERCILE_QUANTILES for continuity with the round-1 pin (fix G); the
#: axis is now the per-year indexed earnings measure, not the career AIME.
AIME_TERCILE_QUANTILES: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)

#: Shared-earnings (combined axis) quintile cut levels and the adjacent
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

#: Minimum observed positive-earnings years for a person's earnings axis to
#: be admitted (reduces the worst partial-career truncation noise in the
#: tercile assignment; disclosed).
MIN_EARNINGS_YEARS = 5

#: Half-width (years) of the pre/post window for the earnings-around-event
#: ratio. The ratio is (mean post-window earnings) / (mean pre-window
#: earnings) over observed positive-earnings years in each side.
EVENT_WINDOW_YEARS = 3

#: A placebo (non-event) window's centre year must sit at least this many
#: years from ANY of the person's marital events (fix E). The referee's
#: placebo convention; the placebo drift deflator detrends the event ratios.
PLACEBO_MIN_EVENT_GAP = 4

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
    docstring): pooling the sparse late-first-marriage age bands does not
    recover a standalone-gateable cell -- no pooled 35-44 tercile cell is
    gate-eligible either (measured 100-seed tolerances 0.679-1.043, all above
    T_max; fix H). The honest reason the sparse cells stay report-only is
    POWER. Returned empty so the build script and tests bind the deliberate
    absence."""
    return {}


@dataclass(frozen=True)
class CoupleEarningsPanel:
    """The marriage x earnings panel and its per-person attribute frame.

    Attributes:
        couples: One row per directed couple (marriage-history record with a
            joinable, both-axis spouse): ``person_id`` (ego),
            ``spouse_person_id``, ``sex``, ``weight``, ``earn_own``,
            ``earn_spouse`` (mean indexed positive-year earnings),
            ``own_tercile``, ``spouse_tercile``, ``shared`` (own + spouse
            axis), ``start_year``, ``start_decade``, and the REPORT-ONLY
            decomposition columns ``aime_proxy_own`` / ``aime_proxy_spouse``
            (the round-1 career-sum AIME proxy) and ``nobs_own`` /
            ``nobs_spouse`` (observed positive-earnings-year counts).
        marital_events: earnings-supply marital transition events
            (``first_marriage`` / ``remarriage``) with ``person_id``,
            ``age``, ``sex``, ``weight``, ``transition``, ``tercile``,
            ``fm_band``.
        marital_exposure: earnings-supply person-years with ``person_id``,
            ``sex``, ``weight``, ``marital_state``, ``tercile``, ``age``,
            ``fm_band`` (the hazard denominators).
        event_windows: One row per marital event with observed earnings
            support: ``person_id``, ``sex``, ``weight`` (the demographic
            most-recent-positive weight; fix F), ``event_type``
            (``marriage`` / ``divorce``), ``ratio`` (post/pre mean earnings).
        attrs: One row per person_id (``person_id`` + ``component_id``, the
            couple-graph connected component the couple-disjoint split draws
            on; fix A).
        earn_tercile_cuts: The committed earnings-axis tercile cut levels
            (t33, t67), fixed on the full earnings supply and applied to
            every half-split.
        placebo_deflators: The fix-E per-group (all / female / male)
            non-event drift deflators the event-window ratios are detrended
            by.
        meta: Provenance / coverage counts recorded in the artifact.
    """

    couples: pd.DataFrame
    marital_events: pd.DataFrame
    marital_exposure: pd.DataFrame
    event_windows: pd.DataFrame
    attrs: pd.DataFrame
    earn_tercile_cuts: tuple[float, float]
    placebo_deflators: dict[str, float]
    meta: dict[str, object]


# --------------------------------------------------------------------------
# Earnings histories + the earnings-axis supply
# --------------------------------------------------------------------------
def person_earnings_histories(
    panel: pd.DataFrame,
) -> tuple[dict[int, dict[int, float]], dict[int, int], dict[int, float]]:
    """Per-person ``{year: earnings}`` history, implied birth year, weight.

    Pure transform over a :func:`populace_dynamics.data.family.
    family_earnings_panel`-shaped frame (``person_id``, ``period``,
    ``earnings``, ``age``, ``weight``). Birth year is the median implied
    ``period - age`` (the R7 convention); the anchor weight is the person's
    chronologically last observed weight (retained for provenance; the gated
    statistics use the demographic most-recent-positive weight instead --
    fix F).
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


def _indexed_positive_years(
    hist: dict[int, float], birth: int, params: SSAParameters
) -> list[float]:
    """The NAWI-indexed values of a person's OBSERVED positive-earnings
    years (creditable-capped then indexed to the age-60 year), no zeros
    fill -- the per-year earnings axis draws on these."""
    creditable = benefits.creditable_history(hist, params)
    indexed = benefits.indexed_history(creditable, birth, params)
    return [indexed[y] for y in indexed if hist.get(y, 0.0) > 0.0]


def _admitted(
    hist: dict[int, float],
    birth: int | None,
    params: SSAParameters,
    min_nawi: int,
    max_nawi: int,
) -> bool:
    """Earnings-axis admission (unchanged from the round-1 draft): the
    age-60 indexing year lands in the NAWI series and at least
    :data:`MIN_EARNINGS_YEARS` positive-earnings years are observed."""
    if birth is None or not (min_nawi <= birth + 60 <= max_nawi):
        return False
    return sum(1 for v in hist.values() if v > 0) >= MIN_EARNINGS_YEARS


def indexed_earnings_supply(
    history: dict[int, dict[int, float]],
    birth_year: dict[int, int],
    params: SSAParameters,
) -> dict[int, float]:
    """The gate-2c earnings axis (fix C): each admitted person's MEAN indexed
    positive-year earnings -- the NAWI-indexed family-file labor income
    averaged over the person's observed positive-earnings years, a per-year
    earnings-capacity measure robust to the observed-years count (unlike the
    round-1 career-sum AIME proxy). Admission is unchanged (age-60 in the
    NAWI series and >= :data:`MIN_EARNINGS_YEARS` positive years), so the
    SELECTED universe is identical; only the scalar changed."""
    min_nawi, max_nawi = min(params.nawi), max(params.nawi)
    supply: dict[int, float] = {}
    for pid, hist in history.items():
        b = birth_year.get(pid)
        if not _admitted(hist, b, params, min_nawi, max_nawi):
            continue
        vals = _indexed_positive_years(hist, int(b), params)
        supply[pid] = float(np.mean(vals)) if vals else 0.0
    return supply


def career_aime_supply(
    history: dict[int, dict[int, float]],
    birth_year: dict[int, int],
    params: SSAParameters,
) -> dict[int, float]:
    """The round-1 career-sum AIME PROXY over the same admitted set --
    REPORT-ONLY, the decomposition baseline the referee's fix-C construct
    read is measured against (its within-couple sorting is dominated by
    observation mechanics; see :func:`assortative_correlation`)."""
    min_nawi, max_nawi = min(params.nawi), max(params.nawi)
    supply: dict[int, float] = {}
    for pid, hist in history.items():
        b = birth_year.get(pid)
        if not _admitted(hist, b, params, min_nawi, max_nawi):
            continue
        supply[pid] = float(benefits.aime(hist, int(b), params))
    return supply


def observed_positive_years(
    history: dict[int, dict[int, float]], supply_ids: Iterable[int]
) -> dict[int, int]:
    """Observed positive-earnings-year count per supply person (fix C
    decomposition: this count anti-correlates within couples)."""
    ids = set(supply_ids)
    return {
        pid: sum(1 for v in history[pid].values() if v > 0)
        for pid in ids
        if pid in history
    }


def couple_components(
    attrs_ids: Iterable[int], couples: pd.DataFrame
) -> dict[int, int]:
    """Fix A: map every split person to its couple-graph connected component
    (id = the minimum person_id in the component). Union-find over the
    directed-couple edges whose BOTH endpoints are split persons; a person in
    no bi-directional couple is its own singleton component. Splitting by
    component keeps both members of every couple on one side (couple-disjoint
    floor)."""
    ids = set(int(x) for x in attrs_ids)
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in zip(
        couples["person_id"].astype(int),
        couples["spouse_person_id"].astype(int),
        strict=False,
    ):
        if a in ids and b in ids:
            union(a, b)
    return {i: find(i) for i in ids}


def _tercile_of(value: float, cuts: tuple[float, float]) -> int:
    return 1 + int(value >= cuts[0]) + int(value >= cuts[1])


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------
def _build_couples(
    episodes: pd.DataFrame,
    earn: dict[int, float],
    aime_proxy: dict[int, float],
    nobs: dict[int, int],
    sex: dict[int, str],
    weight: dict[int, float],
    cuts: tuple[float, float],
) -> pd.DataFrame:
    """Directed both-axis couples with terciles, shared axis, decade, and the
    report-only decomposition columns."""
    ep = episodes[episodes["spouse_person_id"].notna()].copy()
    ep["spouse_person_id"] = ep["spouse_person_id"].astype(int)
    ep["earn_own"] = ep["person_id"].map(earn)
    ep["earn_spouse"] = ep["spouse_person_id"].map(earn)
    ep["sex"] = ep["person_id"].map(sex)
    ep["spouse_sex"] = ep["spouse_person_id"].map(sex)
    ep = ep[
        ep["earn_own"].notna()
        & ep["earn_spouse"].notna()
        & ep["sex"].isin(SEXES)
        & ep["start_year"].notna()
    ].copy()
    ep["weight"] = ep["person_id"].map(weight).fillna(0.0).astype(float)
    ep = ep[ep["weight"] > 0]
    ep["own_tercile"] = ep["earn_own"].map(lambda v: _tercile_of(v, cuts))
    ep["spouse_tercile"] = ep["earn_spouse"].map(
        lambda v: _tercile_of(v, cuts)
    )
    ep["shared"] = ep["earn_own"].astype(float) + ep["earn_spouse"].astype(
        float
    )
    ep["aime_proxy_own"] = ep["person_id"].map(aime_proxy)
    ep["aime_proxy_spouse"] = ep["spouse_person_id"].map(aime_proxy)
    ep["nobs_own"] = ep["person_id"].map(nobs)
    ep["nobs_spouse"] = ep["spouse_person_id"].map(nobs)
    ep["start_year"] = ep["start_year"].astype(int)
    ep["start_decade"] = (ep["start_year"] // 10 * 10).astype(int)
    return ep[
        [
            "person_id",
            "spouse_person_id",
            "sex",
            "spouse_sex",
            "weight",
            "earn_own",
            "earn_spouse",
            "own_tercile",
            "spouse_tercile",
            "shared",
            "start_year",
            "start_decade",
            "aime_proxy_own",
            "aime_proxy_spouse",
            "nobs_own",
            "nobs_spouse",
        ]
    ].reset_index(drop=True)


def _build_marital(
    marital: transitions.MaritalPanel,
    earn: dict[int, float],
    cuts: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earnings-supply marital events + exposure, tercile- and band-tagged."""
    supply = set(earn)
    tmap = {pid: _tercile_of(v, cuts) for pid, v in earn.items()}

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


def _event_years(episodes: pd.DataFrame) -> dict[int, set[int]]:
    """Per person, the set of marital-event years (marriage start_year and
    divorce episode_end_year) the placebo windows must avoid (fix E)."""
    out: dict[int, set[int]] = {}
    for row in episodes.itertuples(index=False):
        pid = int(row.person_id)
        years = out.setdefault(pid, set())
        if not pd.isna(row.start_year):
            years.add(int(row.start_year))
        if getattr(row, "how_ended", None) == "divorce" and not pd.isna(
            row.episode_end_year
        ):
            years.add(int(row.episode_end_year))
    return out


def _window_ratio(hist: dict[int, float], year: int, w: int) -> float | None:
    """The (mean post) / (mean pre) earnings ratio over a +/-w window, or
    None when either side has no observed positive-earnings year."""
    pre = [hist[y] for y in range(year - w, year) if hist.get(y, 0.0) > 0.0]
    post = [
        hist[y]
        for y in range(year + 1, year + w + 1)
        if hist.get(y, 0.0) > 0.0
    ]
    if not pre or not post:
        return None
    return float(np.mean(post) / np.mean(pre))


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
    post window ``[y+1, y+w]`` (``w`` = :data:`EVENT_WINDOW_YEARS`). The
    weight is the demographic most-recent-positive weight (fix F).
    """
    rows: list[dict[str, object]] = []
    plans = (
        ("marriage", "start_year", episodes),
        (
            "divorce",
            "episode_end_year",
            episodes[episodes["how_ended"] == "divorce"],
        ),
    )
    w = EVENT_WINDOW_YEARS
    for event_type, year_col, src in plans:
        sub = src[src[year_col].notna()]
        for row in sub.itertuples(index=False):
            pid = int(row.person_id)
            hist = history.get(pid)
            s = sex.get(pid)
            if hist is None or s not in SEXES:
                continue
            ratio = _window_ratio(hist, int(getattr(row, year_col)), w)
            if ratio is None:
                continue
            rows.append(
                {
                    "person_id": pid,
                    "sex": s,
                    "weight": float(weight.get(pid, 0.0)),
                    "event_type": event_type,
                    "ratio": ratio,
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=["person_id", "sex", "weight", "event_type", "ratio"],
    )
    return frame[frame["weight"] > 0].reset_index(drop=True)


def placebo_drift_deflators(
    episodes: pd.DataFrame,
    history: dict[int, dict[int, float]],
    sex: dict[int, str],
    weight: dict[int, float],
) -> tuple[dict[str, float], dict[str, int]]:
    """Fix E: the non-event drift deflators (all / female / male).

    Every qualifying non-event +/-``EVENT_WINDOW_YEARS`` window (centre year
    >= :data:`PLACEBO_MIN_EVENT_GAP` from ANY of the person's marital events,
    positive-earnings support in both sides) contributes its post/pre ratio,
    weighted by the person's demographic weight spread evenly over that
    person's windows (so each person contributes weight once). The deflator
    is the weighted median -- the generic nominal life-cycle + inflation
    drift a window on this panel shows with no event. The event-window cells
    are scored NET of this deflator, so a real-terms candidate is not failed
    by construction (the round-1 cells were ~77% generic drift). Returned as
    a fixed panel-level deflator (like a price index): applied identically to
    every half-split, so it cancels in the |ln ratio| floor and re-prices
    only the committed reference value.
    """
    ev_years = _event_years(episodes)
    w = EVENT_WINDOW_YEARS
    rows: list[tuple[str, float, float]] = []
    counts = {"all": 0, "female": 0, "male": 0}
    for pid, hist in history.items():
        s = sex.get(pid)
        if s not in SEXES:
            continue
        pw = float(weight.get(pid, 0.0))
        if pw <= 0:
            continue
        pos = [y for y, v in hist.items() if v > 0.0]
        if not pos:
            continue
        evs = ev_years.get(pid, set())
        ratios: list[float] = []
        for y0 in range(min(pos) + w, max(pos) - w + 1):
            if evs and min(abs(y0 - e) for e in evs) < PLACEBO_MIN_EVENT_GAP:
                continue
            r = _window_ratio(hist, y0, w)
            if r is not None:
                ratios.append(r)
        if not ratios:
            continue
        share = pw / len(ratios)
        for r in ratios:
            rows.append((s, share, r))
        counts["all"] += 1
        counts[s] += 1
    frame = pd.DataFrame(rows, columns=["sex", "weight", "ratio"])
    deflators: dict[str, float] = {}
    for group in ("all", "female", "male"):
        g = frame if group == "all" else frame[frame["sex"] == group]
        deflators[group] = (
            _weighted_median(
                g["ratio"].to_numpy(dtype=float),
                g["weight"].to_numpy(dtype=float),
            )
            if len(g)
            else 1.0
        )
    return deflators, counts


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
    demographic weights and the death records, computes the per-year earnings
    axis (fix C) from the certified :func:`populace_dynamics.ss.benefits`
    indexing chain (SSA parameters loaded via :func:`load_ssa_parameters`
    unless injected), and builds the directed-couple, earnings-supply-marital
    and (fix E placebo-deflated) earnings-window frames. Source frames may be
    injected for synthetic unit tests (then no PSID / pe-us read happens).
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

    history, birth_year, _earn_weight = person_earnings_histories(
        earnings_panel
    )
    earn = indexed_earnings_supply(history, birth_year, params)
    aime_proxy = career_aime_supply(history, birth_year, params)
    nobs = observed_positive_years(history, earn)
    cut_lo, cut_hi = (
        np.quantile(
            np.fromiter(earn.values(), dtype=float),
            AIME_TERCILE_QUANTILES,
        )
        if earn
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
    couples = _build_couples(episodes, earn, aime_proxy, nobs, sex, pw, cuts)

    marital = transitions.build_marital_panel(
        marriage_records, death_records, person_weight
    )
    marital_events, marital_exposure = _build_marital(marital, earn, cuts)

    # Fix F: event windows carry the demographic most-recent-positive weight.
    event_windows = _event_windows(episodes, history, sex, pw)
    deflators, placebo_counts = placebo_drift_deflators(
        episodes, history, sex, pw
    )

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
    comp = couple_components(attrs["person_id"], couples)
    attrs["component_id"] = attrs["person_id"].map(comp).astype(int)
    component_sizes = pd.Series(list(comp.values())).value_counts()
    size_hist = component_sizes.value_counts().to_dict()

    # Split-half rank reliability of the career-AIME proxy (fix C
    # decomposition): even/odd observed-year split, AIME on each half.
    reliability, spearman_brown = _split_half_reliability(
        history, birth_year, earn, params
    )

    n_joinable = int(episodes["spouse_person_id"].notna().sum())
    ego_supply = (
        episodes.loc[episodes["spouse_person_id"].notna(), "person_id"]
        .isin(earn)
        .sum()
    )
    meta = {
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "nawi_year_range": [int(min(params.nawi)), int(max(params.nawi))],
        "earnings_income_year_range": _income_year_range(history, earn),
        "n_earnings_supply_persons": len(earn),
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
        "n_couple_components": int(len(set(comp.values()))),
        "component_size_histogram": {
            str(int(k)): int(v) for k, v in sorted(size_hist.items())
        },
        "placebo_deflators": {k: round(v, 4) for k, v in deflators.items()},
        "placebo_person_counts": placebo_counts,
        "aime_proxy_split_half_reliability": reliability,
        "aime_proxy_spearman_brown": spearman_brown,
        "min_earnings_years": MIN_EARNINGS_YEARS,
        "event_window_years": EVENT_WINDOW_YEARS,
    }

    return CoupleEarningsPanel(
        couples=couples,
        marital_events=marital_events,
        marital_exposure=marital_exposure,
        event_windows=event_windows,
        attrs=attrs,
        earn_tercile_cuts=cuts,
        placebo_deflators=deflators,
        meta=meta,
    )


def _income_year_range(
    history: dict[int, dict[int, float]], supply_ids: Iterable[int]
) -> list[int] | None:
    years = [
        y
        for pid in set(supply_ids)
        for y, v in history.get(pid, {}).items()
        if v > 0
    ]
    return [int(min(years)), int(max(years))] if years else None


def _split_half_reliability(
    history: dict[int, dict[int, float]],
    birth_year: dict[int, int],
    supply: dict[int, float],
    params: SSAParameters,
) -> tuple[float | None, float | None]:
    """Even/odd observed-year split-half rank reliability of the career-AIME
    proxy (fix C decomposition): shows the weak career-AIME sorting is NOT
    sampling noise in the proxy."""
    a0: list[float] = []
    a1: list[float] = []
    for pid in supply:
        b = birth_year.get(pid)
        if b is None:
            continue
        years = sorted(history[pid])
        halves = []
        ok = True
        for which in (0, 1):
            sel = {
                y: history[pid][y]
                for i, y in enumerate(years)
                if i % 2 == which
            }
            if sum(1 for v in sel.values() if v > 0) < 1:
                ok = False
                break
            halves.append(float(benefits.aime(sel, int(b), params)))
        if ok:
            a0.append(halves[0])
            a1.append(halves[1])
    if len(a0) < 3:
        return None, None
    rho = float(pd.Series(a0).corr(pd.Series(a1), method="spearman"))
    sb = 2.0 * rho / (1.0 + rho) if rho > -1 else None
    return round(rho, 4), (round(sb, 4) if sb is not None else None)


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
    """First-marriage hazard by earnings tercile x age band x sex."""
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
    """Remarriage hazard by earnings tercile x sex (pooled over age)."""
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
    windows: pd.DataFrame,
    deflators: dict[str, float],
    *,
    weighted: bool,
) -> dict[str, dict[str, float]]:
    """Median (post/pre) earnings ratio around marriage and divorce, DETRENDED
    by the fix-E placebo drift deflator for the group (all / female / male).
    The deflator is a fixed panel-level constant, so it cancels in the
    half-split floor and re-prices only the reference value -- the gated
    quantity is the event increment net of generic nominal drift."""
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
            deflator = deflators.get(group, 1.0) or 1.0
            detrended = _weighted_median(ratios, w) / deflator
            cells[key] = _value_cell(detrended, float(w.sum()), int(len(g)))
    return cells


def _shared_earnings_cells(
    couples: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """Adjacent-quintile cutpoint ratios of the couple's combined axis."""
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
    placebo-detrended around-event earnings-ratio medians, and the
    shared-earnings cutpoint-ratio shape moments. ``person_ids=None`` gives
    the committed reference moments; passing each half of a couple-disjoint
    split (by ``component_id``) gives the noise-floor inputs. The earnings
    tercile cut levels and the placebo deflators are fixed on the full supply
    (``panel.earn_tercile_cuts`` / ``panel.placebo_deflators``), so a half's
    contingency, hazard categories and detrend do not shift.
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
    cells.update(
        _event_window_cells(
            windows, panel.placebo_deflators, weighted=weighted
        )
    )
    cells.update(_shared_earnings_cells(couples, weighted=weighted))
    return cells


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 3:
        return None
    rho = a.corr(b, method="spearman")
    return None if pd.isna(rho) else round(float(rho), 4)


def assortative_correlation(
    panel: CoupleEarningsPanel,
    person_ids: Iterable[int] | None = None,
) -> dict[str, object]:
    """REPORT-ONLY within-couple earnings rank correlation (Spearman), on the
    per-year earnings axis, with the fix-C construct-validity decomposition.

    Overall and by marriage-decade. A correlation is not a scale-free
    log-rate, so it is NEVER gated (the assortative-mating GATE runs on the
    tercile contingency shares); this is the interpretable scalar summary
    reported alongside. The decomposition (career-AIME proxy cross-sex,
    within-sex AIME rank, observed-years count, split-half reliability, and
    the career-AIME by-decade rhos) rides here so the referee's fix-C reading
    -- the JOINT sorts on earnings (rho ~0.49), not on the observation
    mechanics the career-sum proxy conflated (rho ~0.12) -- is auditable and
    the external-anchor bridge does not read as a contradiction.
    """
    couples = panel.couples
    if person_ids is not None:
        couples = couples[couples["person_id"].isin(set(person_ids))]

    by_decade = {}
    for decade in MARRIAGE_DECADE_BANDS:
        sub = couples[couples["start_decade"] == decade]
        by_decade[str(decade)] = {
            "n_directed_couples": int(len(sub)),
            "spearman_earnings": _spearman(
                sub["earn_own"], sub["earn_spouse"]
            ),
            "spearman_aime_proxy": _spearman(
                sub["aime_proxy_own"], sub["aime_proxy_spouse"]
            ),
        }

    # Within-sex AIME-proxy rank (each partner ranked within their OWN sex --
    # removes the cross-sex pooling artifact).
    within = couples.dropna(subset=["sex", "spouse_sex"]).copy()
    if len(within):
        within["r_own"] = within.groupby("sex")["aime_proxy_own"].rank()
        within["r_spouse"] = within.groupby("spouse_sex")[
            "aime_proxy_spouse"
        ].rank()
        within_sex_rho = _spearman(within["r_own"], within["r_spouse"])
    else:
        within_sex_rho = None

    decomposition = {
        "earnings_axis_spearman": _spearman(
            couples["earn_own"], couples["earn_spouse"]
        ),
        "career_aime_proxy_spearman_cross_sex": _spearman(
            couples["aime_proxy_own"], couples["aime_proxy_spouse"]
        ),
        "within_sex_aime_proxy_rank_spearman": within_sex_rho,
        "observed_positive_years_spearman": _spearman(
            couples["nobs_own"], couples["nobs_spouse"]
        ),
        "career_aime_proxy_split_half_reliability": panel.meta.get(
            "aime_proxy_split_half_reliability"
        ),
        "career_aime_proxy_spearman_brown": panel.meta.get(
            "aime_proxy_spearman_brown"
        ),
        "note": (
            "Fix C construct-validity decomposition. The gated earnings axis "
            "is the per-year mean indexed earnings (earnings_axis_spearman "
            "~0.49, literature-scale within-couple sorting). The round-1 "
            "career-sum AIME proxy correlated only ~0.12 cross-sex: cross-sex "
            "tercile pooling halves it (within-sex rank ~0.21) and the "
            "observed-positive-years count anti-correlates within couples "
            "(~ -0.10, single-earner-era head/wife role-years), which the "
            "zeros-filled career sum rewards. The proxy's split-half rank "
            "reliability ~0.98, so the weak career-AIME signal is NOT "
            "sampling noise -- it is what the career-sum proxy measures, "
            "which is why the axis is the per-year measure."
        ),
    }

    return {
        "gated": False,
        "report_only": True,
        "statistic": (
            "within-couple Spearman rank correlation of the per-year indexed "
            "earnings axis (with the career-AIME-proxy decomposition)"
        ),
        "overall": {
            "n_directed_couples": int(len(couples)),
            "spearman_earnings": _spearman(
                couples["earn_own"], couples["earn_spouse"]
            ),
            "spearman_aime_proxy": _spearman(
                couples["aime_proxy_own"], couples["aime_proxy_spouse"]
            ),
        },
        "by_marriage_decade": by_decade,
        "decomposition": decomposition,
        "note": (
            "REPORT-ONLY: a rank correlation is bounded in [-1, 1] and can "
            "cross zero, so it is not a scale-free |ln ratio| and is never "
            "gated. The assortative-mating GATE runs on the tercile "
            "contingency shares (assort_mating.own{o}_spouse{u}) over the "
            "per-year earnings axis; these rhos and the decomposition are the "
            "interpretable summary reported alongside."
        ),
    }
