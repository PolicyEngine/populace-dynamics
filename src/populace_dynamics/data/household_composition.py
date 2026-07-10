"""Household-composition moments from the PSID Family Relationship Matrix.

The gate-2b holdout basis (``gates.yaml`` gate_2b, ``holdout_basis:
[MX23REL]``) is household composition -- *who lives with whom*, and how it
changes wave to wave. This module turns the person-pair Family
Relationship Matrix (:mod:`populace_dynamics.data.relmap`) into a
person-wave household panel and the reference-moment cells the gate-2b
noise floor is measured on, mirroring the marital/fertility surface
:mod:`populace_dynamics.data.transitions` builds for tranche 2a.

Unit and join
-------------
The scored unit is the **person-wave**: one row per enumerated person per
PSID interview year, with that wave's household. Composition is read from
MX23REL and each person-wave is joined to its contemporaneous age and
wave weight (:func:`populace_dynamics.data.panels.demographic_panel`,
``ind2023er``) and to person-constant sex
(:func:`populace_dynamics.data.deaths.read_death_records`, ``ER32000``).
The person split that generates the floor
(:func:`populace_dynamics.harness.panel.split_panel_by_person`) is by
``person_id``, so every wave of a person -- and every wave-to-wave
transition -- lands on one side of the half-split.

Coresidence direction and the code frames (load-bearing)
--------------------------------------------------------
Two DIFFERENT relmap code frames are read, and confusing them inverts the
statistics:

* **Coresidence flags** read ``ego_rel_to_alter`` (``MX8``) -- *ego's*
  relationship *to* alter -- so the code names what EGO is in the pair:
  ego coded **child** (30/33/35/38) => coresident **parent**; ego coded
  **parent** (50/53/55/56) => coresident **child**; ego coded
  **spouse/partner** (20/22) => coresident **spouse**; ego coded
  **grandparent** (66/68/82/87/88) => coresident **grandchild**. Each set
  is own + step + social + foster kin and EXCLUDES in-law-by-marriage
  links (37/57/67/69), stated once here so the four families share one
  inclusion rule.
* **Generation offsets** (for multigen) read ``ego_rel_to_rp`` (``MX7``),
  a SEPARATE frame. Its code 88 is ``first_year_cohabitor`` (generation
  0), NOT the ``MX8`` meaning "step great-grandparent" (+2) -- see
  :data:`populace_dynamics.data.relmap.REL_TO_REFERENCE_PERSON_1983_PLUS`.
  :data:`_GEN_1983_PLUS` maps only codes valid in the MX7 frame.

Multigenerational definition
----------------------------
``multigen`` is **three or more distinct lineal generations present** in
the family unit (the Census B11017 concept), counted over lineal kin only
(collateral relatives -- siblings, nieces, cousins -- are not generations
apart from the reference person). Skipped-generation households
(grandparent + grandchild, no middle generation) are TWO distinct
generations and are NOT multigenerational, matching B11017.

Statistic families
------------------
Point-in-time **stocks** -- person-wave shares by
:data:`COMPOSITION_AGE_BANDS` x sex for ``coresident_spouse`` /
``coresident_parent`` / ``coresident_child`` / ``coresident_grandchild``
and ``multigen``, plus the person-level household-size distribution -- and
wave-to-wave **transitions** (the tranche name is "transitions"):
parental-home exit, spousal-coresidence loss, and multigenerational
entry/exit, measured between a person-wave and the next observed wave at
most :data:`MAX_TRANSITION_GAP_YEARS` years later (the biennial post-1997
PSID cadence; 1-year pre-1997). The sparse older-age tail of the
moderate-rate stock families is recovered by the pre-registered
:data:`AGGREGATION_BANDS` pools.

Weighting
---------
Each person-wave carries its **wave** individual weight (the
contemporaneous weight for a cross-sectional composition statistic, unlike
tranche 2a's person-constant most-recent weight for retrospective
histories); the unweighted rate is reported alongside. Person-waves with
no positive wave weight, no joined age, or ``na`` sex are dropped. Raw
PSID weights are population-scaled only from 1997 (sample-scaled before),
so the pooled weighted estimand is effectively 1997-2023 -- the build
script names this explicitly (``description_claims_exactly_the_scored_
surface``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from populace_dynamics.data import deaths, panels, relmap

__all__ = [
    "START_AGE",
    "MAX_AGE",
    "SEXES",
    "COMPOSITION_AGE_BANDS",
    "HH_SIZE_CLASSES",
    "HH_SIZE_TOP",
    "CORESIDENCE_LINKS",
    "AGGREGATION_BANDS",
    "MAX_TRANSITION_GAP_YEARS",
    "PARENTAL_EXIT_BANDS",
    "SPOUSAL_LOSS_BANDS",
    "HouseholdCompositionPanel",
    "band_label",
    "aggregation_members",
    "build_household_panel",
    "reference_moments",
]

#: Person-wave age universe (matches tranche 2a's START_AGE floor).
START_AGE = 15
MAX_AGE = 120
#: ``na`` sex is dropped; the panel scores the two coded sexes.
SEXES: tuple[str, ...] = ("female", "male")

#: Age bands for the composition shares (open-topped 75+).
COMPOSITION_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (15, 24),
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 120),
)

#: Household-size share classes; sizes at or above :data:`HH_SIZE_TOP`
#: pool into one open ``5+`` cell.
HH_SIZE_CLASSES: tuple[int, ...] = (1, 2, 3, 4)
HH_SIZE_TOP = 5

#: ``MX8`` (ego-to-alter) code sets. The flag name states the coresident
#: relation from EGO's side (see the module docstring): an ego who is a
#: *child* of an alter HAS a coresident parent, etc. Each set is own + step
#: + social + foster kin and EXCLUDES in-law-by-marriage links.
_SPOUSE_LINK = frozenset({20, 22})  # ego is spouse/partner of alter
_CHILD_LINK = frozenset({30, 33, 35, 38})  # ego is a child of alter
_PARENT_LINK = frozenset({50, 53, 55, 56})  # ego is a parent of alter
_GRANDPARENT_LINK = frozenset(
    {66, 68, 82, 87, 88}
)  # ego is a grandparent of alter (in-law 67/69 excluded)

#: Flag name -> the ``MX8`` code set whose presence among ego's alters
#: sets it. Ordered; fixes the emitted cell order.
CORESIDENCE_LINKS: dict[str, frozenset[int]] = {
    "coresident_spouse": _SPOUSE_LINK,
    "coresident_parent": _CHILD_LINK,
    "coresident_child": _PARENT_LINK,
    "coresident_grandchild": _GRANDPARENT_LINK,
}

#: Every per-band family (the coresidence links + ``multigen``); the order
#: the reference moments are emitted in.
_BAND_FAMILIES: tuple[str, ...] = (*CORESIDENCE_LINKS, "multigen")

#: Pre-registered coverage-recovery aggregate bands, fixed a priori BEFORE
#: the corrected-multigen rebuild (referee round 1, finding 6b): the sparse
#: older-age tail of the moderate-rate families. ``multigen`` pools 65+
#: (not 55+) so ``multigen.55-64`` can gate standalone rather than be
#: superseded. ``coresident_spouse``, ``coresident_child`` and ``hh_size``
#: carry no aggregate.
AGGREGATION_BANDS: dict[str, tuple[int, int]] = {
    "coresident_parent": (45, 120),
    "coresident_grandchild": (55, 120),
    "multigen": (65, 120),
}

#: Maximum year gap linking a person-wave to the "next observed wave" for a
#: transition: the biennial post-1997 PSID cadence (2 years), also
#: admitting the 1-year pre-1997 annual step. Attrition gaps beyond this
#: are not linked (a transition would otherwise span many years).
MAX_TRANSITION_GAP_YEARS = 2

#: Transition age bands (broader than the stock bands: transitions carry
#: smaller denominators).
PARENTAL_EXIT_BANDS: tuple[tuple[int, int], ...] = ((15, 24), (25, 34))
SPOUSAL_LOSS_BANDS: tuple[tuple[int, int], ...] = (
    (55, 64),
    (65, 74),
    (75, 120),
)
#: Composition flags whose next-wave value the transitions read.
_TRANSITION_SOURCE_FLAGS = (
    "coresident_parent",
    "coresident_spouse",
    "multigen",
)

#: ``MX7``/``MX12`` rel-to-reference-person code -> lineal generation
#: offset from the reference person, for the multigenerational count. ONLY
#: codes valid in the MX7 frame appear (see
#: :data:`populace_dynamics.data.relmap.REL_TO_REFERENCE_PERSON_1983_PLUS`);
#: MX8-only codes (61-64, 80-82, 87) are deliberately absent so the two
#: frames cannot silently cross. Only LINEAL kin carry an offset;
#: collateral relatives (sibling, niece, cousin, other, nonrelative) map to
#: ``None`` and are excluded from the generation count.
_GEN_1983_PLUS: dict[int, int] = {
    10: 0,  # reference person
    20: 0,  # legal spouse
    22: 0,  # partner
    88: 0,  # first-year cohabitor (MX7 frame; NOT MX8 step-great-grandparent)
    90: 0,  # uncooperative spouse
    92: 0,  # uncooperative partner
    30: -1,  # child
    33: -1,  # stepchild
    35: -1,  # child of partner
    37: -1,  # child-in-law
    38: -1,  # foster child
    83: -1,  # child of first-year cohabitor
    60: -2,  # grandchild
    65: -2,  # great-grandchild
    50: 1,  # parent
    57: 1,  # parent-in-law
    58: 1,  # parent of cohabitor
    66: 2,  # grandparent
    67: 2,  # grandparent of spouse
    68: 2,  # great-grandparent
    69: 2,  # great-grandparent of spouse
}
#: Pre-1983 abbreviated MX7 frame. It has no grandparent code, so a
#: household whose oldest member is the RP's grandparent is under-counted
#: before 1983 (a documented era asymmetry; negligible under the weighting,
#: which is ~99.8% post-1997).
_GEN_PRE1983: dict[int, int] = {
    1: 0,  # reference person
    2: 0,  # spouse or partner
    9: 0,  # husband of reference person
    4: 0,  # sibling
    3: -1,  # child
    6: -2,  # grandchild or great-grandchild
    5: 1,  # parent
}
_RP_ERA_BREAK = 1983
#: Census B11017 multigenerational: three or more distinct generations.
_MULTIGEN_MIN_GENERATIONS = 3


def band_label(lo: int, hi: int) -> str:
    """Flat band label, e.g. ``"25-34"`` or open-topped ``"75+"``."""
    return f"{lo}-{hi}" if hi < 120 else f"{lo}+"


def _band_of(age: int) -> str | None:
    for lo, hi in COMPOSITION_AGE_BANDS:
        if lo <= age <= hi:
            return band_label(lo, hi)
    return None


def aggregation_members() -> dict[str, list[str]]:
    """Map each pre-registered aggregate cell to its per-band members.

    ``coresident_parent.45+|female`` -> the four
    ``coresident_parent.{45-54,55-64,65-74,75+}|female`` per-band cells;
    ``multigen.65+|female`` -> the two ``multigen.{65-74,75+}|female``
    cells (55-64 excluded so it can gate standalone). The build script uses
    this to demote a gating aggregate's members to report-only; the tests
    bind the map.
    """
    out: dict[str, list[str]] = {}
    for flag, (lo, hi) in AGGREGATION_BANDS.items():
        agg = f"{flag}.{band_label(lo, hi)}"
        for sex in SEXES:
            members = [
                f"{flag}.{band_label(blo, bhi)}|{sex}"
                for blo, bhi in COMPOSITION_AGE_BANDS
                if blo >= lo and bhi <= hi
            ]
            out[f"{agg}|{sex}"] = members
    return out


@dataclass(frozen=True)
class HouseholdCompositionPanel:
    """The person-wave household panel and its per-person attribute frame.

    Attributes:
        person_waves: One row per enumerated person-wave with
            ``person_id``, ``year``, ``age``, ``band``, ``sex``,
            ``weight`` (that wave's weight), ``hh_size``, the
            ``coresident_*`` boolean flags, ``multigen``, and the
            transition columns ``has_next`` / ``next_coresident_parent`` /
            ``next_coresident_spouse`` / ``next_multigen``.
        attrs: One row per person (``person_id``), the split/holdout unit.
    """

    person_waves: pd.DataFrame
    attrs: pd.DataFrame


# --------------------------------------------------------------------------
# Pure construction (unit-testable on synthetic relmap-shaped frames)
# --------------------------------------------------------------------------
def _generation(code: int, year: int) -> int | None:
    frame = _GEN_PRE1983 if year < _RP_ERA_BREAK else _GEN_1983_PLUS
    return frame.get(int(code))


def household_roster(rel_map: pd.DataFrame) -> pd.DataFrame:
    """Per person-wave household roster from a relationship-map frame.

    Pure transform over a :func:`populace_dynamics.data.relmap.
    relationship_map`-shaped frame (unit-testable on synthetic rows).
    Returns one row per ``(interview_year, interview_number, person_id)``
    with the person's ``ego_rel_to_rp``, the household size (distinct
    enumerated members), the four ``coresident_*`` flags (from ego's
    ``MX8`` links to alters), and the ``multigen`` household flag (>= 3
    distinct lineal generations present).
    """
    key = ["interview_year", "interview_number", "ego_person_id"]
    hh_key = ["interview_year", "interview_number"]

    roster = rel_map.groupby(key, as_index=False).agg(
        ego_rel_to_rp=("ego_rel_to_rp", "first")
    )
    roster = roster.rename(columns={"ego_person_id": "person_id"})

    size = (
        rel_map.groupby(hh_key)["ego_person_id"]
        .nunique()
        .rename("hh_size")
        .reset_index()
    )
    roster = roster.merge(size, on=hh_key, how="left")

    nonself = rel_map[rel_map["ego_rel_to_alter"] != relmap.SELF]
    for flag, link_codes in CORESIDENCE_LINKS.items():
        hit = (
            nonself[nonself["ego_rel_to_alter"].isin(link_codes)]
            .groupby(key)
            .size()
            .rename("_n")
            .reset_index()
            .rename(columns={"ego_person_id": "person_id"})
        )
        roster = roster.merge(
            hit,
            on=["interview_year", "interview_number", "person_id"],
            how="left",
        )
        roster[flag] = roster["_n"].fillna(0) > 0
        roster = roster.drop(columns="_n")

    codes = roster["ego_rel_to_rp"].tolist()
    years = roster["interview_year"].tolist()
    gen = pd.Series(
        [_generation(c, y) for c, y in zip(codes, years, strict=True)],
        index=roster.index,
    )
    n_gen = (
        roster.assign(_gen=gen)
        .dropna(subset=["_gen"])
        .groupby(hh_key)["_gen"]
        .nunique()
        .rename("_n_gen")
        .reset_index()
    )
    roster = roster.merge(n_gen, on=hh_key, how="left")
    roster["multigen"] = (
        roster["_n_gen"].fillna(0) >= _MULTIGEN_MIN_GENERATIONS
    )
    return roster.drop(columns="_n_gen")


def _add_transitions(person_waves: pd.DataFrame) -> pd.DataFrame:
    """Add next-observed-wave state columns for the transition families.

    ``has_next`` marks a person-wave that has a following observed wave at
    a 1-2 year gap (:data:`MAX_TRANSITION_GAP_YEARS`); the ``next_*``
    columns carry that wave's composition flags. Pure transform.
    """
    pw = person_waves.sort_values(["person_id", "year"]).reset_index(drop=True)
    grp = pw.groupby("person_id", sort=False)
    next_year = grp["year"].shift(-1)
    gap = next_year - pw["year"]
    has_next = (gap >= 1) & (gap <= MAX_TRANSITION_GAP_YEARS)
    pw["has_next"] = has_next.fillna(False).to_numpy()
    for flag in _TRANSITION_SOURCE_FLAGS:
        pw[f"next_{flag}"] = grp[flag].shift(-1)
    return pw


def join_demographics(
    roster: pd.DataFrame,
    demo: pd.DataFrame,
    sex: pd.DataFrame,
) -> pd.DataFrame:
    """Join wave age/weight and person sex; keep usable adult person-waves.

    Pure transform (unit-testable): ``demo`` is a
    :func:`populace_dynamics.data.panels.demographic_panel`-shaped frame
    (``person_id``, ``period``, ``age``, ``weight``), ``sex`` a
    ``person_id`` -> ``sex`` frame. Keeps person-waves with a joined age in
    ``[START_AGE, MAX_AGE]``, a positive wave weight, and a coded sex, then
    adds the next-wave transition columns.
    """
    demo_j = demo[["person_id", "period", "age", "weight"]].rename(
        columns={"period": "interview_year"}
    )
    out = roster.merge(demo_j, on=["person_id", "interview_year"], how="left")
    out = out.merge(sex[["person_id", "sex"]], on="person_id", how="left")
    keep = (
        out["age"].notna()
        & (out["weight"].fillna(0) > 0)
        & out["sex"].isin(SEXES)
    )
    out = out.loc[keep].copy()
    out["age"] = out["age"].astype(int)
    out = out[(out["age"] >= START_AGE) & (out["age"] <= MAX_AGE)]
    out["weight"] = out["weight"].astype(float)
    out["band"] = out["age"].map(_band_of)
    out = out.rename(columns={"interview_year": "year"})
    cols = [
        "person_id",
        "year",
        "age",
        "band",
        "sex",
        "weight",
        "hh_size",
        *CORESIDENCE_LINKS,
        "multigen",
    ]
    out = out[cols].reset_index(drop=True)
    return _add_transitions(out)


def build_household_panel(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    rel_map: pd.DataFrame | None = None,
    demo: pd.DataFrame | None = None,
    sex: pd.DataFrame | None = None,
) -> HouseholdCompositionPanel:
    """Build the person-wave household panel from the staged PSID products.

    Reads MX23REL (:func:`populace_dynamics.data.relmap.relationship_map`),
    the demographic panel and the sex records, then assembles the panel via
    the pure :func:`household_roster` / :func:`join_demographics`
    transforms. The three source frames may be injected (synthetic tests).
    """
    if rel_map is None:
        rel_map = relmap.relationship_map(data_dir=data_dir, nrows=nrows)
    if demo is None:
        demo = panels.demographic_panel(data_dir=data_dir)
    if sex is None:
        sex = deaths.read_death_records(data_dir=data_dir)
    roster = household_roster(rel_map)
    person_waves = join_demographics(roster, demo, sex)
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    return HouseholdCompositionPanel(person_waves=person_waves, attrs=attrs)


# --------------------------------------------------------------------------
# Reference moments (mirror transitions.reference_moments' cell schema)
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


def _weight_series(frame: pd.DataFrame, weighted: bool) -> pd.Series:
    if weighted:
        return frame["weight"]
    return pd.Series(1.0, index=frame.index)


def _band_sex_share_cells(
    df: pd.DataFrame,
    hit: pd.Series,
    prefix: str,
    bands: tuple[tuple[int, int], ...],
    *,
    weighted: bool,
) -> dict[str, dict[str, float]]:
    """``prefix.<band>|<sex>`` weighted-share cells over a masked subframe.

    ``df`` is the exposure (denominator) frame already restricted to the
    at-risk rows; ``hit`` is the boolean event mask aligned to ``df``.
    """
    w = _weight_series(df, weighted)
    grp = pd.DataFrame(
        {
            "band": df["band"].to_numpy(),
            "sex": df["sex"].to_numpy(),
            "w": w.to_numpy(),
            "hit": hit.to_numpy(dtype=bool),
        }
    )
    den = grp.groupby(["band", "sex"])["w"].sum()
    num = (
        grp.assign(wn=grp["w"] * grp["hit"])
        .groupby(["band", "sex"])["wn"]
        .sum()
    )
    cnt = grp[grp["hit"]].groupby(["band", "sex"]).size()

    out: dict[str, dict[str, float]] = {}
    for lo, hi in bands:
        band = band_label(lo, hi)
        for sex in SEXES:
            gk = (band, sex)
            out[f"{prefix}.{band}|{sex}"] = _rate_cell(
                float(num.get(gk, 0.0)),
                float(den.get(gk, 0.0)),
                int(cnt.get(gk, 0)),
            )
    return out


def _stock_cells(
    person_waves: pd.DataFrame, flag: str, *, weighted: bool
) -> dict[str, dict[str, float]]:
    df = person_waves[person_waves["band"].notna()]
    return _band_sex_share_cells(
        df, df[flag], flag, COMPOSITION_AGE_BANDS, weighted=weighted
    )


def _hh_size_cells(
    person_waves: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """Person-level household-size share cells (sizes 1..4 and ``5+``)."""
    w = _weight_series(person_waves, weighted)
    total = float(w.sum())
    out: dict[str, dict[str, float]] = {}
    for size in HH_SIZE_CLASSES:
        mask = (person_waves["hh_size"] == size).to_numpy()
        out[f"hh_size.{size}"] = _rate_cell(
            float(w[mask].sum()), total, int(mask.sum())
        )
    mask = (person_waves["hh_size"] >= HH_SIZE_TOP).to_numpy()
    out[f"hh_size.{HH_SIZE_TOP}+"] = _rate_cell(
        float(w[mask].sum()), total, int(mask.sum())
    )
    return out


def _aggregate_cells(
    person_waves: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """The pre-registered older-age tail aggregate cells."""
    out: dict[str, dict[str, float]] = {}
    for flag, (lo, hi) in AGGREGATION_BANDS.items():
        band = band_label(lo, hi)
        for sex in SEXES:
            sub = person_waves[
                (person_waves["age"] >= lo)
                & (person_waves["age"] <= hi)
                & (person_waves["sex"] == sex)
            ]
            w = _weight_series(sub, weighted)
            hit = sub[flag].to_numpy(dtype=bool)
            out[f"{flag}.{band}|{sex}"] = _rate_cell(
                float(w.to_numpy()[hit].sum()),
                float(w.sum()),
                int(hit.sum()),
            )
    return out


def _transition_band_cells(
    person_waves: pd.DataFrame,
    at_risk_flag: str,
    next_flag: str,
    event_value: bool,
    prefix: str,
    bands: tuple[tuple[int, int], ...],
    *,
    weighted: bool,
) -> dict[str, dict[str, float]]:
    """A wave-to-wave transition family, by age band x sex.

    Exposure = person-waves with ``has_next`` and ``at_risk_flag`` True;
    event = the next wave's ``next_flag == event_value``.
    """
    df = person_waves[
        person_waves["has_next"]
        & person_waves[at_risk_flag]
        & person_waves["band"].notna()
    ]
    hit = df[f"next_{next_flag}"] == event_value
    return _band_sex_share_cells(df, hit, prefix, bands, weighted=weighted)


def _transition_overall_cell(
    person_waves: pd.DataFrame,
    at_risk_value: bool,
    event_value: bool,
    key: str,
    *,
    weighted: bool,
) -> dict[str, dict[str, float]]:
    """A single pooled (all ages x sexes) multigen-transition cell."""
    df = person_waves[
        person_waves["has_next"] & (person_waves["multigen"] == at_risk_value)
    ]
    w = _weight_series(df, weighted)
    hit = (df["next_multigen"] == event_value).to_numpy(dtype=bool)
    return {
        key: _rate_cell(
            float(w.to_numpy()[hit].sum()), float(w.sum()), int(hit.sum())
        )
    }


def _transition_cells(
    person_waves: pd.DataFrame, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """Every wave-to-wave transition cell (the tranche-name families)."""
    cells: dict[str, dict[str, float]] = {}
    cells.update(
        _transition_band_cells(
            person_waves,
            "coresident_parent",
            "coresident_parent",
            False,
            "parental_home_exit",
            PARENTAL_EXIT_BANDS,
            weighted=weighted,
        )
    )
    cells.update(
        _transition_band_cells(
            person_waves,
            "coresident_spouse",
            "coresident_spouse",
            False,
            "spousal_loss",
            SPOUSAL_LOSS_BANDS,
            weighted=weighted,
        )
    )
    cells.update(
        _transition_overall_cell(
            person_waves, False, True, "multigen_entry", weighted=weighted
        )
    )
    cells.update(
        _transition_overall_cell(
            person_waves, True, False, "multigen_exit", weighted=weighted
        )
    )
    return cells


def reference_moments(
    panel: HouseholdCompositionPanel,
    person_ids: Iterable[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Every gate-2b household-composition reference-moment cell.

    The union of the per-band x sex coresidence + multigen stock shares
    (:data:`_BAND_FAMILIES`), the person-level household-size shares, the
    pre-registered older-age tail aggregates, and the wave-to-wave
    transition families. Calling with ``person_ids=None`` gives the
    committed reference moments; calling on each half of a person-disjoint
    split gives the noise-floor inputs.
    """
    pw = panel.person_waves
    if person_ids is not None:
        pw = pw[pw["person_id"].isin(set(person_ids))]
    cells: dict[str, dict[str, float]] = {}
    for flag in _BAND_FAMILIES:
        cells.update(_stock_cells(pw, flag, weighted=weighted))
    cells.update(_hh_size_cells(pw, weighted=weighted))
    cells.update(_aggregate_cells(pw, weighted=weighted))
    cells.update(_transition_cells(pw, weighted=weighted))
    return cells
