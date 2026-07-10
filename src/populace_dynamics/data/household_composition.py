"""Household-composition moments from the PSID Family Relationship Matrix.

The gate-2b holdout basis (``gates.yaml`` gate_2b, ``holdout_basis:
[MX23REL]``) is household composition -- *who lives with whom*. This
module turns the person-pair Family Relationship Matrix
(:mod:`populace_dynamics.data.relmap`) into a person-wave household panel
and the reference-moment cells the gate-2b noise floor is measured on,
mirroring the marital/fertility surface :mod:`populace_dynamics.data.
transitions` builds for tranche 2a.

Unit and join
-------------
The scored unit is the **person-wave**: one row per enumerated person per
PSID interview year, with that wave's household. Composition is read from
MX23REL and each person-wave is joined to its contemporaneous age and
cross-sectional weight (:func:`populace_dynamics.data.panels.
demographic_panel`, ``ind2023er``) and to person-constant sex
(:func:`populace_dynamics.data.deaths.read_death_records`, ``ER32000``).
The person split that generates the floor (:func:`populace_dynamics.
harness.panel.split_panel_by_person`) is by ``person_id``, so every wave
of a person lands on one side of the half-split -- the same
person-disjoint discipline tranche 2a uses.

Coresidence direction (the load-bearing subtlety)
-------------------------------------------------
``ego_rel_to_alter`` (``MX8``) is *ego's* relationship *to* alter, so the
code names what EGO is in the pair, and coresidence is read from ego's
side:

* ego coded **child** (30/33/35/38) => ego has a coresident **parent**
  (``coresident_parent``);
* ego coded **parent** (50/53/55/56) => ego has a coresident **child**
  (``coresident_child``);
* ego coded **spouse/partner** (20/22) => coresident **spouse**
  (symmetric);
* ego coded **grandparent** (66/67/68/69/82/87/88) => coresident
  **grandchild** (``coresident_grandchild``).

Reading the code as the alter's role would invert every gradient (it
would score a 15-24-year-old as *having a child* when they live *with*
their parent); the person-wave rates in the committed floor are the
check.

Statistic families
------------------
Person-wave shares by :data:`COMPOSITION_AGE_BANDS` x sex for
``coresident_spouse`` / ``coresident_parent`` / ``coresident_child`` /
``coresident_grandchild`` and ``multigen`` (a 3+-lineal-generation
household), plus the person-level household-size share distribution
(:data:`HH_SIZE_CLASSES` and an open ``5+`` top). The sparse older-age
tail of the moderate-rate families is recovered by the pre-registered
:data:`AGGREGATION_BANDS` pools (the household analogue of tranche 2a's
``widowhood.45+`` aggregate), fixed a priori.

Weighting
---------
Each person-wave carries its **wave** cross-sectional individual weight
(the contemporaneous weight for a cross-sectional composition statistic,
unlike tranche 2a's person-constant most-recent weight for retrospective
histories); the unweighted rate is reported alongside. Person-waves with
no positive wave weight, no joined age, or ``na`` sex are dropped (the
repo ``weight>0`` convention).
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
#: *child* of an alter HAS a coresident parent, etc.
_SPOUSE_LINK = frozenset({20, 22})  # ego is spouse/partner of alter
_CHILD_LINK = frozenset({30, 33, 35, 38})  # ego is a child of alter
_PARENT_LINK = frozenset({50, 53, 55, 56})  # ego is a parent of alter
_GRANDPARENT_LINK = frozenset(
    {66, 67, 68, 69, 82, 87, 88}
)  # ego is a grandparent of alter

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

#: Pre-registered coverage-recovery aggregate bands (fixed a priori,
#: before the stabilised tolerances are seen): the sparse older-age tail
#: of the moderate-rate families, where the per-band tolerance exceeds the
#: ln(1.5) power cap. The household analogue of tranche 2a's
#: widowhood.45+/45-64 aggregates. ``coresident_spouse``,
#: ``coresident_child`` and ``hh_size`` carry no aggregate.
AGGREGATION_BANDS: dict[str, tuple[int, int]] = {
    "coresident_parent": (45, 120),
    "coresident_grandchild": (55, 120),
    "multigen": (55, 120),
}

#: ``MX7``/``MX12`` relation-to-reference-person code -> lineal generation
#: offset from the reference person, for multigenerational detection. Only
#: LINEAL kin carry an offset (collateral relatives -- sibling stays gen 0,
#: niece/uncle/cousin/other/nonrelative -> ``None``), so ``multigen`` is a
#: 3+-lineal-generation household in the Census sense. relmap's era split
#: gives two frames; both are mapped (the pre-1983 abbreviated frame has no
#: grandparent code, so a household whose oldest member is the RP's
#: grandparent is under-detected there -- a documented era asymmetry).
_GEN_1983_PLUS: dict[int, int] = {
    10: 0,  # reference person
    20: 0,  # legal spouse
    22: 0,  # partner
    90: 0,  # uncooperative spouse
    92: 0,  # uncooperative partner
    30: -1,  # child
    33: -1,  # stepchild
    35: -1,  # child of partner
    37: -1,  # child-in-law
    38: -1,  # foster child
    83: -1,  # child of first-year cohabitor
    60: -2,  # grandchild
    61: -2,
    62: -2,
    63: -2,
    64: -2,
    65: -2,  # great-grandchild
    80: -2,  # social grandchild
    81: -2,  # social great-grandchild
    50: 1,  # parent
    57: 1,  # parent-in-law
    58: 1,  # parent of cohabitor
    66: 2,  # grandparent
    67: 2,  # grandparent of spouse
    68: 2,  # great-grandparent
    69: 2,  # great-grandparent of spouse
    82: 2,  # social great-grandparent
    87: 2,  # step grandparent
    88: 2,  # step great-grandparent
}
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
_MULTIGEN_SPAN = 2


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
    ``coresident_parent.{45-54,55-64,65-74,75+}|female`` per-band cells,
    etc. The build script uses this to demote a gating aggregate's members
    to report-only (supersession), and the tests bind the map.
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
            ``weight`` (that wave's cross-sectional weight), ``hh_size``,
            the ``coresident_*`` boolean flags, and ``multigen``.
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
    relationship_map`-shaped frame (so it is unit-testable on synthetic
    rows). Returns one row per ``(interview_year, interview_number,
    person_id)`` with the person's ``ego_rel_to_rp``, the household size
    (distinct enumerated members in the ``interview_number``), the four
    ``coresident_*`` flags (from ego's ``MX8`` links to alters), and the
    ``multigen`` household flag (lineal-generation span >= 2).
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
    for flag, codes in CORESIDENCE_LINKS.items():
        hit = (
            nonself[nonself["ego_rel_to_alter"].isin(codes)]
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
    span = (
        roster.assign(_gen=gen)
        .dropna(subset=["_gen"])
        .groupby(hh_key)["_gen"]
        .agg(lambda s: s.max() - s.min())
        .rename("_span")
        .reset_index()
    )
    roster = roster.merge(span, on=hh_key, how="left")
    roster["multigen"] = roster["_span"].fillna(0) >= _MULTIGEN_SPAN
    return roster.drop(columns="_span")


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
    ``[START_AGE, MAX_AGE]``, a positive wave weight, and a coded sex.
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
    return out[cols].reset_index(drop=True)


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


def _band_sex_cells(
    person_waves: pd.DataFrame, flag: str, *, weighted: bool
) -> dict[str, dict[str, float]]:
    """One ``flag.<band>|<sex>`` cell per band x sex (weighted share)."""
    df = person_waves[person_waves["band"].notna()]
    w = _weight_series(df, weighted)
    grp = pd.DataFrame(
        {
            "band": df["band"].to_numpy(),
            "sex": df["sex"].to_numpy(),
            "w": w.to_numpy(),
            "hit": df[flag].to_numpy(dtype=bool),
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
    for lo, hi in COMPOSITION_AGE_BANDS:
        band = band_label(lo, hi)
        for sex in SEXES:
            gk = (band, sex)
            out[f"{flag}.{band}|{sex}"] = _rate_cell(
                float(num.get(gk, 0.0)),
                float(den.get(gk, 0.0)),
                int(cnt.get(gk, 0)),
            )
    return out


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


def reference_moments(
    panel: HouseholdCompositionPanel,
    person_ids: Iterable[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Every gate-2b household-composition reference-moment cell.

    The union of the per-band x sex coresidence + multigen shares
    (:data:`_BAND_FAMILIES`), the person-level household-size shares, and
    the pre-registered older-age tail aggregates. Calling with
    ``person_ids=None`` gives the committed reference moments; calling on
    each half of a person-disjoint split gives the noise-floor inputs
    (exactly the tranche-2a pattern).
    """
    pw = panel.person_waves
    if person_ids is not None:
        pw = pw[pw["person_id"].isin(set(person_ids))]
    cells: dict[str, dict[str, float]] = {}
    for flag in _BAND_FAMILIES:
        cells.update(_band_sex_cells(pw, flag, weighted=weighted))
    cells.update(_hh_size_cells(pw, weighted=weighted))
    cells.update(_aggregate_cells(pw, weighted=weighted))
    return cells
