"""Gate-2b candidate 3: custodial conditioning, household bridge, skipped-gen.

Candidate 3 (registration #42 comment 4939960467) is candidate 2
(:mod:`populace_dynamics.models.household_composition_sim_v2`, merged in PR
#134) with EXACTLY THREE frozen deltas, each feeding a DISJOINT family the
candidate-2 grading (comment 4939958136) isolated. Everything candidate 2
cleared or carried -- the certified tranche-2a marital core and maternal
births, the parental-home exit hazard, the multigenerational entry/exit
machinery, ``coresident_parent``, the cohabitation (MX8 code-22) overlay
unioned into ``coresident_spouse``, the observed cah85_23 father->child links,
the retained unlinked shadow kernel, and the household-size composition rule --
is carried BYTE-FAITHFULLY (imported and re-run with candidate 2's exact draw
streams, not copied), and the three deltas draw from a SEPARATE
``SeedSequence([draw_seed, 0xC3])`` isolated from candidate 1's occupancy
stream (``0xB2B``) and candidate 2's delta streams (``0xC2``).

Delta 1 -- **custodial paternal conditioning** (targets ``coresident_child``
male overshoot). Candidate 2 attributes every cah85_23 father->child
biological birth as a coresident child, aging it out under the parental-home
hazard; but biological paternity over-represents coresidence -- non-custodial
fathers do not live with their children (the OVERSHOOT the grading isolated).
Candidate 3 instead lets a father-linked child count as the man's coresident
child in a wave only with a train-fitted probability ``P(linked child
coresident with father | child age band x father marital state)``, drawn per
wave per draw. The probability is estimated on train father->child links whose
child is a joinable PSID person (so coresidence is observable in MX23REL: the
father is coded a parent, MX8 {50,53,55,56}, of that child that wave) and
applied to candidate 2's full linked-child set. The MATERNAL side and the
unlinked shadow kernel are UNTOUCHED (byte-identical to candidate 2).

Delta 2 -- **household bridge** (targets ``hh_size`` 3/4/5+). The gate-2b
reference ``hh_size`` counts every enumerated household member; the composed
ego-centric family unit (``1 + spouse + children + parents``) misses the
non-family coresidents (siblings, other relatives, roomers, ...). Candidate 3
fits the train distribution of the NON-FAMILY member count -- enumerated
``hh_size`` minus the ego's own ``1 + spouse-links + child-links +
parent-links`` -- binned ``0 / 1 / 2+`` by ego age band x sex, samples it per
draw, and adds it ONLY to the ``hh_size`` composition. It feeds no coresidence
family cell (a leak into any coresidence cell is a spec violation).

Delta 3 -- **skipped-generation coresidence** (targets
``coresident_grandchild`` 55+). The locked reference EXCLUDES
skipped-generation households (grandparent + grandchild, no middle generation)
from ``multigen`` (two generations, not three -- the Census B11017 concept),
so the composed grandchild (``multigen AND coresident_child AND NOT
coresident_parent``) misses grandparents raising grandchildren alone.
Candidate 3 fits train entry/exit hazards by age band x sex for the observed
skipped-generation state (``coresident_grandchild AND NOT multigen``), evolves
it from each holdout person's observed initial state, and UNIONS it into the
composed grandchild concept ONLY -- explicitly NOT into ``multigen`` (which
stays byte-identical; the regression risk is named).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import relmap, transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition_sim as hcs
from populace_dynamics.models import household_composition_sim_v2 as hcs2
from populace_dynamics.models.family_transitions.components.fertility import (
    build_fertility_lookup,
)

__all__ = [
    "CUSTODIAL_CHILD_AGE_BANDS",
    "CHILD_CORESIDENCE_MAX_AGE",
    "NONFAMILY_CLASSES",
    "HouseholdCompositionModelV3",
    "father_link_births_with_child",
    "parent_child_coresidence_pairs",
    "family_unit_sizes",
    "fit_custodial_coresidence",
    "fit_nonfamily_distribution",
    "attach_skipgen",
    "fit_skipgen_rates",
    "custodial_linked_child_counts",
    "fit_household_model_v3",
    "simulate_draw_v3",
]

#: Child age bands (in years) for the custodial coresidence probability,
#: fixed a priori to the coresidence life-cycle (infancy/childhood ->
#: adolescence -> young adult -> adult), NOT tuned to the holdout. A linked
#: child older than :data:`CHILD_CORESIDENCE_MAX_AGE` is never counted (the
#: candidate-1 open top for the child roster).
CUSTODIAL_CHILD_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 12),
    (13, 17),
    (18, 24),
    (25, 60),
)
#: The oldest age at which a father-linked child can be a coresident child
#: (mirrors candidate 1's ``CHILD_MAX_LEAVE_AGE`` open top).
CHILD_CORESIDENCE_MAX_AGE = hcs.CHILD_MAX_LEAVE_AGE  # == 60

#: Non-family household-member count classes (delta 2): 0, 1, and the open
#: ``2+`` tail. The registration names exactly this ``0 / 1 / 2+`` support;
#: the ``2+`` class contributes the minimal guaranteed count (2) to hh_size,
#: so the deep composition tail (hh_size.5+) stays the named residual.
NONFAMILY_CLASSES: tuple[str, ...] = ("0", "1", "2+")
_NONFAMILY_CONTRIB = {"0": 0, "1": 1, "2+": 2}

#: Two-state father marital classification for the custodial conditioning:
#: the certified marital core's ``married`` state vs everything else. The
#: salient custodial axis (the grading named the maternal-custodial norm and
#: non-coresident fathers); guarantees dense support in both cells across
#: every child age band.
_MARRIED = "married"
_NOT_MARRIED = "not_married"


def _child_band(age: int) -> str | None:
    """Label the custodial child age band for a child age, or ``None``."""
    for lo, hi in CUSTODIAL_CHILD_AGE_BANDS:
        if lo <= age <= hi:
            return hc.band_label(lo, hi)
    return None


def _marital_binary(state: pd.Series | np.ndarray) -> np.ndarray:
    """Binarize a marital-state series to ``married`` / ``not_married``."""
    s = pd.Series(state)
    return np.where(s.to_numpy() == "married", _MARRIED, _NOT_MARRIED)


@dataclass
class HouseholdCompositionModelV3:
    """Candidate 2's fitted bundle plus the three candidate-3 delta components.

    ``base_v2`` is the byte-faithful candidate-2
    :class:`~populace_dynamics.models.household_composition_sim_v2.
    HouseholdCompositionModelV2` (which itself carries candidate 1); the three
    delta components are all train-fitted (side B only):

    * ``custodial`` -- ``P(linked child coresident | child age band x father
      marital state)`` (delta 1).
    * ``nonfamily`` -- ``P(non-family member count in {0, 1, 2+} | ego age
      band x sex)`` (delta 2).
    * ``skipgen_entry`` / ``skipgen_exit`` -- the skipped-generation
      grandchild entry / exit hazards by age band x sex (delta 3).

    ``father_links_child`` (father, child, birth-year links whose child is a
    joinable PSID person) and ``parent_pairs`` (observed father->child
    coresidence pairs) are seed-independent data attributes reused across the
    per-seed custodial fit.
    """

    base_v2: hcs2.HouseholdCompositionModelV2
    custodial: dict[tuple[str, str], float]
    nonfamily: dict[tuple[str, str], tuple[float, float, float]]
    skipgen_entry: dict[tuple[str, str], float]
    skipgen_exit: dict[tuple[str, str], float]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def base(self) -> hcs.HouseholdCompositionModel:
        """The candidate-1 base bundle (carried through candidate 2)."""
        return self.base_v2.base

    @property
    def father_links(self) -> pd.DataFrame:
        """Candidate 2's (parent_person_id, birth_year) father->child links."""
        return self.base_v2.father_links


# --------------------------------------------------------------------------
# Seed-independent delta inputs (resolved once; the per-seed fits filter them)
# --------------------------------------------------------------------------
def father_link_births_with_child(birth_records: pd.DataFrame) -> pd.DataFrame:
    """Observed cah85_23 father->child links whose child is a PSID person.

    Like :func:`hcs2.father_link_births` but KEEPS the joinable
    ``child_person_id`` (needed to observe, in MX23REL, whether the specific
    child coresides with the father). One row per male-parent biological
    birth with a joinable child id and a birth year.
    """
    from populace_dynamics.data import births

    ev = births.birth_events(birth_records)
    fa = ev[
        (ev["parent_sex"] == "male")
        & (ev["record_type"] == "birth")
        & ev["birth_year"].notna()
        & ev["child_person_id"].notna()
    ]
    return pd.DataFrame(
        {
            "parent_person_id": fa["parent_person_id"].astype("int64"),
            "child_person_id": fa["child_person_id"].astype("int64"),
            "birth_year": fa["birth_year"].astype("int64"),
        }
    ).reset_index(drop=True)


def parent_child_coresidence_pairs(rel_map: pd.DataFrame) -> pd.DataFrame:
    """Observed (father, child, year) coresidence pairs from MX23REL.

    Every ordered pair where ego is coded a PARENT of alter (MX8
    ``{50, 53, 55, 56}`` -- :data:`hc.CORESIDENCE_LINKS['coresident_child']`,
    the exact inclusion rule the reference ``coresident_child`` uses), i.e.
    the alter is a coresident child of ego that wave. Deduplicated to
    ``(year, parent_person_id, child_person_id)``.
    """
    codes = hc.CORESIDENCE_LINKS["coresident_child"]
    links = rel_map[rel_map["ego_rel_to_alter"].isin(codes)]
    out = links[["interview_year", "ego_person_id", "alter_person_id"]].rename(
        columns={
            "interview_year": "year",
            "ego_person_id": "parent_person_id",
            "alter_person_id": "child_person_id",
        }
    )
    return out.drop_duplicates().reset_index(drop=True)


def family_unit_sizes(rel_map: pd.DataFrame) -> pd.DataFrame:
    """Per (person_id, year), the ego-centric nuclear family-unit size.

    ``1 + (# spouse/partner links) + (# child links) + (# parent links)`` --
    the same ego-side MX8 link families the household composition rule
    composes (:data:`hc.CORESIDENCE_LINKS`). The reference ``hh_size`` counts
    every enumerated member, so ``hh_size - family_unit_size`` is the count of
    NON-family household members (siblings, other relatives, roomers, ...) the
    composed family unit structurally omits -- the delta-2 bridge target.
    """
    nonself = rel_map[rel_map["ego_rel_to_alter"] != relmap.SELF]
    fam_codes = (
        hc.CORESIDENCE_LINKS["coresident_spouse"]
        | hc.CORESIDENCE_LINKS["coresident_child"]
        | hc.CORESIDENCE_LINKS["coresident_parent"]
    )
    fam = nonself[nonself["ego_rel_to_alter"].isin(fam_codes)]
    counts = (
        fam.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("_n_fam")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    counts["family_unit_size"] = counts["_n_fam"].astype("int64") + 1
    return counts[["person_id", "year", "family_unit_size"]]


# --------------------------------------------------------------------------
# Delta 1 fit: custodial coresidence probability
# --------------------------------------------------------------------------
def _father_marital_by_year(mpanel: transitions.MaritalPanel) -> pd.DataFrame:
    """Per (person_id, year) observed marital binary for the custodial fit."""
    py = mpanel.person_years[["person_id", "year", "marital_state"]].copy()
    py["marital"] = _marital_binary(py["marital_state"])
    return py[["person_id", "year", "marital"]]


def fit_custodial_coresidence(
    person_waves: pd.DataFrame,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Train ``P(linked child coresident with father | age band x marital)``.

    Exposure: each train father->child link paired with every father wave in
    which the child's age lands in a custodial band. Event: the (father,
    child, year) pair is observed coresident (in ``parent_pairs``). Weighted
    by the father-wave weight; empty (band, marital) strata fall back to the
    pooled rate.
    """
    fl = father_links_child[
        father_links_child["parent_person_id"].isin(train_ids)
    ]
    father_waves = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "weight"]
    ].rename(columns={"person_id": "parent_person_id"})
    # (father, child, birth_year) x (father wave) -> child age per wave.
    expo = fl.merge(father_waves, on="parent_person_id", how="inner")
    if not len(expo):
        return {}, {"n_exposure": 0, "n_coresident": 0}
    expo["child_age"] = expo["year"] - expo["birth_year"]
    expo = expo[
        (expo["child_age"] >= 0)
        & (expo["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    expo["child_band"] = expo["child_age"].map(_child_band)
    expo = expo[expo["child_band"].notna()]
    # Father marital state that wave (observed); uncovered -> not_married.
    expo = expo.merge(
        marital_by_year.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    expo["marital"] = expo["marital"].fillna(_NOT_MARRIED)
    # Coresidence event: the specific (father, child, year) pair is observed.
    pairs = parent_pairs.rename(
        columns={"parent_person_id": "parent_person_id"}
    )
    pairs = pairs.assign(_cores=True)
    expo = expo.merge(
        pairs,
        on=["parent_person_id", "child_person_id", "year"],
        how="left",
    )
    expo["coresident"] = expo["_cores"].fillna(False).astype(bool)

    w = expo["weight"].to_numpy(dtype=np.float64)
    ev = expo["coresident"].to_numpy(dtype=np.float64)
    overall = float((w * ev).sum() / w.sum()) if w.sum() > 0 else 0.0
    custodial: dict[tuple[str, str], float] = {}
    for lo, hi in CUSTODIAL_CHILD_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for marital in (_MARRIED, _NOT_MARRIED):
            sub = expo[
                (expo["child_band"] == band) & (expo["marital"] == marital)
            ]
            if len(sub):
                sw = sub["weight"].to_numpy(dtype=np.float64)
                se = sub["coresident"].to_numpy(dtype=np.float64)
                custodial[(band, marital)] = (
                    float((sw * se).sum() / sw.sum()) if sw.sum() > 0 else 0.0
                )
            else:
                custodial[(band, marital)] = overall
    diag = {
        "n_exposure": int(len(expo)),
        "n_coresident": int(expo["coresident"].sum()),
        "overall_coresidence_rate": round(overall, 5),
    }
    return custodial, diag


# --------------------------------------------------------------------------
# Delta 2 fit: non-family household-member distribution
# --------------------------------------------------------------------------
def fit_nonfamily_distribution(
    person_waves: pd.DataFrame,
    fu_sizes: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[tuple[str, str], tuple[float, float, float]], dict[str, Any]]:
    """Train ``P(non-family member count in {0,1,2+} | age band x sex)``.

    The non-family count is enumerated ``hh_size`` minus the ego family-unit
    size (clipped at 0). Weighted category shares per (band, sex); empty
    strata fall back to the pooled distribution.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "band", "sex", "weight", "hh_size"]
    ].merge(fu_sizes, on=["person_id", "year"], how="left")
    pw["family_unit_size"] = pw["family_unit_size"].fillna(1).astype("int64")
    resid = pw["hh_size"].to_numpy() - pw["family_unit_size"].to_numpy()
    resid = np.clip(resid, 0, None)
    pw = pw.assign(
        _cls=np.where(resid == 0, "0", np.where(resid == 1, "1", "2+"))
    )

    def _dist(frame: pd.DataFrame) -> tuple[float, float, float]:
        w = frame["weight"].to_numpy(dtype=np.float64)
        tot = float(w.sum())
        if tot <= 0:
            return (1.0, 0.0, 0.0)
        return tuple(
            float(w[(frame["_cls"] == c).to_numpy()].sum() / tot)
            for c in NONFAMILY_CLASSES
        )  # type: ignore[return-value]

    overall = _dist(pw)
    nonfamily: dict[tuple[str, str], tuple[float, float, float]] = {}
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            sub = pw[(pw["band"] == band) & (pw["sex"] == sex)]
            nonfamily[(band, sex)] = _dist(sub) if len(sub) else overall
    w = pw["weight"].to_numpy(dtype=np.float64)
    mean_resid = float((w * resid).sum() / w.sum()) if w.sum() > 0 else 0.0
    diag = {
        "n_person_waves": int(len(pw)),
        "overall_p0_p1_p2plus": [round(x, 5) for x in overall],
        "weighted_mean_nonfamily_count": round(mean_resid, 5),
    }
    return nonfamily, diag


# --------------------------------------------------------------------------
# Delta 3 fit: skipped-generation grandchild occupancy
# --------------------------------------------------------------------------
def attach_skipgen(person_waves: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``person_waves`` with ``skipgen`` + next state.

    ``skipgen`` is the observed skipped-generation grandchild state --
    ``coresident_grandchild AND NOT multigen`` -- the grandparent+grandchild
    (no middle generation) coresidence the locked reference excludes from
    ``multigen`` and the composed grandchild misses. ``next_skipgen`` is the
    following observed wave's state, aligned within person.
    """
    pw = person_waves.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw["skipgen"] = pw["coresident_grandchild"].to_numpy(dtype=bool) & ~pw[
        "multigen"
    ].to_numpy(dtype=bool)
    pw["next_skipgen"] = pw.groupby("person_id", sort=False)["skipgen"].shift(
        -1
    )
    return pw


def fit_skipgen_rates(
    train_pw: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Train skipped-generation entry / exit hazards by age band x sex.

    Identical construction to :func:`hcs.fit_multigen_rates`, on the
    ``skipgen`` / ``next_skipgen`` columns.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry_pool = has_next[~has_next["skipgen"]]
    exit_pool = has_next[has_next["skipgen"]]
    entry_overall = hcs._weighted_rate(
        entry_pool, entry_pool["next_skipgen"].to_numpy(dtype=np.float64)
    )
    exit_overall = hcs._weighted_rate(
        exit_pool,
        exit_pool["next_skipgen"].eq(False).to_numpy(dtype=np.float64),
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
                hcs._weighted_rate(
                    e, e["next_skipgen"].to_numpy(dtype=np.float64)
                )
                if len(e)
                else entry_overall
            )
            exit_[(band, sex)] = (
                hcs._weighted_rate(
                    x,
                    x["next_skipgen"].eq(False).to_numpy(dtype=np.float64),
                )
                if len(x)
                else exit_overall
            )
    return entry, exit_


# --------------------------------------------------------------------------
# Fit (train / side B only for every fitted parameter)
# --------------------------------------------------------------------------
def fit_household_model_v3(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    marriage_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    marriage_order_map: pd.DataFrame,
    rel_map: pd.DataFrame,
    train_ids: set[int],
    *,
    father_links_child: pd.DataFrame | None = None,
    parent_pairs: pd.DataFrame | None = None,
    fu_sizes: pd.DataFrame | None = None,
) -> HouseholdCompositionModelV3:
    """Fit candidate 2 (byte-faithful) plus the three candidate-3 deltas.

    Every fitted parameter is estimated on the train complement (side B): the
    candidate-2 bundle via :func:`hcs2.fit_household_model_v2`, the custodial
    coresidence probability (delta 1), the non-family distribution (delta 2)
    and the skipped-generation hazards (delta 3). The seed-independent link /
    coresidence / family-unit frames may be injected (resolved once by the
    run script for reuse across seeds).
    """
    base_v2 = hcs2.fit_household_model_v2(
        hh,
        mpanel,
        demo,
        marriage_records,
        birth_records,
        marriage_order_map,
        rel_map,
        train_ids,
    )

    if father_links_child is None:
        father_links_child = father_link_births_with_child(birth_records)
    if parent_pairs is None:
        parent_pairs = parent_child_coresidence_pairs(rel_map)
    if fu_sizes is None:
        fu_sizes = family_unit_sizes(rel_map)

    marital_by_year = _father_marital_by_year(mpanel)
    custodial, cust_diag = fit_custodial_coresidence(
        hh.person_waves,
        father_links_child,
        parent_pairs,
        marital_by_year,
        train_ids,
    )
    nonfamily, nf_diag = fit_nonfamily_distribution(
        hh.person_waves, fu_sizes, train_ids
    )
    skip_pw = attach_skipgen(hh.person_waves)
    train_skip = skip_pw[skip_pw["person_id"].isin(train_ids)]
    skipgen_entry, skipgen_exit = fit_skipgen_rates(train_skip)

    meta = {
        **base_v2.meta,
        "custodial_n_exposure": cust_diag["n_exposure"],
        "custodial_n_coresident": cust_diag["n_coresident"],
        "custodial_overall_rate": cust_diag.get("overall_coresidence_rate"),
        "nonfamily_overall_p0_p1_p2plus": nf_diag["overall_p0_p1_p2plus"],
        "nonfamily_weighted_mean_count": nf_diag[
            "weighted_mean_nonfamily_count"
        ],
        "skipgen_entry_overall": round(
            float(np.mean(list(skipgen_entry.values()))), 6
        ),
        "skipgen_exit_overall": round(
            float(np.mean(list(skipgen_exit.values()))), 6
        ),
        "skipgen_train_person_waves": int(train_skip["skipgen"].sum()),
    }
    return HouseholdCompositionModelV3(
        base_v2=base_v2,
        custodial=custodial,
        nonfamily=nonfamily,
        skipgen_entry=skipgen_entry,
        skipgen_exit=skipgen_exit,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Delta 1 apply: custodial per-wave coresidence of the linked children
# --------------------------------------------------------------------------
def custodial_linked_child_counts(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    custodial: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Per side-A person-wave, the count of coresident father-linked children.

    Each (father, linked-child) pair is coresident in a father wave with the
    train-fitted probability ``custodial[child band, father marital]``, drawn
    per wave. Returns an int array aligned to ``side_a_pw`` row order (the
    father-linked contribution to the coresident-child count; maternal and
    unlinked-shadow children are counted elsewhere).
    """
    counts = np.zeros(len(side_a_pw), dtype=np.int64)
    if not len(linked_births):
        return counts
    pw = side_a_pw.reset_index(drop=True)
    pw = pw.assign(_row=np.arange(len(pw), dtype=np.int64))
    fw = pw[["person_id", "year", "_row"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    expo = linked_births.merge(fw, on="parent_person_id", how="inner")
    if not len(expo):
        return counts
    expo["child_age"] = expo["year"] - expo["birth_year"]
    expo = expo[
        (expo["child_age"] >= 0)
        & (expo["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    expo["child_band"] = expo["child_age"].map(_child_band)
    expo = expo[expo["child_band"].notna()]
    if not len(expo):
        return counts
    expo = expo.merge(
        marital_sim.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    expo["marital"] = expo["marital"].fillna(_NOT_MARRIED)
    # Deterministic row order for a reproducible draw stream.
    expo = expo.sort_values(["parent_person_id", "birth_year", "_row"])
    expo = expo.reset_index(drop=True)
    prob = np.array(
        [
            custodial.get((b, m), 0.0)
            for b, m in zip(
                expo["child_band"].to_numpy(),
                expo["marital"].to_numpy(),
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    u = rng.random(len(expo))
    coresident = u < prob
    rows = expo["_row"].to_numpy()[coresident]
    np.add.at(counts, rows, 1)
    return counts


def _sim_marital_binary(
    sim_years: pd.DataFrame, side_a_pw: pd.DataFrame
) -> pd.DataFrame:
    """Per side-A (person_id, year), the SIMULATED father marital binary.

    From the certified simulation's ``marital_state``; waves the simulation
    does not cover fall back to the observed coresident-spouse union state
    (``married`` if spouse-coresident) then ``not_married``.
    """
    sim = sim_years[["person_id", "year", "marital_state"]].copy()
    sim["marital"] = _marital_binary(sim["marital_state"])
    out = side_a_pw[["person_id", "year"]].merge(
        sim[["person_id", "year", "marital"]],
        on=["person_id", "year"],
        how="left",
    )
    return out[["person_id", "year", "marital"]]


# --------------------------------------------------------------------------
# Delta 2 apply: sampled non-family member count -> hh_size only
# --------------------------------------------------------------------------
def _sample_nonfamily(
    side_a_pw: pd.DataFrame,
    nonfamily: dict[tuple[str, str], tuple[float, float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a non-family member count per side-A person-wave (added to hh)."""
    n = len(side_a_pw)
    bands = side_a_pw["band"].to_numpy(dtype=object)
    sexes = side_a_pw["sex"].to_numpy()
    p1 = np.zeros(n, dtype=np.float64)
    p2 = np.zeros(n, dtype=np.float64)
    for (band, sex), (_q0, q1, q2) in nonfamily.items():
        mask = (bands == band) & (sexes == sex)
        p1[mask] = q1
        p2[mask] = q2
    u = rng.random(n)
    # class 0 if u < q0; class 1 if u < q0 + q1; else class 2+.
    q0 = 1.0 - p1 - p2
    contrib = np.where(
        u < q0, 0, np.where(u < q0 + p1, 1, _NONFAMILY_CONTRIB["2+"])
    )
    return contrib.astype(np.int64)


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def simulate_draw_v3(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModelV3,
    ids_a: set[int],
    draw_seed: int,
    occupancy_stream_tag: int = 0xB2B,
    delta_stream_tag_v2: int = 0xC2,
    delta_stream_tag_v3: int = 0xC3,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-3 draw of the side-A holdout households.

    Steps 1-5 REPRODUCE candidate 2's draw byte-for-byte -- candidate 1's
    :func:`hcs.simulate_draw` at the same ``draw_seed`` and occupancy tag
    ``0xB2B`` (carried ``coresident_parent`` / ``multigen``), the candidate-2
    cohabitation overlay and father-link / maternal / shadow child roster on
    the same ``0xC2`` streams -- so every carried family is identical to
    candidate 2. The three candidate-3 deltas then draw from a separate
    ``SeedSequence([draw_seed, 0xC3])`` (spawned into custodial / non-family /
    skipped-generation streams) and recompose only their target families.
    """
    base = model.base

    # 1. carried parent / multigen / legal-marriage spouse (candidate 1,
    #    byte-identical; occupancy tag 0xB2B).
    c1_panel = hcs.simulate_draw(
        hh, mpanel, base, ids_a, draw_seed, occupancy_stream_tag
    )
    carried = c1_panel.person_waves[
        [
            "person_id",
            "year",
            "coresident_parent",
            "multigen",
            "coresident_spouse",
        ]
    ]

    side_a_pw = (
        hh.person_waves[hh.person_waves["person_id"].isin(ids_a)]
        .sort_values(["person_id", "year"])
        .reset_index(drop=True)
    )
    side_a_pw = side_a_pw.merge(
        model.base_v2.cohab_flag, on=["person_id", "year"], how="left"
    )
    side_a_pw["cohabiting"] = (
        side_a_pw["cohabiting"].fillna(False).astype(bool)
    )
    aligned = side_a_pw[["person_id", "year"]].merge(
        carried, on=["person_id", "year"], how="left"
    )
    c1_parent = aligned["coresident_parent"].to_numpy(dtype=bool)
    c1_multigen = aligned["multigen"].to_numpy(dtype=bool)
    legal_spouse = aligned["coresident_spouse"].to_numpy(dtype=bool)

    # 2. candidate-2 delta substreams (0xC2), consumed EXACTLY as candidate 2.
    delta_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v2])
    child_ss, cohab_ss = delta_ss.spawn(2)
    child_rng = np.random.default_rng(child_ss)
    cohab_rng = np.random.default_rng(cohab_ss)

    # 3. cohabitation (code-22) overlay unioned into coresident_spouse.
    mats = hcs._padded_person_matrices(side_a_pw)
    pw = mats["pw"]
    row_of = mats["row_of"]
    n_persons, max_waves = mats["n_persons"], mats["max_waves"]
    valid = row_of >= 0
    safe_row = np.where(valid, row_of, 0)
    sex_mat = pw["sex"].to_numpy()[safe_row]
    band_mat = pw["band"].to_numpy(dtype=object)[safe_row]
    obs_cohab = pw["cohabiting"].to_numpy(dtype=bool)[safe_row] & valid
    cohab_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    cohab_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.base_v2.cohab_entry.items():
        cohab_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in model.base_v2.cohab_exit.items():
        cohab_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    cohab_state = hcs._evolve_two_state(
        valid, obs_cohab, cohab_entry_prob, cohab_exit_prob, cohab_rng
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]
    spouse = legal_spouse | cohab_row

    # 4. certified marital core + maternal births (same draw seed as cand 2).
    sim_panel, sim_births = ft.simulate(
        mpanel, ids_a, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"

    # 5. linked / shadow paternal children (candidate-2 child stream 0xC2).
    men_a = mpanel.attrs[
        (mpanel.attrs["sex"] == "male")
        & (mpanel.attrs["person_id"].isin(ids_a))
    ]
    men_ids = set(int(x) for x in men_a["person_id"])
    linked_ids = (
        set(int(x) for x in model.father_links["parent_person_id"]) & men_ids
    )
    paternal_linked = model.father_links[
        model.father_links["parent_person_id"].isin(linked_ids)
    ][["parent_person_id", "birth_year"]].copy()
    paternal_linked["parent_person_id"] = paternal_linked[
        "parent_person_id"
    ].astype("int64")
    paternal_linked["birth_year"] = paternal_linked["birth_year"].astype(
        "int64"
    )
    paternal_linked["_source"] = "linked"
    unlinked_men = men_a[~men_a["person_id"].isin(linked_ids)]
    lookup, decade_map = build_fertility_lookup(
        base.family_transitions.fertility
    )
    paternal_shadow = hcs._paternal_births(
        sim_years, unlinked_men, base.male_gap, lookup, decade_map, child_rng
    )
    paternal_shadow["_source"] = "shadow"

    # Pool + age out on the candidate-2 child stream -- byte-identical draws
    # for the maternal + shadow children (their leave-year windows carry).
    all_births = pd.concat(
        [maternal, paternal_linked, paternal_shadow], ignore_index=True
    )
    all_births = all_births[all_births["parent_person_id"].isin(ids_a)]
    child_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    nonlinked_leaves = child_leaves[child_leaves["_source"] != "linked"]
    child_counts_nonlinked = hcs._coresident_child_counts(
        nonlinked_leaves, side_a_pw
    )

    # 6. candidate-3 delta substreams (0xC3), isolated from 0xB2B / 0xC2.
    c3_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v3])
    custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    custodial_rng = np.random.default_rng(custodial_ss)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # Delta 1: custodial per-wave coresidence of the linked children.
    marital_sim = _sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked = custodial_linked_child_counts(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model.custodial,
        custodial_rng,
    )
    child_counts = child_counts_nonlinked + child_counts_linked

    # compose child / grandchild / hh_size from the byte-identical carried
    # states plus the delta-1 child counts.
    coresident_child, grandchild_composed, hh_size_base = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

    # Delta 3: skipped-generation grandchild occupancy, unioned into
    # coresident_grandchild ONLY (NOT into multigen).
    obs_skipgen = (
        pw["coresident_grandchild"].to_numpy(dtype=bool)[safe_row]
        & ~pw["multigen"].to_numpy(dtype=bool)[safe_row]
        & valid
    )
    skip_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    skip_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.skipgen_entry.items():
        skip_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in model.skipgen_exit.items():
        skip_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    skipgen_state = hcs._evolve_two_state(
        valid, obs_skipgen, skip_entry_prob, skip_exit_prob, skipgen_rng
    )
    skipgen_row = np.zeros(len(pw), dtype=bool)
    skipgen_row[row_of[valid]] = skipgen_state[valid]
    coresident_grandchild = grandchild_composed | skipgen_row

    # Delta 2: sampled non-family member count added to hh_size ONLY.
    nonfamily_count = _sample_nonfamily(pw, model.nonfamily, nonfamily_rng)
    hh_size = (hh_size_base.astype(np.int64) + nonfamily_count).astype(
        np.int64
    )

    sim_pw = pw.copy()
    sim_pw["coresident_spouse"] = spouse
    sim_pw["coresident_parent"] = c1_parent
    sim_pw["coresident_child"] = coresident_child
    sim_pw["coresident_grandchild"] = coresident_grandchild
    sim_pw["multigen"] = c1_multigen
    sim_pw["hh_size"] = hh_size
    sim_pw = sim_pw.drop(
        columns=[
            "has_next",
            "next_coresident_parent",
            "next_coresident_spouse",
            "next_multigen",
            "cohabiting",
        ]
    )
    sim_pw = hc._add_transitions(sim_pw)
    attrs = sim_pw[["person_id"]].drop_duplicates().reset_index(drop=True)
    panel = hc.HouseholdCompositionPanel(person_waves=sim_pw, attrs=attrs)

    diagnostics = {
        "n_maternal_births": int(len(maternal)),
        "n_paternal_linked_births": int(len(paternal_linked)),
        "n_paternal_shadow_births": int(len(paternal_shadow)),
        "n_side_a_men": len(men_ids),
        "n_linked_fathers_side_a": len(linked_ids),
        "n_cohab_person_waves_simulated": int(cohab_row[row_of[valid]].sum()),
        "n_linked_child_coresident_wave_units": int(child_counts_linked.sum()),
        "n_skipgen_person_waves_simulated": int(
            skipgen_row[row_of[valid]].sum()
        ),
        "mean_nonfamily_count_simulated": float(nonfamily_count.mean()),
    }
    return panel, diagnostics
