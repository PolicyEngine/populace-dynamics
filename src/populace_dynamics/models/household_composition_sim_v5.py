"""Gate-2b candidate 5: coupling, the not-married custodial cell, bridge reach.

Candidate 5 (registration #42 comment 4945159933) is candidate 4
(:mod:`populace_dynamics.models.household_composition_sim_v4`, merged in PR
#138) with EXACTLY THREE frozen deltas, each designed against a corrected
mechanism the gate-2b forensics-2 decomposition
(``runs/gate2b_forensics2_v1.json``, grading 4945156926) measured. Everything
candidate 4 cleared or carried -- the certified tranche-2a marital core, the
carried ``coresident_parent`` / ``multigen`` (stock + transitions) /
``parental_home_exit``, AND the candidate-4 ``coresident_spouse`` family (the
legal-marriage registry, the age-refined cohabitation overlay and the
legal-spouse residual overlay) -- is carried BYTE-FAITHFULLY: candidate 5
reproduces candidate 4's draw exactly (candidate 1's ``simulate_draw`` at
``0xB2B``; the candidate-2 streams at ``0xC2``; the candidate-3 streams at
``0xC3``; the candidate-4 legal-residual / non-family-2+ streams at ``0xC4``),
then the three candidate-5 deltas either (i) re-fit an existing hazard /
probability table whose per-wave draw consumes RNG BY SHAPE ONLY -- so the
stream stays byte-identical and only the affected state changes -- or (ii) add a
genuinely new stochastic component on an ISOLATED
``SeedSequence([draw_seed, 0xC5])``.

Delta 1 -- **multigen--adult-child coupling** (``coresident_grandchild``
55+|female, forensics-2 Q6 channel a). The reference COUPLES the multigen state
and a coresident own adult child into the SAME household fact (train joint
``multigen AND child AND NOT-parent`` ~0.0384, FAR above the independence
product ~0.0077); the candidate-4 simulation draws multigen (the carried
candidate-1 hazard) and a coresident child (aged-out custodial / maternal) from
SEPARATE components so their joint collapses to the product ~0.0063. Candidate 5
replaces, FOR 55+ EGOS ONLY, the independent coresident-own-adult-child input to
the composed grandchild with a train-fitted JOINT draw --
``P(coresident own adult child | multigen state, age band, sex)`` applied to the
SIMULATED multigen occupancy on the isolated ``0xC5`` stream -- so the simulated
joint reproduces the reference coupling. SPEC CONSTRAINT (load-bearing): the
multigen occupancy MARGINAL is UNCHANGED -- multigen comes off candidate 1's
``simulate_draw`` unchanged and the coupling only READS it; the coupling
reallocates WHICH multigen egos carry the adult child, not how many egos are
multigen, so every multigen stock / transition cell stays byte-identical to
candidate 4. The coupled indicator feeds the grandchild composition ONLY (never
``coresident_child`` or ``hh_size`` or ``multigen``), exactly as candidate 4's
skip-gen occupancy is unioned into the grandchild alone.

Delta 2 -- **not-married custodial correction** (``coresident_child`` male
cells, forensics-2 Q5). Forensics-2 measured the observable (father, child,
wave) basis OVER-stating coresidence for NOT-married fathers by ~0.096 at school
ages against the less-selected child-record basis (denominator = the child's
OWN enumerated waves). Candidate 5 replaces, for NOT-married linked fathers
only, the observable-basis custodial probability with the child-record-basis
rate by child age band; the young MARRIED gate -- forensics-2 measured FAITHFUL
(the gap REVERSES: child-record 0.966 vs observable 0.916) -- is UNTOUCHED. The
per-child custodial draw is unchanged in shape (byte-identical ``0xC3`` custodial
stream); only the not-married probability moves.

Delta 3 -- **bridge reach + parent_count composition** (``hh_size``,
forensics-2 Q7). The 0.088 size-3 core-vs-actual gap splits EXACTLY into a
non-core-member part ~0.051 (reference size-3 CORES that are truly 4+ households
-- a sibling / roomer present) and a composition part ~0.037. (a) The non-family
bridge's incidence is re-fit CONDITIONAL ON CORE SIZE from train, so size-3
family cores are lifted upward at the train rate (the ``0xC3`` non-family class
draw is unchanged in shape; only its per-core thresholds move). (b) The
``parent_count=2`` assumption -- every coresident-parent ego adds exactly two
parents, forcing the over-produced three-adults route and foreclosing
couple+parent -- is corrected to the train coresident-parent-count composition
(1 vs 2 parents) drawn per ego on the isolated ``0xC5`` stream. Both feed
``hh_size`` ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition_sim as hcs
from populace_dynamics.models import household_composition_sim_v2 as hcs2
from populace_dynamics.models import household_composition_sim_v3 as hcs3
from populace_dynamics.models import household_composition_sim_v4 as hcs4

__all__ = [
    "GRANDCHILD_LO",
    "CORE_SIZE_CAP",
    "COUPLING_AGE_BANDS_55PLUS",
    "HouseholdCompositionModelV5",
    "grandchild_age_band",
    "fit_multigen_child_coupling",
    "build_child_record_exposure",
    "fit_custodial_child_record",
    "fit_nonfamily_by_core",
    "parent_link_counts",
    "fit_parent_count_composition",
    "fit_household_model_v5",
    "custodial_prob_v5",
    "custodial_linked_child_counts_v5",
    "sample_nonfamily_v5",
    "route_of_size3",
    "simulate_draw_v5",
    "coupling_and_gap_checks",
]

#: Delta 1 couples the multigen state and a coresident own adult child for egos
#: at or above this age (the forensics-2 grandchild 55+ channel-a threshold).
GRANDCHILD_LO = 55

#: Delta 3a caps the family-core size the non-family bridge conditions on (a
#: core of 5 or more shares the ``5+`` incidence); the salient cell is core 3.
CORE_SIZE_CAP = 5

#: Delta 1 fits ``P(coresident own child | multigen, band, sex)`` on these
#: composition bands within 55+ (the gate cell is the pooled 55+; the finer age
#: structure keeps the joint dense). A (band, sex, multigen) stratum thinner
#: than :data:`_MIN_STRATUM_N` at-risk waves falls back to the pooled 55+
#: (sex, multigen) rate (candidate convention: dense support or fall back).
COUPLING_AGE_BANDS_55PLUS: tuple[tuple[int, int], ...] = tuple(
    (lo, hi) for lo, hi in hc.COMPOSITION_AGE_BANDS if lo >= GRANDCHILD_LO
)

#: Candidate-5 isolated RNG tag (the two new stochastic components: the coupled
#: adult-child draw and the per-ego parent-count draw). Isolated from
#: ``0xB2B`` / ``0xC2`` / ``0xC3`` / ``0xC4`` so every carried stream is
#: byte-identical to candidate 4.
DELTA_STREAM_TAG_V5 = 0xC5

#: Minimum weighted-observation count for a fitted stratum to be used directly;
#: sparser strata fall back to the pooled marginal (candidate convention).
_MIN_STRATUM_N = hcs4._MIN_STRATUM_N  # 20

_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED


def grandchild_age_band(age: int) -> tuple[int, int] | None:
    """Return the 55+ composition band for an age, or ``None`` if below 55."""
    for lo, hi in COUPLING_AGE_BANDS_55PLUS:
        if lo <= age <= hi:
            return (lo, hi)
    return None


@dataclass
class HouseholdCompositionModelV5:
    """Candidate 4's byte-faithful bundle plus the three candidate-5 deltas.

    ``base_v4`` is the byte-faithful candidate-4
    :class:`~populace_dynamics.models.household_composition_sim_v4.
    HouseholdCompositionModelV4` (which carries candidates 3, 2 and 1). The
    three delta components are all train-fitted (side B only):

    * ``coupling_child_given_multigen`` -- ``P(coresident own child | multigen
      state, 55+ band, sex)`` with a pooled 55+ (sex, multigen) fallback baked
      in for every 55+ band (delta 1).
    * ``coupling_child_pooled`` -- the pooled 55+ (sex, multigen) rates (the
      fallback, recorded for the joint-vs-product check).
    * ``custodial_child_record`` -- the CHILD-RECORD-basis
      ``P(coresident with father | child age band, father marital)`` (delta 2);
      only the ``not_married`` cells are applied (the married gate is carried).
    * ``nonfamily_by_core`` -- ``P(non-family count in {0,1,2+} | core size,
      band, sex)`` for the DENSE (core, band, sex) cells (delta 3a); sparse
      cells fall back to candidate 4's (band, sex) table.
    * ``parent_count_two_share`` -- ``P(two coresident parents | coresident
      parent, band, sex)`` with a pooled fallback baked in (delta 3b).
    """

    base_v4: hcs4.HouseholdCompositionModelV4
    coupling_child_given_multigen: dict[tuple[str, str, bool], float]
    coupling_child_pooled: dict[tuple[str, bool], float]
    custodial_child_record: dict[tuple[str, str], float]
    nonfamily_by_core: dict[tuple[int, str, str], tuple[float, float, float]]
    parent_count_two_share: dict[tuple[str, str], float]
    parent_count_two_pooled: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def base_v3(self) -> hcs3.HouseholdCompositionModelV3:
        return self.base_v4.base_v3

    @property
    def base_v2(self) -> hcs2.HouseholdCompositionModelV2:
        return self.base_v4.base_v2

    @property
    def base(self) -> hcs.HouseholdCompositionModel:
        return self.base_v4.base

    @property
    def father_links(self) -> pd.DataFrame:
        return self.base_v4.father_links


# --------------------------------------------------------------------------
# Delta 1 fit: multigen -- adult-child coupling (P(child | multigen, band, sex))
# --------------------------------------------------------------------------
def fit_multigen_child_coupling(
    person_waves: pd.DataFrame,
    train_ids: set[int],
) -> tuple[
    dict[tuple[str, str, bool], float],
    dict[tuple[str, bool], float],
    dict[str, Any],
]:
    """Train ``P(coresident own child | multigen state, 55+ band, sex)``.

    Weighted, on the raw side-B roster among 55+ person-waves. The reference
    COUPLES the multigen state and a coresident own child (a multigen 55+
    grandmother almost always has the middle-generation adult child present --
    that IS why she is multigen), so this conditional is far above the marginal
    coresident-child rate for ``multigen=True``. A (band, sex, multigen)
    stratum thinner than :data:`_MIN_STRATUM_N` waves falls back to the pooled
    55+ (sex, multigen) rate; the fallback is baked into every 55+ band so the
    apply lookup is total. Returns (table, pooled, diag).
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)]
    older = pw[pw["age"] >= GRANDCHILD_LO]
    pooled: dict[tuple[str, bool], float] = {}
    for sex in hc.SEXES:
        for mg in (False, True):
            sub = older[(older["sex"] == sex) & (older["multigen"] == mg)]
            pooled[(sex, mg)] = hcs._weighted_rate(
                sub, sub["coresident_child"].to_numpy(np.float64)
            )
    table: dict[tuple[str, str, bool], float] = {}
    for lo, hi in COUPLING_AGE_BANDS_55PLUS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            for mg in (False, True):
                sub = older[
                    (older["band"] == band)
                    & (older["sex"] == sex)
                    & (older["multigen"] == mg)
                ]
                table[(band, sex, mg)] = (
                    hcs._weighted_rate(
                        sub, sub["coresident_child"].to_numpy(np.float64)
                    )
                    if len(sub) >= _MIN_STRATUM_N
                    else pooled[(sex, mg)]
                )
    diag = {
        "p_child_given_multigen_true_female_by_band": {
            hc.band_label(lo, hi): round(
                table[(hc.band_label(lo, hi), "female", True)], 5
            )
            for lo, hi in COUPLING_AGE_BANDS_55PLUS
        },
        "pooled_p_child_given_multigen_true_female": round(
            pooled[("female", True)], 5
        ),
        "pooled_p_child_given_multigen_false_female": round(
            pooled[("female", False)], 5
        ),
        "n_train_55plus_female_waves": int(
            len(older[older["sex"] == "female"])
        ),
    }
    return table, pooled, diag


# --------------------------------------------------------------------------
# Delta 2 fit: child-record-basis custodial coresidence (not-married cells)
# --------------------------------------------------------------------------
def build_child_record_exposure(
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    demo: pd.DataFrame,
    rel_map: pd.DataFrame,
) -> pd.DataFrame:
    """Seed-independent child-record custodial exposure (forensics-2 Q5 basis).

    One row per joinable child-wave (the child's OWN enumerated relationship-
    matrix wave, independent of the father-side observation window): the child's
    own wave weight, whether the biological father is coresident that wave, the
    child age band, and the father's observed marital state. Filtered per seed
    to children whose father is in the train ids. Mirrors
    :func:`gate2b_forensics2.q5_custodial_selection`'s child-record basis
    exactly (same denominator, same event).
    """
    fa = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ]
    child_ids = set(int(x) for x in fa["child_person_id"].unique())
    enum = (
        rel_map[["interview_year", "ego_person_id"]]
        .drop_duplicates()
        .rename(
            columns={
                "interview_year": "year",
                "ego_person_id": "child_person_id",
            }
        )
    )
    enum = enum[enum["child_person_id"].isin(child_ids)].copy()
    fa_child = fa.drop_duplicates("child_person_id").set_index(
        "child_person_id"
    )
    enum["birth_year"] = enum["child_person_id"].map(fa_child["birth_year"])
    enum["father_id"] = enum["child_person_id"].map(
        fa_child["parent_person_id"]
    )
    enum["child_age"] = enum["year"] - enum["birth_year"]
    enum = enum[
        (enum["child_age"] >= 0)
        & (enum["child_age"] <= hcs3.CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    enum["child_band"] = enum["child_age"].map(hcs3._child_band)
    enum = enum[enum["child_band"].notna()]
    demo_j = demo[["person_id", "period", "weight"]].rename(
        columns={"person_id": "child_person_id", "period": "year"}
    )
    enum = enum.merge(demo_j, on=["child_person_id", "year"], how="left")
    enum = enum[enum["weight"].fillna(0) > 0].copy()
    pp_fa = parent_pairs.rename(
        columns={"parent_person_id": "father_id"}
    ).assign(_wf=True)
    enum = enum.merge(
        pp_fa, on=["father_id", "child_person_id", "year"], how="left"
    )
    enum["with_father"] = enum["_wf"].fillna(False).astype(bool)
    enum = enum.drop(columns="_wf")
    enum = enum.merge(
        marital_by_year.rename(columns={"person_id": "father_id"}),
        on=["father_id", "year"],
        how="left",
    )
    enum["marital"] = enum["marital"].fillna(_NOT_MARRIED)
    return enum[
        [
            "father_id",
            "child_person_id",
            "year",
            "child_band",
            "weight",
            "with_father",
            "marital",
        ]
    ].reset_index(drop=True)


def fit_custodial_child_record(
    child_record_expo: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Train the child-record-basis ``P(with father | band, marital)``.

    Filtered to children whose father is a train person, weighted by the
    child's own wave weight. Delta 2 applies ONLY the ``not_married`` cells (the
    observable over-statement the young-married gate does not carry); both
    marital states are fit for the record / the gap check.
    """
    sub = child_record_expo[child_record_expo["father_id"].isin(train_ids)]
    rates: dict[tuple[str, str], float] = {}
    for lo, hi in hcs3.CUSTODIAL_CHILD_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for marital in (_MARRIED, _NOT_MARRIED):
            m = (sub["child_band"] == band) & (sub["marital"] == marital)
            frame = sub[m]
            rates[(band, marital)] = (
                hcs._weighted_rate(
                    frame, frame["with_father"].to_numpy(np.float64)
                )
                if len(frame)
                else 0.0
            )
    diag = {
        "n_child_record_waves_train": int(len(sub)),
        "not_married_child_record_by_band": {
            hc.band_label(lo, hi): round(
                rates[(hc.band_label(lo, hi), _NOT_MARRIED)], 5
            )
            for lo, hi in hcs3.CUSTODIAL_CHILD_AGE_BANDS
        },
    }
    return rates, diag


# --------------------------------------------------------------------------
# Delta 3a fit: non-family bridge incidence conditional on core size
# --------------------------------------------------------------------------
def fit_nonfamily_by_core(
    person_waves: pd.DataFrame,
    fu_sizes: pd.DataFrame,
    train_ids: set[int],
) -> tuple[
    dict[tuple[int, str, str], tuple[float, float, float]],
    dict[str, Any],
]:
    """Train ``P(non-family count in {0,1,2+} | core size, band, sex)``.

    The non-family count is enumerated ``hh_size`` minus the ego family-unit
    size (the core), clipped at 0; the core is capped at
    :data:`CORE_SIZE_CAP`. Only DENSE (core, band, sex) cells (>=
    :data:`_MIN_STRATUM_N` waves) are returned; the simulation falls back to
    candidate 4's (band, sex) table for sparse cells. Conditioning on core size
    lets the bridge lift size-3 family cores upward at the train rate (the
    forensics-2 non-core-member mass), which the band-only fit under-produced.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "band", "sex", "weight", "hh_size"]
    ].merge(fu_sizes, on=["person_id", "year"], how="left")
    pw["family_unit_size"] = pw["family_unit_size"].fillna(1).astype("int64")
    resid = np.clip(
        pw["hh_size"].to_numpy() - pw["family_unit_size"].to_numpy(), 0, None
    )
    core = np.clip(pw["family_unit_size"].to_numpy(), 1, CORE_SIZE_CAP).astype(
        np.int64
    )
    pw = pw.assign(
        _cls=np.where(resid == 0, "0", np.where(resid == 1, "1", "2+")),
        _core=core,
    )

    def _dist(frame: pd.DataFrame) -> tuple[float, float, float]:
        w = frame["weight"].to_numpy(np.float64)
        tot = float(w.sum())
        if tot <= 0:
            return (1.0, 0.0, 0.0)
        return tuple(
            float(w[(frame["_cls"] == c).to_numpy()].sum() / tot)
            for c in hcs3.NONFAMILY_CLASSES
        )  # type: ignore[return-value]

    table: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    for core_size in range(1, CORE_SIZE_CAP + 1):
        for lo, hi in hc.COMPOSITION_AGE_BANDS:
            band = hc.band_label(lo, hi)
            for sex in hc.SEXES:
                sub = pw[
                    (pw["_core"] == core_size)
                    & (pw["band"] == band)
                    & (pw["sex"] == sex)
                ]
                if len(sub) >= _MIN_STRATUM_N:
                    table[(core_size, band, sex)] = _dist(sub)
    # Weighted P(non-core member present | core size) diagnostic.
    incidence_by_core: dict[str, float] = {}
    for core_size in range(1, CORE_SIZE_CAP + 1):
        sub = pw[pw["_core"] == core_size]
        incidence_by_core[str(core_size)] = round(
            hcs._weighted_rate(sub, (sub["_cls"] != "0").to_numpy(np.float64)),
            5,
        )
    diag = {
        "n_dense_core_band_sex_cells": int(len(table)),
        "p_noncore_member_present_by_core": incidence_by_core,
        "core_size_cap": CORE_SIZE_CAP,
    }
    return table, diag


# --------------------------------------------------------------------------
# Delta 3b fit: per-ego coresident-parent count composition (1 vs 2)
# --------------------------------------------------------------------------
def parent_link_counts(rel_map: pd.DataFrame) -> pd.DataFrame:
    """Per (person_id, year), the count of coresident-parent links.

    The exact MX8 codes the reference ``coresident_parent`` uses
    (:data:`hc.CORESIDENCE_LINKS['coresident_parent']`); a coresident-parent ego
    has one or two (rarely more) parent links. Delta 3b draws the per-ego parent
    count from this train composition instead of the fixed ``parent_count=2``.
    """
    codes = hc.CORESIDENCE_LINKS["coresident_parent"]
    links = rel_map[rel_map["ego_rel_to_alter"].isin(codes)]
    counts = (
        links.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("n_parent_links")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    return counts[["person_id", "year", "n_parent_links"]]


def fit_parent_count_composition(
    person_waves: pd.DataFrame,
    parent_counts: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[tuple[str, str], float], float, dict[str, Any]]:
    """Train ``P(two coresident parents | coresident parent, band, sex)``.

    Among train coresident-parent person-waves (>=1 parent link), the weighted
    share with two-or-more parent links (drawn as exactly two, the ceiling
    ``parent_count=2`` used). A (band, sex) stratum thinner than
    :data:`_MIN_STRATUM_N` falls back to the pooled coresident-parent rate; the
    fallback is baked into every (band, sex) so the apply lookup is total.
    Returns (table, pooled, diag).
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "band", "sex", "weight", "coresident_parent"]
    ].merge(parent_counts, on=["person_id", "year"], how="left")
    pw["n_parent_links"] = pw["n_parent_links"].fillna(0).astype("int64")
    cp = pw[pw["coresident_parent"] & (pw["n_parent_links"] >= 1)]
    two = (cp["n_parent_links"] >= 2).to_numpy(np.float64)
    pooled = hcs._weighted_rate(cp, two)
    table: dict[tuple[str, str], float] = {}
    for lo, hi in hc.COMPOSITION_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for sex in hc.SEXES:
            sub = cp[(cp["band"] == band) & (cp["sex"] == sex)]
            table[(band, sex)] = (
                hcs._weighted_rate(
                    sub, (sub["n_parent_links"] >= 2).to_numpy(np.float64)
                )
                if len(sub) >= _MIN_STRATUM_N
                else pooled
            )
    diag = {
        "pooled_p_two_parents_given_coresident_parent": round(pooled, 5),
        "p_two_parents_by_band_female": {
            hc.band_label(lo, hi): round(
                table[(hc.band_label(lo, hi), "female")], 5
            )
            for lo, hi in hc.COMPOSITION_AGE_BANDS
        },
        "n_coresident_parent_waves_train": int(len(cp)),
    }
    return table, pooled, diag


# --------------------------------------------------------------------------
# Fit (train / side B only for every fitted parameter)
# --------------------------------------------------------------------------
def fit_household_model_v5(
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
    legal_flag: pd.DataFrame | None = None,
    child_record_expo: pd.DataFrame | None = None,
    parent_counts: pd.DataFrame | None = None,
) -> HouseholdCompositionModelV5:
    """Fit candidate 4 (byte-faithful) plus the three candidate-5 deltas.

    Every fitted parameter is estimated on the train complement (side B): the
    candidate-4 bundle via :func:`hcs4.fit_household_model_v4`, then the three
    deltas. The seed-independent link / coresidence / family-unit / child-record
    / parent-count frames may be injected (resolved once by the run script).
    """
    base_v4 = hcs4.fit_household_model_v4(
        hh,
        mpanel,
        demo,
        marriage_records,
        birth_records,
        marriage_order_map,
        rel_map,
        train_ids,
        father_links_child=father_links_child,
        parent_pairs=parent_pairs,
        fu_sizes=fu_sizes,
        legal_flag=legal_flag,
    )
    if father_links_child is None:
        father_links_child = hcs3.father_link_births_with_child(birth_records)
    if parent_pairs is None:
        parent_pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    if fu_sizes is None:
        fu_sizes = hcs3.family_unit_sizes(rel_map)
    if parent_counts is None:
        parent_counts = parent_link_counts(rel_map)
    if child_record_expo is None:
        marital_by_year = hcs3._father_marital_by_year(mpanel)
        child_record_expo = build_child_record_exposure(
            father_links_child, parent_pairs, marital_by_year, demo, rel_map
        )

    # Delta 1: multigen -- adult-child coupling.
    coupling_table, coupling_pooled, coupling_diag = (
        fit_multigen_child_coupling(hh.person_waves, train_ids)
    )
    # Delta 2: not-married child-record custodial rates.
    child_record, cr_diag = fit_custodial_child_record(
        child_record_expo, train_ids
    )
    # Delta 3a: non-family bridge conditional on core size.
    nonfamily_by_core, nfc_diag = fit_nonfamily_by_core(
        hh.person_waves, fu_sizes, train_ids
    )
    # Delta 3b: per-ego parent-count composition.
    pcc_table, pcc_pooled, pcc_diag = fit_parent_count_composition(
        hh.person_waves, parent_counts, train_ids
    )

    meta = {
        **base_v4.meta,
        "coupling_multigen_child": coupling_diag,
        "custodial_child_record": cr_diag,
        "nonfamily_by_core": nfc_diag,
        "parent_count_composition": pcc_diag,
        "grandchild_coupling_age_lo": GRANDCHILD_LO,
        "core_size_cap": CORE_SIZE_CAP,
    }
    return HouseholdCompositionModelV5(
        base_v4=base_v4,
        coupling_child_given_multigen=coupling_table,
        coupling_child_pooled=coupling_pooled,
        custodial_child_record=child_record,
        nonfamily_by_core=nonfamily_by_core,
        parent_count_two_share=pcc_table,
        parent_count_two_pooled=pcc_pooled,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Delta 2 apply: custodial per-wave coresidence (not-married -> child-record)
# --------------------------------------------------------------------------
def custodial_prob_v5(
    model: HouseholdCompositionModelV5, age: int, era: str, marital: str
) -> float:
    """Custodial coresidence probability with the delta-2 not-married swap.

    NOT-married fathers use the child-record-basis rate by child age band (the
    less-selected measure); MARRIED fathers use candidate 4's observable-basis
    lookup UNCHANGED (the young-married gate the forensics measured faithful).
    """
    if marital == _NOT_MARRIED:
        band = hc.band_label(*hcs4._child_band_bounds(age))
        if (band, _NOT_MARRIED) in model.custodial_child_record:
            return model.custodial_child_record[(band, _NOT_MARRIED)]
    return hcs4._custodial_prob(model.base_v4, age, era, marital)


def custodial_linked_child_counts_v5(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    model: HouseholdCompositionModelV5,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per side-A person-wave, the count of coresident father-linked children.

    Identical exposure construction and draw SHAPE to
    :func:`hcs4.custodial_linked_child_counts_v4` (so the custodial ``0xC3``
    stream is byte-identical in consumption), with the delta-2 probability: the
    not-married custodial cells swap to the child-record basis, married cells
    unchanged. The father-marital gate is applied on the SIMULATED marital
    state, self-consistent, exactly as candidate 4.
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
        & (expo["child_age"] <= hcs3.CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    expo["child_band"] = expo["child_age"].map(hcs3._child_band)
    expo = expo[expo["child_band"].notna()]
    if not len(expo):
        return counts
    expo = expo.merge(
        marital_sim.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    expo["marital"] = expo["marital"].fillna(_NOT_MARRIED)
    # Byte-identical row order to candidate 4 (candidate-3 order).
    expo = expo.sort_values(["parent_person_id", "birth_year", "_row"])
    expo = expo.reset_index(drop=True)
    ages = expo["child_age"].to_numpy()
    years = expo["year"].to_numpy()
    marital = expo["marital"].to_numpy()
    prob = np.array(
        [
            custodial_prob_v5(model, int(a), hcs4.era_of_year(int(y)), str(m))
            for a, y, m in zip(ages, years, marital, strict=True)
        ],
        dtype=np.float64,
    )
    u = rng.random(len(expo))
    coresident = u < prob
    rows = expo["_row"].to_numpy()[coresident]
    np.add.at(counts, rows, 1)
    return counts


# --------------------------------------------------------------------------
# Delta 3a apply: non-family count conditional on the simulated core size
# --------------------------------------------------------------------------
def sample_nonfamily_v5(
    pw: pd.DataFrame,
    model: HouseholdCompositionModelV5,
    class_rng: np.random.Generator,
    count_rng: np.random.Generator,
    core_size: np.ndarray,
) -> np.ndarray:
    """Draw a non-family member count keyed on the SIMULATED core size.

    The 0/1/2+ CLASS draw is byte-identical in shape to candidate 4 (same
    ``class_rng`` on the ``0xC3`` non-family stream), but the per-person
    thresholds now come from the (core size, band, sex) table where dense,
    falling back to candidate 4's (band, sex) shares -- so a size-3 family core
    is lifted upward at the train rate. The 2+ count spread is candidate 4's
    (byte-identical ``0xC4`` draw).
    """
    n = len(pw)
    bands = pw["band"].to_numpy(dtype=object)
    sexes = pw["sex"].to_numpy()
    core = np.clip(np.asarray(core_size, dtype=np.int64), 1, CORE_SIZE_CAP)
    p1 = np.zeros(n, dtype=np.float64)
    p2 = np.zeros(n, dtype=np.float64)
    # Base (band, sex) candidate-4 shares (the fallback for sparse core cells).
    for (band, sex), (_q0, q1, q2) in model.base_v3.nonfamily.items():
        mask = (bands == band) & (sexes == sex)
        p1[mask] = q1
        p2[mask] = q2
    # Refine with the dense (core, band, sex) shares.
    for (core_size_k, band, sex), (
        _q0,
        q1,
        q2,
    ) in model.nonfamily_by_core.items():
        mask = (core == core_size_k) & (bands == band) & (sexes == sex)
        p1[mask] = q1
        p2[mask] = q2
    u = class_rng.random(n)  # 0xC3 non-family class draw (byte-identical)
    q0 = 1.0 - p1 - p2
    cls = np.where(u < q0, 0, np.where(u < q0 + p1, 1, 2))

    # 2+ count spread on the isolated 0xC4 stream (candidate 4, byte-identical).
    u2 = count_rng.random(n)
    twoplus_count = np.full(n, 2, dtype=np.int64)
    for (band, sex), (counts, cum) in model.base_v4.nonfamily_2plus.items():
        mask = (bands == band) & (sexes == sex)
        if not mask.any():
            continue
        idx = np.searchsorted(cum, u2[mask], side="left")
        idx = np.clip(idx, 0, len(counts) - 1)
        twoplus_count[mask] = counts[idx]
    contrib = np.where(cls == 0, 0, np.where(cls == 1, 1, twoplus_count))
    return contrib.astype(np.int64)


def route_of_size3(
    has_spouse: np.ndarray, n_child: np.ndarray, n_parent: np.ndarray
) -> dict[str, np.ndarray]:
    """Mutually-exclusive size-3 composition-route masks (forensics-2 Q7).

    Applied to the simulated family core (spouse / child_counts / n_parents) for
    the gap-closure check; identical partition to
    :func:`gate2b_forensics2._route_of`.
    """
    has_spouse = np.asarray(has_spouse, dtype=bool)
    n_child = np.asarray(n_child, dtype=np.int64)
    n_parent = np.asarray(n_parent, dtype=np.int64)
    couple_child = has_spouse & (n_child >= 1)
    single_two = ~has_spouse & (n_child >= 2)
    couple_parent = has_spouse & (n_child == 0) & (n_parent >= 1)
    three_adults = ~has_spouse & (n_child == 0) & (n_parent >= 1)
    routed = couple_child | single_two | couple_parent | three_adults
    return {
        "couple_plus_child": couple_child,
        "single_parent_plus_two_children": single_two,
        "couple_plus_parent": couple_parent,
        "three_adults": three_adults,
        "other_family_core": ~routed,
    }


def _wshare(weight: np.ndarray, hit: np.ndarray) -> float:
    tot = float(weight.sum())
    return float(weight[hit].sum() / tot) if tot > 0 else 0.0


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def simulate_draw_v5(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModelV5,
    ids_a: set[int],
    draw_seed: int,
    occupancy_stream_tag: int = 0xB2B,
    delta_stream_tag_v2: int = 0xC2,
    delta_stream_tag_v3: int = 0xC3,
    delta_stream_tag_v4: int = 0xC4,
    delta_stream_tag_v5: int = DELTA_STREAM_TAG_V5,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-5 draw of the side-A holdout households.

    Reproduces candidate 4's draw byte-for-byte -- candidate 1's
    ``simulate_draw`` at ``0xB2B``, the candidate-2 streams at ``0xC2``, the
    candidate-3 streams at ``0xC3``, the candidate-4 legal-residual / non-family
    2+ streams at ``0xC4`` -- with the three candidate-5 deltas: the custodial
    probability is re-keyed for not-married fathers (byte-identical ``0xC3``
    draw shape); the non-family class thresholds are re-keyed on the simulated
    core size (byte-identical ``0xC3`` draw shape); and the coupled adult-child
    draw and the per-ego parent-count draw come from a separate
    ``SeedSequence([draw_seed, 0xC5]).spawn(2)``.
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

    mats = hcs._padded_person_matrices(side_a_pw)
    pw = mats["pw"]
    row_of = mats["row_of"]
    n_persons, max_waves = mats["n_persons"], mats["max_waves"]
    valid = row_of >= 0
    safe_row = np.where(valid, row_of, 0)
    age_mat = pw["age"].to_numpy()[safe_row]
    sex_mat = pw["sex"].to_numpy()[safe_row]
    band_mat = pw["band"].to_numpy(dtype=object)[safe_row]
    obs_cohab = pw["cohabiting"].to_numpy(dtype=bool)[safe_row] & valid

    # 3. candidate-4 DELTA 1 (carried): age-refined cohabitation overlay.
    cohab_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    cohab_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.base_v2.cohab_entry.items():
        cohab_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in model.base_v2.cohab_exit.items():
        cohab_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (age, sex), rate in model.base_v4.cohab_entry_age.items():
        cohab_entry_prob[(age_mat == age) & (sex_mat == sex)] = rate
    for (age, sex), rate in model.base_v4.cohab_exit_age.items():
        cohab_exit_prob[(age_mat == age) & (sex_mat == sex)] = rate
    cohab_state = hcs._evolve_two_state(
        valid, obs_cohab, cohab_entry_prob, cohab_exit_prob, cohab_rng
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]

    # 4. candidate-4 delta substreams (0xC4), consumed EXACTLY as candidate 4.
    c4_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v4])
    legal_resid_ss, nonfamily2_ss = c4_ss.spawn(2)
    legal_resid_rng = np.random.default_rng(legal_resid_ss)
    nonfamily2_rng = np.random.default_rng(nonfamily2_ss)

    # candidate-4 DELTA 2 (carried): additive legal-spouse residual overlay.
    lr_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_marg_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.base_v4.legal_residual_entry.items():
        m = (band_mat == band) & (sex_mat == sex)
        lr_exit_prob[m] = model.base_v4.legal_residual_exit[(band, sex)]
        lr_entry_prob[m] = rate
        lr_marg_prob[m] = model.base_v4.legal_residual_marginal[(band, sex)]
    lr_initial = np.zeros((n_persons, max_waves), dtype=bool)
    lr_initial[:, 0] = (
        legal_resid_rng.random(n_persons) < lr_marg_prob[:, 0]
    ) & valid[:, 0]
    lr_state = hcs._evolve_two_state(
        valid, lr_initial, lr_entry_prob, lr_exit_prob, legal_resid_rng
    )
    lr_row = np.zeros(len(pw), dtype=bool)
    lr_row[row_of[valid]] = lr_state[valid]

    spouse = legal_spouse | cohab_row | lr_row

    # 5. certified marital core + maternal births (same draw seed as cand 4).
    sim_panel, sim_births = ft.simulate(
        mpanel, ids_a, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"

    # 6. linked / shadow paternal children (candidate-2 child stream 0xC2).
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
    from populace_dynamics.models.family_transitions.components.fertility import (  # noqa: E501
        build_fertility_lookup,
    )

    lookup, decade_map = build_fertility_lookup(
        base.family_transitions.fertility
    )
    paternal_shadow = hcs._paternal_births(
        sim_years, unlinked_men, base.male_gap, lookup, decade_map, child_rng
    )
    paternal_shadow["_source"] = "shadow"

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

    # candidate-3 delta substreams (0xC3), isolated from 0xB2B/0xC2.
    c3_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v3])
    custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    custodial_rng = np.random.default_rng(custodial_ss)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # candidate-5 delta substreams (0xC5), isolated from 0xB2B/0xC2/0xC3/0xC4.
    c5_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)

    # DELTA 2: custodial per-wave coresidence with the not-married child-record
    # swap. Same draw shape as candidate 4 -> byte-identical custodial stream.
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked = custodial_linked_child_counts_v5(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model,
        custodial_rng,
    )
    child_counts = child_counts_nonlinked + child_counts_linked

    coresident_child, grandchild_composed, _hh_default = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

    # DELTA 3b: per-ego coresident-parent count (1 vs 2) on the 0xC5 stream,
    # replacing the fixed parent_count=2; feeds hh_size ONLY.
    bands_row = pw["band"].to_numpy(dtype=object)
    sexes_row = pw["sex"].to_numpy()
    ages_row = pw["age"].to_numpy()
    two_share = np.full(len(pw), model.parent_count_two_pooled, np.float64)
    for (band, sex), share in model.parent_count_two_share.items():
        two_share[(bands_row == band) & (sexes_row == sex)] = share
    u_pc = parentcount_rng.random(len(pw))
    parent_count_ego = np.where(u_pc < two_share, 2, 1).astype(np.int64)
    n_parents_ego = np.where(c1_parent, parent_count_ego, 0).astype(np.int64)
    hh_size_base = (
        1
        + spouse.astype(np.int64)
        + child_counts.astype(np.int64)
        + n_parents_ego
    ).astype(np.int64)

    # DELTA 1: multigen -- adult-child coupling for 55+ egos. The independent
    # coresident-own-adult-child input to the composed grandchild is replaced by
    # a joint draw P(child | simulated multigen, band, sex) on the 0xC5 stream;
    # the multigen marginal is UNCHANGED. Feeds the grandchild composition ONLY.
    is_55_row = ages_row >= GRANDCHILD_LO
    p_coupled = np.zeros(len(pw), dtype=np.float64)
    for (sex, mg), rate in model.coupling_child_pooled.items():
        p_coupled[is_55_row & (sexes_row == sex) & (c1_multigen == mg)] = rate
    for (band, sex, mg), rate in model.coupling_child_given_multigen.items():
        m = (
            is_55_row
            & (bands_row == band)
            & (sexes_row == sex)
            & (c1_multigen == mg)
        )
        p_coupled[m] = rate
    u_couple = coupling_rng.random(len(pw))
    coupled_child = u_couple < p_coupled
    grandchild_coupled = c1_multigen & coupled_child & (~c1_parent)
    grandchild_final = np.where(
        is_55_row, grandchild_coupled, grandchild_composed
    )

    # DELTA 4 of candidate 4 (carried): 5-year skipped-generation occupancy.
    obs_skipgen = (
        pw["coresident_grandchild"].to_numpy(dtype=bool)[safe_row]
        & ~pw["multigen"].to_numpy(dtype=bool)[safe_row]
        & valid
    )
    skip_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    skip_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.base_v3.skipgen_entry.items():
        skip_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in model.base_v3.skipgen_exit.items():
        skip_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for ((lo, hi), sex), rate in model.base_v4.skipgen_entry_age.items():
        m = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sex)
        skip_entry_prob[m] = rate
    for ((lo, hi), sex), rate in model.base_v4.skipgen_exit_age.items():
        m = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sex)
        skip_exit_prob[m] = rate
    skipgen_state = hcs._evolve_two_state(
        valid, obs_skipgen, skip_entry_prob, skip_exit_prob, skipgen_rng
    )
    skipgen_row = np.zeros(len(pw), dtype=bool)
    skipgen_row[row_of[valid]] = skipgen_state[valid]
    coresident_grandchild = grandchild_final | skipgen_row

    # DELTA 3a: non-family count with the core-size-conditional bridge reach.
    # Class draw byte-identical on 0xC3; 2+ count on 0xC4.
    nonfamily_count = sample_nonfamily_v5(
        pw, model, nonfamily_rng, nonfamily2_rng, hh_size_base
    )
    hh_size = (hh_size_base + nonfamily_count).astype(np.int64)

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

    # ---- Coupling (Q6) + gap (Q7) checks on the side-A draw. ----
    weight = pw["weight"].to_numpy(np.float64)
    mask55f = is_55_row & (sexes_row == "female")
    w55 = weight[mask55f]
    mg55 = c1_multigen[mask55f]
    coup55 = coupled_child[mask55f]
    np55 = ~c1_parent[mask55f]
    joint55 = grandchild_coupled[mask55f]
    p_mg = _wshare(w55, mg55)
    p_coup = _wshare(w55, coup55)
    p_np = _wshare(w55, np55)
    base3 = hh_size_base == 3
    routes = route_of_size3(spouse, child_counts, n_parents_ego)
    cp_mask = c1_parent
    diagnostics = {
        "n_maternal_births": int(len(maternal)),
        "n_paternal_linked_births": int(len(paternal_linked)),
        "n_paternal_shadow_births": int(len(paternal_shadow)),
        "n_side_a_men": len(men_ids),
        "n_linked_fathers_side_a": len(linked_ids),
        "n_cohab_person_waves_simulated": int(cohab_row[row_of[valid]].sum()),
        "n_legal_residual_person_waves_simulated": int(
            lr_row[row_of[valid]].sum()
        ),
        "n_linked_child_coresident_wave_units": int(child_counts_linked.sum()),
        "n_skipgen_person_waves_simulated": int(
            skipgen_row[row_of[valid]].sum()
        ),
        "n_coupled_grandchild_waves_simulated": int(
            grandchild_coupled[is_55_row].sum()
        ),
        "mean_nonfamily_count_simulated": float(nonfamily_count.mean()),
        "mean_nonfamily_count_within_2plus_simulated": (
            float(nonfamily_count[nonfamily_count >= 2].mean())
            if (nonfamily_count >= 2).any()
            else 0.0
        ),
        # Delta 1 coupling check (55+ female).
        "coupling_gc55f_den_wt": float(w55.sum()),
        "coupling_gc55f_multigen": p_mg,
        "coupling_gc55f_coupled_child": p_coup,
        "coupling_gc55f_notparent": p_np,
        "coupling_gc55f_joint_mg_child_notparent": _wshare(w55, joint55),
        "coupling_gc55f_independence_product": p_mg * p_coup * p_np,
        "coupling_gc55f_composed_grandchild": _wshare(
            w55, grandchild_final[mask55f]
        ),
        "coupling_gc55f_skiponly": _wshare(
            w55, (skipgen_row & ~grandchild_final)[mask55f]
        ),
        "coupling_gc55f_union": _wshare(w55, coresident_grandchild[mask55f]),
        # Delta 3 gap-closure check (size-3 family core).
        "size3_core_total": _wshare(weight, base3),
        "size3_full_total": _wshare(weight, hh_size == 3),
        "size3_route_three_adults": _wshare(
            weight, routes["three_adults"] & base3
        ),
        "size3_route_couple_plus_child": _wshare(
            weight, routes["couple_plus_child"] & base3
        ),
        "size3_route_single_parent_plus_two_children": _wshare(
            weight, routes["single_parent_plus_two_children"] & base3
        ),
        "size3_route_couple_plus_parent": _wshare(
            weight, routes["couple_plus_parent"] & base3
        ),
        "mean_n_parents_among_coresident_parent": (
            float(n_parents_ego[cp_mask].mean()) if cp_mask.any() else 0.0
        ),
    }
    return panel, diagnostics


# --------------------------------------------------------------------------
# Delta-specific fit-vs-forensics checks (fit seed's train side B)
# --------------------------------------------------------------------------
def coupling_and_gap_checks(
    model: HouseholdCompositionModelV5,
    forensics: dict[str, Any],
) -> dict[str, Any]:
    """Record the delta-1 joint-vs-product and delta-3 gap-closure fits.

    Computed from the fitted objects on the given seed's train side B (the
    fitted objects are a property of the fit, not the holdout), beside the
    forensics-2 measured quantities the deltas implement.
    """
    q6 = forensics["question_6_grandchild_reference_channels"]
    ref_comp = q6["reference_component_rates_55plus_female"]
    ref_mg = ref_comp["multigen"]
    # Implied fit joint: P(multigen) x P(child | multigen=1) (the coupling
    # replaces the independence product with this joint on the multigen mass).
    p_child_given_mg1 = model.coupling_child_pooled[("female", True)]
    implied_joint = ref_mg * p_child_given_mg1
    delta_1 = {
        "reference_joint_multigen_and_child_and_notparent": ref_comp[
            "multigen_and_child_and_notparent"
        ],
        "reference_independence_product": ref_comp[
            "independence_product_multigen_x_child_x_notparent"
        ],
        "fitted_p_child_given_multigen_true_female": round(
            p_child_given_mg1, 5
        ),
        "fitted_p_child_given_multigen_false_female": round(
            model.coupling_child_pooled[("female", False)], 5
        ),
        "reference_multigen_55plus_female": ref_mg,
        "implied_fit_joint_multigen_x_p_child_given_multigen": round(
            implied_joint, 5
        ),
        "coupling_lifts_joint_toward_reference": bool(
            implied_joint
            > ref_comp["independence_product_multigen_x_child_x_notparent"]
        ),
        "note": (
            "The reference couples multigen and a coresident own child into "
            "one household fact (joint ~5x the independence product). The "
            "coupling draws P(child | multigen=True) ~0.88 on the simulated "
            "multigen mass, lifting the composed joint from the independence "
            "product toward the reference joint; the multigen marginal is "
            "unchanged (carried from candidate 1)."
        ),
    }

    q7 = forensics["question_7_hh_size3_family_core_routes"]
    gap = q7["gap_decomposition"]
    delta_3 = {
        "reference_actual_size3_total": q7["reference_actual_size3_total"],
        "reference_core_size3_total": q7["reference_core_size3_total"],
        "total_gap_sim_core_minus_ref_actual": gap[
            "total_gap_sim_core_minus_ref_actual"
        ],
        "noncore_member_gap_total": gap["noncore_member_gap_total"],
        "composition_gap_total": gap["composition_gap_total"],
        "fitted_p_noncore_member_present_by_core": model.meta[
            "nonfamily_by_core"
        ]["p_noncore_member_present_by_core"],
        "fitted_pooled_p_two_parents": model.meta["parent_count_composition"][
            "pooled_p_two_parents_given_coresident_parent"
        ],
        "note": (
            "The 0.088 size-3 core-vs-actual gap splits into a non-core-member "
            "part ~0.051 (bridge reach conditional on core size) and a "
            "composition part ~0.037 (the parent_count=2 correction). Delta 3a "
            "lifts size-3 cores at the fitted per-core non-core incidence; "
            "delta 3b draws 1 vs 2 parents from the train composition instead "
            "of the fixed 2, relaxing the over-produced three-adults route."
        ),
    }
    return {"delta_1_coupling": delta_1, "delta_3_gap_closure": delta_3}
