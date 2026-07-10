"""Gate-2b candidate 2: cohabitation overlay + paternal child attribution.

Candidate 2 (registration #42 comment 4939456379) is candidate 1
(:mod:`populace_dynamics.models.household_composition_sim`, merged in PR
#133) with EXACTLY TWO frozen deltas; every other component -- the certified
tranche-2a marital core and maternal births, the parental-home exit hazard,
the multigenerational entry/exit machinery, ``coresident_parent``, the
household-size composition rule, and the composed-only grandchild -- is
REUSED byte-faithfully from the candidate-1 module (imported, not copied).

Delta 1 -- **cohabiting-partner overlay.** The gate-2b reference spouse
concept is the MX8 spouse-OR-partner flag (:data:`hc.CORESIDENCE_LINKS`
``coresident_spouse`` = codes ``{20, 22}``). The staged MX23REL formats file
decodes ``20 = 'Legal Spouse'`` and ``22 = 'Partner - the cohabiting
partner to ALTER, not legally married'`` (verified 2026-07-10). The certified
2a registry generates LEGAL marriage (the code-20 mass); candidate 1's
simulated ``coresident_spouse`` therefore under-counts the code-22 cohabiting
partners on covered waves (the young-adult failure the grading isolated). So
candidate 2 fits a cohabitation occupancy state on the code-22 partner spells
ONLY -- the concept mass the legal registry omits -- as train entry/exit
hazards by age band x sex, exactly like the multigen machinery, evolved from
each holdout person's OBSERVED initial cohabitation state. ``coresident_
spouse`` becomes the UNION of the certified legal-marriage state and the
cohabitation state (a person-period is spouse-coresident if either holds;
double-counting is impossible -- the union is on the same ego, and the
code-20 legal mass and code-22 partner mass are disjoint).

Delta 2 -- **paternal child attribution.** Candidate 1 attributes coresident
children of men through a shadow fertility kernel evaluated at the wife's
imputed age (the maternal certified kernel is female/marriage-unconditional,
so the shadow under-attributes -- the male ``coresident_child`` failure the
grading isolated). Candidate 2 attributes each man his OBSERVED cah85_23
father->child birth links (the man's own recorded biological births, a data
initial condition of the same species as the observed initial states), aging
them out under the SAME fitted parental-home-exit hazard; the candidate-1
spousal-gap-shifted shadow kernel is retained ONLY for the residual set of
men with no recorded father link (the linked / unlinked split is recorded in
the artifact).

RNG and byte-faithful carry. The candidate-1 families that cleared
(``coresident_parent``, ``multigen``, and their transitions) are produced by
calling candidate 1's :func:`hcs.simulate_draw` UNCHANGED at the same draw
seed 5200+k and occupancy tag 0xB2B, and their states are read straight off
its panel -- so those states are BYTE-IDENTICAL to candidate 1 and cannot be
perturbed by the two deltas (regression is impossible by construction, not
merely improbable). The certified marital core and maternal births are the
same certified simulate at the same seed. The two registered deltas draw from
a SEPARATE ``SeedSequence([5200 + k, 0xC2])`` (spawned into a child stream and
a cohabitation stream), isolated from the candidate-1 occupancy stream, so
they add mass without touching the carried draws. This is the faithful
realization of the registration's "carried byte-faithfully / the union is
additive only / everything that cleared stays cleared".
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
from populace_dynamics.models.family_transitions.components.fertility import (
    build_fertility_lookup,
)

__all__ = [
    "PARTNER_CODE",
    "HouseholdCompositionModelV2",
    "cohabitation_flag",
    "attach_cohabitation",
    "fit_cohabitation_rates",
    "father_link_births",
    "father_link_coverage",
    "fit_household_model_v2",
    "simulate_draw_v2",
]

#: MX8 ego-to-alter code for a cohabiting partner (NOT a legal spouse). The
#: staged MX23REL formats decode 20='Legal Spouse' and 22='Partner - the
#: cohabiting partner to ALTER, not legally married'; the certified 2a
#: registry already generates the legal-marriage (code-20) mass, so the
#: cohabitation overlay is the code-22 partner mass ONLY.
PARTNER_CODE = relmap.PARTNER  # == 22


@dataclass
class HouseholdCompositionModelV2:
    """Candidate 1's fitted bundle plus the two candidate-2 delta components.

    ``base`` is the byte-faithful candidate-1
    :class:`~populace_dynamics.models.household_composition_sim.
    HouseholdCompositionModel` (all carried components); ``cohab_entry`` /
    ``cohab_exit`` are the train-fitted cohabitation (code-22) occupancy
    hazards (delta 1); ``cohab_flag`` is the per person-wave observed
    cohabitation state (the initial condition for the overlay); and
    ``father_links`` are the observed cah85_23 father->child birth links
    (delta 2), a data attribute (not a fitted parameter).
    """

    base: hcs.HouseholdCompositionModel
    cohab_entry: dict[tuple[str, str], float]
    cohab_exit: dict[tuple[str, str], float]
    cohab_flag: pd.DataFrame
    father_links: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Delta 1: cohabitation (MX8 code-22 partner) occupancy
# --------------------------------------------------------------------------
def cohabitation_flag(rel_map: pd.DataFrame) -> pd.DataFrame:
    """Per (person_id, year), whether ego has a coresident code-22 partner.

    Mirrors :func:`hc.household_roster`'s coresidence-flag construction, but
    for the cohabiting-partner code ONLY (MX8 ``22``), at the (year, person)
    grain the household panel is keyed on. An ego coded a partner of some
    alter that wave HAS a coresident cohabiting partner; the certified legal
    registry does not carry this mass, so it is the overlay's target.
    """
    nonself = rel_map[rel_map["ego_rel_to_alter"] != relmap.SELF]
    partner = nonself[nonself["ego_rel_to_alter"] == PARTNER_CODE]
    hits = (
        partner.groupby(["interview_year", "ego_person_id"])
        .size()
        .rename("_n")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    hits["cohabiting"] = hits["_n"] > 0
    return hits[["person_id", "year", "cohabiting"]]


def attach_cohabitation(
    person_waves: pd.DataFrame, cohab_flag: pd.DataFrame
) -> pd.DataFrame:
    """Return a copy of ``person_waves`` with ``cohabiting`` + next state.

    ``cohabiting`` is the observed code-22 partner state (False where the
    person-wave has no code-22 alter); ``next_cohabiting`` is the following
    observed wave's state (the transition target the hazard is fit on),
    aligned by the same person-sorted ordering the panel uses.
    """
    pw = person_waves.merge(cohab_flag, on=["person_id", "year"], how="left")
    pw["cohabiting"] = pw["cohabiting"].fillna(False).astype(bool)
    pw = pw.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw["next_cohabiting"] = pw.groupby("person_id", sort=False)[
        "cohabiting"
    ].shift(-1)
    return pw


def fit_cohabitation_rates(
    train_pw: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Train cohabitation entry / exit hazards by age band x sex.

    Identical construction to
    :func:`hcs.fit_multigen_rates`, on the ``cohabiting`` / ``next_
    cohabiting`` columns: entry rate among not-cohabiting at-risk waves and
    exit rate among cohabiting at-risk waves, weighted, per (band, sex), with
    empty strata falling back to the pooled overall rate.
    """
    has_next = train_pw[train_pw["has_next"] & train_pw["band"].notna()]
    entry_pool = has_next[~has_next["cohabiting"]]
    exit_pool = has_next[has_next["cohabiting"]]
    entry_overall = hcs._weighted_rate(
        entry_pool, entry_pool["next_cohabiting"].to_numpy(dtype=np.float64)
    )
    exit_overall = hcs._weighted_rate(
        exit_pool,
        exit_pool["next_cohabiting"].eq(False).to_numpy(dtype=np.float64),
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
                    e, e["next_cohabiting"].to_numpy(dtype=np.float64)
                )
                if len(e)
                else entry_overall
            )
            exit_[(band, sex)] = (
                hcs._weighted_rate(
                    x,
                    x["next_cohabiting"].eq(False).to_numpy(dtype=np.float64),
                )
                if len(x)
                else exit_overall
            )
    return entry, exit_


# --------------------------------------------------------------------------
# Delta 2: paternal father->child links (cah85_23)
# --------------------------------------------------------------------------
def father_link_births(birth_records: pd.DataFrame) -> pd.DataFrame:
    """Observed cah85_23 father->child biological birth links.

    One row per recorded biological birth event whose parent is male (a
    father->child link): ``parent_person_id`` (the father) and ``birth_year``
    (the child). Mirrors the fertility-kernel scope the shadow it replaces
    models -- biological births, not adoptions -- so the linked and unlinked
    streams attribute the same KIND of children. NA birth years are dropped
    (the aging-out needs a birth year).
    """
    from populace_dynamics.data import births

    ev = births.birth_events(birth_records)
    fa = ev[
        (ev["parent_sex"] == "male")
        & (ev["record_type"] == "birth")
        & ev["birth_year"].notna()
    ]
    out = pd.DataFrame(
        {
            "parent_person_id": fa["parent_person_id"].astype("int64"),
            "birth_year": fa["birth_year"].astype("int64"),
        }
    )
    return out.reset_index(drop=True)


def father_link_coverage(
    mpanel: transitions.MaritalPanel,
    father_links: pd.DataFrame,
    ids_a: set[int],
) -> dict[str, Any]:
    """The deterministic linked / unlinked father split over side-A men.

    Records how many side-A men carry >= 1 observed father->child link (the
    paternal children attributed from data) versus the residual with none
    (attributed by the retained shadow kernel), plus the ever-married cut
    (the shadow's candidate-1 population). Deterministic given ``ids_a``.
    """
    men = mpanel.attrs[
        (mpanel.attrs["sex"] == "male")
        & (mpanel.attrs["person_id"].isin(ids_a))
    ]
    men_ids = set(int(x) for x in men["person_id"])
    linked_ids = (
        set(int(x) for x in father_links["parent_person_id"]) & men_ids
    )
    linked_records = father_links[
        father_links["parent_person_id"].isin(linked_ids)
    ]
    ever_married = set(
        int(x) for x in men[men["n_marriages"] > 0]["person_id"]
    )
    n_men = len(men_ids)
    return {
        "n_side_a_men": n_men,
        "n_linked_fathers": len(linked_ids),
        "n_unlinked_men": n_men - len(linked_ids),
        "coverage_fraction_men": (
            round(len(linked_ids) / n_men, 4) if n_men else 0.0
        ),
        "n_linked_father_birth_records": int(len(linked_records)),
        "n_ever_married_men": len(ever_married),
        "n_linked_and_ever_married": len(linked_ids & ever_married),
        "n_unlinked_ever_married_shadow_eligible": len(
            (men_ids - linked_ids) & ever_married
        ),
    }


# --------------------------------------------------------------------------
# Fit (train / side B only for every fitted parameter)
# --------------------------------------------------------------------------
def fit_household_model_v2(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    marriage_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    marriage_order_map: pd.DataFrame,
    rel_map: pd.DataFrame,
    train_ids: set[int],
) -> HouseholdCompositionModelV2:
    """Fit candidate 1 (byte-faithful) plus the two candidate-2 deltas.

    Every fitted parameter is estimated on the train complement (side B): the
    candidate-1 bundle via
    :func:`hcs.fit_household_model`, and the cohabitation entry/exit hazards
    on the train code-22 partner spells. The father->child links are observed
    data (not a fitted parameter); they are resolved here once for reuse.
    """
    base = hcs.fit_household_model(
        hh,
        mpanel,
        demo,
        marriage_records,
        birth_records,
        marriage_order_map,
        rel_map,
        train_ids,
    )

    cohab_flag = cohabitation_flag(rel_map)
    pw_cohab = attach_cohabitation(hh.person_waves, cohab_flag)
    train_pw = pw_cohab[pw_cohab["person_id"].isin(train_ids)]
    cohab_entry, cohab_exit = fit_cohabitation_rates(train_pw)

    father_links = father_link_births(birth_records)

    train_cohab_waves = int(train_pw["cohabiting"].sum())
    meta = {
        **base.meta,
        "cohab_entry_overall": round(
            float(np.mean(list(cohab_entry.values()))), 5
        ),
        "cohab_exit_overall": round(
            float(np.mean(list(cohab_exit.values()))), 5
        ),
        "cohab_train_person_waves": train_cohab_waves,
        "n_father_link_births_total": int(len(father_links)),
        "n_distinct_linked_fathers_total": int(
            father_links["parent_person_id"].nunique()
        ),
    }
    return HouseholdCompositionModelV2(
        base=base,
        cohab_entry=cohab_entry,
        cohab_exit=cohab_exit,
        cohab_flag=cohab_flag,
        father_links=father_links,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def simulate_draw_v2(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModelV2,
    ids_a: set[int],
    draw_seed: int,
    occupancy_stream_tag: int = 0xB2B,
    delta_stream_tag: int = 0xC2,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-2 draw of the side-A holdout households.

    The CARRIED families -- ``coresident_parent`` and ``multigen`` (their
    stocks and transitions) -- are produced by calling candidate 1's
    :func:`hcs.simulate_draw` UNCHANGED at the same ``draw_seed`` and
    ``occupancy_stream_tag``, so those states are BYTE-IDENTICAL to candidate
    1 (the parental-home exit and multigen occupancy draws, and therefore the
    families that cleared, cannot be perturbed by the two deltas -- regression
    is impossible by construction). The two registered deltas are then
    overlaid and the household is recomposed: the cohabitation state (delta 1)
    is unioned into ``coresident_spouse``, and the paternal children (delta 2)
    replace the child roster; both delta draws come from a distinct
    ``SeedSequence([draw_seed, delta_stream_tag])`` so they are isolated from
    the candidate-1 occupancy stream. Returns the simulated panel and a
    per-draw diagnostics dict.
    """
    base = model.base

    # 1. Carried families byte-identical to candidate 1: call candidate 1's
    #    simulate_draw unchanged and read the parental / multigen states (and
    #    its legal-marriage coresident_spouse) straight off its panel.
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
        model.cohab_flag, on=["person_id", "year"], how="left"
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

    # Delta substreams (isolated from the candidate-1 occupancy stream).
    delta_ss = np.random.SeedSequence([draw_seed, delta_stream_tag])
    child_ss, cohab_ss = delta_ss.spawn(2)
    child_rng = np.random.default_rng(child_ss)
    cohab_rng = np.random.default_rng(cohab_ss)

    # Delta 1: cohabitation (code-22) occupancy, evolved from the observed
    # initial state, unioned into coresident_spouse.
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
    for (band, sex), rate in model.cohab_entry.items():
        cohab_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in model.cohab_exit.items():
        cohab_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    cohab_state = hcs._evolve_two_state(
        valid, obs_cohab, cohab_entry_prob, cohab_exit_prob, cohab_rng
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]
    spouse = legal_spouse | cohab_row

    # Delta 2: paternal children from observed father links (linked men) +
    # the retained candidate-1 shadow kernel (unlinked residual). The maternal
    # side is the certified kernel, re-derived at the same draw seed (identical
    # to candidate 1's maternal births).
    sim_panel, sim_births = ft.simulate(
        mpanel, ids_a, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")

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
    unlinked_men = men_a[~men_a["person_id"].isin(linked_ids)]
    lookup, decade_map = build_fertility_lookup(
        base.family_transitions.fertility
    )
    paternal_shadow = hcs._paternal_births(
        sim_years, unlinked_men, base.male_gap, lookup, decade_map, child_rng
    )
    all_births = pd.concat(
        [maternal, paternal_linked, paternal_shadow], ignore_index=True
    )
    all_births = all_births[all_births["parent_person_id"].isin(ids_a)]
    child_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    child_counts = hcs._coresident_child_counts(child_leaves, side_a_pw)

    # Recompose the household from the byte-identical carried states plus the
    # two deltas (household size and grandchild are composed, not tuned).
    coresident_child, coresident_grandchild, hh_size = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

    sim_pw = pw.copy()
    sim_pw["coresident_spouse"] = spouse
    sim_pw["coresident_parent"] = c1_parent
    sim_pw["coresident_child"] = coresident_child
    sim_pw["coresident_grandchild"] = coresident_grandchild
    sim_pw["multigen"] = c1_multigen
    sim_pw["hh_size"] = hh_size.astype(np.int64)
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
    }
    return panel, diagnostics
