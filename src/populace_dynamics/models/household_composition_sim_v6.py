"""Gate-2b candidate 6: the four measured levers.

Candidate 6 (registration #42 comment 4946285556) is candidate 5
(:mod:`populace_dynamics.models.household_composition_sim_v5`, merged in PR
#141) with EXACTLY FOUR frozen deltas, each designed against a graded finding
of the gate-2b forensics-3 decomposition (``runs/gate2b_forensics3_v1.json``,
grading 4946281888). Everything candidate 5 cleared or carried -- the certified
tranche-2a marital core, the carried ``coresident_parent`` / ``multigen``
(stock + transitions) / ``parental_home_exit``, the delta-1 multigen--adult-
child coupling at 55+ (which took ``coresident_grandchild`` to 1.00), and the
delta-3b per-ego parent-count composition -- is carried BYTE-FAITHFULLY:
candidate 6 REUSES candidate 5's generator (candidate 1's ``simulate_draw`` at
``0xB2B``; the candidate-2 streams at ``0xC2``; the candidate-3 streams at
``0xC3``; the candidate-4 legal-residual stream at ``0xC4``; the candidate-5
coupling / parent-count streams at ``0xC5``) and re-runs its exact draws, then
the four candidate-6 deltas either (i) re-key an existing probability table
whose per-wave draw consumes RNG BY SHAPE ONLY -- so the stream stays byte-
identical and only the affected state changes -- or (ii) add a genuinely new
stochastic component on an ISOLATED ``SeedSequence([draw_seed, 0xC6])``.

Delta 1 -- **0-4 basis revert** (``coresident_child`` 15-24|male, forensics-3
Q8). The candidate-5 not-married custodial swap to the child-record basis is
REVERTED at child ages 0-4 to the OBSERVABLE basis (candidate 4's father-wave
custodial lookup); the child-record swap STAYS at 5-17|not-married. Forensics-3
adjudicated the child-record 0-4 gap a join-denominator artifact (child-record
+0.048 over observable at 0-4 but LOWER at school ages -- a sign inversion that
marks selective under-enumeration of young children living away from the
father). The reference gate cell is ego-anchored, so the observable basis is
the matched concept at 0-4; the pooled observable there (~0.710) is BELOW the
child-record (~0.742), so the revert drains the young-father over-production.
Same draw shape (byte-identical ``0xC3`` custodial stream); only the not-married
0-4 probability moves.

Delta 2 -- **adult-child exit timing** (``coresident_child`` 35-44|male and
45-54|female, forensics-3 Q8). A single-year child-age home-exit hazard over
ages 18-30 is re-fit on train (weighted exit rate among coresident-parent at-
risk waves by the ego's own age x sex -- the coresident adult child's exit) and
applied to both parent sides as the exit timing:

* **Maternal (own-birth) side** -- the maternal leave-year draw OVERRIDES
  candidate 1's spline hazard at ages 18-30, so maternal adult children age out
  at the empirical single-year rate (the empirical exceeds the spline for sons);
  this is the ``coresident_child`` 45-54|female lever. Draws on the isolated
  ``0xC6`` stream; the shadow leave-year stays byte-identical to candidate 5 (it
  comes off the ``0xC2`` draw unchanged).
* **Linked-married side** -- candidate 4's single-year OBSERVABLE married
  custodial probability ALREADY declines from ~0.85 at child age 18 to ~0.14 at
  age 30 over the 18-30 window: it IS the coresident-adult-child home-exit
  timing for linked-married children (the forensics' "faithful married custodial
  basis"), applied UNCHANGED. A hard leave ON TOP would DOUBLE-COUNT the aging-
  out already in the declining prob and over-drain the older married-linked male
  cells (55-64|male, 65-74|male), contradicting the registration's "modal
  failure: 35-44|male alone". The 35-44|male over-production is therefore the
  linked-father child SUPPLY (not the timing), deferred to candidate 7 as the
  registration anticipated ("if the single-year exit refit does not drain it, c7
  needs a forensics-4 look at linked-father child supply").

The multigen coupling stays at 55+ EXACTLY as candidate 5 fitted it -- extending
it downward to 45-54 is a spec violation (forensics-3: the reference coupling
there is weak, lift x1.75 vs the ~5x at 55+, and the cell already OVER-
produces).

Delta 3 -- **female cohabitation lift at 25-34** (``coresident_spouse``
25-34|female, forensics-3 Q9). The 25-34|female miss is a cohabitation-OVERLAY
shortfall (overlay gap -0.045), NOT the under-produced-legal mechanism of the
male bands (the legal core there is +0.011 over). Candidate 6 re-fits the
single-year cohabitation entry/exit hazards for FEMALES over ages 25-44 (the
same single-year estimator candidate 4's delta 1 used for 15-34) and applies
them as a female override; the candidate-4 legal top-up is explicitly NOT
applied (it would push the already-adequate legal core further over). Same
``0xC2`` cohabitation draw shape (byte-identical); only the female 25-44
entry/exit probabilities move.

Delta 4 -- **count-conditional bridge** (``hh_size`` 3/4/5+, forensics-3 Q10).
Candidate 5 conditioned the non-core INCIDENCE on core size (delta 3a) but drew
the 2+ non-core COUNT from the (band, sex) table INDEPENDENTLY of core size.
Candidate 6 draws the FULL non-core member count from the train joint
``P(non-core count = j | capped core size = k)`` -- the parameterization
forensics-3 PROVED clears hh3/hh4/hh5+ simultaneously on the sim's own core
distribution (0.1887/0.1709/0.1303, all in tolerance). The count draw consumes
the ``0xC3`` non-family stream BY SHAPE ONLY (one uniform per ego, exactly as
candidate 5's class draw); the ``0xC4`` 2+ count stream is retired (candidate
5's legal-residual ``0xC4`` draw is preserved so its spawn stays byte-
identical). The secondary core-5+ fertility deficit is named, carried, and NOT
chased (the honest joint absorbs it).
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
from populace_dynamics.models import household_composition_sim_v5 as hcs5

__all__ = [
    "GRANDCHILD_LO",
    "CORE_SIZE_CAP",
    "CHILD_EXIT_REFIT_LO",
    "CHILD_EXIT_REFIT_HI",
    "COHAB_FEMALE_REFIT_LO",
    "COHAB_FEMALE_REFIT_HI",
    "CUSTODIAL_REVERT_BAND",
    "DELTA_STREAM_TAG_V5",
    "DELTA_STREAM_TAG_V6",
    "HouseholdCompositionModelV6",
    "fit_child_exit_single_year",
    "fit_female_cohab_single_year",
    "fit_nonfamily_count_by_core",
    "fit_household_model_v6",
    "custodial_prob_v6",
    "child_leave_years_refit",
    "custodial_linked_child_counts_v6",
    "sample_nonfamily_count_v6",
    "simulate_draw_v6",
    "bridge_feasibility_convolution",
    "c6_delta_checks",
]

#: Delta 1 couples the multigen state and a coresident own adult child for egos
#: at or above this age (candidate 5, CARRIED). The multigen coupling stays at
#: 55+ EXACTLY; extending it downward to 45-54 is a spec violation (forensics-3:
#: the reference coupling there is weak and the cell already over-produces).
GRANDCHILD_LO = hcs5.GRANDCHILD_LO  # 55

#: Delta 4 caps the family-core size the non-family bridge conditions on (a core
#: of 5 or more shares the ``5+`` count conditional); candidate 5, CARRIED.
CORE_SIZE_CAP = hcs5.CORE_SIZE_CAP  # 5

#: Delta 2 re-fits the coresident-adult-child home-exit hazard on this single-
#: year child-age range (the reference own-child age composition puts the
#: aging-out mass here).
CHILD_EXIT_REFIT_LO = 18
CHILD_EXIT_REFIT_HI = 30

#: Delta 3 re-fits the FEMALE single-year cohabitation entry/exit on this age
#: range (the overlay shortfall band; candidate 4 fit single-year only to 34).
COHAB_FEMALE_REFIT_LO = 25
COHAB_FEMALE_REFIT_HI = 44

#: Delta 1 reverts the not-married custodial cell to the observable basis at
#: this child age band ONLY; the child-record swap stays for 5-17|not-married.
CUSTODIAL_REVERT_BAND: tuple[int, int] = (0, 4)

#: Candidate-5 isolated RNG tag (the coupling + parent-count draws), CARRIED.
DELTA_STREAM_TAG_V5 = hcs5.DELTA_STREAM_TAG_V5  # 0xC5

#: Candidate-6 isolated RNG tag: the one new stochastic component (the maternal
#: single-year leave-year refit). Isolated from 0xB2B / 0xC2 / 0xC3 / 0xC4 /
#: 0xC5 so every carried stream is byte-identical to candidate 5.
DELTA_STREAM_TAG_V6 = 0xC6

_MIN_STRATUM_N = hcs4._MIN_STRATUM_N  # 20
_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED

#: The biennial leave grid (candidate 1's ``_child_leave_years`` steps) that
#: falls inside the delta-2 single-year refit window; the maternal spline is
#: overridden and the linked-married leave is drawn on exactly these ages.
_REFIT_GRID_AGES: tuple[int, ...] = tuple(
    a
    for a in range(hcs.CHILD_MIN_LEAVE_AGE, hcs.CHILD_MAX_LEAVE_AGE, 2)
    if CHILD_EXIT_REFIT_LO <= a <= CHILD_EXIT_REFIT_HI
)


@dataclass
class HouseholdCompositionModelV6:
    """Candidate 5's byte-faithful bundle plus the four candidate-6 deltas.

    ``base_v5`` is the byte-faithful candidate-5
    :class:`~populace_dynamics.models.household_composition_sim_v5.
    HouseholdCompositionModelV5` (which carries candidates 4, 3, 2 and 1 and the
    multigen--adult-child coupling and per-ego parent-count composition). The
    new fitted components (side B only):

    * ``child_exit_single_year`` -- ``P(coresident adult child exits home | ego
      own age in [18, 30], sex)``, the single-year home-exit hazard (delta 2),
      with the candidate-1 spline baked in as the fallback so every grid age is
      total.
    * ``cohab_entry_age_female`` / ``cohab_exit_age_female`` -- the single-year
      FEMALE cohabitation entry / exit hazards over 25-44 (delta 3).
    * ``nonfamily_count_by_core`` -- ``P(non-core member count = j | capped core
      size = k)`` as ``(counts, cumulative)`` per capped core k in 1..5 (delta
      4); replaces candidate 5's incidence-conditional bridge.
    """

    base_v5: hcs5.HouseholdCompositionModelV5
    child_exit_single_year: dict[tuple[int, str], float]
    cohab_entry_age_female: dict[int, float]
    cohab_exit_age_female: dict[int, float]
    nonfamily_count_by_core: dict[int, tuple[np.ndarray, np.ndarray]]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- pass-through accessors to the carried candidate bundles ----
    @property
    def base_v4(self) -> hcs4.HouseholdCompositionModelV4:
        return self.base_v5.base_v4

    @property
    def base_v3(self) -> hcs3.HouseholdCompositionModelV3:
        return self.base_v5.base_v3

    @property
    def base_v2(self) -> hcs2.HouseholdCompositionModelV2:
        return self.base_v5.base_v2

    @property
    def base(self) -> hcs.HouseholdCompositionModel:
        return self.base_v5.base

    @property
    def father_links(self) -> pd.DataFrame:
        return self.base_v5.father_links

    @property
    def custodial_child_record(self) -> dict[tuple[str, str], float]:
        return self.base_v5.custodial_child_record

    @property
    def coupling_child_given_multigen(
        self,
    ) -> dict[tuple[str, str, bool], float]:
        return self.base_v5.coupling_child_given_multigen

    @property
    def coupling_child_pooled(self) -> dict[tuple[str, bool], float]:
        return self.base_v5.coupling_child_pooled

    @property
    def parent_count_two_share(self) -> dict[tuple[str, str], float]:
        return self.base_v5.parent_count_two_share

    @property
    def parent_count_two_pooled(self) -> float:
        return self.base_v5.parent_count_two_pooled


# --------------------------------------------------------------------------
# Delta 2 fit: single-year coresident-adult-child home-exit hazard (18-30)
# --------------------------------------------------------------------------
def fit_child_exit_single_year(
    person_waves: pd.DataFrame,
    parental_exit: hcs.ParentalExitModel,
    train_ids: set[int],
) -> tuple[dict[tuple[int, str], float], dict[str, Any]]:
    """Train the single-year ``P(home exit | coresident adult child age, sex)``.

    The coresident adult child IS a coresident-parent person-wave (a young
    adult still living with a parent). Same at-risk / event construction as
    :func:`hcs.fit_parental_exit` (``coresident_parent`` True with an observed
    next wave; event = the next wave has no coresident parent; weighted), but
    read as a single-year child-age hazard over ``[CHILD_EXIT_REFIT_LO,
    CHILD_EXIT_REFIT_HI]``. A single-year (age, sex) stratum thinner than
    :data:`_MIN_STRATUM_N` falls back to candidate 1's spline prediction at that
    age; the fallback is baked into every grid age so the apply lookup is total.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)]
    at_risk = pw[pw["coresident_parent"] & pw["has_next"]]
    table: dict[tuple[int, str], float] = {}
    diag_emp: dict[str, dict[str, float | int]] = {}
    for age in range(CHILD_EXIT_REFIT_LO, CHILD_EXIT_REFIT_HI + 1):
        for sex in hc.SEXES:
            sub = at_risk[(at_risk["age"] == age) & (at_risk["sex"] == sex)]
            spline = float(
                parental_exit.predict(
                    np.array([float(age)]),
                    np.array([1.0 if sex == "male" else 0.0]),
                )[0]
            )
            if len(sub) >= _MIN_STRATUM_N:
                ev = (
                    sub["next_coresident_parent"]
                    .eq(False)
                    .to_numpy(np.float64)
                )
                rate = hcs._weighted_rate(sub, ev)
            else:
                rate = spline
            table[(age, sex)] = rate
            diag_emp[f"{age}|{sex}"] = {
                "empirical": round(rate, 5),
                "spline": round(spline, 5),
                "n_atrisk": int(len(sub)),
            }
    diag = {
        "refit_age_range": [CHILD_EXIT_REFIT_LO, CHILD_EXIT_REFIT_HI],
        "grid_ages_biennial": list(_REFIT_GRID_AGES),
        "single_year_hazard_vs_spline": diag_emp,
        "n_at_risk_waves_train": int(len(at_risk)),
    }
    return table, diag


# --------------------------------------------------------------------------
# Delta 3 fit: single-year FEMALE cohabitation entry/exit (25-44)
# --------------------------------------------------------------------------
def fit_female_cohab_single_year(
    person_waves: pd.DataFrame,
    cohab_flag: pd.DataFrame,
    train_ids: set[int],
    band_entry: dict[tuple[str, str], float],
    band_exit: dict[tuple[str, str], float],
) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    """Train FEMALE code-22 entry / exit by single year of age over 25-44.

    Same estimator as :func:`hcs4.fit_cohab_single_year` (entry among not-
    cohabiting at-risk waves, exit among cohabiting at-risk waves, weighted),
    restricted to females and extended to age 44 (candidate 4 fit single-year
    only to 34). A single-year stratum thinner than :data:`_MIN_STRATUM_N`
    weighted at-risk waves falls back to the carried candidate-2 female band
    hazard.
    """
    pw = hcs2.attach_cohabitation(person_waves, cohab_flag)
    pw = pw[pw["person_id"].isin(train_ids)]
    hasn = pw[pw["has_next"] & pw["band"].notna()]
    entry: dict[int, float] = {}
    exit_: dict[int, float] = {}
    diag_rows: dict[str, dict[str, float | int]] = {}
    for age in range(COHAB_FEMALE_REFIT_LO, COHAB_FEMALE_REFIT_HI + 1):
        sub = hasn[(hasn["age"] == age) & (hasn["sex"] == "female")]
        band = hc.band_label(*hcs4._band_bounds(age))
        fb_e = band_entry.get((band, "female"), 0.0)
        fb_x = band_exit.get((band, "female"), 0.0)
        ep = sub[~sub["cohabiting"]]
        xp = sub[sub["cohabiting"]]
        e = (
            hcs._weighted_rate(
                ep, ep["next_cohabiting"].fillna(False).to_numpy(np.float64)
            )
            if len(ep) >= _MIN_STRATUM_N
            else fb_e
        )
        x = (
            hcs._weighted_rate(
                xp, xp["next_cohabiting"].eq(False).to_numpy(np.float64)
            )
            if len(xp) >= _MIN_STRATUM_N
            else fb_x
        )
        entry[age] = e
        exit_[age] = x
        diag_rows[str(age)] = {
            "entry": round(e, 5),
            "exit": round(x, 5),
            "equilibrium": round(e / (e + x), 5) if (e + x) > 0 else 0.0,
            "n_entry_atrisk": int(len(ep)),
            "n_exit_atrisk": int(len(xp)),
        }
    diag = {
        "refit_age_range": [COHAB_FEMALE_REFIT_LO, COHAB_FEMALE_REFIT_HI],
        "female_single_year": diag_rows,
    }
    return entry, exit_, diag


# --------------------------------------------------------------------------
# Delta 4 fit: full non-core member count conditional on capped core size
# --------------------------------------------------------------------------
def fit_nonfamily_count_by_core(
    person_waves: pd.DataFrame,
    fu_sizes: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Train ``P(non-core member count = j | capped core size = k)`` in full.

    The non-core count is enumerated ``hh_size`` minus the ego family-unit size
    (the core), clipped at 0; the core is capped at :data:`CORE_SIZE_CAP`. The
    full count distribution (not just 0/1/2+ incidence) is fit per capped core k
    in 1..5 as a ``(counts, cumulative)`` pair for the inverse-CDF draw. This is
    the forensics-3 Q10 parameterization proven to clear hh3/hh4/hh5+
    simultaneously on the sim's own core distribution.
    """
    pw = person_waves[person_waves["person_id"].isin(train_ids)][
        ["person_id", "year", "weight", "hh_size"]
    ].merge(fu_sizes, on=["person_id", "year"], how="left")
    pw["family_unit_size"] = pw["family_unit_size"].fillna(1).astype("int64")
    resid = np.clip(
        pw["hh_size"].to_numpy() - pw["family_unit_size"].to_numpy(), 0, None
    ).astype(np.int64)
    capped = np.clip(
        pw["family_unit_size"].to_numpy(), 1, CORE_SIZE_CAP
    ).astype(np.int64)
    weight = pw["weight"].to_numpy(np.float64)
    table: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    incidence: dict[str, float] = {}
    mean_count: dict[str, float] = {}
    for k in range(1, CORE_SIZE_CAP + 1):
        mask = capped == k
        wk = weight[mask]
        rk = resid[mask]
        tot = float(wk.sum())
        if tot <= 0:
            table[k] = (np.array([0], dtype=np.int64), np.array([1.0]))
            incidence[str(k)] = 0.0
            mean_count[str(k)] = 0.0
            continue
        vals = np.arange(0, int(rk.max()) + 1, dtype=np.int64)
        probs = np.array(
            [float(wk[rk == j].sum() / tot) for j in vals], dtype=np.float64
        )
        cum = np.cumsum(probs)
        cum[-1] = 1.0  # guard the inverse-CDF tail against float drift
        table[k] = (vals, cum)
        incidence[str(k)] = round(float(1.0 - probs[0]), 5)
        mean_count[str(k)] = round(float((vals * probs).sum()), 5)
    diag = {
        "core_size_cap": CORE_SIZE_CAP,
        "noncore_incidence_by_capped_core": incidence,
        "mean_noncore_count_by_capped_core": mean_count,
        "max_noncore_count_by_core": {
            str(k): int(table[k][0][-1]) for k in table
        },
    }
    return table, diag


# --------------------------------------------------------------------------
# Fit (train / side B only for every fitted parameter)
# --------------------------------------------------------------------------
def fit_household_model_v6(
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
) -> HouseholdCompositionModelV6:
    """Fit candidate 5 (byte-faithful) plus the four candidate-6 deltas."""
    base_v5 = hcs5.fit_household_model_v5(
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
        child_record_expo=child_record_expo,
        parent_counts=parent_counts,
    )
    if fu_sizes is None:
        fu_sizes = hcs3.family_unit_sizes(rel_map)

    # Delta 2: single-year child-age home-exit hazard (18-30).
    child_exit, exit_diag = fit_child_exit_single_year(
        hh.person_waves, base_v5.base.parental_exit, train_ids
    )
    # Delta 3: single-year FEMALE cohabitation entry/exit (25-44).
    cohab_entry_f, cohab_exit_f, cohab_diag = fit_female_cohab_single_year(
        hh.person_waves,
        base_v5.base_v2.cohab_flag,
        train_ids,
        base_v5.base_v2.cohab_entry,
        base_v5.base_v2.cohab_exit,
    )
    # Delta 4: full non-core count conditional on capped core size.
    nonfamily_count, nfc_diag = fit_nonfamily_count_by_core(
        hh.person_waves, fu_sizes, train_ids
    )

    meta = {
        **base_v5.meta,
        "child_exit_single_year": exit_diag,
        "female_cohab_single_year": cohab_diag,
        "nonfamily_count_by_core": nfc_diag,
        "custodial_revert_band": list(CUSTODIAL_REVERT_BAND),
        "grandchild_coupling_age_lo": GRANDCHILD_LO,
        "child_exit_refit_range": [CHILD_EXIT_REFIT_LO, CHILD_EXIT_REFIT_HI],
        "cohab_female_refit_range": [
            COHAB_FEMALE_REFIT_LO,
            COHAB_FEMALE_REFIT_HI,
        ],
    }
    return HouseholdCompositionModelV6(
        base_v5=base_v5,
        child_exit_single_year=child_exit,
        cohab_entry_age_female=cohab_entry_f,
        cohab_exit_age_female=cohab_exit_f,
        nonfamily_count_by_core=nonfamily_count,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Delta 1 apply: custodial per-wave coresidence with the 0-4 basis revert
# --------------------------------------------------------------------------
def custodial_prob_v6(
    model: HouseholdCompositionModelV6, age: int, era: str, marital: str
) -> float:
    """Custodial coresidence probability with the delta-1 0-4 revert.

    NOT-married fathers use the child-record-basis rate by child age band EXCEPT
    at the 0-4 band (delta 1: reverted to the observable basis, candidate 4's
    father-wave lookup, because the child-record 0-4 gap is a join-denominator
    artifact and the reference cell is ego-anchored). MARRIED fathers use
    candidate 4's observable lookup UNCHANGED.
    """
    if marital == _NOT_MARRIED:
        lo, hi = hcs4._child_band_bounds(age)
        if (lo, hi) != CUSTODIAL_REVERT_BAND:
            band = hc.band_label(lo, hi)
            if (band, _NOT_MARRIED) in model.custodial_child_record:
                return model.custodial_child_record[(band, _NOT_MARRIED)]
    return hcs4._custodial_prob(model.base_v4, age, era, marital)


# --------------------------------------------------------------------------
# Delta 2 apply: single-year maternal own-birth leave refit
# --------------------------------------------------------------------------
def child_leave_years_refit(
    births: pd.DataFrame,
    parental_exit: hcs.ParentalExitModel,
    child_exit_single_year: dict[tuple[int, str], float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw each child's home-leaving year with the delta-2 18-30 exit refit.

    Mirrors :func:`hcs._child_leave_years` EXACTLY (same biennial grid, same
    uniform child-sex draw, same open top) but overrides the spline hazard with
    the fitted single-year hazard at ages in ``[CHILD_EXIT_REFIT_LO,
    CHILD_EXIT_REFIT_HI]`` (by drawn child sex). Draws on the isolated ``0xC6``
    stream, so the shadow leave-year (kept on the candidate-5 ``0xC2`` draw)
    stays byte-identical.
    """
    n = len(births)
    if n == 0:
        return births.assign(leave_year=np.array([], dtype=np.int64))
    child_male = rng.random(n) < 0.5
    leave_age = np.full(n, hcs.CHILD_MAX_LEAVE_AGE, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    ages = list(range(hcs.CHILD_MIN_LEAVE_AGE, hcs.CHILD_MAX_LEAVE_AGE, 2))
    for age in ages:
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        males = child_male[idx]
        if CHILD_EXIT_REFIT_LO <= age <= CHILD_EXIT_REFIT_HI:
            prob = np.array(
                [
                    child_exit_single_year[(age, "male" if m else "female")]
                    for m in males
                ],
                dtype=np.float64,
            )
        else:
            prob = parental_exit.predict(
                np.full(idx.size, float(age)), males.astype(np.float64)
            )
        u = rng.random(idx.size)
        left = u < prob
        left_idx = idx[left]
        leave_age[left_idx] = age
        alive[left_idx] = False
    birth_year = births["birth_year"].to_numpy(dtype=np.int64)
    return births.assign(leave_year=birth_year + leave_age)


def custodial_linked_child_counts_v6(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    model: HouseholdCompositionModelV6,
    custodial_rng: np.random.Generator,
) -> np.ndarray:
    """Per side-A person-wave, the count of coresident father-linked children.

    Identical exposure construction and custodial draw SHAPE to
    :func:`hcs5.custodial_linked_child_counts_v5` (byte-identical ``0xC3``
    custodial stream), with ONLY the delta-1 0-4 not-married revert in
    :func:`custodial_prob_v6`. The linked-married exit timing (delta 2, the
    other parent side) is NOT a hard leave: candidate 4's single-year observable
    MARRIED custodial probability already declines from ~0.85 at child age 18 to
    ~0.14 at age 30 -- it IS the coresident-adult-child home-exit timing (the
    forensics' "faithful married custodial basis"), applied UNCHANGED. Adding a
    hard leave on top would double-count the aging-out and over-drain the older
    married-linked male cells; the 35-44|male residual is the linked-father
    child SUPPLY, deferred to candidate 7. So this function is byte-identical to
    candidate 5 EXCEPT the not-married 0-4 revert (which only moves the not-
    married 0-4 rows).
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
    # Byte-identical row order to candidate 5 (candidate-3 order).
    expo = expo.sort_values(["parent_person_id", "birth_year", "_row"])
    expo = expo.reset_index(drop=True)
    ages = expo["child_age"].to_numpy()
    years = expo["year"].to_numpy()
    marital = expo["marital"].to_numpy()
    prob = np.array(
        [
            custodial_prob_v6(model, int(a), hcs4.era_of_year(int(y)), str(m))
            for a, y, m in zip(ages, years, marital, strict=True)
        ],
        dtype=np.float64,
    )
    u = custodial_rng.random(
        len(expo)
    )  # 0xC3 custodial stream, byte-identical
    coresident = u < prob
    rows = expo["_row"].to_numpy()[coresident]
    np.add.at(counts, rows, 1)
    return counts


# --------------------------------------------------------------------------
# Delta 4 apply: full non-core count conditional on the simulated core size
# --------------------------------------------------------------------------
def sample_nonfamily_count_v6(
    pw: pd.DataFrame,
    model: HouseholdCompositionModelV6,
    count_rng: np.random.Generator,
    core_size: np.ndarray,
) -> np.ndarray:
    """Draw the FULL non-core member count keyed on the SIMULATED core size.

    Inverse-CDF draw of ``P(non-core count = j | capped core size)`` from the
    train joint (delta 4). Consumes ONE uniform per ego on the ``0xC3`` non-
    family stream -- byte-identical in shape to candidate 5's 0/1/2+ class draw
    -- so the candidate-5 custodial / skip-gen ``0xC3`` streams stay byte-
    identical; the candidate-5 ``0xC4`` 2+ count draw is retired.
    """
    n = len(pw)
    core = np.clip(np.asarray(core_size, dtype=np.int64), 1, CORE_SIZE_CAP)
    u = count_rng.random(n)  # 0xC3 non-family stream (byte-identical shape)
    counts = np.zeros(n, dtype=np.int64)
    for k in range(1, CORE_SIZE_CAP + 1):
        mask = core == k
        if not mask.any():
            continue
        vals, cum = model.nonfamily_count_by_core[k]
        idx = np.searchsorted(cum, u[mask], side="left")
        idx = np.clip(idx, 0, len(vals) - 1)
        counts[mask] = vals[idx]
    return counts


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def simulate_draw_v6(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModelV6,
    ids_a: set[int],
    draw_seed: int,
    occupancy_stream_tag: int = 0xB2B,
    delta_stream_tag_v2: int = 0xC2,
    delta_stream_tag_v3: int = 0xC3,
    delta_stream_tag_v4: int = 0xC4,
    delta_stream_tag_v5: int = DELTA_STREAM_TAG_V5,
    delta_stream_tag_v6: int = DELTA_STREAM_TAG_V6,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-6 draw of the side-A holdout households.

    Reproduces candidate 5's draw byte-for-byte on every carried stream
    (``0xB2B`` / ``0xC2`` / ``0xC3`` / ``0xC4`` / ``0xC5``), with the four
    candidate-6 deltas: the custodial 0-4 not-married revert (delta 1, byte-
    identical ``0xC3`` shape); the maternal single-year leave refit and the
    linked-married adult-child leave (delta 2, on the isolated ``0xC6``); the
    female 25-44 cohabitation override (delta 3, byte-identical ``0xC2`` shape);
    and the count-conditional non-family bridge (delta 4, byte-identical
    ``0xC3`` shape). The multigen coupling stays at 55+ EXACTLY as candidate 5.
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

    # 3. candidate-4 DELTA 1 (carried): age-refined cohabitation overlay, then
    #    candidate-6 DELTA 3: the female 25-44 single-year override.
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
    # DELTA 3: female 25-44 single-year cohabitation override.
    female_mat = sex_mat == "female"
    for age, rate in model.cohab_entry_age_female.items():
        cohab_entry_prob[(age_mat == age) & female_mat] = rate
    for age, rate in model.cohab_exit_age_female.items():
        cohab_exit_prob[(age_mat == age) & female_mat] = rate
    cohab_state = hcs._evolve_two_state(
        valid, obs_cohab, cohab_entry_prob, cohab_exit_prob, cohab_rng
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]

    # 4. candidate-4 delta substreams (0xC4), consumed EXACTLY as candidate 4
    #    for the legal-spouse residual overlay (candidate 5 carried it; the
    #    candidate-5 non-family 2+ spawn is retired by delta 4 but still spawned
    #    so the legal-residual stream stays byte-identical).
    c4_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v4])
    legal_resid_ss, _nonfamily2_ss = c4_ss.spawn(2)
    legal_resid_rng = np.random.default_rng(legal_resid_ss)

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

    # 5. certified marital core + maternal births (same draw seed as cand 5).
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

    # candidate-5 base leave-year draw (0xC2) -- kept byte-identical so the
    # SHADOW leave-year is unchanged; the maternal rows are re-drawn by delta 2.
    base_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    # candidate-3 delta substreams (0xC3), isolated from 0xB2B/0xC2.
    c3_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v3])
    custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    custodial_rng = np.random.default_rng(custodial_ss)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # candidate-5 delta substreams (0xC5): coupling + parent count (CARRIED).
    c5_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)

    # candidate-6 delta substream (0xC6): the maternal single-year leave refit.
    # Isolated from every carried stream.
    mat_leave_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, delta_stream_tag_v6])
    )

    # DELTA 2 (maternal): re-draw the maternal (own-birth) leave-years with the
    # single-year 18-30 exit refit on 0xC6; the shadow leave stays byte-
    # identical (candidate-5 0xC2 draw).
    maternal_births = all_births[all_births["_source"] == "maternal"]
    maternal_leaves = child_leave_years_refit(
        maternal_births,
        base.parental_exit,
        model.child_exit_single_year,
        mat_leave_rng,
    )
    nonlinked_leaves = pd.concat(
        [maternal_leaves, shadow_leaves], ignore_index=True
    )
    child_counts_nonlinked = hcs._coresident_child_counts(
        nonlinked_leaves, side_a_pw
    )

    # DELTA 1 (linked): custodial per-wave coresidence with the 0-4 not-married
    # revert. Byte-identical 0xC3 custodial stream (only the not-married 0-4
    # probability moves). The married linked exit timing is candidate 4's
    # single-year observable custodial decline UNCHANGED (delta 2, other side).
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked = custodial_linked_child_counts_v6(
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

    # DELTA 3b of candidate 5 (CARRIED): per-ego coresident-parent count (1 vs 2)
    # on the 0xC5 stream, feeding hh_size ONLY.
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

    # DELTA 1 of candidate 5 (CARRIED): multigen--adult-child coupling for 55+
    # egos on the 0xC5 stream; the multigen marginal is UNCHANGED and the
    # coupling is applied ONLY at 55+ (never extended downward to 45-54). Feeds
    # the grandchild composition ONLY.
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

    # DELTA 4: non-family count from the train joint P(count | capped core).
    # Class/count draw byte-identical on 0xC3; the 0xC4 2+ count is retired.
    nonfamily_count = sample_nonfamily_count_v6(
        pw, model, nonfamily_rng, hh_size_base
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

    # ---- Per-delta instrumentation on the side-A draw. ----
    weight = pw["weight"].to_numpy(np.float64)
    mask55f = is_55_row & (sexes_row == "female")
    w55 = weight[mask55f]

    def _wshare(w: np.ndarray, hit: np.ndarray) -> float:
        tot = float(w.sum())
        return float(w[hit].sum() / tot) if tot > 0 else 0.0

    # core / non-core split for the delta-4 realized joint.
    core_capped = np.clip(hh_size_base, 1, CORE_SIZE_CAP)
    noncore_present = nonfamily_count > 0
    sim_core_dist = {
        str(k): _wshare(weight, hh_size_base == k) for k in range(1, 9)
    }
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
        "n_maternal_child_coresident_wave_units": int(
            child_counts_nonlinked.sum()
        ),
        "n_skipgen_person_waves_simulated": int(
            skipgen_row[row_of[valid]].sum()
        ),
        "n_coupled_grandchild_waves_simulated": int(
            grandchild_coupled[is_55_row].sum()
        ),
        "mean_nonfamily_count_simulated": float(nonfamily_count.mean()),
        # Delta 1 coupling check (55+ female), carried.
        "coupling_gc55f_den_wt": float(w55.sum()),
        "coupling_gc55f_joint_mg_child_notparent": _wshare(
            w55, grandchild_coupled[mask55f]
        ),
        "coupling_gc55f_union": _wshare(w55, coresident_grandchild[mask55f]),
        # Delta 2 aging-out check (45-54|female maternal adult vs minor split
        # is measured train-side in the forensics; here record the realized
        # 55+ no-coupling-extension invariant and the 45-54 grandchild).
        "no_coupling_below_55_max_p": (
            float(p_coupled[~is_55_row].max()) if (~is_55_row).any() else 0.0
        ),
        # Delta 4 realized non-core incidence by capped core.
        "noncore_incidence_by_core": {
            str(k): _wshare(
                weight[core_capped == k], noncore_present[core_capped == k]
            )
            for k in range(1, CORE_SIZE_CAP + 1)
        },
        "sim_core_size_distribution": sim_core_dist,
        "mean_n_parents_among_coresident_parent": (
            float(n_parents_ego[c1_parent].mean()) if c1_parent.any() else 0.0
        ),
    }
    return panel, diagnostics


# --------------------------------------------------------------------------
# Delta-specific fit-vs-raw checks (fit seed's train side B + forensics-3)
# --------------------------------------------------------------------------
def bridge_feasibility_convolution(
    nonfamily_count_by_core: dict[int, tuple[np.ndarray, np.ndarray]],
    core_size_distribution: dict[str, float],
) -> dict[str, float]:
    """Convolve the fitted count conditional with a core-size distribution.

    Reproduces the forensics-3 Q10 honest-joint counterfactual: for each core
    size k the fitted ``P(non-core = j | capped core)`` is convolved and binned
    into 1/2/3/4/5+. Used to record that the delta-4 parameterization clears
    hh3/hh4/hh5+ on the sim's own core distribution (and to reproduce
    0.1887/0.1709/0.1303 on the committed candidate-5 core distribution).
    """
    hh_imp = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5+": 0.0}
    for k_str, pk in core_size_distribution.items():
        k = int(k_str)
        vals, cum = nonfamily_count_by_core[min(k, CORE_SIZE_CAP)]
        probs = np.diff(np.concatenate([[0.0], cum]))
        for j, pj in zip(vals, probs, strict=True):
            size = k + int(j)
            key = str(size) if size <= 4 else "5+"
            hh_imp[key] += float(pk) * float(pj)
    return hh_imp


def c6_delta_checks(
    model: HouseholdCompositionModelV6,
    forensics: dict[str, Any],
) -> dict[str, Any]:
    """Record the four deltas' fit-vs-raw checks against forensics-3.

    Computed from the fitted objects on the given seed's train side B beside the
    forensics-3 measured quantities each delta implements (Q8 0-4 basis + exit
    channels; Q9 female overlay; Q10 count-conditional feasibility).
    """
    q8 = forensics["question_8_child_cell_triage"]
    q9 = forensics["question_9_spouse_25_34_female_decomposition"]
    q10 = forensics["question_10_hh_size_3_5plus_joint_constraint"]

    # Delta 1: the 0-4 revert basis (child-record vs observable at 0-4 NM).
    adj = q8["adjudication_0_4_basis_15_24_male"]
    delta_1 = {
        "revert_band": list(CUSTODIAL_REVERT_BAND),
        "child_record_stays_bands": ["5-12", "13-17"],
        "forensics_observable_0_4_nm": adj["not_married_0_4"][
            "observable_basis"
        ],
        "forensics_child_record_0_4_nm": adj["not_married_0_4"][
            "child_record_basis"
        ],
        "fitted_child_record_0_4_nm": round(
            model.custodial_child_record[
                (hc.band_label(*CUSTODIAL_REVERT_BAND), _NOT_MARRIED)
            ],
            5,
        ),
        "sign_inversion_confirmed": adj[
            "sign_inverts_between_0_4_and_school_ages"
        ],
        "note": (
            "Delta 1 reverts the not-married 0-4 custodial cell to the "
            "observable (father-wave) basis and keeps the child-record swap at "
            "5-17; the child-record 0-4 rate is a join-denominator artifact "
            "(higher than observable at 0-4, lower at school ages)."
        ),
    }

    # Delta 2: single-year exit hazard vs spline, both parent sides.
    exit_diag = model.meta["child_exit_single_year"][
        "single_year_hazard_vs_spline"
    ]
    grid = model.meta["child_exit_single_year"]["grid_ages_biennial"]
    male_lift = [
        exit_diag[f"{a}|male"]["empirical"] - exit_diag[f"{a}|male"]["spline"]
        for a in grid
    ]
    delta_2 = {
        "refit_grid_ages": grid,
        "single_year_hazard_vs_spline_at_grid": {
            f"{a}|{s}": exit_diag[f"{a}|{s}"] for a in grid for s in hc.SEXES
        },
        "mean_male_empirical_minus_spline_at_grid": round(
            float(np.mean(male_lift)), 5
        ),
        "applied_to": (
            "maternal own-birth leave-year refit (0xC6); linked-married side "
            "is candidate 4's single-year observable declining married custodial "
            "prob (the exit timing already in place)"
        ),
        "coupling_stays_at_55_plus": GRANDCHILD_LO == 55,
        "no_coupling_extension_to_45_54": True,
        "forensics_45_54f_over_produced": q8[
            "adjudication_retention_vs_aging_out_45_54_female"
        ]["cell_is_over_produced"],
        "forensics_45_54f_reference_coupling_lift_ratio": q8[
            "adjudication_retention_vs_aging_out_45_54_female"
        ]["reference_coupling_signature"][
            "reference_coupling_lift_ratio_joint_over_product"
        ],
        "linked_married_35_44m_is_supply_deferred_to_c7": True,
        "note": (
            "The single-year 18-30 exit hazard exceeds candidate 1's spline for "
            "sons; overriding the maternal leave ages maternal adult children "
            "out (the 45-54|female lever). The linked-married side is NOT a hard "
            "leave: candidate 4's single-year observable married custodial prob "
            "already declines ~0.85->~0.14 over 18-30 (it IS the exit timing), "
            "so a leave would double-count and over-drain the older male cells; "
            "35-44|male is the linked-father SUPPLY, deferred to c7. The "
            "multigen coupling stays at 55+ (the 45-54 reference coupling lift "
            "is x1.75, far below the ~5x at 55+ and the cell over-produces, so "
            "extending it downward is a spec violation)."
        ),
    }

    # Delta 3: female overlay gap and the fitted single-year female hazards.
    band25 = q9["per_female_band"]["coresident_spouse.25-34|female"]
    delta_3 = {
        "forensics_target_classification": q9["target_cell_classification"],
        "forensics_legal_gap_25_34f": q9["target_cell_legal_gap"],
        "forensics_overlay_gap_25_34f": q9["target_cell_overlay_gap"],
        "reference_code22_cohab_stock_25_34f": band25[
            "reference_code22_cohab_stock"
        ],
        "sim_cohab_overlay_contribution_25_34f_candidate5": band25[
            "sim_cohab_overlay_contribution"
        ],
        "fitted_female_entry_25_34": {
            str(a): round(model.cohab_entry_age_female[a], 5)
            for a in range(25, 35)
        },
        "fitted_female_exit_25_34": {
            str(a): round(model.cohab_exit_age_female[a], 5)
            for a in range(25, 35)
        },
        "legal_top_up_applied": False,
        "note": (
            "Delta 3 re-fits the FEMALE single-year cohabitation entry/exit "
            "over 25-44 (the overlay under-supplies the code-22 stock by "
            "-0.045); the legal top-up is NOT applied (the legal core is +0.011 "
            "over). At 25-34 the single-year female hazards coincide with "
            "candidate 4's train estimate; the new structure is the 35-44 "
            "single-year refit (candidate 4 fit single-year only to 34)."
        ),
    }

    # Delta 4: reproduce the forensics-3 Q10 feasibility (0.1887/0.1709/0.1303)
    # by convolving the fitted count conditional with the committed candidate-5
    # sim core distribution.
    sim_c5_core = q10["sim_c5_core_size_distribution"]
    implied_on_c5_core = bridge_feasibility_convolution(
        model.nonfamily_count_by_core, sim_c5_core
    )
    forensics_implied = q10["honest_joint_counterfactual"][
        "implied_hh_from_sim_core"
    ]
    ref_hh = q10["reference_hh_size_distribution"]
    tol = {
        "3": q10["honest_joint_counterfactual"]["per_cell"]["hh_size.3"][
            "tolerance"
        ],
        "4": q10["honest_joint_counterfactual"]["per_cell"]["hh_size.4"][
            "tolerance"
        ],
        "5+": q10["honest_joint_counterfactual"]["per_cell"]["hh_size.5+"][
            "tolerance"
        ],
    }

    def _clears(cell: str) -> bool:
        r = implied_on_c5_core[cell]
        a = ref_hh[cell]
        if r <= 0 or a <= 0:
            return False
        return abs(float(np.log(r / a))) <= tol[cell]

    delta_4 = {
        "fitted_noncore_incidence_by_core": model.meta[
            "nonfamily_count_by_core"
        ]["noncore_incidence_by_capped_core"],
        "forensics_reference_noncore_incidence_by_core": q10[
            "reference_noncore_incidence_by_capped_core"
        ],
        "implied_hh_from_committed_c5_core_MINE": {
            k: round(v, 5) for k, v in implied_on_c5_core.items()
        },
        "forensics_implied_hh_from_sim_core": {
            k: round(float(v), 5) for k, v in forensics_implied.items()
        },
        "reproduces_feasibility_0_1887_0_1709_0_1303": {
            "hh_size.3": _clears("3"),
            "hh_size.4": _clears("4"),
            "hh_size.5+": _clears("5+"),
            "all_three_clear_on_committed_c5_core": bool(
                _clears("3") and _clears("4") and _clears("5+")
            ),
        },
        "note": (
            "Delta 4 draws the FULL non-core count from the train joint "
            "P(count | capped core). Convolving the fitted conditional with the "
            "committed candidate-5 sim core distribution reproduces the "
            "forensics-3 Q10 feasibility (~0.1887/0.1709/0.1303, all three in "
            "tolerance); the small deviation from the forensics figures is the "
            "per-seed train conditional. The core-5+ fertility deficit is named "
            "and carried (the honest joint absorbs it)."
        ),
    }

    return {
        "delta_1_zero_four_revert": delta_1,
        "delta_2_adult_child_exit_timing": delta_2,
        "delta_3_female_cohab_lift": delta_3,
        "delta_4_count_conditional_bridge": delta_4,
    }
