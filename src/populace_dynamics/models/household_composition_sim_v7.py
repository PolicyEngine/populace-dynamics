"""Gate-2b candidate 7: enumeration conditioning and episode persistence.

Candidate 7 (registration #42 comment 4948186843) is candidate 6
(:mod:`populace_dynamics.models.household_composition_sim_v6`, merged in PR
#143) with EXACTLY TWO frozen deltas, each designed against a graded finding of
the gate-2b forensics-4 decomposition (``runs/gate2b_forensics4_v1.json``,
grading 4948183531). Everything candidate 6 cleared or carried -- the certified
tranche-2a marital core, the carried ``coresident_parent`` / ``coresident_spouse``
(including the fragile ``coresident_spouse.25-34|female`` cell, carried
UNTOUCHED with its 2/5 split-seed fragility on the record) / ``multigen`` (stock
+ transitions) / ``parental_home_exit``, the delta-1 multigen--adult-child
coupling at 55+ (which took ``coresident_grandchild.55+|female`` to 1.00), and
the delta-4 count-conditional bridge -- is carried BYTE-FAITHFULLY: candidate 7
REUSES candidate 6's generator (candidate 1's ``simulate_draw`` at ``0xB2B``;
the candidate-2 streams at ``0xC2``; the candidate-3 streams at ``0xC3``; the
candidate-4 legal-residual stream at ``0xC4``; the candidate-5 coupling /
parent-count streams at ``0xC5``; the candidate-6 maternal-leave stream at
``0xC6``) and re-runs its exact draws, then the two candidate-7 deltas REPLACE
the linked-father child coresidence draw with a new component drawn on an
ISOLATED ``SeedSequence([draw_seed, 0xC7])`` -- so every carried stream is
byte-identical to candidate 6.

Delta 1 -- **enumeration conditioning** (removes the dominant
``unenumerated_nonjoinable_supply`` channel, forensics-4 Q11: +0.035 at
25-34|male, +0.036 at 35-44|male). The committed candidate-6 linked draw
(:func:`hcs6.custodial_linked_child_counts_v6`) draws coresidence over
``model.father_links`` (``father_link_births``), which does NOT require an
ENUMERATED child: 25.8% of its exposure rows (9,500 of 36,887) are non-joinable
-- biological children with no enumerated household record, whom the reference
roster can never observe coresident. Candidate 7 restricts the paternal-linked
draw to ENUMERATED children (the ``(parent_person_id, birth_year)`` keys present
in ``father_links_child``); a non-joinable linked exposure row cannot be drawn
coresident, and its share is recorded per child band.

Delta 2 -- **episode persistence** (removes the ``spell_length`` fragmentation
channel, forensics-4 Q11: +0.018 at 25-34|male, +0.022 at 35-44|male). The
committed candidate-6 draw applies the faithful per-wave custodial probability
as an INDEPENDENT per-wave occupancy, fragmenting the coresidence spells (sim
mean episode 3.57 waves vs the reference 5.93; sim single-wave share 0.331 vs
0.165). Candidate 7 replaces the independent per-wave draw with a correlated
entry/persist/exit process: a per-father latent frailty ``Z_f`` (shared across
the father's children) that a fitted fraction ``rho`` of linked children follow
across all their waves (a persistent, sibling-synchronized episode -- coresident
while the faithful custodial probability exceeds the shared latent), while
``1 - rho`` of children keep the candidate-6 independent per-wave draw. The
mixture PRESERVES the per-wave custodial marginal by band EXACTLY (a child is
coresident at a wave with probability ``p_c`` whether it follows the shared
latent or the independent draw -- ``rho * p_c + (1 - rho) * p_c = p_c``), so the
faithful per-wave custody probabilities do NOT move; ``rho`` is fitted on train
to the episode-length distribution (target mean ~5.93 waves). The persistent,
sibling-synchronized fraction concentrates coresidence into fewer father-waves,
reshaping the father-wave stock (the spell channel) without touching the
per-wave marginal. The per-draw marginal-preservation check (simulated per-wave
marginal vs train, per band) is recorded.

The shadow (unlinked imputed-paternal) channel (forensics-4 +0.032/+0.060) is a
NAMED RESIDUAL, deliberately untouched -- it is offset by the marital-joint and
non-linked negatives, and touching it without a measurement would be tuning.
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
from populace_dynamics.models import household_composition_sim_v6 as hcs6

__all__ = [
    "GRANDCHILD_LO",
    "CORE_SIZE_CAP",
    "CUSTODIAL_REVERT_BAND",
    "CHILD_CORESIDENCE_MAX_AGE",
    "SPELL_CHILD_MAX_AGE",
    "DELTA_STREAM_TAG_V6",
    "DELTA_STREAM_TAG_V7",
    "EPISODE_BUCKETS",
    "HouseholdCompositionModelV7",
    "enumerated_joinable_keys",
    "episode_length_hist",
    "reference_linked_episode_stats",
    "simulate_linked_episode_coresidence",
    "fit_linked_episode_persistence",
    "fit_household_model_v7",
    "custodial_linked_child_counts_v7",
    "simulate_draw_v7",
    "c7_delta_checks",
]

#: Carried from candidate 6 (the multigen--adult-child coupling stays at 55+).
GRANDCHILD_LO = hcs6.GRANDCHILD_LO  # 55
#: Carried from candidate 6 (the non-family bridge core-size cap).
CORE_SIZE_CAP = hcs6.CORE_SIZE_CAP  # 5
#: Carried from candidate 6 (the delta-1 0-4 not-married custodial revert band).
CUSTODIAL_REVERT_BAND = hcs6.CUSTODIAL_REVERT_BAND  # (0, 4)
#: The oldest child age counted as a coresident linked child (candidate 3).
CHILD_CORESIDENCE_MAX_AGE = hcs3.CHILD_CORESIDENCE_MAX_AGE  # 60
#: Episode lengths are measured over MINOR child ages only (forensics-4).
SPELL_CHILD_MAX_AGE = 17

#: Candidate-6 isolated RNG tag (the maternal single-year leave refit), CARRIED.
DELTA_STREAM_TAG_V6 = hcs6.DELTA_STREAM_TAG_V6  # 0xC6
#: Candidate-7 isolated RNG tag: the one new stochastic component (the linked
#: episode-persistence draw). Isolated from 0xB2B / 0xC2 / 0xC3 / 0xC4 / 0xC5 /
#: 0xC6 so every carried stream is byte-identical to candidate 6.
DELTA_STREAM_TAG_V7 = 0xC7

#: Episode-length reporting buckets (1 / 2 / 3 / 4+ waves), forensics-4.
EPISODE_BUCKETS = ("1", "2", "3", "4+")

_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED
#: The child age bands the custodial marginal is recorded over (delta-2 check).
_CHILD_BANDS = tuple(
    hc.band_label(lo, hi) for lo, hi in hcs3.CUSTODIAL_CHILD_AGE_BANDS
)


@dataclass
class HouseholdCompositionModelV7:
    """Candidate 6's byte-faithful bundle plus the two candidate-7 deltas.

    ``base_v6`` is the byte-faithful candidate-6
    :class:`~populace_dynamics.models.household_composition_sim_v6.
    HouseholdCompositionModelV6` (which carries candidates 5, 4, 3, 2 and 1).
    The two candidate-7 components (both re-key the LINKED child coresidence
    draw; every other component is candidate 6 UNCHANGED):

    * ``joinable_keys`` -- the enumerated ``(parent_person_id, birth_year)``
      keys (from ``father_links_child``); a linked exposure row whose key is not
      here cannot be drawn coresident (delta 1, a deterministic filter, no RNG).
    * ``linked_episode_persistence`` -- ``rho``, the fitted share of linked
      children that follow the persistent, sibling-synchronized frailty episode
      (vs the candidate-6 independent per-wave draw); fitted on train side B to
      the episode-length mean (delta 2).
    """

    base_v6: hcs6.HouseholdCompositionModelV6
    joinable_keys: frozenset[tuple[int, int]]
    linked_episode_persistence: float
    episode_fit: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- pass-through accessors to the carried candidate bundles ----
    @property
    def base_v5(self) -> hcs5.HouseholdCompositionModelV5:
        return self.base_v6.base_v5

    @property
    def base_v4(self) -> hcs4.HouseholdCompositionModelV4:
        return self.base_v6.base_v4

    @property
    def base_v3(self) -> hcs3.HouseholdCompositionModelV3:
        return self.base_v6.base_v3

    @property
    def base_v2(self) -> hcs2.HouseholdCompositionModelV2:
        return self.base_v6.base_v2

    @property
    def base(self) -> hcs.HouseholdCompositionModel:
        return self.base_v6.base

    @property
    def father_links(self) -> pd.DataFrame:
        return self.base_v6.father_links

    @property
    def custodial_child_record(self) -> dict[tuple[str, str], float]:
        return self.base_v6.custodial_child_record

    @property
    def coupling_child_given_multigen(
        self,
    ) -> dict[tuple[str, str, bool], float]:
        return self.base_v6.coupling_child_given_multigen

    @property
    def coupling_child_pooled(self) -> dict[tuple[str, bool], float]:
        return self.base_v6.coupling_child_pooled

    @property
    def parent_count_two_share(self) -> dict[tuple[str, str], float]:
        return self.base_v6.parent_count_two_share

    @property
    def parent_count_two_pooled(self) -> float:
        return self.base_v6.parent_count_two_pooled

    @property
    def child_exit_single_year(self) -> dict[tuple[int, str], float]:
        return self.base_v6.child_exit_single_year

    @property
    def cohab_entry_age_female(self) -> dict[int, float]:
        return self.base_v6.cohab_entry_age_female

    @property
    def cohab_exit_age_female(self) -> dict[int, float]:
        return self.base_v6.cohab_exit_age_female

    @property
    def nonfamily_count_by_core(
        self,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return self.base_v6.nonfamily_count_by_core


# --------------------------------------------------------------------------
# Delta 1: enumerated (joinable) linked exposure keys
# --------------------------------------------------------------------------
def enumerated_joinable_keys(
    father_links_child: pd.DataFrame,
) -> frozenset[tuple[int, int]]:
    """The enumerated ``(parent_person_id, birth_year)`` linked-child keys.

    A linked biological child is ENUMERATED (joinable) when it has an
    observed household record -- i.e. a ``child_person_id`` in
    ``father_links_child`` (:func:`hcs3.father_link_births_with_child`). The
    committed candidate-6 draw coresides over ``model.father_links``
    (``father_link_births``), which does NOT require this; candidate 7
    restricts the draw to these keys (delta 1). Seed-independent: a property of
    the link records, estimated on neither split (no holdout tuning surface).
    """
    fac = father_links_child[["parent_person_id", "birth_year"]].copy()
    fac["parent_person_id"] = fac["parent_person_id"].astype("int64")
    fac["birth_year"] = fac["birth_year"].astype("int64")
    return frozenset(map(tuple, fac.drop_duplicates().to_numpy().tolist()))


# --------------------------------------------------------------------------
# Episode-length histogram (forensics-4 _episode_length_hist, byte-faithful)
# --------------------------------------------------------------------------
def episode_length_hist(
    father_id: np.ndarray,
    child_key: np.ndarray,
    year: np.ndarray,
    coresident: np.ndarray,
) -> tuple[dict[str, float], float, int]:
    """Weighted-free episode-length histogram over father-child coresidence.

    A byte-faithful copy of the forensics-4 ``_episode_length_hist``: an
    EPISODE is a maximal run of coresident waves adjacent in the year-ordered
    per ``(father, child-key)`` sequence. Returns the ``EPISODE_BUCKETS``
    distribution, the mean episode length, and the episode count.
    """
    n = len(father_id)
    empty = ({b: 0.0 for b in EPISODE_BUCKETS}, 0.0, 0)
    if n == 0:
        return empty
    df = pd.DataFrame(
        {
            "fid": np.asarray(father_id, dtype=np.int64),
            "ck": np.asarray(child_key, dtype=np.int64),
            "year": np.asarray(year, dtype=np.int64),
            "cor": np.asarray(coresident, dtype=bool),
        }
    )
    df = df.groupby(["fid", "ck", "year"], as_index=False)["cor"].max()
    df = df.sort_values(["fid", "ck", "year"], kind="stable")
    n = len(df)
    fid = df["fid"].to_numpy()
    ck = df["ck"].to_numpy()
    cor = df["cor"].to_numpy()
    new_pair = np.empty(n, dtype=bool)
    new_pair[0] = True
    new_pair[1:] = (fid[1:] != fid[:-1]) | (ck[1:] != ck[:-1])
    prev_cor = np.empty(n, dtype=bool)
    prev_cor[0] = False
    prev_cor[1:] = cor[:-1]
    start = cor & (new_pair | ~prev_cor)
    if not start.any():
        return empty
    epi_id = np.cumsum(start) - 1
    lengths = np.bincount(epi_id[cor])
    lengths = lengths[lengths > 0]
    total = int(lengths.size)
    if total == 0:
        return empty
    dist = {
        "1": float(np.sum(lengths == 1) / total),
        "2": float(np.sum(lengths == 2) / total),
        "3": float(np.sum(lengths == 3) / total),
        "4+": float(np.sum(lengths >= 4) / total),
    }
    return dist, float(lengths.mean()), total


# --------------------------------------------------------------------------
# Delta 2 target: the train reference episode-length distribution (0-17)
# --------------------------------------------------------------------------
def reference_linked_episode_stats(
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    father_waves: pd.DataFrame,
) -> tuple[dict[str, float], float, int]:
    """The reference episode-length distribution over JOINABLE linked pairs.

    Mirrors the forensics-4 ``q11_reference`` spell block: over the FULL
    father-wave grid (coresident True where the ``(father, child, year)`` pair
    is in ``parent_pairs``, False elsewhere), for enumerated children at child
    ages 0-17, the maximal runs of coresident waves. ``father_waves`` is the
    ``(person_id, year)`` grid of the fathers on the side being measured (train
    side B for the fit target). This is the delta-2 fit TARGET (mean ~5.93).
    """
    fac = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ].copy()
    fac["parent_person_id"] = fac["parent_person_id"].astype("int64")
    fac["child_person_id"] = fac["child_person_id"].astype("int64")
    fac["birth_year"] = fac["birth_year"].astype("int64")
    fw = father_waves[["person_id", "year"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    ref_epi = fac.merge(fw, on="parent_person_id", how="inner")
    ref_epi["child_age"] = ref_epi["year"] - ref_epi["birth_year"]
    ref_epi = ref_epi[
        (ref_epi["child_age"] >= 0)
        & (ref_epi["child_age"] <= SPELL_CHILD_MAX_AGE)
    ]
    pp_cor = parent_pairs[
        ["parent_person_id", "child_person_id", "year"]
    ].assign(_cor=True)
    ref_epi = ref_epi.merge(
        pp_cor,
        on=["parent_person_id", "child_person_id", "year"],
        how="left",
    )
    ref_epi["coresident"] = ref_epi["_cor"].fillna(False).astype(bool)
    return episode_length_hist(
        ref_epi["parent_person_id"].to_numpy(np.int64),
        ref_epi["birth_year"].to_numpy(np.int64),
        ref_epi["year"].to_numpy(np.int64),
        ref_epi["coresident"].to_numpy(bool),
    )


# --------------------------------------------------------------------------
# Delta 2 draw: the persistent, sibling-synchronized frailty episode
# --------------------------------------------------------------------------
def simulate_linked_episode_coresidence(
    father_id: np.ndarray,
    birth_id: np.ndarray,
    prob: np.ndarray,
    persistence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw per-exposure-row coresidence via the delta-2 episode process.

    The rows MUST be pre-sorted by ``(birth_id, year)`` so a child's waves are
    contiguous. A fraction ``persistence`` (``rho``) of children (drawn per
    ``birth_id``) follow a per-FATHER latent frailty ``Z_f`` (shared across the
    father's children) across ALL their waves -- coresident at a wave iff
    ``Z_f < p_c`` there -- a persistent, sibling-synchronized episode; the
    remaining ``1 - rho`` keep the candidate-6 INDEPENDENT per-wave draw
    (``V < p_c``). The mixture preserves the per-wave marginal EXACTLY:
    ``P(coresident) = rho * P(Z_f < p_c) + (1 - rho) * P(V < p_c) = p_c``
    (both ``Z_f`` and ``V`` are uniform, so each term is ``p_c``). Draws
    ``Z_f`` (per father), the persistence selector (per child), then ``V`` (per
    row), in that order -- an isolated stream, so it perturbs no carried draw.
    """
    n = len(prob)
    if n == 0:
        return np.zeros(0, dtype=bool)
    uniq_f, f_inv = np.unique(
        np.asarray(father_id, dtype=np.int64), return_inverse=True
    )
    uniq_b, b_inv = np.unique(
        np.asarray(birth_id, dtype=np.int64), return_inverse=True
    )
    z_father = rng.random(len(uniq_f))  # per-father shared frailty latent
    follows = rng.random(len(uniq_b)) < persistence  # per-child persistent?
    v_idio = rng.random(n)  # per-row independent draw (candidate-6 shape)
    u = np.where(follows[b_inv], z_father[f_inv], v_idio)
    return u < np.asarray(prob, dtype=np.float64)


def _linked_exposure_frame(
    linked_births: pd.DataFrame,
    father_waves: pd.DataFrame,
    marital: pd.DataFrame,
    base_v6: hcs6.HouseholdCompositionModelV6,
    joinable_keys: frozenset[tuple[int, int]],
) -> pd.DataFrame:
    """The linked father-wave x child exposure grid with per-row custody prob.

    Byte-faithful to :func:`hcs6.custodial_linked_child_counts_v6`'s exposure
    construction (merge over ``parent_person_id``; child age in
    ``[0, CHILD_CORESIDENCE_MAX_AGE]``; custodial band; father marital;
    candidate-3 row order), with a stable per-birth id (delta-2 episode key)
    and the delta-1 ``joinable`` flag. ``_row`` indexes ``father_waves`` (reset).
    """
    fw = father_waves.reset_index(drop=True)
    fw = fw.assign(_row=np.arange(len(fw), dtype=np.int64))
    fw = fw[["person_id", "year", "_row"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    lb = linked_births.reset_index(drop=True)
    lb = lb.assign(_birth_id=np.arange(len(lb), dtype=np.int64))
    expo = lb.merge(fw, on="parent_person_id", how="inner")
    if not len(expo):
        return expo.assign(
            child_age=np.array([], dtype=np.int64),
            child_band=np.array([], dtype=object),
            marital=np.array([], dtype=object),
            joinable=np.array([], dtype=bool),
            p_c=np.array([], dtype=np.float64),
        )
    expo["child_age"] = expo["year"] - expo["birth_year"]
    expo = expo[
        (expo["child_age"] >= 0)
        & (expo["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    expo["child_band"] = expo["child_age"].map(hcs3._child_band)
    expo = expo[expo["child_band"].notna()]
    if not len(expo):
        return expo.assign(
            marital=np.array([], dtype=object),
            joinable=np.array([], dtype=bool),
            p_c=np.array([], dtype=np.float64),
        )
    expo = expo.merge(
        marital.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    expo["marital"] = expo["marital"].fillna(_NOT_MARRIED)
    expo = expo.sort_values(["parent_person_id", "birth_year", "_row"])
    expo = expo.reset_index(drop=True)
    idx = pd.MultiIndex.from_arrays(
        [expo["parent_person_id"], expo["birth_year"]]
    )
    expo["joinable"] = np.asarray(idx.isin(joinable_keys), dtype=bool)
    expo["p_c"] = np.array(
        [
            hcs6.custodial_prob_v6(
                base_v6, int(a), hcs4.era_of_year(int(y)), str(m)
            )
            for a, y, m in zip(
                expo["child_age"].to_numpy(),
                expo["year"].to_numpy(),
                expo["marital"].to_numpy(),
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    return expo


def _expected_episode_mean_at_persistence(
    expo_joinable: pd.DataFrame,
    persistence: float,
    rng: np.random.Generator,
) -> float:
    """Mean simulated episode length (child 0-17) at a candidate ``rho``.

    Simulates the delta-2 episode coresidence over the (train) joinable
    exposure and measures the episode-length mean over minor child ages -- the
    monotone-in-``rho`` objective the fit inverts. Train-side only.
    """
    ex = expo_joinable.sort_values(["_birth_id", "year"]).reset_index(
        drop=True
    )
    coresident = simulate_linked_episode_coresidence(
        ex["parent_person_id"].to_numpy(np.int64),
        ex["_birth_id"].to_numpy(np.int64),
        ex["p_c"].to_numpy(np.float64),
        persistence,
        rng,
    )
    minor = ex["child_age"].to_numpy() <= SPELL_CHILD_MAX_AGE
    _dist, mean, _n = episode_length_hist(
        ex["parent_person_id"].to_numpy(np.int64)[minor],
        ex["birth_year"].to_numpy(np.int64)[minor],
        ex["year"].to_numpy(np.int64)[minor],
        coresident[minor],
    )
    return mean


def fit_linked_episode_persistence(
    hh: hc.HouseholdCompositionPanel,
    base_v6: hcs6.HouseholdCompositionModelV6,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    joinable_keys: frozenset[tuple[int, int]],
    train_ids: set[int],
    *,
    fit_seed: int = 0xC7,
    n_bisect: int = 24,
) -> tuple[float, dict[str, Any]]:
    """Fit the delta-2 persistence ``rho`` to the train episode-length mean.

    Measures the TARGET (the reference episode-length mean over joinable linked
    pairs on train side B, child ages 0-17, from ``parent_pairs``), then
    bisects ``rho`` in ``[0, 1]`` so the SIMULATED episode-length mean matches
    it (the mean is monotone increasing in ``rho`` -- ``rho = 0`` is the
    candidate-6 independent per-wave draw, ``rho = 1`` the fully persistent,
    sibling-synchronized episode). If the target exceeds the ``rho = 1`` mean,
    ``rho`` caps at 1. Every quantity is estimated on side B only.
    """
    train_pw = hh.person_waves[hh.person_waves["person_id"].isin(train_ids)]
    father_waves = train_pw[["person_id", "year"]]
    ref_dist, ref_mean, ref_n = reference_linked_episode_stats(
        father_links_child, parent_pairs, father_waves
    )
    # The committed linked exposure basis (== the sim's paternal_linked source),
    # restricted to train fathers -- the exposure the fit simulates over.
    fl = base_v6.father_links[["parent_person_id", "birth_year"]].copy()
    fl["parent_person_id"] = fl["parent_person_id"].astype("int64")
    fl["birth_year"] = fl["birth_year"].astype("int64")
    fl = fl[fl["parent_person_id"].isin(train_ids)]
    expo = _linked_exposure_frame(
        fl, father_waves, marital_by_year, base_v6, joinable_keys
    )
    expo_j = expo[expo["joinable"].to_numpy()].copy()

    def sim_mean(rho: float) -> float:
        return _expected_episode_mean_at_persistence(
            expo_j, rho, np.random.default_rng([fit_seed, 0])
        )

    mean_lo = sim_mean(0.0)
    mean_hi = sim_mean(1.0)
    if ref_mean <= mean_lo:
        rho = 0.0
    elif ref_mean >= mean_hi:
        rho = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if sim_mean(mid) < ref_mean:
                lo = mid
            else:
                hi = mid
        rho = 0.5 * (lo + hi)
    achieved = sim_mean(rho)
    diag = {
        "target_reference_episode_mean_train": ref_mean,
        "target_reference_episode_distribution_train": ref_dist,
        "target_reference_n_episodes_train": ref_n,
        "candidate6_independent_episode_mean_train": mean_lo,
        "fully_persistent_episode_mean_train": mean_hi,
        "fitted_persistence_rho": rho,
        "achieved_episode_mean_at_rho_train": achieved,
        "n_bisect_iterations": n_bisect,
        "spell_child_max_age": SPELL_CHILD_MAX_AGE,
        "note": (
            "rho is the fraction of linked children that follow the persistent, "
            "sibling-synchronized frailty episode (vs the candidate-6 "
            "independent per-wave draw); fitted on train side B so the "
            "simulated episode-length mean matches the reference (~5.93 waves) "
            "while the per-wave custodial marginal is preserved exactly."
        ),
    }
    return rho, diag


# --------------------------------------------------------------------------
# Fit (train / side B only for every fitted parameter)
# --------------------------------------------------------------------------
def fit_household_model_v7(
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
    marital_by_year: pd.DataFrame | None = None,
    fu_sizes: pd.DataFrame | None = None,
    legal_flag: pd.DataFrame | None = None,
    child_record_expo: pd.DataFrame | None = None,
    parent_counts: pd.DataFrame | None = None,
) -> HouseholdCompositionModelV7:
    """Fit candidate 6 (byte-faithful) plus the two candidate-7 deltas."""
    if father_links_child is None:
        father_links_child = hcs3.father_link_births_with_child(birth_records)
    if parent_pairs is None:
        parent_pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    if marital_by_year is None:
        marital_by_year = hcs3._father_marital_by_year(mpanel)

    base_v6 = hcs6.fit_household_model_v6(
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

    # Delta 1: the enumerated (joinable) linked-child keys.
    joinable_keys = enumerated_joinable_keys(father_links_child)
    n_exposure_rows = int(len(base_v6.father_links))
    fl_all = base_v6.father_links[["parent_person_id", "birth_year"]]
    fl_idx = pd.MultiIndex.from_arrays(
        [
            fl_all["parent_person_id"].astype("int64"),
            fl_all["birth_year"].astype("int64"),
        ]
    )
    n_joinable_rows = int(np.asarray(fl_idx.isin(joinable_keys)).sum())

    # Delta 2: the episode-persistence rho, fitted to the train episode mean.
    rho, episode_fit = fit_linked_episode_persistence(
        hh,
        base_v6,
        father_links_child,
        parent_pairs,
        marital_by_year,
        joinable_keys,
        train_ids,
    )

    meta = {
        **base_v6.meta,
        "enumeration_conditioning": {
            "n_linked_exposure_rows": n_exposure_rows,
            "n_joinable_exposure_rows": n_joinable_rows,
            "n_nonjoinable_exposure_rows": n_exposure_rows - n_joinable_rows,
            "n_joinable_keys": len(joinable_keys),
            "nonjoinable_share": (
                round((n_exposure_rows - n_joinable_rows) / n_exposure_rows, 6)
                if n_exposure_rows > 0
                else 0.0
            ),
        },
        "linked_episode_persistence": episode_fit,
        "delta_stream_tag_v6": DELTA_STREAM_TAG_V6,
        "delta_stream_tag_v7": DELTA_STREAM_TAG_V7,
        "custodial_revert_band": list(CUSTODIAL_REVERT_BAND),
        "grandchild_coupling_age_lo": GRANDCHILD_LO,
    }
    return HouseholdCompositionModelV7(
        base_v6=base_v6,
        joinable_keys=joinable_keys,
        linked_episode_persistence=rho,
        episode_fit=episode_fit,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Delta 1 + Delta 2 apply: the enumeration-conditioned, episode-persistent
# linked child coresidence draw (replaces custodial_linked_child_counts_v6)
# --------------------------------------------------------------------------
def custodial_linked_child_counts_v7(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    model: HouseholdCompositionModelV7,
    episode_rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per side-A person-wave, the count of coresident father-linked children.

    Replaces :func:`hcs6.custodial_linked_child_counts_v6`. The exposure
    construction (merge, child-age filter, custodial band, father marital,
    candidate-3 row order) and the per-row faithful custody probability
    ``custodial_prob_v6`` are BYTE-FAITHFUL to candidate 6. The two candidate-7
    deltas:

    * **Delta 1 (enumeration conditioning).** A non-joinable exposure row (its
      ``(parent_person_id, birth_year)`` not in ``model.joinable_keys``) cannot
      be drawn coresident; it is EXCLUDED from the draw and its share recorded
      per child band.
    * **Delta 2 (episode persistence).** Over the JOINABLE exposure, coresidence
      is drawn by :func:`simulate_linked_episode_coresidence` on the isolated
      ``0xC7`` stream -- a fraction ``rho`` of children follow the persistent,
      sibling-synchronized frailty episode, the rest the candidate-6 independent
      per-wave draw. The per-wave custodial marginal is preserved exactly.

    Returns the per-side-A-row coresident-linked-child counts and a diagnostic
    dict (the per-band marginal-preservation check, the non-joinable share by
    band, and the simulated episode-length distribution over child ages 0-17).
    """
    counts = np.zeros(len(side_a_pw), dtype=np.int64)
    diag = {
        "n_linked_child_coresident_wave_units": 0,
        "nonjoinable_share_by_band": {},
        "marginal_preservation_by_band": {},
        "marginal_preservation_max_abs_dev": 0.0,
        "sim_episode_length_distribution": {b: 0.0 for b in EPISODE_BUCKETS},
        "sim_episode_mean_length": 0.0,
        "sim_n_episodes": 0,
        "persistence_rho": float(model.linked_episode_persistence),
    }
    if not len(linked_births):
        return counts, diag
    expo = _linked_exposure_frame(
        linked_births,
        side_a_pw,
        marital_sim,
        model.base_v6,
        model.joinable_keys,
    )
    if not len(expo):
        return counts, diag

    joinable = expo["joinable"].to_numpy(dtype=bool)
    child_band = expo["child_band"].to_numpy(dtype=object)

    # Non-joinable share by child band (delta 1; deterministic given exposure).
    for band in _CHILD_BANDS:
        m = child_band == band
        n_rows = int(m.sum())
        n_nonjoin = int((m & ~joinable).sum())
        diag["nonjoinable_share_by_band"][band] = {
            "n_exposure_rows": n_rows,
            "n_nonjoinable": n_nonjoin,
            "share": (round(n_nonjoin / n_rows, 6) if n_rows else 0.0),
        }

    # Delta 2 draw over the JOINABLE exposure only (sorted by (birth, year)).
    expo_j = (
        expo[joinable]
        .sort_values(["_birth_id", "year"])
        .reset_index(drop=True)
    )
    coresident = simulate_linked_episode_coresidence(
        expo_j["parent_person_id"].to_numpy(np.int64),
        expo_j["_birth_id"].to_numpy(np.int64),
        expo_j["p_c"].to_numpy(np.float64),
        float(model.linked_episode_persistence),
        episode_rng,
    )
    rows = expo_j["_row"].to_numpy()
    np.add.at(counts, rows[coresident], 1)
    diag["n_linked_child_coresident_wave_units"] = int(coresident.sum())

    # Marginal-preservation check by band (delta 2 constraint): the simulated
    # per-wave coresidence share vs the faithful target (mean p_c) over the
    # joinable exposure -- preserved exactly in expectation (mixture identity),
    # so the per-draw deviation is Monte-Carlo only.
    jband = expo_j["child_band"].to_numpy(dtype=object)
    jp = expo_j["p_c"].to_numpy(dtype=np.float64)
    max_dev = 0.0
    for band in _CHILD_BANDS:
        m = jband == band
        if not m.any():
            continue
        target = float(jp[m].mean())
        sim = float(coresident[m].mean())
        dev = abs(sim - target)
        max_dev = max(max_dev, dev)
        diag["marginal_preservation_by_band"][band] = {
            "target_mean_custody_prob": target,
            "sim_coresident_share": sim,
            "abs_deviation": dev,
            "n_joinable_exposure_rows": int(m.sum()),
        }
    diag["marginal_preservation_max_abs_dev"] = max_dev

    # Simulated episode-length distribution over minor child ages (fit-vs-raw).
    minor = expo_j["child_age"].to_numpy() <= SPELL_CHILD_MAX_AGE
    dist, mean, n_epi = episode_length_hist(
        expo_j["parent_person_id"].to_numpy(np.int64)[minor],
        expo_j["birth_year"].to_numpy(np.int64)[minor],
        expo_j["year"].to_numpy(np.int64)[minor],
        coresident[minor],
    )
    diag["sim_episode_length_distribution"] = dist
    diag["sim_episode_mean_length"] = mean
    diag["sim_n_episodes"] = n_epi
    return counts, diag


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def simulate_draw_v7(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModelV7,
    ids_a: set[int],
    draw_seed: int,
    occupancy_stream_tag: int = 0xB2B,
    delta_stream_tag_v2: int = 0xC2,
    delta_stream_tag_v3: int = 0xC3,
    delta_stream_tag_v4: int = 0xC4,
    delta_stream_tag_v5: int = hcs6.DELTA_STREAM_TAG_V5,
    delta_stream_tag_v6: int = DELTA_STREAM_TAG_V6,
    delta_stream_tag_v7: int = DELTA_STREAM_TAG_V7,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-7 draw of the side-A holdout households.

    Reproduces candidate 6's draw byte-for-byte on every carried stream
    (``0xB2B`` / ``0xC2`` / ``0xC3`` / ``0xC4`` / ``0xC5`` / ``0xC6``), then
    REPLACES the linked-father child coresidence draw with the candidate-7
    enumeration-conditioned, episode-persistent draw
    (:func:`custodial_linked_child_counts_v7`) on the ISOLATED ``0xC7`` stream.
    The candidate-6 ``0xC3`` custodial stream is retired (its non-family and
    skip-gen sibling spawns are preserved so those stay byte-identical); every
    other carried family is byte-identical to candidate 6.
    """
    base = model.base

    # 1. carried parent / multigen / legal-marriage spouse (candidate 1, 0xB2B).
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

    # 2. candidate-2 delta substreams (0xC2), consumed EXACTLY as candidate 6.
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

    # 3. candidate-4 DELTA 1 (carried) cohab overlay, then candidate-6 DELTA 3
    #    (carried): the female 25-44 single-year override.
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

    # 4. candidate-4 delta substreams (0xC4), consumed EXACTLY as candidate 6.
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

    # 5. certified marital core + maternal births (same draw seed as cand 6).
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

    # candidate-6 base leave-year draw (0xC2) -- byte-identical so the SHADOW
    # leave-year is unchanged; the maternal rows are re-drawn by carried delta 2.
    base_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    # candidate-3 delta substreams (0xC3), isolated from 0xB2B/0xC2. The
    # custodial spawn is RETIRED by candidate 7 (the linked draw moved to 0xC7);
    # its non-family and skip-gen sibling spawns are preserved so those stay
    # byte-identical to candidate 6.
    c3_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v3])
    _custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # candidate-5 delta substreams (0xC5): coupling + parent count (CARRIED).
    c5_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)

    # candidate-6 delta substream (0xC6): the maternal single-year leave refit
    # (CARRIED). Isolated from every other stream.
    mat_leave_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, delta_stream_tag_v6])
    )

    # candidate-7 delta substream (0xC7): the linked episode-persistence draw
    # (the two candidate-7 deltas). Isolated from every carried stream.
    episode_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, delta_stream_tag_v7])
    )

    # carried DELTA 2 (maternal): the single-year 18-30 exit refit on 0xC6.
    maternal_births = all_births[all_births["_source"] == "maternal"]
    maternal_leaves = hcs6.child_leave_years_refit(
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

    # candidate-7 DELTAS 1 + 2 (linked): enumeration-conditioned, episode-
    # persistent linked child coresidence on the isolated 0xC7 stream.
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked, linked_diag = custodial_linked_child_counts_v7(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model,
        episode_rng,
    )
    child_counts = child_counts_nonlinked + child_counts_linked

    coresident_child, grandchild_composed, _hh_default = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

    # DELTA 3b of candidate 5 (CARRIED): per-ego coresident-parent count on 0xC5.
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
    # coupling is applied ONLY at 55+.
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

    # DELTA 4 of candidate 6 (carried): non-family count from P(count | core).
    nonfamily_count = hcs6.sample_nonfamily_count_v6(
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
        "n_linked_child_coresident_wave_units": linked_diag[
            "n_linked_child_coresident_wave_units"
        ],
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
        "no_coupling_below_55_max_p": (
            float(p_coupled[~is_55_row].max()) if (~is_55_row).any() else 0.0
        ),
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
        # Candidate-7 delta instrumentation.
        "linked_persistence_rho": float(model.linked_episode_persistence),
        "linked_nonjoinable_share_by_band": linked_diag[
            "nonjoinable_share_by_band"
        ],
        "linked_marginal_preservation_by_band": linked_diag[
            "marginal_preservation_by_band"
        ],
        "linked_marginal_preservation_max_abs_dev": linked_diag[
            "marginal_preservation_max_abs_dev"
        ],
        "linked_sim_episode_length_distribution": linked_diag[
            "sim_episode_length_distribution"
        ],
        "linked_sim_episode_mean_length": linked_diag[
            "sim_episode_mean_length"
        ],
        "linked_sim_n_episodes": linked_diag["sim_n_episodes"],
    }
    return panel, diagnostics


# --------------------------------------------------------------------------
# Delta-specific fit-vs-raw checks (fit seed's train side B + forensics-4)
# --------------------------------------------------------------------------
def c7_delta_checks(
    model: HouseholdCompositionModelV7,
    forensics4: dict[str, Any],
) -> dict[str, Any]:
    """Record the two deltas' fit-vs-raw checks against forensics-4.

    Reproduces the forensics-4 Q11 channel-removal ARITHMETIC (the raw
    candidate-6 cell miss minus the two removed channels == the candidate-7
    predicted residual), the non-joinable exposure split, and the episode-length
    fit target -- computed from the fitted model beside the forensics-4 measured
    quantities each delta implements.
    """
    q11 = forensics4["question_11_linked_father_child_supply"]
    enum = model.meta["enumeration_conditioning"]
    epi = model.meta["linked_episode_persistence"]

    # Delta 1: enumeration conditioning (removes unenumerated_nonjoinable).
    per_cell_arith: dict[str, Any] = {}
    for cell in ("coresident_child.25-34|male", "coresident_child.35-44|male"):
        pc = q11["per_cell"][cell]
        ch = pc["channels"]
        raw_miss = pc["cell_miss_sim_minus_reference"]
        removed = ch["unenumerated_nonjoinable_supply"] + ch["spell_length"]
        per_cell_arith[cell] = {
            "raw_cell_miss_candidate6": raw_miss,
            "unenumerated_nonjoinable_channel": ch[
                "unenumerated_nonjoinable_supply"
            ],
            "spell_length_channel": ch["spell_length"],
            "shadow_named_residual": ch["shadow_unlinked_channel"],
            "predicted_residual_after_two_deltas": round(
                raw_miss - removed, 6
            ),
            "tolerance": pc["holdout_committed_candidate6"]["tolerance"],
        }

    delta_1 = {
        "n_linked_exposure_rows": enum["n_linked_exposure_rows"],
        "n_joinable_exposure_rows": enum["n_joinable_exposure_rows"],
        "n_nonjoinable_exposure_rows": enum["n_nonjoinable_exposure_rows"],
        "nonjoinable_share": enum["nonjoinable_share"],
        "forensics_measured_nonjoinable_share": round(9500 / 36887, 6),
        "forensics_n_linked_exposure_rows_train": q11["per_cell"][
            "coresident_child.35-44|male"
        ]["anchors"].get("reference_a_refexp_all_analytic_occupancy"),
        "removes_channel": "unenumerated_nonjoinable_supply",
        "note": (
            "Delta 1 restricts the linked coresidence draw to ENUMERATED "
            "children (the joinable (parent, birth_year) keys); the non-joinable "
            "biological children the committed candidate-6 draw coresides but "
            "the reference roster can never observe (25.8% of linked exposure) "
            "are excluded -- removing the dominant unenumerated_nonjoinable "
            "channel (+0.035/+0.036)."
        ),
    }

    delta_2 = {
        "fitted_persistence_rho": epi["fitted_persistence_rho"],
        "target_reference_episode_mean_train": epi[
            "target_reference_episode_mean_train"
        ],
        "candidate6_independent_episode_mean_train": epi[
            "candidate6_independent_episode_mean_train"
        ],
        "achieved_episode_mean_at_rho_train": epi[
            "achieved_episode_mean_at_rho_train"
        ],
        "forensics_sim_v6_episode_mean": forensics4[
            "question_11_linked_father_child_supply"
        ]["episode_length_distributions"]["sim"]["mean_episode_length"],
        "forensics_reference_episode_mean": forensics4[
            "question_11_linked_father_child_supply"
        ]["episode_length_distributions"]["reference"]["mean_episode_length"],
        "removes_channel": "spell_length",
        "note": (
            "Delta 2 replaces the candidate-6 independent per-wave occupancy "
            "(mean episode 3.57 waves) with a correlated entry/persist/exit "
            "process fitted to the train episode-length mean (~5.93 waves), "
            "CONSTRAINED to preserve the per-wave custodial marginal by band "
            "exactly -- reshaping the father-wave stock (the spell channel, "
            "+0.018/+0.022) without moving the faithful per-wave probabilities."
        ),
    }

    return {
        "delta_1_enumeration_conditioning": delta_1,
        "delta_2_episode_persistence": delta_2,
        "channel_removal_arithmetic": per_cell_arith,
        "shadow_channel_untouched_named_residual": {
            "coresident_child.25-34|male": q11["per_cell"][
                "coresident_child.25-34|male"
            ]["channels"]["shadow_unlinked_channel"],
            "coresident_child.35-44|male": q11["per_cell"][
                "coresident_child.35-44|male"
            ]["channels"]["shadow_unlinked_channel"],
            "note": (
                "The shadow (unlinked imputed-paternal) channel is a NAMED "
                "residual, deliberately untouched -- offset by the marital-joint "
                "and non-linked negatives; touching it without a measurement "
                "would be tuning."
            ),
        },
    }
