"""Gate-2b candidate 8: proven levers + band-signed adult-child retention.

Candidate 8 (registration #42 comment 4948604739) is candidate 7
(:mod:`populace_dynamics.models.household_composition_sim_v7`, merged in PR
#145) with EXACTLY THREE frozen deltas, each designed against a graded finding
of the gate-2b forensics-5 decomposition (``runs/gate2b_forensics5_v1.json``,
grading 4948603337). Everything candidate 7 cleared or carried -- the certified
tranche-2a marital core, the carried ``coresident_parent`` / ``multigen`` (stock
+ transitions) / ``parental_home_exit`` / ``coresident_grandchild`` (the 55+
coupled cell and the 45-54|female composed cell), and every ``coresident_spouse``
band EXCEPT the 25-34|female overlay lifted by delta 2 -- is carried
BYTE-FAITHFULLY: candidate 8 REUSES candidate 7's generator and re-runs its exact
``0xB2B`` / ``0xC2`` / ``0xC3`` / ``0xC4`` / ``0xC5`` / ``0xC6`` / ``0xC7``
streams, then the three candidate-8 deltas modify the composed
``coresident_child`` / ``hh_size`` / ``coresident_spouse.25-34|female`` frame on a
SINGLE ISOLATED ``SeedSequence([draw_seed, 0xC8])`` -- so every carried stream is
byte-identical to candidate 7. ``coresident_parent`` / ``multigen`` /
``coresident_grandchild`` are taken from the candidate-7 composition UNCHANGED
(not recomposed from the delta-modified child), so their gated cells are
byte-identical.

Delta 1 -- **fertility-core lift** (Q15-proven for ``hh_size.5+`` and
``coresident_child.55-64|male``): the composed frame's completed-family-size
distribution is corrected to the train ``3+``-child distribution per parent
cohort (band) x sex, implemented exactly as the Q15 analytic application (swap
``D_sim[S] -> D_train[S]`` holding the sim's own conditional kernels
``K_sim(coresident|S)`` and the ``hh_size|S`` resample kernel). Reproduces the
Q15 headline (``hh_size.5+`` 0.128 -> 0.144; ``55-64|male`` 0.213 -> 0.255).

Delta 2 -- **cohabitation-overlay lift at 25-34|female** (Q16-proven, no
collateral): the currently-non-spouse mass at band ``25-34`` female is lifted by
the forensics-3 Q9 measured -0.045 cohabitation-overlay shortfall (Bernoulli
superposition ``new = old + 0.045 * (1 - old)``), an age-band-specific override;
every other female spouse band is untouched. Reproduces the Q16 headline
(0.588 -> 0.606).

Delta 3 -- **band-signed adult-child retention refit at parent ages 45+** (the
Q14 exit channel, BOTH signs) + the **link-coverage inclusion** at older parent
ages: the coresident-adult-child exit/retention hazards are refit by parent age
band x sex on train, closing the measured EXIT-ORIGIN channel band-signed
(lifting retention at ``65-74`` -- exit -0.022 under -- and REDUCING the
``45-54|female`` over-retention -- +0.079 over), plus the LINK-COVERAGE channel
(-0.020 at ``55-64|male``, -0.016 at ``65-74|male``): the enumerated joinable
children whose links exist in train but sit outside the current draw basis at
those ages enter it, per the Q14 channel definition exactly. The v7
persistence/enumeration interaction (+0.008 / -0.010) is the NAMED residual, left
for a targeted candidate-9 forensics.
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
from populace_dynamics.models import household_composition_sim_v7 as hcs7

__all__ = [
    "GRANDCHILD_LO",
    "CORE_SIZE_CAP",
    "CHILD_CORESIDENCE_MAX_AGE",
    "SPELL_CHILD_MAX_AGE",
    "SIZE_BUCKETS",
    "DELTA_STREAM_TAG_V7",
    "DELTA_STREAM_TAG_V8",
    "COHAB_OVERLAY_LIFT_BAND",
    "COHAB_OVERLAY_LIFT",
    "RETENTION_EXIT_CELLS",
    "LINK_COVERAGE_CELLS",
    "N_FIT_DRAWS",
    "HouseholdCompositionModelV8",
    "size_bucket",
    "train_completed_size",
    "sim_completed_size_row",
    "cell_completed_size_dk",
    "q14_linked_reference_cell",
    "fit_delta3_retention_link",
    "fit_household_model_v8",
    "apply_fertility_core_lift",
    "apply_retention_link_refit",
    "apply_cohab_overlay_lift",
    "simulate_draw_v8",
    "c8_delta_checks",
]

#: Carried from candidate 7 (unchanged).
GRANDCHILD_LO = hcs7.GRANDCHILD_LO  # 55
CORE_SIZE_CAP = hcs7.CORE_SIZE_CAP  # 5
CHILD_CORESIDENCE_MAX_AGE = hcs7.CHILD_CORESIDENCE_MAX_AGE  # 60
SPELL_CHILD_MAX_AGE = hcs7.SPELL_CHILD_MAX_AGE  # 17

#: Candidate-7 isolated RNG tag (the linked episode-persistence draw), CARRIED.
DELTA_STREAM_TAG_V7 = hcs7.DELTA_STREAM_TAG_V7  # 0xC7
#: Candidate-8 isolated RNG tag: the three candidate-8 deltas draw exclusively
#: from ``SeedSequence([draw_seed, 0xC8])``, isolated from every carried stream
#: (0xB2B / 0xC2 / 0xC3 / 0xC4 / 0xC5 / 0xC6 / 0xC7), so every carried family is
#: byte-identical to candidate 7.
DELTA_STREAM_TAG_V8 = 0xC8

#: Completed-family-size buckets (a parent's max coresident own-child count),
#: verbatim from forensics-5 (``gate2b_forensics5.SIZE_BUCKETS``).
SIZE_BUCKETS = ("0", "1", "2", "3", "4+")

#: Delta 2: the forensics-3 Q9 measured cohabitation-overlay shortfall, applied
#: as an age-band-specific Bernoulli superposition at 25-34|female ONLY.
COHAB_OVERLAY_LIFT_BAND = "25-34"
COHAB_OVERLAY_LIFT = 0.045

#: Delta 3: the parent-45+ adult-child cells whose EXIT-ORIGIN channel is closed
#: band-signed (Q14: lift 65-74 under-retention, reduce 45-54|female
#: over-retention). Each closes ``-exit_origin`` (train-fitted).
RETENTION_EXIT_CELLS = (
    "coresident_child.65-74|male",
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
#: Delta 3: the older-male cells whose LINK-COVERAGE channel is closed (Q14:
#: -0.020 at 55-64|male, -0.016 at 65-74|male). Each closes ``-link_coverage``.
LINK_COVERAGE_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
)

#: Train draws used to fit the delta-3 channels (kernel shift + linked v7). The
#: channels are stable across draws; the fit averages this many side-B draws.
N_FIT_DRAWS = 6

_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED
#: The composition age bands the fertility-core swap reassigns within.
_COMPOSITION_BANDS = tuple(
    hc.band_label(lo, hi) for lo, hi in hc.COMPOSITION_AGE_BANDS
)


# --------------------------------------------------------------------------
# Completed-family-size helpers (byte-faithful to forensics-5)
# --------------------------------------------------------------------------
def size_bucket(counts: np.ndarray) -> np.ndarray:
    """Map an integer count array to SIZE_BUCKETS labels (0/1/2/3/4+)."""
    c = np.asarray(counts, dtype=np.int64)
    return np.where(c >= 4, "4+", c.astype(str).astype(object))


def _cell_of(cell: str) -> tuple[str, str]:
    """Split ``coresident_child.55-64|male`` -> ("55-64", "male")."""
    tail = cell.split(".", 1)[1]
    band, sex = tail.split("|")
    return band, sex


def train_completed_size(
    parent_pairs: pd.DataFrame, ids_b: set[int]
) -> dict[int, int]:
    """Per-parent COMPLETED FAMILY SIZE on train side B.

    Byte-faithful to ``gate2b_forensics5.train_completed_size``: the parent's
    maximum coresident own-child count across their waves (from ``parent_pairs``,
    the roster parent-child coresidence pairs). Persons never observed as a
    coresident parent carry size 0.
    """
    pp = parent_pairs[parent_pairs["parent_person_id"].isin(ids_b)]
    if not len(pp):
        return {}
    per_wave = pp.groupby(["parent_person_id", "year"], sort=False).size()
    per_parent = per_wave.groupby("parent_person_id").max()
    return {int(k): int(v) for k, v in per_parent.items()}


def sim_completed_size_row(
    person_id_row: np.ndarray, child_counts_row: np.ndarray
) -> np.ndarray:
    """Per person-wave row, the ego's max coresident child count over waves.

    Byte-faithful to ``gate2b_forensics5._sim_completed_size_row``.
    """
    df = pd.DataFrame({"pid": person_id_row, "cc": child_counts_row})
    mx = df.groupby("pid")["cc"].transform("max")
    return mx.to_numpy(dtype=np.int64)


def _size_dist(sizes: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Weighted completed-family-size distribution over SIZE_BUCKETS."""
    w = np.asarray(weight, dtype=np.float64)
    tot = float(w.sum())
    if tot <= 0:
        return {b: 0.0 for b in SIZE_BUCKETS}
    c = np.asarray(sizes, dtype=np.int64)
    return {
        "0": float(w[c == 0].sum() / tot),
        "1": float(w[c == 1].sum() / tot),
        "2": float(w[c == 2].sum() / tot),
        "3": float(w[c == 3].sum() / tot),
        "4+": float(w[c >= 4].sum() / tot),
    }


def completed_size_dist_by_cell(
    hh: hc.HouseholdCompositionPanel,
    size_map: dict[int, int],
    ids_b: set[int],
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, float]]:
    """Train ``D_train[S]`` per (band, sex) cell and over all side-B waves.

    The fertility-core lift's target: the completed-family-size distribution the
    sim's own core is corrected to, per parent cohort (band) x sex (delta 1).
    """
    pw = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)]
    pw = pw[pw["band"].notna()]
    s_all = (
        pw["person_id"]
        .map(lambda p: size_map.get(int(p), 0))
        .to_numpy(dtype=np.int64)
    )
    w_all = pw["weight"].to_numpy(np.float64)
    d_all = _size_dist(s_all, w_all)
    by_cell: dict[tuple[str, str], dict[str, float]] = {}
    band_arr = pw["band"].to_numpy(dtype=object)
    sex_arr = pw["sex"].to_numpy()
    for band in _COMPOSITION_BANDS:
        for sex in ("male", "female"):
            m = (band_arr == band) & (sex_arr == sex)
            if not m.any():
                by_cell[(band, sex)] = {b: 0.0 for b in SIZE_BUCKETS}
                continue
            by_cell[(band, sex)] = _size_dist(s_all[m], w_all[m])
    return by_cell, d_all


def cell_completed_size_dk(
    pw_cell: pd.DataFrame, size_map: dict[int, int]
) -> tuple[dict[str, float], dict[str, float], float]:
    """Train ``D[S]`` / ``K[S]`` and full rate for one cell's person-waves.

    Byte-faithful to ``gate2b_forensics5._train_cell_dk``: the completed-size
    distribution, the coresidence-given-size kernel, and the full rate
    (== ``sum_S D[S] K[S]`` exactly). Used to fit the delta-3 exit-origin channel
    (``K_train`` for the coefficient term).
    """
    if not len(pw_cell):
        z = {b: 0.0 for b in SIZE_BUCKETS}
        return z, z, 0.0
    s = (
        pw_cell["person_id"]
        .map(lambda p: size_map.get(int(p), 0))
        .to_numpy(dtype=np.int64)
    )
    bucket = size_bucket(s)
    w = pw_cell["weight"].to_numpy(np.float64)
    cor = pw_cell["coresident_child"].to_numpy(bool)
    tot = float(w.sum())
    d: dict[str, float] = {}
    k: dict[str, float] = {}
    for b in SIZE_BUCKETS:
        mb = bucket == b
        wb = float(w[mb].sum())
        d[b] = (wb / tot) if tot > 0 else 0.0
        k[b] = float((w[mb] * cor[mb]).sum() / wb) if wb > 0 else 0.0
    full = sum(d[b] * k[b] for b in SIZE_BUCKETS)
    return d, k, float(full)


# --------------------------------------------------------------------------
# Q14 linked-father anchors (link-coverage + v7 interaction), ported
# --------------------------------------------------------------------------
def q14_linked_reference_cell(
    hh: hc.HouseholdCompositionPanel,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    model: hcs7.HouseholdCompositionModelV7,
    ids_b: set[int],
    cell: str,
) -> dict[str, float]:
    """Reference-side linked anchors at one older-male cell (byte-faithful).

    Ports ``gate2b_forensics5.q14_linked_reference`` for a single male cell.
    Returns, per full cell weight: the observed coresidence with ANY linked
    father child (``linked_any``), the analytic independent-per-wave occupancy
    over the JOINABLE exposure under observed father marital (``a_refexp_j``), and
    the observed coresidence with JOINABLE-enumerated children (``s_join``). The
    Q14 channels are then ``link_coverage = s_join - linked_any`` and
    ``v7_interaction = a_refexp_j - s_join``. Deterministic; no simulation RNG.
    """
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)][
        [
            "person_id",
            "year",
            "age",
            "band",
            "sex",
            "weight",
            "coresident_child",
        ]
    ].copy()
    fl = model.father_links[["parent_person_id", "birth_year"]].copy()
    fl["parent_person_id"] = fl["parent_person_id"].astype("int64")
    fl["birth_year"] = fl["birth_year"].astype("int64")
    linked_father_ids = set(int(x) for x in fl["parent_person_id"].unique())
    joinable_keys = model.joinable_keys
    fac = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ].copy()
    fac["parent_person_id"] = fac["parent_person_id"].astype("int64")
    fac["child_person_id"] = fac["child_person_id"].astype("int64")
    fac["birth_year"] = fac["birth_year"].astype("int64")

    fw = pw_b[["person_id", "year", "band", "sex", "weight"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    expo = fl.merge(fw, on="parent_person_id", how="inner")
    expo["child_age"] = expo["year"] - expo["birth_year"]
    expo = expo[
        (expo["child_age"] >= 0)
        & (expo["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    expo["child_band"] = expo["child_age"].map(hcs3._child_band)
    expo = expo[expo["child_band"].notna()].copy()
    expo = expo.merge(
        marital_by_year.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    expo["marital"] = expo["marital"].fillna(_NOT_MARRIED)
    idx = pd.MultiIndex.from_arrays(
        [expo["parent_person_id"], expo["birth_year"]]
    )
    expo["joinable"] = np.asarray(idx.isin(joinable_keys), dtype=bool)
    prob = np.array(
        [
            hcs6.custodial_prob_v6(
                model.base_v6, int(a), hcs4.era_of_year(int(y)), str(m)
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
    logno_j = np.where(
        expo["joinable"].to_numpy(),
        np.log1p(-np.clip(prob, 0.0, 1.0 - 1e-15)),
        0.0,
    )
    expo["_logno_j"] = logno_j
    grp = expo.groupby(["parent_person_id", "year"], sort=False)
    fw_agg = grp.agg(logno_j=("_logno_j", "sum")).reset_index()
    fw_agg["a_refexp_j"] = 1.0 - np.exp(fw_agg["logno_j"].to_numpy())

    idx2 = pd.MultiIndex.from_arrays(
        [fac["parent_person_id"], fac["birth_year"]]
    )
    fac_j = fac[np.asarray(idx2.isin(joinable_keys), dtype=bool)]
    linked_cor = parent_pairs.merge(
        fac_j, on=["parent_person_id", "child_person_id"], how="inner"
    )
    linked_cor["child_age"] = linked_cor["year"] - linked_cor["birth_year"]
    linked_cor = linked_cor[
        (linked_cor["child_age"] >= 0)
        & (linked_cor["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ]
    cor_fw = (
        linked_cor.groupby(["parent_person_id", "year"])
        .size()
        .rename("n_cor_j")
        .reset_index()
    )

    pw_b = pw_b.assign(
        _linked=pw_b["person_id"].isin(linked_father_ids),
        _male=(pw_b["sex"] == "male"),
    )
    bl, _sx = _cell_of(cell)
    cell_mask = (pw_b["band"] == bl) & pw_b["_male"]
    cw = pw_b.loc[cell_mask, "weight"].to_numpy(np.float64)
    cw_tot = float(cw.sum())
    cc_any = pw_b.loc[cell_mask, "coresident_child"].to_numpy(bool)
    is_link = pw_b.loc[cell_mask, "_linked"].to_numpy(bool)
    ref_full = float((cw * cc_any).sum() / cw_tot) if cw_tot > 0 else 0.0
    ref_linked_any = (
        float((cw * (cc_any & is_link)).sum() / cw_tot) if cw_tot > 0 else 0.0
    )
    cl = (
        pw_b.loc[cell_mask & pw_b["_linked"], ["person_id", "year", "weight"]]
        .merge(
            fw_agg[["parent_person_id", "year", "a_refexp_j"]].rename(
                columns={"parent_person_id": "person_id"}
            ),
            on=["person_id", "year"],
            how="left",
        )
        .merge(
            cor_fw.rename(columns={"parent_person_id": "person_id"}),
            on=["person_id", "year"],
            how="left",
        )
    )
    cl["a_refexp_j"] = cl["a_refexp_j"].fillna(0.0)
    cl["n_cor_j"] = cl["n_cor_j"].fillna(0).astype("int64")
    clw = cl["weight"].to_numpy(np.float64)
    a_j = cl["a_refexp_j"].to_numpy(np.float64)
    n_cor = cl["n_cor_j"].to_numpy(np.int64)
    a_refexp_j = float((clw * a_j).sum() / cw_tot) if cw_tot > 0 else 0.0
    s_join = float((clw * (n_cor > 0)).sum() / cw_tot) if cw_tot > 0 else 0.0
    return {
        "reference_full_rate": ref_full,
        "linked_any": ref_linked_any,
        "s_joinable_restricted": s_join,
        "a_refexp_joinable": a_refexp_j,
        "link_coverage": s_join - ref_linked_any,
        "v7_interaction": a_refexp_j - s_join,
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class HouseholdCompositionModelV8:
    """Candidate 7's byte-faithful bundle plus the three candidate-8 deltas.

    ``base_v7`` is the byte-faithful candidate-7 model (carrying candidates 6-1).
    The three candidate-8 fitted structures:

    * ``completed_size_dist_train`` / ``completed_size_dist_train_all`` -- the
      train completed-family-size distribution per (band, sex) and pooled; the
      delta-1 fertility-core lift swaps ``D_sim[S] -> D_train[S]`` holding the
      sim kernels.
    * ``retention_link_shift`` -- the per-cell delta-3 additive closure of the
      Q14 EXIT-ORIGIN (band-signed) and LINK-COVERAGE channels at parent 45+.
    * ``cohab_overlay_lift`` -- the delta-2 cohabitation-overlay lift (0.045) at
      25-34|female.
    """

    base_v7: hcs7.HouseholdCompositionModelV7
    completed_size_dist_train: dict[tuple[str, str], dict[str, float]]
    completed_size_dist_train_all: dict[str, float]
    retention_link_shift: dict[str, float]
    cohab_overlay_lift: float = COHAB_OVERLAY_LIFT
    delta3_fit: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- pass-through accessors to the carried candidate bundles ----
    @property
    def base_v6(self) -> hcs6.HouseholdCompositionModelV6:
        return self.base_v7.base_v6

    @property
    def base_v5(self) -> hcs5.HouseholdCompositionModelV5:
        return self.base_v7.base_v5

    @property
    def base_v4(self) -> hcs4.HouseholdCompositionModelV4:
        return self.base_v7.base_v4

    @property
    def base_v3(self) -> hcs3.HouseholdCompositionModelV3:
        return self.base_v7.base_v3

    @property
    def base_v2(self) -> hcs2.HouseholdCompositionModelV2:
        return self.base_v7.base_v2

    @property
    def base(self) -> hcs.HouseholdCompositionModel:
        return self.base_v7.base

    @property
    def father_links(self) -> pd.DataFrame:
        return self.base_v7.father_links

    @property
    def joinable_keys(self) -> frozenset[tuple[int, int]]:
        return self.base_v7.joinable_keys

    @property
    def linked_episode_persistence(self) -> float:
        return self.base_v7.linked_episode_persistence

    @property
    def coupling_child_given_multigen(self):
        return self.base_v7.coupling_child_given_multigen

    @property
    def coupling_child_pooled(self):
        return self.base_v7.coupling_child_pooled

    @property
    def parent_count_two_share(self):
        return self.base_v7.parent_count_two_share

    @property
    def parent_count_two_pooled(self) -> float:
        return self.base_v7.parent_count_two_pooled

    @property
    def child_exit_single_year(self):
        return self.base_v7.child_exit_single_year

    @property
    def cohab_entry_age_female(self):
        return self.base_v7.cohab_entry_age_female

    @property
    def cohab_exit_age_female(self):
        return self.base_v7.cohab_exit_age_female

    @property
    def nonfamily_count_by_core(self):
        return self.base_v7.nonfamily_count_by_core


# --------------------------------------------------------------------------
# Delta 3 fit: the band-signed exit-origin + link-coverage channel closure
# --------------------------------------------------------------------------
def fit_delta3_retention_link(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    base_v7: hcs7.HouseholdCompositionModelV7,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    size_map: dict[int, int],
    ids_b: set[int],
    *,
    n_fit_draws: int = N_FIT_DRAWS,
    fit_seed_base: int = 0xC8,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit the delta-3 per-cell additive channel closures on train side B.

    For each delta-3 cell the Q14 decomposition (byte-faithful to forensics-5)
    is measured on ``n_fit_draws`` train draws: the ENDOWMENT (fertility-origin,
    closed by delta 1) and the COEFFICIENT (kernel shift) via the Oaxaca-Blinder
    telescope against the train ``D``/``K``, plus -- for the male cells -- the
    LINK-COVERAGE and v7-interaction split from the linked reference. The
    additive shift applied by delta 3 is:

    * ``RETENTION_EXIT_CELLS`` (65-74|male, 45-54|female, 65-74|female):
      ``-exit_origin`` (band-signed; lifts under-retention, reduces
      over-retention).
    * ``LINK_COVERAGE_CELLS`` (55-64|male, 65-74|male): additionally
      ``-link_coverage`` (the enumerated joinable children outside the draw
      basis entering it).

    The v7 persistence/enumeration interaction is left as the NAMED residual.
    Every quantity is estimated on side B (no holdout tuning surface).
    """
    all_cells = tuple(
        dict.fromkeys(RETENTION_EXIT_CELLS + LINK_COVERAGE_CELLS)
    )
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)]
    pw_b = pw_b[pw_b["band"].notna()]

    # Deterministic train D / K per cell.
    train_dk: dict[str, Any] = {}
    for cell in all_cells:
        bl, sx = _cell_of(cell)
        sub = pw_b[(pw_b["band"] == bl) & (pw_b["sex"] == sx)][
            ["person_id", "weight", "coresident_child"]
        ]
        d, k, full = cell_completed_size_dk(sub, size_map)
        train_dk[cell] = {"d_train": d, "k_train": k, "ref_full": full}

    # Linked reference (deterministic) for the male cells: link + v7.
    linked_ref: dict[str, dict[str, float]] = {}
    for cell in LINK_COVERAGE_CELLS:
        linked_ref[cell] = q14_linked_reference_cell(
            hh,
            father_links_child,
            parent_pairs,
            marital_by_year,
            base_v7,
            ids_b,
            cell,
        )

    # Train draws: measure K_sim / D_sim per cell -> endowment + coefficient.
    endow_acc = {c: [] for c in all_cells}
    coef_acc = {c: [] for c in all_cells}
    sim_full_acc = {c: [] for c in all_cells}
    for k in range(n_fit_draws):
        comp = _compose_v7(hh, mpanel, base_v7, ids_b, fit_seed_base + k)
        s_row = sim_completed_size_row(comp["person_id"], comp["child_counts"])
        bucket_row = size_bucket(s_row)
        band_row = comp["band"]
        sex_row = comp["sex"]
        w_row = comp["weight"]
        cor_row = comp["coresident_child"]
        for cell in all_cells:
            bl, sx = _cell_of(cell)
            m = (band_row == bl) & (sex_row == sx)
            d_train = train_dk[cell]["d_train"]
            k_train = train_dk[cell]["k_train"]
            endow, coef, sim_full, _d, _k = _oaxaca_terms(
                bucket_row[m], cor_row[m], w_row[m], d_train, k_train
            )
            endow_acc[cell].append(endow)
            coef_acc[cell].append(coef)
            sim_full_acc[cell].append(sim_full)

    shift: dict[str, float] = {}
    diag_cells: dict[str, Any] = {}
    for cell in all_cells:
        endow = float(np.mean(endow_acc[cell]))
        coef = float(np.mean(coef_acc[cell]))
        sim_full = float(np.mean(sim_full_acc[cell]))
        ref_full = train_dk[cell]["ref_full"]
        if cell in LINK_COVERAGE_CELLS:
            link = linked_ref[cell]["link_coverage"]
            v7 = linked_ref[cell]["v7_interaction"]
        else:
            link = 0.0
            v7 = 0.0
        exit_origin = coef - link - v7
        s = 0.0
        if cell in RETENTION_EXIT_CELLS:
            s += -exit_origin
        if cell in LINK_COVERAGE_CELLS:
            s += -link
        shift[cell] = float(s)
        diag_cells[cell] = {
            "sim_full_train": sim_full,
            "reference_full_train": ref_full,
            "cell_miss": sim_full - ref_full,
            "fertility_origin": endow,
            "coefficient_kernel_shift": coef,
            "link_coverage": link,
            "v7_persistence_enumeration_interaction": v7,
            "exit_origin": exit_origin,
            "applied_shift": float(s),
            "closes": (
                ["exit_origin", "link_coverage"]
                if cell in RETENTION_EXIT_CELLS and cell in LINK_COVERAGE_CELLS
                else (
                    ["exit_origin"]
                    if cell in RETENTION_EXIT_CELLS
                    else ["link_coverage"]
                )
            ),
        }
    diag = {
        "n_fit_draws": n_fit_draws,
        "per_cell": diag_cells,
        "note": (
            "Delta 3 closes the Q14 EXIT-ORIGIN channel band-signed at parent "
            "45+ (RETENTION_EXIT_CELLS) and the LINK-COVERAGE channel at older "
            "male bands (LINK_COVERAGE_CELLS); the v7 persistence/enumeration "
            "interaction is the named residual. Fitted on side B over "
            f"{n_fit_draws} train draws; the additive shift is applied to the "
            "delta-1-lifted side-A coresident_child cell via a conditional flip."
        ),
    }
    return shift, diag


def _oaxaca_terms(
    cell_bucket: np.ndarray,
    cell_cor: np.ndarray,
    cell_w: np.ndarray,
    d_train: dict[str, float],
    k_train: dict[str, float],
) -> tuple[float, float, float, dict[str, float], dict[str, float]]:
    """Exact Oaxaca endowment / coefficient for one draw's cell (forensics-5)."""
    w = np.asarray(cell_w, dtype=np.float64)
    tot = float(w.sum())
    d_sim: dict[str, float] = {}
    k_sim: dict[str, float] = {}
    for b in SIZE_BUCKETS:
        mb = cell_bucket == b
        wb = float(w[mb].sum())
        d_sim[b] = (wb / tot) if tot > 0 else 0.0
        k_sim[b] = float((w[mb] * cell_cor[mb]).sum() / wb) if wb > 0 else 0.0
    sim_full = sum(d_sim[b] * k_sim[b] for b in SIZE_BUCKETS)
    endowment = sum(
        (d_sim[b] - d_train.get(b, 0.0)) * k_sim[b] for b in SIZE_BUCKETS
    )
    coefficient = sum(
        d_train.get(b, 0.0) * (k_sim[b] - k_train.get(b, 0.0))
        for b in SIZE_BUCKETS
    )
    return float(endowment), float(coefficient), float(sim_full), d_sim, k_sim


# --------------------------------------------------------------------------
# Fit (train / side B only for every fitted parameter)
# --------------------------------------------------------------------------
def fit_household_model_v8(
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
    n_fit_draws: int = N_FIT_DRAWS,
) -> HouseholdCompositionModelV8:
    """Fit candidate 7 (byte-faithful) plus the three candidate-8 deltas."""
    if father_links_child is None:
        father_links_child = hcs3.father_link_births_with_child(birth_records)
    if parent_pairs is None:
        parent_pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    if marital_by_year is None:
        marital_by_year = hcs3._father_marital_by_year(mpanel)

    base_v7 = hcs7.fit_household_model_v7(
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
        marital_by_year=marital_by_year,
        fu_sizes=fu_sizes,
        legal_flag=legal_flag,
        child_record_expo=child_record_expo,
        parent_counts=parent_counts,
    )

    # Delta 1: the train completed-family-size distribution per (band, sex).
    size_map = train_completed_size(parent_pairs, train_ids)
    csd_cell, csd_all = completed_size_dist_by_cell(hh, size_map, train_ids)

    # Delta 3: the band-signed exit-origin + link-coverage channel closures.
    shift, delta3_fit = fit_delta3_retention_link(
        hh,
        mpanel,
        base_v7,
        father_links_child,
        parent_pairs,
        marital_by_year,
        size_map,
        train_ids,
        n_fit_draws=n_fit_draws,
    )

    meta = {
        **base_v7.meta,
        "delta_stream_tag_v7": DELTA_STREAM_TAG_V7,
        "delta_stream_tag_v8": DELTA_STREAM_TAG_V8,
        "fertility_core_lift": {
            "completed_size_dist_train_all": csd_all,
            "size_buckets": list(SIZE_BUCKETS),
            "note": (
                "Delta 1 swaps the sim's completed-family-size distribution to "
                "the train D_train[S] per (band, sex), holding the sim's own "
                "coresidence-given-size and hh_size|size kernels (the Q15 "
                "analytic application realized as a per-person bucket resample)."
            ),
        },
        "cohab_overlay_lift": {
            "band": COHAB_OVERLAY_LIFT_BAND,
            "lift": COHAB_OVERLAY_LIFT,
            "note": (
                "Delta 2 lifts the currently-non-spouse mass at 25-34|female by "
                "the forensics-3 Q9 -0.045 overlay shortfall (Bernoulli "
                "superposition); other female bands untouched."
            ),
        },
        "retention_link_refit": delta3_fit,
    }
    return HouseholdCompositionModelV8(
        base_v7=base_v7,
        completed_size_dist_train=csd_cell,
        completed_size_dist_train_all=csd_all,
        retention_link_shift=shift,
        cohab_overlay_lift=COHAB_OVERLAY_LIFT,
        delta3_fit=delta3_fit,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Delta application helpers (all draw only from the isolated 0xC8 stream)
# --------------------------------------------------------------------------
def _conditional_flip(
    x: np.ndarray,
    weight: np.ndarray,
    subset: np.ndarray,
    target_shift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shift the weighted mean of ``x`` over ``subset`` by ``target_shift``.

    A positive shift flips some ``subset`` rows where ``x`` is False to True
    (adding coresidence); a negative shift flips some True rows to False. The
    per-row flip probability realizes the target shift in expectation (the
    additive-flip realization of an analytic cell-rate adjustment). Returns a new
    boolean array; ``x`` is not mutated.
    """
    out = np.asarray(x, dtype=bool).copy()
    if target_shift == 0.0 or not subset.any():
        return out
    w = np.asarray(weight, dtype=np.float64)
    w_tot = float(w[subset].sum())
    if w_tot <= 0:
        return out
    need = target_shift * w_tot
    if target_shift > 0:
        cand = subset & (~out)
        w_cand = float(w[cand].sum())
        if w_cand <= 0:
            return out
        p = min(max(need / w_cand, 0.0), 1.0)
        u = rng.random(len(out))
        flip = cand & (u < p)
        out[flip] = True
    else:
        cand = subset & out
        w_cand = float(w[cand].sum())
        if w_cand <= 0:
            return out
        p = min(max((-need) / w_cand, 0.0), 1.0)
        u = rng.random(len(out))
        flip = cand & (u < p)
        out[flip] = False
    return out


def apply_fertility_core_lift(
    person_id: np.ndarray,
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    child_counts: np.ndarray,
    coresident_child: np.ndarray,
    hh_size: np.ndarray,
    model: HouseholdCompositionModelV8,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Delta 1: swap ``D_sim[S] -> D_train[S]`` per (band, sex), hold kernels.

    Per person, the completed size ``S_sim`` (max coresident child count over
    waves) is bucketed; per (band, sex) cell the target bucket is redrawn from
    the train ``D_train[cell]`` and ``coresident_child`` is re-emitted from the
    draw's own kernel ``K_sim[cell][bucket]`` while ``hh_size`` is resampled from
    the draw's own rows in ``(cell, bucket)`` -- the Q15 analytic application
    (hold ``K_sim`` / ``H_sim``, swap ``D``). Rows outside the composition bands
    keep their candidate-7 value. Returns the new coresident_child and hh_size
    plus a diagnostic (per-cell sim-vs-counterfactual completed-size means).
    """
    s_sim = sim_completed_size_row(person_id, child_counts)
    bucket_sim = size_bucket(s_sim)
    cor_new = np.asarray(coresident_child, dtype=bool).copy()
    hh_new = np.asarray(hh_size, dtype=np.int64).copy()
    diag_cells: dict[str, Any] = {}
    for bl in _COMPOSITION_BANDS:
        for sx in ("male", "female"):
            cell = (bl, sx)
            m = (band == bl) & (sex == sx)
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            d_train = model.completed_size_dist_train.get(cell)
            if d_train is None:
                continue
            probs = np.array(
                [d_train.get(b, 0.0) for b in SIZE_BUCKETS], dtype=np.float64
            )
            tot = probs.sum()
            if tot <= 0:
                continue
            probs = probs / tot
            # Target bucket per row ~ D_train[cell].
            tgt_idx = rng.choice(len(SIZE_BUCKETS), size=len(idx), p=probs)
            tgt_bucket = np.array(SIZE_BUCKETS, dtype=object)[tgt_idx]
            # The draw's own kernels within the cell.
            w_cell = np.asarray(weight, dtype=np.float64)[idx]
            cor_cell = np.asarray(coresident_child, dtype=bool)[idx]
            hh_cell = np.asarray(hh_size, dtype=np.int64)[idx]
            bkt_cell = bucket_sim[idx]
            k_sim: dict[str, float] = {}
            hh_pool: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for b in SIZE_BUCKETS:
                mb = bkt_cell == b
                wb = float(w_cell[mb].sum())
                k_sim[b] = (
                    float((w_cell[mb] * cor_cell[mb]).sum() / wb)
                    if wb > 0
                    else 0.0
                )
                if mb.any() and wb > 0:
                    hh_pool[b] = (
                        hh_cell[mb],
                        w_cell[mb] / wb,
                    )
                else:
                    hh_pool[b] = (hh_cell, None)
            # Re-emit coresident_child ~ Bernoulli(K_sim[tgt_bucket]).
            k_row = np.array([k_sim[b] for b in tgt_bucket], dtype=np.float64)
            u = rng.random(len(idx))
            cor_new[idx] = u < k_row
            # Resample hh_size from the draw's own (cell, bucket) pool.
            for b in SIZE_BUCKETS:
                sel = np.flatnonzero(tgt_bucket == b)
                if not len(sel):
                    continue
                vals, wprob = hh_pool[b]
                if wprob is None or not len(vals):
                    continue
                pick = rng.choice(len(vals), size=len(sel), p=wprob)
                hh_new[idx[sel]] = vals[pick]
            diag_cells[f"coresident_child.{bl}|{sx}"] = {
                "sim_completed_size_dist": _size_dist(s_sim[idx], w_cell),
                "train_completed_size_dist": d_train,
                "k_sim_given_size": k_sim,
            }
    diag = {"per_cell": diag_cells}
    return cor_new, hh_new, diag


def apply_retention_link_refit(
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    coresident_child: np.ndarray,
    model: HouseholdCompositionModelV8,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Delta 3: additive band-signed exit + link-coverage closure at parent 45+.

    For each delta-3 cell the fitted additive shift ``retention_link_shift[cell]``
    (``-exit_origin`` band-signed, plus ``-link_coverage`` at the older-male
    cells) is realized on the delta-1-lifted coresident_child via a conditional
    flip (:func:`_conditional_flip`). Returns the new coresident_child and a
    per-cell realized-shift diagnostic.
    """
    cor_new = np.asarray(coresident_child, dtype=bool).copy()
    diag: dict[str, Any] = {}
    for cell, shift in model.retention_link_shift.items():
        bl, sx = _cell_of(cell)
        subset = (band == bl) & (sex == sx)
        before = _wshare(weight, cor_new & subset, subset)
        cor_new = _conditional_flip(cor_new, weight, subset, shift, rng)
        after = _wshare(weight, cor_new & subset, subset)
        diag[cell] = {
            "target_shift": float(shift),
            "rate_before": before,
            "rate_after": after,
            "realized_shift": after - before,
        }
    return cor_new, diag


def apply_cohab_overlay_lift(
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    spouse: np.ndarray,
    model: HouseholdCompositionModelV8,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Delta 2: cohabitation-overlay lift at 25-34|female (Bernoulli super.).

    The currently-non-spouse mass at band 25-34 female is flipped to spouse with
    probability ``cohab_overlay_lift`` (0.045), realizing
    ``new = old + 0.045 * (1 - old)`` in expectation. Age-band-specific: no other
    female band and no male band is touched.
    """
    spouse_new = np.asarray(spouse, dtype=bool).copy()
    subset = (
        (band == COHAB_OVERLAY_LIFT_BAND) & (sex == "female") & (~spouse_new)
    )
    before = _wshare(
        weight,
        spouse_new,
        (sex == "female") & (band == COHAB_OVERLAY_LIFT_BAND),
    )
    if subset.any():
        u = rng.random(len(spouse_new))
        flip = subset & (u < model.cohab_overlay_lift)
        spouse_new[flip] = True
    after = _wshare(
        weight,
        spouse_new,
        (sex == "female") & (band == COHAB_OVERLAY_LIFT_BAND),
    )
    diag = {
        "band": COHAB_OVERLAY_LIFT_BAND,
        "lift": float(model.cohab_overlay_lift),
        "rate_before": before,
        "rate_after": after,
        "realized_lift": after - before,
    }
    return spouse_new, diag


def _wshare(weight: np.ndarray, hit: np.ndarray, subset: np.ndarray) -> float:
    w = np.asarray(weight, dtype=np.float64)
    tot = float(w[subset].sum())
    if tot <= 0:
        return 0.0
    return float(w[hit & subset].sum() / tot)


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def _compose_v7(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    base_v7: hcs7.HouseholdCompositionModelV7,
    ids_a: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Run candidate 7's composition and return every array (byte-identical).

    A faithful copy of :func:`hcs7.simulate_draw_v7`'s body up to (but not
    including) the panel build: the same 0xB2B / 0xC2 / 0xC3 / 0xC4 / 0xC5 /
    0xC6 / 0xC7 substreams and train-fitted tables, so every carried array is
    byte-identical to candidate 7. Returns the padded ``pw`` frame plus the
    composed ``spouse`` / ``coresident_parent`` / ``multigen`` /
    ``coresident_child`` / ``coresident_grandchild`` / integer ``child_counts`` /
    ``hh_size`` arrays aligned to the ``pw`` row order. Used by both the delta-3
    fit and :func:`simulate_draw_v8`, so the completed-size basis is the exact
    per-person child COUNT.
    """
    model = base_v7
    base = model.base

    c1_panel = hcs.simulate_draw(hh, mpanel, base, ids_a, draw_seed, 0xB2B)
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

    delta_ss = np.random.SeedSequence([draw_seed, 0xC2])
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

    cohab_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    cohab_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (bnd, sx), rate in model.base_v2.cohab_entry.items():
        cohab_entry_prob[(band_mat == bnd) & (sex_mat == sx)] = rate
    for (bnd, sx), rate in model.base_v2.cohab_exit.items():
        cohab_exit_prob[(band_mat == bnd) & (sex_mat == sx)] = rate
    for (age, sx), rate in model.base_v4.cohab_entry_age.items():
        cohab_entry_prob[(age_mat == age) & (sex_mat == sx)] = rate
    for (age, sx), rate in model.base_v4.cohab_exit_age.items():
        cohab_exit_prob[(age_mat == age) & (sex_mat == sx)] = rate
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

    c4_ss = np.random.SeedSequence([draw_seed, 0xC4])
    legal_resid_ss, _nonfamily2_ss = c4_ss.spawn(2)
    legal_resid_rng = np.random.default_rng(legal_resid_ss)
    lr_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_marg_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (bnd, sx), rate in model.base_v4.legal_residual_entry.items():
        m = (band_mat == bnd) & (sex_mat == sx)
        lr_exit_prob[m] = model.base_v4.legal_residual_exit[(bnd, sx)]
        lr_entry_prob[m] = rate
        lr_marg_prob[m] = model.base_v4.legal_residual_marginal[(bnd, sx)]
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

    sim_panel, sim_births = ft.simulate(
        mpanel, ids_a, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"

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
    base_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    c3_ss = np.random.SeedSequence([draw_seed, 0xC3])
    _custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    c5_ss = np.random.SeedSequence([draw_seed, hcs6.DELTA_STREAM_TAG_V5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)

    mat_leave_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, hcs6.DELTA_STREAM_TAG_V6])
    )
    episode_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V7])
    )

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
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked, _linked_diag = hcs7.custodial_linked_child_counts_v7(
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

    bands_row = pw["band"].to_numpy(dtype=object)
    sexes_row = pw["sex"].to_numpy()
    ages_row = pw["age"].to_numpy()
    two_share = np.full(len(pw), model.parent_count_two_pooled, np.float64)
    for (bnd, sx), share in model.parent_count_two_share.items():
        two_share[(bands_row == bnd) & (sexes_row == sx)] = share
    u_pc = parentcount_rng.random(len(pw))
    parent_count_ego = np.where(u_pc < two_share, 2, 1).astype(np.int64)
    n_parents_ego = np.where(c1_parent, parent_count_ego, 0).astype(np.int64)
    hh_size_base = (
        1
        + spouse.astype(np.int64)
        + child_counts.astype(np.int64)
        + n_parents_ego
    ).astype(np.int64)

    is_55_row = ages_row >= GRANDCHILD_LO
    p_coupled = np.zeros(len(pw), dtype=np.float64)
    for (sx, mg), rate in model.coupling_child_pooled.items():
        p_coupled[is_55_row & (sexes_row == sx) & (c1_multigen == mg)] = rate
    for (bnd, sx, mg), rate in model.coupling_child_given_multigen.items():
        m = (
            is_55_row
            & (bands_row == bnd)
            & (sexes_row == sx)
            & (c1_multigen == mg)
        )
        p_coupled[m] = rate
    u_couple = coupling_rng.random(len(pw))
    coupled_child = u_couple < p_coupled
    grandchild_coupled = c1_multigen & coupled_child & (~c1_parent)
    grandchild_final = np.where(
        is_55_row, grandchild_coupled, grandchild_composed
    )

    obs_skipgen = (
        pw["coresident_grandchild"].to_numpy(dtype=bool)[safe_row]
        & ~pw["multigen"].to_numpy(dtype=bool)[safe_row]
        & valid
    )
    skip_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    skip_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (bnd, sx), rate in model.base_v3.skipgen_entry.items():
        skip_entry_prob[(band_mat == bnd) & (sex_mat == sx)] = rate
    for (bnd, sx), rate in model.base_v3.skipgen_exit.items():
        skip_exit_prob[(band_mat == bnd) & (sex_mat == sx)] = rate
    for ((lo, hi), sx), rate in model.base_v4.skipgen_entry_age.items():
        m = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sx)
        skip_entry_prob[m] = rate
    for ((lo, hi), sx), rate in model.base_v4.skipgen_exit_age.items():
        m = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sx)
        skip_exit_prob[m] = rate
    skipgen_state = hcs._evolve_two_state(
        valid, obs_skipgen, skip_entry_prob, skip_exit_prob, skipgen_rng
    )
    skipgen_row = np.zeros(len(pw), dtype=bool)
    skipgen_row[row_of[valid]] = skipgen_state[valid]
    coresident_grandchild = grandchild_final | skipgen_row

    nonfamily_count = hcs6.sample_nonfamily_count_v6(
        pw, model, nonfamily_rng, hh_size_base
    )
    hh_size = (hh_size_base + nonfamily_count).astype(np.int64)

    return {
        "pw": pw,
        "person_id": pw["person_id"].to_numpy(np.int64),
        "band": pw["band"].to_numpy(dtype=object),
        "sex": pw["sex"].to_numpy(),
        "age": pw["age"].to_numpy(np.int64),
        "weight": pw["weight"].to_numpy(np.float64),
        "spouse": spouse,
        "coresident_parent": c1_parent,
        "multigen": c1_multigen,
        "coresident_child": coresident_child,
        "coresident_grandchild": coresident_grandchild,
        "child_counts": child_counts.astype(np.int64),
        "hh_size": hh_size,
    }


def simulate_draw_v8(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: HouseholdCompositionModelV8,
    ids_a: set[int],
    draw_seed: int,
    delta_stream_tag_v8: int = DELTA_STREAM_TAG_V8,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-8 draw of the side-A holdout households.

    Composes candidate 7 byte-for-byte (:func:`_compose_v7`; carried
    ``coresident_parent`` / ``multigen`` / ``coresident_grandchild`` kept
    UNCHANGED), then applies the three candidate-8 deltas -- fertility-core lift
    (delta 1) and band-signed retention + link-coverage refit (delta 3) on
    ``coresident_child`` / ``hh_size``, and the cohabitation-overlay lift
    (delta 2) on ``coresident_spouse.25-34|female`` -- drawing exclusively from
    the isolated ``SeedSequence([draw_seed, 0xC8])``. With the deltas at their
    zero point the panel is byte-identical to :func:`hcs7.simulate_draw_v7`.
    """
    comp = _compose_v7(hh, mpanel, model.base_v7, ids_a, draw_seed)
    pw = comp["pw"]
    band, sex, weight = comp["band"], comp["sex"], comp["weight"]

    # Isolated 0xC8 substreams: delta 1 (fertility), delta 3 (retention/link),
    # delta 2 (cohab overlay). Isolated from every carried stream.
    c8_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v8])
    fert_ss, retention_ss, cohab_ss = c8_ss.spawn(3)
    fert_rng = np.random.default_rng(fert_ss)
    retention_rng = np.random.default_rng(retention_ss)
    cohab_rng = np.random.default_rng(cohab_ss)

    # Delta 1: fertility-core lift (coresident_child + hh_size).
    cor_d1, hh_d1, fert_diag = apply_fertility_core_lift(
        comp["person_id"],
        band,
        sex,
        weight,
        comp["child_counts"],
        comp["coresident_child"],
        comp["hh_size"],
        model,
        fert_rng,
    )
    # Delta 3: band-signed retention + link-coverage refit (coresident_child).
    cor_d3, retention_diag = apply_retention_link_refit(
        band, sex, weight, cor_d1, model, retention_rng
    )
    # Delta 2: cohabitation-overlay lift (coresident_spouse.25-34|female).
    spouse_d2, cohab_diag = apply_cohab_overlay_lift(
        band, sex, weight, comp["spouse"], model, cohab_rng
    )

    # Build the panel EXACTLY as candidate 7 does (carries byte-identical).
    sim_pw = pw.copy()
    sim_pw["coresident_spouse"] = spouse_d2
    sim_pw["coresident_parent"] = comp["coresident_parent"]
    sim_pw["coresident_child"] = cor_d3
    sim_pw["coresident_grandchild"] = comp["coresident_grandchild"]
    sim_pw["multigen"] = comp["multigen"]
    sim_pw["hh_size"] = hh_d1
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
        "linked_persistence_rho": float(model.linked_episode_persistence),
        "fertility_core_lift": fert_diag,
        "retention_link_refit": retention_diag,
        "cohab_overlay_lift": cohab_diag,
        "delta_stream_tag_v8": delta_stream_tag_v8,
    }
    return panel, diagnostics


# --------------------------------------------------------------------------
# Delta-specific fit-vs-raw reproduction checks (Q14 / Q15 / Q16)
# --------------------------------------------------------------------------
def c8_delta_checks(
    model: HouseholdCompositionModelV8,
    forensics5: dict[str, Any],
) -> dict[str, Any]:
    """Record the three deltas' fit-vs-raw checks against forensics-5.

    Reproduces the Q15 fertility-lever headline (``hh_size.5+`` 0.128 -> 0.144;
    ``coresident_child.55-64|male`` 0.213 -> 0.255), the Q16 cohab-overlay lift
    (0.588 -> 0.606), and the Q14 band-signed exit-origin + link-coverage
    channels each delta implements -- from the fitted model beside the forensics-5
    measured quantities.
    """
    q14 = forensics5["question_14_older_parent_adult_child_supply"]["per_cell"]
    q15 = forensics5["question_15_single_lever_convergence"]
    q16 = forensics5["question_16_fragile_spouse_cell"]

    delta_1 = {
        "mechanism": (
            "swap the sim completed-family-size distribution D_sim[S] to the "
            "train D_train[S] per (band, sex), holding the sim coresidence / "
            "hh_size kernels (Q15 analytic application)"
        ),
        "reproduction_hh_size_5plus": {
            "seed_mean_sim": q15["hh_size_cells"]["hh_size.5+"][
                "seed_mean_sim"
            ],
            "seed_mean_counterfactual_lever": q15["hh_size_cells"][
                "hh_size.5+"
            ]["seed_mean_counterfactual"],
            "seed_mean_reference": q15["hh_size_cells"]["hh_size.5+"][
                "seed_mean_reference"
            ],
            "counterfactual_clears_holdout_failing_seeds": q15[
                "hh_size_cells"
            ]["hh_size.5+"]["counterfactual_clears_holdout_failing_seeds"],
            "headline": "0.128 -> 0.144 (vs reference 0.139)",
        },
        "reproduction_coresident_child_55_64_male": {
            "seed_mean_sim": q15["older_male_supply_cells"][
                "coresident_child.55-64|male"
            ]["seed_mean_sim"],
            "seed_mean_counterfactual_lever": q15["older_male_supply_cells"][
                "coresident_child.55-64|male"
            ]["seed_mean_counterfactual"],
            "seed_mean_reference": q15["older_male_supply_cells"][
                "coresident_child.55-64|male"
            ]["seed_mean_reference"],
            "counterfactual_clears_holdout_failing_seeds": q15[
                "older_male_supply_cells"
            ]["coresident_child.55-64|male"][
                "counterfactual_clears_holdout_failing_seeds"
            ],
            "headline": "0.213 -> 0.255 (vs reference 0.262)",
        },
        "trade_holds_hh_3_4": q15["convergence_verdict"][
            "hh_size_3_4_trade_holds"
        ],
        "cleared_child_cells_hold": q15["convergence_verdict"][
            "cleared_child_cells_hold"
        ],
        "target_completed_size_dist_train_all": (
            model.completed_size_dist_train_all
        ),
    }

    delta_2 = {
        "mechanism": (
            "Bernoulli superposition new = old + 0.045 * (1 - old) on the "
            "currently-non-spouse mass at 25-34|female (Q16 cohab-overlay lift)"
        ),
        "overlay_shortfall_applied": model.cohab_overlay_lift,
        "reproduction_spouse_25_34_female": {
            "seed_mean_sim_full": q16["seed_mean_sim_full"],
            "seed_mean_lifted_full": q16["seed_mean_lifted_full"],
            "lift_clears_holdout_failing_seeds": q16[
                "lift_clears_holdout_failing_seeds"
            ],
            "no_collateral_spouse_cell_moves_out": q16[
                "no_collateral_spouse_cell_moves_out"
            ],
            "headline": "0.588 -> 0.606 (vs reference 0.621)",
        },
    }

    # Delta 3: band-sign fit-vs-raw against the Q14 measured channels.
    band_sign: dict[str, Any] = {}
    fit = model.delta3_fit["per_cell"]
    for cell in tuple(
        dict.fromkeys(RETENTION_EXIT_CELLS + LINK_COVERAGE_CELLS)
    ):
        measured = q14[cell]["channels"]
        fitted = fit[cell]
        band_sign[cell] = {
            "forensics5_measured": {
                "exit_origin": measured["exit_origin"],
                "link_coverage": measured["link_coverage"],
                "v7_persistence_enumeration_interaction": measured[
                    "v7_persistence_enumeration_interaction"
                ],
                "fertility_origin": measured["fertility_origin"],
                "cell_miss": q14[cell]["cell_miss_sim_minus_reference"],
            },
            "candidate8_fit_train": {
                "exit_origin": fitted["exit_origin"],
                "link_coverage": fitted["link_coverage"],
                "v7_persistence_enumeration_interaction": fitted[
                    "v7_persistence_enumeration_interaction"
                ],
                "fertility_origin": fitted["fertility_origin"],
                "cell_miss": fitted["cell_miss"],
            },
            "applied_shift": fitted["applied_shift"],
            "closes": fitted["closes"],
            "exit_sign": (
                "reduce_over_retention"
                if fitted["exit_origin"] > 0
                else "lift_under_retention"
            ),
        }
    delta_3 = {
        "mechanism": (
            "close the Q14 EXIT-ORIGIN channel band-signed at parent 45+ "
            "(lift 65-74 under-retention, reduce 45-54|female over-retention) "
            "plus the LINK-COVERAGE channel at older-male bands; v7 interaction "
            "left as the named residual"
        ),
        "retention_exit_cells": list(RETENTION_EXIT_CELLS),
        "link_coverage_cells": list(LINK_COVERAGE_CELLS),
        "band_sign_fit_vs_raw": band_sign,
        "link_coverage_share": {
            cell: {
                "link_coverage_channel": fit[cell]["link_coverage"],
                "share_of_cell_miss": (
                    fit[cell]["link_coverage"] / fit[cell]["cell_miss"]
                    if fit[cell]["cell_miss"] != 0
                    else None
                ),
            }
            for cell in LINK_COVERAGE_CELLS
        },
    }

    return {
        "delta_1_fertility_core_lift": delta_1,
        "delta_2_cohab_overlay_lift": delta_2,
        "delta_3_retention_link_refit": delta_3,
        "named_residual_v7_interaction": {
            "coresident_child.55-64|male": q14["coresident_child.55-64|male"][
                "channels"
            ]["v7_persistence_enumeration_interaction"],
            "coresident_child.65-74|male": q14["coresident_child.65-74|male"][
                "channels"
            ]["v7_persistence_enumeration_interaction"],
            "note": (
                "The v7 persistence/enumeration interaction (+0.008 / -0.010) "
                "is deliberately UNTOUCHED -- the named residual left for a "
                "targeted candidate-9 forensics."
            ),
        },
    }
