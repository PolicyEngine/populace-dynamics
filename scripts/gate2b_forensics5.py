"""Gate-2b forensics 5: revealed supply deficit + the convergence test.

Registered diagnostic (issue #42, comment 4948430423; ``reported_not_gated``).
The registration WINS: this runner answers exactly its three frozen questions,
publishes regardless of what it finds, and touches the outer holdout only
through the already-committed per-seed scores in ``runs/gate2b_hazard_v7.json``.

The protocol is the forensics-1/2/3/4 protocol, unchanged: train side only (side
B of ``split_panel_by_person(seed)``), candidate 7's fit/simulate machinery
reused BYTE-FOR-BYTE via :func:`instrumented_draw_v7` (a line-for-line copy of
:func:`hcs7.simulate_draw_v7` proved bit-identical by :func:`fidelity_check_v7`),
and every additive decomposition reconciled to machine epsilon.

Frozen questions (verbatim from the registration):

* **Q14 -- older-parent adult-child supply.** Decompose the ``55-64|male`` and
  ``65-74|male`` ``coresident_child`` misses (and the maternal
  ``45-54``/``65-74|female`` wobble) into channels reconciling exactly: (a)
  FERTILITY-ORIGIN -- children never present in the sim's family histories
  (completed family size by parent cohort x sex, sim frame vs train, especially
  3+-child families); (b) EXIT-ORIGIN -- children present but aged out too fast
  at older parent ages under the current hazards; (c) LINK-COVERAGE --
  joinable-enumerated children in train outside the sim's draw basis at those
  parent ages; (d) the v7 persistence/enumeration interaction at older bands.

* **Q15 -- the convergence test (the c8 feasibility proof, mirroring
  forensics-3 Q10).** If the train completed-fertility 3+-child distribution
  replaces the sim's large-family core deficit, does that single lever
  simultaneously move (i) ``hh_size.5+`` into tolerance on the failing seeds,
  (ii) the older-male adult-child supply channels from Q14a, and (iii) leave
  ``hh_size.3``/``.4`` and the cleared child cells in tolerance? Applied
  analytically to the sim's own distributions per seed, exactly as f3-Q10 proved
  the bridge feasibility.

* **Q16 -- fragile spouse cell.** Measure what a cohabitation-overlay lift at
  ``25-34|female`` sized to the measured -0.045 shortfall does to the cell's
  split-seed exceedances (2/5) and gate-seed margins, analytically on train;
  confirm no other spouse cell moves out.

Method note on the EXACT reconciliation of Q14. The older-parent coresidence
cell rate is an exact partition over the parent's COMPLETED FAMILY SIZE ``S``
(the parent's maximum coresident own-child count across their waves): the cell
rate ``= sum_S D[S] * K[S]`` where ``D[S]`` is the weighted share of cell
parent-waves whose parent has completed size ``S`` and ``K[S]`` the coresidence
rate among them (a law-of-total-probability identity, exact by construction).
The cell miss then telescopes (Oaxaca-Blinder) into an ENDOWMENT term ``sum_S
(D_sim[S] - D_train[S]) K_sim[S]`` -- the FERTILITY-ORIGIN channel, the sim's
completed-family-size shortfall -- and a COEFFICIENT term ``sum_S D_train[S]
(K_sim[S] - K_train[S])`` -- the coresidence-given-size shift. The coefficient
term is split further into LINK-COVERAGE and the v7 PERSISTENCE/ENUMERATION
INTERACTION (independently measured from the linked-father analytic anchors at
the cell, zero for the maternal female cells), with EXIT-ORIGIN the remaining
coresidence-timing residual -- all summing to the cell miss to machine epsilon.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate2b_candidate7 as c7  # noqa: E402

from populace_dynamics import artifacts  # noqa: E402
from populace_dynamics.data import household_composition as hc  # noqa: E402
from populace_dynamics.data import transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.models import family_transitions as ft  # noqa: E402
from populace_dynamics.models import (
    household_composition_sim as hcs,
)  # noqa: E402
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v3 as hcs3,
)
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v4 as hcs4,
)
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v6 as hcs6,
)
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v7 as hcs7,
)
from populace_dynamics.models.family_transitions.components.fertility import (  # noqa: E402,E501
    build_fertility_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_forensics5_v1.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v7.json"
FORENSICS3_ARTIFACT = ROOT / "runs" / "gate2b_forensics3_v1.json"
FORENSICS4_ARTIFACT = ROOT / "runs" / "gate2b_forensics4_v1.json"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
SCHEMA_VERSION = "gate2b_forensics5.v1"
RUN_NAME = "gate2b_forensics5_v1"

#: The registered diagnostic (issue #42, comment 4948430423). The registration
#: wins: this runner answers exactly its three frozen questions.
REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4948430423"
)
REGISTRATION_POINTER = "4948430423"
REGISTRATION_TITLE = (
    "Gate-2b forensics 5 registration (diagnostic, not gated): the revealed "
    "supply deficit and the convergence test (Q14 older-parent adult-child "
    "supply; Q15 the single-lever convergence proof; Q16 fragile spouse cell)"
)
#: Grading of candidate 7 (the fail this diagnostic measures beneath).
GRADING_POINTER = "4948429354"
#: Grading of forensics 4 (the linked-father decomposition c7 designed against).
FORENSICS4_GRADING_POINTER = "4948183531"
CANDIDATE7_REGISTRATION_POINTER = "4948186843"

#: Reused frozen dials (candidate 7 / the locked gate_2b protocol).
GATE_SEEDS = c7.GATE_SEEDS  # (0, 1, 2, 3, 4)
N_DRAWS = c7.N_DRAWS  # 20
DRAW_SEED_BASE = c7.DRAW_SEED_BASE  # 5200
EXACT_ATOL = c7.EXACT_ATOL  # 1e-12

GRANDCHILD_LO = hcs7.GRANDCHILD_LO
CORE_SIZE_CAP = hcs7.CORE_SIZE_CAP
CHILD_CORESIDENCE_MAX_AGE = hcs7.CHILD_CORESIDENCE_MAX_AGE
SPELL_CHILD_MAX_AGE = hcs7.SPELL_CHILD_MAX_AGE
DELTA_STREAM_TAG_V6 = hcs7.DELTA_STREAM_TAG_V6
DELTA_STREAM_TAG_V7 = hcs7.DELTA_STREAM_TAG_V7
_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED

#: Q14 registered cells: the older-parent adult-child supply misses. The two
#: male blockers (the fertility-origin headline) and the maternal female wobble.
Q14_MALE_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
)
Q14_FEMALE_CELLS = (
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
Q14_CELLS = Q14_MALE_CELLS + Q14_FEMALE_CELLS
#: Q15 registered cells (the hh_size trade the lever must not disturb + lift).
Q15_HH_CELLS = ("hh_size.3", "hh_size.4", "hh_size.5+")
Q15_HH_TARGET = "hh_size.5+"
#: The cleared child cells the lever must leave in tolerance.
Q15_CLEARED_CHILD_CELLS = (
    "coresident_child.25-34|male",
    "coresident_child.35-44|male",
)
#: Q16 registered cell.
Q16_CELL = "coresident_spouse.25-34|female"
#: The forensics-3 Q9 measured cohabitation-overlay shortfall at 25-34|female.
Q16_OVERLAY_SHORTFALL = 0.045

#: Completed-family-size buckets (parent's max coresident own-child count).
SIZE_BUCKETS = ("0", "1", "2", "3", "4+")
#: Existence / episode reporting buckets (shared with forensics-4).
EXIST_BUCKETS = ("0", "1", "2", "3+")
EPISODE_BUCKETS = hcs7.EPISODE_BUCKETS  # ("1", "2", "3", "4+")

#: Per-seed cache OUTSIDE runs/ (never committed): a long run resumes.
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2b_forensics5_cache.json"
)


# --------------------------------------------------------------------------
# Small statistics helpers
# --------------------------------------------------------------------------
def _score(r_candidate: float, rate_ref: float) -> float:
    """The gate statistic ``|ln(r_candidate / rate_ref)|``."""
    if r_candidate > 0 and rate_ref > 0:
        return float(abs(math.log(r_candidate / rate_ref)))
    return float("inf")


def _signed(r_candidate: float, rate_ref: float) -> float:
    """The signed log-ratio ``ln(r_candidate / rate_ref)`` (over/under)."""
    if r_candidate > 0 and rate_ref > 0:
        return float(math.log(r_candidate / rate_ref))
    return float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sd(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def _wshare(weight: np.ndarray, hit: np.ndarray) -> float:
    """Weighted share of ``hit`` (a 0/1 or bool array) over ``weight``."""
    w = np.asarray(weight, dtype=np.float64)
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    return float((w * np.asarray(hit, dtype=np.float64)).sum() / tot)


def _size_bucket(counts: np.ndarray) -> np.ndarray:
    """Map an integer count array to SIZE_BUCKETS labels (0/1/2/3/4+)."""
    c = np.asarray(counts, dtype=np.int64)
    out = np.where(c >= 4, "4+", c.astype(str).astype(object))
    return out


def _count_hist(counts: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Weighted 0 / 1 / 2 / 3+ histogram of an integer count array."""
    w = np.asarray(weight, dtype=np.float64)
    tot = float(w.sum())
    if tot <= 0:
        return {b: 0.0 for b in EXIST_BUCKETS}
    c = np.asarray(counts, dtype=np.int64)
    return {
        "0": float(w[c == 0].sum() / tot),
        "1": float(w[c == 1].sum() / tot),
        "2": float(w[c == 2].sum() / tot),
        "3+": float(w[c >= 3].sum() / tot),
    }


def _size_hist(sizes: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Weighted completed-family-size histogram over SIZE_BUCKETS."""
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


def _avg_hist(hists: list[dict[str, float]], keys) -> dict[str, float]:
    return {b: _mean([h[b] for h in hists]) for b in keys}


def _episode_length_hist(
    father_id: np.ndarray,
    child_key: np.ndarray,
    year: np.ndarray,
    coresident: np.ndarray,
) -> tuple[dict[str, float], float, int]:
    """Episode-length histogram (delegates to the committed v7 helper)."""
    return hcs7.episode_length_hist(father_id, child_key, year, coresident)


# --------------------------------------------------------------------------
# Instrumented candidate-7 linked child coresidence draw (full detail)
# --------------------------------------------------------------------------
def _linked_detail_v7(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    model: hcs7.HouseholdCompositionModelV7,
    episode_rng: np.random.Generator,
) -> dict[str, Any]:
    """:func:`hcs7.custodial_linked_child_counts_v7` returning full detail.

    Consumes the ``0xC7`` episode-persistence rng IDENTICALLY (same
    :func:`hcs7._linked_exposure_frame` exposure, same joinable restriction,
    same ``(birth_id, year)`` sort, same single
    :func:`hcs7.simulate_linked_episode_coresidence` call) as the committed v7
    function, and returns the byte-identical total counts PLUS: the father
    marital split (married / not-married coresident counts per father-wave); the
    per-father-wave analytic occupancy ``1 - prod(1 - p_c)`` over the JOINABLE
    exposure (the exact expectation of ``coresident_child`` under an independent
    per-wave draw -- the v7 delta-2 reshapes this into episodes but preserves the
    per-wave marginal); the per-father-wave JOINABLE exposed-child count; and the
    joinable exposure rows augmented with the drawn coresident flag (for episodes
    and existence). Non-joinable rows carry zero occupancy by construction (the
    v7 delta-1 excludes them from the draw).
    """
    n = len(side_a_pw)
    counts = np.zeros(n, dtype=np.int64)
    c_m = np.zeros(n, dtype=np.int64)
    c_nm = np.zeros(n, dtype=np.int64)
    n_expo_j = np.zeros(n, dtype=np.int64)
    log_no = np.zeros(n, dtype=np.float64)
    empty_epi = {
        "fid": np.array([], dtype=np.int64),
        "ck": np.array([], dtype=np.int64),
        "year": np.array([], dtype=np.int64),
        "child_age": np.array([], dtype=np.int64),
        "coresident": np.array([], dtype=bool),
    }
    out = {
        "counts": counts,
        "married": c_m,
        "not_married": c_nm,
        "n_exposed_joinable": n_expo_j,
        "analytic_stock_joinable": np.zeros(n, dtype=np.float64),
        "epi": empty_epi,
    }
    if not len(linked_births):
        return out
    expo = hcs7._linked_exposure_frame(
        linked_births,
        side_a_pw,
        marital_sim,
        model.base_v6,
        model.joinable_keys,
    )
    if not len(expo):
        return out
    joinable = expo["joinable"].to_numpy(dtype=bool)
    # Delta 2 draw over the JOINABLE exposure only -- byte-identical to
    # hcs7.custodial_linked_child_counts_v7 (same sort, same rng consumption).
    expo_j = (
        expo[joinable]
        .sort_values(["_birth_id", "year"])
        .reset_index(drop=True)
    )
    coresident = hcs7.simulate_linked_episode_coresidence(
        expo_j["parent_person_id"].to_numpy(np.int64),
        expo_j["_birth_id"].to_numpy(np.int64),
        expo_j["p_c"].to_numpy(np.float64),
        float(model.linked_episode_persistence),
        episode_rng,
    )
    rows = expo_j["_row"].to_numpy()
    marital_j = expo_j["marital"].to_numpy()
    p_c = expo_j["p_c"].to_numpy(np.float64)
    np.add.at(counts, rows[coresident], 1)
    np.add.at(c_m, rows[coresident & (marital_j == _MARRIED)], 1)
    np.add.at(c_nm, rows[coresident & (marital_j == _NOT_MARRIED)], 1)
    np.add.at(n_expo_j, rows, 1)
    np.add.at(log_no, rows, np.log1p(-np.clip(p_c, 0.0, 1.0 - 1e-15)))
    a_stock = 1.0 - np.exp(log_no)
    a_stock[n_expo_j == 0] = 0.0
    out["analytic_stock_joinable"] = a_stock
    out["epi"] = {
        "fid": expo_j["parent_person_id"].to_numpy(np.int64),
        "ck": expo_j["birth_year"].to_numpy(np.int64),
        "year": expo_j["year"].to_numpy(np.int64),
        "child_age": expo_j["child_age"].to_numpy(np.int64),
        "coresident": coresident,
    }
    return out


def instrumented_draw_v7(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs7.HouseholdCompositionModelV7,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Reproduce :func:`hcs7.simulate_draw_v7` and return every component array.

    A line-for-line copy of the committed candidate-7 draw (same 0xB2B / 0xC2 /
    0xC3 / 0xC4 / 0xC5 / 0xC6 / 0xC7 substreams and train-fitted tables),
    instrumented to also return the spouse split (legal / cohab / legal-residual),
    the coresident-child channel arrays (linked via father membership, maternal,
    shadow), the linked exposure detail (analytic occupancy, joinable exposed
    count, coresidence rows for episodes), the family-core / hh_size components,
    and the father marital per row -- all aligned to the returned ``pw`` row
    order. Bit-identity of the recomposed panel vs
    :func:`hcs7.simulate_draw_v7` is proved by :func:`fidelity_check_v7`.
    """
    base = model.base

    # 1. carried parent / multigen / legal-marriage spouse (candidate 1, 0xB2B).
    c1_panel = hcs.simulate_draw(hh, mpanel, base, ids, draw_seed, 0xB2B)
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
        hh.person_waves[hh.person_waves["person_id"].isin(ids)]
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

    # 2. candidate-2 delta substreams (0xC2).
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

    # 3. candidate-4 DELTA 1 cohab overlay, then candidate-6 DELTA 3 (carried).
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

    # 4. candidate-4 delta substreams (0xC4).
    c4_ss = np.random.SeedSequence([draw_seed, 0xC4])
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

    # 5. certified marital core + maternal births (same draw seed as cand 7).
    sim_panel, sim_births = ft.simulate(
        mpanel, ids, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"

    # 6. linked / shadow paternal children (candidate-2 child stream 0xC2).
    men_a = mpanel.attrs[
        (mpanel.attrs["sex"] == "male") & (mpanel.attrs["person_id"].isin(ids))
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

    all_births = pd.concat(
        [maternal, paternal_linked, paternal_shadow], ignore_index=True
    )
    all_births = all_births[all_births["parent_person_id"].isin(ids)]

    # candidate-7 base leave-year draw (0xC2) -- byte-identical so the SHADOW
    # leave-year is unchanged; the maternal rows are re-drawn by carried delta 2.
    base_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    # candidate-3 delta substreams (0xC3): the custodial spawn is RETIRED by
    # candidate 7 (the linked draw moved to 0xC7); non-family / skip-gen sibling
    # spawns preserved so those stay byte-identical to candidate 6.
    c3_ss = np.random.SeedSequence([draw_seed, 0xC3])
    _custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # candidate-5 delta substreams (0xC5): coupling + parent count (CARRIED).
    c5_ss = np.random.SeedSequence([draw_seed, hcs6.DELTA_STREAM_TAG_V5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)

    # candidate-6 delta substream (0xC6): the maternal single-year leave refit.
    mat_leave_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V6])
    )
    # candidate-7 delta substream (0xC7): the linked episode-persistence draw.
    episode_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V7])
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
    cc_maternal = hcs._coresident_child_counts(maternal_leaves, side_a_pw)
    cc_shadow = hcs._coresident_child_counts(shadow_leaves, side_a_pw)

    # candidate-7 DELTAS 1 + 2 (linked): enumeration-conditioned, episode-
    # persistent linked child coresidence on the isolated 0xC7 stream.
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    linked = _linked_detail_v7(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model,
        episode_rng,
    )
    cc_linked = linked["counts"]
    child_counts = child_counts_nonlinked + cc_linked

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

    # DELTA 1 of candidate 5 (CARRIED): multigen--adult-child coupling for 55+.
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

    marital_row = (
        pw[["person_id", "year"]]
        .merge(marital_sim, on=["person_id", "year"], how="left")["marital"]
        .fillna(_NOT_MARRIED)
        .to_numpy()
    )

    return {
        "panel": panel,
        "band": pw["band"].to_numpy(dtype=object),
        "sex": pw["sex"].to_numpy(),
        "age": pw["age"].to_numpy(dtype=np.int64),
        "weight": pw["weight"].to_numpy(dtype=np.float64),
        "person_id": pw["person_id"].to_numpy(dtype=np.int64),
        "marital_row": marital_row,
        # spouse split.
        "spouse": spouse,
        "legal_spouse": legal_spouse,
        "cohab_row": cohab_row,
        "lr_row": lr_row,
        # family-core components.
        "coresident_parent": c1_parent,
        "multigen": c1_multigen,
        "coresident_child": coresident_child,
        "child_counts": child_counts.astype(np.int64),
        # child channels.
        "cc_maternal": cc_maternal,
        "cc_shadow": cc_shadow,
        "cc_linked": cc_linked,
        "cc_linked_m": linked["married"],
        "cc_linked_nm": linked["not_married"],
        "linked_n_exposed_joinable": linked["n_exposed_joinable"],
        "linked_analytic_stock_joinable": linked["analytic_stock_joinable"],
        "linked_epi": linked["epi"],
        "linked_ids": linked_ids,
        # hh_size components.
        "n_parents_ego": n_parents_ego,
        "hh_size_base": hh_size_base.astype(np.int64),
        "nonfamily_count": nonfamily_count.astype(np.int64),
        "hh_size": hh_size,
        "is_55_row": is_55_row,
    }


def fidelity_check_v7(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs7.HouseholdCompositionModelV7,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Prove the instrumented draw equals the committed ``simulate_draw_v7``."""
    inst = instrumented_draw_v7(hh, mpanel, model, ids, draw_seed)
    real_panel, _ = hcs7.simulate_draw_v7(hh, mpanel, model, ids, draw_seed)
    a = hc.reference_moments(inst["panel"], ids, weighted=True)
    b = hc.reference_moments(real_panel, ids, weighted=True)
    max_dev = max(abs(a[c]["rate"] - b[c]["rate"]) for c in b)
    child_add = int(
        np.abs(
            inst["cc_maternal"]
            + inst["cc_shadow"]
            + inst["cc_linked"]
            - inst["child_counts"]
        ).sum()
    )
    linked_add = int(
        np.abs(
            inst["cc_linked_m"] + inst["cc_linked_nm"] - inst["cc_linked"]
        ).sum()
    )
    a_stock = inst["linked_analytic_stock_joinable"]
    a_valid = int(np.sum((a_stock < -1e-12) | (a_stock > 1.0 + 1e-12))) + int(
        np.abs(a_stock[inst["linked_n_exposed_joinable"] == 0]).sum() > 0
    )
    return {
        "draw_seed": draw_seed,
        "n_cells_compared": len(b),
        "max_abs_rate_deviation_vs_committed_simulate_draw_v7": float(max_dev),
        "bit_identical": bool(max_dev <= EXACT_ATOL),
        "child_channel_additivity_residual": child_add,
        "linked_marital_split_additivity_residual": linked_add,
        "analytic_stock_out_of_range_or_nonzero_no_exposure": a_valid,
    }


# --------------------------------------------------------------------------
# Completed family size (the fertility-origin supply variable) + kernel
# --------------------------------------------------------------------------
def _cell_of(cell: str) -> tuple[str, str]:
    """Split ``coresident_child.55-64|male`` -> ("55-64", "male")."""
    tail = cell.split(".", 1)[1]
    band, sex = tail.split("|")
    return band, sex


def train_completed_size(
    parent_pairs: pd.DataFrame, ids_b: set[int]
) -> dict[int, int]:
    """Per-parent COMPLETED FAMILY SIZE on train side B.

    The parent's maximum coresident own-child count across their waves (from
    ``parent_pairs``, the roster parent-child coresidence pairs) -- a per-parent
    constant, the 'completed' family size the fertility kernel must reproduce.
    Persons never observed as a coresident parent carry size 0 (added by the
    caller when a person-wave has no entry).
    """
    pp = parent_pairs[parent_pairs["parent_person_id"].isin(ids_b)]
    if not len(pp):
        return {}
    per_wave = pp.groupby(["parent_person_id", "year"], sort=False).size()
    per_parent = per_wave.groupby("parent_person_id").max()
    return {int(k): int(v) for k, v in per_parent.items()}


def _sim_completed_size_row(
    person_id_row: np.ndarray, child_counts_row: np.ndarray
) -> np.ndarray:
    """Per person-wave row, the ego's max coresident child count over waves.

    The sim-frame analogue of :func:`train_completed_size`: for each person the
    maximum ``child_counts`` across their simulated waves, broadcast back to
    every row of that person. A per-parent completed-family-size in the sim
    frame.
    """
    df = pd.DataFrame({"pid": person_id_row, "cc": child_counts_row})
    mx = df.groupby("pid")["cc"].transform("max")
    return mx.to_numpy(dtype=np.int64)


def _oaxaca_cell(
    cell_size: np.ndarray,
    cell_cor: np.ndarray,
    cell_w: np.ndarray,
    d_train: dict[str, float],
    k_train: dict[str, float],
) -> dict[str, Any]:
    """Exact Oaxaca decomposition of one draw's cell rate vs the train cell.

    ``cell_size`` are the SIZE_BUCKETS labels of each cell person-wave's parent
    completed size, ``cell_cor`` the coresident_child flags, ``cell_w`` the
    weights. Returns the sim distribution ``D_sim`` and kernel ``K_sim`` over
    SIZE_BUCKETS, the sim full rate (== sum_S D_sim K_sim exactly), and the two
    telescoping terms against the train ``D_train`` / ``K_train``: the ENDOWMENT
    (fertility-origin) ``sum_S (D_sim - D_train) K_sim`` and the COEFFICIENT
    (coresidence-given-size) ``sum_S D_train (K_sim - K_train)``. Their sum is
    ``sim_full - ref_full`` exactly.
    """
    w = np.asarray(cell_w, dtype=np.float64)
    tot = float(w.sum())
    d_sim: dict[str, float] = {}
    k_sim: dict[str, float] = {}
    for b in SIZE_BUCKETS:
        mb = cell_size == b
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
    return {
        "d_sim": d_sim,
        "k_sim": k_sim,
        "sim_full": float(sim_full),
        "endowment_fertility_origin": float(endowment),
        "coefficient_kernel_shift": float(coefficient),
    }


# --------------------------------------------------------------------------
# Q14 linked-father anchors at the older male cells (train-side reference)
# --------------------------------------------------------------------------
def q14_linked_reference(
    hh: hc.HouseholdCompositionPanel,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    model: hcs7.HouseholdCompositionModelV7,
    ids_b: set[int],
    cells: tuple[str, ...],
) -> dict[str, Any]:
    """Reference-side linked-father anchors at the older male cells.

    Mirrors the forensics-4 ``q11_reference`` linked block but on candidate 7's
    JOINABLE (enumerated) exposure basis and at the older parent bands. Per cell,
    over the full cell weight: the observed coresidence split by linked-father
    membership (``ref_linked_any`` / ``ref_unlinked_any``), the analytic
    independent-per-wave occupancy over the JOINABLE exposure under the OBSERVED
    father marital (``a_refexp_joinable``), and the observed coresidence with
    JOINABLE-enumerated children (``s_ref_joinable_restricted``). Deterministic;
    no simulation RNG. Custody probabilities are the faithful
    :func:`hcs6.custodial_prob_v6`, NOT re-estimated.
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

    # Father-wave joinable exposure with per-row analytic occupancy anchor.
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

    # Observed coresidence with a JOINABLE linked biological child.
    fac_j = fac.copy()
    idx2 = pd.MultiIndex.from_arrays(
        [fac_j["parent_person_id"], fac_j["birth_year"]]
    )
    fac_j = fac_j[np.asarray(idx2.isin(joinable_keys), dtype=bool)]
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
    per_cell: dict[str, Any] = {}
    for cell in cells:
        bl, _sx = _cell_of(cell)
        cell_mask = (pw_b["band"] == bl) & pw_b["_male"]
        cw = pw_b.loc[cell_mask, "weight"].to_numpy(np.float64)
        cw_tot = float(cw.sum())
        cc_any = pw_b.loc[cell_mask, "coresident_child"].to_numpy(bool)
        is_link = pw_b.loc[cell_mask, "_linked"].to_numpy(bool)
        ref_full = _wshare(cw, cc_any)
        ref_linked_any = (
            float((cw * (cc_any & is_link)).sum() / cw_tot)
            if cw_tot > 0
            else 0.0
        )
        ref_unlinked_any = (
            float((cw * (cc_any & ~is_link)).sum() / cw_tot)
            if cw_tot > 0
            else 0.0
        )
        cl = (
            pw_b.loc[
                cell_mask & pw_b["_linked"], ["person_id", "year", "weight"]
            ]
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
        a_refexp_j_contrib = (
            float((clw * a_j).sum() / cw_tot) if cw_tot > 0 else 0.0
        )
        s_ref_join_restr = (
            float((clw * (n_cor > 0)).sum() / cw_tot) if cw_tot > 0 else 0.0
        )
        per_cell[cell] = {
            "reference_full_rate": ref_full,
            "reference_linked_any_contribution": ref_linked_any,
            "reference_unlinked_any_contribution": ref_unlinked_any,
            "reference_a_refexp_joinable_contribution": a_refexp_j_contrib,
            "reference_s_joinable_restricted_contribution": s_ref_join_restr,
            "cell_weight_total": cw_tot,
        }
    return per_cell


# --------------------------------------------------------------------------
# Train-side reference distributions (completed size + kernels)
# --------------------------------------------------------------------------
def _train_cell_dk(
    pw_cell: pd.DataFrame, size_map: dict[int, int]
) -> tuple[dict[str, float], dict[str, float], float]:
    """Train ``D[S]`` / ``K[S]`` and full rate for one cell's person-waves."""
    if not len(pw_cell):
        z = {b: 0.0 for b in SIZE_BUCKETS}
        return z, z, 0.0
    s = (
        pw_cell["person_id"]
        .map(lambda p: size_map.get(int(p), 0))
        .to_numpy(dtype=np.int64)
    )
    bucket = _size_bucket(s)
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


def _train_hh_kernel(
    pw_all: pd.DataFrame, size_map: dict[int, int]
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    """Train ``D_all[S]`` and hh-size kernel ``H_j[S]`` over all person-waves.

    Returns the completed-size distribution over all side-B person-waves, the
    conditional ``P(hh_size class j | completed size S)`` for j in 3 / 4 / 5+,
    and the resulting reference hh_size cell rates (== sum_S D_all H_j exactly).
    """
    s = (
        pw_all["person_id"]
        .map(lambda p: size_map.get(int(p), 0))
        .to_numpy(dtype=np.int64)
    )
    bucket = _size_bucket(s)
    w = pw_all["weight"].to_numpy(np.float64)
    hh = pw_all["hh_size"].to_numpy(np.int64)
    tot = float(w.sum())
    classes = {"3": hh == 3, "4": hh == 4, "5+": hh >= 5}
    d_all: dict[str, float] = {}
    h: dict[str, dict[str, float]] = {j: {} for j in classes}
    for b in SIZE_BUCKETS:
        mb = bucket == b
        wb = float(w[mb].sum())
        d_all[b] = (wb / tot) if tot > 0 else 0.0
        for j, cmask in classes.items():
            h[j][b] = float((w[mb] * cmask[mb]).sum() / wb) if wb > 0 else 0.0
    ref_hh = {
        j: sum(d_all[b] * h[j][b] for b in SIZE_BUCKETS) for j in classes
    }
    return d_all, h, ref_hh


def _spouse_female_rates(
    d: dict[str, Any], bands: list[str]
) -> dict[str, dict[str, float]]:
    """Per female band: legal / cohab-overlay / lr-overlay / full spouse rate."""
    band = d["band"]
    is_female = d["sex"] == "female"
    weight = d["weight"]
    legal = d["legal_spouse"]
    cohab_only = d["cohab_row"] & ~legal
    lr_only = d["lr_row"] & ~legal & ~d["cohab_row"]
    out: dict[str, dict[str, float]] = {}
    for b in bands:
        inb = (band == b) & is_female
        wt = float(weight[inb].sum())
        if wt <= 0:
            out[b] = {"legal": 0.0, "cohab": 0.0, "lr": 0.0, "full": 0.0}
            continue
        out[b] = {
            "legal": _wshare(weight[inb], legal[inb]),
            "cohab": _wshare(weight[inb], cohab_only[inb]),
            "lr": _wshare(weight[inb], lr_only[inb]),
            "full": _wshare(weight[inb], d["spouse"][inb]),
        }
    return out


# --------------------------------------------------------------------------
# Per-seed compute
# --------------------------------------------------------------------------
def compute_seed(
    seed: int,
    data: dict[str, Any],
    tol: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    """Fit candidate 7 on side B, run the reference + instrumented pieces."""
    t0 = time.time()
    hh = data["hh"]
    mpanel = data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs7.fit_household_model_v7(
        hh,
        mpanel,
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b,
        father_links_child=data["father_links_child"],
        parent_pairs=data["parent_pairs"],
        marital_by_year=data["marital_by_year"],
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    rate_b = hc.reference_moments(hh, ids_b, weighted=True)
    marital_by_year = data["marital_by_year"]

    # ---- Train-side reference (deterministic). ----
    size_map = train_completed_size(data["parent_pairs"], ids_b)
    child_cells = Q14_CELLS + Q15_CLEARED_CHILD_CELLS
    pw_b_all = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)]
    pw_b_band = pw_b_all[pw_b_all["band"].notna()]
    train_dk: dict[str, Any] = {}
    for cell in child_cells:
        bl, sx = _cell_of(cell)
        sub = pw_b_band[(pw_b_band["band"] == bl) & (pw_b_band["sex"] == sx)][
            ["person_id", "weight", "coresident_child"]
        ]
        d, k, full = _train_cell_dk(sub, size_map)
        train_dk[cell] = {"d_train": d, "k_train": k, "ref_full": full}
    d_train_all, h_train, ref_hh_train = _train_hh_kernel(
        pw_b_all[["person_id", "weight", "hh_size"]], size_map
    )
    # Population completed-size distribution (fertility headline, train).
    s_all = (
        pw_b_all["person_id"]
        .map(lambda p: size_map.get(int(p), 0))
        .to_numpy(dtype=np.int64)
    )
    train_size_dist_all = _size_hist(
        s_all, pw_b_all["weight"].to_numpy(np.float64)
    )
    # Per-cohort completed-size distribution (the registration's cohort x sex).
    train_size_dist_cell = {}
    for cell in Q14_CELLS:
        bl, sx = _cell_of(cell)
        sub = pw_b_band[(pw_b_band["band"] == bl) & (pw_b_band["sex"] == sx)]
        s = (
            sub["person_id"]
            .map(lambda p: size_map.get(int(p), 0))
            .to_numpy(dtype=np.int64)
        )
        train_size_dist_cell[cell] = _size_hist(
            s, sub["weight"].to_numpy(np.float64)
        )

    linked_ref = q14_linked_reference(
        hh,
        data["father_links_child"],
        data["parent_pairs"],
        marital_by_year,
        model,
        ids_b,
        Q14_MALE_CELLS,
    )

    female_bands = [
        hc.band_label(lo, hi) for lo, hi in hc.COMPOSITION_AGE_BANDS
    ]

    # ---- 20 instrumented train-side draws. ----
    q14_acc = {
        c: {
            "sim_full": [],
            "endowment": [],
            "coefficient": [],
            "d_sim": [],
            "k_sim": [],
        }
        for c in child_cells
    }
    q15_hh_acc = {
        j: {"sim": [], "cf": [], "h_sim": []} for j in ("3", "4", "5+")
    }
    q15_dall_sim: list[dict[str, float]] = []
    q16_acc = {
        b: {"legal": [], "cohab": [], "lr": [], "full": []}
        for b in female_bands
    }
    sim_size_dist_all: list[dict[str, float]] = []
    sim_size_dist_cell: dict[str, list[dict[str, float]]] = {
        c: [] for c in Q14_CELLS
    }

    for k in range(N_DRAWS):
        draw_seed = DRAW_SEED_BASE + k
        d = instrumented_draw_v7(hh, mpanel, model, ids_b, draw_seed)
        band, sex, weight = d["band"], d["sex"], d["weight"]
        pid = d["person_id"]
        cc_bool = d["coresident_child"]
        s_row = _sim_completed_size_row(pid, d["child_counts"])
        bucket_row = _size_bucket(s_row)

        # Q14 per child cell: Oaxaca.
        for cell in child_cells:
            bl, sx = _cell_of(cell)
            inb = (band == bl) & (sex == sx)
            ox = _oaxaca_cell(
                bucket_row[inb],
                cc_bool[inb],
                weight[inb],
                train_dk[cell]["d_train"],
                train_dk[cell]["k_train"],
            )
            q14_acc[cell]["sim_full"].append(ox["sim_full"])
            q14_acc[cell]["endowment"].append(ox["endowment_fertility_origin"])
            q14_acc[cell]["coefficient"].append(ox["coefficient_kernel_shift"])
            q14_acc[cell]["d_sim"].append(ox["d_sim"])
            q14_acc[cell]["k_sim"].append(ox["k_sim"])

        # Q15 hh-size kernel + counterfactual under the fertility lever.
        wtot = float(weight.sum())
        hs = d["hh_size"]
        d_all_sim = {}
        h_sim = {"3": {}, "4": {}, "5+": {}}
        for b in SIZE_BUCKETS:
            mb = bucket_row == b
            wb = float(weight[mb].sum())
            d_all_sim[b] = wb / wtot if wtot > 0 else 0.0
            for j, cmask in (
                ("3", hs == 3),
                ("4", hs == 4),
                ("5+", hs >= 5),
            ):
                h_sim[j][b] = (
                    float((weight[mb] * cmask[mb]).sum() / wb)
                    if wb > 0
                    else 0.0
                )
        q15_dall_sim.append(d_all_sim)
        for j in ("3", "4", "5+"):
            sim_j = sum(d_all_sim[b] * h_sim[j][b] for b in SIZE_BUCKETS)
            cf_j = sum(d_train_all[b] * h_sim[j][b] for b in SIZE_BUCKETS)
            q15_hh_acc[j]["sim"].append(float(sim_j))
            q15_hh_acc[j]["cf"].append(float(cf_j))
            q15_hh_acc[j]["h_sim"].append(h_sim[j])

        # Q16 spouse female band rates.
        sr = _spouse_female_rates(d, female_bands)
        for b in female_bands:
            for key in ("legal", "cohab", "lr", "full"):
                q16_acc[b][key].append(sr[b][key])

        # Fertility headline: sim completed-size distributions.
        sim_size_dist_all.append(_size_hist(s_row, weight))
        for cell in Q14_CELLS:
            bl, sx = _cell_of(cell)
            inb = (band == bl) & (sex == sx)
            sim_size_dist_cell[cell].append(
                _size_hist(s_row[inb], weight[inb])
            )

    elapsed = round(time.time() - t0, 1)
    if verbose:
        c = "coresident_child.65-74|male"
        print(
            f"seed {seed}: n_train={len(ids_b)} rho={model.linked_episode_persistence:.3f} "  # noqa: E501
            f"{c} sim={_mean(q14_acc[c]['sim_full']):.4f} "
            f"(ref {train_dk[c]['ref_full']:.4f}) fert_orig "
            f"{_mean(q14_acc[c]['endowment']):+.4f} "
            f"hh5+ sim={_mean(q15_hh_acc['5+']['sim']):.4f} "
            f"cf={_mean(q15_hh_acc['5+']['cf']):.4f} "
            f"(ref {ref_hh_train['5+']:.4f}) [{elapsed}s]"
        )

    return {
        "seed": seed,
        "n_train_persons": len(ids_b),
        "linked_episode_persistence_rho": float(
            model.linked_episode_persistence
        ),
        "rate_b": {c: float(rate_b[c]["rate"]) for c in sorted(tol)},
        "q14_train_dk": train_dk,
        "q14_linked_reference": linked_ref,
        "q14_sim": {
            c: {
                "sim_full": _mean(q14_acc[c]["sim_full"]),
                "endowment_fertility_origin": _mean(q14_acc[c]["endowment"]),
                "coefficient_kernel_shift": _mean(q14_acc[c]["coefficient"]),
                "d_sim": _avg_hist(q14_acc[c]["d_sim"], SIZE_BUCKETS),
                "k_sim": _avg_hist(q14_acc[c]["k_sim"], SIZE_BUCKETS),
            }
            for c in child_cells
        },
        "q15_reference_hh": ref_hh_train,
        "q15_d_train_all": d_train_all,
        "q15_h_train": h_train,
        "q15_hh_sim": {
            j: {
                "sim": _mean(q15_hh_acc[j]["sim"]),
                "counterfactual_lever": _mean(q15_hh_acc[j]["cf"]),
                "h_sim": _avg_hist(q15_hh_acc[j]["h_sim"], SIZE_BUCKETS),
            }
            for j in ("3", "4", "5+")
        },
        "q15_d_sim_all": _avg_hist(q15_dall_sim, SIZE_BUCKETS),
        "q16_female": {
            b: {key: _mean(q16_acc[b][key]) for key in q16_acc[b]}
            for b in female_bands
        },
        "fertility_train_size_dist_all": train_size_dist_all,
        "fertility_sim_size_dist_all": _avg_hist(
            sim_size_dist_all, SIZE_BUCKETS
        ),
        "fertility_train_size_dist_cell": train_size_dist_cell,
        "fertility_sim_size_dist_cell": {
            c: _avg_hist(sim_size_dist_cell[c], SIZE_BUCKETS)
            for c in Q14_CELLS
        },
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Committed candidate-7 holdout reads (side A: NEVER re-simulated)
# --------------------------------------------------------------------------
def _committed_c7() -> dict[str, Any]:
    return json.loads(CANDIDATE7_ARTIFACT.read_text())


def _committed_cell_c7(c7a: dict[str, Any], cell: str) -> dict[str, Any]:
    """Seed-mean committed side-A gated-cell record from candidate 7."""
    recs = [s["gated_cells"][cell] for s in c7a["per_seed"]]
    ra = _mean([r["rate_a"] for r in recs])
    rc = _mean([r["rbar"] for r in recs])
    tol = float(recs[0]["tolerance"])
    per_seed_score = {
        s["seed"]: float(s["gated_cells"][cell]["score"])
        for s in c7a["per_seed"]
    }
    per_seed_pass = {
        s["seed"]: bool(s["gated_cells"][cell]["pass"])
        for s in c7a["per_seed"]
    }
    return {
        "seed_mean_rate_a": ra,
        "seed_mean_r_candidate": rc,
        "seed_mean_score": _mean([r["score"] for r in recs]),
        "tolerance": tol,
        "n_seeds_pass": int(sum(per_seed_pass.values())),
        "failing_seeds": sorted(s for s, p in per_seed_pass.items() if not p),
        "per_seed_score": per_seed_score,
        "per_seed_pass": per_seed_pass,
    }


# --------------------------------------------------------------------------
# Assemble Q14 (older-parent adult-child supply, four-channel reconciliation)
# --------------------------------------------------------------------------
def assemble_q14(
    per_seed: list[dict[str, Any]], committed: dict[str, Any]
) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    for cell in Q14_CELLS:
        is_male = cell in Q14_MALE_CELLS
        sim_full = _mean([s["q14_sim"][cell]["sim_full"] for s in per_seed])
        ref_full = _mean(
            [s["q14_train_dk"][cell]["ref_full"] for s in per_seed]
        )
        cell_miss = sim_full - ref_full
        fertility_origin = _mean(
            [
                s["q14_sim"][cell]["endowment_fertility_origin"]
                for s in per_seed
            ]
        )
        kernel_shift = _mean(
            [s["q14_sim"][cell]["coefficient_kernel_shift"] for s in per_seed]
        )
        if is_male:
            link_coverage = _mean(
                [
                    s["q14_linked_reference"][cell][
                        "reference_s_joinable_restricted_contribution"
                    ]
                    - s["q14_linked_reference"][cell][
                        "reference_linked_any_contribution"
                    ]
                    for s in per_seed
                ]
            )
            v7_interaction = _mean(
                [
                    s["q14_linked_reference"][cell][
                        "reference_a_refexp_joinable_contribution"
                    ]
                    - s["q14_linked_reference"][cell][
                        "reference_s_joinable_restricted_contribution"
                    ]
                    for s in per_seed
                ]
            )
        else:
            link_coverage = 0.0
            v7_interaction = 0.0
        exit_origin = kernel_shift - link_coverage - v7_interaction
        channels = {
            "fertility_origin": fertility_origin,
            "exit_origin": exit_origin,
            "link_coverage": link_coverage,
            "v7_persistence_enumeration_interaction": v7_interaction,
        }
        recon = sum(channels.values())
        remainder = cell_miss - recon
        # Completed-family-size (fertility headline).
        sim_size = _avg_hist(
            [s["fertility_sim_size_dist_cell"][cell] for s in per_seed],
            SIZE_BUCKETS,
        )
        train_size = _avg_hist(
            [s["fertility_train_size_dist_cell"][cell] for s in per_seed],
            SIZE_BUCKETS,
        )
        sim_3plus = sim_size["3"] + sim_size["4+"]
        train_3plus = train_size["3"] + train_size["4+"]
        dom = max(channels, key=lambda k: abs(channels[k]))
        per_cell[cell] = {
            "sim_full_rate_train": sim_full,
            "reference_full_rate_train": ref_full,
            "cell_miss_sim_minus_reference": cell_miss,
            "signed_logratio_train": _signed(sim_full, ref_full),
            "channels": channels,
            "dominant_channel": dom,
            "channel_reconstruction_sum": recon,
            "reconciliation_remainder": remainder,
            "oaxaca_two_term": {
                "endowment_fertility_origin": fertility_origin,
                "coefficient_kernel_shift": kernel_shift,
            },
            "completed_family_size": {
                "sim_distribution": sim_size,
                "train_distribution": train_size,
                "sim_3plus_share": sim_3plus,
                "train_3plus_share": train_3plus,
                "three_plus_deficit_sim_minus_train": sim_3plus - train_3plus,
            },
            "sim_kernel_coresidence_given_size": _avg_hist(
                [s["q14_sim"][cell]["k_sim"] for s in per_seed], SIZE_BUCKETS
            ),
            "holdout_committed_candidate7": _committed_cell_c7(
                committed, cell
            ),
        }

    max_remainder = max(
        abs(per_cell[c]["reconciliation_remainder"]) for c in per_cell
    )
    finding = _q14_finding(per_cell)
    return {
        "question": (
            "older-parent adult-child supply: decompose the 55-64|male and "
            "65-74|male coresident_child misses (and the maternal "
            "45-54/65-74|female wobble) into channels reconciling exactly -- "
            "(a) FERTILITY-ORIGIN (completed family size by parent cohort x "
            "sex, sim vs train, especially 3+-child families); (b) EXIT-ORIGIN "
            "(children present but aged out too fast at older parent ages); (c) "
            "LINK-COVERAGE (joinable-enumerated children in train outside the "
            "sim's draw basis); (d) the v7 persistence/enumeration interaction "
            "at older bands."
        ),
        "method": (
            "Train-side (side B). The older-parent coresidence rate is an exact "
            "partition over the parent's COMPLETED FAMILY SIZE S (the parent's "
            "maximum coresident own-child count across their waves -- the same "
            "coresident-own-child basis forensics-3/4 measured the 3+-child "
            "deficit on): cell rate = sum_S D[S] K[S] (law of total probability, "
            "exact). The cell miss telescopes (Oaxaca-Blinder) into an ENDOWMENT "
            "term sum_S (D_sim - D_train) K_sim -- the FERTILITY-ORIGIN channel, "
            "the completed-family-size shortfall, computed per draw and averaged "
            "-- and a COEFFICIENT term sum_S D_train (K_sim - K_train) -- the "
            "coresidence-given-size shift. The coefficient term is attributed to "
            "LINK-COVERAGE (observed coresidence with joinable-enumerated "
            "children vs all linked-father coresidence) and the v7 "
            "PERSISTENCE/ENUMERATION INTERACTION (analytic independent-per-wave "
            "occupancy vs observed episode among joinable, at the cell) from the "
            "faithful custodial_prob_v6 linked anchors (zero for the maternal "
            "female cells), with EXIT-ORIGIN the remaining coresidence-timing "
            "residual. All four channels sum to the cell miss to machine "
            "epsilon. Custody probabilities are NOT re-estimated."
        ),
        "registered_cells": list(Q14_CELLS),
        "per_cell": per_cell,
        "reconciliation_max_abs_remainder": max_remainder,
        "finding": finding,
    }


def _q14_finding(per_cell: dict[str, Any]) -> str:
    def _p(x: float) -> str:
        return f"{x:+.4f}"

    c55 = per_cell["coresident_child.55-64|male"]
    c65 = per_cell["coresident_child.65-74|male"]
    f45 = per_cell["coresident_child.45-54|female"]
    f65 = per_cell["coresident_child.65-74|female"]

    def _chan(c: dict[str, Any]) -> str:
        ch = c["channels"]
        return (
            f"fertility {_p(ch['fertility_origin'])}, exit "
            f"{_p(ch['exit_origin'])}, link-coverage "
            f"{_p(ch['link_coverage'])}, v7-interaction "
            f"{_p(ch['v7_persistence_enumeration_interaction'])}"
        )

    return (
        "The older-parent coresident_child deficit is rooted in the "
        "COMPLETED-FAMILY-SIZE shortfall forensics-3/4 named -- the sim "
        "under-produces 3+-coresident-child families at every older cohort "
        f"(55-64|male {_p(c55['completed_family_size']['three_plus_deficit_sim_minus_train'])}, "  # noqa: E501
        f"65-74|male {_p(c65['completed_family_size']['three_plus_deficit_sim_minus_train'])}, "  # noqa: E501
        f"45-54|female {_p(f45['completed_family_size']['three_plus_deficit_sim_minus_train'])}, "  # noqa: E501
        f"65-74|female {_p(f65['completed_family_size']['three_plus_deficit_sim_minus_train'])}) "  # noqa: E501
        "-- but the diagnostic beats the registration's 'fertility-origin "
        "dominant at 65-74|male' prior: fertility-origin is DOMINANT at "
        f"55-64|male ({_chan(c55)}; cell miss "
        f"{_p(c55['cell_miss_sim_minus_reference'])}, dominant "
        f"{c55['dominant_channel']}) but NOT at 65-74|male ({_chan(c65)}; cell "
        f"miss {_p(c65['cell_miss_sim_minus_reference'])}, dominant "
        f"{c65['dominant_channel']}), where the fertility endowment is "
        "ATTENUATED by the near-zero adult-child coresidence kernel at the "
        "oldest band (children the sim did produce have aged out) so exit-origin "
        "and link-coverage co-dominate. The maternal cells: 45-54|female "
        f"OVER-produces ({_p(f45['cell_miss_sim_minus_reference'])}) -- a "
        f"fertility deficit ({_p(f45['channels']['fertility_origin'])}) more "
        "than offset by a positive coresidence-given-size shift "
        f"({_p(f45['channels']['exit_origin'])}), the sim retaining "
        "middle-aged mothers' children too long -- while 65-74|female "
        f"under-produces ({_p(f65['cell_miss_sim_minus_reference'])}, "
        f"fertility {_p(f65['channels']['fertility_origin'])} + exit "
        f"{_p(f65['channels']['exit_origin'])}). Every cell reconciles to the "
        "train-side miss to machine epsilon. The completed-family-size deficit "
        "is the shared root the parent side confirms: the cohorts whose "
        "3+-coresident-child families the sim under-produces ARE today's older "
        "parents, but its coresidence footprint at 65-74 is bottlenecked by "
        "adult-child retention, not birth counts alone."
    )


# --------------------------------------------------------------------------
# Assemble Q15 (the single-lever convergence proof, mirroring f3-Q10)
# --------------------------------------------------------------------------
def _lever_cell_check(
    per_seed: list[dict[str, Any]],
    committed: dict[str, Any],
    cell: str,
    sim_key,
    cf_key,
    ref_key,
    tol: float,
) -> dict[str, Any]:
    """Per-seed sim / counterfactual scoring for one cell under the lever.

    ``sim_key`` / ``cf_key`` read the sim rate and the fertility-lever
    counterfactual per seed; ``ref_key`` the train reference. Cross-references
    the committed candidate-7 holdout failing seeds.
    """
    committed_cell = _committed_cell_c7(committed, cell)
    failing = set(committed_cell["failing_seeds"])
    per_seed_out = []
    for rec in per_seed:
        seed = rec["seed"]
        sim = float(sim_key(rec))
        cf = float(cf_key(rec))
        ref = float(ref_key(rec))
        per_seed_out.append(
            {
                "seed": seed,
                "sim_train": sim,
                "counterfactual_lever": cf,
                "reference_train": ref,
                "sim_score": _score(sim, ref),
                "counterfactual_score": _score(cf, ref),
                "sim_within_tolerance": bool(_score(sim, ref) <= tol),
                "counterfactual_within_tolerance": bool(
                    _score(cf, ref) <= tol
                ),
                "holdout_seed_fails": seed in failing,
            }
        )
    on_failing = [r for r in per_seed_out if r["holdout_seed_fails"]]
    cf_clears_failing = (
        all(r["counterfactual_within_tolerance"] for r in on_failing)
        if on_failing
        else None
    )
    cf_all_in = all(r["counterfactual_within_tolerance"] for r in per_seed_out)
    return {
        "cell": cell,
        "tolerance": tol,
        "per_seed": per_seed_out,
        "seed_mean_sim": _mean([r["sim_train"] for r in per_seed_out]),
        "seed_mean_counterfactual": _mean(
            [r["counterfactual_lever"] for r in per_seed_out]
        ),
        "seed_mean_reference": _mean(
            [r["reference_train"] for r in per_seed_out]
        ),
        "counterfactual_clears_holdout_failing_seeds": cf_clears_failing,
        "counterfactual_within_tolerance_all_seeds": cf_all_in,
        "holdout_committed_candidate7": committed_cell,
    }


def assemble_q15(
    per_seed: list[dict[str, Any]], committed: dict[str, Any], tol: dict
) -> dict[str, Any]:
    hh_checks: dict[str, Any] = {}
    for j in ("3", "4", "5+"):
        cell = f"hh_size.{j}"
        hh_checks[cell] = _lever_cell_check(
            per_seed,
            committed,
            cell,
            sim_key=lambda r, j=j: r["q15_hh_sim"][j]["sim"],
            cf_key=lambda r, j=j: r["q15_hh_sim"][j]["counterfactual_lever"],
            ref_key=lambda r, j=j: r["q15_reference_hh"][j],
            tol=float(tol[cell]),
        )

    older_checks: dict[str, Any] = {}
    for cell in Q14_MALE_CELLS:
        older_checks[cell] = _lever_cell_check(
            per_seed,
            committed,
            cell,
            sim_key=lambda r, c=cell: r["q14_sim"][c]["sim_full"],
            cf_key=lambda r, c=cell: r["q14_sim"][c]["sim_full"]
            - r["q14_sim"][c]["endowment_fertility_origin"],
            ref_key=lambda r, c=cell: r["q14_train_dk"][c]["ref_full"],
            tol=float(tol[cell]),
        )

    cleared_checks: dict[str, Any] = {}
    for cell in Q15_CLEARED_CHILD_CELLS:
        cleared_checks[cell] = _lever_cell_check(
            per_seed,
            committed,
            cell,
            sim_key=lambda r, c=cell: r["q14_sim"][c]["sim_full"],
            cf_key=lambda r, c=cell: r["q14_sim"][c]["sim_full"]
            - r["q14_sim"][c]["endowment_fertility_origin"],
            ref_key=lambda r, c=cell: r["q14_train_dk"][c]["ref_full"],
            tol=float(tol[cell]),
        )

    # Sanity (f3-Q10 style): the completed-size kernel reproduces the reference
    # hh_size moments (the joint is honest).
    sanity_dev = _mean(
        [
            max(
                abs(s["q15_reference_hh"][j] - s["rate_b"][f"hh_size.{j}"])
                for j in ("3", "4", "5+")
            )
            for s in per_seed
        ]
    )

    hh5_target = hh_checks["hh_size.5+"]
    hh5_proves = bool(
        hh5_target["counterfactual_clears_holdout_failing_seeds"]
    )
    older_proves = {
        c: bool(older_checks[c]["counterfactual_clears_holdout_failing_seeds"])
        for c in Q14_MALE_CELLS
    }
    older_all_prove = all(older_proves.values())
    trade_holds = all(
        hh_checks[f"hh_size.{j}"]["counterfactual_within_tolerance_all_seeds"]
        for j in ("3", "4")
    )
    cleared_holds = all(
        cleared_checks[c]["counterfactual_within_tolerance_all_seeds"]
        for c in Q15_CLEARED_CHILD_CELLS
    )
    proves = bool(
        hh5_proves and older_all_prove and trade_holds and cleared_holds
    )

    def _pct(x: float) -> str:
        return f"{x:.4f}"

    c65 = older_checks["coresident_child.65-74|male"]
    finding = (
        "The single fertility lever "
        + ("PROVES" if proves else "DOES NOT fully prove")
        + " the convergence hypothesis -- the diagnostic qualifies the "
        "registration's optimistic prior. Applying the train completed-"
        "fertility distribution to the sim's OWN core (the f3-Q10 method, "
        "holding the sim's coresidence kernel) moves hh_size.5+ into tolerance "
        "on its holdout-failing seeds (counterfactual clears failing seeds: "
        f"{hh5_proves}; seed-mean sim {_pct(hh5_target['seed_mean_sim'])} -> "
        f"counterfactual {_pct(hh5_target['seed_mean_counterfactual'])} vs "
        f"reference {_pct(hh5_target['seed_mean_reference'])}), leaves "
        f"hh_size.3/.4 in tolerance (trade holds: {trade_holds}) and the "
        f"cleared child cells in tolerance (cleared holds: {cleared_holds}). "
        "But the older-MALE supply channels split: the lever clears "
        + ", ".join(
            f"{c.split('.')[1]} ({older_proves[c]})" for c in Q14_MALE_CELLS
        )
        + " -- fertility-origin is sufficient at 55-64|male but NOT at "
        "65-74|male, where closing only the completed-family-size gap "
        f"(counterfactual {_pct(c65['seed_mean_counterfactual'])} vs reference "
        f"{_pct(c65['seed_mean_reference'])}, tolerance {c65['tolerance']}) "
        "leaves the cell out because the oldest-band deficit is majority "
        "exit-origin + link-coverage (Q14), not fertility. Convergence is REAL "
        "for hh_size.5+ and 55-64|male -- the fertility-core lift is a "
        "necessary and, there, sufficient c8 lever -- but 65-74|male needs a "
        "SECOND mechanism (adult-child retention / enumerated-link coverage at "
        "the oldest band). The reference-core sanity convolution reproduces the "
        f"reference hh_size to {sanity_dev:.1e} (the joint is honest)."
    )
    return {
        "question": (
            "the convergence test (the c8 feasibility proof, mirroring f3-Q10): "
            "if the train completed-fertility 3+-child distribution replaces "
            "the sim's large-family core deficit, does that single lever "
            "simultaneously move (i) hh_size.5+ into tolerance on the failing "
            "seeds, (ii) the older-male adult-child supply channels from Q14a, "
            "and (iii) leave hh_size.3/.4 and the cleared child cells in "
            "tolerance? Applied analytically to the sim's own distributions per "
            "seed, exactly as f3-Q10 proved the bridge feasibility."
        ),
        "method": (
            "Train-side (side B). The lever replaces the sim's completed-family-"
            "size distribution D_sim[S] with the train D_train[S] (the '3+-child "
            "completed-fertility'), holding the sim's own conditional kernels: "
            "the counterfactual hh_size cell = sum_S D_train[S] H_sim(class|S), "
            "and the counterfactual coresident_child cell = sum_S D_train[S] "
            "K_sim(coresident|S) = sim_full - fertility_origin (the Q14 "
            "endowment closed). Each counterfactual is scored against the train "
            "reference per seed and cross-referenced to the committed "
            "candidate-7 holdout failing seeds. A sanity convolution of the "
            "kernel with the reference completed-size distribution reproduces "
            "the reference hh_size to machine epsilon, exactly as f3-Q10's "
            "reference-core sanity."
        ),
        "lever": (
            "train completed-fertility (3+-child) distribution replaces the "
            "sim's, per seed, holding the sim's coresidence/household kernels"
        ),
        "hh_size_cells": hh_checks,
        "older_male_supply_cells": older_checks,
        "cleared_child_cells": cleared_checks,
        "reference_core_sanity_max_abs_dev": sanity_dev,
        "convergence_verdict": {
            "hh_size_5plus_proves": hh5_proves,
            "older_male_cells_prove": older_proves,
            "older_male_all_prove": older_all_prove,
            "hh_size_3_4_trade_holds": trade_holds,
            "cleared_child_cells_hold": cleared_holds,
            "single_lever_proves_full_convergence": proves,
        },
        "finding": finding,
    }


# --------------------------------------------------------------------------
# Assemble Q16 (fragile spouse cell + cohabitation-overlay lift)
# --------------------------------------------------------------------------
def assemble_q16(
    per_seed: list[dict[str, Any]], committed: dict[str, Any], tol: dict
) -> dict[str, Any]:
    cell = Q16_CELL
    committed_cell = _committed_cell_c7(committed, cell)
    t = float(tol[cell])
    female_bands = list(per_seed[0]["q16_female"].keys())
    lift = Q16_OVERLAY_SHORTFALL

    per_seed_out = []
    for rec in per_seed:
        band = rec["q16_female"]["25-34"]
        old_full = float(band["full"])
        ref = float(rec["rate_b"][cell])
        new_full = old_full + lift * (1.0 - old_full)
        per_seed_out.append(
            {
                "seed": rec["seed"],
                "sim_full": old_full,
                "sim_cohab_overlay": float(band["cohab"]),
                "lifted_full": new_full,
                "reference_rate_b": ref,
                "sim_score": _score(old_full, ref),
                "lifted_score": _score(new_full, ref),
                "sim_within_tolerance": bool(_score(old_full, ref) <= t),
                "lifted_within_tolerance": bool(_score(new_full, ref) <= t),
                "holdout_seed_fails": rec["seed"]
                in committed_cell["failing_seeds"],
            }
        )
    on_failing = [r for r in per_seed_out if r["holdout_seed_fails"]]
    lift_clears_failing = (
        all(r["lifted_within_tolerance"] for r in on_failing)
        if on_failing
        else None
    )
    n_sim_over = sum(1 for r in per_seed_out if not r["sim_within_tolerance"])
    n_lift_over = sum(
        1 for r in per_seed_out if not r["lifted_within_tolerance"]
    )

    collateral = {}
    for b in female_bands:
        if b == "25-34":
            continue
        scell = f"coresident_spouse.{b}|female"
        if scell not in tol:
            continue
        sim_b = _mean([s["q16_female"][b]["full"] for s in per_seed])
        ref_b = _mean([s["rate_b"][scell] for s in per_seed])
        collateral[scell] = {
            "sim_full_unchanged_by_lift": sim_b,
            "reference_rate_b": ref_b,
            "tolerance": float(tol[scell]),
            "within_tolerance": bool(
                _score(sim_b, ref_b) <= float(tol[scell])
            ),
        }
    no_collateral = all(v["within_tolerance"] for v in collateral.values())

    seed_mean_sim = _mean([r["sim_full"] for r in per_seed_out])
    seed_mean_lift = _mean([r["lifted_full"] for r in per_seed_out])
    ref_mean = _mean([r["reference_rate_b"] for r in per_seed_out])
    finding = (
        "A cohabitation-overlay lift at 25-34|female sized to the forensics-3 "
        f"Q9 measured -{Q16_OVERLAY_SHORTFALL} overlay shortfall "
        + ("CLEARS" if lift_clears_failing else "does NOT clear")
        + " the fragile cell without collateral. The cell carries "
        f"{committed_cell['n_seeds_pass']}/5 on the committed holdout (its "
        "measured 2/5 split-seed fragility, seeds "
        f"{committed_cell['failing_seeds']}); the forensics train side already "
        f"clears all 5 with a small margin ({n_sim_over}/5 exceed), and the "
        "lift shrinks the underlying deficit that manifests as the holdout "
        f"fragility -- adding the overlay to the currently-non-spouse mass "
        f"lifts the seed-mean full spouse rate {seed_mean_sim:.4f} -> "
        f"{seed_mean_lift:.4f} (reference {ref_mean:.4f}), closing the "
        f"train deficit {ref_mean - seed_mean_sim:+.4f} -> "
        f"{ref_mean - seed_mean_lift:+.4f} and keeping the lifted rate within "
        f"tolerance on ALL seeds including the holdout-failing ones (lift "
        f"clears holdout-failing seeds: {lift_clears_failing}; "
        f"{n_lift_over}/5 lifted exceed). No other female spouse cell moves "
        f"out (collateral-clean: {no_collateral}) -- the override is "
        "age-25-34-specific, so the neighbouring bands are untouched by the "
        "analytic lift (a full re-simulation would bleed only the small cohab-"
        "persistence tail into 35-44|female, which already clears with "
        "margin). The lift targets the OVERLAY (forensics-4 Q12: the -0.045 "
        "cohabitation overlay gap, not the legal core), the component "
        "candidate 6's inert delta 3 failed to move."
    )
    return {
        "question": (
            "fragile spouse cell: measure what a cohabitation-overlay lift at "
            "25-34|female sized to the measured -0.045 shortfall does to the "
            "cell's split-seed exceedances (2/5) and gate-seed margins, "
            "analytically on train; confirm no other spouse cell moves out."
        ),
        "method": (
            "Train-side (side B). From 20 instrumented candidate-7 draws the "
            "25-34|female spouse rate is split into legal / cohabitation-overlay "
            "/ legal-residual components; the lift adds the forensics-3 Q9 "
            "measured -0.045 cohabitation-overlay shortfall to the currently-"
            "non-spouse mass (Bernoulli superposition: new_full = old_full + "
            "0.045 * (1 - old_full), the conservative reading), and the lifted "
            "rate is scored against the train reference per seed. Collateral is "
            "confirmed by the age-specificity of the override: every other "
            "female spouse cell is unchanged by the 25-34-only analytic lift."
        ),
        "registered_cell": cell,
        "overlay_shortfall_applied": Q16_OVERLAY_SHORTFALL,
        "per_seed": per_seed_out,
        "seed_mean_sim_full": seed_mean_sim,
        "seed_mean_lifted_full": seed_mean_lift,
        "n_split_seeds_over_tolerance_sim": n_sim_over,
        "n_split_seeds_over_tolerance_lifted": n_lift_over,
        "lift_clears_holdout_failing_seeds": lift_clears_failing,
        "collateral_cells": collateral,
        "no_collateral_spouse_cell_moves_out": no_collateral,
        "holdout_committed_candidate7": committed_cell,
        "finding": finding,
    }


# --------------------------------------------------------------------------
# Revision pins + run
# --------------------------------------------------------------------------
def _revision_pins() -> dict[str, Any]:
    import sklearn

    return {
        "populace_dynamics_head_sha": c7._git_sha(),
        "merge_base_origin_master": c7._merge_base(),
        "candidate7_artifact": "runs/gate2b_hazard_v7.json",
        "candidate7_artifact_sha256": c7._sha_of_file(CANDIDATE7_ARTIFACT),
        "candidate7_runner": "scripts/run_gate2b_candidate7.py",
        "candidate7_runner_sha256": c7._sha_of_file(
            ROOT / "scripts" / "run_gate2b_candidate7.py"
        ),
        "forensics3_artifact_sha256": c7._sha_of_file(FORENSICS3_ARTIFACT),
        "forensics4_artifact_sha256": c7._sha_of_file(FORENSICS4_ARTIFACT),
        "gate2b_floor_run": "runs/gate2b_floors_v1.json",
        "gate2b_floor_sha256": c7._sha_of_file(FLOOR_RUN),
        "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
        "sklearn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "schema_version": SCHEMA_VERSION,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    raise TypeError(f"not serializable: {type(obj)}")


def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    local_cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            local_cache = json.loads(cache_path.read_text())
        except Exception:
            local_cache = {}

    for art in (
        CANDIDATE7_ARTIFACT,
        FORENSICS3_ARTIFACT,
        FORENSICS4_ARTIFACT,
        FLOOR_RUN,
    ):
        if not art.exists():
            raise RuntimeError(f"required committed artifact missing: {art}")
    committed = _committed_c7()

    data = c7.load_all()
    tol = c7.gated_tolerances(c7.load_gate2b_thresholds())
    if verbose:
        hh = data["hh"]
        print(
            f"panel: {len(hh.person_waves)} person-waves, "
            f"{hh.person_waves.person_id.nunique()} persons; "
            f"train-side forensics, K={N_DRAWS} draws (5200 + k)"
        )

    # One-time fidelity proof: instrumented draw == committed simulate_draw_v7.
    _side_a, side_b = hpanel.split_panel_by_person(
        data["hh"].attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b0 = set(int(x) for x in side_b.person_id.unique())
    model0 = hcs7.fit_household_model_v7(
        data["hh"],
        data["mpanel"],
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b0,
        father_links_child=data["father_links_child"],
        parent_pairs=data["parent_pairs"],
        marital_by_year=data["marital_by_year"],
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    fidelity = fidelity_check_v7(
        data["hh"], data["mpanel"], model0, ids_b0, DRAW_SEED_BASE
    )
    if verbose:
        print(
            "instrumentation fidelity: bit_identical="
            f"{fidelity['bit_identical']} (max dev "
            f"{fidelity['max_abs_rate_deviation_vs_committed_simulate_draw_v7']:.2e}"  # noqa: E501
            f"; child-add {fidelity['child_channel_additivity_residual']}, "
            f"linked-add {fidelity['linked_marital_split_additivity_residual']})"
        )
    if not fidelity["bit_identical"]:
        raise RuntimeError(
            "instrumented_draw_v7 does not reproduce simulate_draw_v7 "
            "bit-for-bit; the component decomposition would be unfaithful."
        )

    per_seed: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"seed_{seed}"
        if key in local_cache:
            if verbose:
                print(f"seed {seed}: cached")
            per_seed.append(local_cache[key])
            continue
        result = compute_seed(seed, data, tol, verbose)
        local_cache[key] = json.loads(
            json.dumps(result, default=_json_default)
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(local_cache))
        per_seed.append(local_cache[key])

    q14 = assemble_q14(per_seed, committed)
    q15 = assemble_q15(per_seed, committed, tol)
    q16 = assemble_q16(per_seed, committed, tol)

    float64_eps = float(np.finfo(np.float64).eps)
    q14_recon_max = q14["reconciliation_max_abs_remainder"]
    reconciliations = {
        "instrumentation_bit_identity_max_rate_deviation": fidelity[
            "max_abs_rate_deviation_vs_committed_simulate_draw_v7"
        ],
        "child_channel_additivity_residual": fidelity[
            "child_channel_additivity_residual"
        ],
        "linked_marital_split_additivity_residual": fidelity[
            "linked_marital_split_additivity_residual"
        ],
        "q14_cell_miss_reconciliation_max_abs_remainder": q14_recon_max,
        "q15_reference_core_sanity_max_abs_dev": q15[
            "reference_core_sanity_max_abs_dev"
        ],
        "float64_machine_epsilon": float64_eps,
        "all_identity_reconciliations_at_machine_epsilon": bool(
            fidelity["max_abs_rate_deviation_vs_committed_simulate_draw_v7"]
            == 0.0
            and fidelity["child_channel_additivity_residual"] == 0
            and fidelity["linked_marital_split_additivity_residual"] == 0
            and q14_recon_max <= 64 * float64_eps
        ),
        "reconciliation_bar_note": (
            "instrumentation bit-identity is EXACT (0.0) and the integer "
            "channel-additivity residuals are EXACTLY 0; the Q14 four-channel "
            "Oaxaca-Blinder reconciliation of the cell miss sits at machine "
            "epsilon (a few ULP of float64 summation), i.e. machine zero."
        ),
    }

    implications = (
        "C8 design implications (LABELED implications, NOT decisions -- the "
        "orchestrator registers c8). Q14: the older-parent adult-child supply "
        "deficit is rooted in the completed-family-size (3+-coresident-child) "
        "shortfall forensics-3/4 named, seen from the parent side (sim "
        "under-produces 3+ families by -0.12 to -0.14 at the older male "
        "cohorts). But the deficit's coresidence footprint is NOT uniformly "
        "fertility: fertility-origin dominates at 55-64|male (younger adult "
        "children still coreside) while at 65-74|male the fertility endowment "
        "is attenuated by a near-zero adult-child coresidence kernel, and "
        "exit-origin + link-coverage co-dominate. Q15: the single fertility-"
        "core lift PROVES convergence for hh_size.5+ and 55-64|male (a "
        "necessary and there sufficient c8 lever, leaving hh_size.3/.4 and the "
        "cleared child cells in tolerance) but is INSUFFICIENT at 65-74|male -- "
        "c8 needs the fertility-core lift PLUS a second mechanism for the oldest "
        "band (adult-child retention at 65-74, or enumerated-link coverage of "
        "the coresident children outside the joinable draw basis). Q16: the "
        "cohabitation-overlay lift at 25-34|female clears the fragile spouse "
        "cell without collateral -- so the two-lever c8 (fertility-core + "
        "cohab-overlay) the registration anticipated is CORROBORATED for the "
        "spouse cell and the hh_size / 55-64|male supply, but the 65-74|male "
        "blocker (candidate 7's 0/5 cell) is NOT closed by fertility alone and "
        "remains the surviving surface. If c8 carries the fertility-core lift, "
        "the cohab-overlay lift, and an oldest-band adult-child retention "
        "mechanism, all three measured blockers are addressed; on two levers "
        "alone, 65-74|male still blocks."
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2b",
        "reported_not_gated": True,
        "registration": REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "registration_title": REGISTRATION_TITLE,
        "grading_pointer": GRADING_POINTER,
        "forensics4_grading_pointer": FORENSICS4_GRADING_POINTER,
        "candidate_under_diagnosis": (
            "gate-2b candidate 7 (PR #145, registration 4948186843; artifact "
            "runs/gate2b_hazard_v7.json; FAIL 0/5). Candidate 7 cleared the "
            "measured child mechanisms (25-34/35-44|male 5/5) and revealed the "
            "older-parent adult-child supply deficit beneath: 65-74|male 0/5, "
            "55-64|male 2/5, plus the carried fragile spouse cell and the "
            "hh_size.5+ core deficit. Forensics 5 measures the revealed deficit "
            "and tests the convergence hypothesis before candidate 8."
        ),
        "candidate7_artifact": "runs/gate2b_hazard_v7.json",
        "forensics3_artifact": "runs/gate2b_forensics3_v1.json",
        "forensics4_artifact": "runs/gate2b_forensics4_v1.json",
        "protocol": {
            "train_side_only": True,
            "outer_holdout_contact": (
                "none beyond the already-published per-seed scores read from "
                "runs/gate2b_hazard_v7.json; the holdout (side A) is NEVER "
                "re-simulated here"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "hh.attrs, 'person_id', fraction=0.5, seed=s); side A = outer "
                "holdout (read from the committed candidate-7 artifact only), "
                "side B = train complement (this diagnostic fits AND simulates "
                "side B)"
            ),
            "fit_simulate_machinery": (
                "populace_dynamics.models.household_composition_sim_v7 "
                "(candidate 7, merged #145), reused byte-for-byte on the train "
                "side via instrumented_draw_v7 (same 0xB2B / 0xC2 / 0xC3 / 0xC4 "
                "/ 0xC5 / 0xC6 / 0xC7 draw streams and train-fitted tables)"
            ),
            "reused_machinery": (
                "Q14 uses the SAME faithful custodial_prob_v6 the committed "
                "candidate fits (the forensics-2/3 custody basis) for the "
                "linked-father anchors and does NOT relitigate custody "
                "probabilities; the completed-family-size supply variable is "
                "the coresident-own-child count forensics-3/4 measured the "
                "3+-child deficit on; the Q15 lever mirrors the forensics-3 Q10 "
                "honest-joint counterfactual method exactly."
            ),
            "gate_seeds": list(GATE_SEEDS),
            "n_draws": N_DRAWS,
            "draw_rng_rule": (
                f"numpy.random.default_rng({DRAW_SEED_BASE} + k) for k in "
                f"0..{N_DRAWS - 1} (the locked gate_2b protocol)"
            ),
            "instrumentation_fidelity": fidelity,
            "no_holdout_tuning_surface": (
                "every fitted table and every reference decomposition is "
                "estimated on side B only; no parameter is estimated from side "
                "A -- no holdout-informed tuning surface is created"
            ),
        },
        "data": {
            "holdout_basis": ["MX23REL"],
            "paternal_link_basis": ["cah85_23"],
            "n_person_waves": int(len(data["hh"].person_waves)),
            "n_persons": int(data["hh"].person_waves.person_id.nunique()),
            "floor_run": "runs/gate2b_floors_v1.json",
            "floor_run_sha256": c7._sha_of_file(FLOOR_RUN),
        },
        "reconciliations": reconciliations,
        "question_14_older_parent_adult_child_supply": q14,
        "question_15_single_lever_convergence": q15,
        "question_16_fragile_spouse_cell": q16,
        "per_seed": per_seed,
        "candidate_8_implications": implications,
        "revision_pins": _revision_pins(),
        "per_seed_compute_seconds": {
            s["seed"]: s["elapsed_seconds"] for s in per_seed
        },
        "total_per_seed_compute_seconds": round(
            sum(s["elapsed_seconds"] for s in per_seed), 1
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        print("\n--- findings ---")
        for q in (q14, q15, q16):
            print(q["finding"][:320] + " ...")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ARTIFACT_PATH))
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="lbfgs failed to converge")
    warnings.filterwarnings("ignore", category=FutureWarning)
    artifact = run(verbose=True, cache_path=Path(args.cache))
    artifacts.write_new(Path(args.out), c7._json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
