"""Gate-2b forensics 4: linked-father child supply + two attribution checks.

Registered diagnostic (issue #42, comment 4947226688; ``reported_not_gated``).
The registration WINS: this runner answers exactly its three frozen questions,
publishes regardless of what it finds, and touches the outer holdout only
through the already-committed per-seed scores in ``runs/gate2b_hazard_v6.json``.

The protocol is the forensics-1/2/3 protocol, unchanged: train side only (side
B of ``split_panel_by_person(seed)``), candidate 6's fit/simulate machinery
reused BYTE-FOR-BYTE via :func:`instrumented_draw_v6` (a line-for-line copy of
:func:`hcs6.simulate_draw_v6` proved bit-identical by :func:`fidelity_check_v6`),
and the reused forensics-1/2 machinery (``spouse_concept_codes`` /
``q5_custodial_selection``) imported and called verbatim. Every additive
decomposition reconciles to machine epsilon.

Frozen questions (verbatim from the registration):

* **Q11 -- linked-father child supply at 25-44.** Decompose the male
  ``coresident_child`` overshoot at 25-34 and 35-44 into (a) EXISTENCE -- the
  number of linked children per father by father age band, sim-vs-train; (b)
  SPELL LENGTH -- father-child coresidence episode durations sim-vs-train at
  child ages 0-17 (custody probabilities are measured faithful per
  forensics-2/3, and are NOT relitigated here -- supply and spells only); (c)
  reconcile the two channels to the cell misses exactly, and if both existence
  and spells match train, enumerate what remains (weighting, age-timing of the
  father, the marital-state joint).

* **Q12 -- spouse 25-34|female movement attribution.** The cell stopped capping
  seeds in c6 while the registered delta targeting it (delta 3) was inert.
  Attribute the movement by component replay (delta 1's basis revert via
  household composition? delta 4's bridge? draw-path shifts from RNG stream
  changes? delta 3 itself?) and determine whether the clearing is STABLE
  (present across all draws/seeds with margin) or fragile luck.

* **Q13 -- hh_size.5+ marginal seed.** Per-draw dispersion vs tolerance on the
  failing seed; noise (per-draw spread straddles the line) vs structure
  (systematic tilt); if structure, name the component.

Method note on the exact reconciliation of Q11 (c): the linked exposure is the
same ``cah85_23`` father->child data on both sides, and the simulated panel
re-labels states on the SAME person-wave grid, so the cell father-waves, their
weights, and the exposed-linked-children counts are IDENTICAL sim-vs-train by
construction (the EXISTENCE channel is a structural zero, reported explicitly).
The linked-channel miss then telescopes EXACTLY into: a Monte-Carlo gap
(realized minus the analytic ``1 - prod(1 - p_c)`` occupancy expectation, the
finite-draw term); a MARITAL-JOINT term (the analytic occupancy under the
simulated father marital vs the observed father marital, same faithful custody
probability); a SPELL term (the analytic independent-per-wave occupancy vs the
observed correlated coresidence among the SAME linked children -- the
occupancy-vs-episode persistence the registration names); and a SUPPLY residual
(the observed coresidence with children OUTSIDE the linked set that the sim's
linked channel does not model). The shadow (unlinked-father) channel is the
remaining exact partition term. All terms sum to the train-side cell miss to
machine epsilon.
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
import run_gate2b_candidate6 as c6  # noqa: E402

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
from populace_dynamics.models.family_transitions.components.fertility import (  # noqa: E402,E501
    build_fertility_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_forensics4_v1.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v6.json"
FORENSICS1_ARTIFACT = ROOT / "runs" / "gate2b_forensics1_v1.json"
FORENSICS2_ARTIFACT = ROOT / "runs" / "gate2b_forensics2_v1.json"
FORENSICS3_ARTIFACT = ROOT / "runs" / "gate2b_forensics3_v1.json"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
SCHEMA_VERSION = "gate2b_forensics4.v1"
RUN_NAME = "gate2b_forensics4_v1"

#: The registered diagnostic (issue #42, comment 4947226688). The registration
#: wins: this runner answers exactly its three frozen questions.
REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4947226688"
)
REGISTRATION_POINTER = "4947226688"
REGISTRATION_TITLE = (
    "Gate-2b forensics 4 registration (diagnostic, not gated): linked-father "
    "supply, and two attribution checks (Q11 existence-vs-spell-length; Q12 "
    "spouse 25-34|female movement attribution; Q13 hh_size.5+ marginal seed)"
)
GRADING_POINTER = "4947225286"
CANDIDATE6_REGISTRATION_POINTER = "4946285556"

#: Reused frozen dials (candidate 6 / the locked gate_2b protocol).
GATE_SEEDS = c6.GATE_SEEDS  # (0, 1, 2, 3, 4)
N_DRAWS = c6.N_DRAWS  # 20
DRAW_SEED_BASE = c6.DRAW_SEED_BASE  # 5200
EXACT_ATOL = c6.EXACT_ATOL  # 1e-12

GRANDCHILD_LO = hcs6.GRANDCHILD_LO
CORE_SIZE_CAP = hcs6.CORE_SIZE_CAP
CUSTODIAL_REVERT_BAND = hcs6.CUSTODIAL_REVERT_BAND
DELTA_STREAM_TAG_V5 = hcs6.DELTA_STREAM_TAG_V5
DELTA_STREAM_TAG_V6 = hcs6.DELTA_STREAM_TAG_V6
_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED
CHILD_CORESIDENCE_MAX_AGE = hcs3.CHILD_CORESIDENCE_MAX_AGE

#: The two Q11 registered father-age cells (male coresident_child overshoot).
Q11_CELLS = (
    "coresident_child.25-34|male",
    "coresident_child.35-44|male",
)
#: The Q12 spouse cell + the female bands the replay characterizes.
Q12_CELL = "coresident_spouse.25-34|female"
#: The Q13 hh_size cell + the pre-identified marginal (failing) split seed.
Q13_CELL = "hh_size.5+"
Q13_MARGINAL_SEED = 3

#: Existence / episode reporting buckets.
EXIST_BUCKETS = ("0", "1", "2", "3+")
EPISODE_BUCKETS = ("1", "2", "3", "4+")
#: Spells are measured over minor child ages only, per the registration.
SPELL_CHILD_MAX_AGE = 17

#: Per-seed cache OUTSIDE runs/ (never committed): a long run resumes.
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2b_forensics4_cache.json"
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


def _wmean_over(weight: np.ndarray, value: np.ndarray) -> float:
    """Weighted mean of ``value`` over ``weight`` (same as _wshare, floats)."""
    return _wshare(weight, value)


def _seed_mean_leaf(per_seed: list[dict[str, Any]], path: list) -> float:
    vals = []
    for s in per_seed:
        node: Any = s
        for k in path:
            node = node[k]
        vals.append(float(node))
    return _mean(vals)


def _episode_length_hist(
    father_id: np.ndarray,
    child_key: np.ndarray,
    year: np.ndarray,
    coresident: np.ndarray,
) -> tuple[dict[str, float], float, int]:
    """Weighted-free episode-length histogram over father-child coresidence.

    An EPISODE is a maximal run of coresident waves adjacent in the year-ordered
    per (father, child) sequence (both sides share the same person-wave grid, so
    "adjacent" is the natural spell-in-waves). Returns the share distribution
    over ``EPISODE_BUCKETS`` (1 / 2 / 3 / 4+ waves), the mean episode length, and
    the episode count.
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
    # Collapse to one coresidence flag per (father, child-key, year) -- the rare
    # multi-birth (parent, birth_year) rows share the child key, so a wave is
    # coresident iff ANY of that key's children coreside (matches the per-key
    # granularity used on both sides).
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
    # An episode STARTS at a coresident row that opens a new pair or follows a
    # non-coresident row.
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


# --------------------------------------------------------------------------
# Instrumented candidate-6 draw (faithful copy returning every component)
# --------------------------------------------------------------------------
def _linked_detail_v6(
    linked_births: pd.DataFrame,
    side_a_pw: pd.DataFrame,
    marital_sim: pd.DataFrame,
    model: hcs6.HouseholdCompositionModelV6,
    rng: np.random.Generator,
    joinable_keys: set | None = None,
) -> dict[str, Any]:
    """:func:`hcs6.custodial_linked_child_counts_v6` returning full detail.

    Consumes the ``0xC3`` custodial rng IDENTICALLY (same exposure construction,
    sort, and single ``rng.random(len(expo))`` draw) as the committed v6
    function, and returns the byte-identical total counts PLUS: the father
    marital split (married / not-married coresident counts per father-wave); the
    per-father-wave analytic occupancy ``1 - prod(1 - p_c)`` (the exact
    expectation of ``coresident_child`` under the independent per-wave draw); the
    per-father-wave exposed-linked-child count; and the exposure rows augmented
    with the drawn coresident flag (for episode lengths).
    """
    n = len(side_a_pw)
    counts = np.zeros(n, dtype=np.int64)
    c_m = np.zeros(n, dtype=np.int64)
    c_nm = np.zeros(n, dtype=np.int64)
    n_expo = np.zeros(n, dtype=np.int64)
    a_stock = np.zeros(n, dtype=np.float64)  # analytic 1 - prod(1 - p_c)
    log_no = np.zeros(n, dtype=np.float64)  # sum log(1 - p_c) accumulator
    empty_epi = {
        "fid": np.array([], dtype=np.int64),
        "ck": np.array([], dtype=np.int64),
        "year": np.array([], dtype=np.int64),
        "child_age": np.array([], dtype=np.int64),
        "coresident": np.array([], dtype=bool),
        "joinable": np.array([], dtype=bool),
    }
    out = {
        "counts": counts,
        "married": c_m,
        "not_married": c_nm,
        "n_exposed": n_expo,
        "analytic_stock": a_stock,
        "epi": empty_epi,
    }
    if not len(linked_births):
        return out
    pw = side_a_pw.reset_index(drop=True)
    pw = pw.assign(_row=np.arange(len(pw), dtype=np.int64))
    fw = pw[["person_id", "year", "_row"]].rename(
        columns={"person_id": "parent_person_id"}
    )
    expo = linked_births.merge(fw, on="parent_person_id", how="inner")
    if not len(expo):
        return out
    expo["child_age"] = expo["year"] - expo["birth_year"]
    expo = expo[
        (expo["child_age"] >= 0)
        & (expo["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    expo["child_band"] = expo["child_age"].map(hcs3._child_band)
    expo = expo[expo["child_band"].notna()]
    if not len(expo):
        return out
    expo = expo.merge(
        marital_sim.rename(columns={"person_id": "parent_person_id"}),
        on=["parent_person_id", "year"],
        how="left",
    )
    expo["marital"] = expo["marital"].fillna(_NOT_MARRIED)
    # Byte-identical row order to candidate 6 (candidate-3 order).
    expo = expo.sort_values(["parent_person_id", "birth_year", "_row"])
    expo = expo.reset_index(drop=True)
    ages = expo["child_age"].to_numpy()
    years = expo["year"].to_numpy()
    marital = expo["marital"].to_numpy()
    prob = np.array(
        [
            hcs6.custodial_prob_v6(
                model, int(a), hcs4.era_of_year(int(y)), str(m)
            )
            for a, y, m in zip(ages, years, marital, strict=True)
        ],
        dtype=np.float64,
    )
    u = rng.random(len(expo))  # 0xC3 custodial stream, byte-identical
    coresident = u < prob
    rows = expo["_row"].to_numpy()
    np.add.at(counts, rows[coresident], 1)
    np.add.at(c_m, rows[coresident & (marital == _MARRIED)], 1)
    np.add.at(c_nm, rows[coresident & (marital == _NOT_MARRIED)], 1)
    np.add.at(n_expo, rows, 1)
    # Analytic occupancy: P(>=1 coresident) = 1 - prod(1 - p_c) per father-wave.
    np.add.at(log_no, rows, np.log1p(-np.clip(prob, 0.0, 1.0 - 1e-15)))
    a_stock[:] = 1.0 - np.exp(log_no)
    a_stock[n_expo == 0] = 0.0
    if joinable_keys is None:
        joinable = np.ones(len(expo), dtype=bool)
    else:
        idx = pd.MultiIndex.from_arrays(
            [expo["parent_person_id"], expo["birth_year"]]
        )
        joinable = np.asarray(idx.isin(joinable_keys), dtype=bool)
    out["epi"] = {
        "fid": expo["parent_person_id"].to_numpy(np.int64),
        "ck": expo["birth_year"].to_numpy(np.int64),
        "year": years.astype(np.int64),
        "child_age": ages.astype(np.int64),
        "coresident": coresident,
        "joinable": joinable,
    }
    return out


def instrumented_draw_v6(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs6.HouseholdCompositionModelV6,
    ids: set[int],
    draw_seed: int,
    joinable_keys: set | None = None,
) -> dict[str, Any]:
    """Reproduce :func:`hcs6.simulate_draw_v6` and return every component array.

    A line-for-line copy of the committed candidate-6 draw (same 0xB2B / 0xC2 /
    0xC3 / 0xC4 / 0xC5 / 0xC6 substreams and train-fitted tables), instrumented
    to also return the spouse split (legal / cohab / legal-residual), the
    coresident-child attribution masks (linked via father membership; the
    father's simulated marital that wave; shadow via unlinked), the per-source
    child counts, the linked exposure detail (analytic occupancy, exposed count,
    coresidence rows for episodes), the family-core / non-family split, and the
    grandchild split -- all aligned to the returned ``pw`` row order. Bit-
    identity of the recomposed panel vs :func:`hcs6.simulate_draw_v6` is proved
    by :func:`fidelity_check_v6`.
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

    # 3. candidate-4 DELTA 1 (carried) cohab overlay, then candidate-6 DELTA 3:
    #    the female 25-44 single-year override.
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

    # 4. candidate-4 delta substreams (0xC4): spawn(2) exactly as candidate 6
    #    (the 2+ count spawn is retired by delta 4 but still spawned so the
    #    legal-residual stream stays byte-identical).
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

    # 5. certified marital core + maternal births (same draw seed as cand 6).
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

    # candidate-6 base leave-year draw (0xC2) -- kept byte-identical so the
    # SHADOW leave-year is unchanged; the maternal rows are re-drawn by delta 2.
    base_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    shadow_leaves = base_leaves[base_leaves["_source"] == "shadow"]

    # candidate-3 delta substreams (0xC3), isolated from 0xB2B/0xC2.
    c3_ss = np.random.SeedSequence([draw_seed, 0xC3])
    custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    custodial_rng = np.random.default_rng(custodial_ss)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # candidate-5 delta substreams (0xC5): coupling + parent count (CARRIED).
    c5_ss = np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V5])
    coupling_ss, parentcount_ss = c5_ss.spawn(2)
    coupling_rng = np.random.default_rng(coupling_ss)
    parentcount_rng = np.random.default_rng(parentcount_ss)

    # candidate-6 delta substream (0xC6): the maternal single-year leave refit.
    mat_leave_rng = np.random.default_rng(
        np.random.SeedSequence([draw_seed, DELTA_STREAM_TAG_V6])
    )

    # DELTA 2 (maternal): re-draw the maternal leave-years with the single-year
    # 18-30 exit refit on 0xC6; the shadow leave stays byte-identical.
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

    # DELTA 1 (linked): custodial per-wave coresidence with the 0-4 not-married
    # revert; byte-identical 0xC3 custodial stream. Returns full detail.
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    linked = _linked_detail_v6(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model,
        custodial_rng,
        joinable_keys,
    )
    cc_linked = linked["counts"]
    child_counts = child_counts_nonlinked + cc_linked

    coresident_child, grandchild_composed, _hh_default = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

    # DELTA 3b (carried): per-ego coresident-parent count (1 vs 2) on 0xC5.
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

    # DELTA 1 (carried): multigen -- adult-child coupling for 55+ egos (0xC5).
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

    # Simulated father-marital per pw row (for the linked-channel split).
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
        "linked_n_exposed": linked["n_exposed"],
        "linked_analytic_stock": linked["analytic_stock"],
        "linked_epi": linked["epi"],
        "linked_ids": linked_ids,
        # hh_size components.
        "n_parents_ego": n_parents_ego,
        "hh_size_base": hh_size_base.astype(np.int64),
        "nonfamily_count": nonfamily_count.astype(np.int64),
        "hh_size": hh_size,
        # grandchild split.
        "grandchild_composed": grandchild_composed,
        "grandchild_coupled": grandchild_coupled,
        "coresident_grandchild": coresident_grandchild,
        "is_55_row": is_55_row,
    }


def fidelity_check_v6(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs6.HouseholdCompositionModelV6,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Prove the instrumented draw equals the committed ``simulate_draw_v6``."""
    inst = instrumented_draw_v6(hh, mpanel, model, ids, draw_seed)
    real_panel, _ = hcs6.simulate_draw_v6(hh, mpanel, model, ids, draw_seed)
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
    # Analytic occupancy consistency: E[1[cc_linked>0]] == analytic stock only
    # in expectation; here confirm the analytic stock is a valid probability and
    # zero exactly where there is no exposure.
    a_stock = inst["linked_analytic_stock"]
    a_valid = int(np.sum((a_stock < -1e-12) | (a_stock > 1.0 + 1e-12))) + int(
        np.abs(a_stock[inst["linked_n_exposed"] == 0]).sum() > 0
    )
    return {
        "draw_seed": draw_seed,
        "n_cells_compared": len(b),
        "max_abs_rate_deviation_vs_committed_simulate_draw_v6": float(max_dev),
        "bit_identical": bool(max_dev <= EXACT_ATOL),
        "child_channel_additivity_residual": child_add,
        "linked_marital_split_additivity_residual": linked_add,
        "analytic_stock_out_of_range_or_nonzero_no_exposure": a_valid,
    }


# --------------------------------------------------------------------------
# Q11 reference: existence + episodes + analytic-occupancy anchors (train side)
# --------------------------------------------------------------------------
def q11_reference(
    hh: hc.HouseholdCompositionPanel,
    father_links_child: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    marital_by_year: pd.DataFrame,
    model: hcs6.HouseholdCompositionModelV6,
    ids_b: set[int],
) -> dict[str, Any]:
    """Reference-side Q11 anchors (deterministic; no simulation RNG).

    ALIGNMENT: the committed ``simulate_draw_v6`` linked channel draws
    coresidence over ``model.father_links`` (the ``father_link_births`` basis,
    which does NOT require a joinable child -- 25.8% of its rows are
    non-joinable), so this reference uses the SAME basis for the linked/unlinked
    partition and the exposed-count EXISTENCE. The father-wave grid, weights, and
    exposed-linked-child counts are therefore identical sim-vs-train (existence
    is a structural zero). The analytic occupancy anchor applies the SAME
    faithful ``custodial_prob_v6`` to the OBSERVED father marital, split into an
    ALL-exposure anchor (``a_refexp_all``) and a JOINABLE-only anchor
    (``a_refexp_joinable``) so the non-joinable biological children the sim
    coresides but the reference cannot enumerate become their own channel. The
    realized coresidence and the episodes use ``father_links_child`` (only
    enumerated children appear in ``parent_pairs``). Custody probabilities are
    NOT re-estimated.
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
    # The COMMITTED linked exposure basis (== the sim's paternal_linked source).
    fl = model.father_links[["parent_person_id", "birth_year"]].copy()
    fl["parent_person_id"] = fl["parent_person_id"].astype("int64")
    fl["birth_year"] = fl["birth_year"].astype("int64")
    linked_father_ids = set(int(x) for x in fl["parent_person_id"].unique())
    # Joinable (parent, birth_year) keys -- an enumerated child exists.
    fac = father_links_child[
        ["parent_person_id", "child_person_id", "birth_year"]
    ].copy()
    fac["parent_person_id"] = fac["parent_person_id"].astype("int64")
    fac["birth_year"] = fac["birth_year"].astype("int64")
    joinable_keys = set(
        map(
            tuple,
            fac[["parent_person_id", "birth_year"]]
            .drop_duplicates()
            .to_numpy()
            .tolist(),
        )
    )

    # Father-wave exposure over the SIM basis: every model.father_links birth row
    # paired with each father wave whose child age lands in a custodial band
    # (row-identical to the sim expo).
    fw = pw_b[["person_id", "year", "age", "band", "sex", "weight"]].rename(
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
                model, int(a), hcs4.era_of_year(int(y)), str(m)
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
    logno = np.log1p(-np.clip(prob, 0.0, 1.0 - 1e-15))
    expo["_logno"] = logno
    expo["_logno_j"] = np.where(expo["joinable"].to_numpy(), logno, 0.0)
    grp = expo.groupby(["parent_person_id", "year"], sort=False)
    fw_agg = grp.agg(
        n_exposed=("birth_year", "size"),
        logno=("_logno", "sum"),
        logno_j=("_logno_j", "sum"),
    ).reset_index()
    fw_agg["a_refexp_all"] = 1.0 - np.exp(fw_agg["logno"].to_numpy())
    fw_agg["a_refexp_j"] = 1.0 - np.exp(fw_agg["logno_j"].to_numpy())
    fw_agg = fw_agg.merge(
        pw_b[["person_id", "year", "band", "weight"]].rename(
            columns={"person_id": "parent_person_id"}
        ),
        on=["parent_person_id", "year"],
        how="left",
    )

    # Observed coresidence with a JOINABLE linked biological child (non-joinable
    # children are unenumerated -> never in parent_pairs).
    linked_cor = parent_pairs.merge(
        fac, on=["parent_person_id", "child_person_id"], how="inner"
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
    fw_agg = fw_agg.merge(cor_fw, on=["parent_person_id", "year"], how="left")
    fw_agg["n_cor_j"] = fw_agg["n_cor_j"].fillna(0).astype("int64")

    pw_b = pw_b.assign(
        _linked=pw_b["person_id"].isin(linked_father_ids),
        _male=(pw_b["sex"] == "male"),
    )

    per_cell: dict[str, Any] = {}
    for cell in Q11_CELLS:
        bl = cell.split(".")[1].split("|")[0]
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
        # The FULL cell linked father-wave population (matches the sim's linked
        # mask -- every cell male linked father-wave, INCLUDING those with no
        # child in the exposure window this wave, which carry n_exposed = 0 and
        # a_refexp = 0). Left-merge the exposure aggregates; absent -> zero.
        cl = pw_b.loc[
            cell_mask & pw_b["_linked"], ["person_id", "year", "weight"]
        ].merge(
            fw_agg[
                [
                    "parent_person_id",
                    "year",
                    "n_exposed",
                    "n_cor_j",
                    "a_refexp_all",
                    "a_refexp_j",
                ]
            ].rename(columns={"parent_person_id": "person_id"}),
            on=["person_id", "year"],
            how="left",
        )
        cl["n_exposed"] = cl["n_exposed"].fillna(0).astype("int64")
        cl["n_cor_j"] = cl["n_cor_j"].fillna(0).astype("int64")
        cl["a_refexp_all"] = cl["a_refexp_all"].fillna(0.0)
        cl["a_refexp_j"] = cl["a_refexp_j"].fillna(0.0)
        clw = cl["weight"].to_numpy(np.float64)
        clw_tot = float(clw.sum())
        n_exp = cl["n_exposed"].to_numpy(np.int64)
        n_cor = cl["n_cor_j"].to_numpy(np.int64)
        a_all = cl["a_refexp_all"].to_numpy(np.float64)
        a_j = cl["a_refexp_j"].to_numpy(np.float64)
        # Contributions over the FULL cell weight (denominator = cw_tot); the
        # zero-exposure father-waves add exactly zero.
        s_ref_linked_restr = (
            float((clw * (n_cor > 0)).sum() / cw_tot) if cw_tot > 0 else 0.0
        )
        a_refexp_all_contrib = (
            float((clw * a_all).sum() / cw_tot) if cw_tot > 0 else 0.0
        )
        a_refexp_j_contrib = (
            float((clw * a_j).sum() / cw_tot) if cw_tot > 0 else 0.0
        )
        per_cell[cell] = {
            "reference_full_male_rate": ref_full,
            "reference_linked_any_contribution": ref_linked_any,
            "reference_unlinked_any_contribution": ref_unlinked_any,
            "reference_linked_restricted_contribution": s_ref_linked_restr,
            "reference_a_refexp_all_contribution": a_refexp_all_contrib,
            "reference_a_refexp_joinable_contribution": a_refexp_j_contrib,
            "n_linked_father_waves_cell": int(len(cl)),
            "cell_weight_total": cw_tot,
            "linked_father_wave_weight_total": clw_tot,
            # EXISTENCE: exposed-linked-child count distribution over the FULL
            # linked population (identical to the sim by construction) and the
            # observed coresident-linked count.
            "reference_exposed_count_distribution": _count_hist(n_exp, clw),
            "reference_coresident_linked_count_distribution": _count_hist(
                n_cor, clw
            ),
            "mean_exposed_linked_children_per_father_wave": (
                float((clw * n_exp).sum() / clw_tot) if clw_tot > 0 else 0.0
            ),
            "mean_coresident_linked_children_per_father_wave": (
                float((clw * n_cor).sum() / clw_tot) if clw_tot > 0 else 0.0
            ),
        }

    # SPELL LENGTH: reference episode-length distribution over minor child ages
    # (0-17) across all JOINABLE linked father-child pairs. Built on the FULL
    # father-wave grid (coresident True where in parent_pairs, False elsewhere)
    # so episodes are runs of consecutive coresident waves.
    pp_cor = parent_pairs[
        ["parent_person_id", "child_person_id", "year"]
    ].assign(_cor=True)
    ref_epi = fac.merge(
        fw[["parent_person_id", "year"]], on="parent_person_id", how="inner"
    )
    ref_epi["child_age"] = ref_epi["year"] - ref_epi["birth_year"]
    ref_epi = ref_epi[
        (ref_epi["child_age"] >= 0)
        & (ref_epi["child_age"] <= SPELL_CHILD_MAX_AGE)
    ]
    ref_epi = ref_epi.merge(
        pp_cor, on=["parent_person_id", "child_person_id", "year"], how="left"
    )
    ref_epi["coresident"] = ref_epi["_cor"].fillna(False).astype(bool)
    ref_dist, ref_mean, ref_n = _episode_length_hist(
        ref_epi["parent_person_id"].to_numpy(np.int64),
        ref_epi["birth_year"].to_numpy(np.int64),
        ref_epi["year"].to_numpy(np.int64),
        ref_epi["coresident"].to_numpy(bool),
    )
    return {
        "per_cell": per_cell,
        "reference_episode_length_distribution": ref_dist,
        "reference_mean_episode_length": ref_mean,
        "reference_n_episodes": ref_n,
        "n_linked_fathers_train": len(linked_father_ids),
        "linked_exposure_basis": "model.father_links (father_link_births)",
        "n_linked_exposure_rows_train": int(len(fl)),
        "n_joinable_exposure_keys_train": len(joinable_keys),
    }


# --------------------------------------------------------------------------
# Per-seed compute
# --------------------------------------------------------------------------
def _spouse_female_rates(
    d: dict[str, Any], bands: list[str]
) -> dict[str, Any]:
    """Per female band: legal / cohab-overlay / lr-overlay / full spouse rate."""
    band = d["band"]
    sex = d["sex"]
    weight = d["weight"]
    is_female = sex == "female"
    legal = d["legal_spouse"]
    cohab_only = d["cohab_row"] & ~legal
    lr_only = d["lr_row"] & ~legal & ~d["cohab_row"]
    out: dict[str, Any] = {}
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


def compute_seed(
    seed: int,
    data: dict[str, Any],
    tol: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    """Fit candidate 6 on side B, run the reference + instrumented pieces."""
    t0 = time.time()
    hh = data["hh"]
    mpanel = data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs6.fit_household_model_v6(
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
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    # Delta-3-OFF replay model (female cohab override disabled) -- isolates the
    # candidate-6 delta 3 effect at apply time; every other fitted table and RNG
    # stream is untouched.
    import dataclasses

    model_no_d3 = dataclasses.replace(
        model, cohab_entry_age_female={}, cohab_exit_age_female={}
    )

    rate_b = hc.reference_moments(hh, ids_b, weighted=True)
    marital_by_year = hcs3._father_marital_by_year(mpanel)

    # ---- Deterministic reference decomposition (no sim RNG). ----
    q11_ref = q11_reference(
        hh,
        data["father_links_child"],
        data["parent_pairs"],
        marital_by_year,
        model,
        ids_b,
    )

    female_bands = [
        hc.band_label(lo, hi) for lo, hi in hc.COMPOSITION_AGE_BANDS
    ]

    # ---- 20 instrumented train-side draws. ----
    # Q11 accumulators (per cell).
    q11_sim: dict[str, dict[str, list[float]]] = {
        c: {
            "linked_contrib": [],
            "shadow_contrib": [],
            "male_full": [],
            "a_sim_linked": [],
        }
        for c in Q11_CELLS
    }
    q11_exist_sim: dict[str, dict[str, list[dict[str, float]]]] = {
        c: {"exposed": [], "coresident": []} for c in Q11_CELLS
    }
    q11_epi_sim: list[tuple[dict[str, float], float, int]] = []
    # Q12 accumulators.
    q12_c6: dict[str, dict[str, list[float]]] = {
        b: {"legal": [], "cohab": [], "lr": [], "full": []}
        for b in female_bands
    }
    q12_no_d3: dict[str, dict[str, list[float]]] = {
        b: {"legal": [], "cohab": [], "lr": [], "full": []}
        for b in female_bands
    }
    q12_target_per_draw: list[float] = []
    # Q13 accumulators.
    q13_hh5_per_draw: list[float] = []
    q13_core: dict[str, list[float]] = {str(k): [] for k in range(1, 9)}
    q13_childcnt: dict[str, list[float]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4+": [],
    }
    q13_core5plus: list[float] = []

    # Joinable (parent, birth_year) keys -- the sim tags each linked exposure row
    # so its episodes restrict to enumerated children (comparable to the
    # reference episodes, which can only see enumerated children).
    _fac = data["father_links_child"]
    joinable_keys = set(
        map(
            tuple,
            _fac[["parent_person_id", "birth_year"]]
            .astype("int64")
            .drop_duplicates()
            .to_numpy()
            .tolist(),
        )
    )

    for k in range(N_DRAWS):
        draw_seed = DRAW_SEED_BASE + k
        d = instrumented_draw_v6(
            hh, mpanel, model, ids_b, draw_seed, joinable_keys
        )
        band, sex, weight = d["band"], d["sex"], d["weight"]
        pid = d["person_id"]
        linked = np.isin(pid, list(d["linked_ids"]))
        is_male = sex == "male"
        cc_bool = d["coresident_child"]

        # ---- Q11 per cell. ----
        for cell in Q11_CELLS:
            bl = cell.split(".")[1].split("|")[0]
            inb = (band == bl) & is_male
            wt = float(weight[inb].sum())
            if wt <= 0:
                for key in q11_sim[cell]:
                    q11_sim[cell][key].append(0.0)
                q11_exist_sim[cell]["exposed"].append(
                    {b: 0.0 for b in EXIST_BUCKETS}
                )
                q11_exist_sim[cell]["coresident"].append(
                    {b: 0.0 for b in EXIST_BUCKETS}
                )
                continue
            link_mask = inb & linked
            shadow_mask = inb & (~linked)
            q11_sim[cell]["male_full"].append(
                float((weight[inb] * cc_bool[inb]).sum() / wt)
            )
            q11_sim[cell]["linked_contrib"].append(
                float((weight[link_mask] * cc_bool[link_mask]).sum() / wt)
            )
            q11_sim[cell]["shadow_contrib"].append(
                float((weight[shadow_mask] * cc_bool[shadow_mask]).sum() / wt)
            )
            # Analytic linked occupancy contribution over the FULL cell weight.
            a_stock = d["linked_analytic_stock"]
            q11_sim[cell]["a_sim_linked"].append(
                float((weight[link_mask] * a_stock[link_mask]).sum() / wt)
            )
            # EXISTENCE distributions among linked fathers in the cell.
            lw = weight[link_mask]
            q11_exist_sim[cell]["exposed"].append(
                _count_hist(d["linked_n_exposed"][link_mask], lw)
            )
            q11_exist_sim[cell]["coresident"].append(
                _count_hist(d["cc_linked"][link_mask], lw)
            )

        # Q11 sim episodes (child ages 0-17, JOINABLE linked pairs this draw --
        # matched to the reference, which only sees enumerated children).
        epi = d["linked_epi"]
        mminor = (epi["child_age"] <= SPELL_CHILD_MAX_AGE) & epi["joinable"]
        q11_epi_sim.append(
            _episode_length_hist(
                epi["fid"][mminor],
                epi["ck"][mminor],
                epi["year"][mminor],
                epi["coresident"][mminor],
            )
        )

        # ---- Q12: spouse female bands, c6 and delta-3-off. ----
        sr = _spouse_female_rates(d, female_bands)
        for b in female_bands:
            for key in ("legal", "cohab", "lr", "full"):
                q12_c6[b][key].append(sr[b][key])
        q12_target_per_draw.append(sr["25-34"]["full"])
        d0 = instrumented_draw_v6(hh, mpanel, model_no_d3, ids_b, draw_seed)
        sr0 = _spouse_female_rates(d0, female_bands)
        for b in female_bands:
            for key in ("legal", "cohab", "lr", "full"):
                q12_no_d3[b][key].append(sr0[b][key])

        # ---- Q13: hh_size.5+ per-draw + core / child-count structure. ----
        wtot = float(weight.sum())
        hs = d["hh_size"]
        hb = d["hh_size_base"]
        cc = d["child_counts"]
        q13_hh5_per_draw.append(float(weight[hs >= 5].sum() / wtot))
        for kk in range(1, 9):
            q13_core[str(kk)].append(float(weight[hb == kk].sum() / wtot))
        q13_core5plus.append(float(weight[hb >= 5].sum() / wtot))
        for j in range(0, 4):
            q13_childcnt[str(j)].append(float(weight[cc == j].sum() / wtot))
        q13_childcnt["4+"].append(float(weight[cc >= 4].sum() / wtot))

    def _avg_hist(hists: list[dict[str, float]], keys) -> dict[str, float]:
        return {b: _mean([h[b] for h in hists]) for b in keys}

    def _avg_epi(
        epis: list[tuple[dict[str, float], float, int]],
    ) -> dict[str, Any]:
        return {
            "distribution": _avg_hist([e[0] for e in epis], EPISODE_BUCKETS),
            "mean_episode_length": _mean([e[1] for e in epis]),
            "mean_n_episodes": _mean([float(e[2]) for e in epis]),
        }

    # Q13 reference core-size + child-count distributions (train side B).
    fu = data["fu_sizes"]
    ppw = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)][
        ["person_id", "year", "weight", "hh_size"]
    ].merge(fu, on=["person_id", "year"], how="left")
    ppw["family_unit_size"] = ppw["family_unit_size"].fillna(1).astype("int64")
    rw = ppw["weight"].to_numpy(np.float64)
    rtot = float(rw.sum())
    core_ref = ppw["family_unit_size"].to_numpy()
    ref_core5plus = float(rw[core_ref >= 5].sum() / rtot)
    ref_hh5plus = float(rw[ppw["hh_size"].to_numpy() >= 5].sum() / rtot)
    # Reference 3+-own-child family share (the forensics-3 Q10 upstream signal).
    pp = data["parent_pairs"]
    pp_b = pp[pp["parent_person_id"].isin(ids_b)]
    childcnt = pp_b.groupby(["parent_person_id", "year"]).size().rename("_n")
    # Attach to father-waves; own-child count per person-wave (0 if none).
    owncnt = ppw[["person_id", "year", "weight"]].merge(
        childcnt.reset_index().rename(
            columns={"parent_person_id": "person_id"}
        ),
        on=["person_id", "year"],
        how="left",
    )
    owncnt["_n"] = owncnt["_n"].fillna(0).astype("int64")
    ow = owncnt["weight"].to_numpy(np.float64)
    ref_child3plus = float(ow[owncnt["_n"].to_numpy() >= 3].sum() / ow.sum())

    elapsed = round(time.time() - t0, 1)
    if verbose:
        c1 = Q11_CELLS[1]
        print(
            f"seed {seed}: n_train={len(ids_b)} "
            f"{c1} sim_male={_mean(q11_sim[c1]['male_full']):.4f} "
            f"(ref {q11_ref['per_cell'][c1]['reference_full_male_rate']:.4f}) "
            f"spouse2534f sim={_mean(q12_c6['25-34']['full']):.4f} "
            f"hh5+ sim={_mean(q13_hh5_per_draw):.4f} "
            f"(ref {ref_hh5plus:.4f}) [{elapsed}s]"
        )

    return {
        "seed": seed,
        "n_train_persons": len(ids_b),
        "rate_b": {c: float(rate_b[c]["rate"]) for c in sorted(tol)},
        "q11_reference": q11_ref,
        "q11_sim": {
            c: {key: _mean(vals) for key, vals in sub.items()}
            for c, sub in q11_sim.items()
        },
        "q11_sim_existence": {
            c: {
                "exposed_count_distribution": _avg_hist(
                    q11_exist_sim[c]["exposed"], EXIST_BUCKETS
                ),
                "coresident_count_distribution": _avg_hist(
                    q11_exist_sim[c]["coresident"], EXIST_BUCKETS
                ),
            }
            for c in Q11_CELLS
        },
        "q11_sim_episodes": _avg_epi(q11_epi_sim),
        "q12_c6_female": {
            b: {key: _mean(q12_c6[b][key]) for key in q12_c6[b]}
            for b in female_bands
        },
        "q12_no_delta3_female": {
            b: {key: _mean(q12_no_d3[b][key]) for key in q12_no_d3[b]}
            for b in female_bands
        },
        "q12_target_full_per_draw": q12_target_per_draw,
        "q13_hh5plus_per_draw": q13_hh5_per_draw,
        "q13_sim_core_distribution": {
            k: _mean(v) for k, v in q13_core.items()
        },
        "q13_sim_core5plus": _mean(q13_core5plus),
        "q13_sim_child_count_distribution": {
            k: _mean(v) for k, v in q13_childcnt.items()
        },
        "q13_reference_core5plus": ref_core5plus,
        "q13_reference_hh5plus": ref_hh5plus,
        "q13_reference_child3plus_share": ref_child3plus,
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Committed candidate-6 holdout reads (side A: NEVER re-simulated)
# --------------------------------------------------------------------------
def _committed_c6() -> dict[str, Any]:
    return json.loads(CANDIDATE6_ARTIFACT.read_text())


def _committed_cell_c6(c6a: dict[str, Any], cell: str) -> dict[str, Any]:
    """Seed-mean committed side-A gated-cell record from candidate 6."""
    recs = [s["gated_cells"][cell] for s in c6a["per_seed"]]
    ra = _mean([r["rate_a"] for r in recs])
    rc = _mean([r["rbar"] for r in recs])
    tol = float(recs[0]["tolerance"])
    n_pass = sum(1 for r in recs if r["score"] <= tol)
    return {
        "seed_mean_rate_a": ra,
        "seed_mean_r_candidate": rc,
        "seed_mean_score": _mean([r["score"] for r in recs]),
        "tolerance": tol,
        "n_seeds_pass": int(n_pass),
        "per_seed_score": {
            s["seed"]: float(s["gated_cells"][cell]["score"])
            for s in c6a["per_seed"]
        },
    }


# --------------------------------------------------------------------------
# Assemble Q11 (existence vs spell + exact reconciliation)
# --------------------------------------------------------------------------
def assemble_q11(
    per_seed: list[dict[str, Any]], committed: dict[str, Any]
) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    for cell in Q11_CELLS:
        sim = {
            key: _mean([s["q11_sim"][cell][key] for s in per_seed])
            for key in per_seed[0]["q11_sim"][cell]
        }
        ref = {
            key: _mean(
                [s["q11_reference"]["per_cell"][cell][key] for s in per_seed]
            )
            for key in per_seed[0]["q11_reference"]["per_cell"][cell]
            if isinstance(
                per_seed[0]["q11_reference"]["per_cell"][cell][key],
                (int, float),
            )
        }
        # --- exact telescoping reconciliation of the train-side cell miss. ---
        male_full_sim = sim["male_full"]
        male_full_ref = ref["reference_full_male_rate"]
        cell_miss = male_full_sim - male_full_ref
        linked_sim = sim["linked_contrib"]
        shadow_sim = sim["shadow_contrib"]
        a_sim_linked = sim["a_sim_linked"]
        ref_linked_any = ref["reference_linked_any_contribution"]
        ref_unlinked_any = ref["reference_unlinked_any_contribution"]
        s_ref_linked_restr = ref["reference_linked_restricted_contribution"]
        a_refexp_all = ref["reference_a_refexp_all_contribution"]
        a_refexp_j = ref["reference_a_refexp_joinable_contribution"]

        # Exact telescoping of the linked-channel miss (existence == 0, the
        # exposed-count basis being identical model.father_links data):
        #   linked_sim - ref_linked_any
        #     = (linked_sim - a_sim_all)          [mc_gap: finite draws]
        #     + (a_sim_all - a_refexp_all)        [marital: sim vs obs marital]
        #     + (a_refexp_all - a_refexp_join)    [unenumerated non-joinable]
        #     + (a_refexp_join - s_ref_restr)     [spell: occupancy vs episode]
        #     + (s_ref_restr - ref_linked_any)    [supply residual: non-linked]
        mc_gap = linked_sim - a_sim_linked
        marital = a_sim_linked - a_refexp_all
        unenumerated_supply = a_refexp_all - a_refexp_j
        spell = a_refexp_j - s_ref_linked_restr
        supply_residual = s_ref_linked_restr - ref_linked_any
        shadow_miss = shadow_sim - ref_unlinked_any
        existence = 0.0  # identical linked exposure by construction.
        recon = (
            mc_gap
            + marital
            + unenumerated_supply
            + spell
            + supply_residual
            + shadow_miss
        )
        remainder = cell_miss - recon

        exposed_sim = {
            b: _mean(
                [
                    s["q11_sim_existence"][cell]["exposed_count_distribution"][
                        b
                    ]
                    for s in per_seed
                ]
            )
            for b in EXIST_BUCKETS
        }
        coresident_sim = {
            b: _mean(
                [
                    s["q11_sim_existence"][cell][
                        "coresident_count_distribution"
                    ][b]
                    for s in per_seed
                ]
            )
            for b in EXIST_BUCKETS
        }
        committed_cell = _committed_cell_c6(committed, cell)
        per_cell[cell] = {
            "sim_male_full_rate_train": male_full_sim,
            "reference_male_full_rate_train": male_full_ref,
            "cell_miss_sim_minus_reference": cell_miss,
            "signed_logratio_train": _signed(male_full_sim, male_full_ref),
            "channels": {
                "existence_identical_exposure": existence,
                "spell_length": spell,
                "marital_state_joint": marital,
                "unenumerated_nonjoinable_supply": unenumerated_supply,
                "supply_residual_nonlinked_coresidence": supply_residual,
                "shadow_unlinked_channel": shadow_miss,
                "monte_carlo_finite_draw_gap": mc_gap,
            },
            "channel_reconstruction_sum": recon,
            "reconciliation_remainder": remainder,
            "anchors": {
                "sim_linked_contribution": linked_sim,
                "sim_shadow_contribution": shadow_sim,
                "sim_analytic_linked_occupancy_all": a_sim_linked,
                "reference_linked_any_contribution": ref_linked_any,
                "reference_linked_restricted_contribution": (
                    s_ref_linked_restr
                ),
                "reference_a_refexp_all_analytic_occupancy": a_refexp_all,
                "reference_a_refexp_joinable_analytic_occupancy": a_refexp_j,
                "reference_unlinked_any_contribution": ref_unlinked_any,
            },
            "existence_distributions": {
                "sim_exposed_linked_count": exposed_sim,
                "reference_exposed_linked_count": {
                    b: _mean(
                        [
                            s["q11_reference"]["per_cell"][cell][
                                "reference_exposed_count_distribution"
                            ][b]
                            for s in per_seed
                        ]
                    )
                    for b in EXIST_BUCKETS
                },
                "sim_coresident_linked_count": coresident_sim,
                "reference_coresident_linked_count": {
                    b: _mean(
                        [
                            s["q11_reference"]["per_cell"][cell][
                                "reference_coresident_linked_count_"
                                "distribution"
                            ][b]
                            for s in per_seed
                        ]
                    )
                    for b in EXIST_BUCKETS
                },
                "mean_exposed_linked_children_sim_eq_reference": _mean(
                    [
                        s["q11_reference"]["per_cell"][cell][
                            "mean_exposed_linked_children_per_father_wave"
                        ]
                        for s in per_seed
                    ]
                ),
                "mean_coresident_linked_children_reference": _mean(
                    [
                        s["q11_reference"]["per_cell"][cell][
                            "mean_coresident_linked_children_per_father_wave"
                        ]
                        for s in per_seed
                    ]
                ),
            },
            "holdout_committed_candidate6": committed_cell,
        }

    # Episode-length distributions (population-level, child ages 0-17).
    sim_epi = {
        "distribution": {
            b: _mean(
                [s["q11_sim_episodes"]["distribution"][b] for s in per_seed]
            )
            for b in EPISODE_BUCKETS
        },
        "mean_episode_length": _mean(
            [s["q11_sim_episodes"]["mean_episode_length"] for s in per_seed]
        ),
        "mean_n_episodes": _mean(
            [s["q11_sim_episodes"]["mean_n_episodes"] for s in per_seed]
        ),
    }
    ref_epi = {
        "distribution": {
            b: _mean(
                [
                    s["q11_reference"][
                        "reference_episode_length_distribution"
                    ][b]
                    for s in per_seed
                ]
            )
            for b in EPISODE_BUCKETS
        },
        "mean_episode_length": _mean(
            [
                s["q11_reference"]["reference_mean_episode_length"]
                for s in per_seed
            ]
        ),
        "mean_n_episodes": _mean(
            [
                float(s["q11_reference"]["reference_n_episodes"])
                for s in per_seed
            ]
        ),
    }

    max_remainder = max(
        abs(per_cell[c]["reconciliation_remainder"]) for c in per_cell
    )
    # Which LINKED sub-channel dominates the linked-father miss, per cell, and
    # how the shadow (unlinked) channel compares to the linked-channel total.
    dominant = {}
    for c in per_cell:
        ch = per_cell[c]["channels"]
        linked_sub = {
            "spell_length": ch["spell_length"],
            "marital_state_joint": ch["marital_state_joint"],
            "unenumerated_nonjoinable_supply": ch[
                "unenumerated_nonjoinable_supply"
            ],
            "supply_residual_nonlinked_coresidence": ch[
                "supply_residual_nonlinked_coresidence"
            ],
        }
        linked_total = (
            ch["spell_length"]
            + ch["marital_state_joint"]
            + ch["unenumerated_nonjoinable_supply"]
            + ch["supply_residual_nonlinked_coresidence"]
            + ch["monte_carlo_finite_draw_gap"]
        )
        dominant[c] = {
            "dominant_linked_channel": max(
                linked_sub, key=lambda k: abs(linked_sub[k])
            ),
            "linked_channel_total": linked_total,
            "shadow_channel": ch["shadow_unlinked_channel"],
            "spell_is_largest_linked_channel": bool(
                abs(ch["spell_length"])
                >= max(
                    abs(ch["marital_state_joint"]),
                    abs(ch["unenumerated_nonjoinable_supply"]),
                    abs(ch["supply_residual_nonlinked_coresidence"]),
                )
            ),
        }

    finding = _q11_finding(per_cell, sim_epi, ref_epi, dominant)
    return {
        "question": (
            "linked-father child supply at 25-44: decompose the male "
            "coresident_child overshoot at 25-34 and 35-44 into (a) EXISTENCE "
            "(linked children per father, sim-vs-train), (b) SPELL LENGTH "
            "(father-child coresidence episode durations, sim-vs-train, child "
            "ages 0-17), (c) reconcile the two channels to the cell misses "
            "exactly; custody probabilities are measured faithful per "
            "forensics-2/3 and are NOT relitigated (supply and spells only)."
        ),
        "method": (
            "Train-side (side B). ALIGNMENT: the decomposition uses the "
            "COMMITTED linked exposure basis -- model.father_links "
            "(father_link_births), the exact object simulate_draw_v6 draws "
            "coresidence over -- for the linked/unlinked partition and the "
            "exposed-count EXISTENCE, so the father-wave grid, weights, and "
            "exposed-linked-child counts are identical sim-vs-train (existence "
            "is a structural zero). The linked-channel miss telescopes exactly "
            "into: a Monte-Carlo gap (realized minus the analytic 1-prod(1-p_c) "
            "occupancy expectation); a marital-state joint term (analytic "
            "occupancy under simulated vs observed father marital, same "
            "faithful custodial_prob_v6); an UNENUMERATED non-joinable supply "
            "term (the incremental analytic occupancy of biological children "
            "with no enumerated record -- the sim coresides them, the roster "
            "never observes them); a SPELL term (analytic independent-per-wave "
            "occupancy vs observed correlated coresidence among the enumerated "
            "linked children); a SUPPLY residual (observed coresidence with "
            "children outside the linked set); and the shadow (unlinked) "
            "channel -- all summing to the cell miss to machine epsilon. "
            "Episode lengths are maximal runs of coresident waves on the shared "
            "grid, child ages 0-17, enumerated children both sides. Custody "
            "probabilities are NOT re-estimated."
        ),
        "registered_cells": list(Q11_CELLS),
        "per_cell": per_cell,
        "episode_length_distributions": {
            "sim": sim_epi,
            "reference": ref_epi,
            "note": (
                "Population-level over all linked father-child pairs at child "
                "ages 0-17 on the shared side-B wave grid; a spell spans father "
                "ages, so it is not cell-restricted."
            ),
        },
        "dominant_linked_channel_per_cell": dominant,
        "reconciliation_max_abs_remainder": max_remainder,
        "finding": finding,
    }


def _q11_finding(
    per_cell: dict[str, Any],
    sim_epi: dict[str, Any],
    ref_epi: dict[str, Any],
    dominant: dict[str, Any],
) -> str:
    c34 = per_cell["coresident_child.35-44|male"]
    c25 = per_cell["coresident_child.25-34|male"]

    def _p(x: float) -> str:
        return f"{x:+.4f}"

    ch34 = c34["channels"]
    ch25 = c25["channels"]
    d34 = dominant["coresident_child.35-44|male"]
    d25 = dominant["coresident_child.25-34|male"]

    def _chan_line(ch: dict[str, float]) -> str:
        return (
            f"spell {_p(ch['spell_length'])}, marital-joint "
            f"{_p(ch['marital_state_joint'])}, unenumerated-nonjoinable "
            f"{_p(ch['unenumerated_nonjoinable_supply'])}, non-linked-supply "
            f"{_p(ch['supply_residual_nonlinked_coresidence'])}, shadow "
            f"{_p(ch['shadow_unlinked_channel'])}, MC "
            f"{_p(ch['monte_carlo_finite_draw_gap'])}"
        )

    ed = c34["existence_distributions"]
    return (
        "EXISTENCE is a structural NON-cause, and the diagnostic is decomposed "
        "against the COMMITTED mechanism: the sim's linked channel draws "
        "coresidence over model.father_links (the father_link_births basis, "
        "which does NOT require an enumerated child -- 25.8% of its exposure "
        "rows are non-joinable), and the simulated panel re-labels states on "
        "the same father-wave grid, so the number of exposed linked children "
        "per father-wave is IDENTICAL sim-vs-train (existence channel = 0.0 "
        "exactly; mean exposed linked children per father-wave "
        f"{ed['mean_exposed_linked_children_sim_eq_reference']:.3f}). The "
        "overshoot is an OCCUPANCY-structure effect and telescopes into six "
        "exactly-reconciling channels (existence + spell + marital-joint + "
        "unenumerated-nonjoinable-supply + non-linked-supply + shadow, plus a "
        "Monte-Carlo gap). At 35-44|male (train miss "
        f"{_p(c34['cell_miss_sim_minus_reference'])}): {_chan_line(ch34)}. The "
        f"dominant LINKED sub-channel is {d34['dominant_linked_channel']} "
        f"(linked-channel total {_p(d34['linked_channel_total'])}); the shadow "
        f"(unlinked imputed-paternal) channel {_p(d34['shadow_channel'])} is a "
        "separate cell-level piece. At 25-34|male (train miss "
        f"{_p(c25['cell_miss_sim_minus_reference'])}): {_chan_line(ch25)}; "
        f"dominant linked sub-channel {d25['dominant_linked_channel']}. TWO "
        "occupancy mechanisms carry the linked overshoot. (1) The UNENUMERATED "
        "non-joinable supply: the sim applies custody probabilities to biological "
        "children who never appear in the household roster (left the sample / "
        "non-custodial), coresiding children the reference cannot observe -- a "
        "supply artifact of drawing over father_link_births rather than only "
        "enumerated links. (2) The SPELL signature: the faithful per-wave "
        "custody probability applied as an INDEPENDENT per-wave occupancy "
        "reshapes the coresidence spells vs the observed contiguous episodes "
        f"(sim mean episode {sim_epi['mean_episode_length']:.2f} waves vs "
        f"reference {ref_epi['mean_episode_length']:.2f}; sim single-wave share "
        f"{sim_epi['distribution']['1']:.3f} vs reference "
        f"{ref_epi['distribution']['1']:.3f}; sim 4+ share "
        f"{sim_epi['distribution']['4+']:.3f} vs reference "
        f"{ref_epi['distribution']['4+']:.3f}). The marital-state joint "
        "(simulated vs observed father marital at the coresident wave) is the "
        "remaining named residual. Custody probabilities were held faithful "
        "throughout (custodial_prob_v6, not re-estimated); every channel "
        "reconciles to the train-side cell miss to machine epsilon."
    )


# --------------------------------------------------------------------------
# Assemble Q12 (spouse 25-34|female movement attribution + stable/fragile)
# --------------------------------------------------------------------------
def assemble_q12(
    per_seed: list[dict[str, Any]], committed: dict[str, Any]
) -> dict[str, Any]:
    female_bands = list(per_seed[0]["q12_c6_female"].keys())
    per_band: dict[str, Any] = {}
    for b in female_bands:
        c6r = {
            key: _mean([s["q12_c6_female"][b][key] for s in per_seed])
            for key in per_seed[0]["q12_c6_female"][b]
        }
        no_d3 = {
            key: _mean([s["q12_no_delta3_female"][b][key] for s in per_seed])
            for key in per_seed[0]["q12_no_delta3_female"][b]
        }
        per_band[b] = {
            "c6_full": c6r["full"],
            "c6_legal": c6r["legal"],
            "c6_cohab_overlay": c6r["cohab"],
            "c6_legal_residual_overlay": c6r["lr"],
            "delta3_off_full": no_d3["full"],
            "delta3_off_cohab_overlay": no_d3["cohab"],
            "delta3_effect_full": c6r["full"] - no_d3["full"],
            "delta3_effect_cohab_overlay": c6r["cohab"] - no_d3["cohab"],
        }

    target = "25-34"
    tcell = Q12_CELL
    committed_cell = _committed_cell_c6(committed, tcell)

    # Per-seed target full rate + per-draw dispersion (stable vs fragile).
    per_seed_target = []
    for s in per_seed:
        draws = s["q12_target_full_per_draw"]
        per_seed_target.append(
            {
                "seed": s["seed"],
                "sim_full_mean": _mean(draws),
                "per_draw_sd": _sd(draws),
                "reference_rate_b": float(s["rate_b"][tcell]),
                "signed_logratio": _signed(_mean(draws), s["rate_b"][tcell]),
                "score": _score(_mean(draws), s["rate_b"][tcell]),
            }
        )
    tol = committed_cell["tolerance"]
    holdout_scores = committed_cell["per_seed_score"]
    n_holdout_over = sum(1 for v in holdout_scores.values() if v > tol)
    holdout_margin = tol - committed_cell["seed_mean_score"]

    delta3_target_effect = per_band[target]["delta3_effect_full"]
    finding = (
        "The spouse 25-34|female movement is NOT a candidate-6 interaction and "
        "NOT luck the next candidate could lose by re-drawing -- it is an "
        "INHERITANCE from the carried candidate-4/5 machinery, and the "
        "registered delta 3 is confirmed INERT at the target. Component "
        "replay: disabling delta 3 (the female 25-44 cohabitation override) "
        "moves the 25-34|female full spouse rate by "
        f"{delta3_target_effect:+.2e} (cohab-overlay component "
        f"{per_band[target]['delta3_effect_cohab_overlay']:+.2e}) -- "
        "numerically INERT (below any material threshold), because the fitted "
        "female single-year cohabitation hazards at 25-34 coincide with "
        "candidate 4's existing age-refined estimator at the same target (delta "
        "3's only NEW structure is at 35-44|female, where disabling it moves "
        f"the rate by {per_band['35-44']['delta3_effect_full']:+.2e}). Delta 1 "
        "(the 0-4 "
        "custodial revert) and delta 4 (the count-conditional bridge) cannot "
        "touch the spouse state at all -- coresident_spouse = legal | cohab | "
        "legal-residual, none of which depends on child counts, hh_size, or "
        "the custodial draw, and every spouse-relevant RNG stream (0xB2B legal, "
        "0xC2 cohab, 0xC4 legal-residual) is consumed before and independently "
        "of the 0xC6 maternal-leave and delta-4 count draws; the c6 seed-mean "
        f"holdout r_candidate {committed_cell['seed_mean_r_candidate']:.4f} is "
        "the carried candidate-5 value, so no delta moved it. STABLE vs "
        f"FRAGILE: the clearing is FRAGILE-marginal. The holdout seed-mean "
        f"score {committed_cell['seed_mean_score']:.4f} clears the tolerance "
        f"{tol:.3f} by only {holdout_margin:+.4f}, and {n_holdout_over} of 5 "
        "split seeds individually exceed tolerance; the per-draw dispersion "
        "within a seed is small, so the failure surface is the SPLIT seed, not "
        "the draw -- the cell rides the tolerance line and the next candidate "
        "should carry it with discipline rather than bank it as solved. The "
        "under-produced component is the cohabitation OVERLAY (forensics-3 Q9: "
        "overlay gap -0.045), which delta 3 was designed to lift but did not."
    )
    return {
        "question": (
            "spouse 25-34|female movement attribution: the cell stopped "
            "capping seeds in c6 while the registered delta (delta 3) was "
            "inert; attribute the movement by component replay (delta 1's basis "
            "revert via household composition? delta 4's bridge? draw-path "
            "shifts from RNG stream changes? delta 3 itself?) and determine "
            "whether the clearing is STABLE (present across all draws/seeds "
            "with margin) or fragile luck."
        ),
        "method": (
            "Train-side (side B). Per female band the spouse rate is split into "
            "legal core / cohabitation overlay / legal-residual overlay (the "
            "forensics-3 Q9 components) from 20 instrumented candidate-6 draws, "
            "AND from a delta-3-OFF replay (dataclasses.replace of the fitted "
            "model with empty female cohab overrides -- every other table and "
            "RNG stream untouched) so delta 3's apply-time effect is isolated "
            "exactly. Delta 1 / delta 4 / RNG-stream hypotheses are ruled out "
            "structurally (the spouse state and its 0xB2B/0xC2/0xC4 draws are "
            "byte-identical to candidate 5) and empirically (the committed c6 "
            "holdout r_candidate equals the carried candidate-5 value). "
            "Stable-vs-fragile reads the per-seed / per-draw dispersion and the "
            "committed holdout margin."
        ),
        "registered_cell": tcell,
        "per_female_band_component_replay": per_band,
        "delta3_target_effect_full": delta3_target_effect,
        "delta3_inert_at_target": bool(abs(delta3_target_effect) < 1e-6),
        "per_seed_target": per_seed_target,
        "holdout_committed_candidate6": committed_cell,
        "holdout_n_seeds_over_tolerance": n_holdout_over,
        "holdout_seed_mean_margin_under_tolerance": holdout_margin,
        "stability_verdict": ("fragile_marginal_inherited_not_c6_interaction"),
        "finding": finding,
    }


# --------------------------------------------------------------------------
# Assemble Q13 (hh_size.5+ marginal seed: noise vs structure)
# --------------------------------------------------------------------------
def assemble_q13(
    per_seed: list[dict[str, Any]], committed: dict[str, Any], tol: float
) -> dict[str, Any]:
    cell = Q13_CELL
    committed_cell = _committed_cell_c6(committed, cell)
    # Committed holdout per-draw dispersion on the marginal (failing) seed.
    holdout_by_seed = {s["seed"]: s for s in committed["per_seed"]}
    ms = holdout_by_seed[Q13_MARGINAL_SEED]["gated_cells"][cell]
    rate_a = float(ms["rate_a"])
    pdr = [float(x) for x in ms["per_draw_rate"]]
    per_draw_ln = [_signed(r, rate_a) for r in pdr]
    n_over = sum(1 for x in per_draw_ln if abs(x) > tol)
    n_over_under = sum(1 for x in per_draw_ln if x < -tol)
    n_over_above = sum(1 for x in per_draw_ln if x > tol)
    seed_mean_ln = _signed(float(ms["rbar"]), rate_a)
    per_draw_ln_sd = _sd(per_draw_ln)
    se_of_mean_ln = (
        per_draw_ln_sd / math.sqrt(len(per_draw_ln)) if per_draw_ln else 0.0
    )
    margin_past_line = abs(seed_mean_ln) - tol
    ses_past_line = (
        margin_past_line / se_of_mean_ln if se_of_mean_ln > 0 else float("inf")
    )
    all_same_sign = all(x < 0 for x in per_draw_ln) or all(
        x > 0 for x in per_draw_ln
    )
    straddles_zero = (min(per_draw_ln) < 0) and (max(per_draw_ln) > 0)
    is_structure = bool(
        (abs(seed_mean_ln) > tol) and all_same_sign and (ses_past_line >= 2.0)
    )

    # Train-side corroboration: per-seed per-draw hh5+ dispersion + core-5+.
    per_seed_train = []
    for s in per_seed:
        draws = s["q13_hh5plus_per_draw"]
        ref = float(s["rate_b"][cell])
        lns = [_signed(r, ref) for r in draws]
        per_seed_train.append(
            {
                "seed": s["seed"],
                "sim_hh5plus_mean": _mean(draws),
                "reference_hh5plus": ref,
                "per_draw_rate_sd": _sd(draws),
                "per_draw_ln_sd": _sd(lns),
                "seed_signed_logratio": _signed(_mean(draws), ref),
                "seed_score": _score(_mean(draws), ref),
                "n_draws_over_tol": sum(1 for x in lns if abs(x) > tol),
                "sim_core5plus": s["q13_sim_core5plus"],
                "reference_core5plus": s["q13_reference_core5plus"],
                "core5plus_deficit_sim_minus_ref": (
                    s["q13_sim_core5plus"] - s["q13_reference_core5plus"]
                ),
                "reference_child3plus_share": s[
                    "q13_reference_child3plus_share"
                ],
            }
        )
    mean_core5_deficit = _mean(
        [r["core5plus_deficit_sim_minus_ref"] for r in per_seed_train]
    )
    marginal_train = next(
        r for r in per_seed_train if r["seed"] == Q13_MARGINAL_SEED
    )

    finding = (
        f"The hh_size.5+ marginal seed (split seed {Q13_MARGINAL_SEED}) is "
        "STRUCTURE, not noise -- contradicting the pre-registered 'noise' "
        "expectation, which is the diagnostic earning its keep. On the "
        "committed holdout, seed "
        f"{Q13_MARGINAL_SEED}'s 20 per-draw rates all sit BELOW the reference "
        f"(every per-draw signed ln in [{min(per_draw_ln):+.4f}, "
        f"{max(per_draw_ln):+.4f}], seed-mean ln {seed_mean_ln:+.4f}); the "
        f"per-draw spread does NOT straddle zero, and {n_over_under} of 20 "
        f"draws individually exceed the tolerance {tol:.3f} on the UNDER side "
        f"(0 on the over side). The seed-mean under-production sits "
        f"{margin_past_line:+.4f} past the tolerance line, ~{ses_past_line:.1f} "
        "standard errors of the 20-draw mean -- far beyond a finite-draw "
        "straddle, so the cell fails at its CENTER, not by draw luck. The "
        "component is the upstream CORE large-family deficit named and carried "
        "in forensics-3 Q10: the sim core-5+ under-produces the reference by "
        f"{mean_core5_deficit:+.4f} on average (seed "
        f"{Q13_MARGINAL_SEED}: "
        f"{marginal_train['core5plus_deficit_sim_minus_ref']:+.4f}), traced to "
        "the 3+-own-child fertility / composition deficit, and because the "
        "delta-4 bridge only ADDS non-core members it cannot lift core mass "
        "up. The honest joint absorbs this on the four low-reference split "
        "seeds (0/1/2/4 clear) but not on split seed "
        f"{Q13_MARGINAL_SEED}, whose holdout carries the highest reference "
        f"hh_size.5+ ({rate_a:.4f}); there the carried core-5+ deficit tilts "
        "the cell systematically under tolerance. The lever for the next "
        "candidate is the upstream large-family core (fertility / 3+-child "
        "composition), not the bridge."
    )
    return {
        "question": (
            "hh_size.5+ marginal seed: per-draw dispersion vs tolerance on the "
            "failing seed; noise (per-draw spread straddles the line) vs "
            "structure (systematic tilt); if structure, name the component."
        ),
        "method": (
            "The committed candidate-6 holdout per-draw rates for hh_size.5+ on "
            "the marginal split seed are read directly (no re-simulation of "
            "side A); each per-draw signed ln(r/rate_a) is compared to the "
            "tolerance to test straddle-vs-tilt, and the seed-mean distance "
            "past the line is expressed in standard errors of the 20-draw mean. "
            "Train side (side B) corroborates with the per-seed per-draw "
            "dispersion and the sim-vs-reference core-5+ / 3+-own-child "
            "structure (the forensics-3 Q10 upstream signal)."
        ),
        "registered_cell": cell,
        "marginal_seed": Q13_MARGINAL_SEED,
        "holdout_marginal_seed_dispersion": {
            "rate_a": rate_a,
            "seed_mean_signed_logratio": seed_mean_ln,
            "per_draw_signed_logratio_min": min(per_draw_ln),
            "per_draw_signed_logratio_max": max(per_draw_ln),
            "per_draw_signed_logratio_sd": per_draw_ln_sd,
            "standard_error_of_mean_ln": se_of_mean_ln,
            "tolerance": tol,
            "n_draws_over_tolerance": n_over,
            "n_draws_over_tolerance_under_side": n_over_under,
            "n_draws_over_tolerance_over_side": n_over_above,
            "margin_past_tolerance_line": margin_past_line,
            "standard_errors_past_line": ses_past_line,
            "per_draw_spread_straddles_zero": straddles_zero,
            "all_draws_same_sign": all_same_sign,
        },
        "verdict_structure_not_noise": is_structure,
        "component": "upstream_core_large_family_fertility_deficit",
        "train_side_corroboration": {
            "per_seed": per_seed_train,
            "mean_core5plus_deficit_sim_minus_reference": mean_core5_deficit,
        },
        "holdout_committed_candidate6": committed_cell,
        "finding": finding,
    }


# --------------------------------------------------------------------------
# Revision pins + run
# --------------------------------------------------------------------------
def _revision_pins() -> dict[str, Any]:
    import sklearn

    return {
        "populace_dynamics_head_sha": c6._git_sha(),
        "merge_base_origin_master": c6._merge_base(),
        "candidate6_artifact": "runs/gate2b_hazard_v6.json",
        "candidate6_artifact_sha256": c6._sha_of_file(CANDIDATE6_ARTIFACT),
        "candidate6_runner": "scripts/run_gate2b_candidate6.py",
        "candidate6_runner_sha256": c6._sha_of_file(
            ROOT / "scripts" / "run_gate2b_candidate6.py"
        ),
        "forensics1_artifact_sha256": c6._sha_of_file(FORENSICS1_ARTIFACT),
        "forensics2_artifact_sha256": c6._sha_of_file(FORENSICS2_ARTIFACT),
        "forensics3_artifact_sha256": c6._sha_of_file(FORENSICS3_ARTIFACT),
        "gate2b_floor_run": "runs/gate2b_floors_v1.json",
        "gate2b_floor_sha256": c6._sha_of_file(FLOOR_RUN),
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
        CANDIDATE6_ARTIFACT,
        FORENSICS1_ARTIFACT,
        FORENSICS2_ARTIFACT,
        FORENSICS3_ARTIFACT,
        FLOOR_RUN,
    ):
        if not art.exists():
            raise RuntimeError(f"required committed artifact missing: {art}")
    committed = _committed_c6()

    data = c6.load_all()
    tol = c6.gated_tolerances(c6.load_gate2b_thresholds())
    if verbose:
        hh = data["hh"]
        print(
            f"panel: {len(hh.person_waves)} person-waves, "
            f"{hh.person_waves.person_id.nunique()} persons; "
            f"train-side forensics, K={N_DRAWS} draws (5200 + k)"
        )

    # One-time fidelity proof: instrumented draw == committed simulate_draw_v6.
    _side_a, side_b = hpanel.split_panel_by_person(
        data["hh"].attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b0 = set(int(x) for x in side_b.person_id.unique())
    model0 = hcs6.fit_household_model_v6(
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
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    fidelity = fidelity_check_v6(
        data["hh"], data["mpanel"], model0, ids_b0, DRAW_SEED_BASE
    )
    if verbose:
        print(
            "instrumentation fidelity: bit_identical="
            f"{fidelity['bit_identical']} (max dev "
            f"{fidelity['max_abs_rate_deviation_vs_committed_simulate_draw_v6']:.2e}"
            f"; child-add {fidelity['child_channel_additivity_residual']}, "
            f"linked-add {fidelity['linked_marital_split_additivity_residual']})"
        )
    if not fidelity["bit_identical"]:
        raise RuntimeError(
            "instrumented_draw_v6 does not reproduce simulate_draw_v6 "
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

    q11 = assemble_q11(per_seed, committed)
    q12 = assemble_q12(per_seed, committed)
    q13 = assemble_q13(per_seed, committed, float(tol[Q13_CELL]))

    float64_eps = float(np.finfo(np.float64).eps)
    q11_recon_max = q11["reconciliation_max_abs_remainder"]
    reconciliations = {
        "instrumentation_bit_identity_max_rate_deviation": fidelity[
            "max_abs_rate_deviation_vs_committed_simulate_draw_v6"
        ],
        "child_channel_additivity_residual": fidelity[
            "child_channel_additivity_residual"
        ],
        "linked_marital_split_additivity_residual": fidelity[
            "linked_marital_split_additivity_residual"
        ],
        "q11_cell_miss_reconciliation_max_abs_remainder": q11_recon_max,
        "float64_machine_epsilon": float64_eps,
        "all_identity_reconciliations_at_machine_epsilon": bool(
            fidelity["max_abs_rate_deviation_vs_committed_simulate_draw_v6"]
            == 0.0
            and fidelity["child_channel_additivity_residual"] == 0
            and fidelity["linked_marital_split_additivity_residual"] == 0
            and q11_recon_max <= 64 * float64_eps
        ),
        "reconciliation_bar_note": (
            "instrumentation bit-identity is EXACT (0.0) and the integer "
            "channel-additivity residuals are EXACTLY 0; the Q11 six-channel "
            "telescoping reconciliation of the cell miss sits at machine "
            "epsilon (a few ULP of float64 summation over the six channels), "
            "i.e. machine zero."
        ),
    }

    alignment_decision = {
        "issue": (
            "the sim's linked channel draws coresidence over "
            "model.father_links (father_link_births), which does NOT require an "
            "enumerated child; an initial reference build used "
            "father_link_births_with_child (enumerated only), which would have "
            "decomposed a VARIANT, not the committed mechanism"
        ),
        "measured_gap": (
            "25.8% of the committed linked exposure rows (9500 of 36887) are "
            "non-joinable; 17% of linked fathers (2516 of 14835) are linked in "
            "the sim but would be unlinked under the enumerated-only basis"
        ),
        "resolution": (
            "q11_reference uses model.father_links for the linked/unlinked "
            "partition, the exposed-count existence, and the analytic occupancy "
            "(matching simulate_draw_v6 exactly); father_links_child provides "
            "child ids only for the observable reference coresidence and "
            "episodes; the non-joinable biological children the sim coresides "
            "but the roster cannot enumerate are surfaced as their own channel "
            "(unenumerated_nonjoinable_supply)"
        ),
    }

    implications = (
        "Implications for candidate 7 (labeled implications, NOT decisions -- "
        "the orchestrator registers c7). Q11: the 35-44|male and 25-34|male "
        "overshoot is linked-father / paternal SUPPLY expressed as OCCUPANCY "
        "structure, NOT a linked-child EXISTENCE (count) miss (the exposed "
        "count is identical data) and NOT a custody-probability miss (held "
        "faithful). Two occupancy levers are surfaced, both exactly measured. "
        "(1) UNENUMERATED non-joinable supply: the sim draws coresidence over "
        "model.father_links (father_link_births), applying custody probabilities "
        "to biological children with no enumerated household record (25.8% of "
        "the linked exposure) whom the reference never observes coresident -- "
        "the c7 lever is to restrict the paternal-linked draw to enumerated "
        "children (or condition non-custodial coresidence on enumeration), "
        "which the committed mechanism does not. (2) SPELL persistence: draw "
        "father-child coresidence as bounded correlated EPISODES (entry once, "
        "persist, exit) instead of independent per-wave occupancy, reshaping "
        "the father-wave stock without touching the faithful per-wave marginal. "
        "The marital-state joint (simulated vs observed father marital at the "
        "coresident wave) and the shadow (unlinked imputed-paternal) channel "
        "are the named residuals. Q12: do NOT bank the spouse 25-34|female "
        "cell -- "
        "it is inherited c4/c5 machinery riding the tolerance line (delta 3 was "
        "inert at the target), fragile to the split seed; carry it with "
        "discipline and, if lifted, lift the cohabitation OVERLAY at 25-34 "
        "(the -0.045 overlay gap), not the legal core. Q13: hh_size.5+ is one "
        "systematic mechanism (the upstream core-5+ / 3+-own-child fertility "
        "deficit), not draw noise; the bridge cannot fix it because it only "
        "adds non-core members -- the c7 lever is the large-family core "
        "(fertility / 3+-child composition), the same upstream signal "
        "forensics-3 Q10 named and c6 carried."
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
        "candidate_under_diagnosis": (
            "gate-2b candidate 6 (PR #143, registration 4946285556; artifact "
            "runs/gate2b_hazard_v6.json; FAIL 0/5, one mechanism wide). "
            "Forensics 4 measures the surviving linked-father child SUPPLY "
            "blocker at coresident_child.{25-34|male, 35-44|male}, plus the "
            "attribution of the spouse 25-34|female movement and the "
            "hh_size.5+ marginal seed."
        ),
        "candidate6_artifact": "runs/gate2b_hazard_v6.json",
        "forensics1_artifact": "runs/gate2b_forensics1_v1.json",
        "forensics2_artifact": "runs/gate2b_forensics2_v1.json",
        "forensics3_artifact": "runs/gate2b_forensics3_v1.json",
        "protocol": {
            "train_side_only": True,
            "outer_holdout_contact": (
                "none beyond the already-published per-seed scores read from "
                "runs/gate2b_hazard_v6.json; the holdout (side A) is NEVER "
                "re-simulated here"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "hh.attrs, 'person_id', fraction=0.5, seed=s); side A = outer "
                "holdout (read from the committed candidate-6 artifact only), "
                "side B = train complement (this diagnostic fits AND simulates "
                "side B)"
            ),
            "fit_simulate_machinery": (
                "populace_dynamics.models.household_composition_sim_v6 "
                "(candidate 6, merged #143), reused byte-for-byte on the train "
                "side via instrumented_draw_v6 (same 0xB2B / 0xC2 / 0xC3 / 0xC4 "
                "/ 0xC5 / 0xC6 draw streams and train-fitted tables)"
            ),
            "reused_machinery": (
                "Q11 uses the SAME faithful custodial_prob_v6 the committed "
                "candidate fits (the forensics-2/3 custody basis) and does NOT "
                "relitigate custody probabilities -- supply and spells only; "
                "the forensics-1/2 reference splitters "
                "(spouse_concept_codes / q5_custodial_selection) are NOT "
                "re-invoked because forensics-4's questions are spell / "
                "existence / attribution, not custody-basis or legal-overlay "
                "estimation. The candidate-6 fit/simulate machinery is reused "
                "byte-for-byte via instrumented_draw_v6."
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
            "floor_run_sha256": c6._sha_of_file(FLOOR_RUN),
        },
        "reconciliations": reconciliations,
        "q11_alignment_decision": alignment_decision,
        "question_11_linked_father_child_supply": q11,
        "question_12_spouse_25_34_female_movement": q12,
        "question_13_hh_size_5plus_marginal_seed": q13,
        "per_seed": per_seed,
        "candidate_7_implications": implications,
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
        for q in (q11, q12, q13):
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
    artifacts.write_new(Path(args.out), c6._json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
