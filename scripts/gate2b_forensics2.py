"""Gate-2b forensics 2 -- concept-class residual decomposition (REPORTED, NOT
GATED).

The registered diagnostic of PolicyEngine/populace-dynamics issue #42, comment
4942005972 ("Gate-2b forensics 2 registration (diagnostic, not gated): the
three concept-class residuals"). Candidate 4 (grading 4942004647) reduced the
failure surface to three residuals that are concept / selection problems, not
hazard fitting; this diagnostic measures each before candidate 5. The
registration wins: this runner answers exactly its three frozen questions.

FROZEN SPEC (comment 4942005972), three questions:

* **Q5 -- custodial selection basis.** Construct the LESS-SELECTED custody
  measure: for every sample child (child's own record) in train, observe which
  family unit the child resides in, independent of the father-side observation
  window; recompute ``P(child coresides with father | child age, father marital
  state)`` on that basis; quantify the selection gap vs the (father, child,
  wave)-observable basis c3/c4 fitted (forensics-1 measured 0.95-0.99
  young-child on the observable subset); reconcile against the
  maternal-coresidence complement (a child not with the father is usually with
  the mother -- the mother-side rates are directly observable and constrain the
  truth).
* **Q6 -- grandchild reference channels.** Decompose the reference's 55+ female
  ``coresident_grandchild`` stock (~0.063 train) by household-structure channel
  from the raw roster: (a) three-generation with ego on top (ego's parent
  absent, ego's child + grandchild present), (b) skipped-generation (grandchild
  without the middle generation), (c) any OTHER roster configurations carrying
  grandparent-grandchild codes (enumerate codes and shares -- including whether
  the middle generation being a child-IN-LAW rather than ego's own child breaks
  the composed multigen AND child AND NOT-parent test); for each channel, the
  simulation's reachable supply under the v4 machinery; identify exactly which
  channel(s) carry the unreachable ~0.039.
* **Q7 -- hh_size.3 family-core routes.** Decompose size-3 households
  sim-vs-reference by composition route (couple+child / single-parent+2-children
  / couple+parent / three-adults / other) on train; for the over-produced
  route(s), identify the upstream driver (child aging-out timing? fertility
  level in the sim's composed frame? marital-state joint) with a quantified
  reconciliation of the 0.271-vs-0.181 core gap.

Train-side only, per the forensics-1 protocol. The candidate-4 fit / simulate
machinery (:mod:`populace_dynamics.models.household_composition_sim_v4`, merged
#138) is fit AND simulated on side B (the train complement of each gate seed's
person-disjoint 50/50 split) and scored against side B's OWN empirical rate;
the outer holdout (side A) is NEVER re-simulated -- only the already-published
per-seed scores from the committed ``runs/gate2b_hazard_v4.json`` are read.
Every fitted table is estimated on side B only (no holdout-informed tuning
surface). The instrumented draw is proved bit-identical to the committed
``simulate_draw_v4`` before any component decomposition is trusted.

Environment: the gate ``.venv-gate`` (scikit-learn < 1.9) with the PSID
products staged (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``). Run from the repository root::

    .venv-gate/bin/python scripts/gate2b_forensics2.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 4 is the spec under diagnosis: its load_all / fit / simulate
# machinery is reused verbatim on the TRAIN side.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate2b_candidate4 as c4  # noqa: E402

from populace_dynamics import artifacts  # noqa: E402
from populace_dynamics.data import (  # noqa: E402
    births,  # noqa: E402
    relmap,
    transitions,
)
from populace_dynamics.data import household_composition as hc  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.models import family_transitions as ft  # noqa: E402
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim as hcs,
)
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v3 as hcs3,
)
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v4 as hcs4,
)
from populace_dynamics.models.family_transitions.components.fertility import (  # noqa: E402,E501
    build_fertility_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_forensics2_v1.json"
CANDIDATE4_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v4.json"
FORENSICS1_ARTIFACT = ROOT / "runs" / "gate2b_forensics1_v1.json"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
SCHEMA_VERSION = "gate2b_forensics2.v1"
RUN_NAME = "gate2b_forensics2_v1"

#: The registered diagnostic (issue #42, comment 4942005972). The registration
#: wins: this runner answers exactly its three frozen questions.
REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4942005972"
)
REGISTRATION_POINTER = "4942005972"
REGISTRATION_TITLE = (
    "Registered diagnostic: gate-2b forensics 2, the three concept-class "
    "residuals (custodial selection basis; grandchild reference channels; "
    "hh_size.3 family-core routes)"
)
GRADING_POINTER = "4942004647"
CANDIDATE4_REGISTRATION_POINTER = "4941160621"

#: Reused frozen dials (candidate 4 / the locked gate_2b protocol).
GATE_SEEDS = c4.GATE_SEEDS  # (0, 1, 2, 3, 4)
N_DRAWS = c4.N_DRAWS  # 20
DRAW_SEED_BASE = c4.DRAW_SEED_BASE  # 5200
EXACT_ATOL = c4.EXACT_ATOL  # 1e-12

#: Q5 custodial child age bands + father marital binary (candidate 3/4's exact
#: observable axes -- so the child-record basis lands on the same cells).
CUSTODIAL_CHILD_AGE_BANDS = hcs3.CUSTODIAL_CHILD_AGE_BANDS
CHILD_CORESIDENCE_MAX_AGE = hcs3.CHILD_CORESIDENCE_MAX_AGE  # 60
_MARRIED = hcs3._MARRIED
_NOT_MARRIED = hcs3._NOT_MARRIED

#: Q6 grandparent-type MX8 (ego->alter) codes. The reference
#: ``coresident_grandchild`` link is {66, 68, 82, 87, 88} (own/social/step
#: grandparent + great-grandparent), EXCLUDING the in-law codes {67, 69} and --
#: an asymmetry -- the social great-grandparent 83. The full grandparent-type
#: inventory (verified against MX23REL_formats.sps lines 100-125) is:
GRANDPARENT_CODES_ALL: tuple[int, ...] = (66, 67, 68, 69, 82, 83, 87, 88)
GRANDPARENT_LINK_REF = hc.CORESIDENCE_LINKS[
    "coresident_grandchild"
]  # frozenset
CHILD_LINK = hc.CORESIDENCE_LINKS["coresident_child"]  # ego IS parent of alter
PARENT_LINK = hc.CORESIDENCE_LINKS[
    "coresident_parent"
]  # ego IS child of alter
#: Ego is parent-in-law / social-parent-in-law of alter (MX8): the middle
#: generation present as a child-IN-LAW rather than ego's own child.
CHILD_INLAW_LINK = frozenset({57, 58})
GRANDCHILD_LO = 55

#: Q7 size-3 composition routes (exact partition on both sim core and the raw
#: reference roster; "other" carries non-core members siblings/roomers/etc.).
Q7_ROUTES = (
    "couple_plus_child",
    "single_parent_plus_two_children",
    "couple_plus_parent",
    "three_adults",
    "other_family_core",
)

#: Per-seed cache OUTSIDE runs/ (never committed): a long run resumes.
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2b_forensics2_cache.json"
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


def _mean_over_seeds(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _wshare(weight: np.ndarray, hit: np.ndarray) -> float:
    """Weighted share of ``hit`` (a 0/1 or bool array) over ``weight``."""
    w = np.asarray(weight, dtype=np.float64)
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    return float((w * np.asarray(hit, dtype=np.float64)).sum() / tot)


# --------------------------------------------------------------------------
# Instrumented candidate-4 draw (faithful copy returning the components)
# --------------------------------------------------------------------------
def instrumented_draw_v4(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs4.HouseholdCompositionModelV4,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Reproduce :func:`hcs4.simulate_draw_v4` and return every component array.

    A line-for-line copy of the committed candidate-4 draw (same 0xB2B / 0xC2 /
    0xC3 / 0xC4 substreams, same train-fitted tables), instrumented to also
    return the family-core components (spouse, coresident-parent, child counts,
    parent count) and the composed-vs-skipgen grandchild split -- all aligned to
    the returned ``pw`` row order. Equality of the recomposed panel against the
    committed :func:`hcs4.simulate_draw_v4` is proved once by
    :func:`fidelity_check_v4`.
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

    # 3. DELTA 1: age-refined cohabitation (code-22) overlay.
    cohab_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    cohab_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.base_v2.cohab_entry.items():
        cohab_entry_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (band, sex), rate in model.base_v2.cohab_exit.items():
        cohab_exit_prob[(band_mat == band) & (sex_mat == sex)] = rate
    for (age, sex), rate in model.cohab_entry_age.items():
        cohab_entry_prob[(age_mat == age) & (sex_mat == sex)] = rate
    for (age, sex), rate in model.cohab_exit_age.items():
        cohab_exit_prob[(age_mat == age) & (sex_mat == sex)] = rate
    cohab_state = hcs._evolve_two_state(
        valid, obs_cohab, cohab_entry_prob, cohab_exit_prob, cohab_rng
    )
    cohab_row = np.zeros(len(pw), dtype=bool)
    cohab_row[row_of[valid]] = cohab_state[valid]

    # 6. candidate-4 delta substreams (0xC4).
    c4_ss = np.random.SeedSequence([draw_seed, 0xC4])
    legal_resid_ss, nonfamily2_ss = c4_ss.spawn(2)
    legal_resid_rng = np.random.default_rng(legal_resid_ss)
    nonfamily2_rng = np.random.default_rng(nonfamily2_ss)

    # DELTA 2: additive legal-spouse residual overlay (0xC4).
    lr_entry_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_exit_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    lr_marg_prob = np.zeros((n_persons, max_waves), dtype=np.float64)
    for (band, sex), rate in model.legal_residual_entry.items():
        m = (band_mat == band) & (sex_mat == sex)
        lr_exit_prob[m] = model.legal_residual_exit[(band, sex)]
        lr_entry_prob[m] = rate
        lr_marg_prob[m] = model.legal_residual_marginal[(band, sex)]
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

    # 4. certified marital core + maternal births.
    sim_panel, sim_births = ft.simulate(
        mpanel, ids, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"

    # 5. linked / shadow paternal children (candidate-2 child stream 0xC2).
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
    child_leaves = hcs._child_leave_years(
        all_births, base.parental_exit, child_rng
    )
    nonlinked_leaves = child_leaves[child_leaves["_source"] != "linked"]
    child_counts_nonlinked = hcs._coresident_child_counts(
        nonlinked_leaves, side_a_pw
    )

    # candidate-3 delta substreams (0xC3).
    c3_ss = np.random.SeedSequence([draw_seed, 0xC3])
    custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    custodial_rng = np.random.default_rng(custodial_ss)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    # DELTA 3a: custodial per-wave coresidence keyed by child age x era.
    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked = hcs4.custodial_linked_child_counts_v4(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model,
        custodial_rng,
    )
    child_counts = child_counts_nonlinked + child_counts_linked

    coresident_child, grandchild_composed, hh_size_base = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

    # DELTA 4: 5-year skipped-generation occupancy (55+).
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
    for ((lo, hi), sex), rate in model.skipgen_entry_age.items():
        m = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sex)
        skip_entry_prob[m] = rate
    for ((lo, hi), sex), rate in model.skipgen_exit_age.items():
        m = (age_mat >= lo) & (age_mat <= hi) & (sex_mat == sex)
        skip_exit_prob[m] = rate
    skipgen_state = hcs._evolve_two_state(
        valid, obs_skipgen, skip_entry_prob, skip_exit_prob, skipgen_rng
    )
    skipgen_row = np.zeros(len(pw), dtype=bool)
    skipgen_row[row_of[valid]] = skipgen_state[valid]
    coresident_grandchild = grandchild_composed | skipgen_row

    # DELTA 3b: non-family count with the 2+ tail spread.
    nonfamily_count = hcs4._sample_nonfamily_v4(
        pw, model, nonfamily_rng, nonfamily2_rng
    )
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

    parent_count = int(base.parent_count)
    n_parents = np.where(c1_parent, parent_count, 0).astype(np.int64)
    return {
        "panel": panel,
        "band": pw["band"].to_numpy(dtype=object),
        "sex": pw["sex"].to_numpy(),
        "age": pw["age"].to_numpy(dtype=np.int64),
        "weight": pw["weight"].to_numpy(dtype=np.float64),
        "person_id": pw["person_id"].to_numpy(dtype=np.int64),
        # family-core components.
        "spouse": spouse,
        "coresident_parent": c1_parent,
        "multigen": c1_multigen,
        "child_counts": child_counts.astype(np.int64),
        "coresident_child": coresident_child,
        "n_parents": n_parents,
        "parent_count": parent_count,
        "hh_size_base": hh_size_base.astype(np.int64),
        "nonfamily_count": nonfamily_count.astype(np.int64),
        "hh_size": hh_size,
        # grandchild split.
        "grandchild_composed": grandchild_composed,
        "skipgen_row": skipgen_row,
        "coresident_grandchild": coresident_grandchild,
    }


def fidelity_check_v4(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs4.HouseholdCompositionModelV4,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Prove the instrumented draw equals the committed ``simulate_draw_v4``."""
    inst = instrumented_draw_v4(hh, mpanel, model, ids, draw_seed)
    real_panel, _ = hcs4.simulate_draw_v4(hh, mpanel, model, ids, draw_seed)
    a = hc.reference_moments(inst["panel"], ids, weighted=True)
    b = hc.reference_moments(real_panel, ids, weighted=True)
    max_dev = max(abs(a[c]["rate"] - b[c]["rate"]) for c in b)
    return {
        "draw_seed": draw_seed,
        "n_cells_compared": len(b),
        "max_abs_rate_deviation_vs_committed_simulate_draw_v4": float(max_dev),
        "bit_identical": bool(max_dev <= EXACT_ATOL),
    }


# --------------------------------------------------------------------------
# Q5: custodial selection basis (deterministic; observable vs child-record)
# --------------------------------------------------------------------------
def _all_parent_links(bh: pd.DataFrame) -> pd.DataFrame:
    """Every joinable biological parent->child link (both sexes), one per row.

    ``cah85_23`` father/mother births whose child is a joinable PSID person.
    Carries ``parent_sex`` so the father and mother sides split cleanly.
    """
    ev = births.birth_events(bh)
    sel = ev[
        (ev["record_type"] == "birth")
        & ev["birth_year"].notna()
        & ev["child_person_id"].notna()
    ]
    return pd.DataFrame(
        {
            "parent_person_id": sel["parent_person_id"].astype("int64"),
            "parent_sex": sel["parent_sex"].astype("string"),
            "child_person_id": sel["child_person_id"].astype("int64"),
            "birth_year": sel["birth_year"].astype("int64"),
        }
    ).reset_index(drop=True)


def q5_custodial_selection(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    rel_map: pd.DataFrame,
    parent_pairs: pd.DataFrame,
    links: pd.DataFrame,
    ids_b: set[int],
) -> dict[str, Any]:
    """Observable-basis vs child-record-basis custody, and maternal complement.

    * Observable basis: exactly :func:`hcs3.fit_custodial_coresidence`
      (denominator = FATHER-waves; the (father, child, wave) triples the c3/c4
      custodial gate is fit on).
    * Child-record basis: denominator = CHILD-waves (each joinable child's own
      enumerated waves, independent of the father-side observation window);
      event = the father is coresident with the child that wave. The maternal
      complement partitions each child-wave into {both, father-only,
      mother-only, neither}, which sums to 1 EXACTLY.
    """
    fa = links[links["parent_sex"] == "male"][
        ["parent_person_id", "child_person_id", "birth_year"]
    ]
    mo = links[links["parent_sex"] == "female"][
        ["parent_person_id", "child_person_id", "birth_year"]
    ]
    marital_by_year = hcs3._father_marital_by_year(mpanel)

    # ---- Observable basis (father-wave denominator; c3/c4's fit). ----
    obs_cust, obs_diag = hcs3.fit_custodial_coresidence(
        hh.person_waves, fa, parent_pairs, marital_by_year, ids_b
    )

    # ---- Child-record basis (child-wave denominator). ----
    fa_b = fa[fa["parent_person_id"].isin(ids_b)]
    child_ids_b = set(int(x) for x in fa_b["child_person_id"].unique())
    # Each joinable child's enumerated FU waves (child appears as ego in the
    # relationship matrix) -- "which family unit the child resides in".
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
    enum = enum[enum["child_person_id"].isin(child_ids_b)].copy()
    # One father + birth-year per child (biological father link).
    fa_child = fa_b.drop_duplicates("child_person_id").set_index(
        "child_person_id"
    )
    enum["birth_year"] = enum["child_person_id"].map(fa_child["birth_year"])
    enum["father_id"] = enum["child_person_id"].map(
        fa_child["parent_person_id"]
    )
    enum["child_age"] = enum["year"] - enum["birth_year"]
    enum = enum[
        (enum["child_age"] >= 0)
        & (enum["child_age"] <= CHILD_CORESIDENCE_MAX_AGE)
    ].copy()
    enum["child_band"] = enum["child_age"].map(hcs3._child_band)
    enum = enum[enum["child_band"].notna()]
    # The child's OWN wave weight (the child-record weight).
    demo_j = demo[["person_id", "period", "weight"]].rename(
        columns={"person_id": "child_person_id", "period": "year"}
    )
    enum = enum.merge(demo_j, on=["child_person_id", "year"], how="left")
    enum = enum[enum["weight"].fillna(0) > 0].copy()
    # Father coresident that wave (the (father, child, year) pair in the roster).
    pp_fa = parent_pairs.rename(
        columns={"parent_person_id": "father_id"}
    ).assign(_wf=True)
    enum = enum.merge(
        pp_fa, on=["father_id", "child_person_id", "year"], how="left"
    )
    enum["with_father"] = enum["_wf"].fillna(False).astype(bool)
    enum = enum.drop(columns="_wf")
    # Mother coresident that wave.
    mo_child = mo.drop_duplicates("child_person_id").set_index(
        "child_person_id"
    )["parent_person_id"]
    enum["mother_id"] = enum["child_person_id"].map(mo_child)
    pp_mo = parent_pairs.rename(
        columns={"parent_person_id": "mother_id"}
    ).assign(_wm=True)
    enum = enum.merge(
        pp_mo, on=["mother_id", "child_person_id", "year"], how="left"
    )
    enum["with_mother"] = enum["_wm"].fillna(False).astype(bool)
    enum = enum.drop(columns="_wm")
    # Father marital state that wave (observed); uncovered -> not_married.
    enum = enum.merge(
        marital_by_year.rename(columns={"person_id": "father_id"}),
        on=["father_id", "year"],
        how="left",
    )
    enum["marital"] = enum["marital"].fillna(_NOT_MARRIED)

    w = enum["weight"].to_numpy(dtype=np.float64)
    wf = enum["with_father"].to_numpy(dtype=bool)
    wm = enum["with_mother"].to_numpy(dtype=bool)

    # ---- Selection-gap table by child age band x father marital state. ----
    gap_table: dict[str, Any] = {}
    for lo, hi in CUSTODIAL_CHILD_AGE_BANDS:
        band = hc.band_label(lo, hi)
        for marital in (_MARRIED, _NOT_MARRIED):
            key = f"{band}|{marital}"
            m = (enum["child_band"] == band) & (enum["marital"] == marital)
            m = m.to_numpy()
            child_rec = _wshare(w[m], wf[m]) if m.any() else 0.0
            obs = float(obs_cust.get((band, marital), 0.0))
            gap_table[key] = {
                "observable_basis_p_coresident": obs,
                "child_record_basis_p_coresident": child_rec,
                "selection_gap_observable_minus_child_record": obs - child_rec,
                "child_record_n_child_waves": int(m.sum()),
                "child_record_weight": float(w[m].sum()),
            }

    # ---- Maternal-complement four-way partition (sums to 1 EXACTLY). ----
    def _partition(mask: np.ndarray) -> dict[str, float]:
        ww, f, mo_ = w[mask], wf[mask], wm[mask]
        both = _wshare(ww, f & mo_)
        f_only = _wshare(ww, f & ~mo_)
        m_only = _wshare(ww, ~f & mo_)
        neither = _wshare(ww, ~f & ~mo_)
        p_father = both + f_only
        p_mother = both + m_only
        return {
            "with_both": both,
            "with_father_only": f_only,
            "with_mother_only": m_only,
            "with_neither": neither,
            "p_with_father": p_father,
            "p_with_mother": p_mother,
            "partition_sum": both + f_only + m_only + neither,
            "partition_reconciliation_remainder": (
                both + f_only + m_only + neither - 1.0
            ),
            "father_complement_identity_remainder": (
                p_father + p_mother - both + neither - 1.0
            ),
            "n_child_waves": int(mask.sum()),
        }

    maternal_overall = _partition(np.ones(len(enum), dtype=bool))
    maternal_by_band: dict[str, Any] = {}
    for lo, hi in CUSTODIAL_CHILD_AGE_BANDS:
        band = hc.band_label(lo, hi)
        maternal_by_band[band] = _partition(
            (enum["child_band"] == band).to_numpy()
        )

    return {
        "observable_overall_diag": obs_diag,
        "observable_p_coresident_by_band_marital": {
            f"{b}|{m}": float(v) for (b, m), v in sorted(obs_cust.items())
        },
        "selection_gap_table": gap_table,
        "child_record_n_child_waves_total": int(len(enum)),
        "maternal_complement_overall": maternal_overall,
        "maternal_complement_by_child_band": maternal_by_band,
        "n_train_father_links": int(len(fa_b)),
        "n_joinable_train_children": int(len(child_ids_b)),
    }


# --------------------------------------------------------------------------
# Q6: grandchild reference channels (deterministic reference decomposition)
# --------------------------------------------------------------------------
def q6_reference_channels(
    hh: hc.HouseholdCompositionPanel,
    rel_map: pd.DataFrame,
    ids_b: set[int],
) -> dict[str, Any]:
    """Decompose the reference 55+F ``coresident_grandchild`` stock by channel.

    Over the side-B roster, every 55+ female person-wave flagged
    ``coresident_grandchild`` is classified into mutually-exclusive channels by
    the ego's raw MX8 links: (a) ego-on-top three-generation with the composed
    test PASSING (multigen AND own-child-present AND NOT own-parent-present),
    (a') ego-on-top with the composed test BREAKING (own-parent present -- 4th
    generation -- or only a child-IN-LAW middle generation), (b)
    skipped-generation (no middle generation present at all), (c) other. The
    channels sum to the reference stock EXACTLY. The grandparent-code inventory
    (weighted share of each MX8 code) is enumerated alongside.
    """
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)]
    older_f = pw_b[(pw_b["sex"] == "female") & (pw_b["age"] >= GRANDCHILD_LO)]
    den = float(older_f["weight"].sum())
    g = older_f[older_f["coresident_grandchild"]][
        ["person_id", "year", "weight", "multigen"]
    ].copy()
    stock = float(g["weight"].sum()) / den if den > 0 else 0.0

    # Reference component rates among ALL 55+|female (the composed test's
    # inputs). multigen AND coresident_child AND NOT coresident_parent is the
    # composed grandchild implication; comparing its rate to the product of the
    # marginals shows whether the reference COUPLES multigen and coresident
    # child (the same household fact) or treats them as independent.
    wof = older_f["weight"].to_numpy(dtype=np.float64)
    ref_mg = older_f["multigen"].to_numpy(dtype=bool)
    ref_ch = older_f["coresident_child"].to_numpy(dtype=bool)
    ref_np = ~older_f["coresident_parent"].to_numpy(dtype=bool)
    ref_component_rates = {
        "multigen": _wshare(wof, ref_mg),
        "coresident_child": _wshare(wof, ref_ch),
        "not_coresident_parent": _wshare(wof, ref_np),
        "multigen_and_child_and_notparent": _wshare(
            wof, ref_mg & ref_ch & ref_np
        ),
        "independence_product_multigen_x_child_x_notparent": (
            _wshare(wof, ref_mg) * _wshare(wof, ref_ch) * _wshare(wof, ref_np)
        ),
    }

    # Ego's non-self MX8 link codes per (person, year).
    ego_ids = set(int(x) for x in g["person_id"].unique())
    rm = rel_map[
        (rel_map["ego_person_id"].isin(ego_ids))
        & (rel_map["ego_rel_to_alter"] != relmap.SELF)
    ].rename(columns={"interview_year": "year", "ego_person_id": "person_id"})
    codes = (
        rm.groupby(["person_id", "year"])["ego_rel_to_alter"]
        .agg(lambda s: frozenset(int(x) for x in s))
        .rename("codes")
        .reset_index()
    )
    g = g.merge(codes, on=["person_id", "year"], how="left")
    g["codes"] = g["codes"].apply(
        lambda s: s if isinstance(s, frozenset) else frozenset()
    )

    def _has(s: frozenset, codeset) -> bool:
        return len(s & set(codeset)) > 0

    has_own_child = g["codes"].apply(lambda c: _has(c, CHILD_LINK))
    has_child_inlaw = g["codes"].apply(lambda c: _has(c, CHILD_INLAW_LINK))
    has_own_parent = g["codes"].apply(lambda c: _has(c, PARENT_LINK))
    multigen = g["multigen"].to_numpy(dtype=bool)
    w = g["weight"].to_numpy(dtype=np.float64)

    composed_test = (
        multigen & has_own_child.to_numpy() & ~has_own_parent.to_numpy()
    )
    mid_present = (has_own_child | has_child_inlaw).to_numpy()
    ch_a = composed_test
    ch_abreak = mid_present & ~composed_test
    ch_skip = ~mid_present
    ch_other = ~ch_a & ~ch_abreak & ~ch_skip

    def _share(mask: np.ndarray) -> float:
        return float(w[mask].sum() / den) if den > 0 else 0.0

    # Sub-splits of the ego-on-top-breaks channel.
    inlaw_only_mid = (
        ~has_own_child & has_child_inlaw
    ).to_numpy() & ~composed_test
    four_gen = (has_own_child & has_own_parent).to_numpy()

    # Grandparent-code weighted share of the stock (of the 55+F denominator).
    code_w: Counter = Counter()
    for codeset, wt in zip(g["codes"], w, strict=True):
        for code in codeset & set(GRANDPARENT_CODES_ALL):
            code_w[code] += wt
    code_share = {
        str(code): {
            "label": relmap.EGO_REL_TO_ALTER[code],
            "weighted_share_of_55plus_female": (
                float(code_w[code] / den) if den > 0 else 0.0
            ),
            "in_reference_link": bool(code in GRANDPARENT_LINK_REF),
        }
        for code in GRANDPARENT_CODES_ALL
        if code in code_w
    }

    channels = {
        "a_ego_on_top_composed_reachable": _share(ch_a),
        "a_prime_ego_on_top_composed_breaks": _share(ch_abreak),
        "b_skipped_generation": _share(ch_skip),
        "c_other": _share(ch_other),
    }
    return {
        "reference_stock_55plus_female": stock,
        "n_grandchild_person_waves": int(len(g)),
        "reference_component_rates_55plus_female": ref_component_rates,
        "channels": channels,
        "channel_reconciliation_remainder": stock - sum(channels.values()),
        "a_prime_breaks_detail": {
            "child_in_law_middle_generation_only": _share(inlaw_only_mid),
            "four_generation_own_parent_present": _share(four_gen),
        },
        "grandparent_code_inventory": code_share,
        "codebook_note": (
            "MX8 (ego->alter) grandparent-type codes verified against "
            "MX23REL_formats.sps (lines 100-125): 66 grandparent, 67 "
            "grandparent-in-law, 68 great-grandparent, 69 "
            "great-grandparent-in-law, 82 social grandparent, 83 social "
            "great-grandparent, 87 step grandparent, 88 step great "
            "grandparent. The reference coresident_grandchild link is {66, 68, "
            "82, 87, 88} -- own/social/step grandparent + own/step "
            "great-grandparent -- EXCLUDING the in-law codes {67, 69} and (an "
            "asymmetry) the social great-grandparent 83. MX8 code 88 is step "
            "great-grandparent here (a grandparent-type, correctly included), "
            "NOT the MX7 first-year-cohabitor meaning (generation 0) -- the 2b "
            "ceremony's code-88 lesson: the frames must not cross."
        ),
    }


# --------------------------------------------------------------------------
# Q7: hh_size.3 family-core routes (deterministic reference decomposition)
# --------------------------------------------------------------------------
def _route_of(
    has_spouse: np.ndarray, n_child: np.ndarray, n_parent: np.ndarray
) -> dict[str, np.ndarray]:
    """Mutually-exclusive size-3 composition-route masks (partition the input).

    Applied identically to the simulated family core (spouse / child_counts /
    n_parents) and the raw reference roster (ego's spouse / own-child /
    own-parent link counts). ``other_family_core`` carries every remaining
    configuration (for the reference: households with a non-core member -- a
    sibling, roomer, in-law -- so the ego's core links sum below 2).
    """
    has_spouse = np.asarray(has_spouse, dtype=bool)
    n_child = np.asarray(n_child, dtype=np.int64)
    n_parent = np.asarray(n_parent, dtype=np.int64)
    couple_child = has_spouse & (n_child >= 1)
    single_two = ~has_spouse & (n_child >= 2)
    couple_parent = has_spouse & (n_child == 0) & (n_parent >= 1)
    three_adults = ~has_spouse & (n_child == 0) & (n_parent >= 1)
    routed = couple_child | single_two | couple_parent | three_adults
    other = ~routed
    return {
        "couple_plus_child": couple_child,
        "single_parent_plus_two_children": single_two,
        "couple_plus_parent": couple_parent,
        "three_adults": three_adults,
        "other_family_core": other,
    }


def q7_reference_routes(
    hh: hc.HouseholdCompositionPanel,
    rel_map: pd.DataFrame,
    ids_b: set[int],
) -> dict[str, Any]:
    """Decompose the reference size-3 stock (actual and core) by route.

    ``actual`` = the reference ``hh_size == 3`` person-waves (the gated
    reference 0.181), classified by the ego's raw roster link counts.
    ``core`` = person-waves whose ego family CORE (1 + spouse + own-children +
    own-parents) equals 3 -- the apples-to-apples construct the sim's
    family-core-only size-3 (0.271) measures. Comparing the two isolates the
    non-core-member households (real size 4+ with a core of 3, or size 3 with a
    non-core member) from the sim's core composition error.
    """
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)]
    den = float(pw_b["weight"].sum())

    # Ego's core link counts per (person, year) over the whole train roster.
    rm = rel_map[
        (rel_map["ego_person_id"].isin(set(pw_b["person_id"].unique())))
        & (rel_map["ego_rel_to_alter"] != relmap.SELF)
    ].rename(columns={"interview_year": "year", "ego_person_id": "person_id"})
    rm = rm.assign(
        is_sp=rm["ego_rel_to_alter"].isin(
            hc.CORESIDENCE_LINKS["coresident_spouse"]
        ),
        is_ch=rm["ego_rel_to_alter"].isin(CHILD_LINK),
        is_pa=rm["ego_rel_to_alter"].isin(PARENT_LINK),
    )
    agg = (
        rm.groupby(["person_id", "year"])
        .agg(
            n_spouse=("is_sp", "sum"),
            n_child=("is_ch", "sum"),
            n_parent=("is_pa", "sum"),
        )
        .reset_index()
    )
    pw = pw_b[["person_id", "year", "weight", "hh_size"]].merge(
        agg, on=["person_id", "year"], how="left"
    )
    for col in ("n_spouse", "n_child", "n_parent"):
        pw[col] = pw[col].fillna(0).astype("int64")
    pw["core_size"] = (
        1
        + (pw["n_spouse"] > 0).astype("int64")
        + pw["n_child"]
        + pw["n_parent"]
    )
    has_spouse = (pw["n_spouse"] > 0).to_numpy()
    n_child = pw["n_child"].to_numpy()
    n_parent = pw["n_parent"].to_numpy()
    w = pw["weight"].to_numpy(dtype=np.float64)
    masks = _route_of(has_spouse, n_child, n_parent)

    def _routes(sel: np.ndarray) -> dict[str, float]:
        out = {
            name: float(w[m & sel].sum() / den) if den > 0 else 0.0
            for name, m in masks.items()
        }
        return out

    actual_sel = (pw["hh_size"] == 3).to_numpy()
    core_sel = (pw["core_size"] == 3).to_numpy()
    actual_routes = _routes(actual_sel)
    core_routes = _routes(core_sel)
    actual_total = float(w[actual_sel].sum() / den) if den > 0 else 0.0
    core_total = float(w[core_sel].sum() / den) if den > 0 else 0.0
    return {
        "reference_actual_size3_total": actual_total,
        "reference_actual_size3_routes": actual_routes,
        "reference_actual_reconciliation_remainder": (
            actual_total - sum(actual_routes.values())
        ),
        "reference_core_size3_total": core_total,
        "reference_core_size3_routes": core_routes,
        "reference_core_reconciliation_remainder": (
            core_total - sum(core_routes.values())
        ),
    }


# --------------------------------------------------------------------------
# Per-seed computation (fit on side B, 20 train-side instrumented draws)
# --------------------------------------------------------------------------
def compute_seed(
    seed: int,
    data: dict[str, Any],
    links: pd.DataFrame,
    verbose: bool,
) -> dict[str, Any]:
    """Fit candidate 4 on side B, run the deterministic + instrumented pieces."""
    t0 = time.time()
    hh = data["hh"]
    mpanel = data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs4.fit_household_model_v4(
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
    )
    rate_b = hc.reference_moments(hh, ids_b, weighted=True)

    # ---- Deterministic reference decompositions (no sim RNG). ----
    q5 = q5_custodial_selection(
        hh,
        mpanel,
        data["demo"],
        data["rel_map"],
        data["parent_pairs"],
        links,
        ids_b,
    )
    q6_ref = q6_reference_channels(hh, data["rel_map"], ids_b)
    q7_ref = q7_reference_routes(hh, data["rel_map"], ids_b)

    # ---- 20 instrumented train-side draws: sim-side supply/routes. ----
    parent_count = int(model.base.parent_count)
    # Q6 sim supply among 55+ female.
    q6_composed: list[float] = []
    q6_skiponly: list[float] = []
    q6_union: list[float] = []
    q6_multigen: list[float] = []
    q6_child: list[float] = []
    q6_notparent: list[float] = []
    q6_multigen_child_notparent: list[float] = []
    # Q7 sim family-core size-3 routes + driver inputs.
    q7_core_total: list[float] = []
    q7_routes: dict[str, list[float]] = {k: [] for k in Q7_ROUTES}
    q7_p_spouse: list[float] = []
    # child-count distribution among coupled egos (the couple+child driver).
    q7_child_dist_couples: dict[str, list[float]] = {
        "0": [],
        "1": [],
        "2": [],
        "3+": [],
    }
    q7_full_size3: list[float] = []

    for k in range(N_DRAWS):
        draw_seed = DRAW_SEED_BASE + k
        d = instrumented_draw_v4(hh, mpanel, model, ids_b, draw_seed)
        age, sex, weight = d["age"], d["sex"], d["weight"]
        older_f = (sex == "female") & (age >= GRANDCHILD_LO)
        wf = weight[older_f]
        composed = d["grandchild_composed"][older_f]
        skiponly = (d["skipgen_row"] & ~d["grandchild_composed"])[older_f]
        union = d["coresident_grandchild"][older_f]
        mg = d["multigen"][older_f]
        ch = d["coresident_child"][older_f]
        npar = ~d["coresident_parent"][older_f]
        q6_composed.append(_wshare(wf, composed))
        q6_skiponly.append(_wshare(wf, skiponly))
        q6_union.append(_wshare(wf, union))
        q6_multigen.append(_wshare(wf, mg))
        q6_child.append(_wshare(wf, ch))
        q6_notparent.append(_wshare(wf, npar))
        q6_multigen_child_notparent.append(_wshare(wf, mg & ch & npar))

        # Q7 sim family-core routes on hh_size_base == 3.
        base3 = d["hh_size_base"] == 3
        wtot = float(weight.sum())
        q7_core_total.append(float(weight[base3].sum() / wtot))
        q7_full_size3.append(float(weight[d["hh_size"] == 3].sum() / wtot))
        has_sp = d["spouse"]
        masks = _route_of(has_sp, d["child_counts"], d["n_parents"])
        for name, m in masks.items():
            q7_routes[name].append(float(weight[m & base3].sum() / wtot))
        q7_p_spouse.append(_wshare(weight, has_sp))
        # child-count distribution among coupled egos (spouse present).
        wc = weight[has_sp]
        cc = d["child_counts"][has_sp]
        for label, mask in (
            ("0", cc == 0),
            ("1", cc == 1),
            ("2", cc == 2),
            ("3+", cc >= 3),
        ):
            q7_child_dist_couples[label].append(_wshare(wc, mask))

    elapsed = round(time.time() - t0, 1)
    if verbose:
        print(
            f"seed {seed}: n_train={len(ids_b)} "
            f"gc55f union={np.mean(q6_union):.4f} "
            f"(ref {q6_ref['reference_stock_55plus_female']:.4f}) "
            f"core_size3={np.mean(q7_core_total):.4f} "
            f"(ref_actual {q7_ref['reference_actual_size3_total']:.4f}) "
            f"[{elapsed}s]"
        )

    return {
        "seed": seed,
        "n_train_persons": len(ids_b),
        "parent_count": parent_count,
        "rate_b_grandchild_55f": float(
            rate_b["coresident_grandchild.55+|female"]["rate"]
        ),
        "rate_b_hh_size3": float(rate_b["hh_size.3"]["rate"]),
        "q5": q5,
        "q6_reference": q6_ref,
        "q6_sim_composed_mean": float(np.mean(q6_composed)),
        "q6_sim_skiponly_mean": float(np.mean(q6_skiponly)),
        "q6_sim_union_mean": float(np.mean(q6_union)),
        "q6_sim_component_rates": {
            "multigen": float(np.mean(q6_multigen)),
            "coresident_child": float(np.mean(q6_child)),
            "not_coresident_parent": float(np.mean(q6_notparent)),
            "multigen_and_child_and_notparent": float(
                np.mean(q6_multigen_child_notparent)
            ),
        },
        "q7_reference": q7_ref,
        "q7_sim_core_size3_total_mean": float(np.mean(q7_core_total)),
        "q7_sim_full_size3_total_mean": float(np.mean(q7_full_size3)),
        "q7_sim_core_routes_mean": {
            k: float(np.mean(v)) for k, v in q7_routes.items()
        },
        "q7_sim_p_spouse_mean": float(np.mean(q7_p_spouse)),
        "q7_sim_child_dist_among_couples_mean": {
            k: float(np.mean(v)) for k, v in q7_child_dist_couples.items()
        },
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Committed candidate-4 reads (side A holdout: NEVER re-simulated)
# --------------------------------------------------------------------------
def _committed_c4() -> dict[str, Any]:
    return json.loads(CANDIDATE4_ARTIFACT.read_text())


def _committed_cell_c4(c4a: dict[str, Any], cell: str) -> dict[str, Any]:
    """Seed-averaged committed side-A gated-cell record from candidate 4."""
    recs = [s["gated_cells"][cell] for s in c4a["per_seed"]]
    return {
        "seed_mean_rate_a": float(np.mean([r["rate_a"] for r in recs])),
        "seed_mean_r_candidate": float(
            np.mean([r["r_candidate"] for r in recs])
        ),
        "seed_mean_score": float(np.mean([r["score"] for r in recs])),
        "tolerance": float(recs[0]["tolerance"]),
        "n_seeds_pass": int(sum(r["pass"] for r in recs)),
    }


# --------------------------------------------------------------------------
# Assemble the three question blocks
# --------------------------------------------------------------------------
def _seed_avg_nested(per_seed: list[dict[str, Any]], path: list) -> Any:
    """Seed-average a nested numeric leaf addressed by ``path`` of keys."""
    vals = []
    for s in per_seed:
        node: Any = s
        for k in path:
            node = node[k]
        vals.append(float(node))
    return _mean_over_seeds(vals)


def assemble_q5(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Q5 -- custodial selection basis: observable vs child-record + maternal."""
    bands = [hc.band_label(lo, hi) for lo, hi in CUSTODIAL_CHILD_AGE_BANDS]
    gap_keys = list(per_seed[0]["q5"]["selection_gap_table"].keys())
    gap_table = {
        key: {
            metric: _seed_avg_nested(
                per_seed, ["q5", "selection_gap_table", key, metric]
            )
            for metric in (
                "observable_basis_p_coresident",
                "child_record_basis_p_coresident",
                "selection_gap_observable_minus_child_record",
            )
        }
        | {
            "child_record_n_child_waves": int(
                np.mean(
                    [
                        s["q5"]["selection_gap_table"][key][
                            "child_record_n_child_waves"
                        ]
                        for s in per_seed
                    ]
                )
            )
        }
        for key in gap_keys
    }
    maternal_overall = {
        metric: _seed_avg_nested(
            per_seed, ["q5", "maternal_complement_overall", metric]
        )
        for metric in (
            "with_both",
            "with_father_only",
            "with_mother_only",
            "with_neither",
            "p_with_father",
            "p_with_mother",
            "partition_sum",
            "partition_reconciliation_remainder",
            "father_complement_identity_remainder",
        )
    }
    maternal_by_band = {
        band: {
            metric: _seed_avg_nested(
                per_seed,
                ["q5", "maternal_complement_by_child_band", band, metric],
            )
            for metric in (
                "with_both",
                "with_father_only",
                "with_mother_only",
                "with_neither",
                "p_with_father",
                "p_with_mother",
                "partition_reconciliation_remainder",
            )
        }
        for band in bands
    }

    # Headline framing from the seed-averaged table.
    young_married = [
        gap_table[f"{b}|{_MARRIED}"] for b in ("0-4", "5-12", "13-17")
    ]
    young_married_gap = float(
        np.mean(
            [
                g["selection_gap_observable_minus_child_record"]
                for g in young_married
            ]
        )
    )
    notm_school = [gap_table[f"{b}|{_NOT_MARRIED}"] for b in ("5-12", "13-17")]
    notm_school_gap = float(
        np.mean(
            [
                g["selection_gap_observable_minus_child_record"]
                for g in notm_school
            ]
        )
    )
    ov = maternal_overall
    finding = (
        "The selection gap is HETEROGENEOUS by father marital state -- the "
        "pre-registered 'materially lower child-record custody at young ages' "
        "holds ONLY for not-married fathers, and REVERSES for married fathers. "
        "On the less-selected child-record basis (denominator = the child's "
        "OWN enumerated waves, independent of the father-side observation "
        "window), young children of MARRIED fathers coreside with the father "
        f"at ~{np.mean([g['child_record_basis_p_coresident'] for g in young_married]):.3f} "
        f"vs the observable-basis ~{np.mean([g['observable_basis_p_coresident'] for g in young_married]):.3f} "
        f"(gap ~{young_married_gap:+.3f}: the child-record basis is if anything "
        "HIGHER -- an enumerated young child is almost always with a married "
        "father, and the father-wave observable is DILUTED by father-waves "
        "after the child has aged out or left). The observable custodial gate "
        "the c4 fit conditions on is therefore FAITHFUL at the young married "
        "ages the male coresident_child overshoot was hypothesized to live in "
        "-- confirming the c4 grading's honest negative: no re-basing of the "
        "young married custodial probability drains the overshoot. Where the "
        "bases DO diverge as pre-registered is the NOT-married school-age "
        f"cells: the child-record P(coresident) runs ~{notm_school_gap:+.3f} "
        "BELOW the observable at 5-17|not_married (the observable over-states "
        "coresidence for unmarried fathers -- the (father, child, wave) "
        "triples where an unmarried father IS observed with the linked child "
        "select upward). The maternal complement pins the truth: across all "
        "joinable child-waves the four-way partition [both "
        f"{ov['with_both']:.3f}, father-only {ov['with_father_only']:.3f}, "
        f"mother-only {ov['with_mother_only']:.3f}, neither "
        f"{ov['with_neither']:.3f}] sums to 1 to machine epsilon, giving "
        f"P(with father) ~{ov['p_with_father']:.3f} against P(with mother) "
        f"~{ov['p_with_mother']:.3f} -- a child not with the father is usually "
        "with the mother, and the mother-side rate is the directly observable "
        "constraint the father-side custodial fit must sit beneath. The c5 "
        "basis is the not-married custodial cells (and the maternal-custodial "
        "split), NOT the young married gate."
    )
    return {
        "question": (
            "custodial selection basis: construct the less-selected "
            "child-record custody measure (each sample child's own enumerated "
            "waves, independent of the father-side observation window); "
            "recompute P(child coresides with father | child age, father "
            "marital) on that basis; quantify the selection gap vs the "
            "(father, child, wave)-observable basis c3/c4 fitted; reconcile "
            "against the maternal-coresidence complement."
        ),
        "method": (
            "Train-side (side B). Observable basis = exactly "
            "hcs3.fit_custodial_coresidence (denominator = FATHER-waves; the "
            "triples the c3/c4 gate fits). Child-record basis = each joinable "
            "cah85_23 child's OWN enumerated relationship-matrix waves as the "
            "denominator (child age in [0, 60]); event = the (father, child, "
            "year) coresidence pair present in the roster; father marital = the "
            "father's observed state that wave. The maternal complement adds "
            "the (mother, child, year) coresidence and partitions each "
            "child-wave into {both, father-only, mother-only, neither}, which "
            "sums to 1 EXACTLY (weighted)."
        ),
        "selection_gap_table_by_band_marital": gap_table,
        "maternal_complement_overall": maternal_overall,
        "maternal_complement_by_child_band": maternal_by_band,
        "young_married_seed_mean_gap": young_married_gap,
        "not_married_school_age_seed_mean_gap": notm_school_gap,
        "n_joinable_train_children_seed_mean": int(
            np.mean([s["q5"]["n_joinable_train_children"] for s in per_seed])
        ),
        "finding": finding,
    }


def assemble_q6(
    per_seed: list[dict[str, Any]], committed: dict[str, Any]
) -> dict[str, Any]:
    """Q6 -- grandchild reference channels + the sim's reachable supply."""
    ref_stock = _seed_avg_nested(
        per_seed, ["q6_reference", "reference_stock_55plus_female"]
    )
    channels = {
        k: _seed_avg_nested(per_seed, ["q6_reference", "channels", k])
        for k in per_seed[0]["q6_reference"]["channels"]
    }
    breaks_detail = {
        k: _seed_avg_nested(
            per_seed, ["q6_reference", "a_prime_breaks_detail", k]
        )
        for k in per_seed[0]["q6_reference"]["a_prime_breaks_detail"]
    }
    # Code inventory: union of codes present, seed-averaged share.
    code_keys = sorted(
        set().union(
            *[
                set(s["q6_reference"]["grandparent_code_inventory"].keys())
                for s in per_seed
            ]
        ),
        key=int,
    )
    code_inventory = {}
    for code in code_keys:
        shares = [
            s["q6_reference"]["grandparent_code_inventory"][code][
                "weighted_share_of_55plus_female"
            ]
            for s in per_seed
            if code in s["q6_reference"]["grandparent_code_inventory"]
        ]
        any_seed = next(
            s
            for s in per_seed
            if code in s["q6_reference"]["grandparent_code_inventory"]
        )
        rec = any_seed["q6_reference"]["grandparent_code_inventory"][code]
        code_inventory[code] = {
            "label": rec["label"],
            "in_reference_link": rec["in_reference_link"],
            "weighted_share_of_55plus_female": _mean_over_seeds(shares),
        }

    sim_composed = _seed_avg_nested(per_seed, ["q6_sim_composed_mean"])
    sim_skiponly = _seed_avg_nested(per_seed, ["q6_sim_skiponly_mean"])
    sim_union = _seed_avg_nested(per_seed, ["q6_sim_union_mean"])
    comp_rates = {
        k: _seed_avg_nested(per_seed, ["q6_sim_component_rates", k])
        for k in per_seed[0]["q6_sim_component_rates"]
    }
    ref_comp = {
        k: _seed_avg_nested(
            per_seed,
            ["q6_reference", "reference_component_rates_55plus_female", k],
        )
        for k in per_seed[0]["q6_reference"][
            "reference_component_rates_55plus_female"
        ]
    }
    unreachable = ref_stock - sim_union
    committed_cell = _committed_cell_c4(
        committed, "coresident_grandchild.55+|female"
    )
    code66_rec = code_inventory.get("66", {})
    code66_share = code66_rec.get("weighted_share_of_55plus_female", 0.0)
    code66_pct = (code66_share / ref_stock * 100.0) if ref_stock else 0.0
    sim_indep = (
        comp_rates["multigen"]
        * comp_rates["coresident_child"]
        * comp_rates["not_coresident_parent"]
    )

    finding = (
        "The unreachable ~0.039 is carried by the ego-on-top three-generation "
        "channel (a), NOT the in-law middle-generation channel the "
        "pre-registration named -- and the break is on the SIMULATION side, "
        "not the reference coding. From the raw side-B roster the reference "
        f"55+|female coresident_grandchild stock ~{ref_stock:.4f} decomposes "
        f"almost entirely into channel (a) ego-on-top three-generation "
        f"~{channels['a_ego_on_top_composed_reachable']:.4f} (multigen AND "
        "own-child-present AND NOT own-parent, the exact composed test) and "
        f"channel (b) skipped-generation ~{channels['b_skipped_generation']:.4f}; "
        f"the ego-on-top-BREAKS channel (a') is only "
        f"~{channels['a_prime_ego_on_top_composed_breaks']:.4f} (of which "
        f"child-in-law-only middle generation "
        f"~{breaks_detail['child_in_law_middle_generation_only']:.4f} and "
        f"four-generation own-parent-present "
        f"~{breaks_detail['four_generation_own_parent_present']:.4f}) and "
        f"channel (c) other ~{channels['c_other']:.4f}. So the composed test "
        "is the RIGHT SHAPE for the reference mass -- most of the reference "
        "grandchildren ARE ego-on-top three-generation with the middle "
        "generation being ego's OWN child (the code inventory is ~"
        f"{code66_pct:.0f}% "
        "plain grandparent code 66; the in-law codes 67/69 are excluded and "
        "carry near-zero mass). The gap is that the SIMULATION cannot build "
        f"the joint: the composed path supplies only ~{sim_composed:.4f} of the "
        f"channel-(a) ~{channels['a_ego_on_top_composed_reachable']:.4f}. The "
        "mechanism is a DECOUPLING, not a level miss on any single component -- "
        f"the reference COUPLES multigen (~{ref_comp['multigen']:.3f} of "
        f"55+|female) and a coresident own child (~{ref_comp['coresident_child']:.3f}) "
        "into the SAME household fact, so their joint (multigen AND child AND "
        f"NOT-parent) is ~{ref_comp['multigen_and_child_and_notparent']:.4f}, "
        f"FAR above their independence product "
        f"~{ref_comp['independence_product_multigen_x_child_x_notparent']:.4f} "
        "(a multigen 55+ grandmother almost always has the middle-generation "
        "adult child present -- that IS why she is multigen). The simulation "
        f"draws multigen ~{comp_rates['multigen']:.3f} (the carried candidate-1 "
        f"hazard) and coresident_child ~{comp_rates['coresident_child']:.3f} "
        "(the aged-out custodial / maternal children) from SEPARATE components "
        f"that do not align: the sim joint ~{comp_rates['multigen_and_child_and_notparent']:.4f} "
        f"sits right at its independence product ~{sim_indep:.4f}. A 55+ "
        "woman's own adult children have aged out of the simulated frame, so "
        "even when she is simulated multigen she rarely also carries a "
        "coresident child, and the composed grandchild collapses. The skip-gen "
        f"delta supplies ~{sim_skiponly:.4f} (covering channel (b) fully), so "
        f"the sim union ~{sim_union:.4f} leaves ~{unreachable:.4f} unreachable "
        "-- and that unreachable mass is channel (a): the composed path's "
        "multigen-AND-coresident-child DECOUPLING at older maternal ages, not "
        "an in-law coding break (channel a' is ~"
        f"{channels['a_prime_ego_on_top_composed_breaks']:.4f}) and not a "
        "skip-gen level shortfall (channel b is covered)."
    )
    return {
        "question": (
            "grandchild reference channels: decompose the reference 55+|female "
            "coresident_grandchild stock (~0.063) by household-structure "
            "channel from the raw roster (ego-on-top three-generation; "
            "skipped-generation; other, incl. whether a child-IN-LAW middle "
            "generation breaks the composed multigen AND child AND NOT-parent "
            "test); the simulation's reachable supply per channel under the v4 "
            "machinery; identify which channel(s) carry the unreachable ~0.039."
        ),
        "method": (
            "Train-side (side B). Reference channels from the raw MX8 roster: "
            "every 55+|female coresident_grandchild person-wave classified by "
            "ego's own-child / child-in-law / own-parent links + the multigen "
            "flag (channels partition the stock EXACTLY). Sim supply from 20 "
            "instrumented candidate-4 draws (bit-identical to "
            "simulate_draw_v4): the composed grandchild (multigen AND child AND "
            "NOT-parent) and skip-gen occupancy rates among 55+|female, plus "
            "each composed component's simulated rate to localize the break."
        ),
        "reference_stock_55plus_female": ref_stock,
        "reference_channels": channels,
        "reference_channel_reconciliation_remainder": (
            ref_stock - sum(channels.values())
        ),
        "reference_component_rates_55plus_female": ref_comp,
        "a_prime_breaks_detail": breaks_detail,
        "grandparent_code_inventory": code_inventory,
        "simulation_reachable_supply": {
            "composed_multigen_path": sim_composed,
            "skipgen_only": sim_skiponly,
            "union": sim_union,
            "union_reconciliation_remainder": (
                sim_union - (sim_composed + sim_skiponly)
            ),
            "unreachable_reference_minus_sim_union": unreachable,
            "composed_component_rates_55plus_female": comp_rates,
            "composed_independence_product_55plus_female": sim_indep,
        },
        "channel_reachability": {
            "a_ego_on_top_composed_reachable": (
                "reachable in principle via the composed path, but the sim "
                f"supplies only ~{sim_composed:.4f} of ~"
                f"{channels['a_ego_on_top_composed_reachable']:.4f} -- the "
                "multigen-and-coresident-child decoupling at older maternal "
                "ages (aged-out own child) is the binding gap"
            ),
            "b_skipped_generation": (
                "reachable via the delta-4 skip-gen occupancy; sim supplies "
                f"~{sim_skiponly:.4f} (covers the channel)"
            ),
            "a_prime_ego_on_top_composed_breaks": "unreachable (near-zero mass)",
            "c_other": "unreachable (near-zero mass)",
        },
        "holdout_committed_candidate4": committed_cell,
        "finding": finding,
    }


def assemble_q7(
    per_seed: list[dict[str, Any]], committed: dict[str, Any]
) -> dict[str, Any]:
    """Q7 -- hh_size.3 family-core routes: the 0.271-vs-0.181 core gap driver."""
    ref_actual = _seed_avg_nested(
        per_seed, ["q7_reference", "reference_actual_size3_total"]
    )
    ref_core = _seed_avg_nested(
        per_seed, ["q7_reference", "reference_core_size3_total"]
    )
    sim_core = _seed_avg_nested(per_seed, ["q7_sim_core_size3_total_mean"])
    sim_full = _seed_avg_nested(per_seed, ["q7_sim_full_size3_total_mean"])
    ref_actual_routes = {
        k: _seed_avg_nested(
            per_seed, ["q7_reference", "reference_actual_size3_routes", k]
        )
        for k in Q7_ROUTES
    }
    ref_core_routes = {
        k: _seed_avg_nested(
            per_seed, ["q7_reference", "reference_core_size3_routes", k]
        )
        for k in Q7_ROUTES
    }
    sim_core_routes = {
        k: _seed_avg_nested(per_seed, ["q7_sim_core_routes_mean", k])
        for k in Q7_ROUTES
    }
    # Route-by-route sim-core minus reference-actual (the 0.271-vs-0.181 gap).
    route_gap_vs_actual = {
        k: sim_core_routes[k] - ref_actual_routes[k] for k in Q7_ROUTES
    }
    # Decompose the gap into composition (sim-core vs ref-core) + non-core
    # members (ref-core vs ref-actual).
    composition_gap = {
        k: sim_core_routes[k] - ref_core_routes[k] for k in Q7_ROUTES
    }
    noncore_gap = {
        k: ref_core_routes[k] - ref_actual_routes[k] for k in Q7_ROUTES
    }
    over = max(route_gap_vs_actual, key=route_gap_vs_actual.get)

    # Driver inputs for the over-produced route.
    p_spouse_sim = _seed_avg_nested(per_seed, ["q7_sim_p_spouse_mean"])
    child_dist = {
        k: _seed_avg_nested(
            per_seed, ["q7_sim_child_dist_among_couples_mean", k]
        )
        for k in ("0", "1", "2", "3+")
    }
    committed_cell = _committed_cell_c4(committed, "hh_size.3")
    parent_count = int(round(np.mean([s["parent_count"] for s in per_seed])))
    total_gap = sim_core - ref_actual
    comp_total = sim_core - ref_core
    noncore_total = ref_core - ref_actual
    over_label = over.replace("_", "+")
    over_sim = sim_core_routes[over]
    over_ref_core = ref_core_routes[over]
    over_ref_actual = ref_actual_routes[over]
    finding = (
        f"The over-produced size-3 core route is {over_label} -- in the "
        f"simulation this is ego + {parent_count} coresident parents (the "
        f"fitted parent_count is {parent_count}, so every coresident-parent "
        "ego adds exactly two parents), i.e. an adult child living with both "
        "parents. But the 0.271-vs-0.181 core gap is DOMINANTLY a "
        "non-core-member CONSTRUCT difference, not a couple-composition error, "
        "and it reconciles EXACTLY. The sim family-core-only size-3 "
        f"~{sim_core:.3f} against the reference ACTUAL size-3 ~{ref_actual:.3f} "
        f"is a ~{total_gap:.3f} gap; the reference CORE size-3 (ego core "
        "1+spouse+own-children+own-parents == 3, the apples-to-apples "
        f"construct) is ~{ref_core:.3f}, splitting the gap into a "
        f"non-core-member part ~{noncore_total:.3f} (ref-core minus ref-actual) "
        f"plus a composition part ~{comp_total:.3f} (sim-core minus ref-core), "
        f"summing to ~{total_gap:.3f} to machine epsilon. The non-core-member "
        "part DOMINATES: reference households whose family core is 3 but whose "
        "ACTUAL size is 4+ because a sibling / roomer / other relative is also "
        "present -- exactly the mass the non-family bridge must lift out of "
        f"size 3. It is most acute on the {over_label} route: ~{over_ref_core:.3f} "
        f"of egos have a three-adult CORE but only ~{over_ref_actual:.3f} are "
        "actual 3-person households (an adult child living with both parents "
        "very often ALSO has a coresident sibling, making the real household "
        f"4+). The composition part ~{comp_total:.3f} is smaller and spread "
        f"across routes ({over_label} ~{over_sim - over_ref_core:+.3f}, "
        "single-parent+2-children "
        f"~{sim_core_routes['single_parent_plus_two_children'] - ref_core_routes['single_parent_plus_two_children']:+.3f}, "
        f"couple+child ~{sim_core_routes['couple_plus_child'] - ref_core_routes['couple_plus_child']:+.3f}); "
        f"the parent_count={parent_count} choice forecloses the couple+parent "
        "route (sim ~0.000) and concentrates adult-children-with-parents at "
        "core-size-3 three-adults. On the registration's driver menu the "
        f"marital joint is NOT the culprit (simulated couple rate ~{p_spouse_sim:.3f}) "
        "and the couple+child core roughly matches the reference core; the "
        "coresident-child-count distribution among couples (0 "
        f"~{child_dist['0']:.3f}, 1 ~{child_dist['1']:.3f}, 2 "
        f"~{child_dist['2']:.3f}, 3+ ~{child_dist['3+']:.3f}) is consistent "
        "with the reference core. The dominant upstream lever is the "
        "non-family bridge's REACH (lift more three-adult and family cores out "
        "of size 3), with a secondary child-aging-out-timing effect on the "
        "three-adults route (adult children retained in the parental home). "
        "Recorded descriptively per the registration (couple+child prior, no "
        "strong mechanism prior -- the prior route is NOT the over-produced "
        "one)."
    )
    return {
        "question": (
            "hh_size.3 family-core routes: decompose size-3 households "
            "sim-vs-reference by composition route (couple+child / "
            "single-parent+2-children / couple+parent / three-adults / other); "
            "for the over-produced route(s) identify the upstream driver "
            "(child aging-out timing / fertility level / marital-state joint) "
            "with a quantified reconciliation of the 0.271-vs-0.181 core gap."
        ),
        "method": (
            "Train-side (side B). Sim family-core routes from 20 instrumented "
            "candidate-4 draws: hh_size_base (1 + spouse + child_counts + "
            "n_parents) == 3 partitioned by (spouse, child count, parent count) "
            "into the five routes. Reference routes from the raw roster: the "
            "ego's own spouse/child/parent link counts, evaluated on both the "
            "ACTUAL hh_size == 3 stock (the gated 0.181) and the CORE-size == 3 "
            "construct (apples-to-apples with the sim core). Routes partition "
            "each total EXACTLY; the gap splits into a composition part "
            "(sim-core - ref-core) and a non-core-member part (ref-core - "
            "ref-actual)."
        ),
        "sim_family_core_size3_total": sim_core,
        "sim_full_size3_total": sim_full,
        "reference_actual_size3_total": ref_actual,
        "reference_core_size3_total": ref_core,
        "sim_core_routes": sim_core_routes,
        "reference_actual_routes": ref_actual_routes,
        "reference_core_routes": ref_core_routes,
        "route_gap_sim_core_minus_reference_actual": route_gap_vs_actual,
        "gap_decomposition": {
            "total_gap_sim_core_minus_ref_actual": total_gap,
            "composition_gap_sim_core_minus_ref_core": composition_gap,
            "noncore_member_gap_ref_core_minus_ref_actual": noncore_gap,
            "composition_gap_total": sim_core - ref_core,
            "noncore_member_gap_total": ref_core - ref_actual,
            "reconciliation_remainder": (
                total_gap - ((sim_core - ref_core) + (ref_core - ref_actual))
            ),
        },
        "over_produced_route": over,
        "driver_inputs": {
            "sim_p_spouse": p_spouse_sim,
            "sim_child_count_distribution_among_couples": child_dist,
        },
        "holdout_committed_candidate4": committed_cell,
        "finding": finding,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _revision_pins() -> dict[str, Any]:
    import sklearn

    return {
        "populace_dynamics_head_sha": c4._git_sha(),
        "merge_base_origin_master": c4._merge_base(),
        "candidate4_artifact": "runs/gate2b_hazard_v4.json",
        "candidate4_artifact_sha256": c4._sha_of_file(CANDIDATE4_ARTIFACT),
        "candidate4_runner": "scripts/run_gate2b_candidate4.py",
        "candidate4_runner_sha256": c4._sha_of_file(
            ROOT / "scripts" / "run_gate2b_candidate4.py"
        ),
        "forensics1_artifact": "runs/gate2b_forensics1_v1.json",
        "forensics1_artifact_sha256": c4._sha_of_file(FORENSICS1_ARTIFACT),
        "gate2b_floor_run": "runs/gate2b_floors_v1.json",
        "gate2b_floor_sha256": c4._sha_of_file(FLOOR_RUN),
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

    for art in (CANDIDATE4_ARTIFACT, FORENSICS1_ARTIFACT, FLOOR_RUN):
        if not art.exists():
            raise RuntimeError(f"required committed artifact missing: {art}")
    committed = _committed_c4()

    data = c4.load_all()
    links = _all_parent_links(data["bh"])
    if verbose:
        hh = data["hh"]
        print(
            f"panel: {len(hh.person_waves)} person-waves, "
            f"{hh.person_waves.person_id.nunique()} persons; "
            f"train-side forensics, K={N_DRAWS} draws (5200 + k)"
        )

    # One-time fidelity proof: instrumented draw == committed simulate_draw_v4.
    _side_a, side_b = hpanel.split_panel_by_person(
        data["hh"].attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b0 = set(int(x) for x in side_b.person_id.unique())
    model0 = hcs4.fit_household_model_v4(
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
    )
    fidelity = fidelity_check_v4(
        data["hh"], data["mpanel"], model0, ids_b0, DRAW_SEED_BASE
    )
    if verbose:
        print(
            "instrumentation fidelity: bit_identical="
            f"{fidelity['bit_identical']} (max dev "
            f"{fidelity['max_abs_rate_deviation_vs_committed_simulate_draw_v4']:.2e})"
        )
    if not fidelity["bit_identical"]:
        raise RuntimeError(
            "instrumented_draw_v4 does not reproduce simulate_draw_v4 "
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
        result = compute_seed(seed, data, links, verbose)
        local_cache[key] = json.loads(
            json.dumps(result, default=_json_default)
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(local_cache))
        per_seed.append(local_cache[key])

    q5 = assemble_q5(per_seed)
    q6 = assemble_q6(per_seed, committed)
    q7 = assemble_q7(per_seed, committed)

    implications = (
        "Implications for candidate 5 (labeled implications, NOT decisions -- "
        "the orchestrator registers c5). Q5: the male coresident_child "
        "overshoot is NOT drainable by re-basing the young-MARRIED custodial "
        "probability -- both bases agree it is ~0.95-0.99 faithful there; the "
        "c5 lever is the NOT-married custodial cells (the observable over-states "
        "coresidence for unmarried fathers by ~10 points against the "
        "child-record basis) and the maternal-custodial split the complement "
        "makes observable (P(mother) is the ceiling P(father) must sit "
        "beneath). Q6: the grandchild 55+|female residual is a DECOUPLING in "
        "the composed path, not the in-law channel or the skip-gen level -- "
        "most reference mass is ego-on-top three-generation with ego's own "
        "child as the middle generation, and the reference couples multigen "
        "and coresident-child into one household fact, but the sim draws them "
        "from separate components (carried multigen hazard vs aged-out "
        "custodial/maternal children) so their joint collapses to the "
        "independence product; a c5 fix must COUPLE the simulated multigen "
        "state to a retained coresident adult child at older maternal ages "
        "(a coresident-child life-cycle extension) OR relax the composed "
        "grandchild's own-child requirement, not raise the skip-gen hazard "
        "further. Q7: the hh_size.3 core overshoot is dominantly the "
        "non-core-member construct (reference size-3 excludes households whose "
        "core is 3 but carry a sibling/roomer, which the non-family bridge "
        "must lift to size 4+); the over-produced route is three-adults (ego + "
        "two coresident parents, forced by parent_count=2), so c5 is a "
        "bridge-REACH fix plus a look at the parent_count=2 assumption and "
        "adult-child retention, not a fertility or marital-joint change. Each "
        "maps to a concept/selection lever rather than a hazard re-fit."
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
            "gate-2b candidate 4 (PR #138, registration 4941160621); the three "
            "concept/selection-class residuals grading 4942004647 isolated: "
            "(1) custodial estimation basis (selection-corrected), (2) the "
            "grandchild reference-channel decomposition, (3) the hh_size.3 "
            "family-core route."
        ),
        "candidate4_artifact": "runs/gate2b_hazard_v4.json",
        "forensics1_artifact": "runs/gate2b_forensics1_v1.json",
        "protocol": {
            "train_side_only": True,
            "outer_holdout_contact": (
                "none beyond the already-published per-seed scores read from "
                "runs/gate2b_hazard_v4.json; the holdout (side A) is NEVER "
                "re-simulated here"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "hh.attrs, 'person_id', fraction=0.5, seed=s); side A = outer "
                "holdout (read from the committed candidate-4 artifact only), "
                "side B = train complement (this diagnostic fits AND simulates "
                "side B)"
            ),
            "fit_simulate_machinery": (
                "populace_dynamics.models.household_composition_sim_v4 "
                "(candidate 4, merged #138), reused byte-for-byte on the train "
                "side via instrumented_draw_v4 (same 0xB2B / 0xC2 / 0xC3 / 0xC4 "
                "draw streams and train-fitted tables)"
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
            "floor_run_sha256": c4._sha_of_file(FLOOR_RUN),
        },
        "question_5_custodial_selection_basis": q5,
        "question_6_grandchild_reference_channels": q6,
        "question_7_hh_size3_family_core_routes": q7,
        "per_seed": per_seed,
        "candidate_5_implications": implications,
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
        for q in (q5, q6, q7):
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
    artifacts.write_new(Path(args.out), c4._json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
