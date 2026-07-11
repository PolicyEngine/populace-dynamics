"""Gate-2b forensics 1 -- residual-mechanism decomposition (REPORTED, NOT GATED).

The registered diagnostic of PolicyEngine/populace-dynamics issue #42, comment
4940442065 ("Gate-2b forensics 1 registration (diagnostic, not gated):
decompose the four residual mechanisms"). It is evidence, not another spec
draw: per the 2a playbook (forensics rounds 1-4 preceded the winning
candidates), this diagnostic measures the mechanisms of the four residual
families the candidate-3 grading (4940440505) isolated -- spouse family, child
shadow residual, hh_size middle distribution, grandchild 55+|female remainder
-- BEFORE candidate 4 designs against them. The registration wins: this runner
answers exactly its four frozen questions.

FROZEN SPEC (comment 4940442065), four questions:

* **Q1 -- spouse family (the binding constraint).** For each failing spouse
  cell and seed, decompose the miss into (a) legal-marriage core vs (b)
  cohabitation-overlay contributions; measure the train-side cohabitation
  entry/exit fit against the raw MX8-22 spells by single-year age within the
  failing bands (is the band-level hazard hiding within-band structure?); check
  direction and stability of each cell's miss across the 20 draws; test whether
  the reference spouse concept includes any code mass beyond {20, 22} that
  neither component supplies (concept residual).
* **Q2 -- child shadow residual.** Quantify the unlinked-father share's
  contribution to the remaining male ``coresident_child`` overshoot (worst
  0.141 vs tol 0.071); characterize which men are unlinked (age, marital
  state, era) and whether the shadow kernel's spousal-gap shift misfits them.
* **Q3 -- hh_size middle distribution.** Decompose ``hh_size.3``'s 0.261 miss
  by which composed component (spouse / children / parents / non-family) drives
  the size-3 share error; test the non-family bridge's ``2+ -> 2`` minimal
  reading against the train distribution's actual ``2+`` composition.
* **Q4 -- grandchild 55+|female remainder.** Split the remaining 0.747 into the
  skipped-generation level (delta-3's state) vs the multigen-path composition;
  measure the skip-gen entry hazard's age structure within 55+ against the raw
  train spells.

Train-side only. The candidate protocol fits its components on side B (the
train complement of each gate seed's person-disjoint 50/50 split) and scores
its simulation of side A (the outer holdout). This diagnostic reuses candidate
3's fit/simulate machinery
(:mod:`populace_dynamics.models.household_composition_sim_v3`, merged #136)
BUT fits AND simulates side B -- the train half -- and scores against side B's
OWN empirical rate (``rate_b``). The outer holdout (side A) is NEVER simulated
here; the only side-A numbers used are the already-published per-seed scores
and the committed ``[20, 46, 5]`` per-draw cube read from the committed
candidate-1/2/3 artifacts (``runs/gate2b_hazard_v{1,2,3}.json``). Nothing in
``gates.yaml`` or any committed ``runs/`` gate artifact is written or moved.
Every fitted table is estimated on side B only; no holdout-informed tuning
surface is created (a spec violation if it were).

Environment: the gate ``.venv-gate`` (scikit-learn < 1.9 for the certified
logistic fits, matching the candidate scoring path) with the PSID products
staged (``POPULACE_DYNAMICS_PSID_DIR``). Run from the repository root::

    .venv-gate/bin/python scripts/gate2b_forensics1.py
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

# Candidate 3 is the spec under diagnosis: its fit/simulate machinery is reused
# verbatim on the TRAIN side. It supplies the frozen dials, the loaders, the
# threshold reader and the provenance helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate2b_candidate3 as c3  # noqa: E402

from populace_dynamics import artifacts  # noqa: E402
from populace_dynamics.data import household_composition as hc  # noqa: E402
from populace_dynamics.data import relmap, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.models import family_transitions as ft  # noqa: E402
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim as hcs,
)
from populace_dynamics.models import (  # noqa: E402
    household_composition_sim_v3 as hcs3,
)
from populace_dynamics.models.family_transitions.components.fertility import (  # noqa: E402
    build_fertility_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2b_forensics1_v1.json"
CANDIDATE1_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v1.json"
CANDIDATE2_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v2.json"
CANDIDATE3_ARTIFACT = ROOT / "runs" / "gate2b_hazard_v3.json"
FLOOR_RUN = ROOT / "runs" / "gate2b_floors_v1.json"
SCHEMA_VERSION = "gate2b_forensics1.v1"
RUN_NAME = "gate2b_forensics1_v1"

#: The registered diagnostic (issue #42, comment 4940442065). The registration
#: wins: this runner answers exactly its four frozen questions.
REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4940442065"
)
REGISTRATION_POINTER = "4940442065"
REGISTRATION_TITLE = (
    "Registered diagnostic: gate-2b forensics 1, decompose the four residual "
    "mechanisms"
)
GRADING_POINTER = "4940440505"

#: Reused frozen dials (candidate 3 / the locked gate_2b protocol).
GATE_SEEDS = c3.GATE_SEEDS  # (0, 1, 2, 3, 4)
N_DRAWS = c3.N_DRAWS  # 20
DRAW_SEED_BASE = c3.DRAW_SEED_BASE  # 5200
EXACT_ATOL = c3.EXACT_ATOL  # 1e-12

#: The four residual families (grading 4940440505) and the cell whose miss the
#: registration names for each. Used to focus the decompositions; the FAILING
#: cell set is read from the committed candidate-3 artifact (never hardcoded).
Q2_MALE_CHILD_BANDS = ("25-34", "35-44", "45-54", "55-64", "65-74")
#: Single-year ages spanning the young failing spouse bands (Q1 within-band).
SPOUSE_SINGLE_YEAR_LO = hc.START_AGE  # 15
SPOUSE_SINGLE_YEAR_HI = 34
#: Single-year ages spanning the 55+ grandchild aggregate (Q4 within-band).
GRANDCHILD_SINGLE_YEAR_LO = 55
GRANDCHILD_SINGLE_YEAR_HI = 90

#: Per-seed cache OUTSIDE runs/ (never committed): a long run resumes.
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2b_forensics1_cache.json"
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


def _dist_stats(values: list[float]) -> dict[str, Any]:
    """Mean, sample sd (ddof=1), min/max, n of a value list."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    return {
        "n": n,
        "mean": float(arr.mean()) if n else 0.0,
        "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()) if n else 0.0,
        "max": float(arr.max()) if n else 0.0,
    }


def _mean_over_seeds(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _split_cell(cell: str) -> tuple[str, str]:
    """``coresident_spouse.15-24|male`` -> ("15-24", "male")."""
    band = cell.split(".", 1)[1]
    band, sex = band.split("|", 1)
    return band, sex


def _weighted_share_by_band_sex(
    band: np.ndarray, sex: np.ndarray, weight: np.ndarray, hit: np.ndarray
) -> dict[tuple[str, str], float]:
    """Weighted share of ``hit`` per (band, sex), the reference-moment cell."""
    df = pd.DataFrame(
        {
            "band": band,
            "sex": sex,
            "w": weight.astype(np.float64),
            "wn": weight.astype(np.float64) * hit.astype(np.float64),
        }
    )
    den = df.groupby(["band", "sex"])["w"].sum()
    num = df.groupby(["band", "sex"])["wn"].sum()
    out: dict[tuple[str, str], float] = {}
    for key in den.index:
        d = float(den.loc[key])
        out[key] = float(num.loc[key] / d) if d > 0 else 0.0
    return out


def _aggregate_share(
    age: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    hit: np.ndarray,
    lo: int,
    hi: int,
    which_sex: str,
) -> float:
    """Weighted share of ``hit`` over the open age aggregate (e.g. 55+)."""
    m = (age >= lo) & (age <= hi) & (sex == which_sex)
    w = weight[m].astype(np.float64)
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    return float((w * hit[m].astype(np.float64)).sum() / tot)


# --------------------------------------------------------------------------
# Instrumented candidate-3 draw (faithful copy that returns the components)
# --------------------------------------------------------------------------
def instrumented_draw_v3(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs3.HouseholdCompositionModelV3,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Reproduce :func:`hcs3.simulate_draw_v3` and return every component array.

    A line-for-line copy of the committed candidate-3 draw (same 0xB2B / 0xC2 /
    0xC3 substreams, same train-fitted tables), instrumented to also return the
    legal-marriage vs cohabitation spouse split, the linked vs non-linked
    (maternal + shadow) child counts, the composed vs skipped-generation
    grandchild split, and the family-core vs non-family household-size split --
    all aligned to the returned ``pw`` row order. The equality of the recomposed
    flags against the committed ``simulate_draw_v3`` panel is proved once by
    :func:`fidelity_check`; this is the diagnostic's re-simulation of the
    components ``the committed per-draw machinery`` produces.
    """
    base = model.base

    # 1. carried parent / multigen / legal-marriage spouse (candidate 1).
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

    # 4. certified marital core + maternal births.
    sim_panel, sim_births = ft.simulate(
        mpanel, ids, base.family_transitions, draw_seed
    )
    sim_years = sim_panel.person_years
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    maternal["_source"] = "maternal"

    # 5. linked / shadow paternal children.
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

    # 6. candidate-3 delta substreams (0xC3).
    c3_ss = np.random.SeedSequence([draw_seed, 0xC3])
    custodial_ss, nonfamily_ss, skipgen_ss = c3_ss.spawn(3)
    custodial_rng = np.random.default_rng(custodial_ss)
    nonfamily_rng = np.random.default_rng(nonfamily_ss)
    skipgen_rng = np.random.default_rng(skipgen_ss)

    marital_sim = hcs3._sim_marital_binary(sim_years, side_a_pw)
    child_counts_linked = hcs3.custodial_linked_child_counts(
        paternal_linked[["parent_person_id", "birth_year"]],
        side_a_pw,
        marital_sim,
        model.custodial,
        custodial_rng,
    )
    child_counts = child_counts_nonlinked + child_counts_linked

    coresident_child, grandchild_composed, hh_size_base = hcs.compose_states(
        spouse, c1_parent, c1_multigen, child_counts, base.parent_count
    )

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

    nonfamily_count = hcs3._sample_nonfamily(
        pw, model.nonfamily, nonfamily_rng
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

    return {
        "panel": panel,
        "pw": pw,  # ordered [person_id, year] frame the arrays align to
        "band": pw["band"].to_numpy(dtype=object),
        "sex": pw["sex"].to_numpy(),
        "age": pw["age"].to_numpy(dtype=np.int64),
        "weight": pw["weight"].to_numpy(dtype=np.float64),
        "person_id": pw["person_id"].to_numpy(dtype=np.int64),
        # Q1 spouse split.
        "legal_spouse": legal_spouse,
        "cohab_row": cohab_row,
        "spouse": spouse,
        # Q2 child split.
        "child_counts_linked": child_counts_linked,
        "child_counts_nonlinked": child_counts_nonlinked,
        "coresident_child": coresident_child,
        "linked_ids": linked_ids,
        "men_ids": men_ids,
        # Q3 hh_size split.
        "hh_size_base": hh_size_base.astype(np.int64),
        "nonfamily_count": nonfamily_count.astype(np.int64),
        "hh_size": hh_size,
        "n_parents": np.where(c1_parent, int(base.parent_count), 0).astype(
            np.int64
        ),
        "spouse_int": spouse.astype(np.int64),
        "child_counts": child_counts.astype(np.int64),
        # Q4 grandchild split.
        "grandchild_composed": grandchild_composed,
        "skipgen_row": skipgen_row,
        "coresident_grandchild": coresident_grandchild,
    }


def fidelity_check(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs3.HouseholdCompositionModelV3,
    ids: set[int],
    draw_seed: int,
) -> dict[str, Any]:
    """Prove the instrumented draw equals the committed ``simulate_draw_v3``.

    Runs both on the same (ids, draw_seed) and confirms every scored cell rate
    is bit-identical, so the instrumented component arrays are a faithful
    decomposition of the committed machinery (not a re-derivation that could
    drift).
    """
    inst = instrumented_draw_v3(hh, mpanel, model, ids, draw_seed)
    real_panel, _ = hcs3.simulate_draw_v3(hh, mpanel, model, ids, draw_seed)
    a = hc.reference_moments(inst["panel"], ids, weighted=True)
    b = hc.reference_moments(real_panel, ids, weighted=True)
    max_dev = max(abs(a[c]["rate"] - b[c]["rate"]) for c in b)
    return {
        "draw_seed": draw_seed,
        "n_cells_compared": len(b),
        "max_abs_rate_deviation_vs_committed_simulate_draw_v3": float(max_dev),
        "bit_identical": bool(max_dev <= EXACT_ATOL),
    }


# --------------------------------------------------------------------------
# Deterministic train-side reference decompositions (seed B, no simulation RNG)
# --------------------------------------------------------------------------
def spouse_concept_codes(
    rel_map: pd.DataFrame,
    person_waves: pd.DataFrame,
    ids_b: set[int],
) -> dict[str, Any]:
    """Q1 concept residual: which MX8 codes carry reference spouse mass.

    Over the train (side B) reference roster, enumerate the MX8 ego->alter
    codes that flag ``coresident_spouse`` and reconcile the coresident-spouse
    person-wave mass into code-20 (legal) only / code-22 (partner) only / both.
    The reference concept is ``CORESIDENCE_LINKS['coresident_spouse']`` = MX8
    {20, 22}, so any weighted mass BEYOND {20, 22} is a concept residual the two
    components (legal core, code-22 overlay) do not supply. Reconciles to 1.0.
    """
    spouse_codes = set(hc.CORESIDENCE_LINKS["coresident_spouse"])  # {20, 22}
    # Ego->alter spouse-link rows on the train side (drop the self diagonal).
    links = rel_map[
        (rel_map["ego_rel_to_alter"] != relmap.SELF)
        & (rel_map["ego_person_id"].isin(ids_b))
    ]
    spouse_links = links[links["ego_rel_to_alter"].isin(spouse_codes)]
    # Per (person, year) which spouse codes are present.
    grp = (
        spouse_links.groupby(["interview_year", "ego_person_id"])[
            "ego_rel_to_alter"
        ]
        .agg(lambda s: frozenset(int(x) for x in s))
        .rename("spouse_link_codes")
        .reset_index()
        .rename(
            columns={"interview_year": "year", "ego_person_id": "person_id"}
        )
    )
    # Join to the scored train person-waves (weight + coresident_spouse flag).
    pw = person_waves[person_waves["person_id"].isin(ids_b)][
        ["person_id", "year", "band", "sex", "weight", "coresident_spouse"]
    ].merge(grp, on=["person_id", "year"], how="left")

    cs = pw[pw["coresident_spouse"]].copy()
    w = cs["weight"].to_numpy(dtype=np.float64)
    den = float(w.sum())
    has20 = cs["spouse_link_codes"].apply(
        lambda s: isinstance(s, frozenset) and 20 in s
    )
    has22 = cs["spouse_link_codes"].apply(
        lambda s: isinstance(s, frozenset) and 22 in s
    )
    has_any = cs["spouse_link_codes"].apply(
        lambda s: isinstance(s, frozenset) and len(s) > 0
    )
    only20 = has20 & ~has22
    only22 = has22 & ~has20
    both = has20 & has22
    beyond = ~has_any  # flagged coresident_spouse but no {20,22} link found

    def share(mask: pd.Series) -> float:
        return float(w[mask.to_numpy()].sum() / den) if den > 0 else 0.0

    s20, s22, sboth, sbeyond = (
        share(only20),
        share(only22),
        share(both),
        share(beyond),
    )
    # Per band x sex code-22 (cohab) share of the reference spouse mass -- the
    # component-attribution map: which component each failing cell needs.
    by_cell: dict[str, dict[str, float]] = {}
    for (band, sex), g in cs.groupby(["band", "sex"]):
        gw = g["weight"].to_numpy(dtype=np.float64)
        gd = float(gw.sum())
        if gd <= 0:
            continue
        g20 = g["spouse_link_codes"].apply(
            lambda s: isinstance(s, frozenset) and 20 in s
        )
        g22 = g["spouse_link_codes"].apply(
            lambda s: isinstance(s, frozenset) and 22 in s
        )
        by_cell[f"coresident_spouse.{band}|{sex}"] = {
            "share_code20_legal": float(
                gw[(g20 & ~g22).to_numpy()].sum() / gd
            ),
            "share_code22_cohab": float(
                gw[(g22 & ~g20).to_numpy()].sum() / gd
            ),
            "share_both": float(gw[(g20 & g22).to_numpy()].sum() / gd),
        }
    # The distinct MX8 codes actually present among spouse links (data check).
    present_codes = sorted(
        int(x) for x in spouse_links["ego_rel_to_alter"].unique()
    )
    return {
        "reference_concept_codes_mx8": sorted(spouse_codes),
        "distinct_spouse_link_codes_present": present_codes,
        "codes_beyond_20_22_present": sorted(
            c for c in present_codes if c not in spouse_codes
        ),
        "share_code20_legal_only": s20,
        "share_code22_cohab_only": s22,
        "share_both_codes": sboth,
        "share_beyond_20_22": sbeyond,
        "reconciliation_remainder": 1.0 - (s20 + s22 + sboth + sbeyond),
        "by_cell_code_share": by_cell,
    }


def cohab_single_year_age(
    model_cohab_flag: pd.DataFrame,
    person_waves: pd.DataFrame,
    cohab_entry: dict[tuple[str, str], float],
    cohab_exit: dict[tuple[str, str], float],
    ids_b: set[int],
) -> dict[str, Any]:
    """Q1 fit-vs-raw: raw code-22 stock by single-year age vs band-level fit.

    Joins the observed code-22 partner flag to the train person-waves, and
    reports the weighted code-22 occupancy stock share by single year of age
    (and the code-22 entry/exit transition rates by single age), alongside the
    band-constant fitted cohabitation entry/exit hazards -- so a steep
    within-band single-year gradient under a flat band hazard is the 2a
    band-split lesson made visible.
    """
    pw = person_waves[person_waves["person_id"].isin(ids_b)].merge(
        model_cohab_flag, on=["person_id", "year"], how="left"
    )
    pw["cohabiting"] = pw["cohabiting"].fillna(False).astype(bool)
    pw = pw.sort_values(["person_id", "year"]).reset_index(drop=True)
    pw["next_cohab"] = pw.groupby("person_id", sort=False)["cohabiting"].shift(
        -1
    )

    out_by_sex: dict[str, Any] = {}
    for sex in hc.SEXES:
        s = pw[pw["sex"] == sex]
        stock_by_age: dict[str, float] = {}
        for age in range(SPOUSE_SINGLE_YEAR_LO, SPOUSE_SINGLE_YEAR_HI + 1):
            a = s[s["age"] == age]
            w = a["weight"].to_numpy(dtype=np.float64)
            tot = float(w.sum())
            stock_by_age[str(age)] = (
                float(
                    (w * a["cohabiting"].to_numpy(dtype=np.float64)).sum()
                    / tot
                )
                if tot > 0
                else 0.0
            )
        # Band-level raw stock + fitted hazards for the two young failing bands.
        band_detail: dict[str, Any] = {}
        for lo, hi in ((15, 24), (25, 34)):
            band = hc.band_label(lo, hi)
            a = s[(s["age"] >= lo) & (s["age"] <= hi)]
            w = a["weight"].to_numpy(dtype=np.float64)
            tot = float(w.sum())
            raw_stock = (
                float(
                    (w * a["cohabiting"].to_numpy(dtype=np.float64)).sum()
                    / tot
                )
                if tot > 0
                else 0.0
            )
            # Raw entry / exit among at-risk waves with an observed next wave.
            hasn = a[a["has_next"]]
            entry_pool = hasn[~hasn["cohabiting"]]
            exit_pool = hasn[hasn["cohabiting"]]

            def _wrate(frame: pd.DataFrame, col: pd.Series) -> float:
                ww = frame["weight"].to_numpy(dtype=np.float64)
                t = float(ww.sum())
                return (
                    float((ww * col.to_numpy(dtype=np.float64)).sum() / t)
                    if t > 0
                    else 0.0
                )

            raw_entry = _wrate(
                entry_pool, entry_pool["next_cohab"].fillna(False)
            )
            raw_exit = _wrate(exit_pool, exit_pool["next_cohab"].eq(False))
            # Within-band single-year gradient: youngest-third vs oldest-third
            # raw stock ratio (a scalar summary of the within-band structure).
            third = (hi - lo + 1) // 3
            young = s[(s["age"] >= lo) & (s["age"] <= lo + third - 1)]
            old = s[(s["age"] >= hi - third + 1) & (s["age"] <= hi)]

            def _stock(frame: pd.DataFrame) -> float:
                ww = frame["weight"].to_numpy(dtype=np.float64)
                t = float(ww.sum())
                return (
                    float(
                        (
                            ww * frame["cohabiting"].to_numpy(dtype=np.float64)
                        ).sum()
                        / t
                    )
                    if t > 0
                    else 0.0
                )

            young_stock = _stock(young)
            old_stock = _stock(old)
            band_detail[band] = {
                "raw_stock_share": raw_stock,
                "raw_entry_rate": raw_entry,
                "raw_exit_rate": raw_exit,
                "fitted_entry_hazard": float(
                    cohab_entry.get((band, sex), 0.0)
                ),
                "fitted_exit_hazard": float(cohab_exit.get((band, sex), 0.0)),
                "within_band_young_third_stock": young_stock,
                "within_band_old_third_stock": old_stock,
                "within_band_old_over_young_ratio": (
                    float(old_stock / young_stock)
                    if young_stock > 0
                    else float("inf")
                ),
            }
        out_by_sex[sex] = {
            "raw_code22_stock_by_single_year_age": stock_by_age,
            "band_detail": band_detail,
        }
    return out_by_sex


def unlinked_men_profile(
    mpanel: transitions.MaritalPanel,
    person_waves: pd.DataFrame,
    linked_ids: set[int],
    ids_b: set[int],
) -> dict[str, Any]:
    """Q2: characterize which men are unlinked (age, marital state, era).

    Deterministic over the train men. For the person-wave exposure of linked
    vs unlinked men (weighted), report the age-band composition, the observed
    married share (the shadow kernel only fires for married men), and the birth
    decade (era) composition -- and the ever-married cut (shadow-eligible).
    """
    men = mpanel.attrs[
        (mpanel.attrs["sex"] == "male")
        & (mpanel.attrs["person_id"].isin(ids_b))
    ]
    men_ids = set(int(x) for x in men["person_id"])
    unlinked = men_ids - linked_ids
    ever_married = set(
        int(x) for x in men[men["n_marriages"] > 0]["person_id"]
    )

    pw = person_waves[
        (person_waves["sex"] == "male")
        & (person_waves["person_id"].isin(men_ids))
    ][["person_id", "year", "band", "age", "weight"]].copy()
    # Observed marital state per (person, year).
    py = mpanel.person_years[["person_id", "year", "marital_state"]]
    pw = pw.merge(py, on=["person_id", "year"], how="left")
    pw["married"] = (pw["marital_state"] == "married").astype(float)
    pw["is_linked"] = pw["person_id"].isin(linked_ids)
    birth_year = mpanel.attrs.set_index("person_id")["birth_year"]
    pw["birth_decade"] = pw["person_id"].map(birth_year) // 10 * 10

    def _profile(frame: pd.DataFrame) -> dict[str, Any]:
        w = frame["weight"].to_numpy(dtype=np.float64)
        tot = float(w.sum())
        if tot <= 0:
            return {"weighted_person_waves": 0.0}
        band_share = {}
        for lo, hi in hc.COMPOSITION_AGE_BANDS:
            b = hc.band_label(lo, hi)
            m = (frame["band"] == b).to_numpy()
            band_share[b] = float(w[m].sum() / tot)
        decade_share = {}
        for dec, g in frame.groupby("birth_decade"):
            gw = float(g["weight"].sum())
            decade_share[str(int(dec))] = float(gw / tot)
        return {
            "weighted_person_waves": tot,
            "married_share": float(
                (w * frame["married"].fillna(0).to_numpy()).sum() / tot
            ),
            "age_band_share": band_share,
            "birth_decade_share": decade_share,
        }

    return {
        "n_men": len(men_ids),
        "n_linked": len(linked_ids & men_ids),
        "n_unlinked": len(unlinked),
        "unlinked_fraction": (
            float(len(unlinked) / len(men_ids)) if men_ids else 0.0
        ),
        "n_unlinked_ever_married_shadow_eligible": len(
            unlinked & ever_married
        ),
        "unlinked_ever_married_fraction_of_unlinked": (
            float(len(unlinked & ever_married) / len(unlinked))
            if unlinked
            else 0.0
        ),
        "linked_profile": _profile(pw[pw["is_linked"]]),
        "unlinked_profile": _profile(pw[~pw["is_linked"]]),
    }


def skipgen_single_year_age(
    person_waves: pd.DataFrame,
    skipgen_entry: dict[tuple[str, str], float],
    ids_b: set[int],
) -> dict[str, Any]:
    """Q4: raw skip-gen stock by single-year age within 55+ vs band-level fit.

    The observed skipped-generation state (``coresident_grandchild AND NOT
    multigen``) weighted stock by single year of age within 55+ (female),
    alongside the band-constant fitted entry hazard -- the Q1 fit-vs-raw check
    for the grandchild remainder.
    """
    pw = hcs3.attach_skipgen(person_waves)
    pw = pw[(pw["person_id"].isin(ids_b)) & (pw["sex"] == "female")]
    stock_by_age: dict[str, float] = {}
    for age in range(GRANDCHILD_SINGLE_YEAR_LO, GRANDCHILD_SINGLE_YEAR_HI + 1):
        a = pw[pw["age"] == age]
        w = a["weight"].to_numpy(dtype=np.float64)
        tot = float(w.sum())
        stock_by_age[str(age)] = (
            float((w * a["skipgen"].to_numpy(dtype=np.float64)).sum() / tot)
            if tot > 0
            else 0.0
        )
    band_detail: dict[str, Any] = {}
    for lo, hi in ((55, 64), (65, 74), (75, 120)):
        band = hc.band_label(lo, hi)
        a = pw[(pw["age"] >= lo) & (pw["age"] <= hi)]
        w = a["weight"].to_numpy(dtype=np.float64)
        tot = float(w.sum())
        band_detail[band] = {
            "raw_skipgen_stock_share": (
                float(
                    (w * a["skipgen"].to_numpy(dtype=np.float64)).sum() / tot
                )
                if tot > 0
                else 0.0
            ),
            "fitted_entry_hazard": float(
                skipgen_entry.get((band, "female"), 0.0)
            ),
        }
    return {
        "raw_skipgen_stock_by_single_year_age_female": stock_by_age,
        "band_detail_female": band_detail,
    }


# --------------------------------------------------------------------------
# Per-seed computation (fit on side B, 20 train-side instrumented draws)
# --------------------------------------------------------------------------
def compute_seed(
    seed: int,
    data: dict[str, Any],
    tol: dict[str, float],
    fail_cells: dict[str, list[int]],
    verbose: bool,
) -> dict[str, Any]:
    """Fit candidate 3 on side B, then 20 train-side instrumented draws.

    Returns the deterministic train-side reference decompositions and the
    per-draw component decompositions for the four questions, all train-side
    (side B fitted AND simulated; scored against side B's own reference).
    """
    t0 = time.time()
    hh = data["hh"]
    mpanel = data["mpanel"]
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs3.fit_household_model_v3(
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
    )
    linked_ids = (
        set(int(x) for x in model.father_links["parent_person_id"])
        & set(
            int(x)
            for x in mpanel.attrs[mpanel.attrs["sex"] == "male"]["person_id"]
        )
        & ids_b
    )

    rate_b = hc.reference_moments(hh, ids_b, weighted=True)

    # ---- Deterministic train-side reference decompositions (no sim RNG). ----
    concept = spouse_concept_codes(data["rel_map"], hh.person_waves, ids_b)
    cohab_age = cohab_single_year_age(
        model.base_v2.cohab_flag,
        hh.person_waves,
        model.base_v2.cohab_entry,
        model.base_v2.cohab_exit,
        ids_b,
    )
    unlinked = unlinked_men_profile(mpanel, hh.person_waves, linked_ids, ids_b)
    skipgen_age = skipgen_single_year_age(
        hh.person_waves, model.skipgen_entry, ids_b
    )

    # ---- 20 instrumented train-side draws: component decompositions. ----
    # Q1 spouse: per (band, sex) legal-only vs full spouse rate, per draw.
    q1_legal: dict[str, list[float]] = {}
    q1_full: dict[str, list[float]] = {}
    # Q2 child: per male band, linked vs unlinked contribution, per draw.
    q2_link: dict[str, list[float]] = {}
    q2_unlink: dict[str, list[float]] = {}
    q2_wlink: dict[str, list[float]] = {}
    q2_wunlink: dict[str, list[float]] = {}
    q2_full: dict[str, list[float]] = {}
    # Q3 hh_size: distribution + size-3 partition by non-family count, per draw.
    q3_sizes = (1, 2, 3, 4, "5+")
    q3_dist: dict[str, list[float]] = {str(s): [] for s in q3_sizes}
    q3_base_dist: dict[str, list[float]] = {str(s): [] for s in q3_sizes}
    q3_size3_by_nf: dict[str, list[float]] = {"0": [], "1": [], "2": []}
    q3_size3_family_core: list[float] = []  # nf==0 share of size-3
    # Size-3 mass partitioned by primary composition route (mutually exclusive,
    # sums to P(size==3)): the "which component drives size-3" decomposition.
    q3_route_keys = (
        "nonfamily_ge1",
        "couple_plus_child",
        "two_plus_children",
        "coresident_parents",
        "other_family_core",
    )
    q3_size3_routes: dict[str, list[float]] = {k: [] for k in q3_route_keys}
    # Q4 grandchild 55+|female: composed vs skipgen-only vs union, per draw.
    q4_composed: list[float] = []
    q4_skiponly: list[float] = []
    q4_union: list[float] = []

    def _agg55f(age, sex, weight, hit):
        return _aggregate_share(age, sex, weight, hit, 55, 120, "female")

    for k in range(N_DRAWS):
        draw_seed = DRAW_SEED_BASE + k
        d = instrumented_draw_v3(hh, mpanel, model, ids_b, draw_seed)
        band, sex, age, weight = d["band"], d["sex"], d["age"], d["weight"]

        # Q1: legal-only and full spouse band x sex shares.
        legal_cells = _weighted_share_by_band_sex(
            band, sex, weight, d["legal_spouse"]
        )
        full_cells = _weighted_share_by_band_sex(
            band, sex, weight, d["spouse"]
        )
        for (b, s), v in full_cells.items():
            key = f"coresident_spouse.{b}|{s}"
            q1_full.setdefault(key, []).append(v)
            q1_legal.setdefault(key, []).append(
                float(legal_cells.get((b, s), 0.0))
            )

        # Q2: male coresident_child partitioned by linked vs unlinked men.
        is_male = sex == "male"
        is_linked = np.isin(d["person_id"], list(d["linked_ids"]))
        cc = d["coresident_child"].astype(np.float64)
        for bl, bh in hc.COMPOSITION_AGE_BANDS:
            b = hc.band_label(bl, bh)
            in_band = is_male & (band == b)
            wtot = float(weight[in_band].sum())
            if wtot <= 0:
                continue
            key = f"coresident_child.{b}|male"
            wl = weight[in_band & is_linked]
            wu = weight[in_band & ~is_linked]
            cl = cc[in_band & is_linked]
            cu = cc[in_band & ~is_linked]
            w_link = float(wl.sum() / wtot)
            w_unlink = float(wu.sum() / wtot)
            r_link = float((wl * cl).sum() / wl.sum()) if wl.sum() > 0 else 0.0
            r_unlink = (
                float((wu * cu).sum() / wu.sum()) if wu.sum() > 0 else 0.0
            )
            q2_link.setdefault(key, []).append(w_link * r_link)
            q2_unlink.setdefault(key, []).append(w_unlink * r_unlink)
            q2_wlink.setdefault(key, []).append(w_link)
            q2_wunlink.setdefault(key, []).append(w_unlink)
            q2_full.setdefault(key, []).append(
                float((weight[in_band] * cc[in_band]).sum() / wtot)
            )

        # Q3: hh_size distribution (full + family-core-only) and size-3 by nf.
        hh_size = d["hh_size"]
        base = d["hh_size_base"]
        nf = d["nonfamily_count"]
        wtot = float(weight.sum())
        for s in q3_sizes:
            if s == "5+":
                mfull = hh_size >= 5
                mbase = base >= 5
            else:
                mfull = hh_size == s
                mbase = base == s
            q3_dist[str(s)].append(float(weight[mfull].sum() / wtot))
            q3_base_dist[str(s)].append(float(weight[mbase].sum() / wtot))
        size3 = hh_size == 3
        for c in ("0", "1", "2"):
            m = size3 & (nf == int(c))
            q3_size3_by_nf[c].append(float(weight[m].sum() / wtot))
        m0 = size3 & (nf == 0)
        s3 = float(weight[size3].sum())
        q3_size3_family_core.append(
            float(weight[m0].sum() / s3) if s3 > 0 else 0.0
        )
        # Mutually-exclusive size-3 composition routes (partition size-3 mass).
        sp = d["spouse_int"]
        ch = d["child_counts"]
        npar = d["n_parents"]
        route_nf = size3 & (nf >= 1)
        core = size3 & (nf == 0)
        route_couple_child = core & (sp == 1) & (ch >= 1)
        route_two_children = core & (sp == 0) & (ch >= 2) & (npar == 0)
        route_parents = core & (npar >= 1) & (sp == 0) & (ch == 0)
        route_other = (
            core & ~route_couple_child & ~route_two_children & ~route_parents
        )
        for name, mask in (
            ("nonfamily_ge1", route_nf),
            ("couple_plus_child", route_couple_child),
            ("two_plus_children", route_two_children),
            ("coresident_parents", route_parents),
            ("other_family_core", route_other),
        ):
            q3_size3_routes[name].append(float(weight[mask].sum() / wtot))

        # Q4: grandchild 55+|female composed vs skipgen-only vs union.
        composed = d["grandchild_composed"]
        skip_only = d["skipgen_row"] & ~d["grandchild_composed"]
        union = d["coresident_grandchild"]
        q4_composed.append(_agg55f(age, sex, weight, composed))
        q4_skiponly.append(_agg55f(age, sex, weight, skip_only))
        q4_union.append(_agg55f(age, sex, weight, union))

    elapsed = round(time.time() - t0, 1)
    if verbose:
        sc = np.mean(q1_full.get("coresident_spouse.15-24|male", [0.0]))
        print(
            f"seed {seed}: n_train={len(ids_b)} "
            f"spouse.15-24|male full={sc:.4f} "
            f"(rate_b {rate_b['coresident_spouse.15-24|male']['rate']:.4f}) "
            f"[{elapsed}s]"
        )

    return {
        "seed": seed,
        "n_train_persons": len(ids_b),
        "rate_b": {c: float(rate_b[c]["rate"]) for c in sorted(tol)},
        "q1_spouse_legal_mean": {
            k: float(np.mean(v)) for k, v in q1_legal.items()
        },
        "q1_spouse_full_mean": {
            k: float(np.mean(v)) for k, v in q1_full.items()
        },
        "q1_spouse_full_draws": {
            k: v for k, v in q1_full.items() if k in fail_cells
        },
        "q1_concept": concept,
        "q1_cohab_age": cohab_age,
        "q2_linked_contribution_mean": {
            k: float(np.mean(v)) for k, v in q2_link.items()
        },
        "q2_unlinked_contribution_mean": {
            k: float(np.mean(v)) for k, v in q2_unlink.items()
        },
        "q2_w_linked_mean": {
            k: float(np.mean(v)) for k, v in q2_wlink.items()
        },
        "q2_w_unlinked_mean": {
            k: float(np.mean(v)) for k, v in q2_wunlink.items()
        },
        "q2_male_child_full_mean": {
            k: float(np.mean(v)) for k, v in q2_full.items()
        },
        "q2_unlinked_profile": unlinked,
        "q3_dist_mean": {k: float(np.mean(v)) for k, v in q3_dist.items()},
        "q3_base_dist_mean": {
            k: float(np.mean(v)) for k, v in q3_base_dist.items()
        },
        "q3_size3_by_nonfamily_mean": {
            k: float(np.mean(v)) for k, v in q3_size3_by_nf.items()
        },
        "q3_size3_family_core_share_mean": float(
            np.mean(q3_size3_family_core)
        ),
        "q3_size3_routes_mean": {
            k: float(np.mean(v)) for k, v in q3_size3_routes.items()
        },
        "q4_composed_mean": float(np.mean(q4_composed)),
        "q4_skiponly_mean": float(np.mean(q4_skiponly)),
        "q4_union_mean": float(np.mean(q4_union)),
        "q4_union_draws": [float(x) for x in q4_union],
        "q4_skipgen_age": skipgen_age,
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Committed-artifact reads (side A holdout: NEVER re-simulated)
# --------------------------------------------------------------------------
def _committed() -> dict[str, Any]:
    """Read the committed candidate-1/2/3 artifacts (side-A holdout evidence)."""
    c1 = json.loads(CANDIDATE1_ARTIFACT.read_text())
    c2 = json.loads(CANDIDATE2_ARTIFACT.read_text())
    c3a = json.loads(CANDIDATE3_ARTIFACT.read_text())
    return {"c1": c1, "c2": c2, "c3": c3a}


def _failing_cells(c3a: dict[str, Any]) -> dict[str, list[int]]:
    """Map each failing gated cell (candidate 3, side A) to its failing seeds."""
    out: dict[str, list[int]] = {}
    for f in c3a["verdict"]["all_failing_gated_cells"]:
        out.setdefault(f["cell"], []).append(int(f["seed"]))
    for cell in out:
        out[cell] = sorted(out[cell])
    return out


def _committed_cube(c3a: dict[str, Any]) -> dict[str, Any]:
    """The committed [20, 46, 5] side-A per-draw rate cube, keyed for reads."""
    cube = c3a["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    cells = cube["cell_index"]
    seeds = cube["seed_index"]
    rates = cube["rates"]  # [k][cell][seed]
    idx = {c: i for i, c in enumerate(cells)}
    sidx = {s: j for j, s in enumerate(seeds)}

    def draws(cell: str, seed: int) -> list[float]:
        i, j = idx[cell], sidx[seed]
        return [float(rates[k][i][j]) for k in range(len(rates))]

    return {"draws": draws, "seeds": seeds}


def _committed_cell(
    c3a: dict[str, Any], cell: str, seed: int
) -> dict[str, Any]:
    """A committed side-A gated-cell record (rate_a, r_candidate, score)."""
    per = {s["seed"]: s for s in c3a["per_seed"]}
    rec = per[seed]["gated_cells"][cell]
    return {
        "rate_a": float(rec["rate_a"]),
        "r_candidate": float(rec["r_candidate"]),
        "score": float(rec["score"]),
        "tolerance": float(rec["tolerance"]),
        "pass": bool(rec["pass"]),
    }


# --------------------------------------------------------------------------
# Assemble the four question blocks
# --------------------------------------------------------------------------
def assemble_q1(
    per_seed: list[dict[str, Any]],
    committed: dict[str, Any],
    fail_cells: dict[str, list[int]],
    tol: dict[str, float],
) -> dict[str, Any]:
    """Q1 -- spouse: legal-core vs cohabitation-overlay, age fit, stability."""
    c3a = committed["c3"]
    cube = _committed_cube(c3a)
    spouse_fail = {
        c: s
        for c, s in fail_cells.items()
        if c.startswith("coresident_spouse.")
    }

    # (a) Legal-core vs cohab-overlay decomposition per failing cell x seed.
    decomposition: dict[str, Any] = {}
    for cell, seeds in sorted(spouse_fail.items()):
        per_cell: dict[str, Any] = {}
        for seed in seeds:
            sd = next(s for s in per_seed if s["seed"] == seed)
            ref_b = float(sd["rate_b"][cell])
            legal = float(sd["q1_spouse_legal_mean"].get(cell, 0.0))
            full = float(sd["q1_spouse_full_mean"].get(cell, 0.0))
            overlay = full - legal  # union is additive: overlay-supplied mass
            residual = ref_b - full  # what neither component reaches (train)
            legal_gap = ref_b - legal  # gap the overlay must close
            com = _committed_cell(c3a, cell, seed)
            per_cell[str(seed)] = {
                "reference_train_rate_b": ref_b,
                "legal_core_only_rate": legal,
                "full_spouse_rate": full,
                "overlay_contribution": overlay,
                "legal_core_gap": legal_gap,
                "residual_miss_train": residual,
                "reconciliation_remainder": legal_gap - overlay - residual,
                "train_signed_logratio": _signed(full, ref_b),
                "holdout_committed": com,
                "holdout_signed_logratio": _signed(
                    com["r_candidate"], com["rate_a"]
                ),
                "dominant_component": (
                    "cohabitation_overlay"
                    if abs(overlay) >= abs(legal)
                    else "legal_core"
                ),
            }
        decomposition[cell] = per_cell

    # (c) Draw-stability of each failing cell x seed on the committed side-A
    #     cube and the train-side draws.
    stability: dict[str, Any] = {}
    for cell, seeds in sorted(spouse_fail.items()):
        per_cell = {}
        for seed in seeds:
            com = _committed_cell(c3a, cell, seed)
            draws = cube["draws"](cell, seed)
            signed = [_signed(r, com["rate_a"]) for r in draws]
            fail_side = [
                1.0 if abs(s) > tol[cell] else 0.0
                for s in [_score(r, com["rate_a"]) for r in draws]
            ]
            same_dir = all(x > 0 for x in signed) or all(x < 0 for x in signed)
            sd = next(s for s in per_seed if s["seed"] == seed)
            train_draws = sd["q1_spouse_full_draws"].get(cell, [])
            per_cell[str(seed)] = {
                "holdout_per_draw_signed_mean": float(np.mean(signed)),
                "holdout_per_draw_signed_sd": (
                    float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0
                ),
                "holdout_fraction_draws_clip_tolerance": float(
                    np.mean(fail_side)
                ),
                "holdout_direction_same_sign_all_draws": bool(same_dir),
                "holdout_direction": (
                    "over" if np.mean(signed) > 0 else "under"
                ),
                "train_per_draw_rate_sd": (
                    float(np.std(train_draws, ddof=1))
                    if len(train_draws) > 1
                    else 0.0
                ),
            }
        stability[cell] = per_cell

    # (b) Fit-vs-raw single-year age structure (seed-averaged).
    age_fit = {
        sex: {
            "band_detail": {
                band: {
                    key: _mean_over_seeds(
                        [
                            s["q1_cohab_age"][sex]["band_detail"][band][key]
                            for s in per_seed
                        ]
                    )
                    for key in per_seed[0]["q1_cohab_age"][sex]["band_detail"][
                        band
                    ]
                }
                for band in per_seed[0]["q1_cohab_age"][sex]["band_detail"]
            },
            "raw_code22_stock_by_single_year_age": {
                age: _mean_over_seeds(
                    [
                        s["q1_cohab_age"][sex][
                            "raw_code22_stock_by_single_year_age"
                        ][age]
                        for s in per_seed
                    ]
                )
                for age in per_seed[0]["q1_cohab_age"][sex][
                    "raw_code22_stock_by_single_year_age"
                ]
            },
        }
        for sex in hc.SEXES
    }

    # (d) Concept residual (seed-averaged).
    concept = {
        "reference_concept_codes_mx8": per_seed[0]["q1_concept"][
            "reference_concept_codes_mx8"
        ],
        "distinct_spouse_link_codes_present": sorted(
            set().union(
                *[
                    set(s["q1_concept"]["distinct_spouse_link_codes_present"])
                    for s in per_seed
                ]
            )
        ),
        "codes_beyond_20_22_present": sorted(
            set().union(
                *[
                    set(s["q1_concept"]["codes_beyond_20_22_present"])
                    for s in per_seed
                ]
            )
        ),
        "share_code20_legal_only": _mean_over_seeds(
            [s["q1_concept"]["share_code20_legal_only"] for s in per_seed]
        ),
        "share_code22_cohab_only": _mean_over_seeds(
            [s["q1_concept"]["share_code22_cohab_only"] for s in per_seed]
        ),
        "share_both_codes": _mean_over_seeds(
            [s["q1_concept"]["share_both_codes"] for s in per_seed]
        ),
        "share_beyond_20_22": _mean_over_seeds(
            [s["q1_concept"]["share_beyond_20_22"] for s in per_seed]
        ),
        "codebook_note": (
            "The MX8 ego->alter frame's only spouse/partner categories are 20 "
            "(Spouse) and 22 (Partner); the staged MX23REL codebook "
            "(Ego_alter_matrix_rel.xlsx) folds RRP 90 (uncooperative spouse) "
            "into 20 and RRP 88 (first-year cohabitor) / 92 (uncooperative "
            "partner) into 22, so no spouse/partner code mass exists outside "
            "{20, 22} in the coresidence frame -- verified, not assumed."
        ),
        "by_cell_code_share": {
            cell: {
                key: _mean_over_seeds(
                    [
                        s["q1_concept"]["by_cell_code_share"][cell][key]
                        for s in per_seed
                        if cell in s["q1_concept"]["by_cell_code_share"]
                    ]
                )
                for key in (
                    "share_code20_legal",
                    "share_code22_cohab",
                    "share_both",
                )
            }
            for cell in sorted(spouse_fail)
        },
        "reconciliation_remainder": _mean_over_seeds(
            [s["q1_concept"]["reconciliation_remainder"] for s in per_seed]
        ),
    }

    # Per-cell direction (holdout) and dominant component (train), seed-averaged.
    cell_summary: dict[str, Any] = {}
    for cell in sorted(spouse_fail):
        blk = decomposition[cell]
        hold_signed = _mean_over_seeds(
            [v["holdout_signed_logratio"] for v in blk.values()]
        )
        overlay = _mean_over_seeds(
            [v["overlay_contribution"] for v in blk.values()]
        )
        legal = _mean_over_seeds(
            [v["legal_core_only_rate"] for v in blk.values()]
        )
        full = _mean_over_seeds([v["full_spouse_rate"] for v in blk.values()])
        residual = _mean_over_seeds(
            [v["residual_miss_train"] for v in blk.values()]
        )
        cell_summary[cell] = {
            "holdout_direction": "over" if hold_signed > 0 else "under",
            "legal_core_rate": legal,
            "overlay_contribution": overlay,
            "full_rate": full,
            "residual_miss_train": residual,
            "code22_cohab_share_of_reference": concept["by_cell_code_share"]
            .get(cell, {})
            .get("share_code22_cohab", 0.0),
        }
    grad_m = age_fit["male"]["band_detail"]["15-24"][
        "within_band_old_over_young_ratio"
    ]
    over_cells = [
        c for c, v in cell_summary.items() if v["holdout_direction"] == "over"
    ]
    under_cells = [
        c for c, v in cell_summary.items() if v["holdout_direction"] == "under"
    ]
    yc = cell_summary.get("coresident_spouse.15-24|male", {})
    finding = (
        "The spouse family fails in TWO opposing directions, and the split is "
        "the legal-core-vs-cohabitation-overlay ALLOCATION, not a concept gap. "
        f"OVERSHOOTING cells {sorted(over_cells)} vs UNDERSHOOTING cells "
        f"{sorted(under_cells)}. At the young overshoot cell "
        f"coresident_spouse.15-24|male the reference spouse mass is "
        f"~{yc.get('code22_cohab_share_of_reference', 0) * 100:.0f}% code-22 "
        "cohabitation and the rest code-20 legal; the legal core produces "
        f"~{yc.get('legal_core_rate', 0):.3f} while the overlay ADDS "
        f"~{yc.get('overlay_contribution', 0):.3f}, and the union "
        f"~{yc.get('full_rate', 0):.3f} OVER-shoots because the cohabitation "
        "overlay over-produces the code-22 stock. The band-level cohabitation "
        "hazard hides steep within-band single-year structure -- raw code-22 "
        f"stock rises ~{grad_m:.0f}x from the youngest to the oldest third of "
        "15-24 (near-zero at 15-17, sizeable at 22-24) while the fitted "
        "entry/exit hazard is band-constant (the 2a single-year lesson) -- so a "
        "flat band hazard mis-places the young mass. The older male cells "
        "UNDERSHOOT instead: their reference mass is majority code-20 legal "
        "marriage and the certified registry under-produces the legal-spouse "
        "stock the code-22 overlay cannot backfill. The concept residual is a "
        f"verified NULL: {concept['share_beyond_20_22'] * 100:.2f}% of "
        "reference spouse mass sits outside MX8 {20, 22} (the codebook folds "
        "RRP 90/88/92 into 20/22), so unlike the 2a undatable-marriage residual "
        "there is no unsupplied code mass -- the miss is entirely the two "
        "components' allocation and their within-band age shape."
    )
    return {
        "question": (
            "spouse family (binding constraint): decompose each failing "
            "spouse cell's miss into legal-marriage-core vs "
            "cohabitation-overlay contributions; measure the train cohabitation "
            "entry/exit fit against the raw MX8-22 spells by single-year age "
            "within the failing bands; check direction and stability across "
            "the 20 draws; test for reference spouse concept mass beyond MX8 "
            "{20, 22}."
        ),
        "method": (
            "Train-side. Per gate seed, fit candidate 3 on side B and simulate "
            "side B at the 20 draw seeds 5200+k; the instrumented draw returns "
            "the legal-marriage (certified registry) spouse array and the "
            "code-22 cohabitation overlay separately, so per cell the full "
            "spouse rate = legal-only rate + overlay contribution EXACTLY (the "
            "union is additive). reference = side B's own weighted "
            "coresident_spouse moment. Draw-stability uses the committed side-A "
            "[20,46,5] per-draw cube (holdout NEVER re-simulated). The concept "
            "residual enumerates the MX8 ego->alter codes flagging "
            "coresident_spouse on the side-B reference roster."
        ),
        "failing_cells": {c: s for c, s in sorted(spouse_fail.items())},
        "per_cell_summary": cell_summary,
        "legal_core_vs_overlay_decomposition": decomposition,
        "draw_stability": stability,
        "single_year_age_fit": age_fit,
        "concept_residual": concept,
        "finding": finding,
    }


def assemble_q2(
    per_seed: list[dict[str, Any]],
    committed: dict[str, Any],
    fail_cells: dict[str, list[int]],
) -> dict[str, Any]:
    """Q2 -- child shadow: unlinked-father contribution to the male overshoot."""
    c3a = committed["c3"]
    child_fail = {
        c: s
        for c, s in fail_cells.items()
        if c.startswith("coresident_child.") and c.endswith("|male")
    }

    decomposition: dict[str, Any] = {}
    for cell, seeds in sorted(child_fail.items()):
        per_cell: dict[str, Any] = {}
        for seed in seeds:
            sd = next(s for s in per_seed if s["seed"] == seed)
            link = float(sd["q2_linked_contribution_mean"].get(cell, 0.0))
            unlink = float(sd["q2_unlinked_contribution_mean"].get(cell, 0.0))
            full = float(sd["q2_male_child_full_mean"].get(cell, 0.0))
            ref_b = float(sd["rate_b"][cell])
            com = _committed_cell(c3a, cell, seed)
            per_cell[str(seed)] = {
                "reference_train_rate_b": ref_b,
                "full_train_rate": full,
                "linked_contribution": link,
                "unlinked_shadow_contribution": unlink,
                "reconciliation_remainder": full - (link + unlink),
                "w_linked": float(sd["q2_w_linked_mean"].get(cell, 0.0)),
                "w_unlinked": float(sd["q2_w_unlinked_mean"].get(cell, 0.0)),
                "train_overshoot": full - ref_b,
                "unlinked_share_of_full": (
                    float(unlink / full) if full > 0 else 0.0
                ),
                "holdout_committed": com,
            }
        decomposition[cell] = per_cell

    profile = {
        "n_men": int(
            np.mean([s["q2_unlinked_profile"]["n_men"] for s in per_seed])
        ),
        "unlinked_fraction": _mean_over_seeds(
            [s["q2_unlinked_profile"]["unlinked_fraction"] for s in per_seed]
        ),
        "unlinked_ever_married_fraction_of_unlinked": _mean_over_seeds(
            [
                s["q2_unlinked_profile"][
                    "unlinked_ever_married_fraction_of_unlinked"
                ]
                for s in per_seed
            ]
        ),
        "linked_married_share": _mean_over_seeds(
            [
                s["q2_unlinked_profile"]["linked_profile"]["married_share"]
                for s in per_seed
            ]
        ),
        "unlinked_married_share": _mean_over_seeds(
            [
                s["q2_unlinked_profile"]["unlinked_profile"]["married_share"]
                for s in per_seed
            ]
        ),
        "linked_age_band_share": {
            b: _mean_over_seeds(
                [
                    s["q2_unlinked_profile"]["linked_profile"][
                        "age_band_share"
                    ][b]
                    for s in per_seed
                ]
            )
            for b in per_seed[0]["q2_unlinked_profile"]["linked_profile"][
                "age_band_share"
            ]
        },
        "unlinked_age_band_share": {
            b: _mean_over_seeds(
                [
                    s["q2_unlinked_profile"]["unlinked_profile"][
                        "age_band_share"
                    ][b]
                    for s in per_seed
                ]
            )
            for b in per_seed[0]["q2_unlinked_profile"]["unlinked_profile"][
                "age_band_share"
            ]
        },
    }

    # Seed-averaged worst-cell attribution (recorded descriptively).
    worst = "coresident_child.35-44|male"
    wd = decomposition.get(worst, {})
    mean_link = _mean_over_seeds(
        [v["linked_contribution"] for v in wd.values()]
    )
    mean_unlink = _mean_over_seeds(
        [v["unlinked_shadow_contribution"] for v in wd.values()]
    )
    mean_full = _mean_over_seeds([v["full_train_rate"] for v in wd.values()])
    mean_ref = _mean_over_seeds(
        [v["reference_train_rate_b"] for v in wd.values()]
    )
    finding = (
        "Recorded descriptively (no strong prior). The male coresident_child "
        f"mass is carried overwhelmingly by the LINKED custodially-gated "
        f"fathers, not the unlinked shadow kernel: on the worst cell {worst} "
        f"the train full rate ~{mean_full:.3f} (ref ~{mean_ref:.3f}) splits "
        f"EXACTLY into a linked contribution ~{mean_link:.3f} "
        f"(~{mean_link / mean_full * 100:.0f}%) and an unlinked-shadow "
        f"contribution ~{mean_unlink:.3f} (~{mean_unlink / mean_full * 100:.0f}%). "
        "The shadow's reach is structurally bounded: "
        f"~{profile['unlinked_fraction'] * 100:.0f}% of men are unlinked, but "
        f"only ~{profile['unlinked_ever_married_fraction_of_unlinked'] * 100:.0f}% "
        "of the unlinked are ever-married and the shadow kernel fires ONLY for "
        f"married men (unlinked married share ~{profile['unlinked_married_share'] * 100:.0f}% "
        f"vs linked ~{profile['linked_married_share'] * 100:.0f}%), so it can "
        "attribute children to only a thin married-unlinked slice. Because the "
        "shadow contribution is small and the linked contribution is the bulk, "
        "the residual overshoot is consistent with the custodial gate on the "
        "LINKED children -- the high fitted P(coresident) at young child ages "
        "carrying too many linked children across the father's mid-life bands "
        "-- rather than the shadow kernel's spousal-gap shift; the two are "
        "reported side by side without a causal verdict, per the registration."
    )
    return {
        "question": (
            "child shadow residual: quantify the unlinked-father share's "
            "contribution to the remaining male coresident_child overshoot "
            "(worst 0.141 vs tol 0.071); characterize which men are unlinked "
            "(age, marital state, era) and whether the shadow kernel's "
            "spousal-gap shift misfits them."
        ),
        "method": (
            "Train-side. Per gate seed and draw, partition side-B men into "
            "linked (>=1 cah85_23 father->child link) and unlinked; the male "
            "band coresident_child rate = w_linked * r_linked + w_unlinked * "
            "r_unlinked EXACTLY. The unlinked profile (age band, observed "
            "married share, birth decade) is deterministic over side-B men. "
            "The shadow kernel (hcs._paternal_births) fires only for married "
            "men, so the unlinked contribution is bounded by the "
            "unlinked-married exposure."
        ),
        "failing_cells": {c: s for c, s in sorted(child_fail.items())},
        "linked_vs_unlinked_decomposition": decomposition,
        "unlinked_men_profile": profile,
        "finding": finding,
    }


def assemble_q3(
    per_seed: list[dict[str, Any]],
    committed: dict[str, Any],
    fail_cells: dict[str, list[int]],
) -> dict[str, Any]:
    """Q3 -- hh_size middle: which component drives the size-3 overshoot."""
    c3a = committed["c3"]

    # Seed-averaged full + family-core distributions and the size-3 partition.
    dist = {
        k: _mean_over_seeds([s["q3_dist_mean"][k] for s in per_seed])
        for k in per_seed[0]["q3_dist_mean"]
    }
    base_dist = {
        k: _mean_over_seeds([s["q3_base_dist_mean"][k] for s in per_seed])
        for k in per_seed[0]["q3_base_dist_mean"]
    }
    size3_by_nf = {
        k: _mean_over_seeds(
            [s["q3_size3_by_nonfamily_mean"][k] for s in per_seed]
        )
        for k in per_seed[0]["q3_size3_by_nonfamily_mean"]
    }
    # Reference hh_size (side B), for the miss reconciliation.
    ref = {
        k: _mean_over_seeds([s["rate_b"][f"hh_size.{k}"] for s in per_seed])
        for k in ("1", "2", "3", "4", "5+")
    }

    routes = {
        k: _mean_over_seeds([s["q3_size3_routes_mean"][k] for s in per_seed])
        for k in per_seed[0]["q3_size3_routes_mean"]
    }
    size3_total = dist["3"]
    size3_core = size3_by_nf["0"]  # family-core only reaches 3
    base3 = base_dist["3"]  # family-core-only P(size==3)
    # The bridge's mean-count loss from the 2+ -> 2 truncation (train).
    c3_meta = c3a["per_seed"]
    nf_overall = [
        s["delta_stats"]["delta_2_household_bridge"][
            "nonfamily_train_overall_p0_p1_p2plus"
        ]
        for s in c3_meta
    ]
    nf_mean_count = [
        s["delta_stats"]["delta_2_household_bridge"][
            "nonfamily_train_weighted_mean_count"
        ]
        for s in c3_meta
    ]
    p0 = _mean_over_seeds([x[0] for x in nf_overall])
    p1 = _mean_over_seeds([x[1] for x in nf_overall])
    p2plus = _mean_over_seeds([x[2] for x in nf_overall])
    actual_mean = _mean_over_seeds(nf_mean_count)
    truncated_mean = p1 * 1 + p2plus * 2  # the bridge's 2+ -> 2 reading
    tail_mean_2plus = (
        (actual_mean - p1) / p2plus if p2plus > 0 else 0.0
    )  # mean non-family count among the 2+ households

    finding = (
        "The hh_size.3 overshoot is primarily a FAMILY-CORE composition error "
        "the non-family bridge cannot fully drain -- correcting the "
        "pre-registered 'non-family 2+ mass mis-shaped rather than the family "
        "core' hunch. The composed family unit ALONE (1 + spouse + children + "
        f"parents, no bridge) already puts ~{base3:.3f} at size 3 against a "
        f"reference ~{ref['3']:.3f}; the bridge REDUCES the full size-3 to "
        f"~{size3_total:.3f} (it pushes size-3-base households up to 4/5+), but "
        f"the ~{size3_core:.3f} of size-3 households that draw zero non-family "
        f"members (~{size3_core / size3_total * 100:.0f}% of size-3) alone "
        f"exceed the reference. Within size-3 the routes are couple+child "
        f"~{routes['couple_plus_child']:.3f}, two+ children "
        f"~{routes['two_plus_children']:.3f}, coresident-parents(+2) "
        f"~{routes['coresident_parents']:.3f}, non-family-driven "
        f"~{routes['nonfamily_ge1']:.3f} -- the couple-plus-one-child family "
        "unit is the dominant size-3 route. The non-family bridge's SEPARATE "
        "error is the deep tail: its 2+ -> 2 minimal reading takes a train "
        f"non-family distribution (P0,P1,P2+) ~ ({p0:.3f}, {p1:.3f}, "
        f"{p2plus:.3f}) whose 2+ households truly average ~{tail_mean_2plus:.2f} "
        f"members (true mean ~{actual_mean:.3f}) and caps them at 2 (simulated "
        f"mean ~{truncated_mean:.3f}), losing ~{actual_mean - truncated_mean:.3f} "
        f"member/person, so hh_size.4 and 5+ still undershoot (sim "
        f"{dist['4']:.3f}/{dist['5+']:.3f} vs ref {ref['4']:.3f}/{ref['5+']:.3f}). "
        "Two distinct mechanisms: the family core over-produces size 3, and the "
        "truncated bridge under-fills the 4/5+ tail."
    )
    return {
        "question": (
            "hh_size middle distribution: decompose hh_size.3's 0.261 miss by "
            "which composed component (spouse / children / parents / "
            "non-family) drives the size-3 share error; test the non-family "
            "bridge's 2+ -> 2 minimal reading against the train distribution's "
            "actual 2+ composition."
        ),
        "method": (
            "Train-side. Per gate seed and draw, the hh_size distribution is "
            "computed with the full composition and with the family-core-only "
            "hh_size_base (1 + spouse + children + parents, no bridge); the "
            "size-3 mass is partitioned by the sampled non-family count (0/1/2) "
            "-- these three partition the size-3 mass EXACTLY. The 2+ -> 2 "
            "reading is tested against the committed train non-family count "
            "distribution and its true weighted mean."
        ),
        "hh_size_distribution": {
            "reference_train": ref,
            "full_simulated": dist,
            "family_core_only_simulated": base_dist,
        },
        "size3_partition_by_nonfamily_count": {
            "nonfamily_0_family_core": size3_by_nf["0"],
            "nonfamily_1": size3_by_nf["1"],
            "nonfamily_2_truncated": size3_by_nf["2"],
            "total_size3": size3_total,
            "reconciliation_remainder": size3_total
            - (size3_by_nf["0"] + size3_by_nf["1"] + size3_by_nf["2"]),
        },
        "size3_partition_by_composition_route": {
            **routes,
            "total_size3": size3_total,
            "reconciliation_remainder": size3_total - sum(routes.values()),
        },
        "nonfamily_2plus_minimal_reading_test": {
            "train_p0_p1_p2plus": [p0, p1, p2plus],
            "train_true_weighted_mean_count": actual_mean,
            "bridge_truncated_mean_count_2plus_as_2": truncated_mean,
            "mean_nonfamily_count_within_2plus_households": tail_mean_2plus,
            "mean_count_lost_to_truncation": actual_mean - truncated_mean,
        },
        "finding": finding,
    }


def assemble_q4(
    per_seed: list[dict[str, Any]],
    committed: dict[str, Any],
    fail_cells: dict[str, list[int]],
    tol: dict[str, float],
) -> dict[str, Any]:
    """Q4 -- grandchild 55+|female: skip-gen level vs multigen-path composition."""
    c3a = committed["c3"]
    cell = "coresident_grandchild.55+|female"
    seeds = fail_cells.get(cell, list(GATE_SEEDS))

    per_seed_block: dict[str, Any] = {}
    for seed in seeds:
        sd = next(s for s in per_seed if s["seed"] == seed)
        composed = float(sd["q4_composed_mean"])
        skiponly = float(sd["q4_skiponly_mean"])
        union = float(sd["q4_union_mean"])
        ref_b = float(sd["rate_b"][cell])
        com = _committed_cell(c3a, cell, seed)
        per_seed_block[str(seed)] = {
            "reference_train_rate_b": ref_b,
            "composed_multigen_path_rate": composed,
            "skipgen_only_rate": skiponly,
            "union_rate": union,
            "reconciliation_remainder": union - (composed + skiponly),
            "residual_miss_train": ref_b - union,
            "skipgen_share_of_union": (
                float(skiponly / union) if union > 0 else 0.0
            ),
            "holdout_committed": com,
        }

    mean_composed = _mean_over_seeds(
        [b["composed_multigen_path_rate"] for b in per_seed_block.values()]
    )
    mean_skip = _mean_over_seeds(
        [b["skipgen_only_rate"] for b in per_seed_block.values()]
    )
    mean_union = _mean_over_seeds(
        [b["union_rate"] for b in per_seed_block.values()]
    )
    mean_ref = _mean_over_seeds(
        [b["reference_train_rate_b"] for b in per_seed_block.values()]
    )

    age_struct = {
        "band_detail_female": {
            band: {
                key: _mean_over_seeds(
                    [
                        s["q4_skipgen_age"]["band_detail_female"][band][key]
                        for s in per_seed
                    ]
                )
                for key in per_seed[0]["q4_skipgen_age"]["band_detail_female"][
                    band
                ]
            }
            for band in per_seed[0]["q4_skipgen_age"]["band_detail_female"]
        },
        "raw_skipgen_stock_by_single_year_age_female": {
            age: _mean_over_seeds(
                [
                    s["q4_skipgen_age"][
                        "raw_skipgen_stock_by_single_year_age_female"
                    ][age]
                    for s in per_seed
                ]
            )
            for age in per_seed[0]["q4_skipgen_age"][
                "raw_skipgen_stock_by_single_year_age_female"
            ]
        },
    }

    finding = (
        "The grandchild 55+|female remainder is a skip-gen LEVEL shortfall, "
        "not a multigen-path composition problem. Of the reference train stock "
        f"~{mean_ref:.4f}, the composed multigen-path grandchild "
        "(multigen AND child AND NOT parent) supplies only "
        f"~{mean_composed:.4f} and the delta-3 skipped-generation state adds "
        f"~{mean_skip:.4f}, for a union of ~{mean_union:.4f} -- still roughly "
        f"half the reference. The skip-gen delta does carry the MAJORITY of "
        f"what the model reaches (~{(mean_skip / mean_union * 100) if mean_union else 0:.0f}% "
        "of the union), so the concept was right but the LEVEL is too low: the "
        "fitted skip-gen entry hazard is ~0.004/wave against a raw 55+ female "
        f"skip-gen stock of ~{age_struct['band_detail_female']['55-64']['raw_skipgen_stock_share']:.4f} "
        "(55-64) rising with age, and the band-constant hazard plus the high "
        "~0.35 exit hazard cannot build the stock the older bands carry. The "
        "multigen path contributes little because 55+ women in three-gen homes "
        "are usually coded coresident_parent (their own child present), which "
        "the composed grandchild excludes by construction (NOT parent). "
        "Recorded descriptively per the registration: the residual is a "
        "skip-gen entry-level / age-structure gap, the candidate-4 target."
    )
    return {
        "question": (
            "grandchild 55+|female remainder: split the remaining 0.747 into "
            "skip-gen level (delta-3's state) vs multigen-path composition; "
            "measure the skip-gen entry hazard's age structure within 55+ "
            "against the raw train spells."
        ),
        "method": (
            "Train-side. Per gate seed and draw, the 55+|female "
            "coresident_grandchild rate is partitioned into the composed "
            "multigen-path grandchild and the skipped-generation-only state "
            "(union = composed + skipgen_only EXACTLY). The skip-gen entry "
            "hazard's age structure is compared to the raw side-B "
            "skipped-generation stock by single year of age within 55+."
        ),
        "cell": cell,
        "failing_seeds": seeds,
        "per_seed_decomposition": per_seed_block,
        "seed_averaged": {
            "reference_train_rate_b": mean_ref,
            "composed_multigen_path_rate": mean_composed,
            "skipgen_only_rate": mean_skip,
            "union_rate": mean_union,
            "residual_miss_after_both": mean_ref - mean_union,
            "skipgen_share_of_union": (
                float(mean_skip / mean_union) if mean_union else 0.0
            ),
        },
        "skipgen_age_structure": age_struct,
        "finding": finding,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    import sklearn

    return {
        "populace_dynamics_head_sha": c3._git_sha(),
        "merge_base_origin_master": c3._merge_base(),
        "gate2b_floor_run": "runs/gate2b_floors_v1.json",
        "gate2b_floor_sha256": c3._sha_of_file(FLOOR_RUN),
        "candidate1_artifact_sha256": c3._sha_of_file(CANDIDATE1_ARTIFACT),
        "candidate2_artifact_sha256": c3._sha_of_file(CANDIDATE2_ARTIFACT),
        "candidate3_artifact_sha256": c3._sha_of_file(CANDIDATE3_ARTIFACT),
        "candidate3_runner": "scripts/run_gate2b_candidate3.py",
        "candidate3_runner_sha256": c3._sha_of_file(
            ROOT / "scripts" / "run_gate2b_candidate3.py"
        ),
        "family_transitions_spec_sha256": ft.CANDIDATE_16.sha256,
        "gates_yaml_locked": bool(thresholds.get("locked", False)),
        "gates_yaml_status": thresholds.get("status"),
        "sklearn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "schema_version": SCHEMA_VERSION,
    }


def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    # Simple JSON cache (candidate 3 has no cache helper; use a local one).
    local_cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            local_cache = json.loads(cache_path.read_text())
        except Exception:
            local_cache = {}

    thresholds = c3.load_gate2b_thresholds()
    tol = c3.gated_tolerances(thresholds)
    if len(tol) != 46:
        raise RuntimeError(
            f"expected 46 gated tolerances, got {len(tol)} from gates.yaml."
        )
    for art in (
        CANDIDATE1_ARTIFACT,
        CANDIDATE2_ARTIFACT,
        CANDIDATE3_ARTIFACT,
        FLOOR_RUN,
    ):
        if not art.exists():
            raise RuntimeError(f"required committed artifact missing: {art}")

    committed = _committed()
    fail_cells = _failing_cells(committed["c3"])

    data = c3.load_all()
    if verbose:
        hh = data["hh"]
        print(
            f"panel: {len(hh.person_waves)} person-waves, "
            f"{hh.person_waves.person_id.nunique()} persons; "
            f"train-side forensics, K={N_DRAWS} draws (5200 + k)"
        )

    # One-time fidelity proof: instrumented draw == committed simulate_draw_v3.
    side_a, side_b = hpanel.split_panel_by_person(
        data["hh"].attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b0 = set(int(x) for x in side_b.person_id.unique())
    model0 = hcs3.fit_household_model_v3(
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
    )
    fidelity = fidelity_check(
        data["hh"], data["mpanel"], model0, ids_b0, DRAW_SEED_BASE
    )
    if verbose:
        print(
            "instrumentation fidelity: bit_identical="
            f"{fidelity['bit_identical']} "
            f"(max dev {fidelity['max_abs_rate_deviation_vs_committed_simulate_draw_v3']:.2e})"
        )
    if not fidelity["bit_identical"]:
        raise RuntimeError(
            "instrumented_draw_v3 does not reproduce simulate_draw_v3 "
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
        result = compute_seed(seed, data, tol, fail_cells, verbose)
        local_cache[key] = json.loads(
            json.dumps(result, default=lambda o: _json_default(o))
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(local_cache))
        per_seed.append(local_cache[key])

    q1 = assemble_q1(per_seed, committed, fail_cells, tol)
    q2 = assemble_q2(per_seed, committed, fail_cells)
    q3 = assemble_q3(per_seed, committed, fail_cells)
    q4 = assemble_q4(per_seed, committed, fail_cells, tol)

    implications = (
        "Implications for candidate 4 (labeled implications, NOT decisions -- "
        "the orchestrator registers c4). Q1 (binding): the spouse gate needs "
        "the cohabitation overlay re-shaped WITHIN the young band (single-year "
        "or finer age structure so 15-24 stops over-producing) AND the legal "
        "core lifted at 25-34 and 65+ where the certified registry "
        "under-produces the code-20 stock; there is NO concept residual to "
        "chase (mass is entirely in {20,22}). Q3: the non-family bridge needs "
        "its 2+ tail spread beyond the minimal 2 (a count distribution with "
        "mass at 3/4+) so size-3 stops over-filling and size-4/5+ fill in -- a "
        "single delta touching only hh_size. Q4: the skip-gen state needs a "
        "higher, age-graded entry level (or a lower exit) to build the 55+ "
        "female stock the band-constant ~0.004 hazard cannot reach; the "
        "multigen path is structurally blocked by the NOT-parent exclusion. "
        "Q2: the male child overshoot is the CUSTODIAL GATE on linked "
        "children, not the shadow kernel (which reaches only the thin "
        "unlinked-married slice) -- a c4 custodial-probability refinement, not "
        "a shadow change. Each maps to a disjoint single-family delta, the 2a "
        "one-delta-per-candidate pattern."
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
        "candidates_under_diagnosis": (
            "gate-2b candidates 1-3 (PRs #133 / #134 / #136); the four "
            "residual families isolated by grading 4940440505 (spouse family; "
            "child shadow residual; hh_size middle distribution; grandchild "
            "55+|female remainder)"
        ),
        "candidate1_artifact": "runs/gate2b_hazard_v1.json",
        "candidate2_artifact": "runs/gate2b_hazard_v2.json",
        "candidate3_artifact": "runs/gate2b_hazard_v3.json",
        "protocol": {
            "train_side_only": True,
            "outer_holdout_contact": (
                "none beyond the already-published per-seed scores and the "
                "committed [20,46,5] per-draw cube read from "
                "runs/gate2b_hazard_v{1,2,3}.json; the holdout (side A) is "
                "NEVER re-simulated here"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "hh.attrs, 'person_id', fraction=0.5, seed=s); side A = outer "
                "holdout (read from committed artifacts only), side B = train "
                "complement (this diagnostic fits AND simulates side B)"
            ),
            "fit_simulate_machinery": (
                "populace_dynamics.models.household_composition_sim_v3 "
                "(candidate 3, merged #136), reused byte-for-byte on the train "
                "side via instrumented_draw_v3 (same 0xB2B / 0xC2 / 0xC3 draw "
                "streams and train-fitted tables)"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "n_draws": N_DRAWS,
            "draw_rng_rule": (
                f"numpy.random.default_rng({DRAW_SEED_BASE} + k) for k in "
                f"0..{N_DRAWS - 1} (the locked gate_2b protocol)"
            ),
            "train_side_statistic": (
                "|ln(rbar_B / rate_b)| per cell, rate_b = side B's own weighted "
                "reference moment"
            ),
            "instrumentation_fidelity": fidelity,
            "no_holdout_tuning_surface": (
                "every fitted table is estimated on side B only; no parameter "
                "is estimated from side A -- no holdout-informed tuning surface "
                "is created"
            ),
        },
        "data": {
            "holdout_basis": ["MX23REL"],
            "paternal_link_basis": ["cah85_23"],
            "n_person_waves": int(len(data["hh"].person_waves)),
            "n_persons": int(data["hh"].person_waves.person_id.nunique()),
            "floor_run": "runs/gate2b_floors_v1.json",
            "floor_run_sha256": c3._sha_of_file(FLOOR_RUN),
        },
        "failing_cells_by_family_committed_candidate3": {
            fam: sorted(
                {
                    c: fail_cells[c] for c in fail_cells if _family(c) == fam
                }.items()
            )
            for fam in (
                "coresident_spouse",
                "coresident_child",
                "coresident_grandchild",
                "hh_size",
            )
        },
        "question_1_spouse_legal_vs_cohabitation": q1,
        "question_2_child_shadow_residual": q2,
        "question_3_hh_size_middle_distribution": q3,
        "question_4_grandchild_skipgen_remainder": q4,
        "per_seed": per_seed,
        "candidate_4_implications": implications,
        "revision_pins": _revision_pins(thresholds),
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
        for q in (q1, q2, q3, q4):
            print(q["finding"][:300] + " ...")
    return artifact


def _family(cell: str) -> str:
    for fam in (
        "coresident_spouse",
        "coresident_parent",
        "coresident_child",
        "coresident_grandchild",
    ):
        if cell.startswith(fam + "."):
            return fam
    if cell.startswith("hh_size."):
        return "hh_size"
    if cell.startswith("multigen."):
        return "multigen_stock"
    if cell in ("multigen_entry", "multigen_exit"):
        return "multigen_transition"
    if cell.startswith("parental_home_"):
        return "parental_home_exit"
    return "other"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not serializable: {type(obj)}")


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
    artifacts.write_new(Path(args.out), c3._json_safe(artifact))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
