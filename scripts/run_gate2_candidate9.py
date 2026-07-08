"""Gate-2 candidate 9 (run 1): candidate 8 + two named deltas.

The NINTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4914111252 (``SPEC_REGISTRATION``): candidate 8's frozen spec
(comment 4912995860) verbatim EXCEPT TWO deltas registered from the gate-2
forensics (#94, ``runs/gate2_forensics_v1.json``). One-shot; no constant
moves after the registration comment; published REGARDLESS of verdict.

The two deltas vs candidate 8 (everything else byte-identical)
--------------------------------------------------------------
**Delta 1 -- observed undatable-marriage lifetime-count initial state
(Q1's exact residual).** Each holdout person's simulated lifetime-marriage
count initializes at their OBSERVED count of marriages that are NOT
representable as datable in-exposure episodes -- the forensics Q1 residual at
per-person resolution -- and then accumulates the simulated datable
transitions exactly as candidate 8 does. That observed residual is
``R[i] = n_marriages[i] - (# datable first_marriage/remarriage transition
events for person i)``: the undatable-start, out-of-window/underage,
undatable-dissolution, separation-origin and MH-count-excess marriages the
forensics :func:`gate2_forensics.reference_residual_breakdown` enumerates
(reused for the reconciliation record). It aggregates EXACTLY to the
forensics reference residual (``mean_lifetime_marriages - in_exposure``,
0.116/0.090 per person male/female) and reconciles to remainder 0.0 -- the
structural deficit the hazard model cannot generate. It is an OBSERVED
initial state (per protocol, like the entry marital state), not a fitted or
simulated quantity, so the hazards are UNTOUCHED and it perturbs NO RNG draw
(a pure post-assembly count add).

**Delta 2 -- unsmoothed low-parity peak fertility (Q2's margin).** The
fertility hazards for parities 0->1 and 1->2 (parity bands 0 and 1) use the
EXACT empirical single-year rate (weighted births / weighted woman-years, no
kernel) at ages 22-38 wherever the single-year cell exposure is at least 200
weighted person-years; the pre-registered triangular kernel is retained
everywhere else (all other ages, all other parities, and any low-exposure
cell in the window). This removes the triangular kernel's peak attenuation
exactly where Q2 locates the broad low-parity under-production (58% of the
0.122 child/woman deficit). Higher parities (2, 3+) are unchanged.

Structural consequences (verified in the artifact)
--------------------------------------------------
Delta 1 changes ONLY ``mean_lifetime_marriages|{male,female}`` (a count add
on the person attribute the moment reads); it touches no marital-state
trajectory, no person-year, and no draw. Delta 2 changes the fertility
THRESHOLD only: the per-year uniform blocks (``rng.random(n_active)`` then
``rng.random(n_fertile)``) are drawn in the SAME order and size as candidate
8 -- the fertility draw count is parity-independent -- so the marriage draw
stream is byte-identical and marital transitions are decoupled from
fertility. Therefore EVERY gated cell that is neither a fertility cell
(``asfr.*``, ``completed_fertility.*``, moved by delta 2) nor a lifetime-
marriage count (moved by delta 1) is BYTE-IDENTICAL to candidate 8; the
``identity_vs_candidate8`` block attests this.

This runner IMPORTS candidate 8's machinery (which chains candidates 6/5/1
and candidate 7's estimator): :func:`fit_components` starts from
``candidate8.fit_components`` and swaps ONLY the fertility table (delta 2);
:func:`simulate_holdout` calls ``candidate8.simulate_holdout`` unchanged and
then adds the observed residual (delta 1). The scoring path, precheck, and
verdict assembly are candidate 1's, imported unchanged.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed holdout-id
sha256, BEFORE any candidate is simulated. A SECOND hard gate (candidate 9's,
train-side, licensed) then verifies the delta-1 reconciliation: per person
``R + datable == n_marriages`` to remainder 0.0, and ``R`` aggregated over the
conditioned train persons equals the forensics reference residual and its
five-bucket decomposition. Any mismatch is a hard stop. Run ONCE; publish
REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate9.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 8 supplies the machinery this build deltas twice: its rescaled
# order-split remarriage, and -- transitively via candidates 6/5/1 -- the
# surviving-spouse widowhood level, the committed NCHS betas, the knot-at-22
# first-marriage spline, divorce, the spousal-gap draw, the single-year
# triangular-kernel fertility, the vectorised simulation, the precheck, the
# verdict assembly and the report-only summary. Only fertility is re-fit
# (delta 2) and only the marriage-count initial state is added (delta 1); the
# forensics module supplies the reused Q1 residual accounting.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import gate2_forensics as forensics  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate8 as c8  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v9.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS_ARTIFACT = ROOT / "runs" / "gate2_forensics_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v9"
RUN_NAME = "gate2_hazard_v9"

#: This run's frozen-spec registration (issue #42, comment 4914111252).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4914111252"
)
#: The candidate-8 spec this build deltas twice (comment 4912995860).
CANDIDATE8_REGISTRATION = c8.SPEC_REGISTRATION
CANDIDATE6_REGISTRATION = c8.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c8.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c8.CANDIDATE1_REGISTRATION
#: The forensics diagnostic the deltas cite (#94, gate2_forensics_v1.json).
FORENSICS_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4913512779"
)

# --- Delta 2 dials (registration comment 4914111252). ---------------------
#: Parity bands whose single-year peak fertility uses the exact rate: 0 (the
#: 0->1 step) and 1 (the 1->2 step). Higher parities are unchanged.
FERT_EXACT_PARITY_BANDS = (0, 1)
#: The age window over which the exact single-year rate replaces the kernel.
FERT_EXACT_AGE_LO = 22
FERT_EXACT_AGE_HI = 38
#: The minimum single-year weighted person-year exposure for the exact rate;
#: below it (and outside the window / above parity 1) the kernel is kept.
FERT_EXACT_MIN_EXPOSURE = 200.0

#: The two named deltas (registration comment 4914111252).
DELTA_VS_CANDIDATE8 = (
    "TWO deltas vs candidate 8. (1) LIFETIME-MARRIAGE COUNT INITIAL STATE: "
    "each holdout person's simulated marriage count initializes at their "
    "OBSERVED count of marriages not representable as datable in-exposure "
    "episodes (the Q1 residual R = n_marriages - datable first_marriage/"
    "remarriage transition events -- undatable start, out-of-window/underage, "
    "undatable dissolution, separation-origin and MH-count-excess marriages, "
    "the forensics reference_residual_breakdown categories), then accumulates "
    "the simulated datable transitions as candidate 8 does; an OBSERVED "
    "initial state per protocol, RNG-neutral, hazards untouched. (2) LOW-"
    "PARITY PEAK FERTILITY: parities 0->1 and 1->2 (parity bands 0,1) use the "
    "exact empirical single-year rate (no kernel) at ages 22-38 where the "
    "single-year cell exposure is >= 200 weighted person-years, kernel "
    "elsewhere; higher parities unchanged. Delta 2 moves only the fertility "
    "threshold (same draw order/size), so the marriage stream is byte-"
    "identical and every non-fertility, non-lifetime-marriage-count gated "
    "cell matches candidate 8 exactly"
)

# --- Frozen dials + pure helpers, reused (byte-identical; imported). -------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
EXACT_ATOL = c1.EXACT_ATOL
FERT_AGE_LO = c5.FERT_AGE_LO  # 15
FERT_AGE_HI = c5.FERT_AGE_HI  # 49
Components = c1.Components

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate9_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA 1: observed undatable-marriage lifetime-count initial state
# --------------------------------------------------------------------------
_RESIDUAL_CACHE: dict[int, dict[int, float]] = {}


def observed_residual_counts(
    panel: transitions.MaritalPanel,
) -> dict[int, float]:
    """Per-person observed residual marriage count (Q1's exact residual).

    ``R[i] = n_marriages[i] - (# datable in-exposure first_marriage and
    remarriage transition events for person i)`` on the reference panel: the
    OBSERVED marriages the simulation's datable episode construction cannot
    represent (undatable start/dissolution years, out-of-window/underage
    starts, separation-origin remarriages, and the ``n_marriages`` (MH) count
    exceeding the datable episode rows). This is the forensics Q1 residual at
    per-person resolution: weighted over the ever-married denominator it
    equals ``mean_lifetime_marriages - in_exposure`` exactly and decomposes
    into :func:`gate2_forensics.reference_residual_breakdown`'s five buckets
    (remainder 0.0). Non-negative and integer-valued on the real file. An
    OBSERVED initial state (like the entry marital state) -- read per person,
    not fitted, not simulated -- so no train/holdout leak and no draw touched.
    Cached per panel (split-independent).
    """
    key = id(panel)
    cached = _RESIDUAL_CACHE.get(key)
    if cached is not None:
        return cached
    attrs = panel.attrs
    starts = panel.events[
        panel.events["transition"].isin(("first_marriage", "remarriage"))
    ]
    ev_count = starts.groupby("person_id").size()
    datable = (
        attrs["person_id"].map(ev_count).fillna(0).astype("int64").to_numpy()
    )
    n_marr = attrs["n_marriages"].to_numpy(dtype="float64")
    residual = n_marr - datable
    # Persons with an undefined observed marriage count (NaN n_marriages) have
    # no observed residual to initialize from, so they contribute 0; they are
    # excluded from mean_lifetime_marriages by its n_marriages >= 1
    # conditioning regardless (NaN >= 1 is False).
    residual = np.where(np.isnan(residual), 0.0, residual)
    out = {
        int(pid): float(r)
        for pid, r in zip(attrs["person_id"].to_numpy(), residual, strict=True)
    }
    _RESIDUAL_CACHE[key] = out
    return out


def _delta1_reconciliation(
    panel: transitions.MaritalPanel,
    mh_records: pd.DataFrame,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Train-side Q1 reconciliation of the delta-1 residual (a hard gate).

    Verifies, BEFORE the one-shot, that the observed residual added to the
    datable count reproduces the forensics Q1 reconciliation to remainder 0.0
    and is consistent with how ``transitions.mean_lifetime_marriages``
    reference counts:

    * per person over the whole panel, ``R + datable == n_marriages`` exactly
      (``per_person_identity_max_abs_residual`` == 0.0): the residual is the
      exact structural complement of the datable in-exposure event count;
    * for each gate seed's TRAIN half and each sex, ``R`` weighted over the
      conditioned ever-married denominator equals the forensics reference
      residual ``mean_lifetime_marriages - in_exposure`` and equals the sum of
      :func:`gate2_forensics.reference_residual_breakdown`'s five buckets
      (``reconciliation_remainder`` == 0.0 to float precision).
    """
    residual = observed_residual_counts(panel)
    attrs = panel.attrs
    starts = panel.events[
        panel.events["transition"].isin(("first_marriage", "remarriage"))
    ]
    ev_count = starts.groupby("person_id").size()
    datable = (
        attrs["person_id"].map(ev_count).fillna(0).astype("int64").to_numpy()
    )
    R_all = attrs["person_id"].map(residual).to_numpy(dtype="float64")
    n_marr_all = attrs["n_marriages"].to_numpy(dtype="float64")
    # The identity R + datable == n_marriages holds over persons with a
    # DEFINED observed marriage count (NaN-n_marriages persons carry R == 0 and
    # never enter mean_lifetime_marriages).
    defined = ~np.isnan(n_marr_all)
    identity = (R_all + datable - n_marr_all)[defined]
    per_person_identity_max = (
        float(np.max(np.abs(identity))) if identity.size else 0.0
    )
    residual_min = float(np.min(R_all[defined])) if defined.any() else 0.0
    n_undefined_n_marriages = int((~defined).sum())

    per_seed_rows: list[dict[str, Any]] = []
    max_abs_remainder = 0.0
    for seed in seeds:
        _side_a, side_b = hpanel.split_panel_by_person(
            panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids_b = set(int(x) for x in side_b.person_id.unique())
        by_sex: dict[str, Any] = {}
        for sex in transitions.SEXES:
            pc = forensics.pathway_cells(panel, ids_b, sex)
            ref_resid = (
                pc["mean_lifetime_marriages"]
                - pc["in_exposure_marriages_per_person"]
            )
            cond = forensics._conditioned_ever_married(attrs, ids_b, sex)
            w = cond["weight"].to_numpy(dtype="float64")
            r_cond = cond["person_id"].map(residual).to_numpy(dtype="float64")
            den = float(w.sum())
            r_agg = float((w * r_cond).sum() / den) if den > 0 else 0.0
            buckets = forensics.reference_residual_breakdown(
                mh_records, attrs, ids_b, sex
            )
            bucket_sum = float(sum(buckets.values()))
            remainder_vs_buckets = ref_resid - bucket_sum
            remainder_vs_agg = r_agg - ref_resid
            max_abs_remainder = max(
                max_abs_remainder,
                abs(remainder_vs_buckets),
                abs(remainder_vs_agg),
            )
            by_sex[sex] = {
                "residual_agg_per_person": r_agg,
                "forensics_reference_residual": ref_resid,
                "forensics_bucket_sum": bucket_sum,
                "remainder_agg_minus_reference": remainder_vs_agg,
                "remainder_reference_minus_buckets": remainder_vs_buckets,
                "buckets": buckets,
            }
        per_seed_rows.append({"seed": seed, "by_sex": by_sex})

    reconciled = bool(
        per_person_identity_max <= EXACT_ATOL and max_abs_remainder <= 1e-9
    )
    return {
        "note": (
            "delta-1 reconciliation (train-side, licensed): the observed "
            "residual R[i] = n_marriages - datable in-exposure marriage events "
            "added to the simulated datable count reproduces the forensics Q1 "
            "reconciliation to remainder 0.0 and matches how "
            "transitions.mean_lifetime_marriages reference counts "
            "(n_marriages)"
        ),
        "per_person_identity_rule": "R + datable == n_marriages (per person)",
        "per_person_identity_max_abs_residual": per_person_identity_max,
        "n_persons_undefined_n_marriages": n_undefined_n_marriages,
        "residual_min_over_defined_persons": residual_min,
        "residual_nonnegative": bool(residual_min >= 0.0),
        "aggregate_reconciliation_max_abs_remainder": float(max_abs_remainder),
        "per_seed": per_seed_rows,
        "reconciled": reconciled,
    }


# --------------------------------------------------------------------------
# DELTA 2: unsmoothed low-parity peak fertility
# --------------------------------------------------------------------------
def _fertility_single_year_counts(
    panel: transitions.MaritalPanel,
    birth_records: pd.DataFrame,
    train_ids: set[int],
    birth_decade: pd.Series,
) -> tuple[
    dict[tuple[int, int, int], float], dict[tuple[int, int, int], float]
]:
    """The exposure/numerator candidate 5 forms internally, reconstructed.

    A byte-identical reconstruction of the single-year weighted denominator
    (``den``: train woman-years by age x parity_band x cohort) and numerator
    (``num``: mother-weighted births by mother_age x parity_band x cohort)
    that :func:`candidate5.fit_fertility_single_year` builds before its kernel
    convolution -- the SAME train women aged 15-49, running parity capped at
    3, births censored at the mother's censor year, and the SAME
    :func:`candidate1._parity_vec`. Returned so the delta-2 exact rate over
    the low-parity peak is ``num / den`` on exactly the cells the kernel
    otherwise smooths.
    """
    py = panel.person_years
    attrs = panel.attrs
    women_ids = set(attrs[attrs["sex"] == "female"]["person_id"]) & train_ids
    lo, hi = FERT_AGE_LO, FERT_AGE_HI
    wy = py[
        py["person_id"].isin(women_ids) & (py["age"] >= lo) & (py["age"] <= hi)
    ][["person_id", "year", "age", "weight"]].copy()

    be = g2f.births.birth_events(birth_records)
    be = be[
        (be["record_type"] == "birth")
        & be["parent_person_id"].isin(women_ids)
        & be["birth_year"].notna()
    ].copy()
    be = be.rename(columns={"parent_person_id": "person_id"})
    be["birth_year"] = be["birth_year"].astype("int64")
    births_by = {
        int(p): np.sort(g["birth_year"].to_numpy())
        for p, g in be.groupby("person_id")
    }

    wy = wy.reset_index(drop=True)
    wy["parity"] = c1._parity_vec(
        wy["person_id"].to_numpy(), wy["year"].to_numpy(), births_by
    )
    wy["decade"] = wy["person_id"].map(birth_decade).to_numpy()
    wy["parity_band"] = np.minimum(wy["parity"].to_numpy(), 3)

    attr_by = attrs.set_index("person_id")
    be["mother_birth"] = (
        be["person_id"].map(attr_by["birth_year"]).astype("float64")
    )
    be["mother_censor"] = (
        be["person_id"].map(attr_by["censor_year"]).astype("float64")
    )
    be["mother_age"] = be["birth_year"] - be["mother_birth"]
    be = be[
        (be["mother_age"] >= lo)
        & (be["mother_age"] <= hi)
        & (be["birth_year"] <= be["mother_censor"])
    ].reset_index(drop=True)
    be["decade"] = be["person_id"].map(birth_decade).to_numpy()
    be["weight"] = be["person_id"].map(attr_by["weight"]).to_numpy()
    be["parity"] = c1._parity_vec(
        be["person_id"].to_numpy(), be["birth_year"].to_numpy(), births_by
    )
    be["parity_band"] = np.minimum(be["parity"].to_numpy(), 3)

    den = (
        wy.groupby(["age", "parity_band", "decade"])["weight"].sum().to_dict()
    )
    num = (
        be.groupby(["mother_age", "parity_band", "decade"])["weight"]
        .sum()
        .to_dict()
    )
    return den, num


def fit_fertility_low_parity_exact(
    panel: transitions.MaritalPanel,
    birth_records: pd.DataFrame,
    train_ids: set[int],
    birth_decade: pd.Series,
) -> tuple[dict[tuple[int, int, int], float], list[tuple[int, int, int]]]:
    """Candidate 5's kernel fertility with the low-parity peak unsmoothed.

    Starts from :func:`candidate5.fit_fertility_single_year` (the triangular-
    kernel single-year table candidates 5/6/8 use, byte-identical), then
    OVERRIDES the parity-0 (0->1) and parity-1 (1->2) cells at ages 22-38
    whose single-year weighted exposure is at least 200 person-years with the
    EXACT empirical rate ``num / den`` -- the kernel is retained on every other
    cell (all other ages, parities 2/3+, and any low-exposure cell in the
    window). Because the override cells always have kernel denominator >= their
    raw exposure > 0, they exist in candidate 5's table already, so the key set
    is unchanged and only the low-parity peak values move. Returns the table
    and the sorted list of overridden ``(age, parity_band, decade)`` cells.
    """
    table = dict(
        c5.fit_fertility_single_year(
            panel, birth_records, train_ids, birth_decade
        )
    )
    den, num = _fertility_single_year_counts(
        panel, birth_records, train_ids, birth_decade
    )
    overridden: list[tuple[int, int, int]] = []
    for (age, pb, dec), exposure in den.items():
        exposure = float(exposure)
        if (
            pb in FERT_EXACT_PARITY_BANDS
            and FERT_EXACT_AGE_LO <= age <= FERT_EXACT_AGE_HI
            and exposure >= FERT_EXACT_MIN_EXPOSURE
        ):
            table[(age, pb, dec)] = float(num.get((age, pb, dec), 0.0)) / (
                exposure
            )
            overridden.append((int(age), int(pb), int(dec)))
    return table, sorted(overridden)


def _fertility_delta_meta(
    table: dict[tuple[int, int, int], float],
    overridden: list[tuple[int, int, int]],
) -> dict[str, Any]:
    """Compact provenance for the delta-2 low-parity exact-rate fertility."""
    by_pb = {0: 0, 1: 0}
    ages = set()
    decades = set()
    for age, pb, dec in overridden:
        by_pb[pb] += 1
        ages.add(age)
        decades.add(dec)
    return {
        "representation": (
            "single-year-of-age rates within parity x birth-decade cohort; "
            "parities 0->1 and 1->2 use the EXACT empirical rate (weighted "
            "births / weighted woman-years, no kernel) at ages 22-38 where the "
            "single-year cell exposure is >= 200 weighted person-years, the "
            "triangular kernel (bandwidth 3) elsewhere; higher parities "
            "unchanged (DELTA 2 vs candidate 8)"
        ),
        "exact_parity_bands": list(FERT_EXACT_PARITY_BANDS),
        "exact_age_range": [FERT_EXACT_AGE_LO, FERT_EXACT_AGE_HI],
        "exact_min_exposure_weighted_person_years": FERT_EXACT_MIN_EXPOSURE,
        "n_cells_total": len(table),
        "n_cells_exact_override": len(overridden),
        "n_exact_override_by_parity_band": {
            str(k): v for k, v in by_pb.items()
        },
        "override_ages": sorted(ages),
        "override_decades": sorted(decades),
    }


# --------------------------------------------------------------------------
# Fitted components (candidate 8's, with the delta-2 fertility swapped)
# --------------------------------------------------------------------------
def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Candidate 8's components with ONLY the fertility table delta'd.

    Starts from :func:`candidate8.fit_components` -- so the rescaled order-
    split remarriage, the surviving-spouse widowhood level, the committed NCHS
    betas, first marriage, divorce and the spousal-gap draw are byte-identical
    to candidate 8 by construction. Then DELTA 2 replaces the fertility table
    with the low-parity-exact single-year construction
    (:func:`fit_fertility_low_parity_exact`). DELTA 1 is NOT a fitted
    component: it is an observed initial state applied at simulation-assembly
    time (:func:`simulate_holdout`).
    """
    base = c8.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # DELTA 2: low-parity peak fertility uses exact single-year rates.
    attr_by = panel.attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")
    fertility, overridden = fit_fertility_low_parity_exact(
        panel, birth_records, train_ids, birth_decade
    )
    base.fertility = fertility
    base.meta["fertility_representation"] = (
        "single-year-of-age rates within parity (0/1/2/3+) x birth-decade "
        "cohort; parities 0->1 and 1->2 EXACT empirical single-year rate (no "
        "kernel) at ages 22-38 where exposure >= 200 weighted person-years, "
        "triangular kernel (bandwidth 3) elsewhere; higher parities unchanged "
        "(DELTA 2 vs candidate 8)"
    )
    base.meta["fertility_low_parity_exact"] = _fertility_delta_meta(
        fertility, overridden
    )
    base.meta["delta_vs_candidate8"] = DELTA_VS_CANDIDATE8
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 8's, with the delta-1 count add)
# --------------------------------------------------------------------------
def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 8's simulation, then the DELTA-1 count initial state.

    Calls :func:`candidate8.simulate_holdout` verbatim -- which uses the
    delta-2 fertility table via ``components.fertility`` and draws the SAME
    per-year uniform blocks in the same order and size as candidate 8 -- then
    applies DELTA 1: each holdout person's simulated lifetime-marriage count is
    initialized at their observed residual (:func:`observed_residual_counts`)
    and the simulated datable episode count is added on top. This is a pure
    post-assembly adjustment of ``sim_panel.attrs['n_marriages']`` (the person
    attribute ``mean_lifetime_marriages`` reads) -- it perturbs NO RNG draw and
    changes NO marital-state trajectory, person-year, or event; only the
    lifetime-marriage count moves.
    """
    sim_panel, sim_births = c8.simulate_holdout(
        panel, holdout_ids, components, sim_seed
    )
    residual = observed_residual_counts(panel)
    add = (
        sim_panel.attrs["person_id"]
        .map(residual)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    sim_panel.attrs["n_marriages"] = (
        sim_panel.attrs["n_marriages"].to_numpy(dtype="float64") + add
    )
    return sim_panel, sim_births


# --------------------------------------------------------------------------
# Per-seed scoring (candidate 1's, calling the candidate-9 fit + simulate)
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    floor: dict[str, Any],
    tol: dict[str, float],
    report_only: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """Fit side B, simulate side A, score every cell against rate_a.

    Identical to :func:`candidate1.score_seed` except it calls the candidate-9
    :func:`fit_components` (delta 2) and :func:`simulate_holdout` (delta 1).
    """
    t0 = time.time()
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    components = fit_components(
        panel, demo, death_records, mh_records, birth_records, order_map, ids_b
    )
    sim_panel, sim_births = simulate_holdout(
        panel, ids_a, components, SIM_SEED_BASE + seed
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    cand = transitions.reference_moments(
        sim_panel, sim_fert, ids_a, weighted=True
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]

    def score_cell(key: str) -> dict[str, Any]:
        rate_a = float(committed_cells[key]["rate_a"])
        r_cand = float(cand[key]["rate"])
        n_cand = int(cand[key]["n_events"])
        if r_cand > 0 and rate_a > 0:
            s = float(abs(math.log(r_cand / rate_a)))
        else:
            s = float("inf")
        return {
            "r_candidate": r_cand,
            "rate_a": rate_a,
            "n_events_candidate": n_cand,
            "log_ratio_abs": s if math.isfinite(s) else None,
            "score": s,
        }

    gated_cells: dict[str, Any] = {}
    n_gated_pass = 0
    for key in sorted(tol):
        rec = score_cell(key)
        rec["tolerance"] = float(tol[key])
        rec["pass"] = bool(rec["score"] <= tol[key])
        n_gated_pass += rec["pass"]
        gated_cells[key] = rec

    report_cells: dict[str, Any] = {}
    for key in sorted(report_only):
        rec = score_cell(key)
        rec["gated"] = False
        report_cells[key] = rec

    seed_pass = n_gated_pass == len(tol)
    elapsed = round(time.time() - t0, 1)
    if verbose:
        fails = [k for k, v in gated_cells.items() if not v["pass"]]
        print(
            f"seed {seed}: {n_gated_pass}/{len(tol)} gated pass "
            f"(seed_pass={seed_pass}); fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "sim_seed": SIM_SEED_BASE + seed,
        "component_meta": components.meta,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Registered modal + movement vs candidate 8
# --------------------------------------------------------------------------
#: The registered modal failure (comment 4914111252): with delta 1 removing
#: the count cells, the 75+ female widow STOCK is the binding residual risk.
REGISTERED_MODAL_CELL = "share_widowed.75+|female"
#: The cells the two deltas target (registration): the lifetime-marriage
#: counts (delta 1) and completed_fertility.c1970s (delta 2).
TARGETED_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
    "completed_fertility.c1970s",
)
#: The cells whose vs-candidate-8 movement is reported: the two count cells
#: (delta 1), the fertility target (delta 2), and the noise-dominated marital
#: cells the registration prices at fresh-draw clip probabilities.
MOVEMENT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
    "completed_fertility.c1970s",
    "asfr.15-19",
    "asfr.20-24",
    "share_divorced.45-54|female",
    "share_widowed.75+|female",
    "widowhood.75+|female",
)


def _cell_moved_by_delta(cell: str) -> bool:
    """Cells the two deltas can move: fertility (delta 2) and the lifetime-
    marriage counts (delta 1). Every other cell is byte-identical to c8."""
    return (
        cell.startswith("asfr.")
        or cell.startswith("completed_fertility.")
        or cell.startswith("mean_lifetime_marriages")
    )


def _candidate8_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-8 -> candidate-9 movement for the tracked cells."""
    if not CANDIDATE8_ARTIFACT.exists():
        return {"available": False}
    a8 = json.loads(CANDIDATE8_ARTIFACT.read_text())
    by8 = {s["seed"]: s for s in a8["per_seed"]}
    by9 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by9)
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in MOVEMENT_CELLS:
        c8_scores = {s: by8[s]["gated_cells"][cell]["score"] for s in seeds}
        c9_scores = {s: by9[s]["gated_cells"][cell]["score"] for s in seeds}
        c8_rate = {
            s: by8[s]["gated_cells"][cell]["r_candidate"] for s in seeds
        }
        c9_rate = {
            s: by9[s]["gated_cells"][cell]["r_candidate"] for s in seeds
        }
        c8_pass = sum(by8[s]["gated_cells"][cell]["pass"] for s in seeds)
        c9_pass = sum(by9[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "tolerance": by9[seeds[0]]["gated_cells"][cell]["tolerance"],
            "candidate8_per_seed_score": c8_scores,
            "candidate9_per_seed_score": c9_scores,
            "candidate8_per_seed_r_candidate": c8_rate,
            "candidate9_per_seed_r_candidate": c9_rate,
            "candidate8_mean_score": float(np.mean(list(c8_scores.values()))),
            "candidate9_mean_score": float(np.mean(list(c9_scores.values()))),
            "candidate8_n_seeds_pass": c8_pass,
            "candidate9_n_seeds_pass": c9_pass,
        }
    return out


def _identity_vs_candidate8(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Non-fertility, non-lifetime-marriage gated cells match candidate 8.

    Delta 2 moves only the fertility threshold (same draw order/size), so the
    marriage draw stream is byte-identical to candidate 8, and delta 1 is an
    RNG-neutral count add on ``n_marriages`` alone. So every gated cell that is
    neither ``asfr.*``/``completed_fertility.*`` (delta 2) nor
    ``mean_lifetime_marriages|*`` (delta 1) must carry candidate 8's exact
    ``r_candidate``. Records the maximum per-cell deviation as the byte-
    identity attestation, and the fertility / lifetime-marriage movement as the
    delta-active attestation.
    """
    if not CANDIDATE8_ARTIFACT.exists():
        return {"available": False}
    a8 = json.loads(CANDIDATE8_ARTIFACT.read_text())
    by8 = {s["seed"]: s for s in a8["per_seed"]}
    max_dev = 0.0
    n_cells = 0
    fert_moved = 0
    count_moved = 0
    fert_max_move = 0.0
    count_max_move = 0.0
    for s in per_seed:
        seed = s["seed"]
        for cell, rec in s["gated_cells"].items():
            c8rec = by8[seed]["gated_cells"][cell]
            move = abs(rec["r_candidate"] - c8rec["r_candidate"])
            if _cell_moved_by_delta(cell):
                if cell.startswith("mean_lifetime_marriages"):
                    count_moved += move > EXACT_ATOL
                    count_max_move = max(count_max_move, move)
                else:
                    fert_moved += move > EXACT_ATOL
                    fert_max_move = max(fert_max_move, move)
            else:
                max_dev = max(max_dev, move)
                n_cells += 1
    return {
        "available": True,
        "note": (
            "delta 2 moves only the fertility threshold (the per-year uniform "
            "blocks are drawn in the same order and size), so the marriage "
            "stream is byte-identical to candidate 8, and delta 1 is an RNG-"
            "neutral count add on n_marriages; every gated cell that is not a "
            "fertility cell (asfr.*, completed_fertility.*) or a lifetime-"
            "marriage count (mean_lifetime_marriages|*) is byte-identical to "
            "candidate 8"
        ),
        "n_cells_checked": n_cells,
        "max_abs_r_candidate_deviation_vs_candidate8": float(max_dev),
        "byte_identical": bool(max_dev <= EXACT_ATOL),
        "n_fertility_cell_movements": fert_moved,
        "max_abs_fertility_r_candidate_move": float(fert_max_move),
        "n_lifetime_marriage_cell_movements": count_moved,
        "max_abs_lifetime_marriage_r_candidate_move": float(count_max_move),
        "delta_active": bool(fert_moved > 0 and count_moved > 0),
    }


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (share_widowed.75+|female), targeted cells, decider."""
    fails_by_cell: dict[str, list[int]] = {}
    for f in verdict["all_failing_gated_cells"]:
        fails_by_cell.setdefault(f["cell"], []).append(f["seed"])

    def track(cell: str) -> dict[str, Any]:
        return {
            "tolerance": per_seed[0]["gated_cells"][cell]["tolerance"],
            "per_seed_score": {
                s["seed"]: s["gated_cells"][cell]["score"] for s in per_seed
            },
            "per_seed_pass": {
                s["seed"]: s["gated_cells"][cell]["pass"] for s in per_seed
            },
            "failed_seeds": sorted(fails_by_cell.get(cell, [])),
        }

    def seeds_pass_if_forgiven(forgiven: set[str]) -> int:
        n = 0
        for s in per_seed:
            ok = all(
                rec["pass"]
                for cell, rec in s["gated_cells"].items()
                if cell not in forgiven
            )
            n += ok
        return n

    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven({REGISTERED_MODAL_CELL})
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    n_pass_no_both = seeds_pass_if_forgiven(
        set(TARGETED_CELLS) | {REGISTERED_MODAL_CELL}
    )
    modal_failed = REGISTERED_MODAL_CELL in fails_by_cell
    gate_pass = verdict["gate_2_pass"]
    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }

    if gate_pass:
        decider = "none (gate passed)"
    else:
        modal_flips = n_pass_no_modal >= 4
        targeted_flips = n_pass_no_targeted >= 4
        both_flips = n_pass_no_both >= 4
        if targeted_flips and modal_flips:
            decider = (
                "both independently decisive (forgiving the registered modal "
                "or the delta-targeted cells alone flips the gate to pass)"
            )
        elif targeted_flips:
            decider = (
                "the delta-targeted cells (forgiving the count/fertility "
                "targets flips >= 4 seeds to pass)"
            )
        elif modal_flips:
            decider = (
                "share_widowed.75+|female (the registered modal alone holds "
                "the gate; forgiving it flips >= 4 seeds to pass)"
            )
        elif both_flips:
            decider = (
                "the registered modal plus the delta-targeted cells jointly "
                "(forgiving all of them flips >= 4 seeds to pass)"
            )
        else:
            decider = (
                "broader than the registered modal + delta-targeted cells "
                "(byte-identical-to-candidate-8 marital cells also hold the "
                "gate below 4 passing seeds)"
            )

    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (with delta 1 removing the count cells, "
            "the 75+ female widow stock is the binding residual risk; "
            "byte-identical to candidate 8, so it holds wherever candidate 8 "
            "failed it)"
        ),
        "modal_cell": REGISTERED_MODAL_CELL,
        "modal_failed": modal_failed,
        "modal_failed_seeds": sorted(
            fails_by_cell.get(REGISTERED_MODAL_CELL, [])
        ),
        "modal_track": track(REGISTERED_MODAL_CELL),
        "modal_is_sole_failing_cell": (
            len(distinct_fail_cells) == 1 and modal_failed
        ),
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "n_seeds_pass_if_modal_and_targeted_forgiven": n_pass_no_both,
            "decider": decider,
            "modal_decided": (not gate_pass) and (n_pass_no_modal >= 4),
        },
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-9 schema + c1-c8 + v8/forensics."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        pins[f"candidate{n}_runner"] = f"scripts/run_gate2_candidate{n}.py"
        pins[f"candidate{n}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{n}.py"
        )
    pins["candidate8_artifact"] = "runs/gate2_hazard_v8.json"
    pins["candidate8_artifact_sha256"] = c1._sha_of_file(CANDIDATE8_ARTIFACT)
    pins["forensics_runner"] = "scripts/gate2_forensics.py"
    pins["forensics_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics.py"
    )
    pins["forensics_artifact"] = "runs/gate2_forensics_v1.json"
    pins["forensics_artifact_sha256"] = c1._sha_of_file(FORENSICS_ARTIFACT)
    pins["delta"] = (
        "delta 1: observed undatable-marriage lifetime-count initial state "
        "(RNG-neutral); delta 2: unsmoothed exact single-year fertility for "
        "parities 0->1 and 1->2 at ages 22-38 where exposure >= 200 py"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 8's model block, edited for the two candidate-9 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a mortality-"
            "composed widowhood component (candidate 8 + two named deltas: an "
            "observed undatable-marriage lifetime-count initial state, and an "
            "unsmoothed exact single-year rate for the low-parity fertility "
            "peak)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate8": DELTA_VS_CANDIDATE8,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- BYTE-"
                "IDENTICAL to candidate 8 (no delta touches first marriage; its "
                "reference cells and the ever-married shares are byte-identical "
                "to candidate 8)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- BYTE-IDENTICAL to candidate 8 (marriage "
                "draws unchanged)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; surviving-spouse marriage-"
                "history widowhood level x candidate 5's committed NCHS betas "
                "-- BYTE-IDENTICAL to candidate 8 (marriage draws unchanged)"
            ),
            "remarriage": (
                "rescaled order-split weighted empirical hazard by years-since-"
                "dissolution band x origin x sex x marriage-order bit (entering "
                "2nd vs 3rd+) -- candidate 8's delta, BYTE-IDENTICAL here "
                "(marriage draws unchanged)"
            ),
            "fertility": (
                "DELTA 2: single-year-of-age rates within parity (0/1/2/3+) x "
                "birth-decade cohort; parities 0->1 and 1->2 use the EXACT "
                "empirical single-year rate (no kernel) at ages 22-38 where the "
                "single-year cell exposure is >= 200 weighted person-years, the "
                "triangular kernel (bandwidth 3) elsewhere; higher parities "
                "unchanged. Removes the kernel's peak attenuation where Q2 "
                "locates the low-parity under-production"
            ),
            "lifetime_marriage_count_initial_state": (
                "DELTA 1: each holdout person's simulated lifetime-marriage "
                "count initializes at their OBSERVED residual (n_marriages "
                "minus datable in-exposure marriage events -- the forensics Q1 "
                "residual: undatable start/dissolution, out-of-window/underage, "
                "separation-origin and MH-count-excess marriages) and "
                "accumulates the simulated datable transitions as candidate 8. "
                "An observed initial state per protocol; RNG-neutral; hazards "
                "untouched"
            ),
        },
        "registered_ambiguity_resolutions": {
            "observed_residual_definition": (
                "R[i] = n_marriages[i] - (# datable in-exposure first_marriage "
                "and remarriage transition events for person i) on the "
                "reference panel; weighted over the ever-married denominator it "
                "equals the forensics reference residual (mean_lifetime_"
                "marriages - in_exposure) and the sum of "
                "reference_residual_breakdown's five buckets (remainder 0.0); "
                "non-negative, integer-valued; applied per holdout person as an "
                "observed initial state (like the entry marital state)"
            ),
            "low_parity_exact_fertility": (
                "parity bands 0 (0->1) and 1 (1->2) only; ages 22-38 inclusive; "
                "the single-year weighted person-year exposure (the kernel's "
                "own denominator before smoothing) must be >= 200 for the exact "
                "rate num/den to replace the kernel value; every other cell "
                "keeps candidate 5's triangular-kernel value byte-identically"
            ),
            "rng_neutrality": (
                "delta 1 adds no draw (a post-assembly count add) and delta 2 "
                "moves only the fertility threshold (the per-year uniform "
                "blocks are drawn in the same order and size), so the marriage "
                "draw stream is byte-identical to candidate 8 and every non-"
                "fertility, non-lifetime-marriage-count gated cell matches "
                "candidate 8 exactly"
            ),
            "everything_else": (
                "the rescaled order-split remarriage, the surviving-spouse "
                "widowhood level, the committed NCHS betas, the knot-at-22 "
                "first-marriage spline, divorce, the spousal-gap draw, the "
                "competing-risk step, the RNG rule default_rng(4200 + seed), "
                "the spawned gap-draw stream, one sequence per person, and the "
                "locked protocol are byte-identical to candidate 8"
            ),
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = c1._load_cache(cache_path)

    thresholds = c1.load_gate2_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    tol = c1.gated_tolerances(thresholds)
    if len(tol) != 46:
        raise RuntimeError(
            f"expected 46 gated tolerances, got {len(tol)} from gates.yaml."
        )
    report_only = list(thresholds["report_only"])

    floor = json.loads(FLOOR_RUN.read_text())
    gated_set = set(floor["gate_partition"]["gate_eligible"])
    if set(tol) != gated_set:
        raise RuntimeError(
            "gates.yaml gated tolerances do not match the floor's "
            "gate_partition; refusing to score a mismatched cell set."
        )

    # Preflight: candidate 8 (the fit base + the movement/identity baseline),
    # candidates 6/5/7 (candidate 8's fit chain) and the forensics artifact
    # (the reused Q1 accounting) must be present, and the candidate-5 NCHS
    # references (candidate 6's fit imports candidate 5's chain).
    for name, path in (
        ("candidate-8", CANDIDATE8_ARTIFACT),
        ("candidate-7", CANDIDATE7_ARTIFACT),
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
        ("forensics", FORENSICS_ARTIFACT),
    ):
        if not path.exists():
            raise RuntimeError(
                f"{name} artifact missing at {path}; required for the run."
            )
    for year, path in c5.NCHS_LIFE_TABLE_PATHS.items():
        if not path.exists():
            raise RuntimeError(
                f"NCHS life-table reference for {year} missing at {path}; "
                "run scripts/fetch_nchs_life_tables_historical.py first."
            )
    c6._committed_beta_v5()  # fail fast if the committed betas drifted

    mh_records = marriage.marriage_history()
    birth_records = g2f.births.birth_history()
    death_records = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, fert, data_meta = g2f.load_panels()
    order_map = c1._order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons"
        )

    # Hard gate 1: bit-exact reproduction of the committed floor.
    precheck = c1.run_precheck(panel, fert, floor)
    if verbose:
        print(
            "precheck all_reproduced_exactly="
            f"{precheck['all_reproduced_exactly']} "
            f"(ref dev={precheck['reference_moments_max_abs_deviation']:.2e}, "
            f"rate_a dev={precheck['rate_a_max_abs_deviation']:.2e}, "
            f"sha_all={precheck['holdout_sha256_all_match']})"
        )
    if not precheck["all_reproduced_exactly"]:
        raise RuntimeError(
            "Scoring path does not reproduce the committed gate-2 floor "
            "(reference moments / per-seed rate_a / holdout sha256) to bit "
            "precision; refusing to proceed."
        )

    # Hard gate 2: the delta-1 reconciliation (train-side, licensed).
    reconciliation = _delta1_reconciliation(panel, mh_records, GATE_SEEDS)
    if verbose:
        print(
            "delta-1 reconciliation reconciled="
            f"{reconciliation['reconciled']} "
            "(per-person identity max="
            f"{reconciliation['per_person_identity_max_abs_residual']:.2e}, "
            "aggregate max remainder="
            f"{reconciliation['aggregate_reconciliation_max_abs_remainder']:.2e}"
            f", residual>=0: {reconciliation['residual_nonnegative']})"
        )
    if not reconciliation["reconciled"]:
        raise RuntimeError(
            "delta-1 reconciliation failed: the observed residual does not "
            "reproduce the forensics Q1 reconciliation to remainder 0.0 on "
            "train; refusing to proceed."
        )

    per_seed: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"seed_{seed}"
        if key in cache:
            if verbose:
                print(f"seed {seed}: cached")
            per_seed.append(cache[key])
            continue
        result = score_seed(
            seed,
            panel,
            fert,
            demo,
            death_records,
            mh_records,
            birth_records,
            order_map,
            floor,
            tol,
            report_only,
            verbose,
        )
        cache[key] = json.loads(json.dumps(result, default=c1._json_default))
        c1._save_cache(cache_path, cache)
        per_seed.append(cache[key])

    verdict = c1.build_verdict(per_seed, tol)
    report_block = c1.report_only_summary(per_seed, report_only)
    seed_conjunction = [
        {
            "seed": s["seed"],
            "n_gated_pass": s["n_gated_pass"],
            "n_gated_fail": s["n_gated_fail"],
            "seed_pass": s["seed_pass"],
        }
        for s in per_seed
    ]

    modal = _modal_failure_check(verdict, per_seed)
    candidate8_comparison = _candidate8_comparison(per_seed)
    identity = _identity_vs_candidate8(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 9",
        "spec_registration": SPEC_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "forensics_diagnostic": FORENSICS_REGISTRATION,
        "forensics_artifact": "runs/gate2_forensics_v1.json (#94)",
        "delta_vs_candidate8": DELTA_VS_CANDIDATE8,
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 + flip #81); "
            "protocol/views/tolerances read at runtime, no threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.40-0.50",
            "conjunction_estimate": 0.45,
            "modal_failure": (
                "share_widowed.75+|female (clipP 0.34, seed 1's boundary "
                "cell): with delta 1 removing the only high-clipP count cell, "
                "the 75+ female widow stock is the binding residual risk, "
                "modally failing one-to-two seeds plus one small-clipP cell "
                "elsewhere and leaving exactly 3 passing seeds"
            ),
            "delta_vs_candidate8": DELTA_VS_CANDIDATE8,
            "registration": SPEC_REGISTRATION,
            "grading_honesty_note": (
                "delta 2 perturbs the shared RNG stream only through the "
                "fertility threshold; the noise-dominated marital cells "
                "(share_divorced.45-54|female, share_widowed.75+|female, "
                "widowhood.75+|female) are in fact byte-identical to candidate "
                "8 (the marriage draw stream is unchanged), so they hold "
                "wherever candidate 8 failed them rather than redrawing -- the "
                "actual per-seed outcome is reported in identity_vs_candidate8 "
                "and candidate8_comparison"
            ),
        },
        "model": _model_block(),
        "protocol": {
            "option": "a (gate-1 mirror; LOCKED gates.yaml gate_2)",
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "the holdout, side B = the train complement"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "sim_rng_rule": "numpy.random.default_rng(4200 + seed)",
            "one_sequence_per_person": True,
            "scored_against": (
                "side A's own empirical rate (rate_a in "
                "runs/gate2_floors_v2.json noise_floor_per_seed)"
            ),
            "statistic": "|ln(r_candidate / rate_a)| per cell",
            "conjunction": (
                "all 46 gated cells per seed AND >= 4 of 5 gate seeds"
            ),
            "weight_definition": (
                "person-constant most-recent positive PSID cross-sectional "
                "weight; every gated statistic weighted, none unweighted"
            ),
        },
        "data": data_meta,
        "precheck": precheck,
        "delta1_reconciliation": reconciliation,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "modal_failure_materialized": modal,
        "candidate8_comparison": candidate8_comparison,
        "identity_vs_candidate8": identity,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "candidate8_registration": CANDIDATE8_REGISTRATION,
            "forensics_diagnostic": "runs/gate2_forensics_v1.json",
            "floor_run": "runs/gate2_floors_v2.json",
            "faithful_candidate_oc": floor["faithful_candidate_oc"][
                "p_gate_pass_4_of_5"
            ],
        },
        "revision_pins": _revision_pins(thresholds),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_2_pass={v['gate_2_pass']} "
            f"({v['n_seeds_pass']}/5 seeds pass)"
        )
        print(f"seed_pass: {v['seed_pass']}")
        print(
            "identity vs candidate 8 (non-fertility, non-count cells) byte_"
            f"identical={identity['byte_identical']} "
            f"(max dev={identity['max_abs_r_candidate_deviation_vs_candidate8']:.2e}"
            f"); registered modal ({REGISTERED_MODAL_CELL}) failed seeds "
            f"{modal['modal_failed_seeds']}; "
            f"decider={modal['decider_analysis']['decider']}"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    artifact = run(verbose=True, cache_path=Path(args.cache))
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
