"""Gate-2b candidate 9: the cohort-scoped fertility lift.

Candidate 9 (registration #42 comment 4948839837) is candidate 8
(:mod:`populace_dynamics.models.household_composition_sim_v8`, merged in PR
#147) with EXACTLY ONE frozen delta against the graded candidate-8 collateral
(grading 4948838962): the delta-1 completed-fertility swap is CONFINED to the
forensics-5-measured deficit cohorts x sex -- ``{55-64, 65-74} x male`` and
``{45-54, 65-74} x female`` -- with every OTHER cohort retaining the sim's own
completed-family-size distribution (candidate-7 behavior). Everything else in
candidate 8 is carried BYTE-FAITHFULLY: the delta-2 cohabitation-overlay lift
at 25-34|female, the delta-3 band-signed adult-child retention refit +
link-coverage inclusion at parent 45+, and every carried candidate-7 family
(``coresident_parent`` / ``coresident_spouse`` except 25-34|female / ``multigen``
/ ``parental_home_exit`` / ``coresident_grandchild``).

The scope change is realized as a WRITE GATE on candidate 8's own
:func:`~populace_dynamics.models.household_composition_sim_v8.apply_fertility_core_lift`:
the composed frame, the isolated ``SeedSequence([draw_seed, 0xC8])`` and every
per-cohort random draw are reproduced BIT-FOR-BIT from candidate 8, and only the
per-cohort WRITES of ``coresident_child`` / ``hh_size`` are gated to the deficit
cohorts. Consequences, exact by construction:

* the deficit cohorts (55-64|male, 65-74|male, 45-54|female, 65-74|female) get
  candidate 8's lift UNCHANGED -- byte-identical ``coresident_child`` / ``hh_size``
  on every draw and seed (their draws and their writes are candidate 8's);
* every non-deficit cohort reverts to the candidate-7 (unlifted) composition
  BYTE-IDENTICALLY -- so the four candidate-8 collateral cells (35-44|male,
  35-44|female, 45-54|male, 55-64|female) return to their candidate-7 cleared
  state, the mechanism the candidate-8 grading localized;
* delta 2, delta 3 and every carried family are byte-identical to candidate 8
  (their ``0xC8`` substreams are spawned independently of the fertility stream,
  and delta 3 operates only on the deficit cohorts, whose delta-1 output is
  candidate 8's).

The priced uncertainty is ``hh_size.5+``: candidate 8's GLOBAL lift cleared it
(0.127 -> 0.144); the scoped lift delivers only the deficit cohorts' share of
that mass, so if a material share came through the middle cohorts' large
families the scoped lift under-delivers it. :func:`scoped_lift_analytic_check`
records the scoped lift's implied TRAIN-SIDE counterfactual for the priced cell,
the two held deficit-male cells and the four collateral cells BEFORE the scored
holdout run, so the artifact shows what the scoping predicts before the holdout
says what it delivers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models import household_composition_sim_v8 as hcs8

__all__ = [
    "FERTILITY_LIFT_CELLS",
    "DEFICIT_COHORTS",
    "COLLATERAL_CELLS",
    "REVERTED_CHILD_CELLS",
    "ANALYTIC_CHECK_CELLS",
    "N_ANALYTIC_DRAWS",
    "ANALYTIC_DRAW_SEED_BASE",
    "HouseholdCompositionModelV9",
    "fit_household_model_v9",
    "apply_scoped_fertility_core_lift",
    "simulate_draw_v9",
    "scoped_lift_analytic_check",
]

#: Carried from candidate 8 (the completed-family-size bucketing + kernels + the
#: two RNG-isolated delta streams are all reused unchanged).
SIZE_BUCKETS = hcs8.SIZE_BUCKETS
DELTA_STREAM_TAG_V7 = hcs8.DELTA_STREAM_TAG_V7  # 0xC7 (carried)
DELTA_STREAM_TAG_V8 = hcs8.DELTA_STREAM_TAG_V8  # 0xC8 (deltas; unchanged tag)

#: The ONE candidate-9 delta: the delta-1 fertility-core swap is confined to the
#: forensics-5-measured 3+-child completed-family-size deficit cohorts x sex.
FERTILITY_LIFT_CELLS = (
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
    "coresident_child.45-54|female",
    "coresident_child.65-74|female",
)
#: The (band, sex) pairs the scoped lift WRITES; every other composition cohort
#: reverts to the candidate-7 distribution.
DEFICIT_COHORTS = frozenset(hcs8._cell_of(c) for c in FERTILITY_LIFT_CELLS)

#: The four candidate-8 collateral cells (grading 4948838962): previously-cleared
#: middle-cohort child cells the GLOBAL lift overshot on the holdout. Under the
#: scoped lift they revert to candidate 7 (their cleared state).
COLLATERAL_CELLS = (
    "coresident_child.35-44|male",
    "coresident_child.35-44|female",
    "coresident_child.45-54|male",
    "coresident_child.55-64|female",
)

#: Every gated coresident_child cell OUTSIDE the deficit scope reverts to the
#: candidate-7 composition byte-identically (the four collateral cells plus the
#: younger cohorts the global lift also moved).
REVERTED_CHILD_CELLS = (
    "coresident_child.15-24|male",
    "coresident_child.15-24|female",
    "coresident_child.25-34|male",
    "coresident_child.25-34|female",
    "coresident_child.35-44|male",
    "coresident_child.35-44|female",
    "coresident_child.45-54|male",
    "coresident_child.55-64|female",
)

#: The cells the pre-run analytic check reports (registration 4948839837): the
#: priced aggregate, the two held deficit-male cells, and the four collateral
#: cells whose scoping predicts a revert-to-cleared.
ANALYTIC_CHECK_CELLS = (
    "hh_size.5+",
    "coresident_child.55-64|male",
    "coresident_child.65-74|male",
) + COLLATERAL_CELLS

#: Train draws averaged for the analytic-check per-cohort D_sim / K_sim / H_sim
#: measurement (deterministic; fixed seeds). A prediction, never gated.
N_ANALYTIC_DRAWS = 8
ANALYTIC_DRAW_SEED_BASE = 0xA9C

#: Candidate 9 reuses candidate 8's fitted bundle verbatim -- the per-cohort
#: train completed-family-size distribution D_train, the delta-3 channel closures
#: and the delta-2 lift are all identical; only the delta-1 APPLICATION scope
#: differs (a simulate-time write gate).
HouseholdCompositionModelV9 = hcs8.HouseholdCompositionModelV8


def fit_household_model_v9(
    *args: Any, **kwargs: Any
) -> hcs8.HouseholdCompositionModelV8:
    """Fit candidate 8's bundle (identical estimator; scope differs at draw).

    Every fitted structure -- the per-(band, sex) train completed-family-size
    distribution, the band-signed delta-3 retention/link closures and the
    delta-2 overlay lift -- is candidate 8's, estimated on side B only. The ONE
    candidate-9 delta is the simulate-time confinement of the delta-1 swap to
    :data:`DEFICIT_COHORTS` (:func:`apply_scoped_fertility_core_lift`); nothing
    in the fit changes, so the model is byte-identical to candidate 8's.
    """
    return hcs8.fit_household_model_v8(*args, **kwargs)


# --------------------------------------------------------------------------
# The ONE delta: the cohort-scoped fertility-core lift (write-gated)
# --------------------------------------------------------------------------
def apply_scoped_fertility_core_lift(
    person_id: np.ndarray,
    band: np.ndarray,
    sex: np.ndarray,
    weight: np.ndarray,
    child_counts: np.ndarray,
    coresident_child: np.ndarray,
    hh_size: np.ndarray,
    model: hcs8.HouseholdCompositionModelV8,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Delta 1 CONFINED to :data:`DEFICIT_COHORTS` (candidate-9 scope).

    A bit-faithful copy of candidate 8's
    :func:`~populace_dynamics.models.household_composition_sim_v8.apply_fertility_core_lift`
    with the per-cohort WRITES of ``coresident_child`` / ``hh_size`` GATED to the
    deficit cohorts. Every per-cohort random draw (the target-bucket resample,
    the coresidence re-emission uniforms and the hh_size pool picks) is drawn for
    ALL composition cohorts in candidate 8's exact order and count, so ``rng`` is
    consumed identically to candidate 8; only the assignment back into the arrays
    is confined to the deficit cohorts. Hence the deficit cohorts are
    byte-identical to candidate 8, and every other cohort keeps its candidate-7
    (input) ``coresident_child`` / ``hh_size``.
    """
    s_sim = hcs8.sim_completed_size_row(person_id, child_counts)
    bucket_sim = hcs8.size_bucket(s_sim)
    cor_new = np.asarray(coresident_child, dtype=bool).copy()
    hh_new = np.asarray(hh_size, dtype=np.int64).copy()
    diag_cells: dict[str, Any] = {}
    for bl in hcs8._COMPOSITION_BANDS:
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
            # Target bucket per row ~ D_train[cell] (drawn for EVERY cohort so
            # the deficit cohorts see candidate 8's exact rng state).
            tgt_idx = rng.choice(len(SIZE_BUCKETS), size=len(idx), p=probs)
            tgt_bucket = np.array(SIZE_BUCKETS, dtype=object)[tgt_idx]
            w_cell = np.asarray(weight, dtype=np.float64)[idx]
            cor_cell = np.asarray(coresident_child, dtype=bool)[idx]
            hh_cell = np.asarray(hh_size, dtype=np.int64)[idx]
            bkt_cell = bucket_sim[idx]
            k_sim: dict[str, float] = {}
            hh_pool: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
            for b in SIZE_BUCKETS:
                mb = bkt_cell == b
                wb = float(w_cell[mb].sum())
                k_sim[b] = (
                    float((w_cell[mb] * cor_cell[mb]).sum() / wb)
                    if wb > 0
                    else 0.0
                )
                if mb.any() and wb > 0:
                    hh_pool[b] = (hh_cell[mb], w_cell[mb] / wb)
                else:
                    hh_pool[b] = (hh_cell, None)
            k_row = np.array([k_sim[b] for b in tgt_bucket], dtype=np.float64)
            u = rng.random(len(idx))
            in_scope = cell in DEFICIT_COHORTS
            # Re-emit coresident_child ~ Bernoulli(K_sim[tgt_bucket]) -- WRITTEN
            # only for the deficit cohorts (else revert to candidate 7).
            if in_scope:
                cor_new[idx] = u < k_row
            # Resample hh_size from the draw's own (cell, bucket) pool -- the
            # pick is drawn for every cohort (rng parity) but WRITTEN only for
            # the deficit cohorts.
            for b in SIZE_BUCKETS:
                sel = np.flatnonzero(tgt_bucket == b)
                if not len(sel):
                    continue
                vals, wprob = hh_pool[b]
                if wprob is None or not len(vals):
                    continue
                pick = rng.choice(len(vals), size=len(sel), p=wprob)
                if in_scope:
                    hh_new[idx[sel]] = vals[pick]
            diag_cells[f"coresident_child.{bl}|{sx}"] = {
                "in_scope": bool(in_scope),
                "sim_completed_size_dist": hcs8._size_dist(s_sim[idx], w_cell),
                "train_completed_size_dist": d_train,
                "k_sim_given_size": k_sim,
            }
    diag = {
        "per_cell": diag_cells,
        "scope_cells": list(FERTILITY_LIFT_CELLS),
        "note": (
            "The delta-1 completed-fertility swap is confined to the deficit "
            "cohorts (55-64|male, 65-74|male, 45-54|female, 65-74|female); "
            "every other composition cohort keeps its candidate-7 "
            "coresident_child / hh_size. All cohorts draw candidate 8's exact "
            "rng, so the deficit cohorts are byte-identical to candidate 8."
        ),
    }
    return cor_new, hh_new, diag


# --------------------------------------------------------------------------
# Simulation (one draw over the side-A holdout)
# --------------------------------------------------------------------------
def simulate_draw_v9(
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    model: hcs8.HouseholdCompositionModelV8,
    ids_a: set[int],
    draw_seed: int,
    delta_stream_tag_v8: int = DELTA_STREAM_TAG_V8,
) -> tuple[hc.HouseholdCompositionPanel, dict[str, Any]]:
    """Simulate one candidate-9 draw of the side-A holdout households.

    Byte-identical to
    :func:`~populace_dynamics.models.household_composition_sim_v8.simulate_draw_v8`
    except the delta-1 fertility-core lift is the SCOPED
    :func:`apply_scoped_fertility_core_lift`. The candidate-7 composition
    (:func:`hcs8._compose_v7`), the isolated ``SeedSequence([draw_seed, 0xC8])``
    and its three spawned substreams, the delta-3 retention/link refit and the
    delta-2 cohab-overlay lift are all candidate 8's, so every carried family and
    both carried deltas are byte-identical to candidate 8, and the deficit
    cohorts' delta-1 output is candidate 8's too.
    """
    comp = hcs8._compose_v7(hh, mpanel, model.base_v7, ids_a, draw_seed)
    pw = comp["pw"]
    band, sex, weight = comp["band"], comp["sex"], comp["weight"]

    # Isolated 0xC8 substreams (candidate 8's): delta 1 (fertility), delta 3
    # (retention/link), delta 2 (cohab overlay). Spawned independently, so the
    # scoped delta 1 does not perturb delta 2 / delta 3.
    c8_ss = np.random.SeedSequence([draw_seed, delta_stream_tag_v8])
    fert_ss, retention_ss, cohab_ss = c8_ss.spawn(3)
    fert_rng = np.random.default_rng(fert_ss)
    retention_rng = np.random.default_rng(retention_ss)
    cohab_rng = np.random.default_rng(cohab_ss)

    # Delta 1 (candidate-9 scope): fertility-core lift CONFINED to the deficit
    # cohorts (coresident_child + hh_size).
    cor_d1, hh_d1, fert_diag = apply_scoped_fertility_core_lift(
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
    # Delta 3 (carried): band-signed retention + link-coverage refit. Operates
    # only on the deficit cohorts (RETENTION_EXIT_CELLS + LINK_COVERAGE_CELLS),
    # whose delta-1 output is candidate 8's, so it is byte-identical.
    cor_d3, retention_diag = hcs8.apply_retention_link_refit(
        band, sex, weight, cor_d1, model, retention_rng
    )
    # Delta 2 (carried): cohabitation-overlay lift (coresident_spouse.25-34|f).
    spouse_d2, cohab_diag = hcs8.apply_cohab_overlay_lift(
        band, sex, weight, comp["spouse"], model, cohab_rng
    )

    # Build the panel EXACTLY as candidate 8 does (carries byte-identical).
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
        "scoped_fertility_core_lift": fert_diag,
        "retention_link_refit": retention_diag,
        "cohab_overlay_lift": cohab_diag,
        "delta_stream_tag_v8": delta_stream_tag_v8,
    }
    return panel, diagnostics


# --------------------------------------------------------------------------
# Pre-run train-side analytic check (registration 4948839837)
# --------------------------------------------------------------------------
def _size_dist_row(
    bucket_cell: np.ndarray, w_cell: np.ndarray
) -> dict[str, float]:
    """Weighted completed-size distribution over SIZE_BUCKETS for one cohort."""
    tot = float(w_cell.sum())
    if tot <= 0:
        return {b: 0.0 for b in SIZE_BUCKETS}
    return {
        b: float(w_cell[bucket_cell == b].sum() / tot) for b in SIZE_BUCKETS
    }


def _kernel_row(
    bucket_cell: np.ndarray, hit_cell: np.ndarray, w_cell: np.ndarray
) -> dict[str, float]:
    """Weighted P(hit | completed-size bucket) for one cohort (K_sim / H_sim)."""
    k: dict[str, float] = {}
    for b in SIZE_BUCKETS:
        mb = bucket_cell == b
        wb = float(w_cell[mb].sum())
        k[b] = float((w_cell[mb] * hit_cell[mb]).sum() / wb) if wb > 0 else 0.0
    return k


def scoped_lift_analytic_check(
    model: hcs8.HouseholdCompositionModelV8,
    hh: hc.HouseholdCompositionPanel,
    mpanel: transitions.MaritalPanel,
    ids_b: set[int],
    reference_b: dict[str, dict[str, Any]],
    tol: dict[str, float],
    *,
    n_draws: int = N_ANALYTIC_DRAWS,
    seed_base: int = ANALYTIC_DRAW_SEED_BASE,
) -> dict[str, Any]:
    """The scoped lift's implied TRAIN-SIDE counterfactual, before the run.

    Applies the Q15 analytic law-of-total-probability to the sim's OWN train
    (side-B) composition, per composition cohort, holding the sim's coresidence-
    given-size kernel ``K_sim`` and hh_size|size kernel ``H_sim`` and swapping the
    completed-size distribution ``D_sim[S] -> D_train[S]`` ONLY for the deficit
    cohorts (the candidate-9 scope), against the GLOBAL swap (every cohort,
    candidate 8). Reports, for the priced ``hh_size.5+`` aggregate, the two held
    deficit-male cells and the four candidate-8 collateral cells:

    * ``sim_train`` -- the sim rate on train (no lift);
    * ``global_counterfactual`` -- candidate 8's swap (every cohort);
    * ``scoped_counterfactual`` -- candidate 9's swap (deficit cohorts only);
    * ``reference_train`` -- the observed side-B rate (``rate_b``);
    * ``predicted_score`` / ``predicted_within_tolerance`` -- the scoped
      counterfactual scored against the train reference at the locked tolerance.

    ``hh_size.5+`` additionally reports ``middle_cohort_share_of_lift`` -- the
    fraction of candidate 8's ``hh_size.5+`` lift that flowed through the
    non-deficit cohorts (the priced quantity: what the scoping forgoes).
    Deterministic (fixed composition seeds); a prediction, never gated.
    """
    import math

    comp_bands = hcs8._COMPOSITION_BANDS
    # Per-cohort accumulators over the analytic train draws.
    child_sim: dict[str, list[float]] = {}
    child_glob: dict[str, list[float]] = {}
    hh5_sim: list[float] = []
    hh5_glob: list[float] = []
    hh5_scoped: list[float] = []

    for k in range(n_draws):
        comp = hcs8._compose_v7(
            hh, mpanel, model.base_v7, ids_b, seed_base + k
        )
        s_row = hcs8.sim_completed_size_row(
            comp["person_id"], comp["child_counts"]
        )
        bucket_row = hcs8.size_bucket(s_row)
        band_row = comp["band"]
        sex_row = comp["sex"]
        w_row = np.asarray(comp["weight"], dtype=np.float64)
        cor_row = np.asarray(comp["coresident_child"], dtype=bool)
        hh5_row = np.asarray(comp["hh_size"], dtype=np.int64) >= 5
        w_total = float(w_row.sum())

        sim_hh5 = float((w_row * hh5_row).sum() / w_total)
        contrib_global = 0.0
        contrib_scoped = 0.0
        for bl in comp_bands:
            for sx in ("male", "female"):
                cell = (bl, sx)
                d_train = model.completed_size_dist_train.get(cell)
                if d_train is None:
                    continue
                mask = (band_row == bl) & (sex_row == sx)
                if not mask.any():
                    continue
                wc = w_row[mask]
                wctot = float(wc.sum())
                if wctot <= 0:
                    continue
                bkt = bucket_row[mask]
                d_sim = _size_dist_row(bkt, wc)
                k_sim = _kernel_row(bkt, cor_row[mask], wc)
                h_sim = _kernel_row(bkt, hh5_row[mask], wc)
                child_cell = f"coresident_child.{bl}|{sx}"
                sim_full = sum(d_sim[b] * k_sim[b] for b in SIZE_BUCKETS)
                cf_full = sum(
                    d_train.get(b, 0.0) * k_sim[b] for b in SIZE_BUCKETS
                )
                child_sim.setdefault(child_cell, []).append(sim_full)
                child_glob.setdefault(child_cell, []).append(cf_full)
                e_sim = sum(d_sim[b] * h_sim[b] for b in SIZE_BUCKETS)
                e_train = sum(
                    d_train.get(b, 0.0) * h_sim[b] for b in SIZE_BUCKETS
                )
                share = wctot / w_total
                delta = share * (e_train - e_sim)
                contrib_global += delta
                if cell in DEFICIT_COHORTS:
                    contrib_scoped += delta
        hh5_sim.append(sim_hh5)
        hh5_glob.append(sim_hh5 + contrib_global)
        hh5_scoped.append(sim_hh5 + contrib_scoped)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _score(cf: float, ref: float) -> float | None:
        if cf > 0 and ref > 0:
            return float(abs(math.log(cf / ref)))
        return None

    cells: dict[str, Any] = {}
    for cell in ANALYTIC_CHECK_CELLS:
        ref = float(reference_b[cell]["rate"])
        t = float(tol[cell])
        if cell == "hh_size.5+":
            sim_v = _mean(hh5_sim)
            glob_v = _mean(hh5_glob)
            scoped_v = _mean(hh5_scoped)
        else:
            sim_v = _mean(child_sim.get(cell, []))
            glob_v = _mean(child_glob.get(cell, []))
            scoped_v = glob_v if cell in FERTILITY_LIFT_CELLS else sim_v
        score = _score(scoped_v, ref)
        cells[cell] = {
            "in_scope": cell in FERTILITY_LIFT_CELLS,
            "reverts_to_candidate7": cell in REVERTED_CHILD_CELLS,
            "sim_train": sim_v,
            "global_counterfactual": glob_v,
            "scoped_counterfactual": scoped_v,
            "reference_train": ref,
            "tolerance": t,
            "predicted_score": score,
            "predicted_within_tolerance": (
                bool(score <= t) if score is not None else None
            ),
        }

    sim_hh5 = _mean(hh5_sim)
    glob_hh5 = _mean(hh5_glob)
    scoped_hh5 = _mean(hh5_scoped)
    global_lift = glob_hh5 - sim_hh5
    scoped_lift = scoped_hh5 - sim_hh5
    middle_share = (
        (global_lift - scoped_lift) / global_lift
        if abs(global_lift) > 1e-12
        else None
    )
    cells["hh_size.5+"]["scoped_share_of_global_lift"] = (
        (scoped_lift / global_lift) if abs(global_lift) > 1e-12 else None
    )
    cells["hh_size.5+"]["middle_cohort_share_of_lift"] = middle_share

    return {
        "method": (
            "Train-side (side B) Q15 law-of-total-probability applied to the "
            "sim's own composition per cohort: swap D_sim[S] -> D_train[S] for "
            "the deficit cohorts only (scoped) vs every cohort (global), holding "
            "the sim's K_sim (coresidence|size) and H_sim (hh_size>=5|size) "
            "kernels; scored against the observed side-B rate_b at the locked "
            f"tolerance. Averaged over {n_draws} deterministic train draws "
            f"(compose seeds {seed_base}+k). A prediction, never gated."
        ),
        "n_draws": n_draws,
        "scope_cells": list(FERTILITY_LIFT_CELLS),
        "collateral_cells": list(COLLATERAL_CELLS),
        "cells": cells,
        "hh_size_5plus_priced": {
            "sim_train": sim_hh5,
            "global_lift_candidate8": global_lift,
            "scoped_lift_candidate9": scoped_lift,
            "middle_cohort_share_of_lift": middle_share,
            "note": (
                "hh_size.5+ is the priced uncertainty: candidate 8's global "
                "lift raised it by global_lift; the scoped lift delivers only "
                "the deficit cohorts' scoped_lift. middle_cohort_share_of_lift "
                "is what the scoping forgoes -- if large it under-delivers "
                "hh_size.5+ (the modal residual)."
            ),
        },
        "prediction_summary": {
            "collateral_cells_predicted_clear": [
                c
                for c in COLLATERAL_CELLS
                if cells[c]["predicted_within_tolerance"]
            ],
            "collateral_cells_predicted_fail": [
                c
                for c in COLLATERAL_CELLS
                if cells[c]["predicted_within_tolerance"] is False
            ],
            "deficit_male_cells_predicted_hold": {
                c: cells[c]["predicted_within_tolerance"]
                for c in (
                    "coresident_child.55-64|male",
                    "coresident_child.65-74|male",
                )
            },
            "hh_size_5plus_predicted_within_tolerance": cells["hh_size.5+"][
                "predicted_within_tolerance"
            ],
        },
    }
