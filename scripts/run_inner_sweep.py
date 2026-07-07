"""The candidate-10 design sweep, scored on the inner-validation harness.

REPORTED, NOT GATED. This sweep touches NO outer holdout data (every
candidate is generated over the INNER holdout carved from each seed's outer
TRAIN complement by :mod:`inner_validation`) and produces the evidence for
selecting candidate 10 BEFORE its single registered outer run. It reads no
gate verdict and writes no gate verdict.

The banked findings after eleven outer runs (issue #42, candidate-9 grading,
comment 4899043614) fix the shared base of every variant here:

* The zero-anchor participation mechanism is SOLVED -- KEEP the participation
  gate refit on the zero-anchor train subpopulation (candidate 9's component
  2 gate), DROP the re-entry-pool restriction (it coupled Q0 to chronically
  low-rank donors), and give Q0 targets NO exposure to any long-memory
  mechanism (at lambda > 0 the blend turned the Q0 near-constant anchor
  coordinate into a low-rank-donor selector, swinging pooled Q0 to -17.9%).
  So EVERY variant below is candidate 7's k-NN machinery + candidate 9's
  zero-anchor participation gate, with the full re-entry pool for all targets
  and Q0 targets' third distance term held at candidate 7's
  ``|u_A(donor) - u_A(target)|`` (no memory exposure).
* The pairs-view C2ST margin is fragile to ANY perturbation (even
  lambda = 0 seeds degraded from candidate 7's 0.516-0.529 to 0.531-0.532),
  which is why the inner harness reports the pairs C2ST margin explicitly.
* The singular unsolved core is long-horizon memory that does not leak into
  short lags or the joint. The variants differ ONLY in that mechanism.

The variants (all deterministic, seeded from the OUTER seed):

* **V0** -- no memory change (candidate 7's distance verbatim on non-Q0
  targets). The control; expected 10-year undershoot.
* **V0-shared** -- V0 but with candidate 7's ORIGINAL shared participation
  gate for ALL targets (no zero-anchor refit), to quantify the refit's
  inner-scale effect alone (V0 vs V0-shared isolates the refit; everything
  else is identical).
* **V1** -- candidate 9's lambda-blend at fixed small lambdas {0.1, 0.2} on
  non-Q0 targets (third term ``|lambda*u_w(donor) + (1-lambda)*u_A(donor) -
  u_A(target)|``), Q0 targets EXEMPT. Tests whether the Q0 exemption rescues
  the blend.
* **V2** -- persistent-state carry: each person carries ``w`` drawn ONCE from
  the 5b-style Gaussian conditional at the anchor using the candidate-3
  z-decomposition, and the non-Q0 third term becomes ``|u_w(donor) -
  Phi(w_target/sigma_w)|`` -- the target-side coordinate is the DRAWN latent
  rather than the raw anchor (decouples anchor noise), Q0 targets exempt.
* **V3** -- horizon-split distances: transitions keep candidate 7's third
  term (anchor) unchanged BUT add a 4th coordinate at weight 0.25 matching
  the person's RUNNING MEAN of already-generated ranks (available backward
  from the anchor) against the donor's from-rank ``u_next`` -- memory from
  the chain's own realized path rather than a donor permanent coordinate, Q0
  targets exempt.

Each variant x each outer seed generates over the inner holdout and is scored
by the full amended-gate mirror (:func:`inner_validation.score_inner_pair`).
The artifact ``runs/inner_sweep_v1.json`` carries every variant x seed
scorecard, the per-variant inner-gate verdict, and a ranking table (which
variants clear ALL amended-gate metrics on >= 4/5 inner seeds, with margins).

Scale caveat (also in :mod:`inner_validation` and the artifact): inner
scales are smaller (~13k inner-train persons -> ~3.3k-4.5k inner-holdout), so
inner C2ST / KS / tail run slightly HOTTER than the outer ~20%-holdout scale.
The sweep RANKS variants and CHECKS margins; it does not predict the outer
run exactly.

Environment (dedicated gate venv; participation gates need populace-fit; the
benefit-space block needs the pinned pe-us oracle)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_inner_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# The protocol / scoring machinery is imported through the inner harness,
# which itself imports it byte-for-byte from the merged baseline runner.
import inner_validation as iv
import numpy as np
import pandas as pd

# The rank machinery, the shared participation gate, and the anchor rows are
# candidate 5b's (byte-for-byte the prior runs').
from run_gate1_baseline import SEEDS, load_filtered_panel
from run_gate1_candidate5b import (
    CellMarginal,
    _gate_sign_draw,
    age_bin,
    anchor_rows,
    fit_cell_marginals,
)

# Candidate 7's k-NN weighted single-record draw and its frozen distance
# weights are imported byte-for-byte (the draw rule and the 1/0.5/0.25 lag
# weights are unchanged; the variants only change the THIRD term / add a
# FOURTH term, and only for non-Q0 targets).
from run_gate1_candidate7 import (
    K_NEIGHBORS,  # noqa: F401 (re-exported for tests)
    W_ANCHOR,
    W_NEXT,
    W_NEXT2,
    _knn_draw,
)

# Candidate 9's donor-pool builder (carries u_w and the anchor-zero flag) and
# its zero-anchor participation gate + the shared gate are imported
# byte-for-byte, so the pools and the participation law are the prior runs'.
from run_gate1_candidate9 import (
    anchor_u_by_person,
    build_donor_pools,
    fit_participation_gate,
    fit_zero_anchor_participation_gate,
)
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "inner_sweep_v1.json"
ARTIFACT_SCHEMA_VERSION = "inner_sweep.v1"

#: The V3 fourth-coordinate weight (the person's running mean of generated
#: ranks vs the donor from-rank u_next). Pinned a priori at candidate 7's
#: anchor weight so the fourth axis carries the same influence as the anchor
#: axis it augments.
W_RUNMEAN = 0.25
#: The frozen V1 lambda grid for the sweep: the two small fixed lambdas the
#: brief pins (the Q0 exemption is the thing under test, not lambda; larger
#: lambda demonstrably breaks the pairs C2ST, so the sweep does not explore
#: it). Reduced to {0.1} only under the tight-budget fallback (recorded).
V1_LAMBDAS = (0.1, 0.2)

#: Fixed integer codes for the generation RNG substream labels. Each label
#: seeds an independent generator via SeedSequence([outer_seed, variant_code,
#: code]); the variant code keeps every variant's draws independent and
#: reproducible from the outer seed alone.
SUBSTREAM_CODES = {"gate": 1, "donor-draw": 2, "re-entry-draw": 3, "wcarry": 4}
#: Distinct variant codes so each variant (and each V1 lambda) draws its own
#: reproducible streams off the outer seed.
VARIANT_CODES = {
    "V0": 10,
    "V0-shared": 11,
    "V1-lam0.1": 12,
    "V1-lam0.2": 13,
    "V2": 14,
    "V3": 15,
}


def _substream(
    outer_seed: int, variant_code: int, label: str
) -> np.random.Generator:
    """A generation RNG for one substream label of one variant/outer seed."""
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(
        np.random.SeedSequence([int(outer_seed), int(variant_code), code])
    )


# --------------------------------------------------------------------------
# V2 -- the persistent-state latent w, drawn once per target person from the
# 5b-style Gaussian conditional at the anchor, using the candidate-3
# z-decomposition (no free parameters).
# --------------------------------------------------------------------------
def draw_wcarry(
    target_ids: np.ndarray,
    anchor_rank_of_person: dict[int, float],
    uw_fit: dict[str, Any],
    pooled_z_mean: float,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Draw each target person's persistent latent ``w`` once (V2).

    The candidate-3 z-decomposition gives the permanent variance
    ``sigma2_perm``, transitory ``sigma2_trans``, and noise ``sigma2_noise``,
    with total z-variance ``gamma_0 = sigma2_perm + sigma2_trans +
    sigma2_noise``. Given a single anchor observation ``z_A = Phi^-1(u_A)`` of
    the pooled-mean-centred z, the permanent component's posterior is the
    exact normal-normal conditional (the 5b stage-4 form
    ``w | z_A ~ N(a z_A, 1 - a^2)`` with ``a^2 = sigma2_perm / gamma_0`` in
    the normalized model), expressed here with the candidate-3 parameters and
    NO free constant:

    * posterior mean ``= (sigma2_perm / gamma_0) * (z_A - pooled_z_mean)``
      (exactly candidate 3's single-observation shrinkage: at ``n_i = 1`` its
      ``V_i = sigma2_trans + sigma2_noise`` and ``w_i = sigma2_perm /
      gamma_0``),
    * posterior variance ``= sigma2_perm * (1 - sigma2_perm / gamma_0)``.

    Each person draws ``w`` ONCE from that conditional (a persistent state,
    not a per-step innovation). Returns ``{person_id: w}``. The caller maps it
    to the k-NN coordinate ``Phi(w / sigma_hat_w)`` -- the same scale as the
    donor ``u_w = Phi(what / sigma_hat_w)``. Zero-anchor persons draw a ``w``
    too, but the caller never reads it (Q0 targets are exempt from the V2
    third term); it is drawn for stream determinism only.
    """
    s2p = float(uw_fit["sigma2_perm"])
    s2t = float(uw_fit["sigma2_trans"])
    s2n = float(uw_fit["sigma2_noise"])
    gamma_0 = s2p + s2t + s2n
    if gamma_0 > 0:
        shrink = s2p / gamma_0
        post_var = s2p * (1.0 - shrink)
    else:
        shrink = 0.0
        post_var = 0.0
    post_sd = float(np.sqrt(max(post_var, 0.0)))

    ids = np.sort(target_ids)
    z_a = np.array(
        [
            norm.ppf(anchor_rank_of_person[int(p)]) - float(pooled_z_mean)
            for p in ids
        ],
        dtype=np.float64,
    )
    mean = shrink * z_a
    noise = rng.standard_normal(ids.size)
    w = mean + post_sd * noise
    return {int(p): float(wi) for p, wi in zip(ids, w, strict=True)}


# --------------------------------------------------------------------------
# The unified generation engine: candidate 7's backward k-NN chain + candidate
# 9's zero-anchor participation gate, with the memory mechanism selected per
# variant and Q0 targets exempt from it.
# --------------------------------------------------------------------------
def generate_variant(
    holdout: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted_shared: Any,
    fitted_zero: Any,
    pools: dict[str, Any],
    memory_mode: str,
    outer_seed: int,
    variant_code: int,
    *,
    lam: float = 0.0,
    uw_of_person: dict[int, float] | None = None,
    uw_fit: dict[str, Any] | None = None,
    pooled_z_mean: float = 0.0,
    sigma_hat_w: float = 1.0,
    use_zero_anchor_gate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward k-NN chain over the holdout with a per-variant memory term.

    Candidate 7's backward chain and candidate 9's zero-anchor participation
    gate, with the banked-findings base (the FULL re-entry pool for every
    target -- NO restriction -- and Q0 targets EXEMPT from the memory term)
    and one of four memory mechanisms on the non-Q0 (positive-anchor)
    targets' THIRD distance term:

    * ``memory_mode="none"`` (V0): third term ``W_ANCHOR * |u_A(donor) - a|``
      (candidate 7 verbatim).
    * ``memory_mode="lambda_blend"`` (V1): third term ``W_ANCHOR *
      |lam*u_w(donor) + (1-lam)*u_A(donor) - a|`` (candidate 9's blend), for
      non-Q0 targets only; ``lam`` fixed.
    * ``memory_mode="wcarry"`` (V2): third term ``W_ANCHOR * |u_w(donor) -
      Phi(w_target/sigma_hat_w)|``; the target coordinate is the drawn
      persistent latent (:func:`draw_wcarry`) rather than the anchor rank.
    * ``memory_mode="running_mean"`` (V3): candidate 7's third term unchanged,
      PLUS a fourth term ``W_RUNMEAN * |u_next(donor) - running_mean|`` where
      ``running_mean`` is the person's running mean of already-generated ranks
      (backward from the anchor), matching donors by their from-rank.

    ``use_zero_anchor_gate`` False (V0-shared) uses the shared gate for ALL
    targets (candidate 7's original participation law), isolating the
    zero-anchor refit's effect. In every mode Q0 targets keep candidate 7's
    ``|u_A(donor) - a|`` third term and the FULL re-entry pool (the re-entry
    restriction is dropped for everyone).

    Returns ``(candidate, diagnostics)``. The candidate holds exactly the
    holdout persons on their observed periods; only earnings are generated,
    the anchor kept at its real value.
    """
    if memory_mode not in {"none", "lambda_blend", "wcarry", "running_mean"}:
        raise ValueError(f"unknown memory_mode {memory_mode!r}")
    if memory_mode == "lambda_blend" and uw_of_person is None:
        raise ValueError("lambda_blend needs uw_of_person")
    if memory_mode == "wcarry" and (uw_of_person is None or uw_fit is None):
        raise ValueError("wcarry needs uw_of_person and uw_fit")

    rng_gate = _substream(outer_seed, variant_code, "gate")
    rng_donor = _substream(outer_seed, variant_code, "donor-draw")
    rng_reentry = _substream(outer_seed, variant_code, "re-entry-draw")
    rng_wcarry = _substream(outer_seed, variant_code, "wcarry")

    pairs_pool = pools["pairs"]
    triples_pool = pools["triples"]
    reentry_pool = pools["reentry"]

    hp = holdout.sort_values(["person_id", "period"]).reset_index(drop=True)
    hp["rank_from_top"] = (
        hp.groupby("person_id")["period"]
        .rank(ascending=False, method="first")
        .astype(int)
        - 1
    )
    hp["depth"] = (
        hp.groupby("person_id")["period"].transform("size").astype(int)
    )
    hp["bin"] = age_bin(hp["age"].to_numpy())

    pids = hp["person_id"].to_numpy()
    periods = hp["period"].to_numpy()
    ages = hp["age"].to_numpy(dtype=np.float64)
    bins = hp["bin"].to_numpy()
    ranks = hp["rank_from_top"].to_numpy()
    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    n_rows = len(hp)
    pos_by_key = {
        (int(pid), int(r)): i
        for i, (pid, r) in enumerate(zip(pids, ranks, strict=True))
    }
    max_depth = int(hp["depth"].max()) if n_rows else 0

    # Continuous anchor rank per target person (frozen stage-2 rule) and the
    # exact zero-anchor flag (read from the anchor rows, not a float compare).
    target_ids = np.sort(hp["person_id"].unique())
    uA_map = anchor_u_by_person(marginals, all_anchor, target_ids)
    anchor_rank_of_person = {int(p): float(uA_map[int(p)]) for p in target_ids}
    ha_all = all_anchor[
        all_anchor.person_id.isin(set(int(x) for x in target_ids))
    ]
    zero_anchor_person = {
        int(r.person_id): bool(float(r.earnings) == 0.0)
        for r in ha_all.itertuples(index=False)
    }

    # V2: draw each target person's persistent latent w once, map to the k-NN
    # coordinate Phi(w/sigma_hat_w) (same scale as the donor u_w). Q0 targets
    # get a value too but it is never read (they are exempt).
    wcoord_of_person: dict[int, float] = {}
    if memory_mode == "wcarry":
        w_of_person = draw_wcarry(
            target_ids,
            anchor_rank_of_person,
            uw_fit,  # type: ignore[arg-type]
            pooled_z_mean,
            rng_wcarry,
        )
        sw = sigma_hat_w if sigma_hat_w > 0 else 1.0
        wcoord_of_person = {
            int(p): float(norm.cdf(w_of_person[int(p)] / sw))
            for p in target_ids
        }

    # Presorted donor arrays (the pinned tie-break order from build_donor_pools).
    tri_u_next = triples_pool["u_next"]
    tri_u_next2 = triples_pool["u_next2"]
    tri_u_A = triples_pool["u_A"]
    tri_u_w = triples_pool["u_w"]
    tri_w = triples_pool["weight"]
    tri_u_prev = triples_pool["u_prev"]
    pair_u_next = pairs_pool["u_next"]
    pair_u_A = pairs_pool["u_A"]
    pair_u_w = pairs_pool["u_w"]
    pair_w = pairs_pool["weight"]
    pair_u_prev = pairs_pool["u_prev"]
    re_u_A = reentry_pool["u_A"]
    re_w = reentry_pool["weight"]
    re_u_prev = reentry_pool["u_prev"]

    # Per-variant donor THIRD-TERM coordinate for the NON-Q0 branch (constant
    # across query rows at a fixed lambda). Q0 always uses u_A.
    if memory_mode == "lambda_blend":
        tri_third = lam * tri_u_w + (1.0 - lam) * tri_u_A
        pair_third = lam * pair_u_w + (1.0 - lam) * pair_u_A
        re_third = lam * reentry_pool["u_w"] + (1.0 - lam) * re_u_A
    elif memory_mode == "wcarry":
        tri_third = tri_u_w
        pair_third = pair_u_w
        re_third = reentry_pool["u_w"]
    else:  # none / running_mean keep the anchor coordinate for the third term
        tri_third = tri_u_A
        pair_third = pair_u_A
        re_third = re_u_A

    # Running-mean-of-generated-ranks per person (V3), updated as the backward
    # chain fixes each period's rank. Seeded at the anchor's own rank so the
    # first backward step already has a path summary.
    running_sum: dict[int, float] = {}
    running_cnt: dict[int, int] = {}
    if memory_mode == "running_mean":
        for p in target_ids:
            running_sum[int(p)] = float(anchor_rank_of_person[int(p)])
            running_cnt[int(p)] = 1

    # Diagnostics accumulators.
    n_clamped = 0
    n_positive_gen = 0
    n_triple_draws = 0
    n_pair_draws = 0
    n_reentry_draws = 0
    n_nonq0_memory_terms = 0
    neighbor_dists: list[float] = []

    for j in range(1, max_depth):
        positions = np.nonzero(ranks == j)[0]
        if positions.size == 0:
            continue
        order = np.argsort(pids[positions], kind="stable")
        positions = positions[order]
        step_pids = pids[positions]

        next_positions = np.array(
            [pos_by_key[(int(pid), j - 1)] for pid in step_pids]
        )
        next_level = gen_earn[next_positions]
        step_is_zero = np.array(
            [zero_anchor_person[int(p)] for p in step_pids], dtype=bool
        )

        # Participation gate: zero-anchor persons use the zero-anchor gate
        # (unless V0-shared), positive-anchor persons the shared gate. One
        # uniform per row from the gate substream in the fixed step order.
        u_gate = rng_gate.random(len(positions))
        signs = np.empty(len(positions), dtype=np.int64)
        pa_step = ~step_is_zero
        if np.any(pa_step):
            signs[pa_step] = _gate_sign_draw(
                fitted_shared,
                next_level[pa_step],
                ages[positions][pa_step],
                u_gate[pa_step],
            )
        if np.any(step_is_zero):
            if use_zero_anchor_gate and fitted_zero is not None:
                gate_za = fitted_zero
            else:
                gate_za = fitted_shared
            signs[step_is_zero] = _gate_sign_draw(
                gate_za,
                next_level[step_is_zero],
                ages[positions][step_is_zero],
                u_gate[step_is_zero],
            )
        is_pos = signs == 1

        vals = np.zeros(len(positions), dtype=np.float64)
        pos_local = np.nonzero(is_pos)[0]
        if pos_local.size:
            a_local = np.array(
                [
                    anchor_rank_of_person[int(step_pids[li])]
                    for li in pos_local
                ],
                dtype=np.float64,
            )
            za_local = step_is_zero[pos_local]
            next_level_pos = next_level[pos_local]
            next_is_pos = next_level_pos > 0

            # Per-row NON-Q0 memory target coordinate (matches the donor
            # third coordinate above): the drawn latent for V2, else the
            # anchor rank a. Q0 rows always use a (exempt).
            if memory_mode == "wcarry":
                third_target = np.array(
                    [wcoord_of_person[int(step_pids[li])] for li in pos_local],
                    dtype=np.float64,
                )
            else:
                third_target = a_local.copy()
            # Q0 (zero-anchor) rows are exempt: their third-term target is the
            # anchor rank a, matched against the donor u_A coordinate.
            third_target[za_local] = a_local[za_local]

            # V3 running-mean coordinate per row (the person's running mean of
            # already-generated ranks). Only used on non-Q0 rows.
            if memory_mode == "running_mean":
                run_target = np.array(
                    [
                        running_sum[int(step_pids[li])]
                        / running_cnt[int(step_pids[li])]
                        for li in pos_local
                    ],
                    dtype=np.float64,
                )
            else:
                run_target = None

            # v1 = rank of the (positive) next period at its own cell.
            v1 = np.full(pos_local.size, np.nan, dtype=np.float64)
            kp = np.nonzero(next_is_pos)[0]
            for m in kp:
                li = pos_local[m]
                gpos = next_positions[li]
                cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                v1[m] = cell.rank(float(gen_earn[gpos]))

            # v2 = rank of the two-steps-later (rank j-2) period, if positive.
            v2 = np.full(pos_local.size, np.nan, dtype=np.float64)
            has_v2 = np.zeros(pos_local.size, dtype=bool)
            for m in kp:
                li = pos_local[m]
                key2 = (int(step_pids[li]), j - 2)
                gpos2 = pos_by_key.get(key2)
                if gpos2 is not None and gen_earn[gpos2] > 0:
                    cell2 = marginals[(int(bins[gpos2]), int(periods[gpos2]))]
                    v2[m] = cell2.rank(float(gen_earn[gpos2]))
                    has_v2[m] = True

            u_prev_local = np.zeros(pos_local.size, dtype=np.float64)

            # Each branch (re-entry / triple / pair) splits internally by
            # za_local: non-Q0 rows use the per-variant third donor coordinate
            # (and V3's fourth axis), Q0 rows use candidate 7's anchor
            # coordinate with no memory term (the banked-findings exemption),
            # all on the FULL pools (no re-entry restriction).

            # ---- Branch A: next zero -> re-entry (full pool, no restriction).
            rp = np.nonzero(~next_is_pos)[0]
            if rp.size:
                _reentry_draw(
                    rp,
                    za_local,
                    third_target,
                    a_local,
                    re_third,
                    re_u_A,
                    re_w,
                    re_u_prev,
                    rng_reentry,
                    u_prev_local,
                    neighbor_dists,
                )
                n_reentry_draws += int(rp.size)

            # ---- Branch B: next positive, v2 exists -> triple.
            tp = np.nonzero(next_is_pos & has_v2)[0]
            if tp.size:
                _transition_draw(
                    tp,
                    za_local,
                    v1[tp],
                    v2[tp],
                    third_target[tp],
                    a_local[tp],
                    (run_target[tp] if run_target is not None else None),
                    tri_u_next,
                    tri_u_next2,
                    tri_third,
                    tri_u_A,
                    tri_w,
                    tri_u_prev,
                    memory_mode,
                    rng_donor,
                    u_prev_local,
                    neighbor_dists,
                    triple=True,
                )
                n_triple_draws += int(tp.size)

            # ---- Branch C: next positive, no v2 -> pair.
            pp = np.nonzero(next_is_pos & ~has_v2)[0]
            if pp.size:
                _transition_draw(
                    pp,
                    za_local,
                    v1[pp],
                    None,
                    third_target[pp],
                    a_local[pp],
                    (run_target[pp] if run_target is not None else None),
                    pair_u_next,
                    None,
                    pair_third,
                    pair_u_A,
                    pair_w,
                    pair_u_prev,
                    memory_mode,
                    rng_donor,
                    u_prev_local,
                    neighbor_dists,
                    triple=False,
                )
                n_pair_draws += int(pp.size)

            n_nonq0_memory_terms += int((~za_local).sum())

            # Earnings = Qhat_pos of the CURRENT (rank-j) period's cell, and
            # update the V3 running mean with this generated rank.
            for m, li in enumerate(pos_local):
                gpos = positions[li]
                cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                up = float(u_prev_local[m])
                q = float(cell.quantile(np.array([up]))[0])
                vals[li] = q
                if cell.wtil.size and (
                    up < cell.wtil[0] or up > cell.wtil[-1]
                ):
                    n_clamped += 1
                if memory_mode == "running_mean":
                    pid_m = int(step_pids[li])
                    running_sum[pid_m] += up
                    running_cnt[pid_m] += 1
            n_positive_gen += int(pos_local.size)
        gen_earn[positions] = vals

    out = hp[["person_id", "period", "earnings", "age", "weight"]].copy()
    out["earnings"] = gen_earn

    n_zero_anchor_holdout = int(
        sum(1 for p in target_ids if zero_anchor_person[int(p)])
    )
    if neighbor_dists:
        pcts = [0, 10, 25, 50, 75, 90, 100]
        nd = {f"p{p}": float(np.percentile(neighbor_dists, p)) for p in pcts}
        nd["mean"] = float(np.mean(neighbor_dists))
    else:
        nd = {}
    diagnostics = {
        "memory_mode": memory_mode,
        "lambda": (float(lam) if memory_mode == "lambda_blend" else None),
        "use_zero_anchor_gate": bool(use_zero_anchor_gate),
        "n_holdout_persons": int(len(target_ids)),
        "n_zero_anchor_holdout_persons": n_zero_anchor_holdout,
        "n_positive_generated": int(n_positive_gen),
        "n_clamped": int(n_clamped),
        "clamped_share": (
            float(n_clamped / n_positive_gen) if n_positive_gen else 0.0
        ),
        "n_triple_draws": int(n_triple_draws),
        "n_pair_draws": int(n_pair_draws),
        "n_reentry_draws": int(n_reentry_draws),
        "n_nonq0_memory_terms": int(n_nonq0_memory_terms),
        "triple_share_of_positive": (
            float(n_triple_draws / (n_triple_draws + n_pair_draws))
            if (n_triple_draws + n_pair_draws)
            else 0.0
        ),
        "neighbor_distance_distribution": nd,
    }
    return out, diagnostics


def _reentry_draw(
    idx: np.ndarray,
    za_local: np.ndarray,
    third_target: np.ndarray,
    a_local: np.ndarray,
    re_third: np.ndarray,
    re_u_A: np.ndarray,
    re_w: np.ndarray,
    re_u_prev: np.ndarray,
    rng: np.random.Generator,
    u_prev_local: np.ndarray,
    neighbor_dists: list[float],
) -> None:
    """Re-entry k-NN draw on the FULL pool (no restriction) for a row set.

    Non-Q0 rows match on the per-variant third donor coordinate
    (``re_third``) against their third-term target; Q0 rows match on the
    donor anchor rank ``re_u_A`` against their anchor rank (candidate 7's
    ``|u_A - a|``, the memory exemption). Split by ``za_local`` so each
    subset uses its own donor coordinate; one uniform per row from the
    re-entry substream in a fixed order.
    """
    nq = idx[~za_local[idx]]
    q0 = idx[za_local[idx]]
    if nq.size:
        dist = np.abs(re_third[None, :] - third_target[nq][:, None])
        u_dr = rng.random(nq.size)
        drawn, kth = _knn_draw(dist, re_w, re_u_prev, u_dr)
        u_prev_local[nq] = drawn
        neighbor_dists.extend(float(x) for x in kth)
    if q0.size:
        dist = np.abs(re_u_A[None, :] - a_local[q0][:, None])
        u_dr = rng.random(q0.size)
        drawn, kth = _knn_draw(dist, re_w, re_u_prev, u_dr)
        u_prev_local[q0] = drawn
        neighbor_dists.extend(float(x) for x in kth)


def _transition_draw(
    idx: np.ndarray,
    za_local: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray | None,
    third_target: np.ndarray,
    a_local: np.ndarray,
    run_target: np.ndarray | None,
    d_u_next: np.ndarray,
    d_u_next2: np.ndarray | None,
    d_third: np.ndarray,
    d_u_A: np.ndarray,
    d_w: np.ndarray,
    d_u_prev: np.ndarray,
    memory_mode: str,
    rng: np.random.Generator,
    u_prev_local: np.ndarray,
    neighbor_dists: list[float],
    *,
    triple: bool,
) -> None:
    """Transition k-NN draw (pair or triple) with the per-variant memory term.

    Two row subsets, each with its own distance (the banked-findings Q0
    exemption):

    * NON-Q0 (positive-anchor) rows -- the per-variant memory term:
      ``W_NEXT|u_next - v1| (+ W_NEXT2|u_next2 - v2| triple) + W_ANCHOR|
      d_third(donor) - third_target| (+ W_RUNMEAN|u_next(donor) - run_target|
      for V3)``. ``d_third`` is the donor's per-variant third coordinate
      (blended rank for V1, ``u_w`` for V2, ``u_A`` for V0/V3); the fourth
      axis is added ONLY when ``memory_mode == "running_mean"`` (V3).
    * Q0 (zero-anchor) rows -- candidate 7's exact distance:
      ``W_NEXT|u_next - v1| (+ W_NEXT2|u_next2 - v2|) + W_ANCHOR|u_A(donor) -
      a|`` -- no memory term, no fourth axis.

    ``idx`` indexes ``u_prev_local`` (the local positive rows); ``v1`` / ``v2``
    / ``third_target`` / ``a_local`` / ``run_target`` are already sliced to
    ``idx``. One uniform per row from the donor substream in a fixed order
    (non-Q0 then Q0).
    """

    def _draw(
        sub_local: np.ndarray,
        third_donor: np.ndarray,
        third_query: np.ndarray,
        add_fourth: bool,
    ) -> None:
        if sub_local.size == 0:
            return
        rows = idx[sub_local]
        dist = W_NEXT * np.abs(d_u_next[None, :] - v1[sub_local][:, None])
        if triple and v2 is not None and d_u_next2 is not None:
            dist = dist + W_NEXT2 * np.abs(
                d_u_next2[None, :] - v2[sub_local][:, None]
            )
        dist = dist + W_ANCHOR * np.abs(
            third_donor[None, :] - third_query[sub_local][:, None]
        )
        if add_fourth and run_target is not None:
            # V3's fourth coordinate: donor from-rank u_next vs the person's
            # running mean of already-generated ranks (chain-path memory).
            dist = dist + W_RUNMEAN * np.abs(
                d_u_next[None, :] - run_target[sub_local][:, None]
            )
        u_dr = rng.random(sub_local.size)
        drawn, kth = _knn_draw(dist, d_w, d_u_prev, u_dr)
        u_prev_local[rows] = drawn
        neighbor_dists.extend(float(x) for x in kth)

    nq = np.nonzero(~za_local[idx])[0]
    q0 = np.nonzero(za_local[idx])[0]
    # Non-Q0 rows: per-variant third donor coordinate + (V3) the fourth axis.
    _draw(
        nq,
        d_third,
        third_target,
        add_fourth=(memory_mode == "running_mean"),
    )
    # Q0 rows: candidate 7's anchor coordinate, no memory, no fourth axis.
    _draw(q0, d_u_A, a_local, add_fourth=False)


# --------------------------------------------------------------------------
# Per-seed fit shared across variants (deterministic given the inner split)
# --------------------------------------------------------------------------
def fit_inner_seed(
    inner_train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    outer_seed: int,
) -> dict[str, Any]:
    """Fit the machinery on one inner-train split (shared across variants).

    Deterministic given the inner split: the per-cell marginals, the u_w
    z-decomposition (candidate 8), the donor pools (candidate 9, carrying u_w
    and the anchor-zero flag), the shared participation gate, and the
    zero-anchor participation gate. The participation gates are seeded from
    the OUTER seed. Every variant on this seed reuses this fit, so the only
    differences are the memory term and (V0-shared) which gate the zero-anchor
    persons use.
    """
    import run_gate1_candidate8 as c8

    marginals = fit_cell_marginals(inner_train)
    uw = c8.build_donor_uw(inner_train, marginals)
    pools = build_donor_pools(
        inner_train, all_anchor, marginals, uw["u_w_of_person"]
    )
    fitted_shared, _ = fit_participation_gate(inner_train, outer_seed)
    fitted_zero, n_zero_pairs = fit_zero_anchor_participation_gate(
        inner_train, all_anchor, outer_seed
    )
    return {
        "marginals": marginals,
        "uw": uw,
        "pools": pools,
        "fitted_shared": fitted_shared,
        "fitted_zero": fitted_zero,
        "n_zero_anchor_train_pairs": int(n_zero_pairs),
    }


# --------------------------------------------------------------------------
# The variant catalog: (name, memory_mode, lambda, use_zero_anchor_gate)
# --------------------------------------------------------------------------
def variant_catalog(v1_lambdas: tuple[float, ...]) -> list[dict[str, Any]]:
    """The sweep's variant list (banked-findings base; memory differs)."""
    catalog: list[dict[str, Any]] = [
        {
            "name": "V0",
            "memory_mode": "none",
            "lambda": None,
            "use_zero_anchor_gate": True,
            "description": (
                "no memory change (candidate 7 distance verbatim on non-Q0 "
                "targets); zero-anchor participation refit ON; full re-entry "
                "pool; Q0 memory-exempt. The control -- expected 10yr "
                "undershoot"
            ),
        },
        {
            "name": "V0-shared",
            "memory_mode": "none",
            "lambda": None,
            "use_zero_anchor_gate": False,
            "description": (
                "V0 but candidate 7's ORIGINAL shared participation gate for "
                "ALL targets (no zero-anchor refit); isolates the refit's "
                "inner-scale effect (V0 vs V0-shared)"
            ),
        },
    ]
    for lam in v1_lambdas:
        catalog.append(
            {
                "name": f"V1-lam{lam}",
                "memory_mode": "lambda_blend",
                "lambda": float(lam),
                "use_zero_anchor_gate": True,
                "description": (
                    f"candidate 9's lambda-blend at fixed lambda={lam} on "
                    "non-Q0 targets (Q0 EXEMPT); tests whether the Q0 "
                    "exemption rescues the blend"
                ),
            }
        )
    catalog.append(
        {
            "name": "V2",
            "memory_mode": "wcarry",
            "lambda": None,
            "use_zero_anchor_gate": True,
            "description": (
                "persistent-state carry: w drawn once per person from the "
                "5b-style Gaussian conditional at the anchor (candidate-3 "
                "z-decomposition); non-Q0 third term |u_w(donor) - "
                "Phi(w_target/sigma_w)| (target = drawn latent, decouples "
                "anchor noise); Q0 EXEMPT"
            ),
        }
    )
    catalog.append(
        {
            "name": "V3",
            "memory_mode": "running_mean",
            "lambda": None,
            "use_zero_anchor_gate": True,
            "description": (
                "horizon-split: candidate 7's third term unchanged PLUS a "
                "4th coordinate at weight 0.25 matching the person's running "
                "mean of already-generated ranks against the donor from-rank "
                "u_next (chain-path memory, not a donor permanent coord); Q0 "
                "EXEMPT"
            ),
        }
    )
    return catalog


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(
    v1_lambdas: tuple[float, ...] = V1_LAMBDAS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full candidate-10 design sweep on the inner harness."""
    started = time.time()
    gate_cfg = iv.load_amended_gate_config()
    view_specs = iv.build_inner_view_specs()

    panel = load_filtered_panel()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    # Anchors on the FULL filtered panel (candidate identity + the reported
    # full-panel Q0 slice would use these; the inner benefit block uses the
    # per-seed TRAIN anchor, see below).
    full_anchor = anchor_rows(panel)

    benefit_params, _ = _load_benefit_oracle()
    if benefit_params is None:
        raise RuntimeError(
            "The amended gate scores a GATED benefit-space block, but the SSA "
            "oracle did not load (set POPULACE_DYNAMICS_PE_US_DIR to the "
            "pinned policyengine-us checkout and rerun). The inner sweep "
            "mirrors the amended gate and refuses to publish without it."
        )
    if verbose:
        print(
            "benefit-space oracle (GATED mirror): pe_us_revision="
            f"{benefit_params.pe_us_revision}"
        )

    catalog = variant_catalog(v1_lambdas)
    # results[variant_name] = list of per-seed scorecards
    results: dict[str, list[dict[str, Any]]] = {v["name"]: [] for v in catalog}
    gen_diag: dict[str, list[dict[str, Any]]] = {
        v["name"]: [] for v in catalog
    }

    for outer_seed in SEEDS:
        seed_t = time.time()
        inner_holdout, inner_train, outer_holdout = iv.inner_split(
            panel, outer_seed
        )
        # The inner benefit-space anchor: the outer TRAIN complement's anchor
        # (the inner population). Its persons cover both inner sides.
        train_persons = set(
            inner_holdout["person_id"].tolist()
            + inner_train["person_id"].tolist()
        )
        train_anchor = full_anchor[
            full_anchor.person_id.isin(train_persons)
        ].reset_index(drop=True)
        benefit_cutpoints = iv.inner_anchor_cutpoints(train_anchor)

        fit = fit_inner_seed(inner_train, full_anchor, outer_seed)

        for v in catalog:
            vt = time.time()
            code = VARIANT_CODES[v["name"]]
            candidate, diag = generate_variant(
                inner_holdout,
                full_anchor,
                fit["marginals"],
                fit["fitted_shared"],
                fit["fitted_zero"],
                fit["pools"],
                v["memory_mode"],
                outer_seed,
                code,
                lam=(v["lambda"] or 0.0),
                uw_of_person=fit["uw"]["u_w_of_person"],
                uw_fit=fit["uw"]["fit"],
                pooled_z_mean=fit["uw"]["pooled_z_mean"],
                sigma_hat_w=fit["uw"]["sigma_hat_w"],
                use_zero_anchor_gate=v["use_zero_anchor_gate"],
            )
            scorecard = iv.score_inner_pair(
                outer_seed,
                inner_holdout,
                candidate,
                train_anchor,
                gate_cfg,
                view_specs,
                benefit_params,
                benefit_cutpoints,
            )
            scorecard["generation_diagnostics"] = diag
            scorecard["n_inner_train_persons"] = int(
                inner_train.person_id.nunique()
            )
            scorecard["n_zero_anchor_inner_train_pairs"] = fit[
                "n_zero_anchor_train_pairs"
            ]
            results[v["name"]].append(scorecard)
            gen_diag[v["name"]].append(diag)
            if verbose:
                pc = scorecard["geometry"]["psid_family_earnings_pairs"][
                    "scores"
                ]["c2st_auc"]
                ac10 = scorecard["battery_values"]["autocorr_log_10yr"]
                bs = scorecard.get("benefit_space")
                q0m = None
                if bs is not None:
                    q0 = bs["by_anchor_quintile"]["quintiles"].get("Q0", {})
                    q0m = (
                        q0.get("distribution", {})
                        .get("mean", {})
                        .get("pct_diff")
                        if q0.get("n_persons")
                        else None
                    )
                print(
                    f"  seed {outer_seed} {v['name']:11s}: "
                    f"pairsC2ST={pc:.4f} ac10={ac10:.3f} "
                    f"geo={scorecard['geometry_pass']} "
                    f"bat={scorecard['battery_pass']} "
                    f"Q0mean%={q0m if q0m is None else round(q0m, 2)} "
                    f"({time.time() - vt:.0f}s)"
                )
        if verbose:
            print(f" seed {outer_seed} done ({time.time() - seed_t:.0f}s)")

    # Per-variant inner-gate verdicts + margin summaries + the ranking table.
    variant_reports: dict[str, Any] = {}
    for v in catalog:
        per_seed = results[v["name"]]
        verdict = iv.inner_gate_verdict(
            per_seed, gate_cfg["benefit_metrics_cfg"]
        )
        margins = iv.summarize_margins(per_seed)
        variant_reports[v["name"]] = {
            "description": v["description"],
            "memory_mode": v["memory_mode"],
            "lambda": v["lambda"],
            "use_zero_anchor_gate": v["use_zero_anchor_gate"],
            "inner_gate_verdict": verdict,
            "margin_summary": margins,
        }

    ranking = _build_ranking(catalog, results, variant_reports)
    refit_effect = _refit_effect(results)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "inner_sweep_v1",
        "purpose": (
            "The candidate-10 design sweep, scored on the inner-validation "
            "harness. REPORTED-NOT-GATED: touches NO outer holdout data, "
            "reads no gate verdict, writes no gate verdict. Produces the "
            "evidence for selecting candidate 10 before its single registered "
            "outer run."
        ),
        "reported_not_gated": True,
        "no_outer_holdout_contact": (
            "every candidate is generated over the INNER holdout carved from "
            "each outer seed's TRAIN complement (split_panel_by_person(train, "
            "'person_id', fraction=0.25, seed=1000+s)); "
            "inner_validation.inner_split asserts the inner pair never shares "
            "a person with the outer holdout"
        ),
        "inner_scale_caveat": (
            "inner scales are smaller (~13k inner-train persons -> ~3.3k-4.5k "
            "inner-holdout persons, ~25k person-periods; donor pools/marginals "
            "from the ~13k-person inner-train vs the outer ~18k-person train), "
            "so inner C2ST / KS / tail run slightly HOTTER than the outer "
            "~20%-holdout scale the locked thresholds were calibrated at. The "
            "sweep RANKS variants and CHECKS margins against the amended "
            "thresholds; it does not predict the outer run exactly. A "
            "hair's-breadth inner pass/fail is a margin, not a verdict."
        ),
        "banked_findings": (
            "after eleven outer runs (issue #42 candidate-9 grading, comment "
            "4899043614): the zero-anchor participation mechanism is SOLVED "
            "(keep the refit gate, drop the re-entry-pool restriction, give "
            "Q0 no memory exposure); the pairs-view C2ST margin is fragile to "
            "any perturbation; the unsolved core is long-horizon memory that "
            "does not leak into short lags or the joint. Every variant here "
            "shares that base and differs ONLY in the long-memory mechanism."
        ),
        "amended_gate_mirror": (
            "gates.yaml gate_1 amended (PR #57/#59): pairs-view full geometry "
            "incl. its C2ST, runs-view coverage (C2ST demoted to reported), "
            "battery vs the committed battery_reference under the locked "
            "tolerances, and the gated benefit-space block (per-seed metrics "
            "fold into the seed geometry verdict; pooled Q0 is a standalone "
            "condition); >=4/5 geometry AND >=4/5 battery AND pooled Q0"
        ),
        "inner_split": {
            "outer_split": (
                "split_panel_by_person(panel, 'person_id', fraction=0.2, "
                "seed=s); the drawn 20% is the untouched outer holdout, the "
                "80% complement is TRAIN"
            ),
            "inner_split": (
                "split_panel_by_person(train, 'person_id', fraction=0.25, "
                "seed=1000+s); the drawn 25% of TRAIN persons is the inner "
                "holdout, the 75% complement is inner-train"
            ),
            "benefit_space_anchor": (
                "the outer TRAIN complement's anchor (the inner population); "
                "positive-anchor quartile edges fixed on that train anchor so "
                "the inner Q0 slice is a fixed property of the inner "
                "population, not the inner seed's draw"
            ),
        },
        "v1_lambdas": list(v1_lambdas),
        "seeds": list(SEEDS),
        "variants": variant_reports,
        "ranking": ranking,
        "refit_effect_v0_vs_v0shared": refit_effect,
        "per_variant_per_seed": {name: results[name] for name in results},
        "revision_pins": _revision_pins(benefit_params),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    if verbose:
        print(f"\nwrote {ARTIFACT_PATH}")
        _print_ranking(ranking)
    return artifact


def _build_ranking(
    catalog: list[dict[str, Any]],
    results: dict[str, list[dict[str, Any]]],
    variant_reports: dict[str, Any],
) -> dict[str, Any]:
    """The ranking table: per variant, which metrics clear >=4/5 inner seeds.

    For each variant, count how many inner seeds pass geometry, battery, and
    (via the verdict) the pooled Q0 gate, and pull out the three brief-named
    focal margins -- the pairs-view C2ST (fragile), the 10-year battery rung
    (the unsolved core), and the pooled Q0 -- with their worst-case inner
    margins. A variant 'clears all amended-gate metrics on >=4/5 inner seeds'
    iff its inner-gate verdict passes (>=4/5 geometry AND >=4/5 battery AND
    pooled Q0). Sorted by (inner_gate_pass, n_geometry_pass + n_battery_pass,
    pooled Q0 margin).
    """
    rows: list[dict[str, Any]] = []
    for v in catalog:
        name = v["name"]
        rep = variant_reports[name]
        verdict = rep["inner_gate_verdict"]
        margins = rep["margin_summary"]

        pairs_c2st = margins.get(
            "geometry:psid_family_earnings_pairs.c2st_auc_max", {}
        )
        ac10 = margins.get("battery:autocorr_log_10yr", {})
        pooled_q0_pct = verdict["pooled_q0_mean_pct_diff"]
        pooled_q0_margin = (
            5.0 - abs(pooled_q0_pct) if pooled_q0_pct is not None else None
        )

        # The binding geometry metric: the gated metric with the worst
        # (minimum) min_margin across seeds, and how many seeds pass it.
        geom_metrics = {
            k: m for k, m in margins.items() if k.startswith("geometry:")
        }
        binding_geom = _binding(geom_metrics)
        bat_metrics = {
            k: m for k, m in margins.items() if k.startswith("battery:")
        }
        binding_bat = _binding(bat_metrics)

        rows.append(
            {
                "variant": name,
                "memory_mode": v["memory_mode"],
                "lambda": v["lambda"],
                "clears_all_on_ge4_seeds": bool(verdict["inner_gate_pass"]),
                "n_geometry_pass": verdict["n_geometry_pass"],
                "n_battery_pass": verdict["n_battery_pass"],
                "pooled_q0_pass": verdict["pooled_q0_pass"],
                "geometry_gate_pass": verdict["geometry_gate_pass"],
                "battery_gate_pass": verdict["battery_gate_pass"],
                "focal_margins": {
                    "pairs_c2st": {
                        "n_pass": pairs_c2st.get("n_pass"),
                        "min_margin": pairs_c2st.get("min_margin"),
                        "mean_margin": pairs_c2st.get("mean_margin"),
                        "worst_seed_value": pairs_c2st.get("worst_seed_value"),
                        "threshold": 0.53,
                    },
                    "battery_10yr": {
                        "n_pass": ac10.get("n_pass"),
                        "min_margin": ac10.get("min_margin"),
                        "mean_margin": ac10.get("mean_margin"),
                        "worst_seed_value": ac10.get("worst_seed_value"),
                        "reference": 0.538879427219318,
                        "tolerance": 0.07,
                    },
                    "pooled_q0": {
                        "pct_diff": pooled_q0_pct,
                        "margin": pooled_q0_margin,
                        "threshold": 5.0,
                        "pass": verdict["pooled_q0_pass"],
                    },
                },
                "binding_geometry_metric": binding_geom,
                "binding_battery_metric": binding_bat,
            }
        )

    def _key(r: dict[str, Any]) -> tuple:
        q0m = r["focal_margins"]["pooled_q0"]["margin"]
        q0m = q0m if q0m is not None else -1e9
        return (
            r["clears_all_on_ge4_seeds"],
            r["n_geometry_pass"] + r["n_battery_pass"],
            q0m,
        )

    rows.sort(key=_key, reverse=True)
    return {
        "note": (
            "which variants clear ALL amended-gate metrics on >=4/5 inner "
            "seeds, with margins. 'clears_all_on_ge4_seeds' = the inner-gate "
            "verdict passes. Focal margins: pairs-view C2ST (fragile), the "
            "10-year battery rung (unsolved core), pooled Q0. min_margin is "
            "the worst-case across the five inner seeds (the binding number); "
            "positive = inside the amended band. Inner scale runs hotter than "
            "outer -- read margins, not hair's-breadth pass/fail."
        ),
        "table": rows,
    }


def _binding(metrics: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The metric with the worst (minimum) min_margin -- the binding one."""
    defined = {
        k: m for k, m in metrics.items() if m.get("min_margin") is not None
    }
    if not defined:
        return None
    key = min(defined, key=lambda k: defined[k]["min_margin"])
    m = defined[key]
    return {
        "metric": key,
        "n_pass": m["n_pass"],
        "min_margin": m["min_margin"],
        "mean_margin": m["mean_margin"],
        "worst_seed_value": m["worst_seed_value"],
    }


def _refit_effect(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """V0 vs V0-shared: the zero-anchor participation refit's inner effect.

    The two are identical except which participation gate the zero-anchor
    persons use (V0 the refit, V0-shared the original shared gate), so their
    difference isolates the refit. Reports the per-seed and pooled pooled-Q0
    mean % (the statistic the refit targets), the generated all-zero share
    proxy via the pooled Q0, and the pairs C2ST (to show the refit does not
    move the geometry).
    """
    if "V0" not in results or "V0-shared" not in results:
        return {"available": False}

    def _q0_series(name: str) -> list[float | None]:
        out: list[float | None] = []
        for s in results[name]:
            bs = s.get("benefit_space")
            val = None
            if bs is not None:
                q0 = bs["by_anchor_quintile"]["quintiles"].get("Q0", {})
                if q0.get("n_persons", 0) > 0:
                    val = q0["distribution"]["mean"]["pct_diff"]
            out.append(val)
        return out

    def _pairs_c2st(name: str) -> list[float]:
        return [
            s["geometry"]["psid_family_earnings_pairs"]["scores"]["c2st_auc"]
            for s in results[name]
        ]

    def _pooled(vals: list[float | None]) -> float | None:
        defined = [v for v in vals if v is not None]
        return float(np.mean(defined)) if defined else None

    v0_q0 = _q0_series("V0")
    vs_q0 = _q0_series("V0-shared")
    return {
        "available": True,
        "note": (
            "V0 (zero-anchor refit) vs V0-shared (candidate 7's original "
            "shared gate); identical otherwise, so the difference isolates "
            "the zero-anchor participation refit at inner scale"
        ),
        "pooled_q0_mean_pct_diff": {
            "V0": _pooled(v0_q0),
            "V0-shared": _pooled(vs_q0),
            "refit_effect": (
                _pooled(v0_q0) - _pooled(vs_q0)
                if _pooled(v0_q0) is not None and _pooled(vs_q0) is not None
                else None
            ),
        },
        "per_seed_q0_mean_pct_diff": {
            "seeds": list(SEEDS),
            "V0": v0_q0,
            "V0-shared": vs_q0,
        },
        "pairs_c2st": {
            "V0": _pairs_c2st("V0"),
            "V0-shared": _pairs_c2st("V0-shared"),
            "note": (
                "the refit should barely move the pairs C2ST (it changes the "
                "zero-anchor participation law, not the geometry of positive "
                "transitions)"
            ),
        },
    }


def _print_ranking(ranking: dict[str, Any]) -> None:
    """Human-readable ranking table to stdout."""
    print("\n==== inner-sweep ranking (reported-not-gated, inner scale) ====")
    print(
        f"{'variant':12s} {'clears':6s} {'geo':4s} {'bat':4s} "
        f"{'pairsC2ST(min_m)':17s} {'ac10(min_m)':13s} {'Q0%(m)':12s}"
    )
    for r in ranking["table"]:
        fm = r["focal_margins"]
        pc = fm["pairs_c2st"]
        ac = fm["battery_10yr"]
        q0 = fm["pooled_q0"]

        def _fmt(x: Any) -> str:
            return f"{x:+.4f}" if isinstance(x, (int, float)) else "  n/a "

        print(
            f"{r['variant']:12s} "
            f"{'YES' if r['clears_all_on_ge4_seeds'] else 'no':6s} "
            f"{r['n_geometry_pass']}/5  {r['n_battery_pass']}/5  "
            f"{pc['n_pass']}/5 {_fmt(pc['min_margin'])}  "
            f"{ac['n_pass']}/5 {_fmt(ac['min_margin'])}  "
            f"{_fmt(q0['pct_diff'])}"
        )


def _load_benefit_oracle() -> tuple[Any, Any]:
    """Load the SSA oracle params, or (None, None)."""
    try:
        from populace_dynamics.ss.params import load_ssa_parameters

        params = load_ssa_parameters()
        return params, None
    except Exception as exc:  # noqa: BLE001
        print(f"benefit-space oracle unavailable ({exc!r})")
        return None, None


def _revision_pins(benefit_params: Any) -> dict[str, Any]:
    """Repo/populace SHAs, schema version, and the SSA oracle pin."""
    import subprocess

    def _sha(cwd: Path) -> str | None:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode()
                .strip()
            )
        except Exception:
            return None

    populace_root = Path("~/PolicyEngine/populace").expanduser()
    pins = {
        "populace_dynamics_sha": _sha(ROOT),
        "populace_repo_sha": _sha(populace_root),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "reported_not_gated": True,
    }
    if benefit_params is not None:
        pins["pe_us_revision"] = getattr(
            benefit_params, "pe_us_revision", None
        )
    return pins


def main() -> None:
    run(verbose=True)


if __name__ == "__main__":
    main()
