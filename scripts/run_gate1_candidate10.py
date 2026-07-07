"""Gate-1 candidate 10: the inner-validated composition (amended gate).

The TWELFTH pre-registered model run of PolicyEngine/populace-dynamics, and
the first whose every constant was selected by NESTED validation (PR #63: an
inner-validation harness that mirrors the amended gate on splits carved from
each outer seed's TRAIN complement, with zero outer-holdout contact) rather
than by outer-gate feedback. This candidate IS the inner sweep's ``V1-lam0.1``
variant at OUTER scale: candidate 7's k-NN conditional-rank-bootstrap
machinery, plus candidate 9's zero-anchor participation regime with its two
poisons removed, plus a FIXED lambda = 0.1 donor-coordinate blend for the
non-Q0 targets.

The candidate-10 spec is registered, frozen before the run, in issue #42's
candidate-10 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4902561460);
every rule below is pinned there and implemented literally. There is no
calibration stage in this run (lambda is FIXED, not chosen against any
score), no re-entry-pool restriction, and no fit-time freedom.

The candidate, per the frozen spec (candidate 7's machinery verbatim plus):

1. **Zero-anchor participation regime** (from candidate 9, poisons removed).
   The participation gate for zero-anchor holdout persons refits on train
   pairs of zero-anchor persons (same features, populace-fit defaults). NO
   re-entry pool restriction: the FULL re-entry pool serves every target (the
   inner sweep's V0 showed pooled Q0 only -0.97% without the restriction vs
   candidate 9's -17.9% with it -- the restriction coupled Q0 to chronically
   low-rank donors). Q0 targets are EXEMPT from every memory coordinate: their
   third distance term stays candidate 7's ``|u_A(donor) - u_A(target)|`` (at
   lambda > 0 the blend turned the Q0 near-constant anchor coordinate into a
   low-rank-donor selector, swinging pooled Q0 to -17.9%).
2. **Fixed lambda = 0.1 donor-coordinate blend** for NON-Q0 targets: the
   third distance term is ``|0.1 * u_w(donor) + 0.9 * u_A(donor) -
   u_A(target)|`` at the standing 0.25 weight, ``u_w`` per candidate 8's
   convention (the shrunk permanent rank from candidate 3's z-decomposition).
   lambda = 0.1 is the inner sweep's winner (battery 4/5 inner seeds; pooled
   Q0 +1.19%; pairs-c2st inner mean margin +0.0007 where the V0-vs-c7 pairing
   measures inner c2st ~0.005-0.01 hotter than outer).

Everything else -- k = 25, the 1/0.5 lag weights, the weighted single-record
draw, no smoothing/jitter, the rank machinery, the gap rule, the re-entry
pools, and the substream seeding -- is byte-identical to the candidate-7
registration. In particular the generation RNG substreams are candidate 7's
two-element ``SeedSequence([seed, code])`` streams (gate / donor-draw /
re-entry-draw), NOT the inner sweep's variant-coded three-element streams (the
sweep coded variants to keep six variants' draws independent within one
process; this single-model outer run uses candidate 7's canonical seeding, as
candidate 9 did).

The generation code paths mirror the inner sweep's ``generate_variant`` at
``memory_mode="lambda_blend"``, ``lam=0.1``, ``use_zero_anchor_gate=True``
EXACTLY: the substream-agnostic draw helpers
:func:`run_inner_sweep._transition_draw` and
:func:`run_inner_sweep._reentry_draw` (which encode the Q0-exempt split, the
full-pool re-entry, and the fixed draw order) are imported byte-for-byte and
driven with the candidate-7 substreams.

The protocol mechanics -- the filter-first load, the person-disjoint 0.2
split per seed, the two locked views, ``panel_scorecard`` scoring, the battery
on the candidate panel vs the committed ``battery_reference`` with locked
definitions, the thresholds read from ``gates.yaml`` at runtime, and the
battery-reference bit-exact precheck -- are IMPORTED from the merged baseline
runner (:mod:`run_gate1_baseline`, PR #40), byte-for-byte the prior runs'.
The rank machinery and the shared participation gate are candidate 5b's (PR
#52); the k-NN draw and anchor quintiles are candidate 7's (PR #55); the u_w
decomposition is candidate 8's (PR #58); the donor pools, the zero-anchor
participation gate, and the AMENDED-gate scoring (the gated benefit-space
block, the per-seed benefit gates, the pooled Q0 gate, the Q0-participation
diagnostics) are candidate 9's (PR #62); and the Q0-exempt full-pool draw
helpers are the inner sweep's (PR #63).

Scored under the amended gate (gates.yaml gate_1, ratified 2026-07-06, PR
#57/#59): the runs-view c2st is demoted to reported-not-gated, a gated
benefit-space block folds its per-seed metrics into each seed's geometry
verdict, and the gate passes iff >=4/5 seeds pass geometry AND >=4/5 seeds
pass battery AND the pooled Q0 band (abs pooled-mean Q0 % <= 5) holds. This is
the amended conjunction run 11 (candidate 9) used.

Determinism. Stage-1 marginals, the u_w decomposition, and the stage-3 donor
pools are deterministic given the split (pure counting/NNLS, no RNG). The
generation draws each of the gate, donor-draw, and re-entry-draw substreams
from its own fixed-label substream of the gate seed, in the batched-by-step,
``person_id``-ordered pass the candidate-2 chain uses. The run reproduces from
the seeds alone. There is no calibration stage, so there is no lambda to
reproduce -- lambda is the fixed constant 0.1.

Environment. The donor pools, the u_w decomposition, and the k-NN draws are
pure numpy/scipy; the two participation gates are ``RegimeGatedQRF`` sign
gates and DO need populace-fit; the benefit-space block needs the SSA oracle
(``POPULACE_DYNAMICS_PE_US_DIR`` -> the pinned policyengine-us checkout). Run
the full gate from the repository root with the PSID family files staged,
using the DEDICATED gate venv (populace-fit pins scikit-learn < 1.9)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_gate1_candidate10.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The protocol machinery is IMPORTED from the merged baseline runner so that
# the filtered-panel load, the person-disjoint split, the view construction,
# the battery definitions, the geometry / battery checks, the threshold
# loading, and the battery-reference reproduction are byte-for-byte identical
# to every prior gate-1 run.
from run_gate1_baseline import (  # noqa: F401 (re-exported for tests)
    AGE_MAX,
    AGE_MIN,
    BATTERY_REFERENCE_RUN,
    PERIOD_MAX,
    PERIOD_MIN,
    PERIOD_STEP,
    SEEDS,
    build_backward_pairs,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    load_filtered_panel,
    load_gate1_thresholds,
    reproduce_battery_reference,
    split_holdout_train,
)

# The rank machinery (cells, CellMarginal, fit_cell_marginals, anchors,
# anchor_u, age_bin) and the shared participation gate's sign-draw are
# candidate 5b's (byte-for-byte the prior runs').
from run_gate1_candidate5b import (  # noqa: F401 (re-exported for tests)
    N_AGE_BINS,
    RANK_CLAMP_HI,
    RANK_CLAMP_LO,
    CellMarginal,
    _gate_sign_draw,
    age_bin,
    anchor_rows,
    anchor_u,
    fit_cell_marginals,
)

# Candidate 7's k-NN weighted single-record draw, its frozen distance weights,
# the anchor-quintile map (diagnostics), and its constants are imported
# byte-for-byte (the draw rule, the 1/0.5/0.25 lag weights, and k = 25 are
# unchanged).
from run_gate1_candidate7 import (  # noqa: F401 (re-exported for tests)
    K_NEIGHBORS,
    N_ANCHOR_QUINTILES,
    W_ANCHOR,
    W_NEXT,
    W_NEXT2,
    _knn_draw,
    anchor_quintile,
)

# Candidate 9's donor-pool builder (carries u_w and the anchor-zero flag), its
# two participation-gate fits, the blend, the reported-not-gated generation
# diagnostics assembler, and -- crucially -- the AMENDED-gate scoring
# (measure_benefit_space, the per-seed benefit gates, the pooled Q0 gate, the
# benefit-space pooling, the Q0-participation diagnostics) are imported
# byte-for-byte, so candidate 10 is scored on the identical amended-gate
# arithmetic run 11 used. Only the generation differs (fixed lambda, no
# re-entry restriction, Q0 memory-exempt).
from run_gate1_candidate9 import (  # noqa: F401 (re-exported for tests)
    _donor_blend,
    _generation_diagnostics,
    _pool_benefit_space,
    _q0_participation_diagnostics,
    anchor_u_by_person,
    build_donor_pools,
    check_benefit_space_per_seed,
    check_pooled_q0,
    fit_participation_gate,
    fit_zero_anchor_participation_gate,
    measure_benefit_space,
    q0_participation_seed,
)

# The Q0-exempt, full-re-entry-pool k-NN draw helpers are the inner sweep's,
# imported byte-for-byte. They take the RNG as a parameter, so candidate 10
# drives them with the candidate-7 substreams while their draw logic (the
# non-Q0/Q0 split, the fixed draw order, the full pools) is EXACTLY the inner
# sweep's V1-lam0.1 generation code path.
from run_inner_sweep import (  # noqa: F401 (re-exported for tests)
    _reentry_draw,
    _transition_draw,
)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_knn_v4.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank_knn.v4"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4902561460"
)
#: The candidate-7 registration this run's base machinery comes from.
BASE_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4896132094"
)
#: The candidate-8 registration the donor permanent rank u_w comes from.
UW_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4897723604"
)
#: The candidate-9 registration the amended-gate scoring + zero-anchor regime
#: come from (this run keeps its gate refit, drops its re-entry restriction and
#: its SMM lambda, and holds Q0 targets memory-exempt).
C9_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4898825218"
)
#: The pre-run forecast comment (logged before the run).
FORECAST_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4902561584"
)

# ---- Frozen constants of the candidate-10 registration -----------------
#: The FIXED donor-coordinate blend weight for NON-Q0 targets. lambda = 0.1 is
#: the inner sweep's winner; it is NOT calibrated against any score in this
#: run (there is no calibration stage). lambda = 0 is candidate 7; lambda = 1
#: candidate 8; 0.1 is a small, fixed memory injection. Pinned a priori.
LAMBDA_FIXED = 0.1

#: Fixed integer codes for the generation RNG substream labels, byte-for-byte
#: candidate 7's. Each label seeds an independent generator via
#: SeedSequence([seed, code]); the three streams are distinct and reproducible
#: from the gate seed. There is NO wcarry stream (that is a V2-only artifact of
#: the inner sweep; lambda_blend never draws from it) and NO variant code (this
#: single model uses candidate 7's canonical two-element seeding).
SUBSTREAM_CODES = {"gate": 1, "donor-draw": 2, "re-entry-draw": 3}


def _substream(seed: int, label: str) -> np.random.Generator:
    """A generation RNG for one fixed substream label off the gate seed.

    Byte-for-byte candidate 7's ``_substream``: two-element
    ``SeedSequence([seed, code])``, no variant code. This is the substream
    seeding the candidate-10 registration pins as "byte-identical to the
    candidate-7 registration".
    """
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


# --------------------------------------------------------------------------
# Stage 4/5 -- generation (holdout): backward k-NN chain + zero-anchor regime,
# fixed lambda = 0.1 blend for non-Q0 targets, full re-entry pool, Q0 exempt.
# --------------------------------------------------------------------------
def generate_candidate(
    holdout: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted_shared: Any,
    fitted_zero: Any,
    pools: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Candidate-10 backward k-NN chain over the holdout (V1-lam0.1, outer).

    This is the inner sweep's ``generate_variant`` at ``memory_mode=
    "lambda_blend"``, ``lam=0.1``, ``use_zero_anchor_gate=True`` -- restated
    here with candidate 7's two-element substream seeding (not the inner
    sweep's variant-coded seeding) and driving the inner sweep's
    substream-agnostic draw helpers (:func:`_transition_draw`,
    :func:`_reentry_draw`) byte-for-byte, so the draw logic is the sweep's
    V1-lam0.1 path exactly.

    For each holdout person: set the continuous anchor rank ``u_A`` and the
    exact zero-anchor flag, keep the anchor at its REAL earnings, and chain
    BACKWARD one conditional draw per observed transition. At each step, for
    each present person, draw participation from the zero-anchor gate
    (zero-anchor persons; falls back to the shared gate if the zero-anchor
    pool was empty) or the shared gate (positive-anchor persons); where
    positive:

    * **NON-Q0 (positive-anchor) targets** use the fixed-lambda blend as their
      third distance coordinate: ``0.25 |0.1*u_w(donor) + 0.9*u_A(donor) -
      u_A(target)|`` on transitions and ``|blend - u_A(target)|`` on re-entry.
    * **Q0 (zero-anchor) targets** are EXEMPT: their third distance term is
      candidate 7's ``|u_A(donor) - u_A(target)|`` (donor coordinate u_A, no
      memory), and re-entry uses the FULL pool (no restriction).

    The blend coordinate is formed once per pool at ``lam=0.1``. The target
    side is always ``u_A`` (exactly candidate 7/8/9). Returns
    ``(candidate, diagnostics)``; the candidate holds exactly the holdout
    persons on their observed periods (only earnings generated, anchor kept),
    and diagnostics carry the reported-not-gated distributions.
    """
    lam = LAMBDA_FIXED

    rng_gate = _substream(seed, "gate")
    rng_donor = _substream(seed, "donor-draw")
    rng_reentry = _substream(seed, "re-entry-draw")

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
    anchor_rank_vals = np.array(
        [anchor_rank_of_person[int(p)] for p in target_ids], dtype=np.float64
    )
    ha_all = all_anchor[
        all_anchor.person_id.isin(set(int(x) for x in target_ids))
    ]
    zero_anchor_person = {
        int(r.person_id): bool(float(r.earnings) == 0.0)
        for r in ha_all.itertuples(index=False)
    }

    # Presorted donor arrays (the pinned tie-break order from
    # build_donor_pools). The blended donor THIRD coordinate for NON-Q0
    # targets is formed once per pool at lam = 0.1 (constant across query
    # rows). The Q0 branch always uses the donor anchor rank u_A.
    tri_u_next = triples_pool["u_next"]
    tri_u_next2 = triples_pool["u_next2"]
    tri_u_A = triples_pool["u_A"]
    tri_blend = _donor_blend(triples_pool["u_w"], tri_u_A, lam)
    tri_w = triples_pool["weight"]
    tri_u_prev = triples_pool["u_prev"]
    pair_u_next = pairs_pool["u_next"]
    pair_u_A = pairs_pool["u_A"]
    pair_blend = _donor_blend(pairs_pool["u_w"], pair_u_A, lam)
    pair_w = pairs_pool["weight"]
    pair_u_prev = pairs_pool["u_prev"]
    re_u_A = reentry_pool["u_A"]
    re_blend = _donor_blend(reentry_pool["u_w"], re_u_A, lam)
    re_w = reentry_pool["weight"]
    re_u_prev = reentry_pool["u_prev"]

    # Diagnostics accumulators.
    n_clamped = 0
    n_positive_gen = 0
    n_triple_draws = 0
    n_pair_draws = 0
    n_reentry_draws = 0
    n_nonq0_memory_terms = 0
    neighbor_dists: list[float] = []
    corner_bottom = np.zeros(N_ANCHOR_QUINTILES, dtype=np.int64)
    corner_top = np.zeros(N_ANCHOR_QUINTILES, dtype=np.int64)
    corner_total = np.zeros(N_ANCHOR_QUINTILES, dtype=np.int64)

    # Backward chain: step j generates the rank-j (earlier) period from the
    # rank-(j-1) (next/later) period already fixed. v1 = rank of period j-1
    # if positive; v2 = rank of period j-2 if it exists and positive.
    for j in range(1, max_depth):
        positions = np.nonzero(ranks == j)[0]
        if positions.size == 0:
            continue
        # Canonical person_id order within the step.
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
        # (component 2), positive-anchor persons the shared gate. One uniform
        # per row from the gate substream in the fixed step order.
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
            gate_za = fitted_zero if fitted_zero is not None else fitted_shared
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

            # NON-Q0 memory target coordinate is the anchor rank a (the blend
            # is applied to the DONOR side; the target side stays u_A exactly
            # as candidate 7/8/9). Q0 rows also use a (exempt). So the target
            # coordinate is a_local for every row; the non-Q0 vs Q0 difference
            # is which DONOR coordinate they match against (blend vs u_A),
            # handled inside the draw helpers by the za_local split.
            third_target = a_local.copy()

            # v1 = rank of the (positive) next period at its own cell.
            v1 = np.full(pos_local.size, np.nan, dtype=np.float64)
            kp = np.nonzero(next_is_pos)[0]
            for m in kp:
                li = pos_local[m]
                gpos = next_positions[li]
                cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                v1[m] = cell.rank(float(gen_earn[gpos]))

            # v2 = rank of the two-steps-later (rank j-2) period, if it exists
            # for this person AND was generated positive.
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
            # za_local: non-Q0 rows use the fixed-lambda blend donor
            # coordinate, Q0 rows use candidate 7's anchor coordinate with no
            # memory term (the banked-findings exemption), ALL on the FULL
            # pools (no re-entry restriction). The draw helpers are the inner
            # sweep's, driven with the candidate-7 substreams.

            # ---- Branch A: next zero -> re-entry (full pool, no restriction).
            rp = np.nonzero(~next_is_pos)[0]
            if rp.size:
                _reentry_draw(
                    rp,
                    za_local,
                    third_target,
                    a_local,
                    re_blend,
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
                    None,
                    tri_u_next,
                    tri_u_next2,
                    tri_blend,
                    tri_u_A,
                    tri_w,
                    tri_u_prev,
                    "lambda_blend",
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
                    None,
                    pair_u_next,
                    None,
                    pair_blend,
                    pair_u_A,
                    pair_w,
                    pair_u_prev,
                    "lambda_blend",
                    rng_donor,
                    u_prev_local,
                    neighbor_dists,
                    triple=False,
                )
                n_pair_draws += int(pp.size)

            n_nonq0_memory_terms += int((~za_local).sum())

            # Corner-mass bookkeeping by the query person's anchor quintile.
            q_local = anchor_quintile(a_local)
            for m in range(pos_local.size):
                qm = int(q_local[m])
                corner_total[qm] += 1
                if u_prev_local[m] < 0.05:
                    corner_bottom[qm] += 1
                elif u_prev_local[m] > 0.95:
                    corner_top[qm] += 1

            # Earnings = Qhat_pos of the CURRENT (rank-j) period's cell.
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
            n_positive_gen += int(pos_local.size)
        gen_earn[positions] = vals

    out = hp[["person_id", "period", "earnings", "age", "weight"]].copy()
    out["earnings"] = gen_earn

    n_zero_anchor_holdout = int(
        sum(1 for p in target_ids if zero_anchor_person[int(p)])
    )
    # The reported-not-gated diagnostics are candidate 9's assembler, called
    # with n_reentry_draws_q0 = 0: candidate 10 has NO zero-anchor-restricted
    # re-entry pool (the full pool serves every target), so the "restricted"
    # re-entry-draw count is zero by construction. pools["n_reentry_q0"] (built
    # but unused) is still reported for provenance.
    diagnostics = _generation_diagnostics(
        pools,
        anchor_rank_vals,
        marginals,
        n_positive_gen,
        n_clamped,
        n_triple_draws,
        n_pair_draws,
        n_reentry_draws,
        0,
        np.asarray(neighbor_dists, dtype=np.float64),
        corner_bottom,
        corner_top,
        corner_total,
        int(len(target_ids)),
        n_zero_anchor_holdout,
    )
    diagnostics["lambda"] = float(lam)
    diagnostics["memory_mode"] = "lambda_blend"
    diagnostics["n_nonq0_memory_terms"] = int(n_nonq0_memory_terms)
    diagnostics["reentry_pool"] = "full (no zero-anchor restriction)"
    diagnostics["q0_memory_exempt"] = True
    return out, diagnostics


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_seed(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    view_specs: dict[str, Any],
    views_cfg: dict[str, Any],
    battery_reference: dict[str, float],
    battery_tol: dict[str, float],
    benefit_metrics_cfg: dict[str, Any],
    benefit_params: Any,
    benefit_cutpoints: np.ndarray | None,
    verbose: bool,
) -> dict[str, Any]:
    """Fit, generate at the fixed lambda, and score candidate 10 for one seed.

    Candidate 9's ``run_seed`` with the SMM-lambda stage removed (lambda is the
    fixed constant 0.1) and the candidate-10 generation (full re-entry pool,
    Q0 memory-exempt). The benefit-space block is GATED under the amendment;
    when ``benefit_params`` is provided it is measured and its per-seed gates
    fold into the seed's geometry verdict. When it is ``None`` the block is
    skipped and the seed's geometry verdict is the locked geometry thresholds
    only -- but the amended gate is not fully evaluable, so :func:`run` refuses
    to publish a verdict without the oracle.
    """
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Stage 1: per-cell marginals on the train complement.
    marginals = fit_cell_marginals(train)

    # Component 1: donor permanent rank u_w from the z-panel (candidate 8's
    # build_donor_uw, byte-for-byte). Imported lazily (populace-fit chain).
    import run_gate1_candidate8 as c8

    uw = c8.build_donor_uw(train, marginals)

    # Stage 3: donor pools (pairs, triples, re-entry) carrying u_w, u_A, and
    # the anchor-zero flag (candidate 9's builder; the reentry_q0 pool it also
    # builds is UNUSED by candidate 10 -- the full re-entry pool serves every
    # target -- but reported for provenance).
    pools = build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )

    # Component 2: the shared participation gate AND the zero-anchor gate
    # (refit on the zero-anchor train subpopulation). Both KEPT (only the
    # re-entry-pool restriction and the SMM lambda are dropped from candidate
    # 9).
    fitted_shared, pairs = fit_participation_gate(train, seed)
    fitted_zero, n_zero_pairs = fit_zero_anchor_participation_gate(
        train, all_anchor, seed
    )

    # Stage 4/5: backward k-NN chain over the holdout at the FIXED lambda = 0.1
    # blend (non-Q0 targets), full re-entry pool, Q0 memory-exempt.
    candidate, diagnostics = generate_candidate(
        holdout,
        all_anchor,
        marginals,
        fitted_shared,
        fitted_zero,
        pools,
        seed,
    )

    # --- geometry (locked thresholds): candidate vs holdout on both views ---
    geometry_by_view: dict[str, Any] = {}
    geometry_thresholds_pass = True
    n_windows: dict[str, int] = {}
    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(candidate, holdout, view, seed=seed)
        checks = check_geometry(scores, views_cfg[vname]["geometry"])
        view_pass = all(c["pass"] for c in checks.values())
        geometry_thresholds_pass = geometry_thresholds_pass and view_pass
        cand_windows, _ = hpanel.project_panel(candidate, view)
        n_windows[vname] = int(len(cand_windows))
        geometry_by_view[vname] = {
            "scores": {k: float(v) for k, v in scores.items()},
            "thresholds": views_cfg[vname]["geometry"],
            "checks": checks,
            "view_pass": bool(view_pass),
        }

    # --- battery: on the CANDIDATE panel, vs committed reference ---
    battery_values = compute_battery(candidate)
    battery_checks = check_battery(
        battery_values, battery_reference, battery_tol
    )
    battery_seed_pass = all(c["pass"] for c in battery_checks.values())

    # Q0 participation diagnostic (reported): generated vs real all-zero share
    # and mean positive periods for the zero-anchor holdout subgroup.
    q0_participation = q0_participation_seed(holdout, candidate, all_anchor)

    # --- benefit-space block (GATED under the amendment) ---
    benefit_space: dict[str, Any] | None = None
    benefit_seed: dict[str, Any] | None = None
    benefit_space_seed_pass: bool | None = None
    if benefit_params is not None and benefit_cutpoints is not None:
        benefit_space = measure_benefit_space(
            seed,
            holdout,
            candidate,
            all_anchor,
            benefit_params,
            benefit_cutpoints,
        )
        benefit_seed = check_benefit_space_per_seed(
            benefit_space, benefit_metrics_cfg
        )
        benefit_space_seed_pass = benefit_seed["benefit_space_seed_pass"]

    # Amended geometry verdict: locked geometry thresholds AND the per-seed
    # benefit-space metrics. If the benefit block is absent the amended
    # geometry verdict is not computable; record the locked-thresholds-only
    # verdict and mark it as such.
    if benefit_space_seed_pass is not None:
        geometry_seed_pass = bool(
            geometry_thresholds_pass and benefit_space_seed_pass
        )
    else:
        geometry_seed_pass = bool(geometry_thresholds_pass)

    result = {
        "seed": seed,
        "n_persons": int(holdout.person_id.nunique()),
        "n_person_periods": int(len(holdout)),
        "n_train_persons": int(train.person_id.nunique()),
        "n_train_pairs": int(len(pairs)),
        "n_zero_anchor_train_pairs": int(n_zero_pairs),
        "n_windows": n_windows,
        "regimes": {
            "shared_participation_gate": fitted_shared.regimes(),
            "zero_anchor_participation_gate": (
                fitted_zero.regimes() if fitted_zero is not None else None
            ),
        },
        "lambda": float(LAMBDA_FIXED),
        "lambda_source": "fixed (inner-sweep winner; no calibration stage)",
        "uw_fit": uw["fit"],
        "pools": {
            "n_pairs": int(pools["n_pairs"]),
            "n_triples": int(pools["n_triples"]),
            "n_reentry": int(pools["n_reentry"]),
            "n_reentry_q0": int(pools["n_reentry_q0"]),
            "n_reentry_q0_note": (
                "the zero-anchor-restricted re-entry pool is built by the "
                "candidate-9 pool builder but UNUSED by candidate 10 (the full "
                "re-entry pool serves every target); reported for provenance"
            ),
        },
        "generation_diagnostics": diagnostics,
        "q0_participation": q0_participation,
        "geometry": geometry_by_view,
        "geometry_thresholds_pass": bool(geometry_thresholds_pass),
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if benefit_space is not None:
        result["benefit_space"] = benefit_space
        result["benefit_space_checks"] = benefit_seed["checks"]
        result["benefit_space_seed_pass"] = bool(benefit_space_seed_pass)
    if verbose:
        d = diagnostics
        bs = ""
        if benefit_space is not None:
            q0 = benefit_space["by_anchor_quintile"]["quintiles"].get("Q0", {})
            q0mean = (
                q0.get("distribution", {}).get("mean", {}).get("pct_diff")
                if q0.get("n_persons")
                else None
            )
            mean_pct = benefit_space["distribution"]["mean"]["pct_diff"]
            bs = (
                f" bs_mean%={mean_pct:+.2f} Q0_mean%={q0mean:+.2f} "
                f"bs_pass={benefit_space_seed_pass}"
                if (mean_pct is not None and q0mean is not None)
                else ""
            )
        pairs_c2st = geometry_by_view["psid_family_earnings_pairs"]["scores"][
            "c2st_auc"
        ]
        print(
            f"seed {seed}: lambda={LAMBDA_FIXED:.1f} "
            f"geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"pairs_c2st={pairs_c2st:.4f} "
            f"ac10={battery_values['autocorr_log_10yr']:.3f} "
            f"clamp={d['clamped_rank_share']['share']:.3f}{bs} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def _load_benefit_oracle() -> tuple[Any, np.ndarray | None]:
    """Load the SSA oracle params + full-panel quartile cuts, or (None, None).

    Byte-for-byte candidate 9's helper. Under the amendment the benefit-space
    block is GATED, so the oracle must be present for a verdict; :func:`run`
    refuses to publish otherwise. Returns ``(None, None)`` on failure and lets
    ``run`` decide.
    """
    try:
        import build_downstream_relevance as ds

        from populace_dynamics.ss.params import load_ssa_parameters

        params = load_ssa_parameters()
        panel = load_filtered_panel()
        all_anchor = anchor_rows(panel)
        cuts = ds.anchor_quintile_cutpoints(all_anchor)
        return params, cuts
    except Exception as exc:  # noqa: BLE001
        print(f"benefit-space oracle unavailable ({exc!r})")
        return None, None


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-10 run (amended gate)."""
    started = time.time()
    thresholds = load_gate1_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_1 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    views_cfg = thresholds["views"]
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }
    # The amended benefit-space block (gated) lives under
    # thresholds.benefit_space; its per-seed metrics and pooled Q0 gate are
    # read from gates.yaml at runtime (no threshold hardcoded).
    benefit_cfg = thresholds.get("benefit_space")
    if benefit_cfg is None:
        raise RuntimeError(
            "gates.yaml gate_1 thresholds carry no benefit_space block; the "
            "amended gate cannot be scored. Expected the ratified amendment "
            "(PR #57/#59) to be live."
        )
    benefit_metrics_cfg = benefit_cfg["metrics"]

    battery_ref_artifact = json.loads(
        (ROOT / BATTERY_REFERENCE_RUN).read_text()
    )
    battery_reference = battery_ref_artifact["battery_reference"]

    panel = load_filtered_panel()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    # Identical battery-reference bit-exact precheck as every prior run: the
    # battery code path must reproduce every committed reference value to float
    # precision before any candidate is scored.
    repro = reproduce_battery_reference(panel)
    if verbose:
        print(
            "battery_reference reproduced exactly: "
            f"{repro['all_committed_values_reproduced_exactly']}"
        )
    if not repro["all_committed_values_reproduced_exactly"]:
        raise RuntimeError(
            "Battery code path does not reproduce the committed "
            "battery_reference to float precision; refusing to proceed "
            "with a divergent definition."
        )

    # Anchors on the FULL filtered panel.
    all_anchor = anchor_rows(panel)

    # The amended gate's benefit-space block is GATED; the SSA oracle must be
    # present. Refuse to publish an amended verdict without it.
    benefit_params, benefit_cutpoints = _load_benefit_oracle()
    if benefit_params is None:
        raise RuntimeError(
            "The amended gate scores a GATED benefit-space block, but the "
            "SSA oracle did not load (set POPULACE_DYNAMICS_PE_US_DIR to the "
            "pinned policyengine-us checkout and rerun). Refusing to publish "
            "an amended-gate verdict without the block."
        )
    if verbose:
        print(
            "benefit-space oracle (GATED): pe_us_revision="
            f"{benefit_params.pe_us_revision}"
        )

    view_specs = {
        "psid_family_earnings_pairs": build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed.append(
            run_seed(
                seed,
                panel,
                all_anchor,
                view_specs,
                views_cfg,
                battery_reference,
                battery_tol,
                benefit_metrics_cfg,
                benefit_params,
                benefit_cutpoints,
                verbose,
            )
        )

    # Amended pass rule (gates.yaml, ratified 2026-07-06):
    #   seed passes geometry iff every locked geometry threshold on every view
    #     holds AND every per-seed benefit_space metric holds;
    #   seed passes battery iff every locked tolerance holds;
    #   gate passes iff >=4/5 geometry AND >=4/5 battery AND the pooled Q0
    #     gate holds.
    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    pooled_q0 = check_pooled_q0(per_seed, benefit_metrics_cfg)
    pooled_q0_pass = pooled_q0["pooled_q0_pass"]
    gate_pass = bool(
        geometry_gate_pass and battery_gate_pass and pooled_q0_pass
    )

    benefit_pooled = _pool_benefit_space(per_seed)
    q0_participation = _q0_participation_diagnostics(per_seed, panel)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_rank_knn_v4",
        "gate": "gate_1",
        "gate_variant": "amended (PR #57/#59): runs-view c2st demoted; "
        "benefit_space block gated; geometry conjoins per-seed benefit_space; "
        ">=4/5 geometry AND >=4/5 battery AND pooled Q0 (same conjunction as "
        "run 11 / candidate 9)",
        "spec_registration": SPEC_REGISTRATION,
        "base_registration": BASE_REGISTRATION,
        "uw_registration": UW_REGISTRATION,
        "c9_registration": C9_REGISTRATION,
        "forecast_registration": FORECAST_REGISTRATION,
        "changes": (
            "candidate 7's machinery verbatim plus the inner-validated "
            "composition (PR #63): (1) the candidate-9 zero-anchor "
            "participation regime with its two poisons removed -- the "
            "participation gate refit on the zero-anchor train subpopulation "
            "is KEPT, but the re-entry-pool restriction is DROPPED (the full "
            "re-entry pool serves every target) and Q0 (zero-anchor) targets "
            "are EXEMPT from every memory coordinate (their third distance "
            "term stays |u_A(donor) - u_A(target)|); (2) a FIXED lambda = 0.1 "
            "donor-coordinate blend for NON-Q0 targets: the third distance "
            "term is |0.1*u_w(donor) + 0.9*u_A(donor) - u_A(target)| at the "
            "0.25 weight, u_w candidate 8's shrunk permanent rank. There is NO "
            "calibration stage (lambda is fixed, not chosen against a score); "
            "candidate 9's train-side SMM lambda selection and its "
            "reentry_q0 restriction are NOT in this run."
        ),
        "description": (
            "Gate-1 candidate 10: the inner-validated composition -- the first "
            "candidate whose every constant was selected by NESTED validation "
            "(PR #63: an inner-validation harness mirroring the amended gate "
            "on splits carved from each outer seed's TRAIN complement, with "
            "zero outer-holdout contact) rather than by outer-gate feedback. "
            "It is the inner sweep's V1-lam0.1 variant at OUTER scale. "
            "Candidate 7's k-NN conditional rank bootstrap (empirical per-cell "
            "quantile marginals supply the magnitude; a continuous "
            "nonparametric transition law supplies the dynamics) with: (a) the "
            "candidate-9 zero-anchor participation gate (refit on the "
            "zero-anchor train subpopulation, same features and populace-fit "
            "defaults); (b) NO re-entry-pool restriction -- the full re-entry "
            "pool serves every target (the inner sweep's V0 showed pooled Q0 "
            "-0.97% without the restriction vs candidate 9's -17.9% with it); "
            "(c) Q0 targets EXEMPT from every memory coordinate (their third "
            "distance term stays candidate 7's |u_A(donor) - u_A(target)|); "
            "and (d) a FIXED lambda = 0.1 donor-coordinate blend for the "
            "non-Q0 targets -- the k-NN third distance term becomes "
            "|0.1*u_w(donor) + 0.9*u_A(donor) - u_A(target)| at candidate 7's "
            "0.25 weight, where u_w is candidate 8's shrunk permanent rank "
            "(candidate 3's stage-1 decomposition on the z-panel). lambda = "
            "0.1 is the inner sweep's winner (battery 4/5 inner seeds; pooled "
            "Q0 +1.19%; pairs-c2st inner mean margin +0.0007 where the "
            "V0-vs-c7 pairing measures inner c2st ~0.005-0.01 hotter than "
            "outer); it is FIXED, not calibrated against any score in this "
            "run. Everything else -- k=25, the 1/0.5 lag weights, the weighted "
            "single-record draw, no smoothing/jitter, the rank machinery, the "
            "gap rule, and the substream seeding (candidate 7's two-element "
            "SeedSequence([seed, code]) gate/donor-draw/re-entry-draw streams) "
            "-- is byte-identical to candidate 7. The generation code paths "
            "mirror the inner sweep's generate_variant at "
            "memory_mode=lambda_blend, lam=0.1, use_zero_anchor_gate=True "
            "exactly (its Q0-exempt full-pool draw helpers _transition_draw / "
            "_reentry_draw are imported byte-for-byte and driven with the "
            "candidate-7 substreams). Registered frozen before the run in "
            "issue #42 (see spec_registration). Scored under the AMENDED gate "
            "live in gates.yaml (PR #57/#59), the same conjunction run 11 "
            "used: geometry conjoins the locked geometry thresholds (runs-view "
            "c2st demoted to reported-not-gated) and the per-seed "
            "benefit-space metrics; the gate needs >=4/5 geometry AND >=4/5 "
            "battery AND the pooled Q0 band. Protocol machinery imported "
            "byte-for-byte from the baseline runner (PR #40); rank machinery "
            "and shared gate from candidate 5b (PR #52); the k-NN draw and "
            "anchor quintiles from candidate 7 (PR #55); the u_w decomposition "
            "from candidate 8 (PR #58); the donor pools, the zero-anchor gate, "
            "and the amended-gate scoring from candidate 9 (PR #62); the "
            "Q0-exempt full-pool draw helpers and the inner-validation "
            "selection from the design sweep (PR #63); the benefit-space "
            "functional from PR #56."
        ),
        "model": {
            "class": (
                "k-NN conditional rank bootstrap with a FIXED "
                "anchor/permanent-rank donor blend (lambda = 0.1) for non-Q0 "
                "targets and a zero-anchor participation regime with a full "
                "(unrestricted) re-entry pool and Q0 memory-exemption "
                "(quantile marginal + continuous empirical conditional draws)"
            ),
            "stochastic": True,
            "populace_fit_used": True,
            "populace_fit_scope": (
                "the shared and zero-anchor participation gates "
                "(RegimeGatedQRF sign gates); the donor pools, the u_w "
                "decomposition, and the k-NN draws use pure numpy/scipy"
            ),
            "calibration": (
                "none in this run: lambda is the FIXED constant 0.1 (the "
                "inner-sweep winner), not chosen against any score. The u_w "
                "decomposition is candidate 3's deterministic grid-rho NNLS on "
                "TRAIN only; the bootstrap has no rescaling freedom. Candidate "
                "9's train-side SMM lambda selection is NOT in this run."
            ),
            "inner_validation": {
                "selected_by": (
                    "nested validation (PR #63): the inner-validation harness "
                    "mirrors the amended gate on person-disjoint inner splits "
                    "carved from each outer seed's TRAIN complement "
                    "(split_panel_by_person(train, 'person_id', fraction=0.25, "
                    "seed=1000+s)); it never touches the outer holdout"
                ),
                "this_candidate_is": (
                    "the inner sweep's V1-lam0.1 variant at OUTER scale"
                ),
                "inner_evidence": (
                    "inner geometry 3/5, battery 4/5, pooled Q0 +1.19% "
                    "(inner_gate_pass False on geometry, one pairs-C2ST seed "
                    "short); pairs-C2ST inner mean margin +0.0007, worst inner "
                    "seed 0.5365; battery 10yr one miss at -0.005"
                ),
                "registered_risk": (
                    "the inner-to-outer c2st heat correction (~0.005-0.01) may "
                    "under-deliver on the worst inner seed (0.5365, needing "
                    "~0.007); one inner seed's 10-year rung sat 0.005 outside "
                    "tolerance; the pooled Q0 result must generalize from "
                    "inner (+1.19%) to outer scale"
                ),
                "sweep_artifact": "runs/inner_sweep_v1.json",
                "harness": "scripts/inner_validation.py",
            },
            "change_1_fixed_donor_blend": {
                "third_distance_term_nonq0": (
                    "|0.1*u_w(donor) + 0.9*u_A(donor) - u_A(target)| at weight "
                    "0.25 (transitions) / bare (re-entry); the TARGET side "
                    "stays u_A, exactly as candidate 7/8/9"
                ),
                "third_distance_term_q0": (
                    "|u_A(donor) - u_A(target)| (candidate 7 verbatim; Q0 "
                    "targets are memory-exempt, no blend)"
                ),
                "lambda": LAMBDA_FIXED,
                "lambda_fixed": True,
                "lambda_provenance": (
                    "the inner sweep's winner (V1-lam0.1): battery 4/5 inner "
                    "seeds, pooled Q0 +1.19%, pairs-c2st inner mean margin "
                    "+0.0007; FIXED, not calibrated against any outer or inner "
                    "score in THIS run"
                ),
                "u_w": (
                    "candidate 8's donor permanent rank u_w = "
                    "Phi(what/sigma_hat_w) from candidate 3's stage-1 "
                    "decomposition on the z-panel (imported byte-for-byte)"
                ),
            },
            "change_2_zero_anchor_participation_regime": {
                "trigger": (
                    "holdout persons with zero anchor earnings (their "
                    "chronologically last observed period is zero)"
                ),
                "participation_gate": (
                    "a RegimeGatedQRF sign gate refit ONLY on train pairs "
                    "whose person has zero anchor earnings, with the SAME "
                    "features (earnings at t, age at t-2; target earnings at "
                    "t-2; sample_weight the earlier-period weight) and the "
                    "SAME populace-fit defaults; a conditional refit, no dial "
                    "(candidate 9's component 2 gate, KEPT verbatim)"
                ),
                "reentry_pool": (
                    "the FULL re-entry pool serves every target -- NO "
                    "zero-anchor restriction (candidate 9's reentry_q0 "
                    "restriction is DROPPED; the inner sweep's V0 showed the "
                    "restriction swung pooled Q0 to -17.9% by coupling Q0 to "
                    "chronically low-rank donors, vs -0.97% without it)"
                ),
                "q0_memory_exempt": (
                    "Q0 (zero-anchor) targets are exempt from every memory "
                    "coordinate: their third distance term stays candidate 7's "
                    "|u_A(donor) - u_A(target)| (at lambda > 0 the blend turned "
                    "the Q0 near-constant anchor coordinate into a "
                    "low-rank-donor selector)"
                ),
                "positive_anchor": (
                    "positive-anchor (non-Q0) persons keep the shared gate and "
                    "use the fixed-lambda blend third term"
                ),
                "empty_pool_fallback": (
                    "if a seed's zero-anchor pair pool is empty the shared "
                    "gate is used (never observed empty on the real panel)"
                ),
                "not_candidate_8_attachment": (
                    "candidate 10 does NOT adopt candidate 8's attachment "
                    "distance; the non-Q0 third distance term is the "
                    "fixed-lambda blend, the Q0 third term the anchor rank"
                ),
            },
            "donor_pools": {
                "pairs": (
                    "train backward-adjacent pairs among positives: "
                    "(u_prev, u_next); each carries u_A, u_w, anchor_zero and "
                    "the earlier-period weight"
                ),
                "triples": (
                    "pairs whose person is also positive at the next-later "
                    "observed period after u_next: (u_prev, u_next, u_next2)"
                ),
                "reentry": (
                    "train pairs where the LATER period is zero and the "
                    "EARLIER is positive (candidate 6's re-entry pairs): "
                    "(u_prev); the FULL pool serves every target"
                ),
                "reentry_q0": (
                    "the zero-anchor-restricted subset is BUILT by the "
                    "candidate-9 pool builder but UNUSED by candidate 10; "
                    "reported for provenance"
                ),
                "tie_break_order": (
                    "records pinned in a stable (person_id, period_prev) sort "
                    "that fixes the k-NN tie-break (byte-for-byte candidate 7)"
                ),
            },
            "knn": {
                "k": K_NEIGHBORS,
                "distance_triples_nonq0": (
                    "|u_next - v1| + 0.5|u_next2 - v2| + "
                    "0.25|0.1*u_w + 0.9*u_A - a|"
                ),
                "distance_pairs_nonq0": (
                    "|u_next - v1| + 0.25|0.1*u_w + 0.9*u_A - a|"
                ),
                "distance_reentry_nonq0": ("|0.1*u_w + 0.9*u_A - a|"),
                "distance_triples_q0": (
                    "|u_next - v1| + 0.5|u_next2 - v2| + 0.25|u_A - a|"
                ),
                "distance_pairs_q0": "|u_next - v1| + 0.25|u_A - a|",
                "distance_reentry_q0": "|u_A - a|",
                "weights": {
                    "w_next": W_NEXT,
                    "w_next2": W_NEXT2,
                    "w_anchor": W_ANCHOR,
                },
                "draw": (
                    "one record drawn with probability proportional to its "
                    "weight among the k=25 nearest (seeded substream); "
                    "generated u_prev is that record's u_prev exactly "
                    "(byte-for-byte candidate 7's _knn_draw)"
                ),
                "no_smoothing": (
                    "no binning, no smoothing, no within-bin jitter; earnings "
                    "= Qhat_pos of the target cell at u_prev"
                ),
            },
            "generation": {
                "anchor": (
                    "chronologically last observed period held at real "
                    "earnings"
                ),
                "anchor_rank": (
                    "positive anchor u_A = rhat(anchor value) at its cell; "
                    "zero anchor u_A = p0/2 of the cell (continuous)"
                ),
                "chain": (
                    "backward one conditional draw per observed transition "
                    "(standing gap rule)"
                ),
                "participation": (
                    "positive-anchor: the shared candidate-2 backward regime "
                    "gate; zero-anchor: the zero-anchor gate (change 2). Both "
                    "on (next generated level, current age), populace-fit "
                    "defaults, drawn from the same gate substream in the fixed "
                    "step order"
                ),
                "code_path": (
                    "mirrors the inner sweep's generate_variant at "
                    "memory_mode=lambda_blend, lam=0.1, "
                    "use_zero_anchor_gate=True; the substream-agnostic draw "
                    "helpers run_inner_sweep._transition_draw / _reentry_draw "
                    "are imported byte-for-byte and driven with the "
                    "candidate-7 substreams (the Q0-exempt non-Q0/Q0 split, "
                    "the full-pool re-entry, and the fixed draw order are the "
                    "sweep's exactly)"
                ),
                "rng": (
                    "distinct fixed-label substreams of the gate seed: gate, "
                    "donor-draw, re-entry-draw (byte-for-byte candidate 7's "
                    "two-element SeedSequence([seed, code])); NO variant code, "
                    "NO wcarry stream (this single model uses candidate 7's "
                    "canonical seeding, not the inner sweep's variant-coded "
                    "streams)"
                ),
                "candidate_panel_pin": (
                    "exactly the holdout persons on exactly their observed "
                    "periods; only earnings generated; anchor keeps real "
                    "value; person_id/period/age/weight copied from holdout"
                ),
            },
        },
        "protocol": {
            "filter": (
                f"age {AGE_MIN}-{AGE_MAX}, reference years "
                f"{PERIOD_MIN}-{PERIOD_MAX}, positive weights (applied "
                "before the split)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel, 'person_id', fraction=0.2, seed=s); the drawn 20% "
                "is the holdout, the complement is the training set "
                "(imported from the baseline runner)"
            ),
            "seeds": list(SEEDS),
            "views": {
                "psid_family_earnings_pairs": {"window": 2, "period_step": 2},
                "psid_family_earnings_runs": {"window": 3, "period_step": 2},
            },
            "scoring": (
                "panel_scorecard(candidate, holdout, view, seed=s) per "
                "locked view; battery on the candidate panel vs committed "
                "battery_reference; benefit-space PIA-proxy gaps per seed "
                "(gated under the amendment)"
            ),
            "pass_rule": (
                "AMENDED (gates.yaml, ratified 2026-07-06): seed passes "
                "geometry iff every locked geometry threshold on every locked "
                "view holds AND every per-seed benefit_space metric (abs mean "
                "%, abs median %, each gated decile d3-d9 %, weighted KS) "
                "holds; seed passes battery iff every locked tolerance holds; "
                "gate passes iff >=4/5 geometry AND >=4/5 battery AND the "
                "pooled Q0 gate (abs pooled-mean Q0 % <= 5) holds"
            ),
        },
        "battery_reference_reproduction": repro,
        "battery_reference_run": BATTERY_REFERENCE_RUN,
        "per_seed": per_seed,
        "seed_conjunction": [
            {
                "seed": s["seed"],
                "lambda": s["lambda"],
                "geometry_thresholds_pass": s["geometry_thresholds_pass"],
                "benefit_space_seed_pass": s.get("benefit_space_seed_pass"),
                "geometry_pass": s["geometry_pass"],
                "battery_pass": s["battery_pass"],
            }
            for s in per_seed
        ],
        "lambda_by_seed": {str(s["seed"]): s["lambda"] for s in per_seed},
        "knn_context": {
            "note": (
                "Reported-not-gated per seed: the u_w decomposition, "
                "neighbor-distance distribution, triple-vs-pair usage share, "
                "donor-record reuse, drawn corner mass by anchor quintile, and "
                "the clamped-rank share. lambda is the fixed constant 0.1 (no "
                "calibration in this run). None of these enters the pass/fail "
                "beyond the amended gate's named blocks."
            ),
            "lambda": LAMBDA_FIXED,
            "lambda_fixed": True,
            "per_seed": [
                {
                    "seed": s["seed"],
                    "lambda": s["lambda"],
                    "n_pairs": s["pools"]["n_pairs"],
                    "n_triples": s["pools"]["n_triples"],
                    "n_reentry": s["pools"]["n_reentry"],
                    "n_reentry_q0_built_unused": s["pools"]["n_reentry_q0"],
                    "n_zero_anchor_train_pairs": s[
                        "n_zero_anchor_train_pairs"
                    ],
                    "uw_fit": s["uw_fit"],
                    "neighbor_distance_distribution": s[
                        "generation_diagnostics"
                    ]["neighbor_distance_distribution"],
                    "triple_pair_usage": s["generation_diagnostics"][
                        "triple_pair_usage"
                    ],
                    "donor_reuse": s["generation_diagnostics"]["donor_reuse"],
                    "drawn_corner_mass_by_anchor_quintile": s[
                        "generation_diagnostics"
                    ]["drawn_corner_mass_by_anchor_quintile"],
                    "anchor_rank_distribution": s["generation_diagnostics"][
                        "anchor_rank_distribution"
                    ],
                    "cell_count_distribution": s["generation_diagnostics"][
                        "cell_count_distribution"
                    ],
                    "clamped_rank_share": s["generation_diagnostics"][
                        "clamped_rank_share"
                    ]["share"],
                }
                for s in per_seed
            ],
        },
        "benefit_space_gated": {
            "note": (
                "GATED under the amendment. Per-seed metrics fold into the "
                "geometry verdict; the pooled Q0 gate is a standalone gate "
                "condition."
            ),
            "metrics_config": benefit_metrics_cfg,
            "pooled_q0_gate": pooled_q0,
            "pooled": benefit_pooled,
            "per_seed_checks": [
                {
                    "seed": s["seed"],
                    "benefit_space_seed_pass": s.get(
                        "benefit_space_seed_pass"
                    ),
                    "checks": s.get("benefit_space_checks"),
                }
                for s in per_seed
            ],
        },
        "q0_participation_diagnostics": q0_participation,
        "verdict": {
            "n_seeds": len(SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "pooled_q0_pass": bool(pooled_q0_pass),
            "pooled_q0_mean_pct_diff": pooled_q0["pooled_q0_mean_pct_diff"],
            "gate_1_pass": gate_pass,
            "rule": (
                ">=4/5 seeds geometry (locked geometry thresholds AND "
                "per-seed benefit_space) AND >=4/5 seeds battery AND pooled "
                "Q0 gate"
            ),
        },
        "revision_pins": _revision_pins(benefit_params),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5, "
            f"pooled_Q0={v['pooled_q0_mean_pct_diff']:+.2f} "
            f"pass={v['pooled_q0_pass']})"
        )
    return artifact


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
        "gates_yaml_locked": True,
    }
    if benefit_params is not None:
        pins["pe_us_revision"] = getattr(
            benefit_params, "pe_us_revision", None
        )
    return pins


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
