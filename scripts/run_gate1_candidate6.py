"""Gate-1 candidate 6: anchored empirical rank-transition kernel.

The EIGHTH pre-registered model run of PolicyEngine/populace-dynamics.
Run 7 (candidate 5b) validated the rank-space architecture -- train-cell
empirical quantile marginals, the proven regime gate, real anchors --
and rejected latent-Gaussian dynamics (the mobility diagonal churned to
0.44-0.46 against the 0.604 +/- 0.05 band, and the classifier read the
latent-Gaussian joint at c2st 0.67/0.73). This candidate keeps the
architecture and replaces the latent dynamics with a NONPARAMETRIC
transition law estimated by COUNTING: an anchored empirical
rank-transition kernel. There is no calibration stage -- the kernel has
zero free parameters beyond the pinned discretization; the iterated
dynamics are whatever the counts imply, and the run publishes them.

The candidate-6 spec is registered, frozen before the run, in issue
#42's candidate-6 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4895401373);
every rule below -- the rank machinery, the anchor bins, the kernel
counting with add-one smoothing, the re-entry distributions, the
backward draws, and the substream labels -- is pinned there and
implemented LITERALLY. No threshold is hardcoded, no model choice is
tuned against holdout scores, and there is no fit-time freedom at all
(the counting estimator has no rescaling freedom). The run is one shot;
the outcome publishes whether it passes or fails.

The candidate, per the frozen spec (each stage implemented literally):

1. **Rank machinery** -- identical to the candidate-5b registration
   (five-year age bins x period cells; weighted empirical quantile
   function ``Qhat_pos`` and rank function ``rhat`` with the same
   interpolation and ``[0.001, 0.999]`` clamps; zero share ``p0`` per
   cell). Reused byte-for-byte from :mod:`run_gate1_candidate5b`.
2. **Anchor bins.** Each person's anchor rank ``u_A``: for positive
   anchors, ``rhat`` at the anchor's cell; for zero anchors, ``p0 / 2``
   of the cell. Anchor bin ``A`` = which fifth of ``(0, 1)`` ``u_A``
   falls in (five equal bins).
3. **Kernel estimation (train, counting only).** For every train
   backward-adjacent pair among positives (both periods positive,
   consecutive observed periods), compute ``(u_prev, u_next)`` via
   ``rhat`` in each period's own cell, and the person's anchor bin
   ``A``. Accumulate weighted counts into a 5 x 20 x 20 array
   ``N[A, bin(u_next), bin(u_prev)]`` over twenty equal rank bins, using
   the pair's earlier-period person weight. Add-one (Laplace) smoothing
   per row; row-normalize to a transition kernel ``P[A, j, .]``.
4. **Re-entry distribution (train, counting).** Per anchor bin, the
   weighted distribution over twenty rank bins of ``u_prev`` for train
   pairs where the LATER period is zero and the earlier is positive
   (rank at re-entry into work, walking backward); add-one smoothed and
   normalized. If a person's chain crosses a zero spell, the next
   positive period's rank draws from this distribution.
5. **Generation (holdout, backward).** Anchor keeps its real value; its
   rank ``u_A`` and bin ``A`` as in stage 2. Participation per step from
   the regime machinery exactly as in the candidate-2 registration
   (backward gate on next generated level and age, populace-fit
   defaults). For a positive step whose next period is positive: draw
   the previous rank bin ``j`` from ``P[A, bin(u_next), .]``, then
   ``u_prev`` uniform within bin ``j``. For a positive step whose next
   period is zero: draw from the anchor bin's re-entry distribution,
   then uniform within bin. Earnings = ``Qhat_pos`` of the period's cell
   at ``u_prev``. One kernel step per observed-period transition
   regardless of gap width (the standing gap rule). All RNG substreams
   seeded from the gate seed with fixed labels (gate, kernel-bin,
   within-bin, re-entry).

The protocol mechanics -- the filter-first load, the person-disjoint
0.2 split per seed, the two locked views, ``panel_scorecard`` scoring,
the battery on the candidate panel vs the committed ``battery_reference``
with locked definitions, the thresholds read from ``gates.yaml`` at
runtime, the seed-level conjunction (>=4/5 both blocks), and the
battery-reference bit-exact precheck -- are IMPORTED from the merged
baseline runner (:mod:`run_gate1_baseline`, pull request 40),
byte-for-byte the prior runs'. The rank machinery (cells,
``CellMarginal``, ``fit_cell_marginals``, anchors, ``anchor_u``) and the
participation gate (``fit_participation_gate``, ``_gate_sign_draw``) are
imported from the merged candidate-5b runner
(:mod:`run_gate1_candidate5b`, pull request 52). Only the kernel counting
and the backward kernel generation are local.

Determinism. Stage-1 marginals and stages 3-4 counting are deterministic
given the split (pure weighted counting, no RNG). Stage-5 generation
draws each of the gate, kernel-bin, within-bin, and re-entry substreams
from its own fixed-label substream of the gate seed, in the
batched-by-step, ``person_id``-ordered pass the candidate-2 chain uses.
The run reproduces from the seeds alone.

Environment. The kernel itself is pure numpy counting and needs NO
populace-fit; the participation gate is a ``RegimeGatedQRF`` sign gate
and DOES need populace-fit. Run the full gate from the repository root
with the PSID family files staged, using the DEDICATED gate venv
(populace-fit pins scikit-learn < 1.9, which the repo's ``.venv``
violates; see populace #318):

    .venv-gate/bin/python scripts/run_gate1_candidate6.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The protocol machinery is IMPORTED from the merged baseline runner so
# that the filtered-panel load, the person-disjoint split, the view
# construction, the battery definitions, the geometry / battery checks,
# the threshold loading, and the battery-reference reproduction are
# byte-for-byte identical to every prior gate-1 run.
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
# anchor_u) and the participation gate (fit_participation_gate,
# _gate_sign_draw) are IMPORTED from the merged candidate-5b runner so
# that the quantile/rank maps, the anchor-rank rule, and the
# candidate-2 backward sign-gate draw are byte-for-byte candidate 5b's.
# Only the kernel counting and the backward kernel generation are local.
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
    fit_participation_gate,
)
from scipy.stats import norm  # noqa: F401 (parity with 5b import surface)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_kernel_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank_kernel.v1"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4895401373"
)

# ---- Frozen constants of the registration (applied literally) ----------
#: Number of anchor bins: five equal fifths of (0, 1).
N_ANCHOR_BINS = 5
#: Number of rank bins for the transition kernel and re-entry
#: distributions: twenty equal bins of (0, 1).
N_RANK_BINS = 20
#: Add-one (Laplace) smoothing count added per kernel row and per
#: re-entry distribution before normalization.
LAPLACE = 1.0

#: Fixed integer codes for the generation RNG substream labels. Each
#: label seeds an independent generator via SeedSequence([seed, code]),
#: so the four streams are distinct and reproducible from the gate seed.
SUBSTREAM_CODES = {"gate": 1, "kernel-bin": 2, "within-bin": 3, "re-entry": 4}


def _substream(seed: int, label: str) -> np.random.Generator:
    """A generation RNG for one fixed substream label off the gate seed."""
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


# --------------------------------------------------------------------------
# Bin maps (five anchor bins; twenty rank bins) -- applied literally
# --------------------------------------------------------------------------
def anchor_bin(u_a: np.ndarray) -> np.ndarray:
    """Which fifth of ``(0, 1)`` the anchor rank falls in: index ``0..4``.

    Five equal bins; ``bin = clip(floor(u_A * 5), 0, 4)``. Anchor ranks
    are ``rhat`` outputs clamped to ``[0.001, 0.999]`` (positive anchors)
    or ``p0 / 2`` (zero anchors), so they lie strictly inside ``(0, 1)``.
    """
    u_a = np.asarray(u_a, dtype=np.float64)
    idx = np.floor(u_a * N_ANCHOR_BINS).astype(np.int64)
    return np.clip(idx, 0, N_ANCHOR_BINS - 1)


def rank_bin(u: np.ndarray) -> np.ndarray:
    """Twenty equal rank bins of ``(0, 1)``: index ``0..19``.

    ``bin = clip(floor(u * 20), 0, 19)``. Applied to ``rhat`` ranks (in
    ``[0.001, 0.999]``) when counting and to generated ``u_next`` values
    when drawing.
    """
    u = np.asarray(u, dtype=np.float64)
    idx = np.floor(u * N_RANK_BINS).astype(np.int64)
    return np.clip(idx, 0, N_RANK_BINS - 1)


# --------------------------------------------------------------------------
# Stage 2 helper - anchor rank and bin per person (uses 5b's anchor_u)
# --------------------------------------------------------------------------
def anchor_u_and_bin(
    marginals: dict[tuple[int, int], CellMarginal],
    all_anchor: pd.DataFrame,
    person_ids: np.ndarray,
) -> dict[int, tuple[float, int]]:
    """Anchor rank ``u_A`` and anchor bin ``A`` for a set of persons.

    ``u_A`` from the frozen rule (:func:`run_gate1_candidate5b.anchor_u`:
    positive anchor -> ``rhat`` at its cell; zero anchor -> ``p0 / 2`` of
    the cell), evaluated on the person's anchor row (their
    chronologically last observed period). ``A`` = :func:`anchor_bin` of
    ``u_A``. Returns ``{person_id: (u_A, A)}`` for the requested persons.
    """
    wanted = set(int(x) for x in person_ids)
    sub = all_anchor[all_anchor.person_id.isin(wanted)]
    out: dict[int, tuple[float, int]] = {}
    for row in sub.itertuples(index=False):
        u_a = anchor_u(
            marginals,
            float(row.earnings),
            float(row.age),
            int(row.period),
        )
        a = int(anchor_bin(np.array([u_a]))[0])
        out[int(row.person_id)] = (float(u_a), a)
    return out


# --------------------------------------------------------------------------
# Stages 3-4 - kernel and re-entry estimation (train, weighted counting)
# --------------------------------------------------------------------------
def estimate_kernel(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
) -> dict[str, Any]:
    """Count the anchored rank-transition kernel and re-entry laws.

    Builds the train backward-adjacent pairs (the candidate-2 /
    baseline ``build_backward_pairs``: one row per person-period ``t``
    whose ``t-2`` is observed, carrying earnings at ``t`` [the LATER
    period], earnings at ``t-2`` [the EARLIER period], and the earlier
    period's weight ``weight_tm2``). For each pair the person's anchor
    bin ``A`` is looked up from the anchor rows.

    * **Kernel** ``N[A, bin(u_next), bin(u_prev)]`` accumulates the
      earlier-period weight over pairs where BOTH endpoints are positive,
      with ``u_prev = rhat`` at the earlier period's own cell and
      ``u_next = rhat`` at the later period's own cell (twenty rank
      bins). Add-one smoothing per row, then row-normalize over
      ``bin(u_prev)`` to ``P[A, j, .]``.
    * **Re-entry** ``Reentry[A, bin(u_prev)]`` accumulates the
      earlier-period weight over pairs where the LATER period is zero and
      the EARLIER is positive (``u_prev = rhat`` at the earlier period's
      own cell). Add-one smoothing, then normalize to a distribution.

    Returns the normalized kernel ``P`` (5 x 20 x 20), the normalized
    re-entry distributions ``R`` (5 x 20), the raw (pre-smoothing)
    weighted pair counts per anchor bin (kernel and re-entry), and the
    unweighted pair counts per anchor bin (for the diagnostics).
    """
    pairs = build_backward_pairs(train)

    # Anchor bin per pair (via the pair's person_id).
    ab_map = anchor_u_and_bin(
        marginals, all_anchor, pairs["person_id"].to_numpy()
    )
    a_of_pair = np.array(
        [ab_map[int(pid)][1] for pid in pairs["person_id"].to_numpy()],
        dtype=np.int64,
    )

    earn_next = pairs["earnings"].to_numpy(dtype=np.float64)  # period t
    earn_prev = pairs["earnings_tm2"].to_numpy(dtype=np.float64)  # t-2
    age_next = pairs["age"].to_numpy(dtype=np.float64)
    age_prev = pairs["age_tm2"].to_numpy(dtype=np.float64)
    period_next = pairs["period"].to_numpy(dtype=np.int64)
    period_prev = (period_next - PERIOD_STEP).astype(np.int64)
    w_prev = pairs["weight_tm2"].to_numpy(dtype=np.float64)  # earlier weight
    bin_next_cell = age_bin(age_next)
    bin_prev_cell = age_bin(age_prev)

    # Raw weighted counts (pre-smoothing) and pair tallies per anchor bin.
    N = np.zeros((N_ANCHOR_BINS, N_RANK_BINS, N_RANK_BINS), dtype=np.float64)
    Rn = np.zeros((N_ANCHOR_BINS, N_RANK_BINS), dtype=np.float64)
    kernel_pairs_per_bin = np.zeros(N_ANCHOR_BINS, dtype=np.int64)
    reentry_pairs_per_bin = np.zeros(N_ANCHOR_BINS, dtype=np.int64)
    kernel_w_per_bin = np.zeros(N_ANCHOR_BINS, dtype=np.float64)
    reentry_w_per_bin = np.zeros(N_ANCHOR_BINS, dtype=np.float64)

    both_pos = (earn_next > 0) & (earn_prev > 0)
    reenter = (earn_next == 0) & (earn_prev > 0)

    for k in np.nonzero(both_pos)[0]:
        cm_prev = marginals[(int(bin_prev_cell[k]), int(period_prev[k]))]
        cm_next = marginals[(int(bin_next_cell[k]), int(period_next[k]))]
        u_prev = cm_prev.rank(float(earn_prev[k]))
        u_next = cm_next.rank(float(earn_next[k]))
        A = int(a_of_pair[k])
        jb = int(rank_bin(np.array([u_next]))[0])
        ib = int(rank_bin(np.array([u_prev]))[0])
        N[A, jb, ib] += float(w_prev[k])
        kernel_pairs_per_bin[A] += 1
        kernel_w_per_bin[A] += float(w_prev[k])

    for k in np.nonzero(reenter)[0]:
        cm_prev = marginals[(int(bin_prev_cell[k]), int(period_prev[k]))]
        u_prev = cm_prev.rank(float(earn_prev[k]))
        A = int(a_of_pair[k])
        ib = int(rank_bin(np.array([u_prev]))[0])
        Rn[A, ib] += float(w_prev[k])
        reentry_pairs_per_bin[A] += 1
        reentry_w_per_bin[A] += float(w_prev[k])

    # Add-one (Laplace) smoothing per row, then row-normalize.
    N_smooth = N + LAPLACE
    P = N_smooth / N_smooth.sum(axis=2, keepdims=True)

    Rn_smooth = Rn + LAPLACE
    R = Rn_smooth / Rn_smooth.sum(axis=1, keepdims=True)

    return {
        "P": P,
        "R": R,
        "N_raw": N,
        "R_raw": Rn,
        "kernel_pairs_per_bin": kernel_pairs_per_bin,
        "reentry_pairs_per_bin": reentry_pairs_per_bin,
        "kernel_w_per_bin": kernel_w_per_bin,
        "reentry_w_per_bin": reentry_w_per_bin,
        "n_pairs": int(len(pairs)),
        "n_kernel_pairs": int(both_pos.sum()),
        "n_reentry_pairs": int(reenter.sum()),
    }


def _draw_bin(pmf_rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Draw a rank bin per row from its pmf, the gate's exact convention.

    Mirrors ``FittedRegimeGatedQRF._gate_draw``: the cumulative
    distribution over the bins, ``chosen = (cumulative >= u).argmax`` --
    the first bin whose inclusive cumulative mass reaches ``u``. ``u``
    comes from the caller's fixed substream. ``pmf_rows`` is ``(n, 20)``
    (one proper distribution per row); returns one bin index ``0..19``
    per row.
    """
    cumulative = np.cumsum(pmf_rows, axis=1)
    chosen = (cumulative >= u[:, None]).argmax(axis=1)
    return chosen.astype(np.int64)


# --------------------------------------------------------------------------
# Stage 5 - generation (holdout): backward kernel chain + regime gate
# --------------------------------------------------------------------------
def generate_candidate(
    holdout: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted: Any,
    kernel: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank-kernel generative candidate panel over the holdout persons.

    For each holdout person: set the anchor rank ``u_A`` and bin ``A``
    from the frozen rule, keep the anchor at its REAL earnings, and chain
    BACKWARD one kernel step per observed transition. At each step, for
    each present person, draw participation from the candidate-2 regime
    gate on (next generated level, current age); where positive:

    * if the next period's generated earnings is positive, draw the
      previous rank bin from ``P[A, bin(u_next), .]`` (kernel-bin
      substream) and place ``u_prev`` uniform within that bin (within-bin
      substream);
    * if the next period's generated earnings is zero, draw the previous
      rank bin from the anchor bin's re-entry distribution ``R[A, .]``
      (re-entry substream) and place ``u_prev`` uniform within that bin;

    then earnings = ``Qhat_pos`` of the period's cell at ``u_prev``.

    The pass is batched by step-from-anchor and ordered by ``person_id``
    within a step (the candidate-2 chain structure), so the gate,
    kernel-bin, within-bin, and re-entry substreams each consume their
    draws in a fixed order. Returns ``(candidate, diagnostics)`` where the
    candidate holds exactly the holdout persons on exactly their observed
    periods (only earnings generated; anchor kept), and diagnostics carry
    the reported-not-gated distributions.
    """
    P = kernel["P"]
    R = kernel["R"]

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

    # Anchor rank + bin per holdout person (frozen stage-2 rule).
    holdout_ids = np.sort(hp["person_id"].unique())
    ab_map = anchor_u_and_bin(marginals, all_anchor, holdout_ids)
    anchor_rank_vals = np.array(
        [ab_map[int(pid)][0] for pid in holdout_ids], dtype=np.float64
    )
    anchor_bin_of_person = {
        int(pid): ab_map[int(pid)][1] for pid in holdout_ids
    }

    rng_gate = _substream(seed, "gate")
    rng_kernel = _substream(seed, "kernel-bin")
    rng_within = _substream(seed, "within-bin")
    rng_reentry = _substream(seed, "re-entry")

    # Diagnostics accumulators.
    n_clamped = 0
    n_positive_gen = 0
    n_kernel_draws = 0
    n_reentry_draws = 0

    # Backward chain: step j generates the rank-j (earlier) period from
    # the rank-(j-1) (next/later) period already fixed, then participation
    # and magnitude for the rank-j generated period.
    for j in range(1, max_depth):
        positions = np.nonzero(ranks == j)[0]
        if positions.size == 0:
            continue
        # Canonical person_id order within the step.
        order = np.argsort(pids[positions], kind="stable")
        positions = positions[order]
        step_pids = pids[positions]

        # Next period (rank j-1) positions, already generated/anchored.
        next_positions = np.array(
            [pos_by_key[(int(pid), j - 1)] for pid in step_pids]
        )
        next_level = gen_earn[next_positions]

        # Participation gate on (next generated level, current age),
        # drawn exactly as candidate 2 with u from the gate substream.
        u_gate = rng_gate.random(len(positions))
        signs = _gate_sign_draw(fitted, next_level, ages[positions], u_gate)
        is_pos = signs == 1

        vals = np.zeros(len(positions), dtype=np.float64)
        pos_local = np.nonzero(is_pos)[0]
        if pos_local.size:
            # Anchor bin A per positive person at this step.
            A_pos = np.array(
                [anchor_bin_of_person[int(step_pids[li])] for li in pos_local],
                dtype=np.int64,
            )
            next_level_pos = next_level[pos_local]
            next_is_pos = next_level_pos > 0

            # Chosen previous rank bin per positive person (0..19),
            # filled from the two mutually exclusive branches.
            chosen_bin = np.empty(pos_local.size, dtype=np.int64)

            # Branch 1: next positive -> kernel P[A, bin(u_next), .].
            kp = np.nonzero(next_is_pos)[0]
            if kp.size:
                # u_next = rhat at the NEXT period's own cell.
                u_next = np.empty(kp.size, dtype=np.float64)
                for m, li in enumerate(pos_local[kp]):
                    gpos = next_positions[li]
                    cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                    u_next[m] = cell.rank(float(gen_earn[gpos]))
                jbin = rank_bin(u_next)
                pmf = P[A_pos[kp], jbin, :]
                u_k = rng_kernel.random(kp.size)
                chosen_bin[kp] = _draw_bin(pmf, u_k)
                n_kernel_draws += int(kp.size)

            # Branch 2: next zero -> re-entry R[A, .].
            rp = np.nonzero(~next_is_pos)[0]
            if rp.size:
                pmf_r = R[A_pos[rp], :]
                u_r = rng_reentry.random(rp.size)
                chosen_bin[rp] = _draw_bin(pmf_r, u_r)
                n_reentry_draws += int(rp.size)

            # u_prev uniform within the chosen bin, one within-bin draw
            # per generated-positive period (person_id order).
            u_within = rng_within.random(pos_local.size)
            u_prev = (chosen_bin.astype(np.float64) + u_within) / N_RANK_BINS

            # Earnings = Qhat_pos of the CURRENT (rank-j) period's cell.
            for m, li in enumerate(pos_local):
                gpos = positions[li]
                cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                q = float(cell.quantile(np.array([u_prev[m]]))[0])
                vals[li] = q
                # Clamped-rank bookkeeping: the quantile clamps u outside
                # the cell's [wtil[0], wtil[-1]] plotting-position span.
                if cell.wtil.size and (
                    u_prev[m] < cell.wtil[0] or u_prev[m] > cell.wtil[-1]
                ):
                    n_clamped += 1
            n_positive_gen += int(pos_local.size)
        gen_earn[positions] = vals

    out = hp[["person_id", "period", "earnings", "age", "weight"]].copy()
    out["earnings"] = gen_earn

    diagnostics = _generation_diagnostics(
        kernel,
        anchor_rank_vals,
        marginals,
        n_positive_gen,
        n_clamped,
        n_kernel_draws,
        n_reentry_draws,
        int(len(holdout_ids)),
    )
    return out, diagnostics


def _generation_diagnostics(
    kernel: dict[str, Any],
    anchor_rank_vals: np.ndarray,
    marginals: dict[tuple[int, int], CellMarginal],
    n_positive_gen: int,
    n_clamped: int,
    n_kernel_draws: int,
    n_reentry_draws: int,
    n_holdout_persons: int,
) -> dict[str, Any]:
    """Assemble the reported-not-gated diagnostics for one seed.

    Carries: the kernel diagonal mass by anchor bin (mean over rows of
    ``P[A, j, j]``), the kernel corner masses (top-bin -> top-bin
    ``P[A, 19, 19]`` and bottom -> bottom ``P[A, 0, 0]``), the pair
    counts per anchor bin (kernel and re-entry, weighted and unweighted),
    the re-entry distributions ``R[A, .]``, the anchor-rank decile
    histogram, the per-cell train positive-count summary, and the
    clamped-rank share.
    """
    P = kernel["P"]
    R = kernel["R"]

    # Kernel diagonal mass by anchor bin: mean over rows j of P[A, j, j].
    diag_mass = {
        f"A{A}": float(np.mean(np.diagonal(P[A])))
        for A in range(N_ANCHOR_BINS)
    }
    # Corner masses by anchor bin.
    corner_top = {
        f"A{A}": float(P[A, N_RANK_BINS - 1, N_RANK_BINS - 1])
        for A in range(N_ANCHOR_BINS)
    }
    corner_bottom = {f"A{A}": float(P[A, 0, 0]) for A in range(N_ANCHOR_BINS)}

    pair_counts = {
        f"A{A}": {
            "kernel_pairs": int(kernel["kernel_pairs_per_bin"][A]),
            "reentry_pairs": int(kernel["reentry_pairs_per_bin"][A]),
            "kernel_weight": float(kernel["kernel_w_per_bin"][A]),
            "reentry_weight": float(kernel["reentry_w_per_bin"][A]),
        }
        for A in range(N_ANCHOR_BINS)
    }
    reentry_dist = {
        f"A{A}": [float(v) for v in R[A]] for A in range(N_ANCHOR_BINS)
    }

    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(anchor_rank_vals, bins=edges)
    anchor_rank_dist = {
        f"[{edges[i]:.1f},{edges[i + 1]:.1f})": int(hist[i])
        for i in range(len(hist))
    }

    cell_counts = np.array(
        [cm.n_pos for cm in marginals.values()], dtype=np.int64
    )
    cell_count_summary = {
        "n_cells": int(len(cell_counts)),
        "min": int(cell_counts.min()),
        "p25": int(np.percentile(cell_counts, 25)),
        "median": int(np.median(cell_counts)),
        "p75": int(np.percentile(cell_counts, 75)),
        "max": int(cell_counts.max()),
    }

    return {
        "n_holdout_persons": int(n_holdout_persons),
        "kernel_diagonal_mass_by_anchor_bin": diag_mass,
        "kernel_corner_mass": {
            "top_to_top": corner_top,
            "bottom_to_bottom": corner_bottom,
        },
        "pair_counts_per_anchor_bin": pair_counts,
        "reentry_distributions": reentry_dist,
        "n_kernel_pairs": int(kernel["n_kernel_pairs"]),
        "n_reentry_pairs": int(kernel["n_reentry_pairs"]),
        "n_train_pairs": int(kernel["n_pairs"]),
        "anchor_rank_distribution": anchor_rank_dist,
        "cell_count_distribution": cell_count_summary,
        "generation_draw_counts": {
            "n_kernel_draws": int(n_kernel_draws),
            "n_reentry_draws": int(n_reentry_draws),
        },
        "clamped_rank_share": {
            "n_positive_generated": int(n_positive_gen),
            "n_clamped": int(n_clamped),
            "share": (
                float(n_clamped / n_positive_gen) if n_positive_gen else 0.0
            ),
            "note": (
                "share of generated POSITIVE periods whose rank u_prev "
                "fell outside the cell's plotting-position span "
                "[wtil[0], wtil[-1]] and was clamped to the cell's "
                "[y_min, y_max]"
            ),
        },
    }


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
    verbose: bool,
) -> dict[str, Any]:
    """Fit, count, generate, and score candidate 6 for one seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Stage 1: per-cell marginals on the train complement.
    marginals = fit_cell_marginals(train)

    # Stages 3-4: kernel and re-entry counts on the train pairs.
    kernel = estimate_kernel(train, all_anchor, marginals)

    # Stage 5: participation gate (train complement) + backward chain.
    fitted, pairs = fit_participation_gate(train, seed)
    candidate, diagnostics = generate_candidate(
        holdout, all_anchor, marginals, fitted, kernel, seed
    )

    # --- geometry: score candidate vs holdout on both locked views ---
    geometry_by_view: dict[str, Any] = {}
    geometry_seed_pass = True
    n_windows: dict[str, int] = {}
    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(candidate, holdout, view, seed=seed)
        checks = check_geometry(scores, views_cfg[vname]["geometry"])
        view_pass = all(c["pass"] for c in checks.values())
        geometry_seed_pass = geometry_seed_pass and view_pass
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

    result = {
        "seed": seed,
        "n_persons": int(holdout.person_id.nunique()),
        "n_person_periods": int(len(holdout)),
        "n_train_persons": int(train.person_id.nunique()),
        "n_train_pairs": int(len(pairs)),
        "n_windows": n_windows,
        "regimes": {"participation_gate": fitted.regimes()},
        "kernel": {
            "n_kernel_pairs": int(kernel["n_kernel_pairs"]),
            "n_reentry_pairs": int(kernel["n_reentry_pairs"]),
        },
        "generation_diagnostics": diagnostics,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if verbose:
        d = diagnostics
        diag_mean = float(
            np.mean(list(d["kernel_diagonal_mass_by_anchor_bin"].values()))
        )
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"mob_diag={battery_values['mobility_diagonal']:.3f} "
            f"kernel_diag_mean={diag_mean:.3f} "
            f"clamp={d['clamped_rank_share']['share']:.3f} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-6 run."""
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

    # Identical battery-reference bit-exact precheck as every prior run:
    # the battery code path must reproduce every committed reference value
    # to float precision before any candidate is scored.
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

    # Anchors on the FULL filtered panel (a person's last observed period
    # is a property of the panel, computed once and sliced per split).
    all_anchor = anchor_rows(panel)

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
                verbose,
            )
        )

    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    gate_pass = geometry_gate_pass and battery_gate_pass

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_rank_kernel_v1",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "description": (
            "Gate-1 candidate 6: anchored empirical rank-transition "
            "kernel. Empirical per-cell quantile marginals (five-year age "
            "bin x period, byte-for-byte candidate 5b's) supply the "
            "magnitude; a nonparametric transition law estimated by "
            "COUNTING supplies the dynamics. For every train "
            "backward-adjacent pair among positives, weighted counts "
            "accumulate into N[A, bin(u_next), bin(u_prev)] over five "
            "anchor bins and twenty rank bins, add-one smoothed and "
            "row-normalized to a transition kernel P[A, j, .]; a per-"
            "anchor-bin re-entry distribution counts u_prev where the "
            "later period is zero. Generation keeps each holdout person's "
            "real anchor, conditions on their anchor bin, and chains "
            "backward one kernel step per observed transition -- drawing "
            "the previous rank bin from P (next positive) or the re-entry "
            "distribution (next zero) and placing u_prev uniform within "
            "bin -- reading earnings from the period's cell quantile at "
            "u_prev. Participation reuses the candidate-2 backward regime "
            "gate. No calibration stage: the kernel has zero free "
            "parameters beyond the pinned discretization. Registered "
            "frozen before the run in issue #42 (see spec_registration). "
            "Candidate scored against the held-out PSID family earnings "
            "panel geometry (two locked views) and the locked moment "
            "battery, per the locked seed-level conjunction in gates.yaml "
            "(pull request 39). Protocol machinery imported byte-for-byte "
            "from the baseline runner (pull request 40); rank machinery "
            "and participation gate from the candidate-5b runner (pull "
            "request 52)."
        ),
        "model": {
            "class": "anchored empirical rank-transition kernel (quantile marginal + counted kernel)",
            "stochastic": True,
            "populace_fit_used": True,
            "populace_fit_scope": (
                "participation gate only (RegimeGatedQRF sign gate); the "
                "kernel counting and generation use pure numpy"
            ),
            "calibration": "none (zero free parameters beyond discretization)",
            "marginal": {
                "cells": (
                    "five-year age bins {25-29,...,55-59} x calendar period"
                ),
                "p0": "weighted zero share per cell (train only)",
                "quantile": (
                    "weighted empirical quantile Qhat_pos: sort positive y_k "
                    "with weights w_k, plotting position "
                    "wtil_k = (cum_k - 0.5 w_k)/W, linear interpolation of "
                    "(wtil_k, y_k) clamped to [y_min, y_max]"
                ),
                "rank": (
                    "inverse interpolation rhat: (y_k -> wtil_k) clamped to "
                    "[0.001, 0.999]"
                ),
                "source": "byte-for-byte candidate 5b (pull request 52)",
            },
            "anchor_bins": {
                "n_anchor_bins": N_ANCHOR_BINS,
                "rule": (
                    "u_A = rhat(anchor value) at its cell for a positive "
                    "anchor; u_A = p0/2 of the cell for a zero anchor; "
                    "anchor bin A = clip(floor(u_A * 5), 0, 4)"
                ),
            },
            "kernel": {
                "n_rank_bins": N_RANK_BINS,
                "array": "N[A, bin(u_next), bin(u_prev)] (5 x 20 x 20)",
                "pairs": (
                    "train backward-adjacent pairs among positives (both "
                    "periods positive, consecutive observed periods 2 years "
                    "apart)"
                ),
                "u_prev": "rhat at the earlier period's own cell",
                "u_next": "rhat at the later period's own cell",
                "weight": "the pair's earlier-period person weight (weight_tm2)",
                "smoothing": "add-one (Laplace) per row",
                "normalization": (
                    "row-normalized over bin(u_prev) to P[A, j, .]"
                ),
            },
            "reentry": {
                "array": "Reentry[A, bin(u_prev)] (5 x 20)",
                "pairs": (
                    "train pairs where the LATER period is zero and the "
                    "EARLIER is positive"
                ),
                "u_prev": "rhat at the earlier period's own cell",
                "weight": "the pair's earlier-period person weight (weight_tm2)",
                "smoothing": "add-one (Laplace) then normalized",
            },
            "generation": {
                "anchor": (
                    "chronologically last observed period held at real "
                    "earnings"
                ),
                "anchor_rank": (
                    "positive anchor u_A = rhat(anchor value) at its cell; "
                    "zero anchor u_A = p0/2 of the cell; anchor bin A"
                ),
                "chain": (
                    "backward one kernel step per observed transition "
                    "(standing gap rule)"
                ),
                "positive_next_positive": (
                    "draw previous rank bin from P[A, bin(u_next), .], then "
                    "u_prev uniform within bin"
                ),
                "positive_next_zero": (
                    "draw previous rank bin from the anchor bin's re-entry "
                    "distribution R[A, .], then u_prev uniform within bin"
                ),
                "period_draw": (
                    "earnings = Qhat_pos of the period's cell at u_prev"
                ),
                "participation": (
                    "candidate-2 backward regime gate (RegimeGatedQRF sign "
                    "gate) on (next generated level, current age), trained "
                    "on the 80% complement, populace-fit defaults"
                ),
                "rng": (
                    "distinct fixed-label substreams of the gate seed: "
                    "gate, kernel-bin, within-bin, re-entry"
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
                "battery_reference (imported from the baseline runner)"
            ),
            "pass_rule": (
                "seed passes geometry iff every locked threshold on every "
                "locked view holds; seed passes battery iff every locked "
                "tolerance holds; gate passes iff >=4/5 seeds pass geometry "
                "AND >=4/5 seeds pass battery"
            ),
        },
        "battery_reference_reproduction": repro,
        "battery_reference_run": BATTERY_REFERENCE_RUN,
        "per_seed": per_seed,
        "seed_conjunction": [
            {
                "seed": s["seed"],
                "geometry_pass": s["geometry_pass"],
                "battery_pass": s["battery_pass"],
            }
            for s in per_seed
        ],
        "kernel_context": {
            "note": (
                "Reported-not-gated per seed: the kernel diagonal mass by "
                "anchor bin, the kernel corner masses (top->top and "
                "bottom->bottom), the pair counts per anchor bin, the "
                "re-entry distributions, the anchor-rank distribution, and "
                "the clamped-rank share. None enters the geometry or "
                "battery pass/fail; the gate rule names only those two "
                "families."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "n_kernel_pairs": s["kernel"]["n_kernel_pairs"],
                    "n_reentry_pairs": s["kernel"]["n_reentry_pairs"],
                    "kernel_diagonal_mass_by_anchor_bin": s[
                        "generation_diagnostics"
                    ]["kernel_diagonal_mass_by_anchor_bin"],
                    "kernel_corner_mass": s["generation_diagnostics"][
                        "kernel_corner_mass"
                    ],
                    "pair_counts_per_anchor_bin": s["generation_diagnostics"][
                        "pair_counts_per_anchor_bin"
                    ],
                    "reentry_distributions": s["generation_diagnostics"][
                        "reentry_distributions"
                    ],
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
        "verdict": {
            "n_seeds": len(SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "gate_1_pass": bool(gate_pass),
            "rule": ">=4/5 seeds geometry AND >=4/5 seeds battery",
        },
        "revision_pins": _revision_pins(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5)"
        )
    return artifact


def _revision_pins() -> dict[str, Any]:
    """Repo/populace SHAs and schema version for provenance."""
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
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "populace_repo_sha": _sha(populace_root),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml_locked": True,
    }


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
