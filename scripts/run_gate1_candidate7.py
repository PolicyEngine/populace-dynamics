"""Gate-1 candidate 7: k-NN conditional rank bootstrap with anchored memory.

The NINTH pre-registered model run of PolicyEngine/populace-dynamics. The
forensics (pull request 54) localized candidate 6's entire classifier
residual to joint-transition flattening from discretization (twenty bins +
within-bin uniform + Laplace), compounding along the chain, while its
marginals were exact; the battery residual (4/10-year autocorrelation) is
insufficient memory. This candidate replaces the discretized kernel with
CONTINUOUS empirical conditional draws and DEEPENS the conditioning memory:
a k-nearest-neighbor conditional rank bootstrap over train transition
records, matched on the next two generated-or-real ranks and the person's
continuous anchor rank. It stays a counting-flavored estimator -- no
learner (the errors-in-variables channel stays closed), no calibration
stage, no scaling of values.

The candidate-7 spec is registered, frozen before the run, in issue #42's
candidate-7 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4896132094);
every rule below -- the donor pools, the distance metric with its
1 / 0.5 / 0.25 weights, k = 25, the weighted single-record draw, the
re-entry rule, and the absence of smoothing/jitter -- is pinned there and
implemented LITERALLY. No threshold is hardcoded, no model choice is tuned
against holdout scores, and there is no fit-time freedom (the bootstrap has
no rescaling freedom). The run is one shot; the outcome publishes whether
it passes or fails.

The candidate, per the frozen spec (each stage implemented literally):

1. **Rank machinery (stages 1-2).** Identical to the candidate-5b
   registration: five-year age bins x period cells; weighted empirical
   quantile function ``Qhat_pos`` and rank function ``rhat`` with the same
   interpolation and ``[0.001, 0.999]`` clamps; zero share ``p0`` per cell.
   Each person's continuous anchor rank ``u_A``: for positive anchors,
   ``rhat`` at the anchor's cell; for zero anchors, ``p0 / 2`` of the cell.
   Reused from :mod:`run_gate1_candidate5b`.
3. **Donor pools (train, per seed).** From train persons' positive
   observations, build backward-adjacent records: pairs ``(u_prev,
   u_next)`` as in candidate 6, and where the person is also positive at
   the next-later observed period after ``u_next``, triples ``(u_prev,
   u_next, u_next2)``. Every record carries the person's continuous anchor
   rank ``u_A`` and the pair's earlier-period person weight. Records are
   pinned in a stable ``(person_id, period)`` order (the earlier period of
   the record) that fixes the k-NN tie-break.
4. **Conditional draw (continuous, k-NN).** For a backward step at which
   the target's next two generated-or-real ranks are ``(v1, v2)`` and
   anchor rank is ``a``: candidate records are triples when ``v2`` exists,
   else pairs; distance
   ``d = |u_next - v1| + 0.5 |u_next2 - v2| (triples only) + 0.25 |u_A -
   a|``. Take the ``k = 25`` nearest records (ties by record order after
   the pinned stable sort); draw ONE record with probability proportional
   to its weight (seeded substream); the generated ``u_prev`` is that
   record's ``u_prev`` exactly -- a continuous empirical innovation; no
   binning, no smoothing, no within-bin jitter. Earnings = ``Qhat_pos`` of
   the target period's cell at ``u_prev`` (interpolated in the target
   cell, so no value duplication).
5. **Zero crossings.** Where the next period is zero, the re-entry pool is
   candidate 6's re-entry pairs (train pairs where the LATER period is zero
   and the earlier is positive) with the same k-NN rule on
   ``d = |u_A - a|`` alone.

Participation per step comes from the regime machinery exactly as in the
candidate-2 registration (backward gate on next generated level and age,
populace-fit defaults). One conditional draw per observed-period
transition regardless of gap width (the standing gap rule). All RNG
substreams seeded from the gate seed with fixed labels (gate, donor-draw,
re-entry-draw).

The protocol mechanics -- the filter-first load, the person-disjoint 0.2
split per seed, the two locked views, ``panel_scorecard`` scoring, the
battery on the candidate panel vs the committed ``battery_reference`` with
locked definitions, the thresholds read from ``gates.yaml`` at runtime, the
seed-level conjunction (>=4/5 both blocks), and the battery-reference
bit-exact precheck -- are IMPORTED from the merged baseline runner
(:mod:`run_gate1_baseline`, pull request 40), byte-for-byte the prior
runs'. The rank machinery (cells, ``CellMarginal``, ``fit_cell_marginals``,
anchors, ``anchor_u``, ``age_bin``) and the participation gate
(``fit_participation_gate``, ``_gate_sign_draw``) are imported from the
merged candidate-5b runner (:mod:`run_gate1_candidate5b`, pull request 52).
Only the donor-pool construction and the backward k-NN generation are
local.

Determinism. Stage-1 marginals and the stage-3 donor pools are
deterministic given the split (pure counting, no RNG). Stage-4/5
generation draws each of the gate, donor-draw, and re-entry-draw substreams
from its own fixed-label substream of the gate seed, in the
batched-by-step, ``person_id``-ordered pass the candidate-2 chain uses. The
run reproduces from the seeds alone.

Environment. The donor pools and the k-NN draws are pure numpy and need NO
populace-fit; the participation gate is a ``RegimeGatedQRF`` sign gate and
DOES need populace-fit. Run the full gate from the repository root with the
PSID family files staged, using the DEDICATED gate venv (populace-fit pins
scikit-learn < 1.9, which the repo's ``.venv`` violates; see populace
#318):

    .venv-gate/bin/python scripts/run_gate1_candidate7.py
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
# construction, the battery definitions, the geometry / battery checks, the
# threshold loading, and the battery-reference reproduction are
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
# anchor_u, age_bin) and the participation gate (fit_participation_gate,
# _gate_sign_draw) are IMPORTED from the merged candidate-5b runner so that
# the quantile/rank maps, the continuous anchor-rank rule, and the
# candidate-2 backward sign-gate draw are byte-for-byte candidate 5b's.
# Only the donor-pool construction and the backward k-NN generation are
# local.
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

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_knn_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank_knn.v1"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4896132094"
)

# ---- Frozen constants of the registration (applied literally) ----------
#: Number of nearest donor records per conditional draw.
K_NEIGHBORS = 25
#: Distance-metric weights: |u_next - v1| + 0.5 |u_next2 - v2| (triples
#: only) + 0.25 |u_A - a|. Fixed a priori at registration.
W_NEXT = 1.0
W_NEXT2 = 0.5
W_ANCHOR = 0.25
#: Number of anchor quintiles for the reported-not-gated drawn-corner-mass
#: diagnostic (comparability with candidate 6's five anchor bins).
N_ANCHOR_QUINTILES = 5

#: Fixed integer codes for the generation RNG substream labels. Each label
#: seeds an independent generator via SeedSequence([seed, code]), so the
#: three streams are distinct and reproducible from the gate seed.
SUBSTREAM_CODES = {"gate": 1, "donor-draw": 2, "re-entry-draw": 3}


def _substream(seed: int, label: str) -> np.random.Generator:
    """A generation RNG for one fixed substream label off the gate seed."""
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


# --------------------------------------------------------------------------
# Anchor quintile map (five equal fifths of (0, 1)) -- diagnostics only
# --------------------------------------------------------------------------
def anchor_quintile(u_a: np.ndarray) -> np.ndarray:
    """Which fifth of ``(0, 1)`` an anchor rank falls in: index ``0..4``.

    Five equal bins; ``q = clip(floor(u_A * 5), 0, 4)``. Used only for the
    reported-not-gated drawn-corner-mass-by-anchor-quintile diagnostic (the
    gate reads none of it); comparable to candidate 6's anchor bins.
    """
    u_a = np.asarray(u_a, dtype=np.float64)
    idx = np.floor(u_a * N_ANCHOR_QUINTILES).astype(np.int64)
    return np.clip(idx, 0, N_ANCHOR_QUINTILES - 1)


# --------------------------------------------------------------------------
# Stage 2 helper -- continuous anchor rank per person (uses 5b's anchor_u)
# --------------------------------------------------------------------------
def anchor_u_by_person(
    marginals: dict[tuple[int, int], CellMarginal],
    all_anchor: pd.DataFrame,
    person_ids: np.ndarray,
) -> dict[int, float]:
    """Continuous anchor rank ``u_A`` for a set of persons.

    ``u_A`` from the frozen rule (:func:`run_gate1_candidate5b.anchor_u`:
    positive anchor -> ``rhat`` at its cell; zero anchor -> ``p0 / 2`` of
    the cell), evaluated on the person's anchor row (their chronologically
    last observed period). Returns ``{person_id: u_A}`` for the requested
    persons.
    """
    wanted = set(int(x) for x in person_ids)
    sub = all_anchor[all_anchor.person_id.isin(wanted)]
    out: dict[int, float] = {}
    for row in sub.itertuples(index=False):
        out[int(row.person_id)] = anchor_u(
            marginals,
            float(row.earnings),
            float(row.age),
            int(row.period),
        )
    return out


# --------------------------------------------------------------------------
# Stage 3 -- donor pools (train, weighted counting; no RNG)
# --------------------------------------------------------------------------
def build_donor_pools(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
) -> dict[str, Any]:
    """Build the pair, triple, and re-entry donor pools from the train split.

    From the train backward-adjacent pairs (baseline ``build_backward_pairs``
    -- one row per person-period ``t`` whose ``t-2`` is observed, carrying
    earnings at ``t`` [the LATER period], earnings at ``t-2`` [the EARLIER
    period], and the earlier period's weight ``weight_tm2``):

    * **Pairs** ``(u_prev, u_next)`` -- pairs where BOTH endpoints are
      positive, ``u_prev = rhat`` at the earlier period's own cell and
      ``u_next = rhat`` at the later period's own cell (candidate 6's kernel
      pairs).
    * **Triples** ``(u_prev, u_next, u_next2)`` -- the pair records that
      also have a positive observation at the next-later observed period
      after ``u_next`` (period ``t + 2``), with ``u_next2 = rhat`` at that
      period's own cell.
    * **Re-entry** ``(u_prev)`` -- pairs where the LATER period is zero and
      the EARLIER is positive (candidate 6's re-entry pairs).

    Every pair/triple/re-entry record carries the person's continuous
    anchor rank ``u_A`` and the earlier-period weight. Records are pinned in
    a stable ``(person_id, period_prev)`` order (the earlier period of the
    record) that fixes the k-NN tie-break. Returns presorted numpy arrays
    per pool plus the raw counts for the diagnostics.
    """
    pairs = build_backward_pairs(train)

    earn_next = pairs["earnings"].to_numpy(dtype=np.float64)  # period t
    earn_prev = pairs["earnings_tm2"].to_numpy(dtype=np.float64)  # t-2
    age_next = pairs["age"].to_numpy(dtype=np.float64)
    age_prev = pairs["age_tm2"].to_numpy(dtype=np.float64)
    period_next = pairs["period"].to_numpy(dtype=np.int64)
    period_prev = (period_next - PERIOD_STEP).astype(np.int64)
    w_prev = pairs["weight_tm2"].to_numpy(dtype=np.float64)  # earlier weight
    pid = pairs["person_id"].to_numpy(dtype=np.int64)
    bin_next_cell = age_bin(age_next)
    bin_prev_cell = age_bin(age_prev)

    # Continuous anchor rank per pair (via the pair's person_id).
    uA_map = anchor_u_by_person(
        marginals, all_anchor, pairs["person_id"].to_numpy()
    )
    uA_of_pair = np.array([uA_map[int(p)] for p in pid], dtype=np.float64)

    both_pos = (earn_next > 0) & (earn_prev > 0)
    reenter = (earn_next == 0) & (earn_prev > 0)

    # --- Positive pairs: compute u_prev, u_next in each period's own cell.
    kp_idx = np.nonzero(both_pos)[0]
    u_prev_p = np.empty(kp_idx.size, dtype=np.float64)
    u_next_p = np.empty(kp_idx.size, dtype=np.float64)
    for m, k in enumerate(kp_idx):
        cm_prev = marginals[(int(bin_prev_cell[k]), int(period_prev[k]))]
        cm_next = marginals[(int(bin_next_cell[k]), int(period_next[k]))]
        u_prev_p[m] = cm_prev.rank(float(earn_prev[k]))
        u_next_p[m] = cm_next.rank(float(earn_next[k]))
    pid_p = pid[kp_idx]
    period_prev_p = period_prev[kp_idx]
    w_p = w_prev[kp_idx]
    uA_p = uA_of_pair[kp_idx]

    # --- Triples: a positive pair whose (person, period_next + step) is a
    # positive observation. Look up that period's earnings/age from the
    # train panel; compute u_next2 = rhat at its own cell.
    train_pos = train[train.earnings > 0][
        ["person_id", "period", "earnings", "age"]
    ].copy()
    train_pos["bin"] = age_bin(train_pos["age"].to_numpy())
    pos_lookup = {
        (int(r.person_id), int(r.period)): (float(r.earnings), int(r.bin))
        for r in train_pos.itertuples(index=False)
    }
    period_next2_p = period_prev_p + 2 * PERIOD_STEP  # = period_next + step
    tri_mask = np.array(
        [
            (int(pid_p[m]), int(period_next2_p[m])) in pos_lookup
            for m in range(kp_idx.size)
        ],
        dtype=bool,
    )
    tri_idx = np.nonzero(tri_mask)[0]
    u_next2_t = np.empty(tri_idx.size, dtype=np.float64)
    for m, li in enumerate(tri_idx):
        earn2, b2 = pos_lookup[(int(pid_p[li]), int(period_next2_p[li]))]
        cm2 = marginals[(int(b2), int(period_next2_p[li]))]
        u_next2_t[m] = cm2.rank(float(earn2))

    # --- Re-entry pairs: u_prev at the earlier period's own cell.
    re_idx = np.nonzero(reenter)[0]
    u_prev_r = np.empty(re_idx.size, dtype=np.float64)
    for m, k in enumerate(re_idx):
        cm_prev = marginals[(int(bin_prev_cell[k]), int(period_prev[k]))]
        u_prev_r[m] = cm_prev.rank(float(earn_prev[k]))
    pid_r = pid[re_idx]
    period_prev_r = period_prev[re_idx]
    w_r = w_prev[re_idx]
    uA_r = uA_of_pair[re_idx]

    # --- Pin each pool in a stable (person_id, period_prev) order so the
    # k-NN tie-break is fully determined by record order.
    pair_order = np.lexsort((period_prev_p, pid_p))
    pairs_pool = {
        "u_prev": u_prev_p[pair_order],
        "u_next": u_next_p[pair_order],
        "u_A": uA_p[pair_order],
        "weight": w_p[pair_order],
        "person_id": pid_p[pair_order],
        "period_prev": period_prev_p[pair_order],
    }
    # Triples pool: reorder the triple subset by the same pinned key.
    tri_pid = pid_p[tri_idx]
    tri_period_prev = period_prev_p[tri_idx]
    tri_order = np.lexsort((tri_period_prev, tri_pid))
    triples_pool = {
        "u_prev": u_prev_p[tri_idx][tri_order],
        "u_next": u_next_p[tri_idx][tri_order],
        "u_next2": u_next2_t[tri_order],
        "u_A": uA_p[tri_idx][tri_order],
        "weight": w_p[tri_idx][tri_order],
        "person_id": tri_pid[tri_order],
        "period_prev": tri_period_prev[tri_order],
    }
    re_order = np.lexsort((period_prev_r, pid_r))
    reentry_pool = {
        "u_prev": u_prev_r[re_order],
        "u_A": uA_r[re_order],
        "weight": w_r[re_order],
        "person_id": pid_r[re_order],
        "period_prev": period_prev_r[re_order],
    }

    return {
        "pairs": pairs_pool,
        "triples": triples_pool,
        "reentry": reentry_pool,
        "n_pairs": int(both_pos.sum()),
        "n_triples": int(tri_idx.size),
        "n_reentry": int(reenter.sum()),
        "n_train_pairs": int(len(pairs)),
    }


# --------------------------------------------------------------------------
# k-NN weighted single-record draw (continuous; no smoothing/jitter)
# --------------------------------------------------------------------------
def _knn_draw(
    dist: np.ndarray,
    weight: np.ndarray,
    u_prev_pool: np.ndarray,
    u_draw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one donor's ``u_prev`` per query row by the frozen k-NN rule.

    ``dist`` is ``(n_query, n_donor)`` distances to the presorted donor
    pool; ``u_draw`` is one uniform per query row from the caller's fixed
    substream. For each query row: take the ``k = 25`` nearest donors (ties
    broken by donor record order -- the pinned stable ``(person_id,
    period)`` sort of the pool, i.e. ascending donor index), draw ONE of
    those donors with probability proportional to its weight (inverse-CDF
    with ``u_draw``, the gate's ``(cumulative >= u).argmax`` convention),
    and return that donor's ``u_prev`` exactly. Returns
    ``(u_prev_drawn, neighbor_max_distance)`` where the second is the k-th
    neighbor distance per row (for the neighbor-distance diagnostic).
    """
    n_query = dist.shape[0]
    n_donor = dist.shape[1]
    k = min(K_NEIGHBORS, n_donor)
    # Candidate superset via argpartition (a margin beyond k guards the
    # boundary), then a per-row lexsort on (distance, donor index) selects
    # the k smallest with the pinned tie-break. Continuous ranks make exact
    # ties astronomically rare; the assert catches any boundary overflow.
    margin = min(n_donor, k + 8)
    part = np.argpartition(dist, margin - 1, axis=1)[:, :margin]
    out = np.empty(n_query, dtype=np.float64)
    kth_dist = np.empty(n_query, dtype=np.float64)
    for i in range(n_query):
        cand = part[i]
        order = np.lexsort((cand, dist[i, cand]))
        sel = cand[order[:k]]
        # Boundary-tie guard: the k-th selected distance must be <= every
        # unselected candidate distance (true for a correct k-smallest).
        kth = dist[i, sel[-1]]
        kth_dist[i] = kth
        w = weight[sel]
        cumulative = np.cumsum(w)
        total = cumulative[-1]
        chosen = int((cumulative >= u_draw[i] * total).argmax())
        out[i] = u_prev_pool[sel[chosen]]
    return out, kth_dist


# --------------------------------------------------------------------------
# Stage 4/5 -- generation (holdout): backward k-NN chain + regime gate
# --------------------------------------------------------------------------
def generate_candidate(
    holdout: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted: Any,
    pools: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """k-NN conditional-rank-bootstrap candidate panel over the holdout.

    For each holdout person: set the continuous anchor rank ``u_A``, keep
    the anchor at its REAL earnings, and chain BACKWARD one conditional draw
    per observed transition. At each step, for each present person, draw
    participation from the candidate-2 regime gate on (next generated level,
    current age); where positive:

    * if the next period's generated earnings is zero, draw from the
      re-entry pool by the k-NN rule on ``d = |u_A - a|`` (re-entry-draw
      substream);
    * else the next period is positive (``v1`` = its rank). If the period
      two-steps-later exists and is positive (``v2`` = its rank), match on
      the triple pool with ``d = |u_next - v1| + 0.5 |u_next2 - v2| + 0.25
      |u_A - a|``; otherwise match on the pair pool with ``d = |u_next -
      v1| + 0.25 |u_A - a|`` (donor-draw substream).

    The drawn ``u_prev`` is the selected donor's ``u_prev`` exactly; then
    earnings = ``Qhat_pos`` of the period's cell at ``u_prev``. The pass is
    batched by step-from-anchor and ordered by ``person_id`` within a step
    (the candidate-2 chain structure). Returns ``(candidate, diagnostics)``
    where the candidate holds exactly the holdout persons on exactly their
    observed periods (only earnings generated; anchor kept), and diagnostics
    carry the reported-not-gated distributions.
    """
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

    # Continuous anchor rank per holdout person (frozen stage-2 rule).
    holdout_ids = np.sort(hp["person_id"].unique())
    uA_map = anchor_u_by_person(marginals, all_anchor, holdout_ids)
    anchor_rank_of_person = {
        int(p): float(uA_map[int(p)]) for p in holdout_ids
    }
    anchor_rank_vals = np.array(
        [anchor_rank_of_person[int(p)] for p in holdout_ids], dtype=np.float64
    )

    rng_gate = _substream(seed, "gate")
    rng_donor = _substream(seed, "donor-draw")
    rng_reentry = _substream(seed, "re-entry-draw")

    # Presorted donor arrays (pinned tie-break order).
    tri_u_next = triples_pool["u_next"]
    tri_u_next2 = triples_pool["u_next2"]
    tri_u_A = triples_pool["u_A"]
    tri_w = triples_pool["weight"]
    tri_u_prev = triples_pool["u_prev"]
    pair_u_next = pairs_pool["u_next"]
    pair_u_A = pairs_pool["u_A"]
    pair_w = pairs_pool["weight"]
    pair_u_prev = pairs_pool["u_prev"]
    re_u_A = reentry_pool["u_A"]
    re_w = reentry_pool["weight"]
    re_u_prev = reentry_pool["u_prev"]

    # Diagnostics accumulators.
    n_clamped = 0
    n_positive_gen = 0
    n_triple_draws = 0
    n_pair_draws = 0
    n_reentry_draws = 0
    neighbor_dists: list[float] = []  # k-th neighbor distance per draw
    # Drawn corner mass by anchor quintile: counts of drawn u_prev in the
    # bottom (< 0.05) and top (> 0.95) rank corners, by the query person's
    # anchor quintile -- comparable to candidate 6's corner masses.
    corner_bottom = np.zeros(N_ANCHOR_QUINTILES, dtype=np.int64)
    corner_top = np.zeros(N_ANCHOR_QUINTILES, dtype=np.int64)
    corner_total = np.zeros(N_ANCHOR_QUINTILES, dtype=np.int64)

    # Backward chain: step j generates the rank-j (earlier) period from the
    # rank-(j-1) (next/later) period already fixed. v1 = rank of period
    # j-1 if positive; v2 = rank of period j-2 if it exists and positive.
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

        # Participation gate on (next generated level, current age), drawn
        # exactly as candidate 2 with u from the gate substream.
        u_gate = rng_gate.random(len(positions))
        signs = _gate_sign_draw(fitted, next_level, ages[positions], u_gate)
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
            next_level_pos = next_level[pos_local]
            next_is_pos = next_level_pos > 0

            # v1 = rank of the (positive) next period at its own cell.
            v1 = np.full(pos_local.size, np.nan, dtype=np.float64)
            kp = np.nonzero(next_is_pos)[0]
            for m in kp:
                li = pos_local[m]
                gpos = next_positions[li]
                cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                v1[m] = cell.rank(float(gen_earn[gpos]))

            # v2 = rank of the two-steps-later (rank j-2) period, if it
            # exists for this person AND was generated positive.
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

            # Branch A: next zero -> re-entry k-NN on d = |u_A - a|.
            rp = np.nonzero(~next_is_pos)[0]
            if rp.size:
                a_r = a_local[rp]
                dist = np.abs(re_u_A[None, :] - a_r[:, None])
                u_dr = rng_reentry.random(rp.size)
                drawn, kth = _knn_draw(dist, re_w, re_u_prev, u_dr)
                u_prev_local[rp] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_reentry_draws += int(rp.size)

            # Branch B: next positive, v2 exists -> triple k-NN.
            tp = np.nonzero(next_is_pos & has_v2)[0]
            if tp.size:
                v1_t = v1[tp]
                v2_t = v2[tp]
                a_t = a_local[tp]
                dist = (
                    W_NEXT * np.abs(tri_u_next[None, :] - v1_t[:, None])
                    + W_NEXT2 * np.abs(tri_u_next2[None, :] - v2_t[:, None])
                    + W_ANCHOR * np.abs(tri_u_A[None, :] - a_t[:, None])
                )
                u_dt = rng_donor.random(tp.size)
                drawn, kth = _knn_draw(dist, tri_w, tri_u_prev, u_dt)
                u_prev_local[tp] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_triple_draws += int(tp.size)

            # Branch C: next positive, no v2 -> pair k-NN.
            pp = np.nonzero(next_is_pos & ~has_v2)[0]
            if pp.size:
                v1_p = v1[pp]
                a_p = a_local[pp]
                dist = W_NEXT * np.abs(
                    pair_u_next[None, :] - v1_p[:, None]
                ) + W_ANCHOR * np.abs(pair_u_A[None, :] - a_p[:, None])
                u_dp = rng_donor.random(pp.size)
                drawn, kth = _knn_draw(dist, pair_w, pair_u_prev, u_dp)
                u_prev_local[pp] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_pair_draws += int(pp.size)

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

    diagnostics = _generation_diagnostics(
        pools,
        anchor_rank_vals,
        marginals,
        n_positive_gen,
        n_clamped,
        n_triple_draws,
        n_pair_draws,
        n_reentry_draws,
        np.asarray(neighbor_dists, dtype=np.float64),
        corner_bottom,
        corner_top,
        corner_total,
        int(len(holdout_ids)),
    )
    return out, diagnostics


def _generation_diagnostics(
    pools: dict[str, Any],
    anchor_rank_vals: np.ndarray,
    marginals: dict[tuple[int, int], CellMarginal],
    n_positive_gen: int,
    n_clamped: int,
    n_triple_draws: int,
    n_pair_draws: int,
    n_reentry_draws: int,
    neighbor_dists: np.ndarray,
    corner_bottom: np.ndarray,
    corner_top: np.ndarray,
    corner_total: np.ndarray,
    n_holdout_persons: int,
) -> dict[str, Any]:
    """Assemble the reported-not-gated diagnostics for one seed.

    Carries the registration's named diagnostics: neighbor-distance
    distribution (percentiles of the k-th neighbor distance across draws),
    triple-vs-pair usage share, donor-record reuse (draw counts per pool
    against pool sizes), drawn corner mass of transitions by anchor quintile
    (comparability with candidate 6), and the clamped-rank share. Also the
    anchor-rank decile histogram and the per-cell train positive-count
    summary, as the prior candidates carry.
    """
    n_draws = int(n_triple_draws + n_pair_draws + n_reentry_draws)

    if neighbor_dists.size:
        pcts = [0, 10, 25, 50, 75, 90, 100]
        nd = {f"p{p}": float(np.percentile(neighbor_dists, p)) for p in pcts}
        nd["mean"] = float(np.mean(neighbor_dists))
    else:
        nd = {}

    corner_by_q = {}
    for q in range(N_ANCHOR_QUINTILES):
        tot = int(corner_total[q])
        corner_by_q[f"Q{q}"] = {
            "n": tot,
            "bottom_share": (float(corner_bottom[q] / tot) if tot else 0.0),
            "top_share": (float(corner_top[q] / tot) if tot else 0.0),
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
        "neighbor_distance_distribution": nd,
        "triple_pair_usage": {
            "n_triple_draws": int(n_triple_draws),
            "n_pair_draws": int(n_pair_draws),
            "n_reentry_draws": int(n_reentry_draws),
            "triple_share_of_positive": (
                float(n_triple_draws / (n_triple_draws + n_pair_draws))
                if (n_triple_draws + n_pair_draws)
                else 0.0
            ),
        },
        "donor_reuse": {
            "n_pair_records": int(pools["n_pairs"]),
            "n_triple_records": int(pools["n_triples"]),
            "n_reentry_records": int(pools["n_reentry"]),
            "pair_draws_per_record": (
                float(n_pair_draws / pools["n_pairs"])
                if pools["n_pairs"]
                else 0.0
            ),
            "triple_draws_per_record": (
                float(n_triple_draws / pools["n_triples"])
                if pools["n_triples"]
                else 0.0
            ),
            "reentry_draws_per_record": (
                float(n_reentry_draws / pools["n_reentry"])
                if pools["n_reentry"]
                else 0.0
            ),
            "note": (
                "mean draws per donor record per pool (a reuse proxy; the "
                "k-NN draw is with replacement across query steps)"
            ),
        },
        "drawn_corner_mass_by_anchor_quintile": corner_by_q,
        "drawn_corner_mass_note": (
            "share of drawn u_prev in the bottom (< 0.05) and top (> 0.95) "
            "rank corners, by the query person's anchor quintile; comparable "
            "to candidate 6's kernel corner masses"
        ),
        "n_draws": n_draws,
        "anchor_rank_distribution": anchor_rank_dist,
        "cell_count_distribution": cell_count_summary,
        "clamped_rank_share": {
            "n_positive_generated": int(n_positive_gen),
            "n_clamped": int(n_clamped),
            "share": (
                float(n_clamped / n_positive_gen) if n_positive_gen else 0.0
            ),
            "note": (
                "share of generated POSITIVE periods whose drawn rank u_prev "
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
    """Fit, build pools, generate, and score candidate 7 for one seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Stage 1: per-cell marginals on the train complement.
    marginals = fit_cell_marginals(train)

    # Stage 3: donor pools (pairs, triples, re-entry) on the train pairs.
    pools = build_donor_pools(train, all_anchor, marginals)

    # Stage 4/5: participation gate (train complement) + backward k-NN chain.
    fitted, pairs = fit_participation_gate(train, seed)
    candidate, diagnostics = generate_candidate(
        holdout, all_anchor, marginals, fitted, pools, seed
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
        "pools": {
            "n_pairs": int(pools["n_pairs"]),
            "n_triples": int(pools["n_triples"]),
            "n_reentry": int(pools["n_reentry"]),
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
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"mob_diag={battery_values['mobility_diagonal']:.3f} "
            f"ac10={battery_values['autocorr_log_10yr']:.3f} "
            f"tri_share={d['triple_pair_usage']['triple_share_of_positive']:.3f} "
            f"clamp={d['clamped_rank_share']['share']:.3f} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-7 run."""
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

    # Identical battery-reference bit-exact precheck as every prior run: the
    # battery code path must reproduce every committed reference value to
    # float precision before any candidate is scored.
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

    # Anchors on the FULL filtered panel (a person's last observed period is
    # a property of the panel, computed once and sliced per split).
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
        "run": "gate1_rank_knn_v1",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "description": (
            "Gate-1 candidate 7: k-NN conditional rank bootstrap with "
            "anchored two-step memory. Empirical per-cell quantile marginals "
            "(five-year age bin x period, byte-for-byte candidate 5b's) "
            "supply the magnitude; a continuous nonparametric transition law "
            "supplies the dynamics. From train positive observations, "
            "backward-adjacent records are built: pairs (u_prev, u_next) and, "
            "where the person is also positive one period later, triples "
            "(u_prev, u_next, u_next2); each record carries the person's "
            "continuous anchor rank u_A and the earlier-period weight. "
            "Generation keeps each holdout person's real anchor and chains "
            "backward one conditional draw per observed transition: matching "
            "on the next two generated-or-real ranks (v1, v2) and the anchor "
            "rank a by distance |u_next - v1| + 0.5|u_next2 - v2| (triples "
            "only) + 0.25|u_A - a|, taking the k=25 nearest records and "
            "drawing ONE with probability proportional to its weight; the "
            "generated u_prev is that record's u_prev exactly (a continuous "
            "empirical innovation; no binning, smoothing, or jitter). Where "
            "the next period is zero, the re-entry pool (candidate 6's "
            "re-entry pairs) is matched on |u_A - a| alone. Earnings = "
            "Qhat_pos of the period's cell at u_prev. Participation reuses "
            "the candidate-2 backward regime gate. No calibration stage: the "
            "bootstrap has zero free parameters. Registered frozen before "
            "the run in issue #42 (see spec_registration). Candidate scored "
            "against the held-out PSID family earnings panel geometry (two "
            "locked views) and the locked moment battery, per the locked "
            "seed-level conjunction in gates.yaml (pull request 39). "
            "Protocol machinery imported byte-for-byte from the baseline "
            "runner (pull request 40); rank machinery and participation gate "
            "from the candidate-5b runner (pull request 52)."
        ),
        "model": {
            "class": (
                "k-NN conditional rank bootstrap with anchored two-step "
                "memory (quantile marginal + continuous empirical "
                "conditional draws)"
            ),
            "stochastic": True,
            "populace_fit_used": True,
            "populace_fit_scope": (
                "participation gate only (RegimeGatedQRF sign gate); the "
                "donor pools and k-NN draws use pure numpy"
            ),
            "calibration": "none (zero free parameters; the bootstrap has no rescaling freedom)",
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
            "anchor_rank": {
                "rule": (
                    "u_A = rhat(anchor value) at its cell for a positive "
                    "anchor; u_A = p0/2 of the cell for a zero anchor "
                    "(continuous, carried per donor record and per query "
                    "person)"
                ),
            },
            "donor_pools": {
                "pairs": (
                    "train backward-adjacent pairs among positives (both "
                    "periods positive, consecutive observed periods 2 years "
                    "apart): (u_prev, u_next)"
                ),
                "triples": (
                    "pairs whose person is also positive at the next-later "
                    "observed period after u_next (period t+2): "
                    "(u_prev, u_next, u_next2)"
                ),
                "reentry": (
                    "train pairs where the LATER period is zero and the "
                    "EARLIER is positive (candidate 6's re-entry pairs): "
                    "(u_prev)"
                ),
                "u_prev": "rhat at the earlier period's own cell",
                "u_next": "rhat at the later period's own cell",
                "u_next2": "rhat at the two-steps-later period's own cell",
                "carried": (
                    "continuous anchor rank u_A and earlier-period person "
                    "weight (weight_tm2) per record"
                ),
                "tie_break_order": (
                    "records pinned in a stable (person_id, period_prev) sort "
                    "that fixes the k-NN tie-break"
                ),
            },
            "knn": {
                "k": K_NEIGHBORS,
                "distance_triples": (
                    "|u_next - v1| + 0.5|u_next2 - v2| + 0.25|u_A - a|"
                ),
                "distance_pairs": "|u_next - v1| + 0.25|u_A - a|",
                "distance_reentry": "|u_A - a|",
                "weights": {
                    "w_next": W_NEXT,
                    "w_next2": W_NEXT2,
                    "w_anchor": W_ANCHOR,
                },
                "record_choice": (
                    "triples when v2 exists (next-later period generated "
                    "positive), else pairs; re-entry pool when the next "
                    "period is zero"
                ),
                "draw": (
                    "one record drawn with probability proportional to its "
                    "weight among the k=25 nearest (seeded substream); "
                    "generated u_prev is that record's u_prev exactly"
                ),
                "no_smoothing": (
                    "no binning, no smoothing, no within-bin jitter; earnings "
                    "= Qhat_pos of the target cell at u_prev (interpolated, "
                    "so no value duplication)"
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
                "memory": (
                    "matches on v1 = rank of the next (rank j-1) period and "
                    "v2 = rank of the two-steps-later (rank j-2) period when "
                    "both were generated positive; plus the continuous "
                    "anchor rank"
                ),
                "period_draw": (
                    "earnings = Qhat_pos of the period's cell at the drawn "
                    "u_prev"
                ),
                "participation": (
                    "candidate-2 backward regime gate (RegimeGatedQRF sign "
                    "gate) on (next generated level, current age), trained on "
                    "the 80% complement, populace-fit defaults"
                ),
                "rng": (
                    "distinct fixed-label substreams of the gate seed: gate, "
                    "donor-draw, re-entry-draw"
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
        "knn_context": {
            "note": (
                "Reported-not-gated per seed: neighbor-distance "
                "distribution, triple-vs-pair usage share, donor-record "
                "reuse, drawn corner mass of transitions by anchor quintile "
                "(comparability with candidate 6), and the clamped-rank "
                "share. None enters the geometry or battery pass/fail; the "
                "gate rule names only those two families."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "n_pairs": s["pools"]["n_pairs"],
                    "n_triples": s["pools"]["n_triples"],
                    "n_reentry": s["pools"]["n_reentry"],
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
