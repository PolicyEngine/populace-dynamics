"""Gate-1 candidate 9: calibrated memory and a zero-anchor participation regime.

The ELEVENTH pre-registered model run of PolicyEngine/populace-dynamics, and
the FIRST candidate scored under the AMENDED gate (PR #57/#59: the runs-view
c2st demotion, plus a gated benefit-space block). It is candidate 7's
machinery VERBATIM with two registered changes -- both targeted at the two
remaining failures with the two diagnosed mechanisms:

* the battery's 10-year rung (candidate 7 undershoots at 0.459; candidate
  8's pure permanent-rank matching overshoots at 0.670 -- the reference band
  0.539 +/- 0.07 sits between), addressed by an SMM-calibrated blend of the
  donor's anchor rank and shrunk permanent rank (component 1);
* the pooled Q0 benefit band (the Q0 forensics, PR #61, localize c7's +9.3%
  entirely to a participation-law error -- the shared regime gate resurrects
  never-workers, +20.7pp zero-to-positive conversion, the level channel
  inside the noise floor), addressed by a zero-anchor participation regime
  (component 2).

The candidate-9 spec is registered, frozen before the run, in issue #42's
candidate-9 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4898825218);
every rule below is pinned there and implemented LITERALLY. No threshold is
hardcoded, no model choice is tuned against holdout scores. The only
calibration is the registered TRAIN-side SMM for lambda. The run is one
shot; the outcome publishes whether it passes or fails.

The two registered changes (everything else byte-identical to candidate 7):

1. **Long memory -- SMM-calibrated donor-coordinate blend.** The k-NN
   distance's third term becomes
   ``|lambda * u_w(donor) + (1 - lambda) * u_A(donor) - u_A(target)|`` at the
   SAME 0.25 weight (bare on re-entry, matching candidate 7's re-entry
   weight), where ``u_w`` is the donor's shrunk permanent rank exactly as
   candidate 8 computed it (candidate 3's stage-1 decomposition applied to
   the z-panel) and ``u_A`` the anchor ranks as in candidate 7. ``lambda`` is
   calibrated PER SEED on the TRAIN split only, by simulated method of
   moments in the 5b tradition: grid ``lambda in {0, 0.1, ..., 1.0}``; at
   each ``lambda``, generate with the FULL candidate machinery for a fixed
   train subsample (the first 2,000 train persons by ``person_id``, anchored
   at their real anchors, participation per component 2) and compute the
   battery autocorrelation ladder (locked definitions, lags 1/2/5) on the
   generated panel; pick the ``lambda`` minimizing equal-weight SSE against
   the train subsample's own ladder; ties to the SMALLER ``lambda``.
   ``lambda = 0`` reproduces candidate 7's matching; ``lambda = 1`` candidate
   8's; the calibration interpolates the demonstrated bracket by iterated
   dynamics, not by one-step fits.
2. **Zero-anchor participation regime.** Holdout persons with zero anchor
   earnings get a participation gate fit ONLY on train pairs whose person has
   zero anchor earnings (same features, same populace-fit defaults as the
   shared gate -- a conditional refit on the subpopulation whose law the
   shared gate provably misstates, not a target-matching dial); their
   re-entry rank pool is likewise restricted to zero-anchor train donors (the
   forensics measured the unrestricted pool importing attached persons'
   higher pre-gap ranks, +0.027 mean rank). Positive-anchor persons keep the
   shared gate and pools exactly as candidate 7.

Everything else -- ``k = 25``, the 1 / 0.5 lag weights, the weighted
single-record draw, no smoothing or jitter, the rank machinery, the gap
rule, the substream seeding -- is byte-identical to the candidate-7
registration. Note that candidate 9 does NOT adopt candidate 8's
substitution-2 attachment distance: the third distance term is the blend
above for ALL targets; the zero-anchor handling is the participation-gate
refit and the re-entry-pool restriction only.

Scored under the AMENDED gate (live in ``gates.yaml``, PR #57/#59):
GEOMETRY now conjoins the locked geometry thresholds on both views (the
runs-view c2st demoted to reported-not-gated) AND the per-seed benefit-space
metrics; the gate needs >= 4/5 geometry AND >= 4/5 battery AND the pooled Q0
band. The benefit-space measurements use the pinned PIA-proxy functional
from :mod:`build_downstream_relevance` (``POPULACE_DYNAMICS_PE_US_DIR`` points
at the pinned policyengine-us checkout).

The protocol mechanics -- the filter-first load, the person-disjoint 0.2
split per seed, the locked views, ``panel_scorecard`` scoring, the battery on
the candidate panel vs the committed ``battery_reference`` with locked
definitions, the thresholds read from ``gates.yaml`` at runtime, and the
battery-reference bit-exact precheck -- are IMPORTED from the merged baseline
runner (:mod:`run_gate1_baseline`, pull request 40), byte-for-byte the prior
runs'. The rank machinery (cells, ``CellMarginal``, ``fit_cell_marginals``,
anchors, ``anchor_u``, ``age_bin``) and the shared participation gate
(``_gate_sign_draw``) are imported from the merged candidate-5b runner (pull
request 52). Candidate 7's ``_knn_draw`` and ``anchor_quintile`` are imported
from the merged candidate-7 runner (pull request 55). The ``u_w``
decomposition (``build_z_panel``, ``build_donor_uw``) is imported from the
merged candidate-8 runner (pull request 58). The benefit-space functional
(``panel_pia_proxy``, ``distribution_gaps``, ``by_quintile``,
``anchor_quintile_cutpoints``) is imported from the merged PR-56 script. Only
the blended donor pools, the SMM-lambda calibration, the zero-anchor gate
refit + restricted re-entry pool, and the backward k-NN generation with the
blend are local.

Determinism. Stage-1 marginals, the ``u_w`` decomposition, the donor pools,
and the per-seed SMM-lambda calibration are deterministic given the split
(pure counting / grid NNLS / seeded common-random-number generation).
Stage-4/5 generation draws each of the gate, donor-draw, and re-entry-draw
substreams from its own fixed-label substream of the gate seed, in the
batched-by-step, ``person_id``-ordered pass the candidate-2 chain uses. The
run reproduces from the seeds alone.

Environment. The donor pools, the ``u_w`` decomposition, the SMM calibration,
and the k-NN draws are pure numpy/scipy and need NO populace-fit; the
participation gates (shared and zero-anchor) are ``RegimeGatedQRF`` sign
gates and DO need populace-fit. Run the full gate from the repository root
with the PSID family files staged, using the DEDICATED gate venv (populace-
fit pins scikit-learn < 1.9, which the repo's ``.venv`` violates; see
populace #318), and point the benefit-space oracle at the pinned pe-us
checkout::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_gate1_candidate9.py
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
    _MOMENT_KW,
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
# anchor_u, age_bin) and the shared participation gate (_gate_sign_draw) are
# IMPORTED from the merged candidate-5b runner so the quantile/rank maps, the
# continuous anchor-rank rule, and the candidate-2 backward sign-gate draw are
# byte-for-byte candidate 5b's.
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

# Candidate 7's k-NN weighted single-record draw and its anchor-quintile map
# are imported byte-for-byte (the draw rule and the reported corner-mass
# quintiles are unchanged by the blend).
from run_gate1_candidate7 import (  # noqa: F401 (re-exported for tests)
    K_NEIGHBORS,
    N_ANCHOR_QUINTILES,
    W_ANCHOR,
    W_NEXT,
    W_NEXT2,
    _knn_draw,
    anchor_quintile,
)

# The u_w z-panel decomposition (candidate 3's stage-1 machinery applied to
# z = Phi^-1(rank)) is imported byte-for-byte from the merged candidate-8
# runner; it is used verbatim (the donor permanent rank is unchanged, only
# how the third distance term combines u_w with the donor anchor rank).
# Imported LAZILY inside run_seed, not at module top level: candidate 8 pulls
# candidate 3 -> candidate 2 -> populace.fit at top level, so a module-level
# import here would pull populace-fit into every importer -- including the
# artifact-only consistency tests and the pure-numpy tests, which must run
# under the repo .venv without populace-fit.
from populace_dynamics.harness import moments
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_knn_v3.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank_knn.v3"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4898825218"
)
#: The candidate-7 registration this run substitutes into (base machinery).
BASE_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4896132094"
)
#: The candidate-8 registration the donor permanent rank u_w comes from.
UW_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4897723604"
)

# ---- Frozen constants of the candidate-9 registration ------------------
#: Component-1 SMM lambda grid: the eleven blend weights {0, 0.1, ..., 1.0}.
#: lambda = 0 reproduces candidate 7 (donor u_A); lambda = 1 candidate 8
#: (donor u_w). Pinned a priori.
LAMBDA_GRID = tuple(round(0.1 * i, 1) for i in range(11))
#: The SMM subsample size: the first 2,000 TRAIN persons by person_id
#: (anchored at their real anchors), generated with the full machinery at
#: each grid lambda. Pinned a priori.
SMM_SUBSAMPLE = 2000
#: The SMM autocorrelation ladder lags (biennial): 1/2/5 = 2/4/10 years,
#: the locked battery-autocorrelation definitions.
SMM_LAGS = (1, 2, 5)

#: Fixed integer codes for the generation RNG substream labels. Each label
#: seeds an independent generator via SeedSequence([seed, code]); byte-for-
#: byte candidate 7's (substream seeding is unchanged). The SMM-generation
#: pass uses a distinct fixed code so its draws never collide with the
#: holdout pass's.
SUBSTREAM_CODES = {"gate": 1, "donor-draw": 2, "re-entry-draw": 3}
#: A distinct substream code for the SMM-lambda train-subsample generation,
#: so the calibration pass is reproducible and independent of the holdout
#: pass (it never reads the holdout substreams).
SMM_STREAM_CODE = 6


def _substream(seed: int, label: str) -> np.random.Generator:
    """A generation RNG for one fixed substream label off the gate seed."""
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


def _smm_substream(seed: int, lam_idx: int, label: str) -> np.random.Generator:
    """An RNG for the SMM pass at grid point ``lam_idx``, one per substream.

    Seeded from ``SeedSequence([seed, SMM_STREAM_CODE, lam_idx, code])`` so
    every grid lambda draws its own reproducible stream and the calibration
    is deterministic from the gate seed. Distinct from the holdout-pass
    substreams (which omit ``SMM_STREAM_CODE``).
    """
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), SMM_STREAM_CODE, int(lam_idx), code]
        )
    )


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
    positive anchor -> ``rhat`` at its cell; zero anchor -> ``p0 / 2`` of the
    cell), evaluated on the person's anchor row (their chronologically last
    observed period). Returns ``{person_id: u_A}`` for the requested persons.
    Byte-for-byte candidate 7's helper.
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
# Zero-anchor participation gate (component 2): refit on the zero-anchor
# train subpopulation with the SAME features and populace-fit defaults.
# --------------------------------------------------------------------------
def fit_participation_gate(train: pd.DataFrame, seed: int) -> Any:
    """Fit the shared candidate-2 backward regime gate on the train complement.

    Byte-for-byte candidate 5b's shared gate: reuses the baseline/candidate-2
    backward pairs exactly (predictors = earnings at ``t`` and age at
    ``t-2``, target = earnings at ``t-2``, ``sample_weight`` = the earlier-
    period weight) and fits a ``RegimeGatedQRF`` at populace-fit defaults.
    Only its SIGN GATE is used at generation. Seeded from the gate seed.
    """
    from populace.fit.qrf import RegimeGatedQRF

    pairs = build_backward_pairs(train)
    model = RegimeGatedQRF(seed=seed)  # populace-fit defaults
    fitted = model.fit(
        pairs,
        predictors=["earnings", "age_tm2"],
        targets=["earnings_tm2"],
        weights="weight_tm2",
    )
    return fitted, pairs


def fit_zero_anchor_participation_gate(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    seed: int,
) -> tuple[Any, int]:
    """Component 2: participation gate on the ZERO-ANCHOR train subpopulation.

    A conditional refit of the shared gate on exactly the train pairs whose
    PERSON has zero anchor earnings (their chronologically last observed
    period is zero), with the SAME features (earnings at ``t``, age at
    ``t-2``; target earnings at ``t-2``; sample_weight the earlier-period
    weight) and the SAME populace-fit defaults, seeded from the gate seed.
    This is a conditional refit on the subpopulation whose participation law
    the shared gate provably misstates (PR #61: the shared gate resurrects
    never-workers at zero anchor), NOT a target-matching dial -- no threshold,
    no free constant.

    Returns ``(fitted_gate, n_pairs)``. If the zero-anchor pair pool is empty
    on a seed (never observed on the real panel), returns ``(None, 0)`` and
    the caller falls back to the shared gate for zero-anchor targets.
    """
    from populace.fit.qrf import RegimeGatedQRF

    zero_ids = set(
        int(p) for p in all_anchor[all_anchor.earnings == 0].person_id
    )
    pairs = build_backward_pairs(train)
    za_pairs = pairs[pairs["person_id"].isin(zero_ids)].reset_index(drop=True)
    n_pairs = int(len(za_pairs))
    if n_pairs == 0:
        return None, 0
    model = RegimeGatedQRF(seed=seed)  # populace-fit defaults
    fitted = model.fit(
        za_pairs,
        predictors=["earnings", "age_tm2"],
        targets=["earnings_tm2"],
        weights="weight_tm2",
    )
    return fitted, n_pairs


# --------------------------------------------------------------------------
# Stage 3 -- donor pools (train, weighted counting; no RNG)
# --------------------------------------------------------------------------
def build_donor_pools(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    u_w_of_person: dict[int, float],
) -> dict[str, Any]:
    """Build the pair, triple, and re-entry donor pools from the train split.

    Candidate 7's ``build_donor_pools`` in structure (pairs where both
    endpoints are positive, triples with a further positive period, and
    re-entry records where the later period is zero and the earlier is
    positive; each pinned in a stable ``(person_id, period_prev)`` sort that
    fixes the k-NN tie-break), with two additions -- each carried on every
    record and never itself a tuned dial:

    * ``u_w`` -- the donor person's shrunk permanent rank (component 1),
      attached via ``person_id`` (constant across a person's records).
    * ``anchor_zero`` -- whether the donor person's own anchor earnings are
      zero, so the zero-anchor re-entry pool can be sliced (component 2's
      re-entry restriction).

    Every record also carries the donor's continuous anchor rank ``u_A`` (the
    ``(1 - lambda)`` coordinate of the blend), the earlier-period weight, and
    the pinned key. Returns presorted numpy arrays per pool plus the raw
    counts and the zero-anchor re-entry pool size for the diagnostics.
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

    # Continuous anchor rank per pair (via the pair's person_id) -- the donor
    # anchor coordinate of the component-1 blend.
    uA_map = anchor_u_by_person(
        marginals, all_anchor, pairs["person_id"].to_numpy()
    )
    uA_of_pair = np.array([uA_map[int(p)] for p in pid], dtype=np.float64)

    # Component-1 donor permanent rank per pair (constant per person).
    uW_of_pair = np.array(
        [u_w_of_person[int(p)] for p in pid], dtype=np.float64
    )

    # Component-2 anchor-zero flag (per person): whether the donor's own
    # anchor earnings are zero (used to slice the zero-anchor re-entry pool).
    anchor_zero_of_person = {
        int(r.person_id): bool(float(r.earnings) == 0.0)
        for r in all_anchor.itertuples(index=False)
    }
    azero_of_pair = np.array(
        [anchor_zero_of_person[int(p)] for p in pid], dtype=bool
    )

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
    uW_p = uW_of_pair[kp_idx]
    azero_p = azero_of_pair[kp_idx]

    # --- Triples: a positive pair whose (person, period_next + step) is a
    # positive observation. Look up that period's earnings/age from the train
    # panel; compute u_next2 = rhat at its own cell.
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
    uW_r = uW_of_pair[re_idx]
    azero_r = azero_of_pair[re_idx]

    # --- Pin each pool in a stable (person_id, period_prev) order so the
    # k-NN tie-break is fully determined by record order (candidate 7's key).
    pair_order = np.lexsort((period_prev_p, pid_p))
    pairs_pool = {
        "u_prev": u_prev_p[pair_order],
        "u_next": u_next_p[pair_order],
        "u_A": uA_p[pair_order],
        "u_w": uW_p[pair_order],
        "anchor_zero": azero_p[pair_order],
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
        "u_w": uW_p[tri_idx][tri_order],
        "anchor_zero": azero_p[tri_idx][tri_order],
        "weight": w_p[tri_idx][tri_order],
        "person_id": tri_pid[tri_order],
        "period_prev": tri_period_prev[tri_order],
    }
    re_order = np.lexsort((period_prev_r, pid_r))
    reentry_pool = {
        "u_prev": u_prev_r[re_order],
        "u_A": uA_r[re_order],
        "u_w": uW_r[re_order],
        "anchor_zero": azero_r[re_order],
        "weight": w_r[re_order],
        "person_id": pid_r[re_order],
        "period_prev": period_prev_r[re_order],
    }

    # Component-2 zero-anchor re-entry pool: the subset of the re-entry pool
    # whose donor person's own anchor earnings are zero (used only for
    # zero-anchor targets' re-entry draws).
    re_mask_q0 = reentry_pool["anchor_zero"]
    reentry_q0 = {k: v[re_mask_q0] for k, v in reentry_pool.items()}

    return {
        "pairs": pairs_pool,
        "triples": triples_pool,
        "reentry": reentry_pool,
        "reentry_q0": reentry_q0,
        "n_pairs": int(both_pos.sum()),
        "n_triples": int(tri_idx.size),
        "n_reentry": int(reenter.sum()),
        "n_reentry_q0": int(reentry_q0["u_prev"].size),
        "n_train_pairs": int(len(pairs)),
    }


# --------------------------------------------------------------------------
# Component-1 blend: the donor coordinate d = lambda*u_w + (1-lambda)*u_A
# --------------------------------------------------------------------------
def _donor_blend(u_w: np.ndarray, u_A: np.ndarray, lam: float) -> np.ndarray:
    """The blended donor coordinate ``lambda * u_w + (1 - lambda) * u_A``.

    ``lambda = 0`` -> the donor anchor rank (candidate 7); ``lambda = 1`` ->
    the donor permanent rank (candidate 8). The k-NN third term is
    ``|blend - u_A(target)|``.
    """
    return lam * u_w + (1.0 - lam) * u_A


# --------------------------------------------------------------------------
# Stage 4/5 -- generation (backward k-NN chain + regime gate) over a target
# person set (the holdout, or the SMM train subsample)
# --------------------------------------------------------------------------
def _generate_over_targets(
    target_panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted_shared: Any,
    fitted_zero: Any,
    pools: dict[str, Any],
    lam: float,
    seed: int,
    rng_gate: np.random.Generator,
    rng_donor: np.random.Generator,
    rng_reentry: np.random.Generator,
    collect_diagnostics: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward k-NN chain over an arbitrary target person set.

    Candidate 7's backward chain with the two candidate-9 changes:

    * the k-NN third distance term is
      ``0.25 |lambda * u_w(donor) + (1 - lambda) * u_A(donor) - u_A(target)|``
      on transitions and ``|blend - u_A(target)|`` on re-entry (the blend is
      component 1; the target side stays ``u_A``, exactly as candidate 7/8);
    * zero-anchor targets draw their participation from the zero-anchor gate
      (``fitted_zero``; falls back to ``fitted_shared`` if the zero-anchor
      pool was empty) and their re-entry innovations from the zero-anchor-
      restricted re-entry pool (component 2). Positive-anchor targets use the
      shared gate and the full pools exactly as candidate 7.

    ``rng_gate`` / ``rng_donor`` / ``rng_reentry`` are the caller's fixed
    substreams (holdout pass or SMM pass), so the same generation code serves
    both the scored holdout and the SMM-lambda train subsample. Returns
    ``(candidate, diagnostics)``; ``diagnostics`` is populated only when
    ``collect_diagnostics`` (the holdout pass).
    """
    pairs_pool = pools["pairs"]
    triples_pool = pools["triples"]
    reentry_pool = pools["reentry"]
    reentry_q0 = pools["reentry_q0"]

    hp = target_panel.sort_values(["person_id", "period"]).reset_index(
        drop=True
    )
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
    # zero-anchor flag (read from the anchor rows directly, so it is exact --
    # not a float comparison on u_A).
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

    # Presorted donor arrays (pinned tie-break order): full pools plus the
    # zero-anchor-restricted re-entry pool. The blended donor coordinate is
    # formed per pool ONCE at this lambda (constant across query rows).
    tri_u_next = triples_pool["u_next"]
    tri_u_next2 = triples_pool["u_next2"]
    tri_blend = _donor_blend(triples_pool["u_w"], triples_pool["u_A"], lam)
    tri_w = triples_pool["weight"]
    tri_u_prev = triples_pool["u_prev"]
    pair_u_next = pairs_pool["u_next"]
    pair_blend = _donor_blend(pairs_pool["u_w"], pairs_pool["u_A"], lam)
    pair_w = pairs_pool["weight"]
    pair_u_prev = pairs_pool["u_prev"]
    re_blend = _donor_blend(reentry_pool["u_w"], reentry_pool["u_A"], lam)
    re_w = reentry_pool["weight"]
    re_u_prev = reentry_pool["u_prev"]
    re_q0_blend = _donor_blend(reentry_q0["u_w"], reentry_q0["u_A"], lam)
    re_q0_w = reentry_q0["weight"]
    re_q0_u_prev = reentry_q0["u_prev"]
    have_q0_reentry = re_q0_u_prev.size > 0

    # Diagnostics accumulators (holdout pass only).
    n_clamped = 0
    n_positive_gen = 0
    n_triple_draws = 0
    n_pair_draws = 0
    n_reentry_draws = 0
    n_reentry_draws_q0 = 0
    n_zero_anchor_positive_gen = 0
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

        # Participation gate on (next generated level, current age): the
        # zero-anchor persons use the zero-anchor gate (component 2); the
        # positive-anchor persons the shared gate. Draw from the SAME gate
        # substream in the fixed step order (one uniform per row), so the
        # stream is consumed identically regardless of the split.
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

            # ---- Branch A: next zero -> re-entry ----------------------------
            # Positive-anchor re-entry -> full re-entry pool, blend distance.
            rp = np.nonzero(~next_is_pos)[0]
            rp_pa = rp[~za_local[rp]]
            if rp_pa.size:
                a_r = a_local[rp_pa]
                dist = np.abs(re_blend[None, :] - a_r[:, None])
                u_dr = rng_reentry.random(rp_pa.size)
                drawn, kth = _knn_draw(dist, re_w, re_u_prev, u_dr)
                u_prev_local[rp_pa] = drawn
                if collect_diagnostics:
                    neighbor_dists.extend(float(x) for x in kth)
                n_reentry_draws += int(rp_pa.size)
            # Zero-anchor re-entry -> zero-anchor-restricted re-entry pool
            # (component 2), blend distance. Falls back to the full pool only
            # if the restricted pool is empty on this seed.
            rp_za = rp[za_local[rp]]
            if rp_za.size:
                a_r = a_local[rp_za]
                if have_q0_reentry:
                    dist = np.abs(re_q0_blend[None, :] - a_r[:, None])
                    u_dr = rng_reentry.random(rp_za.size)
                    drawn, kth = _knn_draw(dist, re_q0_w, re_q0_u_prev, u_dr)
                    n_reentry_draws_q0 += int(rp_za.size)
                else:
                    dist = np.abs(re_blend[None, :] - a_r[:, None])
                    u_dr = rng_reentry.random(rp_za.size)
                    drawn, kth = _knn_draw(dist, re_w, re_u_prev, u_dr)
                u_prev_local[rp_za] = drawn
                if collect_diagnostics:
                    neighbor_dists.extend(float(x) for x in kth)
                n_reentry_draws += int(rp_za.size)

            # ---- Branch B: next positive, v2 exists -> triple ---------------
            tp = np.nonzero(next_is_pos & has_v2)[0]
            if tp.size:
                v1_t = v1[tp]
                v2_t = v2[tp]
                a_t = a_local[tp]
                dist = (
                    W_NEXT * np.abs(tri_u_next[None, :] - v1_t[:, None])
                    + W_NEXT2 * np.abs(tri_u_next2[None, :] - v2_t[:, None])
                    + W_ANCHOR * np.abs(tri_blend[None, :] - a_t[:, None])
                )
                u_dt = rng_donor.random(tp.size)
                drawn, kth = _knn_draw(dist, tri_w, tri_u_prev, u_dt)
                u_prev_local[tp] = drawn
                if collect_diagnostics:
                    neighbor_dists.extend(float(x) for x in kth)
                n_triple_draws += int(tp.size)

            # ---- Branch C: next positive, no v2 -> pair ---------------------
            pp = np.nonzero(next_is_pos & ~has_v2)[0]
            if pp.size:
                v1_p = v1[pp]
                a_p = a_local[pp]
                dist = W_NEXT * np.abs(
                    pair_u_next[None, :] - v1_p[:, None]
                ) + W_ANCHOR * np.abs(pair_blend[None, :] - a_p[:, None])
                u_dp = rng_donor.random(pp.size)
                drawn, kth = _knn_draw(dist, pair_w, pair_u_prev, u_dp)
                u_prev_local[pp] = drawn
                if collect_diagnostics:
                    neighbor_dists.extend(float(x) for x in kth)
                n_pair_draws += int(pp.size)

            if collect_diagnostics:
                # Corner-mass bookkeeping by the query person's anchor
                # quintile.
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
                if (
                    collect_diagnostics
                    and cell.wtil.size
                    and (up < cell.wtil[0] or up > cell.wtil[-1])
                ):
                    n_clamped += 1
            n_positive_gen += int(pos_local.size)
            n_zero_anchor_positive_gen += int(za_local.sum())
        gen_earn[positions] = vals

    out = hp[["person_id", "period", "earnings", "age", "weight"]].copy()
    out["earnings"] = gen_earn

    diagnostics: dict[str, Any] = {}
    if collect_diagnostics:
        n_zero_anchor_holdout = int(
            sum(1 for p in target_ids if zero_anchor_person[int(p)])
        )
        diagnostics = _generation_diagnostics(
            pools,
            anchor_rank_vals,
            marginals,
            n_positive_gen,
            n_clamped,
            n_triple_draws,
            n_pair_draws,
            n_reentry_draws,
            n_reentry_draws_q0,
            np.asarray(neighbor_dists, dtype=np.float64),
            corner_bottom,
            corner_top,
            corner_total,
            int(len(target_ids)),
            n_zero_anchor_holdout,
        )
    return out, diagnostics


def generate_candidate(
    holdout: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted_shared: Any,
    fitted_zero: Any,
    pools: dict[str, Any],
    lam: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """k-NN conditional-rank-bootstrap candidate panel over the holdout.

    Runs :func:`_generate_over_targets` over the holdout persons with the
    holdout substreams and the calibrated ``lambda`` (component 1) and the
    zero-anchor participation regime (component 2), collecting the reported-
    not-gated diagnostics.
    """
    rng_gate = _substream(seed, "gate")
    rng_donor = _substream(seed, "donor-draw")
    rng_reentry = _substream(seed, "re-entry-draw")
    return _generate_over_targets(
        holdout,
        all_anchor,
        marginals,
        fitted_shared,
        fitted_zero,
        pools,
        lam,
        seed,
        rng_gate,
        rng_donor,
        rng_reentry,
        collect_diagnostics=True,
    )


def _generation_diagnostics(
    pools: dict[str, Any],
    anchor_rank_vals: np.ndarray,
    marginals: dict[tuple[int, int], CellMarginal],
    n_positive_gen: int,
    n_clamped: int,
    n_triple_draws: int,
    n_pair_draws: int,
    n_reentry_draws: int,
    n_reentry_draws_q0: int,
    neighbor_dists: np.ndarray,
    corner_bottom: np.ndarray,
    corner_top: np.ndarray,
    corner_total: np.ndarray,
    n_holdout_persons: int,
    n_zero_anchor_holdout: int,
) -> dict[str, Any]:
    """Assemble the reported-not-gated diagnostics for one seed.

    Candidate 7's named diagnostics (neighbor-distance distribution, triple-
    vs-pair usage share, donor-record reuse, drawn corner mass by anchor
    quintile, clamped-rank share, anchor-rank histogram, per-cell train
    positive-count summary) plus the candidate-9 additions: the count of
    zero-anchor re-entry draws routed through the restricted pool and the
    count of zero-anchor holdout persons.
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
        "n_zero_anchor_holdout_persons": int(n_zero_anchor_holdout),
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
        "zero_anchor_reentry": {
            "n_reentry_draws_q0": int(n_reentry_draws_q0),
            "n_reentry_draws_total": int(n_reentry_draws),
            "q0_share_of_reentry_draws": (
                float(n_reentry_draws_q0 / n_reentry_draws)
                if n_reentry_draws
                else 0.0
            ),
            "note": (
                "re-entry draws routed through the zero-anchor-restricted "
                "re-entry pool (component 2) vs all re-entry draws; the "
                "remainder used the full re-entry pool (positive-anchor "
                "targets)"
            ),
        },
        "donor_reuse": {
            "n_pair_records": int(pools["n_pairs"]),
            "n_triple_records": int(pools["n_triples"]),
            "n_reentry_records": int(pools["n_reentry"]),
            "n_reentry_records_q0": int(pools["n_reentry_q0"]),
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
            "to candidate 6's kernel corner masses and candidate 7's"
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
# Component 1 -- per-seed SMM-lambda calibration (TRAIN only)
# --------------------------------------------------------------------------
def _smm_train_subsample(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
) -> pd.DataFrame:
    """The fixed SMM subsample: the first 2,000 TRAIN persons by person_id.

    Deterministic given the split (no RNG): the person set is the numerically
    smallest ``SMM_SUBSAMPLE`` ``person_id`` in ``train``; the subsample is
    those persons' full observed support in the filtered panel (so the
    autocorrelation ladder is measured on the same rows for real and
    generated). Each person is anchored at its real anchor (its chronological
    last observed period), exactly as the holdout persons are.
    """
    ids = np.sort(train["person_id"].unique())[:SMM_SUBSAMPLE]
    id_set = set(int(x) for x in ids)
    sub = train[train.person_id.isin(id_set)].reset_index(drop=True)
    return sub


def _autocorr_ladder(panel: pd.DataFrame) -> dict[int, float]:
    """The locked battery autocorrelation ladder (lags 1/2/5) on a panel.

    Byte-for-byte the battery / SMM definition:
    ``moments.autocorrelation`` on log earnings among positives, weighted,
    biennial lags 1/2/5 (= 2/4/10 years). Returns ``{lag: value}``; a lag
    with too few positive pairs yields ``nan`` (handled by the SSE caller).
    """
    ac = moments.autocorrelation(
        panel, lags=SMM_LAGS, period_step=PERIOD_STEP, log=True, **_MOMENT_KW
    )
    return {int(r.lag): float(r.value) for r in ac.itertuples()}


def calibrate_lambda(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted_shared: Any,
    fitted_zero: Any,
    pools: dict[str, Any],
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Component-1 SMM: choose ``lambda`` on the TRAIN subsample.

    For each grid ``lambda in {0, 0.1, ..., 1.0}``: generate with the FULL
    candidate machinery over the fixed 2,000-person train subsample (real
    anchors, the zero-anchor participation regime), compute the battery
    autocorrelation ladder (lags 1/2/5) on the generated panel, and score its
    equal-weight SSE against the train subsample's OWN real ladder (the
    target). Pick the ``lambda`` minimizing the SSE; ties to the SMALLER
    ``lambda``. All draws are seeded from the gate seed via
    :func:`_smm_substream` (an SMM-specific substream per grid point), so the
    calibration is deterministic and independent of the holdout pass.

    Returns the chosen ``lambda`` index and value, the target ladder, and the
    full grid trace (each grid point's simulated ladder and SSE) -- reported
    in the artifact.
    """
    sub = _smm_train_subsample(train, all_anchor)
    target = _autocorr_ladder(sub)
    target_vec = np.array([target[lag] for lag in SMM_LAGS], dtype=np.float64)

    grid_trace: list[dict[str, Any]] = []
    best_idx: int | None = None
    best_sse = np.inf
    best_sim: dict[int, float] | None = None
    for lam_idx, lam in enumerate(LAMBDA_GRID):
        rng_gate = _smm_substream(seed, lam_idx, "gate")
        rng_donor = _smm_substream(seed, lam_idx, "donor-draw")
        rng_reentry = _smm_substream(seed, lam_idx, "re-entry-draw")
        gen, _ = _generate_over_targets(
            sub,
            all_anchor,
            marginals,
            fitted_shared,
            fitted_zero,
            pools,
            float(lam),
            seed,
            rng_gate,
            rng_donor,
            rng_reentry,
            collect_diagnostics=False,
        )
        sim = _autocorr_ladder(gen)
        sim_vec = np.array(
            [sim.get(lag, np.nan) for lag in SMM_LAGS], dtype=np.float64
        )
        # Equal-weight SSE against the train subsample's own ladder. A nan
        # simulated moment (too few positive pairs) is treated as an infinite
        # penalty so a degenerate lambda never wins.
        if np.any(~np.isfinite(sim_vec)):
            sse = float("inf")
        else:
            sse = float(np.sum((sim_vec - target_vec) ** 2))
        grid_trace.append(
            {
                "lambda": float(lam),
                "simulated_autocorr": {
                    f"lag{lag}": (
                        float(sim[lag])
                        if lag in sim and np.isfinite(sim[lag])
                        else None
                    )
                    for lag in SMM_LAGS
                },
                "sse": (sse if np.isfinite(sse) else None),
            }
        )
        # Strict "<" keeps the FIRST (smallest) lambda on ties.
        if sse < best_sse:
            best_sse = sse
            best_idx = lam_idx
            best_sim = dict(sim)
        if verbose:
            print(
                f"    lambda={lam:.1f} sse="
                f"{sse if np.isfinite(sse) else float('nan'):.6g} "
                f"ac=({sim.get(1, float('nan')):.3f},"
                f"{sim.get(2, float('nan')):.3f},"
                f"{sim.get(5, float('nan')):.3f})"
            )
    assert best_idx is not None
    return {
        "lambda": float(LAMBDA_GRID[best_idx]),
        "lambda_index": int(best_idx),
        "min_sse": (best_sse if np.isfinite(best_sse) else None),
        "n_smm_persons": int(sub.person_id.nunique()),
        "smm_subsample_target": int(SMM_SUBSAMPLE),
        "target_autocorr": {
            f"lag{lag}": float(target[lag]) for lag in SMM_LAGS
        },
        "chosen_simulated_autocorr": (
            {
                f"lag{lag}": (
                    float(best_sim[lag])
                    if best_sim is not None and lag in best_sim
                    else None
                )
                for lag in SMM_LAGS
            }
            if best_sim is not None
            else {}
        ),
        "lambda_grid": list(LAMBDA_GRID),
        "grid_trace": grid_trace,
    }


# --------------------------------------------------------------------------
# Amended-gate benefit-space block (GATED under PR #57/#59)
# --------------------------------------------------------------------------
def measure_benefit_space(
    seed: int,
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    all_anchor: pd.DataFrame,
    params: Any,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    """PIA-proxy gap block via the pinned PR-56 functional (now GATED).

    Pushes both the real holdout histories and the candidate-9 generated
    histories through :func:`build_downstream_relevance.panel_pia_proxy` (the
    statute-shaped PIA proxy, byte-for-byte the merged PR-56 functional),
    aligns the two on ``person_id`` (same person set, same rows, only earnings
    differ), and reports the weighted distribution-gap block, the person-level
    block, and the by-anchor-quintile concentration -- with Q0 (zero anchor
    earnings) called out (the pooled Q0 gate reads it). ``cutpoints`` are the
    seed-stable full-panel positive-anchor quartile edges. Under the amended
    gate this block's per-seed metrics conjoin into the geometry verdict and
    its pooled Q0 gate is a standalone gate condition.
    """
    import build_downstream_relevance as ds

    real_px = ds.panel_pia_proxy(holdout, all_anchor, params)
    cand_px = ds.panel_pia_proxy(candidate, all_anchor, params)
    merged = real_px.merge(
        cand_px, on="person_id", suffixes=("_real", "_cand")
    )
    assert np.allclose(
        merged["weight_real"].to_numpy(), merged["weight_cand"].to_numpy()
    ), "anchor weights diverged between sides"
    merged = merged.rename(columns={"weight_real": "weight"}).drop(
        columns=["weight_cand"]
    )
    assert (
        len(merged) == len(real_px) == len(cand_px)
    ), "candidate and real person sets differ"

    real = merged["pia_proxy_real"].to_numpy(dtype=np.float64)
    cand = merged["pia_proxy_cand"].to_numpy(dtype=np.float64)
    w = merged["weight"].to_numpy(dtype=np.float64)
    quintile_merged = merged.rename(
        columns={"pia_proxy_real": "pia_real", "pia_proxy_cand": "pia_cand"}
    )[["person_id", "pia_real", "pia_cand", "weight"]]

    return {
        "seed": seed,
        "n_persons": int(len(merged)),
        "n_real_zero_proxy": int(np.sum(real == 0.0)),
        "n_candidate_zero_proxy": int(np.sum(cand == 0.0)),
        "distribution": ds.distribution_gaps(cand, w, real, w),
        "person_level": ds.person_level_errors(cand, real, w),
        "by_anchor_quintile": ds.by_quintile(
            quintile_merged, all_anchor, cutpoints
        ),
    }


def check_benefit_space_per_seed(
    benefit_space: dict[str, Any],
    metrics_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the PER-SEED benefit-space gates from gates.yaml on one seed.

    The amended pass rule folds these into the seed's GEOMETRY verdict:
    ``abs_mean_pct_diff_max`` (|mean %| <= 5), ``abs_median_pct_diff_max``
    (|median %| <= 5), ``decile_pct_diff_max`` on the gated deciles d3..d9
    (|decile %| <= 5 each), and ``weighted_ks_max`` (KS <= 0.0599). The
    pooled Q0 gate (``abs_q0_mean_pct_diff_max``, scope pooled_across_seeds)
    is NOT checked here -- it is a standalone gate condition computed across
    seeds. A pct-diff that is ``None`` (real denominator zero) is a FAIL for
    that gated metric (the metric is not satisfiable), surfaced explicitly.
    """
    dist = benefit_space["distribution"]
    checks: dict[str, dict[str, Any]] = {}

    def _abs_le(name: str, value: float | None, thr: float) -> bool:
        if value is None:
            checks[name] = {
                "value": None,
                "threshold": thr,
                "comparison": "|.| <=",
                "pass": False,
                "note": "pct-diff undefined (real denominator zero)",
            }
            return False
        passed = abs(float(value)) <= thr
        checks[name] = {
            "value": float(value),
            "abs_value": abs(float(value)),
            "threshold": thr,
            "comparison": "|.| <=",
            "pass": bool(passed),
        }
        return passed

    def _le(name: str, value: float, thr: float) -> bool:
        passed = float(value) <= thr
        checks[name] = {
            "value": float(value),
            "threshold": thr,
            "comparison": "<=",
            "pass": bool(passed),
        }
        return passed

    all_pass = True

    mean_thr = float(metrics_cfg["abs_mean_pct_diff_max"]["value"])
    all_pass &= _abs_le(
        "abs_mean_pct_diff", dist["mean"]["pct_diff"], mean_thr
    )
    med_thr = float(metrics_cfg["abs_median_pct_diff_max"]["value"])
    all_pass &= _abs_le(
        "abs_median_pct_diff", dist["median"]["pct_diff"], med_thr
    )

    dec_cfg = metrics_cfg["decile_pct_diff_max"]
    dec_thr = float(dec_cfg["value"])
    for dkey in dec_cfg["deciles_gated"]:
        val = dist["deciles"][dkey]["pct_diff"]
        all_pass &= _abs_le(f"decile_{dkey}_pct_diff", val, dec_thr)

    ks_thr = float(metrics_cfg["weighted_ks_max"]["value"])
    all_pass &= _le("weighted_ks", dist["ks_distance"], ks_thr)

    return {"checks": checks, "benefit_space_seed_pass": bool(all_pass)}


def q0_participation_seed(
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    all_anchor: pd.DataFrame,
) -> dict[str, Any]:
    """Zero-anchor participation diagnostic for one seed (reported).

    Among the holdout persons whose anchor earnings are zero (the Q0
    subgroup), the generated panel's all-zero share (persons with no positive
    generated period ANYWHERE in their history) and mean positive-period
    count, next to the same statistics on the REAL holdout zero-anchor
    persons. The registration asks for exactly this comparison: whether the
    zero-anchor regime (change 2) keeps the generated participation law from
    resurrecting never-workers (PR #61's +20.7pp zero-to-positive
    conversion). The candidate anchor itself is held at the real zero, so the
    all-zero share and positive-period counts are read over each person's
    NON-anchor periods (the periods the chain actually generates).
    """
    zero_ids = set(
        int(p) for p in all_anchor[all_anchor.earnings == 0].person_id
    )
    hold_ids = set(int(p) for p in holdout["person_id"].unique())
    q0_ids = zero_ids & hold_ids

    # Anchor period per person (held at real earnings on both sides); the
    # generated participation law is read on the person's NON-anchor rows.
    anchor_period = dict(
        zip(
            all_anchor["person_id"].to_numpy(),
            all_anchor["period"].to_numpy(),
            strict=True,
        )
    )

    def _stats(panel: pd.DataFrame) -> dict[str, Any]:
        sub = panel[panel["person_id"].isin(q0_ids)]
        n_all_zero = 0
        pos_counts: list[int] = []
        n_persons = 0
        for pid, g in sub.groupby("person_id", sort=True):
            ap = int(anchor_period[int(pid)])
            non_anchor = g[g["period"] != ap]
            npos = int((non_anchor["earnings"].to_numpy() > 0).sum())
            pos_counts.append(npos)
            if npos == 0:
                n_all_zero += 1
            n_persons += 1
        return {
            "n_persons": int(n_persons),
            "all_zero_share": (
                float(n_all_zero / n_persons) if n_persons else 0.0
            ),
            "mean_positive_periods": (
                float(np.mean(pos_counts)) if pos_counts else 0.0
            ),
            "note": (
                "over each zero-anchor person's NON-anchor periods (the "
                "anchor is held at the real zero on both sides)"
            ),
        }

    return {
        "n_q0_persons": int(len(q0_ids)),
        "generated": _stats(candidate),
        "real": _stats(holdout),
    }


def check_pooled_q0(
    per_seed: list[dict[str, Any]],
    metrics_cfg: dict[str, Any],
) -> dict[str, Any]:
    """The pooled (across-seed mean) Q0 gate from gates.yaml.

    ``abs_q0_mean_pct_diff_max`` (value 5, scope pooled_across_seeds): the
    absolute across-seed mean of the Q0 subgroup's mean PIA-proxy % gap must
    be ``<= 5``. Q0 = zero anchor earnings; its per-seed real-vs-real floor is
    noisy (~900 persons/side, near-zero denominators), so the gate pools the
    seed means (real is unbiased: pooled |.| ~2.7%). Seeds where Q0 is empty
    or the % gap is undefined are skipped in the mean (Q0 is populated on
    every real seed here).
    """
    thr = float(metrics_cfg["abs_q0_mean_pct_diff_max"]["value"])
    q0_means: list[float] = []
    per_seed_q0: list[dict[str, Any]] = []
    for s in per_seed:
        bs = s.get("benefit_space")
        val: float | None = None
        if bs is not None:
            q0 = bs["by_anchor_quintile"]["quintiles"].get("Q0", {})
            if q0.get("n_persons", 0) > 0:
                val = q0["distribution"]["mean"]["pct_diff"]
        per_seed_q0.append(
            {
                "seed": s["seed"],
                "q0_mean_pct_diff": (float(val) if val is not None else None),
            }
        )
        if val is not None:
            q0_means.append(float(val))
    if q0_means:
        pooled = float(np.mean(q0_means))
        passed = abs(pooled) <= thr
    else:
        pooled = None
        passed = False
    return {
        "pooled_q0_mean_pct_diff": pooled,
        "abs_pooled_q0_mean_pct_diff": (
            abs(pooled) if pooled is not None else None
        ),
        "threshold": thr,
        "comparison": "|.| <=",
        "n_seeds_with_q0": len(q0_means),
        "per_seed_q0": per_seed_q0,
        "pooled_q0_pass": bool(passed),
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
    benefit_metrics_cfg: dict[str, Any],
    benefit_params: Any,
    benefit_cutpoints: np.ndarray | None,
    verbose: bool,
) -> dict[str, Any]:
    """Fit, calibrate lambda, generate, and score candidate 9 for one seed.

    The benefit-space block is GATED under the amendment; when
    ``benefit_params`` is provided it is measured and its per-seed gates fold
    into the seed's geometry verdict. When it is ``None`` (SSA oracle absent)
    the block is skipped and the seed's geometry verdict is the locked
    geometry thresholds only -- but the amended gate is not fully evaluable,
    so :func:`run` refuses to publish a verdict without the oracle.
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
    # the anchor-zero flag; plus the zero-anchor-restricted re-entry pool.
    pools = build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )

    # Component 2: the shared participation gate AND the zero-anchor gate
    # (refit on the zero-anchor train subpopulation).
    fitted_shared, pairs = fit_participation_gate(train, seed)
    fitted_zero, n_zero_pairs = fit_zero_anchor_participation_gate(
        train, all_anchor, seed
    )

    # Component 1: per-seed SMM-lambda calibration on the TRAIN subsample.
    smm = calibrate_lambda(
        train,
        all_anchor,
        marginals,
        fitted_shared,
        fitted_zero,
        pools,
        seed,
        verbose=verbose,
    )
    lam = float(smm["lambda"])

    # Stage 4/5: backward k-NN chain over the holdout with the calibrated
    # lambda and the zero-anchor regime.
    candidate, diagnostics = generate_candidate(
        holdout,
        all_anchor,
        marginals,
        fitted_shared,
        fitted_zero,
        pools,
        lam,
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

    # Q0 participation diagnostic (reported): generated vs real all-zero
    # share and mean positive periods for the zero-anchor holdout subgroup.
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
    # benefit-space metrics (mean %, median %, gated deciles, KS). If the
    # benefit block is absent the amended geometry verdict is not computable;
    # record the locked-thresholds-only verdict and mark it as such.
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
        "lambda_calibration": smm,
        "lambda": lam,
        "uw_fit": uw["fit"],
        "pools": {
            "n_pairs": int(pools["n_pairs"]),
            "n_triples": int(pools["n_triples"]),
            "n_reentry": int(pools["n_reentry"]),
            "n_reentry_q0": int(pools["n_reentry_q0"]),
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
        print(
            f"seed {seed}: lambda={lam:.1f} "
            f"geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"mob_diag={battery_values['mobility_diagonal']:.3f} "
            f"ac10={battery_values['autocorr_log_10yr']:.3f} "
            f"za_reentry={d['zero_anchor_reentry']['q0_share_of_reentry_draws']:.3f} "
            f"clamp={d['clamped_rank_share']['share']:.3f}{bs} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def _load_benefit_oracle() -> tuple[Any, np.ndarray | None]:
    """Load the SSA oracle params + full-panel quartile cuts, or (None, None).

    Under the amendment the benefit-space block is GATED, so the oracle must
    be present for a verdict; :func:`run` refuses to publish otherwise. This
    helper returns ``(None, None)`` on failure and lets ``run`` decide.
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
    """Execute the full pre-registered gate-1 candidate-9 run (amended gate)."""
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

    # Identical battery-reference bit-exact precheck as every prior run.
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
        "run": "gate1_rank_knn_v3",
        "gate": "gate_1",
        "gate_variant": "amended (PR #57/#59): runs-view c2st demoted; "
        "benefit_space block gated; geometry conjoins per-seed benefit_space; "
        ">=4/5 geometry AND >=4/5 battery AND pooled Q0",
        "spec_registration": SPEC_REGISTRATION,
        "base_registration": BASE_REGISTRATION,
        "uw_registration": UW_REGISTRATION,
        "changes": (
            "candidate 7's machinery verbatim with two registered changes: "
            "(1) an SMM-calibrated donor-coordinate blend "
            "|lambda*u_w(donor) + (1-lambda)*u_A(donor) - u_A(target)| at the "
            "0.25 weight, lambda chosen per seed by train-side SMM on the "
            "autocorrelation ladder; (2) a zero-anchor participation regime "
            "(gate refit on the zero-anchor train subpopulation + zero-anchor-"
            "restricted re-entry pool). Candidate 9 does NOT adopt candidate "
            "8's attachment distance."
        ),
        "description": (
            "Gate-1 candidate 9: calibrated memory and a zero-anchor "
            "participation regime -- the first candidate scored under the "
            "amended gate. Candidate 7's k-NN conditional rank bootstrap "
            "(empirical per-cell quantile marginals supply the magnitude; a "
            "continuous nonparametric transition law supplies the dynamics) "
            "with two registered changes. Change 1 (long memory): the k-NN "
            "third distance term becomes |lambda*u_w(donor) + "
            "(1-lambda)*u_A(donor) - u_A(target)| at candidate 7's 0.25 "
            "weight, where u_w is candidate 8's shrunk permanent rank (candidate "
            "3's stage-1 decomposition on the z-panel) and u_A the anchor "
            "ranks; lambda is calibrated per seed on the TRAIN split by "
            "simulated method of moments in the 5b tradition -- grid "
            "{0,0.1,...,1.0}, at each lambda generate the full machinery over "
            "the first 2,000 train persons (real anchors, the zero-anchor "
            "regime) and score the battery autocorrelation ladder (lags "
            "1/2/5) SSE against the subsample's own ladder, picking the "
            "minimizing lambda (ties to the smaller). lambda=0 is candidate 7, "
            "lambda=1 candidate 8; the calibration interpolates the bracket by "
            "iterated dynamics. Change 2 (zero-anchor participation regime): "
            "zero-anchor holdout persons draw participation from a gate refit "
            "ONLY on zero-anchor train pairs (same features, same populace-fit "
            "defaults -- a conditional refit, not a dial) and their re-entry "
            "innovations from a re-entry pool restricted to zero-anchor train "
            "donors; positive-anchor persons keep the shared gate and full "
            "pools exactly as candidate 7. Everything else -- k=25, the 1/0.5 "
            "lag weights, the weighted single-record draw, no smoothing/jitter, "
            "the rank machinery, the gap rule, the substream seeding -- is "
            "byte-identical to candidate 7. Registered frozen before the run "
            "in issue #42 (see spec_registration). Scored under the AMENDED "
            "gate live in gates.yaml (PR #57/#59): geometry conjoins the "
            "locked geometry thresholds (runs-view c2st demoted to "
            "reported-not-gated) and the per-seed benefit-space metrics; the "
            "gate needs >=4/5 geometry AND >=4/5 battery AND the pooled Q0 "
            "band. Protocol machinery imported byte-for-byte from the baseline "
            "runner (PR #40); rank machinery and shared gate from candidate 5b "
            "(PR #52); the k-NN draw and anchor quintiles from candidate 7 (PR "
            "#55); the u_w decomposition from candidate 8 (PR #58); the "
            "benefit-space functional from PR #56."
        ),
        "model": {
            "class": (
                "k-NN conditional rank bootstrap with an SMM-calibrated "
                "anchor/permanent-rank donor blend and a zero-anchor "
                "participation regime (quantile marginal + continuous "
                "empirical conditional draws)"
            ),
            "stochastic": True,
            "populace_fit_used": True,
            "populace_fit_scope": (
                "the shared and zero-anchor participation gates "
                "(RegimeGatedQRF sign gates); the donor pools, the u_w "
                "decomposition, the SMM calibration, and the k-NN draws use "
                "pure numpy/scipy"
            ),
            "calibration": (
                "one registered stage: the TRAIN-side SMM for lambda (grid "
                "{0,0.1,...,1.0}, autocorrelation-ladder SSE on the first "
                "2,000 train persons, ties to the smaller lambda). No holdout "
                "score is read; the u_w decomposition is candidate 3's "
                "deterministic grid-rho NNLS on TRAIN only; the bootstrap has "
                "no rescaling freedom."
            ),
            "change_1_donor_blend": {
                "third_distance_term": (
                    "|lambda*u_w(donor) + (1-lambda)*u_A(donor) - "
                    "u_A(target)| at weight 0.25 (transitions) / bare "
                    "(re-entry); the TARGET side stays u_A, exactly as "
                    "candidate 7/8"
                ),
                "u_w": (
                    "candidate 8's donor permanent rank u_w = "
                    "Phi(what/sigma_hat_w) from candidate 3's stage-1 "
                    "decomposition on the z-panel (imported byte-for-byte)"
                ),
                "lambda_grid": list(LAMBDA_GRID),
                "smm": {
                    "tradition": "5b (train-side SMM, common random numbers)",
                    "subsample": (
                        f"the first {SMM_SUBSAMPLE} train persons by "
                        "person_id, anchored at their real anchors, "
                        "participation per the zero-anchor regime"
                    ),
                    "moments": (
                        "battery autocorrelation ladder, lags 1/2/5 (= "
                        "2/4/10 years), locked definitions"
                    ),
                    "objective": (
                        "equal-weight SSE of the generated ladder against the "
                        "train subsample's own ladder; pick the minimizing "
                        "lambda, ties to the smaller"
                    ),
                    "iterated_dynamics": (
                        "generation runs the full candidate machinery per "
                        "grid point (not a one-step fit)"
                    ),
                    "seeding": (
                        "an SMM-specific substream per grid point off the "
                        "gate seed (SeedSequence[seed, 6, lambda_index, "
                        "code]); independent of the holdout pass"
                    ),
                },
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
                    "SAME populace-fit defaults; a conditional refit, no dial"
                ),
                "reentry_pool": (
                    "the re-entry innovations for zero-anchor targets are "
                    "drawn from a re-entry pool restricted to zero-anchor "
                    "train donors (the forensics measured the unrestricted "
                    "pool importing attached persons' higher pre-gap ranks, "
                    "+0.027 mean rank)"
                ),
                "positive_anchor_unchanged": (
                    "positive-anchor persons keep the shared gate and the "
                    "full pools exactly as candidate 7"
                ),
                "empty_pool_fallback": (
                    "if a seed's zero-anchor pair pool is empty the shared "
                    "gate is used; if the zero-anchor re-entry pool is empty "
                    "the full re-entry pool is used (never observed empty on "
                    "the real panel)"
                ),
                "not_candidate_8_attachment": (
                    "candidate 9 does NOT adopt candidate 8's substitution-2 "
                    "attachment distance (|d_age|/40 + |d_nobs|/13); the "
                    "third distance term is the blend above for ALL targets"
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
                    "(u_prev)"
                ),
                "reentry_q0": (
                    "the subset of the re-entry pool whose donor person's own "
                    "anchor earnings are zero, used only for zero-anchor "
                    "targets' re-entry draws"
                ),
                "tie_break_order": (
                    "records pinned in a stable (person_id, period_prev) sort "
                    "that fixes the k-NN tie-break (byte-for-byte candidate 7)"
                ),
            },
            "knn": {
                "k": K_NEIGHBORS,
                "distance_triples": (
                    "|u_next - v1| + 0.5|u_next2 - v2| + "
                    "0.25|lambda*u_w + (1-lambda)*u_A - a|"
                ),
                "distance_pairs": (
                    "|u_next - v1| + 0.25|lambda*u_w + (1-lambda)*u_A - a|"
                ),
                "distance_reentry": ("|lambda*u_w + (1-lambda)*u_A - a|"),
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
                    "defaults, drawn from the same gate substream in the "
                    "fixed step order"
                ),
                "rng": (
                    "distinct fixed-label substreams of the gate seed: gate, "
                    "donor-draw, re-entry-draw (byte-for-byte candidate 7); "
                    "the SMM pass uses an independent SMM substream per grid "
                    "point"
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
                "Reported-not-gated per seed: the lambda calibration (target "
                "and grid ladders), neighbor-distance distribution, triple-vs-"
                "pair usage share, the zero-anchor re-entry split, donor-"
                "record reuse, drawn corner mass by anchor quintile, the "
                "clamped-rank share, and the u_w decomposition. None of these "
                "enters the pass/fail beyond the amended gate's named blocks."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "lambda": s["lambda"],
                    "lambda_calibration": s["lambda_calibration"],
                    "n_pairs": s["pools"]["n_pairs"],
                    "n_triples": s["pools"]["n_triples"],
                    "n_reentry": s["pools"]["n_reentry"],
                    "n_reentry_q0": s["pools"]["n_reentry_q0"],
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
                    "zero_anchor_reentry": s["generation_diagnostics"][
                        "zero_anchor_reentry"
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


def _q0_participation_diagnostics(
    per_seed: list[dict[str, Any]],
    panel: pd.DataFrame,
) -> dict[str, Any]:
    """Q0 participation diagnostics: generated vs real all-zero share (pooled).

    The registration asks for the zero-anchor subgroup's participation
    behaviour under the refit regime: the generated all-zero share (persons
    whose entire generated non-anchor history is zero) and the mean
    positive-period count, per seed, next to the same statistics on the REAL
    holdout zero-anchor persons. This measures whether the zero-anchor regime
    (change 2) keeps the generated participation law from resurrecting
    never-workers (PR #61's +20.7pp zero-to-positive conversion). Reported,
    not itself gated (the pooled Q0 benefit gate is the gated Q0 condition).
    """
    rows = []
    gen_azs: list[float] = []
    real_azs: list[float] = []
    gen_mpp: list[float] = []
    real_mpp: list[float] = []
    for s in per_seed:
        qp = s.get("q0_participation")
        if qp is None:
            continue
        rows.append(
            {
                "seed": s["seed"],
                "n_q0_persons": qp["n_q0_persons"],
                "generated": qp["generated"],
                "real": qp["real"],
            }
        )
        gen_azs.append(float(qp["generated"]["all_zero_share"]))
        real_azs.append(float(qp["real"]["all_zero_share"]))
        gen_mpp.append(float(qp["generated"]["mean_positive_periods"]))
        real_mpp.append(float(qp["real"]["mean_positive_periods"]))

    def _mean(v: list[float]) -> float | None:
        return float(np.mean(v)) if v else None

    return {
        "note": (
            "Per seed, the zero-anchor subgroup's generated vs real all-zero "
            "share and mean positive-period count (over each person's "
            "non-anchor periods); measures whether the zero-anchor regime "
            "(change 2) stops the shared gate from resurrecting never-workers "
            "(PR #61's +20.7pp zero-to-positive conversion)."
        ),
        "pooled": {
            "generated_all_zero_share_mean": _mean(gen_azs),
            "real_all_zero_share_mean": _mean(real_azs),
            "all_zero_share_gap": (
                _mean(gen_azs) - _mean(real_azs)
                if gen_azs and real_azs
                else None
            ),
            "generated_mean_positive_periods_mean": _mean(gen_mpp),
            "real_mean_positive_periods_mean": _mean(real_mpp),
        },
        "per_seed": rows,
    }


def _pool_benefit_space(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the benefit-space block across seeds (mean/sd/min/max, Q0 out).

    Mirrors PR #56's pooling. Q0 is called out (the pooled Q0 gate reads its
    mean). Returns an empty marker if no seed carried the block.
    """
    rows = [s["benefit_space"] for s in per_seed if "benefit_space" in s]
    if not rows:
        return {"available": False}

    def _summ(vals: list[Any]) -> dict[str, Any]:
        arr = np.array([v for v in vals if v is not None], dtype=np.float64)
        if arr.size == 0:
            return {
                "mean": None,
                "sd": None,
                "min": None,
                "max": None,
                "n_seeds": 0,
            }
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
        }

    def _dig(block: dict[str, Any], path: tuple[str, ...]) -> Any:
        node: Any = block
        for k in path:
            node = node[k]
        return node

    dist = {
        "mean_pct_diff": _summ(
            [_dig(r, ("distribution", "mean", "pct_diff")) for r in rows]
        ),
        "median_pct_diff": _summ(
            [_dig(r, ("distribution", "median", "pct_diff")) for r in rows]
        ),
        "gini_difference": _summ(
            [_dig(r, ("distribution", "gini", "difference")) for r in rows]
        ),
        "ks_distance": _summ(
            [_dig(r, ("distribution", "ks_distance")) for r in rows]
        ),
    }
    decile = {}
    for i in range(1, 10):
        key = f"d{i}"
        decile[key] = _summ(
            [
                _dig(r, ("distribution", "deciles", key, "pct_diff"))
                for r in rows
            ]
        )
    dist["decile_pct_diff"] = decile

    person_level = {
        k: _summ([_dig(r, ("person_level", k)) for r in rows])
        for k in (
            "weighted_mae",
            "weighted_rmse",
            "weighted_share_within_5pct",
            "weighted_share_within_10pct",
        )
    }

    stat_paths: dict[str, tuple[str, ...]] = {
        "mean_n_persons": ("n_persons",),
        "mean_pct_diff": ("distribution", "mean", "pct_diff"),
        "median_pct_diff": ("distribution", "median", "pct_diff"),
        "ks_distance": ("distribution", "ks_distance"),
        "gini_difference": ("distribution", "gini", "difference"),
        "weighted_mae": ("person_level", "weighted_mae"),
        "weighted_rmse": ("person_level", "weighted_rmse"),
        "weighted_share_within_5pct": (
            "person_level",
            "weighted_share_within_5pct",
        ),
        "weighted_share_within_10pct": (
            "person_level",
            "weighted_share_within_10pct",
        ),
    }
    by_q: dict[str, Any] = {}
    for q in range(5):
        key = f"Q{q}"
        blocks = [
            r["by_anchor_quintile"]["quintiles"][key]
            for r in rows
            if r["by_anchor_quintile"]["quintiles"][key].get("n_persons", 0)
            > 0
        ]
        if not blocks:
            by_q[key] = {"n_seeds_present": 0}
            continue
        entry: dict[str, Any] = {"n_seeds_present": len(blocks)}
        for name, path in stat_paths.items():
            vals = []
            for b in blocks:
                node: Any = b
                ok = True
                for k in path:
                    if isinstance(node, dict) and k in node:
                        node = node[k]
                    else:
                        ok = False
                        break
                if ok and node is not None:
                    vals.append(float(node))
            entry[name] = float(np.mean(vals)) if vals else None
        by_q[key] = entry

    return {
        "available": True,
        "gated": True,
        "functional": (
            "pinned PR-56 statute-shaped PIA proxy "
            "(build_downstream_relevance.panel_pia_proxy), applied "
            "identically to candidate 9's holdout and the real holdout"
        ),
        "distribution": dist,
        "person_level": person_level,
        "by_anchor_quintile": by_q,
        "candidate7_reference": {
            "note": (
                "candidate 7's committed downstream numbers (PR #56, "
                "runs/downstream_relevance_c7_v1.json) for comparison"
            ),
            "mean_pct_diff": 1.8477489952384396,
            "median_pct_diff": 1.0171410947686295,
            "ks_distance": 0.02535050245252706,
            "Q0_mean_pct_diff": 9.295292319680367,
            "Q0_median_pct_diff": 22.915959347981033,
            "Q0_ks_distance": 0.09679229074277092,
        },
        "candidate8_reference": {
            "note": (
                "candidate 8's committed Q0 mean % (runs/gate1_rank_knn_v2."
                "json benefit_space_reported_not_gated) for comparison"
            ),
        },
    }


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
