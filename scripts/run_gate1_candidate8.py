"""Gate-1 candidate 8: permanent-rank donor matching with attachment-aware
zero conditioning.

The TENTH pre-registered model run of PolicyEngine/populace-dynamics, and
the last planned candidate iteration before the governance track resolves.
It is candidate 7's machinery VERBATIM with EXACTLY two registered
substitutions -- both conditioning refinements that add no tuned constant.
Candidate 7 left two residuals with one shared cause (insufficient
long-horizon memory: 10-year autocorrelation short on 3/5 seeds, runs-view
c2st over-threshold on 3/5), and the downstream analysis (pull request 56)
exposed a decision-relevant weakness the gate never isolated: the
zero-anchor subgroup's generated careers overstate PIA-proxy by +9.3% mean.

The candidate-8 spec is registered, frozen before the run, in issue #42's
candidate-8 comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4897723604);
every rule below is pinned there and implemented LITERALLY. No threshold
is hardcoded, no model choice is tuned against holdout scores, and there
is no fit-time freedom (the bootstrap has no rescaling freedom; the u_w
decomposition is candidate 3's deterministic grid-rho NNLS, calibrated on
TRAIN only). The run is one shot; the outcome publishes whether it passes
or fails.

The two registered substitutions (everything else byte-identical to
candidate 7):

1. **Long memory -- donors matched on their permanent rank.** Every train
   donor record carries ``u_w = Phi(what / sigma_hat_w)``, where ``what``
   is the donor person's correlated-noise-shrunk within-person mean of
   ``z = Phi^-1(rank)`` over their positive observations (candidate 3's
   stage-1 machinery applied to the Z-PANEL: within-person autocovariances,
   grid-rho NNLS decomposition, per-person shrinkage weights) and
   ``sigma_hat_w^2`` is the decomposition's permanent variance. The k-NN
   distance's third term becomes ``|u_w(donor) - u_A(target)|`` at the SAME
   0.25 weight as candidate 7 (which used ``|u_A(donor) - u_A(target)|``).
   The target's anchor rank ``u_A`` is production-available and stays; the
   donor's side upgrades from a single noisy anchor observation to the
   shrunk full-career permanent estimate. Zero new dials; one substitution.
2. **Zero-anchor conditioning -- attachment structure, not a constant.**
   For holdout persons with zero anchor earnings (``u_A = p0 / 2``
   identically, so the anchor term carries nothing), ALL their k-NN
   distances (transitions and re-entry) replace the third term with
   ``|Delta age at the step| / 40 + |Delta n_observed_periods| / 13``,
   where ``n_observed_periods`` is the person's observed-period count in
   the filtered panel and the scales are the ranges of the respective
   variables (pinned as range widths, not tuned). Donor pools for these
   draws are restricted to train records from persons whose own anchor
   earnings are zero.

Everything else -- donor pools, ``k = 25``, the 1 / 0.5 lag weights, the
weighted single-record draw, no smoothing or jitter, the re-entry pools,
the regime gate, the rank machinery, the gap rule, the substream seeding
-- is byte-identical to the candidate-7 registration. Scored under the
CURRENT locked gate (the amendment of the parallel proposal pull request
changes nothing until refereed and ratified); the artifact additionally
REPORTS the proposed benefit-space block's measurements (PIA-proxy gaps
including Q0, via the pinned functional of pull request 56) so this run
carries evidence for both the locked and the proposed standard.

The protocol mechanics -- the filter-first load, the person-disjoint 0.2
split per seed, the two locked views, ``panel_scorecard`` scoring, the
battery on the candidate panel vs the committed ``battery_reference`` with
locked definitions, the thresholds read from ``gates.yaml`` at runtime,
the seed-level conjunction (>=4/5 both blocks), and the battery-reference
bit-exact precheck -- are IMPORTED from the merged baseline runner
(:mod:`run_gate1_baseline`, pull request 40), byte-for-byte the prior
runs'. The rank machinery (cells, ``CellMarginal``, ``fit_cell_marginals``,
anchors, ``anchor_u``, ``age_bin``) and the participation gate
(``fit_participation_gate``, ``_gate_sign_draw``) are imported from the
merged candidate-5b runner (:mod:`run_gate1_candidate5b`, pull request
52). Candidate 7's ``_knn_draw`` and ``anchor_quintile`` are imported from
the merged candidate-7 runner (:mod:`run_gate1_candidate7`, pull request
55). The u_w decomposition machinery (``pooled_autocovariances``,
``fit_three_component``, ``person_effects_pa``) is imported from the
merged candidate-3 runner (:mod:`run_gate1_candidate3`, pull request 44)
and applied to the z-panel. Only the u_w z-panel construction, the
Q0-conditioned donor pools, and the backward k-NN generation with the two
substitutions are local.

Determinism. Stage-1 marginals, the u_w decomposition, and the stage-3
donor pools are deterministic given the split (pure counting / grid NNLS,
no RNG). Stage-4/5 generation draws each of the gate, donor-draw, and
re-entry-draw substreams from its own fixed-label substream of the gate
seed, in the batched-by-step, ``person_id``-ordered pass the candidate-2
chain uses. The run reproduces from the seeds alone.

Environment. The donor pools, the u_w decomposition, and the k-NN draws
are pure numpy/scipy and need NO populace-fit; the participation gate is a
``RegimeGatedQRF`` sign gate and DOES need populace-fit. Run the full gate
from the repository root with the PSID family files staged, using the
DEDICATED gate venv (populace-fit pins scikit-learn < 1.9, which the
repo's ``.venv`` violates; see populace #318):

    .venv-gate/bin/python scripts/run_gate1_candidate8.py
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

# Candidate 3's stage-1 decomposition machinery (pooled_autocovariances,
# fit_three_component, person_effects_pa) is applied byte-for-byte to the
# Z-PANEL (z = Phi^-1(rank)) rather than to the log-earnings residual panel
# (substitution 1). It is imported LAZILY inside build_donor_uw, not at
# module top level: the merged candidate-3 module imports candidate 2, which
# imports populace.fit at top level, so a module-level import here would
# pull populace-fit into every importer -- including the artifact-only
# consistency tests and the pure-numpy k-NN-draw test, which must run under
# the repo .venv without populace-fit (the baseline runner defers its own
# populace import for exactly this reason; see its header). Deferring the
# candidate-3 import keeps this module importable under the repo .venv; the
# tests that actually exercise the u_w decomposition importorskip populace.
# The rank machinery (cells, CellMarginal, fit_cell_marginals, anchors,
# anchor_u, age_bin) and the participation gate (fit_participation_gate,
# _gate_sign_draw) are IMPORTED from the merged candidate-5b runner so that
# the quantile/rank maps, the continuous anchor-rank rule, and the
# candidate-2 backward sign-gate draw are byte-for-byte candidate 5b's.
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

# Candidate 7's k-NN weighted single-record draw and its anchor-quintile
# map are imported byte-for-byte (the draw rule and the reported corner-mass
# quintiles are unchanged by both substitutions).
from run_gate1_candidate7 import (  # noqa: F401 (re-exported for tests)
    K_NEIGHBORS,
    N_ANCHOR_QUINTILES,
    W_ANCHOR,
    W_NEXT,
    W_NEXT2,
    _knn_draw,
    anchor_quintile,
)
from scipy.stats import norm

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_knn_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank_knn.v2"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4897723604"
)
#: The candidate-7 registration this run substitutes into (reported for
#: provenance; candidate 8 changes exactly two conditioning terms).
BASE_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4896132094"
)

# ---- Frozen constants of the candidate-8 registration ------------------
#: Substitution 2 attachment-conditioning scales, pinned as the RANGE
#: WIDTHS of the respective covariates (not tuned): age over the locked
#: 25-59 filter uses the registered width 40; the observed-period count
#: over the 13 biennial reference years 1998-2022 uses width 13 (the count
#: ranges 1..13 on the filtered panel). The registration fixes these two
#: numbers verbatim.
ZERO_ANCHOR_AGE_SCALE = 40.0
ZERO_ANCHOR_NOBS_SCALE = 13.0

#: Fixed integer codes for the generation RNG substream labels. Each label
#: seeds an independent generator via SeedSequence([seed, code]), so the
#: three streams are distinct and reproducible from the gate seed. These
#: codes are byte-for-byte candidate 7's (substream seeding is unchanged).
SUBSTREAM_CODES = {"gate": 1, "donor-draw": 2, "re-entry-draw": 3}


def _substream(seed: int, label: str) -> np.random.Generator:
    """A generation RNG for one fixed substream label off the gate seed."""
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


# --------------------------------------------------------------------------
# Substitution 1 -- donor permanent rank u_w from the z-panel decomposition
# --------------------------------------------------------------------------
def build_z_panel(
    train: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
) -> pd.DataFrame:
    """The z-panel: ``z = Phi^-1(rank)`` for every positive train row.

    For each train person-period with positive earnings, the rank is
    ``rhat`` at that observation's OWN cell (five-year age bin x period,
    the same map candidate 7 uses to form ``u_prev`` / ``u_next``), and
    ``z = Phi^-1(rank)``. Because ``rhat`` clamps to ``[0.001, 0.999]``,
    every ``z`` is finite. Returns a frame with ``person_id``, ``period``,
    ``r`` (the z value), the column layout candidate 3's stage-1
    autocovariance/shrinkage machinery consumes (it reads ``r`` as the
    residual it decomposes -- here the residual IS z).
    """
    tr = train[train.earnings > 0][
        ["person_id", "period", "earnings", "age"]
    ].copy()
    tr["bin"] = age_bin(tr["age"].to_numpy())
    earn = tr["earnings"].to_numpy(dtype=np.float64)
    bins = tr["bin"].to_numpy()
    periods = tr["period"].to_numpy()
    z = np.empty(len(tr), dtype=np.float64)
    for i in range(len(tr)):
        cm = marginals[(int(bins[i]), int(periods[i]))]
        z[i] = norm.ppf(cm.rank(float(earn[i])))
    return pd.DataFrame(
        {
            "person_id": tr["person_id"].to_numpy(),
            "period": periods,
            "r": z,
        }
    )


def build_donor_uw(
    train: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
) -> dict[str, Any]:
    """Per-person donor permanent rank ``u_w`` and the z-panel decomposition.

    Applies candidate 3's stage-1 machinery to the z-panel:

    * ``pooled_autocovariances`` -> the biennial-lag within-person
      autocovariances ``gamma_k`` (k = 0..5) of the pooled-mean-centred z,
    * ``fit_three_component`` -> the grid-rho NNLS decomposition
      ``(sigma2_perm, sigma2_trans, sigma2_noise, rho)``,
    * ``person_effects_pa`` -> ``what_i`` = the correlated-noise-shrunk
      within-person mean of the centred z (candidate 3's ``perm``).

    Then ``u_w = Phi(what_i / sigma_hat_w)`` with
    ``sigma_hat_w = sqrt(sigma2_perm)`` (the decomposition's permanent
    standard deviation). ``u_w`` is a per-PERSON quantity, attached to every
    donor record of that person across all pools. Persons with no positive
    observation (no z-panel rows) get ``what_i = 0`` from candidate 3's
    rule, so ``u_w = Phi(0) = 0.5``; such persons contribute no donor
    records (a donor record requires a positive pair), so their ``u_w`` is
    never read at match time -- it is carried only for completeness.

    Returns the ``{person_id: u_w}`` map, the fit block (reported, not
    gated), and the pooled z-mean used to centre.
    """
    # Imported here (not at module top) so the module stays importable under
    # the repo .venv: candidate 3 -> candidate 2 -> populace.fit at top
    # level, and only this fit path needs it. The three functions are
    # byte-for-byte candidate 3's; GAMMA_LAGS/RHO_GRID are its frozen grid.
    from run_gate1_candidate3 import (
        GAMMA_LAGS,
        RHO_GRID,
        fit_three_component,
        person_effects_pa,
        pooled_autocovariances,
    )

    z_panel = build_z_panel(train, marginals)
    gamma, gamma_counts, pooled_mean = pooled_autocovariances(z_panel)
    fit = fit_three_component(gamma)
    train_ids = train["person_id"].unique()
    perm = person_effects_pa(z_panel, fit, pooled_mean, train_ids)

    sigma_w = float(np.sqrt(fit["sigma2_perm"]))
    what = perm["perm"].to_numpy(dtype=np.float64)
    if sigma_w > 0:
        u_w_vals = norm.cdf(what / sigma_w)
    else:
        # Degenerate permanent variance (never seen on real data): every
        # person collapses to the prior median rank.
        u_w_vals = np.full(what.shape, 0.5, dtype=np.float64)
    pids = perm["person_id"].to_numpy()
    u_w_of_person = {
        int(p): float(u) for p, u in zip(pids, u_w_vals, strict=True)
    }

    n_pos = perm["n_pos"].to_numpy(dtype=np.float64)
    pos_mask = n_pos > 0
    u_w_pos = u_w_vals[pos_mask]

    def _pct(a: np.ndarray, p: int) -> float:
        return float(np.percentile(a, p)) if a.size else float("nan")

    fit_block = {
        "gamma": {str(k): float(gamma[k]) for k in GAMMA_LAGS},
        "gamma_pair_counts": {
            str(k): int(gamma_counts[k]) for k in GAMMA_LAGS
        },
        "pooled_z_mean": float(pooled_mean),
        "rho": float(fit["rho"]),
        "sigma2_perm": float(fit["sigma2_perm"]),
        "sigma2_trans": float(fit["sigma2_trans"]),
        "sigma2_noise": float(fit["sigma2_noise"]),
        "sigma_hat_w": sigma_w,
        "moment_sse": float(fit["sse"]),
        "implied_perm_share": float(fit["implied_perm_share"]),
        "rho_at_grid_boundary": bool(
            fit["rho"] <= RHO_GRID[0] or fit["rho"] >= RHO_GRID[-1]
        ),
        "n_z_panel_rows": int(len(z_panel)),
        "n_persons_with_positive_obs": int(pos_mask.sum()),
        "u_w_distribution_positive_obs": {
            "min": _pct(u_w_pos, 0),
            "p10": _pct(u_w_pos, 10),
            "p25": _pct(u_w_pos, 25),
            "median": _pct(u_w_pos, 50),
            "p75": _pct(u_w_pos, 75),
            "p90": _pct(u_w_pos, 90),
            "max": _pct(u_w_pos, 100),
            "mean": float(np.mean(u_w_pos)) if u_w_pos.size else float("nan"),
        },
        "rho_grid": list(RHO_GRID),
        "gamma_lags_biennial": list(GAMMA_LAGS),
    }
    return {
        "u_w_of_person": u_w_of_person,
        "fit": fit_block,
        "pooled_z_mean": float(pooled_mean),
        "sigma_hat_w": sigma_w,
    }


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
    persons. Byte-for-byte candidate 7's helper.
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
    u_w_of_person: dict[int, float],
) -> dict[str, Any]:
    """Build the pair, triple, and re-entry donor pools from the train split.

    Candidate 7's ``build_donor_pools`` VERBATIM in structure (pairs where
    both endpoints are positive, triples with a further positive period, and
    re-entry records where the later period is zero and the earlier is
    positive; each pinned in a stable ``(person_id, period_prev)`` sort that
    fixes the k-NN tie-break), with three additions required by the two
    substitutions -- each carried on every record and never itself a tuned
    dial:

    * ``u_w`` -- the donor person's permanent rank (substitution 1),
      attached via ``person_id`` (constant across a person's records).
    * ``age_prev`` -- the age at the record's EARLIER period ``period_prev``
      (the period whose rank ``u_prev`` is the drawn innovation), so a
      zero-anchor target's attachment term can read ``|Delta age at the
      step|`` (substitution 2).
    * ``n_obs`` -- the donor person's observed-period count in the FILTERED
      panel (substitution 2's second attachment covariate).
    * ``anchor_zero`` -- whether the donor person's own anchor earnings are
      zero, so the Q0-restricted pools can be sliced (substitution 2's
      donor-pool restriction).

    Returns presorted numpy arrays per pool plus the raw counts and the
    Q0-restricted pool sizes for the diagnostics.
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

    # Continuous anchor rank per pair (via the pair's person_id) -- reported
    # for the u_w-vs-u_A diagnostic; not the match key on the donor side.
    uA_map = anchor_u_by_person(
        marginals, all_anchor, pairs["person_id"].to_numpy()
    )
    uA_of_pair = np.array([uA_map[int(p)] for p in pid], dtype=np.float64)

    # Substitution-1 donor permanent rank per pair (constant per person).
    uW_of_pair = np.array(
        [u_w_of_person[int(p)] for p in pid], dtype=np.float64
    )

    # Substitution-2 attachment covariates: donor observed-period count in
    # the filtered panel (per person) and the anchor-zero flag (per person).
    n_obs_of_person = (
        train.groupby("person_id")["period"].size().astype(np.int64)
    )
    n_obs_map = {int(k): int(v) for k, v in n_obs_of_person.items()}
    nobs_of_pair = np.array([n_obs_map[int(p)] for p in pid], dtype=np.float64)
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
    age_prev_p = age_prev[kp_idx]
    w_p = w_prev[kp_idx]
    uA_p = uA_of_pair[kp_idx]
    uW_p = uW_of_pair[kp_idx]
    nobs_p = nobs_of_pair[kp_idx]
    azero_p = azero_of_pair[kp_idx]

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
    age_prev_r = age_prev[re_idx]
    w_r = w_prev[re_idx]
    uA_r = uA_of_pair[re_idx]
    uW_r = uW_of_pair[re_idx]
    nobs_r = nobs_of_pair[re_idx]
    azero_r = azero_of_pair[re_idx]

    # --- Pin each pool in a stable (person_id, period_prev) order so the
    # k-NN tie-break is fully determined by record order (candidate 7's key).
    pair_order = np.lexsort((period_prev_p, pid_p))
    pairs_pool = {
        "u_prev": u_prev_p[pair_order],
        "u_next": u_next_p[pair_order],
        "u_A": uA_p[pair_order],
        "u_w": uW_p[pair_order],
        "age_prev": age_prev_p[pair_order],
        "n_obs": nobs_p[pair_order],
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
        "age_prev": age_prev_p[tri_idx][tri_order],
        "n_obs": nobs_p[tri_idx][tri_order],
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
        "age_prev": age_prev_r[re_order],
        "n_obs": nobs_r[re_order],
        "anchor_zero": azero_r[re_order],
        "weight": w_r[re_order],
        "person_id": pid_r[re_order],
        "period_prev": period_prev_r[re_order],
    }

    # Q0-restricted pools (substitution 2 donor restriction): the subset of
    # each pool whose donor person's own anchor earnings are zero.
    def _restrict(pool: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        mask = pool["anchor_zero"]
        return {k: v[mask] for k, v in pool.items()}

    pairs_q0 = _restrict(pairs_pool)
    triples_q0 = _restrict(triples_pool)
    reentry_q0 = _restrict(reentry_pool)

    return {
        "pairs": pairs_pool,
        "triples": triples_pool,
        "reentry": reentry_pool,
        "pairs_q0": pairs_q0,
        "triples_q0": triples_q0,
        "reentry_q0": reentry_q0,
        "n_pairs": int(both_pos.sum()),
        "n_triples": int(tri_idx.size),
        "n_reentry": int(reenter.sum()),
        "n_pairs_q0": int(pairs_q0["u_prev"].size),
        "n_triples_q0": int(triples_q0["u_prev"].size),
        "n_reentry_q0": int(reentry_q0["u_prev"].size),
        "n_train_pairs": int(len(pairs)),
    }


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

    Candidate 7's backward chain VERBATIM except the third distance term,
    which the two registered substitutions rewrite BY TARGET TYPE:

    * **Positive-anchor target** (``u_A != p0/2`` group): the third term is
      ``0.25 |u_w(donor) - u_A(target)|`` on transitions (the donor's ``u_w``
      replacing candidate 7's donor ``u_A``) and ``|u_w(donor) -
      u_A(target)|`` on re-entry (candidate 7's re-entry carried the anchor
      term bare, weight 1.0; substitution 1 swaps only its donor argument).
      The full donor pools are used.
    * **Zero-anchor target** (``u_A = p0/2`` identically): the third term is
      ``0.25 (|Delta age|/40 + |Delta n_obs|/13)`` on transitions and
      ``|Delta age|/40 + |Delta n_obs|/13`` on re-entry (matching candidate
      7's bare re-entry weight), where ``Delta age`` is the target's age at
      the generated (rank-j) period minus the donor's age at its earlier
      period, and ``Delta n_obs`` is the target's observed-period count
      minus the donor's. The Q0-RESTRICTED donor pools (donor anchor
      earnings zero) are used.

    The first two distance terms (``|u_next - v1|`` at weight 1.0 and
    ``0.5 |u_next2 - v2|`` on triples) are unchanged for both target types.
    The drawn ``u_prev`` is the selected donor's ``u_prev`` exactly; then
    earnings = ``Qhat_pos`` of the period's cell at ``u_prev``. Returns
    ``(candidate, diagnostics)`` where the candidate holds exactly the
    holdout persons on exactly their observed periods (only earnings
    generated; anchor kept).
    """
    pairs_pool = pools["pairs"]
    triples_pool = pools["triples"]
    reentry_pool = pools["reentry"]
    pairs_q0 = pools["pairs_q0"]
    triples_q0 = pools["triples_q0"]
    reentry_q0 = pools["reentry_q0"]

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
    depths = hp["depth"].to_numpy()
    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    n_rows = len(hp)
    pos_by_key = {
        (int(pid), int(r)): i
        for i, (pid, r) in enumerate(zip(pids, ranks, strict=True))
    }
    max_depth = int(hp["depth"].max()) if n_rows else 0

    # Continuous anchor rank per holdout person (frozen stage-2 rule) and
    # the zero-anchor flag (u_A == p0/2 identically <=> anchor earnings 0).
    holdout_ids = np.sort(hp["person_id"].unique())
    uA_map = anchor_u_by_person(marginals, all_anchor, holdout_ids)
    anchor_rank_of_person = {
        int(p): float(uA_map[int(p)]) for p in holdout_ids
    }
    anchor_rank_vals = np.array(
        [anchor_rank_of_person[int(p)] for p in holdout_ids], dtype=np.float64
    )
    # Zero-anchor targets: those whose anchor earnings are zero (the group
    # the frozen anchor rule sends to u_A = p0/2). Read the flag from the
    # anchor rows directly, so it is exact (not a float comparison on u_A).
    ha_all = all_anchor[
        all_anchor.person_id.isin(set(int(x) for x in holdout_ids))
    ]
    zero_anchor_person = {
        int(r.person_id): bool(float(r.earnings) == 0.0)
        for r in ha_all.itertuples(index=False)
    }
    # Target observed-period count = the person's depth (observed periods in
    # the filtered panel; the holdout carries the person's full support).
    depth_of_person = {
        int(p): int(d) for p, d in zip(pids, depths, strict=True)
    }

    rng_gate = _substream(seed, "gate")
    rng_donor = _substream(seed, "donor-draw")
    rng_reentry = _substream(seed, "re-entry-draw")

    # Presorted donor arrays (pinned tie-break order) -- full pools.
    tri_u_next = triples_pool["u_next"]
    tri_u_next2 = triples_pool["u_next2"]
    tri_u_w = triples_pool["u_w"]
    tri_w = triples_pool["weight"]
    tri_u_prev = triples_pool["u_prev"]
    pair_u_next = pairs_pool["u_next"]
    pair_u_w = pairs_pool["u_w"]
    pair_w = pairs_pool["weight"]
    pair_u_prev = pairs_pool["u_prev"]
    re_u_w = reentry_pool["u_w"]
    re_w = reentry_pool["weight"]
    re_u_prev = reentry_pool["u_prev"]

    # Diagnostics accumulators.
    n_clamped = 0
    n_positive_gen = 0
    n_triple_draws = 0
    n_pair_draws = 0
    n_reentry_draws = 0
    n_triple_draws_q0 = 0
    n_pair_draws_q0 = 0
    n_reentry_draws_q0 = 0
    neighbor_dists: list[float] = []
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
            # Per-query anchor rank a, zero-anchor flag, target attachment
            # covariates (age at the generated period, observed-period
            # count), all in the fixed step order.
            a_local = np.array(
                [
                    anchor_rank_of_person[int(step_pids[li])]
                    for li in pos_local
                ],
                dtype=np.float64,
            )
            za_local = np.array(
                [zero_anchor_person[int(step_pids[li])] for li in pos_local],
                dtype=bool,
            )
            age_step_local = ages[positions][pos_local]
            nobs_local = np.array(
                [depth_of_person[int(step_pids[li])] for li in pos_local],
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

            # Convenience index sets over the local positives, split by the
            # branch (re-entry / triple / pair) AND the target type
            # (positive-anchor uses substitution 1 + full pools; zero-anchor
            # uses substitution 2 + Q0-restricted pools).
            pa = ~za_local  # positive-anchor mask over pos_local

            # ---- Branch A: next zero -> re-entry ----------------------------
            rp = np.nonzero(~next_is_pos)[0]
            # A1: positive-anchor re-entry -> substitution 1, full re-entry
            # pool, distance = |u_w(donor) - a| (bare, matching c7).
            rp_pa = rp[pa[rp]]
            if rp_pa.size:
                a_r = a_local[rp_pa]
                dist = np.abs(re_u_w[None, :] - a_r[:, None])
                u_dr = rng_reentry.random(rp_pa.size)
                drawn, kth = _knn_draw(dist, re_w, re_u_prev, u_dr)
                u_prev_local[rp_pa] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_reentry_draws += int(rp_pa.size)
            # A2: zero-anchor re-entry -> substitution 2, Q0 re-entry pool,
            # distance = |Delta age|/40 + |Delta n_obs|/13 (bare).
            rp_za = rp[~pa[rp]]
            if rp_za.size:
                age_q = age_step_local[rp_za]
                nobs_q = nobs_local[rp_za]
                dist = (
                    np.abs(reentry_q0["age_prev"][None, :] - age_q[:, None])
                    / ZERO_ANCHOR_AGE_SCALE
                    + np.abs(reentry_q0["n_obs"][None, :] - nobs_q[:, None])
                    / ZERO_ANCHOR_NOBS_SCALE
                )
                u_dr = rng_reentry.random(rp_za.size)
                drawn, kth = _knn_draw(
                    dist,
                    reentry_q0["weight"],
                    reentry_q0["u_prev"],
                    u_dr,
                )
                u_prev_local[rp_za] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_reentry_draws += int(rp_za.size)
                n_reentry_draws_q0 += int(rp_za.size)

            # ---- Branch B: next positive, v2 exists -> triple ---------------
            tp = np.nonzero(next_is_pos & has_v2)[0]
            # B1: positive-anchor triple -> substitution 1, full triple pool.
            tp_pa = tp[pa[tp]]
            if tp_pa.size:
                v1_t = v1[tp_pa]
                v2_t = v2[tp_pa]
                a_t = a_local[tp_pa]
                dist = (
                    W_NEXT * np.abs(tri_u_next[None, :] - v1_t[:, None])
                    + W_NEXT2 * np.abs(tri_u_next2[None, :] - v2_t[:, None])
                    + W_ANCHOR * np.abs(tri_u_w[None, :] - a_t[:, None])
                )
                u_dt = rng_donor.random(tp_pa.size)
                drawn, kth = _knn_draw(dist, tri_w, tri_u_prev, u_dt)
                u_prev_local[tp_pa] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_triple_draws += int(tp_pa.size)
            # B2: zero-anchor triple -> substitution 2, Q0 triple pool.
            tp_za = tp[~pa[tp]]
            if tp_za.size:
                v1_t = v1[tp_za]
                v2_t = v2[tp_za]
                age_q = age_step_local[tp_za]
                nobs_q = nobs_local[tp_za]
                dist = (
                    W_NEXT
                    * np.abs(triples_q0["u_next"][None, :] - v1_t[:, None])
                    + W_NEXT2
                    * np.abs(triples_q0["u_next2"][None, :] - v2_t[:, None])
                    + W_ANCHOR
                    * (
                        np.abs(
                            triples_q0["age_prev"][None, :] - age_q[:, None]
                        )
                        / ZERO_ANCHOR_AGE_SCALE
                        + np.abs(
                            triples_q0["n_obs"][None, :] - nobs_q[:, None]
                        )
                        / ZERO_ANCHOR_NOBS_SCALE
                    )
                )
                u_dt = rng_donor.random(tp_za.size)
                drawn, kth = _knn_draw(
                    dist,
                    triples_q0["weight"],
                    triples_q0["u_prev"],
                    u_dt,
                )
                u_prev_local[tp_za] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_triple_draws += int(tp_za.size)
                n_triple_draws_q0 += int(tp_za.size)

            # ---- Branch C: next positive, no v2 -> pair ---------------------
            pp = np.nonzero(next_is_pos & ~has_v2)[0]
            # C1: positive-anchor pair -> substitution 1, full pair pool.
            pp_pa = pp[pa[pp]]
            if pp_pa.size:
                v1_p = v1[pp_pa]
                a_p = a_local[pp_pa]
                dist = W_NEXT * np.abs(
                    pair_u_next[None, :] - v1_p[:, None]
                ) + W_ANCHOR * np.abs(pair_u_w[None, :] - a_p[:, None])
                u_dp = rng_donor.random(pp_pa.size)
                drawn, kth = _knn_draw(dist, pair_w, pair_u_prev, u_dp)
                u_prev_local[pp_pa] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_pair_draws += int(pp_pa.size)
            # C2: zero-anchor pair -> substitution 2, Q0 pair pool.
            pp_za = pp[~pa[pp]]
            if pp_za.size:
                v1_p = v1[pp_za]
                age_q = age_step_local[pp_za]
                nobs_q = nobs_local[pp_za]
                dist = W_NEXT * np.abs(
                    pairs_q0["u_next"][None, :] - v1_p[:, None]
                ) + W_ANCHOR * (
                    np.abs(pairs_q0["age_prev"][None, :] - age_q[:, None])
                    / ZERO_ANCHOR_AGE_SCALE
                    + np.abs(pairs_q0["n_obs"][None, :] - nobs_q[:, None])
                    / ZERO_ANCHOR_NOBS_SCALE
                )
                u_dp = rng_donor.random(pp_za.size)
                drawn, kth = _knn_draw(
                    dist,
                    pairs_q0["weight"],
                    pairs_q0["u_prev"],
                    u_dp,
                )
                u_prev_local[pp_za] = drawn
                neighbor_dists.extend(float(x) for x in kth)
                n_pair_draws += int(pp_za.size)
                n_pair_draws_q0 += int(pp_za.size)

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
        sum(1 for p in holdout_ids if zero_anchor_person[int(p)])
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
        n_triple_draws_q0,
        n_pair_draws_q0,
        n_reentry_draws_q0,
        np.asarray(neighbor_dists, dtype=np.float64),
        corner_bottom,
        corner_top,
        corner_total,
        int(len(holdout_ids)),
        n_zero_anchor_holdout,
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
    n_triple_draws_q0: int,
    n_pair_draws_q0: int,
    n_reentry_draws_q0: int,
    neighbor_dists: np.ndarray,
    corner_bottom: np.ndarray,
    corner_top: np.ndarray,
    corner_total: np.ndarray,
    n_holdout_persons: int,
    n_zero_anchor_holdout: int,
) -> dict[str, Any]:
    """Assemble the reported-not-gated diagnostics for one seed.

    Carries candidate 7's named diagnostics (neighbor-distance distribution,
    triple-vs-pair usage share, donor-record reuse, drawn corner mass by
    anchor quintile, clamped-rank share, anchor-rank histogram, per-cell
    train positive-count summary) PLUS the candidate-8 additions: the split
    of draws between the full (positive-anchor) and the Q0-restricted
    (zero-anchor) pools, and the count of zero-anchor holdout persons. The
    u_w distributions and the Q0 pool sizes are carried at the seed level in
    :func:`run_seed` (they are properties of the train fit, not the
    generation pass).
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

    n_pos_total = int(n_pair_draws + n_triple_draws)
    n_pos_q0 = int(n_pair_draws_q0 + n_triple_draws_q0)
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
        "zero_anchor_draw_split": {
            "n_pair_draws_q0": int(n_pair_draws_q0),
            "n_triple_draws_q0": int(n_triple_draws_q0),
            "n_reentry_draws_q0": int(n_reentry_draws_q0),
            "n_positive_draws_q0": n_pos_q0,
            "n_positive_draws_total": n_pos_total,
            "q0_share_of_positive_draws": (
                float(n_pos_q0 / n_pos_total) if n_pos_total else 0.0
            ),
            "note": (
                "draws routed through the Q0-restricted (zero-anchor-donor) "
                "pools with the substitution-2 attachment distance, vs all "
                "positive draws; the remainder used the full pools with the "
                "substitution-1 u_w distance"
            ),
        },
        "donor_reuse": {
            "n_pair_records": int(pools["n_pairs"]),
            "n_triple_records": int(pools["n_triples"]),
            "n_reentry_records": int(pools["n_reentry"]),
            "n_pair_records_q0": int(pools["n_pairs_q0"]),
            "n_triple_records_q0": int(pools["n_triples_q0"]),
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
# Downstream benefit-space block (REPORTED, not gated) -- pinned PR-56
# functional applied to candidate 8
# --------------------------------------------------------------------------
def measure_benefit_space(
    seed: int,
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    all_anchor: pd.DataFrame,
    params: Any,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    """PIA-proxy gap block for candidate 8, via the pinned PR-56 functional.

    REPORTED, NOT GATED. Pushes both the real holdout histories and the
    candidate-8 generated histories through
    :func:`build_downstream_relevance.panel_pia_proxy` (the statute-shaped
    PIA proxy, byte-for-byte the merged pull-request-56 functional), aligns
    the two on ``person_id`` (same person set, same rows, only earnings
    differ), and reports the weighted distribution-gap block (mean/median %,
    decile % gaps, weighted Gini difference, weighted KS distance), the
    weighted person-level block, and the by-anchor-quintile concentration --
    with Q0 (zero anchor earnings) called out, the subgroup pull request 56
    flagged at +9.3% mean on candidate 7. ``cutpoints`` are the seed-stable
    full-panel positive-anchor quartile edges.
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
    benefit_params: Any,
    benefit_cutpoints: np.ndarray | None,
    verbose: bool,
) -> dict[str, Any]:
    """Fit, build pools, generate, and score candidate 8 for one seed.

    When ``benefit_params`` is provided the reported-not-gated benefit-space
    block is measured for the seed (the pinned PR-56 PIA-proxy functional
    applied to candidate 8's holdout vs the real holdout); when it is
    ``None`` the block is skipped (so the gate itself never depends on the
    SSA oracle being present).
    """
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Stage 1: per-cell marginals on the train complement.
    marginals = fit_cell_marginals(train)

    # Substitution 1: donor permanent rank u_w from the z-panel (candidate
    # 3's stage-1 decomposition applied to z = Phi^-1(rank)).
    uw = build_donor_uw(train, marginals)

    # Stage 3: donor pools (pairs, triples, re-entry) with u_w, the
    # attachment covariates, and the anchor-zero flag; plus the Q0-restricted
    # pools (substitution 2 donor restriction).
    pools = build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )

    # Stage 4/5: participation gate (train complement) + backward k-NN chain
    # with the two substitutions.
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

    # --- benefit-space block (reported, not gated) ---
    benefit_space: dict[str, Any] | None = None
    if benefit_params is not None and benefit_cutpoints is not None:
        benefit_space = measure_benefit_space(
            seed,
            holdout,
            candidate,
            all_anchor,
            benefit_params,
            benefit_cutpoints,
        )

    result = {
        "seed": seed,
        "n_persons": int(holdout.person_id.nunique()),
        "n_person_periods": int(len(holdout)),
        "n_train_persons": int(train.person_id.nunique()),
        "n_train_pairs": int(len(pairs)),
        "n_windows": n_windows,
        "regimes": {"participation_gate": fitted.regimes()},
        "uw_fit": uw["fit"],
        "pools": {
            "n_pairs": int(pools["n_pairs"]),
            "n_triples": int(pools["n_triples"]),
            "n_reentry": int(pools["n_reentry"]),
            "n_pairs_q0": int(pools["n_pairs_q0"]),
            "n_triples_q0": int(pools["n_triples_q0"]),
            "n_reentry_q0": int(pools["n_reentry_q0"]),
        },
        "generation_diagnostics": diagnostics,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if benefit_space is not None:
        result["benefit_space"] = benefit_space
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
                f" bs_mean%={mean_pct:+.2f} " f"Q0_mean%={q0mean:+.2f}"
                if (mean_pct is not None and q0mean is not None)
                else ""
            )
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"mob_diag={battery_values['mobility_diagonal']:.3f} "
            f"ac10={battery_values['autocorr_log_10yr']:.3f} "
            f"tri_share={d['triple_pair_usage']['triple_share_of_positive']:.3f} "
            f"q0_share={d['zero_anchor_draw_split']['q0_share_of_positive_draws']:.3f} "
            f"clamp={d['clamped_rank_share']['share']:.3f}{bs} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def _load_benefit_oracle() -> tuple[Any, np.ndarray | None]:
    """Load the SSA oracle params + full-panel quartile cuts, or (None, None).

    The benefit-space block is reported-not-gated; if the SSA oracle cannot
    load (e.g. ``POPULACE_DYNAMICS_PE_US_DIR`` unset off the gate machine),
    the run proceeds with the block skipped rather than failing the gate.
    """
    try:
        import build_downstream_relevance as ds

        from populace_dynamics.ss.params import load_ssa_parameters

        params = load_ssa_parameters()
        panel = load_filtered_panel()
        all_anchor = anchor_rows(panel)
        cuts = ds.anchor_quintile_cutpoints(all_anchor)
        return params, cuts
    except Exception as exc:  # noqa: BLE001 (reported-not-gated best effort)
        print(f"benefit-space oracle unavailable ({exc!r}); block skipped")
        return None, None


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-8 run."""
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

    # Reported-not-gated benefit-space oracle (loaded once; skipped if
    # unavailable). The quartile cuts are the seed-stable full-panel
    # positive-anchor edges the PR-56 functional uses.
    benefit_params, benefit_cutpoints = _load_benefit_oracle()
    if verbose and benefit_params is not None:
        print(
            "benefit-space oracle: pe_us_revision="
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
                benefit_params,
                benefit_cutpoints,
                verbose,
            )
        )

    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    gate_pass = geometry_gate_pass and battery_gate_pass

    benefit_pooled = _pool_benefit_space(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_rank_knn_v2",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "base_registration": BASE_REGISTRATION,
        "substitutions": (
            "candidate 7's machinery verbatim with exactly two registered "
            "substitutions: (1) donor-side permanent-rank u_w replaces the "
            "donor anchor rank in the third distance term (same 0.25 "
            "weight); (2) age/observed-span attachment conditioning with "
            "zero-anchor-donor pools for zero-anchor targets"
        ),
        "description": (
            "Gate-1 candidate 8: permanent-rank donor matching with "
            "attachment-aware zero conditioning. Candidate 7's k-NN "
            "conditional rank bootstrap (empirical per-cell quantile "
            "marginals supply the magnitude; a continuous nonparametric "
            "transition law supplies the dynamics) with EXACTLY two "
            "registered substitutions, both conditioning refinements with no "
            "new tuned constants. Substitution 1: every train donor record "
            "carries u_w = Phi(what/sigma_hat_w), the donor person's "
            "correlated-noise-shrunk permanent rank from candidate 3's "
            "stage-1 decomposition APPLIED TO THE Z-PANEL (z = Phi^-1(rank) "
            "of positive observations); the k-NN third term becomes "
            "|u_w(donor) - u_A(target)| at candidate 7's 0.25 weight, "
            "upgrading the donor side from a single noisy anchor draw to the "
            "shrunk full-career permanent estimate. Substitution 2: for "
            "zero-anchor holdout persons (u_A = p0/2 identically) every k-NN "
            "distance replaces the third term with |Delta age|/40 + "
            "|Delta n_observed_periods|/13 (scales pinned as the range "
            "widths, not tuned) and the donor pool is restricted to train "
            "records whose person's own anchor earnings are zero. Everything "
            "else -- donor pools, k=25, the 1/0.5 lag weights, the weighted "
            "single-record draw, no smoothing/jitter, the re-entry pools, "
            "the regime gate, the rank machinery, the gap rule, the "
            "substream seeding -- is byte-identical to candidate 7. "
            "Registered frozen before the run in issue #42 (see "
            "spec_registration). Candidate scored against the held-out PSID "
            "family earnings panel geometry (two locked views) and the "
            "locked moment battery, per the locked seed-level conjunction in "
            "gates.yaml (pull request 39). Protocol machinery imported "
            "byte-for-byte from the baseline runner (pull request 40); rank "
            "machinery and participation gate from candidate 5b (pull "
            "request 52); the k-NN draw and anchor quintiles from candidate "
            "7 (pull request 55); the u_w decomposition from candidate 3 "
            "(pull request 44); the reported benefit-space functional from "
            "pull request 56. The artifact additionally REPORTS the proposed "
            "benefit-space block (PIA-proxy gaps including Q0) so this run "
            "carries evidence for both the locked and the proposed standard."
        ),
        "model": {
            "class": (
                "k-NN conditional rank bootstrap with anchored two-step "
                "memory, permanent-rank donor matching, and attachment-aware "
                "zero conditioning (quantile marginal + continuous empirical "
                "conditional draws)"
            ),
            "stochastic": True,
            "populace_fit_used": True,
            "populace_fit_scope": (
                "participation gate only (RegimeGatedQRF sign gate); the "
                "donor pools, the u_w decomposition, and the k-NN draws use "
                "pure numpy/scipy"
            ),
            "calibration": (
                "none (zero free parameters; the bootstrap has no rescaling "
                "freedom; the u_w decomposition is candidate 3's "
                "deterministic grid-rho NNLS on TRAIN only)"
            ),
            "substitution_1_donor_permanent_rank": {
                "what": (
                    "u_w = Phi(what / sigma_hat_w) per donor person, where "
                    "what is the correlated-noise-shrunk within-person mean "
                    "of z = Phi^-1(rank) over the person's positive "
                    "observations and sigma_hat_w = sqrt(sigma2_perm)"
                ),
                "z_panel": (
                    "z = Phi^-1(rhat) where rhat is the observation's rank at "
                    "its own (age-bin x period) cell -- the same map "
                    "candidate 7 forms u_prev/u_next with; ranks clamp to "
                    "[0.001, 0.999] so z is finite"
                ),
                "decomposition": (
                    "candidate 3's stage-1 machinery on the z-panel: pooled "
                    "within-person autocovariances gamma_k (biennial lags "
                    "0..5) of the pooled-mean-centred z; grid-rho NNLS "
                    "three-component fit (permanent + AR(1) transitory + "
                    "noise); correlated-noise shrunk person effect what_i = "
                    "w_i * mean_i(centred z)"
                ),
                "distance_change": (
                    "the k-NN third term |u_A(donor) - u_A(target)| becomes "
                    "|u_w(donor) - u_A(target)| at the SAME 0.25 weight "
                    "(transitions) / bare weight (re-entry); the TARGET side "
                    "stays u_A"
                ),
                "applies_to": "positive-anchor targets (u_A != p0/2)",
                "rho_grid": per_seed[0]["uw_fit"]["rho_grid"],
                "gamma_lags_biennial": per_seed[0]["uw_fit"][
                    "gamma_lags_biennial"
                ],
            },
            "substitution_2_zero_anchor_conditioning": {
                "trigger": (
                    "holdout persons with zero anchor earnings (u_A = p0/2 "
                    "identically, so the anchor term is uninformative)"
                ),
                "distance_third_term": (
                    "|Delta age at the step|/40 + |Delta n_observed_periods|"
                    "/13 replaces the anchor term on ALL their distances "
                    "(transitions and re-entry)"
                ),
                "delta_age": (
                    "target age at the generated (rank-j) period minus the "
                    "donor's age at its earlier period (period_prev, whose "
                    "rank u_prev is the drawn innovation)"
                ),
                "delta_n_observed_periods": (
                    "target observed-period count in the filtered panel minus "
                    "the donor person's observed-period count"
                ),
                "scales": {
                    "age": ZERO_ANCHOR_AGE_SCALE,
                    "n_observed_periods": ZERO_ANCHOR_NOBS_SCALE,
                    "note": (
                        "pinned as the range widths of the respective "
                        "covariates (age over the 25-59 filter; the "
                        "observed-period count over the 13 biennial reference "
                        "years, count range 1..13); NOT tuned"
                    ),
                },
                "donor_pool_restriction": (
                    "train records whose person's own anchor earnings are "
                    "zero (the Q0-restricted pair/triple/re-entry pools)"
                ),
                "weighting": (
                    "the substitution-2 attachment term keeps candidate 7's "
                    "third-term weight (0.25 on transitions; bare on "
                    "re-entry, where c7's only term was bare); the first two "
                    "terms (1.0|u_next-v1|, 0.5|u_next2-v2|) are unchanged"
                ),
            },
            "donor_pools": {
                "pairs": (
                    "train backward-adjacent pairs among positives: "
                    "(u_prev, u_next); each carries u_A, u_w, age_prev, "
                    "n_obs, anchor_zero and the earlier-period weight"
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
                "q0_restricted": (
                    "the subset of each pool whose donor person's own anchor "
                    "earnings are zero, used only for zero-anchor targets"
                ),
                "tie_break_order": (
                    "records pinned in a stable (person_id, period_prev) sort "
                    "that fixes the k-NN tie-break (byte-for-byte candidate 7)"
                ),
            },
            "knn": {
                "k": K_NEIGHBORS,
                "distance_triples_positive_anchor": (
                    "|u_next - v1| + 0.5|u_next2 - v2| + 0.25|u_w - a|"
                ),
                "distance_pairs_positive_anchor": "|u_next - v1| + 0.25|u_w - a|",
                "distance_reentry_positive_anchor": "|u_w - a|",
                "distance_triples_zero_anchor": (
                    "|u_next - v1| + 0.5|u_next2 - v2| + "
                    "0.25(|d_age|/40 + |d_nobs|/13)"
                ),
                "distance_pairs_zero_anchor": (
                    "|u_next - v1| + 0.25(|d_age|/40 + |d_nobs|/13)"
                ),
                "distance_reentry_zero_anchor": "|d_age|/40 + |d_nobs|/13",
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
                    "candidate-2 backward regime gate (RegimeGatedQRF sign "
                    "gate) on (next generated level, current age), trained on "
                    "the 80% complement, populace-fit defaults"
                ),
                "rng": (
                    "distinct fixed-label substreams of the gate seed: gate, "
                    "donor-draw, re-entry-draw (byte-for-byte candidate 7)"
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
                "distribution, triple-vs-pair usage share, the zero-anchor "
                "(Q0) draw split, donor-record reuse (full and "
                "Q0-restricted), drawn corner mass by anchor quintile, the "
                "clamped-rank share, and the u_w decomposition + "
                "distribution. None enters the geometry or battery pass/fail; "
                "the gate rule names only those two families."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "n_pairs": s["pools"]["n_pairs"],
                    "n_triples": s["pools"]["n_triples"],
                    "n_reentry": s["pools"]["n_reentry"],
                    "n_pairs_q0": s["pools"]["n_pairs_q0"],
                    "n_triples_q0": s["pools"]["n_triples_q0"],
                    "n_reentry_q0": s["pools"]["n_reentry_q0"],
                    "uw_fit": s["uw_fit"],
                    "neighbor_distance_distribution": s[
                        "generation_diagnostics"
                    ]["neighbor_distance_distribution"],
                    "triple_pair_usage": s["generation_diagnostics"][
                        "triple_pair_usage"
                    ],
                    "zero_anchor_draw_split": s["generation_diagnostics"][
                        "zero_anchor_draw_split"
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
        "benefit_space_reported_not_gated": benefit_pooled,
        "verdict": {
            "n_seeds": len(SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "gate_1_pass": bool(gate_pass),
            "rule": ">=4/5 seeds geometry AND >=4/5 seeds battery",
        },
        "revision_pins": _revision_pins(benefit_params),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5)"
        )
    return artifact


def _pool_benefit_space(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the reported-not-gated benefit-space block across seeds.

    Mirrors pull request 56's pooling (mean/sd/min/max across seeds of each
    scalar distribution-gap and person-level statistic, and the mean across
    seeds of the by-quintile concentration, Q0 called out). Returns an empty
    marker when the benefit-space block is absent (SSA oracle unavailable).
    """
    rows = [s["benefit_space"] for s in per_seed if "benefit_space" in s]
    if not rows:
        return {
            "available": False,
            "note": (
                "benefit-space block not measured (SSA oracle unavailable at "
                "run time; set POPULACE_DYNAMICS_PE_US_DIR and rerun to "
                "populate it -- reported-not-gated, so the gate verdict is "
                "unaffected)"
            ),
        }

    def _summ(vals: list[float]) -> dict[str, float]:
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

    # By-quintile pooling (mean across seeds where the quintile is present),
    # Q0 called out (the subgroup PR-56 flagged at +9.3% mean on c7).
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
        "reported_not_gated": True,
        "functional": (
            "pinned pull-request-56 statute-shaped PIA proxy "
            "(build_downstream_relevance.panel_pia_proxy), applied "
            "identically to candidate 8's holdout and the real holdout"
        ),
        "success_criterion_pct": 5.0,
        "distribution": dist,
        "person_level": person_level,
        "by_anchor_quintile": by_q,
        "candidate7_reference": {
            "note": (
                "candidate 7's committed downstream numbers (pull request "
                "56, runs/downstream_relevance_c7_v1.json) for comparison"
            ),
            "mean_pct_diff": 1.8477489952384396,
            "median_pct_diff": 1.0171410947686295,
            "ks_distance": 0.02535050245252706,
            "Q0_mean_pct_diff": 9.295292319680367,
            "Q0_median_pct_diff": 22.915959347981033,
            "Q0_ks_distance": 0.09679229074277092,
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
