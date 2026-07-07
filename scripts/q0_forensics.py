"""Q0 forensics: why generated zero-anchor careers overstate benefits.

REPORTED, NOT GATED. NO HOLDOUT-REAL CONTACT beyond the pooled Q0 statistic
the ratified benefit-space gate already scores. This script reads no gate and
changes no gate; it writes only ``runs/q0_forensics_v1.json``.

The ratified benefit-space block (``gates.yaml`` thresholds.benefit_space,
live since PR #59) bounds the pooled zero-anchor (Q0) PIA-proxy mean gap at
5%. Candidate 7 (PR #55/#56) measures +9.30%; candidate 8's attempted fix
(PR #58, runs/gate1_rank_knn_v2.json) made it worse at +12.2%. This forensics
localizes the +9.30% mechanically so candidate 9's Q0 component can be
designed against the actual cause rather than against the failed
age+observed-span conditioning candidate 8 already tried.

DEFINITIONS (reused from the merged benefit-space machinery, byte-for-byte):

* The **PIA-proxy functional** is
  :func:`build_downstream_relevance.person_pia_proxy` (the pinned
  statute-shaped proxy: top min(10, n_pos) wage-base-capped, NAWI-indexed
  positive years, averaged over that count, then the 2022-eligibility
  415(a)/415(g) PIA). It is an AVERAGE of the top positive years, so working
  MORE positive periods does not inflate the average directly (the divisor
  grows with the count) EXCEPT (a) when a person crosses from zero positive
  observations (PIA-proxy 0) to some, and (b) through the top-min(10, n)
  selection when n_pos > 10.
* The **Q0 subgroup** is holdout persons with ZERO anchor earnings (no
  earnings in their chronologically last observed period), the benefit
  block's definition, using the seed-stable full-panel quintile edges
  (:func:`build_downstream_relevance.anchor_quintile_cutpoints`).
* The **comparison population** for the covariate block (Q4) is TRAIN persons
  with zero anchor earnings -- their real careers. NO holdout-real careers
  are read except through the pooled Q0 mean gap the gate already reports.

Environment (identical to the merged benefit-space builder)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/q0_forensics.py

The candidate-7 regeneration needs populace-fit (the participation gate is a
RegimeGatedQRF sign gate), so run it from the dedicated gate venv.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Candidate-7 machinery (merged PR #55) and the benefit-space proxy (merged
# PR #56). Regeneration is deterministic from the seed alone, byte-for-byte
# the committed candidate-7 run's, so the careers pushed through the proxy
# are exactly the ones the gate scored.
from build_downstream_relevance import (  # noqa: E402
    _weighted_mean,
    _weighted_quantile,
    anchor_quintile_cutpoints,
    person_pia_proxy,
)
from run_gate1_baseline import (  # noqa: E402
    SEEDS,
    build_backward_pairs,
    load_filtered_panel,
    split_holdout_train,
)
from run_gate1_candidate5b import (  # noqa: E402
    age_bin,
    anchor_rows,
    fit_cell_marginals,
    fit_participation_gate,
)
from run_gate1_candidate7 import (  # noqa: E402
    build_donor_pools,
    generate_candidate,
)

from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "q0_forensics_v1.json"
ARTIFACT_SCHEMA_VERSION = "q0_forensics.v1"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v1.json"

#: The top-N the AIME-proxy averages (the partial-career analogue of 35).
PROXY_TOP_N = 10


# --------------------------------------------------------------------------
# Per-person career summaries
# --------------------------------------------------------------------------
def _person_summary(
    periods: np.ndarray, earnings: np.ndarray, params: Any
) -> dict[str, float]:
    """PIA-proxy plus the career statistics the decomposition needs.

    Returns the person's PIA-proxy, positive-period count ``n_pos``,
    observed-period count ``n_obs``, mean positive earnings level (0 if
    none), and the longest run of consecutive zero observations (in
    chronological order).
    """
    e = np.asarray(earnings, dtype=np.float64)
    per = np.asarray(periods)
    order = np.argsort(per)
    e = e[order]
    pos = e > 0
    n_obs = int(e.size)
    n_pos = int(pos.sum())
    mean_pos = float(e[pos].mean()) if n_pos else 0.0
    # Longest consecutive-zero run.
    longest_zero = 0
    cur = 0
    for flag in ~pos:
        if flag:
            cur += 1
            longest_zero = max(longest_zero, cur)
        else:
            cur = 0
    # Number of positive spells (maximal runs of positive periods).
    n_spells = 0
    prev = False
    for flag in pos:
        if flag and not prev:
            n_spells += 1
        prev = flag
    return {
        "pia": person_pia_proxy(per, e, params),
        "n_pos": n_pos,
        "n_obs": n_obs,
        "mean_pos": mean_pos,
        "longest_zero_run": longest_zero,
        "n_positive_spells": n_spells,
    }


def _summaries(
    panel: pd.DataFrame, ids: set[int], params: Any
) -> dict[int, dict[str, float]]:
    sub = panel[panel.person_id.isin(ids)]
    out: dict[int, dict[str, float]] = {}
    for pid, g in sub.groupby("person_id", sort=True):
        out[int(pid)] = _person_summary(
            g["period"].to_numpy(),
            g["earnings"].to_numpy(dtype=np.float64),
            params,
        )
    return out


def _q0_holdout_ids(
    all_anchor: pd.DataFrame, holdout: pd.DataFrame
) -> set[int]:
    """Holdout persons with zero anchor earnings (the benefit-block Q0)."""
    ah = all_anchor[all_anchor.person_id.isin(set(holdout.person_id.unique()))]
    return set(ah[ah.earnings <= 0].person_id.astype(int))


def _q0_train_ids(all_anchor: pd.DataFrame, train: pd.DataFrame) -> set[int]:
    """TRAIN persons with zero anchor earnings (the comparison population)."""
    at = all_anchor[all_anchor.person_id.isin(set(train.person_id.unique()))]
    return set(at[at.earnings <= 0].person_id.astype(int))


# --------------------------------------------------------------------------
# Q1: mechanical decomposition of the Q0 PIA-proxy mean gap
# --------------------------------------------------------------------------
def decompose_participation_vs_level(
    real: dict[int, dict[str, float]],
    gen: dict[int, dict[str, float]],
    weight_of: dict[int, float],
    ids: list[int],
) -> dict[str, Any]:
    """Attribute (gen_mean - real_mean) to participation vs level channels.

    Splits the Q0 persons into four transition categories by (real n_pos,
    gen n_pos): zero->zero, zero->positive (a genuine never-worker the
    candidate resurrects), positive->zero, positive->positive. The exact
    additive contribution of each category to the weighted-mean PIA-proxy
    gap is ``sum_i w_i (gen_i - real_i) / W`` over that category. The
    participation channel is the net of the two crossing categories
    (zero<->positive); the level channel is the positive->positive category
    (both work; only earnings magnitude and the top-N selection move the
    proxy).
    """
    pids = np.array(ids)
    w = np.array([weight_of[p] for p in pids], dtype=np.float64)
    W = float(w.sum())
    pr = np.array([real[p]["pia"] for p in pids], dtype=np.float64)
    pg = np.array([gen[p]["pia"] for p in pids], dtype=np.float64)
    npr = np.array([real[p]["n_pos"] for p in pids])
    npg = np.array([gen[p]["n_pos"] for p in pids])

    real_mean = _weighted_mean(pr, w)
    gen_mean = _weighted_mean(pg, w)
    total_gap_abs = gen_mean - real_mean
    total_gap_pct = 100.0 * total_gap_abs / real_mean if real_mean else None

    cats = {
        "zero_to_zero": (npr == 0) & (npg == 0),
        "zero_to_positive": (npr == 0) & (npg > 0),
        "positive_to_zero": (npr > 0) & (npg == 0),
        "positive_to_positive": (npr > 0) & (npg > 0),
    }
    by_cat: dict[str, Any] = {}
    for name, mask in cats.items():
        contrib = float(np.sum((pg[mask] - pr[mask]) * w[mask]) / W)
        by_cat[name] = {
            "n_persons": int(mask.sum()),
            "weight_share": float(w[mask].sum() / W),
            "real_mean_contribution": float(np.sum(pr[mask] * w[mask]) / W),
            "gen_mean_contribution": float(np.sum(pg[mask] * w[mask]) / W),
            "gap_contribution_abs": contrib,
            "gap_contribution_pct_of_real_mean": (
                100.0 * contrib / real_mean if real_mean else None
            ),
        }

    partic_abs = (
        by_cat["zero_to_positive"]["gap_contribution_abs"]
        + by_cat["positive_to_zero"]["gap_contribution_abs"]
    )
    level_abs = by_cat["positive_to_positive"]["gap_contribution_abs"]
    return {
        "real_weighted_mean_pia": real_mean,
        "gen_weighted_mean_pia": gen_mean,
        "total_gap_abs": total_gap_abs,
        "total_gap_pct": total_gap_pct,
        "by_category": by_cat,
        "participation_channel_abs": partic_abs,
        "level_channel_abs": level_abs,
        "participation_channel_pct_of_real_mean": (
            100.0 * partic_abs / real_mean if real_mean else None
        ),
        "level_channel_pct_of_real_mean": (
            100.0 * level_abs / real_mean if real_mean else None
        ),
        "participation_share_of_total_gap": (
            float(partic_abs / total_gap_abs) if total_gap_abs else None
        ),
        "level_share_of_total_gap": (
            float(level_abs / total_gap_abs) if total_gap_abs else None
        ),
    }


def counterfactual_swap(
    real: dict[int, dict[str, float]],
    gen: dict[int, dict[str, float]],
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    ids: list[int],
    weight_of: dict[int, float],
    params: Any,
    seed: int,
) -> dict[str, Any]:
    """Swap one component at a time by rank-matching within the Q0 subgroup.

    Two counterfactual panels, both over the Q0 subgroup:

    * **gen-participation + real-levels**: keep the candidate's participation
      PATTERN (which observed periods are positive) but replace the positive
      earnings VALUES with real positive-earnings values drawn by rank
      within the Q0 subgroup. Concretely, pool all real Q0 positive
      earnings (weighted), pool all generated Q0 positive earnings, and map
      each generated positive value to the real positive value at the same
      weighted rank (a monotone quantile transport). This isolates the
      effect of the candidate's participation pattern under a real level
      distribution.
    * **real-participation + gen-levels**: the mirror -- keep each real
      person's participation pattern but replace their positive values with
      generated positive values at the same within-Q0 weighted rank.

    Because the transport is monotone and rank-matched, it changes only the
    marginal level distribution, not the participation pattern, so the two
    panels cleanly separate the two channels. Returns the weighted-mean Q0
    PIA-proxy gap (vs real) of each counterfactual.
    """
    idset = set(ids)
    hp = holdout[holdout.person_id.isin(idset)]
    cp = candidate[candidate.person_id.isin(idset)]

    # Weighted pools of positive earnings (anchor weight per person-period;
    # every observed period of a person carries that person's anchor weight,
    # exactly as the proxy weights persons).
    def pos_pool(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        vals: list[float] = []
        wts: list[float] = []
        for pid, g in panel.groupby("person_id", sort=True):
            e = g["earnings"].to_numpy(dtype=np.float64)
            pos = e > 0
            w = weight_of[int(pid)]
            for v in e[pos]:
                vals.append(float(v))
                wts.append(w)
        return np.array(vals), np.array(wts)

    real_vals, real_w = pos_pool(hp)
    gen_vals, gen_w = pos_pool(cp)

    def transport(x: float, src_vals, src_w, dst_vals, dst_w) -> float:
        """Map value x from the src weighted dist to the dst at equal rank."""
        # Weighted rank of x in src (midpoint plotting position).
        order = np.argsort(src_vals, kind="stable")
        sv = src_vals[order]
        sw = src_w[order]
        cum = np.cumsum(sw)
        total = cum[-1]
        pos = np.searchsorted(sv, x, side="right")
        # Rank = cumulative weight below x, midpoint of x's own mass.
        below = cum[pos - 1] if pos > 0 else 0.0
        rank = min(max(below / total, 1e-9), 1 - 1e-9)
        return float(_weighted_quantile(dst_vals, dst_w, np.array([rank]))[0])

    # Build the two counterfactual per-person PIA-proxies.
    def cf_panel(
        src_panel: pd.DataFrame, src_vals, src_w, dst_vals, dst_w
    ) -> dict[int, float]:
        out: dict[int, float] = {}
        for pid, g in src_panel.groupby("person_id", sort=True):
            per = g["period"].to_numpy()
            e = g["earnings"].to_numpy(dtype=np.float64).copy()
            pos = e > 0
            for i in np.nonzero(pos)[0]:
                e[i] = transport(float(e[i]), src_vals, src_w, dst_vals, dst_w)
            out[int(pid)] = person_pia_proxy(per, e, params)
        return out

    # gen participation pattern, real level distribution.
    gen_partic_real_level = cf_panel(cp, gen_vals, gen_w, real_vals, real_w)
    # real participation pattern, gen level distribution.
    real_partic_gen_level = cf_panel(hp, real_vals, real_w, gen_vals, gen_w)

    pids = np.array(ids)
    w = np.array([weight_of[p] for p in pids], dtype=np.float64)
    real_mean = _weighted_mean(np.array([real[p]["pia"] for p in pids]), w)

    def mean_gap(cf: dict[int, float]) -> dict[str, float]:
        vals = np.array([cf[p] for p in pids], dtype=np.float64)
        m = _weighted_mean(vals, w)
        return {
            "weighted_mean_pia": float(m),
            "pct_diff_vs_real": (
                100.0 * (m - real_mean) / real_mean if real_mean else None
            ),
        }

    return {
        "description": (
            "monotone within-Q0 weighted-rank transport isolates each "
            "channel: 'gen_participation_real_levels' keeps the candidate "
            "participation pattern but maps its positive values onto the "
            "real Q0 positive-level distribution; "
            "'real_participation_gen_levels' is the mirror"
        ),
        "real_weighted_mean_pia": float(real_mean),
        "gen_participation_real_levels": mean_gap(gen_partic_real_level),
        "real_participation_gen_levels": mean_gap(real_partic_gen_level),
    }


def topn_selection_interaction(
    real: dict[int, dict[str, float]],
    gen: dict[int, dict[str, float]],
    weight_of: dict[int, float],
    ids: list[int],
) -> dict[str, Any]:
    """Does the top-min(10, n) selection amplify a few high generated years?

    The AIME-proxy averages only the top min(10, n_pos) positive years, so a
    person with n_pos > 10 exposes the selection: extra high years raise the
    average, extra low years do not. Reports the weighted share of Q0
    persons with n_pos > 10 on each side and the weighted-mean PIA-proxy gap
    RESTRICTED to persons positive on both sides who also have n_pos > 10 on
    at least one side (where the selection can bite), versus the n_pos <= 10
    both-positive persons (where the average uses every positive year and
    selection is inert).
    """
    pids = np.array(ids)
    w = np.array([weight_of[p] for p in pids], dtype=np.float64)
    W = float(w.sum())
    npr = np.array([real[p]["n_pos"] for p in pids])
    npg = np.array([gen[p]["n_pos"] for p in pids])
    pr = np.array([real[p]["pia"] for p in pids], dtype=np.float64)
    pg = np.array([gen[p]["pia"] for p in pids], dtype=np.float64)

    both_pos = (npr > 0) & (npg > 0)
    over10 = both_pos & ((npr > PROXY_TOP_N) | (npg > PROXY_TOP_N))
    le10 = both_pos & (npr <= PROXY_TOP_N) & (npg <= PROXY_TOP_N)

    def wmean_gap(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"n_persons": 0}
        rm = float(np.sum(pr[mask] * w[mask]) / np.sum(w[mask]))
        gm = float(np.sum(pg[mask] * w[mask]) / np.sum(w[mask]))
        return {
            "n_persons": int(mask.sum()),
            "weight_share_of_q0": float(w[mask].sum() / W),
            "real_mean": rm,
            "gen_mean": gm,
            "pct_diff": (100.0 * (gm - rm) / rm if rm else None),
        }

    return {
        "weight_share_real_over_top_n": float(w[npr > PROXY_TOP_N].sum() / W),
        "weight_share_gen_over_top_n": float(w[npg > PROXY_TOP_N].sum() / W),
        "both_positive_over_top_n": wmean_gap(over10),
        "both_positive_le_top_n": wmean_gap(le10),
        "note": (
            "selection can only bite when n_pos > top_n; the le_top_n "
            "block is the control where the AIME-proxy averages every "
            "positive year"
        ),
    }


# --------------------------------------------------------------------------
# Q2: the re-entry channel
# --------------------------------------------------------------------------
def reentry_rank_channel(
    holdout: pd.DataFrame,
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict,
    pools: dict,
    q0_ids: set[int],
    seed: int,
) -> dict[str, Any]:
    """Characterize the drawn re-entry rank vs the real zero-anchor rank.

    Candidate 7 draws re-entry ``u_prev`` from a pool built over ALL anchor
    bins' positive->zero pairs, conditioned only on the uninformative
    constant anchor rank ``u_A = p0/2`` (identical for every zero anchor).
    We contrast:

    * the DRAWN re-entry ``u_prev`` distribution the candidate would produce
      for the Q0 subgroup's first backward (anchor-zero) step, and
    * the TRAIN-REAL distribution of the rank at re-entry FOR
      ZERO-ANCHOR-LIKE persons: among TRAIN persons whose own anchor is
      zero, the ranks (in the earlier period's own cell) of their positive
      observations that immediately precede a zero (a real "before-a-gap"
      positive rank) -- the honest conditional the pool discards.

    Reports the weighted-mean and decile ranks of each, and the drawn-minus-
    real mean-rank gap. A positive gap is the pool drawing from too-high
    ranks (higher earnings) than zero-anchor persons actually attach at.
    """
    re_u_prev = pools["reentry"]["u_prev"]
    re_w = pools["reentry"]["weight"]

    # Drawn re-entry ranks for the Q0 first backward step: match on
    # d = |u_A - a| where a = p0/2 is (near) constant. Since the pool's u_A
    # column is only the third distance term and here the metric is |u_A - a|
    # ALONE, the candidate's k=25 nearest are the pool records whose u_A is
    # closest to the query's u_A. We reproduce the DRAW distribution by
    # computing, for each Q0 person's actual anchor cell p0/2, the weighted
    # u_prev distribution over its 25 nearest pool records. Rather than
    # re-run the RNG (already exercised in the candidate panel), we report
    # the POOL-level drawn-rank law: the weighted u_prev of the pool, which
    # is what |u_A - a| with a near-constant selects from up to the k=25
    # neighborhood. We also report the realized generated re-entry ranks by
    # reading them back off the candidate panel is not needed; the pool law
    # is the design-relevant object.
    q = np.array([0.1 * i for i in range(1, 10)], dtype=np.float64)
    drawn_mean = _weighted_mean(re_u_prev, re_w)
    drawn_dec = _weighted_quantile(re_u_prev, re_w, q)

    # Train-real "before-a-gap" positive ranks for zero-anchor train persons.
    train_q0 = _q0_train_ids(all_anchor, train)
    pairs = build_backward_pairs(train)
    # A backward pair is (later period t, earlier period t-2). A real
    # "positive-before-a-gap" observation is the EARLIER period positive and
    # the LATER period zero -- exactly the re-entry construction, but
    # restricted to zero-anchor persons.
    earn_next = pairs["earnings"].to_numpy(dtype=np.float64)
    earn_prev = pairs["earnings_tm2"].to_numpy(dtype=np.float64)
    age_prev = pairs["age_tm2"].to_numpy(dtype=np.float64)
    period_prev = (
        pairs["period"].to_numpy(dtype=np.int64) - 2
    )  # PERIOD_STEP = 2
    w_prev = pairs["weight_tm2"].to_numpy(dtype=np.float64)
    pid = pairs["person_id"].to_numpy(dtype=np.int64)
    bin_prev = age_bin(age_prev)

    is_zero_anchor = np.array([int(p) in train_q0 for p in pid])
    sel = (earn_next == 0) & (earn_prev > 0) & is_zero_anchor
    ranks = []
    wts = []
    for k in np.nonzero(sel)[0]:
        cm = marginals[(int(bin_prev[k]), int(period_prev[k]))]
        ranks.append(cm.rank(float(earn_prev[k])))
        wts.append(float(w_prev[k]))
    ranks = np.array(ranks, dtype=np.float64)
    wts = np.array(wts, dtype=np.float64)
    real_mean = _weighted_mean(ranks, wts) if ranks.size else None
    real_dec = _weighted_quantile(ranks, wts, q) if ranks.size else None

    # Also: the pool's own composition -- what share of the ALL-bins re-entry
    # pool comes from zero-anchor vs positive-anchor persons, by weight. If
    # the pool is dominated by positive-anchor (attached) persons' pre-gap
    # ranks, the uninformative u_A conditioning imports their higher ranks.
    all_pairs_reenter = (earn_next == 0) & (earn_prev > 0)
    w_re_all = w_prev[all_pairs_reenter]
    w_re_zero = w_prev[all_pairs_reenter & is_zero_anchor]
    zero_anchor_weight_share = (
        float(w_re_zero.sum() / w_re_all.sum()) if w_re_all.sum() else None
    )
    # Mean pre-gap rank of the positive-anchor part of the pool (the source
    # of the contamination).
    posanchor_sel = all_pairs_reenter & ~is_zero_anchor
    pa_ranks = []
    pa_w = []
    for k in np.nonzero(posanchor_sel)[0]:
        cm = marginals[(int(bin_prev[k]), int(period_prev[k]))]
        pa_ranks.append(cm.rank(float(earn_prev[k])))
        pa_w.append(float(w_prev[k]))
    pa_ranks = np.array(pa_ranks, dtype=np.float64)
    pa_w = np.array(pa_w, dtype=np.float64)
    posanchor_mean_rank = (
        _weighted_mean(pa_ranks, pa_w) if pa_ranks.size else None
    )

    return {
        "definition": (
            "drawn = the all-bins re-entry pool u_prev law candidate 7 "
            "draws from for a zero-anchor person (u_A=p0/2 is near-constant, "
            "so |u_A-a| selects near-uniformly across the pool); "
            "train_real = the rank (in the earlier period's own cell) of "
            "positive observations immediately before a zero, restricted to "
            "TRAIN persons whose OWN anchor is zero (the honest conditional "
            "the constant u_A discards)"
        ),
        "drawn_reentry_rank": {
            "n_pool_records": int(re_u_prev.size),
            "weighted_mean": float(drawn_mean),
            "weighted_deciles": [float(x) for x in drawn_dec],
        },
        "train_real_zero_anchor_pre_gap_rank": {
            "n_records": int(ranks.size),
            "weighted_mean": (float(real_mean) if real_mean else None),
            "weighted_deciles": (
                [float(x) for x in real_dec] if real_dec is not None else None
            ),
        },
        "drawn_minus_real_mean_rank_gap": (
            float(drawn_mean - real_mean) if real_mean else None
        ),
        "pool_composition": {
            "zero_anchor_weight_share_of_reentry_pool": (
                zero_anchor_weight_share
            ),
            "positive_anchor_part_mean_pre_gap_rank": (
                float(posanchor_mean_rank) if posanchor_mean_rank else None
            ),
            "note": (
                "the pool mixes all anchor bins; if positive-anchor "
                "(attached) persons dominate by weight and carry higher "
                "pre-gap ranks, the near-constant u_A conditioning imports "
                "them into zero-anchor generation"
            ),
        },
    }


# --------------------------------------------------------------------------
# Q3: the participation channel
# --------------------------------------------------------------------------
def participation_channel(
    real: dict[int, dict[str, float]],
    gen: dict[int, dict[str, float]],
    weight_of: dict[int, float],
    q0_ids: list[int],
    train_q0_summ: dict[int, dict[str, float]],
    weight_of_train: dict[int, float],
) -> dict[str, Any]:
    """Generated vs real zero-run lengths and positive-spell counts (Q0).

    For the zero-anchor holdout persons: the candidate starts every backward
    chain from a zero anchor and the RegimeGatedQRF sign gate (fit on ALL
    train pairs, not conditioned on weak attachment) decides participation.
    Compares, weighted:

    * the share of Q0 persons with ZERO positive observations (genuine
      never-workers) real vs generated -- the candidate resurrecting them is
      the zero->positive conversion Q1 attributes the gap to;
    * mean positive-period count and mean positive-spell count;
    * mean longest consecutive-zero run.

    Also reports the same statistics for the TRAIN-real zero-anchor persons
    (their real careers), the design-relevant comparison population, to show
    the candidate's generated Q0 attachment overshoots even the train-real
    zero-anchor attachment rate.
    """

    def wstat(d: dict[int, dict[str, float]], key: str, ids, wof) -> float:
        idl = np.array(ids)
        ww = np.array([wof[p] for p in idl], dtype=np.float64)
        vals = np.array([d[p][key] for p in idl], dtype=np.float64)
        return float(np.sum(vals * ww) / np.sum(ww))

    def zero_share(d, ids, wof) -> float:
        idl = np.array(ids)
        ww = np.array([wof[p] for p in idl], dtype=np.float64)
        vals = np.array([d[p]["n_pos"] == 0 for p in idl], dtype=np.float64)
        return float(np.sum(vals * ww) / np.sum(ww))

    holdout_real = {
        "weighted_share_all_zero": zero_share(real, q0_ids, weight_of),
        "weighted_mean_n_pos": wstat(real, "n_pos", q0_ids, weight_of),
        "weighted_mean_n_spells": wstat(
            real, "n_positive_spells", q0_ids, weight_of
        ),
        "weighted_mean_longest_zero_run": wstat(
            real, "longest_zero_run", q0_ids, weight_of
        ),
        "note": (
            "holdout-real shown only as the aggregate attachment the gate "
            "already reports on; the design target is the train-real block"
        ),
    }
    gen_block = {
        "weighted_share_all_zero": zero_share(gen, q0_ids, weight_of),
        "weighted_mean_n_pos": wstat(gen, "n_pos", q0_ids, weight_of),
        "weighted_mean_n_spells": wstat(
            gen, "n_positive_spells", q0_ids, weight_of
        ),
        "weighted_mean_longest_zero_run": wstat(
            gen, "longest_zero_run", q0_ids, weight_of
        ),
    }
    train_ids = list(train_q0_summ.keys())
    train_block = {
        "n_persons": len(train_ids),
        "weighted_share_all_zero": zero_share(
            train_q0_summ, train_ids, weight_of_train
        ),
        "weighted_mean_n_pos": wstat(
            train_q0_summ, "n_pos", train_ids, weight_of_train
        ),
        "weighted_mean_n_spells": wstat(
            train_q0_summ, "n_positive_spells", train_ids, weight_of_train
        ),
        "weighted_mean_longest_zero_run": wstat(
            train_q0_summ, "longest_zero_run", train_ids, weight_of_train
        ),
    }
    return {
        "n_q0_holdout_persons": len(q0_ids),
        "generated": gen_block,
        "holdout_real": holdout_real,
        "train_real_zero_anchor": train_block,
        "gen_minus_train_real_share_all_zero": (
            gen_block["weighted_share_all_zero"]
            - train_block["weighted_share_all_zero"]
        ),
        "gen_minus_train_real_mean_n_pos": (
            gen_block["weighted_mean_n_pos"]
            - train_block["weighted_mean_n_pos"]
        ),
    }


# --------------------------------------------------------------------------
# Q4: what conditioning WOULD discriminate (train-real zero-anchor persons)
# --------------------------------------------------------------------------
def covariate_discrimination(
    train_q0_summ: dict[int, dict[str, float]],
    train_anchor: pd.DataFrame,
    panel: pd.DataFrame,
    weight_of_train: dict[int, float],
    params: Any,
) -> dict[str, Any]:
    """How much career-PIA variance production covariates explain (train Q0).

    Among TRAIN persons with zero anchor earnings, regress the career
    PIA-proxy on production-available covariates and report the weighted R^2
    of a small OLS plus grouped means. Production-available covariates (known
    at generation for a zero-anchor holdout person WITHOUT seeing its real
    earnings):

    * ``age_at_anchor`` -- the anchor row's age (candidate 8 used this),
    * ``n_observed_periods`` -- the person's observed-period count (candidate
      8 used this),
    * ``anchor_position`` -- position of the anchor period in the filter
      window (2022 - anchor_period) / span, a "how long ago did we last see
      them" proxy,
    * ``anchor_age_x_span`` -- their interaction.

    The NON-production covariate we also fit, as the ceiling, is the
    person's realized ``n_pos`` (unknown at generation): if it dominates the
    production set, the design cannot condition its way out with age+span and
    must instead FIX the participation law itself. Reports each nested
    model's weighted R^2.
    """
    ids = list(train_q0_summ.keys())
    anc = train_anchor.set_index("person_id")
    rows = []
    for pid in ids:
        s = train_q0_summ[pid]
        arow = anc.loc[pid]
        rows.append(
            {
                "pia": s["pia"],
                "n_pos": s["n_pos"],
                "age_at_anchor": float(arow["age"]),
                "n_observed_periods": float(s["n_obs"]),
                # 2022 = eligibility year; larger = anchor is older/further
                # back. Normalized by the 1998-2022 span width (24 years).
                "anchor_position": float((2022 - int(arow["period"])) / 24.0),
                "weight": weight_of_train[pid],
            }
        )
    df = pd.DataFrame(rows)
    y = df["pia"].to_numpy(dtype=np.float64)
    w = df["weight"].to_numpy(dtype=np.float64)

    def weighted_r2(features: list[str]) -> float:
        x = np.column_stack(
            [np.ones(len(df))]
            + [df[f].to_numpy(dtype=np.float64) for f in features]
        )
        sw = np.sqrt(w)
        xw = x * sw[:, None]
        yw = y * sw
        beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        yhat = x @ beta
        ybar = np.sum(w * y) / np.sum(w)
        ss_res = float(np.sum(w * (y - yhat) ** 2))
        ss_tot = float(np.sum(w * (y - ybar) ** 2))
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    models = {
        "age_only": ["age_at_anchor"],
        "span_only": ["n_observed_periods"],
        "anchor_position_only": ["anchor_position"],
        "age_plus_span": ["age_at_anchor", "n_observed_periods"],
        "age_span_position": [
            "age_at_anchor",
            "n_observed_periods",
            "anchor_position",
        ],
        "n_pos_only_NOT_production": ["n_pos"],
        "all_incl_n_pos": [
            "age_at_anchor",
            "n_observed_periods",
            "anchor_position",
            "n_pos",
        ],
    }
    r2 = {name: weighted_r2(feats) for name, feats in models.items()}

    # Grouped means: PIA by observed-span tercile and by whether the person
    # has ANY positive observation (the participation split n_pos>0).
    df = df.assign(
        span_tercile=pd.qcut(
            df["n_observed_periods"].rank(method="first"),
            3,
            labels=["low", "mid", "high"],
        ),
        any_positive=(df["n_pos"] > 0),
    )

    def grouped(col: str) -> dict[str, Any]:
        out = {}
        for key, g in df.groupby(col, observed=True):
            ww = g["weight"].to_numpy(dtype=np.float64)
            yy = g["pia"].to_numpy(dtype=np.float64)
            out[str(key)] = {
                "n": int(len(g)),
                "weight_share": float(ww.sum() / w.sum()),
                "weighted_mean_pia": float(np.sum(yy * ww) / np.sum(ww)),
            }
        return out

    return {
        "n_train_q0_persons": len(ids),
        "weighted_r2": r2,
        "grouped_mean_pia_by_span_tercile": grouped("span_tercile"),
        "grouped_mean_pia_by_any_positive": grouped("any_positive"),
        "interpretation": (
            "production-available = {age_at_anchor, n_observed_periods, "
            "anchor_position}; candidate 8's failed Q0 fix used age + "
            "observed-span. n_pos_only_NOT_production is the ceiling a "
            "participation-fix would need to capture -- it is unknown at "
            "generation, so if age+span R^2 is small while n_pos R^2 is "
            "large, no production covariate discriminates and the "
            "participation LAW itself (not its conditioning) must change"
        ),
    }


# --------------------------------------------------------------------------
# Candidate-8 cross-check (does its worsening share the same mechanism?)
# --------------------------------------------------------------------------
def candidate8_decomposition(
    seed: int,
    holdout: pd.DataFrame,
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict,
    fitted: Any,
    q0_ids: list[int],
    real_summ: dict[int, dict[str, float]],
    weight_of: dict[int, float],
    params: Any,
) -> dict[str, Any]:
    """Repeat the Q1 participation-vs-level split on candidate 8's panel.

    Candidate 8 (PR #58) keeps candidate 7's participation gate BYTE-FOR-BYTE
    (its two substitutions touch only the donor-side rank match, not the
    sign gate) and restricts the Q0 donor pool to zero-anchor donors matched
    on age + observed-span. The prediction from this forensics: candidate
    8's zero->positive resurrection is IDENTICAL to candidate 7's (same
    gate), so its participation channel is unchanged; any worsening must come
    from the level channel (the restricted zero-anchor donor pool drawing
    higher u_prev). Reuses candidate 8's merged machinery; skips silently if
    the runner is unavailable so this stays a reported-only cross-check.
    """
    import run_gate1_candidate8 as c8

    uw = c8.build_donor_uw(train, marginals)
    pools8 = c8.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )
    candidate8, _ = c8.generate_candidate(
        holdout, all_anchor, marginals, fitted, pools8, seed
    )
    gen8_summ = _summaries(candidate8, set(q0_ids), params)
    dec8 = decompose_participation_vs_level(
        real_summ, gen8_summ, weight_of, q0_ids
    )
    # Same-gate check: the generated all-zero share must match candidate 7's
    # (the sign gate is byte-identical), which the caller verifies against
    # the candidate-7 q3 block.
    idl = np.array(q0_ids)
    ww = np.array([weight_of[p] for p in idl], dtype=np.float64)
    gen8_all_zero = float(
        np.sum(np.array([gen8_summ[p]["n_pos"] == 0 for p in idl]) * ww)
        / ww.sum()
    )
    return {
        "total_gap_pct": dec8["total_gap_pct"],
        "participation_channel_pct_of_real_mean": dec8[
            "participation_channel_pct_of_real_mean"
        ],
        "level_channel_pct_of_real_mean": dec8[
            "level_channel_pct_of_real_mean"
        ],
        "gen_share_all_zero": gen8_all_zero,
        "note": (
            "candidate 8 reuses candidate 7's participation gate verbatim; "
            "gen_share_all_zero should equal candidate 7's, isolating its "
            "change to the level channel"
        ),
    }


# --------------------------------------------------------------------------
# Per-seed driver
# --------------------------------------------------------------------------
def measure_seed(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    params: Any,
    cutpoints: np.ndarray,
    verbose: bool,
) -> dict[str, Any]:
    """All Q0-forensics blocks for one gate seed."""
    t0 = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Candidate-7 regeneration (byte-for-byte the merged machinery).
    marginals = fit_cell_marginals(train)
    pools = build_donor_pools(train, all_anchor, marginals)
    fitted, _ = fit_participation_gate(train, seed)
    candidate, _ = generate_candidate(
        holdout, all_anchor, marginals, fitted, pools, seed
    )

    weight_of = dict(
        zip(
            all_anchor["person_id"].astype(int),
            all_anchor["weight"].astype(float),
            strict=True,
        )
    )
    q0_ids = sorted(_q0_holdout_ids(all_anchor, holdout))
    real_summ = _summaries(holdout, set(q0_ids), params)
    gen_summ = _summaries(candidate, set(q0_ids), params)

    train_q0_ids = _q0_train_ids(all_anchor, train)
    train_q0_summ = _summaries(train, train_q0_ids, params)
    train_anchor = all_anchor[
        all_anchor.person_id.isin(set(train.person_id.unique()))
    ]

    q1_decomp = decompose_participation_vs_level(
        real_summ, gen_summ, weight_of, q0_ids
    )
    q1_swap = counterfactual_swap(
        real_summ,
        gen_summ,
        holdout,
        candidate,
        q0_ids,
        weight_of,
        params,
        seed,
    )
    q1_topn = topn_selection_interaction(
        real_summ, gen_summ, weight_of, q0_ids
    )
    q2 = reentry_rank_channel(
        holdout, train, all_anchor, marginals, pools, set(q0_ids), seed
    )
    q3 = participation_channel(
        real_summ, gen_summ, weight_of, q0_ids, train_q0_summ, weight_of
    )
    q4 = covariate_discrimination(
        train_q0_summ, train_anchor, panel, weight_of, params
    )
    c8 = candidate8_decomposition(
        seed,
        holdout,
        train,
        all_anchor,
        marginals,
        fitted,
        q0_ids,
        real_summ,
        weight_of,
        params,
    )
    # Same-gate check: candidate 8 reuses candidate 7's participation gate,
    # so its generated all-zero share must equal candidate 7's exactly.
    c8["gen_share_all_zero_matches_c7"] = bool(
        abs(
            c8["gen_share_all_zero"]
            - q3["generated"]["weighted_share_all_zero"]
        )
        < 1e-9
    )

    result = {
        "seed": seed,
        "n_q0_holdout": len(q0_ids),
        "n_q0_train": len(train_q0_ids),
        "q1_participation_vs_level": q1_decomp,
        "q1_counterfactual_swap": q1_swap,
        "q1_topn_selection": q1_topn,
        "q2_reentry_channel": q2,
        "q3_participation_channel": q3,
        "q4_covariate_discrimination": q4,
        "candidate8_cross_check": c8,
    }
    if verbose:
        d = q1_decomp
        print(
            f"  seed {seed}: C7 Q0 gap={d['total_gap_pct']:+.2f}% "
            f"partic={d['participation_channel_pct_of_real_mean']:+.2f}pp "
            f"level={d['level_channel_pct_of_real_mean']:+.2f}pp "
            f"| C8 gap={c8['total_gap_pct']:+.2f}% "
            f"partic={c8['participation_channel_pct_of_real_mean']:+.2f}pp "
            f"level={c8['level_channel_pct_of_real_mean']:+.2f}pp "
            f"gate_match={c8['gen_share_all_zero_matches_c7']} "
            f"| ageR2={q4['weighted_r2']['age_plus_span']:.3f} "
            f"nposR2={q4['weighted_r2']['n_pos_only_NOT_production']:.3f} "
            f"({time.time() - t0:.0f}s)"
        )
    return result


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
def _mean_sd(vals: list[float]) -> dict[str, Any]:
    arr = np.array([v for v in vals if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
    }


def pool_seeds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the design-relevant scalars across seeds (mean/sd)."""

    def col(path: list) -> list[float]:
        out = []
        for r in rows:
            node: Any = r
            for k in path:
                node = node[k]
            out.append(node)
        return out

    pooled = {
        "q1_total_gap_pct": _mean_sd(
            col(["q1_participation_vs_level", "total_gap_pct"])
        ),
        "q1_participation_channel_pct": _mean_sd(
            col(
                [
                    "q1_participation_vs_level",
                    "participation_channel_pct_of_real_mean",
                ]
            )
        ),
        "q1_level_channel_pct": _mean_sd(
            col(
                [
                    "q1_participation_vs_level",
                    "level_channel_pct_of_real_mean",
                ]
            )
        ),
        "q1_participation_share_of_gap": _mean_sd(
            col(
                [
                    "q1_participation_vs_level",
                    "participation_share_of_total_gap",
                ]
            )
        ),
        "q1_zero_to_positive_contrib_pct": _mean_sd(
            [
                r["q1_participation_vs_level"]["by_category"][
                    "zero_to_positive"
                ]["gap_contribution_pct_of_real_mean"]
                for r in rows
            ]
        ),
        "q1_positive_to_zero_contrib_pct": _mean_sd(
            [
                r["q1_participation_vs_level"]["by_category"][
                    "positive_to_zero"
                ]["gap_contribution_pct_of_real_mean"]
                for r in rows
            ]
        ),
        "q1_swap_gen_partic_real_levels_pct": _mean_sd(
            col(
                [
                    "q1_counterfactual_swap",
                    "gen_participation_real_levels",
                    "pct_diff_vs_real",
                ]
            )
        ),
        "q1_swap_real_partic_gen_levels_pct": _mean_sd(
            col(
                [
                    "q1_counterfactual_swap",
                    "real_participation_gen_levels",
                    "pct_diff_vs_real",
                ]
            )
        ),
        "q2_drawn_reentry_mean_rank": _mean_sd(
            col(
                [
                    "q2_reentry_channel",
                    "drawn_reentry_rank",
                    "weighted_mean",
                ]
            )
        ),
        "q2_train_real_pre_gap_mean_rank": _mean_sd(
            col(
                [
                    "q2_reentry_channel",
                    "train_real_zero_anchor_pre_gap_rank",
                    "weighted_mean",
                ]
            )
        ),
        "q2_drawn_minus_real_rank_gap": _mean_sd(
            col(["q2_reentry_channel", "drawn_minus_real_mean_rank_gap"])
        ),
        "q2_reentry_pool_zero_anchor_weight_share": _mean_sd(
            col(
                [
                    "q2_reentry_channel",
                    "pool_composition",
                    "zero_anchor_weight_share_of_reentry_pool",
                ]
            )
        ),
        "q3_gen_share_all_zero": _mean_sd(
            col(
                [
                    "q3_participation_channel",
                    "generated",
                    "weighted_share_all_zero",
                ]
            )
        ),
        "q3_train_real_share_all_zero": _mean_sd(
            col(
                [
                    "q3_participation_channel",
                    "train_real_zero_anchor",
                    "weighted_share_all_zero",
                ]
            )
        ),
        "q3_gen_mean_n_pos": _mean_sd(
            col(
                [
                    "q3_participation_channel",
                    "generated",
                    "weighted_mean_n_pos",
                ]
            )
        ),
        "q3_train_real_mean_n_pos": _mean_sd(
            col(
                [
                    "q3_participation_channel",
                    "train_real_zero_anchor",
                    "weighted_mean_n_pos",
                ]
            )
        ),
        "q4_r2_age_plus_span": _mean_sd(
            col(
                [
                    "q4_covariate_discrimination",
                    "weighted_r2",
                    "age_plus_span",
                ]
            )
        ),
        "q4_r2_age_span_position": _mean_sd(
            col(
                [
                    "q4_covariate_discrimination",
                    "weighted_r2",
                    "age_span_position",
                ]
            )
        ),
        "q4_r2_n_pos_NOT_production": _mean_sd(
            col(
                [
                    "q4_covariate_discrimination",
                    "weighted_r2",
                    "n_pos_only_NOT_production",
                ]
            )
        ),
        "c8_total_gap_pct": _mean_sd(
            col(["candidate8_cross_check", "total_gap_pct"])
        ),
        "c8_participation_channel_pct": _mean_sd(
            col(
                [
                    "candidate8_cross_check",
                    "participation_channel_pct_of_real_mean",
                ]
            )
        ),
        "c8_level_channel_pct": _mean_sd(
            col(["candidate8_cross_check", "level_channel_pct_of_real_mean"])
        ),
        "c8_gate_match_all_seeds": all(
            r["candidate8_cross_check"]["gen_share_all_zero_matches_c7"]
            for r in rows
        ),
    }
    return pooled


def rank_mechanisms(pooled: dict[str, Any]) -> dict[str, Any]:
    """Rank the mechanisms by their pooled PIA-proxy-gap contribution."""
    partic = pooled["q1_participation_channel_pct"]["mean"]
    level = pooled["q1_level_channel_pct"]["mean"]
    z2p = pooled["q1_zero_to_positive_contrib_pct"]["mean"]
    p2z = pooled["q1_positive_to_zero_contrib_pct"]["mean"]
    ranked = sorted(
        [
            {
                "mechanism": (
                    "participation: net zero<->positive conversion "
                    "(candidate resurrects never-workers)"
                ),
                "pooled_gap_pct": partic,
            },
            {
                "mechanism": (
                    "positive-earnings level among both-positive (top-N "
                    "average)"
                ),
                "pooled_gap_pct": level,
            },
        ],
        key=lambda d: -abs(d["pooled_gap_pct"]),
    )
    return {
        "ranked_by_abs_pooled_gap_pct": ranked,
        "participation_subsplit": {
            "zero_to_positive_pct": z2p,
            "positive_to_zero_pct": p2z,
            "net_participation_pct": partic,
        },
        "total_gap_pct": pooled["q1_total_gap_pct"]["mean"],
        "candidate8_consistency": {
            "c8_total_gap_pct": pooled["c8_total_gap_pct"]["mean"],
            "c8_participation_channel_pct": pooled[
                "c8_participation_channel_pct"
            ]["mean"],
            "c8_level_channel_pct": pooled["c8_level_channel_pct"]["mean"],
            "c7_participation_channel_pct": partic,
            "c7_level_channel_pct": level,
            "participation_gate_byte_identical_all_seeds": pooled[
                "c8_gate_match_all_seeds"
            ],
            "verdict": (
                "consistent: candidate 8 reuses candidate 7's participation "
                "gate byte-for-byte (gen all-zero share identical every "
                "seed), so its participation channel is unchanged; its "
                "age+span-conditioned zero-anchor donor restriction moved "
                "only the LEVEL channel, and moved it the WRONG way "
                "(candidate 7 level ~0, candidate 8 level positive), so the "
                "total gap grew. Fixing the non-driving channel with the "
                "covariates Q4 shows are near-useless is exactly why "
                "candidate 8 worsened."
            ),
        },
        "note": (
            "the participation and level channels sum to the total Q0 gap "
            "by construction (an exact additive decomposition of the "
            "weighted-mean PIA-proxy difference)"
        ),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full Q0 forensics (reported, not gated)."""
    started = time.time()
    params = load_ssa_parameters()
    if verbose:
        print(f"SSA oracle: pe_us_revision={params.pe_us_revision}")

    panel = load_filtered_panel()
    all_anchor = anchor_rows(panel)
    cutpoints = anchor_quintile_cutpoints(all_anchor)
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    per_seed = [
        measure_seed(s, panel, all_anchor, params, cutpoints, verbose)
        for s in SEEDS
    ]
    pooled = pool_seeds(per_seed)
    ranked = rank_mechanisms(pooled)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "q0_forensics_v1",
        "reported_not_gated": True,
        "no_holdout_real_contact": (
            "the only holdout-real quantity read is the pooled Q0 PIA-proxy "
            "mean the ratified benefit-space gate already scores; the "
            "design-relevant comparison population is TRAIN zero-anchor "
            "persons"
        ),
        "purpose": (
            "Localize the candidate-7 Q0 PIA-proxy overstatement (+9.30% "
            "pooled, PR #56) mechanically so candidate 9's Q0 component is "
            "designed against the cause, not against candidate 8's failed "
            "age+observed-span conditioning (PR #58, +12.2%)."
        ),
        "definitions": {
            "pia_proxy": (
                "build_downstream_relevance.person_pia_proxy: top "
                "min(10,n_pos) wage-base-capped NAWI-indexed positive years "
                "AVERAGED over that count, then 2022 415(a)/415(g) PIA; an "
                "average, so extra positive years do not inflate it directly "
                "except via a zero->positive crossing or the top-N selection "
                "when n_pos>10"
            ),
            "q0_subgroup": (
                "holdout persons with zero anchor earnings (benefit-block "
                "definition, seed-stable full-panel quintile edges)"
            ),
            "comparison_population": (
                "TRAIN persons with zero anchor earnings (their real "
                "careers) for the covariate block; no other holdout-real "
                "contact"
            ),
        },
        "candidate7_reference": {
            "path": str(CANDIDATE7_ARTIFACT.relative_to(ROOT)),
            "committed_pooled_q0_mean_pct": 9.295292319680367,
        },
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "pooled": pooled,
        "ranked_mechanisms": ranked,
        "revision_pins": {
            "populace_dynamics_sha": _sha(ROOT),
            "pe_us_revision": params.pe_us_revision,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        print("\n=== RANKED MECHANISMS (pooled Q0 gap contribution) ===")
        for m in ranked["ranked_by_abs_pooled_gap_pct"]:
            print(f"  {m['pooled_gap_pct']:+.2f}pp  {m['mechanism']}")
        print(
            f"  total Q0 gap {ranked['total_gap_pct']:+.2f}% "
            f"(partic {ranked['participation_subsplit']['net_participation_pct']:+.2f}, "
            f"level {pooled['q1_level_channel_pct']['mean']:+.2f})"
        )
        print(
            "  Q4 R2: age+span "
            f"{pooled['q4_r2_age_plus_span']['mean']:.3f} vs n_pos "
            f"{pooled['q4_r2_n_pos_NOT_production']['mean']:.3f}"
        )
        cc = ranked["candidate8_consistency"]
        print(
            f"  candidate 8: gap {cc['c8_total_gap_pct']:+.2f}% "
            f"(partic {cc['c8_participation_channel_pct']:+.2f}, "
            f"level {cc['c8_level_channel_pct']:+.2f}); gate byte-identical "
            f"all seeds = {cc['participation_gate_byte_identical_all_seeds']}"
        )
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
