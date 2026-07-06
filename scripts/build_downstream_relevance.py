"""Downstream relevance of candidate 7: what the residual costs in benefit space.

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. Its
one purpose is to translate the candidate-7 classifier residual (the
population-view C2ST/geometry signal the locked gate scores) into the
downstream benefit-magnitude space a Social Security analysis actually
consumes, so a possible FUTURE public gate amendment (which would require a
fresh referee round) can weigh whether that residual matters where it would
be spent. Nothing here touches ``gates.yaml`` or any committed gate artifact.

The functional here is a STATUTE-SHAPED PROXY, NOT the full 42 USC 415(b)
computation. The filtered panel's careers are partial (biennial PSID
observation, prime-age 25-59 window, reference years 1998-2022), so a
faithful highest-35 AIME cannot be formed. The proxy below is a deliberate,
constant-scaled stand-in that (a) is monotone in lifetime earnings, (b)
routes through the real 415(a)/415(g) PIA bend-point formula so the benefit
progressivity is exact, and (c) uses a scale that cancels in every
real-vs-candidate comparison. Every artifact and doc says so explicitly.

What it does, per gate seed (0-4):

* Regenerate candidate 7 deterministically via the merged candidate-7
  machinery (:mod:`run_gate1_candidate7`): the locked filter-first load, the
  person-disjoint 0.2 split, the train-fit cell marginals / donor pools /
  participation gate, and the backward k-NN chain. The candidate panel holds
  exactly the holdout persons on exactly their observed periods; only
  earnings are generated (each person's anchor stays at its real earnings).
* Push BOTH the real holdout histories and the generated histories through
  the pinned proxy functional and measure how far apart the resulting
  PIA-proxy distributions are.
* Anchor those gaps against a real-vs-real noise floor built at the same
  scale (the ctx20 construction, applied to the seed's TRAIN split).

The proxy functional (pinned; applied identically to both sides):

1. For each person, take their observed POSITIVE-earnings periods within the
   locked filter (age 25-59, periods 1998-2022).
2. Cap each year's earnings at that year's wage base
   (:meth:`SSAParameters.wage_base_for`).
3. Index each capped year to 2022 by the NAWI ratio
   ``nawi[2022] / nawi[year]``.
4. ``AIME-proxy = sum(top min(10, n_pos) indexed values) / (count * 12 * 2)``
   where ``count = min(10, n_pos)``. The ``* 2`` reflects biennial
   observation (each observed year proxies a two-year window); it is a
   constant scale that cancels in real-vs-candidate comparisons but keeps
   magnitudes AIME-like.
5. ``PIA-proxy = pia(AIME-proxy, 2022, params)`` -- the 2022-eligibility
   415(a)/415(g) formula.
6. Persons with zero positive observations carry ``PIA-proxy = 0`` and are
   INCLUDED (both sides generate the same person set).

Every person is weighted by their anchor-period weight (the weight on their
chronologically last observed period, via
:func:`run_gate1_candidate5b.anchor_rows`).

Oracle parameters (NAWI, wage base, PIA bend points and factors) load ONCE
from the pinned policyengine-us checkout and the loader's recorded revision
is written into the artifact. Point the loader at the pin:

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/build_downstream_relevance.py

The candidate-7 regeneration needs populace-fit (the participation gate is a
RegimeGatedQRF sign gate), so run it from the dedicated gate venv.
"""

from __future__ import annotations

import hashlib
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

# Candidate-7 machinery (merged in pull request 55): the filter-first load,
# the person-disjoint 0.2 split, the train marginals / donor pools /
# participation gate, and the backward k-NN generation. Regeneration is
# deterministic from the seed alone, byte-for-byte the committed candidate-7
# run's, so the histories pushed through the proxy are exactly the ones the
# gate scored.
from run_gate1_baseline import (  # noqa: E402
    SEEDS,
    load_filtered_panel,
    split_holdout_train,
)
from run_gate1_candidate5b import (  # noqa: E402
    anchor_rows,
    fit_cell_marginals,
    fit_participation_gate,
)
from run_gate1_candidate7 import (  # noqa: E402
    build_donor_pools,
    generate_candidate,
)

from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.benefits import pia  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "downstream_relevance_c7_v1.json"
ARTIFACT_SCHEMA_VERSION = "downstream_relevance_c7.v1"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v1.json"

#: The eligibility year the proxy PIA formula is evaluated at (fixed).
ELIGIBILITY_YEAR = 2022
#: Highest-N indexed years the AIME proxy averages (min with the count of
#: positive observations); the partial-career analogue of the statute's 35.
PROXY_TOP_N = 10
#: Months per year and the biennial-window scale in the AIME-proxy divisor.
_MONTHS = 12
_BIENNIAL_SCALE = 2
#: Deciles reported for the distribution-gap block.
DECILES = tuple(round(0.1 * i, 1) for i in range(1, 10))
#: Number of anchor-earnings quintiles for the concentration block.
N_QUINTILES = 5
#: The paper's benefit-space success criterion (Phase-1 acceptance table,
#: docs/evaluation-and-model-selection.md and
#: docs/operationalizing-longitudinal-construction.md): "AIME distribution
#: for retired workers -- within 5 percent on key percentiles".
SUCCESS_CRITERION_PCT = 5.0
SUCCESS_CRITERION_TEXT = (
    "AIME distribution for retired workers: within 5 percent on key "
    "percentiles (paper Phase-1 acceptance table, "
    "docs/evaluation-and-model-selection.md 'AIME key percentiles: within "
    "5 percent'). The PIA-proxy is a monotone statute-shaped transform of "
    "the AIME-proxy, so its percentile gaps map onto this criterion."
)


# --------------------------------------------------------------------------
# Weighted statistics (midpoint-plotting-position convention, matching
# populace_dynamics.harness.metrics._weighted_quantile)
# --------------------------------------------------------------------------
def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Weighted quantiles at levels ``q`` (midpoint plotting positions).

    Byte-for-byte the harness convention
    (:func:`populace_dynamics.harness.metrics._weighted_quantile`): sort by
    value, place each atom at the midpoint of its normalized cumulative
    weight, and linearly interpolate. Reduces to the linear (numpy "linear")
    quantile when weights are equal.
    """
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    positions = (cumulative - 0.5 * w) / total
    return np.interp(q, positions, v)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini coefficient of a non-negative distribution.

    Standard weighted-Gini estimator on the weighted Lorenz curve: sort by
    value, form the weighted cumulative population and cumulative value, and
    sum trapezoids. Zeros are legitimate values (persons with no positive
    observations carry PIA-proxy 0), so the distribution can include a mass
    at 0; the estimator handles that directly. Returns 0.0 for a degenerate
    (all-equal or all-zero) distribution.
    """
    order = np.argsort(values, kind="stable")
    v = values[order].astype(np.float64)
    w = weights[order].astype(np.float64)
    total_w = np.sum(w)
    total_v = np.sum(w * v)
    if total_w <= 0 or total_v <= 0:
        return 0.0
    # Cumulative population share (midpoint) and cumulative value share.
    cum_w = np.cumsum(w)
    pop = cum_w / total_w
    cum_v = np.cumsum(w * v)
    lorenz = cum_v / total_v
    # Trapezoidal area under the Lorenz curve, anchored at (0, 0).
    pop_prev = np.concatenate([[0.0], pop[:-1]])
    lorenz_prev = np.concatenate([[0.0], lorenz[:-1]])
    area = np.sum((pop - pop_prev) * (lorenz + lorenz_prev) / 2.0)
    gini = 1.0 - 2.0 * area
    # Numerical guard: clamp into [0, 1].
    return float(min(1.0, max(0.0, gini)))


def _weighted_ks(
    a_vals: np.ndarray,
    a_w: np.ndarray,
    b_vals: np.ndarray,
    b_w: np.ndarray,
) -> float:
    """Weighted Kolmogorov-Smirnov distance between two 1-D distributions.

    The maximum absolute difference of the two weighted empirical CDFs,
    evaluated on the pooled sorted support with right-continuous (<=) CDFs.
    Weights are normalized within each sample, so the statistic is a proper
    distance in [0, 1] regardless of the two samples' total weights.
    """
    grid = np.unique(np.concatenate([a_vals, b_vals]))
    wa = a_w / np.sum(a_w)
    wb = b_w / np.sum(b_w)
    # Weighted CDF at each grid point: sum of normalized weight at values
    # <= grid point. searchsorted on sorted values gives the cutoff.
    oa = np.argsort(a_vals, kind="stable")
    ob = np.argsort(b_vals, kind="stable")
    a_sorted = a_vals[oa]
    b_sorted = b_vals[ob]
    a_wcum = np.cumsum(wa[oa])
    b_wcum = np.cumsum(wb[ob])
    ia = np.searchsorted(a_sorted, grid, side="right") - 1
    ib = np.searchsorted(b_sorted, grid, side="right") - 1
    cdf_a = np.where(ia >= 0, a_wcum[np.clip(ia, 0, len(a_wcum) - 1)], 0.0)
    cdf_b = np.where(ib >= 0, b_wcum[np.clip(ib, 0, len(b_wcum) - 1)], 0.0)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _pct_diff(candidate: float, real: float) -> float | None:
    """Signed percent difference of ``candidate`` from ``real``.

    ``None`` when ``real`` is zero (undefined), so callers can drop the
    point rather than divide by zero. Returned in percent (x 100).
    """
    if real == 0.0:
        return None
    return 100.0 * (candidate - real) / real


# --------------------------------------------------------------------------
# The pinned statute-shaped proxy functional
# --------------------------------------------------------------------------
def person_pia_proxy(
    periods: np.ndarray,
    earnings: np.ndarray,
    params: Any,
) -> float:
    """PIA-proxy for one person from their observed (period, earnings) rows.

    Statute-shaped proxy, NOT the full 415(b) AIME. Steps (pinned):

    1. keep positive-earnings periods only,
    2. cap each at that year's wage base,
    3. index each to 2022 by ``nawi[2022] / nawi[year]``,
    4. ``AIME-proxy = sum(top min(10, n_pos)) / (count * 12 * 2)``,
    5. ``PIA-proxy = pia(AIME-proxy, 2022, params)``.

    Zero positive observations -> ``0.0`` (the person is still included).
    """
    pos = earnings > 0
    if not np.any(pos):
        return 0.0
    yrs = periods[pos].astype(int)
    earn = earnings[pos].astype(np.float64)
    nawi_2022 = params.nawi[ELIGIBILITY_YEAR]
    indexed = np.empty(earn.size, dtype=np.float64)
    for i in range(earn.size):
        year = int(yrs[i])
        capped = min(float(earn[i]), params.wage_base_for(year))
        indexed[i] = capped * nawi_2022 / params.nawi[year]
    n_pos = indexed.size
    count = min(PROXY_TOP_N, n_pos)
    top = np.sort(indexed)[::-1][:count]
    aime_proxy = float(np.sum(top)) / (count * _MONTHS * _BIENNIAL_SCALE)
    return float(pia(aime_proxy, ELIGIBILITY_YEAR, params))


def panel_pia_proxy(
    panel: pd.DataFrame,
    anchor: pd.DataFrame,
    params: Any,
) -> pd.DataFrame:
    """Per-person PIA-proxy and anchor weight for a person-period panel.

    ``panel`` carries ``person_id``, ``period``, ``earnings``; ``anchor``
    carries one row per person with the anchor-period ``weight`` (from
    :func:`anchor_rows` on the full filtered panel). Returns a frame with
    ``person_id``, ``pia_proxy``, ``weight`` (anchor weight) for exactly the
    panel's persons, ordered by ``person_id``. Persons absent from
    ``panel`` do not appear; every panel person appears exactly once
    (zero-positive persons carry ``pia_proxy = 0``).
    """
    weight_of = dict(
        zip(
            anchor["person_id"].to_numpy(),
            anchor["weight"].to_numpy(),
            strict=True,
        )
    )
    rows = []
    for pid, g in panel.groupby("person_id", sort=True):
        proxy = person_pia_proxy(
            g["period"].to_numpy(),
            g["earnings"].to_numpy(dtype=np.float64),
            params,
        )
        rows.append(
            {
                "person_id": int(pid),
                "pia_proxy": float(proxy),
                "weight": float(weight_of[int(pid)]),
            }
        )
    return pd.DataFrame(rows, columns=["person_id", "pia_proxy", "weight"])


# --------------------------------------------------------------------------
# Measurement blocks
# --------------------------------------------------------------------------
def distribution_gaps(
    cand: np.ndarray,
    cand_w: np.ndarray,
    real: np.ndarray,
    real_w: np.ndarray,
) -> dict[str, Any]:
    """Weighted distribution-gap block: two independent PIA-proxy samples.

    Applies to real-vs-candidate (person-aligned, but read here only as two
    weighted samples) and to real-vs-real (disjoint persons). Reports the
    weighted mean and median percent difference, the percent gap at each
    decile, the weighted Gini difference, and the weighted KS distance.
    Percent differences are ``candidate`` relative to ``real``.
    """
    q = np.array(DECILES, dtype=np.float64)
    cand_dec = _weighted_quantile(cand, cand_w, q)
    real_dec = _weighted_quantile(real, real_w, q)
    decile_gaps = {}
    for level, cd, rd in zip(q, cand_dec, real_dec, strict=True):
        decile_gaps[f"d{int(round(level * 10))}"] = {
            "candidate": float(cd),
            "real": float(rd),
            "pct_diff": _pct_diff(float(cd), float(rd)),
        }

    cand_mean = _weighted_mean(cand, cand_w)
    real_mean = _weighted_mean(real, real_w)
    cand_median = float(_weighted_quantile(cand, cand_w, np.array([0.5]))[0])
    real_median = float(_weighted_quantile(real, real_w, np.array([0.5]))[0])

    cand_gini = _weighted_gini(cand, cand_w)
    real_gini = _weighted_gini(real, real_w)

    return {
        "mean": {
            "candidate": cand_mean,
            "real": real_mean,
            "pct_diff": _pct_diff(cand_mean, real_mean),
        },
        "median": {
            "candidate": cand_median,
            "real": real_median,
            "pct_diff": _pct_diff(cand_median, real_median),
        },
        "deciles": decile_gaps,
        "gini": {
            "candidate": cand_gini,
            "real": real_gini,
            "difference": float(cand_gini - real_gini),
        },
        "ks_distance": _weighted_ks(cand, cand_w, real, real_w),
    }


def person_level_errors(
    cand: np.ndarray,
    real: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    """Weighted person-level error block (real vs candidate, same persons).

    Weighted MAE and RMSE of ``candidate - real``; weighted share of persons
    whose candidate PIA-proxy is within 5 percent and within 10 percent of
    their real value. Persons whose real PIA-proxy is zero have an undefined
    percent error; they are counted in ``n_persons`` and ``n_real_zero`` but
    excluded from the within-tolerance denominators (which use only
    real-positive persons), so the shares are well-defined percentages.
    """
    err = cand - real
    total_w = float(np.sum(weight))
    mae = float(np.sum(np.abs(err) * weight) / total_w)
    rmse = float(np.sqrt(np.sum((err**2) * weight) / total_w))

    real_pos = real > 0
    w_pos = weight[real_pos]
    total_w_pos = float(np.sum(w_pos)) if w_pos.size else 0.0
    if total_w_pos > 0:
        pct = np.abs(err[real_pos]) / real[real_pos]
        within5 = float(np.sum(w_pos[pct <= 0.05]) / total_w_pos)
        within10 = float(np.sum(w_pos[pct <= 0.10]) / total_w_pos)
    else:
        within5 = 0.0
        within10 = 0.0

    return {
        "weighted_mae": mae,
        "weighted_rmse": rmse,
        "weighted_share_within_5pct": within5,
        "weighted_share_within_10pct": within10,
        "n_persons": int(real.size),
        "n_real_zero": int(np.sum(~real_pos)),
        "within_tolerance_note": (
            "within-5%/10% shares are over real-POSITIVE persons only "
            "(percent error is undefined at real PIA-proxy 0); MAE and RMSE "
            "are over all persons"
        ),
    }


def anchor_quintile_cutpoints(anchor: pd.DataFrame) -> np.ndarray:
    """Seed-stable anchor-earnings quintile edges from the FULL panel.

    Roughly a weighted fifth of persons have ZERO anchor earnings (no
    earnings in their chronologically last observed period), so a naive
    20/40/60/80 split of all anchor earnings puts the entire zero mass
    exactly at the first edge and the bottom bin is degenerate and unstable
    across seeds. Instead the quintiles are defined once, on the full
    filtered panel, as:

    * **Q0** -- zero anchor earnings (the no-current-earnings group;
      unambiguously the lowest, ~20% by weight),
    * **Q1..Q4** -- the weighted 25/50/75th percentiles of POSITIVE anchor
      earnings.

    Returns the three positive-anchor quartile cut values. Computed on the
    full panel so every seed and every quintile shares identical edges (the
    map is a fixed property of the panel, not the seed's holdout).
    """
    earn = anchor["earnings"].to_numpy(dtype=np.float64)
    w = anchor["weight"].to_numpy(dtype=np.float64)
    pos = earn > 0
    return _weighted_quantile(earn[pos], w[pos], np.array([0.25, 0.5, 0.75]))


def _assign_quintiles(
    anchor: pd.DataFrame,
    person_ids: np.ndarray,
    cutpoints: np.ndarray,
) -> dict[int, int]:
    """Map each person to a quintile 0..4 using the fixed full-panel edges.

    Zero anchor earnings -> Q0; positive anchor earnings -> Q1..Q4 by the
    positive-anchor quartile ``cutpoints`` (``np.searchsorted`` right, so a
    person on a cut goes to the higher bin).
    """
    sub = anchor[anchor["person_id"].isin(set(int(p) for p in person_ids))]
    q_of: dict[int, int] = {}
    for pid, e in zip(
        sub["person_id"].to_numpy(),
        sub["earnings"].to_numpy(dtype=np.float64),
        strict=True,
    ):
        if e <= 0:
            q_of[int(pid)] = 0
        else:
            q = 1 + int(np.searchsorted(cutpoints, e, side="right"))
            q_of[int(pid)] = min(q, N_QUINTILES - 1)
    return q_of


def by_quintile(
    merged: pd.DataFrame,
    anchor: pd.DataFrame,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    """Distribution gaps and person-level errors within each anchor quintile.

    ``merged`` carries ``person_id``, ``pia_real``, ``pia_cand``, ``weight``
    (anchor weight) for the same persons. Quintiles use the seed-stable
    full-panel edges (Q0 = zero anchor earnings; Q1..Q4 = positive-anchor
    quartiles). Answers whether the residual concentrates anywhere along the
    lifetime-earnings distribution.
    """
    q_of = _assign_quintiles(anchor, merged["person_id"].to_numpy(), cutpoints)
    merged = merged.assign(
        quintile=[q_of[int(p)] for p in merged["person_id"].to_numpy()]
    )
    out: dict[str, Any] = {
        "definition": (
            "Q0 = zero anchor earnings (no current-period earnings); "
            "Q1..Q4 = weighted quartiles of positive anchor earnings, "
            "edges fixed once on the full filtered panel (seed-stable)"
        ),
        "positive_anchor_quartile_cutpoints": [float(c) for c in cutpoints],
        "quintiles": {},
    }
    for q in range(N_QUINTILES):
        g = merged[merged["quintile"] == q]
        if len(g) == 0:
            out["quintiles"][f"Q{q}"] = {"n_persons": 0}
            continue
        real = g["pia_real"].to_numpy(dtype=np.float64)
        cand = g["pia_cand"].to_numpy(dtype=np.float64)
        w = g["weight"].to_numpy(dtype=np.float64)
        out["quintiles"][f"Q{q}"] = {
            "n_persons": int(len(g)),
            "distribution": distribution_gaps(cand, w, real, w),
            "person_level": person_level_errors(cand, real, w),
        }
    return out


# --------------------------------------------------------------------------
# Per-seed drivers
# --------------------------------------------------------------------------
def measure_seed_candidate(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    params: Any,
    cutpoints: np.ndarray,
    verbose: bool,
) -> dict[str, Any]:
    """Real-holdout-vs-candidate measurement for one gate seed.

    Regenerates candidate 7 deterministically over the holdout, forms the
    PIA-proxy per person on both the real holdout and the candidate panel
    (same person set, same rows, only earnings differ), and computes the
    distribution-gap, person-level, and by-quintile blocks. ``cutpoints``
    are the seed-stable full-panel positive-anchor quartile edges.
    """
    t0 = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Candidate-7 regeneration (byte-for-byte the merged machinery).
    marginals = fit_cell_marginals(train)
    pools = build_donor_pools(train, all_anchor, marginals)
    fitted, _ = fit_participation_gate(train, seed)
    candidate, _ = generate_candidate(
        holdout, all_anchor, marginals, fitted, pools, seed
    )

    real_px = panel_pia_proxy(holdout, all_anchor, params)
    cand_px = panel_pia_proxy(candidate, all_anchor, params)

    # Same persons on both sides -- align by person_id.
    merged = real_px.merge(
        cand_px, on="person_id", suffixes=("_real", "_cand")
    )
    # The two panel_pia_proxy calls carry identical anchor weights per
    # person, so weight_real == weight_cand; keep one.
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

    result = {
        "seed": seed,
        "n_persons": int(len(merged)),
        "n_person_periods_holdout": int(len(holdout)),
        "n_real_zero_proxy": int(np.sum(real == 0.0)),
        "n_candidate_zero_proxy": int(np.sum(cand == 0.0)),
        "pools": {
            "n_pairs": int(pools["n_pairs"]),
            "n_triples": int(pools["n_triples"]),
            "n_reentry": int(pools["n_reentry"]),
        },
        "distribution": distribution_gaps(cand, w, real, w),
        "person_level": person_level_errors(cand, real, w),
        "by_anchor_quintile": by_quintile(
            quintile_merged, all_anchor, cutpoints
        ),
    }
    if verbose:
        d = result["distribution"]
        pl = result["person_level"]
        print(
            f"  candidate seed {seed}: mean%={d['mean']['pct_diff']:+.2f} "
            f"median%={d['median']['pct_diff']:+.2f} "
            f"KS={d['ks_distance']:.4f} "
            f"gini_diff={d['gini']['difference']:+.4f} "
            f"MAE={pl['weighted_mae']:.1f} "
            f"w5={pl['weighted_share_within_5pct']:.3f} "
            f"({time.time() - t0:.0f}s)"
        )
    return result


def measure_seed_noise(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    params: Any,
    verbose: bool,
) -> dict[str, Any]:
    """Real-vs-real noise-anchor measurement for one gate seed.

    Two disjoint real draws from the seed's TRAIN split via the ctx20
    construction (:mod:`scripts.build_gate1_floor_artifacts`): draw 40% of
    the train persons (``split_panel_by_person``, fraction=0.4,
    seed=1000+s), halve it person-disjointly (fraction=0.5, seed=s), and
    push both halves through the same proxy functional. The two halves are
    DISJOINT persons, so only the distribution-gap block applies;
    person-level errors and by-quintile concentration are n/a across
    disjoint persons.
    """
    t0 = time.time()
    _, train = split_holdout_train(panel, seed)
    forty, _ = hpanel.split_panel_by_person(
        train, "person_id", fraction=0.4, seed=1000 + seed
    )
    side_a, side_b = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=seed
    )

    a_px = panel_pia_proxy(side_a, all_anchor, params)
    b_px = panel_pia_proxy(side_b, all_anchor, params)

    a = a_px["pia_proxy"].to_numpy(dtype=np.float64)
    aw = a_px["weight"].to_numpy(dtype=np.float64)
    b = b_px["pia_proxy"].to_numpy(dtype=np.float64)
    bw = b_px["weight"].to_numpy(dtype=np.float64)

    result = {
        "seed": seed,
        "construction": (
            "ctx20 on the seed's TRAIN split: split_panel_by_person(train, "
            "'person_id', fraction=0.4, seed=1000+s) then "
            "split_panel_by_person(forty, 'person_id', fraction=0.5, "
            "seed=s); side A scored against side B"
        ),
        "n_persons_side_a": int(len(a_px)),
        "n_persons_side_b": int(len(b_px)),
        "distribution": distribution_gaps(a, aw, b, bw),
        "person_level": "n/a (disjoint persons)",
        "by_anchor_quintile": "n/a (disjoint persons)",
    }
    if verbose:
        d = result["distribution"]
        print(
            f"  noise   seed {seed}: mean%={d['mean']['pct_diff']:+.2f} "
            f"median%={d['median']['pct_diff']:+.2f} "
            f"KS={d['ks_distance']:.4f} "
            f"gini_diff={d['gini']['difference']:+.4f} "
            f"(A={len(a_px)},B={len(b_px)}) ({time.time() - t0:.0f}s)"
        )
    return result


# --------------------------------------------------------------------------
# Pooling and context
# --------------------------------------------------------------------------
def _pool_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sd across seeds of each scalar distribution-gap statistic."""

    def collect(path: list[str]) -> list[float]:
        vals = []
        for r in rows:
            node: Any = r["distribution"]
            for key in path:
                node = node[key]
            if node is not None:
                vals.append(float(node))
        return vals

    def summary(vals: list[float]) -> dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
        }

    pooled = {
        "mean_pct_diff": summary(collect(["mean", "pct_diff"])),
        "median_pct_diff": summary(collect(["median", "pct_diff"])),
        "gini_difference": summary(collect(["gini", "difference"])),
        "ks_distance": summary(collect(["ks_distance"])),
    }
    decile_pooled = {}
    for level in DECILES:
        key = f"d{int(round(level * 10))}"
        decile_pooled[key] = summary(collect(["deciles", key, "pct_diff"]))
    pooled["decile_pct_diff"] = decile_pooled
    return pooled


def _pool_person_level(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sd across seeds of each scalar person-level statistic."""

    def summary(key: str) -> dict[str, float]:
        arr = np.array(
            [float(r["person_level"][key]) for r in rows], dtype=np.float64
        )
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
        }

    return {
        "weighted_mae": summary("weighted_mae"),
        "weighted_rmse": summary("weighted_rmse"),
        "weighted_share_within_5pct": summary("weighted_share_within_5pct"),
        "weighted_share_within_10pct": summary("weighted_share_within_10pct"),
    }


def _pool_by_quintile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean across seeds of the by-quintile concentration statistics.

    For each quintile Q0..Q4: the mean over seeds (where the quintile is
    non-empty) of the mean/median % gap, KS distance, Gini difference,
    weighted MAE/RMSE, and the within-5%/10% shares, plus the mean person
    count. Answers, pooled, where the residual concentrates.
    """
    # (stat name -> key path into a per-seed quintile block).
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

    def _dig(block: dict[str, Any], path: tuple[str, ...]) -> float:
        node: Any = block
        for k in path:
            node = node[k]
        return float(node)

    out: dict[str, Any] = {}
    for q in range(N_QUINTILES):
        key = f"Q{q}"
        blocks = [
            r["by_anchor_quintile"]["quintiles"][key]
            for r in rows
            if r["by_anchor_quintile"]["quintiles"][key].get("n_persons", 0)
            > 0
        ]
        if not blocks:
            out[key] = {"n_seeds_present": 0}
            continue
        entry: dict[str, Any] = {"n_seeds_present": len(blocks)}
        for name, path in stat_paths.items():
            entry[name] = float(np.mean([_dig(b, path) for b in blocks]))
        out[key] = entry
    return out


def build_context_row(
    candidate_pooled: dict[str, Any],
    noise_pooled: dict[str, Any],
) -> dict[str, Any]:
    """Place each pooled distributional gap against noise and the +/-5% bar.

    For the mean %, median %, and each decile % gap: the candidate-vs-real
    magnitude, the real-vs-real noise magnitude at the same scale, and two
    verdicts -- whether the candidate gap is within the noise anchor, and
    whether it is within the +/-5% benefit-space criterion. All comparisons
    are on absolute percent magnitude.
    """

    def place(cand_stat: dict[str, float], noise_stat: dict[str, float]):
        cand_abs = abs(cand_stat["mean"])
        noise_abs = abs(noise_stat["mean"])
        return {
            "candidate_pct": cand_stat["mean"],
            "noise_pct": noise_stat["mean"],
            "candidate_abs_pct": cand_abs,
            "noise_abs_pct": noise_abs,
            "within_noise_anchor": bool(cand_abs <= noise_abs),
            "within_5pct_criterion": bool(cand_abs <= SUCCESS_CRITERION_PCT),
        }

    rows = {
        "mean_pct_diff": place(
            candidate_pooled["mean_pct_diff"],
            noise_pooled["mean_pct_diff"],
        ),
        "median_pct_diff": place(
            candidate_pooled["median_pct_diff"],
            noise_pooled["median_pct_diff"],
        ),
    }
    decile_rows = {}
    for level in DECILES:
        key = f"d{int(round(level * 10))}"
        decile_rows[key] = place(
            candidate_pooled["decile_pct_diff"][key],
            noise_pooled["decile_pct_diff"][key],
        )
    rows["deciles"] = decile_rows

    # KS and Gini difference are absolute distances/differences (not percent)
    # -- report candidate vs noise and the "within noise" verdict only.
    rows["ks_distance"] = {
        "candidate": candidate_pooled["ks_distance"]["mean"],
        "noise": noise_pooled["ks_distance"]["mean"],
        "within_noise_anchor": bool(
            candidate_pooled["ks_distance"]["mean"]
            <= noise_pooled["ks_distance"]["mean"]
        ),
    }
    rows["gini_difference"] = {
        "candidate": candidate_pooled["gini_difference"]["mean"],
        "noise": noise_pooled["gini_difference"]["mean"],
        "within_noise_anchor": bool(
            abs(candidate_pooled["gini_difference"]["mean"])
            <= abs(noise_pooled["gini_difference"]["mean"])
        ),
    }

    all_dist_within_noise = all(
        v["within_noise_anchor"]
        for v in [rows["mean_pct_diff"], rows["median_pct_diff"]]
        + list(decile_rows.values())
        + [rows["ks_distance"], rows["gini_difference"]]
    )
    all_dist_within_5pct = all(
        v["within_5pct_criterion"]
        for v in [rows["mean_pct_diff"], rows["median_pct_diff"]]
        + list(decile_rows.values())
    )
    rows["summary"] = {
        "success_criterion_pct": SUCCESS_CRITERION_PCT,
        "success_criterion_text": SUCCESS_CRITERION_TEXT,
        "all_distribution_gaps_within_noise_anchor": bool(
            all_dist_within_noise
        ),
        "all_percent_gaps_within_5pct_criterion": bool(all_dist_within_5pct),
    }
    return rows


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


def _candidate7_reference() -> dict[str, Any]:
    """Reference the committed candidate-7 artifact by content hash + shas."""
    raw = CANDIDATE7_ARTIFACT.read_bytes()
    art = json.loads(raw)
    return {
        "run": art.get("run"),
        "schema_version": art.get("schema_version"),
        "spec_registration": art.get("spec_registration"),
        "verdict": art.get("verdict"),
        "revision_pins": art.get("revision_pins"),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(CANDIDATE7_ARTIFACT.relative_to(ROOT)),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full downstream-relevance measurement (reported, not gated)."""
    started = time.time()

    # Oracle parameters loaded ONCE; the loader records the pe-us revision.
    params = load_ssa_parameters()
    if verbose:
        print(f"SSA oracle parameters: pe_us_revision={params.pe_us_revision}")

    panel = load_filtered_panel()
    all_anchor = anchor_rows(panel)
    # Seed-stable anchor-earnings quintile edges, fixed once on the full
    # filtered panel (Q0 = zero anchor earnings; Q1..Q4 = positive-anchor
    # quartiles), so every seed's concentration block shares identical bins.
    quintile_cutpoints = anchor_quintile_cutpoints(all_anchor)
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons; "
            f"{len(all_anchor)} anchors; positive-anchor quartile cuts "
            f"{[round(float(c)) for c in quintile_cutpoints]}"
        )

    candidate_seeds: list[dict[str, Any]] = []
    noise_seeds: list[dict[str, Any]] = []
    for seed in SEEDS:
        candidate_seeds.append(
            measure_seed_candidate(
                seed, panel, all_anchor, params, quintile_cutpoints, verbose
            )
        )
        noise_seeds.append(
            measure_seed_noise(seed, panel, all_anchor, params, verbose)
        )

    candidate_pooled = {
        "distribution": _pool_distribution(candidate_seeds),
        "person_level": _pool_person_level(candidate_seeds),
        "by_anchor_quintile": _pool_by_quintile(candidate_seeds),
    }
    noise_pooled = {"distribution": _pool_distribution(noise_seeds)}

    context = build_context_row(
        candidate_pooled["distribution"], noise_pooled["distribution"]
    )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "downstream_relevance_c7_v1",
        "reported_not_gated": True,
        "purpose": (
            "Translate the candidate-7 classifier residual into benefit "
            "space (a statute-shaped PIA proxy) to inform whether a future "
            "public gate amendment is warranted; this reads no gate and "
            "changes no gate."
        ),
        "not_full_415b": (
            "The functional is a STATUTE-SHAPED PROXY, not the full 42 USC "
            "415(b) AIME: the filtered panel's careers are partial (biennial "
            "PSID, prime age 25-59, reference years 1998-2022), so a faithful "
            "highest-35 AIME cannot be formed. The proxy is monotone in "
            "lifetime earnings, routes through the exact 415(a)/415(g) PIA "
            "bend-point formula, and uses a constant scale (top min(10,n) "
            "indexed years / (count*12*2)) that cancels in every "
            "real-vs-candidate comparison."
        ),
        "functional": {
            "steps": [
                "positive-earnings periods only, within the locked filter "
                "(age 25-59, periods 1998-2022)",
                "cap each year at that year's wage base (wage_base_for)",
                "index each to 2022 by nawi[2022]/nawi[year]",
                "AIME-proxy = sum(top min(10, n_pos) indexed) / "
                "(min(10, n_pos) * 12 * 2)",
                "PIA-proxy = pia(AIME-proxy, 2022, params) "
                "[2022-eligibility 415(a)/415(g)]",
                "zero positive observations -> PIA-proxy 0 (person included)",
            ],
            "eligibility_year": ELIGIBILITY_YEAR,
            "top_n": PROXY_TOP_N,
            "biennial_scale": _BIENNIAL_SCALE,
            "weight": (
                "each person weighted by their anchor-period weight (the "
                "weight on their chronologically last observed period)"
            ),
            "person_set": (
                "both sides generate the same person set on the same "
                "observed periods; only earnings differ (candidate anchor "
                "held at real earnings)"
            ),
            "anchor_quintiles": {
                "definition": (
                    "Q0 = zero anchor earnings (no current-period earnings, "
                    "~20% by weight); Q1..Q4 = weighted quartiles of "
                    "positive anchor earnings, edges fixed once on the full "
                    "filtered panel so bins are seed-stable"
                ),
                "positive_anchor_quartile_cutpoints": [
                    float(c) for c in quintile_cutpoints
                ],
            },
        },
        "oracle": {
            "source": "policyengine-us (loaded once via load_ssa_parameters)",
            "pe_us_revision": params.pe_us_revision,
            "pe_us_dir_env": "POPULACE_DYNAMICS_PE_US_DIR",
            "eligibility_year": ELIGIBILITY_YEAR,
            "bend_points_2022": list(params.bend_points(ELIGIBILITY_YEAR)),
            "pia_factors": list(params.pia_factors),
            "nawi_2022": params.nawi[ELIGIBILITY_YEAR],
        },
        "candidate7_reference": _candidate7_reference(),
        "protocol": {
            "seeds": list(SEEDS),
            "split": (
                "per gate seed s: the locked person-disjoint 0.2 holdout "
                "(split_holdout_train); candidate 7 regenerated over the "
                "holdout via the merged machinery; real holdout and "
                "candidate pushed through the proxy"
            ),
            "noise_anchor": (
                "real vs real at the same scale: the ctx20 construction on "
                "the seed's TRAIN split (fraction=0.4 seed=1000+s, then "
                "fraction=0.5 seed=s), both disjoint halves pushed through "
                "the proxy; distribution gaps only (person-level and "
                "by-quintile are n/a across disjoint persons)"
            ),
            "measurements": [
                "distribution gaps of PIA-proxy: mean/median % diff, decile "
                "% gaps, weighted Gini difference, weighted KS distance",
                "person-level (real vs candidate, same persons): weighted "
                "MAE and RMSE of (candidate - real), weighted share within "
                "5% and within 10% of real",
                "the same by anchor-earnings quintile (concentration)",
                "noise anchor: identical distribution measures for "
                "real-vs-real",
            ],
        },
        "success_criterion": {
            "pct": SUCCESS_CRITERION_PCT,
            "text": SUCCESS_CRITERION_TEXT,
        },
        "candidate_per_seed": candidate_seeds,
        "noise_per_seed": noise_seeds,
        "candidate_pooled": candidate_pooled,
        "noise_pooled": noise_pooled,
        "context": context,
        "revision_pins": {
            "populace_dynamics_sha": _sha(ROOT),
            "pe_us_revision": params.pe_us_revision,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        s = context["summary"]
        print(
            "\nSUMMARY: all distribution gaps within noise anchor = "
            f"{s['all_distribution_gaps_within_noise_anchor']}; "
            "all percent gaps within +/-5% = "
            f"{s['all_percent_gaps_within_5pct_criterion']}"
        )
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
