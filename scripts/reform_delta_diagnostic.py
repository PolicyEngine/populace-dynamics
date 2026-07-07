"""Reform-delta diagnostic: do synthetic histories reproduce reform incidence?

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It is
the run-13 analogue of the PR #56 downstream-relevance artifact, extended from
benefit-space *levels* to reform *deltas*. The gate-passing generator
(candidate 11, run 13) produces earnings histories whose PIA-proxy LEVELS
match real histories inside the anchor floor; policy simulation needs more.
This diagnostic asks whether the SAME synthetic histories reproduce the
*impact of a reform* -- the per-person and distributional benefit deltas --
that the real histories imply, on the locked protocol. It gates nothing; the
strongest claim it can support is "consistent with the real-vs-real floor".

Frozen spec: issue #42 comment 4906177609 (registration URL below). Where this
module and the registration disagree, the registration wins.

Model. Candidate 11 exactly as committed: the candidate-10/11 generation
machinery is IMPORTED byte-for-byte (:func:`run_gate1_candidate10.run_seed`'s
fit+generate prefix -- stage-1 marginals, candidate-8 donor permanent rank
u_w, candidate-9 donor pools + shared and zero-anchor participation gates, the
candidate-10 backward k-NN chain at the fixed lambda=0.1, full re-entry pool,
Q0 memory-exempt). Nothing in the generation is re-implemented and no modeling
constant moves. Locked seeds 0-4, filter-first, person-disjoint 20% holdouts,
train-complement fits, candidate trajectories on the holdout support.

Benefit functional. :func:`build_downstream_relevance.person_pia_proxy` /
``panel_pia_proxy``, unchanged (statute-shaped proxy, NOT the full 42 USC
415(b); the committed disclaimer is carried verbatim in :data:`NOT_FULL_415B`).
A reform enters ONLY through a thin params wrapper delegating to the pinned
:class:`~populace_dynamics.ss.params.SSAParameters`; the pinned oracle modules
(``ss/params.py``, ``ss/benefits.py``, ``build_downstream_relevance.py``) are
NOT edited.

* **Reform A (bottom-loaded): first PIA factor 0.90 -> 0.95.** The wrapper
  overrides ``pia_factors -> (0.95, f2, f3)`` where ``f2``/``f3`` are the
  baseline's own values, read from the loaded params at runtime (never
  hardcoded). Illustrative mechanical reform; incidence concentrates in the
  low deciles / concave region. No policy position implied.
* **Reform B (top-loaded): eliminate the taxable-maximum cap in the AIME
  step** (benefit side only). The wrapper overrides
  ``wage_base_for(year) -> float('inf')``; ``pia_factors`` and the bend points
  delegate to baseline. Opposite-incidence shape (gains concentrate at the
  cap).

Measurement. Per person: Delta = reform PIA-proxy - baseline PIA-proxy (2022
monthly dollars, and % of the side's weighted-mean baseline). Per reform, per
seed, real holdout vs generated candidate, anchor-weighted (each person at the
weight of their chronologically last observed period):

1. weighted mean Delta and median Delta (generated-vs-real gap);
2. incidence curve: mean Delta by DECILE GROUP of the side's OWN weighted
   baseline distribution (d1-d10 recorded; the d3-d9 gated-decile convention
   is the comparison set; d1/d2/d10 are reported, not compared -- the
   near-zero-baseline bottom and the fragile top);
3. winners share (Delta > $1/month);
4. zero-anchor (Q0) subgroup weighted mean Delta, per-seed signed for both
   sides and pooled across seeds;
5. paired per-person corr(Delta_real, Delta_gen) and weighted mean
   |Delta_gen - Delta_real| -- REPORTED DESCRIPTIVELY ONLY (labelled so in the
   artifact): generation conditions on the anchor, so within-person draw noise
   dilutes paired agreement, and no real-vs-real analogue exists (a person has
   one real history).

Floor (comparison basis for metrics 1-4). For each reform, the real-vs-real
person-disjoint half-split analogue -- the :mod:`build_pia_proxy_floor`
construction (ctx20: per seed draw 40% of persons, fraction=0.4 seed=1000+s,
halve it person-disjointly fraction=0.5 seed=s, giving two DISJOINT ~20%
real samples A and B) -- with the SAME Delta metrics, same 5 seeds. The A-vs-B
gap defines the noise scale for the real-vs-generated gap. "Consistent with
the floor" is the strongest claim; there is no pass/fail verdict. The paired
stats (metric 5) have no floor: A and B are disjoint persons, so no person
alignment exists.

Scope honesty. PSID holdout persons on their observed support -- a validation
of reform-delta fidelity, NOT a deployment simulation: no CPS integration, no
demographics/claiming (gate-2/3 territory), no aggregate cost scoring against
SSA baselines. Artifact ``runs/reform_delta_diagnostic_v1.json``,
``reported_not_gated: true``; the evidence PR publishes regardless of how the
comparison lands.

Environment. Candidate generation needs populace-fit (the participation gates
are ``RegimeGatedQRF`` sign gates); the proxy needs the SSA oracle
(``POPULACE_DYNAMICS_PE_US_DIR`` -> the pinned policyengine-us checkout). Run
from the repository root with the PSID family files staged, in the DEDICATED
gate venv (populace-fit pins scikit-learn < 1.9)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/reform_delta_diagnostic.py
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The pinned PIA-proxy functional and its weighted-statistic helpers, plus the
# seed-stable anchor-quintile machinery, imported byte-for-byte from the merged
# downstream-relevance builder -- single source of truth (PR #56). Importing it
# does NOT pull populace.fit (that chain is deferred inside the candidate-fit
# functions this module imports lazily).
from build_downstream_relevance import (  # noqa: E402
    ELIGIBILITY_YEAR,
    _assign_quintiles,
    _weighted_mean,
    _weighted_quantile,
    anchor_quintile_cutpoints,
    panel_pia_proxy,
)

# The locked protocol: seeds, the filter-first panel load, and the
# person-disjoint 0.2 holdout/train split (imported from the baseline runner).
from run_gate1_baseline import (  # noqa: E402
    SEEDS,
    load_filtered_panel,
    split_holdout_train,
)

# The anchor rows (one row per person = their chronologically last observed
# period) supply each person's anchor weight and their zero-anchor flag.
from run_gate1_candidate5b import anchor_rows  # noqa: E402

from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "reform_delta_diagnostic_v1.json"
ARTIFACT_SCHEMA_VERSION = "reform_delta_diagnostic.v1"
RUN_NAME = "reform_delta_diagnostic_v1"

#: This diagnostic's frozen-spec registration (issue #42 comment 4906177609).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4906177609"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4906177609"

#: The candidate-11 (run 13) gate artifact whose generator this diagnostic
#: reuses byte-for-byte; referenced for provenance.
CANDIDATE11_ARTIFACT = ROOT / "runs" / "gate1_rank_knn_v5.json"

#: The proxy-not-full-415(b) disclaimer, carried VERBATIM from the pinned
#: functional's home (scripts/build_downstream_relevance.py / the committed
#: runs/downstream_relevance_c7_v1.json). A test binds this to the committed
#: downstream artifact so any drift is caught.
NOT_FULL_415B = (
    "The functional is a STATUTE-SHAPED PROXY, not the full 42 USC "
    "415(b) AIME: the filtered panel's careers are partial (biennial "
    "PSID, prime age 25-59, reference years 1998-2022), so a faithful "
    "highest-35 AIME cannot be formed. The proxy is monotone in "
    "lifetime earnings, routes through the exact 415(a)/415(g) PIA "
    "bend-point formula, and uses a constant scale (top min(10,n) "
    "indexed years / (count*12*2)) that cancels in every "
    "real-vs-candidate comparison."
)

#: The zero-anchor-earnings subgroup index (Q0) in the seed-stable full-panel
#: anchor-earnings quintiles (byte-for-byte the floor convention).
Q0_INDEX = 0
#: Winner threshold: a person is a winner if their monthly Delta exceeds $1.
WINNER_THRESHOLD_DOLLARS = 1.0
#: Number of decile GROUPS for the incidence curve (mean Delta by own-baseline
#: decile). d1 = lowest-baseline tenth ... d10 = highest.
N_DECILE_GROUPS = 10
#: The gated-decile comparison convention: d3-d9 are compared against the
#: floor; d1/d2 (near-zero baseline) and d10 (fragile top) are reported only.
COMPARED_DECILES = ("d3", "d4", "d5", "d6", "d7", "d8", "d9")
REPORTED_ONLY_DECILES = ("d1", "d2", "d10")
#: Absolute tolerance for the DESCRIPTIVE ``within_floor`` boolean only. Where
#: a reform leaves a decile's Delta exactly constant (e.g. Reform A's flat
#: first-bracket bump above the first bend point), both the real-vs-gen gap and
#: the floor gap are 0 up to floating-point residue (~1e-14 from weighted means
#: of a constant); this guard keeps a 0-vs-0 knife-edge reading "within" rather
#: than flipping on fp noise. It is far below every real gap (>= ~1e-3) and
#: touches no scale value -- only the annotation flag.
WITHIN_FLOOR_ABS_TOL = 1e-9


# ==========================================================================
# Reform params wrappers (thin; delegate to the loaded SSAParameters)
# ==========================================================================
class _ReformParams:
    """Thin wrapper delegating every attribute to the loaded SSAParameters.

    The pinned :class:`SSAParameters` is a frozen dataclass; a reform enters
    ONLY by overriding the single method/attribute it changes, delegating
    everything else. The pinned oracle modules are never edited.
    """

    def __init__(self, base: Any) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes not defined on the wrapper itself.
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)


class ReformA(_ReformParams):
    """Bottom-loaded: first PIA factor 0.90 -> 0.95.

    ``pia_factors`` becomes ``(0.95, f2, f3)`` where ``f2``/``f3`` are the
    baseline's own second/third factors, read from the loaded params (never
    hardcoded). The bend points and the wage base delegate to baseline.
    """

    def __init__(self, base: Any) -> None:
        super().__init__(base)
        _f1, f2, f3 = base.pia_factors
        self.pia_factors = (0.95, f2, f3)


class ReformB(_ReformParams):
    """Top-loaded: eliminate the taxable-maximum cap in the AIME step.

    ``wage_base_for(year)`` returns ``inf`` (no capping in the proxy's
    creditable-earnings step); ``pia_factors`` and the bend points delegate to
    baseline. Benefit side only.
    """

    def wage_base_for(self, year: int) -> float:
        return float("inf")


#: Registry of the reforms (name -> wrapper factory), in report order.
REFORMS: dict[str, type[_ReformParams]] = {
    "reform_a": ReformA,
    "reform_b": ReformB,
}


# ==========================================================================
# Per-person deltas and the anchor-weighted delta metrics
# ==========================================================================
def panel_deltas(
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    baseline_params: Any,
    reform_params: Any,
) -> pd.DataFrame:
    """Per-person baseline & reform PIA-proxy and their Delta on one panel.

    Both proxies use the SAME pinned :func:`panel_pia_proxy`; the reform enters
    only through ``reform_params``. Returns a frame with ``person_id``,
    ``weight`` (anchor weight), ``baseline``, ``reform``, ``delta`` over
    exactly the panel's persons (one row each; a zero-positive-earnings person
    carries ``baseline = reform = 0`` and ``delta = 0``). Both calls group by
    ``person_id`` with ``sort=True``, so the two frames align row-for-row.
    """
    base_px = panel_pia_proxy(panel, all_anchor, baseline_params)
    reform_px = panel_pia_proxy(panel, all_anchor, reform_params)
    assert np.array_equal(
        base_px["person_id"].to_numpy(), reform_px["person_id"].to_numpy()
    ), "baseline and reform proxy person orders diverged"
    assert np.allclose(
        base_px["weight"].to_numpy(), reform_px["weight"].to_numpy()
    ), "baseline and reform anchor weights diverged"
    out = pd.DataFrame(
        {
            "person_id": base_px["person_id"].to_numpy(),
            "weight": base_px["weight"].to_numpy(dtype=np.float64),
            "baseline": base_px["pia_proxy"].to_numpy(dtype=np.float64),
            "reform": reform_px["pia_proxy"].to_numpy(dtype=np.float64),
        }
    )
    out["delta"] = out["reform"] - out["baseline"]
    return out


def _weighted_decile_groups(
    values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Assign each element to a decile GROUP 0..9 of the weighted distribution.

    Sort by value, place each atom at the midpoint of its normalized
    cumulative weight (the same midpoint plotting-position convention the
    harness :func:`_weighted_quantile` uses), and floor 10x that position into
    ``0..9``. A mass point (e.g. the zero-baseline persons) fills the lowest
    groups by cumulative weight; group membership is a partition of ~a tenth of
    the weight each.
    """
    order = np.argsort(values, kind="stable")
    w = weights[order].astype(np.float64)
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    positions = (cumulative - 0.5 * w) / total
    group_sorted = np.clip((positions * N_DECILE_GROUPS).astype(int), 0, 9)
    groups = np.empty(values.shape[0], dtype=np.int64)
    groups[order] = group_sorted
    return groups


def _q0_mean_delta(
    df: pd.DataFrame, all_anchor: pd.DataFrame, cutpoints: np.ndarray
) -> tuple[float | None, int]:
    """Anchor-weighted mean Delta over the zero-anchor (Q0) persons of ``df``.

    Q0 membership uses the seed-stable full-panel anchor-earnings quintile
    edges (Q0 = zero anchor earnings), byte-for-byte the floor convention.
    Returns ``(mean_delta_or_None, n_q0_persons)``.
    """
    q_of = _assign_quintiles(all_anchor, df["person_id"].to_numpy(), cutpoints)
    mask = np.array(
        [q_of[int(p)] == Q0_INDEX for p in df["person_id"].to_numpy()],
        dtype=bool,
    )
    if not np.any(mask):
        return None, 0
    d = df["delta"].to_numpy(dtype=np.float64)[mask]
    w = df["weight"].to_numpy(dtype=np.float64)[mask]
    return _weighted_mean(d, w), int(mask.sum())


def delta_metrics(
    df: pd.DataFrame, all_anchor: pd.DataFrame, cutpoints: np.ndarray
) -> dict[str, Any]:
    """Anchor-weighted Delta metrics on ONE side (real / generated / a floor half).

    All metrics are weighted by the anchor weight. ``decile_mean_delta`` bins
    persons into ten groups by the side's OWN weighted baseline distribution
    and reports the weighted mean Delta within each; ``q0_mean_delta`` is the
    zero-anchor subgroup weighted mean Delta. Percent figures are the Delta as
    a percent of the side's weighted-mean baseline (``None`` when that is 0).
    """
    d = df["delta"].to_numpy(dtype=np.float64)
    b = df["baseline"].to_numpy(dtype=np.float64)
    w = df["weight"].to_numpy(dtype=np.float64)
    total_w = float(np.sum(w))

    mean_delta = _weighted_mean(d, w)
    median_delta = float(_weighted_quantile(d, w, np.array([0.5]))[0])
    mean_baseline = _weighted_mean(b, w)

    groups = _weighted_decile_groups(b, w)
    decile_mean_delta: dict[str, float | None] = {}
    for k in range(N_DECILE_GROUPS):
        sel = groups == k
        decile_mean_delta[f"d{k + 1}"] = (
            _weighted_mean(d[sel], w[sel]) if np.any(sel) else None
        )

    winners_share = float(np.sum(w[d > WINNER_THRESHOLD_DOLLARS]) / total_w)
    q0_mean_delta, n_q0 = _q0_mean_delta(df, all_anchor, cutpoints)

    return {
        "n_persons": int(len(df)),
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "mean_baseline": mean_baseline,
        "mean_delta_pct_of_mean_baseline": (
            100.0 * mean_delta / mean_baseline
            if mean_baseline != 0.0
            else None
        ),
        "median_delta_pct_of_mean_baseline": (
            100.0 * median_delta / mean_baseline
            if mean_baseline != 0.0
            else None
        ),
        "decile_mean_delta": decile_mean_delta,
        "winners_share": winners_share,
        "q0_mean_delta": q0_mean_delta,
        "n_q0_persons": int(n_q0),
    }


def _weighted_corr(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> float | None:
    """Weighted Pearson correlation; ``None`` if either side is degenerate."""
    w = w.astype(np.float64)
    sw = float(np.sum(w))
    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    cov = float(np.sum(w * (x - mx) * (y - my)) / sw)
    vx = float(np.sum(w * (x - mx) ** 2) / sw)
    vy = float(np.sum(w * (y - my) ** 2) / sw)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return float(cov / np.sqrt(vx * vy))


def paired_stats(
    real_df: pd.DataFrame, gen_df: pd.DataFrame
) -> dict[str, Any]:
    """Paired per-person Delta agreement (real vs generated, same persons).

    REPORTED DESCRIPTIVELY ONLY. Weighted corr(Delta_real, Delta_gen) and the
    weighted mean |Delta_gen - Delta_real| over the shared holdout persons.
    Generation conditions on the anchor, so within-person draw noise dilutes
    the paired agreement, and there is no real-vs-real analogue (a person has
    one real history) -- hence no floor for these.
    """
    merged = real_df[["person_id", "weight", "delta"]].merge(
        gen_df[["person_id", "delta"]],
        on="person_id",
        suffixes=("_real", "_gen"),
    )
    dr = merged["delta_real"].to_numpy(dtype=np.float64)
    dg = merged["delta_gen"].to_numpy(dtype=np.float64)
    w = merged["weight"].to_numpy(dtype=np.float64)
    return {
        "corr": _weighted_corr(dr, dg, w),
        "weighted_mean_abs_diff": _weighted_mean(np.abs(dg - dr), w),
        "n_persons": int(len(merged)),
        "note": (
            "DESCRIPTIVE ONLY: generation conditions on the anchor so "
            "within-person draw noise dilutes paired agreement; no "
            "real-vs-real analogue exists (one real history per person), so "
            "these carry no floor and no verdict"
        ),
    }


# ==========================================================================
# Candidate-11 fit + generate (imported machinery; nothing re-implemented)
# ==========================================================================
def fit_and_generate_candidate11(
    seed: int, panel: pd.DataFrame, all_anchor: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit candidate 11 on the seed's train split; generate the holdout panel.

    Byte-for-byte the fit+generate prefix of
    :func:`run_gate1_candidate10.run_seed` (candidate 11 reuses candidate 10's
    generation exactly): stage-1 per-cell marginals, the candidate-8 donor
    permanent rank ``u_w``, the candidate-9 donor pools and the shared +
    zero-anchor participation gates, then the candidate-10 backward k-NN chain
    (fixed lambda=0.1, full re-entry pool, Q0 memory-exempt). The candidate/8
    modules are imported here (not at module top) so the artifact-consistency
    tests import this builder WITHOUT pulling the populace-fit chain, exactly
    as the run_seed pattern defers it. The generation is deterministic from the
    seed and does NOT depend on the reform. Returns ``(holdout, candidate)``;
    the candidate holds exactly the holdout persons on their observed periods,
    only earnings generated, anchor kept at real earnings.
    """
    import run_gate1_candidate8 as c8
    import run_gate1_candidate10 as c10

    holdout, train = split_holdout_train(panel, seed)
    marginals = c10.fit_cell_marginals(train)
    uw = c8.build_donor_uw(train, marginals)
    pools = c10.build_donor_pools(
        train, all_anchor, marginals, uw["u_w_of_person"]
    )
    fitted_shared, _pairs = c10.fit_participation_gate(train, seed)
    fitted_zero, _n_zero = c10.fit_zero_anchor_participation_gate(
        train, all_anchor, seed
    )
    candidate, _diagnostics = c10.generate_candidate(
        holdout, all_anchor, marginals, fitted_shared, fitted_zero, pools, seed
    )
    return holdout, candidate


# ==========================================================================
# Per-seed measurement (real-vs-generated and the real-vs-real floor)
# ==========================================================================
def measure_realgen_seed(
    reform_params: Any,
    baseline_params: Any,
    all_anchor: pd.DataFrame,
    cutpoints: np.ndarray,
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    """Real-holdout-vs-generated Delta metrics for one seed and one reform.

    The candidate is the holdout persons resimulated (same person set, same
    observed periods, anchor kept), so the two sides are person-aligned. Both
    the distributional metrics (per side) and the paired descriptive stats are
    computed; the reform enters only through ``reform_params``.
    """
    real_df = panel_deltas(holdout, all_anchor, baseline_params, reform_params)
    gen_df = panel_deltas(
        candidate, all_anchor, baseline_params, reform_params
    )
    return {
        "real": delta_metrics(real_df, all_anchor, cutpoints),
        "generated": delta_metrics(gen_df, all_anchor, cutpoints),
        "paired": paired_stats(real_df, gen_df),
    }


def measure_floor_seed(
    seed: int,
    reform_params: Any,
    baseline_params: Any,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    """Real-vs-real Delta-metric floor for one seed and one reform.

    The :mod:`build_pia_proxy_floor` ctx20 construction on the FULL filtered
    panel: draw 40% of persons (``split_panel_by_person`` fraction=0.4
    seed=1000+s), halve it person-disjointly (fraction=0.5 seed=s), giving two
    DISJOINT ~20%-of-persons real samples A and B, and push both through the
    same baseline/reform proxy. The two halves are disjoint persons, so only
    the distributional Delta metrics apply (no paired stats). The A-vs-B gap is
    the noise scale for the real-vs-generated gap at the same scale.
    """
    forty, _ = hpanel.split_panel_by_person(
        panel, "person_id", fraction=0.4, seed=1000 + seed
    )
    side_a, side_b = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=seed
    )
    a_df = panel_deltas(side_a, all_anchor, baseline_params, reform_params)
    b_df = panel_deltas(side_b, all_anchor, baseline_params, reform_params)
    return {
        "side_a": delta_metrics(a_df, all_anchor, cutpoints),
        "side_b": delta_metrics(b_df, all_anchor, cutpoints),
    }


# ==========================================================================
# Pooling (mean/sd across seeds; every headline recomputes from per-seed)
# ==========================================================================
def _summary(values: list[float]) -> dict[str, Any]:
    """Mean/sd/min/max/n and the raw per-seed values (float64, ddof=1 sd).

    Byte-for-byte the committed-floor convention
    (:func:`build_pia_proxy_floor._summary`). An empty list yields a
    zero/empty summary (guards degenerate subgroups).
    """
    arr = np.array([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "sd": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n_seeds": 0,
            "values": [],
        }
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
        "values": [float(v) for v in arr],
    }


def _pool_gap(signed: list[float | None]) -> dict[str, Any]:
    """Pool a list of per-seed signed gaps into abs and signed summaries.

    ``abs.mean`` is the typical per-seed magnitude (the noise/discrepancy
    scale for the mean/median/winners/decile metrics).
    ``abs_of_pooled_signed_mean`` is ``|mean_s(signed)|`` (the pooled-signed
    magnitude, the scale used for the Q0 metric, where the per-seed subgroup is
    noisy but the across-seed mean cancels for real-vs-real). ``None`` entries
    (a degenerate subgroup on some seed) are dropped and the surviving count is
    recorded.
    """
    kept = [float(v) for v in signed if v is not None]
    return {
        "per_seed_signed": kept,
        "abs": _summary([abs(v) for v in kept]),
        "signed": _summary(kept),
        "abs_of_pooled_signed_mean": (
            abs(float(np.mean(kept))) if kept else 0.0
        ),
    }


def _side_pool(
    rows: list[dict[str, Any]], side: str, key: str
) -> dict[str, Any]:
    """Across-seed summary of one side's scalar metric (drops ``None``)."""
    return _summary([r[side][key] for r in rows if r[side][key] is not None])


def _scalar_gap_block(
    rows: list[dict[str, Any]],
    floor_rows: list[dict[str, Any]],
    key: str,
    pooled_signed: bool,
) -> dict[str, Any]:
    """Real-vs-gen vs real-vs-real floor comparison for one scalar metric.

    ``rows`` are the per-seed real/generated blocks; ``floor_rows`` the per-seed
    side_a/side_b blocks. The real-vs-gen per-seed gap is
    ``generated - real``; the floor per-seed gap is ``side_a - side_b``. When
    ``pooled_signed`` (the Q0 metric) the compared scale is
    ``|mean_s(gap)|``; otherwise it is ``mean_s(|gap|)``.
    """
    realgen_gap = _pool_gap(
        [
            (
                (r["generated"][key] - r["real"][key])
                if (
                    r["generated"][key] is not None
                    and r["real"][key] is not None
                )
                else None
            )
            for r in rows
        ]
    )
    floor_gap = _pool_gap(
        [
            (
                (f["side_a"][key] - f["side_b"][key])
                if (
                    f["side_a"][key] is not None
                    and f["side_b"][key] is not None
                )
                else None
            )
            for f in floor_rows
        ]
    )
    if pooled_signed:
        realgen_scale = realgen_gap["abs_of_pooled_signed_mean"]
        floor_scale = floor_gap["abs_of_pooled_signed_mean"]
        scale_kind = "abs_of_pooled_signed_mean (|across-seed mean of gap|)"
    else:
        realgen_scale = realgen_gap["abs"]["mean"]
        floor_scale = floor_gap["abs"]["mean"]
        scale_kind = "abs.mean (mean of per-seed |gap|)"
    return {
        "real_pooled": _side_pool(rows, "real", key),
        "generated_pooled": _side_pool(rows, "generated", key),
        "floor_side_a_pooled": _side_pool(floor_rows, "side_a", key),
        "floor_side_b_pooled": _side_pool(floor_rows, "side_b", key),
        "realgen_gap": realgen_gap,
        "floor_gap": floor_gap,
        "scale_kind": scale_kind,
        "realgen_scale": realgen_scale,
        "floor_scale": floor_scale,
        "within_floor": bool(
            realgen_scale <= floor_scale + WITHIN_FLOOR_ABS_TOL
        ),
    }


def _decile_gap_block(
    rows: list[dict[str, Any]], floor_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-decile incidence-curve gap vs the floor, plus the d3-d9 headline.

    For each decile group d1-d10: the real-vs-gen per-seed gap (generated -
    real mean Delta in that group) and the floor per-seed gap (side_a -
    side_b), pooled to ``mean_s(|gap|)``. The gated-decile convention compares
    d3-d9; d1/d2/d10 are reported only. The headline is the MAX over d3-d9 of
    the real-vs-gen scale against the MAX over d3-d9 of the floor scale.
    """
    per_decile: dict[str, Any] = {}
    for k in range(N_DECILE_GROUPS):
        dkey = f"d{k + 1}"
        realgen_gap = _pool_gap(
            [
                (
                    (
                        r["generated"]["decile_mean_delta"][dkey]
                        - r["real"]["decile_mean_delta"][dkey]
                    )
                    if (
                        r["generated"]["decile_mean_delta"][dkey] is not None
                        and r["real"]["decile_mean_delta"][dkey] is not None
                    )
                    else None
                )
                for r in rows
            ]
        )
        floor_gap = _pool_gap(
            [
                (
                    (
                        f["side_a"]["decile_mean_delta"][dkey]
                        - f["side_b"]["decile_mean_delta"][dkey]
                    )
                    if (
                        f["side_a"]["decile_mean_delta"][dkey] is not None
                        and f["side_b"]["decile_mean_delta"][dkey] is not None
                    )
                    else None
                )
                for f in floor_rows
            ]
        )
        per_decile[dkey] = {
            "real_pooled_mean_delta": _summary(
                [
                    r["real"]["decile_mean_delta"][dkey]
                    for r in rows
                    if r["real"]["decile_mean_delta"][dkey] is not None
                ]
            ),
            "generated_pooled_mean_delta": _summary(
                [
                    r["generated"]["decile_mean_delta"][dkey]
                    for r in rows
                    if r["generated"]["decile_mean_delta"][dkey] is not None
                ]
            ),
            "realgen_gap": realgen_gap,
            "floor_gap": floor_gap,
            "realgen_scale": realgen_gap["abs"]["mean"],
            "floor_scale": floor_gap["abs"]["mean"],
            "within_floor": bool(
                realgen_gap["abs"]["mean"]
                <= floor_gap["abs"]["mean"] + WITHIN_FLOOR_ABS_TOL
            ),
            "compared": dkey in COMPARED_DECILES,
        }
    max_realgen = max(per_decile[d]["realgen_scale"] for d in COMPARED_DECILES)
    max_floor = max(per_decile[d]["floor_scale"] for d in COMPARED_DECILES)
    argmax_realgen = max(
        COMPARED_DECILES, key=lambda d: per_decile[d]["realgen_scale"]
    )
    return {
        "scale_kind": "abs.mean (mean of per-seed |gap|) per decile group",
        "compared_deciles": list(COMPARED_DECILES),
        "reported_only_deciles": list(REPORTED_ONLY_DECILES),
        "per_decile": per_decile,
        "max_realgen_scale_d3_d9": max_realgen,
        "max_floor_scale_d3_d9": max_floor,
        "argmax_realgen_decile_d3_d9": argmax_realgen,
        "within_floor_d3_d9": bool(
            max_realgen <= max_floor + WITHIN_FLOOR_ABS_TOL
        ),
    }


def _paired_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Across-seed pool of the DESCRIPTIVE paired stats (corr, mean |diff|)."""
    return {
        "note": (
            "DESCRIPTIVE ONLY -- no floor, no verdict. Generation conditions "
            "on the anchor so within-person draw noise dilutes the paired "
            "agreement; a person has one real history, so no real-vs-real "
            "paired analogue exists."
        ),
        "corr": _summary(
            [
                r["paired"]["corr"]
                for r in rows
                if r["paired"]["corr"] is not None
            ]
        ),
        "weighted_mean_abs_diff": _summary(
            [r["paired"]["weighted_mean_abs_diff"] for r in rows]
        ),
    }


def pool_reform(
    rows: list[dict[str, Any]], floor_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Pooled comparison for one reform: metrics 1-4 vs floor, 5 descriptive."""
    return {
        "mean_delta": _scalar_gap_block(
            rows, floor_rows, "mean_delta", pooled_signed=False
        ),
        "median_delta": _scalar_gap_block(
            rows, floor_rows, "median_delta", pooled_signed=False
        ),
        "winners_share": _scalar_gap_block(
            rows, floor_rows, "winners_share", pooled_signed=False
        ),
        "q0_mean_delta": _scalar_gap_block(
            rows, floor_rows, "q0_mean_delta", pooled_signed=True
        ),
        "decile_incidence": _decile_gap_block(rows, floor_rows),
        "paired_descriptive": _paired_pool(rows),
    }


# ==========================================================================
# Provenance
# ==========================================================================
def _sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _gates_amendment_state() -> dict[str, Any]:
    """The gate-1 lock + ratified-amendment state, parsed from gates.yaml."""
    doc = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gate1 = doc["gates"]["gate_1"]
    thresholds = gate1.get("thresholds", {})
    history = gate1.get("amendment_history", []) or []
    rules = gate1.get("amendment_rules", {}) or {}
    return {
        "gate_1_locked": bool(thresholds.get("locked", False)),
        "amendments_ratified": [
            {"id": a.get("id"), "ratified": a.get("ratified")} for a in history
        ],
        "amendment_rules": sorted(rules.keys()),
    }


def _candidate11_reference() -> dict[str, Any]:
    """Reference the committed candidate-11 (run 13) gate artifact if present."""
    if not CANDIDATE11_ARTIFACT.exists():
        return {
            "path": str(CANDIDATE11_ARTIFACT.relative_to(ROOT)),
            "present": False,
            "note": (
                "the candidate-11 gate artifact is not committed in this "
                "worktree; the generator is reused from the run_gate1_"
                "candidate10/11 machinery regardless"
            ),
        }
    import hashlib

    raw = CANDIDATE11_ARTIFACT.read_bytes()
    art = json.loads(raw)
    return {
        "path": str(CANDIDATE11_ARTIFACT.relative_to(ROOT)),
        "present": True,
        "run": art.get("run"),
        "schema_version": art.get("schema_version"),
        "spec_registration": art.get("spec_registration"),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _revision_pins(params: Any) -> dict[str, Any]:
    """pe-us SHA, sklearn version, repo SHA, and the gates.yaml amendment state.

    The scikit-learn version is recorded per the ratified
    ``amendment_rules.classifier_version_pin`` (the generator's participation
    gates are version-sensitive); the gate is locked under amendment 2.
    """
    import sklearn

    return {
        "populace_dynamics_sha": _sha(ROOT),
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "sklearn_version": str(sklearn.__version__),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml": _gates_amendment_state(),
    }


# ==========================================================================
# Driver
# ==========================================================================
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full reform-delta diagnostic (reported, not gated)."""
    started = time.time()

    params = load_ssa_parameters()
    baseline_params = params
    if verbose:
        print(f"SSA oracle parameters: pe_us_revision={params.pe_us_revision}")

    panel = load_filtered_panel()
    all_anchor = anchor_rows(panel)
    cutpoints = anchor_quintile_cutpoints(all_anchor)
    n_persons = int(panel.person_id.nunique())
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, {n_persons} "
            f"persons; {len(all_anchor)} anchors; positive-anchor quartile "
            f"cuts {[round(float(c)) for c in cutpoints]}"
        )

    f1, f2, f3 = params.pia_factors
    bend_2022 = params.bend_points(ELIGIBILITY_YEAR)
    wage_base_2022 = params.wage_base_for(ELIGIBILITY_YEAR)
    reform_defs = {
        "baseline": {
            "kind": "untouched pinned oracle params",
            "pia_factors": [float(f1), float(f2), float(f3)],
            "bend_points_2022": [float(bend_2022[0]), float(bend_2022[1])],
            "wage_base_2022": float(wage_base_2022),
        },
        "reform_a": {
            "name": "Reform A (bottom-loaded)",
            "change": "first PIA factor 0.90 -> 0.95",
            "mechanism": (
                "params wrapper overrides pia_factors -> (0.95, f2, f3); "
                "f2/f3 are the baseline's own factors, read from params at "
                "runtime (not hardcoded)"
            ),
            "pia_factors": [0.95, float(f2), float(f3)],
            "baseline_pia_factors": [float(f1), float(f2), float(f3)],
            "bend_points_2022": [float(bend_2022[0]), float(bend_2022[1])],
            "incidence": "concentrates in the low deciles / concave region",
        },
        "reform_b": {
            "name": "Reform B (top-loaded)",
            "change": (
                "eliminate the taxable-maximum cap in the AIME step "
                "(benefit side only): wage_base_for(year) -> inf"
            ),
            "mechanism": (
                "params wrapper overrides wage_base_for(year) -> "
                "float('inf'); pia_factors and the bend points delegate to "
                "baseline"
            ),
            "baseline_wage_base_2022": float(wage_base_2022),
            "incidence": "gains concentrate at the cap (top of the distribution)",
        },
    }

    # Generate the candidate ONCE per seed (the generation does not depend on
    # the reform); reuse it for every reform. Also compute the real-vs-real
    # floor halves per seed (independent of the candidate).
    reforms_results: dict[str, Any] = {}
    per_seed_by_reform: dict[str, list[dict[str, Any]]] = {
        name: [] for name in REFORMS
    }
    floor_by_reform: dict[str, list[dict[str, Any]]] = {
        name: [] for name in REFORMS
    }

    for seed in SEEDS:
        t0 = time.time()
        holdout, candidate = fit_and_generate_candidate11(
            seed, panel, all_anchor
        )
        for name, factory in REFORMS.items():
            reform_params = factory(baseline_params)
            realgen = measure_realgen_seed(
                reform_params,
                baseline_params,
                all_anchor,
                cutpoints,
                holdout,
                candidate,
            )
            floor = measure_floor_seed(
                seed,
                reform_params,
                baseline_params,
                panel,
                all_anchor,
                cutpoints,
            )
            per_seed_by_reform[name].append({"seed": seed, **realgen})
            floor_by_reform[name].append({"seed": seed, **floor})
        if verbose:
            for name in REFORMS:
                rg = per_seed_by_reform[name][-1]
                fl = floor_by_reform[name][-1]
                print(
                    f"seed {seed} {name}: "
                    f"real_mean={rg['real']['mean_delta']:+.2f} "
                    f"gen_mean={rg['generated']['mean_delta']:+.2f} "
                    f"win_real={rg['real']['winners_share']:.3f} "
                    f"win_gen={rg['generated']['winners_share']:.3f} "
                    f"Q0_real={_fmt(rg['real']['q0_mean_delta'])} "
                    f"Q0_gen={_fmt(rg['generated']['q0_mean_delta'])} "
                    f"corr={_fmt(rg['paired']['corr'])} "
                    f"floorA_mean={fl['side_a']['mean_delta']:+.2f} "
                    f"floorB_mean={fl['side_b']['mean_delta']:+.2f}"
                )
        if verbose:
            print(
                f"  seed {seed} fit+generate+measure {time.time() - t0:.0f}s"
            )

    for name in REFORMS:
        reforms_results[name] = {
            "definition": reform_defs[name],
            "per_seed": per_seed_by_reform[name],
            "floor_per_seed": floor_by_reform[name],
            "pooled": pool_reform(
                per_seed_by_reform[name], floor_by_reform[name]
            ),
        }

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "purpose": (
            "Do the gate-passing generator's synthetic earnings histories "
            "(candidate 11, run 13) reproduce the reform INCIDENCE -- the "
            "per-person and distributional benefit deltas -- that real "
            "histories imply? This reads no gate and changes no gate; it is "
            "the run-13 analogue of the PR #56 downstream-relevance artifact, "
            "extended from benefit-space levels to reform deltas. The "
            "strongest claim it can support is 'consistent with the "
            "real-vs-real floor'; no pass/fail verdict exists."
        ),
        "not_full_415b": NOT_FULL_415B,
        "comparison_basis": (
            "Metrics 1-4 are compared against a real-vs-real person-disjoint "
            "half-split floor (the pia_proxy_floor ctx20 construction applied "
            "to the SAME Delta metrics, same 5 seeds): the A-vs-B gap is the "
            "noise scale for the real-vs-generated gap at the same "
            "~20%-of-persons scale. 'Consistent with the floor' is the "
            "strongest claim. Metric 5 (paired per-person) is DESCRIPTIVE "
            "ONLY -- disjoint persons across A/B give it no floor."
        ),
        "within_floor_abs_tol": WITHIN_FLOOR_ABS_TOL,
        "within_floor_note": (
            "within_floor is a DESCRIPTIVE annotation (realgen_scale <= "
            "floor_scale + within_floor_abs_tol), not a verdict; the tolerance "
            "only resolves 0-vs-0 floating-point knife-edges where a reform "
            "leaves a decile's Delta exactly constant"
        ),
        "reforms": reform_defs,
        "functional": {
            "source": (
                "scripts/build_downstream_relevance.py:person_pia_proxy / "
                "panel_pia_proxy (imported, not re-implemented; single source "
                "of truth)"
            ),
            "reform_mechanism": (
                "a reform enters ONLY through a thin params wrapper delegating "
                "to the loaded SSAParameters; the pinned oracle modules "
                "(ss/params.py, ss/benefits.py, build_downstream_relevance.py) "
                "are not edited"
            ),
            "eligibility_year": ELIGIBILITY_YEAR,
            "weight": (
                "each person weighted by their anchor-period weight (the "
                "weight on their chronologically last observed period)"
            ),
            "delta_definition": (
                "Delta = reform PIA-proxy - baseline PIA-proxy (2022 monthly "
                "dollars); both reforms weakly increase the PIA-proxy, so "
                "Delta >= 0"
            ),
        },
        "model": {
            "class": (
                "candidate 11 (run 13) = candidate 10's spec: k-NN "
                "conditional rank bootstrap with a FIXED lambda=0.1 "
                "anchor/permanent-rank donor blend for non-Q0 targets and a "
                "zero-anchor participation regime (full re-entry pool, Q0 "
                "memory-exempt)"
            ),
            "generation_source": (
                "run_gate1_candidate10.run_seed fit+generate prefix, imported "
                "byte-for-byte (fit_cell_marginals; candidate-8 build_donor_uw; "
                "candidate-9 build_donor_pools, fit_participation_gate, "
                "fit_zero_anchor_participation_gate; candidate-10 "
                "generate_candidate). No modeling constant moved; generation "
                "is deterministic from the seed and independent of the reform."
            ),
            "populace_fit_used": True,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "filter": (
                "age 25-59, reference years 1998-2022, positive weights "
                "(applied before the split)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person(panel, "
                "'person_id', fraction=0.2, seed=s); the drawn 20% is the "
                "holdout, the complement is the training set (locked "
                "protocol, imported from the baseline runner)"
            ),
            "floor_construction": (
                "real-vs-real ctx20 on the FULL filtered panel "
                "(build_pia_proxy_floor): split_panel_by_person(panel, "
                "'person_id', fraction=0.4, seed=1000+s) then "
                "split_panel_by_person(forty, 'person_id', fraction=0.5, "
                "seed=s); two DISJOINT ~20%-of-persons halves A and B, each "
                "pushed through the same baseline/reform proxy"
            ),
            "measurements": [
                "1. weighted mean Delta and median Delta (dollars; and % of "
                "the side's weighted-mean baseline); generated-vs-real gap",
                "2. mean Delta by decile group of the side's own weighted "
                "baseline distribution (d1-d10; compared on d3-d9, d1/d2/d10 "
                "reported)",
                "3. winners share (Delta > $1/month)",
                "4. zero-anchor (Q0) subgroup weighted mean Delta (per-seed "
                "signed for both sides; pooled across seeds)",
                "5. paired per-person corr(Delta_real, Delta_gen) and weighted "
                "mean |Delta_gen - Delta_real| (DESCRIPTIVE ONLY)",
            ],
            "comparison": (
                "metrics 1-4 vs the real-vs-real A-vs-B floor at the same "
                "scale; metric 5 descriptive only (no floor)"
            ),
        },
        "scope_honesty": (
            "PSID holdout persons on their observed support -- a validation of "
            "reform-delta fidelity, NOT a deployment simulation: no CPS "
            "integration, no demographics/claiming (gate-2/3 territory), no "
            "aggregate cost scoring against SSA baselines"
        ),
        "candidate11_reference": _candidate11_reference(),
        "n_person_periods": int(len(panel)),
        "n_persons": n_persons,
        "positive_anchor_quartile_cutpoints": [float(c) for c in cutpoints],
        "reform_results": reforms_results,
        "revision_pins": _revision_pins(params),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        for name in REFORMS:
            _print_reform_summary(name, reforms_results[name]["pooled"])
    return artifact


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _print_reform_summary(name: str, pooled: dict[str, Any]) -> None:
    md = pooled["mean_delta"]
    q0 = pooled["q0_mean_delta"]
    dec = pooled["decile_incidence"]
    win = pooled["winners_share"]
    corr = pooled["paired_descriptive"]["corr"]
    print(
        f"\n{name}: mean Delta real={md['real_pooled']['mean']:+.2f} "
        f"gen={md['generated_pooled']['mean']:+.2f} | "
        f"gap={md['realgen_scale']:.3f} floor={md['floor_scale']:.3f} "
        f"within={md['within_floor']} || "
        f"d3-d9 max gap={dec['max_realgen_scale_d3_d9']:.3f} "
        f"floor={dec['max_floor_scale_d3_d9']:.3f} "
        f"within={dec['within_floor_d3_d9']} || "
        f"winners gap={win['realgen_scale']:.4f} floor={win['floor_scale']:.4f} "
        f"within={win['within_floor']} || "
        f"Q0 pooled gap={q0['realgen_scale']:.3f} floor={q0['floor_scale']:.3f} "
        f"within={q0['within_floor']} || "
        f"paired corr={corr['mean']:.3f}"
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, default=_json_default) + "\n"
    )
    print(f"\nwrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
