"""Gate-1 candidate 5b: rank-space generative dynamics.

The SEVENTH pre-registered model run of PolicyEngine/populace-dynamics,
and the first of the GENERATIVE track after the splicing benchmark (5a).
It composes the field's third and fourth strategies (paper sec.
gate-1): empirical quantile marginals (rank space -- the marginal
cannot break by construction, run 4's lesson), a fixed-form latent
persistence model no learner can rescale away (runs 2-3's lesson), and
simulated-moment calibration so the iterated -- not one-step --
dynamics match the ladder (the baseline's lesson). Participation reuses
the regime machinery that has passed in every run.

The candidate-5b spec is registered, frozen before the run, in issue
#42's candidate-5b comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4894920407);
every stage below -- cells, the quantile/rank formulas, the latent
model, the SMM grid and tie-breaks, the anchor conditionals, and the
substream seeding -- is pinned there and implemented LITERALLY. No
threshold is hardcoded, no model choice is tuned against holdout
scores, and the only fit-time freedom is the registered SMM grid, which
calibrates on TRAIN only. The run is one shot; the outcome publishes
whether it passes or fails.

The candidate, per the frozen spec (each stage implemented literally):

1. **Marginal machinery (train only).** Cells = five-year age bins
   {25-29, ..., 55-59} x calendar period. Per cell: ``p0`` = weighted
   zero share; for positive earnings, the weighted empirical quantile
   function ``Qhat_pos`` (sort ``y_k`` with weights ``w_k``, plotting
   position ``wtil_k = (cum_k - 0.5 w_k) / W``, linear interpolation of
   ``(wtil_k, y_k)`` clamped to ``[y_min, y_max]``) and its inverse the
   rank function ``rhat`` (``y_k -> wtil_k``, clamped to
   ``[0.001, 0.999]``).
2. **Latent model.** ``z = a w + b x + c eps`` with ``w ~ N(0,1)``
   person-permanent, ``x`` stationary AR(1) N(0,1) coefficient ``rho``,
   ``eps`` i.i.d. N(0,1), ``a,b,c >= 0``, ``a^2+b^2+c^2 = 1``;
   ``U = Phi(z)``. Because the three variances sum to one, ``z`` is
   marginally N(0,1) and ``U`` marginally Uniform(0,1).
3. **SMM calibration (train only).** Participation is independent of the
   magnitude latent, so the among-positives autocorrelation of the
   deployed process equals that of the magnitude process; calibration
   simulates the magnitude process alone. Subsample = the first 5,000
   train persons with positive anchors ordered by ``person_id``, each on
   their actual positive-period support, anchored at their real anchor
   exactly as generation does, internal RNG seeded from the gate seed.
   Targets: the train panel's battery autocorrelation at lags 1/2/5
   (locked definitions, among positives). Grid: ``a^2`` in
   {0.25,...,0.60}, ``rho`` in {0.55,...,0.95}, ``c^2`` in
   {0.05,...,0.35}, requiring ``b^2 = 1 - a^2 - c^2 >= 0.05``; one
   simulation per point; equal-weight SSE over the three targets; ties
   broken by smaller ``a^2``, then smaller ``rho``, then smaller
   ``c^2``.
4. **Generation (holdout).** Anchor keeps its real value. For a positive
   anchor ``u_A = rhat`` at the anchor's cell of the anchor value; for a
   zero anchor ``u_A = p0 / 2`` of the cell; ``z_A = Phi^-1(u_A)``.
   Initialize by the exact Gaussian conditionals:
   ``w | z_A ~ N(a z_A, 1 - a^2)``, then
   ``x_A | z_A, w ~ N(b (z_A - a w) / (b^2 + c^2), c^2 / (b^2 + c^2))``.
   Chain backward over the person's observed periods:
   ``x_prev = rho x_next + sqrt(1 - rho^2) nu`` with ``nu ~ N(0,1)``,
   one step per observed-period transition regardless of gap width (the
   standing gap rule). Per generated period ``z = a w + b x + c eps``
   with fresh ``eps``, ``U = Phi(z)``. Participation comes from the
   regime machinery exactly as in the candidate-2 registration (backward
   gate on next generated level and age, trained on the 80% complement,
   populace-fit defaults); where positive, earnings = ``Qhat_pos`` of the
   period's cell at ``U``. All RNG streams seeded from the gate seed with
   distinct fixed substream labels (``w``, ``x``-innovations, ``eps``,
   ``gate``).

The protocol mechanics -- the filter-first load, the person-disjoint
0.2 split per seed, the two locked views, ``panel_scorecard`` scoring,
the battery on the candidate panel vs the committed ``battery_reference``
with locked definitions, the thresholds read from ``gates.yaml`` at
runtime, the seed-level conjunction (>=4/5 both blocks), and the
battery-reference bit-exact precheck -- are IMPORTED from the merged
baseline runner (:mod:`run_gate1_baseline`, pull request 40),
byte-for-byte the prior runs'. Only the generative model (marginals,
latent, SMM, backward generation) is local.

Determinism. Stage-1 marginals and Stage-3 SMM are deterministic given
the split (the SMM's shocks are drawn once from the gate seed and reused
across every grid point -- common random numbers, so grid SSE reflects
parameters, not noise). Stage-4 generation draws each of ``w``,
``x``-innovations, ``eps``, and the participation gate from its own
fixed-label substream of the gate seed, in the batched-by-step,
``person_id``-ordered pass the candidate-2 chain uses. The run
reproduces from the seeds alone.

Two front-end environments. The SMM magnitude simulation is pure
numpy/scipy Gaussian algebra plus quantile interpolation and needs NO
populace-fit; it is vectorized so ~500 grid points x 5,000 persons run
in minutes. The participation gate is a ``RegimeGatedQRF`` sign gate and
DOES need populace-fit. Run the full gate from the repository root with
the PSID family files staged, using the DEDICATED gate venv
(populace-fit pins scikit-learn < 1.9, which the repo's ``.venv``
violates; see populace #318):

    .venv-gate/bin/python scripts/run_gate1_candidate5b.py
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
# byte-for-byte identical to every prior gate-1 run. Only the generative
# model is local. The baseline module defers its populace-fit import to
# its fit path; this candidate reaches populace-fit only through the
# participation gate (below).
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
from scipy.stats import norm

from populace_dynamics.harness import moments
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_rank_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate1_rank.v1"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4894920407"
)

# ---- Frozen constants of the registration (applied literally) ----------
#: Five-year age-bin lower edges {25-29, ..., 55-59}: bin index =
#: clip((age - 25) // 5, 0, 6) over the locked age filter 25-59.
AGE_BIN_WIDTH = 5
N_AGE_BINS = 7
#: Rank clamp for the inverse quantile (rank) function.
RANK_CLAMP_LO, RANK_CLAMP_HI = 0.001, 0.999
#: SMM subsample size (first N train persons with positive anchors).
SMM_SUBSAMPLE = 5000
#: SMM autocorrelation lags (biennial => 2/4/10 years), the locked
#: battery-autocorrelation lags among positives.
SMM_LAGS = (1, 2, 5)
#: The registered SMM grid axes (inclusive ranges at step 0.05).
A2_GRID = tuple(round(0.25 + 0.05 * i, 10) for i in range(8))  # .25-.60
RHO_GRID = tuple(round(0.55 + 0.05 * i, 10) for i in range(9))  # .55-.95
C2_GRID = tuple(round(0.05 + 0.05 * i, 10) for i in range(7))  # .05-.35
#: b^2 floor: a grid point is admissible iff b^2 = 1 - a^2 - c^2 >= this.
B2_FLOOR = 0.05

#: Fixed integer codes for the generation RNG substream labels. Each
#: label seeds an independent generator via SeedSequence([seed, code]),
#: so the four streams are distinct and reproducible from the gate seed.
SUBSTREAM_CODES = {"w": 1, "x": 2, "eps": 3, "gate": 4}
#: Fixed code for the SMM's single internal RNG (seeded from the gate
#: seed; its shocks are drawn once and reused across grid points).
SMM_STREAM_CODE = 5

# Battery column order used by the moments autocorrelation call (same as
# the baseline's ``_MOMENT_KW``: id/period/value/weight on the panel).
_MOMENT_KW = dict(
    id_col="person_id",
    period_col="period",
    value_col="earnings",
    weight_col="weight",
)


# --------------------------------------------------------------------------
# Cells (five-year age bin x calendar period)
# --------------------------------------------------------------------------
def age_bin(age: np.ndarray) -> np.ndarray:
    """Five-year age-bin index in ``0..6`` for the locked 25-59 filter.

    Bin ``k`` covers ages ``25 + 5k .. 29 + 5k``; the clip keeps the
    exact boundary age 59 in the top bin (6) and guards against any
    out-of-filter age.
    """
    age = np.asarray(age, dtype=np.float64)
    idx = np.floor((age - AGE_MIN) / AGE_BIN_WIDTH).astype(np.int64)
    return np.clip(idx, 0, N_AGE_BINS - 1)


def _substream(seed: int, label: str) -> np.random.Generator:
    """A generation RNG for one fixed substream label off the gate seed."""
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


# --------------------------------------------------------------------------
# Stage 1 - marginal machinery (train only): p0, Qhat_pos, rhat per cell
# --------------------------------------------------------------------------
class CellMarginal:
    """Per-cell weighted zero share ``p0`` with quantile/rank maps.

    Holds, for one (age-bin, period) cell fitted on the train split:

    * ``p0`` -- the weighted share of zero earnings in the cell.
    * ``wtil`` / ``yval`` -- the plotting-position grid ``(wtil_k, y_k)``
      of the positive-earnings values, sorted ascending by ``y_k``, with
      ``wtil_k = (cum_k - 0.5 w_k) / W`` (``cum_k`` the inclusive
      cumulative positive weight, ``W`` the total positive weight).
    * ``ymin`` / ``ymax`` -- the clamp bounds ``[y_min, y_max]`` for the
      quantile map.

    ``quantile(u)`` linearly interpolates ``(wtil, yval)`` at ``u``,
    clamped to ``[ymin, ymax]`` (``np.interp`` clamps to the end values
    for ``u`` outside ``[wtil[0], wtil[-1]]``, which for a single
    positive value returns that value). ``rank(y)`` is the inverse
    interpolation ``(yval -> wtil)`` clamped to ``[0.001, 0.999]``.
    """

    __slots__ = ("p0", "wtil", "yval", "ymin", "ymax", "n_pos", "w_total")

    def __init__(
        self,
        p0: float,
        wtil: np.ndarray,
        yval: np.ndarray,
        n_pos: int,
        w_total: float,
    ) -> None:
        self.p0 = float(p0)
        self.wtil = wtil
        self.yval = yval
        self.n_pos = int(n_pos)
        self.w_total = float(w_total)
        self.ymin = float(yval[0]) if len(yval) else np.nan
        self.ymax = float(yval[-1]) if len(yval) else np.nan

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Weighted empirical quantile at ranks ``u`` (clamped values)."""
        vals = np.interp(u, self.wtil, self.yval)
        return np.clip(vals, self.ymin, self.ymax)

    def rank(self, y: float) -> float:
        """Inverse quantile (rank) at ``y``, clamped to [0.001, 0.999]."""
        r = float(np.interp(y, self.yval, self.wtil))
        return float(np.clip(r, RANK_CLAMP_LO, RANK_CLAMP_HI))


def _plotting_positions(
    y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted ``(wtil_k, y_k)`` with the registered plotting position.

    Sort ``y`` ascending (stable) with its weights, then
    ``wtil_k = (cum_k - 0.5 w_k) / W`` where ``cum_k`` is the inclusive
    cumulative weight and ``W = sum w``. This is the weighted empirical
    CDF plotting position the harness uses elsewhere
    (:func:`populace_dynamics.harness.moments._weighted_quantile_bin`,
    :func:`age_profile`), applied here to positive earnings within a
    cell.
    """
    order = np.argsort(y, kind="stable")
    ys = y[order]
    ws = w[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    wtil = (cum - 0.5 * ws) / total
    return wtil, ys


def fit_cell_marginals(
    train: pd.DataFrame,
) -> dict[tuple[int, int], CellMarginal]:
    """Fit ``p0``, ``Qhat_pos``, and ``rhat`` for every train cell.

    Cells are (age-bin, period). Within a cell ``p0`` is the weighted
    zero share; the positive-earnings plotting-position grid defines the
    quantile and rank maps. Returns ``{(bin, period): CellMarginal}``.
    The train panel is the seed's 80% complement.
    """
    tr = train[["earnings", "age", "period", "weight"]].copy()
    tr["bin"] = age_bin(tr["age"].to_numpy())
    out: dict[tuple[int, int], CellMarginal] = {}
    for (b, p), g in tr.groupby(["bin", "period"]):
        earn = g["earnings"].to_numpy(dtype=np.float64)
        wt = g["weight"].to_numpy(dtype=np.float64)
        is_pos = earn > 0
        w_total = float(wt.sum())
        w_zero = float(wt[~is_pos].sum())
        p0 = w_zero / w_total if w_total > 0 else 1.0
        if is_pos.any():
            wtil, ys = _plotting_positions(earn[is_pos], wt[is_pos])
            n_pos = int(is_pos.sum())
            w_pos = float(wt[is_pos].sum())
        else:
            wtil = np.empty(0, dtype=np.float64)
            ys = np.empty(0, dtype=np.float64)
            n_pos = 0
            w_pos = 0.0
        out[(int(b), int(p))] = CellMarginal(p0, wtil, ys, n_pos, w_pos)
    return out


# --------------------------------------------------------------------------
# Anchors (chronologically last observed period per person)
# --------------------------------------------------------------------------
def anchor_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per person: their chronologically LAST observed period.

    Anchor = the person's maximum ``period`` in the filtered panel, with
    that row's ``earnings``, ``age``, ``weight`` (the SAME anchor the
    baseline chain and every prior candidate use). Deterministic:
    periods are unique per person-period, so the ``idxmax`` is unique.
    """
    idx = panel.groupby("person_id")["period"].idxmax()
    cols = ["person_id", "period", "earnings", "age", "weight"]
    return panel.loc[idx, cols].reset_index(drop=True)


def anchor_u(
    marginals: dict[tuple[int, int], CellMarginal],
    earnings: float,
    age: float,
    period: int,
) -> float:
    """Anchor rank ``u_A`` per the frozen rule.

    For a positive anchor ``u_A = rhat`` at the anchor's cell of the
    anchor value; for a zero anchor ``u_A = p0 / 2`` of the cell. The
    cell is (age-bin of the anchor age, anchor period).
    """
    cell = marginals[(int(age_bin(np.array([age]))[0]), int(period))]
    if earnings > 0:
        return cell.rank(float(earnings))
    return 0.5 * cell.p0


# --------------------------------------------------------------------------
# Stage 3 - SMM calibration (train only), vectorized magnitude process
# --------------------------------------------------------------------------
def _build_smm_support(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
) -> dict[str, Any]:
    """Fixed ragged support for the SMM subsample's magnitude process.

    The subsample is the first :data:`SMM_SUBSAMPLE` train persons with
    positive anchors ordered by ``person_id``. Each contributes its
    actual positive-period support (the periods where its earnings are
    positive), ordered chronologically DESCENDING (rank 0 = the person's
    latest positive period). Because the SMM subsample is drawn from
    persons with positive anchors, rank 0 is exactly the anchor, held at
    its real earnings; the remaining positive periods receive simulated
    magnitudes.

    Returns arrays laid out on a dense ``(n_persons, max_rank)`` grid
    (invalid slots masked), the per-slot cell index, ``z_A`` at each
    person's anchor, the real anchor earnings, per-slot weights, and the
    lag-pair index (earlier-slot, later-slot, earlier-weight) for each
    SMM lag -- everything that does not depend on ``(a, b, c, rho)``, so
    the grid search only recomputes the Gaussian algebra.
    """
    train_ids = set(train["person_id"].unique())
    tr_anchor = all_anchor[all_anchor.person_id.isin(train_ids)]
    pos_anchor = (
        tr_anchor[tr_anchor.earnings > 0]
        .sort_values("person_id")
        .reset_index(drop=True)
    )
    sub_ids = pos_anchor["person_id"].to_numpy()[:SMM_SUBSAMPLE]
    sub_set = set(int(x) for x in sub_ids)

    sub = train[train.person_id.isin(sub_set) & (train.earnings > 0)].copy()
    sub["bin"] = age_bin(sub["age"].to_numpy())
    # Chronologically DESCENDING within person (rank 0 = latest period).
    sub = sub.sort_values(
        ["person_id", "period"], ascending=[True, False]
    ).reset_index(drop=True)
    sub["rank"] = sub.groupby("person_id").cumcount()

    n_persons = len(sub_ids)
    pid_order = sub_ids  # ascending person_id (already sorted)
    pid_to_row = {int(pid): i for i, pid in enumerate(pid_order)}
    depth = sub.groupby("person_id").size()
    depth = depth.reindex(pid_order).to_numpy().astype(np.int64)
    max_rank = int(depth.max())

    # Dense grids (person x rank), invalid slots masked.
    valid = np.zeros((n_persons, max_rank), dtype=bool)
    cell_idx = np.full((n_persons, max_rank), -1, dtype=np.int64)
    earn_real = np.zeros((n_persons, max_rank), dtype=np.float64)
    wt_grid = np.zeros((n_persons, max_rank), dtype=np.float64)
    period_grid = np.full((n_persons, max_rank), -1, dtype=np.int64)

    # Stable cell ordering so the per-cell interpolation loop is fixed.
    cell_keys = sorted(
        {
            (int(b), int(p))
            for b, p in zip(sub["bin"], sub["period"], strict=True)
        }
    )
    cell_key_to_i = {k: i for i, k in enumerate(cell_keys)}

    rows = sub[["person_id", "rank", "bin", "period", "earnings", "weight"]]
    for pid, rk, b, p, earn, wt in rows.itertuples(index=False):
        i = pid_to_row[int(pid)]
        j = int(rk)
        valid[i, j] = True
        cell_idx[i, j] = cell_key_to_i[(int(b), int(p))]
        earn_real[i, j] = float(earn)
        wt_grid[i, j] = float(wt)
        period_grid[i, j] = int(p)

    # z_A per person from the anchor (rank 0 is always a positive anchor).
    return {
        "n_persons": n_persons,
        "max_rank": max_rank,
        "depth": depth,
        "valid": valid,
        "cell_idx": cell_idx,
        "cell_keys": cell_keys,
        "earn_real": earn_real,
        "wt_grid": wt_grid,
        "period_grid": period_grid,
        "sub_ids": pid_order,
    }


def _lag_pairs_from_support(
    period_grid: np.ndarray,
    valid: np.ndarray,
    wt_grid: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat (earlier-slot, later-slot, earlier-weight) index for one lag.

    Mirrors :func:`moments.autocorrelation`'s ``_pair`` on the biennial
    panel: an earlier period ``t`` pairs with the later period
    ``t + lag * period_step`` when both are positive (present in the
    support). Weight = the EARLIER period's weight, matching the locked
    autocorrelation definition. Slots are addressed as flat indices into
    the ``(n_persons, max_rank)`` grid.
    """
    n_persons, max_rank = period_grid.shape
    step = lag * PERIOD_STEP
    # Map (person, period) -> flat slot for O(1) lookup of the later obs.
    slot_of: dict[tuple[int, int], int] = {}
    for i in range(n_persons):
        for j in range(max_rank):
            if valid[i, j]:
                slot_of[(i, int(period_grid[i, j]))] = i * max_rank + j
    earlier: list[int] = []
    later: list[int] = []
    ew: list[float] = []
    for i in range(n_persons):
        for j in range(max_rank):
            if not valid[i, j]:
                continue
            t = int(period_grid[i, j])
            later_slot = slot_of.get((i, t + step))
            if later_slot is not None:
                earlier.append(i * max_rank + j)
                later.append(later_slot)
                ew.append(float(wt_grid[i, j]))
    return (
        np.asarray(earlier, dtype=np.int64),
        np.asarray(later, dtype=np.int64),
        np.asarray(ew, dtype=np.float64),
    )


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation, the locked autocorrelation formula."""
    if len(x) < 2:
        return float("nan")
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    return float(cov / (sx * sy))


def _draw_smm_shocks(
    support: dict[str, Any], seed: int
) -> dict[str, np.ndarray]:
    """Draw the SMM's standard-normal shocks once from the gate seed.

    Common random numbers: the same underlying normals feed every grid
    point, so the grid SSE reflects ``(a, b, c, rho)`` and not simulation
    noise. Shocks are ``eta_w`` (person permanent), ``eta_x0`` (anchor
    ``x`` initialization), ``nu`` (backward AR innovations per rank), and
    ``eps`` (idiosyncratic per rank), drawn in that fixed order from a
    single generator seeded from the gate seed.
    """
    n = support["n_persons"]
    m = support["max_rank"]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), SMM_STREAM_CODE])
    )
    eta_w = rng.standard_normal(n)
    eta_x0 = rng.standard_normal(n)
    nu = rng.standard_normal((n, m))
    eps = rng.standard_normal((n, m))
    return {"eta_w": eta_w, "eta_x0": eta_x0, "nu": nu, "eps": eps}


def _simulate_magnitude_grid_u(
    support: dict[str, Any],
    shocks: dict[str, np.ndarray],
    z_a: np.ndarray,
    a: float,
    b: float,
    c: float,
    rho: float,
) -> np.ndarray:
    """Vectorized ``U`` grid for the magnitude process at one parameter.

    Applies the exact Gaussian conditionals and the backward AR(1) chain
    over the dense ``(person, rank)`` grid:

    * ``w = a z_A + sqrt(1 - a^2) eta_w`` (person permanent);
    * ``x_0 = b (z_A - a w) / (b^2 + c^2) + sqrt(c^2 / (b^2 + c^2))
      eta_x0`` (the anchor conditional at rank 0);
    * ``x_j = rho x_{j-1} + sqrt(1 - rho^2) nu_j`` for ``j >= 1``;
    * ``z_j = a w + b x_j + c eps_j``; ``U_j = Phi(z_j)``.

    Returns ``U`` on the dense grid (invalid slots carry an arbitrary
    finite value, never read downstream). ``z_A`` and the shocks are
    fixed across the grid; only ``(a, b, c, rho)`` vary here.
    """
    n = support["n_persons"]
    m = support["max_rank"]
    a2 = a * a
    bc = b * b + c * c
    w = a * z_a + np.sqrt(max(0.0, 1.0 - a2)) * shocks["eta_w"]
    x = np.empty((n, m), dtype=np.float64)
    x[:, 0] = (b * (z_a - a * w) / bc) + np.sqrt(c * c / bc) * shocks["eta_x0"]
    root = np.sqrt(max(0.0, 1.0 - rho * rho))
    for j in range(1, m):
        x[:, j] = rho * x[:, j - 1] + root * shocks["nu"][:, j]
    z = a * w[:, None] + b * x + c * shocks["eps"]
    return norm.cdf(z)


def _u_to_earnings_grid(
    support: dict[str, Any],
    cell_marginals_by_i: list[CellMarginal],
    u_grid: np.ndarray,
) -> np.ndarray:
    """Map ``U`` to earnings via each slot's cell quantile (vectorized).

    Groups valid slots by cell index and applies that cell's
    ``Qhat_pos`` to their ranks in one ``np.interp`` per cell. Invalid
    slots and the anchor slot are left at zero here; the caller overlays
    the real anchor earnings.
    """
    n = support["n_persons"]
    m = support["max_rank"]
    cell_idx = support["cell_idx"]
    valid = support["valid"]
    out = np.zeros((n, m), dtype=np.float64)
    flat_cell = cell_idx.reshape(-1)
    flat_u = u_grid.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_out = out.reshape(-1)
    for ci, cm in enumerate(cell_marginals_by_i):
        sel = flat_valid & (flat_cell == ci)
        if sel.any():
            flat_out[sel] = cm.quantile(flat_u[sel])
    return out


def run_smm(
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    seed: int,
) -> dict[str, Any]:
    """Grid-search ``(a^2, rho, c^2)`` to match train autocorrelation.

    Simulates the magnitude process on the fixed 5,000-person support at
    every admissible grid point (``b^2 = 1 - a^2 - c^2 >= 0.05``),
    computes the among-positives autocorrelation at lags 1/2/5 with the
    locked definition, and scores equal-weight SSE against the train
    panel's battery autocorrelation. Ties break by smaller ``a^2``, then
    smaller ``rho``, then smaller ``c^2``. Returns the chosen parameters,
    the target vs simulated moments, the minimum SSE, and the full grid
    trace.
    """
    support = _build_smm_support(train, all_anchor)
    shocks = _draw_smm_shocks(support, seed)

    # Per-slot cell marginals in the support's fixed cell order.
    cell_marginals_by_i = [marginals[key] for key in support["cell_keys"]]

    # z_A per person from the (positive) anchor value at its cell.
    z_a = np.empty(support["n_persons"], dtype=np.float64)
    earn_real = support["earn_real"]
    for i in range(support["n_persons"]):
        anchor_val = float(earn_real[i, 0])
        cm = cell_marginals_by_i[int(support["cell_idx"][i, 0])]
        u_a = cm.rank(anchor_val)
        z_a[i] = norm.ppf(u_a)

    # Fixed lag-pair index and the earlier-period weights per lag.
    lag_pairs = {
        lag: _lag_pairs_from_support(
            support["period_grid"],
            support["valid"],
            support["wt_grid"],
            lag,
        )
        for lag in SMM_LAGS
    }

    # Targets: the train panel's battery autocorrelation among positives.
    ac_train = moments.autocorrelation(
        train, lags=SMM_LAGS, period_step=PERIOD_STEP, log=True, **_MOMENT_KW
    )
    target = {int(r.lag): float(r.value) for r in ac_train.itertuples()}
    target_vec = np.array([target[lag] for lag in SMM_LAGS])

    def simulate_moments(a2: float, c2: float, rho: float) -> np.ndarray:
        a = np.sqrt(a2)
        c = np.sqrt(c2)
        b = np.sqrt(max(0.0, 1.0 - a2 - c2))
        u_grid = _simulate_magnitude_grid_u(support, shocks, z_a, a, b, c, rho)
        earn = _u_to_earnings_grid(support, cell_marginals_by_i, u_grid)
        # Overlay the real anchor earnings at rank 0 (kept as generation
        # keeps the anchor); other positive slots use simulated values.
        earn[:, 0] = earn_real[:, 0]
        flat_earn = earn.reshape(-1)
        moms = []
        for lag in SMM_LAGS:
            e_idx, l_idx, ew = lag_pairs[lag]
            # Lag pairs reference only valid (positive) support slots, so
            # the logs are always finite; take them on the referenced
            # values (not the whole grid, whose masked slots are zero).
            moms.append(
                _weighted_corr(
                    np.log(flat_earn[e_idx]),
                    np.log(flat_earn[l_idx]),
                    ew,
                )
            )
        return np.asarray(moms)

    best: dict[str, Any] | None = None
    grid_trace: list[dict[str, Any]] = []
    for a2 in A2_GRID:
        for rho in RHO_GRID:
            for c2 in C2_GRID:
                b2 = 1.0 - a2 - c2
                if b2 < B2_FLOOR:
                    continue
                sim = simulate_moments(a2, c2, rho)
                sse = float(np.sum((sim - target_vec) ** 2))
                grid_trace.append(
                    {
                        "a2": float(a2),
                        "rho": float(rho),
                        "c2": float(c2),
                        "b2": float(round(b2, 10)),
                        "sim_autocorr": {
                            f"lag{lag}": float(sim[k])
                            for k, lag in enumerate(SMM_LAGS)
                        },
                        "sse": sse,
                    }
                )
                # Tie-break key: (sse, a2, rho, c2) all ascending.
                key = (sse, a2, rho, c2)
                if best is None or key < best["_key"]:
                    best = {
                        "_key": key,
                        "a2": float(a2),
                        "rho": float(rho),
                        "c2": float(c2),
                        "b2": float(round(b2, 10)),
                        "sse": sse,
                        "sim_autocorr": {
                            f"lag{lag}": float(sim[k])
                            for k, lag in enumerate(SMM_LAGS)
                        },
                    }

    assert best is not None
    a2 = best["a2"]
    c2 = best["c2"]
    b2 = best["b2"]
    return {
        "a2": a2,
        "b2": b2,
        "c2": c2,
        "rho": best["rho"],
        "a": float(np.sqrt(a2)),
        "b": float(np.sqrt(b2)),
        "c": float(np.sqrt(c2)),
        "min_sse": best["sse"],
        "target_autocorr": {f"lag{lag}": target[lag] for lag in SMM_LAGS},
        "simulated_autocorr": best["sim_autocorr"],
        "n_smm_persons": int(support["n_persons"]),
        "n_grid_points_evaluated": len(grid_trace),
        "grid_trace": grid_trace,
    }


# --------------------------------------------------------------------------
# Stage 4 - generation (holdout): backward latent chain + regime gate
# --------------------------------------------------------------------------
def fit_participation_gate(train: pd.DataFrame, seed: int) -> Any:
    """Fit the candidate-2 backward regime gate on the train complement.

    Reuses the baseline/candidate-2 backward pairs exactly (predictors =
    earnings at ``t`` and age at ``t-2``, target = earnings at ``t-2``,
    ``sample_weight`` = the earlier-period weight) and fits a
    ``RegimeGatedQRF`` at populace-fit defaults. Only its SIGN GATE
    (zero-vs-positive classifier) is used at generation to decide
    participation; the magnitude comes from the rank-space quantile map,
    not from the fitted forest. Seeded from the gate seed.
    """
    # Imported here so the pure-SMM path (and its tests) need no
    # populace-fit; only the participation gate reaches it.
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


def _gate_sign_draw(
    fitted: Any,
    next_level: np.ndarray,
    age: np.ndarray,
    u_gate: np.ndarray,
) -> np.ndarray:
    """Draw a sign per row from the fitted gate, exactly as candidate 2.

    Replicates ``FittedRegimeGatedQRF._gate_draw`` bit-for-bit -- the
    cumulative proba over ``gate.classes_``, ``chosen = (cumulative >=
    u).argmax(axis=1)``, mapped back through ``classes_`` -- but with the
    ``u`` supplied from this candidate's GATE substream of the gate seed
    (the registration pins the gate stream), rather than the fitted
    model's internal draw RNG. Conditions on (next generated level,
    current age), the candidate-2 backward predictors. Returns one sign
    code (``-1`` / ``0`` / ``1``) per row; participation is ``sign == 1``.
    """
    model = fitted._target_models["earnings_tm2"]
    gate = model.gate
    columns = model.columns
    feat = pd.DataFrame({"earnings": next_level, "age_tm2": age})
    x = feat.loc[:, list(columns)].to_numpy(dtype=np.float64)
    proba = np.asarray(gate.predict_proba(x))
    cumulative = np.cumsum(proba, axis=1)
    chosen = (cumulative >= u_gate[:, None]).argmax(axis=1)
    return np.asarray(gate.classes_)[chosen]


def generate_candidate(
    holdout: pd.DataFrame,
    all_anchor: pd.DataFrame,
    marginals: dict[tuple[int, int], CellMarginal],
    fitted: Any,
    params: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank-space generative candidate panel over the holdout persons.

    For each holdout person: set the anchor rank ``u_A`` from the frozen
    rule, initialize ``(w, x_A)`` by the exact Gaussian conditionals,
    chain ``x`` backward one step per observed transition, and at each
    generated period draw ``eps``, form ``U = Phi(a w + b x + c eps)``,
    draw participation from the regime gate on (next generated level,
    current age), and where positive set earnings = ``Qhat_pos`` of the
    period's cell at ``U``. The anchor keeps its REAL earnings.

    The pass is batched by step-from-anchor and ordered by ``person_id``
    within a step -- the candidate-2 chain structure -- so the ``w``,
    ``x``-innovation, ``eps``, and gate substreams each consume their
    draws in a fixed order. Returns ``(candidate, diagnostics)`` where the
    candidate holds exactly the holdout persons on exactly their observed
    periods (only earnings generated; anchor kept), and diagnostics carry
    the reported-not-gated distributions (anchor rank, cell count, clamped
    rank share).
    """
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    rho = float(params["rho"])
    a2 = a * a
    bc = b * b + c * c

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

    # Person-indexed state, ordered by person_id (deterministic passes).
    holdout_ids = np.sort(hp["person_id"].unique())
    person_pos = {int(pid): k for k, pid in enumerate(holdout_ids)}
    n_persons = len(holdout_ids)

    # Anchor rows for the holdout persons, in person_id order.
    ha = (
        all_anchor[all_anchor.person_id.isin(set(int(x) for x in holdout_ids))]
        .sort_values("person_id")
        .reset_index(drop=True)
    )

    # --- state init: z_A, w, x at anchor (rank 0) --------------------------
    z_a = np.empty(n_persons, dtype=np.float64)
    anchor_rank_vals = np.empty(n_persons, dtype=np.float64)
    for row in ha.itertuples(index=False):
        k = person_pos[int(row.person_id)]
        u_a = anchor_u(
            marginals,
            float(row.earnings),
            float(row.age),
            int(row.period),
        )
        anchor_rank_vals[k] = u_a
        z_a[k] = norm.ppf(u_a)

    rng_w = _substream(seed, "w")
    rng_x = _substream(seed, "x")
    rng_eps = _substream(seed, "eps")
    rng_gate = _substream(seed, "gate")

    # Draw w and the anchor x_A in person_id order (fixed substream order).
    eta_w = rng_w.standard_normal(n_persons)
    w = a * z_a + np.sqrt(max(0.0, 1.0 - a2)) * eta_w
    eta_x0 = rng_x.standard_normal(n_persons)
    x_state = (b * (z_a - a * w) / bc) + np.sqrt(c * c / bc) * eta_x0

    # Diagnostics accumulators.
    n_clamped = 0
    n_positive_gen = 0

    root = np.sqrt(max(0.0, 1.0 - rho * rho))
    # Backward chain: step j draws x at rank j from rank j-1, then
    # participation and magnitude for the rank-j generated period.
    for j in range(1, max_depth):
        positions = np.nonzero(ranks == j)[0]
        if positions.size == 0:
            continue
        # Canonical person_id order within the step.
        order = np.argsort(pids[positions], kind="stable")
        positions = positions[order]
        step_pids = pids[positions]
        step_person_k = np.array([person_pos[int(pid)] for pid in step_pids])
        # Advance x for exactly the persons present at this step.
        nu = rng_x.standard_normal(len(positions))
        x_step = rho * x_state[step_person_k] + root * nu
        x_state[step_person_k] = x_step

        # Latent draw -> U for the generated period.
        eps = rng_eps.standard_normal(len(positions))
        z = a * w[step_person_k] + b * x_step + c * eps
        u = norm.cdf(z)

        # Participation gate on (next generated level, current age),
        # drawn exactly as candidate 2 with u from the gate substream.
        next_positions = np.array(
            [pos_by_key[(int(pid), j - 1)] for pid in step_pids]
        )
        next_level = gen_earn[next_positions]
        u_gate = rng_gate.random(len(positions))
        signs = _gate_sign_draw(fitted, next_level, ages[positions], u_gate)
        is_pos = signs == 1

        # Magnitude where positive: Qhat_pos of the period's cell at U.
        vals = np.zeros(len(positions), dtype=np.float64)
        if is_pos.any():
            pos_local = np.nonzero(is_pos)[0]
            for li in pos_local:
                gpos = positions[li]
                cell = marginals[(int(bins[gpos]), int(periods[gpos]))]
                q = float(cell.quantile(np.array([u[li]]))[0])
                vals[li] = q
                # Clamped-rank bookkeeping: the quantile clamps u outside
                # the cell's [wtil[0], wtil[-1]] plotting-position span.
                if cell.wtil.size and (
                    u[li] < cell.wtil[0] or u[li] > cell.wtil[-1]
                ):
                    n_clamped += 1
            n_positive_gen += int(is_pos.sum())
        gen_earn[positions] = vals

    out = hp[["person_id", "period", "earnings", "age", "weight"]].copy()
    out["earnings"] = gen_earn

    # Anchor-rank distribution (reported-not-gated): decile histogram of
    # u_A across holdout persons.
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(anchor_rank_vals, bins=edges)
    anchor_rank_dist = {
        f"[{edges[i]:.1f},{edges[i + 1]:.1f})": int(hist[i])
        for i in range(len(hist))
    }

    # Cell-count distribution (reported-not-gated): train positive counts
    # per cell touched by the holdout, summarized.
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

    diagnostics = {
        "n_holdout_persons": int(n_persons),
        "chosen_parameters": {
            "a2": float(params["a2"]),
            "b2": float(params["b2"]),
            "c2": float(params["c2"]),
            "rho": float(params["rho"]),
        },
        "smm": {
            "min_sse": float(params["min_sse"]),
            "target_autocorr": params["target_autocorr"],
            "simulated_autocorr": params["simulated_autocorr"],
            "n_smm_persons": int(params["n_smm_persons"]),
            "n_grid_points_evaluated": int(params["n_grid_points_evaluated"]),
        },
        "anchor_rank_distribution": anchor_rank_dist,
        "cell_count_distribution": cell_count_summary,
        "clamped_rank_share": {
            "n_positive_generated": int(n_positive_gen),
            "n_clamped": int(n_clamped),
            "share": (
                float(n_clamped / n_positive_gen) if n_positive_gen else 0.0
            ),
            "note": (
                "share of generated POSITIVE periods whose rank U fell "
                "outside the cell's plotting-position span [wtil[0], "
                "wtil[-1]] and was clamped to the cell's [y_min, y_max]"
            ),
        },
    }
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
    verbose: bool,
) -> dict[str, Any]:
    """Fit, calibrate, generate, and score candidate 5b for one seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)

    # Stage 1: per-cell marginals on the train complement.
    marginals = fit_cell_marginals(train)

    # Stage 3: SMM grid on the fixed 5,000-person magnitude support.
    smm = run_smm(train, all_anchor, marginals, seed)

    # Stage 4: participation gate (train complement) + backward chain.
    fitted, pairs = fit_participation_gate(train, seed)
    candidate, diagnostics = generate_candidate(
        holdout, all_anchor, marginals, fitted, smm, seed
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
        "smm": {
            "chosen": {
                "a2": smm["a2"],
                "b2": smm["b2"],
                "c2": smm["c2"],
                "rho": smm["rho"],
            },
            "min_sse": smm["min_sse"],
            "target_autocorr": smm["target_autocorr"],
            "simulated_autocorr": smm["simulated_autocorr"],
            "n_smm_persons": smm["n_smm_persons"],
            "n_grid_points_evaluated": smm["n_grid_points_evaluated"],
        },
        "generation_diagnostics": diagnostics,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if verbose:
        s = smm
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"a2={s['a2']:.2f} rho={s['rho']:.2f} c2={s['c2']:.2f} "
            f"sse={s['min_sse']:.4f} "
            f"clamp={diagnostics['clamped_rank_share']['share']:.3f} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-5b run."""
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
        "run": "gate1_rank_v1",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "description": (
            "Gate-1 candidate 5b: rank-space generative dynamics. "
            "Empirical per-cell quantile marginals (five-year age bin x "
            "period) supply the magnitude; a fixed-form latent process "
            "z = a w + b x + c eps (person-permanent w, AR(1) x, "
            "idiosyncratic eps; a^2+b^2+c^2=1) drives the rank U = Phi(z) "
            "at which each period reads its cell's weighted empirical "
            "quantile; the latent's (a^2, rho, c^2) are calibrated by "
            "simulated method of moments on the TRAIN panel's "
            "among-positives autocorrelation at lags 1/2/5 over a "
            "registered grid; participation reuses the candidate-2 "
            "backward regime gate. Anchor keeps its real value; the state "
            "initializes by the exact Gaussian conditionals and chains "
            "backward one step per observed transition. Registered frozen "
            "before the run in issue #42 (see spec_registration). "
            "Candidate scored against the held-out PSID family earnings "
            "panel geometry (two locked views) and the locked moment "
            "battery, per the locked seed-level conjunction in gates.yaml "
            "(pull request 39). Protocol machinery imported byte-for-byte "
            "from the baseline runner (pull request 40)."
        ),
        "model": {
            "class": "rank-space generative (quantile marginal + latent Gaussian + SMM)",
            "stochastic": True,
            "populace_fit_used": True,
            "populace_fit_scope": (
                "participation gate only (RegimeGatedQRF sign gate); the "
                "magnitude simulation and SMM use pure numpy/scipy"
            ),
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
            },
            "latent": {
                "form": "z = a w + b x + c eps; U = Phi(z)",
                "w": "N(0,1) person-permanent",
                "x": "stationary AR(1) N(0,1), coefficient rho",
                "eps": "i.i.d. N(0,1)",
                "constraint": "a,b,c >= 0; a^2 + b^2 + c^2 = 1",
            },
            "smm": {
                "objective": (
                    "equal-weight SSE over among-positives autocorrelation "
                    "at lags 1/2/5 (locked definition) vs the TRAIN panel "
                    "targets"
                ),
                "subsample": (
                    "first 5,000 train persons with positive anchors "
                    "ordered by person_id, on their actual positive-period "
                    "support, anchored at their real anchor"
                ),
                "grid": {
                    "a2": list(A2_GRID),
                    "rho": list(RHO_GRID),
                    "c2": list(C2_GRID),
                    "b2_floor": B2_FLOOR,
                },
                "tie_breaks": ["smaller a2", "smaller rho", "smaller c2"],
                "rng": (
                    "one internal generator seeded from the gate seed; "
                    "shocks drawn once and reused across grid points "
                    "(common random numbers)"
                ),
            },
            "generation": {
                "anchor": (
                    "chronologically last observed period held at real "
                    "earnings"
                ),
                "anchor_rank": (
                    "positive anchor u_A = rhat(anchor value) at its cell; "
                    "zero anchor u_A = p0/2 of the cell; z_A = Phi^-1(u_A)"
                ),
                "state_init": (
                    "w | z_A ~ N(a z_A, 1 - a^2); "
                    "x_A | z_A, w ~ N(b (z_A - a w)/(b^2+c^2), "
                    "c^2/(b^2+c^2))"
                ),
                "chain": (
                    "backward x_prev = rho x_next + sqrt(1-rho^2) nu, one "
                    "step per observed transition (standing gap rule)"
                ),
                "period_draw": (
                    "z = a w + b x + c eps (fresh eps); U = Phi(z); where "
                    "the regime gate draws positive, earnings = Qhat_pos of "
                    "the period's cell at U"
                ),
                "participation": (
                    "candidate-2 backward regime gate (RegimeGatedQRF sign "
                    "gate) on (next generated level, current age), trained "
                    "on the 80% complement, populace-fit defaults"
                ),
                "rng": (
                    "distinct fixed-label substreams of the gate seed: w, "
                    "x-innovations, eps, gate"
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
        "smm_context": {
            "note": (
                "Reported-not-gated per seed: the chosen (a^2, b^2, c^2, "
                "rho), the SMM target-vs-simulated autocorrelation at lags "
                "1/2/5, and the minimum SSE. None enters the geometry or "
                "battery pass/fail; the gate rule names only those two "
                "families. Comparability note: the stage-1 decomposition "
                "estimates from candidate 3's artifact (sigma2_perm ~ "
                "0.39-0.44, rho ~ 0.76-0.79) live in LOG-EARNINGS space; "
                "5b's (a^2, rho) live in the latent GAUSSIAN COPULA space "
                "(z), so the two are not directly comparable -- the "
                "copula rho governs rank persistence, not log-level "
                "persistence."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "a2": s["smm"]["chosen"]["a2"],
                    "b2": s["smm"]["chosen"]["b2"],
                    "c2": s["smm"]["chosen"]["c2"],
                    "rho": s["smm"]["chosen"]["rho"],
                    "min_sse": s["smm"]["min_sse"],
                    "target_autocorr": s["smm"]["target_autocorr"],
                    "simulated_autocorr": s["smm"]["simulated_autocorr"],
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
