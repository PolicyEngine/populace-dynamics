"""Leakage-safe forward earnings law for the M6 projection engine.

The fitted law is a biennial conditional-rank chain.  All estimates use
earnings observations dated no later than 2014.  Odd projection years are a
deterministic carry; even years draw participation and then one donor rank,
which is mapped to a nominal level through a calendar-invariant,
NAWI-normalized age marginal.

This module deliberately owns the small pieces of candidate-5b/8/10
machinery used at deployment.  The gate runner scripts are not installed as
part of the package, so importing those scripts here would make the engine
depend on repository layout rather than on fitted inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import ndtr, ndtri

from populace_dynamics.engine.candidates import (
    RANK_REFRESH_OPERATION_ID,
    RANK_REFRESH_OPERATION_KIND,
    CandidateSpec,
)
from populace_dynamics.engine.rng import seed_from_generator

__all__ = [
    "AGE_BIN_WIDTH",
    "AGE_MAX",
    "AGE_MIN",
    "CellMarginal",
    "ForwardEarningsFit",
    "ForwardEarningsGenerator",
    "K_NEIGHBORS",
    "N_AGE_BINS",
    "ProjectedWageIndex",
    "RankRefreshFitAudit",
    "RankRefreshPreflightAbort",
    "RANK_CLAMP_HI",
    "RANK_CLAMP_LO",
    "SUBSTREAM_CODES",
    "age_bin",
    "fit_age_marginals",
    "fit_forward_earnings",
    "fit_projected_wage_index",
    "validate_rank_refresh_fit",
]

BOUNDARY_YEAR = 2014
PERIOD_STEP = 2
AGE_MIN = 25
AGE_MAX = 64
AGE_BIN_WIDTH = 5
N_AGE_BINS = 8
RANK_CLAMP_LO = 0.001
RANK_CLAMP_HI = 0.999

K_NEIGHBORS = 25
W_CURRENT = 1.0
W_PRIOR = 0.5
W_ANCHOR = 0.25
LAMBDA_FIXED = 0.1

GAMMA_LAGS = (0, 1, 2, 3, 4, 5)
RHO_GRID = tuple(round(0.50 + 0.01 * i, 2) for i in range(46))

SUBSTREAM_CODES = {
    "gate": 1,
    "donor-draw": 2,
    "re-entry-draw": 3,
    "memory-refresh-gate": 4,
    "memory-refresh-rank": 5,
}

FRAME_COLUMNS = (
    "person_id",
    "age",
    "sex",
    "u_w",
    "realized_earn_2014",
    "realized_earn_2012",
    "earnings",
    "gen_earn_w2",
    "gen_earn_w4",
)


@dataclass(frozen=True)
class RankRefreshFitAudit:
    """Publishable exact-bin eligibility record for the refresh operation."""

    source: str
    sort: tuple[str, str]
    target_age_bin_width: int
    k: int
    counts_by_bin: Mapping[str, int]
    checksums_by_bin: Mapping[str, str]
    partition_sha256: str
    empty_bins: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort", tuple(self.sort))
        object.__setattr__(
            self,
            "counts_by_bin",
            MappingProxyType(dict(self.counts_by_bin)),
        )
        object.__setattr__(
            self,
            "checksums_by_bin",
            MappingProxyType(dict(self.checksums_by_bin)),
        )
        object.__setattr__(self, "empty_bins", tuple(self.empty_bins))

    def __reduce__(self):
        return (
            type(self),
            (
                self.source,
                self.sort,
                self.target_age_bin_width,
                self.k,
                dict(self.counts_by_bin),
                dict(self.checksums_by_bin),
                self.partition_sha256,
                self.empty_bins,
            ),
        )

    @property
    def eligible(self) -> bool:
        return not self.empty_bins

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sort": list(self.sort),
            "target_age_bin_width": self.target_age_bin_width,
            "k": self.k,
            "counts_by_bin": dict(self.counts_by_bin),
            "checksums_by_bin": dict(self.checksums_by_bin),
            "partition_sha256": self.partition_sha256,
            "empty_bins": list(self.empty_bins),
            "empty_pool_check_passed": self.eligible,
            "eligible": self.eligible,
            "disposition": (
                "REGISTERABLE"
                if self.eligible
                else "NO_REGISTERABLE_EARNINGS_REFRESH_FIT"
            ),
        }


class RankRefreshPreflightAbort(RuntimeError):
    """A typed, publishable stop for a non-registerable exact-bin fit."""

    def __init__(self, audit: RankRefreshFitAudit) -> None:
        self.audit = audit
        bins = ", ".join(str(value) for value in audit.empty_bins)
        super().__init__(
            "NO_REGISTERABLE_EARNINGS_REFRESH_FIT: empty exact target-age "
            f"bin(s) {bins}; the ratified law permits no donor fallback"
        )

    def __reduce__(self):
        return (type(self), (self.audit,))


class QRFModel(Protocol):
    """The fit surface used from ``populace.fit.qrf``."""

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        predictors: list[str],
        targets: list[str],
        weights: str,
    ) -> Any: ...


QRFModelFactory = Callable[..., QRFModel]


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing column(s) {missing}")


def age_bin(age: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Return the pinned five-year age bin, clipping outside 25--64."""
    values = np.asarray(age, dtype=np.float64)
    index = np.floor((values - AGE_MIN) / AGE_BIN_WIDTH).astype(np.int64)
    return np.clip(index, 0, N_AGE_BINS - 1)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_mapping_checksum(mapping: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        array = np.ascontiguousarray(np.asarray(mapping[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _frame_checksum(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(list(columns), kind="stable").reset_index(
        drop=True
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _rank_refresh_probability(
    candidate_spec: CandidateSpec | None,
) -> float | None:
    if candidate_spec is None:
        return None
    operation = candidate_spec.operation(RANK_REFRESH_OPERATION_KIND)
    if operation is None:
        return None
    if operation.implementation_id != RANK_REFRESH_OPERATION_ID:
        raise ValueError(
            "candidate selects an unregistered forward-earnings rank refresh: "
            f"{operation.implementation_id!r}"
        )
    params = dict(operation.params)
    if set(params) != {"q", "substream_codes"}:
        raise ValueError(
            "rank-refresh operation parameters must be exactly q and "
            "substream_codes"
        )
    expected_codes = {
        "memory-refresh-gate": SUBSTREAM_CODES["memory-refresh-gate"],
        "memory-refresh-rank": SUBSTREAM_CODES["memory-refresh-rank"],
    }
    if dict(params["substream_codes"]) != expected_codes:
        raise ValueError("candidate rank-refresh substream binding drifted")
    q = float(params["q"])
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("rank-refresh q must lie in [0, 1]")
    return q


class CellMarginal:
    """Certified weighted zero mass and positive quantile/rank grid."""

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
        self.wtil = np.asarray(wtil, dtype=np.float64)
        self.yval = np.asarray(yval, dtype=np.float64)
        self.n_pos = int(n_pos)
        self.w_total = float(w_total)
        self.ymin = float(self.yval[0]) if len(self.yval) else np.nan
        self.ymax = float(self.yval[-1]) if len(self.yval) else np.nan

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Weighted empirical positive quantile, with certified clamping."""
        values = np.interp(u, self.wtil, self.yval)
        return np.clip(values, self.ymin, self.ymax)

    def rank(self, y: float) -> float:
        """Inverse positive quantile, with certified rank clamping."""
        rank = float(np.interp(y, self.yval, self.wtil))
        return float(np.clip(rank, RANK_CLAMP_LO, RANK_CLAMP_HI))


def _plotting_positions(
    y: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate-5b's stable weighted plotting-position grid."""
    order = np.argsort(y, kind="stable")
    values = y[order]
    weights = weight[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    return (cumulative - 0.5 * weights) / total, values


@dataclass(frozen=True)
class ProjectedWageIndex:
    """Pre-cutoff actual NAWI values and the pinned log-linear projection."""

    actual: Mapping[int, float]
    intercept: float
    slope: float
    boundary_year: int = BOUNDARY_YEAR

    def projected(self, year: int) -> float:
        """Return ``exp(alpha + beta * year)`` from the 2005--2014 fit."""
        return float(np.exp(self.intercept + self.slope * int(year)))

    def normalization_index(self, year: int) -> float:
        """Use realized pre-cutoff NAWI and projected post-cutoff NAWI."""
        year = int(year)
        if year <= self.boundary_year:
            try:
                return float(self.actual[year])
            except KeyError as error:
                raise KeyError(
                    f"pre-cutoff NAWI is unavailable for {year}"
                ) from error
        return self.projected(year)


def fit_projected_wage_index(
    nawi: Mapping[int, float], *, boundary_year: int = BOUNDARY_YEAR
) -> ProjectedWageIndex:
    """Fit ``ln(NAWI) ~ year`` over the pinned trailing decade."""
    years = np.arange(boundary_year - 9, boundary_year + 1, dtype=np.float64)
    values = np.asarray(
        [float(nawi[int(year)]) for year in years], dtype=np.float64
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("NAWI fit values must be finite and positive")
    centered = years - years.mean()
    logs = np.log(values)
    slope = float(
        np.dot(centered, logs - logs.mean()) / np.dot(centered, centered)
    )
    intercept = float(logs.mean() - slope * years.mean())
    actual = {
        int(year): float(value)
        for year, value in zip(years, values, strict=True)
    }
    return ProjectedWageIndex(
        actual=actual,
        intercept=intercept,
        slope=slope,
        boundary_year=int(boundary_year),
    )


def _precutoff_nawi(
    panel: pd.DataFrame,
    nawi: Mapping[int, float],
    *,
    boundary_year: int,
) -> dict[int, float]:
    years = sorted(
        int(year)
        for year in pd.to_numeric(panel["period"], errors="raise").unique()
        if int(year) <= boundary_year
    )
    required = sorted(
        set(years) | set(range(boundary_year - 9, boundary_year + 1))
    )
    out: dict[int, float] = {}
    for year in required:
        value = float(nawi[year])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"NAWI[{year}] must be finite and positive")
        out[year] = value
    return out


def fit_age_marginals(
    estimation_panel: pd.DataFrame,
    nawi: Mapping[int, float],
) -> dict[int, CellMarginal]:
    """Fit eight pooled NAWI-normalized age-bin marginals."""
    _require_columns(
        estimation_panel,
        ("earnings", "age", "period", "weight"),
        "earnings estimation panel",
    )
    frame = estimation_panel[["earnings", "age", "period", "weight"]].copy()
    frame["bin"] = age_bin(frame["age"].to_numpy())
    periods = frame["period"].to_numpy(dtype=np.int64)
    index = np.asarray([float(nawi[int(year)]) for year in periods])
    earnings = frame["earnings"].to_numpy(dtype=np.float64)
    frame["normalized_earnings"] = np.where(
        earnings > 0, earnings / index, 0.0
    )

    result: dict[int, CellMarginal] = {}
    for bin_index, cell_frame in frame.groupby("bin", sort=True):
        level = cell_frame["normalized_earnings"].to_numpy(dtype=np.float64)
        weight = cell_frame["weight"].to_numpy(dtype=np.float64)
        positive = level > 0
        total_weight = float(weight.sum())
        if total_weight <= 0:
            raise ValueError(
                f"age bin {int(bin_index)} has no positive weight"
            )
        p0 = float(weight[~positive].sum() / total_weight)
        if not positive.any():
            raise ValueError(
                f"age bin {int(bin_index)} has no positive earnings"
            )
        wtil, values = _plotting_positions(level[positive], weight[positive])
        result[int(bin_index)] = CellMarginal(
            p0,
            wtil,
            values,
            int(positive.sum()),
            float(weight[positive].sum()),
        )
    missing = sorted(set(range(N_AGE_BINS)) - set(result))
    if missing:
        raise ValueError(
            f"earnings marginal fit is missing age bin(s) {missing}"
        )
    return result


def _normalized_rank(
    marginals: Mapping[int, CellMarginal],
    level: float,
    age: float,
    year: int,
    index: Mapping[int, float],
) -> float:
    if level <= 0:
        raise ValueError("positive-rank lookup received a non-positive level")
    cell = marginals[int(age_bin([age])[0])]
    return cell.rank(float(level) / float(index[int(year)]))


def _anchor_rank(
    marginals: Mapping[int, CellMarginal],
    level: float,
    age: float,
    year: int,
    index: Mapping[int, float],
) -> float:
    cell = marginals[int(age_bin([age])[0])]
    if level > 0:
        return cell.rank(float(level) / float(index[int(year)]))
    return 0.5 * cell.p0


def _forward_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[["person_id", "period", "earnings", "age", "weight"]].copy()
    later = base.rename(
        columns={
            "earnings": "earnings_tp2",
            "age": "age_tp2",
            "weight": "weight_tp2",
        }
    )
    later["period"] = later["period"].to_numpy(dtype=np.int64) - PERIOD_STEP
    pairs = base.merge(
        later[
            [
                "person_id",
                "period",
                "earnings_tp2",
                "age_tp2",
                "weight_tp2",
            ]
        ],
        on=["person_id", "period"],
        how="inner",
        validate="one_to_one",
    )
    pairs["period_tp2"] = (
        pairs["period"].to_numpy(dtype=np.int64) + PERIOD_STEP
    )
    return pairs


def _pooled_autocovariances(
    residuals: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, int], float]:
    frame = residuals[["person_id", "period", "r"]].copy()
    pooled_mean = float(frame["r"].to_numpy(dtype=np.float64).mean())
    frame["rc"] = frame["r"].to_numpy(dtype=np.float64) - pooled_mean
    gamma: dict[int, float] = {}
    counts: dict[int, int] = {}
    centered = frame["rc"].to_numpy(dtype=np.float64)
    gamma[0] = float(np.mean(centered**2))
    counts[0] = int(centered.size)
    for lag in range(1, 6):
        lag_years = PERIOD_STEP * lag
        left = frame.rename(columns={"period": "p_l", "rc": "rc_l"})[
            ["person_id", "p_l", "rc_l"]
        ]
        right = frame.rename(columns={"period": "p_r", "rc": "rc_r"})
        right = right.assign(p_l=right["p_r"] - lag_years)
        joined = left.merge(
            right[["person_id", "p_l", "rc_r"]],
            on=["person_id", "p_l"],
            how="inner",
        )
        products = (joined["rc_l"] * joined["rc_r"]).to_numpy(dtype=np.float64)
        gamma[lag] = (
            float(np.mean(products)) if products.size else float("nan")
        )
        counts[lag] = int(products.size)
    if not np.isfinite([gamma[lag] for lag in GAMMA_LAGS]).all():
        raise ValueError(
            "u_w fit needs positive observations at every 0--10 year lag"
        )
    return gamma, counts, pooled_mean


def _fit_three_component(gamma: Mapping[int, float]) -> dict[str, float]:
    moments = np.asarray([gamma[lag] for lag in GAMMA_LAGS], dtype=np.float64)
    best: dict[str, float] | None = None
    for rho in RHO_GRID:
        design = np.zeros((6, 3), dtype=np.float64)
        design[:, 0] = 1.0
        design[0, 1] = 1.0
        design[0, 2] = 1.0
        for lag in range(1, 6):
            design[lag, 1] = rho**lag
        variance, residual_norm = nnls(design, moments)
        sse = float(residual_norm**2)
        if best is None or sse < best["sse"]:
            best = {
                "rho": float(rho),
                "sigma2_perm": float(variance[0]),
                "sigma2_trans": float(variance[1]),
                "sigma2_noise": float(variance[2]),
                "sse": sse,
            }
    assert best is not None
    gamma0 = float(gamma[0])
    best["gamma_0"] = gamma0
    best["implied_perm_share"] = (
        best["sigma2_perm"] / gamma0 if gamma0 > 0 else 0.0
    )
    return best


def _person_effects(
    residuals: pd.DataFrame,
    fit: Mapping[str, float],
    pooled_mean: float,
    person_ids: np.ndarray,
) -> pd.DataFrame:
    s2_perm = float(fit["sigma2_perm"])
    s2_trans = float(fit["sigma2_trans"])
    s2_noise = float(fit["sigma2_noise"])
    rho = float(fit["rho"])
    frame = residuals[["person_id", "period", "r"]].copy()
    frame["rc"] = frame["r"].to_numpy(dtype=np.float64) - pooled_mean
    rows: list[dict[str, float | int]] = []
    for person_id, group in frame.groupby("person_id", sort=True):
        periods = group["period"].to_numpy(dtype=np.float64)
        centered = group["rc"].to_numpy(dtype=np.float64)
        count = int(len(periods))
        lag = np.abs(periods[:, None] - periods[None, :]) / PERIOD_STEP
        ar_sum = float(np.sum(rho**lag))
        noise_variance = (s2_trans * ar_sum + count * s2_noise) / count**2
        shrink = (
            s2_perm / (s2_perm + noise_variance)
            if s2_perm + noise_variance > 0
            else 0.0
        )
        rows.append(
            {
                "person_id": int(person_id),
                "perm": shrink * float(centered.mean()),
                "n_pos": count,
            }
        )
    effects = pd.DataFrame(rows, columns=["person_id", "perm", "n_pos"])
    result = pd.DataFrame({"person_id": person_ids.astype(np.int64)}).merge(
        effects, on="person_id", how="left", validate="one_to_one"
    )
    result["perm"] = result["perm"].fillna(0.0)
    result["n_pos"] = result["n_pos"].fillna(0)
    return result


def _fit_u_w(
    panel: pd.DataFrame,
    marginals: Mapping[int, CellMarginal],
    index: Mapping[int, float],
) -> tuple[dict[int, float], dict[str, Any]]:
    positive = panel[panel["earnings"] > 0][
        ["person_id", "period", "earnings", "age"]
    ].copy()
    ranks = np.asarray(
        [
            _normalized_rank(
                marginals,
                float(row.earnings),
                float(row.age),
                int(row.period),
                index,
            )
            for row in positive.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    z_panel = pd.DataFrame(
        {
            "person_id": positive["person_id"].to_numpy(dtype=np.int64),
            "period": positive["period"].to_numpy(dtype=np.int64),
            "r": ndtri(ranks),
        }
    )
    gamma, counts, pooled_mean = _pooled_autocovariances(z_panel)
    component_fit = _fit_three_component(gamma)
    person_ids = np.sort(panel["person_id"].unique().astype(np.int64))
    effects = _person_effects(z_panel, component_fit, pooled_mean, person_ids)
    sigma_w = float(np.sqrt(component_fit["sigma2_perm"]))
    if sigma_w > 0:
        values = ndtr(effects["perm"].to_numpy(dtype=np.float64) / sigma_w)
    else:
        values = np.full(len(effects), 0.5, dtype=np.float64)
    mapping = {
        int(person_id): float(value)
        for person_id, value in zip(
            effects["person_id"].to_numpy(), values, strict=True
        )
    }
    diagnostics = {
        "gamma": {str(lag): float(gamma[lag]) for lag in GAMMA_LAGS},
        "gamma_pair_counts": {
            str(lag): int(counts[lag]) for lag in GAMMA_LAGS
        },
        "pooled_z_mean": float(pooled_mean),
        **component_fit,
        "sigma_hat_w": sigma_w,
        "n_z_panel_rows": int(len(z_panel)),
    }
    return mapping, diagnostics


def _build_donor_pools(
    panel: pd.DataFrame,
    pairs: pd.DataFrame,
    anchors: pd.DataFrame,
    marginals: Mapping[int, CellMarginal],
    index: Mapping[int, float],
    u_w_by_person: Mapping[int, float],
) -> dict[str, dict[str, np.ndarray]]:
    anchor_rank = {
        int(row.person_id): _anchor_rank(
            marginals,
            float(row.earnings),
            float(row.age),
            int(row.period),
            index,
        )
        for row in anchors.itertuples(index=False)
    }
    person_id = pairs["person_id"].to_numpy(dtype=np.int64)
    source_level = pairs["earnings"].to_numpy(dtype=np.float64)
    target_level = pairs["earnings_tp2"].to_numpy(dtype=np.float64)
    source_age = pairs["age"].to_numpy(dtype=np.float64)
    target_age = pairs["age_tp2"].to_numpy(dtype=np.float64)
    source_period = pairs["period"].to_numpy(dtype=np.int64)
    target_period = pairs["period_tp2"].to_numpy(dtype=np.int64)
    target_weight = pairs["weight_tp2"].to_numpy(dtype=np.float64)

    both_positive = (source_level > 0) & (target_level > 0)
    positive_index = np.flatnonzero(both_positive)
    u_source = np.asarray(
        [
            _normalized_rank(
                marginals,
                source_level[position],
                source_age[position],
                source_period[position],
                index,
            )
            for position in positive_index
        ]
    )
    u_target = np.asarray(
        [
            _normalized_rank(
                marginals,
                target_level[position],
                target_age[position],
                target_period[position],
                index,
            )
            for position in positive_index
        ]
    )
    positive_person = person_id[positive_index]
    positive_period = target_period[positive_index]
    positive_weight = target_weight[positive_index]
    positive_anchor = np.asarray(
        [anchor_rank[int(pid)] for pid in positive_person]
    )
    positive_u_w = np.asarray(
        [u_w_by_person[int(pid)] for pid in positive_person]
    )

    positive_rows = panel[panel["earnings"] > 0]
    prior_lookup = {
        (int(row.person_id), int(row.period)): (
            float(row.earnings),
            float(row.age),
        )
        for row in positive_rows.itertuples(index=False)
    }
    prior_keys = [
        (int(pid), int(period - 2))
        for pid, period in zip(
            positive_person, source_period[positive_index], strict=True
        )
    ]
    triple_mask = np.asarray(
        [key in prior_lookup for key in prior_keys], dtype=bool
    )
    triple_positions = np.flatnonzero(triple_mask)
    u_prior = np.asarray(
        [
            _normalized_rank(
                marginals,
                prior_lookup[prior_keys[position]][0],
                prior_lookup[prior_keys[position]][1],
                prior_keys[position][1],
                index,
            )
            for position in triple_positions
        ]
    )

    pair_order = np.lexsort((positive_period, positive_person))
    pair_pool = {
        "u_t": u_source[pair_order],
        "u_tp2": u_target[pair_order],
        "u_A": positive_anchor[pair_order],
        "u_w": positive_u_w[pair_order],
        "weight": positive_weight[pair_order],
        "person_id": positive_person[pair_order],
        "period_tp2": positive_period[pair_order],
    }
    triple_person = positive_person[triple_positions]
    triple_period = positive_period[triple_positions]
    triple_order = np.lexsort((triple_period, triple_person))
    triple_pool = {
        "u_t": u_source[triple_positions][triple_order],
        "u_tm2": u_prior[triple_order],
        "u_tp2": u_target[triple_positions][triple_order],
        "u_A": positive_anchor[triple_positions][triple_order],
        "u_w": positive_u_w[triple_positions][triple_order],
        "weight": positive_weight[triple_positions][triple_order],
        "person_id": triple_person[triple_order],
        "period_tp2": triple_period[triple_order],
    }

    reentry_index = np.flatnonzero((source_level == 0) & (target_level > 0))
    reentry_person = person_id[reentry_index]
    reentry_period = target_period[reentry_index]
    reentry_target = np.asarray(
        [
            _normalized_rank(
                marginals,
                target_level[position],
                target_age[position],
                target_period[position],
                index,
            )
            for position in reentry_index
        ]
    )
    reentry_anchor = np.asarray(
        [anchor_rank[int(pid)] for pid in reentry_person]
    )
    reentry_u_w = np.asarray(
        [u_w_by_person[int(pid)] for pid in reentry_person]
    )
    reentry_order = np.lexsort((reentry_period, reentry_person))
    reentry_pool = {
        "u_tp2": reentry_target[reentry_order],
        "u_A": reentry_anchor[reentry_order],
        "u_w": reentry_u_w[reentry_order],
        "weight": target_weight[reentry_index][reentry_order],
        "person_id": reentry_person[reentry_order],
        "period_tp2": reentry_period[reentry_order],
    }
    return {
        "pairs": pair_pool,
        "triples": triple_pool,
        "reentry": reentry_pool,
    }


def _build_stable_pools(
    pairs: pd.DataFrame,
    pair_pool: Mapping[str, np.ndarray],
) -> tuple[dict[int, dict[str, np.ndarray]], RankRefreshFitAudit]:
    """Partition the incumbent positive-pair pool by exact target age."""
    positive = pairs.loc[
        (pairs["earnings"] > 0) & (pairs["earnings_tp2"] > 0)
    ].copy()
    person = positive["person_id"].to_numpy(dtype=np.int64)
    period = positive["period_tp2"].to_numpy(dtype=np.int64)
    order = np.lexsort((period, person))
    positive = positive.iloc[order].reset_index(drop=True)
    if not np.array_equal(
        positive["person_id"].to_numpy(dtype=np.int64),
        np.asarray(pair_pool["person_id"], dtype=np.int64),
    ) or not np.array_equal(
        positive["period_tp2"].to_numpy(dtype=np.int64),
        np.asarray(pair_pool["period_tp2"], dtype=np.int64),
    ):
        raise AssertionError(
            "stable-pool source does not align with incumbent pair order"
        )
    if not positive["age_tp2"].between(AGE_MIN, AGE_MAX).all():
        raise AssertionError("stable-pool target ages escaped 25-64")

    bins = age_bin(positive["age_tp2"].to_numpy(dtype=np.float64))
    by_bin: dict[int, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    checksums: dict[str, str] = {}
    empty: list[int] = []
    for bin_index in range(N_AGE_BINS):
        selected = bins == bin_index
        pool = {
            name: np.asarray(values)[selected]
            for name, values in pair_pool.items()
        }
        count = int(selected.sum())
        weights = np.asarray(pool["weight"], dtype=np.float64)
        if count == 0:
            empty.append(bin_index)
        elif not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(
                f"stable target-age bin {bin_index} has invalid weights"
            )
        by_bin[bin_index] = pool
        counts[str(bin_index)] = count
        checksums[str(bin_index)] = _array_mapping_checksum(pool)
    if sum(counts.values()) != len(pair_pool["person_id"]):
        raise AssertionError("stable age-bin partition lost donor rows")

    audit = RankRefreshFitAudit(
        source="incumbent positive-to-positive pair pool",
        sort=("person_id", "period_tp2"),
        target_age_bin_width=AGE_BIN_WIDTH,
        k=K_NEIGHBORS,
        counts_by_bin=counts,
        checksums_by_bin=checksums,
        partition_sha256=_canonical_sha256(
            {
                "age_bin": bins,
                **{
                    name: np.asarray(value)
                    for name, value in pair_pool.items()
                },
            }
        ),
        empty_bins=tuple(empty),
    )
    return by_bin, audit


def _default_qrf_factory(*, seed: int) -> QRFModel:
    try:
        from populace.fit.qrf import RegimeGatedQRF
    except ImportError as error:
        raise ImportError(
            "forward earnings refitting needs populace-fit; install its "
            "dedicated environment or inject qrf_factory"
        ) from error
    return RegimeGatedQRF(seed=seed)


@dataclass(frozen=True)
class ForwardEarningsFit:
    """Fitted generator plus its leakage-auditable estimation surfaces."""

    generator: ForwardEarningsGenerator
    estimation_panel: pd.DataFrame
    forward_pairs: pd.DataFrame
    anchors: pd.DataFrame
    n_zero_anchor_pairs: int
    u_w_diagnostics: Mapping[str, Any]
    rank_refresh_fit_audit: RankRefreshFitAudit | None = None
    q_invariant_fit_signature_sha256: str | None = None


def fit_forward_earnings(
    panel: pd.DataFrame,
    nawi: Mapping[int, float],
    *,
    seed: int,
    boundary_year: int = BOUNDARY_YEAR,
    qrf_factory: QRFModelFactory | None = None,
    candidate_spec: CandidateSpec | None = None,
) -> ForwardEarningsFit:
    """Fit the pinned forward gates, rank pools, marginals, and frame state."""
    rank_refresh_q = _rank_refresh_probability(candidate_spec)
    _require_columns(
        panel,
        ("person_id", "period", "earnings", "age", "weight"),
        "earnings panel",
    )
    period = pd.to_numeric(panel["period"], errors="raise")
    cutoff_panel = panel.loc[
        (period <= boundary_year) & (panel["weight"] > 0)
    ].copy()
    cutoff_panel = cutoff_panel.sort_values(
        ["person_id", "period"]
    ).reset_index(drop=True)
    if cutoff_panel.duplicated(["person_id", "period"]).any():
        raise ValueError("earnings panel has duplicate person-period rows")
    estimation = cutoff_panel.loc[
        cutoff_panel["age"].between(AGE_MIN, AGE_MAX)
    ].copy()
    estimation = estimation.sort_values(["person_id", "period"]).reset_index(
        drop=True
    )
    if estimation.empty:
        raise ValueError("forward earnings refit has no admissible rows")
    actual_index = _precutoff_nawi(
        cutoff_panel, nawi, boundary_year=boundary_year
    )
    wage_projection = fit_projected_wage_index(
        nawi, boundary_year=boundary_year
    )
    wage_projection = ProjectedWageIndex(
        actual=actual_index,
        intercept=wage_projection.intercept,
        slope=wage_projection.slope,
        boundary_year=boundary_year,
    )
    marginals = fit_age_marginals(estimation, actual_index)
    u_w_by_person, u_w_diagnostics = _fit_u_w(
        estimation, marginals, actual_index
    )

    anchors = cutoff_panel[cutoff_panel["period"] == boundary_year][
        ["person_id", "period", "earnings", "age", "weight"]
    ].copy()
    if anchors["person_id"].duplicated().any():
        raise ValueError("forward earnings fit has duplicate 2014 anchors")
    anchor_ids = set(anchors["person_id"].astype(int))
    if not anchor_ids:
        raise ValueError("forward earnings fit has no realized-2014 anchors")
    u_w_by_person = {
        person_id: float(u_w_by_person.get(person_id, 0.5))
        for person_id in anchor_ids
    }
    pairs = _forward_pairs(estimation)
    pairs = pairs[pairs["person_id"].isin(anchor_ids)].reset_index(drop=True)
    if pairs.empty:
        raise ValueError(
            "forward earnings refit has no adjacent biennial pairs"
        )

    factory = qrf_factory or _default_qrf_factory
    fit_kwargs = {
        "predictors": ["earnings", "age_tp2"],
        "targets": ["earnings_tp2"],
        "weights": "weight_tp2",
    }
    shared_gate = factory(seed=int(seed)).fit(pairs, **fit_kwargs)
    zero_ids = set(
        anchors.loc[anchors["earnings"] == 0, "person_id"].astype(int)
    )
    zero_pairs = pairs[pairs["person_id"].isin(zero_ids)].reset_index(
        drop=True
    )
    zero_gate = (
        factory(seed=int(seed)).fit(zero_pairs, **fit_kwargs)
        if len(zero_pairs)
        else None
    )
    pools = _build_donor_pools(
        estimation,
        pairs,
        anchors,
        marginals,
        actual_index,
        u_w_by_person,
    )
    stable_pools: Mapping[int, Mapping[str, np.ndarray]] | None = None
    refresh_audit: RankRefreshFitAudit | None = None
    if rank_refresh_q is not None:
        stable_pools, refresh_audit = _build_stable_pools(
            pairs, pools["pairs"]
        )

    realized_2014 = {
        int(row.person_id): float(row.earnings)
        for row in anchors.itertuples(index=False)
    }
    rows_2012 = cutoff_panel[
        cutoff_panel["period"] == boundary_year - PERIOD_STEP
    ]
    realized_2012 = {
        int(row.person_id): float(row.earnings)
        for row in rows_2012.itertuples(index=False)
    }
    generator = ForwardEarningsGenerator(
        shared_gate=shared_gate,
        zero_anchor_gate=zero_gate,
        marginals=marginals,
        pools=pools,
        wage_index=wage_projection,
        u_w_by_person=u_w_by_person,
        realized_earn_2014_by_person=realized_2014,
        realized_earn_2012_by_person=realized_2012,
        boundary_year=boundary_year,
        rank_refresh_q=rank_refresh_q,
        stable_pools=stable_pools,
        rank_refresh_fit_audit=refresh_audit,
    )
    fit_signature = (
        None
        if rank_refresh_q is None
        else _q_invariant_fit_signature(
            generator,
            estimation,
            pairs,
            anchors,
            u_w_diagnostics,
        )
    )
    return ForwardEarningsFit(
        generator=generator,
        estimation_panel=estimation,
        forward_pairs=pairs,
        anchors=anchors.reset_index(drop=True),
        n_zero_anchor_pairs=int(len(zero_pairs)),
        u_w_diagnostics=u_w_diagnostics,
        rank_refresh_fit_audit=refresh_audit,
        q_invariant_fit_signature_sha256=fit_signature,
    )


def _substream(seed: int, label: str) -> np.random.Generator:
    code = SUBSTREAM_CODES[label]
    return np.random.default_rng(np.random.SeedSequence([int(seed), code]))


def _gate_sign_draw(
    fitted: Any,
    current_level: np.ndarray,
    target_age: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Candidate-10's externally-driven RegimeGatedQRF sign draw."""
    if hasattr(fitted, "draw_sign"):
        return np.asarray(
            fitted.draw_sign(current_level, target_age, uniforms),
            dtype=np.int64,
        )
    model = fitted._target_models["earnings_tp2"]
    features = pd.DataFrame({"earnings": current_level, "age_tp2": target_age})
    values = features.loc[:, list(model.columns)].to_numpy(dtype=np.float64)
    probability = np.asarray(model.gate.predict_proba(values))
    cumulative = np.cumsum(probability, axis=1)
    selected = (cumulative >= uniforms[:, None]).argmax(axis=1)
    return np.asarray(model.gate.classes_)[selected]


def _knn_draw(
    distance: np.ndarray,
    weight: np.ndarray,
    donor_rank: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Candidate-7's weighted one-record draw with its stable tie-break."""
    n_query, n_donor = distance.shape
    if n_donor == 0:
        raise ValueError("forward earnings selected an empty donor pool")
    k = min(K_NEIGHBORS, n_donor)
    margin = min(n_donor, k + 8)
    candidates = np.argpartition(distance, margin - 1, axis=1)[:, :margin]
    output = np.empty(n_query, dtype=np.float64)
    for row in range(n_query):
        candidate = candidates[row]
        order = np.lexsort((candidate, distance[row, candidate]))
        selected = candidate[order[:k]]
        selected_weight = weight[selected]
        cumulative = np.cumsum(selected_weight)
        total = cumulative[-1]
        if not np.isfinite(total) or total <= 0:
            raise ValueError("forward earnings donor weights are not positive")
        chosen = int((cumulative >= uniforms[row] * total).argmax())
        output[row] = donor_rank[selected[chosen]]
    return output


def _gate_fit_signature(
    fitted_gate: Any | None, pairs: pd.DataFrame
) -> Mapping[str, Any]:
    if fitted_gate is None:
        return {"present": False}
    record: dict[str, Any] = {
        "present": True,
        "class": (
            f"{type(fitted_gate).__module__}."
            f"{type(fitted_gate).__qualname__}"
        ),
    }
    try:
        record["state_pickle_protocol"] = 5
        record["state_sha256"] = hashlib.sha256(
            pickle.dumps(fitted_gate, protocol=5)
        ).hexdigest()
    except (pickle.PickleError, TypeError, AttributeError):
        record["state_sha256"] = _canonical_sha256(
            getattr(fitted_gate, "__dict__", {})
        )

    target_models = getattr(fitted_gate, "_target_models", None)
    if target_models is None or "earnings_tp2" not in target_models:
        return record
    target = target_models["earnings_tp2"]
    columns = tuple(target.columns)
    gate = target.gate
    if gate is None:
        return record
    ordered = pairs.sort_values(["person_id", "period_tp2"], kind="stable")
    features = np.ascontiguousarray(
        ordered.loc[:, list(columns)].to_numpy(dtype="<f8")
    )
    classes = np.ascontiguousarray(np.asarray(gate.classes_, dtype="<i8"))
    probability = np.ascontiguousarray(
        np.asarray(gate.predict_proba(features), dtype="<f8")
    )
    digest = hashlib.sha256()
    for array in (features, classes, probability):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    record.update(
        {
            "columns": list(columns),
            "classes": classes,
            "canonical_surface_rows": len(features),
            "canonical_surface_sha256": digest.hexdigest(),
        }
    )
    return record


def _q_invariant_fit_signature(
    generator: ForwardEarningsGenerator,
    estimation_panel: pd.DataFrame,
    forward_pairs: pd.DataFrame,
    anchors: pd.DataFrame,
    u_w_diagnostics: Mapping[str, Any],
) -> str:
    marginal_payload = {
        str(index): {
            "p0": cell.p0,
            "wtil": cell.wtil,
            "yval": cell.yval,
            "n_pos": cell.n_pos,
            "w_total": cell.w_total,
        }
        for index, cell in generator.marginals.items()
    }
    stable = generator.stable_pools
    record = {
        "checksums": {
            "estimation_rows": _frame_checksum(
                estimation_panel,
                ("person_id", "period", "earnings", "age", "weight"),
            ),
            "forward_pairs": _frame_checksum(
                forward_pairs,
                (
                    "person_id",
                    "period",
                    "period_tp2",
                    "earnings",
                    "earnings_tp2",
                    "age",
                    "age_tp2",
                    "weight",
                    "weight_tp2",
                ),
            ),
            "anchors": _frame_checksum(
                anchors,
                ("person_id", "period", "earnings", "age", "weight"),
            ),
            "marginals": _canonical_sha256(marginal_payload),
            "u_w": _canonical_sha256(sorted(generator.u_w_by_person.items())),
            "wage_index": _canonical_sha256(
                {
                    "actual": generator.wage_index.actual,
                    "intercept": generator.wage_index.intercept,
                    "slope": generator.wage_index.slope,
                    "boundary_year": generator.wage_index.boundary_year,
                }
            ),
        },
        "donor_pools": {
            name: _array_mapping_checksum(pool)
            for name, pool in generator.pools.items()
        },
        "stable_pools": (
            None
            if stable is None
            else {
                str(index): _array_mapping_checksum(pool)
                for index, pool in stable.items()
            }
        ),
        "stable_pool_audit": (
            None
            if generator.rank_refresh_fit_audit is None
            else generator.rank_refresh_fit_audit.as_dict()
        ),
        "u_w_diagnostics": u_w_diagnostics,
        "participation_gates": {
            "shared": _gate_fit_signature(
                generator.shared_gate, forward_pairs
            ),
            "zero_anchor": _gate_fit_signature(
                generator.zero_anchor_gate, forward_pairs
            ),
        },
    }
    return _canonical_sha256(record)


@dataclass(frozen=True)
class ForwardEarningsGenerator:
    """State-through-frame implementation of the pinned M6 forward law."""

    shared_gate: Any
    zero_anchor_gate: Any | None
    marginals: Mapping[int, CellMarginal]
    pools: Mapping[str, Mapping[str, np.ndarray]]
    wage_index: ProjectedWageIndex
    u_w_by_person: Mapping[int, float]
    realized_earn_2014_by_person: Mapping[int, float]
    realized_earn_2012_by_person: Mapping[int, float]
    boundary_year: int = BOUNDARY_YEAR
    rank_refresh_q: float | None = None
    stable_pools: Mapping[int, Mapping[str, np.ndarray]] | None = None
    rank_refresh_fit_audit: RankRefreshFitAudit | None = None

    def __post_init__(self) -> None:
        bindings = (
            self.rank_refresh_q,
            self.stable_pools,
            self.rank_refresh_fit_audit,
        )
        n_supplied = sum(value is not None for value in bindings)
        if n_supplied not in (0, len(bindings)):
            raise ValueError(
                "rank-refresh q, stable pools, and fit audit must be bound "
                "together"
            )
        if not n_supplied:
            return
        assert self.rank_refresh_q is not None
        assert self.stable_pools is not None
        if (
            not np.isfinite(self.rank_refresh_q)
            or not 0.0 <= self.rank_refresh_q <= 1.0
        ):
            raise ValueError("rank-refresh q must lie in [0, 1]")
        if set(self.stable_pools) != set(range(N_AGE_BINS)):
            raise ValueError("stable pools do not cover all exact age bins")

    def materialize_initial_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add the concrete period-zero state required by ``generate``."""
        _require_columns(
            frame, ("person_id", "year", "age", "sex"), "initial frame"
        )
        years = pd.to_numeric(frame["year"], errors="raise").unique()
        if len(years) != 1 or int(years[0]) != self.boundary_year:
            raise ValueError(
                f"forward earnings state must be materialized at {self.boundary_year}"
            )
        person_ids = frame["person_id"].to_numpy(dtype=np.int64)
        missing_2014 = sorted(
            int(pid)
            for pid in person_ids
            if int(pid) not in self.realized_earn_2014_by_person
        )
        missing_u_w = sorted(
            int(pid)
            for pid in person_ids
            if int(pid) not in self.u_w_by_person
        )
        if missing_2014 or missing_u_w:
            raise ValueError(
                "initial frame has no fitted earnings state for "
                f"2014={missing_2014}, u_w={missing_u_w}"
            )
        realized_2014 = np.asarray(
            [
                self.realized_earn_2014_by_person[int(pid)]
                for pid in person_ids
            ],
            dtype=np.float64,
        )
        realized_2012 = np.asarray(
            [
                # An unobserved 2012 start-lag stays missing: the pinned 2016
                # pair ramp never conditions on it, and the adapter replaces
                # gen_earn_w4 with realized 2014 before the 2018 triple draw.
                self.realized_earn_2012_by_person.get(int(pid), np.nan)
                for pid in person_ids
            ],
            dtype=np.float64,
        )
        out = frame.copy()
        out["u_w"] = np.asarray(
            [self.u_w_by_person[int(pid)] for pid in person_ids],
            dtype=np.float64,
        )
        out["realized_earn_2014"] = realized_2014
        out["realized_earn_2012"] = realized_2012
        out["earnings"] = realized_2014
        out["gen_earn_w2"] = realized_2014
        out["gen_earn_w4"] = realized_2012
        return out

    def rank_to_level(
        self, rank: np.ndarray, age: np.ndarray, year: int
    ) -> np.ndarray:
        """Map positive ranks to nominal projected levels."""
        rank = np.asarray(rank, dtype=np.float64)
        age = np.asarray(age, dtype=np.float64)
        if rank.shape != age.shape:
            raise ValueError("rank and age must have the same shape")
        output = np.empty(rank.shape, dtype=np.float64)
        bins = age_bin(age)
        for bin_index in np.unique(bins):
            selected = bins == bin_index
            output[selected] = self.marginals[int(bin_index)].quantile(
                rank[selected]
            )
        return output * self.wage_index.projected(year)

    def level_to_rank(
        self, level: np.ndarray, age: np.ndarray, year: int
    ) -> np.ndarray:
        """Invert the deployed marginal for positive carried levels."""
        level = np.asarray(level, dtype=np.float64)
        age = np.asarray(age, dtype=np.float64)
        if level.shape != age.shape:
            raise ValueError("level and age must have the same shape")
        if np.any(level <= 0):
            raise ValueError(
                "positive-rank lookup received a non-positive level"
            )
        normalized = level / self.wage_index.normalization_index(year)
        bins = age_bin(age)
        return np.asarray(
            [
                self.marginals[int(bin_index)].rank(float(value))
                for bin_index, value in zip(bins, normalized, strict=True)
            ],
            dtype=np.float64,
        )

    def _anchor_ranks(
        self, level: np.ndarray, current_age: np.ndarray, target_year: int
    ) -> np.ndarray:
        anchor_age = current_age - (target_year - self.boundary_year)
        bins = age_bin(anchor_age)
        normalized = level / self.wage_index.normalization_index(
            self.boundary_year
        )
        output = np.empty(len(level), dtype=np.float64)
        for index, (value, bin_index) in enumerate(
            zip(normalized, bins, strict=True)
        ):
            cell = self.marginals[int(bin_index)]
            output[index] = (
                cell.rank(float(value)) if level[index] > 0 else 0.5 * cell.p0
            )
        return output

    @staticmethod
    def _third_coordinate(
        pool: Mapping[str, np.ndarray], q0: bool
    ) -> np.ndarray:
        if q0:
            return pool["u_A"]
        return LAMBDA_FIXED * pool["u_w"] + (1.0 - LAMBDA_FIXED) * pool["u_A"]

    def _draw_reentry(
        self,
        indices: np.ndarray,
        q0: np.ndarray,
        anchor_rank: np.ndarray,
        rng: np.random.Generator,
        output_rank: np.ndarray,
    ) -> None:
        pool = self.pools["reentry"]
        for is_q0 in (False, True):
            selected = indices[q0[indices] == is_q0]
            if not len(selected):
                continue
            coordinate = self._third_coordinate(pool, bool(is_q0))
            distance = np.abs(
                coordinate[None, :] - anchor_rank[selected][:, None]
            )
            uniforms = rng.random(len(selected))
            output_rank[selected] = _knn_draw(
                distance, pool["weight"], pool["u_tp2"], uniforms
            )

    def _draw_transition(
        self,
        indices: np.ndarray,
        q0: np.ndarray,
        current_rank: np.ndarray,
        prior_rank: np.ndarray,
        anchor_rank: np.ndarray,
        rng: np.random.Generator,
        output_rank: np.ndarray,
        *,
        triple: bool,
    ) -> None:
        pool = self.pools["triples" if triple else "pairs"]
        for is_q0 in (False, True):
            selected = indices[q0[indices] == is_q0]
            if not len(selected):
                continue
            coordinate = self._third_coordinate(pool, bool(is_q0))
            distance = W_CURRENT * np.abs(
                pool["u_t"][None, :] - current_rank[selected][:, None]
            )
            if triple:
                distance += W_PRIOR * np.abs(
                    pool["u_tm2"][None, :] - prior_rank[selected][:, None]
                )
            distance += W_ANCHOR * np.abs(
                coordinate[None, :] - anchor_rank[selected][:, None]
            )
            uniforms = rng.random(len(selected))
            output_rank[selected] = _knn_draw(
                distance, pool["weight"], pool["u_tp2"], uniforms
            )

    def _draw_rank_refresh(
        self,
        indices: np.ndarray,
        q0: np.ndarray,
        anchor_rank: np.ndarray,
        target_age: np.ndarray,
        gate_rng: np.random.Generator,
        stable_rng: np.random.Generator,
        output_rank: np.ndarray,
    ) -> None:
        """Draw both isolated uniforms before applying the q threshold."""
        assert self.rank_refresh_q is not None
        assert self.stable_pools is not None
        refresh_uniforms = gate_rng.random(len(indices))
        stable_uniforms = stable_rng.random(len(indices))
        stable_ranks = np.full(len(indices), np.nan, dtype=np.float64)
        target_bins = age_bin(target_age)
        for bin_index in range(N_AGE_BINS):
            pool = self.stable_pools[bin_index]
            for is_q0 in (False, True):
                selected_positions = np.flatnonzero(
                    (target_bins[indices] == bin_index)
                    & (q0[indices] == is_q0)
                )
                if not len(selected_positions):
                    continue
                query_indices = indices[selected_positions]
                coordinate = self._third_coordinate(pool, is_q0)
                distance = np.abs(
                    coordinate[None, :] - anchor_rank[query_indices][:, None]
                )
                stable_ranks[selected_positions] = _knn_draw(
                    distance,
                    np.asarray(pool["weight"]),
                    np.asarray(pool["u_tp2"]),
                    stable_uniforms[selected_positions],
                )
        if len(indices) and not np.isfinite(stable_ranks).all():
            raise AssertionError(
                "stable donor draw left an eligible rank unset"
            )
        refreshed = refresh_uniforms < self.rank_refresh_q
        output_rank[indices[refreshed]] = stable_ranks[refreshed]

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw one projection year's levels, with no RNG use on odd years."""
        year = int(year)
        if year % PERIOD_STEP:
            _require_columns(frame, ("earnings",), "forward earnings frame")
            return frame["earnings"].to_numpy(dtype=np.float64, copy=True)
        if year <= self.boundary_year:
            raise ValueError("forward earnings even draws begin after 2014")
        validate_rank_refresh_fit(self)
        _require_columns(frame, FRAME_COLUMNS, "forward earnings frame")

        # Select only the pinned conditioning surface.  In particular, this
        # intentionally never asks pandas for the frame's realized ``nawi``.
        conditioning = frame.loc[:, list(FRAME_COLUMNS)]
        order = np.argsort(
            conditioning["person_id"].to_numpy(dtype=np.int64), kind="stable"
        )
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        sorted_frame = conditioning.iloc[order]
        current_age = sorted_frame["age"].to_numpy(dtype=np.float64)
        current_level = sorted_frame["earnings"].to_numpy(dtype=np.float64)
        lag_level = sorted_frame["gen_earn_w2"].to_numpy(dtype=np.float64)
        prior_level = sorted_frame["gen_earn_w4"].to_numpy(dtype=np.float64)
        anchor_level = sorted_frame["realized_earn_2014"].to_numpy(
            dtype=np.float64
        )
        # Schema validation reads u_w and sex, while the pinned donor-side
        # lambda law correctly leaves both out of target distances.
        target_u_w = sorted_frame["u_w"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(target_u_w).all()
            or sorted_frame["sex"].isna().any()
        ):
            raise ValueError("forward earnings frame has invalid u_w or sex")
        required_numeric = np.column_stack(
            (current_age, current_level, lag_level, anchor_level)
        )
        prior_invalid = (
            np.any(prior_level < 0)
            or np.isinf(prior_level).any()
            or (
                year >= self.boundary_year + 2 * PERIOD_STEP
                and np.isnan(prior_level).any()
            )
        )
        if (
            not np.isfinite(required_numeric).all()
            or np.any(required_numeric[:, 1:] < 0)
            or prior_invalid
        ):
            raise ValueError(
                "forward earnings frame has invalid numeric state"
            )

        seed = seed_from_generator(rng)
        gate_rng = _substream(seed, "gate")
        donor_rng = _substream(seed, "donor-draw")
        reentry_rng = _substream(seed, "re-entry-draw")
        refresh_gate_rng = None
        refresh_rank_rng = None
        if self.rank_refresh_q is not None:
            refresh_gate_rng = _substream(seed, "memory-refresh-gate")
            refresh_rank_rng = _substream(seed, "memory-refresh-rank")

        anchor_rank = self._anchor_ranks(anchor_level, current_age, year)
        q0 = anchor_level == 0
        gate_uniform = gate_rng.random(len(sorted_frame))
        signs = np.empty(len(sorted_frame), dtype=np.int64)
        positive_anchor = ~q0
        if positive_anchor.any():
            signs[positive_anchor] = _gate_sign_draw(
                self.shared_gate,
                current_level[positive_anchor],
                current_age[positive_anchor],
                gate_uniform[positive_anchor],
            )
        if q0.any():
            gate = (
                self.zero_anchor_gate
                if self.zero_anchor_gate is not None
                else self.shared_gate
            )
            signs[q0] = _gate_sign_draw(
                gate,
                current_level[q0],
                current_age[q0],
                gate_uniform[q0],
            )

        generated = np.zeros(len(sorted_frame), dtype=np.float64)
        participating = np.flatnonzero(signs == 1)
        if len(participating):
            current_rank = np.full(len(sorted_frame), np.nan, dtype=np.float64)
            current_positive = lag_level > 0
            current_indices = np.flatnonzero(current_positive)
            if len(current_indices):
                current_rank[current_indices] = self.level_to_rank(
                    lag_level[current_indices],
                    current_age[current_indices] - PERIOD_STEP,
                    year - PERIOD_STEP,
                )

            prior_rank = np.full(len(sorted_frame), np.nan, dtype=np.float64)
            prior_positive = prior_level > 0
            prior_indices = np.flatnonzero(prior_positive)
            if len(prior_indices):
                prior_rank[prior_indices] = self.level_to_rank(
                    prior_level[prior_indices],
                    current_age[prior_indices] - 2 * PERIOD_STEP,
                    year - 2 * PERIOD_STEP,
                )

            output_rank = np.full(len(sorted_frame), np.nan, dtype=np.float64)
            reentry = participating[~current_positive[participating]]
            triple = participating[
                current_positive[participating]
                & prior_positive[participating]
                & (year >= self.boundary_year + 2 * PERIOD_STEP)
            ]
            pair = participating[
                current_positive[participating]
                & ~np.isin(participating, triple)
            ]

            # Candidate-10's frozen branch/subset draw order.
            self._draw_reentry(
                reentry, q0, anchor_rank, reentry_rng, output_rank
            )
            self._draw_transition(
                triple,
                q0,
                current_rank,
                prior_rank,
                anchor_rank,
                donor_rng,
                output_rank,
                triple=True,
            )
            self._draw_transition(
                pair,
                q0,
                current_rank,
                prior_rank,
                anchor_rank,
                donor_rng,
                output_rank,
                triple=False,
            )
            if self.rank_refresh_q is not None:
                assert refresh_gate_rng is not None
                assert refresh_rank_rng is not None
                continuers = participating[current_positive[participating]]
                self._draw_rank_refresh(
                    continuers,
                    q0,
                    anchor_rank,
                    current_age,
                    refresh_gate_rng,
                    refresh_rank_rng,
                    output_rank,
                )
            generated[participating] = self.rank_to_level(
                output_rank[participating], current_age[participating], year
            )
        return generated[inverse]


def validate_rank_refresh_fit(generator: ForwardEarningsGenerator) -> None:
    """Abort a selected refresh law when any exact donor bin is empty."""
    audit = generator.rank_refresh_fit_audit
    if audit is not None and not audit.eligible:
        raise RankRefreshPreflightAbort(audit)
