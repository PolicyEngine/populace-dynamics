"""The held-out panel-moment battery (paper sec. 5, surface 5).

Structure-sensitive moments computed on person-period panels under
weights: earnings-mobility matrices, weighted moments and
autocorrelation of (log) earnings changes, age-earnings profiles,
zero-earnings spell structure, and discrete-state transition rates.
Every function returns a tidy ``pandas.DataFrame`` so candidate and
holdout batteries align on keys, and :func:`moment_distance` reduces an
aligned pair to one number for gate scoring.

These moments complement the geometry blocks (energy distance, PRDC,
C2ST) that :mod:`populace_dynamics.harness.panel` runs on trajectory
windows: geometry under a subsample cap is weak exactly where
structure lives, so the battery scores the named quantities directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "weighted_moments",
    "mobility_matrix",
    "change_moments",
    "autocorrelation",
    "age_profile",
    "zero_spells",
    "transition_rates",
    "moment_distance",
]


def _validate(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing column(s) {missing}.")


def _pair(
    df: pd.DataFrame,
    id_col: str,
    period_col: str,
    columns: list[str],
    horizon: int,
    period_step: int,
) -> pd.DataFrame:
    """Self-merge a panel with itself ``horizon`` steps ahead.

    Returns rows carrying ``<col>`` at t and ``<col>_next`` at
    t + horizon * period_step for every person-period where both
    observations exist. Gaps in observation break pairs naturally,
    which is how entry and exit are handled.
    """
    base = df[[id_col, period_col, *columns]].copy()
    ahead = base.copy()
    ahead[period_col] = ahead[period_col] - horizon * period_step
    paired = base.merge(
        ahead,
        on=[id_col, period_col],
        suffixes=("", "_next"),
        how="inner",
    )
    return paired


def weighted_moments(x: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Weighted mean, sd, skewness, and excess kurtosis of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must have positive total.")
    mean = float(np.average(x, weights=w))
    centered = x - mean
    var = float(np.average(centered**2, weights=w))
    sd = float(np.sqrt(var))
    if sd == 0:
        return {"mean": mean, "sd": 0.0, "skew": 0.0, "kurtosis": 0.0}
    skew = float(np.average(centered**3, weights=w) / sd**3)
    kurt = float(np.average(centered**4, weights=w) / sd**4 - 3.0)
    return {"mean": mean, "sd": sd, "skew": skew, "kurtosis": kurt}


def _weighted_quantile_bin(
    x: np.ndarray, w: np.ndarray, n_bins: int
) -> np.ndarray:
    """Assign weighted-quantile bin labels ``1..n_bins`` to ``x``."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    cum = np.cumsum(w[order])
    ranks[order] = (cum - 0.5 * w[order]) / cum[-1]
    bins = np.minimum((ranks * n_bins).astype(np.int64) + 1, n_bins)
    return bins


def mobility_matrix(
    df: pd.DataFrame,
    *,
    id_col: str,
    period_col: str,
    value_col: str,
    weight_col: str,
    horizon: int = 1,
    period_step: int = 1,
    n_bins: int = 5,
    zero_bin: bool = True,
) -> pd.DataFrame:
    """Weighted origin-to-destination transition probabilities.

    Origin and destination positions are weighted-quantile bins of the
    positive values within each period; exact zeros occupy a dedicated
    bin ``0`` when ``zero_bin`` is set. Returns tidy rows
    ``{origin, destination, probability, weight}`` with probabilities
    row-normalized over destinations.
    """
    _validate(df, [id_col, period_col, value_col, weight_col])
    work = df[[id_col, period_col, value_col, weight_col]].copy()

    def assign(group: pd.DataFrame) -> pd.Series:
        values = group[value_col].to_numpy(dtype=np.float64)
        weights = group[weight_col].to_numpy(dtype=np.float64)
        out = np.zeros(len(group), dtype=np.int64)
        positive = values > 0 if zero_bin else np.ones(len(group), bool)
        if positive.sum() > 0:
            out[positive] = _weighted_quantile_bin(
                values[positive], weights[positive], n_bins
            )
        return pd.Series(out, index=group.index)

    work["bin"] = (
        work.groupby(period_col, group_keys=False)
        .apply(assign, include_groups=False)
        .astype(np.int64)
    )
    paired = _pair(
        work, id_col, period_col, ["bin", weight_col], horizon, period_step
    )
    counts = (
        paired.groupby(["bin", "bin_next"])[weight_col]
        .sum()
        .rename("weight")
        .reset_index()
    )
    totals = counts.groupby("bin")["weight"].transform("sum")
    counts["probability"] = counts["weight"] / totals
    return counts.rename(columns={"bin": "origin", "bin_next": "destination"})[
        ["origin", "destination", "probability", "weight"]
    ]


def change_moments(
    df: pd.DataFrame,
    *,
    id_col: str,
    period_col: str,
    value_col: str,
    weight_col: str,
    by: list[str] | None = None,
    horizon: int = 1,
    period_step: int = 1,
    log: bool = True,
) -> pd.DataFrame:
    """Weighted moments of (log) value changes at ``horizon``.

    With ``log`` set, rows where either endpoint is non-positive drop
    (zero transitions belong to :func:`zero_spells` and the mobility
    matrix's zero bin, not to log changes). Returns one tidy row per
    group with mean, sd, skew, and excess kurtosis of the change.
    """
    by = list(by or [])
    _validate(df, [id_col, period_col, value_col, weight_col, *by])
    paired = _pair(
        df,
        id_col,
        period_col,
        [value_col, weight_col, *by],
        horizon,
        period_step,
    )
    if log:
        keep = (paired[value_col] > 0) & (paired[f"{value_col}_next"] > 0)
        paired = paired.loc[keep]
        change = np.log(paired[f"{value_col}_next"]) - np.log(
            paired[value_col]
        )
    else:
        change = paired[f"{value_col}_next"] - paired[value_col]
    paired = paired.assign(_change=change)
    rows: list[dict] = []
    groups = paired.groupby(by) if by else [((), paired)]
    for key, group in groups:
        stats = weighted_moments(
            group["_change"].to_numpy(),
            group[weight_col].to_numpy(),
        )
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)), **stats})
    long = pd.DataFrame(rows).melt(
        id_vars=by, var_name="moment", value_name="value"
    )
    return long.sort_values([*by, "moment"]).reset_index(drop=True)


def autocorrelation(
    df: pd.DataFrame,
    *,
    id_col: str,
    period_col: str,
    value_col: str,
    weight_col: str,
    lags: tuple[int, ...] = (1, 2, 5),
    period_step: int = 1,
    log: bool = True,
) -> pd.DataFrame:
    """Weighted autocorrelation of (log) values at each lag."""
    _validate(df, [id_col, period_col, value_col, weight_col])
    rows = []
    for lag in lags:
        paired = _pair(
            df,
            id_col,
            period_col,
            [value_col, weight_col],
            lag,
            period_step,
        )
        if log:
            keep = (paired[value_col] > 0) & (paired[f"{value_col}_next"] > 0)
            paired = paired.loc[keep]
            x = np.log(paired[value_col].to_numpy(dtype=np.float64))
            y = np.log(paired[f"{value_col}_next"].to_numpy(dtype=np.float64))
        else:
            x = paired[value_col].to_numpy(dtype=np.float64)
            y = paired[f"{value_col}_next"].to_numpy(dtype=np.float64)
        w = paired[weight_col].to_numpy(dtype=np.float64)
        if len(x) < 2:
            rows.append({"lag": lag, "value": np.nan})
            continue
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.average((x - mx) * (y - my), weights=w)
        sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
        sy = np.sqrt(np.average((y - my) ** 2, weights=w))
        rows.append({"lag": lag, "value": float(cov / (sx * sy))})
    return pd.DataFrame(rows)


def age_profile(
    df: pd.DataFrame,
    *,
    age_col: str,
    value_col: str,
    weight_col: str,
    by: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Weighted mean and quantiles of the value by age (and groups)."""
    by = list(by or [])
    _validate(df, [age_col, value_col, weight_col, *by])
    rows = []
    for key, group in df.groupby([age_col, *by]):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value_col].to_numpy(dtype=np.float64)
        weights = group[weight_col].to_numpy(dtype=np.float64)
        order = np.argsort(values, kind="stable")
        cum = np.cumsum(weights[order])
        grid = (cum - 0.5 * weights[order]) / cum[-1]
        row = dict(zip([age_col, *by], key, strict=True))
        row["mean"] = float(np.average(values, weights=weights))
        for q in quantiles:
            row[f"q{int(q * 100)}"] = float(np.interp(q, grid, values[order]))
        rows.append(row)
    wide = pd.DataFrame(rows)
    return (
        wide.melt(
            id_vars=[age_col, *by], var_name="statistic", value_name="value"
        )
        .sort_values([age_col, *by, "statistic"])
        .reset_index(drop=True)
    )


def zero_spells(
    df: pd.DataFrame,
    *,
    id_col: str,
    period_col: str,
    value_col: str,
    weight_col: str,
    period_step: int = 1,
) -> pd.DataFrame:
    """Weighted zero-spell structure: entry, exit, and spell lengths.

    Entry is P(y_{t+1} = 0 | y_t > 0); exit is P(y_{t+1} > 0 |
    y_t = 0), both over observed consecutive pairs. Spell lengths
    count maximal runs of consecutive zero observations per person,
    weighted by the person's weight at the spell's first period.
    """
    _validate(df, [id_col, period_col, value_col, weight_col])
    paired = _pair(
        df,
        id_col,
        period_col,
        [value_col, weight_col],
        1,
        period_step,
    )
    zero_now = paired[value_col] == 0
    zero_next = paired[f"{value_col}_next"] == 0
    w = paired[weight_col]
    rows = [
        {
            "statistic": "entry_rate",
            "value": (
                float(w[~zero_now & zero_next].sum() / w[~zero_now].sum())
                if w[~zero_now].sum() > 0
                else np.nan
            ),
        },
        {
            "statistic": "exit_rate",
            "value": (
                float(w[zero_now & ~zero_next].sum() / w[zero_now].sum())
                if w[zero_now].sum() > 0
                else np.nan
            ),
        },
    ]
    work = df.sort_values([id_col, period_col])
    zeros = work[work[value_col] == 0]
    if len(zeros):
        gap = zeros.groupby(id_col)[period_col].diff() != period_step
        spell_id = gap.cumsum()
        lengths = zeros.groupby([id_col, spell_id]).agg(
            length=(period_col, "size"),
            weight=(weight_col, "first"),
        )
        mean_length = float(
            np.average(lengths["length"], weights=lengths["weight"])
        )
    else:
        mean_length = np.nan
    rows.append({"statistic": "mean_spell_length", "value": mean_length})
    return pd.DataFrame(rows)


def transition_rates(
    df: pd.DataFrame,
    *,
    id_col: str,
    period_col: str,
    state_col: str,
    weight_col: str,
    horizon: int = 1,
    period_step: int = 1,
) -> pd.DataFrame:
    """Weighted discrete-state transition probabilities.

    Serves the family-transition moments: pass marital or household
    status as ``state_col``. Returns tidy rows ``{origin, destination,
    probability, weight}`` row-normalized over destinations.
    """
    _validate(df, [id_col, period_col, state_col, weight_col])
    paired = _pair(
        df,
        id_col,
        period_col,
        [state_col, weight_col],
        horizon,
        period_step,
    )
    counts = (
        paired.groupby([state_col, f"{state_col}_next"])[weight_col]
        .sum()
        .rename("weight")
        .reset_index()
    )
    totals = counts.groupby(state_col)["weight"].transform("sum")
    counts["probability"] = counts["weight"] / totals
    return counts.rename(
        columns={state_col: "origin", f"{state_col}_next": "destination"}
    )[["origin", "destination", "probability", "weight"]]


def moment_distance(
    candidate: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    value_col: str = "value",
    weight_col: str | None = None,
) -> float:
    """Reduce an aligned candidate/holdout battery pair to one number.

    Aligns on every shared non-value column and returns the (optionally
    weighted) mean absolute difference of ``value_col``. Rows present
    on one side only raise, so a candidate cannot win by omission.
    """
    keys = [
        c
        for c in candidate.columns
        if c in holdout.columns and c not in (value_col, weight_col)
    ]
    merged = candidate.merge(
        holdout,
        on=keys,
        suffixes=("_cand", "_hold"),
        how="outer",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", keys]
        raise ValueError(
            "Candidate and holdout batteries do not align on "
            f"{len(missing)} row(s); first: "
            f"{missing.iloc[0].to_dict()}."
        )
    diff = (merged[f"{value_col}_cand"] - merged[f"{value_col}_hold"]).abs()
    if weight_col is not None:
        return float(np.average(diff, weights=merged[f"{weight_col}_hold"]))
    return float(diff.mean())
