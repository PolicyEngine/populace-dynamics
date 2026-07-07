"""Held-out reproduction check for the SSA claim-age reference (#74, B2).

This builds ``runs/claiming_reference_v1.json`` -- a REPORTED, not
gated, artifact that states the standard a *future* claiming gate would
lock, and measures it now. The standard is out-of-sample Supplement-year
reproduction: fit the collapsed eight-category age-at-entitlement shares
on entitlement years **1998-2019** and predict the held-out years
**2020-2022**, recording the maximum absolute share deviation
(percentage points) under two rules:

* ``nearest_year`` -- predict each held-out year with the last in-sample
  year (2019). This is the module's documented default fallback for
  out-of-range requests (:func:`populace_dynamics.claiming._resolve_year`),
  so its held-out error is the honest cost of that default.
* ``linear_trend`` -- per (sex, category) ordinary-least-squares fit on
  the fit years, extrapolated to each held-out year.

Nothing here changes a gate; the artifact is pinned like the other
``runs/`` floors so the number is reproducible and test-bound.

Run from the repository root::

    .venv/bin/python scripts/build_claiming_reference.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics.claiming import load_claim_age_reference

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "runs" / "claiming_reference_v1.json"
REFERENCE_REL = "data/external/ssa_claim_ages_2023supplement.json"

SCHEMA_VERSION = "claiming_reference.v1"
RUN = "claiming_reference_v1"

FIT_YEARS = list(range(1998, 2020))  # 1998-2019 inclusive
HOLDOUT_YEARS = [2020, 2021, 2022]
SEXES = ("male", "female")

#: The stable eight-way partition (present in every year, unlike the raw
#: FRA sub-columns), evaluated as published percentages (0-100 scale).
CATEGORIES = (
    "age62",
    "age63",
    "age64",
    "age65",
    "age66",
    "disability_conversion",
    "age67_69",
    "age70plus",
)


def _series(ref, sex: str, category: str, years: list[int]) -> list[float]:
    return [ref.row(sex, y)["categories"][category] for y in years]


def _predict_nearest(fit_values: list[float], _year: int) -> float:
    """Nearest-year rule: last in-sample value, for any held-out year."""
    return fit_values[-1]


def _predict_trend(
    fit_years: list[int], fit_values: list[float], year: int
) -> float:
    """Linear-trend rule: OLS on the fit years, extrapolated."""
    slope, intercept = np.polyfit(fit_years, fit_values, 1)
    return float(intercept + slope * year)


def _evaluate(ref, rule: str) -> dict[str, Any]:
    per_cell: list[dict[str, Any]] = []
    for sex in SEXES:
        for category in CATEGORIES:
            fit_values = _series(ref, sex, category, FIT_YEARS)
            for year in HOLDOUT_YEARS:
                actual = ref.row(sex, year)["categories"][category]
                if rule == "nearest_year":
                    predicted = _predict_nearest(fit_values, year)
                elif rule == "linear_trend":
                    predicted = _predict_trend(FIT_YEARS, fit_values, year)
                else:
                    raise ValueError(rule)
                per_cell.append(
                    {
                        "sex": sex,
                        "year": year,
                        "category": category,
                        "predicted": round(predicted, 4),
                        "actual": actual,
                        "deviation": round(predicted - actual, 4),
                    }
                )
    abs_devs = [abs(c["deviation"]) for c in per_cell]
    argmax_cell = max(per_cell, key=lambda c: abs(c["deviation"]))
    return {
        "max_abs_deviation": round(max(abs_devs), 4),
        "mean_abs_deviation": round(float(np.mean(abs_devs)), 4),
        "rmse": round(float(np.sqrt(np.mean(np.square(abs_devs)))), 4),
        "argmax": argmax_cell,
        "n_cells": len(per_cell),
        "per_cell": per_cell,
    }


def build() -> dict[str, Any]:
    ref = load_claim_age_reference()
    nearest = _evaluate(ref, "nearest_year")
    trend = _evaluate(ref, "linear_trend")
    better = (
        "nearest_year"
        if nearest["max_abs_deviation"] <= trend["max_abs_deviation"]
        else "linear_trend"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "reported_not_gated": True,
        "purpose": (
            "Out-of-sample reproduction check for the SSA claim-age "
            "reference (task B2, #74). Changes no gate. States the "
            "standard a FUTURE claiming gate would lock -- fit 1998-2019, "
            "predict 2020-2022 collapsed-category shares -- and measures "
            "it now under the module's default nearest-year rule and a "
            "linear-trend alternative."
        ),
        "reference": {
            "artifact": REFERENCE_REL,
            "schema_version": ref.schema_version,
            "max_abs_residual": ref.validation["max_abs_residual"],
            "categories": list(CATEGORIES),
        },
        "protocol": {
            "fit_years": [FIT_YEARS[0], FIT_YEARS[-1]],
            "fit_years_inclusive": True,
            "holdout_years": HOLDOUT_YEARS,
            "sexes": list(SEXES),
            "categories": list(CATEGORIES),
            "metric": (
                "absolute deviation in published percentage points "
                "(share on a 0-100 scale)"
            ),
            "rules": {
                "nearest_year": (
                    "predict each held-out year with the last in-sample "
                    "year (2019); the module's documented default fallback"
                ),
                "linear_trend": (
                    "per (sex, category) OLS on the fit years, "
                    "extrapolated to each held-out year"
                ),
            },
            "default_rule": "nearest_year",
        },
        "results": {
            "nearest_year": nearest,
            "linear_trend": trend,
        },
        "headline": {
            "default_rule": "nearest_year",
            "nearest_year_max_abs_deviation": nearest["max_abs_deviation"],
            "linear_trend_max_abs_deviation": trend["max_abs_deviation"],
            "better_rule": better,
            "units": "percentage points of share (0-100 scale)",
        },
        "build": {
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "built_by": "scripts/build_claiming_reference.py",
        },
    }


def main() -> None:
    artifact = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    head = artifact["headline"]
    print(
        f"wrote {OUT_PATH}\n"
        f"  nearest-year max|dev| = "
        f"{head['nearest_year_max_abs_deviation']} pp "
        f"(at {artifact['results']['nearest_year']['argmax']['sex']} "
        f"{artifact['results']['nearest_year']['argmax']['year']} "
        f"{artifact['results']['nearest_year']['argmax']['category']})\n"
        f"  linear-trend max|dev| = "
        f"{head['linear_trend_max_abs_deviation']} pp "
        f"(at {artifact['results']['linear_trend']['argmax']['sex']} "
        f"{artifact['results']['linear_trend']['argmax']['year']} "
        f"{artifact['results']['linear_trend']['argmax']['category']})\n"
        f"  better rule = {head['better_rule']}"
    )


if __name__ == "__main__":
    main()
