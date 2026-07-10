"""Candidate-16 single-year fertility component, free of script imports.

Candidate 16 retains candidate 5's fertility component through the frozen
reuse chain at ``scripts/run_gate2_candidate16.py:214-238``.  This module ports
the triangular kernel and weighted fit from
``scripts/run_gate2_candidate5.py:197-205,310-418``, the parity helper from
``scripts/run_gate2_candidate1.py:617-634``, and the lookup from
``scripts/run_gate2_candidate5.py:421-439``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.data import births, transitions

__all__ = [
    "FERTILITY_AGE_HI",
    "FERTILITY_AGE_LO",
    "FERTILITY_KERNEL_BANDWIDTH",
    "build_fertility_lookup",
    "fertility_probabilities",
    "fit_fertility",
    "triangular_kernel_weights",
]

# Frozen range and bandwidth; source: scripts/run_gate2_candidate5.py:197-205.
FERTILITY_AGE_LO: int = min(lo for lo, _ in transitions.ASFR_AGE_BANDS)
FERTILITY_AGE_HI: int = max(hi for _, hi in transitions.ASFR_AGE_BANDS)
FERTILITY_KERNEL_BANDWIDTH: int = 3


def triangular_kernel_weights() -> dict[int, float]:
    """Return the frozen bandwidth-three triangular kernel.

    Preserves ``scripts/run_gate2_candidate5.py:312-326`` exactly: integer
    offsets ``-2..2`` receive weights ``1/3, 2/3, 1, 2/3, 1/3``.
    """
    h = FERTILITY_KERNEL_BANDWIDTH
    weights: dict[int, float] = {}
    for d in range(-(h - 1), h):
        w = 1.0 - abs(d) / h
        if w > 0.0:
            weights[d] = w
    return weights


def _parity(
    person_ids: np.ndarray,
    years: np.ndarray,
    births_by: dict[int, np.ndarray],
) -> np.ndarray:
    """Return prior-birth counts using candidate 1's exact grouping order.

    Ported from ``scripts/run_gate2_candidate1.py:617-634``.
    """
    out = np.zeros(len(person_ids), dtype=np.int64)
    order = np.argsort(person_ids, kind="stable")
    sp = person_ids[order]
    sy = years[order]
    starts = np.searchsorted(sp, np.unique(sp), side="left")
    bounds = np.append(starts, len(sp))
    for i in range(len(starts)):
        seg = slice(bounds[i], bounds[i + 1])
        pid = int(sp[bounds[i]])
        arr = births_by.get(pid)
        if arr is None:
            continue
        out[order[seg]] = np.searchsorted(arr, sy[seg], side="left")
    return out


def fit_fertility(
    panel: transitions.MaritalPanel,
    birth_records: pd.DataFrame,
    train_ids: set[int],
    birth_decade: pd.Series,
) -> dict[tuple[int, int, int], float]:
    """Fit single-year fertility within parity and cohort, smoothed over age.

    The weighted woman-year denominator, mother-weighted birth numerator,
    running parity, censoring, grouping, set iteration, nested loop, and
    accumulation order preserve
    ``scripts/run_gate2_candidate5.py:329-418`` exactly. Keys are
    ``(age, min(parity, 3), birth_decade)`` for ages 15 through 49.
    """
    py = panel.person_years
    attrs = panel.attrs
    women_ids = set(attrs[attrs["sex"] == "female"]["person_id"]) & train_ids
    lo, hi = FERTILITY_AGE_LO, FERTILITY_AGE_HI
    wy = py[
        py["person_id"].isin(women_ids) & (py["age"] >= lo) & (py["age"] <= hi)
    ][["person_id", "year", "age", "weight"]].copy()

    be = births.birth_events(birth_records)
    be = be[
        (be["record_type"] == "birth")
        & be["parent_person_id"].isin(women_ids)
        & be["birth_year"].notna()
    ].copy()
    be = be.rename(columns={"parent_person_id": "person_id"})
    be["birth_year"] = be["birth_year"].astype("int64")
    births_by = {
        int(p): np.sort(g["birth_year"].to_numpy())
        for p, g in be.groupby("person_id")
    }

    wy = wy.reset_index(drop=True)
    wy["parity"] = _parity(
        wy["person_id"].to_numpy(), wy["year"].to_numpy(), births_by
    )
    wy["decade"] = wy["person_id"].map(birth_decade).to_numpy()
    wy["parity_band"] = np.minimum(wy["parity"].to_numpy(), 3)

    attr_by = attrs.set_index("person_id")
    be["mother_birth"] = (
        be["person_id"].map(attr_by["birth_year"]).astype("float64")
    )
    be["mother_censor"] = (
        be["person_id"].map(attr_by["censor_year"]).astype("float64")
    )
    be["mother_age"] = be["birth_year"] - be["mother_birth"]
    be = be[
        (be["mother_age"] >= lo)
        & (be["mother_age"] <= hi)
        & (be["birth_year"] <= be["mother_censor"])
    ].reset_index(drop=True)
    be["decade"] = be["person_id"].map(birth_decade).to_numpy()
    be["weight"] = be["person_id"].map(attr_by["weight"]).to_numpy()
    be["parity"] = _parity(
        be["person_id"].to_numpy(), be["birth_year"].to_numpy(), births_by
    )
    be["parity_band"] = np.minimum(be["parity"].to_numpy(), 3)

    den = (
        wy.groupby(["age", "parity_band", "decade"])["weight"].sum().to_dict()
    )
    num = (
        be.groupby(["mother_age", "parity_band", "decade"])["weight"]
        .sum()
        .to_dict()
    )

    strata = {(int(pb), int(dec)) for (_a, pb, dec) in den}
    kernel = triangular_kernel_weights()
    table: dict[tuple[int, int, int], float] = {}
    for pb, dec in strata:
        for age in range(lo, hi + 1):
            num_s = 0.0
            den_s = 0.0
            for d, w in kernel.items():
                a = age + d
                num_s += w * float(num.get((a, pb, dec), 0.0))
                den_s += w * float(den.get((a, pb, dec), 0.0))
            if den_s > 0.0:
                table[(age, pb, dec)] = num_s / den_s
    return table


def build_fertility_lookup(
    table: dict[tuple[int, int, int], float],
) -> tuple[np.ndarray, dict[int, int]]:
    """Build candidate 16's dense fertility array and decade index.

    This ports the fertility portion of
    ``scripts/run_gate2_candidate16.py:786-792`` exactly.  The result is indexed
    ``[age - 15, parity_band, decade_index]``.
    """
    decades = sorted({d for (_a, _p, d) in table})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = FERTILITY_AGE_HI - FERTILITY_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1)), dtype=np.float64)
    for (age, pb, d), v in table.items():
        fert_arr[age - FERTILITY_AGE_LO, pb, decade_map[d]] = v
    return fert_arr, decade_map


def fertility_probabilities(
    age: np.ndarray,
    parity: np.ndarray,
    decade_index: np.ndarray,
    lookup: np.ndarray,
) -> np.ndarray:
    """Look up candidate-16 single-year fertility probabilities.

    The clipping, parity cap, safe missing-cohort index, and final mask preserve
    ``scripts/run_gate2_candidate5.py:421-439`` exactly.
    """
    ai = np.clip(age - FERTILITY_AGE_LO, 0, lookup.shape[0] - 1)
    pb = np.minimum(parity, 3)
    safe = np.where(decade_index >= 0, decade_index, 0)
    vals = lookup[ai, pb, safe]
    return np.where(decade_index >= 0, vals, 0.0)
