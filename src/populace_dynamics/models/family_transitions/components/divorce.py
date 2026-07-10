"""Candidate-16 empirical divorce component, free of script imports.

Candidate 16 retains candidate 1's divorce fit and lookup through the frozen
reuse chain documented at ``scripts/run_gate2_candidate16.py:214-238``.  This
module ports the fit from ``scripts/run_gate2_candidate1.py:384-414``, its
marriage-order join from ``scripts/run_gate2_candidate1.py:485-499``, and the
probability lookup from ``scripts/run_gate2_candidate1.py:884-889``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.data import transitions

__all__ = [
    "DIVORCE_DURATION_BANDS",
    "DIVORCE_DURATION_LOWERS",
    "divorce_probabilities",
    "fit_divorce",
]

# Frozen bands; source: scripts/run_gate2_candidate1.py:332-338.
DIVORCE_DURATION_BANDS: tuple[tuple[int, int], ...] = (
    transitions.DIVORCE_DURATION_BANDS
)
DIVORCE_DURATION_LOWERS: np.ndarray = np.array(
    [lo for lo, _ in DIVORCE_DURATION_BANDS], dtype=np.int64
)


def _band_indices(
    values: np.ndarray, lowers: np.ndarray, n_bands: int
) -> np.ndarray:
    """Return clipped band indices from candidate 1 lines 345-347.

    The exact source is ``scripts/run_gate2_candidate1.py:345-347``.
    """
    return np.clip(
        np.searchsorted(lowers, values, side="right") - 1,
        0,
        n_bands - 1,
    )


def _attach_order(
    frame: pd.DataFrame, order_map: pd.DataFrame, *, dur_col: str
) -> pd.DataFrame:
    """Attach marriage order using candidate 1's exact start-year join.

    Ported from ``scripts/run_gate2_candidate1.py:485-499``.
    """
    frame = frame.copy()
    frame["current_start"] = (
        frame["year"].to_numpy() - frame[dur_col].astype("int64").to_numpy()
    )
    merged = frame.merge(
        order_map.rename(columns={"start_year": "current_start"}),
        on=["person_id", "current_start"],
        how="left",
    )
    merged["order"] = merged["order"].fillna(1).astype("int64")
    return merged


def fit_divorce(
    train_py: pd.DataFrame,
    train_events: pd.DataFrame,
    order_map: pd.DataFrame,
) -> np.ndarray:
    """Fit the weighted duration-by-order divorce table with add-one smoothing.

    Inputs are the already train-restricted person-year and event frames.  The
    selection, grouping, mean-weight prior, nested loop, and arithmetic preserve
    ``scripts/run_gate2_candidate1.py:384-414`` exactly.  The returned array is
    indexed ``[duration_band, order >= 2]``.
    """
    married = train_py[train_py["marital_state"] == "married"].copy()
    married = _attach_order(married, order_map, dur_col="marriage_duration")
    div_ev = train_events[
        (train_events["transition"] == "divorce")
        & train_events["marriage_duration"].notna()
    ].copy()
    div_ev = _attach_order(div_ev, order_map, dur_col="marriage_duration")
    wbar_married = float(married["weight"].mean())
    married["dur_band"] = _band_indices(
        married["marriage_duration"].astype("int64").to_numpy(),
        DIVORCE_DURATION_LOWERS,
        len(DIVORCE_DURATION_BANDS),
    )
    married["ord_bit"] = (married["order"] >= 2).astype(int)
    div_ev["dur_band"] = _band_indices(
        div_ev["marriage_duration"].astype("int64").to_numpy(),
        DIVORCE_DURATION_LOWERS,
        len(DIVORCE_DURATION_BANDS),
    )
    div_ev["ord_bit"] = (div_ev["order"] >= 2).astype(int)
    div_den = married.groupby(["dur_band", "ord_bit"])["weight"].sum()
    div_num = div_ev.groupby(["dur_band", "ord_bit"])["weight"].sum()
    div_table = np.zeros((len(DIVORCE_DURATION_BANDS), 2), dtype=np.float64)
    for b in range(len(DIVORCE_DURATION_BANDS)):
        for o in (0, 1):
            wnum = float(div_num.get((b, o), 0.0))
            wden = float(div_den.get((b, o), 0.0))
            div_table[b, o] = (wnum + wbar_married) / (
                wden + 2.0 * wbar_married
            )
    return div_table


def divorce_probabilities(
    duration: np.ndarray, order: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Look up candidate-16 divorce probabilities.

    This is the operation-for-operation port of
    ``scripts/run_gate2_candidate1.py:884-889`` retained by candidate 16.
    """
    bands = _band_indices(
        duration, DIVORCE_DURATION_LOWERS, len(DIVORCE_DURATION_BANDS)
    )
    ocol = (order >= 2).astype(np.int64)
    return table[bands, ocol]
