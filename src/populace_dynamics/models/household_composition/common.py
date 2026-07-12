"""Shared, frozen candidate-9 household-composition primitives.

Every numerical helper in this module is copied from
``household_composition_sim.py:82-131,219-224,388-462,584-609``.  Keeping
these operations in one module prevents component boundaries from changing
array layout, row order, or random-number consumption.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def weighted_rate(frame: pd.DataFrame, event: np.ndarray) -> float:
    """Return the weighted event rate.

    Copied from ``household_composition_sim.py:219-224``.
    """
    w = frame["weight"].to_numpy(dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    return float((w * event).sum() / total)


def restricted_cubic_basis(
    x: np.ndarray, knots: tuple[float, ...]
) -> np.ndarray:
    """Return the Harrell restricted-cubic-spline basis.

    Copied from ``household_composition_sim.py:112-131``.
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    n_knots = len(k)
    t1, tkm1, tk = k[0], k[-2], k[-1]
    denom = tk - tkm1
    scale = (tk - t1) ** 2
    cols = [x.copy()]
    for j in range(n_knots - 2):
        tj = k[j]
        term = (
            np.maximum(x - tj, 0.0) ** 3
            - np.maximum(x - tkm1, 0.0) ** 3 * (tk - tj) / denom
            + np.maximum(x - tk, 0.0) ** 3 * (tkm1 - tj) / denom
        ) / scale
        cols.append(term)
    return np.column_stack(cols)


def padded_person_matrices(side_a_pw: pd.DataFrame) -> dict[str, Any]:
    """Reshape ordered person-waves into ``[person, wave]`` matrices.

    Copied from ``household_composition_sim.py:388-418``.
    """
    pw = side_a_pw.sort_values(["person_id", "year"]).reset_index(drop=True)
    pids = pw["person_id"].to_numpy()
    order = np.arange(len(pw))
    first = np.concatenate([[True], pids[1:] != pids[:-1]])
    person_start = order[first]
    wave_idx = order - np.repeat(
        person_start, np.diff(np.concatenate([person_start, [len(pw)]]))
    )
    unique_pids = pids[first]
    n_persons = len(unique_pids)
    max_waves = int(wave_idx.max()) + 1
    row_of = np.full((n_persons, max_waves), -1, dtype=np.int64)
    person_of_row = np.repeat(
        np.arange(n_persons),
        np.diff(np.concatenate([person_start, [len(pw)]])),
    )
    row_of[person_of_row, wave_idx] = order
    return {
        "pw": pw,
        "unique_pids": unique_pids,
        "n_persons": n_persons,
        "max_waves": max_waves,
        "row_of": row_of,
        "person_of_row": person_of_row,
        "wave_idx": wave_idx,
    }


def evolve_absorbing_exit(
    valid: np.ndarray,
    initial: np.ndarray,
    exit_prob: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve an absorbing true-to-false occupancy.

    Copied from ``household_composition_sim.py:421-441``.
    """
    n_persons, max_waves = valid.shape
    state = np.zeros((n_persons, max_waves), dtype=bool)
    cur = initial[:, 0] & valid[:, 0]
    state[:, 0] = cur
    for w in range(1, max_waves):
        u = rng.random(n_persons)
        exited = cur & (u < exit_prob[:, w - 1])
        cur = cur & ~exited & valid[:, w]
        state[:, w] = cur
    return state


def evolve_two_state(
    valid: np.ndarray,
    initial: np.ndarray,
    entry_prob: np.ndarray,
    exit_prob: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve a two-state entry/exit occupancy.

    Copied from ``household_composition_sim.py:444-462``.
    """
    n_persons, max_waves = valid.shape
    state = np.zeros((n_persons, max_waves), dtype=bool)
    cur = initial[:, 0] & valid[:, 0]
    state[:, 0] = cur
    for w in range(1, max_waves):
        u = rng.random(n_persons)
        enters = (~cur) & (u < entry_prob[:, w - 1])
        exits = cur & (u < exit_prob[:, w - 1])
        cur = np.where(cur, cur & ~exits, enters) & valid[:, w]
        state[:, w] = cur
    return state


def coresident_child_counts(
    child_leaves: pd.DataFrame, side_a_pw: pd.DataFrame
) -> np.ndarray:
    """Count coresident children for each person-wave.

    Copied from ``household_composition_sim.py:584-609``.
    """
    counts = np.zeros(len(side_a_pw), dtype=np.int64)
    if not len(child_leaves):
        return counts
    pw = side_a_pw.reset_index(drop=True)
    pw = pw.assign(_row=np.arange(len(pw), dtype=np.int64))
    children = child_leaves.rename(columns={"parent_person_id": "person_id"})[
        ["person_id", "birth_year", "leave_year"]
    ]
    merged = pw[["person_id", "year", "_row"]].merge(
        children, on="person_id", how="inner"
    )
    hit = (merged["year"] >= merged["birth_year"]) & (
        merged["year"] < merged["leave_year"]
    )
    agg = merged.loc[hit].groupby("_row").size()
    counts[agg.index.to_numpy()] = agg.to_numpy()
    return counts


def weighted_share(
    weight: np.ndarray, hit: np.ndarray, subset: np.ndarray
) -> float:
    """Return a weighted share on ``subset``.

    Copied from ``household_composition_sim_v8.py:1014-1019``.
    """
    w = np.asarray(weight, dtype=np.float64)
    total = float(w[subset].sum())
    if total <= 0:
        return 0.0
    return float(w[hit & subset].sum() / total)
