"""The M4 disability-hazard candidate: a forward Markov simulator over the
self-reported work-limitation panel (roadmap #113, M4; gate ``gate_m4``).

This is the CANDIDATE the locked ``gate_m4`` contract scores -- "the M4
module's hazard machinery" of
:mod:`populace_dynamics.data.disability`, refit per split seed on the TRAIN
half and simulated on the HOLDOUT support. It carries no free
hyperparameter: the fitted quantities ARE the module's own weighted
band x sex reference rates on the train half (side B), so the internal
gate cells are genuine candidate-vs-PSID *reproduction* cells (the floor
:mod:`scripts.build_m4_gate_floors` priced the real-vs-real half-split
null of the same statistic).

The generative machinery
------------------------
Fit (train half, side B), all via the unmodified
:mod:`populace_dynamics.data.disability` estimators so the fitted numbers
are byte-identical to that module's reference moments on side B:

* ``incidence[band, sex]`` -- the not-disabled -> disabled per-interval
  hazard (:func:`disability.incidence_cells`);
* ``recovery[band, sex]``  -- the disabled -> not-disabled *exit* hazard
  (:func:`disability.recovery_cells`; recovery counts every exit from
  disabled, retirement included);
* ``prevalence0[band, sex]`` -- the disabled-occupancy stock
  (:func:`disability.prevalence_cells`), the fitted initial-state
  distribution at each episode start;
* ``exit_retirement_share[sex]`` -- among non-death disability exits in
  the FRA-crossing window, the weighted share to ``retired``
  (:func:`build_m4_gate_floors.conversion_exit_shares`), the split that
  turns a simulated recovery into a retirement conversion vs a
  return-to-work near FRA.

Simulate (holdout half, side A), preserving the observed *support* -- the
same persons, waves, ages, sexes and START-WAVE weights -- and
re-drawing only the disability/retirement STATE, per person over their
grid, from a start-wave-weighted forward chain:

* at each episode start (a person's first observed wave, and the first
  wave after any gap longer than :data:`disability.MAX_INTERVAL`) the
  state is drawn ``disabled ~ Bernoulli(prevalence0[band, sex])`` -- the
  fitted stock, re-seeded because the gap breaks the transition grid
  exactly as it does in the real panel (no pair forms across it);
* across each grid-adjacent step the state advances by the fitted
  hazards: a not-disabled person onsets with ``incidence``; a disabled
  person exits with ``recovery`` and, if it exits inside the FRA-crossing
  window, converts to ``retired`` with ``exit_retirement_share`` (else
  returns to work); ``retired`` is absorbing (the automatic FRA
  conversion, :mod:`populace_dynamics.disability_conversion`).

The simulated states are handed straight back to
:func:`disability.build_transition_pairs`, so a simulated
:class:`disability.DisabilityPanel` is scored by the SAME
:func:`disability.reference_moments` /
:func:`build_m4_gate_floors.prevalence_shares` /
:func:`build_m4_gate_floors.conversion_exit_shares` code the floor used
on the real halves -- the candidate never re-implements a rate.

Determinism
-----------
:func:`simulate_draw` is a pure function of ``(panel, model, holdout_ids,
draw_seed)``: it sorts the holdout person-years by ``(person_id,
period)`` and consumes ``numpy.random.default_rng(draw_seed)`` in a fixed
wave-position order, so two calls at the same draw seed reproduce
byte-identically (pinned by the candidate reproduction test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from populace_dynamics.data import disability

__all__ = [
    "DisabilityHazardModel",
    "CONVERSION_EXIT_WINDOW",
    "fit",
    "simulate_draw",
]

#: The FRA-crossing window in which a disabled -> not-disabled exit may be
#: a retirement conversion (the module's own conversion window, ages
#: 60-67 inclusive). Outside it every exit is a return-to-work.
CONVERSION_EXIT_WINDOW: tuple[int, int] = disability.CONVERSION_WINDOW

#: Number of :data:`disability.DI_AGE_BANDS`; lookup row 0 is the
#: "no band" sentinel (ages outside 20-66) carrying a zero hazard.
_N_BANDS = len(disability.DI_AGE_BANDS)
_SEX_INDEX = {sex: i for i, sex in enumerate(disability.SEXES)}


@dataclass(frozen=True)
class DisabilityHazardModel:
    """Fitted band x sex hazards + the initial-state stock and FRA split.

    All arrays are shape ``(1 + N_BANDS, len(SEXES))``: row 0 is the
    "no band" sentinel (a zero hazard / zero stock for ages outside the
    :data:`disability.DI_AGE_BANDS` 20-66 range), rows ``1 + j`` carry
    band ``DI_AGE_BANDS[j]``. Sex column order is
    :data:`disability.SEXES` (``female``, ``male``).

    Attributes:
        incidence: not-disabled -> disabled per-interval hazard.
        recovery: disabled -> (not-disabled OR retired) exit hazard.
        prevalence0: fitted disabled-occupancy stock, the episode-start
            initial-state distribution.
        exit_retirement_share: per sex, ``P(retired | disability exit)``
            inside :data:`CONVERSION_EXIT_WINDOW`.
        n_train_persons: the side-B person count the fit used.
        train_moments_note: provenance string.
    """

    incidence: np.ndarray
    recovery: np.ndarray
    prevalence0: np.ndarray
    exit_retirement_share: np.ndarray
    n_train_persons: int
    train_moments_note: str

    @property
    def meta(self) -> dict[str, object]:
        """A JSON-safe fit summary for the run artifact (report-only)."""
        bands = ["none", *[f"{lo}-{hi}" for lo, hi in disability.DI_AGE_BANDS]]
        return {
            "model": "disability_hazard_sim (M4 candidate 1)",
            "estimator": (
                "unmodified populace_dynamics.data.disability weighted "
                "band x sex reference rates on the train half (side B); "
                "start-wave weights; no free hyperparameter"
            ),
            "n_train_persons": int(self.n_train_persons),
            "bands": bands,
            "sexes": list(disability.SEXES),
            "incidence_by_band_sex": self._grid(self.incidence),
            "recovery_by_band_sex": self._grid(self.recovery),
            "prevalence0_by_band_sex": self._grid(self.prevalence0),
            "exit_retirement_share_by_sex": {
                sex: float(self.exit_retirement_share[_SEX_INDEX[sex]])
                for sex in disability.SEXES
            },
        }

    @staticmethod
    def _grid(arr: np.ndarray) -> dict[str, dict[str, float]]:
        bands = ["none", *[f"{lo}-{hi}" for lo, hi in disability.DI_AGE_BANDS]]
        return {
            band: {
                sex: float(arr[bi, _SEX_INDEX[sex]])
                for sex in disability.SEXES
            }
            for bi, band in enumerate(bands)
        }


def _band_index_array(ages: np.ndarray) -> np.ndarray:
    """Vectorised :func:`disability._band_of`: ``1 + j`` for band ``j``,
    else 0 (the "no band" sentinel)."""
    out = np.zeros(len(ages), dtype=np.int64)
    for j, (lo, hi) in enumerate(disability.DI_AGE_BANDS):
        out[(ages >= lo) & (ages <= hi)] = j + 1
    return out


def _fit_grid(cells: dict[str, dict[str, float]], prefix: str) -> np.ndarray:
    """Pack a ``{prefix}.{band}|{sex}`` rate map into a
    ``(1 + N_BANDS, len(SEXES))`` grid (row 0 the zero sentinel)."""
    grid = np.zeros((1 + _N_BANDS, len(disability.SEXES)), dtype=np.float64)
    for j, (lo, hi) in enumerate(disability.DI_AGE_BANDS):
        band = disability.band_label(lo, hi)
        for sex, si in _SEX_INDEX.items():
            grid[j + 1, si] = cells[f"{prefix}.{band}|{sex}"]["rate"]
    return grid


def fit(
    panel: disability.DisabilityPanel, train_ids: set[int]
) -> DisabilityHazardModel:
    """Fit the hazard machinery on the train half (side B).

    Every fitted number is the unmodified
    :mod:`populace_dynamics.data.disability` weighted reference rate on
    ``train_ids``, so a "train copy" scores at the noise floor and the
    fit introduces no statistic the floor did not price.
    """
    incidence = _fit_grid(
        disability.incidence_cells(panel, train_ids), "incidence"
    )
    recovery = _fit_grid(
        disability.recovery_cells(panel, train_ids), "recovery"
    )
    prevalence0 = _fit_grid(
        disability.prevalence_cells(panel, train_ids), "prevalence"
    )

    # Retirement share among non-death disability exits in the FRA window
    # (the same statistic build_m4_gate_floors.conversion_exit_shares
    # measures; imported lazily to avoid a scripts <-> src import cycle).
    from build_m4_gate_floors import conversion_exit_shares

    ce = conversion_exit_shares(panel, train_ids)
    exit_share = np.zeros(len(disability.SEXES), dtype=np.float64)
    for sex, si in _SEX_INDEX.items():
        exit_share[si] = ce[sex]["share"]

    return DisabilityHazardModel(
        incidence=incidence,
        recovery=recovery,
        prevalence0=prevalence0,
        exit_retirement_share=exit_share,
        n_train_persons=len(train_ids),
        train_moments_note=(
            "weighted band x sex reference rates on side B "
            "(populace_dynamics.data.disability, start-wave weights)"
        ),
    )


def simulate_draw(
    panel: disability.DisabilityPanel,
    model: DisabilityHazardModel,
    holdout_ids: set[int],
    draw_seed: int,
) -> disability.DisabilityPanel:
    """Simulate one draw of the holdout half's disability trajectories.

    Preserves the holdout *support* (persons, waves, ages, sexes,
    start-wave weights) and re-draws only the disability/retirement
    state, per person over their grid, from the fitted forward chain.
    Returns a simulated :class:`disability.DisabilityPanel` whose
    ``person_years`` carry simulated ``disabled`` / ``retired`` flags and
    whose ``pairs`` are rebuilt by the unmodified
    :func:`disability.build_transition_pairs`, so the candidate is scored
    by the very rate code the floor used.

    Deterministic in ``(panel, model, holdout_ids, draw_seed)``.
    """
    py = panel.person_years
    sub = py[py["person_id"].isin(holdout_ids)]
    sub = sub.sort_values(["person_id", "period"], kind="stable")

    pid = sub["person_id"].to_numpy()
    period = sub["period"].to_numpy(dtype=np.int64)
    age = sub["age"].to_numpy(dtype=np.int64)
    weight = sub["weight"].to_numpy(dtype=np.float64)
    sex = sub["sex"].to_numpy()
    n = len(pid)

    sex_idx = np.zeros(n, dtype=np.int64)
    for s, si in _SEX_INDEX.items():
        sex_idx[sex == s] = si

    band_here = _band_index_array(age)

    # Episode structure: first-in-person and post-gap waves re-initialise.
    first_in_person = np.ones(n, dtype=bool)
    if n > 1:
        first_in_person[1:] = pid[1:] != pid[:-1]
    interval = np.full(n, disability.MAX_INTERVAL + 1_000_000, dtype=np.int64)
    if n > 1:
        interval[1:] = period[1:] - period[:-1]
    interval[first_in_person] = disability.MAX_INTERVAL + 1_000_000
    is_init = first_in_person | (interval > disability.MAX_INTERVAL)

    # Wave position within person (contiguous per person after the sort).
    group_id = np.cumsum(first_in_person) - 1
    group_start = np.flatnonzero(first_in_person)
    wave_pos = np.arange(n) - group_start[group_id]

    lo_win, hi_win = CONVERSION_EXIT_WINDOW
    rng = np.random.default_rng(draw_seed)
    disabled = np.zeros(n, dtype=bool)
    retired = np.zeros(n, dtype=bool)

    max_pos = int(wave_pos.max()) if n else -1
    for t in range(max_pos + 1):
        rows = np.flatnonzero(wave_pos == t)
        if rows.size == 0:
            continue
        m = rows.size
        u_state = rng.random(m)  # onset / init / exit
        u_split = rng.random(m)  # return-to-work vs retirement at FRA

        prev = np.maximum(rows - 1, 0)
        prev_disabled = disabled[prev]
        prev_retired = retired[prev]
        b_here = band_here[rows]
        b_prev = band_here[prev]
        sx = sex_idx[rows]
        age_prev = age[prev]

        init_m = is_init[rows]
        trans_m = ~init_m
        from_nd = trans_m & (~prev_disabled) & (~prev_retired)
        from_d = trans_m & prev_disabled
        from_r = trans_m & prev_retired

        new_d = np.zeros(m, dtype=bool)
        new_r = np.zeros(m, dtype=bool)

        # Episode start: draw the fitted disabled-occupancy stock.
        p0 = model.prevalence0[b_here, sx]
        new_d |= init_m & (u_state < p0)

        # Not-disabled -> disabled onset (incidence).
        p_inc = model.incidence[b_prev, sx]
        new_d |= from_nd & (u_state < p_inc)

        # Disabled -> exit (recovery); split retirement vs return-to-work.
        p_rec = model.recovery[b_prev, sx]
        exit_now = from_d & (u_state < p_rec)
        stay_d = from_d & ~exit_now
        new_d |= stay_d
        in_window = (age_prev >= lo_win) & (age_prev <= hi_win)
        ret_share = model.exit_retirement_share[sx]
        to_retired = exit_now & in_window & (u_split < ret_share)
        new_r |= to_retired
        # exits not to retirement return to work (not-disabled): nothing set.

        # Retired is absorbing.
        new_r |= from_r

        disabled[rows] = new_d
        retired[rows] = new_r

    sim = pd.DataFrame(
        {
            "person_id": pid,
            "period": period,
            "age": age,
            "sex": pd.array(sex, dtype="string"),
            "weight": weight,
            "disabled": disabled,
            "retired": retired,
        }
    )
    # status_code is not scored; carry a faithful code so the frame reads
    # like read_disability_status output (5 disabled, 4 retired, else 1).
    status_code = np.ones(n, dtype=np.int64)
    status_code[retired] = disability.RETIRED_CODE
    status_code[disabled] = disability.DISABLED_CODE
    sim["status_code"] = status_code

    pairs = disability.build_transition_pairs(sim)
    person_years = sim[
        [
            "person_id",
            "period",
            "age",
            "sex",
            "weight",
            "status_code",
            "disabled",
            "retired",
        ]
    ].reset_index(drop=True)
    return disability.DisabilityPanel(person_years=person_years, pairs=pairs)
