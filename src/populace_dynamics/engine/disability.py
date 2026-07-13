"""M6 adapter for the certified M4 reproduction-mode disability simulator.

The M4 public simulator accepts an integer seed and constructs its own random
generator.  M6 needs a generator supplied by its draw/module/period stream
registry, while the fitted M4 parameters and frozen model implementation remain
untouched.  :func:`simulate_reproduction` is therefore an engine-side copy of
that state-transition path with the sole interface change ``draw_seed -> rng``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from populace_dynamics.data import disability
from populace_dynamics.models.disability_hazard_sim import (
    DisabilityHazardModel,
)

from .support import StartWaveWeightSnapshot

__all__ = ["simulate_reproduction"]

_SEX_INDEX = {sex: index for index, sex in enumerate(disability.SEXES)}


def _band_index_array(ages: np.ndarray) -> np.ndarray:
    """Return the M4 lookup row for each age (zero is the no-band row)."""
    out = np.zeros(len(ages), dtype=np.int64)
    for index, (lower, upper) in enumerate(disability.DI_AGE_BANDS):
        out[(ages >= lower) & (ages <= upper)] = index + 1
    return out


def simulate_reproduction(
    panel: disability.DisabilityPanel,
    model: DisabilityHazardModel,
    holdout_ids: set[int],
    rng: np.random.Generator | None,
    *,
    start_weights: StartWaveWeightSnapshot,
    rng_by_period: Mapping[int, np.random.Generator] | None = None,
) -> disability.DisabilityPanel:
    """Redraw M4 states on realized support using an injected generator.

    Persons, periods, ages, sexes, and start-wave weights are preserved.  Only
    disability and retirement state are redrawn.  Random-number consumption and
    output assembly match ``disability_hazard_sim.simulate_draw`` exactly when
    ``rng`` is ``default_rng(draw_seed)``.
    """
    if (rng is None) == (rng_by_period is None):
        raise ValueError("supply exactly one of rng or rng_by_period")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    person_years = panel.person_years
    sub = person_years[person_years["person_id"].isin(holdout_ids)]
    sub = start_weights.apply(sub)
    sub = sub.sort_values(["person_id", "period"], kind="stable")

    person_id = sub["person_id"].to_numpy()
    period = sub["period"].to_numpy(dtype=np.int64)
    age = sub["age"].to_numpy(dtype=np.int64)
    weight = sub["weight"].to_numpy(dtype=np.float64)
    sex = sub["sex"].to_numpy()
    n_rows = len(person_id)

    sex_index = np.zeros(n_rows, dtype=np.int64)
    for label, index in _SEX_INDEX.items():
        sex_index[sex == label] = index
    band_here = _band_index_array(age)

    first_in_person = np.ones(n_rows, dtype=bool)
    if n_rows > 1:
        first_in_person[1:] = person_id[1:] != person_id[:-1]
    interval = np.full(
        n_rows,
        disability.MAX_INTERVAL + 1_000_000,
        dtype=np.int64,
    )
    if n_rows > 1:
        interval[1:] = period[1:] - period[:-1]
    interval[first_in_person] = disability.MAX_INTERVAL + 1_000_000
    is_initial = first_in_person | (interval > disability.MAX_INTERVAL)

    group_id = np.cumsum(first_in_person) - 1
    group_start = np.flatnonzero(first_in_person)
    wave_position = np.arange(n_rows) - group_start[group_id]

    lower_window, upper_window = disability.CONVERSION_WINDOW
    disabled = np.zeros(n_rows, dtype=bool)
    retired = np.zeros(n_rows, dtype=bool)

    max_position = int(wave_position.max()) if n_rows else -1
    for position in range(max_position + 1):
        rows = np.flatnonzero(wave_position == position)
        if rows.size == 0:
            continue
        count = rows.size
        if rng_by_period is None:
            assert rng is not None
            state_uniform = rng.random(count)
            split_uniform = rng.random(count)
        else:
            missing_periods = set(period[rows]) - set(rng_by_period)
            if missing_periods:
                raise ValueError(
                    f"missing disability RNG period(s): {sorted(missing_periods)}"
                )
            state_uniform = np.asarray(
                [rng_by_period[int(period[row])].random() for row in rows]
            )
            split_uniform = np.asarray(
                [rng_by_period[int(period[row])].random() for row in rows]
            )

        previous = np.maximum(rows - 1, 0)
        previous_disabled = disabled[previous]
        previous_retired = retired[previous]
        band_current = band_here[rows]
        band_previous = band_here[previous]
        sex_current = sex_index[rows]
        age_previous = age[previous]

        initial = is_initial[rows]
        transition = ~initial
        from_not_disabled = transition & ~previous_disabled & ~previous_retired
        from_disabled = transition & previous_disabled
        from_retired = transition & previous_retired

        new_disabled = np.zeros(count, dtype=bool)
        new_retired = np.zeros(count, dtype=bool)

        initial_probability = model.prevalence0[band_current, sex_current]
        new_disabled |= initial & (state_uniform < initial_probability)

        incidence_probability = model.incidence[band_previous, sex_current]
        new_disabled |= from_not_disabled & (
            state_uniform < incidence_probability
        )

        recovery_probability = model.recovery[band_previous, sex_current]
        exits = from_disabled & (state_uniform < recovery_probability)
        new_disabled |= from_disabled & ~exits
        in_window = (age_previous >= lower_window) & (
            age_previous <= upper_window
        )
        retirement_share = model.exit_retirement_share[sex_current]
        new_retired |= exits & in_window & (split_uniform < retirement_share)
        new_retired |= from_retired

        disabled[rows] = new_disabled
        retired[rows] = new_retired

    simulated = pd.DataFrame(
        {
            "person_id": person_id,
            "period": period,
            "age": age,
            "sex": pd.array(sex, dtype="string"),
            "weight": weight,
            "disabled": disabled,
            "retired": retired,
        }
    )
    status_code = np.ones(n_rows, dtype=np.int64)
    status_code[retired] = disability.RETIRED_CODE
    status_code[disabled] = disability.DISABLED_CODE
    simulated["status_code"] = status_code
    prior_disabled = np.zeros(n_rows, dtype=bool)
    prior_retired = np.zeros(n_rows, dtype=bool)
    if n_rows > 1:
        prior_disabled[1:] = disabled[:-1]
        prior_retired[1:] = retired[:-1]
    simulated["di_converted"] = (
        ~first_in_person
        & (interval <= disability.MAX_INTERVAL)
        & prior_disabled
        & ~prior_retired
        & retired
    )

    pairs = disability.build_transition_pairs(simulated)
    simulated_person_years = simulated[
        [
            "person_id",
            "period",
            "age",
            "sex",
            "weight",
            "status_code",
            "disabled",
            "retired",
            "di_converted",
        ]
    ].reset_index(drop=True)
    return disability.DisabilityPanel(
        person_years=simulated_person_years,
        pairs=pairs,
    )
