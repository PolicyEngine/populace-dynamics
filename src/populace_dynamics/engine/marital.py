"""Engine-side candidate-16 simulation with injected random generators.

The certified family-transition simulator accepts an integer seed and creates
its own generators.  M6 needs a period axis without changing that frozen
implementation, so this module ports the candidate-16 state machine and takes
the main and spousal-gap generators as explicit inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import couple_earnings as ce
from populace_dynamics.data import transitions
from populace_dynamics.engine.loop import MaritalStepResult
from populace_dynamics.models.couple_formation_sim_v1 import CommittedAxis
from populace_dynamics.models.couple_formation_sim_v2 import (
    FirstMarriageEarningsModifier,
    apply_first_marriage_modifier,
)
from populace_dynamics.models.family_transitions import simulator as ft_sim
from populace_dynamics.models.family_transitions.components.divorce import (
    divorce_probabilities,
)
from populace_dynamics.models.family_transitions.components.fertility import (
    FERTILITY_AGE_HI,
    FERTILITY_AGE_LO,
    fertility_probabilities,
)
from populace_dynamics.models.family_transitions.components.initial_states import (
    apply_entry_widowed,
)
from populace_dynamics.models.family_transitions.components.remarriage import (
    remarriage_probabilities,
)
from populace_dynamics.models.family_transitions.components.spousal_age_gap import (
    draw_spousal_gaps,
)
from populace_dynamics.models.family_transitions.components.widowhood import (
    widowhood_probabilities,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)

_STATE = {"never_married": 0, "married": 1, "divorced": 2, "widowed": 3}
_STATE_ABSORB = 4


def _simulate_candidate16_with_generators(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: FittedFamilyTransitions,
    main_rng: np.random.Generator,
    gap_rng: np.random.Generator,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Port candidate 16 exactly, replacing only RNG construction."""
    attrs = panel.attrs[panel.attrs["person_id"].isin(holdout_ids)].copy()
    attrs = attrs.sort_values("person_id").reset_index(drop=True)
    n_people = len(attrs)
    person_id = attrs["person_id"].to_numpy(dtype=np.int64)
    birth_year = attrs["birth_year"].to_numpy(dtype=np.float64)
    sex = attrs["sex"].to_numpy()
    is_male = (sex == "male").astype(np.float64)
    start_year = attrs["start_exposure_year"].to_numpy(dtype=np.int64)
    end_year = attrs["censor_year"].to_numpy(dtype=np.int64)
    birth_decade = (birth_year // 10 * 10).astype(np.int64)
    support_stratum = components.initial_states.support.align(person_id)

    person_years = panel.person_years
    entry = (
        person_years[person_years["person_id"].isin(holdout_ids)]
        .sort_values("year")
        .groupby("person_id", as_index=False)
        .first()
    )
    entry_state = (
        entry.set_index("person_id")["marital_state"]
        .reindex(person_id)
        .to_numpy()
    )
    entry_duration = entry.set_index("person_id")["marriage_duration"].reindex(
        person_id
    )
    entry_years_since = entry.set_index("person_id")[
        "years_since_dissolution"
    ].reindex(person_id)

    state = np.zeros(n_people, dtype=np.int64)
    current_start = np.full(n_people, -1, dtype=np.int64)
    order = np.zeros(n_people, dtype=np.int64)
    dissolution_year = np.full(n_people, -1, dtype=np.int64)
    open_start = np.full(n_people, -1, dtype=np.int64)
    open_order = np.zeros(n_people, dtype=np.int64)
    for index in range(n_people):
        observed_state = entry_state[index]
        if pd.isna(observed_state) or observed_state == "never_married":
            state[index] = 0
        elif observed_state == "married":
            state[index] = 1
            duration = entry_duration.iloc[index]
            duration_value = int(duration) if not pd.isna(duration) else 0
            current_start[index] = int(start_year[index]) - duration_value
            order[index] = 1
            open_start[index] = current_start[index]
            open_order[index] = 1
        elif observed_state in ("divorced", "widowed"):
            state[index] = _STATE[observed_state]
            years_since = entry_years_since.iloc[index]
            years_since_value = (
                int(years_since) if not pd.isna(years_since) else 0
            )
            dissolution_year[index] = (
                int(start_year[index]) - years_since_value
            )
            order[index] = 1
        else:
            state[index] = _STATE_ABSORB

    gap = np.zeros(n_people, dtype=np.float64)
    entry_married = np.nonzero(state == 1)[0]
    if entry_married.size:
        gap[entry_married] = draw_spousal_gaps(
            gap_rng,
            entry_married,
            current_start[entry_married].astype(np.float64)
            - birth_year[entry_married],
            is_male,
            components.spousal_age_gaps,
        )

    lookups = ft_sim._build_lookups(components)
    episode_person: list[int] = []
    episode_order: list[int] = []
    episode_start: list[int] = []
    episode_end: list[Any] = []
    episode_how: list[str] = []

    def close_episode(indices: np.ndarray, how: str, year: int) -> None:
        for index in indices:
            episode_person.append(int(person_id[index]))
            episode_order.append(int(open_order[index]))
            episode_start.append(int(open_start[index]))
            episode_end.append(int(year))
            episode_how.append(how)

    first_simulation_year = int(start_year.min())
    last_simulation_year = int(end_year.max())
    for year in range(first_simulation_year, last_simulation_year + 1):
        active = (start_year <= year) & (year <= end_year)
        active_index = np.nonzero(active)[0]
        if active_index.size == 0:
            continue
        age = year - birth_year[active_index]
        uniform = main_rng.random(active_index.size)
        state_snapshot = state[active_index]

        never_married = state_snapshot == 0
        if never_married.any():
            subset = active_index[never_married]
            probability = components.first_marriage.predict(
                age[never_married],
                is_male[subset],
                birth_decade[subset],
            )
            marrying = uniform[never_married] < probability
            global_index = subset[marrying]
            order[global_index] += 1
            current_start[global_index] = year
            state[global_index] = 1
            open_start[global_index] = year
            open_order[global_index] = order[global_index]
            if global_index.size:
                gap[global_index] = draw_spousal_gaps(
                    gap_rng,
                    global_index,
                    year - birth_year[global_index],
                    is_male,
                    components.spousal_age_gaps,
                )

        married = state_snapshot == 1
        if married.any():
            subset = active_index[married]
            duration = (year - current_start[subset]).astype(np.int64)
            divorce_probability = divorce_probabilities(
                duration, order[subset], components.divorce
            )
            # Retained to preserve candidate 16's operation sequence.
            _spouse_age = age[married] + gap[subset]
            widowhood_probability = widowhood_probabilities(
                age[married],
                is_male[subset],
                support_stratum[subset],
                components.widowhood.lookup,
            )
            married_uniform = uniform[married]
            divorcing = married_uniform < divorce_probability
            becoming_widowed = (~divorcing) & (
                married_uniform < divorce_probability + widowhood_probability
            )
            divorce_index = subset[divorcing]
            close_episode(divorce_index, "divorce", year)
            state[divorce_index] = 2
            dissolution_year[divorce_index] = year
            widow_index = subset[becoming_widowed]
            close_episode(widow_index, "widowhood", year)
            state[widow_index] = 3
            dissolution_year[widow_index] = year

        dissolved = (state_snapshot == 2) | (state_snapshot == 3)
        if dissolved.any():
            subset = active_index[dissolved]
            years_since = (year - dissolution_year[subset]).astype(np.int64)
            origin = state_snapshot[dissolved]
            probability = remarriage_probabilities(
                age[dissolved],
                years_since,
                origin,
                is_male[subset],
                lookups.remarriage,
            )
            remarrying = uniform[dissolved] < probability
            global_index = subset[remarrying]
            order[global_index] += 1
            current_start[global_index] = year
            state[global_index] = 1
            dissolution_year[global_index] = -1
            open_start[global_index] = year
            open_order[global_index] = order[global_index]
            if global_index.size:
                gap[global_index] = draw_spousal_gaps(
                    gap_rng,
                    global_index,
                    year - birth_year[global_index],
                    is_male,
                    components.spousal_age_gaps,
                )

    still_married = np.nonzero(state == 1)[0]
    for index in still_married:
        episode_person.append(int(person_id[index]))
        episode_order.append(int(open_order[index]))
        episode_start.append(int(open_start[index]))
        episode_end.append(pd.NA)
        episode_how.append("intact")

    simulated_panel = ft_sim._assemble_panel(
        attrs,
        episode_person,
        episode_order,
        episode_start,
        episode_end,
        episode_how,
    )
    apply_entry_widowed(
        simulated_panel, components.initial_states.entry_widowed
    )
    residual = components.initial_states.marriage_residual_by_person
    observed_addition = (
        simulated_panel.attrs["person_id"]
        .map(residual)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    simulated_panel.attrs["n_marriages"] = (
        simulated_panel.attrs["n_marriages"].to_numpy(dtype="float64")
        + observed_addition
    )
    return simulated_panel, _empty_births()


def _empty_births() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_person_id": pd.Series(dtype="int64"),
            "birth_year": pd.Series(dtype="Int64"),
            "birth_order": pd.Series(dtype="Int64"),
            "record_type": pd.Series(dtype="string"),
            "is_event": pd.Series(dtype="bool"),
        }
    )


def simulate_maternal_births(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: FittedFamilyTransitions,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run candidate-16's fertility kernel on the step-4 stream."""
    attrs = panel.attrs[panel.attrs["person_id"].isin(holdout_ids)].copy()
    attrs = attrs.sort_values("person_id").reset_index(drop=True)
    if attrs.empty:
        return _empty_births()
    person_id = attrs["person_id"].to_numpy(dtype=np.int64)
    birth_year = attrs["birth_year"].to_numpy(dtype=np.float64)
    sex = attrs["sex"].to_numpy()
    start_year = attrs["start_exposure_year"].to_numpy(dtype=np.int64)
    end_year = attrs["censor_year"].to_numpy(dtype=np.int64)
    birth_decade = (birth_year // 10 * 10).astype(np.int64)
    lookups = ft_sim._build_lookups(components)
    decade_index = np.array(
        [
            lookups.fertility_decade.get(int(decade), -1)
            for decade in birth_decade
        ],
        dtype=np.int64,
    )
    parity = np.zeros(len(attrs), dtype=np.int64)
    out_person: list[int] = []
    out_year: list[int] = []
    out_order: list[int] = []
    for year in range(int(start_year.min()), int(end_year.max()) + 1):
        age = (year - birth_year).astype(np.int64)
        fertile = (
            (start_year <= year)
            & (year <= end_year)
            & (sex == "female")
            & (age >= FERTILITY_AGE_LO)
            & (age <= FERTILITY_AGE_HI)
        )
        indices = np.flatnonzero(fertile)
        if not indices.size:
            continue
        probability = fertility_probabilities(
            age[indices],
            parity[indices],
            decade_index[indices],
            lookups.fertility,
        )
        born = rng.random(indices.size) < probability
        born_indices = indices[born]
        for index in born_indices:
            out_person.append(int(person_id[index]))
            out_year.append(year)
            out_order.append(int(parity[index]) + 1)
        parity[born_indices] += 1
    return pd.DataFrame(
        {
            "parent_person_id": np.asarray(out_person, dtype=np.int64),
            "birth_year": pd.array(out_year, dtype="Int64"),
            "birth_order": pd.array(out_order, dtype="Int64"),
            "record_type": pd.array(
                ["birth"] * len(out_person), dtype="string"
            ),
            "is_event": np.ones(len(out_person), dtype=bool),
        }
    )


def simulate_marital_step(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: FittedFamilyTransitions,
    modifier: FirstMarriageEarningsModifier,
    permanent_axis: CommittedAxis,
    *,
    main_rng: np.random.Generator,
    gap_rng: np.random.Generator,
) -> MaritalStepResult:
    """Run authoritative step 3 and attach the no-RNG gate-2c view."""
    simulated_panel, births = _simulate_candidate16_with_generators(
        panel,
        holdout_ids,
        components,
        main_rng,
        gap_rng,
    )
    events, exposure = ce._build_marital(
        simulated_panel,
        permanent_axis.earn,
        permanent_axis.cuts,
    )
    weighted_events, check = apply_first_marriage_modifier(
        events,
        modifier,
        exposure,
    )
    return MaritalStepResult(
        panel=simulated_panel,
        sim_years=simulated_panel.person_years,
        births=births,
        weighted_events=weighted_events,
        exposure=exposure,
        modifier_check=check,
    )
