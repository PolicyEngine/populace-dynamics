"""Vectorized annual simulator for the flattened candidate-16 registry.

The state machine, competing-risk priority, RNG topology, observed-state
application, and output assembly port
``scripts/run_gate2_candidate16.py:748-1079``. The assembly helper is copied
from ``scripts/run_gate2_candidate1.py:925-969``. No frozen runner is imported
or redirected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import transitions
from populace_dynamics.models.family_transitions.components.divorce import (
    divorce_probabilities,
)
from populace_dynamics.models.family_transitions.components.fertility import (
    FERTILITY_AGE_HI,
    FERTILITY_AGE_LO,
    build_fertility_lookup,
    fertility_probabilities,
)
from populace_dynamics.models.family_transitions.components.initial_states import (
    apply_entry_widowed,
)
from populace_dynamics.models.family_transitions.components.remarriage import (
    build_remarriage_lookup,
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

__all__ = ["simulate"]

_STATE = {"never_married": 0, "married": 1, "divorced": 2, "widowed": 3}
_STATE_ABSORB = 4


@dataclass(frozen=True)
class _SimulationLookups:
    remarriage: np.ndarray
    fertility: np.ndarray
    fertility_decade: dict[int, int]


def _build_lookups(components: FittedFamilyTransitions) -> _SimulationLookups:
    fertility, fertility_decade = build_fertility_lookup(components.fertility)
    return _SimulationLookups(
        remarriage=build_remarriage_lookup(components.remarriage),
        fertility=fertility,
        fertility_decade=fertility_decade,
    )


def _assemble_panel(
    attrs: pd.DataFrame,
    episode_person: list[int],
    episode_order: list[int],
    episode_start: list[int],
    episode_end: list[Any],
    episode_how: list[str],
) -> transitions.MaritalPanel:
    """Build a simulated panel through the reference assembly functions."""
    marriage_count = (
        pd.Series(episode_person).value_counts()
        if episode_person
        else pd.Series(dtype="int64")
    )
    simulated_attrs = attrs.copy()
    simulated_attrs["n_marriages"] = (
        simulated_attrs["person_id"]
        .map(marriage_count)
        .fillna(0)
        .astype("float64")
    )
    episodes = pd.DataFrame(
        {
            "person_id": np.array(episode_person, dtype=np.int64),
            "marriage_order": pd.array(episode_order, dtype="Int64"),
            "start_year": pd.array(episode_start, dtype="Int64"),
            "start_month": pd.array(
                [pd.NA] * len(episode_person), dtype="Int64"
            ),
            "episode_end_year": pd.array(episode_end, dtype="Int64"),
            "how_ended": pd.array(episode_how, dtype="string"),
            "spouse_person_id": pd.array(
                [pd.NA] * len(episode_person), dtype="Int64"
            ),
            "last_known_status": pd.array(
                [pd.NA] * len(episode_person), dtype="string"
            ),
        }
    )
    episodes["episode_duration_years"] = (
        episodes["episode_end_year"] - episodes["start_year"]
    ).astype("Int64")
    changepoints = transitions._changepoints(episodes, simulated_attrs)
    person_years = transitions._assign_state(
        transitions._person_years_frame(simulated_attrs), changepoints
    )
    events = transitions._events_frame(episodes, simulated_attrs)
    return transitions.MaritalPanel(
        person_years=person_years,
        events=events,
        attrs=simulated_attrs,
    )


def simulate(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: FittedFamilyTransitions,
    simulation_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Simulate one candidate-16 draw over the holdout persons.

    Candidate 16 draws one active-person uniform block before hazard lookup,
    uses divorce priority over widowhood, snapshots state before all transition
    blocks, and draws a separate fertile-woman block. The main generator and
    spawned gap generator preserve
    ``scripts/run_gate2_candidate16.py:826-1079`` exactly.
    """
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
    parity = np.zeros(n_people, dtype=np.int64)
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

    rng = np.random.default_rng(simulation_seed)
    gap_seed_sequence = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_sequence)
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

    lookups = _build_lookups(components)
    fertility_decade_index = np.array(
        [
            lookups.fertility_decade.get(int(decade), -1)
            for decade in birth_decade
        ],
        dtype=np.int64,
    )

    episode_person: list[int] = []
    episode_order: list[int] = []
    episode_start: list[int] = []
    episode_end: list[Any] = []
    episode_how: list[str] = []
    birth_person: list[int] = []
    simulated_birth_year: list[int] = []
    birth_order: list[int] = []

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
        uniform = rng.random(active_index.size)
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
            # Candidate 16 retains this inert spouse-age computation so its
            # spawned gap stream stays identical to the frozen run.
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

        age_all = (year - birth_year).astype(np.int64)
        fertile = (
            active
            & (sex == "female")
            & (age_all >= FERTILITY_AGE_LO)
            & (age_all <= FERTILITY_AGE_HI)
        )
        fertile_index = np.nonzero(fertile)[0]
        if fertile_index.size:
            fertility_uniform = rng.random(fertile_index.size)
            fertile_age = (year - birth_year[fertile_index]).astype(np.int64)
            probability = fertility_probabilities(
                fertile_age,
                parity[fertile_index],
                fertility_decade_index[fertile_index],
                lookups.fertility,
            )
            born = fertility_uniform < probability
            global_index = fertile_index[born]
            for index in global_index:
                birth_person.append(int(person_id[index]))
                simulated_birth_year.append(int(year))
                birth_order.append(int(parity[index]) + 1)
            parity[global_index] += 1

    still_married = np.nonzero(state == 1)[0]
    for index in still_married:
        episode_person.append(int(person_id[index]))
        episode_order.append(int(open_order[index]))
        episode_start.append(int(open_start[index]))
        episode_end.append(pd.NA)
        episode_how.append("intact")

    simulated_panel = _assemble_panel(
        attrs,
        episode_person,
        episode_order,
        episode_start,
        episode_end,
        episode_how,
    )
    simulated_births = pd.DataFrame(
        {
            "parent_person_id": np.array(birth_person, dtype=np.int64),
            "birth_year": pd.array(simulated_birth_year, dtype="Int64"),
            "birth_order": pd.array(birth_order, dtype="Int64"),
            "record_type": pd.array(
                ["birth"] * len(birth_person), dtype="string"
            ),
            "is_event": np.ones(len(birth_person), dtype=bool),
        }
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
    return simulated_panel, simulated_births
