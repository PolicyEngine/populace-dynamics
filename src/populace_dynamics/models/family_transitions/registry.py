"""Flattened component registry for resolved candidate 16.

Unlike the frozen candidate ladder, runtime resolution never imports an
ancestor runner or redirects a function's globals. Candidate 16 is a complete,
immutable list of implementation IDs whose fitters live under this package.
The effective lineage and final overrides are documented in
``scripts/run_gate2_candidate16.py:645-713``; each component module cites the
precise frozen source statements it ports.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from populace_dynamics.data import transitions
from populace_dynamics.models.family_transitions.components.divorce import (
    fit_divorce,
)
from populace_dynamics.models.family_transitions.components.fertility import (
    fit_fertility,
)
from populace_dynamics.models.family_transitions.components.first_marriage import (
    fit_first_marriage,
)
from populace_dynamics.models.family_transitions.components.initial_states import (
    ObservedInitialStates,
    fit_observed_initial_states,
)
from populace_dynamics.models.family_transitions.components.remarriage import (
    fit_remarriage,
)
from populace_dynamics.models.family_transitions.components.spousal_age_gap import (
    fit_spousal_age_gaps,
)
from populace_dynamics.models.family_transitions.components.widowhood import (
    fit_widowhood,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)

__all__ = [
    "CANDIDATE_16",
    "REGISTRY",
    "CandidateSpec",
    "ComponentDefinition",
    "ComponentRef",
    "ComponentRegistry",
    "FitContext",
]


@dataclass(frozen=True)
class FitContext:
    """Train split and source records shared by all component fitters."""

    panel: transitions.MaritalPanel
    demographic_panel: pd.DataFrame
    marriage_records: pd.DataFrame
    birth_records: pd.DataFrame
    marriage_order_map: pd.DataFrame
    train_ids: frozenset[int]


@dataclass(frozen=True)
class ComponentRef:
    """Immutable selection of one registered component implementation."""

    kind: str
    implementation_id: str
    params: Mapping[str, Any]


ComponentFitter = Callable[[FitContext, dict[str, Any]], Any]


@dataclass(frozen=True)
class ComponentDefinition:
    """One concrete implementation and its explicit fit dependency adapter."""

    kind: str
    implementation_id: str
    fitter: ComponentFitter
    source: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    """A fully materialized candidate with no runtime inheritance."""

    candidate_id: str
    contract_revision: str
    components: tuple[ComponentRef, ...]

    def canonical_dict(self) -> dict[str, Any]:
        """Return the deterministic resolved specification payload."""
        return {
            "candidate_id": self.candidate_id,
            "contract_revision": self.contract_revision,
            "components": [
                {
                    "kind": component.kind,
                    "implementation_id": component.implementation_id,
                    "params": json.loads(json.dumps(dict(component.params))),
                }
                for component in self.components
            ],
        }

    @property
    def sha256(self) -> str:
        """Hash the complete resolved spec rather than an ancestor chain."""
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class ComponentRegistry:
    """Resolve and fit immutable component definitions by implementation ID."""

    def __init__(self, definitions: tuple[ComponentDefinition, ...]) -> None:
        keyed: dict[tuple[str, str], ComponentDefinition] = {}
        for definition in definitions:
            key = (definition.kind, definition.implementation_id)
            if key in keyed:
                raise ValueError(f"duplicate component definition: {key!r}")
            keyed[key] = definition
        self._definitions = MappingProxyType(keyed)

    def definition(self, reference: ComponentRef) -> ComponentDefinition:
        """Return the exact definition selected by ``reference``."""
        key = (reference.kind, reference.implementation_id)
        try:
            definition = self._definitions[key]
        except KeyError as error:
            raise KeyError(
                f"unregistered family-transition component: {key}"
            ) from error
        if reference.params != definition.params:
            raise ValueError(
                f"parameters for {key!r} do not match its registered "
                f"implementation: selected={dict(reference.params)!r}, "
                f"registered={dict(definition.params)!r}"
            )
        return definition

    def fit(
        self, spec: CandidateSpec, context: FitContext
    ) -> FittedFamilyTransitions:
        """Fit a fully resolved candidate in declared dependency order."""
        fitted: dict[str, Any] = {}
        implementation_ids: dict[str, str] = {}
        for reference in spec.components:
            if reference.kind in fitted:
                raise ValueError(
                    f"candidate repeats component kind {reference.kind!r}"
                )
            definition = self.definition(reference)
            fitted[reference.kind] = definition.fitter(context, fitted)
            implementation_ids[reference.kind] = reference.implementation_id
        required = {
            "initial_states",
            "first_marriage",
            "divorce",
            "widowhood",
            "remarriage",
            "fertility",
            "spousal_age_gap",
        }
        if set(fitted) != required:
            raise ValueError(
                "candidate component kinds differ from the complete model: "
                f"missing={sorted(required - set(fitted))}, "
                f"extra={sorted(set(fitted) - required)}"
            )
        return FittedFamilyTransitions(
            first_marriage=fitted["first_marriage"],
            divorce=fitted["divorce"],
            widowhood=fitted["widowhood"],
            remarriage=fitted["remarriage"],
            fertility=fitted["fertility"],
            initial_states=fitted["initial_states"],
            spousal_age_gaps=fitted["spousal_age_gap"],
            implementation_ids=implementation_ids,
        )


def _fit_initial_states(context: FitContext, _: dict[str, Any]) -> Any:
    return fit_observed_initial_states(
        context.panel, context.demographic_panel
    )


def _train_frames(
    context: FitContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_person_years = context.panel.person_years[
        context.panel.person_years["person_id"].isin(context.train_ids)
    ]
    train_events = context.panel.events[
        context.panel.events["person_id"].isin(context.train_ids)
    ]
    attrs = context.panel.attrs.set_index("person_id")
    birth_decade = (attrs["birth_year"] // 10 * 10).astype("int64")
    return train_person_years, train_events, birth_decade


def _fit_first_marriage(context: FitContext, _: dict[str, Any]) -> Any:
    train_person_years, train_events, birth_decade = _train_frames(context)
    never_married = train_person_years[
        train_person_years["marital_state"] == "never_married"
    ][["person_id", "year", "age", "sex", "weight"]].copy()
    never_married["birth_decade"] = (
        never_married["person_id"].map(birth_decade).to_numpy()
    )
    first_marriages = train_events[
        train_events["transition"] == "first_marriage"
    ]
    event_years = {
        (int(person_id), int(year))
        for person_id, year in zip(
            first_marriages["person_id"].to_numpy(),
            first_marriages["year"].to_numpy(),
            strict=True,
        )
    }
    return fit_first_marriage(never_married, event_years)


def _fit_divorce(context: FitContext, _: dict[str, Any]) -> Any:
    train_person_years, train_events, _ = _train_frames(context)
    return fit_divorce(
        train_person_years,
        train_events,
        context.marriage_order_map,
    )


def _fit_remarriage(context: FitContext, _: dict[str, Any]) -> Any:
    return fit_remarriage(context.panel, set(context.train_ids))


def _fit_fertility(context: FitContext, _: dict[str, Any]) -> Any:
    _, _, birth_decade = _train_frames(context)
    return fit_fertility(
        context.panel,
        context.birth_records,
        set(context.train_ids),
        birth_decade,
    )


def _fit_gap(context: FitContext, _: dict[str, Any]) -> Any:
    return fit_spousal_age_gaps(
        context.marriage_records,
        context.panel.attrs,
        set(context.train_ids),
    )


def _fit_widowhood(context: FitContext, fitted: dict[str, Any]) -> Any:
    initial_states: ObservedInitialStates = fitted["initial_states"]
    return fit_widowhood(
        context.panel,
        set(context.train_ids),
        initial_states.support.by_person,
    )


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze JSON-like component parameters."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _params(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _deep_freeze(value) for key, value in values.items()}
    )


_INITIAL_STATE_PARAMS = _params(support_threshold_age=75)
_FIRST_MARRIAGE_PARAMS = _params(knots=[20, 22, 25, 30, 40])
_DIVORCE_PARAMS = _params(duration_bands=[[0, 4], [5, 9], [10, 19], [20, 120]])
_REMARRIAGE_PARAMS = _params(
    age_bands=[[18, 34], [35, 49], [50, 64], [65, 74], [75, 120]]
)
_FERTILITY_PARAMS = _params(age_range=[15, 49], bandwidth=3)
_SPOUSAL_GAP_PARAMS = _params(min_weighted_couples=200)
_WIDOWHOOD_PARAMS = _params(
    age_bands=[
        [18, 34],
        [35, 44],
        [45, 54],
        [55, 64],
        [65, 74],
        [75, 84],
        [85, 120],
    ],
    support_strata=[0, 1],
    period_trend_applied=False,
)


REGISTRY = ComponentRegistry(
    (
        ComponentDefinition(
            kind="initial_states",
            implementation_id="observed_residual_entry_widowed_support75.v1",
            fitter=_fit_initial_states,
            source="candidate9:189-232;c12:469-526,1103-1133;c16:410-486",
            params=_INITIAL_STATE_PARAMS,
        ),
        ComponentDefinition(
            kind="first_marriage",
            implementation_id="logit_ncs_age_sex_cohort_knots20_22_25_30_40.v1",
            fitter=_fit_first_marriage,
            source="candidate1:187-315;candidate2:147-175;candidate3:187-238",
            params=_FIRST_MARRIAGE_PARAMS,
        ),
        ComponentDefinition(
            kind="divorce",
            implementation_id="empirical_duration_order_add_one.v1",
            fitter=_fit_divorce,
            source="candidate1:384-414,485-499,884-889",
            params=_DIVORCE_PARAMS,
        ),
        ComponentDefinition(
            kind="remarriage",
            implementation_id="empirical_age5_ysd_origin_sex_add_one.v1",
            fitter=_fit_remarriage,
            source="candidate10:260-328,395-414;candidate11:163-171",
            params=_REMARRIAGE_PARAMS,
        ),
        ComponentDefinition(
            kind="fertility",
            implementation_id="single_age_triangular_bw3_parity_cohort.v1",
            fitter=_fit_fertility,
            source="candidate5:312-439",
            params=_FERTILITY_PARAMS,
        ),
        ComponentDefinition(
            kind="spousal_age_gap",
            implementation_id="empirical_age4_sex_adjacent_fallback.v1",
            fitter=_fit_gap,
            source="candidate12:317-460,793-831",
            params=_SPOUSAL_GAP_PARAMS,
        ),
        ComponentDefinition(
            kind="widowhood",
            implementation_id="mh85_23_age7_sex_support75_untrended.v1",
            fitter=_fit_widowhood,
            source="candidate14:379-430;candidate15:391-423;candidate16:497-634,795-823",
            params=_WIDOWHOOD_PARAMS,
        ),
    )
)


CANDIDATE_16 = CandidateSpec(
    candidate_id="candidate16_registry_v1",
    contract_revision="gate_2_amendment_1",
    components=(
        ComponentRef(
            "initial_states",
            "observed_residual_entry_widowed_support75.v1",
            _INITIAL_STATE_PARAMS,
        ),
        ComponentRef(
            "first_marriage",
            "logit_ncs_age_sex_cohort_knots20_22_25_30_40.v1",
            _FIRST_MARRIAGE_PARAMS,
        ),
        ComponentRef(
            "divorce",
            "empirical_duration_order_add_one.v1",
            _DIVORCE_PARAMS,
        ),
        ComponentRef(
            "remarriage",
            "empirical_age5_ysd_origin_sex_add_one.v1",
            _REMARRIAGE_PARAMS,
        ),
        ComponentRef(
            "fertility",
            "single_age_triangular_bw3_parity_cohort.v1",
            _FERTILITY_PARAMS,
        ),
        ComponentRef(
            "spousal_age_gap",
            "empirical_age4_sex_adjacent_fallback.v1",
            _SPOUSAL_GAP_PARAMS,
        ),
        ComponentRef(
            "widowhood",
            "mh85_23_age7_sex_support75_untrended.v1",
            _WIDOWHOOD_PARAMS,
        ),
    ),
)
