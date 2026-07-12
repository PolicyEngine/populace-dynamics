"""Immutable component registry for resolved gate-2b candidate 9.

Unlike the frozen v1--v9 ladder, registry resolution never imports an ancestor
household simulator.  The effective lineage is flattened into ten named
components whose copied sources are cited in their module docstrings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.models.household_composition.components.child_attribution import (
    enumerated_joinable_keys,
    father_link_births,
    fit_custodial_child_record,
    fit_custodial_era,
    fit_linked_episode_persistence,
)
from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
    COHAB_OVERLAY_LIFT,
    cohabitation_flag,
    fit_cohab_single_year,
    fit_cohabitation_rates,
    fit_female_cohab_single_year,
)
from populace_dynamics.models.household_composition.components.fertility_core_lift import (
    N_FIT_DRAWS,
    completed_size_dist_by_cell,
    fit_retention_link,
    train_completed_size,
)
from populace_dynamics.models.household_composition.components.household_size import (
    fit_parent_count_composition,
)
from populace_dynamics.models.household_composition.components.legal_spouse_residual import (
    fit_legal_residual,
)
from populace_dynamics.models.household_composition.components.marital_core_adapter import (
    fit_family_transitions,
    fit_male_gap,
)
from populace_dynamics.models.household_composition.components.multigenerational_occupancy import (
    fit_multigen_child_coupling,
    fit_multigen_rates,
)
from populace_dynamics.models.household_composition.components.nonfamily_bridge import (
    fit_nonfamily_count_by_core,
)
from populace_dynamics.models.household_composition.components.parental_home_exit import (
    fit_child_exit_single_year,
    fit_parental_exit,
)
from populace_dynamics.models.household_composition.components.skip_generation_state import (
    attach_skipgen,
    fit_skipgen_5yr,
    fit_skipgen_rates,
)
from populace_dynamics.models.household_composition.fitted import (
    FittedHouseholdComposition,
)

__all__ = [
    "CANDIDATE_9",
    "REGISTRY",
    "CandidateSpec",
    "ComponentDefinition",
    "ComponentRef",
    "ComponentRegistry",
    "FitContext",
    "fit_context_from_sources",
]


@dataclass(frozen=True)
class FitContext:
    """Train split and seed-independent data shared by component fitters."""

    hh: hc.HouseholdCompositionPanel
    mpanel: transitions.MaritalPanel
    demographic_panel: pd.DataFrame
    marriage_records: pd.DataFrame
    birth_records: pd.DataFrame
    marriage_order_map: pd.DataFrame
    relationship_map: pd.DataFrame
    train_ids: frozenset[int]
    father_links_child: pd.DataFrame
    parent_pairs: pd.DataFrame
    marital_by_year: pd.DataFrame
    family_unit_sizes: pd.DataFrame
    legal_spouse_flag: pd.DataFrame
    child_record_exposure: pd.DataFrame
    parent_counts: pd.DataFrame


def fit_context_from_sources(
    sources: Mapping[str, Any], train_ids: set[int]
) -> FitContext:
    """Build a context from :func:`household_composition.data.load_sources`."""
    return FitContext(
        hh=sources["hh"],
        mpanel=sources["mpanel"],
        demographic_panel=sources["demo"],
        marriage_records=sources["mh"],
        birth_records=sources["bh"],
        marriage_order_map=sources["order_map"],
        relationship_map=sources["rel_map"],
        train_ids=frozenset(train_ids),
        father_links_child=sources["father_links_child"],
        parent_pairs=sources["parent_pairs"],
        marital_by_year=sources["marital_by_year"],
        family_unit_sizes=sources["fu_sizes"],
        legal_spouse_flag=sources["legal_flag"],
        child_record_exposure=sources["child_record_expo"],
        parent_counts=sources["parent_counts"],
    )


@dataclass(frozen=True)
class ComponentRef:
    """Immutable selection of one component implementation."""

    kind: str
    implementation_id: str
    params: Mapping[str, Any]


ComponentFitter = Callable[[FitContext, dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class ComponentDefinition:
    """One registered component implementation and its frozen parameters."""

    kind: str
    implementation_id: str
    fitter: ComponentFitter
    source: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    """Fully resolved, ordered candidate specification."""

    candidate_id: str
    contract_revision: str
    components: tuple[ComponentRef, ...]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "contract_revision": self.contract_revision,
            "components": [
                {
                    "kind": component.kind,
                    "implementation_id": component.implementation_id,
                    "params": _thaw(component.params),
                }
                for component in self.components
            ],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class ComponentRegistry:
    """Registry that fits a resolved candidate in dependency order."""

    def __init__(self, definitions: tuple[ComponentDefinition, ...]) -> None:
        keyed: dict[tuple[str, str], ComponentDefinition] = {}
        for definition in definitions:
            key = (definition.kind, definition.implementation_id)
            if key in keyed:
                raise ValueError(f"duplicate component definition: {key!r}")
            keyed[key] = definition
        self._definitions = MappingProxyType(keyed)

    def definition(self, reference: ComponentRef) -> ComponentDefinition:
        key = (reference.kind, reference.implementation_id)
        try:
            definition = self._definitions[key]
        except KeyError as error:
            raise KeyError(
                f"unregistered household-composition component: {key}"
            ) from error
        if reference.params != definition.params:
            raise ValueError(f"parameters for {key!r} do not match registry")
        return definition

    def fit(
        self, spec: CandidateSpec, context: FitContext
    ) -> FittedHouseholdComposition:
        fitted: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        implementation_ids: dict[str, str] = {}
        seen_kinds: set[str] = set()
        for reference in spec.components:
            if reference.kind in seen_kinds:
                raise ValueError(f"candidate repeats {reference.kind!r}")
            definition = self.definition(reference)
            values = dict(definition.fitter(context, fitted))
            component_meta = values.pop("_meta", {})
            overlap = set(values) & set(fitted)
            if overlap:
                raise ValueError(
                    f"component {reference.kind!r} repeats fields {overlap}"
                )
            fitted.update(values)
            meta[reference.kind] = component_meta
            implementation_ids[reference.kind] = reference.implementation_id
            seen_kinds.add(reference.kind)
        required_kinds = {
            "marital_core_adapter",
            "cohabitation_overlay",
            "legal_spouse_residual",
            "parental_home_exit",
            "multigenerational_occupancy",
            "child_attribution",
            "skip_generation_state",
            "nonfamily_bridge",
            "household_size",
            "fertility_core_lift",
        }
        if seen_kinds != required_kinds:
            raise ValueError(
                "candidate component kinds differ from complete model: "
                f"missing={sorted(required_kinds - seen_kinds)}, "
                f"extra={sorted(seen_kinds - required_kinds)}"
            )
        return FittedHouseholdComposition(
            **fitted,
            component_meta=meta,
            implementation_ids=implementation_ids,
        )


def _fit_marital(context: FitContext, _: dict[str, Any]) -> Mapping[str, Any]:
    """Port ``household_composition_sim.py:332-357`` into the registry."""
    family = fit_family_transitions(
        context.mpanel,
        context.demographic_panel,
        context.marriage_records,
        context.birth_records,
        context.marriage_order_map,
        set(context.train_ids),
    )
    return {"family_transitions": family, "male_gap": fit_male_gap(family)}


def _fit_cohabitation(
    context: FitContext, _: dict[str, Any]
) -> Mapping[str, Any]:
    """Port fits from ``household_composition_sim_v2.py:285-343``,
    ``household_composition_sim_v4.py:701-708``, and
    ``household_composition_sim_v6.py:471-478``.
    """
    flag = cohabitation_flag(context.relationship_map)
    from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
        attach_cohabitation,
    )

    person_waves = attach_cohabitation(context.hh.person_waves, flag)
    train = person_waves[person_waves["person_id"].isin(context.train_ids)]
    entry, exit_ = fit_cohabitation_rates(train)
    entry_age, exit_age = fit_cohab_single_year(
        context.hh.person_waves,
        flag,
        set(context.train_ids),
        entry,
        exit_,
    )
    entry_female, exit_female, diag = fit_female_cohab_single_year(
        context.hh.person_waves,
        flag,
        set(context.train_ids),
        entry,
        exit_,
    )
    return {
        "cohab_flag": flag,
        "cohab_entry": entry,
        "cohab_exit": exit_,
        "cohab_entry_age": entry_age,
        "cohab_exit_age": exit_age,
        "cohab_entry_age_female": entry_female,
        "cohab_exit_age_female": exit_female,
        "cohab_overlay_lift": COHAB_OVERLAY_LIFT,
        "_meta": diag,
    }


def _fit_legal(
    context: FitContext, fitted: dict[str, Any]
) -> Mapping[str, Any]:
    """Port ``household_composition_sim_v4.py:709-716``."""
    entry, exit_, marginal, target, diag = fit_legal_residual(
        context.hh,
        context.mpanel,
        fitted["family_transitions"],
        set(context.train_ids),
        context.legal_spouse_flag,
    )
    return {
        "legal_residual_entry": entry,
        "legal_residual_exit": exit_,
        "legal_residual_marginal": marginal,
        "legal_residual_target": target,
        "_meta": diag,
    }


def _fit_parental(context: FitContext, _: dict[str, Any]) -> Mapping[str, Any]:
    """Port ``household_composition_sim.py:353-354`` and
    ``household_composition_sim_v6.py:467-470``.
    """
    train = context.hh.person_waves[
        context.hh.person_waves["person_id"].isin(context.train_ids)
    ]
    parental = fit_parental_exit(train)
    child_exit, diag = fit_child_exit_single_year(
        context.hh.person_waves, parental, set(context.train_ids)
    )
    return {
        "parental_exit": parental,
        "child_exit_single_year": child_exit,
        "_meta": diag,
    }


def _fit_multigen(context: FitContext, _: dict[str, Any]) -> Mapping[str, Any]:
    """Port ``household_composition_sim.py:355`` and
    ``household_composition_sim_v5.py:579-582``.
    """
    train = context.hh.person_waves[
        context.hh.person_waves["person_id"].isin(context.train_ids)
    ]
    entry, exit_ = fit_multigen_rates(train)
    coupling, pooled, diag = fit_multigen_child_coupling(
        context.hh.person_waves, set(context.train_ids)
    )
    return {
        "multigen_entry": entry,
        "multigen_exit": exit_,
        "coupling_child_given_multigen": coupling,
        "coupling_child_pooled": pooled,
        "_meta": diag,
    }


def _fit_children(
    context: FitContext, fitted: dict[str, Any]
) -> Mapping[str, Any]:
    """Port fits from ``household_composition_sim_v4.py:717-731``,
    ``household_composition_sim_v5.py:583-586``, and
    ``household_composition_sim_v7.py:619-640``.
    """
    custodial_era, age_marital, band_marital, overall, era_diag = (
        fit_custodial_era(
            context.hh.person_waves,
            context.father_links_child,
            context.parent_pairs,
            context.marital_by_year,
            set(context.train_ids),
        )
    )
    child_record, record_diag = fit_custodial_child_record(
        context.child_record_exposure, set(context.train_ids)
    )
    links = father_link_births(context.birth_records)
    joinable = enumerated_joinable_keys(context.father_links_child)
    partial = SimpleNamespace(
        **fitted,
        father_links=links,
        custodial_era=custodial_era,
        custodial_age_marital=age_marital,
        custodial_band_marital=band_marital,
        custodial_overall=overall,
        custodial_child_record=child_record,
    )
    persistence, persistence_diag = fit_linked_episode_persistence(
        context.hh.person_waves,
        partial,
        context.father_links_child,
        context.parent_pairs,
        context.marital_by_year,
        joinable,
        set(context.train_ids),
    )
    return {
        "father_links": links,
        "custodial_era": custodial_era,
        "custodial_age_marital": age_marital,
        "custodial_band_marital": band_marital,
        "custodial_overall": overall,
        "custodial_child_record": child_record,
        "joinable_keys": joinable,
        "linked_episode_persistence": persistence,
        "_meta": {
            "custodial_era": era_diag,
            "child_record": record_diag,
            "episode_persistence": persistence_diag,
        },
    }


def _fit_skipgen(context: FitContext, _: dict[str, Any]) -> Mapping[str, Any]:
    """Port ``household_composition_sim_v4.py:736-741``."""
    person_waves = attach_skipgen(context.hh.person_waves)
    train = person_waves[person_waves["person_id"].isin(context.train_ids)]
    entry, exit_ = fit_skipgen_rates(train)
    entry_age, exit_age = fit_skipgen_5yr(train, entry, exit_)
    return {
        "skipgen_entry": entry,
        "skipgen_exit": exit_,
        "skipgen_entry_age": entry_age,
        "skipgen_exit_age": exit_age,
    }


def _fit_nonfamily(
    context: FitContext, _: dict[str, Any]
) -> Mapping[str, Any]:
    """Port ``household_composition_sim_v6.py:479-482``."""
    table, diag = fit_nonfamily_count_by_core(
        context.hh.person_waves,
        context.family_unit_sizes,
        set(context.train_ids),
    )
    return {"nonfamily_count_by_core": table, "_meta": diag}


def _fit_household_size(
    context: FitContext, _: dict[str, Any]
) -> Mapping[str, Any]:
    """Port ``household_composition_sim_v5.py:591-594``."""
    table, pooled, diag = fit_parent_count_composition(
        context.hh.person_waves,
        context.parent_counts,
        set(context.train_ids),
    )
    return {
        "parent_count_two_share": table,
        "parent_count_two_pooled": pooled,
        "_meta": diag,
    }


def _fit_fertility(
    context: FitContext, fitted: dict[str, Any]
) -> Mapping[str, Any]:
    """Port ``household_composition_sim_v8.py:746-761``."""
    size_map = train_completed_size(
        context.parent_pairs, set(context.train_ids)
    )
    by_cell, overall = completed_size_dist_by_cell(
        context.hh, size_map, set(context.train_ids)
    )
    partial = SimpleNamespace(**fitted)
    shift, diag = fit_retention_link(
        context.hh,
        context.mpanel,
        partial,
        context.father_links_child,
        context.parent_pairs,
        context.marital_by_year,
        size_map,
        set(context.train_ids),
        n_fit_draws=N_FIT_DRAWS,
    )
    return {
        "completed_size_dist_train": by_cell,
        "completed_size_dist_train_all": overall,
        "retention_link_shift": shift,
        "delta3_fit": diag,
        "_meta": diag,
    }


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _params(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _deep_freeze(value) for key, value in values.items()}
    )


_EMPTY = _params()
_COHAB_PARAMS = _params(
    band_rates=True,
    single_year=[15, 34],
    female_single_year=[25, 44],
    overlay_lift={"cell": "25-34|female", "probability": 0.045},
)
_LEGAL_PARAMS = _params(core_seed=5200, mixing_exit_floor=0.5)
_PARENTAL_PARAMS = _params(knots=[16, 19, 23, 30, 45], child_refit=[18, 30])
_MULTIGEN_PARAMS = _params(coupling_age_lo=55)
_CHILD_PARAMS = _params(
    child_age_bands=[[0, 4], [5, 12], [13, 17], [18, 24], [25, 60]],
    persistence_fit_seed=0xC7,
    persistence_bisections=24,
)
_SKIPGEN_PARAMS = _params(
    fine_bands=[[55, 59], [60, 64], [65, 69], [70, 74], [75, 79], [80, 120]]
)
_NONFAMILY_PARAMS = _params(core_size_cap=5, full_count_conditional=True)
_HOUSEHOLD_SIZE_PARAMS = _params(parent_count_support=[1, 2])
_FERTILITY_PARAMS = _params(
    size_buckets=["0", "1", "2", "3", "4+"],
    scope_cells=[
        "55-64|male",
        "65-74|male",
        "45-54|female",
        "65-74|female",
    ],
    n_retention_fit_draws=6,
)


REGISTRY = ComponentRegistry(
    (
        ComponentDefinition(
            "marital_core_adapter",
            "family_transitions.candidate16_adapter.v1",
            _fit_marital,
            "household_composition_sim.py:315-357,612-642",
            _EMPTY,
        ),
        ComponentDefinition(
            "cohabitation_overlay",
            "code22_band_single_year_female_lift.v1",
            _fit_cohabitation,
            "household_composition_sim_v2.py:116-206;v4.py:216-260;v6.py:307-363;v8.py:971-1011",
            _COHAB_PARAMS,
        ),
        ComponentDefinition(
            "legal_spouse_residual",
            "code20_residual_stationary_topup.v1",
            _fit_legal,
            "household_composition_sim_v4.py:273-410",
            _LEGAL_PARAMS,
        ),
        ComponentDefinition(
            "parental_home_exit",
            "rcs_exit_plus_adult_child_single_year.v1",
            _fit_parental,
            "household_composition_sim.py:112-216;v6.py:251-301,533-578",
            _PARENTAL_PARAMS,
        ),
        ComponentDefinition(
            "multigenerational_occupancy",
            "band_hazards_plus_55_child_coupling.v1",
            _fit_multigen,
            "household_composition_sim.py:227-272;v5.py:196-258",
            _MULTIGEN_PARAMS,
        ),
        ComponentDefinition(
            "child_attribution",
            "maternal_shadow_custodial_enumerated_frailty.v1",
            _fit_children,
            "household_composition_sim_v2.py:212-236;v3.py:177-338;v4.py:416-534;v5.py:264-376;v6.py:510-527;v7.py:223-791",
            _CHILD_PARAMS,
        ),
        ComponentDefinition(
            "skip_generation_state",
            "band_plus_55_five_year_hazards.v1",
            _fit_skipgen,
            "household_composition_sim_v3.py:395-458;v4.py:614-651",
            _SKIPGEN_PARAMS,
        ),
        ComponentDefinition(
            "nonfamily_bridge",
            "full_count_given_capped_core.v1",
            _fit_nonfamily,
            "household_composition_sim_v3.py:224-251;v6.py:369-424,655-681",
            _NONFAMILY_PARAMS,
        ),
        ComponentDefinition(
            "household_size",
            "composed_core_parent_count_mixture.v1",
            _fit_household_size,
            "household_composition_sim.py:82-109;v5.py:456-521",
            _HOUSEHOLD_SIZE_PARAMS,
        ),
        ComponentDefinition(
            "fertility_core_lift",
            "cohort_scoped_completed_size_retention_link.v1",
            _fit_fertility,
            "household_composition_sim_v8.py:150-968;v9.py:151-256",
            _FERTILITY_PARAMS,
        ),
    )
)


CANDIDATE_9 = CandidateSpec(
    candidate_id="gate2b_candidate9_registry_v1",
    contract_revision="gate_2b_locked",
    components=tuple(
        ComponentRef(
            definition.kind,
            definition.implementation_id,
            definition.params,
        )
        for definition in REGISTRY._definitions.values()
    ),
)
