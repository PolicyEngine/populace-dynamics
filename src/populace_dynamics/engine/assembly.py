"""Assembly of cutoff-refitted objects into the ordered projection engine.

The certified registries operate on their native panel types, while M6's
public loop carries one canonical person-year slice.  This module is the
explicit alignment layer between those surfaces: callers supply pure builders
from a slice to the native marital and household panels, and the assembly
injects fitted objects, period streams, step-3 state, step-4 births, and M4's
realized-support reproduction result.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import pandas as pd

from populace_dynamics.data import disability as disability_data
from populace_dynamics.data import household_composition as hc_data
from populace_dynamics.data import transitions
from populace_dynamics.engine.composition import (
    composition_rngs_from_registry,
    simulate_candidate9_injected,
)
from populace_dynamics.engine.disability import simulate_reproduction
from populace_dynamics.engine.earnings_domain import wrap_earnings_domain
from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodContext,
    PeriodModules,
)
from populace_dynamics.engine.marital import simulate_marital_step
from populace_dynamics.engine.refit import M6RefitBundle
from populace_dynamics.engine.rng import ProjectionModule
from populace_dynamics.engine.steps import (
    AgeSexMortalityModel,
    ClaimingSchedule,
    EarningsGenerator,
    FertilityDraws,
    advance_age,
    apply_claiming,
    apply_earnings,
    apply_fertility,
    apply_mortality,
    simulate_fertility,
)
from populace_dynamics.engine.support import StartWaveWeightSnapshot
from populace_dynamics.models.disability_hazard_sim import (
    DisabilityHazardModel,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)
from populace_dynamics.models.household_composition.components.marital_core_adapter import (
    fit_male_gap,
)
from populace_dynamics.models.household_composition.fitted import (
    FittedHouseholdComposition,
)


class MaritalPanelBuilder(Protocol):
    """Build one period's native C16 panel and simulation population."""

    def __call__(
        self, frame: pd.DataFrame, context: PeriodContext
    ) -> tuple[transitions.MaritalPanel, set[int]]: ...


class HouseholdPanelBuilder(Protocol):
    """Build candidate-9's native support for one reconciliation period."""

    def __call__(
        self, frame: pd.DataFrame, context: PeriodContext
    ) -> tuple[hc_data.HouseholdCompositionPanel, set[int]]: ...


@dataclass(frozen=True)
class CertifiedEngineInputs:
    """All fitted objects and native-panel seams required by M6."""

    family: FittedFamilyTransitions
    modifier: object
    permanent_axis: object
    household: FittedHouseholdComposition
    mortality: AgeSexMortalityModel
    disability: DisabilityHazardModel
    claiming: ClaimingSchedule
    earnings: EarningsGenerator
    marital_panel_builder: MaritalPanelBuilder
    household_panel_builder: HouseholdPanelBuilder
    disability_panel: disability_data.DisabilityPanel
    disability_ids: set[int]
    start_weights: StartWaveWeightSnapshot
    male_gap: float

    @classmethod
    def from_refit_bundle(
        cls,
        bundle: M6RefitBundle,
        *,
        mortality: AgeSexMortalityModel,
        earnings: EarningsGenerator | None = None,
        marital_panel_builder: MaritalPanelBuilder,
        household_panel_builder: HouseholdPanelBuilder,
        disability_panel: disability_data.DisabilityPanel,
        disability_ids: set[int],
        start_weights: StartWaveWeightSnapshot,
    ) -> CertifiedEngineInputs:
        """Bind a complete, non-serialized cutoff-refit bundle."""
        required = {
            "family": bundle.family,
            "household": bundle.household,
            "modifier": bundle.modifier,
            "disability": bundle.disability,
            "claiming_pmfs": bundle.claiming_pmfs,
            "earnings": bundle.earnings if earnings is None else earnings,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"refit bundle is incomplete: {missing}")
        assert bundle.family is not None
        assert bundle.household is not None
        assert bundle.modifier is not None
        assert bundle.claiming_pmfs is not None
        default_earnings = (
            bundle.earnings.generator if bundle.earnings is not None else None
        )
        selected_earnings = (
            earnings if earnings is not None else default_earnings
        )
        assert selected_earnings is not None
        # Candidate 9 retains its independently refitted embedded C16 object
        # in ``bundle`` for the injected-vs-internal pre-flight.  The scored
        # engine binds an immutable local copy to the authoritative step-3 C16
        # fit, so downstream fertility and composition cannot silently fall
        # back to a second marital law.
        household_fit = replace(
            bundle.household.fitted,
            family_transitions=bundle.family.fitted,
            male_gap=fit_male_gap(bundle.family.fitted),
        )
        return cls(
            family=bundle.family.fitted,
            modifier=bundle.modifier.modifier,
            permanent_axis=bundle.modifier.axis,
            household=household_fit,
            mortality=mortality,
            disability=bundle.disability,
            claiming=ClaimingSchedule(bundle.claiming_pmfs),
            earnings=selected_earnings,
            marital_panel_builder=marital_panel_builder,
            household_panel_builder=household_panel_builder,
            disability_panel=disability_panel,
            disability_ids=set(disability_ids),
            start_weights=start_weights,
            male_gap=float(household_fit.male_gap),
        )


def _fit_digest(value: object) -> str:
    try:
        payload = pickle.dumps(value, protocol=5)
    except (pickle.PickleError, TypeError) as error:
        raise ValueError(
            "fitted marital core is not byte-comparable"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _merge_period_columns(
    frame: pd.DataFrame,
    updates: pd.DataFrame,
    *,
    year: int,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    current = updates[updates["year"] == year]
    if current.empty:
        return frame.copy()
    current = current[["person_id", *columns]]
    if current["person_id"].duplicated().any():
        raise ValueError("period update has duplicate person rows")
    out = frame.drop(
        columns=[c for c in columns if c in frame], errors="ignore"
    )
    return out.merge(
        current, on="person_id", how="left", validate="one_to_one"
    )


@dataclass
class _AssemblyState:
    draw_index: int | None = None
    marital_projection: MaritalStepResult | None = None
    marital_ids: set[int] = field(default_factory=set)
    fertility: dict[int, FertilityDraws] = field(default_factory=dict)
    roster_absent_births: dict[int, dict[str, object]] = field(
        default_factory=dict
    )
    disability_projection: disability_data.DisabilityPanel | None = None
    household_projection: hc_data.HouseholdCompositionPanel | None = None

    def reset(self, draw_index: int) -> None:
        """Start one projection invocation with draw-local cached cores."""
        self.draw_index = int(draw_index)
        self.marital_projection = None
        self.marital_ids.clear()
        self.fertility.clear()
        self.roster_absent_births.clear()
        self.disability_projection = None
        self.household_projection = None


M6_DRAW_OUTPUTS_KEY = "m6_draw_outputs"


def _publish_draw_output(
    context: PeriodContext, name: str, value: object
) -> None:
    collector = context.metadata.get(M6_DRAW_OUTPUTS_KEY)
    if collector is None:
        return
    if not isinstance(collector, dict):
        raise TypeError(
            f"context metadata {M6_DRAW_OUTPUTS_KEY!r} must be a dict"
        )
    collector[name] = value


def assemble_period_modules(inputs: CertifiedEngineInputs) -> PeriodModules:
    """Construct the real fitted-object adapters consumed by ProjectionEngine."""
    embedded = inputs.household.family_transitions
    if _fit_digest(inputs.family) != _fit_digest(embedded):
        raise ValueError(
            "candidate-9 embedded marital core is not byte-identical to "
            "the authoritative candidate-16 fit"
        )
    state = _AssemblyState()
    earnings = wrap_earnings_domain(inputs.earnings)

    def ensure_draw(context: PeriodContext, *, start: bool = False) -> None:
        # Mortality is the first ordered step, so ``start`` also distinguishes
        # a repeated invocation at the same draw index.  The other adapters
        # keep the defensive draw-index check for focused direct-call tests.
        if start or state.draw_index != context.draw_index:
            state.reset(context.draw_index)

    def initialize(frame):
        materialize = getattr(earnings, "materialize_initial_frame", None)
        return materialize(frame) if materialize is not None else frame.copy()

    def mortality_step(frame, context, rng):
        ensure_draw(context, start=context.period_index == 1)
        survivors = apply_mortality(
            frame, context, rng, model=inputs.mortality
        )
        collector = context.metadata.get(M6_DRAW_OUTPUTS_KEY)
        if collector is not None:
            if not isinstance(collector, dict):
                raise TypeError(
                    f"context metadata {M6_DRAW_OUTPUTS_KEY!r} must be a dict"
                )
            survivor_ids = set(survivors["person_id"])
            mortality = frame[["person_id", "age", "sex", "weight"]].copy()
            mortality["cal_year"] = context.year
            mortality["death"] = ~mortality["person_id"].isin(survivor_ids)
            mortality["exposure"] = np.where(mortality["death"], 0.5, 1.0)
            collector.setdefault("mortality_slices", []).append(mortality)
        return survivors

    def marital_step(frame, context, rng):
        del rng
        ensure_draw(context)
        if context.rng_registry is None:
            raise RuntimeError(
                "certified assembly requires the M6 RNG registry"
            )
        if state.marital_projection is None:
            panel, ids = inputs.marital_panel_builder(frame, context)
            state.marital_projection = simulate_marital_step(
                panel,
                ids,
                inputs.family,
                inputs.modifier,
                inputs.permanent_axis,
                main_rng=context.rng_registry.generator(
                    0, ProjectionModule.MARITAL_CORE
                ),
                gap_rng=context.rng_registry.child_generator(
                    0, ProjectionModule.MARITAL_CORE, 1
                ),
            )
            state.marital_ids = set(ids)
            _publish_draw_output(context, "marital", state.marital_projection)
        return state.marital_projection

    def fertility_step(frame, context, marital, rng):
        ensure_draw(context)
        marital_columns = tuple(
            column
            for column in (
                "marital_state",
                "marriage_order",
                "marriage_duration",
                "years_since_dissolution",
            )
            if column in marital.sim_years.columns
        )
        frame = _merge_period_columns(
            frame,
            marital.sim_years,
            year=context.year,
            columns=marital_columns,
        )
        reconciled = apply_fertility(
            frame,
            context,
            marital,
            rng,
            components=inputs.family,
            holdout_ids=state.marital_ids,
            male_gap=inputs.male_gap,
            birth_store=state.fertility,
            roster_absent_births=state.roster_absent_births,
        )
        _publish_draw_output(
            context,
            "roster_absent_births",
            dict(state.roster_absent_births),
        )
        return reconciled

    def disability_step(frame, context, rng):
        del rng
        ensure_draw(context)
        if state.disability_projection is None:
            if context.rng_registry is None:
                raise RuntimeError(
                    "certified assembly requires the M6 RNG registry"
                )
            start_year = context.year - context.period_index
            end_year = start_year + context.rng_registry.n_periods
            support_rows = inputs.disability_panel.person_years[
                inputs.disability_panel.person_years["period"].between(
                    start_year + 1, end_year
                )
            ].copy()
            support_panel = disability_data.DisabilityPanel(
                person_years=support_rows,
                pairs=disability_data.build_transition_pairs(support_rows),
            )
            periods = sorted(
                int(value)
                for value in support_rows["period"].unique()
                if int(value) > start_year
            )
            by_period = {}
            for period in periods:
                period_index = period - start_year
                if period_index > context.rng_registry.n_periods:
                    continue
                by_period[period] = context.rng_registry.generator(
                    period_index, ProjectionModule.DISABILITY
                )
            state.disability_projection = simulate_reproduction(
                support_panel,
                inputs.disability,
                inputs.disability_ids,
                None,
                start_weights=inputs.start_weights,
                rng_by_period=by_period,
            )
            _publish_draw_output(
                context, "disability", state.disability_projection
            )
        updates = state.disability_projection.person_years.rename(
            columns={"period": "year"}
        )
        return _merge_period_columns(
            frame,
            updates,
            year=context.year,
            columns=("disabled", "retired", "status_code", "di_converted"),
        )

    def earnings_step(frame, context, rng):
        return apply_earnings(frame, context, rng, model=earnings)

    def claiming_step(frame, context, rng):
        return apply_claiming(frame, context, rng, schedule=inputs.claiming)

    def household_step(frame, context, marital, rng):
        del rng
        ensure_draw(context)
        if context.rng_registry is None:
            raise RuntimeError(
                "certified assembly requires the M6 RNG registry"
            )
        if state.household_projection is None:
            panel, ids = inputs.household_panel_builder(frame, context)
            household_fertility = simulate_fertility(
                marital,
                inputs.family,
                state.marital_ids,
                inputs.male_gap,
                context.rng_registry.generator(0, ProjectionModule.FERTILITY),
            )
            simulated, diagnostics = simulate_candidate9_injected(
                panel,
                inputs.household,
                ids,
                marital,
                composition_rngs_from_registry(context.rng_registry, 0),
                fertility=household_fertility,
            )
            state.household_projection = simulated
            _publish_draw_output(context, "household", simulated)
            _publish_draw_output(context, "household_diagnostics", diagnostics)
        return _merge_period_columns(
            frame,
            state.household_projection.person_waves,
            year=context.year,
            columns=(
                "coresident_spouse",
                "coresident_parent",
                "coresident_child",
                "coresident_grandchild",
                "multigen",
                "hh_size",
            ),
        )

    return PeriodModules(
        mortality=mortality_step,
        aging=advance_age,
        marital_core=marital_step,
        fertility=fertility_step,
        disability=disability_step,
        earnings=earnings_step,
        claiming=claiming_step,
        household_composition=household_step,
        initialize=initialize,
    )
