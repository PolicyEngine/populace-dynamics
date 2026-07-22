"""Production input-plan factory for the registered M6 candidate-2 run.

Invoke the one-shot runner with::

    python scripts/run_gate_m6_candidate2.py \
        --registration-id REGISTRATION_8 \
        --input-factory registered_m6_candidate2_inputs:build_input_plan

Plan construction reads only fields dated no later than ``T*=2014`` for the
demographic, disability, and relationship panels. Earnings mirror the
ratified run-7 field-dating rule: the sole post-``T*`` collection wave is 2015,
whose labor fields measure reference-year 2014; its collection-wave age,
sequence, relationship, weight, and interview fields are the registered
family-panel covariates. The marriage, birth, and death files are
retrospective products; they are read only to establish the registered
``<=T*`` histories, immediately passed through the engine's field-aware
history truncation, and never used as holdout truth.

The full-window run-7 factory is reachable only through ``load_full_inputs``.
The candidate-2 runner invokes that callback after both registered fits and
the selected-C convergence preflight complete. It then carries the exact
preflighted ``M6RefitInputs`` object into the complete harness bundle.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import registered_m6_inputs

from populace_dynamics.data import (
    births,
    deaths,
    disability,
    family,
    household_composition,
    marriage,
    panels,
    relmap,
    transitions,
)
from populace_dynamics.engine import refit as refit_engine
from populace_dynamics.engine.refit import BOUNDARY_YEAR
from populace_dynamics.harness.m6_candidate2_runner import (
    M6Candidate2InputPlan,
)
from populace_dynamics.harness.m6_inputs import (
    DEFAULT_EARNINGS_SEED,
    M6HarnessInputs,
    M6RawInputs,
    assemble_m6_fit_inputs,
)
from populace_dynamics.models.household_composition import data as hc_data
from populace_dynamics.models.household_composition import (
    registry as hc_registry,
)
from populace_dynamics.models.household_composition.components.child_attribution import (
    build_child_record_exposure,
    father_link_births_with_child,
    father_marital_by_year,
    parent_child_coresidence_pairs,
)
from populace_dynamics.models.household_composition.components.household_size import (
    parent_link_counts,
)
from populace_dynamics.models.household_composition.components.legal_spouse_residual import (
    legal_spouse_flag,
)
from populace_dynamics.models.household_composition.components.nonfamily_bridge import (
    family_unit_sizes,
)

RELATIONSHIP_CHUNK_SIZE = 250_000
TRAIN_EARNINGS_WAVES = tuple(
    wave for wave in family.FAMILY_WAVES if wave - 1 <= BOUNDARY_YEAR
)


def _assert_bounded(frame: pd.DataFrame, column: str, label: str) -> None:
    """Reject a supposedly train-only frame containing a future row."""
    values = pd.to_numeric(frame[column], errors="coerce")
    future = values.notna() & (values > BOUNDARY_YEAR)
    if future.any():
        maximum = int(values[future].max())
        raise RuntimeError(
            f"{label} acquired post-{BOUNDARY_YEAR} truth (max={maximum})"
        )


def _censor_future_exact_deaths(records: pd.DataFrame) -> pd.DataFrame:
    """Hide exact deaths after ``T*`` while retaining the sex source.

    The 2023 individual file is a retrospective product and the registered
    fits need its person-constant sex field. An exact death later than the
    fitting boundary is not admissible history, so its ``death_year`` is
    masked before mortality exposure or marital state is constructed.
    """
    out = records.copy()
    years = pd.to_numeric(out["death_year"], errors="coerce")
    out.loc[years > BOUNDARY_YEAR, "death_year"] = pd.NA
    out["death_year"] = out["death_year"].astype("Int64")
    return out


def _load_train_household_sources(
    demographic: pd.DataFrame,
    death_records: pd.DataFrame,
    retrospective_death_records: pd.DataFrame,
) -> dict[str, Any]:
    """Build the exact cutoff context without scored future panels.

    Marriage, birth, and death histories are retrospective single products.
    The temporary native marital panel mirrors candidate 1's construction so
    the shared field-aware truncator sees the same history, then the context
    is reduced to ``<=T*`` before this function returns.
    """
    relationship_waves = tuple(
        sorted(int(value) for value in demographic["period"].unique())
    )
    relationship_map = relmap.relationship_map(
        waves=relationship_waves,
        chunksize=RELATIONSHIP_CHUNK_SIZE,
    )
    _assert_bounded(
        relationship_map,
        "interview_year",
        "relationship map",
    )

    roster = household_composition.household_roster(relationship_map)
    person_waves = household_composition.join_demographics(
        roster,
        demographic,
        death_records,
    )
    household_panel = household_composition.HouseholdCompositionPanel(
        person_waves=person_waves,
        attrs=(
            person_waves[["person_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        ),
    )

    marriage_records = marriage.marriage_history()
    birth_records = births.birth_history()
    positive_weight = demographic[demographic["weight"] > 0]
    person_weight = (
        positive_weight.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    marital_panel = transitions.build_marital_panel(
        marriage_records,
        retrospective_death_records,
        person_weight,
    )
    order_map = hc_data.marriage_order_map(marriage_records)

    father_links_child = father_link_births_with_child(birth_records)
    parent_pairs = parent_child_coresidence_pairs(relationship_map)
    marital_by_year = father_marital_by_year(marital_panel)
    family_sizes = family_unit_sizes(relationship_map)
    legal_flag = legal_spouse_flag(relationship_map)
    parent_counts = parent_link_counts(relationship_map)
    child_record_exposure = build_child_record_exposure(
        father_links_child,
        parent_pairs,
        marital_by_year,
        demographic,
        relationship_map,
    )
    sources = {
        "hh": household_panel,
        "mpanel": marital_panel,
        "demo": demographic,
        "mh": marriage_records,
        "bh": birth_records,
        "rel_map": relationship_map,
        "order_map": order_map,
        "father_links_child": father_links_child,
        "parent_pairs": parent_pairs,
        "marital_by_year": marital_by_year,
        "fu_sizes": family_sizes,
        "legal_flag": legal_flag,
        "parent_counts": parent_counts,
        "child_record_expo": child_record_exposure,
    }
    train_ids = {
        int(value) for value in household_panel.person_waves["person_id"]
    }
    context = hc_registry.fit_context_from_sources(sources, train_ids)
    bounded = refit_engine._truncate_household_context(
        context,
        BOUNDARY_YEAR,
    )
    return {
        "hh": bounded.hh,
        "mpanel": bounded.mpanel,
        "demo": bounded.demographic_panel,
        "mh": bounded.marriage_records,
        "bh": bounded.birth_records,
        "rel_map": bounded.relationship_map,
        "order_map": bounded.marriage_order_map,
        "father_links_child": bounded.father_links_child,
        "parent_pairs": bounded.parent_pairs,
        "marital_by_year": bounded.marital_by_year,
        "fu_sizes": bounded.family_unit_sizes,
        "legal_flag": bounded.legal_spouse_flag,
        "parent_counts": bounded.parent_counts,
        "child_record_expo": bounded.child_record_exposure,
    }


def load_train_only_raw_inputs() -> M6RawInputs:
    """Acquire the production fit sources without a holdout-truth read."""
    registered_m6_inputs.assert_pe_us_version()
    registered_m6_inputs.assert_pe_us_param_dir()
    params = registered_m6_inputs.boundary_ssa_parameters()
    claiming_reference = registered_m6_inputs.load_claiming_reference()

    demographic = panels.demographic_panel(max_period=BOUNDARY_YEAR)
    _assert_bounded(demographic, "period", "demographic panel")
    retrospective_death_records = deaths.read_death_records()
    death_records = _censor_future_exact_deaths(retrospective_death_records)
    household_sources = _load_train_household_sources(
        demographic,
        death_records,
        retrospective_death_records,
    )

    earnings_panel = family.family_earnings_panel(waves=TRAIN_EARNINGS_WAVES)
    _assert_bounded(earnings_panel, "period", "earnings panel")
    disability_status = disability.read_disability_status(
        max_period=BOUNDARY_YEAR
    )
    _assert_bounded(disability_status, "period", "disability status")

    mortality_external_rates, mortality_exposure = (
        registered_m6_inputs._pad_below_25_projection_coverage(
            registered_m6_inputs.nchs_2010_external_rates(),
            registered_m6_inputs.mortality_exposure_adapter(
                demographic,
                death_records,
            ),
            boundary_year=BOUNDARY_YEAR,
        )
    )
    _assert_bounded(
        mortality_exposure,
        "event_year",
        "mortality exposure",
    )
    _assert_bounded(
        mortality_exposure,
        "required_interview_year",
        "mortality exposure interview provenance",
    )
    return M6RawInputs(
        household_sources=household_sources,
        death_records=death_records,
        earnings_panel=earnings_panel,
        disability_status=disability_status,
        ssa_params=params,
        ssa_params_vintage=registered_m6_inputs.SSA_PARAMS_VINTAGE,
        claiming_reference=claiming_reference,
        mortality_exposure=mortality_exposure,
        mortality_external_rates=mortality_external_rates,
        mortality_external_vintage=(
            registered_m6_inputs.MORTALITY_EXTERNAL_VINTAGE
        ),
    )


def build_input_plan() -> M6Candidate2InputPlan:
    """Build fit-only state and defer every full-window reader."""
    raw = load_train_only_raw_inputs()
    fit_inputs = assemble_m6_fit_inputs(
        raw,
        boundary_year=BOUNDARY_YEAR,
        earnings_seed=DEFAULT_EARNINGS_SEED,
    )
    full_inputs: M6HarnessInputs | None = None

    def load_full_inputs() -> M6HarnessInputs:
        nonlocal full_inputs
        if full_inputs is None:
            assembled = registered_m6_inputs.build_inputs()
            full_inputs = replace(assembled, refit_inputs=fit_inputs)
        return full_inputs

    return M6Candidate2InputPlan(
        fit_inputs=fit_inputs,
        load_full_inputs=load_full_inputs,
    )
