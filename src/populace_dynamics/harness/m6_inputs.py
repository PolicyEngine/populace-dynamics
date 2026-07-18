"""Production input construction for the M6 scored-run harness.

The refit layer is intentionally reader-free.  This module is its application
boundary: :func:`load_m6_raw_inputs` invokes the certified PSID readers, while
:func:`assemble_m6_inputs` is a pure transform over caller-owned frames.  Tests
therefore exercise the complete wiring without contacting staged PSID data.

External SSA, claiming, and mortality references are never selected here.
They must be supplied by the caller with explicit vintages no later than the
2014 temporal boundary.  The full PSID panels are retained for truth/support
construction, but every training identifier and modifier weight is derived
from observations dated no later than that boundary; the engine refit entry
point performs the field-aware row truncation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics import claiming
from populace_dynamics.data import deaths, disability, family, transitions
from populace_dynamics.data.household_composition import (
    HouseholdCompositionPanel,
)
from populace_dynamics.engine.panel_builders import PanelBuilderInputs
from populace_dynamics.engine.refit import (
    BOUNDARY_YEAR,
    M6RefitInputs,
    claiming_pmfs_from_reference,
    prepare_mortality_refit_inputs,
    validate_external_vintage,
)
from populace_dynamics.harness.m6_cells import (
    build_anchor_frame,
    disability_pairs,
    earnings_frame,
    marital_tables_from_panel,
    mortality_slices,
    presence_by_wave,
)
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition as hc
from populace_dynamics.models.household_composition import data as hc_data
from populace_dynamics.models.household_composition.components.cohabitation_overlay import (
    cohabitation_flag,
)

DEFAULT_EARNINGS_SEED = 5200

_HOUSEHOLD_SOURCE_KEYS = frozenset(
    {
        "hh",
        "mpanel",
        "demo",
        "mh",
        "bh",
        "rel_map",
        "order_map",
        "father_links_child",
        "parent_pairs",
        "marital_by_year",
        "fu_sizes",
        "legal_flag",
        "parent_counts",
        "child_record_expo",
    }
)
_MORTALITY_RATE_COLUMNS = frozenset(
    {
        "lower_age",
        "upper_age",
        "age_band",
        "sex",
        "central_rate",
    }
)

__all__ = [
    "DEFAULT_EARNINGS_SEED",
    "M6HarnessInputs",
    "M6InputProvenance",
    "M6RawInputs",
    "M6TruthTables",
    "assemble_m6_fit_inputs",
    "assemble_m6_inputs",
    "load_m6_inputs",
    "load_m6_raw_inputs",
]


@dataclass(frozen=True)
class M6RawInputs:
    """Caller-supplied reader outputs and explicitly-vintaged references."""

    household_sources: Mapping[str, Any]
    death_records: pd.DataFrame
    earnings_panel: pd.DataFrame
    disability_status: pd.DataFrame
    ssa_params: Any
    ssa_params_vintage: int
    claiming_reference: claiming.ClaimAgeReference
    mortality_exposure: pd.DataFrame
    mortality_external_rates: pd.DataFrame
    mortality_external_vintage: int


@dataclass(frozen=True)
class M6TruthTables:
    """Truth/support tables built once through the shared floor machinery."""

    anchor: pd.DataFrame
    presence: dict[int, set[int]]
    mortality: pd.DataFrame
    marital_events: pd.DataFrame
    marital_person_years: pd.DataFrame
    disability_pairs: pd.DataFrame
    earnings: pd.DataFrame


@dataclass(frozen=True)
class M6InputProvenance:
    """JSON-ready evidence for the M6 input and leakage boundary."""

    boundary_year: int
    earnings_seed: int
    readers: Mapping[str, str]
    external_vintages: Mapping[str, int]
    external_details: Mapping[str, str]
    source_rows: Mapping[str, int]
    source_max_year: Mapping[str, int | None]
    truth_rows: Mapping[str, int]
    training_population: Mapping[str, int]
    certified_full_window_artifacts_read: bool = False
    certified_full_window_artifacts_written: bool = False

    def to_artifact(self) -> dict[str, Any]:
        """Return a plain JSON-serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class M6HarnessInputs:
    """Complete caller-owned input bundle consumed by the scored runner."""

    refit_inputs: M6RefitInputs
    panel_builder_inputs: PanelBuilderInputs
    truth: M6TruthTables
    demographic_panel: pd.DataFrame
    earnings_panel: pd.DataFrame
    disability_status: pd.DataFrame
    disability_panel: disability.DisabilityPanel
    death_records: pd.DataFrame
    provenance: M6InputProvenance


@dataclass(frozen=True)
class _PreparedRefitInputs:
    """Shared fit state, before ``M6TruthTables`` are derived."""

    refit_inputs: M6RefitInputs
    family_train_ids: frozenset[int]
    household_train_ids: frozenset[int]
    disability_train_ids: frozenset[int]


def _require_columns(
    frame: pd.DataFrame, columns: set[str] | frozenset[str], label: str
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")


def _max_year(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame or frame.empty:
        return None
    years = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(years.max()) if len(years) else None


def _validate_boundary(boundary_year: int) -> int:
    boundary = int(boundary_year)
    if boundary != BOUNDARY_YEAR:
        raise ValueError(
            f"the M6 scored harness requires boundary_year={BOUNDARY_YEAR}"
        )
    return boundary


def _validate_household_sources(sources: Mapping[str, Any]) -> None:
    missing = _HOUSEHOLD_SOURCE_KEYS - set(sources)
    if missing:
        raise ValueError(
            "household source bundle is missing keys " f"{sorted(missing)}"
        )
    if not isinstance(sources["mpanel"], transitions.MaritalPanel):
        raise TypeError("household_sources['mpanel'] must be a MaritalPanel")
    if not isinstance(sources["hh"], HouseholdCompositionPanel):
        raise TypeError(
            "household_sources['hh'] must be a HouseholdCompositionPanel"
        )


def _validate_external_inputs(raw: M6RawInputs, boundary_year: int) -> None:
    validate_external_vintage(
        "SSA earnings-index parameters",
        raw.ssa_params_vintage,
        boundary_year=boundary_year,
    )
    nawi = getattr(raw.ssa_params, "nawi", None)
    if not isinstance(nawi, Mapping):
        raise TypeError("ssa_params must expose a year-keyed nawi mapping")
    earnings_years = pd.to_numeric(
        raw.earnings_panel.get("period", pd.Series(dtype="float64")),
        errors="coerce",
    ).dropna()
    required_nawi = {
        int(year) for year in earnings_years if int(year) <= boundary_year
    } | set(range(boundary_year - 9, boundary_year + 1))
    missing_nawi = sorted(required_nawi - {int(year) for year in nawi})
    if missing_nawi:
        raise ValueError(
            "ssa_params.nawi is missing pre-boundary years "
            f"{missing_nawi[:10]}"
        )
    invalid_nawi = [
        year
        for year in required_nawi
        if not np.isfinite(float(nawi[year])) or float(nawi[year]) <= 0
    ]
    if invalid_nawi:
        raise ValueError(
            "ssa_params.nawi must be positive and finite for years "
            f"{sorted(invalid_nawi)[:10]}"
        )

    # Materializing the PMFs here verifies both the vintage and that the
    # caller's reference contains a usable pre-boundary entitlement year.
    claiming_pmfs_from_reference(
        raw.claiming_reference, boundary_year=boundary_year
    )

    _require_columns(
        raw.mortality_external_rates,
        _MORTALITY_RATE_COLUMNS,
        "mortality external rates",
    )
    # This validates the declared vintage and the conservative event/interview
    # dating rule.  The raw exposure remains in M6RefitInputs, whose refit path
    # repeats the same truncation immediately before fitting.
    prepare_mortality_refit_inputs(
        raw.mortality_exposure,
        raw.mortality_external_rates,
        external_vintage_year=raw.mortality_external_vintage,
        boundary_year=boundary_year,
    )


def _preboundary_weights(
    demographic_panel: pd.DataFrame, boundary_year: int
) -> pd.Series:
    _require_columns(
        demographic_panel,
        frozenset({"person_id", "period", "weight"}),
        "demographic panel",
    )
    period = pd.to_numeric(demographic_panel["period"], errors="coerce")
    weight = pd.to_numeric(demographic_panel["weight"], errors="coerce")
    admissible = demographic_panel[(period <= boundary_year) & (weight > 0)]
    if admissible.empty:
        raise ValueError("demographic panel has no positive-weight <=T* rows")
    return (
        admissible.sort_values(["person_id", "period"])
        .groupby("person_id", sort=False)
        .tail(1)
        .set_index("person_id")["weight"]
        .astype("float64")
    )


def _family_context(
    sources: Mapping[str, Any], train_ids: frozenset[int]
) -> ft.FitContext:
    return ft.FitContext(
        panel=sources["mpanel"],
        demographic_panel=sources["demo"],
        marriage_records=sources["mh"],
        birth_records=sources["bh"],
        marriage_order_map=sources["order_map"],
        train_ids=train_ids,
    )


def _prepare_refit_inputs(
    raw: M6RawInputs,
    *,
    sources: Mapping[str, Any],
    disability_panel: disability.DisabilityPanel,
    boundary_year: int,
    earnings_seed: int,
) -> _PreparedRefitInputs:
    """Build only the inputs consumed by ``refit_m6_components``.

    This helper deliberately has no anchor, presence, panel-builder, or truth
    argument.  Keeping the fit state separate lets a registered runner execute
    a designed fit preflight before it derives ``M6TruthTables`` or constructs
    the complete ``M6HarnessInputs`` bundle.  Its inputs are caller-owned raw
    source objects; this helper makes no claim that acquiring those objects was
    reader-free.
    """
    preboundary_demo = sources["demo"]
    preboundary_demo = preboundary_demo[
        (preboundary_demo["period"] <= boundary_year)
        & (preboundary_demo["weight"] > 0)
    ]
    preboundary_demo_ids = {
        int(value) for value in preboundary_demo["person_id"].unique()
    }
    family_train_ids = frozenset(
        preboundary_demo_ids
        & {
            int(value)
            for value in sources["mpanel"].attrs["person_id"].unique()
        }
    )
    household_train_ids = frozenset(
        int(value)
        for value in sources["hh"]
        .person_waves.loc[
            sources["hh"].person_waves["year"] <= boundary_year,
            "person_id",
        ]
        .unique()
    )
    disability_train_ids = frozenset(
        int(value)
        for value in disability_panel.person_years.loc[
            disability_panel.person_years["period"] <= boundary_year,
            "person_id",
        ].unique()
    )
    family_context = _family_context(sources, family_train_ids)
    household_context = hc.fit_context_from_sources(
        sources, set(household_train_ids)
    )
    modifier_weights = _preboundary_weights(sources["demo"], boundary_year)
    refit_inputs = M6RefitInputs(
        family_context=family_context,
        household_context=household_context,
        earnings_panel=raw.earnings_panel,
        earnings_seed=earnings_seed,
        modifier_marital_panel=sources["mpanel"],
        modifier_interview_years=sources["demo"]["period"],
        modifier_marriage_records=sources["mh"],
        modifier_person_weight=modifier_weights,
        ssa_params=raw.ssa_params,
        ssa_params_vintage=int(raw.ssa_params_vintage),
        modifier_train_ids=set(family_train_ids),
        disability_panel=disability_panel,
        disability_train_ids=set(disability_train_ids),
        claiming_reference=raw.claiming_reference,
        mortality_exposure=raw.mortality_exposure,
        mortality_external_rates=raw.mortality_external_rates,
        mortality_external_vintage=int(raw.mortality_external_vintage),
    )
    return _PreparedRefitInputs(
        refit_inputs=refit_inputs,
        family_train_ids=family_train_ids,
        household_train_ids=household_train_ids,
        disability_train_ids=disability_train_ids,
    )


def _source_provenance(
    *,
    raw: M6RawInputs,
    disability_panel: disability.DisabilityPanel,
    truth: M6TruthTables,
    boundary_year: int,
    earnings_seed: int,
    family_train_ids: frozenset[int],
    household_train_ids: frozenset[int],
    disability_train_ids: frozenset[int],
) -> M6InputProvenance:
    sources = raw.household_sources
    ssa_revision = str(getattr(raw.ssa_params, "pe_us_revision", "unknown"))
    return M6InputProvenance(
        boundary_year=boundary_year,
        earnings_seed=earnings_seed,
        readers={
            "household_sources": (
                "populace_dynamics.models.household_composition.data."
                "load_sources"
            ),
            "death_records": (
                "populace_dynamics.data.deaths.read_death_records"
            ),
            "earnings_panel": (
                "populace_dynamics.data.family.family_earnings_panel"
            ),
            "disability_status": (
                "populace_dynamics.data.disability." "read_disability_status"
            ),
            "external_inputs": "caller_supplied",
        },
        external_vintages={
            "ssa_parameters": int(raw.ssa_params_vintage),
            "claiming_reference": int(raw.claiming_reference.supplement_year),
            "mortality_reference": int(raw.mortality_external_vintage),
        },
        external_details={
            "ssa_revision": ssa_revision,
            "claiming_schema": raw.claiming_reference.schema_version,
            "claiming_table": raw.claiming_reference.table,
            "mortality_reference": "caller_supplied_external_rates",
        },
        source_rows={
            "demographic": len(sources["demo"]),
            "marriage_records": len(sources["mh"]),
            "birth_records": len(sources["bh"]),
            "relationship_map": len(sources["rel_map"]),
            "household_person_waves": len(sources["hh"].person_waves),
            "marital_person_years": len(sources["mpanel"].person_years),
            "marital_events": len(sources["mpanel"].events),
            "death_records": len(raw.death_records),
            "earnings": len(raw.earnings_panel),
            "disability_status": len(raw.disability_status),
            "disability_person_years": len(disability_panel.person_years),
            "disability_pairs": len(disability_panel.pairs),
            "mortality_exposure": len(raw.mortality_exposure),
            "mortality_external_rates": len(raw.mortality_external_rates),
        },
        source_max_year={
            "demographic": _max_year(sources["demo"], "period"),
            "marriage_records": _max_year(sources["mh"], "start_year"),
            "birth_records": _max_year(sources["bh"], "birth_year"),
            "relationship_map": _max_year(
                sources["rel_map"], "interview_year"
            ),
            "household_person_waves": _max_year(
                sources["hh"].person_waves, "year"
            ),
            "earnings": _max_year(raw.earnings_panel, "period"),
            "disability_status": _max_year(raw.disability_status, "period"),
            "mortality_exposure": _max_year(
                raw.mortality_exposure, "event_year"
            ),
        },
        truth_rows={
            "anchor": len(truth.anchor),
            "mortality": len(truth.mortality),
            "marital_events": len(truth.marital_events),
            "marital_person_years": len(truth.marital_person_years),
            "disability_pairs": len(truth.disability_pairs),
            "earnings": len(truth.earnings),
        },
        training_population={
            "family": len(family_train_ids),
            "household": len(household_train_ids),
            "modifier": len(family_train_ids),
            "disability": len(disability_train_ids),
        },
    )


def assemble_m6_fit_inputs(
    raw: M6RawInputs,
    *,
    boundary_year: int = BOUNDARY_YEAR,
    earnings_seed: int = DEFAULT_EARNINGS_SEED,
) -> M6RefitInputs:
    """Assemble fit state without deriving the M6 holdout truth tables.

    The returned object is sufficient for ``refit_m6_components``.  In
    particular, this path does not build the M6 anchor, realized-population
    inputs, presence map, or any mortality, marital, disability, or earnings
    ``M6TruthTables`` member.  A registered candidate can therefore fit and run
    a designed preflight before lazily constructing ``M6HarnessInputs``.

    ``raw`` is deliberately caller-owned and may already contain full-window
    source frames produced by external readers.  Deferring or restricting those
    reads belongs to the registered candidate's input factory, not this pure
    assembly seam.
    """
    boundary = _validate_boundary(boundary_year)
    seed = int(earnings_seed)
    if seed < 0:
        raise ValueError("earnings_seed must be non-negative")
    _validate_household_sources(raw.household_sources)
    sources = raw.household_sources
    _require_columns(
        sources["demo"],
        frozenset({"person_id", "period", "age", "weight", "interview"}),
        "demographic panel",
    )
    _require_columns(
        raw.death_records,
        frozenset({"person_id", "sex", "death_year"}),
        "death records",
    )
    _require_columns(
        raw.earnings_panel,
        frozenset({"person_id", "period", "earnings", "age", "weight"}),
        "earnings panel",
    )
    _validate_external_inputs(raw, boundary)
    disability_panel = disability.build_disability_panel(
        raw.disability_status, raw.death_records
    )
    return _prepare_refit_inputs(
        raw,
        sources=sources,
        disability_panel=disability_panel,
        boundary_year=boundary,
        earnings_seed=seed,
    ).refit_inputs


def assemble_m6_inputs(
    raw: M6RawInputs,
    *,
    boundary_year: int = BOUNDARY_YEAR,
    earnings_seed: int = DEFAULT_EARNINGS_SEED,
) -> M6HarnessInputs:
    """Purely assemble certified reader outputs into the M6 run inputs."""
    boundary = _validate_boundary(boundary_year)
    seed = int(earnings_seed)
    if seed < 0:
        raise ValueError("earnings_seed must be non-negative")
    _validate_household_sources(raw.household_sources)
    sources = raw.household_sources
    _require_columns(
        sources["demo"],
        frozenset({"person_id", "period", "age", "weight", "interview"}),
        "demographic panel",
    )
    _require_columns(
        raw.death_records,
        frozenset({"person_id", "sex", "death_year"}),
        "death records",
    )
    _require_columns(
        raw.earnings_panel,
        frozenset({"person_id", "period", "earnings", "age", "weight"}),
        "earnings panel",
    )
    _validate_external_inputs(raw, boundary)

    disability_panel = disability.build_disability_panel(
        raw.disability_status, raw.death_records
    )
    anchor = build_anchor_frame(sources["demo"])
    if anchor.empty:
        raise ValueError("demographic panel has no M6 holdout anchor rows")
    builders = PanelBuilderInputs.from_realized_histories(
        anchor=anchor,
        marriage_records=sources["mh"],
        death_records=raw.death_records,
        household=sources["hh"],
        cohabitation=cohabitation_flag(sources["rel_map"]),
    )

    present = presence_by_wave(sources["demo"])
    marital_events, marital_person_years = marital_tables_from_panel(
        builders.marital, anchor, present
    )
    truth = M6TruthTables(
        anchor=anchor,
        presence=present,
        mortality=mortality_slices(sources["demo"], raw.death_records, anchor),
        marital_events=marital_events,
        marital_person_years=marital_person_years,
        disability_pairs=disability_pairs(
            raw.disability_status, raw.death_records, anchor
        ),
        earnings=earnings_frame(raw.earnings_panel, anchor),
    )

    prepared = _prepare_refit_inputs(
        raw,
        sources=sources,
        disability_panel=disability_panel,
        boundary_year=boundary,
        earnings_seed=seed,
    )
    provenance = _source_provenance(
        raw=raw,
        disability_panel=disability_panel,
        truth=truth,
        boundary_year=boundary,
        earnings_seed=seed,
        family_train_ids=prepared.family_train_ids,
        household_train_ids=prepared.household_train_ids,
        disability_train_ids=prepared.disability_train_ids,
    )
    return M6HarnessInputs(
        refit_inputs=prepared.refit_inputs,
        panel_builder_inputs=builders,
        truth=truth,
        demographic_panel=sources["demo"],
        earnings_panel=raw.earnings_panel,
        disability_status=raw.disability_status,
        disability_panel=disability_panel,
        death_records=raw.death_records,
        provenance=provenance,
    )


def load_m6_raw_inputs(
    *,
    ssa_params: Any,
    ssa_params_vintage: int,
    claiming_reference: claiming.ClaimAgeReference,
    mortality_exposure: pd.DataFrame,
    mortality_external_rates: pd.DataFrame,
    mortality_external_vintage: int,
) -> M6RawInputs:
    """Read the certified PSID sources; select no external reference."""
    return M6RawInputs(
        household_sources=hc_data.load_sources(),
        death_records=deaths.read_death_records(),
        earnings_panel=family.family_earnings_panel(),
        disability_status=disability.read_disability_status(),
        ssa_params=ssa_params,
        ssa_params_vintage=int(ssa_params_vintage),
        claiming_reference=claiming_reference,
        mortality_exposure=mortality_exposure,
        mortality_external_rates=mortality_external_rates,
        mortality_external_vintage=int(mortality_external_vintage),
    )


def load_m6_inputs(
    *,
    ssa_params: Any,
    ssa_params_vintage: int,
    claiming_reference: claiming.ClaimAgeReference,
    mortality_exposure: pd.DataFrame,
    mortality_external_rates: pd.DataFrame,
    mortality_external_vintage: int,
    boundary_year: int = BOUNDARY_YEAR,
    earnings_seed: int = DEFAULT_EARNINGS_SEED,
) -> M6HarnessInputs:
    """Read certified sources and assemble the complete production bundle."""
    raw = load_m6_raw_inputs(
        ssa_params=ssa_params,
        ssa_params_vintage=ssa_params_vintage,
        claiming_reference=claiming_reference,
        mortality_exposure=mortality_exposure,
        mortality_external_rates=mortality_external_rates,
        mortality_external_vintage=mortality_external_vintage,
    )
    return assemble_m6_inputs(
        raw, boundary_year=boundary_year, earnings_seed=earnings_seed
    )
