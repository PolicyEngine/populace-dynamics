"""One-shot orchestration for the registered M6 temporal-holdout run.

This module is intentionally split into small phases.  The production defaults
bind the cutoff refit, realized-population builder, projection engine, shared
cell reductions, and one-shot artifact writer.  Tests can replace the phase
operations with synthetic seams; importing this module never reads PSID data,
runs a projection, or writes an artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from populace_dynamics import artifacts
from populace_dynamics.data import disability as disability_data
from populace_dynamics.data import transitions
from populace_dynamics.engine.assembly import (
    M6_DRAW_OUTPUTS_KEY,
    CertifiedEngineInputs,
    assemble_period_modules,
)
from populace_dynamics.engine.earnings_domain import (
    earnings_domain_person_ids as fitted_earnings_domain_person_ids,
)
from populace_dynamics.engine.loop import ProjectionEngine, ProjectionResult
from populace_dynamics.engine.panel_builders import (
    household_panel_builder,
    marital_panel_builder,
)
from populace_dynamics.engine.refit import (
    BOUNDARY_YEAR,
    EARNINGS_SPEC_REGISTRATION,
    EARNINGS_SPEC_SHA256,
    M6RefitBundle,
    fit_mortality_model,
    prepare_m6_preflight_context,
    refit_m6_components,
)
from populace_dynamics.engine.support import PresenceBasis
from populace_dynamics.harness.m6_cells import (
    DISABILITY_BANDS,
    GATED_FLOW_YEARS,
    MARITAL_BANDS,
    SEXES,
    SHOCK_EARN_YEARS,
    SHOCK_FLOW_YEARS,
    band_of,
    disability_cells,
    earnings_cells,
    marital_cells,
    mortality_cells,
)
from populace_dynamics.harness.m6_inputs import M6HarnessInputs
from populace_dynamics.harness.m6_population import (
    M6RealizedPopulation,
    build_realized_population,
    subset_realized_population,
)
from populace_dynamics.harness.m6_preflight import (
    Candidate9PreflightInputs,
    recertification_payload,
    run_candidate9_recertification,
    verify_external_sign_path,
)
from populace_dynamics.harness.m6_projection import (
    prepare_gated_realized_support,
    prepare_projected_disability,
    prepare_projected_marital,
    project_earnings_on_realized_support,
)
from populace_dynamics.harness.m6_reporting import (
    assemble_report_only_payload,
    build_alignment_displacement,
    build_entrant_diagnostics,
    build_mortality_anchor_disclosure,
    build_not_certified_surface,
    build_redrawn_seed_comparison,
    build_shock_window_diagnostics,
)
from populace_dynamics.harness.m6_scoring import (
    M6GateContract,
    M6GateScore,
    M6SeedScore,
    aggregate_gate,
    recompute_domain_earnings_floor,
    reduce_gated_cells,
    restrict_earnings_domain_support,
    score_gate_seed,
    side_a_person_ids,
)

SCHEMA_VERSION = "gate_m6_candidate1.v1"
CANDIDATE_NUMBER = 1
PROJECTION_END_YEAR = 2022
FROZEN_FLOOR_RUN = "runs/m6_holdout_floors_v3.json"
FROZEN_FLOOR_SHA256 = (
    "e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77"
)
DEFAULT_OUTPUT = Path("runs/gate_m6_candidate1_v1.json")
PHASE_ORDER = (
    "refit",
    "preflight_1",
    "preflight_2",
    "project_and_score",
    "report_only",
    "assemble_and_write",
)

# Superseded registrations, each publicly graded on issue #42 as unable to
# authorize a scored run: the pre-amendment design registration cited by
# section 2.8 (stop: no harness), the second registration (stop: no <= T*
# external-reference binding; graded 4967433717), and the third registration
# (harness and certified inputs both existed, but the run FAILED TO EXECUTE with
# two pre-scoring crashes -- the QRF-import env miss and the demographic-seed
# sex integration defect this patch fixes; graded 4972045579).  Useful lineage,
# but none can authorize a scored run.
_KNOWN_STALE_REGISTRATIONS = frozenset(
    {"4962640241", "4967241464", "4971244215"}
)
_REGISTRATION = re.compile(
    r"^(?:[0-9]{7,}|https://github\.com/[^/]+/[^/]+/issues/42"
    r"#issuecomment-[0-9]+)$"
)


@dataclass(frozen=True)
class M6ResolvedContract:
    """The allow-listed gate protocol and verified frozen floor."""

    contract: M6GateContract
    floor_artifact: Mapping[str, Any]
    floor_path: str
    floor_sha256: str


@dataclass(frozen=True)
class M6RefitPhase:
    """Objects created in phase 1 and reused without serialization."""

    bundle: M6RefitBundle | Any
    mortality: Any
    population: M6RealizedPopulation | Any
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class M6SeedRun:
    """One split seed's gate score plus non-gating draw diagnostics."""

    seed: int
    score: M6SeedScore | Any
    side_a_units: Mapping[str, int]
    draw_reports: tuple[Mapping[str, Any], ...] = ()
    truth_cells: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class M6RunnerOperations:
    """Injectable phase seams used by synthetic orchestration tests."""

    resolve_contract: Callable[[Path], M6ResolvedContract]
    refit: Callable[[M6HarnessInputs], M6RefitPhase]
    preflight_1: Callable[
        [M6HarnessInputs, M6RefitPhase, M6GateContract], Mapping[str, Any]
    ]
    preflight_2: Callable[
        [M6HarnessInputs, M6RefitPhase, M6GateContract], Mapping[str, Any]
    ]
    score_seed: Callable[
        [M6HarnessInputs, M6RefitPhase, M6GateContract, int], M6SeedRun
    ]
    aggregate: Callable[
        [M6GateContract, Sequence[M6SeedScore | Any]], M6GateScore | Any
    ]
    domain_floor: Callable[
        [M6HarnessInputs, M6RefitPhase, M6ResolvedContract], Mapping[str, Any]
    ]
    report_only: Callable[
        [
            M6HarnessInputs,
            M6RefitPhase,
            M6ResolvedContract,
            Sequence[M6SeedRun],
            Mapping[str, Any],
        ],
        Mapping[str, Any],
    ]
    write: Callable[[Path, Mapping[str, Any]], None]


def validate_registration_id(value: str) -> str:
    """Require a fresh issue-42 comment identifier with no baked-in default."""
    registration = str(value).strip()
    if not _REGISTRATION.fullmatch(registration):
        raise ValueError(
            "registration-id must be a fresh issue #42 comment id or its "
            "full GitHub issue-comment URL"
        )
    numeric = registration.rsplit("-", 1)[-1]
    if numeric in _KNOWN_STALE_REGISTRATIONS:
        raise ValueError(
            "registration-id is the earlier design registration; create a "
            "fresh issue #42 registration before the scored run"
        )
    return registration


def _sanitize_gate_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fields section 2.8.8 permits the scored harness to read."""
    raw_views = block.get("views")
    raw_scoring = block.get("scoring")
    if not isinstance(raw_views, Mapping) or not isinstance(
        raw_scoring, Mapping
    ):
        raise ValueError("gate_m6 has no machine-readable views/scoring")
    views: dict[str, Any] = {}
    for name, raw in raw_views.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"gate_m6 view {name!r} is not a mapping")
        derivations = raw.get("derivations")
        if not isinstance(derivations, Mapping):
            raise ValueError(f"gate_m6 view {name!r} has no derivations")
        views[str(name)] = {
            "family": raw.get("family"),
            "split_unit": raw.get("split_unit"),
            "quantity_type": raw.get("quantity_type"),
            "floor_run": raw.get("floor_run"),
            "tolerances": raw.get("tolerances"),
            "derivations": {
                "floor_run": derivations.get("floor_run"),
                "rules": derivations.get("rules"),
            },
        }
    return {
        "locked": block.get("locked"),
        "status": block.get("status"),
        "floor_run": block.get("floor_run"),
        "floor_run_sha256": block.get("floor_run_sha256"),
        "views": views,
        "scoring": {
            "gate_seeds": raw_scoring.get("gate_seeds"),
            "conjunction": raw_scoring.get("conjunction"),
            "mixed_k": raw_scoring.get("mixed_k"),
        },
    }


def contract_from_gate_document(
    document: Mapping[str, Any],
) -> M6GateContract:
    """Resolve exactly ``gates.gate_m6`` and ignore every sibling gate."""
    gates = document.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("gates.yaml has no gates mapping")
    block = gates.get("gate_m6")
    if not isinstance(block, Mapping):
        raise ValueError("gates.yaml has no gate_m6 block")
    contract = M6GateContract.from_block(_sanitize_gate_block(block))
    if (
        contract.floor_run != FROZEN_FLOOR_RUN
        or contract.floor_run_sha256 != FROZEN_FLOOR_SHA256
    ):
        raise ValueError("gate_m6 no longer pins the frozen v3 floor")
    return contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_m6_contract(root: Path) -> M6ResolvedContract:
    """Load the allow-listed M6 protocol and byte-verify its frozen floor."""
    repository = root.resolve()
    gate_path = repository / "gates.yaml"
    document = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("gates.yaml must contain a mapping")
    contract = contract_from_gate_document(document)
    floor_path = (repository / contract.floor_run).resolve()
    try:
        relative = floor_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError(
            "gate_m6 floor path escapes the repository"
        ) from error
    if relative != FROZEN_FLOOR_RUN:
        raise ValueError("gate_m6 floor path is not the frozen v3 artifact")
    actual_sha = _sha256(floor_path)
    if actual_sha != contract.floor_run_sha256:
        raise ValueError(
            "frozen M6 floor sha256 mismatch: "
            f"{actual_sha} != {contract.floor_run_sha256}"
        )
    floor = json.loads(floor_path.read_text(encoding="utf-8"))
    if not isinstance(floor, Mapping):
        raise ValueError("frozen M6 floor artifact must be a mapping")
    return M6ResolvedContract(
        contract=contract,
        floor_artifact=floor,
        floor_path=relative,
        floor_sha256=actual_sha,
    )


def _plain_provenance(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_artifact"):
        return value.to_artifact()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def _refit_lineage(bundle: M6RefitBundle) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name in ("family", "household", "earnings", "modifier", "mortality"):
        component = getattr(bundle, name, None)
        if component is not None and hasattr(component, "provenance"):
            components[name] = _plain_provenance(component.provenance)
    resolved_specs = dict(bundle.registry_spec_sha256s)
    resolved_specs["forward_earnings_adapter"] = EARNINGS_SPEC_SHA256
    return {
        "boundary_year": int(bundle.boundary_year),
        "earnings_spec_registration": EARNINGS_SPEC_REGISTRATION,
        "earnings_spec_sha256": EARNINGS_SPEC_SHA256,
        "resolved_spec_sha256s": resolved_specs,
        "refit_provenance": components,
        "certified_full_window_artifacts_read": False,
        "certified_full_window_artifacts_written": False,
    }


def refit_m6_phase(inputs: M6HarnessInputs) -> M6RefitPhase:
    """Refit every component at 2014 and materialize the realized population."""
    bundle = refit_m6_components(
        inputs.refit_inputs,
        boundary_year=BOUNDARY_YEAR,
    )
    if bundle.mortality is None or bundle.earnings is None:
        raise RuntimeError("M6 refit did not return mortality/earnings")
    mortality = fit_mortality_model(bundle.mortality)
    domain = fitted_earnings_domain_person_ids(
        bundle.earnings.generator
    ) & frozenset(int(value) for value in inputs.truth.anchor["person_id"])
    population = build_realized_population(
        demographic_panel=inputs.demographic_panel,
        death_records=inputs.death_records,
        earnings_panel=inputs.earnings_panel,
        disability_panel=inputs.disability_panel,
        panel_builder_inputs=inputs.panel_builder_inputs,
        earnings_domain_ids=domain,
    )
    return M6RefitPhase(
        bundle=bundle,
        mortality=mortality,
        population=population,
        lineage=_refit_lineage(bundle),
    )


def _run_m6_preflight_1(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    contract: M6GateContract,
) -> Mapping[str, Any]:
    bundle = phase.bundle
    if any(
        getattr(bundle, name, None) is None
        for name in ("family", "household", "modifier")
    ):
        raise RuntimeError("M6 refit is incomplete for pre-flight 1")
    truncated = prepare_m6_preflight_context(
        inputs.refit_inputs, boundary_year=BOUNDARY_YEAR
    )
    holdout_ids = set(truncated.train_ids)
    result = run_candidate9_recertification(
        Candidate9PreflightInputs(
            marital_panel=truncated.mpanel,
            household_panel=truncated.hh,
            holdout_ids={int(value) for value in holdout_ids},
            family=bundle.family.fitted,
            modifier=bundle.modifier.modifier,
            permanent_axis=bundle.modifier.axis,
            household=bundle.household.fitted,
        ),
        draw_indices=tuple(range(contract.n_draws)),
    )
    return recertification_payload(result)


def run_m6_preflight_2(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    contract: M6GateContract,
) -> Mapping[str, Any]:
    """Verify the certified external earnings sign-gate on a synthetic probe."""
    del inputs, contract
    if phase.bundle.earnings is None:
        raise RuntimeError("M6 refit is incomplete for pre-flight 2")
    return verify_external_sign_path(phase.bundle.earnings.generator).as_dict()


def _gated(frame: pd.DataFrame, person_ids: set[object]) -> pd.DataFrame:
    out = frame[frame["person_id"].isin(person_ids)].copy()
    if "window" in out:
        out = out[out["window"] == "gated"].copy()
    return out.reset_index(drop=True)


def _flow_presence_frame(
    population: M6RealizedPopulation,
    periods: Sequence[int],
    *,
    period_column: str,
) -> pd.DataFrame:
    """Expand odd-wave realized presence onto scored interval keys."""
    rows = [
        {"person_id": person_id, period_column: int(period)}
        for period in periods
        for person_id in sorted(
            population.presence.get(
                int(period) if int(period) % 2 else int(period) - 1,
                set(),
            )
        )
    ]
    return pd.DataFrame(rows, columns=["person_id", period_column])


def _truth_cells(
    inputs: M6HarnessInputs,
    household_ids: set[object],
    person_ids: set[object],
    domain_ids: set[object],
) -> dict[str, dict[str, Any]]:
    truth = inputs.truth
    earnings_truth = _gated(truth.earnings, person_ids)
    _projection_side, earnings_truth = restrict_earnings_domain_support(
        earnings_truth,
        earnings_truth,
        domain_ids,
    )
    return reduce_gated_cells(
        _gated(truth.marital_events, household_ids),
        _gated(truth.marital_person_years, household_ids),
        _gated(truth.disability_pairs, person_ids),
        earnings_truth,
    )


def _project_side(
    phase: M6RefitPhase,
    population: M6RealizedPopulation,
    *,
    draw_index: int,
) -> tuple[ProjectionResult, dict[str, Any]]:
    """Project one split side with a newly assembled, empty-cache engine."""
    collector: dict[str, Any] = {}
    certified = CertifiedEngineInputs.from_refit_bundle(
        phase.bundle,
        mortality=phase.mortality,
        marital_panel_builder=marital_panel_builder,
        household_panel_builder=household_panel_builder,
        disability_panel=population.disability_panel,
        disability_ids=set(population.disability_ids),
        start_weights=population.start_weights,
    )
    engine = ProjectionEngine(assemble_period_modules(certified))
    metadata = population.projection_metadata()
    metadata[M6_DRAW_OUTPUTS_KEY] = collector
    result = engine.project(
        population.initial_slice,
        end_year=PROJECTION_END_YEAR,
        draw_index=draw_index,
        metadata=metadata,
    )
    return result, collector


def _require_collected_panel(
    collector: Mapping[str, Any], key: str, expected: type
) -> Any:
    value = collector.get(key)
    if not isinstance(value, expected):
        raise RuntimeError(f"M6 projection did not publish {key!r}")
    return value


def _projected_cells(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    household_population: M6RealizedPopulation,
    person_population: M6RealizedPopulation,
    *,
    draw_index: int,
) -> tuple[
    dict[str, dict[str, Any]],
    ProjectionResult,
    Mapping[str, Any],
    ProjectionResult,
    Mapping[str, Any],
]:
    household_result, household_collector = _project_side(
        phase, household_population, draw_index=draw_index
    )
    person_result, person_collector = _project_side(
        phase, person_population, draw_index=draw_index
    )
    marital = _require_collected_panel(household_collector, "marital", object)
    marital_panel = getattr(marital, "panel", None)
    if not isinstance(marital_panel, transitions.MaritalPanel):
        raise RuntimeError("M6 marital collector has no native panel")
    disability_panel = _require_collected_panel(
        person_collector, "disability", disability_data.DisabilityPanel
    )
    projected_events, projected_person_years = prepare_projected_marital(
        marital_panel,
        household_population.anchor,
        household_population.presence,
    )
    projected_disability = prepare_projected_disability(
        disability_panel, person_population.anchor
    )
    marital_truth = _gated(
        inputs.truth.marital_person_years,
        set(household_population.holdout_ids),
    )
    marital_support = prepare_gated_realized_support(
        projected_person_years,
        marital_truth,
        realized_presence=_flow_presence_frame(
            household_population,
            GATED_FLOW_YEARS,
            period_column="year",
        ),
        start_weights=household_population.start_weights,
        presence_basis=PresenceBasis.START_OF_INTERVAL,
        period_column="year",
    )
    projected_person_years = marital_support.projection

    disability_truth = _gated(
        inputs.truth.disability_pairs,
        set(person_population.holdout_ids),
    )
    disability_support = prepare_gated_realized_support(
        projected_disability,
        disability_truth,
        realized_presence=_flow_presence_frame(
            person_population,
            (2015, 2017),
            period_column="start_wave",
        ),
        start_weights=person_population.start_weights,
        presence_basis=PresenceBasis.START_OF_INTERVAL,
        period_column="start_wave",
    )
    projected_disability = disability_support.projection
    if phase.bundle.earnings is None:
        raise RuntimeError("M6 earnings refit is missing")
    projected_earnings = project_earnings_on_realized_support(
        initial_slice=person_population.initial_slice,
        truth_support=person_population.earnings_support,
        generator=phase.bundle.earnings.generator,
        domain_person_ids=person_population.earnings_domain_ids,
        all_person_ids=person_population.holdout_ids,
        draw_index=draw_index,
    )
    projected_earnings, earnings_truth = restrict_earnings_domain_support(
        projected_earnings,
        person_population.earnings_support,
        person_population.earnings_domain_ids,
    )
    earnings_support = prepare_gated_realized_support(
        projected_earnings,
        earnings_truth,
        realized_presence=earnings_truth[
            ["person_id", "period"]
        ].drop_duplicates(),
        start_weights=person_population.start_weights,
        presence_basis=PresenceBasis.EXACT_WAVE,
        period_column="period",
    )
    projected_earnings = earnings_support.projection
    cells = reduce_gated_cells(
        projected_events,
        projected_person_years,
        projected_disability,
        projected_earnings,
    )
    return (
        cells,
        household_result,
        household_collector,
        person_result,
        person_collector,
    )


def _prepare_marital_years(
    panel: transitions.MaritalPanel,
    anchor: pd.DataFrame,
    presence: Mapping[int, set[int]],
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[frame["year"].isin(years)].copy()
        out = out.merge(
            anchor[["person_id", "weight"]],
            on="person_id",
            how="inner",
            suffixes=("_raw", ""),
        )
        year = out["year"].to_numpy(dtype=np.int64)
        opening = np.where(year % 2 == 1, year, year - 1)
        keep = np.asarray(
            [
                person_id in presence.get(int(wave), set())
                for person_id, wave in zip(
                    out["person_id"], opening, strict=True
                )
            ],
            dtype=bool,
        )
        out = out.loc[keep].copy()
        out["band"] = out["age"].map(
            lambda value: band_of(value, MARITAL_BANDS)
        )
        return out[out["band"].notna() & out["sex"].isin(SEXES)].reset_index(
            drop=True
        )

    return prepare(panel.events), prepare(panel.person_years)


def _prepare_disability_starts(
    panel: disability_data.DisabilityPanel,
    anchor: pd.DataFrame,
    starts: tuple[int, ...],
) -> pd.DataFrame:
    person_years = panel.person_years.sort_values(
        ["person_id", "period"], kind="stable"
    )
    grouped = person_years.groupby("person_id", sort=False)
    next_period = grouped["period"].shift(-1)
    next_disabled = grouped["disabled"].shift(-1)
    gap = next_period - person_years["period"]
    keep = (
        next_period.notna()
        & gap.between(1, disability_data.MAX_INTERVAL)
        & person_years["period"].isin(starts)
    )
    pairs = pd.DataFrame(
        {
            "person_id": person_years.loc[keep, "person_id"].to_numpy(),
            "sex": person_years.loc[keep, "sex"].to_numpy(),
            "age": person_years.loc[keep, "age"].to_numpy(dtype=np.int64),
            "start_wave": person_years.loc[keep, "period"].to_numpy(
                dtype=np.int64
            ),
            "from_disabled": person_years.loc[keep, "disabled"].to_numpy(
                dtype=bool
            ),
            "to_disabled": next_disabled.loc[keep].to_numpy(dtype=bool),
        }
    )
    pairs = pairs.merge(
        anchor[["person_id", "weight"]], on="person_id", how="inner"
    )
    pairs["band"] = pairs["age"].map(
        lambda value: band_of(value, DISABILITY_BANDS)
    )
    return pairs[pairs["band"].notna() & pairs["sex"].isin(SEXES)].reset_index(
        drop=True
    )


def _projected_mortality_cells(
    collector: Mapping[str, Any],
    years: tuple[int, ...],
    presence: Mapping[int, set[int]],
) -> dict[str, dict[str, float]]:
    slices = collector.get("mortality_slices", ())
    if not isinstance(slices, Sequence) or not slices:
        raise RuntimeError("M6 projection published no mortality slices")
    frame = pd.concat(list(slices), ignore_index=True)
    frame = frame[frame["cal_year"].isin(years)].copy()
    calendar_year = frame["cal_year"].to_numpy(dtype=np.int64)
    opening_wave = np.where(
        calendar_year % 2 == 1, calendar_year, calendar_year - 1
    )
    present = np.asarray(
        [
            person_id in presence.get(int(wave), set())
            for person_id, wave in zip(
                frame["person_id"], opening_wave, strict=True
            )
        ],
        dtype=bool,
    )
    frame = frame.loc[present].copy()
    frame["band"] = frame["age"].map(
        lambda value: band_of(
            value,
            (
                (25, 34),
                (35, 44),
                (45, 54),
                (55, 64),
                (65, 74),
                (75, 84),
                (85, 120),
            ),
        )
    )
    return mortality_cells(frame[frame["band"].notna()])


def _live_earnings(
    result: ProjectionResult,
    truth_support: pd.DataFrame,
    years: tuple[int, ...],
) -> pd.DataFrame:
    panel = result.panel
    required = {"person_id", "year", "earnings"}
    if missing := required - set(panel):
        raise RuntimeError(f"projected panel lacks earnings fields {missing}")
    projected = panel[panel["year"].isin(years)][
        ["person_id", "year", "earnings"]
    ].rename(columns={"year": "period"})
    metadata = truth_support.drop(columns="earnings", errors="ignore")
    return projected.merge(
        metadata,
        on=["person_id", "period"],
        how="inner",
        validate="one_to_one",
    )


def _surface_pair(
    truth: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    return {"truth": dict(truth), "projection": dict(projection)}


def _draw_report(
    inputs: M6HarnessInputs,
    household_population: M6RealizedPopulation,
    person_population: M6RealizedPopulation,
    household_result: ProjectionResult,
    household_collector: Mapping[str, Any],
    person_result: ProjectionResult,
    person_collector: Mapping[str, Any],
    *,
    draw_index: int,
) -> Mapping[str, Any]:
    marital_result = household_collector["marital"]
    marital_panel = marital_result.panel
    disability_panel = person_collector["disability"]
    gated_ev, gated_py = _prepare_marital_years(
        marital_panel,
        household_population.anchor,
        household_population.presence,
        GATED_FLOW_YEARS,
    )
    shock_ev, shock_py = _prepare_marital_years(
        marital_panel,
        household_population.anchor,
        household_population.presence,
        SHOCK_FLOW_YEARS,
    )
    truth_gated_ev = _gated(
        inputs.truth.marital_events, set(household_population.holdout_ids)
    )
    truth_gated_py = _gated(
        inputs.truth.marital_person_years,
        set(household_population.holdout_ids),
    )
    truth_shock_ev = inputs.truth.marital_events[
        inputs.truth.marital_events["person_id"].isin(
            household_population.holdout_ids
        )
        & (inputs.truth.marital_events["window"] == "shock")
    ]
    truth_shock_py = inputs.truth.marital_person_years[
        inputs.truth.marital_person_years["person_id"].isin(
            household_population.holdout_ids
        )
        & (inputs.truth.marital_person_years["window"] == "shock")
    ]
    gated_marital_truth = marital_cells(truth_gated_ev, truth_gated_py)
    gated_marital_projection = marital_cells(gated_ev, gated_py)
    widowhood_truth = {
        key: value
        for key, value in gated_marital_truth.items()
        if key.startswith("widowhood.")
    }
    widowhood_projection = {
        key: value
        for key, value in gated_marital_projection.items()
        if key.startswith("widowhood.")
    }

    truth_mortality = inputs.truth.mortality[
        inputs.truth.mortality["person_id"].isin(
            household_population.holdout_ids
        )
    ]
    gated_truth_mortality = mortality_cells(
        truth_mortality[truth_mortality["window"] == "gated"]
    )
    shock_truth_mortality = mortality_cells(
        truth_mortality[truth_mortality["window"] == "shock"]
    )
    shock_disability_truth = inputs.truth.disability_pairs[
        inputs.truth.disability_pairs["person_id"].isin(
            person_population.holdout_ids
        )
        & (inputs.truth.disability_pairs["window"] == "shock")
    ]
    shock_disability_projection = _prepare_disability_starts(
        disability_panel, person_population.anchor, (2019,)
    )
    shock_earnings_truth = inputs.truth.earnings[
        inputs.truth.earnings["person_id"].isin(person_population.holdout_ids)
    ]
    shock_earnings_projection = _live_earnings(
        person_result,
        person_population.earnings_support,
        (2018, *SHOCK_EARN_YEARS),
    )
    truth_earn_cells = earnings_cells(
        shock_earnings_truth,
        level_years=SHOCK_EARN_YEARS,
        change_years=(2018, *SHOCK_EARN_YEARS),
    )
    projected_earn_cells = earnings_cells(
        shock_earnings_projection,
        level_years=SHOCK_EARN_YEARS,
        change_years=(2018, *SHOCK_EARN_YEARS),
    )

    initial_ids = set(household_population.holdout_ids)
    synthetic_ids = set(household_result.panel["person_id"]) - initial_ids
    return {
        "draw_index": draw_index,
        "draw_seed": 5200 + draw_index,
        "not_certified": {
            "mortality_drift": _surface_pair(
                gated_truth_mortality,
                _projected_mortality_cells(
                    household_collector,
                    GATED_FLOW_YEARS,
                    household_population.presence,
                ),
            ),
            "widowhood": _surface_pair(widowhood_truth, widowhood_projection),
        },
        "shock_window": {
            "machine_reason": "exogenous_shock_outside_model_class",
            "mortality": _surface_pair(
                shock_truth_mortality,
                _projected_mortality_cells(
                    household_collector,
                    SHOCK_FLOW_YEARS,
                    household_population.presence,
                ),
            ),
            "marital": _surface_pair(
                marital_cells(truth_shock_ev, truth_shock_py),
                marital_cells(shock_ev, shock_py),
            ),
            "disability": _surface_pair(
                disability_cells(shock_disability_truth),
                disability_cells(shock_disability_projection),
            ),
            "earnings": _surface_pair(truth_earn_cells, projected_earn_cells),
        },
        "entrants": {
            # Every synthetic ID allocated by this closed-panel engine is a
            # step-4 materialized maternal birth.  The cached marital core's
            # separate births object is not the production step-4 draw.
            "synthetic_births": len(synthetic_ids),
            "immigrant_cohorts": 0,
            "synthetic_persons": len(synthetic_ids),
            "scheduled_realized_openers": sum(
                len(frame)
                for frame in household_population.scheduled_entries_by_year.values()
            ),
        },
        "trace": {
            "household_years": [
                trace.year for trace in household_result.traces
            ],
            "person_years": [trace.year for trace in person_result.traces],
        },
    }


def score_m6_seed(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    contract: M6GateContract,
    seed: int,
) -> M6SeedRun:
    """Project the two correlation-respecting side-A halves for K draws."""
    population = phase.population
    household_ids = side_a_person_ids(
        population.anchor, split_unit="household", seed=seed
    )
    person_ids = side_a_person_ids(
        population.anchor, split_unit="person", seed=seed
    )
    household_population = subset_realized_population(
        population, household_ids
    )
    person_population = subset_realized_population(population, person_ids)
    truth_cells = _truth_cells(
        inputs,
        set(household_ids),
        set(person_ids),
        set(person_population.earnings_domain_ids),
    )
    projected_draw_cells: list[Mapping[str, Mapping[str, Any]]] = []
    draw_reports: list[Mapping[str, Any]] = []
    for draw_index in range(contract.n_draws):
        (
            projected,
            household_result,
            household_collector,
            person_result,
            person_collector,
        ) = _projected_cells(
            inputs,
            phase,
            household_population,
            person_population,
            draw_index=draw_index,
        )
        projected_draw_cells.append(projected)
        draw_reports.append(
            _draw_report(
                inputs,
                household_population,
                person_population,
                household_result,
                household_collector,
                person_result,
                person_collector,
                draw_index=draw_index,
            )
        )
    score = score_gate_seed(
        contract,
        seed=seed,
        truth_cells=truth_cells,
        projected_draw_cells=projected_draw_cells,
        n_side_a_units=None,
    )
    return M6SeedRun(
        seed=seed,
        score=score,
        side_a_units={
            "household": int(
                household_population.anchor["household_id"].nunique()
            ),
            "person": int(len(person_ids)),
        },
        draw_reports=tuple(draw_reports),
        truth_cells=truth_cells,
    )


def domain_earnings_floor_check(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    resolved: M6ResolvedContract,
) -> Mapping[str, Any]:
    """Run the required truth-only, two-directional domain-floor audit."""
    return recompute_domain_earnings_floor(
        phase.population.anchor,
        inputs.truth.earnings,
        phase.population.earnings_domain_ids,
        resolved.contract,
        frozen_floor_artifact=resolved.floor_artifact,
    )


def build_report_only(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    resolved: M6ResolvedContract,
    seed_runs: Sequence[M6SeedRun],
    domain_floor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Assemble every non-gating surface without changing the verdict."""
    del resolved
    n_closed = int(len(phase.population.holdout_ids))
    n_domain = int(len(phase.population.earnings_domain_ids))
    anchor = getattr(phase.population, "anchor", None)
    if isinstance(anchor, pd.DataFrame) and {
        "person_id",
        "anchor_wave",
    }.issubset(anchor):
        later_earnings_entrants = int(
            (
                (anchor["anchor_wave"] > 2015)
                & ~anchor["person_id"].isin(
                    phase.population.earnings_domain_ids
                )
            ).sum()
        )
    else:
        later_earnings_entrants = n_closed - n_domain
    shock_runs = []
    not_certified_measurements = {
        "mortality_drift": [],
        "widowhood": [],
    }
    entrant_draws = []
    for run in seed_runs:
        seed_shock = []
        seed_mortality = []
        seed_widowhood = []
        seed_entrants = []
        for draw in run.draw_reports:
            shock = draw["shock_window"]
            flow_truth: dict[str, Any] = {}
            flow_projection: dict[str, Any] = {}
            for module in ("mortality", "marital", "disability"):
                for cell, record in shock[module]["truth"].items():
                    flow_truth[f"{module}:{cell}"] = record
                for cell, record in shock[module]["projection"].items():
                    flow_projection[f"{module}:{cell}"] = record
            seed_shock.append(
                build_shock_window_diagnostics(
                    flow_projected=flow_projection,
                    flow_truth=flow_truth,
                    earnings_projected=shock["earnings"]["projection"],
                    earnings_truth=shock["earnings"]["truth"],
                )
            )
            seed_mortality.append(draw["not_certified"]["mortality_drift"])
            seed_widowhood.append(draw["not_certified"]["widowhood"])
            seed_entrants.append(draw["entrants"])
        shock_runs.append({"seed": run.seed, "draws": seed_shock})
        not_certified_measurements["mortality_drift"].append(
            {"seed": run.seed, "draws": seed_mortality}
        )
        not_certified_measurements["widowhood"].append(
            {"seed": run.seed, "draws": seed_widowhood}
        )
        entrant_draws.append({"seed": run.seed, "draws": seed_entrants})

    shock_window = {
        "status": "computed" if shock_runs else "not_computed",
        "gated": False,
        "per_seed_draws": shock_runs,
    }
    not_certified = build_not_certified_surface(not_certified_measurements)
    reference_entrants = (
        seed_runs[0].draw_reports[0]["entrants"]
        if seed_runs and seed_runs[0].draw_reports
        else {
            "synthetic_births": 0,
            "immigrant_cohorts": 0,
            "synthetic_persons": 0,
        }
    )
    entrant_counts = build_entrant_diagnostics(
        synthetic_births=int(reference_entrants["synthetic_births"]),
        immigrant_cohorts=int(reference_entrants["immigrant_cohorts"]),
        later_earnings_entrants=later_earnings_entrants,
        marked_no_earnings_state=n_closed - n_domain,
        synthetic_person_ids=int(reference_entrants["synthetic_persons"]),
    )
    entrant_counts["reference_draw"] = {
        "seed": seed_runs[0].seed if seed_runs else None,
        "draw_index": 0 if seed_runs and seed_runs[0].draw_reports else None,
    }
    entrant_counts["ensemble_draw_counts"] = entrant_draws
    realized_seed_cells = {
        str(run.seed): dict(run.truth_cells or {}) for run in seed_runs
    }
    mortality_anchor = build_mortality_anchor_disclosure()
    if phase.bundle.mortality is not None:
        mortality_anchor["fitted_external_reference_rates"] = (
            phase.bundle.mortality.external_rates.to_dict(orient="records")
        )
    report = assemble_report_only_payload(
        shock_window=shock_window,
        not_certified=not_certified,
        entrants=entrant_counts,
        alignment=build_alignment_displacement(None, None),
        redrawn_seed=build_redrawn_seed_comparison(
            realized_seed_cells=realized_seed_cells
        ),
        mortality_anchor=mortality_anchor,
    )
    return {
        "domain_earnings_floor": dict(domain_floor),
        "family_b": report,
        "family_c": {
            "gating": False,
            "optional": True,
            "registered_fingerprints": [],
            "pass": None,
        },
    }


def _score_artifact(
    gate_score: M6GateScore | Any,
    seed_runs: Sequence[M6SeedRun],
) -> dict[str, Any]:
    payload = dict(gate_score.to_artifact())
    units = {run.seed: dict(run.side_a_units) for run in seed_runs}
    for record in payload.get("per_seed", []):
        seed = int(record["seed"])
        record["n_side_a_units"] = units[seed]
    return payload


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or (value is not None and not isinstance(value, str)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def assemble_m6_artifact(
    *,
    registration_id: str,
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    resolved: M6ResolvedContract,
    seed_runs: Sequence[M6SeedRun],
    gate_score: M6GateScore | Any,
    preflight_1: Mapping[str, Any],
    preflight_2: Mapping[str, Any],
    report_only: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete payload in memory before the exclusive write."""
    contract = resolved.contract
    family_a = _score_artifact(gate_score, seed_runs)
    passed = bool(gate_score.passed)
    valid = bool(gate_score.valid)
    provenance = _plain_provenance(getattr(inputs, "provenance", {}))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": "gate_m6_candidate1",
            "phase_order": list(PHASE_ORDER),
            "boundary_year": BOUNDARY_YEAR,
            "projection_end_year": PROJECTION_END_YEAR,
            "build_only_guard": False,
        },
        "candidate": {
            "number": CANDIDATE_NUMBER,
            "description": "M6 composed projection engine per design 2.8",
        },
        "registration": {
            "issue": 42,
            "registration_id": registration_id,
            "fresh_registration_required": True,
        },
        "lineage": {
            **dict(phase.lineage),
            "floor_run": resolved.floor_path,
            "floor_sha256": resolved.floor_sha256,
        },
        "gate": {
            "name": "gate_m6",
            "status": "locked",
            "cells": [
                {
                    "cell": rule.cell,
                    "family": rule.family,
                    "split_unit": rule.split_unit,
                    "metric": rule.metric,
                    "tolerance": rule.tolerance,
                    "k": rule.k,
                    "rounding": rule.rounding,
                }
                for rule in contract.cells
            ],
            "tolerance_source": "locked gates.yaml gate_m6 views",
            "tolerances_recomputed": False,
        },
        "protocol": {
            "gate_seeds": list(contract.gate_seeds),
            "required_seed_passes": contract.required_seed_passes,
            "n_draws": contract.n_draws,
            "draw_index": list(range(contract.n_draws)),
            "draw_seeds": list(contract.draw_seeds),
            "split_fraction": 0.5,
            "split_units": {
                "marital": "household",
                "disability": "person",
                "earnings": "person",
            },
            "earnings_support": "realized_support_intersect_2014_domain",
            "earnings_support_symmetric_both_sides": True,
            "evaluation_support_adapter": (
                "engine.support.prepare_evaluation_support"
            ),
            "evaluation_mode": "GATED_REALIZED",
            "presence_basis": {
                "marital_disability": "START_OF_INTERVAL",
                "earnings": "EXACT_WAVE",
            },
            "undefined_draw_invalidates": True,
            "regenerated_surface_required": True,
        },
        "verdict": {
            "valid": valid,
            "pass": passed,
            "family_a_pass": passed,
            "family_b_gated": False,
            "family_c_gated": False,
            "publishes_regardless": True,
            "certifies_nothing_about_mortality_drift": True,
            "earnings_certification": (
                "M6-first-certified forward earnings law; no gate_1 "
                "backward-law certificate transfers"
            ),
        },
        "family_a": family_a,
        "family_b": report_only["family_b"],
        "family_c": report_only["family_c"],
        "preflights": {
            "candidate9_recertification": dict(preflight_1),
            "earnings_sign_path": dict(preflight_2),
        },
        "earnings_domain_floor_self_check": report_only[
            "domain_earnings_floor"
        ],
        "provenance": provenance,
        "fence": {
            "gates_yaml_read": "gate_m6 protocol/cells only",
            "post_boundary_macro_on_scored_path": False,
            "evaluation_mode": "GATED_REALIZED",
            "forward_realized_inputs": "rejected_by_contract",
            "certified_full_window_artifacts_read": False,
            "certified_full_window_artifacts_written": False,
        },
        "publishes_regardless": True,
    }
    return _json_safe(artifact)


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    artifacts.write_new(path, artifact, sidecar=True)


def default_operations() -> M6RunnerOperations:
    return M6RunnerOperations(
        resolve_contract=resolve_m6_contract,
        refit=refit_m6_phase,
        preflight_1=_run_m6_preflight_1,
        preflight_2=run_m6_preflight_2,
        score_seed=score_m6_seed,
        aggregate=aggregate_gate,
        domain_floor=domain_earnings_floor_check,
        report_only=build_report_only,
        write=_write_artifact,
    )


def _ensure_exclusive_targets(path: Path) -> None:
    for target in (path, Path(f"{path}.env.json")):
        if os.path.lexists(target):
            raise FileExistsError(
                f"{target} already exists; choose a fresh one-shot output"
            )


def guard_registered_m6_run(
    *,
    registration_id: str,
    output: Path | str = DEFAULT_OUTPUT,
    root: Path | str | None = None,
) -> None:
    """Validate the one-shot request before any external input is loaded.

    The CLI calls this before importing its registered input factory.  The
    execution entry point repeats the same checks after inputs exist, closing
    the time-of-check/time-of-use window around the exclusive artifact path.
    """
    validate_registration_id(registration_id)
    repository = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    destination = Path(output)
    if not destination.is_absolute():
        destination = repository / destination
    _ensure_exclusive_targets(destination)
    resolve_m6_contract(repository)


def execute_registered_m6_run(
    inputs: M6HarnessInputs,
    *,
    registration_id: str,
    output: Path | str = DEFAULT_OUTPUT,
    root: Path | str | None = None,
    operations: M6RunnerOperations | None = None,
) -> dict[str, Any]:
    """Execute the ordered registered pass and write only after every phase."""
    registration = validate_registration_id(registration_id)
    repository = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    destination = Path(output)
    if not destination.is_absolute():
        destination = repository / destination
    _ensure_exclusive_targets(destination)
    ops = operations or default_operations()
    resolved = ops.resolve_contract(repository)

    phase = ops.refit(inputs)
    preflight_1 = ops.preflight_1(inputs, phase, resolved.contract)
    preflight_2 = ops.preflight_2(inputs, phase, resolved.contract)

    seed_runs = tuple(
        ops.score_seed(inputs, phase, resolved.contract, seed)
        for seed in resolved.contract.gate_seeds
    )
    if tuple(run.seed for run in seed_runs) != resolved.contract.gate_seeds:
        raise RuntimeError("seed phase returned results out of protocol order")
    gate_score = ops.aggregate(
        resolved.contract, [run.score for run in seed_runs]
    )
    domain_floor = ops.domain_floor(inputs, phase, resolved)
    report = ops.report_only(inputs, phase, resolved, seed_runs, domain_floor)
    artifact = assemble_m6_artifact(
        registration_id=registration,
        inputs=inputs,
        phase=phase,
        resolved=resolved,
        seed_runs=seed_runs,
        gate_score=gate_score,
        preflight_1=preflight_1,
        preflight_2=preflight_2,
        report_only=report,
    )
    ops.write(destination, artifact)
    return artifact


__all__ = [
    "CANDIDATE_NUMBER",
    "DEFAULT_OUTPUT",
    "FROZEN_FLOOR_RUN",
    "FROZEN_FLOOR_SHA256",
    "M6RefitPhase",
    "M6ResolvedContract",
    "M6RunnerOperations",
    "M6SeedRun",
    "PHASE_ORDER",
    "SCHEMA_VERSION",
    "assemble_m6_artifact",
    "contract_from_gate_document",
    "default_operations",
    "execute_registered_m6_run",
    "guard_registered_m6_run",
    "resolve_m6_contract",
    "validate_registration_id",
]
