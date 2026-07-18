"""Leakage-safe refit entry points for the M6 temporal holdout.

The functions in this module are deliberately pure at the repository
boundary: every estimation frame, external reference, parameter bundle, and
fitter is supplied by the caller.  No function resolves a path, reads a
certified run, consults ``gates.yaml``, or writes a fitted artifact.

The frozen specifications and newly fitted estimates are separate objects.
The family and household registry adapters retain the certified
``CandidateSpec`` hashes while recording that their estimates came from an
estimation window ending at ``T*``.  Annual flow inputs can additionally carry
``required_interview_year``; such a row is usable only when both its event date
and the interview needed to date it are no later than ``T*``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from populace_dynamics import claiming
from populace_dynamics.data import couple_earnings as ce
from populace_dynamics.data import disability, transitions
from populace_dynamics.data import household_composition as hc_data
from populace_dynamics.models import couple_formation_sim_v1 as couple_v1
from populace_dynamics.models import couple_formation_sim_v2 as couple_v2
from populace_dynamics.models import disability_hazard_sim as m4
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition as hc

from .candidates import CandidateSpec
from .forward_earnings import (
    ForwardEarningsGenerator,
    fit_forward_earnings,
    validate_rank_refresh_fit,
)
from .steps import AgeSexMortalityModel

__all__ = [
    "BOUNDARY_YEAR",
    "EarningsChainedRefit",
    "ExternalVintage",
    "M6RefitBundle",
    "M6RefitInputs",
    "ModifierRefit",
    "MortalityRefitInputs",
    "QRFModelFactory",
    "RefitProvenance",
    "RegistryRefit",
    "claiming_pmfs_from_reference",
    "fit_mortality_model",
    "prepare_mortality_refit_inputs",
    "prepare_m6_preflight_context",
    "refit_disability",
    "refit_earnings_chained_generator",
    "refit_family_transitions",
    "refit_first_marriage_modifier",
    "refit_household_composition",
    "refit_m6_components",
    "truncate_estimation_frame",
    "validate_external_vintage",
]

BOUNDARY_YEAR = 2014
REQUIRED_INTERVIEW_COLUMN = "required_interview_year"

EARNINGS_SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4905323933"
)
EARNINGS_SPEC = {
    "candidate": "m6_forward_mirror_of_gate1_candidate11",
    "shared_qrf": {
        "implementation": "populace.fit.qrf.RegimeGatedQRF",
        "parameters": "defaults",
        "predictors": ["earnings", "age_tp2"],
        "target": "earnings_tp2",
        "weight": "weight_tp2",
    },
    "zero_anchor_qrf": {
        "implementation": "populace.fit.qrf.RegimeGatedQRF",
        "parameters": "defaults",
        "population": "zero_anchor_train_pairs",
    },
    "rank_bootstrap": {
        "k": 25,
        "distance_weights": [1.0, 0.5, 0.25],
        "lambda": 0.1,
        "full_reentry_pool": True,
        "q0_memory_exempt": True,
    },
    "age_support": [25, 64],
    "age_bins": 8,
    "level_map": "pooled_age_bin_nawi_normalized_cell_marginal",
    "wage_index_projection": "ols_log_nawi_2005_2014",
}
EARNINGS_SPEC_SHA256 = hashlib.sha256(
    json.dumps(EARNINGS_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True)
class RefitProvenance:
    """Provenance that distinguishes a truncated fit from a frozen spec."""

    boundary_year: int
    estimation_rule: str
    n_rows: Mapping[str, int]
    max_year: Mapping[str, int | None]
    certified_full_window_artifacts_read: bool = False
    certified_full_window_artifacts_written: bool = False


@dataclass(frozen=True)
class RegistryRefit:
    """A registry result carrying the unchanged spec and new-fit provenance."""

    candidate_id: str
    spec_sha256: str
    fitted: Any
    provenance: RefitProvenance


@dataclass(frozen=True)
class ExternalVintage:
    """Caller-declared external-reference vintage admitted to a refit."""

    name: str
    vintage_year: int
    boundary_year: int


@dataclass(frozen=True)
class EarningsChainedRefit:
    """The fitted M6 forward chain and its truncated support."""

    generator: ForwardEarningsGenerator
    shared_gate: Any
    zero_anchor_gate: Any | None
    estimation_panel: pd.DataFrame
    forward_pairs: pd.DataFrame
    anchors: pd.DataFrame
    n_zero_anchor_pairs: int
    u_w_diagnostics: Mapping[str, Any]
    seed: int
    spec_registration: str
    adapter_spec_sha256: str
    provenance: RefitProvenance
    engine_candidate_id: str | None = None
    engine_candidate_spec_sha256: str | None = None
    q_invariant_fit_signature_sha256: str | None = None
    rank_refresh_fit_audit: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ModifierRefit:
    """Gate-2c permanent earnings axis plus its first-marriage modifier."""

    axis: couple_v1.CommittedAxis
    modifier: couple_v2.FirstMarriageEarningsModifier
    ssa_vintage: ExternalVintage
    provenance: RefitProvenance


@dataclass(frozen=True)
class MortalityRefitInputs:
    """Temporally admissible mortality exposure and external reference."""

    exposure: pd.DataFrame
    external_rates: pd.DataFrame
    external_vintage: ExternalVintage
    provenance: RefitProvenance


@dataclass(frozen=True)
class M6RefitBundle:
    """Collected fits for one boundary, without serializing any artifact."""

    boundary_year: int
    family: RegistryRefit | None = None
    household: RegistryRefit | None = None
    earnings: EarningsChainedRefit | None = None
    modifier: ModifierRefit | None = None
    disability: Any | None = None
    claiming_pmfs: Mapping[tuple[str, int], Mapping[int, float]] | None = None
    mortality: MortalityRefitInputs | None = None

    @property
    def registry_spec_sha256s(self) -> dict[str, str]:
        """Return the certified registry hashes represented in the bundle."""
        out: dict[str, str] = {}
        if self.family is not None:
            out["family_transitions"] = self.family.spec_sha256
        if self.household is not None:
            out["household_composition"] = self.household.spec_sha256
        if self.modifier is not None:
            # Gate-2c's modifier is built around the same certified C16
            # transition spec.  Keep the role-specific key even though the
            # digest intentionally duplicates family_transitions.
            out["first_marriage_modifier_core"] = (
                couple_v2.CERTIFIED_SPEC.sha256
            )
        return out


@dataclass(frozen=True)
class M6RefitInputs:
    """Caller-owned inputs for the complete no-artifact refit entry point."""

    family_context: ft.FitContext
    household_context: hc.FitContext
    earnings_panel: pd.DataFrame
    earnings_seed: int
    modifier_marital_panel: transitions.MaritalPanel
    modifier_interview_years: pd.Series | np.ndarray
    modifier_marriage_records: pd.DataFrame
    modifier_person_weight: pd.Series
    ssa_params: Any
    ssa_params_vintage: int
    modifier_train_ids: set[int]
    disability_panel: disability.DisabilityPanel
    disability_train_ids: set[int]
    claiming_reference: claiming.ClaimAgeReference
    mortality_exposure: pd.DataFrame
    mortality_external_rates: pd.DataFrame
    mortality_external_vintage: int


class _QRFModel(Protocol):
    def fit(
        self,
        frame: pd.DataFrame,
        *,
        predictors: list[str],
        targets: list[str],
        weights: str,
    ) -> Any: ...


class QRFModelFactory(Protocol):
    """Factory seam for candidate-11's optional ``populace-fit`` dependency."""

    def __call__(self, *, seed: int) -> _QRFModel: ...


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing column(s) {missing}")


def _integer_years(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"{label} contains a non-numeric year")
    return numeric


def truncate_estimation_frame(
    frame: pd.DataFrame,
    *,
    boundary_year: int = BOUNDARY_YEAR,
    year_column: str,
    flow: bool = False,
    required_interview_column: str = REQUIRED_INTERVIEW_COLUMN,
    require_interview_provenance: bool = True,
    label: str = "estimation frame",
) -> pd.DataFrame:
    """Return rows admissible under the field-aware ``<=T*`` rule.

    A level or stock row is dated only by ``year_column``.  A flow row must
    also be dated no later than the interview required to establish the event.
    Callers may set ``require_interview_provenance=False`` only for a source
    whose dating is independently known not to depend on a survey interview.
    """
    _require_columns(frame, (year_column,), label)
    years = _integer_years(frame[year_column], f"{label}.{year_column}")
    keep = years.notna() & (years <= int(boundary_year))
    if flow:
        if required_interview_column not in frame:
            if require_interview_provenance:
                raise ValueError(
                    f"{label} flow rows require "
                    f"{required_interview_column!r} for the conservative "
                    "biennial boundary rule"
                )
        else:
            interview = _integer_years(
                frame[required_interview_column],
                f"{label}.{required_interview_column}",
            )
            keep &= interview.notna() & (interview <= int(boundary_year))
    return frame.loc[keep].copy().reset_index(drop=True)


def _truncate_optional_flow(
    frame: pd.DataFrame,
    *,
    year_column: str,
    boundary_year: int,
    label: str,
) -> pd.DataFrame:
    """Apply interview dating when supplied, otherwise enforce event year."""
    return truncate_estimation_frame(
        frame,
        boundary_year=boundary_year,
        year_column=year_column,
        flow=REQUIRED_INTERVIEW_COLUMN in frame,
        label=label,
    )


def _truncate_event_history(
    frame: pd.DataFrame,
    *,
    year_column: str,
    event_flag: str,
    fallback_interview_column: str | None,
    boundary_year: int,
    label: str,
) -> pd.DataFrame:
    """Keep placeholders and only actual events dated inside the fit window."""
    if frame.empty:
        return frame.copy()
    _require_columns(frame, (year_column, event_flag), label)
    years = _integer_years(frame[year_column], f"{label}.{year_column}")
    event = frame[event_flag].fillna(False).astype(bool)
    keep = ~event | (years.notna() & (years <= boundary_year))
    interview_column = (
        REQUIRED_INTERVIEW_COLUMN
        if REQUIRED_INTERVIEW_COLUMN in frame
        else fallback_interview_column
    )
    if interview_column is None or interview_column not in frame:
        if event.any():
            raise ValueError(
                f"{label} actual events require interview-year provenance"
            )
    else:
        interview = _integer_years(
            frame[interview_column],
            f"{label}.{interview_column}",
        )
        keep &= ~event | (interview.notna() & (interview <= boundary_year))
    return frame.loc[keep].copy().reset_index(drop=True)


def _truncate_marital_panel(
    panel: transitions.MaritalPanel,
    boundary_year: int,
    *,
    interview_years: pd.Series | np.ndarray | None = None,
) -> transitions.MaritalPanel:
    def with_interview_year(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if REQUIRED_INTERVIEW_COLUMN in frame:
            return frame
        if interview_years is None:
            raise ValueError(
                f"{label} requires interview-year provenance for annual flows"
            )
        waves = np.sort(
            pd.to_numeric(pd.Series(interview_years), errors="coerce")
            .dropna()
            .unique()
        )
        if waves.size == 0:
            raise ValueError("interview-year calendar is empty")
        out = frame.copy()
        years = pd.to_numeric(out["year"], errors="coerce").to_numpy()
        position = np.searchsorted(waves, years, side="left")
        required = np.full(len(out), np.nan)
        available = position < waves.size
        required[available] = waves[position[available]]
        out[REQUIRED_INTERVIEW_COLUMN] = required
        return out

    person_years = truncate_estimation_frame(
        with_interview_year(panel.person_years, "marital person-years"),
        year_column="year",
        boundary_year=boundary_year,
        flow=True,
        label="marital person-years",
    )
    events = truncate_estimation_frame(
        with_interview_year(panel.events, "marital events"),
        year_column="year",
        boundary_year=boundary_year,
        flow=True,
        label="marital events",
    )
    attrs = panel.attrs.copy()
    if "censor_year" in attrs:
        attrs["censor_year"] = np.minimum(
            pd.to_numeric(attrs["censor_year"], errors="coerce"),
            boundary_year,
        )
    present = set(person_years["person_id"])
    attrs = attrs[attrs["person_id"].isin(present)].reset_index(drop=True)
    return transitions.MaritalPanel(
        person_years=person_years,
        events=events,
        attrs=attrs,
    )


def _truncate_marriage_records(
    records: pd.DataFrame, boundary_year: int
) -> pd.DataFrame:
    truncated = _truncate_event_history(
        records,
        year_column="start_year",
        event_flag="is_marriage",
        fallback_interview_column="most_recent_report_year",
        boundary_year=boundary_year,
        label="marriage records",
    )
    for column in ("end_year", "separation_year"):
        if column in truncated:
            values = _integer_years(
                truncated[column], f"marriage records.{column}"
            )
            truncated.loc[values > boundary_year, column] = pd.NA
    if "most_recent_report_year" in truncated:
        values = _integer_years(
            truncated["most_recent_report_year"],
            "marriage records.most_recent_report_year",
        )
        truncated["most_recent_report_year"] = values.clip(
            upper=boundary_year
        ).astype("Int64")
    if "n_marriages" in truncated:
        counts = (
            truncated[truncated["is_marriage"].fillna(False)]
            .groupby("person_id")
            .size()
        )
        truncated["n_marriages"] = (
            truncated["person_id"].map(counts).fillna(0).astype("Int64")
        )
    return truncated


def _truncate_birth_records(
    records: pd.DataFrame, boundary_year: int
) -> pd.DataFrame:
    return _truncate_event_history(
        records,
        year_column="birth_year",
        event_flag="is_event",
        fallback_interview_column="most_recent_child_report_year",
        boundary_year=boundary_year,
        label="birth records",
    )


def _refresh_marital_estimation_state(
    panel: transitions.MaritalPanel,
    marriage_records: pd.DataFrame,
    *,
    demographic_panel: pd.DataFrame | None = None,
    person_weight: pd.Series | None = None,
) -> transitions.MaritalPanel:
    """Remove full-window count/weight state from a truncated marital panel."""
    if person_weight is None and demographic_panel is not None:
        if {"person_id", "period", "weight"}.issubset(demographic_panel):
            positive = demographic_panel[demographic_panel["weight"] > 0]
            person_weight = (
                positive.sort_values("period")
                .groupby("person_id")
                .tail(1)
                .set_index("person_id")["weight"]
            )

    attrs = panel.attrs.copy()
    if "n_marriages" in attrs and "is_marriage" in marriage_records:
        counts = (
            marriage_records[marriage_records["is_marriage"].fillna(False)]
            .groupby("person_id")
            .size()
        )
        attrs["n_marriages"] = (
            attrs["person_id"].map(counts).fillna(0).astype("float64")
        )
    person_years = panel.person_years.copy()
    events = panel.events.copy()
    if person_weight is not None:
        weights = pd.Series(person_weight, dtype="float64")
        valid = set(weights[weights > 0].index)
        attrs = attrs[attrs["person_id"].isin(valid)].copy()
        person_years = person_years[
            person_years["person_id"].isin(valid)
        ].copy()
        events = events[events["person_id"].isin(valid)].copy()
        for frame in (attrs, person_years, events):
            if "weight" in frame:
                frame["weight"] = frame["person_id"].map(weights).to_numpy()
    return transitions.MaritalPanel(
        person_years=person_years.reset_index(drop=True),
        events=events.reset_index(drop=True),
        attrs=attrs.reset_index(drop=True),
    )


def _truncate_family_context(
    context: ft.FitContext, boundary_year: int
) -> ft.FitContext:
    panel = _truncate_marital_panel(
        context.panel,
        boundary_year,
        interview_years=context.demographic_panel["period"],
    )
    demographic = truncate_estimation_frame(
        context.demographic_panel,
        boundary_year=boundary_year,
        year_column="period",
        label="demographic panel",
    )
    marriages = _truncate_marriage_records(
        context.marriage_records, boundary_year
    )
    panel = _refresh_marital_estimation_state(
        panel,
        marriages,
        demographic_panel=demographic,
    )
    births = _truncate_birth_records(context.birth_records, boundary_year)
    order_map = truncate_estimation_frame(
        context.marriage_order_map,
        boundary_year=boundary_year,
        year_column="start_year",
        label="marriage order map",
    )
    valid_ids = set(panel.attrs["person_id"])
    return ft.FitContext(
        panel=panel,
        demographic_panel=demographic,
        marriage_records=marriages,
        birth_records=births,
        marriage_order_map=order_map,
        train_ids=frozenset(set(context.train_ids) & valid_ids),
    )


def _max_year(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.max()) if len(values) else None


def _registry_provenance(
    context: ft.FitContext | hc.FitContext,
    boundary_year: int,
) -> RefitProvenance:
    return RefitProvenance(
        boundary_year=boundary_year,
        estimation_rule=(
            "all estimation rows dated <=T*; flow rows also require a "
            "<=T* interview when required_interview_year is supplied"
        ),
        n_rows={
            "marital_person_years": (
                len(context.panel.person_years)
                if isinstance(context, ft.FitContext)
                else len(context.mpanel.person_years)
            ),
            "marital_events": (
                len(context.panel.events)
                if isinstance(context, ft.FitContext)
                else len(context.mpanel.events)
            ),
            "demographic": len(context.demographic_panel),
        },
        max_year={
            "marital_person_years": _max_year(
                (
                    context.panel.person_years
                    if isinstance(context, ft.FitContext)
                    else context.mpanel.person_years
                ),
                "year",
            ),
            "marital_events": _max_year(
                (
                    context.panel.events
                    if isinstance(context, ft.FitContext)
                    else context.mpanel.events
                ),
                "year",
            ),
            "demographic": _max_year(context.demographic_panel, "period"),
        },
    )


def refit_family_transitions(
    context: ft.FitContext,
    *,
    boundary_year: int = BOUNDARY_YEAR,
    registry: Any = ft.REGISTRY,
) -> RegistryRefit:
    """Refit the unchanged candidate-16 registry on caller-supplied inputs."""
    truncated = _truncate_family_context(context, boundary_year)
    fitted = registry.fit(ft.CANDIDATE_16, truncated)
    return RegistryRefit(
        candidate_id=ft.CANDIDATE_16.candidate_id,
        spec_sha256=ft.CANDIDATE_16.sha256,
        fitted=fitted,
        provenance=_registry_provenance(truncated, boundary_year),
    )


def _truncate_year_frame(
    frame: pd.DataFrame, boundary_year: int, label: str
) -> pd.DataFrame:
    return truncate_estimation_frame(
        frame,
        boundary_year=boundary_year,
        year_column="year",
        label=label,
    )


def _rebuild_household_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute next-wave columns after removing post-boundary waves."""
    out = frame.sort_values(["person_id", "year"]).reset_index(drop=True)
    grouped = out.groupby("person_id", sort=False)
    next_year = grouped["year"].shift(-1)
    gap = next_year - out["year"]
    out["has_next"] = (
        (gap >= 1) & (gap <= hc_data.MAX_TRANSITION_GAP_YEARS)
    ).fillna(False)
    for flag in ("coresident_parent", "coresident_spouse", "multigen"):
        if flag in out:
            out[f"next_{flag}"] = grouped[flag].shift(-1)
    return out


def _truncate_household_context(
    context: hc.FitContext, boundary_year: int
) -> hc.FitContext:
    person_waves = _truncate_year_frame(
        context.hh.person_waves, boundary_year, "household person-waves"
    )
    person_waves = _rebuild_household_transitions(person_waves)
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    household_panel = hc_data.HouseholdCompositionPanel(
        person_waves=person_waves,
        attrs=attrs,
    )
    relationship = truncate_estimation_frame(
        context.relationship_map,
        boundary_year=boundary_year,
        year_column="interview_year",
        label="relationship map",
    )
    demographic = truncate_estimation_frame(
        context.demographic_panel,
        boundary_year=boundary_year,
        year_column="period",
        label="demographic panel",
    )
    marriages = _truncate_marriage_records(
        context.marriage_records, boundary_year
    )
    mpanel = _truncate_marital_panel(
        context.mpanel,
        boundary_year,
        interview_years=context.demographic_panel["period"],
    )
    mpanel = _refresh_marital_estimation_state(
        mpanel,
        marriages,
        demographic_panel=demographic,
    )
    valid_ids = set(attrs["person_id"])
    return hc.FitContext(
        hh=household_panel,
        mpanel=mpanel,
        demographic_panel=demographic,
        marriage_records=marriages,
        birth_records=_truncate_birth_records(
            context.birth_records, boundary_year
        ),
        marriage_order_map=truncate_estimation_frame(
            context.marriage_order_map,
            boundary_year=boundary_year,
            year_column="start_year",
            label="marriage order map",
        ),
        relationship_map=relationship,
        train_ids=frozenset(set(context.train_ids) & valid_ids),
        father_links_child=truncate_estimation_frame(
            context.father_links_child,
            boundary_year=boundary_year,
            year_column="birth_year",
            label="father-child links",
        ),
        parent_pairs=_truncate_year_frame(
            context.parent_pairs, boundary_year, "parent pairs"
        ),
        marital_by_year=_truncate_year_frame(
            context.marital_by_year, boundary_year, "marital-by-year"
        ),
        family_unit_sizes=_truncate_year_frame(
            context.family_unit_sizes, boundary_year, "family-unit sizes"
        ),
        legal_spouse_flag=_truncate_year_frame(
            context.legal_spouse_flag, boundary_year, "legal-spouse flag"
        ),
        child_record_exposure=_truncate_year_frame(
            context.child_record_exposure,
            boundary_year,
            "child-record exposure",
        ),
        parent_counts=_truncate_year_frame(
            context.parent_counts, boundary_year, "parent counts"
        ),
    )


def prepare_m6_preflight_context(
    inputs: M6RefitInputs,
    *,
    boundary_year: int = BOUNDARY_YEAR,
) -> hc.FitContext:
    """Return the exact cutoff household context used by pre-flight 1.

    The transfer check must simulate on the same holdout-blind native panels
    supplied to the candidate-9 refit.  Exposing the existing field-aware
    truncation here prevents a runner from accidentally reaching back to the
    full-window ``M6RefitInputs.household_context``.
    """
    if int(boundary_year) != BOUNDARY_YEAR:
        raise ValueError(
            f"the M6 scored harness requires boundary_year={BOUNDARY_YEAR}"
        )
    return _truncate_household_context(
        inputs.household_context, int(boundary_year)
    )


def refit_household_composition(
    context: hc.FitContext,
    *,
    boundary_year: int = BOUNDARY_YEAR,
    registry: Any = hc.REGISTRY,
) -> RegistryRefit:
    """Refit the unchanged candidate-9 registry on caller-supplied inputs."""
    truncated = _truncate_household_context(context, boundary_year)
    fitted = registry.fit(hc.CANDIDATE_9, truncated)
    return RegistryRefit(
        candidate_id=hc.CANDIDATE_9.candidate_id,
        spec_sha256=hc.CANDIDATE_9.sha256,
        fitted=fitted,
        provenance=_registry_provenance(truncated, boundary_year),
    )


def _default_qrf_factory(*, seed: int) -> _QRFModel:
    try:
        from populace.fit.qrf import RegimeGatedQRF
    except ImportError as error:
        raise ImportError(
            "candidate-11 refitting needs populace-fit; install its dedicated "
            "environment or inject qrf_factory for a synthetic/unit fit"
        ) from error
    return RegimeGatedQRF(seed=seed)


def refit_earnings_chained_generator(
    panel: pd.DataFrame,
    nawi: Mapping[int, float],
    *,
    seed: int,
    boundary_year: int = BOUNDARY_YEAR,
    qrf_factory: QRFModelFactory | None = None,
    candidate_spec: CandidateSpec | None = None,
) -> EarningsChainedRefit:
    """Refit the pinned forward conditional-rank law on ``<=T*`` rows."""
    fitted = fit_forward_earnings(
        panel,
        nawi,
        seed=seed,
        boundary_year=boundary_year,
        qrf_factory=qrf_factory or _default_qrf_factory,
        candidate_spec=candidate_spec,
    )
    validate_rank_refresh_fit(fitted.generator)
    provenance = RefitProvenance(
        boundary_year=boundary_year,
        estimation_rule=(
            "forward earnings income-reference period <=T*; age 25-64; "
            "calendar-invariant NAWI-normalized marginal"
        ),
        n_rows={
            "estimation_panel": len(fitted.estimation_panel),
            "forward_pairs": len(fitted.forward_pairs),
            "zero_anchor_pairs": fitted.n_zero_anchor_pairs,
        },
        max_year={
            "earnings_reference_year": _max_year(
                fitted.estimation_panel, "period"
            )
        },
    )
    return EarningsChainedRefit(
        generator=fitted.generator,
        shared_gate=fitted.generator.shared_gate,
        zero_anchor_gate=fitted.generator.zero_anchor_gate,
        estimation_panel=fitted.estimation_panel,
        forward_pairs=fitted.forward_pairs,
        anchors=fitted.anchors,
        n_zero_anchor_pairs=fitted.n_zero_anchor_pairs,
        u_w_diagnostics=fitted.u_w_diagnostics,
        seed=int(seed),
        spec_registration=EARNINGS_SPEC_REGISTRATION,
        adapter_spec_sha256=EARNINGS_SPEC_SHA256,
        provenance=provenance,
        engine_candidate_id=(
            None if candidate_spec is None else candidate_spec.candidate_id
        ),
        engine_candidate_spec_sha256=(
            None if candidate_spec is None else candidate_spec.sha256
        ),
        q_invariant_fit_signature_sha256=(
            fitted.q_invariant_fit_signature_sha256
        ),
        rank_refresh_fit_audit=(
            None
            if fitted.rank_refresh_fit_audit is None
            else fitted.rank_refresh_fit_audit.as_dict()
        ),
    )


def validate_external_vintage(
    name: str,
    vintage_year: int,
    *,
    boundary_year: int = BOUNDARY_YEAR,
) -> ExternalVintage:
    """Reject an external reference that was not available by ``T*``."""
    vintage_year = int(vintage_year)
    boundary_year = int(boundary_year)
    if vintage_year > boundary_year:
        raise ValueError(
            f"{name} vintage {vintage_year} is post-T* ({boundary_year})"
        )
    return ExternalVintage(name, vintage_year, boundary_year)


def refit_first_marriage_modifier(
    fitted_family: ft.FittedFamilyTransitions,
    marital_panel: transitions.MaritalPanel,
    earnings_panel: pd.DataFrame,
    marriage_records: pd.DataFrame,
    person_weight: pd.Series,
    params: Any,
    *,
    params_vintage: int,
    train_ids: set[int],
    interview_years: pd.Series | np.ndarray | None = None,
    boundary_year: int = BOUNDARY_YEAR,
) -> ModifierRefit:
    """Build the <=``T*`` permanent earnings axis and refit gate-2c's modifier."""
    vintage = validate_external_vintage(
        "SSA earnings-index parameters",
        params_vintage,
        boundary_year=boundary_year,
    )
    earnings = truncate_estimation_frame(
        earnings_panel,
        boundary_year=boundary_year,
        year_column="period",
        label="gate-2c earnings panel",
    )
    marriages = _truncate_marriage_records(marriage_records, boundary_year)
    mpanel = _truncate_marital_panel(
        marital_panel,
        boundary_year,
        interview_years=interview_years,
    )
    mpanel = _refresh_marital_estimation_state(
        mpanel,
        marriages,
        person_weight=person_weight,
    )
    history, birth_year, _ = ce.person_earnings_histories(earnings)
    earn = ce.indexed_earnings_supply(history, birth_year, params)
    if not earn:
        raise ValueError("gate-2c permanent earnings supply is empty")
    values = np.fromiter(earn.values(), dtype=np.float64)
    cuts = tuple(
        float(value)
        for value in np.quantile(values, ce.AIME_TERCILE_QUANTILES)
    )
    decile_edges = np.quantile(values, np.arange(1, couple_v1.N_DECILES) / 10)
    decile = np.searchsorted(decile_edges, values, side="right")
    supply_by_decile = tuple(
        np.sort(values[decile == index])
        for index in range(couple_v1.N_DECILES)
    )
    sex = (
        marriages.drop_duplicates("person_id")
        .set_index("person_id")["sex"]
        .to_dict()
    )
    axis = couple_v1.CommittedAxis(
        earn={int(key): float(value) for key, value in earn.items()},
        cuts=cuts,
        decile_edges=decile_edges,
        supply_by_decile=supply_by_decile,
        sex={int(key): str(value) for key, value in sex.items()},
        birth_year={int(key): int(value) for key, value in birth_year.items()},
        person_weight={
            int(key): float(value) for key, value in person_weight.items()
        },
    )
    valid_ids = set(mpanel.attrs["person_id"])
    modifier = couple_v2.fit_first_marriage_modifier(
        fitted_family,
        mpanel,
        axis,
        set(train_ids) & valid_ids,
    )
    return ModifierRefit(
        axis=axis,
        modifier=modifier,
        ssa_vintage=vintage,
        provenance=RefitProvenance(
            boundary_year=boundary_year,
            estimation_rule=(
                "permanent earnings capacity and marital flows refit <=T*"
            ),
            n_rows={
                "earnings": len(earnings),
                "marital_person_years": len(mpanel.person_years),
                "marital_events": len(mpanel.events),
            },
            max_year={
                "earnings": _max_year(earnings, "period"),
                "marital": _max_year(mpanel.person_years, "year"),
            },
        ),
    )


def refit_disability(
    panel: disability.DisabilityPanel,
    *,
    train_ids: set[int] | None = None,
    boundary_year: int = BOUNDARY_YEAR,
    fitter: Callable[[disability.DisabilityPanel, set[int]], Any] = m4.fit,
) -> Any:
    """Route a <=``T*`` panel to M4 after rebuilding adjacent-wave pairs."""
    person_years = _truncate_optional_flow(
        panel.person_years,
        year_column="period",
        boundary_year=boundary_year,
        label="disability person-years",
    )
    pairs = disability.build_transition_pairs(person_years)
    truncated = disability.DisabilityPanel(
        person_years=person_years,
        pairs=pairs,
    )
    present = set(person_years["person_id"])
    ids = present if train_ids is None else present & set(train_ids)
    return fitter(truncated, ids)


def claiming_pmfs_from_reference(
    reference: claiming.ClaimAgeReference,
    *,
    boundary_year: int = BOUNDARY_YEAR,
) -> dict[tuple[str, int], dict[int, float]]:
    """Materialize <=``T*`` claim PMFs from a caller-supplied vintage."""
    validate_external_vintage(
        "claiming reference",
        reference.supplement_year,
        boundary_year=boundary_year,
    )
    years = [year for year in reference.years() if year <= boundary_year]
    if not years:
        raise ValueError("claiming reference has no entitlement year <=T*")
    return {
        (sex, year): claiming.claim_age_pmf(
            sex,
            year,
            exclude_conversions=True,
            reference=reference,
        )
        for sex in ("female", "male")
        for year in years
    }


def prepare_mortality_refit_inputs(
    exposure: pd.DataFrame,
    external_rates: pd.DataFrame,
    *,
    external_vintage_year: int,
    boundary_year: int = BOUNDARY_YEAR,
    event_year_column: str = "event_year",
) -> MortalityRefitInputs:
    """Validate mortality's <=``T*`` flow support and external vintage."""
    vintage = validate_external_vintage(
        "mortality reference",
        external_vintage_year,
        boundary_year=boundary_year,
    )
    truncated = truncate_estimation_frame(
        exposure,
        boundary_year=boundary_year,
        year_column=event_year_column,
        flow=True,
        label="mortality exposure",
    )
    return MortalityRefitInputs(
        exposure=truncated,
        external_rates=external_rates.copy(),
        external_vintage=vintage,
        provenance=RefitProvenance(
            boundary_year=boundary_year,
            estimation_rule=(
                "death flow event and required dating interview <=T*; "
                "external mortality vintage <=T*"
            ),
            n_rows={
                "exposure": len(truncated),
                "external_rates": len(external_rates),
            },
            max_year={
                "mortality_event": _max_year(truncated, event_year_column)
            },
        ),
    )


def fit_mortality_model(
    prepared: MortalityRefitInputs,
) -> AgeSexMortalityModel:
    """Fit the NCHS x PSID-band annual mortality probabilities.

    Exposure rows use the fixed boundary weight.  The PSID-to-external central
    rate ratio is estimated within each band/sex and applied to the supplied
    external central rate; the product is converted to a one-year death
    probability.  Algebraically the fitted-window level equals the PSID
    central rate while retaining the explicit external-anchor decomposition.
    """
    exposure_columns = (
        "age_band",
        "sex",
        "start_weight",
        "exposure",
        "death",
    )
    external_columns = (
        "lower_age",
        "upper_age",
        "age_band",
        "sex",
        "central_rate",
    )
    _require_columns(prepared.exposure, exposure_columns, "mortality exposure")
    _require_columns(
        prepared.external_rates,
        external_columns,
        "external mortality rates",
    )
    external = prepared.external_rates.copy()
    bands = tuple(
        sorted(
            {
                (int(row.lower_age), int(row.upper_age))
                for row in external.itertuples(index=False)
            }
        )
    )
    probability: dict[tuple[str, str], float] = {}
    for row in external.itertuples(index=False):
        cell = prepared.exposure[
            (prepared.exposure["age_band"] == row.age_band)
            & (prepared.exposure["sex"] == row.sex)
        ]
        weighted_exposure = float(
            (cell["start_weight"] * cell["exposure"]).sum()
        )
        weighted_deaths = float((cell["start_weight"] * cell["death"]).sum())
        if weighted_exposure <= 0 or float(row.central_rate) <= 0:
            raise ValueError(
                f"undefined mortality fit cell {row.age_band}|{row.sex}"
            )
        psid_rate = weighted_deaths / weighted_exposure
        ratio = psid_rate / float(row.central_rate)
        aligned_rate = float(row.central_rate) * ratio
        probability[(str(row.age_band), str(row.sex))] = float(
            -np.expm1(-aligned_rate)
        )
    return AgeSexMortalityModel(bands=bands, probability=probability)


def refit_m6_components(
    inputs: M6RefitInputs,
    *,
    boundary_year: int = BOUNDARY_YEAR,
    qrf_factory: QRFModelFactory | None = None,
    earnings_candidate_spec: CandidateSpec | None = None,
    family_registry: Any = ft.REGISTRY,
    household_registry: Any = hc.REGISTRY,
    disability_fitter: Callable[
        [disability.DisabilityPanel, set[int]], Any
    ] = m4.fit,
) -> M6RefitBundle:
    """Refit every composed M6 object without reading or writing artifacts.

    The same immutable registry specs are selected, but every estimate and
    permanent-axis input is rebuilt after the single ``<=T*`` truncation
    policy.  External claiming, mortality, and SSA-index vintages must also be
    admissible at the boundary.
    """
    family = refit_family_transitions(
        inputs.family_context,
        boundary_year=boundary_year,
        registry=family_registry,
    )
    household = refit_household_composition(
        inputs.household_context,
        boundary_year=boundary_year,
        registry=household_registry,
    )
    earnings = refit_earnings_chained_generator(
        inputs.earnings_panel,
        inputs.ssa_params.nawi,
        seed=inputs.earnings_seed,
        boundary_year=boundary_year,
        qrf_factory=qrf_factory,
        candidate_spec=earnings_candidate_spec,
    )
    modifier = refit_first_marriage_modifier(
        family.fitted,
        inputs.modifier_marital_panel,
        inputs.earnings_panel,
        inputs.modifier_marriage_records,
        inputs.modifier_person_weight,
        inputs.ssa_params,
        params_vintage=inputs.ssa_params_vintage,
        train_ids=inputs.modifier_train_ids,
        interview_years=inputs.modifier_interview_years,
        boundary_year=boundary_year,
    )
    disability_fit = refit_disability(
        inputs.disability_panel,
        train_ids=inputs.disability_train_ids,
        boundary_year=boundary_year,
        fitter=disability_fitter,
    )
    claiming_pmfs = claiming_pmfs_from_reference(
        inputs.claiming_reference,
        boundary_year=boundary_year,
    )
    mortality = prepare_mortality_refit_inputs(
        inputs.mortality_exposure,
        inputs.mortality_external_rates,
        external_vintage_year=inputs.mortality_external_vintage,
        boundary_year=boundary_year,
    )
    return M6RefitBundle(
        boundary_year=boundary_year,
        family=family,
        household=household,
        earnings=earnings,
        modifier=modifier,
        disability=disability_fit,
        claiming_pmfs=claiming_pmfs,
        mortality=mortality,
    )
