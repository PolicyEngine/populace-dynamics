"""Pure report-only diagnostics for the M6 temporal-holdout run.

Nothing in this module can change a gate verdict.  The functions accept frames
or already-reduced summaries produced by the runner and return JSON-ready
records for phase 5 of the M6 harness.  In particular, missing successor-gate
machinery is represented as unavailable; it is never replaced by a zero,
empty comparison, or synthetic PASS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.harness.m6_cells import (
    SHOCK_EARN_YEARS,
    SHOCK_FLOW_YEARS,
)

__all__ = [
    "NOT_CERTIFIED_ORDER",
    "assemble_report_only_payload",
    "build_alignment_displacement",
    "build_entrant_diagnostics",
    "build_mortality_anchor_disclosure",
    "build_not_certified_surface",
    "build_redrawn_seed_comparison",
    "build_shock_window_diagnostics",
    "compare_diagnostic_cells",
]


SHOCK_REASON = "exogenous_shock_outside_model_class"
REDRAWN_UNAVAILABLE = "successor_forward_seed_machinery_out_of_scope"
REDRAWN_MARGIN_UNRESOLVED = "pre_named_margin_absent_from_ratified_spec"

# The first nine entries follow the reporting order required by sections 2.8.7
# and 4.10, including the earnings-survivorship channel made explicit by F1.
# The last two retain the standing registry disclosures at the same prominence
# as the PASS claim.  They are declarations, not data-dependent conclusions.
NOT_CERTIFIED_ORDER: tuple[str, ...] = (
    "mortality_drift",
    "widowhood",
    "shock_window_2020_2022",
    "entrants_open_panel",
    "autocorrelation_lag5",
    "forward_projection_2100_extrapolation",
    "stock_margins",
    "remarriage_above_pooled_working_age",
    "forward_earnings_survivorship",
    "spec_selection_in_sample",
    "forward_earnings_law_not_gate1_certified",
)

_NOT_CERTIFIED_DETAILS = {
    "mortality_drift": (
        "Mortality drift is not certified. It remains report-only and is the "
        "first limitation named beside the PASS claim."
    ),
    "widowhood": (
        "Widowhood does not clear the admissible pooled surface and remains "
        "report-only."
    ),
    "shock_window_2020_2022": (
        "The 2020-2022 shock window is outside the engine model class and is "
        "partitioned out of every gated set."
    ),
    "entrants_open_panel": (
        "The gate covers the closed panel only; synthetic births, immigrant "
        "cohorts, and other open additions remain report-only."
    ),
    "autocorrelation_lag5": (
        "Lag-5 persistence exceeds the eight-year holdout span and is not "
        "certified."
    ),
    "forward_projection_2100_extrapolation": (
        "The gate covers the temporal holdout only, not extrapolation through "
        "2100."
    ),
    "stock_margins": (
        "End-window marital and disability stock margins are report-only."
    ),
    "remarriage_above_pooled_working_age": (
        "Only pooled working-age remarriage is gated; the 65+ tail remains "
        "report-only."
    ),
    "forward_earnings_survivorship": (
        "Gated earnings use realized support and do not certify mortality's "
        "effect on the earnings composition through survivorship."
    ),
    "spec_selection_in_sample": (
        "The gated surface structure was selected on the full window; the "
        "holdout is out of sample for fitted parameters, not structure."
    ),
    "forward_earnings_law_not_gate1_certified": (
        "The forward earnings law is first certified, if at all, by M6; no "
        "gate-1 backward-law certificate transfers."
    ),
}

_ENTRANT_BRIDGES = {
    "synthetic_births": {
        "requirement": "fertility_to_person_roster_materialization",
        "detail": (
            "Fertility births must be materialized as synthetic person rows "
            "in their birth year before an entrant margin can be interpreted."
        ),
    },
    "immigrant_cohorts": {
        "requirement": "immigrant_cohort_to_person_roster_materialization",
        "detail": (
            "Immigrant entry cohorts require an explicit entry-year person-row "
            "bridge; no PSID truth row may be imputed for them."
        ),
    },
}


def _json_scalar(value: Any) -> Any:
    """Convert common numpy/pandas scalar values to JSON-safe Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return _json_scalar(value)


def _cell_scalar(value: Any) -> float | None:
    """Extract the conventional scalar from a diagnostic cell summary."""
    if isinstance(value, Mapping):
        for key in ("rate", "value", "score"):
            if key in value:
                return _cell_scalar(value[key])
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def compare_diagnostic_cells(
    projected: Mapping[str, Any] | None,
    truth: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compare reduced projected/truth cells without applying a threshold.

    The original cell records are preserved, while ``absolute_gap`` and
    ``absolute_log_ratio`` are conveniences for numeric rate/value cells.  A
    missing or non-positive value yields ``None`` rather than an invented
    diagnostic.
    """
    projected = projected or {}
    truth = truth or {}
    out: list[dict[str, Any]] = []
    for cell in sorted(set(projected) | set(truth)):
        projected_record = projected.get(cell)
        truth_record = truth.get(cell)
        p_value = _cell_scalar(projected_record)
        t_value = _cell_scalar(truth_record)
        absolute_gap = (
            abs(p_value - t_value)
            if p_value is not None and t_value is not None
            else None
        )
        absolute_log_ratio = (
            abs(float(np.log(p_value / t_value)))
            if p_value is not None
            and t_value is not None
            and p_value > 0
            and t_value > 0
            else None
        )
        out.append(
            {
                "cell": str(cell),
                "projected": _json_value(projected_record),
                "truth": _json_value(truth_record),
                "absolute_gap": _json_scalar(absolute_gap),
                "absolute_log_ratio": _json_scalar(absolute_log_ratio),
                "status": (
                    "computed"
                    if p_value is not None and t_value is not None
                    else "missing_or_non_numeric"
                ),
            }
        )
    return out


def build_shock_window_diagnostics(
    *,
    flow_projected: Mapping[str, Any] | None = None,
    flow_truth: Mapping[str, Any] | None = None,
    earnings_projected: Mapping[str, Any] | None = None,
    earnings_truth: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the dual-axis shock cells as a non-gating diagnostic."""
    flow = compare_diagnostic_cells(flow_projected, flow_truth)
    earnings = compare_diagnostic_cells(earnings_projected, earnings_truth)
    return {
        "status": "computed" if flow or earnings else "not_computed",
        "gated": False,
        "machine_reason": SHOCK_REASON,
        "axes": {
            "flows": {
                "axis": "event_year",
                "years": list(SHOCK_FLOW_YEARS),
            },
            "earnings": {
                "axis": "reference_year",
                "years": list(SHOCK_EARN_YEARS),
                "unobserved_reference_years": [2021],
            },
        },
        "flow_cells": flow,
        "earnings_cells": earnings,
        "disposition": "published_report_only_never_gated",
    }


def build_not_certified_surface(
    measurements: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the ordered negative surface that accompanies any verdict."""
    measurements = measurements or {}
    unknown = set(measurements) - set(NOT_CERTIFIED_ORDER)
    if unknown:
        raise ValueError(
            "unknown not_certified measurements: "
            f"{sorted(str(item) for item in unknown)}"
        )
    return [
        {
            "order": order,
            "margin": margin,
            "status": "not_certified",
            "gated": False,
            "detail": _NOT_CERTIFIED_DETAILS[margin],
            "measurement": _json_value(measurements.get(margin)),
        }
        for order, margin in enumerate(NOT_CERTIFIED_ORDER, start=1)
    ]


def _count_record(value: int | Mapping[int, int]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        by_year = {int(year): int(count) for year, count in value.items()}
        if any(count < 0 for count in by_year.values()):
            raise ValueError("entrant counts cannot be negative")
        return {
            "total": int(sum(by_year.values())),
            "by_year": [
                {"year": year, "count": by_year[year]}
                for year in sorted(by_year)
            ],
        }
    count = int(value)
    if count < 0:
        raise ValueError("entrant counts cannot be negative")
    return {"total": count, "by_year": None}


def build_entrant_diagnostics(
    *,
    synthetic_births: int | Mapping[int, int],
    immigrant_cohorts: int | Mapping[int, int],
    later_earnings_entrants: int | Mapping[int, int],
    marked_no_earnings_state: int | Mapping[int, int],
    synthetic_person_ids: int | Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Publish family-B and earnings-only open-addition counts.

    ``later_earnings_entrants`` are deliberately kept separate from family B:
    they remain closed-panel members for flow modules but are open additions
    for the 2014-anchored earnings chain.  Counts must come from explicit
    runner markers; this function performs no demographic inference.
    """
    family_b = {
        "synthetic_births": _count_record(synthetic_births),
        "immigrant_cohorts": _count_record(immigrant_cohorts),
    }
    if synthetic_person_ids is not None:
        family_b["synthetic_person_ids"] = _count_record(synthetic_person_ids)
    return {
        "status": "reported",
        "gated": False,
        "family_b_open_additions": family_b,
        "earnings_module_open_additions": {
            "later_earnings_entrants": _count_record(later_earnings_entrants),
            "marked_no_earnings_state": _count_record(
                marked_no_earnings_state
            ),
            "classification": (
                "closed_panel_for_flows_open_addition_for_earnings_only"
            ),
            "non_scored_seam_rule": "earnings_zero_with_domain_marker_false",
            "benefit_level_disclosure": (
                "AIME, PIA, and claiming levels are understated for marked "
                "people and must not be read at face value."
            ),
        },
        "bridge_requirements": _json_value(_ENTRANT_BRIDGES),
        "truth_basis": "no_psid_truth_for_family_b_open_additions",
    }


def build_alignment_displacement(
    before: pd.DataFrame | None,
    after: pd.DataFrame | None,
    *,
    key_columns: Sequence[str] = ("person_id", "year"),
    value_columns: Sequence[str] = (),
    unavailable_reason: str = "alignment_layer_output_not_collected",
) -> dict[str, Any]:
    """Compute per-year maximum report-only alignment displacement.

    ``before`` is the unaligned (and scored) path; ``after`` is the aligned
    report-only path.  Exact key equality is required so a missing row cannot
    masquerade as zero displacement.  If neither frame is supplied, an
    explicit unavailable record is returned.
    """
    if before is None and after is None:
        return {
            "status": "not_computed",
            "reason": unavailable_reason,
            "gated": False,
            "scored_path": "unaligned",
            "alignment_path": "report_only",
            "per_year_maximum": None,
            "maximum_alignment_displacement": None,
            "n_intervened_rows": None,
        }
    if before is None or after is None:
        raise ValueError("before and after must be supplied together")
    keys = tuple(key_columns)
    values = tuple(value_columns)
    if "year" not in keys:
        raise ValueError("key_columns must include year")
    if not values:
        raise ValueError("value_columns must name at least one aligned field")
    required = set(keys) | set(values)
    for label, frame in (("before", before), ("after", after)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} alignment frame is missing {sorted(missing)}"
            )
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{label} alignment keys are not unique")

    merged = before[list(keys) + list(values)].merge(
        after[list(keys) + list(values)],
        on=list(keys),
        how="outer",
        suffixes=("_before", "_after"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("alignment before/after keys differ")

    displacement_columns: list[str] = []
    for field in values:
        before_values = pd.to_numeric(
            merged[f"{field}_before"], errors="raise"
        )
        after_values = pd.to_numeric(merged[f"{field}_after"], errors="raise")
        displacement = f"__displacement_{field}"
        merged[displacement] = (after_values - before_values).abs()
        displacement_columns.append(displacement)

    merged["__row_max"] = merged[displacement_columns].max(axis=1)
    per_year: list[dict[str, Any]] = []
    for year, group in merged.groupby("year", sort=True):
        per_field = {
            field: float(group[f"__displacement_{field}"].max())
            for field in values
        }
        per_year.append(
            {
                "year": int(year),
                "maximum": float(group["__row_max"].max()),
                "by_field": per_field,
            }
        )
    maximum = float(merged["__row_max"].max()) if len(merged) else 0.0
    n_intervened = int((merged["__row_max"] > 0).sum())
    return {
        "status": "computed",
        "gated": False,
        "scored_path": "unaligned",
        "alignment_path": "report_only",
        "value_columns": list(values),
        "per_year_maximum": per_year,
        "maximum_alignment_displacement": maximum,
        "n_intervened_rows": n_intervened,
        "disclosure": (
            "Gate scores use the unaligned projection; alignment intervention "
            "magnitudes are published only."
        ),
    }


def build_redrawn_seed_comparison(
    *,
    realized_seed_cells: Mapping[str, Any] | None,
    redrawn_seed_cells: Mapping[str, Any] | None = None,
    margin_name: str | None = None,
    margin_bound: float | None = None,
) -> dict[str, Any]:
    """Publish the decision-5b comparison without inventing its missing law.

    Revision 7 requires a pre-named margin but supplies neither its identifier
    nor bound, and section 2.8.1 explicitly excludes the successor forward-seed
    machinery that would produce the re-drawn state.  Callers may supply those
    registered values later.  Until then the artifact records the gap rather
    than fabricating a comparison or a margin result.
    """
    margin = {
        "name": margin_name,
        "bound": _json_scalar(margin_bound),
        "status": (
            "supplied"
            if margin_name is not None and margin_bound is not None
            else "unresolved"
        ),
        "unresolved_reason": (
            None
            if margin_name is not None and margin_bound is not None
            else REDRAWN_MARGIN_UNRESOLVED
        ),
        "gated": False,
        "successor_flip_seed": True,
    }
    if redrawn_seed_cells is None:
        return {
            "status": "unavailable",
            "reason": REDRAWN_UNAVAILABLE,
            "gated": False,
            "realized_seed_cells_published": _json_value(realized_seed_cells),
            "redrawn_seed_cells": None,
            "comparison": None,
            "margin": margin,
            "pass": None,
        }

    comparison = compare_diagnostic_cells(
        redrawn_seed_cells, realized_seed_cells
    )
    return {
        "status": "computed",
        "reason": None,
        "gated": False,
        "realized_seed_cells_published": _json_value(realized_seed_cells),
        "redrawn_seed_cells": _json_value(redrawn_seed_cells),
        "comparison": comparison,
        "margin": margin,
        # This report-only module never converts the comparison into a verdict.
        "pass": None,
    }


def build_mortality_anchor_disclosure(
    anchor_comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the SSA/NCHS anchor and its calibration circularity."""
    return {
        "deliverable": "ssa_nchs_life_table_mortality_anchor",
        "status": (
            "computed" if anchor_comparison is not None else "not_supplied"
        ),
        "gated": False,
        "required_before_m7_lock_flip": True,
        "anchor_sources": ["SSA_life_table", "NCHS_life_table"],
        "comparison": _json_value(anchor_comparison),
        "circularity_disclosure": (
            "The engine mortality input is itself NCHS x PSID-band anchored. "
            "Agreement with the SSA/NCHS anchor validates the alignment path, "
            "not independent mortality drift."
        ),
        "external_level_log_ratio_gated": False,
    }


def assemble_report_only_payload(
    *,
    shock_window: Mapping[str, Any],
    not_certified: Sequence[Mapping[str, Any]],
    entrants: Mapping[str, Any],
    alignment: Mapping[str, Any],
    redrawn_seed: Mapping[str, Any],
    mortality_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble phase-5 records and enforce their non-gating disposition."""
    components: dict[str, Any] = {
        "shock_window": _json_value(shock_window),
        "not_certified": _json_value(list(not_certified)),
        "entrants": _json_value(entrants),
        "alignment_displacement": _json_value(alignment),
        "redrawn_t_star_seed_comparison": _json_value(redrawn_seed),
        "mortality_anchor": _json_value(mortality_anchor),
    }
    records: list[Mapping[str, Any]] = [
        shock_window,
        entrants,
        alignment,
        redrawn_seed,
        mortality_anchor,
        *not_certified,
    ]
    if any(record.get("gated") is not False for record in records):
        raise ValueError("every report-only component must set gated=False")
    return {
        "phase": "report_only",
        "gated": False,
        "changes_gate_verdict": False,
        "publishes_regardless": True,
        **components,
    }
