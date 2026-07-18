"""Pure scoring machinery for the M6 temporal-holdout harness.

The module reduces already-prepared realized or projected long frames to the
locked v3 cell surface.  It never reads PSID data, runs the projection engine,
or chooses thresholds.  In particular, the earnings-domain filter is explicit
and symmetric: callers must supply projected values on the realized support,
then both sides are intersected with the 2014-anchored earnings domain.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.harness import panel as hpanel
from populace_dynamics.harness.m6_cells import (
    METRIC_CAP,
    WEAK_POWER_P_GATE_FLOOR,
    _score,
    _tol,
    coarsened_disability_cells,
    coarsened_marital_cells,
    earnings_cells,
    oc_4of5,
    run_floor,
)

N_DRAWS = 20
DRAW_SEED_BASE = 5200
GATE_SEEDS = (0, 1, 2, 3, 4)
REQUIRED_SEED_PASSES = 4
FROZEN_FLOOR_RUN = "runs/m6_holdout_floors_v4.json"
FROZEN_FLOOR_SHA256 = (
    "4cd2d01a9fd76064e701ae77a9226208cbae94d743f76f502d3d0a5f657d9523"
)

GATED_CELL_NAMES = (
    "divorce.18-44",
    "earn_autocorr_lag2",
    "earn_dlog_mean.prime",
    "earn_dlog_sd.older",
    "earn_mob_h1_diag",
    "earn_p10.prime",
    "earn_zero_rate.older",
    "first_marriage.18-29|female",
    "incidence.20-66",
    "recovery.20-66",
    "remarriage.18-64",
)

EARNINGS_CELL_NAMES = tuple(
    cell for cell in GATED_CELL_NAMES if cell.startswith("earn_")
)

_FIRST_MARRIAGE_BANDS = ((18, 29), (30, 44), (45, 64), (65, 120))
_DIVORCE_BANDS = ((18, 44), (45, 120))
_REMARRIAGE_BANDS = ((18, 64), (65, 120))
_DISABILITY_BANDS = ((20, 66),)
_CONJUNCTION = re.compile(r"(?P<required>\d+)\s+of\s+(?P<total>\d+)")


@dataclass(frozen=True)
class M6CellRule:
    """One locked v3 cell and its scoring metadata."""

    cell: str
    family: str
    split_unit: str
    metric: str
    tolerance: float
    k: int
    rounding: int


@dataclass(frozen=True)
class M6GateContract:
    """The read-only scoring fields extracted from ``gates.gate_m6``."""

    cells: tuple[M6CellRule, ...]
    gate_seeds: tuple[int, ...]
    required_seed_passes: int
    n_draws: int = N_DRAWS
    draw_seed_base: int = DRAW_SEED_BASE
    floor_run: str = FROZEN_FLOOR_RUN
    floor_run_sha256: str = ""

    @property
    def by_cell(self) -> dict[str, M6CellRule]:
        return {rule.cell: rule for rule in self.cells}

    @property
    def cell_names(self) -> tuple[str, ...]:
        return tuple(rule.cell for rule in self.cells)

    @property
    def draw_seeds(self) -> tuple[int, ...]:
        return tuple(self.draw_seed_base + k for k in range(self.n_draws))

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> M6GateContract:
        """Validate only the locked protocol/cell fields the harness may read."""
        if block.get("locked") is not True or block.get("status") != "locked":
            raise ValueError("gate_m6 must be locked")
        floor_run = block.get("floor_run")
        floor_sha = block.get("floor_run_sha256")
        if (floor_run, floor_sha) != (
            FROZEN_FLOOR_RUN,
            FROZEN_FLOOR_SHA256,
        ):
            raise ValueError("gate_m6 frozen v3 floor binding changed")

        raw_views = block.get("views")
        if not isinstance(raw_views, Mapping):
            raise ValueError("gate_m6 views must be a mapping")
        rules: list[M6CellRule] = []
        for raw_view in raw_views.values():
            if not isinstance(raw_view, Mapping):
                raise ValueError("gate_m6 view must be a mapping")
            family = str(raw_view.get("family", ""))
            split_unit = str(raw_view.get("split_unit", ""))
            if raw_view.get("quantity_type") != "flow":
                raise ValueError("gate_m6 scored views must remain flows")
            tolerances = raw_view.get("tolerances")
            derivations = raw_view.get("derivations")
            if not isinstance(tolerances, Mapping) or not isinstance(
                derivations, Mapping
            ):
                raise ValueError("gate_m6 view lacks locked derivations")
            if (
                raw_view.get("floor_run") != floor_run
                or derivations.get("floor_run") != floor_run
            ):
                raise ValueError("gate_m6 view does not use the frozen floor")
            raw_rules = derivations.get("rules")
            if not isinstance(raw_rules, Mapping) or set(raw_rules) != set(
                tolerances
            ):
                raise ValueError("gate_m6 derivation keys do not match")
            for cell, raw_tolerance in tolerances.items():
                derivation = raw_rules[cell]
                if not isinstance(derivation, Mapping):
                    raise ValueError(
                        f"gate_m6 derivation for {cell!r} is invalid"
                    )
                if derivation.get("key") != cell:
                    raise ValueError(
                        f"gate_m6 derivation key for {cell!r} changed"
                    )
                metric = str(derivation.get("metric", ""))
                if metric not in METRIC_CAP:
                    raise ValueError(f"unsupported M6 metric {metric!r}")
                tolerance = float(raw_tolerance)
                if (
                    int(derivation.get("k", -1)) != 3
                    or int(derivation.get("rounding", -1)) != 3
                ):
                    raise ValueError(
                        f"gate_m6 derivation protocol changed for {cell!r}"
                    )
                if (
                    not math.isfinite(tolerance)
                    or tolerance <= 0
                    or tolerance > METRIC_CAP[metric] + 1e-12
                ):
                    raise ValueError(
                        f"gate_m6 tolerance for {cell!r} is invalid"
                    )
                rules.append(
                    M6CellRule(
                        cell=str(cell),
                        family=family,
                        split_unit=split_unit,
                        metric=metric,
                        tolerance=tolerance,
                        k=int(derivation["k"]),
                        rounding=int(derivation.get("rounding", 3)),
                    )
                )

        rules.sort(key=lambda rule: rule.cell)
        if tuple(rule.cell for rule in rules) != GATED_CELL_NAMES:
            raise ValueError(
                "gate_m6 cells do not match the frozen v3 surface"
            )
        if len({rule.cell for rule in rules}) != len(rules):
            raise ValueError("gate_m6 contains duplicate cells")
        for rule in rules:
            expected_family = (
                "earnings"
                if rule.cell.startswith("earn_")
                else (
                    "disability"
                    if rule.cell.startswith(("incidence", "recovery"))
                    else "marital"
                )
            )
            expected_split = (
                "household" if expected_family == "marital" else "person"
            )
            if (rule.family, rule.split_unit) != (
                expected_family,
                expected_split,
            ):
                raise ValueError(
                    f"gate_m6 family/split changed for {rule.cell!r}"
                )

        scoring = block.get("scoring")
        if not isinstance(scoring, Mapping):
            raise ValueError("gate_m6 scoring protocol is missing")
        seeds = tuple(int(seed) for seed in scoring.get("gate_seeds", ()))
        if seeds != GATE_SEEDS:
            raise ValueError("gate_m6 gate seeds changed")
        match = _CONJUNCTION.search(str(scoring.get("conjunction", "")))
        if match is None:
            raise ValueError("gate_m6 conjunction is not machine-readable")
        required = int(match.group("required"))
        total = int(match.group("total"))
        if (required, total) != (REQUIRED_SEED_PASSES, len(seeds)):
            raise ValueError("gate_m6 conjunction changed")
        if scoring.get("mixed_k") != {"flow": 3, "stock": 4, "margin": 3}:
            raise ValueError("gate_m6 mixed-k protocol changed")
        return cls(
            cells=tuple(rules),
            gate_seeds=seeds,
            required_seed_passes=required,
            floor_run=floor_run,
            floor_run_sha256=floor_sha,
        )


@dataclass(frozen=True)
class M6CellScore:
    """One cell's K-draw reduction at one gate seed."""

    cell: str
    metric: str
    tolerance: float
    rate_a: float
    per_draw_rates: tuple[float | None, ...]
    rbar: float | None
    score: float | None
    passed: bool
    undefined_draw_indices: tuple[int, ...]
    per_draw_rate_sd: float | None
    max_per_draw_score: float | None
    max_per_draw_abs_ln: float | None
    regenerated: bool

    def to_artifact(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "rate_a": self.rate_a,
            "per_draw_rate": list(self.per_draw_rates),
            "rbar": self.rbar,
            "score": self.score,
            "tolerance": self.tolerance,
            "pass": self.passed,
            "undefined_draw_indices": list(self.undefined_draw_indices),
            "per_draw_rate_sd": self.per_draw_rate_sd,
            "max_per_draw_score": self.max_per_draw_score,
            "max_per_draw_abs_ln": self.max_per_draw_abs_ln,
            "regenerated": self.regenerated,
        }


@dataclass(frozen=True)
class M6SeedScore:
    """All 11 cells and the all-cell conjunction for one gate seed."""

    seed: int
    cells: tuple[M6CellScore, ...]
    n_side_a_units: int | None
    n_cells_passed: int
    undefined_draw_cells: tuple[str, ...]
    non_regenerated_cells: tuple[str, ...]
    valid: bool
    passed: bool

    def to_artifact(self, *, worst_n: int = 5) -> dict[str, Any]:
        def severity(cell: M6CellScore) -> float:
            if cell.score is None:
                return float("inf")
            return (
                cell.score / cell.tolerance
                if cell.tolerance > 0
                else float("inf")
            )

        worst = sorted(self.cells, key=severity, reverse=True)[:worst_n]
        return {
            "seed": self.seed,
            "n_side_a_units": self.n_side_a_units,
            "seed_pass": self.passed,
            "valid": self.valid,
            "n_cells_pass": self.n_cells_passed,
            "n_cells_fail": len(self.cells) - self.n_cells_passed,
            "undefined_draw_cells": list(self.undefined_draw_cells),
            "non_regenerated_cells": list(self.non_regenerated_cells),
            "worst_cells": [
                {
                    "cell": cell.cell,
                    "score": cell.score,
                    "tolerance": cell.tolerance,
                    "score_over_tolerance": (
                        None if cell.score is None else severity(cell)
                    ),
                }
                for cell in worst
            ],
            "gated_cells": {
                cell.cell: cell.to_artifact() for cell in self.cells
            },
        }


@dataclass(frozen=True)
class M6GateScore:
    """The registered 4-of-5 verdict plus run-conformance guards."""

    seeds: tuple[M6SeedScore, ...]
    n_seeds_passed: int
    valid: bool
    passed: bool

    def to_artifact(self) -> dict[str, Any]:
        cells = [cell for seed in self.seeds for cell in seed.cells]
        draw_sds = [
            cell.per_draw_rate_sd
            for cell in cells
            if cell.per_draw_rate_sd is not None
        ]
        draw_abs_logs = [
            cell.max_per_draw_abs_ln
            for cell in cells
            if cell.max_per_draw_abs_ln is not None
        ]
        regenerated_surface = bool(cells) and all(
            cell.regenerated for cell in cells
        )
        identity_candidate = bool(cells) and all(
            not cell.undefined_draw_indices
            and all(
                value is not None and value == cell.rate_a
                for value in cell.per_draw_rates
            )
            for cell in cells
        )
        return {
            "valid": self.valid,
            "pass": self.passed,
            "n_seeds_pass": self.n_seeds_passed,
            "seed_pass": {str(seed.seed): seed.passed for seed in self.seeds},
            "conformance": {
                "regenerated_surface": regenerated_surface,
                "identity_candidate": identity_candidate,
                "max_across_draw_sd": max(draw_sds) if draw_sds else None,
                "max_per_draw_abs_ln": (
                    max(draw_abs_logs) if draw_abs_logs else None
                ),
                "note": (
                    "every scored seed-cell has non-zero across-draw "
                    "dispersion; the projected surface is regenerated"
                    if regenerated_surface
                    else "at least one scored seed-cell has zero or "
                    "undefined across-draw dispersion; the run is "
                    "non-conformant"
                ),
            },
            "per_seed": [seed.to_artifact() for seed in self.seeds],
        }


def earnings_domain_person_ids(
    population_ids: Iterable[object],
    realized_earn_2014_by_person: Mapping[object, Any],
    u_w_by_person: Mapping[object, Any],
) -> frozenset[object]:
    """Return the exact 2014-state domain required by the certified initializer."""
    return (
        frozenset(population_ids)
        & frozenset(realized_earn_2014_by_person)
        & frozenset(u_w_by_person)
    )


def side_a_person_ids(
    anchor: pd.DataFrame,
    *,
    split_unit: str,
    seed: int,
) -> frozenset[object]:
    """Select side A under the contract's person/household split."""
    split_column = {
        "person": "person_id",
        "household": "household_id",
    }.get(split_unit)
    if split_column is None:
        raise ValueError(f"unsupported M6 split unit {split_unit!r}")
    missing = {"person_id", split_column} - set(anchor)
    if missing:
        raise ValueError(f"anchor is missing split columns {sorted(missing)}")
    columns = (
        ["person_id"]
        if split_column == "person_id"
        else [
            "person_id",
            split_column,
        ]
    )
    persons = anchor[columns].drop_duplicates("person_id")
    left, _ = hpanel.split_panel_by_person(
        persons,
        split_column,
        fraction=0.5,
        seed=seed,
    )
    return frozenset(left.person_id.tolist())


def restrict_earnings_domain_support(
    projection: pd.DataFrame,
    truth: pd.DataFrame,
    domain_person_ids: Iterable[object],
    *,
    periods: tuple[int, ...] = (2014, 2016, 2018),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect both sides with realized support and the earnings domain.

    The truth rows define the realized EXACT_WAVE-class support.  Projection
    rows outside it (including the live survivor frame's shape) are ignored;
    every realized in-domain key must nevertheless be present exactly once on
    the projected side, or scoring aborts.
    """
    required = {"person_id", "period"}
    for label, frame in (("projection", projection), ("truth", truth)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} earnings missing {sorted(missing)}")
    domain = frozenset(domain_person_ids)
    truth_side = truth[
        truth.person_id.isin(domain) & truth.period.isin(periods)
    ].copy()
    truth_keys = pd.MultiIndex.from_frame(truth_side[["person_id", "period"]])
    if truth_keys.has_duplicates:
        raise ValueError("truth earnings has duplicate person-period support")

    projection_domain = projection[
        projection.person_id.isin(domain) & projection.period.isin(periods)
    ].copy()
    projection_keys = pd.MultiIndex.from_frame(
        projection_domain[["person_id", "period"]]
    )
    if projection_keys.has_duplicates:
        raise ValueError(
            "projected earnings has duplicate person-period support"
        )
    projection_side = projection_domain.loc[
        projection_keys.isin(truth_keys)
    ].copy()
    selected_keys = pd.MultiIndex.from_frame(
        projection_side[["person_id", "period"]]
    )
    if set(selected_keys) != set(truth_keys):
        missing = list(set(truth_keys) - set(selected_keys))[:10]
        raise ValueError(
            "symmetric earnings scoring requires every realized in-domain "
            f"person-period on the projected side; missing {missing}"
        )
    sort = ["person_id", "period"]
    return (
        projection_side.sort_values(sort).reset_index(drop=True),
        truth_side.sort_values(sort).reset_index(drop=True),
    )


def _reduce_gated_cells(
    marital_events: pd.DataFrame,
    marital_person_years: pd.DataFrame,
    disability_transition_pairs: pd.DataFrame,
    earnings: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Reduce prepared long frames without choosing an empty disposition."""
    cells: dict[str, dict[str, Any]] = {}
    cells.update(
        coarsened_marital_cells(
            marital_events,
            marital_person_years,
            "first_marriage",
            _FIRST_MARRIAGE_BANDS,
            False,
        )
    )
    cells.update(
        coarsened_marital_cells(
            marital_events,
            marital_person_years,
            "divorce",
            _DIVORCE_BANDS,
            True,
        )
    )
    cells.update(
        coarsened_marital_cells(
            marital_events,
            marital_person_years,
            "remarriage",
            _REMARRIAGE_BANDS,
            True,
        )
    )
    cells.update(
        coarsened_disability_cells(
            disability_transition_pairs,
            "incidence",
            _DISABILITY_BANDS,
            True,
        )
    )
    cells.update(
        coarsened_disability_cells(
            disability_transition_pairs,
            "recovery",
            _DISABILITY_BANDS,
            True,
        )
    )
    cells.update(earnings_cells(earnings))
    return cells


def reduce_gated_cells(
    marital_events: pd.DataFrame,
    marital_person_years: pd.DataFrame,
    disability_transition_pairs: pd.DataFrame,
    earnings: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Strictly reduce a truth surface to the 11 locked v3 cells."""
    cells = _reduce_gated_cells(
        marital_events,
        marital_person_years,
        disability_transition_pairs,
        earnings,
    )
    missing = set(GATED_CELL_NAMES) - set(cells)
    if missing:
        raise ValueError(
            f"M6 reduction left gated cells undefined: {sorted(missing)}"
        )
    return {cell: dict(cells[cell]) for cell in GATED_CELL_NAMES}


def _undefined_projected_cell(name: str) -> dict[str, Any]:
    if not name.startswith("earn_"):
        return {"rate": None, "metric": "log_ratio", "undefined": True}
    metric = {
        "earn_autocorr_lag2": "abs_gap_corr",
        "earn_dlog_mean.prime": "abs_gap_log",
    }.get(name, "log_ratio")
    return {"value": None, "metric": metric, "undefined": True}


def reduce_projected_gated_cells(
    marital_events: pd.DataFrame,
    marital_person_years: pd.DataFrame,
    disability_transition_pairs: pd.DataFrame,
    earnings: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Reduce projection rows, typing omitted locked cells as undefined."""
    cells = _reduce_gated_cells(
        marital_events,
        marital_person_years,
        disability_transition_pairs,
        earnings,
    )
    for name in set(GATED_CELL_NAMES) - set(cells):
        cells[name] = _undefined_projected_cell(name)
    return {cell: dict(cells[cell]) for cell in GATED_CELL_NAMES}


def _cell_value(record: Mapping[str, Any]) -> float | None:
    raw = record.get("rate", record.get("value"))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def cell_value_bytes(
    cells: Mapping[str, Mapping[str, Any]],
    names: Iterable[str] = GATED_CELL_NAMES,
) -> bytes:
    """Canonical name + IEEE-754 bytes for a cell-function identity check."""
    payload = bytearray()
    for name in sorted(names):
        if name not in cells:
            raise ValueError(f"missing cell {name!r} from identity payload")
        value = _cell_value(cells[name])
        if value is None:
            raise ValueError(f"cell {name!r} is undefined in identity payload")
        encoded = name.encode("utf-8")
        payload.extend(struct.pack(">I", len(encoded)))
        payload.extend(encoded)
        payload.extend(struct.pack(">d", value))
    return bytes(payload)


def _candidate_record(value: float, metric: str) -> dict[str, Any]:
    if metric == "log_ratio":
        return {"rate": value, "metric": metric}
    return {"value": value, "metric": metric}


def score_gate_seed(
    contract: M6GateContract,
    *,
    seed: int,
    truth_cells: Mapping[str, Mapping[str, Any]],
    projected_draw_cells: Sequence[Mapping[str, Mapping[str, Any]]],
    n_side_a_units: int | None = None,
) -> M6SeedScore:
    """Score the K-draw mean once and enforce undefined/regeneration guards."""
    if seed not in contract.gate_seeds:
        raise ValueError(f"seed {seed} is outside the gate_m6 protocol")
    if len(projected_draw_cells) != contract.n_draws:
        raise ValueError(
            f"gate_m6 requires {contract.n_draws} draws, got "
            f"{len(projected_draw_cells)}"
        )
    expected = set(contract.cell_names)
    if set(truth_cells) != expected:
        raise ValueError("truth cells do not match the locked gate surface")

    results: list[M6CellScore] = []
    for rule in contract.cells:
        truth_value = _cell_value(truth_cells[rule.cell])
        if truth_value is None:
            raise ValueError(f"truth cell {rule.cell!r} is undefined")
        values: list[float | None] = []
        undefined: list[int] = []
        for draw_index, draw in enumerate(projected_draw_cells):
            record = draw.get(rule.cell)
            value = None if record is None else _cell_value(record)
            if value is None or (rule.metric == "log_ratio" and value <= 0):
                undefined.append(draw_index)
                values.append(None)
            else:
                values.append(value)

        if rule.metric == "log_ratio" and truth_value <= 0:
            undefined = list(range(contract.n_draws))
        defined = [value for value in values if value is not None]
        if undefined:
            rbar = None
            score = None
            sd = None
            max_score = None
            max_abs_ln = None
            regenerated = False
            passed = False
        else:
            rates = np.asarray(defined, dtype=np.float64)
            rbar = float(rates.mean())
            score = _score(
                {rule.cell: _candidate_record(rbar, rule.metric)},
                {rule.cell: dict(truth_cells[rule.cell])},
                rule.cell,
            )
            sd = float(rates.std(ddof=1)) if len(rates) > 1 else 0.0
            draw_scores = [
                _score(
                    {rule.cell: _candidate_record(float(value), rule.metric)},
                    {rule.cell: dict(truth_cells[rule.cell])},
                    rule.cell,
                )
                for value in rates
            ]
            finite_scores = [
                value for value in draw_scores if value is not None
            ]
            max_score = max(finite_scores) if finite_scores else None
            max_abs_ln = max_score if rule.metric == "log_ratio" else None
            regenerated = bool(np.any(rates != rates[0]))
            passed = score is not None and score <= rule.tolerance
        results.append(
            M6CellScore(
                cell=rule.cell,
                metric=rule.metric,
                tolerance=rule.tolerance,
                rate_a=truth_value,
                per_draw_rates=tuple(values),
                rbar=rbar,
                score=score,
                passed=passed,
                undefined_draw_indices=tuple(undefined),
                per_draw_rate_sd=sd,
                max_per_draw_score=max_score,
                max_per_draw_abs_ln=max_abs_ln,
                regenerated=regenerated,
            )
        )

    undefined_cells = tuple(
        result.cell for result in results if result.undefined_draw_indices
    )
    non_regenerated = tuple(
        result.cell
        for result in results
        if not result.undefined_draw_indices and not result.regenerated
    )
    valid = not undefined_cells and not non_regenerated
    n_passed = sum(result.passed for result in results)
    return M6SeedScore(
        seed=seed,
        cells=tuple(results),
        n_side_a_units=n_side_a_units,
        n_cells_passed=n_passed,
        undefined_draw_cells=undefined_cells,
        non_regenerated_cells=non_regenerated,
        valid=valid,
        passed=valid and n_passed == len(results),
    )


def aggregate_gate(
    contract: M6GateContract,
    seeds: Sequence[M6SeedScore],
) -> M6GateScore:
    """Apply the locked all-cell-per-seed and 4-of-5 conjunctions."""
    by_seed = {result.seed: result for result in seeds}
    if set(by_seed) != set(contract.gate_seeds) or len(by_seed) != len(seeds):
        raise ValueError("seed results do not match the gate_m6 protocol")
    ordered = tuple(by_seed[seed] for seed in contract.gate_seeds)
    valid = all(result.valid for result in ordered)
    n_passed = sum(result.passed for result in ordered)
    return M6GateScore(
        seeds=ordered,
        n_seeds_passed=n_passed,
        valid=valid,
        passed=valid and n_passed >= contract.required_seed_passes,
    )


def _frozen_cells(
    artifact: Mapping[str, Any] | None,
) -> Mapping[str, Mapping[str, Any]] | None:
    if artifact is None:
        return None
    floor = artifact.get("floor")
    if isinstance(floor, Mapping) and isinstance(floor.get("cells"), Mapping):
        return floor["cells"]
    if all(isinstance(value, Mapping) for value in artifact.values()):
        return artifact  # already the floor.cells mapping
    raise ValueError("frozen floor artifact has no floor.cells mapping")


def recompute_domain_earnings_floor(
    anchor: pd.DataFrame,
    earnings: pd.DataFrame,
    domain_person_ids: Iterable[object],
    contract: M6GateContract,
    *,
    frozen_floor_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reprice the earnings half-split on the 2014-state domain, report-only.

    The locked tolerances remain the scored contract.  This self-check publishes
    both the effective OC of those tolerances on the restricted domain and the
    counterfactual re-derived domain tolerances.  Either weak power (< 0.90) or
    the opposite-direction vacuity guard escalates to a floors-ceremony finding.
    """
    earnings_rules = [
        rule for rule in contract.cells if rule.family == "earnings"
    ]
    names = [rule.cell for rule in earnings_rules]
    if tuple(sorted(names)) != EARNINGS_CELL_NAMES:
        raise ValueError("contract earnings cells do not match v3")
    required_anchor = {"person_id", "household_id"}
    if missing := required_anchor - set(anchor):
        raise ValueError(f"anchor missing columns {sorted(missing)}")
    if "person_id" not in earnings:
        raise ValueError("earnings frame is missing person_id")

    domain = frozenset(domain_person_ids)
    domain_anchor = anchor[anchor.person_id.isin(domain)].copy()
    domain_earnings = earnings[earnings.person_id.isin(domain)].copy()
    if domain_anchor.empty:
        raise ValueError("earnings domain is empty")

    def compute(person_ids: set[object]) -> dict[str, Any]:
        domain_ids = set(person_ids) & domain
        return earnings_cells(
            domain_earnings[domain_earnings.person_id.isin(domain_ids)]
        )

    floor, _ = run_floor(anchor, compute, "person_id")
    missing_cells = set(names) - set(floor)
    if missing_cells:
        raise ValueError(
            "domain earnings floor left cells undefined: "
            f"{sorted(missing_cells)}"
        )
    selected_floor = {name: floor[name] for name in names}
    locked_tolerances = {rule.cell: rule.tolerance for rule in earnings_rules}
    raw_domain_tolerances = {
        rule.cell: _tol(
            selected_floor[rule.cell]["mean"],
            selected_floor[rule.cell]["sd"],
            rule.k,
        )
        for rule in earnings_rules
    }
    domain_tolerances = {
        rule.cell: min(
            raw_domain_tolerances[rule.cell], METRIC_CAP[rule.metric]
        )
        for rule in earnings_rules
    }
    effective_oc = oc_4of5(selected_floor, locked_tolerances, names)
    rederived_oc = oc_4of5(selected_floor, domain_tolerances, names)

    frozen = _frozen_cells(frozen_floor_artifact)
    frozen_selected = (
        None if frozen is None else {name: frozen[name] for name in names}
    )
    frozen_oc = (
        None
        if frozen_selected is None
        else oc_4of5(frozen_selected, locked_tolerances, names)
    )

    at_cap = [
        rule.cell
        for rule in earnings_rules
        if raw_domain_tolerances[rule.cell] >= METRIC_CAP[rule.metric] - 1e-12
    ]
    near_unfailable = [
        cell
        for cell, record in rederived_oc["per_cell"].items()
        if record["cell_pass_prob"] == 1.0
    ]
    near_tautological_oc = rederived_oc["p_gate_pass_4_of_5"] == 1.0 and bool(
        near_unfailable
    )
    weak_power = effective_oc["p_gate_pass_4_of_5"] < WEAK_POWER_P_GATE_FLOOR
    vacuity = bool(at_cap) or near_tautological_oc
    escalates = weak_power or vacuity

    per_cell: dict[str, Any] = {}
    for rule in earnings_rules:
        domain_stats = selected_floor[rule.cell]
        frozen_stats = (
            None if frozen_selected is None else frozen_selected[rule.cell]
        )
        per_cell[rule.cell] = {
            "metric": rule.metric,
            "locked_tolerance": rule.tolerance,
            "domain_raw_tolerance": raw_domain_tolerances[rule.cell],
            "domain_capped_tolerance": domain_tolerances[rule.cell],
            "tolerance_delta_vs_locked": (
                domain_tolerances[rule.cell] - rule.tolerance
            ),
            "at_metric_cap": rule.cell in at_cap,
            "domain_floor": domain_stats,
            "frozen_floor": frozen_stats,
            "realized_sigma_delta_vs_frozen": (
                None
                if frozen_stats is None
                else domain_stats["realized_sigma"]
                - frozen_stats["realized_sigma"]
            ),
        }

    frozen_p_gate = (
        None if frozen_oc is None else frozen_oc["p_gate_pass_4_of_5"]
    )
    return {
        "truth_side_only": True,
        "report_only": True,
        "frozen_tolerances_remain_gated_contract": True,
        "n_domain_persons": int(domain_anchor.person_id.nunique()),
        "n_domain_earnings_rows": int(len(domain_earnings)),
        "per_cell": per_cell,
        "oc": {
            "frozen_v3_earnings": frozen_oc,
            "locked_tolerances_on_domain": effective_oc,
            "domain_rederived_tolerances": rederived_oc,
            "locked_domain_delta_vs_frozen_v3": (
                None
                if frozen_p_gate is None
                else effective_oc["p_gate_pass_4_of_5"] - frozen_p_gate
            ),
        },
        "two_directional_escalation": {
            "weak_power_floor": WEAK_POWER_P_GATE_FLOOR,
            "near_unpassable": weak_power,
            "domain_tolerances_at_metric_cap": at_cap,
            "near_unfailable_cells": near_unfailable,
            "near_tautological_oc": near_tautological_oc,
            "vacuity": vacuity,
            "escalates_to_floors_ceremony_finding": escalates,
            "rule": (
                "escalate if the locked-tolerance domain earnings OC is below "
                "0.90 OR a domain-derived gated tolerance reaches its metric "
                "cap OR the rounded domain OC is 1.0 with a rounded-1.0 cell"
            ),
        },
    }
