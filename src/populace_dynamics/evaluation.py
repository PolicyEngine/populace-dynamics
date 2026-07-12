"""Generic evaluation of locked K-draw gate contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import Enum
from operator import index
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from populace_dynamics.contract import ContractRef

__all__ = [
    "ArtifactCube",
    "CellEvaluation",
    "ComparisonOperator",
    "DrawAggregation",
    "GateEvaluation",
    "GateSpec",
    "ReferenceRate",
    "ScoreStatistic",
    "ScoringSpec",
    "SeedEvaluation",
    "ThresholdSpec",
    "derive_tolerance",
    "evaluate_gate",
    "load_gate",
]

_DEFAULT_GATES_PATH = Path(__file__).resolve().parents[2] / "gates.yaml"
_CONJUNCTION_PATTERN = re.compile(
    r">=\s*(?P<required>\d+)\s+of(?:\s+the)?\s+(?P<total>\d+)",
    re.IGNORECASE,
)
_DRAW_STREAM_PATTERN = re.compile(r"(?P<base>\d+)\s*\+\s*k")


class DrawAggregation(str, Enum):
    """Supported reduction of registered draws."""

    MEAN_RATES_THEN_SCORE = "mean_rates_then_score"


class ScoreStatistic(str, Enum):
    """Supported cell score functions."""

    ABSOLUTE_LOG_RATIO = "absolute_log_ratio"


class ComparisonOperator(str, Enum):
    """Supported tolerance comparison operators."""

    LESS_THAN_OR_EQUAL = "<="


@dataclass(frozen=True)
class ScoringSpec:
    """Typed scoring and comparison semantics for a locked gate."""

    aggregation: DrawAggregation
    statistic: ScoreStatistic
    operator: ComparisonOperator


@dataclass(frozen=True)
class ThresholdSpec:
    """One locked tolerance and its exact floor derivation."""

    cell: str
    tolerance: Decimal
    floor_run: str
    floor_statistic: str
    floor_key: str
    k: Decimal
    rounding: int
    derived: Decimal


@dataclass(frozen=True)
class ReferenceRate:
    """One locked floor reference rate for a gated cell and seed."""

    cell: str
    seed: int
    rate: float


@dataclass(frozen=True)
class GateSpec:
    """Typed scoring surface loaded from one locked gate block."""

    name: str
    contract: ContractRef
    floor_run: str
    n_draws: int
    draw_seeds: tuple[int, ...]
    seeds: tuple[int, ...]
    required_seed_passes: int
    scoring: ScoringSpec
    thresholds: tuple[ThresholdSpec, ...]
    reference_rates: tuple[ReferenceRate, ...]

    @property
    def cells(self) -> tuple[str, ...]:
        """Return gated cell names in canonical order."""
        return tuple(threshold.cell for threshold in self.thresholds)

    @property
    def threshold_by_cell(self) -> dict[str, ThresholdSpec]:
        """Return threshold metadata indexed by gated cell."""
        return {threshold.cell: threshold for threshold in self.thresholds}

    @property
    def reference_by_seed(self) -> dict[int, dict[str, float]]:
        """Return locked floor rates indexed by seed and cell."""
        references = {seed: {} for seed in self.seeds}
        for reference in self.reference_rates:
            references[reference.seed][reference.cell] = reference.rate
        return references


@dataclass(frozen=True)
class ArtifactCube:
    """A canonical draw-by-cell-by-seed rate cube."""

    draw_seeds: tuple[int, ...]
    cells: tuple[str, ...]
    seeds: tuple[int, ...]
    rates: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the cube's draw, cell, and seed dimensions."""
        return (len(self.draw_seeds), len(self.cells), len(self.seeds))

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any]) -> ArtifactCube:
        """Validate and convert a canonical artifact cube mapping."""
        try:
            declared_shape = tuple(
                _exact_int(value, "shape") for value in block["shape"]
            )
            draw_seeds = tuple(
                _exact_int(value, "draw seed")
                for value in block["k_index_draw_seeds"]
            )
            raw_cells = tuple(block["cell_index"])
            if not all(isinstance(value, str) for value in raw_cells):
                raise ValueError("Artifact cube cells must be strings")
            cells = raw_cells
            seeds = tuple(
                _exact_int(value, "gate seed") for value in block["seed_index"]
            )
            raw_rates = _as_sequence(block["rates"], "rates")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Artifact cube metadata is invalid") from error

        if len(declared_shape) != 3:
            raise ValueError("Artifact cube shape must have three dimensions")
        if len(set(draw_seeds)) != len(draw_seeds):
            raise ValueError("Artifact cube draw seeds must be unique")
        if len(set(cells)) != len(cells):
            raise ValueError("Artifact cube cells must be unique")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Artifact cube seeds must be unique")

        rates: list[tuple[tuple[float, ...], ...]] = []
        for draw_index, raw_draw in enumerate(raw_rates):
            draw = _as_sequence(raw_draw, f"rates[{draw_index}]")
            converted_cells: list[tuple[float, ...]] = []
            for cell_index, raw_cell in enumerate(draw):
                seed_rates = _as_sequence(
                    raw_cell,
                    f"rates[{draw_index}][{cell_index}]",
                )
                converted_seed_rates: list[float] = []
                for value in seed_rates:
                    try:
                        rate = float(value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Artifact cube contains a non-numeric rate"
                        ) from error
                    if not math.isfinite(rate):
                        raise ValueError(
                            "Artifact cube contains an undefined rate"
                        )
                    converted_seed_rates.append(rate)
                converted_cells.append(tuple(converted_seed_rates))
            rates.append(tuple(converted_cells))

        cube = cls(
            draw_seeds=draw_seeds,
            cells=cells,
            seeds=seeds,
            rates=tuple(rates),
        )
        if len(cube.rates) != len(cube.draw_seeds):
            raise ValueError("Artifact cube rates do not match draw shape")
        if declared_shape != cube.shape:
            raise ValueError(
                f"Artifact cube declared shape {declared_shape} does not "
                f"match indices {cube.shape}"
            )
        for draw in cube.rates:
            if len(draw) != len(cube.cells):
                raise ValueError("Artifact cube rates do not match cell shape")
            if any(len(cell) != len(cube.seeds) for cell in draw):
                raise ValueError("Artifact cube rates do not match seed shape")
        return cube


@dataclass(frozen=True)
class CellEvaluation:
    """Recomputed score and verdict for one cell and gate seed."""

    cell: str
    seed: int
    mean_rate: float
    reference_rate: float
    score: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class SeedEvaluation:
    """All gated cell results and their seed-level conjunction."""

    seed: int
    cells: tuple[CellEvaluation, ...]
    n_cells_passed: int
    passed: bool


@dataclass(frozen=True)
class GateEvaluation:
    """Typed result of evaluating every cell, seed, and conjunction."""

    gate: GateSpec
    draw_seeds: tuple[int, ...]
    seeds: tuple[SeedEvaluation, ...]
    n_seeds_pass: int
    passed: bool

    @property
    def failures(self) -> tuple[CellEvaluation, ...]:
        """Return every failing cell in seed and cell order."""
        return tuple(
            cell
            for seed in self.seeds
            for cell in seed.cells
            if not cell.passed
        )


def _as_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Artifact cube {field} must be a sequence")
    return value


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error


def _as_decimal(value: Decimal | int | float | str, field: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a decimal number")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be a decimal number") from error


def derive_tolerance(
    mean: Decimal | int | float | str,
    sd: Decimal | int | float | str,
    k: Decimal | int | float | str,
    rounding: int,
) -> Decimal:
    """Derive ``mean + k * sd`` with explicit decimal half-even rounding."""
    decimal_rounding = _exact_int(rounding, "rounding")
    if decimal_rounding < 0:
        raise ValueError("rounding must be non-negative")
    quantum = Decimal(1).scaleb(-decimal_rounding)
    return (
        _as_decimal(mean, "mean") + _as_decimal(k, "k") * _as_decimal(sd, "sd")
    ).quantize(quantum, rounding=ROUND_HALF_EVEN)


def _find_gate(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []

    def visit(node: Mapping[str, Any]) -> None:
        for key, value in node.items():
            if key == name and isinstance(value, Mapping):
                matches.append(value)
            if isinstance(value, Mapping):
                visit(value)

    visit(document)
    if not matches:
        raise KeyError(f"Gate {name!r} was not found")
    if len(matches) != 1:
        raise ValueError(f"Gate {name!r} is not unique")
    return matches[0]


def _floor_statistic(
    artifact: Mapping[str, Any],
    floor_key: str,
) -> tuple[str, Mapping[str, Any]]:
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for name, block in artifact.items():
        if not isinstance(block, Mapping) or floor_key not in block:
            continue
        record = block[floor_key]
        if isinstance(record, Mapping) and {"mean", "sd"} <= set(record):
            matches.append((name, record))
    if len(matches) != 1:
        raise ValueError(
            f"Floor key {floor_key!r} does not have one exact derivation"
        )
    return matches[0]


def _load_thresholds(
    gate_name: str,
    block: Mapping[str, Any],
    root: Path,
) -> tuple[str, tuple[ThresholdSpec, ...]]:
    views = block.get("views")
    floor_run = block.get("floor_run")
    if not isinstance(views, Mapping) or not isinstance(floor_run, str):
        raise ValueError(f"Locked gate {gate_name!r} has no typed views")

    artifacts: dict[str, Mapping[str, Any]] = {}
    thresholds: dict[str, ThresholdSpec] = {}
    for view_name, raw_view in views.items():
        if not isinstance(raw_view, Mapping):
            raise ValueError(f"Gate view {view_name!r} must be a mapping")
        tolerances = raw_view.get("tolerances")
        derivations = raw_view.get("derivations")
        if not isinstance(tolerances, Mapping) or not isinstance(
            derivations, Mapping
        ):
            raise ValueError(f"Gate view {view_name!r} is not derived")
        rules = derivations.get("rules")
        view_floor_run = derivations.get("floor_run")
        if not isinstance(rules, Mapping) or set(rules) != set(tolerances):
            raise ValueError(
                f"Gate view {view_name!r} derivation keys do not match"
            )
        if (
            view_floor_run != floor_run
            or raw_view.get("floor_run") != floor_run
        ):
            raise ValueError(
                f"Gate view {view_name!r} does not use {floor_run}"
            )
        if floor_run not in artifacts:
            artifact = json.loads(
                (root / floor_run).read_text(encoding="utf-8"),
                parse_float=Decimal,
            )
            if not isinstance(artifact, Mapping):
                raise ValueError(
                    f"Floor artifact {floor_run} must be a mapping"
                )
            artifacts[floor_run] = artifact

        for cell, tolerance_value in tolerances.items():
            if cell in thresholds:
                raise ValueError(f"Duplicate gated cell {cell!r}")
            rule = rules[cell]
            if not isinstance(rule, Mapping) or rule.get("key") != cell:
                raise ValueError(
                    f"Gate cell {cell!r} has an invalid derivation"
                )
            k = _as_decimal(rule["k"], "k")
            rounding = _exact_int(rule.get("rounding", 3), "rounding")
            statistic_name, stats = _floor_statistic(
                artifacts[floor_run],
                str(rule["key"]),
            )
            derived = derive_tolerance(
                stats["mean"],
                stats["sd"],
                k,
                rounding,
            )
            tolerance = _as_decimal(tolerance_value, "tolerance")
            if derived != tolerance:
                raise ValueError(
                    f"Gate cell {cell!r} tolerance {tolerance} does not "
                    f"equal its exact derivation {derived}"
                )
            thresholds[cell] = ThresholdSpec(
                cell=str(cell),
                tolerance=tolerance,
                floor_run=floor_run,
                floor_statistic=statistic_name,
                floor_key=str(rule["key"]),
                k=k,
                rounding=rounding,
                derived=derived,
            )
    return floor_run, tuple(thresholds[cell] for cell in sorted(thresholds))


def _load_scoring(
    gate_name: str,
    block: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> ScoringSpec:
    statistic = str(block.get("statistic", ""))
    pass_rule = str(protocol.get("pass_rule", ""))
    estimator_text = " ".join(
        str(value)
        for value in (
            protocol.get("estimator", ""),
            protocol.get("candidate", ""),
            pass_rule,
        )
    ).lower()
    compact_statistic = "".join(statistic.lower().split())
    if (
        "|ln(" not in compact_statistic
        or "rbar" not in compact_statistic
        or "rate_a" not in compact_statistic
        or "mean" not in estimator_text
        or "scored once" not in estimator_text
        or "not the mean" not in estimator_text
        or "<=" not in pass_rule
    ):
        raise ValueError(
            f"Gate {gate_name!r} uses unsupported scoring semantics"
        )
    return ScoringSpec(
        aggregation=DrawAggregation.MEAN_RATES_THEN_SCORE,
        statistic=ScoreStatistic.ABSOLUTE_LOG_RATIO,
        operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
    )


def _load_draw_seeds(
    gate_name: str,
    protocol: Mapping[str, Any],
    cube_schema: Mapping[str, Any],
    n_draws: int,
) -> tuple[int, ...]:
    stream_text = " ".join(
        str(value)
        for value in (
            protocol.get("candidate_draw_stream", ""),
            protocol.get("candidate", ""),
            protocol.get("estimator", ""),
            cube_schema.get("rule", ""),
        )
    )
    bases = {
        int(match.group("base"))
        for match in _DRAW_STREAM_PATTERN.finditer(stream_text)
    }
    if len(bases) != 1:
        raise ValueError(f"Gate {gate_name!r} has no unique K-draw stream")
    base = bases.pop()
    return tuple(base + draw for draw in range(n_draws))


def _load_reference_rates(
    gate_name: str,
    root: Path,
    floor_run: str,
    seeds: tuple[int, ...],
    cells: tuple[str, ...],
) -> tuple[ReferenceRate, ...]:
    artifact = json.loads(
        (root / floor_run).read_text(encoding="utf-8"),
        parse_float=Decimal,
    )
    rows = artifact.get("noise_floor_per_seed")
    if not isinstance(rows, list):
        raise ValueError(f"Gate {gate_name!r} floor has no per-seed rates")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(
                f"Gate {gate_name!r} floor seed must be a mapping"
            )
        seed = _exact_int(row.get("seed"), "floor seed")
        if seed in by_seed or not isinstance(row.get("cells"), Mapping):
            raise ValueError(f"Gate {gate_name!r} floor seed is invalid")
        by_seed[seed] = row["cells"]
    if set(by_seed) != set(seeds):
        raise ValueError(f"Gate {gate_name!r} floor seeds do not match")

    references: list[ReferenceRate] = []
    for seed in seeds:
        for cell in cells:
            record = by_seed[seed].get(cell)
            if not isinstance(record, Mapping) or "rate_a" not in record:
                raise ValueError(
                    f"Gate {gate_name!r} floor lacks {cell!r} at seed {seed}"
                )
            rate = float(record["rate_a"])
            if not math.isfinite(rate):
                raise ValueError(
                    f"Gate {gate_name!r} floor rate is not finite"
                )
            references.append(ReferenceRate(cell=cell, seed=seed, rate=rate))
    return tuple(references)


def load_gate(
    name: str,
    *,
    gates_path: str | Path = _DEFAULT_GATES_PATH,
) -> GateSpec:
    """Load one uniquely named, locked K-draw gate from ``gates.yaml``.

    The loader validates each structured floor derivation before exposing its
    tolerance, so evaluation never silently consumes stale contract numbers.
    """
    path = Path(gates_path).resolve()
    contract = ContractRef.current(root=path.parent, path=path)
    contract_text = path.read_text(encoding="utf-8")
    if ContractRef.current(root=path.parent, path=path) != contract:
        raise RuntimeError("Contract changed while it was being loaded")
    document = yaml.safe_load(contract_text)
    if not isinstance(document, Mapping) or not isinstance(
        document.get("gates"), Mapping
    ):
        raise ValueError("gates.yaml must contain a gates mapping")
    gate = _find_gate(document["gates"], name)
    raw_thresholds = gate.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError(f"Gate {name!r} has no locked thresholds")
    if (
        raw_thresholds.get("locked") is not True
        or gate.get("locked", True) is not True
    ):
        raise ValueError(f"Gate {name!r} must be locked")
    for candidate in (gate, raw_thresholds):
        if "status" in candidate and candidate["status"] != "locked":
            raise ValueError(f"Gate {name!r} must have locked status")

    floor_run, thresholds = _load_thresholds(name, raw_thresholds, path.parent)
    protocol = raw_thresholds.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"Gate {name!r} has no protocol")
    seeds = tuple(
        _exact_int(seed, "gate seed")
        for seed in protocol.get("gate_seeds", ())
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Gate {name!r} must define unique gate seeds")
    schema = protocol.get("fresh_run_artifact_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(f"Gate {name!r} has no fresh-run artifact schema")
    cube_schema = schema.get("per_draw_per_cell_rates")
    if not isinstance(cube_schema, Mapping):
        raise ValueError(f"Gate {name!r} has no rate-cube schema")
    try:
        shape = tuple(
            _exact_int(value, "cube shape") for value in cube_schema["shape"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Gate {name!r} has an invalid cube shape") from error
    if (
        len(shape) != 3
        or shape[0] <= 0
        or shape[1:] != (len(thresholds), len(seeds))
    ):
        raise ValueError(
            f"Gate {name!r} cube shape does not match its surface"
        )
    candidate_draws = protocol.get("candidate_draws", shape[0])
    if _exact_int(candidate_draws, "candidate draws") != shape[0]:
        raise ValueError(f"Gate {name!r} candidate draw count is inconsistent")
    draw_seeds = _load_draw_seeds(name, protocol, cube_schema, shape[0])
    scoring = _load_scoring(name, raw_thresholds, protocol)

    pass_rule = protocol.get("pass_rule")
    match = _CONJUNCTION_PATTERN.search(str(pass_rule))
    if match is None:
        raise ValueError(f"Gate {name!r} has no machine-readable conjunction")
    required = int(match.group("required"))
    total = int(match.group("total"))
    if total != len(seeds) or not 0 < required <= total:
        raise ValueError(f"Gate {name!r} conjunction does not match its seeds")

    references = _load_reference_rates(
        name,
        path.parent,
        floor_run,
        seeds,
        tuple(threshold.cell for threshold in thresholds),
    )

    return GateSpec(
        name=name,
        contract=contract,
        floor_run=floor_run,
        n_draws=shape[0],
        draw_seeds=draw_seeds,
        seeds=seeds,
        required_seed_passes=required,
        scoring=scoring,
        thresholds=thresholds,
        reference_rates=references,
    )


def _validated_references(
    spec: GateSpec,
    reference_rates: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    if set(reference_rates) != set(spec.seeds):
        raise ValueError("Reference seeds do not match the locked gate")
    references: dict[int, dict[str, float]] = {}
    locked_references = spec.reference_by_seed
    expected_cells = set(spec.cells)
    for seed in spec.seeds:
        raw_cells = reference_rates[seed]
        if set(raw_cells) != expected_cells:
            raise ValueError(
                f"Seed {seed} reference cells do not match the locked gate"
            )
        references[seed] = {}
        for cell, raw_rate in raw_cells.items():
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError) as error:
                raise ValueError("Reference rate must be numeric") from error
            if not math.isfinite(rate):
                raise ValueError("Reference rate must be finite")
            if rate != locked_references[seed][cell]:
                raise ValueError(
                    f"Seed {seed} reference rate for {cell!r} does not "
                    "match the locked floor"
                )
            references[seed][cell] = rate
    return references


def evaluate_gate(
    gate: str | GateSpec,
    cube: ArtifactCube | Mapping[str, Any],
    reference_rates: Mapping[int, Mapping[str, float]],
    *,
    gates_path: str | Path = _DEFAULT_GATES_PATH,
) -> GateEvaluation:
    """Recompute K-draw cell scores, seed conjunctions, and gate verdict."""
    spec = (
        load_gate(gate, gates_path=gates_path)
        if isinstance(gate, str)
        else gate
    )
    artifact_cube = (
        cube
        if isinstance(cube, ArtifactCube)
        else ArtifactCube.from_mapping(cube)
    )
    expected_shape = (spec.n_draws, len(spec.cells), len(spec.seeds))
    if artifact_cube.shape != expected_shape:
        raise ValueError(
            f"Artifact cube shape {artifact_cube.shape} does not match "
            f"locked shape {expected_shape}"
        )
    if artifact_cube.draw_seeds != spec.draw_seeds:
        raise ValueError(
            "Artifact cube draw seeds do not match the locked gate"
        )
    if set(artifact_cube.cells) != set(spec.cells):
        raise ValueError("Artifact cube cells do not match the locked gate")
    if artifact_cube.seeds != spec.seeds:
        raise ValueError("Artifact cube seeds do not match the locked gate")

    references = _validated_references(spec, reference_rates)
    if spec.scoring != ScoringSpec(
        aggregation=DrawAggregation.MEAN_RATES_THEN_SCORE,
        statistic=ScoreStatistic.ABSOLUTE_LOG_RATIO,
        operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
    ):
        raise ValueError(
            f"Gate {spec.name!r} uses unsupported scoring semantics"
        )
    cell_positions = {
        cell: index for index, cell in enumerate(artifact_cube.cells)
    }
    thresholds = spec.threshold_by_cell
    seed_results: list[SeedEvaluation] = []
    for seed_index, seed in enumerate(spec.seeds):
        cell_results: list[CellEvaluation] = []
        for cell in spec.cells:
            cell_index = cell_positions[cell]
            draw_rates = np.asarray(
                [
                    artifact_cube.rates[draw][cell_index][seed_index]
                    for draw in range(spec.n_draws)
                ],
                dtype=np.float64,
            )
            mean_rate = float(draw_rates.mean())
            reference_rate = references[seed][cell]
            score = (
                float(abs(math.log(mean_rate / reference_rate)))
                if mean_rate > 0.0 and reference_rate > 0.0
                else float("inf")
            )
            tolerance = float(thresholds[cell].tolerance)
            cell_results.append(
                CellEvaluation(
                    cell=cell,
                    seed=seed,
                    mean_rate=mean_rate,
                    reference_rate=reference_rate,
                    score=score,
                    tolerance=tolerance,
                    passed=score <= tolerance,
                )
            )
        n_cells_passed = sum(result.passed for result in cell_results)
        seed_results.append(
            SeedEvaluation(
                seed=seed,
                cells=tuple(cell_results),
                n_cells_passed=n_cells_passed,
                passed=n_cells_passed == len(spec.cells),
            )
        )

    n_seeds_pass = sum(result.passed for result in seed_results)
    return GateEvaluation(
        gate=spec,
        draw_seeds=artifact_cube.draw_seeds,
        seeds=tuple(seed_results),
        n_seeds_pass=n_seeds_pass,
        passed=n_seeds_pass >= spec.required_seed_passes,
    )
