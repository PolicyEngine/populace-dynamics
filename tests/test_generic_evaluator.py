"""Always-runnable generic Gate-2 evaluator reproduction tests."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from populace_dynamics.evaluation import (
    ArtifactCube,
    ComparisonOperator,
    DrawAggregation,
    GateEvaluation,
    ScoreStatistic,
    ScoringSpec,
    derive_tolerance,
    evaluate_gate,
    load_gate,
)

ROOT = Path(__file__).resolve().parents[1]
GATES_PATH = ROOT / "gates.yaml"

REPRODUCTION_CASES = (
    (
        "gate_2",
        "runs/gate2_hazard_v16.json",
        (True, True, False, True, True),
        "completed_fertility.c1970s",
    ),
    (
        "gate_2b",
        "runs/gate2b_hazard_v9.json",
        (True, True, True, False, True),
        "hh_size.5+",
    ),
    (
        "gate_2c",
        "runs/gate2c_floors_v1.json",
        (True, True, True, True, True),
        None,
    ),
)


def _load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text())


def _reference_rates(floor: dict, cells: tuple[str, ...]) -> dict:
    return {
        row["seed"]: {cell: row["cells"][cell]["rate_a"] for cell in cells}
        for row in floor["noise_floor_per_seed"]
    }


def _candidate_case(gate_name: str, artifact_path: str):
    spec = load_gate(gate_name, gates_path=GATES_PATH)
    artifact = _load_json(artifact_path)
    floor = _load_json(spec.floor_run)
    cube = artifact["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    return spec, artifact, cube, _reference_rates(floor, spec.cells)


def _gate2c_floor_case():
    spec = load_gate("gate_2c", gates_path=GATES_PATH)
    floor = _load_json(spec.floor_run)
    references = _reference_rates(floor, spec.cells)
    by_seed = {
        row["seed"]: row["cells"] for row in floor["noise_floor_per_seed"]
    }
    rates = [
        [
            [by_seed[seed][cell]["rate_b"] for seed in spec.seeds]
            for cell in spec.cells
        ]
        for _ in range(spec.n_draws)
    ]
    cube = {
        "shape": [spec.n_draws, len(spec.cells), len(spec.seeds)],
        "k_index_draw_seeds": list(range(5200, 5200 + spec.n_draws)),
        "cell_index": list(spec.cells),
        "seed_index": list(spec.seeds),
        "rates": rates,
    }
    return spec, floor, cube, references


@pytest.mark.parametrize(
    "gate_name,artifact_path,expected_seed_pass,expected_failure",
    REPRODUCTION_CASES,
)
def test__given_committed_cube__then_generic_evaluator_reproduces_verdict(
    gate_name,
    artifact_path,
    expected_seed_pass,
    expected_failure,
):
    # Given
    if gate_name == "gate_2c":
        spec, artifact, cube, references = _gate2c_floor_case()
    else:
        spec, artifact, cube, references = _candidate_case(
            gate_name, artifact_path
        )

    # When
    result = evaluate_gate(spec, cube, references)

    # Then
    assert isinstance(result, GateEvaluation)
    assert tuple(seed.passed for seed in result.seeds) == expected_seed_pass
    assert result.n_seeds_pass == sum(expected_seed_pass)
    assert result.passed is True
    assert {failure.cell for failure in result.failures} == (
        {expected_failure} if expected_failure else set()
    )

    if gate_name == "gate_2c":
        assert (
            result.n_seeds_pass
            == artifact["training_copy_check"]["n_seeds_pass"]
        )
        assert (
            result.passed == artifact["training_copy_check"]["passes_4_of_5"]
        )
        by_seed = {
            row["seed"]: row["cells"]
            for row in artifact["noise_floor_per_seed"]
        }
        for seed in result.seeds:
            for cell in seed.cells:
                assert cell.score == pytest.approx(
                    by_seed[seed.seed][cell.cell]["log_ratio_abs"],
                    abs=1e-15,
                )
    else:
        committed_by_seed = {row["seed"]: row for row in artifact["per_seed"]}
        for seed in result.seeds:
            committed_cells = committed_by_seed[seed.seed]["gated_cells"]
            for cell in seed.cells:
                committed = committed_cells[cell.cell]
                assert cell.mean_rate == committed["rbar"]
                assert cell.reference_rate == committed["rate_a"]
                assert cell.score == committed["score"]
                assert cell.tolerance == committed["tolerance"]
                assert cell.passed == committed["pass"]


@pytest.mark.parametrize(
    "gate_name,n_cells",
    (("gate_2", 46), ("gate_2b", 46), ("gate_2c", 27)),
)
def test__given_locked_gate_name__then_typed_contract_is_loaded(
    gate_name,
    n_cells,
):
    # Given / When
    spec = load_gate(gate_name, gates_path=GATES_PATH)

    # Then
    assert spec.name == gate_name
    assert len(spec.cells) == n_cells
    assert spec.n_draws == 20
    assert spec.draw_seeds == tuple(range(5200, 5220))
    assert spec.seeds == (0, 1, 2, 3, 4)
    assert spec.required_seed_passes == 4
    assert spec.scoring == ScoringSpec(
        aggregation=DrawAggregation.MEAN_RATES_THEN_SCORE,
        statistic=ScoreStatistic.ABSOLUTE_LOG_RATIO,
        operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
    )
    assert len(spec.reference_rates) == n_cells * len(spec.seeds)
    assert spec.contract.path == "gates.yaml"
    assert all(
        threshold.derived == threshold.tolerance
        for threshold in spec.thresholds
    )


def test__given_unlocked_gate__then_loader_rejects_it():
    # Given / When / Then
    with pytest.raises(ValueError, match="locked"):
        load_gate("gate_3", gates_path=GATES_PATH)


def test__given_noncanonical_cube_shape__then_evaluator_rejects_it():
    # Given
    spec, _, cube, references = _candidate_case(
        "gate_2b", "runs/gate2b_hazard_v9.json"
    )
    malformed = copy.deepcopy(cube)
    malformed["rates"] = malformed["rates"][:-1]

    # When / Then
    with pytest.raises(ValueError, match="shape"):
        evaluate_gate(spec, malformed, references)


@pytest.mark.parametrize(
    "draw_seeds",
    (
        list(range(6200, 6220)),
        list(reversed(range(5200, 5220))),
        [5200.5, *range(5201, 5220)],
    ),
)
def test__given_wrong_draw_stream__then_evaluator_rejects_it(draw_seeds):
    # Given
    spec, _, cube, references = _candidate_case(
        "gate_2b", "runs/gate2b_hazard_v9.json"
    )
    malformed = copy.deepcopy(cube)
    malformed["k_index_draw_seeds"] = draw_seeds

    # When / Then
    with pytest.raises(ValueError, match="draw seed|metadata"):
        evaluate_gate(spec, malformed, references)


def test__given_missing_reference_cell__then_evaluator_rejects_it():
    # Given
    spec, _, cube, references = _candidate_case(
        "gate_2b", "runs/gate2b_hazard_v9.json"
    )
    missing = copy.deepcopy(references)
    missing[0].pop(spec.cells[0])

    # When / Then
    with pytest.raises(ValueError, match="reference cells"):
        evaluate_gate(spec, cube, missing)


def test__given_changed_reference_rate__then_evaluator_rejects_it():
    # Given
    spec, _, cube, references = _candidate_case(
        "gate_2b", "runs/gate2b_hazard_v9.json"
    )
    changed = copy.deepcopy(references)
    changed[0][spec.cells[0]] *= 1.01

    # When / Then
    with pytest.raises(ValueError, match="locked floor"):
        evaluate_gate(spec, cube, changed)


def test__given_unsupported_scorer__then_evaluator_refuses_a_verdict():
    # Given
    spec, _, cube, references = _gate2c_floor_case()
    unsupported = replace(
        spec,
        scoring=ScoringSpec(
            aggregation=DrawAggregation.MEAN_RATES_THEN_SCORE,
            statistic="absolute_difference",
            operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
        ),
    )

    # When / Then
    with pytest.raises(ValueError, match="unsupported scoring"):
        evaluate_gate(unsupported, cube, references)


def test__given_decimal_half_step__then_derivation_uses_half_even_rounding():
    # Given / When
    even_down = derive_tolerance("0.1005", "0", "4", 3)
    even_up = derive_tolerance("0.1015", "0", "4", 3)

    # Then
    assert even_down == Decimal("0.100")
    assert even_up == Decimal("0.102")


def test__given_one_breached_cell__then_seed_conjunction_fails():
    # Given
    spec, _, cube, references = _gate2c_floor_case()
    mutated = copy.deepcopy(cube)
    cell_index = 0
    seed_index = 0
    cell = spec.cells[cell_index]
    tolerance = float(spec.threshold_by_cell[cell].tolerance)
    breached_rate = references[0][cell] * math.exp(tolerance + 0.01)
    for draw in range(spec.n_draws):
        mutated["rates"][draw][cell_index][seed_index] = breached_rate

    # When
    result = evaluate_gate(spec, mutated, references)

    # Then
    assert result.seeds[0].passed is False
    assert result.seeds[0].n_cells_passed == len(spec.cells) - 1
    assert [failure.cell for failure in result.failures] == [cell]
    assert result.n_seeds_pass == 4
    assert result.passed is True


def test__given_nonpositive_candidate_rate__then_cell_fails_with_infinite_score():
    # Given
    spec, _, cube, references = _gate2c_floor_case()
    mutated = copy.deepcopy(cube)
    for draw in range(spec.n_draws):
        mutated["rates"][draw][0][0] = 0.0

    # When
    result = evaluate_gate(
        spec, ArtifactCube.from_mapping(mutated), references
    )

    # Then
    failed = result.failures[0]
    assert failed.cell == spec.cells[0]
    assert failed.seed == 0
    assert math.isinf(failed.score)
