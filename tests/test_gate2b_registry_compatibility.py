"""Bind the registry certificate to the frozen candidate-9 artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from populace_dynamics.models.household_composition.registry import CANDIDATE_9

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_PATH = ROOT / "runs" / "gate2b_registry_certificate_v1.json"
REFERENCE_PATH = ROOT / "runs" / "gate2b_hazard_v9.json"
RUNNER_PATH = ROOT / "scripts" / "run_gate2b_registry.py"
IMPLEMENTATION_ROOT = (
    ROOT / "src" / "populace_dynamics" / "models" / "household_composition"
)
REFERENCE_SHA256 = (
    "0bb2599efc70cbeebff1801b9484245ff031209e07bcda0898b9d2db338a748d"
)
REFERENCE_N_BYTES = 869_757
MISMATCH_CAUSE = (
    "registry replay differs from the frozen candidate-9 quantity; inspect "
    "component operation order and the recorded numerical environment"
)


def _as_float64(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "n_bytes": len(payload),
    }


def _coordinate(
    index: tuple[int, ...],
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for position, axis in zip(index, axes, strict=True):
        result[f"{axis}_index"] = position
        result[axis] = labels[axis][position]
    return result


def _expected_float_mismatches(
    actual: np.ndarray,
    expected: np.ndarray,
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    rows = np.argwhere(actual.view(np.uint64) != expected.view(np.uint64))
    result = []
    for row in rows:
        index = tuple(int(value) for value in row)
        actual_value = float(actual[index])
        expected_value = float(expected[index])
        result.append(
            {
                **_coordinate(index, axes, labels),
                "reference": expected_value,
                "reference_hex": expected_value.hex(),
                "registry": actual_value,
                "registry_hex": actual_value.hex(),
                "abs_deviation": float(abs(actual_value - expected_value)),
                "cause": MISMATCH_CAUSE,
            }
        )
    return result


def _assert_float_comparison(
    claim: dict[str, Any],
    actual_values: Any,
    expected_values: Any,
    *,
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> bool:
    actual = _as_float64(actual_values)
    expected = _as_float64(expected_values)
    assert actual.shape == expected.shape
    max_abs = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    bit_equal = bool(
        np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    )
    assert claim["shape"] == list(actual.shape)
    assert claim["max_abs_deviation"] == max_abs
    assert claim["ieee_uint64_equal"] is bit_equal
    assert claim["mismatches"] == _expected_float_mismatches(
        actual, expected, axes, labels
    )
    return bit_equal


def _expected_exact_mismatches(
    actual: np.ndarray,
    expected: np.ndarray,
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **_coordinate(tuple(int(value) for value in row), axes, labels),
            "reference": expected[tuple(row)],
            "registry": actual[tuple(row)],
            "cause": MISMATCH_CAUSE,
        }
        for row in np.argwhere(actual != expected)
    ]


def _assert_exact_array_comparison(
    claim: dict[str, Any],
    actual_values: Any,
    expected_values: Any,
    *,
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> bool:
    actual = np.asarray(actual_values, dtype=object)
    expected = np.asarray(expected_values, dtype=object)
    assert actual.shape == expected.shape
    mismatches = _expected_exact_mismatches(actual, expected, axes, labels)
    exact = not mismatches
    assert claim == {
        "exact_match": exact,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }
    return exact


def _assert_structured_comparison(
    claim: dict[str, Any], actual: Any, expected: Any
) -> bool:
    exact = actual == expected
    assert claim == {
        "exact_match": exact,
        "n_mismatches": int(not exact),
        "mismatches": (
            []
            if exact
            else [
                {
                    "reference": expected,
                    "registry": actual,
                    "cause": MISMATCH_CAUSE,
                }
            ]
        ),
    }
    return exact


def _reference_matrix(
    per_seed: list[dict[str, Any]], cells: list[str], field: str
) -> list[list[Any]]:
    return [
        [seed_result["gated_cells"][cell][field] for seed_result in per_seed]
        for cell in cells
    ]


def _per_seed_cube(
    per_seed: list[dict[str, Any]], cells: list[str], n_draws: int
) -> list[list[list[float]]]:
    return [
        [
            [
                seed_result["gated_cells"][cell]["per_draw_rate"][draw]
                for seed_result in per_seed
            ]
            for cell in cells
        ]
        for draw in range(n_draws)
    ]


def _means_from_draws(cube: Any) -> list[list[float]]:
    """Reproduce the frozen contiguous one-cell-at-a-time mean operation."""
    n_draws = len(cube)
    n_cells = len(cube[0])
    n_seeds = len(cube[0][0])
    return [
        [
            float(
                np.asarray(
                    [cube[draw][cell][seed] for draw in range(n_draws)],
                    dtype=np.float64,
                ).mean()
            )
            for seed in range(n_seeds)
        ]
        for cell in range(n_cells)
    ]


def _scores_from_means(means: Any, rate_a: Any) -> list[list[float]]:
    return [
        [
            float(abs(math.log(means[cell][seed] / rate_a[cell][seed])))
            for seed in range(len(means[cell]))
        ]
        for cell in range(len(means))
    ]


def _family_of(cell: str) -> str:
    if cell.startswith("coresident_spouse."):
        return "coresident_spouse"
    if cell.startswith("coresident_parent."):
        return "coresident_parent"
    if cell.startswith("coresident_child."):
        return "coresident_child"
    if cell.startswith("coresident_grandchild."):
        return "coresident_grandchild"
    if cell.startswith("multigen."):
        return "multigen_stock"
    if cell in ("multigen_entry", "multigen_exit"):
        return "multigen_transition"
    if cell.startswith("parental_home_exit."):
        return "parental_home_exit"
    if cell.startswith("hh_size."):
        return "hh_size"
    return "other"


def _derived_verdict(
    cells: list[str],
    seeds: list[int],
    means: Any,
    rate_a: Any,
    scores: Any,
    tolerances: Any,
    passes: Any,
) -> dict[str, Any]:
    seed_pass = {
        str(seed): all(passes[cell][seed_index] for cell in range(len(cells)))
        for seed_index, seed in enumerate(seeds)
    }
    all_failing = [
        {
            "cell": cells[cell],
            "seed": seed,
            "family": _family_of(cells[cell]),
            "score": scores[cell][seed_index],
            "tolerance": tolerances[cell][seed_index],
            "r_candidate": means[cell][seed_index],
            "rate_a": rate_a[cell][seed_index],
        }
        for seed_index, seed in enumerate(seeds)
        for cell in range(len(cells))
        if not passes[cell][seed_index]
    ]
    n_seeds_pass = sum(seed_pass.values())
    return {
        "n_gate_seeds": len(seeds),
        "n_gated_cells": len(cells),
        "seed_pass": seed_pass,
        "n_seeds_pass": n_seeds_pass,
        "gate_2b_pass": n_seeds_pass >= 4,
        "rule": (
            "A seed passes iff every one of the 46 gated cells holds "
            "(|ln(rbar / rate_a)| <= locked tolerance); the gate passes iff "
            ">= 4 of the 5 gate seeds pass."
        ),
        "all_failing_gated_cells": all_failing,
    }


def test__given_gate2b_registry_certificate__then_every_c9_claim_recomputes():
    """Recompute the cube-to-verdict chain and all compatibility claims."""
    certificate_bytes = CERTIFICATE_PATH.read_bytes()
    reference_bytes = REFERENCE_PATH.read_bytes()
    certificate = json.loads(certificate_bytes)
    reference = json.loads(reference_bytes)
    assert (
        certificate["schema_version"]
        == "gate2b_registry_compatibility_certificate_v1"
    )

    reference_identity = _file_identity(REFERENCE_PATH)
    assert reference_identity == certificate["reference"]
    assert reference_identity == {
        "path": "runs/gate2b_hazard_v9.json",
        "sha256": REFERENCE_SHA256,
        "n_bytes": REFERENCE_N_BYTES,
    }
    assert certificate["runner"] == _file_identity(RUNNER_PATH)
    implementation_paths = sorted(IMPLEMENTATION_ROOT.rglob("*.py"))
    assert certificate["implementation_files"] == [
        _file_identity(path) for path in implementation_paths
    ]

    spec_claim = dict(certificate["resolved_candidate_spec"])
    spec_sha = spec_claim.pop("sha256")
    assert spec_claim == CANDIDATE_9.canonical_dict()
    assert spec_sha == CANDIDATE_9.sha256
    canonical_spec = json.dumps(
        spec_claim, sort_keys=True, separators=(",", ":")
    ).encode()
    assert spec_sha == hashlib.sha256(canonical_spec).hexdigest()

    canonical_cube = reference["fresh_run_artifact_schema"][
        "per_draw_per_cell_rates"
    ]
    draw_seeds = canonical_cube["k_index_draw_seeds"]
    cells = canonical_cube["cell_index"]
    seeds = canonical_cube["seed_index"]
    reference_cube = canonical_cube["rates"]
    assert canonical_cube["shape"] == [20, 46, 5]
    per_seed_by_value = {row["seed"]: row for row in reference["per_seed"]}
    per_seed = [per_seed_by_value[seed] for seed in seeds]
    assert (
        _as_float64(reference_cube).view(np.uint64).tolist()
        == _as_float64(_per_seed_cube(per_seed, cells, len(draw_seeds)))
        .view(np.uint64)
        .tolist()
    )

    protocol = certificate["protocol"]
    labels = {"draw": draw_seeds, "cell": cells, "seed": seeds}
    assert protocol["shape"] == canonical_cube["shape"]
    assert protocol["axes"] == {
        "order": ["draw", "cell", "seed"],
        "draw_seeds": draw_seeds,
        "cells": cells,
        "seeds": seeds,
    }
    assert protocol["single_draw_provenance_seeds"] == [
        4200 + seed for seed in seeds
    ]

    evaluation = certificate["registry_evaluation"]
    comparisons = certificate["comparisons"]
    assert set(comparisons) == {
        "per_draw_per_cell_rates",
        "per_cell_means",
        "per_cell_scores",
        "rate_a",
        "tolerances",
        "per_cell_pass",
        "seed_verdicts",
        "seed_conjunction",
        "verdict",
    }
    float_results = [
        _assert_float_comparison(
            comparisons["per_draw_per_cell_rates"],
            evaluation["per_draw_per_cell_rates"],
            reference_cube,
            axes=("draw", "cell", "seed"),
            labels=labels,
        )
    ]

    derived_means = _means_from_draws(evaluation["per_draw_per_cell_rates"])
    registry_means = evaluation["per_cell_means"]
    for field in ("rbar", "r_candidate"):
        assert _as_float64(registry_means[field]).view(np.uint64).tolist() == (
            _as_float64(derived_means).view(np.uint64).tolist()
        )
        float_results.append(
            _assert_float_comparison(
                comparisons["per_cell_means"][field],
                registry_means[field],
                _reference_matrix(per_seed, cells, field),
                axes=("cell", "seed"),
                labels=labels,
            )
        )

    rate_a = evaluation["rate_a"]
    tolerances = evaluation["tolerances"]
    derived_scores = _scores_from_means(derived_means, rate_a)
    assert (
        _as_float64(evaluation["per_cell_scores"]).view(np.uint64).tolist()
        == _as_float64(derived_scores).view(np.uint64).tolist()
    )
    for evaluation_key, reference_field in (
        ("per_cell_scores", "score"),
        ("rate_a", "rate_a"),
        ("tolerances", "tolerance"),
    ):
        float_results.append(
            _assert_float_comparison(
                comparisons[evaluation_key],
                evaluation[evaluation_key],
                _reference_matrix(per_seed, cells, reference_field),
                axes=("cell", "seed"),
                labels=labels,
            )
        )

    derived_pass = [
        [
            derived_scores[cell][seed] <= tolerances[cell][seed]
            for seed in range(len(seeds))
        ]
        for cell in range(len(cells))
    ]
    assert evaluation["per_cell_pass"] == derived_pass
    reference_pass = _reference_matrix(per_seed, cells, "pass")
    cell_pass_equal = _assert_exact_array_comparison(
        comparisons["per_cell_pass"],
        evaluation["per_cell_pass"],
        reference_pass,
        axes=("cell", "seed"),
        labels=labels,
    )
    derived_seed_conjunction = [
        {
            "seed": seed,
            "n_gated_pass": sum(
                derived_pass[cell][seed_index] for cell in range(len(cells))
            ),
            "n_gated_fail": sum(
                not derived_pass[cell][seed_index]
                for cell in range(len(cells))
            ),
            "seed_pass": all(
                derived_pass[cell][seed_index] for cell in range(len(cells))
            ),
        }
        for seed_index, seed in enumerate(seeds)
    ]
    derived_seed_verdicts = [
        row["seed_pass"] for row in derived_seed_conjunction
    ]
    assert evaluation["seed_verdicts"] == derived_seed_verdicts
    assert evaluation["seed_conjunction"] == derived_seed_conjunction
    seed_verdicts_equal = _assert_exact_array_comparison(
        comparisons["seed_verdicts"],
        evaluation["seed_verdicts"],
        [row["seed_pass"] for row in per_seed],
        axes=("seed",),
        labels=labels,
    )
    seed_conjunction_equal = _assert_structured_comparison(
        comparisons["seed_conjunction"],
        evaluation["seed_conjunction"],
        reference["seed_conjunction"],
    )
    derived_verdict = _derived_verdict(
        cells,
        seeds,
        derived_means,
        rate_a,
        derived_scores,
        tolerances,
        derived_pass,
    )
    assert evaluation["verdict"] == derived_verdict
    verdict_equal = _assert_structured_comparison(
        comparisons["verdict"], evaluation["verdict"], reference["verdict"]
    )

    gate_results = []
    registry_cube_array = _as_float64(evaluation["per_draw_per_cell_rates"])
    reference_cube_array = _as_float64(reference_cube)
    reference_blocks = reference["per_family_decomposition"]
    assert set(certificate["per_gate_block"]) == set(reference_blocks)
    assert len(reference_blocks) == 8
    for block, block_result in reference_blocks.items():
        indices = [cells.index(cell) for cell in block_result["cells"]]
        block_claim = certificate["per_gate_block"][block]
        block_labels = {**labels, "cell": block_result["cells"]}
        assert block_claim["cells"] == block_result["cells"]
        gate_results.append(
            _assert_float_comparison(
                block_claim["per_draw_per_cell_rates"],
                registry_cube_array[:, indices, :],
                reference_cube_array[:, indices, :],
                axes=("draw", "cell", "seed"),
                labels=block_labels,
            )
        )
        for field in ("rbar", "r_candidate"):
            gate_results.append(
                _assert_float_comparison(
                    block_claim["per_cell_means"][field],
                    _as_float64(registry_means[field])[indices, :],
                    _as_float64(_reference_matrix(per_seed, cells, field))[
                        indices, :
                    ],
                    axes=("cell", "seed"),
                    labels=block_labels,
                )
            )
        gate_results.append(
            _assert_float_comparison(
                block_claim["per_cell_scores"],
                _as_float64(evaluation["per_cell_scores"])[indices, :],
                _as_float64(_reference_matrix(per_seed, cells, "score"))[
                    indices, :
                ],
                axes=("cell", "seed"),
                labels=block_labels,
            )
        )
        gate_results.append(
            _assert_exact_array_comparison(
                block_claim["per_cell_pass"],
                np.asarray(evaluation["per_cell_pass"], dtype=object)[
                    indices, :
                ],
                np.asarray(reference_pass, dtype=object)[indices, :],
                axes=("cell", "seed"),
                labels=block_labels,
            )
        )

    recomputed_pass = all(
        (
            *float_results,
            cell_pass_equal,
            seed_verdicts_equal,
            seed_conjunction_equal,
            verdict_equal,
            *gate_results,
        )
    )
    assert recomputed_pass
    assert certificate["certificate_pass"] is True
