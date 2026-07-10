"""Build a complete candidate-16 registry compatibility certificate."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from populace_dynamics.models.family_transitions.evaluation import (
    SINGLE_DRAW_PROVENANCE_SEEDS,
    EvaluationResult,
)
from populace_dynamics.models.family_transitions.registry import CANDIDATE_16

__all__ = ["build_compatibility_certificate"]

_MISMATCH_CAUSE = (
    "registry replay differs from the frozen candidate-16 quantity; inspect "
    "component operation order and the recorded numerical environment"
)


def _normalized(value: Any) -> Any:
    """Apply the same JSON key/value normalization as the written artifact."""
    return json.loads(json.dumps(value))


def _reference_matrix(
    per_seed: list[dict[str, Any]], cells: list[str], field: str
) -> list[list[Any]]:
    return [
        [seed_result["gated_cells"][cell][field] for seed_result in per_seed]
        for cell in cells
    ]


def _reference_cube(
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


def _coordinate(
    index: tuple[int, ...],
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> dict[str, Any]:
    coordinate: dict[str, Any] = {}
    for position, axis in zip(index, axes, strict=True):
        coordinate[f"{axis}_index"] = int(position)
        if axis in labels:
            coordinate[axis] = labels[axis][position]
    return coordinate


def _float_claim(
    actual_values: Any,
    reference_values: Any,
    *,
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> dict[str, Any]:
    actual = np.ascontiguousarray(np.asarray(actual_values, dtype=np.float64))
    reference = np.ascontiguousarray(
        np.asarray(reference_values, dtype=np.float64)
    )
    if actual.shape != reference.shape:
        raise ValueError(
            f"comparison shape mismatch: {actual.shape} != {reference.shape}"
        )
    deviations = np.abs(actual - reference)
    max_abs_deviation = float(deviations.max()) if actual.size else 0.0
    actual_bits = actual.view(np.uint64)
    reference_bits = reference.view(np.uint64)
    mismatch_indices = np.argwhere(actual_bits != reference_bits)
    mismatches: list[dict[str, Any]] = []
    for row in mismatch_indices:
        index = tuple(int(value) for value in row)
        actual_value = float(actual[index])
        reference_value = float(reference[index])
        mismatches.append(
            {
                **_coordinate(index, axes, labels),
                "reference": reference_value,
                "reference_hex": reference_value.hex(),
                "registry": actual_value,
                "registry_hex": actual_value.hex(),
                "abs_deviation": float(abs(actual_value - reference_value)),
                "cause": _MISMATCH_CAUSE,
            }
        )
    return {
        "shape": list(actual.shape),
        "max_abs_deviation": max_abs_deviation,
        "ieee_uint64_equal": not mismatches,
        "mismatches": mismatches,
    }


def _exact_array_claim(
    actual_values: Any,
    reference_values: Any,
    *,
    axes: tuple[str, ...],
    labels: dict[str, list[Any]],
) -> dict[str, Any]:
    actual = np.asarray(actual_values, dtype=object)
    reference = np.asarray(reference_values, dtype=object)
    if actual.shape != reference.shape:
        raise ValueError(
            f"comparison shape mismatch: {actual.shape} != {reference.shape}"
        )
    mismatch_indices = np.argwhere(actual != reference)
    mismatches = [
        {
            **_coordinate(tuple(int(value) for value in row), axes, labels),
            "reference": reference[tuple(row)],
            "registry": actual[tuple(row)],
            "cause": _MISMATCH_CAUSE,
        }
        for row in mismatch_indices
    ]
    return {
        "exact_match": not mismatches,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }


def _structured_claim(actual: Any, reference: Any) -> dict[str, Any]:
    exact = actual == reference
    return {
        "exact_match": exact,
        "n_mismatches": int(not exact),
        "mismatches": (
            []
            if exact
            else [
                {
                    "reference": reference,
                    "registry": actual,
                    "cause": _MISMATCH_CAUSE,
                }
            ]
        ),
    }


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "n_bytes": len(payload),
    }


def build_compatibility_certificate(
    root: Path,
    evaluation: EvaluationResult,
    *,
    runner_path: Path,
) -> dict[str, Any]:
    """Diff every scored candidate-16 quantity and return the certificate."""
    reference_path = root / "runs" / "gate2_hazard_v16.json"
    reference = json.loads(reference_path.read_text())
    registry_evaluation = _normalized(evaluation.certificate_payload())
    seeds = evaluation.seeds
    cells = evaluation.cells
    draw_seeds = evaluation.draw_seeds
    by_seed = {row["seed"]: row for row in reference["per_seed"]}
    per_seed = [by_seed[seed] for seed in seeds]
    canonical_cube = reference["fresh_run_artifact_schema"][
        "per_draw_per_cell_rates"
    ]
    if (
        canonical_cube["shape"] != [len(draw_seeds), len(cells), len(seeds)]
        or canonical_cube["k_index_draw_seeds"] != draw_seeds
        or canonical_cube["cell_index"] != cells
        or canonical_cube["seed_index"] != seeds
    ):
        raise RuntimeError("candidate-16 canonical cube axes do not match")
    reference_rates = canonical_cube["rates"]
    per_seed_rates = _reference_cube(per_seed, cells, len(draw_seeds))
    canonical_bits = np.asarray(reference_rates, dtype=np.float64).view(
        np.uint64
    )
    per_seed_bits = np.asarray(per_seed_rates, dtype=np.float64).view(
        np.uint64
    )
    if not np.array_equal(canonical_bits, per_seed_bits):
        raise RuntimeError(
            "candidate-16 canonical cube differs from its per-seed rate copies"
        )
    labels = {"draw": draw_seeds, "cell": cells, "seed": seeds}
    cube_axes = ("draw", "cell", "seed")
    matrix_axes = ("cell", "seed")

    comparisons: dict[str, Any] = {
        "per_draw_per_cell_rates": _float_claim(
            registry_evaluation["per_draw_per_cell_rates"],
            reference_rates,
            axes=cube_axes,
            labels=labels,
        ),
        "per_cell_means": {
            field: _float_claim(
                registry_evaluation["per_cell_means"][field],
                _reference_matrix(per_seed, cells, field),
                axes=matrix_axes,
                labels=labels,
            )
            for field in ("rbar", "r_candidate")
        },
        "per_cell_scores": _float_claim(
            registry_evaluation["per_cell_scores"],
            _reference_matrix(per_seed, cells, "score"),
            axes=matrix_axes,
            labels=labels,
        ),
        "rate_a": _float_claim(
            registry_evaluation["rate_a"],
            _reference_matrix(per_seed, cells, "rate_a"),
            axes=matrix_axes,
            labels=labels,
        ),
        "tolerances": _float_claim(
            registry_evaluation["tolerances"],
            _reference_matrix(per_seed, cells, "tolerance"),
            axes=matrix_axes,
            labels=labels,
        ),
        "per_cell_pass": _exact_array_claim(
            registry_evaluation["per_cell_pass"],
            _reference_matrix(per_seed, cells, "pass"),
            axes=matrix_axes,
            labels=labels,
        ),
        "seed_verdicts": _exact_array_claim(
            registry_evaluation["seed_verdicts"],
            [row["seed_pass"] for row in per_seed],
            axes=("seed",),
            labels=labels,
        ),
        "seed_conjunction": _structured_claim(
            registry_evaluation["seed_conjunction"],
            reference["seed_conjunction"],
        ),
        "verdict": _structured_claim(
            registry_evaluation["verdict"], reference["verdict"]
        ),
    }

    registry_cube = np.asarray(
        registry_evaluation["per_draw_per_cell_rates"], dtype=np.float64
    )
    reference_cube = np.asarray(reference_rates, dtype=np.float64)
    per_gate_block: dict[str, Any] = {}
    for block, block_result in reference["verdict"]["per_block"].items():
        block_cells = block_result["cells"]
        indices = [cells.index(cell) for cell in block_cells]
        block_labels = {**labels, "cell": block_cells}
        per_gate_block[block] = {
            "cells": block_cells,
            "per_draw_per_cell_rates": _float_claim(
                registry_cube[:, indices, :],
                reference_cube[:, indices, :],
                axes=cube_axes,
                labels=block_labels,
            ),
            "per_cell_means": {
                field: _float_claim(
                    np.asarray(
                        registry_evaluation["per_cell_means"][field],
                        dtype=np.float64,
                    )[indices, :],
                    np.asarray(
                        _reference_matrix(per_seed, cells, field),
                        dtype=np.float64,
                    )[indices, :],
                    axes=matrix_axes,
                    labels=block_labels,
                )
                for field in ("rbar", "r_candidate")
            },
            "per_cell_scores": _float_claim(
                np.asarray(
                    registry_evaluation["per_cell_scores"], dtype=np.float64
                )[indices, :],
                np.asarray(
                    _reference_matrix(per_seed, cells, "score"),
                    dtype=np.float64,
                )[indices, :],
                axes=matrix_axes,
                labels=block_labels,
            ),
            "per_cell_pass": _exact_array_claim(
                np.asarray(registry_evaluation["per_cell_pass"], dtype=object)[
                    indices, :
                ],
                np.asarray(
                    _reference_matrix(per_seed, cells, "pass"), dtype=object
                )[indices, :],
                axes=matrix_axes,
                labels=block_labels,
            ),
        }

    float_claims = [
        comparisons["per_draw_per_cell_rates"],
        *comparisons["per_cell_means"].values(),
        comparisons["per_cell_scores"],
        comparisons["rate_a"],
        comparisons["tolerances"],
    ]
    exact_claims = [
        comparisons["per_cell_pass"],
        comparisons["seed_verdicts"],
        comparisons["seed_conjunction"],
        comparisons["verdict"],
    ]
    for block in per_gate_block.values():
        float_claims.extend(
            [
                block["per_draw_per_cell_rates"],
                *block["per_cell_means"].values(),
                block["per_cell_scores"],
            ]
        )
        exact_claims.append(block["per_cell_pass"])
    certificate_pass = all(
        claim["max_abs_deviation"] == 0.0 and claim["ieee_uint64_equal"]
        for claim in float_claims
    ) and all(claim["exact_match"] for claim in exact_claims)

    implementation_directory = (
        root / "src" / "populace_dynamics" / "models" / "family_transitions"
    )
    implementation_files = [
        _file_identity(path, root)
        for path in sorted(implementation_directory.rglob("*.py"))
    ]
    return {
        "schema_version": "registry_compatibility_certificate_v1",
        "reference": _file_identity(reference_path, root),
        "runner": _file_identity(runner_path, root),
        "implementation_files": implementation_files,
        "resolved_candidate_spec": {
            **CANDIDATE_16.canonical_dict(),
            "sha256": CANDIDATE_16.sha256,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": {
            "shape": [len(draw_seeds), len(cells), len(seeds)],
            "axes": {
                "order": ["draw", "cell", "seed"],
                "draw_seeds": draw_seeds,
                "cells": cells,
                "seeds": seeds,
            },
            "single_draw_provenance_seeds": list(SINGLE_DRAW_PROVENANCE_SEEDS),
            "outer_rng_note": (
                "default_rng(4200 + seed) is constructed as inherited "
                "single-draw provenance but not consumed by the amended "
                "scorer; scored simulations consume default_rng(5200 + k)"
            ),
        },
        "registry_evaluation": registry_evaluation,
        "comparisons": comparisons,
        "per_gate_block": per_gate_block,
        "certificate_pass": certificate_pass,
    }
