"""Gate-2b evaluation for the candidate-9 component registry.

The one-draw moment builder and K=20 mean-of-draws scorer port
``scripts/run_gate2b_candidate9.py:442-630``. Cube assembly ports
``scripts/run_gate2b_candidate9.py:671-700`` and verdict assembly ports
``scripts/run_gate2b_candidate9.py:757-814``. This module contains no frozen
runner or ``household_composition_sim_v*`` imports; the registry runner is only
a CLI and artifact boundary around :func:`evaluate_candidate9`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as harness_panel
from populace_dynamics.models.household_composition.data import load_sources
from populace_dynamics.models.household_composition.registry import (
    CANDIDATE_9,
    REGISTRY,
    fit_context_from_sources,
)
from populace_dynamics.models.household_composition.simulator import simulate

__all__ = [
    "DRAW_SEEDS",
    "GATE_SEEDS",
    "SINGLE_DRAW_PROVENANCE_SEEDS",
    "EvaluationResult",
    "evaluate_candidate9",
]

GATE_SEEDS = (0, 1, 2, 3, 4)
SIMULATION_SEED_BASE = 4200
DRAW_SEED_BASE = 5200
N_DRAWS = 20
DRAW_SEEDS = tuple(DRAW_SEED_BASE + draw for draw in range(N_DRAWS))
SINGLE_DRAW_PROVENANCE_SEEDS = tuple(
    SIMULATION_SEED_BASE + seed for seed in GATE_SEEDS
)


@dataclass(frozen=True)
class EvaluationResult:
    """All candidate quantities certified against the frozen c9 artifact."""

    cells: list[str]
    seeds: list[int]
    draw_seeds: list[int]
    per_draw_per_cell_rates: list[list[list[float]]]
    per_cell_means: dict[str, list[list[float]]]
    per_cell_scores: list[list[float]]
    rate_a: list[list[float]]
    tolerances: list[list[float]]
    per_cell_pass: list[list[bool]]
    seed_verdicts: list[bool]
    seed_conjunction: list[dict[str, Any]]
    verdict: dict[str, Any]

    def certificate_payload(self) -> dict[str, Any]:
        """Return the JSON-ready registry evaluation embedded in a certificate."""
        return {
            "per_draw_per_cell_rates": self.per_draw_per_cell_rates,
            "per_cell_means": self.per_cell_means,
            "per_cell_scores": self.per_cell_scores,
            "rate_a": self.rate_a,
            "tolerances": self.tolerances,
            "per_cell_pass": self.per_cell_pass,
            "seed_verdicts": self.seed_verdicts,
            "seed_conjunction": self.seed_conjunction,
            "verdict": self.verdict,
        }


def _load_contract(root: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Load the locked gate-2b tolerances and committed side-A rates.

    This ports the contract checks at
    ``scripts/run_gate2b_candidate9.py:1268-1288``.
    """
    thresholds = yaml.safe_load((root / "gates.yaml").read_text())["gates"][
        "gate_2"
    ]["gate_2b"]["thresholds"]
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2b must remain locked for registry evaluation"
        )
    tolerances: dict[str, float] = {}
    for view in thresholds["views"].values():
        for cell, value in view["tolerances"].items():
            tolerances[cell] = float(value)
    if len(tolerances) != 46:
        raise RuntimeError(f"expected 46 gated cells, found {len(tolerances)}")
    floor = json.loads((root / "runs" / "gate2b_floors_v1.json").read_text())
    if set(tolerances) != set(floor["gate_partition"]["gate_eligible"]):
        raise RuntimeError("locked tolerances and floor gate partition differ")
    return tolerances, floor


def _draw_moments(
    sources: dict[str, Any],
    holdout_ids: set[int],
    fitted: Any,
    draw_seed: int,
) -> dict[str, dict[str, float]]:
    """Simulate and measure one draw in candidate-9 operation order."""
    simulated_panel = simulate(
        sources["hh"],
        sources["mpanel"],
        fitted,
        holdout_ids,
        draw_seed,
    )
    return hc.reference_moments(simulated_panel, holdout_ids, weighted=True)


def _score_seed(
    seed: int,
    sources: dict[str, Any],
    floor: dict[str, Any],
    tolerances: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    """Fit and score one gate seed exactly as candidate 9 did."""
    started = time.time()
    side_a, side_b = harness_panel.split_panel_by_person(
        sources["hh"].attrs,
        "person_id",
        fraction=0.5,
        seed=seed,
    )
    holdout_ids = set(int(value) for value in side_a.person_id.unique())
    train_ids = frozenset(int(value) for value in side_b.person_id.unique())
    context = fit_context_from_sources(sources, train_ids)
    fitted = REGISTRY.fit(CANDIDATE_9, context)

    # Construct candidate 9's inherited 4200 + seed outer stream for protocol
    # provenance, exactly as the merged registry precedent does.  It is not
    # advanced or coupled to the amended K=20 draws at 5200 + k.
    outer_rng = np.random.default_rng(SIMULATION_SEED_BASE + seed)
    del outer_rng

    committed = {row["seed"]: row for row in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]
    cells = sorted(tolerances)
    per_draw_rate: dict[str, list[float]] = {cell: [] for cell in cells}
    per_draw_denominator: dict[str, list[float]] = {cell: [] for cell in cells}
    for draw_seed in DRAW_SEEDS:
        moments = _draw_moments(sources, holdout_ids, fitted, draw_seed)
        for cell in cells:
            record = moments[cell]
            per_draw_rate[cell].append(float(record["rate"]))
            per_draw_denominator[cell].append(float(record.get("den_wt", 0.0)))

    undefined = [
        {
            "cell": cell,
            "draw_k": draw,
            "draw_seed": DRAW_SEEDS[draw],
        }
        for cell in cells
        for draw in range(N_DRAWS)
        if per_draw_denominator[cell][draw] <= 0.0
    ]
    if undefined:
        raise RuntimeError(f"undefined gated registry draws: {undefined!r}")

    gated_cells: dict[str, dict[str, Any]] = {}
    n_pass = 0
    for cell in cells:
        rate_a = float(committed[cell]["rate_a"])
        rates = np.asarray(per_draw_rate[cell], dtype=np.float64)
        # Preserve c9's contiguous, one-cell-at-a-time reduction topology.
        mean = float(rates.mean())
        score = (
            float(abs(math.log(mean / rate_a)))
            if mean > 0 and rate_a > 0
            else float("inf")
        )
        passed = bool(score <= tolerances[cell])
        n_pass += passed
        gated_cells[cell] = {
            "r_candidate": mean,
            "rbar": mean,
            "rate_a": rate_a,
            "score": score,
            "per_draw_rate": [float(rate) for rate in rates],
            "tolerance": float(tolerances[cell]),
            "pass": passed,
        }
    seed_pass = n_pass == len(cells)
    result = {
        "seed": seed,
        "gated_cells": gated_cells,
        "n_gated": len(cells),
        "n_gated_pass": n_pass,
        "n_gated_fail": len(cells) - n_pass,
        "seed_pass": bool(seed_pass),
    }
    if verbose:
        failures = [
            cell for cell, record in gated_cells.items() if not record["pass"]
        ]
        elapsed = time.time() - started
        print(
            f"seed {seed}: {n_pass}/{len(cells)} pass; "
            f"failures={failures}; {elapsed:.1f}s"
        )
    return result


_FAMILY_PREFIX = (
    ("coresident_spouse", lambda key: key.startswith("coresident_spouse.")),
    ("coresident_parent", lambda key: key.startswith("coresident_parent.")),
    ("coresident_child", lambda key: key.startswith("coresident_child.")),
    (
        "coresident_grandchild",
        lambda key: key.startswith("coresident_grandchild."),
    ),
    ("multigen_stock", lambda key: key.startswith("multigen.")),
    (
        "multigen_transition",
        lambda key: key in ("multigen_entry", "multigen_exit"),
    ),
    ("parental_home_exit", lambda key: key.startswith("parental_home_exit.")),
    ("hh_size", lambda key: key.startswith("hh_size.")),
)


def _family_of(cell: str) -> str:
    for family, predicate in _FAMILY_PREFIX:
        if predicate(cell):
            return family
    return "other"


def _build_verdict(
    per_seed: list[dict[str, Any]], tolerances: dict[str, float]
) -> dict[str, Any]:
    """Assemble the exact frozen candidate-9 gate verdict."""
    seed_pass = {row["seed"]: row["seed_pass"] for row in per_seed}
    n_seeds_pass = sum(seed_pass.values())
    all_failing = [
        {
            "cell": cell,
            "seed": seed_result["seed"],
            "family": _family_of(cell),
            "score": seed_result["gated_cells"][cell]["score"],
            "tolerance": seed_result["gated_cells"][cell]["tolerance"],
            "r_candidate": seed_result["gated_cells"][cell]["r_candidate"],
            "rate_a": seed_result["gated_cells"][cell]["rate_a"],
        }
        for seed_result in per_seed
        for cell in sorted(tolerances)
        if not seed_result["gated_cells"][cell]["pass"]
    ]
    return {
        "n_gate_seeds": len(per_seed),
        "n_gated_cells": len(tolerances),
        "seed_pass": seed_pass,
        "n_seeds_pass": n_seeds_pass,
        "gate_2b_pass": bool(n_seeds_pass >= 4),
        "rule": (
            "A seed passes iff every one of the 46 gated cells holds "
            "(|ln(rbar / rate_a)| <= locked tolerance); the gate passes iff "
            ">= 4 of the 5 gate seeds pass."
        ),
        "all_failing_gated_cells": all_failing,
    }


def evaluate_candidate9(
    root: Path,
    *,
    verbose: bool = True,
) -> EvaluationResult:
    """Fit and score resolved candidate 9 on seeds 0-4 and draws 5200-5219."""
    tolerances, floor = _load_contract(root)
    sources = load_sources()
    per_seed = [
        _score_seed(seed, sources, floor, tolerances, verbose)
        for seed in GATE_SEEDS
    ]
    cells = sorted(tolerances)
    seeds = list(GATE_SEEDS)
    by_seed = {row["seed"]: row for row in per_seed}
    cube = [
        [
            [
                float(
                    by_seed[seed]["gated_cells"][cell]["per_draw_rate"][draw]
                )
                for seed in seeds
            ]
            for cell in cells
        ]
        for draw in range(N_DRAWS)
    ]

    def matrix(field: str) -> list[list[Any]]:
        return [
            [by_seed[seed]["gated_cells"][cell][field] for seed in seeds]
            for cell in cells
        ]

    seed_conjunction = [
        {
            "seed": row["seed"],
            "n_gated_pass": row["n_gated_pass"],
            "n_gated_fail": row["n_gated_fail"],
            "seed_pass": row["seed_pass"],
        }
        for row in per_seed
    ]
    return EvaluationResult(
        cells=cells,
        seeds=seeds,
        draw_seeds=list(DRAW_SEEDS),
        per_draw_per_cell_rates=cube,
        per_cell_means={
            "rbar": matrix("rbar"),
            "r_candidate": matrix("r_candidate"),
        },
        per_cell_scores=matrix("score"),
        rate_a=matrix("rate_a"),
        tolerances=matrix("tolerance"),
        per_cell_pass=matrix("pass"),
        seed_verdicts=[row["seed_pass"] for row in per_seed],
        seed_conjunction=seed_conjunction,
        verdict=_build_verdict(per_seed, tolerances),
    )
