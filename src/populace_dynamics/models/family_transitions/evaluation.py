"""Gate-2 evaluation for the candidate-16 component registry.

The one-draw moment builder and K=20 mean-of-draws scorer port
``scripts/run_gate2_candidate10.py:777-941``. Verdict assembly ports
``scripts/run_gate2_candidate1.py:1142-1228``. This module contains no frozen
runner imports; ``scripts/run_gate2_registry.py`` is only a CLI and artifact
boundary around :func:`evaluate_candidate16`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from populace_dynamics.data import (
    births,
    deaths,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.harness import panel as harness_panel
from populace_dynamics.models.family_transitions.common import (
    marriage_order_map,
)
from populace_dynamics.models.family_transitions.registry import (
    CANDIDATE_16,
    REGISTRY,
    FitContext,
)
from populace_dynamics.models.family_transitions.simulator import simulate

__all__ = [
    "DRAW_SEEDS",
    "GATE_SEEDS",
    "SINGLE_DRAW_PROVENANCE_SEEDS",
    "EvaluationResult",
    "evaluate_candidate16",
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
class _SourceBundle:
    panel: transitions.MaritalPanel
    fertility_panel: transitions.FertilityPanel
    demographic_panel: pd.DataFrame
    marriage_records: pd.DataFrame
    birth_records: pd.DataFrame
    order_map: pd.DataFrame


@dataclass(frozen=True)
class EvaluationResult:
    """All candidate quantities certified against the frozen c16 artifact."""

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


def _load_sources() -> _SourceBundle:
    """Build the same marital and fertility panels as the frozen evaluation.

    This copies ``scripts/build_gate2_floors.py:148-186`` into the registry
    application boundary, using only source-package data readers.
    """
    marriage_records = marriage.marriage_history()
    death_records = deaths.read_death_records()
    birth_records = births.birth_history()
    demographic_panel = panels.demographic_panel()
    positive_weight = demographic_panel[demographic_panel.weight > 0]
    person_weight = (
        positive_weight.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    marital_panel = transitions.build_marital_panel(
        marriage_records, death_records, person_weight
    )
    fertility_panel = transitions.build_fertility_panel(
        marital_panel, birth_records
    )
    return _SourceBundle(
        panel=marital_panel,
        fertility_panel=fertility_panel,
        demographic_panel=demographic_panel,
        marriage_records=marriage_records,
        birth_records=birth_records,
        order_map=marriage_order_map(marriage_records),
    )


def _load_contract(root: Path) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = yaml.safe_load((root / "gates.yaml").read_text())["gates"][
        "gate_2"
    ]["thresholds"]
    if not thresholds.get("locked", False):
        raise RuntimeError("gate_2 must remain locked for registry evaluation")
    tolerances: dict[str, float] = {}
    for view in thresholds["views"].values():
        for cell, value in view["tolerances"].items():
            tolerances[cell] = float(value)
    if len(tolerances) != 46:
        raise RuntimeError(f"expected 46 gated cells, found {len(tolerances)}")
    floor = json.loads((root / "runs" / "gate2_floors_v2.json").read_text())
    if set(tolerances) != set(floor["gate_partition"]["gate_eligible"]):
        raise RuntimeError("locked tolerances and floor gate partition differ")
    return tolerances, floor


def _draw_moments(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Any,
    draw_seed: int,
) -> dict[str, dict[str, float]]:
    simulated_panel, simulated_births = simulate(
        panel, holdout_ids, components, draw_seed
    )
    simulated_fertility = transitions.build_fertility_panel(
        simulated_panel, simulated_births
    )
    return transitions.reference_moments(
        simulated_panel,
        simulated_fertility,
        holdout_ids,
        weighted=True,
    )


def _score_seed(
    seed: int,
    sources: _SourceBundle,
    floor: dict[str, Any],
    tolerances: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    started = time.time()
    side_a, side_b = harness_panel.split_panel_by_person(
        sources.panel.attrs,
        "person_id",
        fraction=0.5,
        seed=seed,
    )
    holdout_ids = set(int(value) for value in side_a.person_id.unique())
    train_ids = frozenset(int(value) for value in side_b.person_id.unique())
    context = FitContext(
        panel=sources.panel,
        demographic_panel=sources.demographic_panel,
        marriage_records=sources.marriage_records,
        birth_records=sources.birth_records,
        marriage_order_map=sources.order_map,
        train_ids=train_ids,
    )
    components = REGISTRY.fit(CANDIDATE_16, context)

    # Candidate 16 records default_rng(4200 + seed) as the inherited outer
    # single-draw stream but does not consume it in the amended K=20 scorer.
    # Constructing it makes the registry protocol explicit without advancing
    # or coupling it to any scored draw.
    outer_rng = np.random.default_rng(SIMULATION_SEED_BASE + seed)
    del outer_rng

    committed = {row["seed"]: row for row in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]
    cells = sorted(tolerances)
    per_draw_rate: dict[str, list[float]] = {cell: [] for cell in cells}
    per_draw_denominator: dict[str, list[float]] = {cell: [] for cell in cells}
    for draw_seed in DRAW_SEEDS:
        moments = _draw_moments(
            sources.panel, holdout_ids, components, draw_seed
        )
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


_FAMILY_PREFIX = {
    "first_marriage": lambda key: key.startswith("first_marriage"),
    "divorce": lambda key: key.startswith("divorce."),
    "widowhood": lambda key: key.startswith("widowhood."),
    "remarriage": lambda key: key.startswith("remarriage."),
    "occupancy": lambda key: key.startswith("ever_married_by_")
    and "|" in key
    or key.startswith("mean_lifetime_marriages"),
    "nuptiality_cohort": lambda key: key.startswith("ever_married_by_40.c"),
    "stock_occupancy": lambda key: key.startswith("share_"),
    "fertility": lambda key: key.startswith("asfr.")
    or key.startswith("completed_fertility."),
}


def _family_of(cell: str) -> str:
    for family, predicate in _FAMILY_PREFIX.items():
        if predicate(cell):
            return family
    return "other"


def _build_verdict(
    per_seed: list[dict[str, Any]], tolerances: dict[str, float]
) -> dict[str, Any]:
    seed_pass = {row["seed"]: row["seed_pass"] for row in per_seed}
    n_seeds_pass = sum(seed_pass.values())
    families = sorted({_family_of(cell) for cell in tolerances})
    per_block: dict[str, Any] = {}
    for family in families:
        cells = sorted(
            cell for cell in tolerances if _family_of(cell) == family
        )
        by_seed: dict[int, dict[str, int]] = {}
        failing: list[dict[str, Any]] = []
        for seed_result in per_seed:
            n_pass = sum(
                seed_result["gated_cells"][cell]["pass"] for cell in cells
            )
            by_seed[seed_result["seed"]] = {
                "n_pass": n_pass,
                "n_cells": len(cells),
            }
            for cell in cells:
                record = seed_result["gated_cells"][cell]
                if not record["pass"]:
                    failing.append(
                        {
                            "cell": cell,
                            "seed": seed_result["seed"],
                            "score": record["score"],
                            "tolerance": record["tolerance"],
                            "r_candidate": record["r_candidate"],
                            "rate_a": record["rate_a"],
                        }
                    )
        per_block[family] = {
            "cells": cells,
            "n_cells": len(cells),
            "per_seed_pass": by_seed,
            "failing_cells": failing,
        }
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
        "gate_2_pass": bool(n_seeds_pass >= 4),
        "rule": (
            "A seed passes iff every one of the 46 gated cells holds "
            "(|ln(r_candidate / rate_a)| <= locked tolerance); the gate "
            "passes iff >= 4 of the 5 gate seeds pass."
        ),
        "per_block": per_block,
        "all_failing_gated_cells": all_failing,
    }


def evaluate_candidate16(
    root: Path,
    *,
    verbose: bool = True,
) -> EvaluationResult:
    """Fit and score resolved candidate 16 on seeds 0-4 and draws 5200-5219."""
    tolerances, floor = _load_contract(root)
    sources = _load_sources()
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
