"""Synthetic-only tests for the §2.8.4 M6 scoring composition."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.harness.m6_cells import (
    earnings_cells,
    oc_4of5,
    run_floor,
)
from populace_dynamics.harness.m6_scoring import (
    DRAW_SEED_BASE,
    EARNINGS_CELL_NAMES,
    GATE_SEEDS,
    GATED_CELL_NAMES,
    M6CellRule,
    M6GateContract,
    aggregate_gate,
    earnings_domain_person_ids,
    recompute_domain_earnings_floor,
    reduce_gated_cells,
    restrict_earnings_domain_support,
    score_gate_seed,
)
from populace_dynamics.harness.panel import split_panel_by_person


def _contract() -> M6GateContract:
    rules = []
    for cell in GATED_CELL_NAMES:
        family = (
            "earnings"
            if cell.startswith("earn_")
            else (
                "disability"
                if cell.startswith(("incidence", "recovery"))
                else "marital"
            )
        )
        metric = {
            "earn_autocorr_lag2": "abs_gap_corr",
            "earn_dlog_mean.prime": "abs_gap_log",
        }.get(cell, "log_ratio")
        rules.append(
            M6CellRule(
                cell=cell,
                family=family,
                split_unit="household" if family == "marital" else "person",
                metric=metric,
                tolerance=0.5,
                k=3,
                rounding=3,
            )
        )
    return M6GateContract(
        cells=tuple(rules),
        gate_seeds=GATE_SEEDS,
        required_seed_passes=4,
        floor_run_sha256="synthetic",
    )


def _truth_cells(contract: M6GateContract) -> dict[str, dict]:
    cells = {}
    for index, rule in enumerate(contract.cells):
        value = 0.2 + index / 100
        cells[rule.cell] = (
            {"rate": value}
            if rule.metric == "log_ratio"
            else {"value": value, "metric": rule.metric}
        )
    return cells


def _draws(
    contract: M6GateContract,
    truth: dict[str, dict],
    *,
    tilt: float = 1.0,
) -> list[dict[str, dict]]:
    draws = []
    for k in range(contract.n_draws):
        direction = -1.0 if k % 2 == 0 else 1.0
        draw = {}
        for rule in contract.cells:
            record = truth[rule.cell]
            value = float(record.get("rate", record.get("value")))
            if rule.metric == "log_ratio":
                value *= tilt * (1.0 + 0.01 * direction)
                draw[rule.cell] = {"rate": value}
            else:
                value = value * tilt + 0.001 * direction
                draw[rule.cell] = {"value": value, "metric": rule.metric}
        draws.append(draw)
    return draws


def _earnings_panel(n_people: int = 80) -> pd.DataFrame:
    rows = []
    for person_id in range(n_people):
        older = person_id >= n_people // 2
        cohort = "older" if older else "prime"
        age = 50 if older else 35
        base = 20_000.0 + 173.0 * person_id
        for step, period in enumerate((2014, 2016, 2018)):
            zero = older and person_id % 7 == 0 and step != 1
            growth = 1.0 + step * (0.02 + 0.003 * (person_id % 9))
            rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "age": age + 2 * step,
                    "earnings": 0.0 if zero else base * growth,
                    "weight": 1.0 + (person_id % 5) / 10,
                    "cohort": cohort,
                }
            )
    return pd.DataFrame(rows)


def test_reduce_gated_cells_composes_the_frozen_v3_surface():
    marital_py = pd.DataFrame(
        [
            {
                "person_id": 1,
                "age": 25,
                "sex": "female",
                "weight": 1.0,
                "marital_state": "never_married",
            },
            {
                "person_id": 2,
                "age": 35,
                "sex": "male",
                "weight": 2.0,
                "marital_state": "married",
            },
            {
                "person_id": 3,
                "age": 50,
                "sex": "female",
                "weight": 3.0,
                "marital_state": "divorced",
            },
        ]
    )
    marital_events = pd.DataFrame(
        [
            {**marital_py.iloc[0].to_dict(), "transition": "first_marriage"},
            {**marital_py.iloc[1].to_dict(), "transition": "divorce"},
            {**marital_py.iloc[2].to_dict(), "transition": "remarriage"},
        ]
    )
    disability = pd.DataFrame(
        [
            {
                "person_id": 4,
                "age": 30,
                "sex": "female",
                "weight": 1.0,
                "from_disabled": False,
                "to_disabled": True,
            },
            {
                "person_id": 5,
                "age": 50,
                "sex": "male",
                "weight": 1.0,
                "from_disabled": True,
                "to_disabled": False,
            },
        ]
    )
    cells = reduce_gated_cells(
        marital_events,
        marital_py,
        disability,
        _earnings_panel(),
    )
    assert tuple(cells) == GATED_CELL_NAMES
    assert cells["first_marriage.18-29|female"]["rate"] == 1.0
    assert cells["divorce.18-44"]["rate"] == 1.0
    assert cells["remarriage.18-64"]["rate"] == 1.0
    assert cells["incidence.20-66"]["rate"] == 1.0
    assert cells["recovery.20-66"]["rate"] == 1.0


def test_score_seed_means_draws_once_and_applies_both_conformance_guards():
    contract = _contract()
    truth = _truth_cells(contract)
    valid = score_gate_seed(
        contract,
        seed=0,
        truth_cells=truth,
        projected_draw_cells=_draws(contract, truth),
        n_side_a_units=123,
    )
    assert valid.valid is True
    assert valid.passed is True
    assert valid.n_cells_passed == 11
    assert all(cell.regenerated for cell in valid.cells)
    assert all(cell.rbar == pytest.approx(cell.rate_a) for cell in valid.cells)

    bad_draws = _draws(contract, truth)
    bad_draws[0]["earn_p10.prime"] = {"rate": 0.0}
    constant = truth["divorce.18-44"]["rate"]
    for draw in bad_draws:
        draw["divorce.18-44"] = {"rate": constant}
    invalid = score_gate_seed(
        contract,
        seed=0,
        truth_cells=truth,
        projected_draw_cells=bad_draws,
    )
    assert invalid.valid is False
    assert invalid.passed is False
    assert invalid.undefined_draw_cells == ("earn_p10.prime",)
    assert invalid.non_regenerated_cells == ("divorce.18-44",)
    artifact = invalid.to_artifact()
    assert artifact["n_cells_fail"] >= 1
    assert len(artifact["worst_cells"]) == 5


def test_gate_uses_the_seed_level_four_of_five_conjunction():
    contract = _contract()
    truth = _truth_cells(contract)
    seeds = []
    for seed in contract.gate_seeds:
        tilt = 2.0 if seed == 4 else 1.0
        seeds.append(
            score_gate_seed(
                contract,
                seed=seed,
                truth_cells=truth,
                projected_draw_cells=_draws(contract, truth, tilt=tilt),
            )
        )
    result = aggregate_gate(contract, seeds)
    assert result.valid is True
    assert result.n_seeds_passed == 4
    assert result.passed is True
    conformance = result.to_artifact()["conformance"]
    assert conformance["regenerated_surface"] is True
    assert conformance["identity_candidate"] is False
    assert conformance["max_across_draw_sd"] > 0
    assert conformance["max_per_draw_abs_ln"] > 0

    copied = aggregate_gate(
        contract,
        [
            score_gate_seed(
                contract,
                seed=seed,
                truth_cells=truth,
                projected_draw_cells=[truth] * contract.n_draws,
            )
            for seed in contract.gate_seeds
        ],
    )
    copied_conformance = copied.to_artifact()["conformance"]
    assert copied_conformance["regenerated_surface"] is False
    assert copied_conformance["identity_candidate"] is True


def test_earnings_domain_filter_is_both_sided_and_separate_from_reduction():
    truth = pd.DataFrame(
        [
            {"person_id": pid, "period": year, "earnings": pid * 100 + year}
            for pid in (1, 2, 3)
            for year in (2014, 2016, 2018)
        ]
    )
    projection = pd.concat(
        [
            truth.assign(earnings=truth.earnings * 1.01),
            pd.DataFrame(
                [{"person_id": 1, "period": 2020, "earnings": 999.0}]
            ),
        ],
        ignore_index=True,
    )
    domain = earnings_domain_person_ids(
        (1, 2, 3),
        {1: 10.0, 2: 20.0},
        {1: 0.1, 2: 0.2, 3: 0.3},
    )
    projected, realized = restrict_earnings_domain_support(
        projection, truth, domain
    )
    assert domain == frozenset({1, 2})
    assert set(projected.person_id) == set(realized.person_id) == {1, 2}
    assert len(projected) == len(realized) == 6
    assert set(projected.period) == {2014, 2016, 2018}

    missing = projection[
        ~((projection.person_id == 2) & (projection.period == 2018))
    ]
    with pytest.raises(ValueError, match="every realized in-domain"):
        restrict_earnings_domain_support(missing, truth, domain)


def test_domain_floor_recompute_publishes_both_escalation_directions():
    contract = _contract()
    earnings = _earnings_panel()
    anchor = pd.DataFrame(
        {
            "person_id": np.arange(80),
            "household_id": np.arange(80),
        }
    )
    tripwire_anchor = pd.DataFrame(
        {"person_id": [1, 2, 3], "household_id": [1, 2, 3]}
    )
    tripwire_domain = {1, 3}
    full_left, _ = split_panel_by_person(tripwire_anchor, "person_id", seed=1)
    domain_left, _ = split_panel_by_person(
        tripwire_anchor[tripwire_anchor.person_id.isin(tripwire_domain)],
        "person_id",
        seed=1,
    )
    assert set(full_left.person_id) & tripwire_domain == {3}
    assert set(domain_left.person_id) == set()

    domain = set(range(80)) - {1}
    domain_earnings = earnings[earnings.person_id.isin(domain)].copy()

    def compute(person_ids):
        selected = set(person_ids) & domain
        return earnings_cells(
            domain_earnings[domain_earnings.person_id.isin(selected)]
        )

    manual_floor, _ = run_floor(anchor, compute, "person_id")
    legacy_floor, _ = run_floor(
        anchor[anchor.person_id.isin(domain)], compute, "person_id"
    )
    assert any(
        manual_floor[cell][statistic] != legacy_floor[cell][statistic]
        for cell in EARNINGS_CELL_NAMES
        for statistic in ("mean", "sd")
    )
    result = recompute_domain_earnings_floor(
        anchor,
        earnings,
        domain,
        contract,
    )
    assert result["truth_side_only"] is True
    assert result["frozen_tolerances_remain_gated_contract"] is True
    assert set(result["per_cell"]) == set(EARNINGS_CELL_NAMES)
    for cell in EARNINGS_CELL_NAMES:
        published = result["per_cell"][cell]["domain_floor"]
        assert published["mean"] == pytest.approx(manual_floor[cell]["mean"])
        assert published["sd"] == pytest.approx(manual_floor[cell]["sd"])
    locked_tolerances = {
        rule.cell: rule.tolerance
        for rule in contract.cells
        if rule.family == "earnings"
    }
    assert result["oc"]["locked_tolerances_on_domain"] == oc_4of5(
        manual_floor,
        locked_tolerances,
        EARNINGS_CELL_NAMES,
    )
    escalation = result["two_directional_escalation"]
    assert set(escalation) >= {
        "near_unpassable",
        "vacuity",
        "escalates_to_floors_ceremony_finding",
    }
    assert escalation["escalates_to_floors_ceremony_finding"] == (
        escalation["near_unpassable"] or escalation["vacuity"]
    )
    assert (
        result["oc"]["locked_tolerances_on_domain"]["p_gate_pass_4_of_5"]
        >= 0.0
    )
    assert contract.draw_seeds[0] == DRAW_SEED_BASE


def test_aggregate_rejects_missing_or_duplicate_seed_results():
    contract = _contract()
    truth = _truth_cells(contract)
    one = score_gate_seed(
        contract,
        seed=0,
        truth_cells=truth,
        projected_draw_cells=_draws(contract, truth),
    )
    with pytest.raises(ValueError, match="do not match"):
        aggregate_gate(contract, [one])
    with pytest.raises(ValueError, match="do not match"):
        aggregate_gate(contract, [one, replace(one, seed=0)])
