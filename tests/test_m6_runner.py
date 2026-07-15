"""Synthetic orchestration tests for the one-shot M6 runner.

These tests never load PSID data and never invoke a projection.  The production
phases are replaced with recording seams so ordering and commit-last behavior
can be checked directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import populace_dynamics.harness.m6_runner as runner_module
from populace_dynamics.harness.m6_runner import (
    FROZEN_FLOOR_SHA256,
    M6RefitPhase,
    M6ResolvedContract,
    M6RunnerOperations,
    M6SeedRun,
    build_report_only,
    contract_from_gate_document,
    execute_registered_m6_run,
    guard_registered_m6_run,
    resolve_m6_contract,
    validate_registration_id,
)
from populace_dynamics.harness.m6_scoring import M6GateContract

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _FakeSeedScore:
    seed: int


@dataclass(frozen=True)
class _FakeGateScore:
    seeds: tuple[_FakeSeedScore, ...]
    valid: bool = True
    passed: bool = True

    def to_artifact(self):
        return {
            "valid": self.valid,
            "pass": self.passed,
            "per_seed": [
                {"seed": score.seed, "n_side_a_units": None}
                for score in self.seeds
            ],
        }


def _synthetic_operations(events, *, fail_at=None):
    contract = M6GateContract(
        cells=(),
        gate_seeds=(0, 1),
        required_seed_passes=1,
        n_draws=2,
        floor_run="synthetic-floor.json",
        floor_run_sha256="synthetic-sha",
    )

    def record(name, value=None):
        events.append(name)
        if fail_at == name:
            raise RuntimeError(f"synthetic failure at {name}")
        return value

    def resolve(_root):
        return record(
            "resolve",
            M6ResolvedContract(
                contract=contract,
                floor_artifact={},
                floor_path="synthetic-floor.json",
                floor_sha256="synthetic-sha",
            ),
        )

    def refit(_inputs):
        return record(
            "refit",
            M6RefitPhase(
                bundle=None,
                mortality=None,
                population=None,
                lineage={"synthetic": True},
            ),
        )

    def preflight_1(_inputs, _phase, _contract):
        return record("preflight_1", {"passed": True})

    def preflight_2(_inputs, _phase, _contract):
        return record(
            "preflight_2",
            {"branch": "certified_target_models_reconstruction"},
        )

    def score_seed(_inputs, _phase, _contract, seed):
        return record(
            f"score_{seed}",
            M6SeedRun(
                seed=seed,
                score=_FakeSeedScore(seed),
                side_a_units={"household": 2, "person": 3},
            ),
        )

    def aggregate(_contract, scores):
        return record("aggregate", _FakeGateScore(tuple(scores)))

    def domain_floor(_inputs, _phase, _resolved):
        return record(
            "domain_floor",
            {
                "truth_side_only": True,
                "two_directional_escalation": {
                    "escalates_to_floors_ceremony_finding": False
                },
            },
        )

    def report(_inputs, _phase, _resolved, _runs, domain):
        return record(
            "report_only",
            {
                "domain_earnings_floor": domain,
                "family_b": {"gating": False},
                "family_c": {"gating": False},
            },
        )

    def write(_path, _artifact):
        record("write")

    return M6RunnerOperations(
        resolve_contract=resolve,
        refit=refit,
        preflight_1=preflight_1,
        preflight_2=preflight_2,
        score_seed=score_seed,
        aggregate=aggregate,
        domain_floor=domain_floor,
        report_only=report,
        write=write,
    )


def test_runner_orders_phases_and_commits_last(tmp_path):
    events = []
    inputs = SimpleNamespace(provenance={"synthetic": True})
    artifact = execute_registered_m6_run(
        inputs,
        registration_id="9999999999",
        output=tmp_path / "fresh.json",
        root=ROOT,
        operations=_synthetic_operations(events),
    )
    assert events == [
        "resolve",
        "refit",
        "preflight_1",
        "preflight_2",
        "score_0",
        "score_1",
        "aggregate",
        "domain_floor",
        "report_only",
        "write",
    ]
    assert artifact["verdict"]["pass"] is True
    assert artifact["family_a"]["per_seed"][0]["n_side_a_units"] == {
        "household": 2,
        "person": 3,
    }


def test_runner_writes_nothing_when_an_earlier_phase_fails(tmp_path):
    events = []
    with pytest.raises(RuntimeError, match="preflight_1"):
        execute_registered_m6_run(
            SimpleNamespace(provenance={}),
            registration_id="9999999999",
            output=tmp_path / "never.json",
            root=ROOT,
            operations=_synthetic_operations(events, fail_at="preflight_1"),
        )
    assert events == ["resolve", "refit", "preflight_1"]
    assert "write" not in events
    assert not (tmp_path / "never.json").exists()


def test_runner_payload_is_deterministic_for_identical_phases(tmp_path):
    first_events = []
    second_events = []
    inputs = SimpleNamespace(provenance={"source": "synthetic"})
    first = execute_registered_m6_run(
        inputs,
        registration_id="9999999999",
        output=tmp_path / "one.json",
        root=ROOT,
        operations=_synthetic_operations(first_events),
    )
    second = execute_registered_m6_run(
        inputs,
        registration_id="9999999999",
        output=tmp_path / "two.json",
        root=ROOT,
        operations=_synthetic_operations(second_events),
    )
    assert first == second


def test_gate_fence_ignores_sibling_gates_and_non_protocol_fields():
    document = yaml.safe_load((ROOT / "gates.yaml").read_text())

    class Poison:
        def __getattribute__(self, name):
            raise AssertionError(f"forbidden gate field accessed: {name}")

    document["gates"]["forbidden_sibling"] = Poison()
    document["gates"]["gate_m6"]["not_certified"] = Poison()
    contract = contract_from_gate_document(document)
    assert len(contract.cells) == 11
    assert contract.floor_run_sha256 == FROZEN_FLOOR_SHA256


def test_frozen_floor_is_byte_verified():
    resolved = resolve_m6_contract(ROOT)
    assert resolved.floor_sha256 == FROZEN_FLOOR_SHA256
    assert resolved.floor_path == "runs/m6_holdout_floors_v3.json"


def test_report_only_marks_unavailable_paths_without_fabricating_zero():
    phase = M6RefitPhase(
        bundle=SimpleNamespace(mortality=None),
        mortality=None,
        population=SimpleNamespace(
            holdout_ids=frozenset({1, 2}),
            earnings_domain_ids=frozenset({1}),
        ),
        lineage={},
    )
    resolved = M6ResolvedContract(
        contract=M6GateContract(
            cells=(), gate_seeds=(), required_seed_passes=0
        ),
        floor_artifact={},
        floor_path="synthetic",
        floor_sha256="synthetic",
    )
    report = build_report_only(
        SimpleNamespace(), phase, resolved, (), {"truth_side_only": True}
    )["family_b"]
    alignment = report["alignment_displacement"]
    redrawn = report["redrawn_t_star_seed_comparison"]
    assert alignment["status"] == "not_computed"
    assert alignment["maximum_alignment_displacement"] is None
    assert redrawn["status"] == "unavailable"
    assert redrawn["pass"] is None


def test_report_only_lifts_roster_absent_birth_reconciliation():
    phase = M6RefitPhase(
        bundle=SimpleNamespace(mortality=None),
        mortality=None,
        population=SimpleNamespace(
            holdout_ids=frozenset({1, 2}),
            earnings_domain_ids=frozenset({1}),
        ),
        lineage={},
    )
    resolved = M6ResolvedContract(
        contract=M6GateContract(
            cells=(), gate_seeds=(), required_seed_passes=0
        ),
        floor_artifact={},
        floor_path="synthetic",
        floor_sha256="synthetic",
    )
    shock = {
        "machine_reason": "synthetic",
        **{
            module: {"truth": {}, "projection": {}}
            for module in ("mortality", "marital", "disability", "earnings")
        },
    }
    seed_run = M6SeedRun(
        seed=0,
        score=_FakeSeedScore(0),
        side_a_units={"household": 1, "person": 1},
        draw_reports=(
            {
                "shock_window": shock,
                "not_certified": {
                    "mortality_drift": {},
                    "widowhood": {},
                },
                "entrants": {
                    "synthetic_births": 3,
                    "immigrant_cohorts": 0,
                    "synthetic_persons": 3,
                    "scheduled_realized_openers": 2,
                    "roster_absent_births": {
                        2020: {
                            "dropped_parent_ids": [782173],
                            "dropped_count": 1,
                        }
                    },
                },
            },
        ),
    )

    entrants = build_report_only(
        SimpleNamespace(),
        phase,
        resolved,
        (seed_run,),
        {"truth_side_only": True},
    )["family_b"]["entrants"]

    assert entrants["roster_absent_births"] == {
        "2020": {
            "dropped_parent_ids": [782173],
            "dropped_count": 1,
        }
    }
    assert entrants["scheduled_realized_openers"] == {
        "total": 2,
        "by_year": None,
    }


@pytest.mark.parametrize(
    "registration",
    [
        "",
        "not-a-comment",
        "4962640241",
        "4967241464",
        "4971244215",
        "4973199058",
        "4976428384",
        "4981073550",
    ],
)
def test_registration_must_be_fresh_and_explicit(registration):
    # All six graded pre-scoring terminations (registrations 1-6) are stale;
    # registration 3 (4971244215) crashed pre-scoring, graded 4972045579,
    # registration 4 (4973199058) fired the pre-flight-1 designed abort, graded
    # 4973798460 (root-caused in forensics 4973982118), and registration 5
    # (4976428384) crashed in the seed-1 scoring projection on the marital
    # entry-row gap, graded 4979269487 (forensics 4979437110, closed by 3g),
    # and registration 6 (4981073550) crashed in fertility materialization,
    # graded 4984699959 (forensics 4984997277, closed by 3h).
    with pytest.raises(ValueError):
        validate_registration_id(registration)


def test_registration_accepts_comment_id_or_issue_url():
    assert validate_registration_id("9999999999") == "9999999999"
    url = (
        "https://github.com/PolicyEngine/populace-dynamics/issues/42"
        "#issuecomment-9999999999"
    )
    assert validate_registration_id(url) == url


def test_pre_input_guard_rejects_stale_registration_before_contract_read(
    monkeypatch, tmp_path
):
    touched = False

    def forbidden(_root):
        nonlocal touched
        touched = True
        raise AssertionError("contract read before registration validation")

    monkeypatch.setattr(runner_module, "resolve_m6_contract", forbidden)
    with pytest.raises(ValueError, match="earlier design registration"):
        guard_registered_m6_run(
            registration_id="4962640241",
            output=tmp_path / "fresh.json",
            root=ROOT,
        )
    assert touched is False
