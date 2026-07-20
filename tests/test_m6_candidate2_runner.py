"""Guard and orchestration tests for the hash-bound M6 candidate-2 runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics import claiming
from populace_dynamics.data import (
    disability as disability_data,
)
from populace_dynamics.data import (
    household_composition as household_data,
)
from populace_dynamics.data import (
    transitions,
)
from populace_dynamics.engine import assembly
from populace_dynamics.engine import candidates as engine_candidates
from populace_dynamics.engine import refit as refit_module
from populace_dynamics.engine.composition import (
    RECERTIFICATION_CHANNEL_SETS,
    Candidate9RecertificationFailure,
    RecertificationCell,
    RecertificationResult,
)
from populace_dynamics.engine.loop import (
    MaritalStepResult,
    PeriodModules,
    ProjectionEngine,
)
from populace_dynamics.engine.panel_builders import PanelBuilderInputs
from populace_dynamics.harness import m6_candidate2_runner as runner
from populace_dynamics.harness import m6_cells
from populace_dynamics.harness.m6_candidate2_runner import (
    M6Candidate2Bindings,
    M6Candidate2Preparation,
    M6Candidate2RunnerOperations,
    M6SourceIdentity,
)
from populace_dynamics.harness.m6_runner import (
    M6RefitPhase,
    M6ResolvedContract,
    M6SeedRun,
)
from populace_dynamics.harness.m6_scoring import M6CellRule, M6GateContract
from populace_dynamics.models.family_transitions import (
    registry as family_candidates,
)
from populace_dynamics.models.household_composition.components.child_attribution import (
    father_marital_by_year,
)
from populace_dynamics.ss.params import SSAParameters

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _FakeCellScore:
    cell: str
    score: float | None
    rate_a: float | None = 0.1
    per_draw_rates: tuple[float | None, ...] = (0.11, 0.12)
    rbar: float | None = 0.115
    undefined_draw_indices: tuple[int, ...] = ()
    regenerated: bool = True


@dataclass(frozen=True)
class _FakeSeedScore:
    seed: int
    cells: tuple[_FakeCellScore, ...]
    valid: bool = True
    passed: bool = True


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


def _fake_seed(
    seed: int,
    *,
    scores: dict[str, float | None] | None = None,
    valid: bool = True,
) -> _FakeSeedScore:
    values = {
        name: tolerance - 0.01
        for name, tolerance in runner.MUST_NOT_REGRESS_TOLERANCES.items()
    }
    values[runner.FIRST_MARRIAGE_GATE_CELL] = 0.1
    values.update(scores or {})
    cells = tuple(
        _FakeCellScore(name, value) for name, value in values.items()
    )
    return _FakeSeedScore(seed=seed, cells=cells, valid=valid, passed=valid)


def _fake_gate(
    *,
    passed: bool = True,
    valid: bool = True,
    seed_scores: dict[int, dict[str, float | None]] | None = None,
) -> _FakeGateScore:
    overrides = seed_scores or {}
    return _FakeGateScore(
        tuple(
            _fake_seed(seed, scores=overrides.get(seed), valid=valid)
            for seed in runner.MUST_NOT_REGRESS_SEEDS
        ),
        valid=valid,
        passed=passed,
    )


def _candidate_specs() -> dict:
    return {
        "family_transitions": {
            "candidate_id": "m6_candidate2_registry_v1",
            "contract_revision": "synthetic-family",
            "components": [
                {
                    "kind": "first_marriage",
                    "implementation_id": "synthetic-first-marriage",
                    "params": {
                        "selected_c": 0.001,
                        "selection_ledger_sha256": "f" * 64,
                        "final_fit_checksums": {"rows": "e" * 64},
                    },
                }
            ],
        },
        "engine_operations": {
            "candidate_id": "m6_candidate2_engine_v1",
            "contract_revision": "synthetic-engine",
            "operations": [],
        },
    }


def _conformance_evidence(*, full: bool = False) -> dict:
    carriers = {2006: 1, 2008: 1, 2010: 1}
    paired_rows = [
        {
            "pseudo_boundary": boundary,
            "draw_seed": seed,
            "postrepair_carriers_verified": carriers[boundary],
            "relabel_event_count": 0,
            "relabel_f6_weight": 0.0,
            **{
                invariant: True
                for invariant in runner.CONFORMANCE_PAIRED_INVARIANTS
            },
        }
        for boundary in runner.CONFORMANCE_EVALUATION_YEARS
        for seed in range(7200, 7240)
    ]
    if full:
        atomic_rows = []
        aggregates = []
        for boundary, years in runner.CONFORMANCE_EVALUATION_YEARS.items():
            for year in years:
                required = year if year % 2 else year + 1
                for origin in ("divorced", "widowed"):
                    for sex in ("female", "male"):
                        for band in ("18-29", "30-44", "45-64"):
                            key = {
                                "pseudo_boundary": boundary,
                                "evaluation_year": year,
                                "required_interview_year": required,
                                "entry_origin": origin,
                                "sex": sex,
                                "age_band": band,
                            }
                            atomic_rows.extend(
                                {
                                    **key,
                                    "draw_seed": seed,
                                    "from_transition": "first_marriage",
                                    "to_transition": "remarriage",
                                    "event_count": 0,
                                    "f6_weight": 0.0,
                                    "first_marriage_count_delta": 0,
                                    "first_marriage_f6_weight_delta": 0.0,
                                    "remarriage_count_delta": 0,
                                    "remarriage_f6_weight_delta": 0.0,
                                }
                                for seed in range(7200, 7240)
                            )
                            summaries = {
                                metric: {
                                    "max_per_draw": 0.0,
                                    "mean_per_draw": 0.0,
                                    "min_per_draw": 0.0,
                                    "sum_across_draws": 0.0,
                                }
                                for metric in runner.CONFORMANCE_AGGREGATE_METRICS
                            }
                            aggregates.append(
                                {**key, "n_draws": 40, **summaries}
                            )
    else:
        atomic_rows = [{"synthetic": True}]
        aggregates = [{"synthetic": True}]
    return {
        "schema": "m6.candidate2.entry_dissolved_conformance.v1",
        "status": "PASS",
        "authority": {
            "candidate_number": 2,
            "repair_merge": runner.LANDED_REPAIR_COMMIT,
        },
        "source_bindings": {
            name: "1" * 64 for name in runner.CONFORMANCE_SOURCE_BINDING_KEYS
        },
        "disposition": {
            "condition_4": "PASS",
            "condition_5": "PASS",
            "semantic_change_detected": False,
            "gate_or_floor_byte_change_authorized": False,
        },
        "information_boundary": {
            "candidate_output_contact": False,
            "floor_artifact_read": False,
            "gate_contract_read": False,
            "gate_scorer_called": False,
            "gate_tolerance_read": False,
            "post_2014_values_entered_computation": False,
            "runs_path_read_or_written": False,
        },
        "protocol": {
            "independent_of_candidate2_first_marriage_estimator": True,
            "complete_zero_rows_published": True,
        },
        "boundaries": [
            {
                "pseudo_boundary": boundary,
                "entry_dissolved_carriers": count,
            }
            for boundary, count in carriers.items()
        ],
        "paired_draw_invariants": {
            "all_pass": True,
            "draw_count": 120,
            "rows": paired_rows,
        },
        "relabel_ledger": {
            "atomic_row_count": len(atomic_rows),
            "atomic_rows_sha256": hashlib.sha256(
                runner._canonical_json_bytes(atomic_rows)
            ).hexdigest(),
            "atomic_rows": atomic_rows,
            "aggregates_by_boundary_year_origin_sex_age": aggregates,
        },
    }


def _matching_bindings(**changes) -> M6Candidate2Bindings:
    values = runner.PREREGISTERED_VALUES
    bindings = M6Candidate2Bindings(
        source=M6SourceIdentity("a" * 40),
        design_commit=values["design_commit"],
        design_blob_sha256s={
            path: "b" * 64 for path in runner.DESIGN_BLOB_PATHS
        },
        floor_run="synthetic-floor.json",
        floor_run_sha256=values["floor_run_sha256"],
        candidate_number=2,
        candidate_spec_ids={
            "family_transitions": "m6_candidate2_registry_v1",
            "engine_operations": "m6_candidate2_engine_v1",
        },
        candidate_spec_sha256s={
            "family_transitions": values["candidate_spec.family_transitions"],
            "engine_operations": values["candidate_spec.engine_operations"],
        },
        candidate_specs=_candidate_specs(),
        selection_evidence_sha256s={
            name: values[f"selection_evidence.{name}"]
            for name in runner.SELECTION_EVIDENCE_PATHS
        },
        first_marriage_selection_ledger={
            "schema": "synthetic",
            "selection": {"selected_c": 0.001, "candidates": []},
            "boundaries": {},
        },
        entry_dissolved_conformance_sha256="9" * 64,
        entry_dissolved_conformance=_conformance_evidence(),
        dependency_sha256="c" * 64,
        sorted_pip_freeze=("alpha==1", "zeta==2"),
        runtime_identity={
            "python": "3.13.12",
            "numpy": "2.4.2",
            "pandas": "2.3.3",
            "scipy": "1.17.0",
        },
        environment_sidecar_path=f"{runner.DEFAULT_OUTPUT}.env.json",
        environment_sidecar_sha256="d" * 64,
        contract_ref={
            "blob_sha": "e" * 40,
            "head_sha": "a" * 40,
            "path": "gates.yaml",
        },
    )
    return replace(bindings, **changes)


def _resolved() -> M6ResolvedContract:
    contract = M6GateContract(
        cells=(),
        gate_seeds=runner.MUST_NOT_REGRESS_SEEDS,
        required_seed_passes=runner.MUST_NOT_REGRESS_REQUIRED_SEEDS,
        n_draws=2,
        floor_run="synthetic-floor.json",
        floor_run_sha256=runner.PREREGISTERED_VALUES["floor_run_sha256"],
    )
    return M6ResolvedContract(
        contract=contract,
        floor_artifact={},
        floor_path="synthetic-floor.json",
        floor_sha256=runner.PREREGISTERED_VALUES["floor_run_sha256"],
    )


def _first_marriage_resolved() -> M6ResolvedContract:
    resolved = _resolved()
    rules = [
        M6CellRule(
            cell=runner.FIRST_MARRIAGE_GATE_CELL,
            family="marital",
            split_unit="household",
            metric="log_ratio",
            tolerance=0.356,
            k=3,
            rounding=3,
        )
    ]
    for cell, tolerance in runner.MUST_NOT_REGRESS_TOLERANCES.items():
        family = (
            "earnings"
            if cell.startswith("earn_")
            else (
                "disability"
                if cell.startswith(("incidence", "recovery"))
                else "marital"
            )
        )
        rules.append(
            M6CellRule(
                cell=cell,
                family=family,
                split_unit="household" if family == "marital" else "person",
                metric="log_ratio",
                tolerance=tolerance,
                k=3,
                rounding=3,
            )
        )
    return replace(
        resolved,
        contract=replace(resolved.contract, cells=tuple(rules)),
    )


def _sidecar() -> bytes:
    return runner._canonical_json_bytes(
        {
            "contract": {
                "blob_sha": "e" * 40,
                "head_sha": "a" * 40,
                "path": "gates.yaml",
            },
            "environment": {"python": "3.13.12"},
        }
    )


def _preparation(tmp_path: Path) -> M6Candidate2Preparation:
    destination = tmp_path / runner.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True)
    sidecar = _sidecar()
    bindings = replace(
        _matching_bindings(),
        environment_sidecar_sha256=hashlib.sha256(sidecar).hexdigest(),
    )
    return M6Candidate2Preparation(
        registration_id="9999999999",
        repository=tmp_path,
        destination=destination,
        bindings=bindings,
        resolved=_resolved(),
        sidecar_payload=sidecar,
    )


def _replace_surface(
    bindings: M6Candidate2Bindings, surface: str
) -> M6Candidate2Bindings:
    bad = "f" * 64
    if surface == "design_commit":
        return replace(bindings, design_commit="f" * 40)
    if surface == "floor_run_sha256":
        return replace(bindings, floor_run_sha256=bad)
    prefix, name = surface.split(".", 1)
    if prefix == "candidate_spec":
        values = dict(bindings.candidate_spec_sha256s)
        values[name] = bad
        return replace(bindings, candidate_spec_sha256s=values)
    values = dict(bindings.selection_evidence_sha256s)
    values[name] = bad
    return replace(bindings, selection_evidence_sha256s=values)


def test_dirty_tree_aborts_before_any_data_read(monkeypatch, tmp_path):
    events = []

    def git(_repository, *arguments):
        events.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("status", "--porcelain"):
            return "?? dirty-marker"
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_git_text", git)
    monkeypatch.setattr(
        runner,
        "collect_candidate2_bindings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repository data read after dirty status")
        ),
    )
    (tmp_path / runner.DEFAULT_OUTPUT.parent).mkdir()

    with pytest.raises(RuntimeError, match="dirty worktree"):
        runner.guard_registered_m6_candidate2_run(
            registration_id="9999999999",
            root=tmp_path,
        )

    assert events == [("rev-parse", "HEAD"), ("status", "--porcelain")]


def test_candidate1_output_path_is_hard_refused(monkeypatch, tmp_path):
    assert runner.CANDIDATE1_OUTPUT == Path("runs/gate_m6_candidate1_v1.json")
    assert runner.DEFAULT_OUTPUT == Path("runs/gate_m6_candidate2_v1.json")
    monkeypatch.setattr(
        runner,
        "capture_source_identity",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("Git reached for a forbidden output")
        ),
    )

    with pytest.raises(ValueError, match="hard-refuses"):
        runner.guard_registered_m6_candidate2_run(
            registration_id="9999999999",
            output=Path("runs/gate_m6_candidate1_v1.json"),
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "registration",
    [
        runner.CANDIDATE1_REGISTRATION_ID,
        (
            "https://github.com/PolicyEngine/populace-dynamics/issues/42"
            f"#issuecomment-{runner.CANDIDATE1_REGISTRATION_ID}"
        ),
    ],
)
def test_candidate1_registration_id_is_hard_refused(registration):
    with pytest.raises(ValueError, match="fresh registration 8"):
        runner.validate_candidate2_registration_id(registration)


def test_every_non_candidate2_output_path_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "capture_source_identity",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("Git reached for a forbidden output")
        ),
    )

    with pytest.raises(ValueError, match="writes exclusively"):
        runner.guard_registered_m6_candidate2_run(
            registration_id="9999999999",
            output=Path("elsewhere.json"),
            root=tmp_path,
        )


@pytest.mark.parametrize("occupied", ["primary", "sidecar"])
def test_existing_primary_or_sidecar_is_refused(
    occupied, monkeypatch, tmp_path
):
    destination = tmp_path / runner.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True)
    target = (
        destination
        if occupied == "primary"
        else Path(f"{destination}.env.json")
    )
    target.write_text("reserved", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "capture_source_identity",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("Git reached after occupied-path refusal")
        ),
    )

    with pytest.raises(FileExistsError, match="one-shot"):
        runner.guard_registered_m6_candidate2_run(
            registration_id="9999999999",
            root=tmp_path,
        )

    assert target.read_text(encoding="utf-8") == "reserved"


@pytest.mark.parametrize("surface", runner.PREREGISTERED_COMPARISON_KEYS)
def test_preregistered_binding_mismatch_aborts_before_data_read(
    surface, monkeypatch, tmp_path
):
    destination = tmp_path / runner.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True)
    mismatched = _replace_surface(_matching_bindings(), surface)
    monkeypatch.setattr(
        runner,
        "capture_source_identity",
        lambda _root: M6SourceIdentity("a" * 40),
    )
    monkeypatch.setattr(runner, "_assert_imported_source_tree", lambda _: None)
    monkeypatch.setattr(
        runner, "assert_candidate2_identity_is_frozen", lambda _: None
    )
    monkeypatch.setattr(
        runner,
        "collect_candidate2_bindings",
        lambda *args, **kwargs: (mismatched, _resolved(), _sidecar()),
    )

    with pytest.raises(RuntimeError, match=surface):
        runner.guard_registered_m6_candidate2_run(
            registration_id="9999999999",
            root=tmp_path,
        )

    assert not destination.exists()
    assert not Path(f"{destination}.env.json").exists()


def test_record_only_surfaces_are_not_in_comparison_scope():
    assert runner.PREREGISTERED_COMPARISON_KEYS == (
        "design_commit",
        "floor_run_sha256",
        "candidate_spec.family_transitions",
        "candidate_spec.engine_operations",
        "selection_evidence.qstar_q_grid",
        "selection_evidence.first_marriage_c",
        "selection_evidence.remarriage_no_op",
    )
    altered = replace(
        _matching_bindings(),
        source=M6SourceIdentity("9" * 40),
        design_blob_sha256s={"anything": "8" * 64},
        dependency_sha256="7" * 64,
        sorted_pip_freeze=("different==1",),
        runtime_identity={
            "python": "different",
            "numpy": "different",
            "pandas": "different",
            "scipy": "different",
        },
        environment_sidecar_sha256="6" * 64,
        entry_dissolved_conformance_sha256="5" * 64,
        entry_dissolved_conformance={"record_only": "different"},
    )
    runner.assert_preregistered_bindings(altered)


def test_candidate2_identity_comes_from_registry_and_is_not_candidate16():
    identity = runner.resolve_candidate2_identity()
    assert identity.number == next(
        number
        for number, spec in engine_candidates.REGISTRY.items()
        if spec is engine_candidates.CANDIDATE_2
    )
    assert identity.number == 2
    assert identity.engine_spec is engine_candidates.REGISTRY[identity.number]
    assert identity.family_spec is family_candidates.M6_CANDIDATE_2
    assert identity.family_spec is not family_candidates.CANDIDATE_16
    assert identity.family_spec_sha256 != family_candidates.CANDIDATE_16.sha256
    assert identity.first_marriage_params["selected_c"] == 0.001
    runner.assert_candidate2_identity_is_frozen(identity)


def test_candidate2_identity_rejects_candidate16(monkeypatch):
    monkeypatch.setattr(
        family_candidates,
        "M6_CANDIDATE_2",
        family_candidates.CANDIDATE_16,
    )
    with pytest.raises(RuntimeError, match="forbidden candidate 16"):
        runner.resolve_candidate2_identity()


def test_candidate2_identity_rejects_non_first_marriage_sibling_drift(
    monkeypatch,
):
    spec = family_candidates.M6_CANDIDATE_2
    components = tuple(
        (
            replace(
                component, params={**dict(component.params), "drift": True}
            )
            if component.kind == "divorce"
            else component
        )
        for component in spec.components
    )
    monkeypatch.setattr(
        family_candidates,
        "M6_CANDIDATE_2",
        replace(spec, components=components),
    )
    with pytest.raises(RuntimeError, match="non-first-marriage component"):
        runner.resolve_candidate2_identity()


def test_registry_values_are_semantically_bound_to_selection_ledgers():
    identity = runner.resolve_candidate2_identity()
    checksums = {"coefficient_sha256": "c" * 64}
    frozen = replace(
        identity,
        family_spec=SimpleNamespace(candidate_id="m6_candidate2_registry_v1"),
        first_marriage_params={
            "selected_c": 0.001,
            "selection_ledger_sha256": "placeholder",
            "final_fit_checksums": checksums,
        },
    )
    first_marriage = runner._canonical_json_bytes(
        {
            "selection": {"selected_c": 0.001},
            "final_fit": {"fit_audit": {"checksums": checksums}},
        }
    )
    frozen = replace(
        frozen,
        first_marriage_params={
            **dict(frozen.first_marriage_params),
            "selection_ledger_sha256": hashlib.sha256(
                first_marriage
            ).hexdigest(),
        },
    )
    qstar = runner._canonical_json_bytes({"selector": {"selected_q": 0.55}})
    runner._assert_selection_semantics(
        frozen,
        {
            "first_marriage_c": first_marriage,
            "qstar_q_grid": qstar,
        },
    )
    with pytest.raises(RuntimeError, match="final-fit checksums differ"):
        runner._assert_selection_semantics(
            replace(
                frozen,
                first_marriage_params={
                    **dict(frozen.first_marriage_params),
                    "final_fit_checksums": {"coefficient_sha256": "d" * 64},
                },
            ),
            {
                "first_marriage_c": first_marriage,
                "qstar_q_grid": qstar,
            },
        )
    with pytest.raises(RuntimeError, match="exact selection ledger"):
        runner._assert_selection_semantics(
            replace(
                frozen,
                first_marriage_params={
                    **dict(frozen.first_marriage_params),
                    "selection_ledger_sha256": "d" * 64,
                },
            ),
            {
                "first_marriage_c": first_marriage,
                "qstar_q_grid": qstar,
            },
        )
    with pytest.raises(RuntimeError, match="selected_c differs"):
        runner._assert_selection_semantics(
            replace(
                frozen,
                first_marriage_params={
                    **dict(frozen.first_marriage_params),
                    "selected_c": 0.003,
                },
            ),
            {
                "first_marriage_c": first_marriage,
                "qstar_q_grid": qstar,
            },
        )
    with pytest.raises(RuntimeError, match="engine q differs"):
        runner._assert_selection_semantics(
            frozen,
            {
                "first_marriage_c": first_marriage,
                "qstar_q_grid": runner._canonical_json_bytes(
                    {"selector": {"selected_q": 0.6}}
                ),
            },
        )


def test_prefreeze_identity_aborts_before_binding_collection(monkeypatch):
    called = False

    def collect(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("binding collection reached")

    monkeypatch.setattr(runner, "collect_candidate2_bindings", collect)
    monkeypatch.setattr(
        runner,
        "capture_source_identity",
        lambda _root: M6SourceIdentity("a" * 40),
    )
    monkeypatch.setattr(runner, "_assert_imported_source_tree", lambda _: None)
    monkeypatch.setattr(
        family_candidates,
        "M6_CANDIDATE_2",
        family_candidates.M6_CANDIDATE_2_PREFREEZE,
    )
    with pytest.raises(RuntimeError, match="selected_c=None"):
        runner.guard_registered_m6_candidate2_run(
            registration_id="9999999999",
            root=ROOT,
        )
    assert called is False


def test_live_preregistered_file_and_spec_hashes_are_exact():
    head = runner._git_text(ROOT, "rev-parse", "HEAD")
    identity = runner.resolve_candidate2_identity()
    assert (
        identity.family_spec_sha256
        == runner.PREREGISTERED_VALUES["candidate_spec.family_transitions"]
    )
    assert (
        identity.engine_spec_sha256
        == runner.PREREGISTERED_VALUES["candidate_spec.engine_operations"]
    )
    for name, path in runner.SELECTION_EVIDENCE_PATHS.items():
        assert (
            hashlib.sha256(runner._head_blob(ROOT, head, path)).hexdigest()
            == runner.PREREGISTERED_VALUES[f"selection_evidence.{name}"]
        )


def test_binding_collection_records_design_hashes_from_captured_commit(
    monkeypatch, tmp_path
):
    payloads = {
        path: f"HEAD blob for {path}".encode()
        for path in runner.DESIGN_BLOB_PATHS
    }
    calls = []

    def head_blob(_root, commit, path):
        calls.append((commit, path))
        return payloads.get(path, b"{}")

    monkeypatch.setattr(
        runner,
        "_head_blob",
        head_blob,
    )
    monkeypatch.setattr(
        runner,
        "_live_gate_document",
        lambda _root, _commit: {
            "gates": {"gate_m6": {"design_commit": "design"}}
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_pinned_contract",
        lambda *_args: _resolved(),
    )
    monkeypatch.setattr(
        runner,
        "_assert_selection_semantics",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        runner,
        "_validated_entry_dissolved_conformance",
        lambda _blob, **_kwargs: _conformance_evidence(),
    )
    monkeypatch.setattr(
        runner,
        "_dependency_snapshot",
        lambda: (("alpha==1",), "d" * 64),
    )
    monkeypatch.setattr(
        runner,
        "_sidecar_snapshot",
        lambda *_args: (
            _sidecar(),
            hashlib.sha256(_sidecar()).hexdigest(),
            {
                "python": "3",
                "numpy": "2",
                "pandas": "2",
                "scipy": "1",
            },
            {"blob_sha": "b", "head_sha": "c", "path": "gates.yaml"},
        ),
    )
    source = M6SourceIdentity("a" * 40)
    bindings, _, _ = runner.collect_candidate2_bindings(
        tmp_path,
        source=source,
        identity=runner.resolve_candidate2_identity(),
    )
    assert bindings.design_blob_sha256s == {
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in payloads.items()
    }
    assert all(
        (source.commit_sha, path) in calls for path in runner.DESIGN_BLOB_PATHS
    )
    assert (
        source.commit_sha,
        runner.CONFORMANCE_EVIDENCE_PATH,
    ) in calls


def test_live_entry_dissolved_conformance_ledger_is_bound_and_validated():
    head = runner._git_text(ROOT, "rev-parse", "HEAD")
    blob = runner._head_blob(ROOT, head, runner.CONFORMANCE_EVIDENCE_PATH)
    evidence = runner._validated_entry_dissolved_conformance(blob)
    ledger = evidence["relabel_ledger"]
    assert evidence["status"] == "PASS"
    assert evidence["authority"]["candidate_number"] == 2
    assert ledger["atomic_row_count"] == 5280
    assert len(ledger["aggregates_by_boundary_year_origin_sex_age"]) == 132
    assert (
        hashlib.sha256(
            runner._canonical_json_bytes(ledger["atomic_rows"])
        ).hexdigest()
        == ledger["atomic_rows_sha256"]
    )


def test_entry_dissolved_conformance_guard_rejects_internal_drift():
    def validate(document):
        return runner._validated_entry_dissolved_conformance(
            runner._canonical_json_bytes(document),
        )

    evidence = _conformance_evidence(full=True)
    validate(evidence)
    atomic = evidence["relabel_ledger"]["atomic_rows"][0]
    atomic["event_count"] = 1
    with pytest.raises(RuntimeError, match="atomic relabel ledger hash"):
        validate(evidence)
    atomic["event_count"] = 0
    evidence["relabel_ledger"]["atomic_rows_sha256"] = hashlib.sha256(
        runner._canonical_json_bytes(evidence["relabel_ledger"]["atomic_rows"])
    ).hexdigest()
    paired = evidence["paired_draw_invariants"]["rows"][0]
    paired["relabel_event_count"] = 1
    with pytest.raises(RuntimeError, match="atomic/paired relabel totals"):
        validate(evidence)
    paired["relabel_event_count"] = 0
    atomic.update(
        {
            "event_count": -1,
            "f6_weight": -2.0,
            "first_marriage_count_delta": 1,
            "first_marriage_f6_weight_delta": 2.0,
            "remarriage_count_delta": -1,
            "remarriage_f6_weight_delta": -2.0,
        }
    )
    evidence["relabel_ledger"]["atomic_rows_sha256"] = hashlib.sha256(
        runner._canonical_json_bytes(evidence["relabel_ledger"]["atomic_rows"])
    ).hexdigest()
    with pytest.raises(RuntimeError, match="atomic relabel transfer"):
        validate(evidence)
    atomic.update(
        {
            "event_count": 0,
            "f6_weight": 0.0,
            "first_marriage_count_delta": 0,
            "first_marriage_f6_weight_delta": 0.0,
            "remarriage_count_delta": 0,
            "remarriage_f6_weight_delta": 0.0,
        }
    )
    evidence["relabel_ledger"]["atomic_rows_sha256"] = hashlib.sha256(
        runner._canonical_json_bytes(evidence["relabel_ledger"]["atomic_rows"])
    ).hexdigest()
    evidence["paired_draw_invariants"]["all_pass"] = False
    with pytest.raises(RuntimeError, match="paired-draw invariants"):
        validate(evidence)
    evidence["paired_draw_invariants"]["all_pass"] = True
    invariant = runner.CONFORMANCE_PAIRED_INVARIANTS[0]
    evidence["paired_draw_invariants"]["rows"][0][invariant] = False
    with pytest.raises(RuntimeError, match="invariant row is not PASS"):
        validate(evidence)
    evidence["paired_draw_invariants"]["rows"][0][invariant] = True
    aggregate = evidence["relabel_ledger"][
        "aggregates_by_boundary_year_origin_sex_age"
    ].pop()
    with pytest.raises(RuntimeError, match="relabel ledger is incomplete"):
        validate(evidence)
    evidence["relabel_ledger"][
        "aggregates_by_boundary_year_origin_sex_age"
    ].append(aggregate)
    evidence["information_boundary"]["candidate_output_contact"] = True
    with pytest.raises(RuntimeError, match="information boundary"):
        validate(evidence)
    evidence["information_boundary"]["candidate_output_contact"] = False
    first = runner.CONFORMANCE_SOURCE_BINDING_KEYS[0]
    original_source = evidence["source_bindings"][first]
    evidence["source_bindings"][first] = "g" * 64
    with pytest.raises(RuntimeError, match="source bindings"):
        validate(evidence)
    evidence["source_bindings"][first] = original_source


def test_dependency_digest_uses_sorted_pip_freeze_with_lf_terminator():
    lines, payload = runner._canonicalize_pip_freeze(
        "zeta==2\nalpha==1\neditable @ file:///tmp/source\n"
    )
    assert lines == (
        "alpha==1",
        "editable @ file:///tmp/source",
        "zeta==2",
    )
    assert payload == (b"alpha==1\neditable @ file:///tmp/source\nzeta==2\n")
    assert runner._sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_transport_disclosure_uses_canonical_panel_birth_decade():
    observed = {}

    class Model:
        def transport_diagnostics(self, age, is_male, decade):
            observed["age"] = age.tolist()
            observed["is_male"] = is_male.tolist()
            observed["decade"] = decade.tolist()
            return SimpleNamespace(
                target_birth_decade=decade,
                mapped_birth_decade=decade.copy(),
                global_boundary_evaluated=np.zeros(len(age), dtype=bool),
                cohort_boundary_evaluated=np.zeros(len(age), dtype=bool),
            )

    frame = pd.DataFrame(
        {
            "person_id": [1],
            "year": [2015],
            "age": [25],
            "sex": ["female"],
            "weight": [2.5],
            "marital_state": ["never_married"],
            "window": ["gated"],
        }
    )
    fit_attrs = pd.DataFrame({"person_id": [1], "birth_year": [1979]})
    truth_attrs = pd.DataFrame({"person_id": [1], "birth_year": [1989]})
    inputs = runner.M6HarnessInputs(
        refit_inputs=SimpleNamespace(
            family_context=SimpleNamespace(
                panel=SimpleNamespace(attrs=fit_attrs)
            )
        ),
        panel_builder_inputs=SimpleNamespace(
            marital=SimpleNamespace(attrs=truth_attrs)
        ),
        truth=SimpleNamespace(marital_person_years=frame),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )
    phase = M6RefitPhase(
        bundle=SimpleNamespace(
            family=SimpleNamespace(
                fitted=SimpleNamespace(first_marriage=Model())
            )
        ),
        mortality=None,
        population=None,
        lineage={},
    )
    result = runner._first_marriage_transport_disclosure(
        inputs,
        phase,
        _first_marriage_resolved(),
        {"passed": True},
    )
    assert observed == {
        "age": [25.0],
        "is_male": [False],
        "decade": [1980],
    }
    cell = result["gated_cell_transport"]["first_marriage.18-29|female"]
    assert cell["in_support"] == {"rows": 1, "f6_weight": 2.5}
    assert result["completed_before_projection_or_score"] is True


def test_must_not_regress_uses_exact_thresholds_and_four_of_five():
    target = "earn_dlog_sd.older"
    gate = _fake_gate(
        seed_scores={
            0: {target: runner.MUST_NOT_REGRESS_TOLERANCES[target]},
            4: {target: runner.MUST_NOT_REGRESS_TOLERANCES[target] + 0.001},
        }
    )
    result = runner._must_not_regress_artifact(gate)
    assert result["tolerances"] == dict(runner.MUST_NOT_REGRESS_TOLERANCES)
    assert result["required_seed_passes"] == 4
    assert result["per_seed"][0]["cells"][target]["pass"] is True
    assert result["n_seeds_pass"] == 4
    assert result["pass"] is True


def test_must_not_regress_fails_when_only_three_seeds_clear():
    target = "earn_zero_rate.older"
    tolerance = runner.MUST_NOT_REGRESS_TOLERANCES[target]
    result = runner._must_not_regress_artifact(
        _fake_gate(
            seed_scores={
                0: {target: tolerance + 0.001},
                1: {target: tolerance + 0.001},
            }
        )
    )
    assert result["n_seeds_pass"] == 3
    assert result["pass"] is False


def test_gate_pass_regression_fail_is_never_an_accepted_pass():
    target = "earn_zero_rate.older"
    tolerance = runner.MUST_NOT_REGRESS_TOLERANCES[target]
    regression = runner._must_not_regress_artifact(
        _fake_gate(
            passed=True,
            seed_scores={
                0: {target: tolerance + 0.004},
                1: {target: tolerance + 0.004},
            },
        )
    )
    acceptance = runner._candidate2_acceptance(
        {"valid": True, "pass": True},
        regression,
    )
    assert acceptance["conjunction"] == {
        "valid": True,
        "pass": False,
        "status": "GATE_PASS_REGRESSION_FAIL",
        "requires_gate_contract_and_must_not_regress": True,
    }


@pytest.mark.parametrize(
    ("gate_valid", "gate_pass", "regression_valid", "expected"),
    [
        (True, True, True, "PASS"),
        (True, False, True, "FAIL"),
        (False, False, False, "INVALID"),
    ],
)
def test_candidate2_acceptance_statuses(
    gate_valid, gate_pass, regression_valid, expected
):
    regression = runner._must_not_regress_artifact(
        _fake_gate(valid=regression_valid)
    )
    acceptance = runner._candidate2_acceptance(
        {"valid": gate_valid, "pass": gate_pass},
        regression,
    )
    assert acceptance["conjunction"]["status"] == expected
    assert acceptance["conjunction"]["pass"] is (expected == "PASS")


def test_must_not_regress_rejects_missing_seed_or_cell():
    gate = _fake_gate()
    with pytest.raises(RuntimeError, match="seeds differ"):
        runner._must_not_regress_artifact(replace(gate, seeds=gate.seeds[:-1]))
    missing = replace(
        gate.seeds[0],
        cells=tuple(
            cell
            for cell in gate.seeds[0].cells
            if cell.cell != "divorce.18-44"
        ),
    )
    with pytest.raises(RuntimeError, match="missing cells"):
        runner._must_not_regress_artifact(
            replace(gate, seeds=(missing, *gate.seeds[1:]))
        )


def test_must_not_regress_validity_uses_only_its_five_target_cells():
    unrelated_invalid = runner._must_not_regress_artifact(
        _fake_gate(valid=False)
    )
    assert unrelated_invalid["valid"] is True
    first = _fake_gate().seeds[0]
    bad_cells = tuple(
        replace(cell, score=None) if cell.cell == "divorce.18-44" else cell
        for cell in first.cells
    )
    result = runner._must_not_regress_artifact(
        replace(
            _fake_gate(),
            seeds=(replace(first, cells=bad_cells), *_fake_gate().seeds[1:]),
        )
    )
    assert result["valid"] is False
    assert result["pass"] is False


def test_first_marriage_decomposition_fails_closed_on_invalid_paired_arm():
    runs = tuple(
        M6SeedRun(
            seed=seed,
            score=_fake_seed(seed),
            side_a_units={"household": 10},
        )
        for seed in runner.MUST_NOT_REGRESS_SEEDS
    )
    first = runs[0]
    cells = tuple(
        (
            replace(cell, regenerated=False)
            if cell.cell == runner.FIRST_MARRIAGE_GATE_CELL
            else cell
        )
        for cell in first.score.cells
    )
    incumbent = (
        replace(first, score=replace(first.score, cells=cells)),
        *runs[1:],
    )
    delta = runner._first_marriage_estimator_delta(
        runs,
        incumbent,
        _resolved().contract,
    )
    assert delta["status"] == "INVALID"
    assert delta["per_seed"][0]["postrepair_incumbent"]["valid"] is False
    acceptance = runner._candidate2_acceptance(
        {"valid": True, "pass": True},
        {"valid": True, "pass": True},
    )
    assert acceptance["conjunction"]["status"] == "PASS"
    verdict = runner._candidate2_run_verdict(
        acceptance["conjunction"],
        {"status": "INVALID"},
    )
    assert verdict["status"] == "INVALID"
    assert verdict["pass"] is False


def test_sidecar_is_canonical_and_sha_is_bound_in_main_artifact(tmp_path):
    preparation = _preparation(tmp_path)
    bindings = preparation.bindings
    phase = M6RefitPhase(
        bundle=None,
        mortality=None,
        population=None,
        lineage={
            "resolved_spec_sha256s": {
                "family_transitions": bindings.candidate_spec_sha256s[
                    "family_transitions"
                ],
                "engine_candidate": bindings.candidate_spec_sha256s[
                    "engine_operations"
                ],
            },
            "engine_candidate_id": bindings.candidate_spec_ids[
                "engine_operations"
            ],
        },
    )
    runs = tuple(
        M6SeedRun(
            seed=seed,
            score=_fake_seed(seed),
            side_a_units={"person": 2},
        )
        for seed in runner.MUST_NOT_REGRESS_SEEDS
    )
    artifact = runner.assemble_m6_candidate2_artifact(
        registration_id="9999999999",
        inputs=SimpleNamespace(provenance={}),
        phase=phase,
        resolved=_resolved(),
        seed_runs=runs,
        gate_score=_FakeGateScore(tuple(run.score for run in runs)),
        first_marriage_diagnostics={"passed": True},
        first_marriage_estimator_delta=(
            runner._first_marriage_estimator_delta(
                runs,
                runs,
                _resolved().contract,
            )
        ),
        preflight_1={"passed": True},
        preflight_2={"passed": True},
        report_only={
            "domain_earnings_floor": {"truth_side_only": True},
            "family_b": {"gating": False},
            "family_c": {"gating": False},
        },
        bindings=bindings,
    )
    runner.write_new_candidate2_artifact(
        preparation.destination,
        artifact,
        preparation.sidecar_payload,
    )
    sidecar_path = Path(f"{preparation.destination}.env.json")
    assert sidecar_path.read_bytes() == preparation.sidecar_payload
    assert (
        runner._canonical_json_bytes(json.loads(sidecar_path.read_bytes()))
        == sidecar_path.read_bytes()
    )
    assert artifact["integrity"]["environment_sidecar"]["sha256"] == (
        hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    )
    assert artifact["runtime_identity"] == dict(bindings.runtime_identity)
    assert artifact["dependency_snapshot"]["sha256"] == (
        bindings.dependency_sha256
    )
    assert artifact["dependency_snapshot"]["sorted_pip_freeze"] == list(
        bindings.sorted_pip_freeze
    )
    integrity = artifact["integrity"]
    assert integrity["source"] == {
        "git_commit": bindings.source.commit_sha,
        "worktree_clean": True,
        "tree_pin": "git commit",
    }
    assert integrity["design"] == {
        "gate_m6_design_commit": bindings.design_commit,
        "head_blob_sha256s": dict(bindings.design_blob_sha256s),
    }
    assert integrity["floor"] == {
        "path": bindings.floor_run,
        "sha256": bindings.floor_run_sha256,
    }
    for name, digest in bindings.candidate_spec_sha256s.items():
        assert integrity["candidate_specs"][name] == {
            "candidate_id": bindings.candidate_spec_ids[name],
            "canonical_json_sha256": digest,
            "canonical_spec": bindings.candidate_specs[name],
        }
    for name, digest in bindings.selection_evidence_sha256s.items():
        assert integrity["selection_evidence"][name] == {
            "path": runner.SELECTION_EVIDENCE_PATHS[name],
            "sha256": digest,
        }
    assert set(integrity["preregistered_comparisons"]) == set(
        runner.PREREGISTERED_COMPARISON_KEYS
    )
    assert integrity["entry_dissolved_conformance"] == {
        "path": runner.CONFORMANCE_EVIDENCE_PATH,
        "sha256": bindings.entry_dissolved_conformance_sha256,
        "record_and_publish": True,
        "covered_by_source_commit": True,
    }
    first_marriage = artifact["preflights"]["first_marriage_fit_and_transport"]
    assert first_marriage["passed"] is True
    assert first_marriage["selection_ledger"] == (
        bindings.first_marriage_selection_ledger
    )
    assert artifact["verdict"]["status"] == "PASS"
    assert set(artifact["candidate2_acceptance"]) == {
        "gate_contract_result",
        "must_not_regress_result",
        "conjunction",
    }
    decomposition = artifact["first_marriage_effect_decomposition"]
    assert decomposition["status"] == "COMPLETE"
    relabel = decomposition["landed_conformance_relabel"]["relabel_ledger"]
    source_relabel = bindings.entry_dissolved_conformance["relabel_ledger"]
    assert relabel["atomic_row_count"] == source_relabel["atomic_row_count"]
    assert (
        relabel["atomic_rows_sha256"] == source_relabel["atomic_rows_sha256"]
    )
    assert relabel["aggregates_by_boundary_year_origin_sex_age"] == (
        source_relabel["aggregates_by_boundary_year_origin_sex_age"]
    )
    assert relabel["full_atomic_rows_embedded"] is False
    assert relabel["full_atomic_rows_bound_by_source_sha256"] is True
    assert (
        decomposition["support_aware_estimator_delta"]["status"] == "COMPLETE"
    )


def test_pair_writer_removes_primary_when_sidecar_write_fails(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    artifact = {
        "integrity": {
            "environment_sidecar": {
                "sha256": hashlib.sha256(
                    preparation.sidecar_payload
                ).hexdigest()
            }
        }
    }
    original = runner._write_exclusive
    calls = []

    def fail_second(path, payload):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("synthetic sidecar failure")
        original(path, payload)

    monkeypatch.setattr(runner, "_write_exclusive", fail_second)
    with pytest.raises(OSError, match="sidecar failure"):
        runner.write_new_candidate2_artifact(
            preparation.destination,
            artifact,
            preparation.sidecar_payload,
        )
    assert not preparation.destination.exists()
    assert not Path(f"{preparation.destination}.env.json").exists()


@pytest.mark.parametrize("race_target", ["primary", "sidecar"])
def test_pair_writer_refuses_atomic_create_races(
    race_target, monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    sidecar = Path(f"{preparation.destination}.env.json")
    artifact = {
        "integrity": {
            "environment_sidecar": {
                "sha256": hashlib.sha256(
                    preparation.sidecar_payload
                ).hexdigest()
            }
        }
    }
    original = runner._write_exclusive
    target = preparation.destination if race_target == "primary" else sidecar

    def race(path, payload):
        if path == target:
            path.write_bytes(b"competing writer")
        original(path, payload)

    monkeypatch.setattr(runner, "_write_exclusive", race)
    with pytest.raises(FileExistsError, match="one-shot"):
        runner.write_new_candidate2_artifact(
            preparation.destination,
            artifact,
            preparation.sidecar_payload,
        )
    assert target.read_bytes() == b"competing writer"
    if race_target == "sidecar":
        assert not preparation.destination.exists()


@pytest.mark.parametrize("failure", ["noncanonical", "sha_mismatch"])
def test_pair_writer_refuses_unbound_sidecar_before_write(failure, tmp_path):
    preparation = _preparation(tmp_path)
    sidecar = preparation.sidecar_payload
    expected = hashlib.sha256(sidecar).hexdigest()
    if failure == "noncanonical":
        sidecar += b"\n"
    else:
        expected = "0" * 64
    artifact = {"integrity": {"environment_sidecar": {"sha256": expected}}}
    match = "not canonical" if failure == "noncanonical" else "exact"
    with pytest.raises(ValueError, match=match):
        runner.write_new_candidate2_artifact(
            preparation.destination,
            artifact,
            sidecar,
        )
    assert not preparation.destination.exists()
    assert not Path(f"{preparation.destination}.env.json").exists()


@pytest.mark.parametrize("occupied", ["primary", "sidecar"])
def test_pair_writer_directly_refuses_existing_targets(occupied, tmp_path):
    preparation = _preparation(tmp_path)
    target = (
        preparation.destination
        if occupied == "primary"
        else Path(f"{preparation.destination}.env.json")
    )
    target.write_bytes(b"reserved")
    artifact = {
        "integrity": {
            "environment_sidecar": {
                "sha256": hashlib.sha256(
                    preparation.sidecar_payload
                ).hexdigest()
            }
        }
    }
    with pytest.raises(FileExistsError, match="one-shot"):
        runner.write_new_candidate2_artifact(
            preparation.destination,
            artifact,
            preparation.sidecar_payload,
        )
    assert target.read_bytes() == b"reserved"


def test_pair_writer_hard_refuses_candidate1_path(tmp_path):
    sidecar = _sidecar()
    artifact = {
        "integrity": {
            "environment_sidecar": {
                "sha256": hashlib.sha256(sidecar).hexdigest()
            }
        }
    }
    path = tmp_path / "runs/gate_m6_candidate1_v1.json"
    path.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="hard-refuses"):
        runner.write_new_candidate2_artifact(path, artifact, sidecar)
    assert not path.exists()


def test_default_refit_binds_both_candidate2_specs(monkeypatch):
    sentinel = object()
    observed = {}

    def refit(inputs, **kwargs):
        observed["inputs"] = inputs
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(runner, "refit_m6_components", refit)
    monkeypatch.setattr(
        runner, "assert_candidate2_identity_is_frozen", lambda _: None
    )
    inputs = object()
    assert runner._fit_candidate2(inputs) is sentinel
    identity = runner.resolve_candidate2_identity()
    assert observed == {
        "inputs": inputs,
        "boundary_year": runner.BOUNDARY_YEAR,
        "family_candidate_spec": identity.family_spec,
        "earnings_candidate_spec": identity.engine_spec,
    }


def test_postrepair_incumbent_comparator_swaps_only_first_marriage(
    monkeypatch,
):
    @dataclass(frozen=True)
    class Fitted:
        first_marriage: object
        divorce: object
        widowhood: object
        remarriage: object
        fertility: object
        initial_states: object
        spousal_age_gaps: object
        implementation_ids: dict[str, str]

    carried = {
        name: object()
        for name in (
            "divorce",
            "widowhood",
            "remarriage",
            "fertility",
            "initial_states",
            "spousal_age_gaps",
        )
    }
    candidate_fitted = Fitted(
        first_marriage=object(),
        **carried,
        implementation_ids={"first_marriage": "candidate2"},
    )
    incumbent_first_marriage = object()
    incumbent_fitted = replace(
        candidate_fitted,
        first_marriage=incumbent_first_marriage,
        implementation_ids={"first_marriage": "incumbent"},
    )
    candidate_family = refit_module.RegistryRefit(
        candidate_id="m6_candidate2_registry_v1",
        spec_sha256="a" * 64,
        fitted=candidate_fitted,
        provenance=object(),
    )
    incumbent_family = refit_module.RegistryRefit(
        candidate_id=family_candidates.CANDIDATE_16.candidate_id,
        spec_sha256=family_candidates.CANDIDATE_16.sha256,
        fitted=incumbent_fitted,
        provenance=object(),
    )
    other = {
        name: object()
        for name in (
            "household",
            "earnings",
            "disability",
            "claiming_pmfs",
            "mortality",
        )
    }
    axis = SimpleNamespace(
        earn={1: 2.0},
        cuts=(1.0, 2.0),
        decile_edges=np.asarray([1.0, 2.0]),
        supply_by_decile=(np.asarray([1.0]),),
        sex={1: "female"},
        birth_year={1: 1980},
        person_weight={1: 3.0},
    )
    candidate_modifier_model = object()
    candidate_modifier = refit_module.ModifierRefit(
        axis=axis,
        modifier=candidate_modifier_model,
        ssa_vintage=object(),
        provenance=object(),
    )
    candidate_bundle = refit_module.M6RefitBundle(
        boundary_year=runner.BOUNDARY_YEAR,
        family=candidate_family,
        modifier=candidate_modifier,
        **other,
    )
    monkeypatch.setattr(
        runner,
        "refit_family_transitions",
        lambda *_args, **_kwargs: incumbent_family,
    )
    comparator_modifier = object()
    fresh_modifier = replace(
        candidate_modifier,
        axis=SimpleNamespace(
            earn={1: 2.0},
            cuts=(1.0, 2.0),
            decile_edges=np.asarray([1.0, 2.0]),
            supply_by_decile=(np.asarray([1.0]),),
            sex={1: "female"},
            birth_year={1: 1980},
            person_weight={1: 3.0},
        ),
        modifier=comparator_modifier,
    )
    monkeypatch.setattr(
        runner,
        "refit_first_marriage_modifier",
        lambda *_args, **_kwargs: fresh_modifier,
    )
    fit_inputs = SimpleNamespace(
        family_context=object(),
        modifier_marital_panel=object(),
        earnings_panel=object(),
        modifier_marriage_records=object(),
        modifier_person_weight=object(),
        ssa_params=object(),
        ssa_params_vintage=2014,
        modifier_train_ids=set(),
        modifier_interview_years=object(),
    )
    comparator = runner._fit_postrepair_incumbent_first_marriage(
        fit_inputs,
        candidate_bundle,
    )
    assert comparator.family.fitted.first_marriage is incumbent_first_marriage
    for name, value in carried.items():
        assert getattr(comparator.family.fitted, name) is value
    for name, value in other.items():
        assert getattr(comparator, name) is value
    assert comparator.modifier.axis is axis
    assert comparator.modifier.modifier is comparator_modifier
    assert comparator.modifier.ssa_vintage is candidate_modifier.ssa_vintage
    assert comparator.modifier.provenance is candidate_modifier.provenance
    assert comparator.family.candidate_id == (
        family_candidates.CANDIDATE_16.candidate_id
    )


def test_family_refit_records_an_explicit_sibling_spec(monkeypatch):
    calls = []
    context = object()

    class Registry:
        def fit(self, spec, observed):
            calls.append((spec, observed))
            return "fitted"

    monkeypatch.setattr(
        refit_module,
        "_truncate_family_context",
        lambda observed, _boundary: observed,
    )
    monkeypatch.setattr(
        refit_module,
        "_registry_provenance",
        lambda observed, boundary: {
            "observed": observed,
            "boundary": boundary,
        },
    )
    spec = SimpleNamespace(
        candidate_id="m6_candidate2_registry_v1",
        sha256="f" * 64,
    )
    result = refit_module.refit_family_transitions(
        context,
        candidate_spec=spec,
        registry=Registry(),
    )
    assert calls == [(spec, context)]
    assert result.candidate_id == spec.candidate_id
    assert result.spec_sha256 == spec.sha256


class _SyntheticProbabilityGate:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, values):
        return np.tile(np.asarray([[0.25, 0.75]]), (len(values), 1))


class _SyntheticQRFOutcome:
    columns = ("earnings", "age_tp2")
    gate = _SyntheticProbabilityGate()


class _SyntheticFittedQRF:
    def __init__(self):
        self._target_models = {
            "earnings_tp2": _SyntheticQRFOutcome(),
        }


class _SyntheticQRFFactory:
    def __call__(self, *, seed):
        del seed

        class Model:
            def fit(self, frame, **kwargs):
                del frame, kwargs
                return _SyntheticFittedQRF()

        return Model()


def _synthetic_ssa_params() -> SSAParameters:
    years = list(range(2000, 2017))
    return SSAParameters(
        nawi={year: 100.0 + year for year in years},
        wage_base={1900: 1_000_000.0},
        pia_factors=(0.9, 0.32, 0.15),
        fra_months_by_birth_year=[(1900, 804)],
        early_monthly_rates=(0.0, 0.0),
        early_first_bracket_months=36,
        pe_us_revision="synthetic-2014-vintage",
    )


def _synthetic_claim_reference() -> claiming.ClaimAgeReference:
    categories = {
        "age62": 40.0,
        "age63": 10.0,
        "age64": 10.0,
        "age65": 10.0,
        "age66": 10.0,
        "disability_conversion": 10.0,
        "age67_69": 6.0,
        "age70plus": 4.0,
    }
    record = {
        "number_thousands": 1,
        "average_age": 64.0,
        "raw": {},
        "categories": categories,
        "fra_at": {"share": 0.0, "at_age": 66},
    }
    return claiming.ClaimAgeReference(
        schema_version="ssa_claim_ages.v1",
        table="synthetic",
        supplement_year=2014,
        raw_columns=(),
        collapsed_categories=tuple(categories),
        provenance={},
        validation={},
        fra_schedule={},
        _data={
            "female": {"2014": record},
            "male": {"2014": record},
        },
    )


def _synthetic_base_marital_panel() -> transitions.MaritalPanel:
    person_years = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "year": [2012, 2014, 2015, 2013, 2015],
            "required_interview_year": [2013, 2015, 2015, 2013, 2015],
            "age": [30, 32, 33, 40, 42],
            "sex": ["female", "female", "female", "male", "male"],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0],
            "marital_state": [
                "never_married",
                "never_married",
                "married",
                "married",
                "married",
            ],
            "marriage_duration": pd.array(
                [pd.NA, pd.NA, 0, 5, 7], dtype="Int64"
            ),
            "years_since_dissolution": pd.array([pd.NA] * 5, dtype="Int64"),
        }
    )
    events = pd.DataFrame(
        {
            "person_id": [1, 2],
            "year": [2014, 2013],
            "required_interview_year": [2015, 2013],
            "age": [32, 40],
            "sex": ["female", "male"],
            "weight": [1.0, 2.0],
            "transition": ["first_marriage", "divorce"],
            "marriage_duration": pd.array([pd.NA, 5], dtype="Int64"),
            "years_since_dissolution": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "origin": pd.array([pd.NA, pd.NA], dtype="string"),
        }
    )
    attrs = pd.DataFrame(
        {
            "person_id": [1, 2],
            "birth_year": [1982.0, 1973.0],
            "censor_year": [2023.0, 2023.0],
            "weight": [1.0, 2.0],
            "n_marriages": [2.0, 1.0],
        }
    )
    return transitions.MaritalPanel(person_years, events, attrs)


def _synthetic_marriage_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "sex": ["female", "female", "male"],
            "birth_year": pd.array([1982, 1982, 1973], dtype="Int64"),
            "start_year": pd.array([2012, 2018, 2010], dtype="Int64"),
            "required_interview_year": pd.array(
                [2013, 2019, 2011], dtype="Int64"
            ),
            "end_year": pd.array([2018, pd.NA, 2016], dtype="Int64"),
            "separation_year": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "most_recent_report_year": pd.array(
                [2023, 2023, 2023], dtype="Int64"
            ),
            "n_marriages": pd.array([2, 2, 1], dtype="Int64"),
            "is_marriage": [True, True, True],
            "spouse_person_id": pd.array([2, 2, 1], dtype="Int64"),
        }
    )


def _synthetic_family_context():
    base_panel = _synthetic_base_marital_panel()
    attrs = base_panel.attrs.copy()
    attrs["sex"] = ["female", "male"]
    attrs["most_recent_report_year"] = 2014
    attrs["death_year"] = pd.array([pd.NA, pd.NA], dtype="Int64")
    attrs["start_exposure_year"] = attrs["birth_year"] + 15
    patterns = (
        ("female", 1970, (18, 24, 25)),
        ("female", 1980, (18, 24, 25)),
        ("female", 1990, (18, 23)),
        ("male", 1970, (18, 24, 25)),
        ("male", 1980, (18, 24, 25)),
    )
    rows = []
    event_years = set()
    person_id = 101
    for sex, cohort, ages in patterns:
        for age in ages:
            for is_event in (False, True):
                year = cohort + age
                rows.append(
                    {
                        "person_id": person_id,
                        "year": year,
                        "age": age,
                        "sex": sex,
                        "weight": 1.0,
                        "birth_decade": cohort,
                    }
                )
                if is_event:
                    event_years.add((person_id, year))
                person_id += 1
    training = pd.DataFrame(rows)
    person_years = training.copy()
    person_years["required_interview_year"] = person_years["year"]
    person_years["marital_state"] = "never_married"
    person_years["marriage_duration"] = pd.array(
        [pd.NA] * len(person_years), dtype="Int64"
    )
    person_years["years_since_dissolution"] = pd.array(
        [pd.NA] * len(person_years), dtype="Int64"
    )
    event_mask = [
        (int(pid), int(year)) in event_years
        for pid, year in zip(
            person_years["person_id"],
            person_years["year"],
            strict=True,
        )
    ]
    events = person_years.loc[
        event_mask,
        [
            "person_id",
            "year",
            "required_interview_year",
            "age",
            "sex",
            "weight",
        ],
    ].copy()
    events["transition"] = "first_marriage"
    for column in ("marriage_duration", "years_since_dissolution"):
        events[column] = pd.array([pd.NA] * len(events), dtype="Int64")
    events["origin"] = pd.array([pd.NA] * len(events), dtype="string")
    training_attrs = (
        training.sort_values(["person_id", "year"])
        .groupby("person_id", as_index=False)
        .first()[["person_id", "birth_decade", "sex", "weight"]]
        .rename(columns={"birth_decade": "birth_year"})
    )
    training_attrs["censor_year"] = 2014
    training_attrs["n_marriages"] = 1
    training_attrs["most_recent_report_year"] = 2014
    training_attrs["death_year"] = pd.array(
        [pd.NA] * len(training_attrs), dtype="Int64"
    )
    training_attrs["start_exposure_year"] = training_attrs["birth_year"] + 15
    panel = transitions.MaritalPanel(
        person_years=pd.concat(
            [base_panel.person_years, person_years], ignore_index=True
        ),
        events=pd.concat([base_panel.events, events], ignore_index=True),
        attrs=pd.concat([attrs, training_attrs], ignore_index=True),
    )
    training_ids = training_attrs["person_id"].astype(int).tolist()
    demographic = pd.concat(
        [
            pd.DataFrame(
                {
                    "person_id": [1, 1, 2, 2],
                    "period": [2013, 2015, 2013, 2015],
                }
            ),
            pd.DataFrame(
                {
                    "person_id": training_ids,
                    "period": [2013] * len(training_ids),
                }
            ),
        ],
        ignore_index=True,
    )
    birth_records = pd.DataFrame(
        {
            "parent_person_id": [1, 1],
            "birth_year": pd.array([2010, 2017], dtype="Int64"),
            "required_interview_year": pd.array([2011, 2017], dtype="Int64"),
            "is_event": [True, True],
            "record_type": ["birth", "birth"],
            "birth_order": pd.array([1, 2], dtype="Int64"),
            "parent_sex": ["female", "female"],
            "child_person_id": pd.array([10, 11], dtype="Int64"),
        }
    )
    return family_candidates.FitContext(
        panel=panel,
        demographic_panel=demographic,
        marriage_records=_synthetic_marriage_records(),
        birth_records=birth_records,
        marriage_order_map=pd.DataFrame(
            {
                "person_id": [1, 1, 2],
                "start_year": [2012, 2018, 2010],
                "order": [1, 2, 1],
            }
        ),
        train_ids=frozenset({1, 2, *training_ids}),
    )


def _refit_registered_family_core(spec):
    return refit_module.refit_family_transitions(
        _synthetic_family_context(),
        candidate_spec=spec,
    ).fitted


def test_fit_digest_accepts_populated_candidate2_audit_core():
    core = _refit_registered_family_core(family_candidates.M6_CANDIDATE_2)
    audit = core.first_marriage.fit_audit
    assert audit is not None
    assert audit.checksums
    assert audit.support

    digest = assembly._fit_digest(core)
    assert len(digest) == 64
    int(digest, 16)


def test_every_live_registered_family_core_is_digestable():
    specs = (
        family_candidates.CANDIDATE_16,
        family_candidates.M6_CANDIDATE_2,
    )

    def fully_registered(spec):
        try:
            for component in spec.components:
                family_candidates.REGISTRY.definition(component)
        except (KeyError, ValueError):
            return False
        return True

    discovered = {
        value.candidate_id
        for value in vars(family_candidates).values()
        if isinstance(value, family_candidates.CandidateSpec)
        and fully_registered(value)
    }
    assert discovered == {spec.candidate_id for spec in specs}

    for spec in specs:
        digest = assembly._fit_digest(_refit_registered_family_core(spec))
        assert len(digest) == 64
        int(digest, 16)


def _synthetic_household_context():
    person_waves = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "year": [2011, 2013, 2015, 2013, 2015],
            "age": [29, 31, 33, 40, 42],
            "band": ["25-34", "25-34", "25-34", "35-44", "35-44"],
            "sex": ["female", "female", "female", "male", "male"],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0],
            "hh_size": [2, 2, 3, 1, 2],
            "coresident_spouse": [False, True, True, False, True],
            "coresident_parent": [True, False, False, True, True],
            "coresident_child": [False, False, True, False, False],
            "coresident_grandchild": [False] * 5,
            "multigen": [False, False, True, False, False],
        }
    )
    person_waves.loc[
        (person_waves["person_id"] == 2) & (person_waves["year"] == 2015),
        "year",
    ] = 2014
    person_waves = household_data._add_transitions(person_waves)
    household = household_data.HouseholdCompositionPanel(
        person_waves=person_waves,
        attrs=pd.DataFrame({"person_id": [1, 2]}),
    )
    marital = _synthetic_base_marital_panel()
    marital_years = marital.person_years.copy()
    row = (marital_years["person_id"] == 1) & (marital_years["year"] == 2014)
    marital_years.loc[row, ["year", "required_interview_year", "age"]] = [
        2013,
        2013,
        31,
    ]
    prebirth = (marital_years["person_id"] == 1) & (
        marital_years["year"] == 2015
    )
    marital_years.loc[
        prebirth,
        ["year", "required_interview_year", "age", "marital_state"],
    ] = [2008, 2013, 26, "never_married"]
    marital_years.loc[
        prebirth, ["marriage_duration", "years_since_dissolution"]
    ] = pd.NA
    marital_events = marital.events.copy()
    row = (marital_events["person_id"] == 1) & (
        marital_events["transition"] == "first_marriage"
    )
    marital_events.loc[row, ["year", "required_interview_year", "age"]] = [
        2012,
        2013,
        30,
    ]
    marital_attrs = marital.attrs.copy()
    marital_attrs["sex"] = ["female", "male"]
    marital_attrs["most_recent_report_year"] = 2014
    marital_attrs["death_year"] = pd.array([pd.NA, pd.NA], dtype="Int64")
    marital_attrs["start_exposure_year"] = marital_attrs["birth_year"] + 15
    marital = transitions.MaritalPanel(
        marital_years, marital_events, marital_attrs
    )
    birth_records = pd.DataFrame(
        {
            "is_event": [True],
            "parent_person_id": pd.Series([1], dtype="int64"),
            "birth_year": pd.Series([2010], dtype="Int64"),
            "required_interview_year": pd.Series([2011], dtype="Int64"),
            "birth_order": pd.Series([1], dtype="Int64"),
            "record_type": pd.Series(["birth"], dtype="string"),
            "parent_sex": pd.Series(["female"], dtype="string"),
            "child_person_id": pd.Series([10], dtype="Int64"),
        }
    )
    relationship_map = pd.DataFrame(
        {
            "interview_year": [2013, 2015],
            "ego_rel_to_alter": [10, 10],
            "ego_person_id": [1, 1],
        }
    )
    empty_links = pd.DataFrame(
        {
            "parent_person_id": pd.Series(dtype="int64"),
            "child_person_id": pd.Series(dtype="int64"),
            "birth_year": pd.Series(dtype="int64"),
        }
    )
    empty_pairs = pd.DataFrame(
        {
            "parent_person_id": pd.Series(dtype="int64"),
            "child_person_id": pd.Series(dtype="int64"),
            "year": pd.Series(dtype="int64"),
        }
    )
    child_exposure = pd.DataFrame(
        {
            "father_id": pd.Series(dtype="int64"),
            "child_person_id": pd.Series(dtype="int64"),
            "year": pd.Series(dtype="int64"),
            "child_band": pd.Series(dtype="string"),
            "weight": pd.Series(dtype="float64"),
            "with_father": pd.Series(dtype=bool),
            "marital": pd.Series(dtype=bool),
        }
    )

    def person_year(name):
        return pd.DataFrame(
            {
                "person_id": [1, 2],
                "year": [2013, 2013],
                name: [1, 1],
            }
        )

    from populace_dynamics.models.household_composition import registry

    return registry.FitContext(
        hh=household,
        mpanel=marital,
        demographic_panel=pd.DataFrame(
            {
                "person_id": [1, 1, 2, 2],
                "period": [2013, 2015, 2013, 2015],
            }
        ),
        marriage_records=_synthetic_marriage_records(),
        birth_records=birth_records,
        marriage_order_map=pd.DataFrame(
            {
                "person_id": [1, 1, 2],
                "start_year": [2012, 2018, 2010],
                "order": [1, 2, 1],
            }
        ),
        relationship_map=relationship_map,
        train_ids=frozenset({1, 2}),
        father_links_child=empty_links,
        parent_pairs=empty_pairs,
        marital_by_year=father_marital_by_year(marital),
        family_unit_sizes=person_year("family_unit_size"),
        legal_spouse_flag=person_year("legal_spouse_obs"),
        child_record_exposure=child_exposure,
        parent_counts=person_year("n_parent_links"),
    )


def _synthetic_fit_earnings() -> pd.DataFrame:
    rows = []
    periods = tuple(range(2002, 2016, 2))
    for bin_index, age in enumerate(range(27, 67, 5)):
        person_id = bin_index + 1
        for wave_index, period in enumerate(periods):
            earnings = float(10_000 + 1_000 * bin_index + 125 * wave_index)
            if person_id == 1 and period == 2014:
                earnings = 0.0
            rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "earnings": earnings,
                    "age": age,
                    "weight": float(person_id),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_refit_inputs():
    family = _synthetic_family_context()
    disability_years = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "period": [2011, 2013, 2011, 2013],
            "required_interview_year": [2011, 2013, 2011, 2013],
            "age": [40, 42, 50, 52],
            "sex": ["female", "female", "male", "male"],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "status_code": [1, 5, 5, 1],
            "disabled": [False, True, True, False],
            "retired": [False, False, False, False],
        }
    )
    disability_panel = disability_data.DisabilityPanel(
        disability_years,
        disability_data.build_transition_pairs(disability_years),
    )
    return refit_module.M6RefitInputs(
        family_context=family,
        household_context=_synthetic_household_context(),
        earnings_panel=_synthetic_fit_earnings(),
        earnings_seed=17,
        modifier_marital_panel=family.panel,
        modifier_interview_years=(
            family.panel.person_years["required_interview_year"]
        ),
        modifier_marriage_records=family.marriage_records,
        modifier_person_weight=(
            family.panel.attrs.set_index("person_id")["weight"]
        ),
        ssa_params=_synthetic_ssa_params(),
        ssa_params_vintage=2014,
        modifier_train_ids=set(family.train_ids),
        disability_panel=disability_panel,
        disability_train_ids={1, 2},
        claiming_reference=_synthetic_claim_reference(),
        mortality_exposure=pd.DataFrame(
            {
                "event_year": [2013, 2013],
                "required_interview_year": [2013, 2013],
                "age_band": ["0+", "0+"],
                "sex": ["female", "male"],
                "start_weight": [1.0, 1.0],
                "exposure": [1.0, 1.0],
                "death": [0.01, 0.01],
            }
        ),
        mortality_external_rates=pd.DataFrame(
            {
                "lower_age": [0, 0],
                "upper_age": [120, 120],
                "age_band": ["0+", "0+"],
                "sex": ["female", "male"],
                "central_rate": [0.001, 0.001],
            }
        ),
        mortality_external_vintage=2014,
    )


def _transport_universe_fixture(tmp_path: Path) -> SimpleNamespace:
    entrant = 9_999
    truth_attrs = pd.DataFrame(
        {
            "person_id": [1, entrant],
            "sex": ["female", "female"],
            "birth_year": [1989.0, 2000.0],
            "most_recent_report_year": [2019.0, 2019.0],
            "n_marriages": [0.0, 0.0],
            "death_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "censor_year": [2019.0, 2019.0],
            "weight": [2.5, 3.0],
            "start_exposure_year": [2004.0, 2015.0],
        }
    )
    person_years = transitions._assign_state(
        transitions._person_years_frame(truth_attrs),
        pd.DataFrame(),
    )
    empty_events = pd.DataFrame(
        columns=[
            "person_id",
            "year",
            "age",
            "sex",
            "weight",
            "transition",
        ]
    )
    marital = transitions.MaritalPanel(
        person_years=person_years,
        events=empty_events,
        attrs=truth_attrs,
    )
    anchor = pd.DataFrame(
        {
            "person_id": [1, entrant],
            "household_id": [101, 202],
            "weight": [2.5, 3.0],
            "anchor_wave": [2015, 2017],
        }
    )
    presence = {
        2013: set(),
        2015: {1},
        2017: {1, entrant},
        2019: {1, entrant},
        2021: set(),
    }
    _, marital_person_years = m6_cells.marital_tables_from_panel(
        marital,
        anchor,
        presence,
    )
    household = household_data.HouseholdCompositionPanel(
        person_waves=pd.DataFrame(
            {
                "person_id": [1, entrant],
                "year": [2015, 2017],
                "coresident_spouse": [False, False],
                "coresident_parent": [False, False],
                "coresident_child": [False, False],
                "coresident_grandchild": [False, False],
                "multigen": [False, False],
                "hh_size": [1, 1],
            }
        ),
        attrs=pd.DataFrame({"person_id": [1, entrant]}),
    )
    cohabitation = pd.DataFrame(
        {
            "person_id": [1, entrant],
            "year": [2015, 2017],
            "cohabiting": [False, False],
        }
    )
    builders = PanelBuilderInputs(
        anchor=anchor,
        marital=marital,
        household=household,
        cohabitation=cohabitation,
    )
    demographic_panel = pd.DataFrame(
        {
            "person_id": [1, 1, 1, entrant, entrant],
            "period": [2015, 2017, 2019, 2017, 2019],
            "age": [26, 28, 30, 17, 19],
            "sequence": [1, 1, 1, 1, 1],
            "relationship": [10, 10, 10, 10, 10],
            "weight": [2.5, 2.5, 2.5, 3.0, 3.0],
            "interview": [101, 101, 101, 202, 202],
        }
    )
    death_records = pd.DataFrame(
        {
            "person_id": [1, entrant],
            "sex": pd.array(["female", "female"], dtype="string"),
            "death_year": pd.array([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    earnings_rows = []
    for person_id, birth_year, weight in (
        (1, 1989, 2.5),
        (entrant, 2000, 3.0),
    ):
        for period in (2014, 2016, 2018, 2020):
            earnings_rows.append(
                {
                    "person_id": person_id,
                    "period": period,
                    "age": period - birth_year,
                    "weight": weight,
                    "earnings": 20_000.0 + 100.0 * (period - 2014),
                }
            )
    earnings_panel = pd.DataFrame(earnings_rows)
    disability_panel = disability_data.DisabilityPanel(
        person_years=pd.DataFrame(
            {
                "person_id": [1, 1, 1, entrant, entrant],
                "period": [2015, 2017, 2019, 2017, 2019],
                "status_code": [1, 1, 1, 1, 1],
                "disabled": [False, False, False, False, False],
                "retired": [False, False, False, False, False],
            }
        ),
        pairs=pd.DataFrame(),
    )

    fit_inputs = _synthetic_refit_inputs()
    inputs = runner.M6HarnessInputs(
        refit_inputs=fit_inputs,
        panel_builder_inputs=builders,
        truth=SimpleNamespace(
            anchor=anchor,
            marital_person_years=marital_person_years,
        ),
        demographic_panel=demographic_panel,
        earnings_panel=earnings_panel,
        disability_status=object(),
        disability_panel=disability_panel,
        death_records=death_records,
        provenance={"synthetic_transport_universe": True},
    )
    preparation = replace(
        _preparation(tmp_path),
        resolved=_first_marriage_resolved(),
    )
    return SimpleNamespace(
        fit_inputs=fit_inputs,
        inputs=inputs,
        bundle=None,
        preparation=preparation,
        post_tstar_person_id=entrant,
        projection_calls=[],
    )


def _synthetic_operations(
    events, *, bundle=None
) -> M6Candidate2RunnerOperations:
    candidate_bundle = object() if bundle is None else bundle
    incumbent_bundle = object()
    candidate_score_ids = set()

    def record(name, value=None):
        events.append(name)
        return value

    def materialize(_inputs, observed_bundle):
        assert observed_bundle is candidate_bundle
        bindings = _matching_bindings()
        return record(
            "materialize",
            M6RefitPhase(
                bundle=candidate_bundle,
                mortality=None,
                population=None,
                lineage={
                    "resolved_spec_sha256s": {
                        "family_transitions": (
                            bindings.candidate_spec_sha256s[
                                "family_transitions"
                            ]
                        ),
                        "engine_candidate": bindings.candidate_spec_sha256s[
                            "engine_operations"
                        ],
                    },
                    "engine_candidate_id": bindings.candidate_spec_ids[
                        "engine_operations"
                    ],
                },
            ),
        )

    def materialize_incumbent(candidate_phase, observed_bundle):
        assert candidate_phase.bundle is candidate_bundle
        assert observed_bundle is incumbent_bundle
        return record(
            "materialize_incumbent",
            replace(candidate_phase, bundle=incumbent_bundle),
        )

    def score(_inputs, phase, _contract, seed):
        arm = "incumbent" if phase.bundle is incumbent_bundle else "candidate"
        result = M6SeedRun(
            seed=seed,
            score=_fake_seed(seed),
            side_a_units={"person": 2},
        )
        if arm == "candidate":
            candidate_score_ids.add(id(result.score))
        return record(
            f"score_{arm}_{seed}",
            result,
        )

    def aggregate(_contract, scores):
        assert len(scores) == len(runner.MUST_NOT_REGRESS_SEEDS)
        assert all(id(score) in candidate_score_ids for score in scores)
        return record("aggregate", _FakeGateScore(tuple(scores)))

    return M6Candidate2RunnerOperations(
        fit=lambda _inputs: record("fit", candidate_bundle),
        first_marriage_preflight=lambda _bundle: record(
            "first_marriage_preflight", {"passed": True}
        ),
        fit_postrepair_incumbent=lambda _inputs, observed: (
            record("fit_incumbent", incumbent_bundle)
            if observed is candidate_bundle
            else (_ for _ in ()).throw(AssertionError("wrong bundle"))
        ),
        materialize=materialize,
        materialize_postrepair_incumbent=materialize_incumbent,
        first_marriage_disclosure=lambda *_args: record(
            "first_marriage_disclosure", {"passed": True}
        ),
        preflight_1=lambda *_args: record("preflight_1", {"passed": True}),
        preflight_2=lambda *_args: record("preflight_2", {"passed": True}),
        score_seed=score,
        aggregate=aggregate,
        domain_floor=lambda *_args: record(
            "domain_floor", {"truth_side_only": True}
        ),
        report_only=lambda *_args: record(
            "report_only",
            {
                "domain_earnings_floor": {"truth_side_only": True},
                "family_b": {"gating": False},
                "family_c": {"gating": False},
            },
        ),
        write=lambda *_args: record("write"),
    )


def test_candidate2_default_preflight_threads_registered_family_spec(
    monkeypatch,
):
    identity = runner.resolve_candidate2_identity()
    calls = []

    monkeypatch.setattr(
        runner,
        "resolve_candidate2_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        runner,
        "assert_candidate2_identity_is_frozen",
        lambda observed: calls.append(("identity", observed)),
    )

    def capture(inputs, phase, contract, *, family_candidate_spec=None):
        calls.append(
            (
                "preflight_1",
                inputs,
                phase,
                contract,
                family_candidate_spec,
            )
        )
        return {"passed": True}

    monkeypatch.setattr(runner.m6_runner, "_run_m6_preflight_1", capture)
    operations = runner.default_operations()
    inputs = object()
    phase = object()
    contract = object()

    assert operations.preflight_1(inputs, phase, contract) == {"passed": True}
    assert calls == [
        ("identity", identity),
        (
            "preflight_1",
            inputs,
            phase,
            contract,
            identity.family_spec,
        ),
    ]


def _transport_universe_operations(
    fixture: SimpleNamespace,
    events: list[str],
    disclosure,
) -> M6Candidate2RunnerOperations:
    operations = runner.default_operations()

    def fit(fit_inputs):
        assert fit_inputs is fixture.fit_inputs
        events.append("fit")
        fixture.bundle = operations.fit(fit_inputs)
        return fixture.bundle

    def fit_preflight(bundle):
        assert bundle is fixture.bundle
        events.append("first_marriage_preflight")
        return runner._preflight_candidate2_first_marriage(bundle)

    def materialize(inputs, bundle):
        assert bundle is fixture.bundle
        events.append("materialize")
        return operations.materialize(inputs, bundle)

    def fit_incumbent(inputs, bundle):
        assert bundle is fixture.bundle
        events.append("fit_incumbent")
        return operations.fit_postrepair_incumbent(inputs, bundle)

    def materialize_incumbent(phase, bundle):
        assert phase.bundle is fixture.bundle
        events.append("materialize_incumbent")
        return operations.materialize_postrepair_incumbent(phase, bundle)

    def transport_disclosure(inputs, phase, resolved, fit_preflight):
        events.append("first_marriage_disclosure")
        return disclosure(inputs, phase, resolved, fit_preflight)

    def preflight_1(inputs, phase, contract):
        assert phase.bundle is fixture.bundle
        assert contract is fixture.preparation.resolved.contract
        events.append("preflight_1")
        return operations.preflight_1(inputs, phase, contract)

    def preflight_2(inputs, phase, contract):
        assert inputs is fixture.inputs
        assert phase.bundle is fixture.bundle
        assert contract is fixture.preparation.resolved.contract
        events.append("preflight_2")
        return runner.m6_runner.run_m6_preflight_2(inputs, phase, contract)

    def score_seed(inputs, phase, contract, seed):
        arm = "candidate" if phase.bundle is fixture.bundle else "incumbent"
        events.append(f"score_{arm}_{seed}")
        return runner.m6_runner.score_m6_seed(
            inputs,
            phase,
            contract,
            seed,
        )

    def aggregate(contract, scores):
        assert len(scores) == len(runner.MUST_NOT_REGRESS_SEEDS)
        events.append("aggregate")
        return runner.m6_runner.aggregate_gate(contract, scores)

    def write(path, artifact, sidecar):
        events.append("write")
        runner.write_new_candidate2_artifact(path, artifact, sidecar)

    def domain_floor(*_args):
        events.append("domain_floor")
        return {"truth_side_only": True}

    def report_only(*_args):
        events.append("report_only")
        return {
            "domain_earnings_floor": {"truth_side_only": True},
            "family_b": {"gating": False},
            "family_c": {"gating": False},
        }

    return replace(
        operations,
        fit=fit,
        first_marriage_preflight=fit_preflight,
        fit_postrepair_incumbent=fit_incumbent,
        materialize=materialize,
        materialize_postrepair_incumbent=materialize_incumbent,
        first_marriage_disclosure=transport_disclosure,
        preflight_1=preflight_1,
        preflight_2=preflight_2,
        score_seed=score_seed,
        aggregate=aggregate,
        domain_floor=domain_floor,
        report_only=report_only,
        write=write,
    )


def _install_transport_universe_test_seams(
    monkeypatch,
    fixture: SimpleNamespace,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    registered_identity = runner.resolve_candidate2_identity()
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: fixture.preparation,
    )
    monkeypatch.setattr(
        runner,
        "_revalidate_source_identity",
        lambda _preparation: None,
    )
    monkeypatch.setattr(
        runner,
        "assert_candidate2_identity_is_frozen",
        lambda _identity: None,
    )

    def synthetic_identity():
        if fixture.bundle is None:
            return registered_identity
        model = fixture.bundle.family.fitted.first_marriage
        assert model is not None and model.fit_audit is not None
        return replace(
            registered_identity,
            first_marriage_params={
                **registered_identity.first_marriage_params,
                "selected_c": 0.001,
                "final_fit_checksums": dict(model.fit_audit.checksums),
            },
        )

    monkeypatch.setattr(
        runner,
        "resolve_candidate2_identity",
        synthetic_identity,
    )
    monkeypatch.setattr(
        refit_module,
        "_default_qrf_factory",
        _SyntheticQRFFactory(),
    )
    truth_cells = {
        rule.cell: {"rate": 0.2, "n_events": 1, "n_at_risk": 5}
        for rule in fixture.preparation.resolved.contract.cells
    }
    monkeypatch.setattr(
        runner.m6_runner,
        "_truth_cells",
        lambda *_args, **_kwargs: truth_cells,
    )
    monkeypatch.setattr(
        runner.m6_runner,
        "side_a_person_ids",
        lambda anchor, **_kwargs: set(anchor["person_id"]),
    )

    def passthrough(frame, _context, _rng):
        return frame.copy()

    def aging(frame, context, _rng):
        out = frame.copy()
        out["year"] = context.year
        return out

    def earnings(frame, context, _rng):
        out = frame.copy()
        out["synthetic_score_signal"] = (
            0.99 if context.draw_index == 0 else 1.01
        )
        return out

    def marital_core(_frame, _context, _rng):
        return MaritalStepResult(pd.DataFrame(), pd.DataFrame())

    def marital_reader(frame, _context, _marital, _rng):
        return frame.copy()

    modules = PeriodModules(
        mortality=passthrough,
        aging=aging,
        marital_core=marital_core,
        fertility=marital_reader,
        disability=passthrough,
        earnings=earnings,
        claiming=passthrough,
        household_composition=marital_reader,
    )

    def projected_cells(
        _inputs,
        _phase,
        household_population,
        _person_population,
        *,
        draw_index,
    ):
        initial = household_population.initial_slice[["person_id", "year"]]
        result = ProjectionEngine(modules).project(
            initial,
            end_year=runner.m6_runner.PROJECTION_END_YEAR,
            start_year=2014,
            draw_index=draw_index,
            metadata=household_population.projection_metadata(),
        )
        final_slice = result.slices[-1]
        fixture.projection_calls.append(
            {
                "draw_index": draw_index,
                "trace_years": [trace.year for trace in result.traces],
                "final_person_ids": set(final_slice["person_id"]),
            }
        )
        multiplier = float(final_slice["synthetic_score_signal"].mean())
        cells = {
            key: {"rate": record["rate"] * multiplier}
            for key, record in truth_cells.items()
        }
        return cells, result, {}, result, {}

    monkeypatch.setattr(
        runner.m6_runner,
        "_projected_cells",
        projected_cells,
    )
    monkeypatch.setattr(
        runner.m6_runner,
        "_draw_report",
        lambda *_args, draw_index: {"draw_index": draw_index},
    )


def test_transport_universe_end_to_end_scores_and_writes_artifact(
    monkeypatch,
    tmp_path,
):
    fixture = _transport_universe_fixture(tmp_path)
    _install_transport_universe_test_seams(monkeypatch, fixture)
    events = []
    operations = _transport_universe_operations(
        fixture,
        events,
        runner._first_marriage_transport_disclosure,
    )

    def load_full_inputs():
        events.append("load_full_inputs")
        return fixture.inputs

    artifact = runner.execute_registered_m6_candidate2_run(
        runner.M6Candidate2InputPlan(
            fit_inputs=fixture.fit_inputs,
            load_full_inputs=load_full_inputs,
        ),
        registration_id="9999999999",
        root=tmp_path,
        operations=operations,
        preparation=fixture.preparation,
    )

    assert events == [
        "fit",
        "first_marriage_preflight",
        "fit_incumbent",
        "load_full_inputs",
        "materialize",
        "materialize_incumbent",
        "first_marriage_disclosure",
        "preflight_1",
        "preflight_2",
        "score_candidate_0",
        "score_candidate_1",
        "score_candidate_2",
        "score_candidate_3",
        "score_candidate_4",
        "score_incumbent_0",
        "score_incumbent_1",
        "score_incumbent_2",
        "score_incumbent_3",
        "score_incumbent_4",
        "aggregate",
        "domain_floor",
        "report_only",
        "write",
    ]
    assert len(fixture.projection_calls) == 20
    assert all(
        call["trace_years"] == list(range(2015, 2023))
        for call in fixture.projection_calls
    )
    assert all(
        call["final_person_ids"] == {1, fixture.post_tstar_person_id}
        for call in fixture.projection_calls
    )
    fit_ids = set(fixture.fit_inputs.family_context.panel.attrs["person_id"])
    truth_ids = set(
        fixture.inputs.panel_builder_inputs.marital.attrs["person_id"]
    )
    support_ids = set(fixture.inputs.truth.marital_person_years["person_id"])
    assert truth_ids - fit_ids == {fixture.post_tstar_person_id}
    assert support_ids == truth_ids

    transport = artifact["preflights"]["first_marriage_fit_and_transport"][
        "gated_cell_transport"
    ][runner.FIRST_MARRIAGE_GATE_CELL]
    assert transport["at_risk_rows"] == 6
    assert transport["at_risk_f6_weight"] == 16.0
    assert transport["in_support"] == {"rows": 4, "f6_weight": 10.0}
    assert transport["age_out_of_support"] == {
        "rows": 0,
        "f6_weight": 0.0,
    }
    assert transport["unseen_cohort"] == {"rows": 2, "f6_weight": 6.0}
    recertification = artifact["preflights"]["candidate9_recertification"]
    assert recertification["passed"] is True
    assert len(recertification["cells"]) == 15
    assert all(cell["passed"] for cell in recertification["cells"])
    assert artifact["preflights"]["earnings_sign_path"] == {
        "branch": "certified_target_models_reconstruction",
        "gates_checked": ["shared_gate", "zero_anchor_gate"],
        "probe_rows": 2,
        "output_signs": [0, 1, 0, 1],
    }
    assert artifact["verdict"]["status"] == "PASS"
    assert fixture.preparation.destination.exists()
    sidecar = Path(f"{fixture.preparation.destination}.env.json")
    assert sidecar.exists()
    assert json.loads(fixture.preparation.destination.read_text()) == artifact
    assert sidecar.read_bytes() == fixture.preparation.sidecar_payload


def test_transport_universe_pre_fix_join_reproduces_designed_abort(
    monkeypatch,
    tmp_path,
):
    fixture = _transport_universe_fixture(tmp_path)
    _install_transport_universe_test_seams(monkeypatch, fixture)
    events = []

    def pre_fix_disclosure(inputs, phase, resolved, fit_preflight):
        pre_fix_marital = replace(
            inputs.panel_builder_inputs.marital,
            attrs=inputs.refit_inputs.family_context.panel.attrs,
        )
        pre_fix_builders = replace(
            inputs.panel_builder_inputs,
            marital=pre_fix_marital,
        )
        pre_fix_inputs = replace(
            inputs,
            panel_builder_inputs=pre_fix_builders,
        )
        return runner._first_marriage_transport_disclosure(
            pre_fix_inputs,
            phase,
            resolved,
            fit_preflight,
        )

    operations = _transport_universe_operations(
        fixture,
        events,
        pre_fix_disclosure,
    )

    def load_full_inputs():
        events.append("load_full_inputs")
        return fixture.inputs

    with pytest.raises(runner.M6Candidate2DesignedAbort) as caught:
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=fixture.fit_inputs,
                load_full_inputs=load_full_inputs,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=operations,
            preparation=fixture.preparation,
        )

    report = caught.value.report
    assert report["status"] == "FIRST_MARRIAGE_PRESCORE_ABORT"
    assert report["abort"] == {
        "type": "FirstMarriagePreflightAbort",
        "message": (
            "first-marriage diagnostic support lacks canonical birth year"
        ),
        "stage": "transport_disclosure",
        "converted_to_gate_fail": False,
    }
    assert report["fences"] == {
        "full_inputs_loaded": True,
        "materialization_completed": True,
        "projection_started": False,
        "score_started": False,
        "candidate_artifact_written": False,
    }
    assert events == [
        "fit",
        "first_marriage_preflight",
        "fit_incumbent",
        "load_full_inputs",
        "materialize",
        "materialize_incumbent",
        "first_marriage_disclosure",
    ]
    assert fixture.projection_calls == []
    assert not fixture.preparation.destination.exists()
    assert not Path(f"{fixture.preparation.destination}.env.json").exists()


def test_candidate2_runner_orders_phases_and_commits_last(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runner, "_revalidate_source_identity", lambda _preparation: None
    )
    fit_inputs = object()
    full_inputs = runner.M6HarnessInputs(
        refit_inputs=fit_inputs,
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )

    def load_full_inputs():
        events.append("load_full_inputs")
        return full_inputs

    artifact = runner.execute_registered_m6_candidate2_run(
        runner.M6Candidate2InputPlan(
            fit_inputs=fit_inputs,
            load_full_inputs=load_full_inputs,
        ),
        registration_id="9999999999",
        root=tmp_path,
        operations=_synthetic_operations(events),
        preparation=preparation,
    )
    assert events == [
        "fit",
        "first_marriage_preflight",
        "fit_incumbent",
        "load_full_inputs",
        "materialize",
        "materialize_incumbent",
        "first_marriage_disclosure",
        "preflight_1",
        "preflight_2",
        "score_candidate_0",
        "score_candidate_1",
        "score_candidate_2",
        "score_candidate_3",
        "score_candidate_4",
        "score_incumbent_0",
        "score_incumbent_1",
        "score_incumbent_2",
        "score_incumbent_3",
        "score_incumbent_4",
        "aggregate",
        "domain_floor",
        "report_only",
        "write",
    ]
    assert artifact["candidate"]["number"] == 2
    assert artifact["run"]["name"] == "gate_m6_candidate2"
    assert artifact["integrity"]["source"]["git_commit"] == "a" * 40


def test_source_drift_at_final_revalidation_prevents_publication(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    revalidations = 0

    def revalidate(_preparation):
        nonlocal revalidations
        revalidations += 1
        if revalidations == 2:
            raise RuntimeError("source commit changed")

    monkeypatch.setattr(runner, "_revalidate_source_identity", revalidate)
    fit_inputs = object()
    full_inputs = runner.M6HarnessInputs(
        refit_inputs=fit_inputs,
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )
    with pytest.raises(RuntimeError, match="source commit changed"):
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=fit_inputs,
                load_full_inputs=lambda: full_inputs,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=_synthetic_operations(events),
            preparation=preparation,
        )
    assert revalidations == 2
    assert events[-1] == "report_only"
    assert "write" not in events


def test_full_inputs_must_reuse_the_exact_preflighted_fit_inputs(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runner, "_revalidate_source_identity", lambda _preparation: None
    )
    full_inputs = runner.M6HarnessInputs(
        refit_inputs=object(),
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )
    with pytest.raises(RuntimeError, match="exact preflighted fit-input"):
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=object(),
                load_full_inputs=lambda: full_inputs,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=_synthetic_operations(events),
            preparation=preparation,
        )
    assert events == ["fit", "first_marriage_preflight", "fit_incumbent"]


def test_transport_disclosure_abort_is_structured_after_materialization(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runner, "_revalidate_source_identity", lambda _preparation: None
    )
    fit_inputs = object()
    full_inputs = runner.M6HarnessInputs(
        refit_inputs=fit_inputs,
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )
    operations = _synthetic_operations(events)

    def abort(*_args):
        events.append("first_marriage_disclosure")
        raise runner.FirstMarriagePreflightAbort("synthetic transport abort")

    operations = replace(operations, first_marriage_disclosure=abort)
    with pytest.raises(runner.M6Candidate2DesignedAbort) as caught:
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=fit_inputs,
                load_full_inputs=lambda: full_inputs,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=operations,
            preparation=preparation,
        )
    assert caught.value.report["status"] == "FIRST_MARRIAGE_PRESCORE_ABORT"
    assert caught.value.report["fences"] == {
        "full_inputs_loaded": True,
        "materialization_completed": True,
        "projection_started": False,
        "score_started": False,
        "candidate_artifact_written": False,
    }
    assert events == [
        "fit",
        "first_marriage_preflight",
        "fit_incumbent",
        "materialize",
        "materialize_incumbent",
        "first_marriage_disclosure",
    ]


def test_preflight1_failure_publishes_structured_signed_table(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runner,
        "_revalidate_source_identity",
        lambda _preparation: None,
    )
    fit_inputs = object()
    full_inputs = runner.M6HarnessInputs(
        refit_inputs=fit_inputs,
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )
    cells = []
    for channel_set, names in RECERTIFICATION_CHANNEL_SETS.items():
        for name in names:
            failed = name == "legal_core"
            injected_mean = 0.21 if failed else 0.2
            internal_mean = 0.2
            cells.append(
                RecertificationCell(
                    channel_set=channel_set,
                    cell=name,
                    injected_mean=injected_mean,
                    internal_mean=internal_mean,
                    absolute_delta=abs(injected_mean - internal_mean),
                    sigma_of_mean_difference=0.001,
                    tolerance=0.003,
                    passed=not failed,
                )
            )
    failure = Candidate9RecertificationFailure(
        RecertificationResult(
            sigma_multiplier=3.0,
            cells=tuple(cells),
        ),
        ("legal_spouse_residual/legal_core: synthetic wiring defect",),
    )
    operations = _synthetic_operations(events)

    def abort_preflight(*_args):
        events.append("preflight_1")
        raise failure

    operations = replace(operations, preflight_1=abort_preflight)

    def load_full_inputs():
        events.append("load_full_inputs")
        return full_inputs

    with pytest.raises(runner.M6Candidate2DesignedAbort) as caught:
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=fit_inputs,
                load_full_inputs=load_full_inputs,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=operations,
            preparation=preparation,
        )

    report = caught.value.report
    assert report["status"] == "CANDIDATE9_RECERTIFICATION_PRESCORE_ABORT"
    assert report["abort"] == {
        "type": "Candidate9RecertificationFailure",
        "message": str(failure),
        "stage": "preflight_1",
        "converted_to_gate_fail": False,
    }
    assert report["fences"] == {
        "full_inputs_loaded": True,
        "materialization_completed": True,
        "projection_started": False,
        "score_started": False,
        "candidate_artifact_written": False,
    }
    recertification = report["preflights"]["candidate9_recertification"]
    assert recertification["passed"] is False
    assert recertification["sigma_multiplier"] == 3.0
    rows = recertification["cells"]
    keys = [(row["channel_set"], row["cell"]) for row in rows]
    assert len(keys) == len(set(keys)) == 15
    assert keys == [
        (channel_set, name)
        for channel_set, names in RECERTIFICATION_CHANNEL_SETS.items()
        for name in names
    ]
    assert all(
        row["signed_delta"]
        == pytest.approx(row["injected_mean"] - row["internal_mean"])
        for row in rows
    )
    assert [row["cell"] for row in rows if not row["passed"]] == ["legal_core"]
    assert events == [
        "fit",
        "first_marriage_preflight",
        "fit_incumbent",
        "load_full_inputs",
        "materialize",
        "materialize_incumbent",
        "first_marriage_disclosure",
        "preflight_1",
    ]
    assert not preparation.destination.exists()
    assert not Path(f"{preparation.destination}.env.json").exists()


def test_preflight1_unrelated_error_is_not_converted(monkeypatch, tmp_path):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runner,
        "_revalidate_source_identity",
        lambda _preparation: None,
    )
    fit_inputs = object()
    full_inputs = runner.M6HarnessInputs(
        refit_inputs=fit_inputs,
        panel_builder_inputs=object(),
        truth=object(),
        demographic_panel=object(),
        earnings_panel=object(),
        disability_status=object(),
        disability_panel=object(),
        death_records=object(),
        provenance={},
    )
    failure = RuntimeError("unrelated preflight defect")
    operations = _synthetic_operations(events)

    def abort_preflight(*_args):
        events.append("preflight_1")
        raise failure

    operations = replace(operations, preflight_1=abort_preflight)

    with pytest.raises(RuntimeError) as caught:
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=fit_inputs,
                load_full_inputs=lambda: full_inputs,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=operations,
            preparation=preparation,
        )

    assert caught.value is failure
    assert not preparation.destination.exists()
    assert not Path(f"{preparation.destination}.env.json").exists()


def test_first_marriage_designed_abort_precedes_truth_load_and_write(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    events = []
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runner, "_revalidate_source_identity", lambda _preparation: None
    )
    operations = _synthetic_operations(events)

    class Audit:
        def canonical_dict(self):
            return {"eligible": False, "gradient_inf_norm": 2.0}

    model = SimpleNamespace(fit_audit=Audit())
    bundle = SimpleNamespace(
        family=SimpleNamespace(fitted=SimpleNamespace(first_marriage=model))
    )
    monkeypatch.setattr(
        runner,
        "recompute_support_aware_first_marriage_checksums",
        lambda observed: (
            {"model": "replayed"}
            if observed is model
            else (_ for _ in ()).throw(AssertionError("wrong model"))
        ),
    )

    def fit(_inputs):
        events.append("fit")
        return bundle

    def abort(_bundle):
        events.append("first_marriage_preflight")
        raise runner.FirstMarriagePreflightAbort("synthetic no-fit")

    operations = replace(
        operations,
        fit=fit,
        first_marriage_preflight=abort,
    )

    def poisoned_full_loader():
        raise AssertionError("holdout truth loaded after designed abort")

    with pytest.raises(runner.M6Candidate2DesignedAbort) as caught:
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=object(),
                load_full_inputs=poisoned_full_loader,
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=operations,
            preparation=preparation,
        )
    assert events == ["fit", "first_marriage_preflight"]
    assert caught.value.report["status"] == (
        "NO_REGISTERABLE_FIRST_MARRIAGE_FIT"
    )
    assert caught.value.report["fences"] == {
        "full_inputs_loaded": False,
        "materialization_completed": False,
        "projection_started": False,
        "score_started": False,
        "candidate_artifact_written": False,
    }
    diagnostics = caught.value.report["first_marriage_diagnostics"]
    assert diagnostics["fit_audit"] == {
        "eligible": False,
        "gradient_inf_norm": 2.0,
    }
    assert diagnostics["recomputed_checksums"] == {"model": "replayed"}
    assert diagnostics["selection_ledger"] == (
        preparation.bindings.first_marriage_selection_ledger
    )
    assert diagnostics["registered_spec_params"]["selected_c"] == 0.001
    assert diagnostics["hazard_table_ages_18_29"] is None
    assert not preparation.destination.exists()
    assert not Path(f"{preparation.destination}.env.json").exists()


def test_designed_abort_refuses_binding_drift_before_publication(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    changed = replace(
        preparation,
        bindings=replace(preparation.bindings, dependency_sha256="8" * 64),
    )
    guarded = iter((preparation, changed))
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: next(guarded),
    )
    monkeypatch.setattr(
        runner, "_revalidate_source_identity", lambda _preparation: None
    )
    events = []
    operations = replace(
        _synthetic_operations(events),
        first_marriage_preflight=lambda _bundle: (_ for _ in ()).throw(
            runner.FirstMarriagePreflightAbort("synthetic no-fit")
        ),
    )
    with pytest.raises(RuntimeError, match="designed-abort publication"):
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=object(),
                load_full_inputs=lambda: (_ for _ in ()).throw(
                    AssertionError("truth loader reached")
                ),
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=operations,
            preparation=preparation,
        )


def test_execution_rejects_guarded_binding_drift_before_fit(
    monkeypatch, tmp_path
):
    preparation = _preparation(tmp_path)
    changed = replace(
        preparation,
        bindings=replace(
            preparation.bindings,
            dependency_sha256="9" * 64,
        ),
    )
    monkeypatch.setattr(
        runner,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: changed,
    )
    events = []
    with pytest.raises(RuntimeError, match="input-plan factory"):
        runner.execute_registered_m6_candidate2_run(
            runner.M6Candidate2InputPlan(
                fit_inputs=object(),
                load_full_inputs=lambda: (_ for _ in ()).throw(
                    AssertionError("truth loader reached")
                ),
            ),
            registration_id="9999999999",
            root=tmp_path,
            operations=_synthetic_operations(events),
            preparation=preparation,
        )
    assert events == []
