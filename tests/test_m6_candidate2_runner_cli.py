"""Synthetic CLI ordering checks for the M6 candidate-2 runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from populace_dynamics.harness.m6_candidate2_runner import DEFAULT_OUTPUT

ROOT = Path(__file__).resolve().parents[1]


def _script_module():
    path = ROOT / "scripts" / "run_gate_m6_candidate2.py"
    spec = importlib.util.spec_from_file_location("m6_candidate2_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_has_no_candidate_default_and_guards_before_input_import(
    monkeypatch, capsys
):
    module = _script_module()
    events = []
    preparation = object()

    class Plan:
        pass

    parser = module.parser()
    assert "candidate" not in {action.dest for action in parser._actions}
    assert DEFAULT_OUTPUT == Path("runs/gate_m6_candidate2_v1.json")
    monkeypatch.setattr(module, "M6Candidate2InputPlan", Plan)
    monkeypatch.setattr(
        module,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: events.append("guard") or preparation,
    )

    def input_factory(_specification):
        events.append("input_factory_import")
        return lambda: events.append("input_plan_build") or Plan()

    monkeypatch.setattr(module, "_input_factory", input_factory)

    def execute(*_args, **kwargs):
        events.append("execute")
        assert kwargs["preparation"] is preparation
        return {"verdict": {"valid": True, "pass": True, "status": "PASS"}}

    monkeypatch.setattr(
        module,
        "execute_registered_m6_candidate2_run",
        execute,
    )
    assert (
        module.main(
            [
                "--registration-id",
                "9999999999",
                "--input-factory",
                "registered:inputs",
            ]
        )
        == 0
    )
    assert events == [
        "guard",
        "input_factory_import",
        "input_plan_build",
        "execute",
    ]
    printed = capsys.readouterr().out
    assert "M6 candidate 2 PASS" in printed
    assert str(DEFAULT_OUTPUT) in printed
    assert "rank-refresh law" in printed


def test_cli_publishes_structured_designed_abort(monkeypatch, capsys):
    module = _script_module()
    preparation = object()

    class Plan:
        pass

    monkeypatch.setattr(module, "M6Candidate2InputPlan", Plan)
    monkeypatch.setattr(
        module,
        "guard_registered_m6_candidate2_run",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(module, "_input_factory", lambda _spec: Plan)
    report = {
        "schema_version": "gate_m6_candidate2.designed_abort.v1",
        "status": "NO_REGISTERABLE_FIRST_MARRIAGE_FIT",
        "fences": {"candidate_artifact_written": False},
    }

    def abort(*_args, **_kwargs):
        raise module.M6Candidate2DesignedAbort(report)

    monkeypatch.setattr(module, "execute_registered_m6_candidate2_run", abort)
    assert (
        module.main(
            [
                "--registration-id",
                "9999999999",
                "--input-factory",
                "registered:inputs",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out) == report


def test_input_factory_module_must_be_tracked_under_guarded_source():
    module = _script_module()
    try:
        module._input_factory("json:loads")
    except RuntimeError as error:
        assert "outside the guarded source tree" in str(error)
    else:
        raise AssertionError("external input factory was accepted")
