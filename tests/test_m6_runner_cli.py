"""Synthetic governance checks for the M6 one-shot CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script_module():
    path = ROOT / "scripts" / "run_gate_m6_candidate1.py"
    spec = importlib.util.spec_from_file_location("m6_runner_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_guards_before_input_import_and_prints_required_caveats(
    monkeypatch, capsys, tmp_path
):
    module = _script_module()
    events = []

    class Inputs:
        pass

    monkeypatch.setattr(module, "M6HarnessInputs", Inputs)
    monkeypatch.setattr(
        module,
        "guard_registered_m6_run",
        lambda **kwargs: events.append("guard"),
    )

    def input_factory(_specification):
        events.append("input_factory_import")
        return lambda: Inputs()

    monkeypatch.setattr(module, "_input_factory", input_factory)

    def execute(*args, **kwargs):
        events.append("execute")
        return {"verdict": {"valid": True, "pass": True}}

    monkeypatch.setattr(module, "execute_registered_m6_run", execute)
    output = tmp_path / "fresh.json"
    assert (
        module.main(
            [
                "--registration-id",
                "9999999999",
                "--input-factory",
                "registered:inputs",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    assert events == ["guard", "input_factory_import", "execute"]
    printed = capsys.readouterr().out
    assert "PASS" in printed
    assert "Certifies nothing about mortality drift" in printed
    assert "M6-first-certified forward law" in printed
    assert "no gate_1" in printed
