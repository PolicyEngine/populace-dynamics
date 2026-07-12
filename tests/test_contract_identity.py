"""Contract identity and opt-in artifact sidecar tests."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy
import sklearn

from populace_dynamics.artifacts import write_new
from populace_dynamics.contract import (
    ContractRef,
    contract_revision,
    environment_block,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test__given_checkout__then_contract_revision_matches_gates_blob():
    # Given
    expected = _git("rev-parse", "HEAD:gates.yaml")

    # When
    actual = contract_revision(ROOT)

    # Then
    assert actual == expected


def test__given_dirty_contract__then_contract_identity_is_refused(tmp_path):
    # Given
    repository = tmp_path / "repository"
    repository.mkdir()
    contract = repository / "gates.yaml"
    contract.write_text("locked: true\n", encoding="utf-8")
    _fixture_git(repository, "init", "-b", "master")
    _fixture_git(repository, "add", "gates.yaml")
    _fixture_git(
        repository,
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract-test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Add contract",
    )
    contract.write_text("locked: false\n", encoding="utf-8")

    # When / Then
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        contract_revision(repository)


def test__given_checkout__then_contract_ref_binds_blob_head_and_path():
    # Given / When
    ref = ContractRef.current(ROOT)

    # Then
    assert ref == ContractRef(
        blob_sha=_git("rev-parse", "HEAD:gates.yaml"),
        head_sha=_git("rev-parse", "HEAD"),
        path="gates.yaml",
    )
    with pytest.raises(FrozenInstanceError):
        ref.path = "different.yaml"


def test__given_running_process__then_environment_block_pins_versions():
    # Given
    expected = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }

    # When
    environment = environment_block()

    # Then
    assert environment == expected
    assert all(
        isinstance(value, str) and value for value in environment.values()
    )


def test__given_default_artifact_write__then_no_sidecar_is_created(tmp_path):
    # Given
    destination = tmp_path / "candidate.json"
    payload = {"candidate": 1}

    # When
    write_new(destination, payload)

    # Then
    assert json.loads(destination.read_text()) == payload
    assert not Path(f"{destination}.env.json").exists()


def test__given_sidecar_opt_in__then_identity_and_environment_are_written(
    tmp_path,
):
    # Given
    destination = tmp_path / "candidate.json"
    payload = '{"candidate": 1}\n'

    # When
    write_new(destination, payload, sidecar=True)

    # Then
    assert destination.read_text() == payload
    sidecar = Path(f"{destination}.env.json")
    metadata = json.loads(sidecar.read_text())
    assert metadata == {
        "environment": environment_block(),
        "contract": {
            "blob_sha": contract_revision(ROOT),
            "head_sha": _git("rev-parse", "HEAD"),
            "path": "gates.yaml",
        },
    }


def test__given_existing_sidecar__then_primary_artifact_is_not_created(
    tmp_path,
):
    # Given
    destination = tmp_path / "candidate.json"
    sidecar = Path(f"{destination}.env.json")
    original = b"reserved sidecar bytes\n"
    sidecar.write_bytes(original)

    # When / Then
    with pytest.raises(FileExistsError, match="one-shot rule"):
        write_new(destination, {"candidate": 1}, sidecar=True)
    assert not destination.exists()
    assert sidecar.read_bytes() == original


def test__given_written_pair__then_repeat_write_changes_neither_file(tmp_path):
    # Given
    destination = tmp_path / "candidate.json"
    write_new(destination, {"candidate": 1}, sidecar=True)
    sidecar = Path(f"{destination}.env.json")
    original_artifact = destination.read_bytes()
    original_sidecar = sidecar.read_bytes()

    # When / Then
    with pytest.raises(FileExistsError, match="one-shot rule"):
        write_new(destination, {"candidate": 2}, sidecar=True)
    assert destination.read_bytes() == original_artifact
    assert sidecar.read_bytes() == original_sidecar
