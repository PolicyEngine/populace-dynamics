import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from populace_dynamics.artifacts import write_new

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "runs" / "legacy_manifest_v1.json"
GENERATED_AT_COMMIT = "747966cd7426a2a2292ce3dd093a8fe65d9a9351"
LEGACY_PATTERNS = (
    "scripts/run_gate2_candidate*.py",
    "runs/gate2_hazard_v*.json",
    "scripts/gate2_forensics*",
    "runs/gate2_forensics*",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _legacy_paths() -> list[str]:
    paths = {"gates.yaml"}
    for pattern in LEGACY_PATTERNS:
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.glob(pattern)
            if path.is_file()
        )
    return sorted(paths)


def test_legacy_manifest_matches_working_tree():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert set(manifest) == {
        "schema_version",
        "generated_at_commit",
        "contract_revision",
        "entries",
    }
    assert manifest["schema_version"] == "legacy_manifest.v1"
    assert manifest["generated_at_commit"] == GENERATED_AT_COMMIT
    assert manifest["contract_revision"] == _git(
        "rev-parse", "HEAD:gates.yaml"
    )

    entries = manifest["entries"]
    manifest_paths = [entry["path"] for entry in entries]
    assert len(manifest_paths) == len(set(manifest_paths))
    assert set(manifest_paths) == set(_legacy_paths())
    is_shallow = _git("rev-parse", "--is-shallow-repository") == "true"

    for entry in entries:
        assert set(entry) == {
            "path",
            "sha256",
            "n_bytes",
            "originating_commit",
        }
        relative_path = entry["path"]
        payload = (ROOT / relative_path).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["n_bytes"] == len(payload)
        assert len(entry["originating_commit"]) == 40
        if not is_shallow:
            assert entry["originating_commit"] == _git(
                "log",
                "--diff-filter=A",
                "-1",
                "--format=%H",
                "--",
                relative_path,
            )


def test_write_new_refuses_to_overwrite(tmp_path):
    destination = tmp_path / "candidate17.json"
    original = (
        json.dumps({"candidate": 17, "verdict": "fail"}, indent=2) + "\n"
    )

    write_new(destination, original)
    original_bytes = destination.read_bytes()
    assert destination.read_text(encoding="utf-8") == original

    with pytest.raises(FileExistsError, match="one-shot rule"):
        write_new(destination, '{"candidate": 17, "verdict": "pass"}\n')
    with pytest.raises(FileExistsError, match="one-shot rule"):
        write_new(destination, object())

    assert destination.read_bytes() == original_bytes
