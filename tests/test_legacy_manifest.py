import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from populace_dynamics.artifacts import write_new
from scripts import build_legacy_manifest as legacy_manifest_builder

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


def _fixture_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_fixture(root: Path, message: str) -> None:
    _fixture_git(root, "add", "--all")
    _fixture_git(
        root,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest-test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        message,
    )


def _amend_fixture(root: Path) -> None:
    _fixture_git(root, "add", "--all")
    _fixture_git(
        root,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest-test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--amend",
        "--no-edit",
    )


def _write_fixture_manifest(root: Path, manifest: dict[str, object]) -> None:
    manifest_path = root / "runs" / "legacy_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _fresh_fixture_manifest() -> dict[str, object]:
    entries = [
        legacy_manifest_builder._entry(path)
        for path in legacy_manifest_builder._legacy_paths()
    ]
    return {
        "schema_version": legacy_manifest_builder.SCHEMA_VERSION,
        "generated_at_commit": legacy_manifest_builder.GENERATED_AT_COMMIT,
        "contract_revision": legacy_manifest_builder._git(
            "rev-parse", "HEAD:gates.yaml"
        ),
        "entries": entries,
    }


@pytest.fixture
def legacy_manifest_repo(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    scripts_path = root / "scripts"
    runs_path = root / "runs"
    scripts_path.mkdir(parents=True)
    runs_path.mkdir()
    (root / "gates.yaml").write_text(
        "contract: ratified-parent\n", encoding="utf-8"
    )
    (scripts_path / "run_gate2_candidate1.py").write_text(
        'CANDIDATE = "frozen"\n', encoding="utf-8"
    )
    (runs_path / "gate2_hazard_v1.json").write_text(
        '{"status": "frozen"}\n', encoding="utf-8"
    )

    _fixture_git(root, "init", "-b", "master")
    _commit_fixture(root, "Add contract and frozen evidence")

    monkeypatch.setattr(legacy_manifest_builder, "ROOT", root)
    monkeypatch.setattr(
        legacy_manifest_builder,
        "MANIFEST_PATH",
        runs_path / "legacy_manifest_v1.json",
    )
    _write_fixture_manifest(root, legacy_manifest_builder.build_manifest())
    _commit_fixture(root, "Add legacy manifest")
    return root


def _commit_contract_transition(root: Path) -> None:
    (root / "gates.yaml").write_text(
        "contract: ratified-transition-with-longer-content\n",
        encoding="utf-8",
    )
    _commit_fixture(root, "Ratify contract transition")


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


def test_transition_mode_accepts_only_ratified_contract_movement(
    legacy_manifest_repo,
):
    root = legacy_manifest_repo
    manifest_path = root / "runs" / "legacy_manifest_v1.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    _commit_contract_transition(root)

    legacy_manifest_builder.main(["--transition"])

    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_top_level = {key for key in before if before[key] != after[key]}
    assert changed_top_level == {"contract_revision", "entries"}
    assert after["contract_revision"] == _fixture_git(
        root, "rev-parse", "HEAD:gates.yaml"
    )

    before_by_path = {entry["path"]: entry for entry in before["entries"]}
    after_by_path = {entry["path"]: entry for entry in after["entries"]}
    changed_entries = {
        path
        for path in before_by_path
        if before_by_path[path] != after_by_path[path]
    }
    assert changed_entries == {"gates.yaml"}
    assert list(before_by_path) == list(after_by_path)
    changed_gate_fields = {
        key
        for key in before_by_path["gates.yaml"]
        if before_by_path["gates.yaml"][key]
        != after_by_path["gates.yaml"][key]
    }
    assert changed_gate_fields == {"sha256", "n_bytes"}


@pytest.mark.parametrize(
    "movement",
    ["changed", "added", "removed", "reordered"],
)
def test_transition_mode_rejects_non_gates_movement(
    legacy_manifest_repo,
    movement,
):
    root = legacy_manifest_repo
    manifest_path = root / "runs" / "legacy_manifest_v1.json"
    candidate_path = root / "scripts" / "run_gate2_candidate1.py"
    if movement == "changed":
        candidate_path.write_text('CANDIDATE = "moved"\n', encoding="utf-8")
    elif movement == "added":
        (root / "scripts" / "run_gate2_candidate2.py").write_text(
            'CANDIDATE = "added"\n', encoding="utf-8"
        )
    elif movement == "removed":
        candidate_path.unlink()
    (root / "gates.yaml").write_text(
        "contract: ratified-transition-with-longer-content\n",
        encoding="utf-8",
    )
    _commit_fixture(root, "Move contract and frozen evidence")
    current_manifest = _fresh_fixture_manifest()
    if movement == "reordered":
        current_manifest["entries"][1:] = reversed(
            current_manifest["entries"][1:]
        )
    _write_fixture_manifest(root, current_manifest)
    _amend_fixture(root)
    before = manifest_path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match="Transition may only change contract_revision and gates.yaml",
    ):
        legacy_manifest_builder.main(["--transition"])

    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("mismatch", ["contract_revision", "gates_entry"])
def test_transition_mode_rejects_parent_manifest_mismatch(
    legacy_manifest_repo,
    mismatch,
):
    root = legacy_manifest_repo
    manifest_path = root / "runs" / "legacy_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mismatch == "contract_revision":
        manifest["contract_revision"] = "0" * 40
    else:
        gates_entry = next(
            entry
            for entry in manifest["entries"]
            if entry["path"] == "gates.yaml"
        )
        gates_entry["sha256"] = "0" * 64
    _write_fixture_manifest(root, manifest)
    _commit_fixture(root, "Corrupt parent manifest")
    _commit_contract_transition(root)
    _write_fixture_manifest(root, _fresh_fixture_manifest())
    _amend_fixture(root)
    before = manifest_path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match="Parent manifest does not match parent gates.yaml",
    ):
        legacy_manifest_builder.main(["--transition"])

    assert manifest_path.read_bytes() == before


def test_default_mode_keeps_append_only_behavior(legacy_manifest_repo):
    root = legacy_manifest_repo
    manifest_path = root / "runs" / "legacy_manifest_v1.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_path = root / "scripts" / "run_gate2_candidate2.py"
    new_path.write_text('CANDIDATE = "new"\n', encoding="utf-8")
    _commit_fixture(root, "Add later legacy evidence")

    legacy_manifest_builder.main([])

    appended = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert appended["entries"][:-1] == before["entries"]
    assert appended["entries"][-1]["path"] == (
        "scripts/run_gate2_candidate2.py"
    )
    _commit_fixture(root, "Append later evidence to manifest")

    _commit_contract_transition(root)
    before_rejection = manifest_path.read_bytes()
    with pytest.raises(
        RuntimeError,
        match="Existing manifest has unexpected contract_revision",
    ):
        legacy_manifest_builder.main([])
    assert manifest_path.read_bytes() == before_rejection


def test_transition_mode_is_idempotent_after_manifest_commit(
    legacy_manifest_repo,
):
    root = legacy_manifest_repo
    manifest_path = root / "runs" / "legacy_manifest_v1.json"
    _commit_contract_transition(root)
    legacy_manifest_builder.main(["--transition"])
    _amend_fixture(root)
    transitioned = manifest_path.read_bytes()

    legacy_manifest_builder.main(["--transition"])

    assert manifest_path.read_bytes() == transitioned
    assert _fixture_git(root, "status", "--porcelain") == ""


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
