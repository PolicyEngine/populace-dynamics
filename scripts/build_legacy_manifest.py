"""Build the byte-level manifest for frozen Gate-2 legacy evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "runs" / "legacy_manifest_v1.json"
SCHEMA_VERSION = "legacy_manifest.v1"

# This is the clean origin/master snapshot from which manifest v1 was
# generated. Pinning it avoids the impossible self-reference of embedding the
# SHA of the commit that first adds the manifest inside that same commit.
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


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _legacy_paths() -> list[str]:
    paths = {"gates.yaml"}
    for pattern in LEGACY_PATTERNS:
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.glob(pattern)
            if path.is_file()
        )
    return sorted(paths)


def _assert_matches_head(relative_path: str) -> None:
    working_blob = _git("hash-object", "--", relative_path)
    head_blob = _git("rev-parse", f"HEAD:{relative_path}")
    if working_blob != head_blob:
        raise RuntimeError(
            f"{relative_path} differs from HEAD; refusing to record dirty "
            "legacy bytes"
        )


def _entry(relative_path: str) -> dict[str, object]:
    _assert_matches_head(relative_path)
    payload = (ROOT / relative_path).read_bytes()
    originating_commit = _git(
        "log",
        "--diff-filter=A",
        "-1",
        "--format=%H",
        "--",
        relative_path,
    )
    if not originating_commit:
        raise RuntimeError(f"No originating commit found for {relative_path}")
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "n_bytes": len(payload),
        "originating_commit": originating_commit,
    }


def _existing_manifest() -> dict[str, object] | None:
    head_manifest = subprocess.run(
        ["git", "show", "HEAD:runs/legacy_manifest_v1.json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if head_manifest.returncode == 0:
        payload = head_manifest.stdout
    elif MANIFEST_PATH.exists():
        payload = MANIFEST_PATH.read_text(encoding="utf-8")
    else:
        return None

    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise RuntimeError("Existing legacy manifest must be a JSON object")
    return manifest


def _append_only_entries(
    current_entries: list[dict[str, object]],
    existing_manifest: dict[str, object] | None,
    contract_revision: str,
) -> list[dict[str, object]]:
    if existing_manifest is None:
        return current_entries

    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("generated_at_commit", GENERATED_AT_COMMIT),
        ("contract_revision", contract_revision),
    ):
        if existing_manifest.get(key) != expected:
            raise RuntimeError(f"Existing manifest has unexpected {key}")

    existing_entries = existing_manifest.get("entries")
    if not isinstance(existing_entries, list):
        raise RuntimeError("Existing manifest entries must be a list")

    current_by_path = {str(entry["path"]): entry for entry in current_entries}
    pinned_entries: list[dict[str, object]] = []
    pinned_paths: set[str] = set()
    for entry in existing_entries:
        if not isinstance(entry, dict) or "path" not in entry:
            raise RuntimeError("Existing manifest contains an invalid entry")
        path = entry["path"]
        if not isinstance(path, str):
            raise RuntimeError("Existing manifest path must be a string")
        if path in pinned_paths:
            raise RuntimeError(f"Duplicate existing manifest entry: {path}")
        pinned_paths.add(path)
        if current_by_path.get(path) != entry:
            raise RuntimeError(
                f"Append-only manifest entry changed or disappeared: {path}"
            )
        pinned_entries.append(entry)

    return pinned_entries + [
        entry for entry in current_entries if entry["path"] not in pinned_paths
    ]


def _manifest_payload(
    contract_revision: str,
    entries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_commit": GENERATED_AT_COMMIT,
        "contract_revision": contract_revision,
        "entries": entries,
    }


def _manifest_at(revision: str) -> dict[str, object]:
    manifest_path = "runs/legacy_manifest_v1.json"
    result = subprocess.run(
        ["git", "show", f"{revision}:{manifest_path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Transition requires the parent commit to contain the legacy "
            "manifest"
        )
    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Parent legacy manifest must contain valid JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise RuntimeError("Parent legacy manifest must be a JSON object")
    return manifest


def _validated_parent_entries(
    parent: str,
    parent_manifest: dict[str, object],
) -> list[dict[str, object]]:
    expected_keys = {
        "schema_version",
        "generated_at_commit",
        "contract_revision",
        "entries",
    }
    if set(parent_manifest) != expected_keys:
        raise RuntimeError("Parent legacy manifest has unexpected fields")
    if parent_manifest["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(
            "Parent legacy manifest has unexpected schema_version"
        )
    if parent_manifest["generated_at_commit"] != GENERATED_AT_COMMIT:
        raise RuntimeError(
            "Parent legacy manifest has unexpected generated_at_commit"
        )

    try:
        parent_contract_revision = _git("rev-parse", f"{parent}:gates.yaml")
        parent_gates = _git_bytes("show", f"{parent}:gates.yaml")
        originating_commit = _git(
            "log",
            "--diff-filter=A",
            "-1",
            "--format=%H",
            parent,
            "--",
            "gates.yaml",
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Transition requires the parent commit to contain gates.yaml"
        ) from error
    expected_gates_entry = {
        "path": "gates.yaml",
        "sha256": hashlib.sha256(parent_gates).hexdigest(),
        "n_bytes": len(parent_gates),
        "originating_commit": originating_commit,
    }

    entries = parent_manifest["entries"]
    if not isinstance(entries, list):
        raise RuntimeError("Parent legacy manifest entries must be a list")
    parent_by_path: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry:
            raise RuntimeError(
                "Parent legacy manifest contains an invalid entry"
            )
        path = entry["path"]
        if not isinstance(path, str):
            raise RuntimeError("Parent legacy manifest path must be a string")
        if path in parent_by_path:
            raise RuntimeError(
                f"Duplicate parent legacy manifest entry: {path}"
            )
        parent_by_path[path] = entry

    if (
        parent_manifest["contract_revision"] != parent_contract_revision
        or parent_by_path.get("gates.yaml") != expected_gates_entry
    ):
        raise RuntimeError("Parent manifest does not match parent gates.yaml")
    return entries


def _transition_entries(
    current_entries: list[dict[str, object]],
    existing_manifest: dict[str, object] | None,
    contract_revision: str,
) -> list[dict[str, object]]:
    if existing_manifest is None:
        raise RuntimeError("Transition requires an existing legacy manifest")
    try:
        parent = _git("rev-parse", "HEAD^")
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Transition requires a parent commit") from error
    parent_manifest = _manifest_at(parent)
    parent_entries = _validated_parent_entries(parent, parent_manifest)
    current_by_path = {str(entry["path"]): entry for entry in current_entries}
    parent_paths = [str(entry["path"]) for entry in parent_entries]
    if set(current_by_path) != set(parent_paths):
        raise RuntimeError(
            "Transition may only change contract_revision and gates.yaml; "
            "legacy entries cannot be added or removed"
        )

    transitioned_entries: list[dict[str, object]] = []
    for parent_entry in parent_entries:
        path = str(parent_entry["path"])
        current_entry = current_by_path[path]
        if path != "gates.yaml" and current_entry != parent_entry:
            raise RuntimeError(
                "Transition may only change contract_revision and gates.yaml; "
                f"legacy entry moved: {path}"
            )
        transitioned_entries.append(current_entry)

    expected_manifest = _manifest_payload(
        contract_revision,
        transitioned_entries,
    )
    if (
        existing_manifest != parent_manifest
        and existing_manifest != expected_manifest
    ):
        raise RuntimeError(
            "Transition may only change contract_revision and gates.yaml; "
            "the current manifest is neither the parent nor the permitted "
            "transition manifest"
        )
    return transitioned_entries


def build_manifest(*, transition: bool = False) -> dict[str, object]:
    """Return the deterministic legacy manifest."""
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError(
            "Full Git history is required to recover originating commits"
        )
    current_entries = [_entry(path) for path in _legacy_paths()]
    contract_revision = _git("rev-parse", "HEAD:gates.yaml")
    existing_manifest = _existing_manifest()
    if transition:
        entries = _transition_entries(
            current_entries,
            existing_manifest,
            contract_revision,
        )
    else:
        entries = _append_only_entries(
            current_entries,
            existing_manifest,
            contract_revision,
        )
    return _manifest_payload(contract_revision, entries)


def main(argv: list[str] | None = None) -> None:
    """Regenerate the checked-in manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition",
        action="store_true",
        help="record a ratified gates.yaml transition from the parent commit",
    )
    args = parser.parse_args(argv)
    manifest = build_manifest(transition=args.transition)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
