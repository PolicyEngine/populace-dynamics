"""Build the byte-level manifest for frozen Gate-2 legacy evidence."""

from __future__ import annotations

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


def build_manifest() -> dict[str, object]:
    """Return the deterministic legacy manifest."""
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError(
            "Full Git history is required to recover originating commits"
        )
    current_entries = [_entry(path) for path in _legacy_paths()]
    contract_revision = _git("rev-parse", "HEAD:gates.yaml")
    entries = _append_only_entries(
        current_entries, _existing_manifest(), contract_revision
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_commit": GENERATED_AT_COMMIT,
        "contract_revision": contract_revision,
        "entries": entries,
    }


def main() -> None:
    """Regenerate the checked-in manifest."""
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
