"""Safe helpers for writing one-shot run artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from populace_dynamics.contract import ContractRef, environment_block


def _already_exists(destination: Path) -> FileExistsError:
    return FileExistsError(
        f"{destination} already exists; the one-shot rule requires a "
        "new artifact path"
    )


def _payload(data: object) -> str | bytes:
    if isinstance(data, (str, bytes)):
        return data
    return json.dumps(data, indent=2) + "\n"


def _write_exclusive(destination: Path, payload: str | bytes) -> None:
    created = False
    try:
        if isinstance(payload, bytes):
            with destination.open("xb") as output:
                created = True
                output.write(payload)
        else:
            with destination.open("x", encoding="utf-8", newline="") as output:
                created = True
                output.write(payload)
    except FileExistsError:
        raise _already_exists(destination) from None
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def write_new(
    path: str | Path,
    data: object,
    *,
    sidecar: bool = False,
) -> None:
    """Write an artifact without permitting an existing file to be replaced.

    Strings and bytes are written verbatim. Other values are serialized as
    indented JSON with a trailing newline. If ``sidecar`` is true, an
    environment and contract reference is written to ``<path>.env.json``.
    """
    destination = Path(path)
    sidecar_destination = Path(f"{destination}.env.json")
    targets = (destination, sidecar_destination) if sidecar else (destination,)
    for target in targets:
        if os.path.lexists(target):
            raise _already_exists(target)

    artifact_payload = _payload(data)
    sidecar_payload = None
    if sidecar:
        sidecar_payload = _payload(
            {
                "environment": environment_block(),
                "contract": asdict(ContractRef.current()),
            }
        )

    primary_written = False
    try:
        _write_exclusive(destination, artifact_payload)
        primary_written = True
        if sidecar_payload is not None:
            _write_exclusive(sidecar_destination, sidecar_payload)
    except BaseException:
        if primary_written:
            destination.unlink(missing_ok=True)
        raise
