"""Safe helpers for writing one-shot run artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _already_exists(destination: Path) -> FileExistsError:
    return FileExistsError(
        f"{destination} already exists; the one-shot rule requires a "
        "new artifact path"
    )


def write_new(path: str | Path, data: object) -> None:
    """Write an artifact without permitting an existing file to be replaced.

    Strings and bytes are written verbatim. Other values are serialized as
    indented JSON with a trailing newline.
    """
    destination = Path(path)
    if os.path.lexists(destination):
        raise _already_exists(destination)

    if isinstance(data, (str, bytes)):
        payload = data
    else:
        payload = json.dumps(data, indent=2) + "\n"

    try:
        if isinstance(payload, bytes):
            with destination.open("xb") as output:
                output.write(payload)
        else:
            with destination.open("x", encoding="utf-8", newline="") as output:
                output.write(payload)
    except FileExistsError:
        raise _already_exists(destination) from None
