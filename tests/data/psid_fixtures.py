"""Shared helper to synthesize a tiny fixed-width PSID product.

The demographic-reader unit tests build a real (miniature) PSID product
-- a fixed-width ``.txt`` plus an SPSS ``.sps`` setup file -- so they
exercise the same label-verification and colspec-conversion path the
readers use on the staged files, without touching the multi-MB real
data. This mirrors ``tests/data/test_family.py``'s ``_write_product``:
the DATA LIST positions are derived from the field widths so they can
never drift out of sync with the right-justified text.
"""

from __future__ import annotations

from pathlib import Path


def write_product(
    directory: Path,
    sps_name: str,
    txt_name: str,
    fields: list[tuple[str, int, str, list[int]]],
) -> None:
    """Write a fixture PSID product from field tables.

    Args:
        directory: Product subdirectory to create (e.g.
            ``tmp_path / "mh85_23"``).
        sps_name: SPSS setup filename (e.g. ``"MH85_23.sps"``).
        txt_name: Fixed-width data filename (e.g. ``"MH85_23.txt"``).
        fields: ``(name, width, label, values)`` per variable. Column
            positions are computed from the widths, and each row's text
            is right-justified to the width, so the layout the reader
            parses always matches the bytes it reads.
    """
    directory.mkdir(parents=True, exist_ok=True)
    specs = []
    position = 1
    for name, width, _, _ in fields:
        end = position + width - 1
        specs.append(f"      {name:<15} {position} - {end}")
        position += width
    labels = "\n".join(
        f'   {name:<12} "{label}"' for name, _, label, _ in fields
    )
    sps = (
        "DATA LIST FILE = PSID FIXED /\n"
        + "\n".join(specs)
        + "\n.\n\nVARIABLE LABELS\n"
        + labels
        + "\n.\n"
    )
    (directory / sps_name).write_text(sps)
    n_rows = len(fields[0][3])
    lines = [
        "".join(f"{values[i]:>{width}}" for _, width, _, values in fields)
        for i in range(n_rows)
    ]
    (directory / txt_name).write_text("\n".join(lines) + "\n")
