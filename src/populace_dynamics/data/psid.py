"""Fixed-width loader for PSID packaged-data products.

The Panel Study of Income Dynamics (PSID) distributes each data
product as a fixed-width ``.txt`` file plus a companion SPSS setup
file (``.sps``) that documents the column layout and human-readable
variable labels. PSID variable names are opaque codes (``ER30004``,
``MH6``, ``MX2``, ...) so the labels are the only practical way to
find a variable of interest -- e.g. the label ``"AGE OF INDIVIDUAL
68"`` tells you both what the variable measures and which wave (1968)
it belongs to.

This module parses the ``.sps`` setup files to recover the column
layout and labels, then reads only the requested columns out of the
(large) fixed-width ``.txt`` file with :func:`pandas.read_fwf`.

SPSS DATA LIST column convention
---------------------------------
PSID ``.sps`` files declare columns in a ``DATA LIST`` block shaped
like::

    DATA LIST FILE = PSID FIXED /
          ER30000         1 - 1         ER30001         2 - 5

Each ``NAME START - END`` triple gives **1-indexed, inclusive**
character positions within the fixed-width record (so ``ER30000`` is
the single character at position 1, and ``ER30001`` spans positions
2 through 5 inclusive, i.e. 4 characters wide). This is the
convention returned by :func:`parse_sps_layout` in the ``start`` and
``end`` columns.

:func:`pandas.read_fwf`, by contrast, wants ``colspecs`` as
**0-indexed, half-open** ``(start, end)`` tuples (Python slice
semantics: ``record[start:end]``). Converting from the SPSS
convention to the pandas convention only requires shifting the start
down by one -- the end position needs no change, because "1-indexed
inclusive end" and "0-indexed exclusive end" happen to be the same
integer (e.g. SPSS ``2 - 5`` is characters 2,3,4,5; as a 0-indexed
half-open slice that is ``record[1:5]``, i.e. characters at 0-indexed
positions 1,2,3,4, which are the same four characters). This module
performs and documents that conversion in :func:`read_psid`.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import pandas as pd

__all__ = [
    "PRODUCTS",
    "parse_sps_layout",
    "parse_sps_labels",
    "find_variables",
    "verify_labels",
    "product_sps_path",
    "read_psid",
]


# Registry of staged PSID packaged-data products: product key ->
# (subdir name, fixed-width .txt filename, .sps setup filename).
# Derived from inspecting the staged directories under
# ~/PolicyEngine/psid-data/ (2026-07-03); see that directory's
# README.md for provenance.
PRODUCTS: dict[str, tuple[str, str, str]] = {
    "ind2023er": ("ind2023er", "IND2023ER.txt", "IND2023ER.sps"),
    "mh85_23": ("mh85_23", "MH85_23.txt", "MH85_23.sps"),
    "cah85_23": ("cah85_23", "CAH85_23.txt", "CAH85_23.sps"),
    "MX23REL": ("MX23REL", "MX23REL.txt", "MX23REL.sps"),
}

# Products whose full-column read is large enough to warrant a
# warning when the caller passes columns=None.
_LARGE_PRODUCTS = {"ind2023er"}

_DEFAULT_DATA_DIR_ENV = "POPULACE_DYNAMICS_PSID_DIR"
_DEFAULT_DATA_DIR = Path("~/PolicyEngine/psid-data").expanduser()

_README_POINTER = (
    "see ~/PolicyEngine/psid-data/README.md for what should be "
    "staged and how to fetch it"
)

# Matches a "NAME  START - END" triple in a DATA LIST block, e.g.
# "ER30001         2 - 5". Variable names are alphanumeric PSID
# codes that may carry a trailing letter suffix (e.g. "ER33294A").
_COLSPEC_RE = re.compile(r"(\S+)\s+(\d+)\s*-\s*(\d+)")

# Matches a "NAME  "LABEL"" pair in a VARIABLE LABELS block, e.g.
# 'ER30004      "AGE OF INDIVIDUAL                     68"'.
_LABEL_RE = re.compile(r'^\s*(\S+)\s+"([^"]*)"', re.MULTILINE)


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Resolve the PSID data directory from arg, env var, or default.

    Precedence: explicit ``data_dir`` argument, then the
    ``POPULACE_DYNAMICS_PSID_DIR`` environment variable, then
    ``~/PolicyEngine/psid-data``.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser()
    env_value = os.environ.get(_DEFAULT_DATA_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_DATA_DIR


def parse_sps_layout(sps_path: str | Path) -> pd.DataFrame:
    """Parse the ``DATA LIST`` column layout from a PSID ``.sps`` file.

    Args:
        sps_path: Path to the SPSS setup file.

    Returns:
        A DataFrame with columns ``["name", "start", "end", "width"]``
        in file order, one row per variable. ``start`` and ``end`` are
        **1-indexed and inclusive**, matching the SPSS DATA LIST
        convention printed in the file itself (e.g. ``ER30001``
        spanning columns 2 through 5 inclusive appears as
        ``start=2, end=5, width=4``). This is *not* the convention
        :func:`pandas.read_fwf` expects for ``colspecs`` -- see the
        module docstring for the conversion.

    Raises:
        FileNotFoundError: If ``sps_path`` does not exist.
        ValueError: If no ``DATA LIST`` block can be found.
    """
    sps_path = Path(sps_path)
    if not sps_path.is_file():
        raise FileNotFoundError(
            f"PSID setup file not found: {sps_path} ({_README_POINTER})"
        )
    text = sps_path.read_text(errors="replace")

    match = re.search(r"DATA LIST\b.*?/(.*?)\n\s*\.\s*\n", text, re.DOTALL)
    if not match:
        raise ValueError(
            f"No DATA LIST block found in {sps_path}; the file may "
            "not be a PSID SPSS setup file"
        )
    block = match.group(1)

    names: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for name, start, end in _COLSPEC_RE.findall(block):
        names.append(name)
        starts.append(int(start))
        ends.append(int(end))

    if not names:
        raise ValueError(
            f"DATA LIST block in {sps_path} contained no column "
            "specifications"
        )

    return pd.DataFrame(
        {
            "name": names,
            "start": starts,
            "end": ends,
            "width": [e - s + 1 for s, e in zip(starts, ends, strict=True)],
        }
    )


def parse_sps_labels(sps_path: str | Path) -> dict[str, str]:
    """Parse the ``VARIABLE LABELS`` block from a PSID ``.sps`` file.

    Args:
        sps_path: Path to the SPSS setup file.

    Returns:
        A dict mapping variable name to its human-readable label,
        with surrounding whitespace stripped. PSID labels often
        carry the survey wave/year as trailing digits (e.g.
        ``"AGE OF INDIVIDUAL 68"`` for the 1968 wave), since PSID
        variable names themselves are opaque codes.

    Raises:
        FileNotFoundError: If ``sps_path`` does not exist.
        ValueError: If no ``VARIABLE LABELS`` block can be found.
    """
    sps_path = Path(sps_path)
    if not sps_path.is_file():
        raise FileNotFoundError(
            f"PSID setup file not found: {sps_path} ({_README_POINTER})"
        )
    text = sps_path.read_text(errors="replace")

    match = re.search(
        r"VARIABLE LABELS\s*\n(.*?)\n\s*\.\s*\n", text, re.DOTALL
    )
    if not match:
        raise ValueError(
            f"No VARIABLE LABELS block found in {sps_path}; the "
            "file may not be a PSID SPSS setup file"
        )
    block = match.group(1)

    labels = {name: label.strip() for name, label in _LABEL_RE.findall(block)}
    if not labels:
        raise ValueError(
            f"VARIABLE LABELS block in {sps_path} contained no "
            "name/label pairs"
        )
    return labels


def find_variables(labels: dict[str, str], pattern: str) -> dict[str, str]:
    """Filter a labels dict by a case-insensitive regex over labels.

    Since PSID variable names are opaque codes, this is the
    label-driven way to build a "varmap" of variables of interest --
    e.g. ``find_variables(labels, r"\\bAGE OF INDIVIDUAL\\b")`` finds
    every wave's age variable, each carrying its own year in the
    label (``"... 68"``, ``"... 69"``, ...).

    Args:
        labels: A name -> label dict, as returned by
            :func:`parse_sps_labels`.
        pattern: A regular expression tested against each label
            (case-insensitively) with :func:`re.search`.

    Returns:
        The subset of ``labels`` whose label text matches
        ``pattern``, preserving input order.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    return {
        name: label for name, label in labels.items() if regex.search(label)
    }


def verify_labels(
    labels: dict[str, str],
    expected: dict[str, str],
    *,
    context: str,
) -> None:
    """Assert each variable's label matches the expected text, or raise.

    This is the label-verification discipline the demographic readers
    (:mod:`populace_dynamics.data.marriage`, ``births``, ``relmap``)
    share: PSID variable names are opaque codes, so a reader that hard
    maps ``MH12 -> "how the marriage ended"`` is only correct while the
    release's column-to-name assignment holds. Checking the label at
    read time turns a re-layout (a shifted or renamed column) into a
    loud failure instead of a silently mismapped column.

    Comparison normalizes runs of whitespace, because PSID pads label
    columns with variable spacing. Raises :class:`ValueError` naming the
    first mismatching variable, its actual label, and the expected one;
    ``context`` names the product in the message (e.g. ``"mh85_23"``).
    """
    for var, want in expected.items():
        actual = " ".join(labels.get(var, "").split())
        wanted = " ".join(want.split())
        if actual != wanted:
            raise ValueError(
                f"{context}: variable {var} label {actual!r} does not "
                f"match the expected layout ({wanted!r}). The release "
                "layout may have changed."
            )


def product_sps_path(product: str, data_dir: Path | None = None) -> Path:
    """Resolve the ``.sps`` setup path for a staged product.

    Convenience for readers that need the labels (via
    :func:`parse_sps_labels`) before reading columns, without
    re-deriving the staged-directory layout.

    Raises:
        KeyError: If ``product`` is not a recognized product key.
    """
    if product not in PRODUCTS:
        raise KeyError(
            f"Unknown PSID product {product!r}; expected one of "
            f"{sorted(PRODUCTS)}"
        )
    subdir, _, sps_name = PRODUCTS[product]
    return _resolve_data_dir(data_dir) / subdir / sps_name


def read_psid(
    product: str,
    columns: list[str] | None = None,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read selected columns from a staged PSID fixed-width product.

    Args:
        product: One of the keys in :data:`PRODUCTS` (currently
            ``"ind2023er"``, ``"mh85_23"``, ``"cah85_23"``,
            ``"MX23REL"``).
        columns: Variable names to read, matching names in the
            product's ``.sps`` layout. ``None`` reads every column
            (a warning is emitted for large products, since e.g.
            ``ind2023er`` has 2771 columns across 85,536 rows).
        data_dir: Directory containing the product subdirectories.
            If ``None``, resolved from the
            ``POPULACE_DYNAMICS_PSID_DIR`` environment variable, and
            failing that, ``~/PolicyEngine/psid-data``.
        nrows: If given, read only this many rows (useful for
            smoke-testing against the real, multi-hundred-MB files).

    Returns:
        A DataFrame with one column per requested variable (or every
        variable in layout order, if ``columns`` is ``None``), and
        one row per fixed-width record.

    Raises:
        KeyError: If ``product`` is not a recognized product key.
        FileNotFoundError: If the resolved data directory, ``.sps``
            file, or ``.txt`` file does not exist.
        KeyError: If a requested column name is not present in the
            product's layout.
    """
    if product not in PRODUCTS:
        raise KeyError(
            f"Unknown PSID product {product!r}; expected one of "
            f"{sorted(PRODUCTS)}"
        )
    subdir, txt_name, sps_name = PRODUCTS[product]

    resolved_dir = _resolve_data_dir(data_dir)
    product_dir = resolved_dir / subdir
    if not product_dir.is_dir():
        raise FileNotFoundError(
            f"PSID product directory not found: {product_dir} "
            f"({_README_POINTER})"
        )

    sps_path = product_dir / sps_name
    txt_path = product_dir / txt_name
    if not sps_path.is_file():
        raise FileNotFoundError(
            f"PSID setup file not found: {sps_path} ({_README_POINTER})"
        )
    if not txt_path.is_file():
        raise FileNotFoundError(
            f"PSID fixed-width data file not found: {txt_path} "
            f"({_README_POINTER})"
        )

    layout = parse_sps_layout(sps_path)
    layout_names = list(layout["name"])

    if columns is None:
        if product in _LARGE_PRODUCTS:
            warnings.warn(
                f"read_psid({product!r}) with columns=None reads all "
                f"{len(layout_names)} columns; pass an explicit "
                "columns=[...] list to read only what you need",
                stacklevel=2,
            )
        selected_names = layout_names
    else:
        missing = [c for c in columns if c not in set(layout_names)]
        if missing:
            raise KeyError(
                f"Column(s) {missing} not found in {sps_path} layout"
            )
        selected_names = list(columns)

    layout_by_name = layout.set_index("name")
    # Convert from the SPSS DATA LIST convention (1-indexed,
    # inclusive start/end) to the pandas read_fwf convention
    # (0-indexed, half-open [start, end)). Only the start needs to
    # shift down by one -- see the module docstring for why the end
    # position is numerically unchanged between the two conventions.
    colspecs = [
        (
            int(layout_by_name.loc[name, "start"]) - 1,
            int(layout_by_name.loc[name, "end"]),
        )
        for name in selected_names
    ]

    return pd.read_fwf(
        txt_path,
        colspecs=colspecs,
        names=selected_names,
        header=None,
        nrows=nrows,
    )
