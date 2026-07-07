"""Person-pair relationship map from the PSID Family Relationship Matrix.

The 1968-2023 Family Relationship Matrix File (PSID packaged product
1188, ``MX23REL``) records, for every responding family unit in every
wave, the relationship between each ordered pair of enumerated members
(ego, alter). It resolves household composition and couple/parent-child
links over time -- the join fabric behind the gate-2 marriage/
survivorship moments and the caregiver anchors, complementing the
event-dated marriage (:mod:`populace_dynamics.data.marriage`) and
childbirth (:mod:`populace_dynamics.data.births`) histories.

Person identifiers
------------------
Same convention as the rest of the stack. Ego is ``MX5``/``MX6`` and
alter ``MX10``/``MX11``, so ``ego_person_id = MX5 * 1000 + MX6`` and
``alter_person_id = MX10 * 1000 + MX11`` are directly joinable to the
individual file's ``ER30001 * 1000 + ER30002``. Every row carries valid
ego and alter ids -- the matrix enumerates only actual family members,
so (unlike the spouse/child links in the history files) the id columns
carry no missing sentinels.

Size and memory
---------------
The file is large: 3,485,034 rows, one per ordered within-family pair
per wave (about 250 MB as a nine-column int64 frame; the diagonal
``ego_rel_to_alter == 10`` "self" rows are ~26% of it). Read it
efficiently:

* :func:`relationship_map` reads only the nine substantive columns, not
  all twelve.
* Pass ``ego_rel_to_alter=`` (e.g. ``SPOUSE`` for couple links, or
  ``{SPOUSE, PARTNER}``) and/or ``waves=`` to slice by relationship or
  year, and ``drop_self=True`` to drop the diagonal.
* Pass ``chunksize=`` to stream the fixed-width read and apply those
  filters per chunk, so a slice (e.g. all spouse pairs, ~328k rows)
  never materializes the whole file. With no ``chunksize`` the full
  (filtered) frame is built in one read.

Relationship codes
-------------------
``ego_rel_to_rp`` (``MX7``) and ``alter_rel_to_rp`` (``MX12``) give each
person's relation to the family's reference person; ``ego_rel_to_alter``
(``MX8``) gives ego's relation to alter. The codes are kept raw (they
are a ~50-value frame) and documented here as module constants for
decoding.

Documented judgment calls (verified against the staged file / codebook,
2026-07-07)
-----------------------------------------------------------------------
* **Era-split relation-to-reference-person frame.** ``MX7``/``MX12`` use
  an abbreviated frame (codes 1-9) in interview waves 1968-1982 and a
  detailed frame (10, 20, 22, 30, ... 98) from 1983 on. The two are
  disjoint and both documented (:data:`REL_TO_REFERENCE_PERSON_PRE1983`,
  :data:`REL_TO_REFERENCE_PERSON_1983_PLUS`); :func:`rel_to_reference_person`
  returns the right one for a wave. ``MX8`` (ego-to-alter) uses one frame
  across all waves.
* **Codes retained, not decoded in place.** The frame keeps raw integer
  relationship codes so it stays a compact join key; callers decode via
  the exported constants. Tests assert the observed codes are a subset
  of the documented frames (a ~50-code era-split frame need not exhaust
  every value in every release).
* **No id sentinels.** ``MX5``/``MX6``/``MX10``/``MX11`` carry no 0 or
  9999 sentinels in the staged file (every enumerated pair is two real
  persons), so ego/alter ids are always populated. Sequence numbers
  (``MX4``/``MX9``) are 1-20 (present family members).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from populace_dynamics.data import psid

__all__ = [
    "PRODUCT",
    "relationship_map",
    "rel_to_reference_person",
    "REL_TO_REFERENCE_PERSON_PRE1983",
    "REL_TO_REFERENCE_PERSON_1983_PLUS",
    "EGO_REL_TO_ALTER",
    "SELF",
    "SPOUSE",
    "PARTNER",
    "CHILD",
    "PARENT",
]

#: PSID product key for the Family Relationship Matrix File.
PRODUCT = "MX23REL"

#: Every variable on the file, mapped to its exact documented label; the
#: whole layout is label-verified at read.
_VARS: dict[str, str] = {
    "MX1": "RELEASE NUMBER",
    "MX2": "INTERVIEW YEAR",
    "MX3": "INTERVIEW NUMBER",
    "MX4": "EGO SEQUENCE NUMBER",
    "MX5": "EGO 1968 INTERVIEW NUMBER",
    "MX6": "EGO PERSON NUMBER",
    "MX7": "EGO RELATION TO REFERENCE PERSON",
    "MX8": "EGO RELATION TO ALTER",
    "MX9": "ALTER SEQUENCE NUMBER",
    "MX10": "ALTER 1968 INTERVIEW NUMBER",
    "MX11": "ALTER PERSON NUMBER",
    "MX12": "ALTER RELATION TO REFERENCE PERSON",
}

#: The nine substantive columns actually read (MX1 release is skipped).
_READ_VARS: tuple[str, ...] = (
    "MX2",
    "MX3",
    "MX4",
    "MX5",
    "MX6",
    "MX7",
    "MX8",
    "MX9",
    "MX10",
    "MX11",
    "MX12",
)

#: MX7/MX12 relation-to-reference-person, interview waves 1968-1982.
REL_TO_REFERENCE_PERSON_PRE1983: dict[int, str] = {
    1: "reference_person",
    2: "spouse_or_partner",
    3: "child",
    4: "sibling",
    5: "parent",
    6: "grandchild_or_great_grandchild",
    7: "other_relative",
    8: "nonrelative",
    9: "husband_of_reference_person",
}

#: MX7/MX12 relation-to-reference-person, interview waves 1983 onward.
REL_TO_REFERENCE_PERSON_1983_PLUS: dict[int, str] = {
    10: "reference_person",
    20: "legal_spouse",
    22: "partner",
    30: "child",
    33: "stepchild",
    35: "child_of_partner",
    37: "child_in_law",
    38: "foster_child",
    40: "sibling",
    47: "sibling_in_law",
    48: "sibling_of_cohabitor",
    50: "parent",
    57: "parent_in_law",
    58: "parent_of_cohabitor",
    60: "grandchild",
    65: "great_grandchild",
    66: "grandparent",
    67: "grandparent_of_spouse",
    68: "great_grandparent",
    69: "great_grandparent_of_spouse",
    70: "niece_or_nephew",
    71: "niece_or_nephew_of_spouse",
    72: "uncle_or_aunt",
    73: "uncle_or_aunt_of_spouse",
    74: "cousin",
    75: "cousin_of_spouse",
    83: "child_of_first_year_cohabitor",
    88: "first_year_cohabitor",
    90: "uncooperative_spouse",
    92: "uncooperative_partner",
    95: "other_relative",
    96: "other_relative_of_spouse",
    97: "other_relative_of_cohabitor",
    98: "nonrelative",
}

#: MX8 ego-to-alter relationship (one frame across all waves).
EGO_REL_TO_ALTER: dict[int, str] = {
    10: "self",
    20: "legal_spouse",
    22: "partner",
    30: "child",
    33: "stepchild",
    35: "social_child",
    37: "child_in_law",
    38: "foster_child",
    39: "social_child_in_law",
    40: "sibling",
    43: "step_sibling",
    45: "social_sibling",
    47: "sibling_in_law",
    48: "social_sibling_in_law",
    50: "parent",
    53: "step_parent",
    55: "social_parent",
    56: "foster_parent",
    57: "parent_in_law",
    58: "social_parent_in_law",
    60: "grandchild",
    61: "step_grandchild",
    62: "grandchild_in_law",
    63: "step_great_grandchild",
    64: "great_grandchild_in_law",
    65: "great_grandchild",
    66: "grandparent",
    67: "grandparent_in_law",
    68: "great_grandparent",
    69: "great_grandparent_in_law",
    70: "niece_or_nephew",
    71: "niece_or_nephew_by_marriage",
    72: "uncle_or_aunt",
    73: "uncle_or_aunt_by_marriage",
    74: "cousin",
    75: "cousin_by_marriage",
    80: "social_grandchild",
    81: "social_great_grandchild",
    82: "social_grandparent",
    83: "social_great_grandparent",
    84: "social_niece_or_nephew",
    85: "social_uncle_or_aunt",
    86: "social_cousin",
    87: "step_grandparent",
    88: "step_great_grandparent",
    89: "other_foster_relative",
    94: "undetermined",
    95: "other_relative",
    96: "other_relative_by_marriage",
    97: "other_social_relative",
    98: "nonrelative",
}

#: Convenience MX8 (ego-to-alter) codes for the common links.
SELF = 10
SPOUSE = 20
PARTNER = 22
CHILD = 30
PARENT = 50

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "interview_year",
    "interview_number",
    "ego_person_id",
    "ego_sequence",
    "ego_rel_to_rp",
    "ego_rel_to_alter",
    "alter_person_id",
    "alter_sequence",
    "alter_rel_to_rp",
)

_RP_ERA_BREAK = 1983


def rel_to_reference_person(wave: int) -> dict[int, str]:
    """Return the MX7/MX12 code frame for a given interview wave.

    The abbreviated 1968-1982 frame or the detailed 1983-onward frame,
    per the documented era split.
    """
    if wave < _RP_ERA_BREAK:
        return REL_TO_REFERENCE_PERSON_PRE1983
    return REL_TO_REFERENCE_PERSON_1983_PLUS


def _as_set(codes: int | Iterable[int] | None) -> set[int] | None:
    if codes is None:
        return None
    if isinstance(codes, int):
        return {codes}
    return set(codes)


def _build(raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble the join-friendly pair frame from raw MX columns.

    Pure transform, split out so id construction and the relationship
    filters are unit-testable on synthetic rows.
    """
    return pd.DataFrame(
        {
            "interview_year": raw["MX2"].astype("int64"),
            "interview_number": raw["MX3"].astype("int64"),
            "ego_person_id": (
                raw["MX5"].astype("int64") * 1000 + raw["MX6"].astype("int64")
            ).astype("int64"),
            "ego_sequence": raw["MX4"].astype("int64"),
            "ego_rel_to_rp": raw["MX7"].astype("int64"),
            "ego_rel_to_alter": raw["MX8"].astype("int64"),
            "alter_person_id": (
                raw["MX10"].astype("int64") * 1000
                + raw["MX11"].astype("int64")
            ).astype("int64"),
            "alter_sequence": raw["MX9"].astype("int64"),
            "alter_rel_to_rp": raw["MX12"].astype("int64"),
        }
    )


def _select(
    frame: pd.DataFrame,
    waves: set[int] | None,
    rels: set[int] | None,
    drop_self: bool,
) -> pd.DataFrame:
    """Apply the wave / relationship / drop-self row filters."""
    mask = pd.Series(True, index=frame.index)
    if waves is not None:
        mask &= frame["interview_year"].isin(waves)
    if rels is not None:
        mask &= frame["ego_rel_to_alter"].isin(rels)
    if drop_self:
        mask &= frame["ego_rel_to_alter"] != SELF
    return frame.loc[mask].reset_index(drop=True)


def _empty() -> pd.DataFrame:
    raw = pd.DataFrame({v: pd.Series(dtype="int64") for v in _READ_VARS})
    return _build(raw)


def relationship_map(
    *,
    waves: Iterable[int] | None = None,
    ego_rel_to_alter: int | Iterable[int] | None = None,
    drop_self: bool = False,
    data_dir: Path | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Read the Family Relationship Matrix into a join-friendly pair frame.

    Columns: ``interview_year``, ``interview_number``, ``ego_person_id``,
    ``ego_sequence``, ``ego_rel_to_rp``, ``ego_rel_to_alter``,
    ``alter_person_id``, ``alter_sequence``, ``alter_rel_to_rp``. All
    ``int64``. Labels are verified before the fixed-width read, so a
    re-layout fails loudly.

    Args:
        waves: Keep only these interview years (any iterable of years).
        ego_rel_to_alter: Keep only these ``MX8`` relationship codes
            (an int or iterable, e.g. :data:`SPOUSE` for couple links).
        drop_self: Drop the diagonal ``ego_rel_to_alter == SELF`` rows.
        data_dir: Staged-data directory (see
            :func:`populace_dynamics.data.psid.read_psid`).
        nrows: Cap the fixed-width read (for smoke tests).
        chunksize: Stream the read in chunks of this many rows, applying
            the filters per chunk. Use with ``ego_rel_to_alter``/``waves``
            to extract a slice without materializing the ~3.5M-row file
            (see the module docstring on memory).

    Returns:
        The filtered pair frame, row order preserved from the file.
    """
    sps_path = psid.product_sps_path(PRODUCT, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    psid.verify_labels(labels, _VARS, context=PRODUCT)
    layout = psid.parse_sps_layout(sps_path).set_index("name")
    names = list(_READ_VARS)
    # Convert the SPSS 1-indexed inclusive layout to pandas 0-indexed
    # half-open colspecs (start - 1, end); see the psid module docstring.
    colspecs = [
        (int(layout.loc[n, "start"]) - 1, int(layout.loc[n, "end"]))
        for n in names
    ]
    txt_path = sps_path.parent / psid.PRODUCTS[PRODUCT][1]

    wave_set = _as_set(waves)
    rel_set = _as_set(ego_rel_to_alter)

    reader = pd.read_fwf(
        txt_path,
        colspecs=colspecs,
        names=names,
        header=None,
        nrows=nrows,
        chunksize=chunksize,
    )
    if chunksize is None:
        return _select(_build(reader), wave_set, rel_set, drop_self)

    parts = [
        _select(_build(chunk), wave_set, rel_set, drop_self)
        for chunk in reader
    ]
    if not parts:
        return _empty()
    return pd.concat(parts, ignore_index=True)
